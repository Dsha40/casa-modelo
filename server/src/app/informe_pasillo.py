"""src/app/informe_pasillo.py — «Informe del pasillo», el PDF del dashboard
(13-oct-2026).

## Qué es

El botón «Exportar PDF» del dashboard de tienda genera este informe, con el
formato que el usuario diseñó a partir de los informes de la sesión
(`Informe_Pasillo_2026-10-09.pdf`): cabecera azul, resumen en cuatro
tarjetas, tráfico, demografía (anillo de género, edades, cómo se decidió),
afluencia por hora, anaqueles (visitas, agarres, producto y género), stock
(antes y después, SIN la foto del cliente), mapas de calor por anaquel con
sus productos, permanencia y el mapa de calor general.

## Cómo

  1. `datos(desde, hasta, camara)` lee TODO de disco, en solo lectura: la
     base de tienda, el registro de seguimientos, los anaqueles (totales y
     `anaqueles_eventos`), las alertas de stock y los mapas de calor.
  2. `html(datos)` monta una página autocontenida (imágenes en base64).
  3. `a_pdf(html)` la imprime con Edge/Chrome sin ventana. Sin navegador
     (p. ej. el servidor de Linux) devuelve None y `informe_pdf.py` cae al
     informe clásico de reportlab.

Cada sección se arma aparte: si una falla, sale «sin datos» y el resto del
informe sigue. Las cifras salen de las mismas tablas que el dashboard.

Las fotos de stock medidas sobre la imagen gris de un corte de iVMS (hasta
el 13-oct-2026 el stock medía también durante los cortes) no se muestran, y
un agarre que solo confirmaron ellas no se cuenta. El informe lo dice en una
nota.
"""
from __future__ import annotations

import base64
import datetime as dt
import glob
import html as _html
import json
import logging
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

SERVIDOR = Path(__file__).resolve().parents[2]
_MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep',
          'oct', 'nov', 'dic']
_ORDEN_EDAD = ['Nino', 'Niño', '0-12', '13-17', '18-25', 'Joven', '26-35',
               '36-50', 'Adulto', '51-65', 'Mayor', '65+', 'Desconocido']
_UMBRAL_PERSONAL = 15          # entradas en el periodo: «personal probable»
_EDAD_BONITA = {'Nino': 'Niño', 'Desconocido': 'desconocida'}


# ── Formato ──────────────────────────────────────────────────────────────
def _dia(fecha: dt.date) -> str:
    return f'{fecha.day}-{_MESES[fecha.month - 1]}'


def _h12(hora: int, minuto: Optional[int] = None) -> str:
    sufijo = 'a. m.' if hora % 24 < 12 else 'p. m.'
    h = hora % 12 or 12
    return f'{h}:{minuto:02d} {sufijo}' if minuto is not None else f'{h} {sufijo}'


def _hora_de(t: float) -> str:
    lt = time.localtime(t)
    return _h12(lt.tm_hour, lt.tm_min)


def _pct(a: float, b: float) -> str:
    return f'{100 * a / b:.1f} %'.replace('.', ',') if b else '—'


def _seg(s: Optional[float]) -> str:
    if s is None:
        return '—'
    if s >= 3600:
        return f'{int(s // 3600)} h {int(s % 3600 // 60)} min'
    if s >= 120:
        return f'{int(s // 60)} min'
    return f'{s:.1f} s'.replace('.', ',')


def _e(texto: Any) -> str:
    return _html.escape(str(texto))


def _png_b64(datos: bytes, tipo: str = 'jpeg') -> str:
    return f'data:image/{tipo};base64,' + base64.b64encode(datos).decode('ascii')


# ── Ventana ──────────────────────────────────────────────────────────────
def _fechas(desde: str, hasta: str) -> List[dt.date]:
    a = dt.date.fromisoformat(desde)
    b = dt.date.fromisoformat(hasta)
    if b < a:
        a, b = b, a
    # Tope de 62 días: los ÚLTIMOS (la cabecera dice el periodo de verdad).
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)][-62:]


def _limites(fechas: List[dt.date]) -> Tuple[float, float]:
    a = time.mktime(fechas[0].timetuple())
    b = time.mktime((fechas[-1] + dt.timedelta(days=1)).timetuple())
    return a, b


# ── Fuentes ──────────────────────────────────────────────────────────────
def _bd() -> sqlite3.Connection:
    from ..analityc.core import tienda_almacen
    ruta = Path(tienda_almacen.ruta_bd()).resolve()
    return sqlite3.connect(f'{ruta.as_uri()}?mode=ro', uri=True)


def _camara_principal(con, a: float, b: float) -> Optional[str]:
    fila = con.execute(
        "select device_id, count(*) n from tienda_capturas where ts between ? and ? "
        "group by device_id order by n desc limit 1", (a, b)).fetchone()
    return fila[0] if fila else None


def _seguimientos(cam: str, a: float, b: float, fechas: List[dt.date]) -> int:
    ruta = SERVIDOR / 'output' / f'analytics_log_{cam}.jsonl'
    if not ruta.is_file():
        return 0
    prefijos = tuple(f'"{f.isoformat()}T' for f in fechas)
    vistos = set()
    with open(ruta, encoding='utf-8', errors='replace') as fh:
        for linea in fh:
            if '"person_detected"' not in linea or not any(p in linea for p in prefijos):
                continue
            try:
                e = json.loads(linea)
                t = dt.datetime.fromisoformat(e['timestamp']).timestamp()
            except Exception:                               # noqa: BLE001
                continue
            if a <= t < b:
                vistos.add((e.get('data') or {}).get('track_id'))
    return len(vistos)


def _nombre_camara(cam: str, con) -> Tuple[str, str]:
    """('cámara 3', 'iVMS-4200 (3)')"""
    fila = con.execute("select camera from tienda_capturas where device_id=? "
                       "order by ts desc limit 1", (cam,)).fetchone()
    visible = fila[0] if fila else cam
    m = re.search(r'-c(\d+)-', cam)
    return (f'cámara {m.group(1)}' if m else visible), visible


def _genero_de_pid(pid: Optional[str], capturas: List[tuple]) -> Optional[str]:
    """Género de la ficha de esa persona (identidad o número), si lo tiene."""
    if not pid:
        return None
    if pid.startswith('prov:'):
        sufijo = f"_t{pid.split(':', 1)[1]}"
        gs = [c[2] for c in capturas if c[0].endswith(sufijo) and c[2] in ('Hombre', 'Mujer')]
    else:
        gs = [c[2] for c in capturas if c[4] == pid and c[2] in ('Hombre', 'Mujer')]
    return Counter(gs).most_common(1)[0][0] if gs else None


def _eventos_anaqueles(cam: str, fechas: List[dt.date], a: float, b: float
                       ) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """(eventos de visitas/agarres en la ventana, completos?). Sin archivo
    de eventos para algún día (anteriores al 13-oct-2026), se recupera lo que
    quede en el log del servidor (agarres y visitas, con su hora)."""
    from ..analityc.core import anaqueles_eventos
    salida: List[Dict[str, Any]] = []
    completos = True
    sin_archivo = []
    for f in fechas:
        ev = anaqueles_eventos.leer(cam, f.isoformat())
        if ev is None:
            sin_archivo.append(f.isoformat())
            continue
        salida += [e for e in ev if a <= float(e.get('t') or 0) < b]
        # El día en que empezó a guardarse (13-oct-2026): lo de antes del
        # primer evento guardado, del log.
        primero = min((float(e.get('t') or 0) for e in ev), default=None)
        if primero is not None:
            previos = [e for e in _eventos_del_log(cam, [f.isoformat()], a, b)
                       if e['t'] < primero - 30]
            if previos:
                completos = False
                salida += previos
    if sin_archivo:
        completos = False
        salida += _eventos_del_log(cam, sin_archivo, a, b)
    return sorted(salida, key=lambda e: e.get('t') or 0), completos


def _eventos_del_log(cam: str, fechas: List[str], a: float, b: float) -> List[Dict[str, Any]]:
    salida = []
    patron_ag = re.compile(r'AGARRE #\d+ en (.+?)(?: → (.+?))? \((\w+)\)')
    patron_vi = re.compile(r'visita #\d+ a (.+?) \((\w+)\)')
    for ruta in glob.glob(str(SERVIDOR / 'logs' / 'server.log*')):
        try:
            with open(ruta, encoding='utf-8', errors='replace') as fh:
                for linea in fh:
                    if linea[:10] not in fechas or f'anaqueles[{cam}]' not in linea:
                        continue
                    try:
                        t = time.mktime(time.strptime(linea[:19], '%Y-%m-%d %H:%M:%S'))
                    except ValueError:
                        continue
                    if not a <= t < b:
                        continue
                    m = patron_ag.search(linea)
                    if m:
                        salida.append({'t': t, 'tipo': 'agarre', 'area_nombre': m.group(1),
                                       'producto': m.group(2), 'pid_corto': m.group(3)})
                        continue
                    m = patron_vi.search(linea)
                    if m:
                        salida.append({'t': t, 'tipo': 'visita', 'area_nombre': m.group(1),
                                       'pid_corto': m.group(2)})
        except OSError:
            continue
    return salida


# ── Datos del informe ────────────────────────────────────────────────────
def datos(desde: str, hasta: str, camara: Optional[str] = None) -> Dict[str, Any]:
    # Las fichas se indexan en la base solo cuando alguien abre la galería
    # del dashboard (`tienda_capturas.sincronizar`): se ponen al día antes de
    # leerlas, como hace la galería. Barato: solo parsea lo nuevo.
    try:
        from ..analityc.core import tienda_capturas
        tienda_capturas.sincronizar()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("informe del pasillo: índice de capturas sin poner al día (%s)", exc)
    con = _bd()
    try:
        return _datos(con, desde, hasta, camara)
    finally:
        con.close()


def _datos(con, desde: str, hasta: str, camara: Optional[str]) -> Dict[str, Any]:
    fechas = _fechas(desde, hasta)
    a, b = _limites(fechas)
    cam = camara or _camara_principal(con, a, b)
    d: Dict[str, Any] = {'desde': fechas[0], 'hasta': fechas[-1], 'camara': cam,
                         'generado': time.strftime('%Y-%m-%d %H:%M'), 'errores': []}
    if not cam:
        d['vacio'] = True
        return d
    d['camara_corta'], d['camara_visible'] = _nombre_camara(cam, con)
    caps = con.execute(
        "select stem, ts, genero, edad, person_uuid, coalesce(no_es_persona, 0) "
        "from tienda_capturas where device_id=? and ts >= ? and ts < ? order by ts",
        (cam, a, b)).fetchall()
    caps = [c for c in caps if not c[5]]
    d['capturas'] = caps
    clas = dict(con.execute(
        "select stem, solo_cuerpo from tienda_clasificaciones where camera_id=? "
        "and ts >= ? and ts < ?", (cam, a - 600, b + 600)).fetchall())
    eventos = con.execute(
        "select ts, persistent_id, tipo, permanencia_s from tienda_eventos "
        "where camera_id=? and ts >= ? and ts < ? order by ts", (cam, a, b)).fetchall()
    d['vacio'] = not caps and not eventos

    # Horas con datos (para el periodo y la afluencia)
    horas = sorted({(time.localtime(c[1]).tm_year, time.localtime(c[1]).tm_mon,
                     time.localtime(c[1]).tm_mday, time.localtime(c[1]).tm_hour) for c in caps}
                   | {(time.localtime(e[0]).tm_year, time.localtime(e[0]).tm_mon,
                       time.localtime(e[0]).tm_mday, time.localtime(e[0]).tm_hour) for e in eventos})
    por_dia: Dict[dt.date, List[int]] = defaultdict(list)
    for y, m, dd, h in horas:
        por_dia[dt.date(y, m, dd)].append(h)
    d['tramos'] = [(f, min(hs), max(hs) + 1) for f, hs in sorted(por_dia.items())]

    # Resumen
    ident = [c for c in caps if c[4]]
    clasif = [c for c in caps if c[2] in ('Hombre', 'Mujer')]
    salidas = [e for e in eventos if e[2] == 'salida' and e[3] is not None]
    estancias = [float(e[3]) for e in salidas]
    por_persona: Dict[str, float] = defaultdict(float)
    for e in salidas:
        por_persona[e[1]] += float(e[3])
    entradas = [e for e in eventos if e[2] == 'entrada']
    d['resumen'] = {
        'capturadas': len(caps), 'identificadas': len(ident),
        'personas_ident': len({c[4] for c in ident}), 'clasificadas': len(clasif),
        'perm_mediana': statistics.median(estancias) if estancias else None,
        'perm_persona': statistics.median(por_persona.values()) if por_persona else None,
    }
    try:
        d['seguimientos'] = _seguimientos(cam, a, b, fechas)
    except Exception as exc:                                # noqa: BLE001
        d['seguimientos'] = None
        d['errores'].append(f'seguimientos: {exc}')
    d['trafico'] = {
        'entraron': len({e[1] for e in entradas}),
        'diez_s': sum(1 for v in por_persona.values() if v >= 10),
        'entradas': len(entradas),
    }

    # Demografía
    edades = defaultdict(Counter)
    for c in clasif:
        edades[c[2]][c[3] or 'Desconocido'] += 1
    d['demo'] = {
        'hombres': sum(1 for c in clasif if c[2] == 'Hombre'),
        'mujeres': sum(1 for c in clasif if c[2] == 'Mujer'),
        'edades': edades,
        'edad_total': sum((edades[g] for g in edades), Counter()),
        'solo_cuerpo': sum(1 for c in clasif if clas.get(c[0]) == 1),
        'con_cara': sum(1 for c in clasif if clas.get(c[0]) == 0),
        'con_id': sum(1 for c in clasif if c[4]),
    }

    # Afluencia por hora
    caps_h = Counter((dt.date.fromtimestamp(c[1]), time.localtime(c[1]).tm_hour) for c in caps)
    ent_h = Counter((dt.date.fromtimestamp(e[0]), time.localtime(e[0]).tm_hour) for e in entradas)
    d['afluencia'] = sorted(set(caps_h) | set(ent_h))
    d['caps_h'], d['ent_h'] = caps_h, ent_h
    # Personal probable: muchas entradas de la misma identidad
    cuenta = Counter(e[1] for e in entradas if e[1])
    d['personal'] = []
    for pid, n in cuenta.most_common(3):
        if n < _UMBRAL_PERSONAL:
            break
        ts = [e[0] for e in entradas if e[1] == pid]
        hora = Counter(time.localtime(t).tm_hour for t in ts).most_common(1)[0][0]
        d['personal'].append({'n': n, 'desde': min(ts), 'hasta': max(ts), 'hora': hora})

    # Las fotos de stock, antes que los anaqueles: un agarre que solo
    # confirmaron bajadas medidas durante un corte de iVMS no cuenta.
    try:
        fotos = _fotos_stock(cam, a, b)
    except Exception as exc:                                # noqa: BLE001
        fotos = []
        d['errores'].append(f'stock: {exc}')
    # Anaqueles
    try:
        d['anaqueles'] = _datos_anaqueles(cam, fechas, a, b, caps, fotos)
    except Exception as exc:                                # noqa: BLE001
        d['anaqueles'] = None
        d['errores'].append(f'anaqueles: {exc}')
    # Stock
    try:
        d['stock'], d['stock_en_corte'] = _datos_stock(fotos, d.get('anaqueles'))
    except Exception as exc:                                # noqa: BLE001
        d['stock'], d['stock_en_corte'] = [], 0
        d['errores'].append(f'stock: {exc}')
    # Permanencia
    d['perm'] = {
        'n': len(estancias),
        'mediana': statistics.median(estancias) if estancias else None,
        'p90': sorted(estancias)[int(0.9 * (len(estancias) - 1))] if estancias else None,
        'max': max(estancias) if estancias else None,
        'cortas': (100 * sum(1 for x in estancias if x <= 5) / len(estancias)) if estancias else None,
        'personas': len(por_persona),
        'persona_mediana': statistics.median(por_persona.values()) if por_persona else None,
        'total': sum(por_persona.values()),
    }
    # Mapa de calor general: la hora con más entradas (o capturas)
    try:
        d['mapa_general'] = _mapa_general(cam, ent_h or caps_h)
    except Exception as exc:                                # noqa: BLE001
        d['mapa_general'] = None
        d['errores'].append(f'mapa general: {exc}')
    try:
        d['personal_camiseta'] = _datos_personal(cam, fechas, a, b)
    except Exception as exc:                                # noqa: BLE001
        d['personal_camiseta'] = None
        d['errores'].append(f'personal de camiseta: {exc}')
    return d


def _datos_personal(cam: str, fechas, a: float, b: float) -> Optional[Dict[str, Any]]:
    """Permanencia y mapa PROPIOS del personal de camiseta (19-oct-2026,
    `personal_mapa.py`): por hora, segundos con algún empleado en el área y
    segundos-persona. Solo las horas que empiezan dentro del periodo."""
    from ..analityc.core.anaqueles_registro import _carpeta_nombre
    from ..analityc.core import personal_mapa
    carpeta = SERVIDOR / 'output' / 'personal_permanencia' / _carpeta_nombre(cam)
    horas: List[Tuple[dt.date, int, float, float]] = []
    for f in fechas:
        try:
            datos_dia = json.loads((carpeta / f'{f.isoformat()}.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for h, v in sorted(datos_dia.get('horas', {}).items()):
            inicio = time.mktime(dt.datetime(f.year, f.month, f.day, int(h)).timetuple())
            if a <= inicio < b and v.get('con_personal_s'):
                horas.append((f, int(h), float(v['con_personal_s']), float(v.get('persona_s', 0))))
    if not horas:
        return None
    por_hora = Counter({(f, h): con for f, h, con, _p in horas})
    return {'horas': horas,
            'con_personal_s': sum(x[2] for x in horas),
            'persona_s': sum(x[3] for x in horas),
            'mapa': _mapa_general(personal_mapa.clave_mapa(cam), por_hora)}


def _datos_anaqueles(cam: str, fechas, a: float, b: float, caps,
                     fotos: Iterable[Dict[str, Any]] = ()) -> Optional[Dict[str, Any]]:
    from ..analityc.core.anaqueles_registro import ultimo_registro
    registros = [(f, ultimo_registro(cam, f.isoformat())) for f in fechas]
    registros = [(f, r) for f, r in registros if r and r.get('areas')]
    if not registros:
        return None
    nombres: Dict[int, str] = {}
    productos: Dict[int, Dict[int, str]] = defaultdict(dict)
    totales: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    for _f, r in registros:
        for k, ar in r['areas'].items():
            i = int(k)
            nombres[i] = ar.get('nombre') or f'Área {i + 1}'
            for pk, p in (ar.get('productos') or {}).items():
                productos[i][int(pk)] = (p or {}).get('nombre') or f'Producto {int(pk) + 1}'
            totales[i][0] += int(ar.get('visitas') or 0)
            totales[i][1] += int(ar.get('agarres') or 0)
    eventos, completos = _eventos_anaqueles(cam, fechas, a, b)
    por_area = {i: [0, 0] for i in nombres}
    agarres = []
    en_corte = 0
    a_mano = 0
    fotos = list(fotos)
    from . import agarres_descartados
    descartes = agarres_descartados.leer(cam)
    for e in eventos:
        i = e.get('area')
        if i is None:
            i = next((k for k, n in nombres.items() if n == e.get('area_nombre')), None)
        if i is None:
            continue
        e['area'] = i
        if e.get('tipo') == 'visita':
            por_area[i][0] += 1
        elif e.get('tipo') == 'agarre':
            if _confirmado_en_corte(e['t'], i, fotos):
                en_corte += 1
                continue
            if agarres_descartados.descartado(e['t'], i, descartes)[0]:
                a_mano += 1
                continue
            por_area[i][1] += 1
            genero = _genero_de_pid(e.get('pid'), caps)
            if genero is None and e.get('pid_corto'):
                corto = e['pid_corto']
                # Cifras = número de seguimiento (la ficha acaba en _t<número>);
                # si no, final de la identidad.
                genero = next((c[2] for c in caps if c[2] in ('Hombre', 'Mujer') and (
                    c[0].endswith(corto) if corto.isdigit()
                    else bool(c[4]) and c[4].endswith(corto))), None)
            if genero is None and e.get('genero') in ('Hombre', 'Mujer'):
                genero = e['genero']
            agarres.append({'t': e['t'], 'area': i, 'producto': e.get('producto'),
                            'persona': e.get('pid') or e.get('pid_corto'), 'genero': genero})
    if not completos and not eventos:
        por_area = {i: list(totales[i]) for i in nombres}      # sin horas: totales del día
    return {'nombres': nombres, 'productos': productos, 'por_area': por_area,
            'agarres': agarres, 'completos': completos, 'en_corte': en_corte,
            'a_mano': a_mano,
            'ultimo_registro': registros[-1]}


_FOTO = re.compile(r'^(\d{4}-\d{2}-\d{2})_(\d{6})_(\d+)_(\d+)(_cerca|_bajada)?\.jpg$')


def _separadores(img) -> List[int]:
    """Centro de cada franja que separa los paneles de la foto de una alerta
    (ANTES | AHORA | QUIÉN AGARRÓ): 8 px del color del marco (30, 30, 30).
    Con JPEG el marco no queda exacto: columna oscura (media ~30) y lisa."""
    import numpy as np
    h, w = img.shape[:2]
    zona = img[32:max(33, h - 34)].astype(np.int16)
    marco = (np.abs(zona.mean(axis=(0, 2)) - 30) < 12) & (zona.std(axis=(0, 2)) < 12)
    columnas = np.where(marco)[0]
    if not len(columnas):
        return []
    grupos = np.split(columnas, np.where(np.diff(columnas) > 1)[0] + 1)
    return [int(g.mean()) for g in grupos if 4 <= len(g) <= 14 and 20 < g.mean() < w - 20]


def _limites_paneles(img) -> Optional[Tuple[int, int]]:
    """(fin del ANTES, fin del AHORA) de la foto de una alerta.

    ANTES y AHORA son el MISMO recuadro ampliado al mismo alto
    (`anaqueles_stock.componer_evidencia`): miden lo mismo, así que el fin del
    AHORA cae al doble que el del ANTES (o en el borde, si no hay persona).
    Se busca ese par entre las franjas oscuras. 15-oct-2026: en los botes
    oscuros de Nescafé los huecos entre botes también parecían franjas
    (10:41:07: 79, 153, 228, 334, 645, 669) y se cortaba en la segunda, 153:
    el PDF enseñaba solo un trozo del ANTES. El par bueno era 334 y 669."""
    w = img.shape[1]
    cortes = _separadores(img)
    mejor = None
    for c1 in cortes:
        for c2 in cortes + [w]:
            error = abs(c2 - 2 * c1)
            if c2 > c1 and error <= 4 and (
                    mejor is None or (error, -c1) < (mejor[0], -mejor[1])):
                mejor = (error, c1, c2)
    return (mejor[1], mejor[2]) if mejor else None


def _antes_ahora(ruta: Path) -> Optional[bytes]:
    """Recorta la foto de la alerta a «ANTES | AHORA» (sin el panel del
    cliente)."""
    import cv2
    img = cv2.imread(str(ruta))
    if img is None:
        return None
    limites = _limites_paneles(img)
    if limites:
        img = img[:, :limites[1]]
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


#: Un panel con la imagen gris de un corte de iVMS es liso y de ese gris.
#: Medido el 13-oct-2026 en las 18 fotos de stock guardadas: el AHORA de las
#: dos bajadas falsas de las 08:11 da desviación 0,1 y 7,7 y media 77,0-77,6;
#: todos los demás paneles, desviación 41-68. Pide las dos cosas: una balda
#: vacía y lisa de otro color no se toma por un corte.
_GRIS_IVMS = 78.0
_GRIS_MARGEN = 12.0
_LISO_PANEL = 20.0


def _medida_en_corte(ruta: Path) -> bool:
    """¿El ANTES o el AHORA de la foto es la imagen gris de un corte de
    iVMS? Entonces la medida no vale: hasta el 13-oct-2026 el stock medía
    también durante los cortes (dos bajadas falsas a las 08:11 de ese día)."""
    import cv2
    img = cv2.imread(str(ruta))
    if img is None:
        return False
    h, w = img.shape[:2]
    limites = _limites_paneles(img)
    bordes = [0, *limites] if limites else [0] + _separadores(img) + [w]
    for x0, x1 in list(zip(bordes[:-1], bordes[1:]))[:2]:          # ANTES y AHORA
        panel = img[32:max(33, h - 34), x0 + 6:max(x0 + 7, x1 - 6)]
        gris = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        if gris.std() < _LISO_PANEL and abs(gris.mean() - _GRIS_IVMS) < _GRIS_MARGEN:
            return True
    return False


def _fotos_stock(cam: str, a: float, b: float) -> List[Dict[str, Any]]:
    """Las fotos de stock de la ventana, de la más vieja a la más nueva, con
    `en_corte` si se midieron sobre la imagen gris de un corte."""
    from ..analityc.core.anaqueles_registro import _carpeta_nombre, raiz_salida
    carpeta = raiz_salida() / _carpeta_nombre(cam) / 'stock_alertas'
    if not carpeta.is_dir():
        return []
    salida = []
    for f in sorted(carpeta.glob('*.jpg')):
        m = _FOTO.match(f.name)
        if not m:
            continue
        t = time.mktime(time.strptime(f'{m.group(1)} {m.group(2)}', '%Y-%m-%d %H%M%S'))
        if not a <= t < b:
            continue
        meta = {}
        try:
            meta = json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
        salida.append({'ruta': f, 't': t, 'tipo': (m.group(5) or '_agotado')[1:],
                       'area': int(m.group(3)), 'pos': int(m.group(4)), 'meta': meta,
                       'en_corte': _medida_en_corte(f)})
    return salida


def _confirmado_en_corte(t: float, area: int, fotos: List[Dict[str, Any]]) -> bool:
    """¿Al agarre solo lo confirmaron bajadas medidas durante un corte? El
    agarre se cuenta en el mismo segundo que la bajada que lo confirma (log
    del 13-oct-2026: bajada a las 08:11:25, agarre a las 08:11:26): ±3 s."""
    cerca = [f for f in fotos if f['tipo'] == 'bajada' and f['area'] == area
             and abs(f['t'] - float(t)) <= 3.0]
    return bool(cerca) and all(f['en_corte'] for f in cerca)


def _datos_stock(fotos: List[Dict[str, Any]], anaq) -> Tuple[List[Dict[str, Any]], int]:
    """(las 6 últimas fotos que valen, cuántas se quitan por ser de un corte)."""
    salida = []
    for s in fotos:
        if s['en_corte']:
            continue
        area, pos, meta = s['area'], s['pos'], s['meta']
        nombre_area = (anaq or {}).get('nombres', {}).get(area, f'Área {area + 1}')
        producto = meta.get('producto') or (anaq or {}).get('productos', {}).get(area, {}).get(pos) or f'Producto {pos + 1}'
        imagen = _antes_ahora(s['ruta'])
        if imagen:
            salida.append({'t': s['t'], 'tipo': s['tipo'], 'area': nombre_area, 'producto': producto,
                           'cambio': meta.get('cambio'), 'bajadas': meta.get('bajadas'),
                           'imagen': imagen})
    return salida[-6:], sum(1 for s in fotos if s['en_corte'])


_COLORES_PROD = [(220, 90, 50), (40, 160, 240), (60, 170, 60), (70, 60, 220),
                 (200, 70, 200), (40, 170, 200), (150, 150, 40)]


def _mapa_area(cam: str, idx: int, registro: Dict[str, Any], nombres) -> Optional[bytes]:
    import cv2
    import numpy as np

    from ..analityc.core import anaqueles_eventos
    from ..analityc.core.analytics.heatmap import render_calor_sobre
    from ..analityc.core.anaqueles_registro import ref_ruta
    ruta = ref_ruta(cam, idx)
    area = (registro.get('areas') or {}).get(str(idx))
    if ruta is None or area is None:
        return None
    img = cv2.imread(str(ruta))
    if img is None or img.size == 0:
        return None
    escala = 1.0
    if img.shape[1] < 700:
        escala = 700.0 / img.shape[1]
        img = cv2.resize(img, (700, max(1, int(img.shape[0] * escala))), interpolation=cv2.INTER_CUBIC)
    grid = np.asarray(area.get('grid') or [], dtype=np.float32)
    sub = (f"{int(area.get('agarres') or 0)} agarres - {int(area.get('visitas') or 0)} visitas - "
           f"{registro.get('fecha')}")
    titulo = (nombres.get(idx) or f'Area {idx + 1}').encode('ascii', 'replace').decode('ascii')
    out = render_calor_sobre(img, grid, alpha_max=0.6, con_leyenda=True, titulo=titulo, subtitulo=sub)
    diseno = anaqueles_eventos.diseno(cam).get(str(idx)) or {}
    for k, p in enumerate(diseno.get('productos') or []):
        try:
            poli = (np.asarray(p['poligono'], dtype=float) * escala).astype(np.int32)
            color = _COLORES_PROD[k % len(_COLORES_PROD)]
            cv2.polylines(out, [poli], True, color, 2, cv2.LINE_AA)
            x, y = int(poli[:, 0].min()), int(poli[:, 1].min())
            texto = f"Producto {int(p.get('pos', k)) + 1}"
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw + 6, y), color, -1)
            cv2.putText(out, texto, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:                                   # noqa: BLE001
            continue
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


def _mapa_general(cam: str, por_hora: Counter) -> Optional[Dict[str, Any]]:
    """El mapa de la hora con más movimiento. Se salta las horas pintadas
    sobre la imagen gris de un corte de iVMS (hasta el 13-oct-2026 el fondo
    del mapa podía quedarse en gris: p. ej. el de las 18h del 12-oct)."""
    from ..analityc.core import cortes_video
    from ..analityc.core.analytics.config import AnalyticsConfig
    raiz = Path(AnalyticsConfig.HEATMAP_DIR)
    if not raiz.is_absolute():
        raiz = SERVIDOR / raiz
    for (fecha, hora), _n in por_hora.most_common():
        ruta = raiz / 'history' / cam / f'{fecha.isoformat()}_{hora:02d}.png'
        if ruta.is_file():
            import cv2
            img = cv2.imread(str(ruta))
            if img is None or cortes_video.sin_video(img):
                continue
            if img.shape[1] > 1000:
                img = cv2.resize(img, (1000, int(img.shape[0] * 1000 / img.shape[1])))
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return {'fecha': fecha, 'hora': hora, 'imagen': buf.tobytes()}
    return None


# ── HTML ─────────────────────────────────────────────────────────────────
_CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Segoe UI', Arial, sans-serif; color: #1f2937; font-size: 11.5px;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; background: #fff; }
.cab { background: linear-gradient(90deg, #0d1b45 0%, #1d3468 100%); color: #fff;
       padding: 30px 40px 26px; position: relative; }
.cab h1 { margin: 0 0 14px; font-size: 27px; font-weight: 700; }
.cab p { margin: 3px 0; font-size: 12.5px; color: #dfe6f5; }
.cab b { color: #fff; }
.pill { position: absolute; top: 30px; right: 40px; border: 1px solid rgba(255,255,255,.75);
        border-radius: 999px; padding: 5px 14px; font-size: 11.5px; color: #fff; }
.cuerpo { padding: 0 40px 24px; }
.aviso { background: #e9effb; border-left: 4px solid #3e6fd9; padding: 10px 16px; margin: 16px 0 6px; font-size: 12px; }
h2 { font-size: 16.5px; margin: 22px 0 12px; padding-bottom: 7px; border-bottom: 1.5px solid #d6dae3; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tarjeta { border: 1px solid #dfe3ec; border-radius: 10px; background: #f7f8fb; padding: 13px 15px; break-inside: avoid; }
.blanca { background: #fff; }
.kpi .et { font-size: 10px; letter-spacing: .06em; text-transform: uppercase; color: #4b5563; }
.kpi .num { font-size: 28px; font-weight: 700; color: #111827; margin: 6px 0 8px; }
.kpi .num small { font-size: 13px; color: #3e6fd9; margin-left: 2px; }
.kpi .txt { font-size: 10.5px; color: #4b5563; line-height: 1.35; }
.tarjeta h3 { font-size: 13.5px; margin: 2px 0 12px; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { background: #eef0f5; text-align: left; padding: 8px 10px; font-weight: 700; }
td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }
.der { text-align: right; }
tr.tot td { background: #eef0f5; font-weight: 700; }
tr.grupo td { background: #e9effb; font-weight: 700; }
.nota { font-size: 10.5px; color: #4b5563; margin-top: 10px; line-height: 1.4; }
.caja { background: #e9effb; border-radius: 8px; padding: 10px 13px; margin-top: 10px; font-size: 10.5px; line-height: 1.4; }
.dos { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
.anillo { width: 124px; height: 124px; border-radius: 50%; position: relative; flex: none; }
.anillo::after { content: ''; position: absolute; inset: 28px; background: #fff; border-radius: 50%; }
.fila-anillo { display: flex; align-items: center; gap: 22px; margin-bottom: 14px; }
.ley { font-size: 11.5px; } .ley div { margin: 6px 0; display: flex; align-items: center; gap: 8px; }
.cuad { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
.barras .b { display: grid; grid-template-columns: 74px 1fr 34px; align-items: center; gap: 8px; margin: 7px 0; font-size: 11px; }
.barras .et2 { text-align: right; color: #374151; }
.anchas .b { grid-template-columns: 150px 1fr 34px; }
.pista { background: #eef1f6; border-radius: 6px; height: 14px; overflow: hidden; }
.relleno { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #3e6fd9, #5b8cf0); }
.naranja { background: linear-gradient(90deg, #f0a020, #f4b64d); }
.barras .n { font-weight: 700; }
.mini { display: inline-block; height: 12px; border-radius: 6px; background: linear-gradient(90deg, #3e6fd9, #5b8cf0); vertical-align: middle; margin-right: 10px; }
.apilada { display: flex; height: 18px; border-radius: 6px; overflow: hidden; font-size: 10px; color: #fff; font-weight: 700; }
.apilada div { display: flex; align-items: center; justify-content: center; }
.salto { break-before: page; padding-top: 24px; }
img.foto { display: block; margin: 0 auto; max-width: 100%; border-radius: 6px; }
.pie { display: flex; justify-content: space-between; color: #6b7280; font-size: 10px; border-top: 1px solid #d6dae3; margin-top: 26px; padding-top: 10px; }
.vacio { color: #6b7280; font-style: italic; }
"""


def _barra(et: str, n: int, maximo: int, clase: str = '') -> str:
    ancho = 100 * n / maximo if maximo else 0
    return (f'<div class="b"><div class="et2">{_e(et)}</div><div class="pista">'
            f'<div class="relleno {clase}" style="width:{ancho:.1f}%"></div></div>'
            f'<div class="n">{n}</div></div>')


def _periodo_texto(d) -> Tuple[str, str]:
    tramos = d.get('tramos') or []
    if not tramos:
        f0, f1 = d['desde'], d['hasta']
        t = _dia(f0) if f0 == f1 else f'{_dia(f0)} – {_dia(f1)}'
        return t, t
    cab = ' y '.join(f'<b>{_dia(f)}, {_h12(h0)} – {_h12(h1)}</b>' for f, h0, h1 in tramos)
    aviso = ' y '.join(f'de <b>{_h12(h0, 0)} a {_h12(h1, 0)}</b> del {_dia(f)}' for f, h0, h1 in tramos)
    return cab, aviso


def html(d: Dict[str, Any]) -> str:
    rango = (_dia(d['desde']) if d['desde'] == d['hasta']
             else f"{_dia(d['desde'])} al {_dia(d['hasta'])}")
    cab, aviso = _periodo_texto(d) if not d.get('vacio') else (rango, rango)
    modo = 'modo Anaqueles' if d.get('anaqueles') else 'modo Tiendas'
    cam = d.get('camara_corta') or 'sin cámara'
    partes = [f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe del pasillo {d['desde']}</title><style>{_CSS}</style></head><body>
<div class="cab"><h1>Informe del pasillo</h1>
<p>Periodo: {cab} (hora de Venezuela)</p>
<p>Fecha del informe: <b>{_e(d['hasta'].isoformat())}</b></p>
<div class="pill">Pasillo · {_e(cam)} · {modo}</div></div>
<div class="cuerpo"><div class="aviso">Pasillo, {_e(cam)}, {aviso}.</div>"""]
    if d.get('vacio'):
        partes.append('<p class="vacio">No hay datos registrados en ese periodo: o ninguna '
                      'cámara de tienda transmitía, o el servidor estuvo apagado. No es un '
                      'periodo sin gente: es un periodo sin medir.</p></div></body></html>')
        return ''.join(partes)
    for seccion in (_sec_resumen, _sec_trafico, _sec_demografia, _sec_afluencia,
                    _sec_anaqueles, _sec_stock, _sec_mapas_area, _sec_permanencia,
                    _sec_mapa_general, _sec_personal):
        try:
            partes.append(seccion(d))
        except Exception as exc:                            # noqa: BLE001
            logger.warning("informe del pasillo: sección %s sin datos: %s", seccion.__name__, exc)
            partes.append(f'<p class="vacio">Sin datos para esta sección ({_e(exc)}).</p>')
    horario = ' · '.join(f'{_dia(f)} {_h12(h0)} – {_h12(h1)}' for f, h0, h1 in d.get('tramos') or [])
    partes.append(f'<div class="pie"><span>Informe del pasillo — datos del {rango}</span>'
                  f'<span>Horario: {_e(horario)}</span></div></div></body></html>')
    return ''.join(partes)


def _sec_resumen(d) -> str:
    r = d['resumen']
    n = r['capturadas']
    return f"""<h2>Resumen del periodo</h2><div class="kpis">
<div class="tarjeta kpi"><div class="et">Personas capturadas</div><div class="num">{n} <small>100 %</small></div>
<div class="txt">Una ficha con foto por cada persona detectada y seguida. Quien sale de plano y vuelve sin ser reconocido cuenta dos veces.</div></div>
<div class="tarjeta kpi"><div class="et">Identificadas</div><div class="num">{r['identificadas']} <small>{_pct(r['identificadas'], n)}</small></div>
<div class="txt">Cara reconocida lo bastante bien para darle identidad: {r['personas_ident']} personas distintas.</div></div>
<div class="tarjeta kpi"><div class="et">Clasificadas</div><div class="num">{r['clasificadas']} <small>{_pct(r['clasificadas'], n)}</small></div>
<div class="txt">El sistema decidió su género y su edad aproximada.</div></div>
<div class="tarjeta kpi"><div class="et">Permanencia mediana</div><div class="num">{_seg(r['perm_mediana'])}</div>
<div class="txt">Por estancia en el área de conteo. Por persona, sumando estancias: {_seg(r['perm_persona'])}.</div></div></div>"""


def _sec_trafico(d) -> str:
    t = d['trafico']
    seg = d.get('seguimientos')
    return f"""<div class="tarjeta blanca" style="margin-top:14px"><h3>Tráfico</h3><table>
<tr><th>Medida</th><th class="der">Cantidad</th></tr>
<tr><td>Seguimientos</td><td class="der">{seg if seg is not None else '—'}</td></tr>
<tr><td>Personas capturadas</td><td class="der">{d['resumen']['capturadas']}</td></tr>
<tr><td>Entraron al área de conteo (personas reconocidas)</td><td class="der">{t['entraron']}</td></tr>
<tr><td>Se quedaron 10 s o más en el área (sumando todas sus estancias)</td><td class="der">{t['diez_s']}</td></tr></table>
<div class="nota">Los seguimientos inflan mucho: cada vez que alguien queda tapado o sale del plano empieza uno nuevo. No sirven como número de personas.</div></div>"""


def _sec_demografia(d) -> str:
    g = d['demo']
    h, m = g['hombres'], g['mujeres']
    total = h + m
    ph = 100 * h / total if total else 0
    anillo = (f'background: conic-gradient(#3e6fd9 0 {ph:.1f}%, #f0a020 {ph:.1f}% 100%)'
              if total else 'background:#eef1f6')
    filas = []
    for gen, etiqueta in (('Hombre', 'Hombres'), ('Mujer', 'Mujeres')):
        ed = g['edades'].get(gen) or Counter()
        texto = ', '.join(f'{_EDAD_BONITA.get(k, k).lower()} ×{v}' for k, v in ed.most_common())
        filas.append(f"<tr><td>{etiqueta}</td><td class='der'>{sum(ed.values())}</td><td>{_e(texto) or '—'}</td></tr>")
    et = g['edad_total']
    orden = [k for k in _ORDEN_EDAD if et.get(k)] + [k for k in et if k not in _ORDEN_EDAD]
    maximo = max(et.values()) if et else 0
    barras_edad = ''.join(_barra(_EDAD_BONITA.get(k, k), et[k], maximo) for k in orden) or '<p class="vacio">Sin edades.</p>'
    decididos = g['solo_cuerpo'] + g['con_cara']
    m2 = max(g['solo_cuerpo'], g['con_cara'], 1)
    return f"""<h2>Demografía</h2><div class="dos">
<div class="tarjeta blanca"><h3>Distribución de género</h3>
<div class="fila-anillo"><div class="anillo" style="{anillo}"></div>
<div class="ley"><div><span class="cuad" style="background:#3e6fd9"></span>Hombre — {h} ({_pct(h, total)})</div>
<div><span class="cuad" style="background:#f0a020"></span>Mujer — {m} ({_pct(m, total)})</div></div></div>
<table><tr><th>Género</th><th class="der">Personas</th><th>Edades</th></tr>{''.join(filas)}
<tr class="tot"><td>Total</td><td class="der">{total}</td><td></td></tr></table>
<div class="nota">Clasificadas: {g['con_id']} con identidad y {total - g['con_id']} sin identidad.</div></div>
<div class="tarjeta blanca"><h3>Rango de edad</h3><div class="barras">{barras_edad}</div>
<h3 style="margin-top:16px">Cómo se decidió el género</h3><div class="barras">
{_barra('Solo cuerpo', g['solo_cuerpo'], m2)}{_barra('Con la cara', g['con_cara'], m2, 'naranja')}</div>
<div class="caja">La rama corporal decidió el <b>{_pct(g['solo_cuerpo'], decididos)}</b> de los casos, sin ver la cara.</div></div></div>"""


def _sec_afluencia(d) -> str:
    claves = d['afluencia']
    if not claves:
        return '<h2 class="salto">Afluencia por hora</h2><p class="vacio">Sin datos por hora.</p>'
    maximo = max(d['caps_h'].values()) if d['caps_h'] else 1
    filas, dia_actual = [], None
    for f, hora in claves:
        if f != dia_actual:
            dia_actual = f
            filas.append(f'<tr class="grupo"><td colspan="3">{_dia(f)}</td></tr>')
        n = d['caps_h'].get((f, hora), 0)
        filas.append(f'<tr><td>{_h12(hora)}</td><td><span class="mini" style="width:{180 * n / maximo:.0f}px"></span>'
                     f'<b>{n}</b></td><td class="der">{d["ent_h"].get((f, hora), 0)}</td></tr>')
    tot_c, tot_e = sum(d['caps_h'].values()), sum(d['ent_h'].values())
    filas.append(f'<tr class="tot"><td>Total</td><td><b>{tot_c}</b></td><td class="der">{tot_e}</td></tr>')
    (fm, hm), _n = d['caps_h'].most_common(1)[0] if d['caps_h'] else ((d['desde'], 0), 0)
    nota = f'La hora con más gente fue las {_h12(hm)} del {_dia(fm)}.'
    for p in d['personal'][:1]:
        nota += (f' <b>Las entradas de las {_h12(p["hora"])} están infladas por personal probable:</b> una identidad '
                 f'entró {p["n"]} veces entre las {_hora_de(p["desde"])} y las {_hora_de(p["hasta"])} '
                 '(deducción por su comportamiento, no comprobado en imagen).')
    return f"""<h2 class="salto">Afluencia por hora</h2><div class="tarjeta blanca"><table>
<tr><th>Hora</th><th>Personas capturadas</th><th class="der">Entradas al área</th></tr>{''.join(filas)}</table>
<div class="caja">{nota}</div></div>"""


def _sec_anaqueles(d) -> str:
    an = d.get('anaqueles')
    if not an:
        return '<h2>Anaqueles</h2><p class="vacio">Sin datos de anaqueles en el periodo.</p>'
    nombres, prods = an['nombres'], an['productos']

    def desc(i):
        lista = [prods[i][k] for k in sorted(prods.get(i, {}))][:3]
        return f"{nombres[i]} · {', '.join(lista)}" if lista else nombres[i]
    maximo = max([v for v, _a in an['por_area'].values()] + [1])
    barras = ''.join(_barra(desc(i), an['por_area'][i][0], maximo) for i in sorted(an['por_area']))
    tabla = ''.join(f"<tr><td>{_e(desc(i))}</td><td class='der'>{an['por_area'][i][0]}</td>"
                    f"<td class='der'>{an['por_area'][i][1]}</td></tr>" for i in sorted(an['por_area']))
    alertas = [s for s in d.get('stock') or [] if s['tipo'] in ('agotado', 'cerca')]
    notas = ''
    if not an['completos']:
        notas += ('<div class="caja">Parte del periodo es anterior al registro de cada visita y agarre '
                  '(13-oct-2026): esos datos salen del log del servidor o de los totales del día.</div>')
    n = an.get('en_corte') or 0
    if n:
        notas += ('<div class="caja">' + (
            '1 agarre no se cuenta: lo confirmó una medida de stock tomada' if n == 1 else
            f'{n} agarres no se cuentan: los confirmaron medidas de stock tomadas')
            + ' mientras la cámara no daba vídeo (imagen gris de iVMS).</div>')
    n = an.get('a_mano') or 0
    if n:
        notas += ('<div class="caja">' + (
            '1 agarre no se cuenta: se revisó en imagen y era falso' if n == 1 else
            f'{n} agarres no se cuentan: se revisaron en imagen y eran falsos')
            + ' (empleado recolocando o reponiendo, producto tapado o imagen borrosa).</div>')
    for s in alertas[:2]:
        tipo = 'posible agotado' if s['tipo'] == 'agotado' else 'cerca de vaciarse'
        notas += (f'<div class="caja"><b>Aviso de «{tipo}»:</b> {_e(s["producto"])} a las '
                  f'{_hora_de(s["t"])} del {_dia(dt.date.fromtimestamp(s["t"]))} (sin verificar).</div>')
    ag = an['agarres']
    lista = ''.join(f"<tr><td>{_dia(dt.date.fromtimestamp(x['t']))}, {_hora_de(x['t'])}</td>"
                    f"<td class='der'>{_e(nombres.get(x['area'], x['area']).replace('Anaquel ', ''))}</td>"
                    f"<td>{_e(x['producto'] or 'sin producto')}</td></tr>" for x in ag) \
        or "<tr><td colspan='3' class='vacio'>Ningún agarre en el periodo.</td></tr>"
    # Por producto y género
    por_prod: Dict[str, Dict[str, Any]] = {}
    for x in ag:
        p = por_prod.setdefault(x['producto'] or 'sin producto',
                                {'n': 0, 'personas': set(), 'H': 0, 'M': 0, 'S': 0, 'area': x['area']})
        p['n'] += 1
        p['personas'].add(x['persona'])
        p['H' if x['genero'] == 'Hombre' else 'M' if x['genero'] == 'Mujer' else 'S'] += 1
    maxp = max([p['n'] for p in por_prod.values()] + [1])
    apiladas, filas_p = '', ''
    for nombre, p in sorted(por_prod.items(), key=lambda kv: -kv[1]['n']):
        segs = ''.join(f'<div style="width:{100 * p[k] / p["n"]:.1f}%;background:{c}">{p[k]}</div>'
                       for k, c in (('H', '#3e6fd9'), ('M', '#f0a020'), ('S', '#9aa3b2')) if p[k])
        apiladas += (f'<div class="b" style="grid-template-columns:120px 1fr 26px"><div>{_e(nombre)}<br>'
                     f'<small style="color:#6b7280">{_e(nombres.get(p["area"], ""))}</small></div>'
                     f'<div class="apilada" style="width:{100 * p["n"] / maxp:.0f}%">{segs}</div>'
                     f'<div class="n">{p["n"]}</div></div>')
        filas_p += (f"<tr><td>{_e(nombre)}</td><td class='der'>{p['n']}</td><td class='der'>{len(p['personas'])}</td>"
                    f"<td class='der'>{p['H']}</td><td class='der'>{p['M']}</td><td class='der'>{p['S']}</td></tr>")
    personas = len({x['persona'] for x in ag})
    nota_ag = (f'Los {len(ag)} agarres los hicieron {personas} personas. El género sale de la ficha de cada '
               'persona; «sin género» es que el sistema no llegó a decidirlo.') if ag else ''
    bloque_prod = '' if not ag else f"""<div class="tarjeta blanca" style="margin-top:14px"><h3>Agarres por producto y género</h3>
<div class="dos"><div class="barras">{apiladas}
<div class="ley" style="display:flex;gap:14px;margin-top:8px"><div><span class="cuad" style="background:#3e6fd9"></span>Hombres</div>
<div><span class="cuad" style="background:#f0a020"></span>Mujeres</div><div><span class="cuad" style="background:#9aa3b2"></span>Sin género</div></div></div>
<table><tr><th>Producto</th><th class="der">Agarres</th><th class="der">Personas</th><th class="der">H</th><th class="der">M</th><th class="der">Sin género</th></tr>{filas_p}</table></div>
<div class="caja">{nota_ag}</div></div>"""
    return f"""<h2>Anaqueles</h2><div class="dos">
<div class="tarjeta blanca"><h3>Visitas y agarres por área</h3><div class="barras anchas">{barras}</div>
<table><tr><th>Área</th><th class="der">Visitas</th><th class="der">Agarres</th></tr>{tabla}</table>{notas}</div>
<div class="tarjeta blanca"><h3>Agarres del periodo</h3><table><tr><th>Hora</th><th class="der">Anaquel</th><th>Producto asignado</th></tr>{lista}</table></div></div>
{bloque_prod}"""


def _sec_stock(d) -> str:
    st = d.get('stock') or []
    n = d.get('stock_en_corte') or 0
    nota = ('' if not n else '<p class="vacio">' + (
        'Se quitó 1 foto de stock tomada' if n == 1 else f'Se quitaron {n} fotos de stock tomadas')
        + ' mientras la cámara no daba vídeo (imagen gris de iVMS): no son bajadas reales.</p>')
    if not st:
        return ('<h2 class="salto">Bajadas de stock</h2><p class="vacio">Ninguna bajada ni alerta de stock '
                'en el periodo.</p>' + nota)
    piezas = []
    for s in st:
        texto = {'bajada': 'disminuyó el stock de este producto',
                 'cerca': f'cerca de vaciarse ({s.get("bajadas") or "?"} bajadas desde la última foto de balda surtida)',
                 'agotado': 'posible agotado: el hueco es casi total'}[s['tipo']]
        piezas.append(f'<div class="tarjeta blanca" style="margin-bottom:12px"><img class="foto" style="max-height:200px" '
                      f'src="{_png_b64(s["imagen"])}"><div class="caja"><b>{_e(s["producto"])} ({_e(s["area"])}), '
                      f'{_dia(dt.date.fromtimestamp(s["t"]))} a las {_hora_de(s["t"])}: {texto}.</b> '
                      'La foto compara el producto antes y después del agarre.</div></div>')
    return '<h2 class="salto">Bajadas de stock</h2>' + ''.join(piezas) + nota


def _sec_mapas_area(d) -> str:
    an = d.get('anaqueles')
    if not an:
        return ''
    fecha, registro = an['ultimo_registro']
    piezas = []
    for i in sorted(an['nombres']):
        imagen = _mapa_area(d['camara'], i, registro, an['nombres'])
        if imagen is None:
            continue
        prods = an['productos'].get(i, {})
        lista = ' · '.join(f'Producto {k + 1}: {_e(v)}' for k, v in sorted(prods.items()))
        a = (registro.get('areas') or {}).get(str(i)) or {}
        piezas.append(f'<div style="margin-bottom:14px"><img class="foto" style="max-height:230px" src="{_png_b64(imagen)}">'
                      f'<div class="nota"><b>{_e(an["nombres"][i])}</b> — captura acumulada del día {_dia(fecha)}'
                      f'{"" if (a.get("visitas") or a.get("agarres")) else "; ese día no tuvo visitas ni agarres"}.'
                      f'{" " + lista if lista else ""}</div></div>')
    if not piezas:
        return ''
    return '<h2>Mapas de calor por anaquel (dónde meten la mano)</h2><div class="tarjeta blanca">' + ''.join(piezas) + '</div>'


def _sec_permanencia(d) -> str:
    p = d['perm']
    if not p['n']:
        return '<h2>Permanencia en el área de conteo</h2><p class="vacio">Sin estancias registradas.</p>'
    return f"""<h2>Permanencia en el área de conteo</h2><div class="dos">
<div class="tarjeta blanca"><h3>Cada estancia ({p['n']} entradas y salidas)</h3><table>
<tr><th>Medida</th><th class="der">Tiempo</th></tr>
<tr><td>Mediana</td><td class="der"><b>{_seg(p['mediana'])}</b></td></tr>
<tr><td>9 de cada 10</td><td class="der">menos de {_seg(p['p90'])}</td></tr>
<tr><td>La más larga</td><td class="der">{_seg(p['max'])}</td></tr>
<tr><td>Duran 5 s o menos</td><td class="der">{p['cortas']:.0f} %</td></tr></table></div>
<div class="tarjeta blanca"><h3>Por persona ({p['personas']} personas)</h3><div class="dos">
<div class="tarjeta kpi"><div class="et">Mediana</div><div class="num">{_seg(p['persona_mediana'])}</div><div class="txt">Sumando todas sus estancias.</div></div>
<div class="tarjeta kpi"><div class="et">Entre todas</div><div class="num">{_seg(p['total'])}</div><div class="txt">Tiempo total dentro del área.</div></div></div></div></div>"""


def _sec_mapa_general(d) -> str:
    m = d.get('mapa_general')
    if not m:
        return ''
    return f"""<h2>Mapa de calor general</h2><div class="tarjeta blanca">
<img class="foto" style="max-height:260px" src="{_png_b64(m['imagen'])}">
<div class="nota"><b>Pasillo, {_e(d.get('camara_corta'))}</b> — mapa de calor de la hora de las {_h12(m['hora'])} del {_dia(m['fecha'])}
(azul: poca actividad · rojo: mucha). Suma la posición de los pies de cada persona en cada fotograma, así que quien
se queda parado calienta más que quien cruza.</div></div>"""


def _sec_personal(d) -> str:
    p = d.get('personal_camiseta')
    if not p:
        return ''
    filas = ''.join(f'<tr><td>{_dia(f)} · {_h12(h)}</td><td class="der">{_seg(con)}</td>'
                    f'<td class="der">{_seg(per)}</td></tr>' for f, h, con, per in p['horas'])
    mapa = ''
    if p.get('mapa'):
        m = p['mapa']
        mapa = (f'<div class="tarjeta blanca" style="margin-top:12px"><img class="foto" style="max-height:240px" '
                f'src="{_png_b64(m["imagen"])}"><div class="nota">Mapa de calor <b>del personal</b> de la hora '
                f'de las {_h12(m["hora"])} del {_dia(m["fecha"])}: dónde trabajó el personal de camiseta.</div></div>')
    return f"""<h2>Personal de tienda</h2><div class="tarjeta blanca"><table>
<tr><th>Hora</th><th class="der">Con personal en el área</th><th class="der">Tiempo por persona sumado</th></tr>
{filas}<tr class="tot"><td>Total</td><td class="der">{_seg(p['con_personal_s'])}</td><td class="der">{_seg(p['persona_s'])}</td></tr></table>
<div class="nota">Lo reconoce el sistema por la camiseta de Santa Rosa (aprox. 9 de cada 10 empleados) y queda
<b>fuera de todas las cifras de clientes</b> de este informe. «Con personal» no suma doble si hay dos a la vez;
«por persona» sí. El vigilante y mantenimiento tampoco cuentan como clientes, pero no se incluyen aquí.</div></div>{mapa}"""


# ── PDF ──────────────────────────────────────────────────────────────────
def navegador() -> Optional[str]:
    """Edge o Chrome para imprimir sin ventana (ELDE_NAVEGADOR_PDF manda)."""
    candidatos = [os.environ.get('ELDE_NAVEGADOR_PDF', ''),
                  r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                  r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
                  r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                  r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe']
    candidatos += [shutil.which(n) or '' for n in ('msedge', 'google-chrome', 'chromium', 'chromium-browser', 'chrome')]
    return next((c for c in candidatos if c and os.path.isfile(c)), None)


def a_pdf(pagina: str, tope_s: float = 90.0) -> Optional[bytes]:
    """La página impresa a PDF con el navegador, o None si no hay o falla."""
    exe = navegador()
    if exe is None:
        return None
    # `ignore_cleanup_errors`: en Windows el navegador puede tardar en soltar
    # su perfil temporal; que eso no tire un PDF ya hecho.
    with tempfile.TemporaryDirectory(prefix='informe_pasillo_',
                                     ignore_cleanup_errors=True) as carpeta:
        entrada = Path(carpeta) / 'informe.html'
        salida = Path(carpeta) / 'informe.pdf'
        entrada.write_text(pagina, encoding='utf-8')
        orden = [exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                 f'--user-data-dir={Path(carpeta) / "perfil"}', '--no-first-run',
                 '--virtual-time-budget=4000', f'--print-to-pdf={salida}', entrada.as_uri()]
        try:
            subprocess.run(orden, capture_output=True, timeout=tope_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("informe del pasillo: el navegador no imprimió (%s)", exc)
            return None
        if salida.is_file() and salida.stat().st_size > 1000:
            return salida.read_bytes()
        logger.warning("informe del pasillo: el navegador no dejó PDF")
        return None


def construir(desde: str, hasta: str, camara: Optional[str] = None) -> str:
    return html(datos(desde, hasta, camara))
