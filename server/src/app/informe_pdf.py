"""src/app/informe_pdf.py — informe PDF del día del dashboard de tienda.

## Qué es

Un documento AUTOCONTENIDO y detallado de lo que el sistema detectó en el
día: resumen ejecutivo, anaqueles (visitas/agarres/conversión con el calor
sobre la foto real del estante), demografía, mapas de calor por cámara, el
registro de personas detectadas, el personal enrolado y una sección de
metodología que explica cada término — para que el PDF se entienda sin
haber visto nunca el dashboard.

**Desde el 13-oct-2026 el botón «Exportar PDF» da el «Informe del pasillo»**
(`informe_pasillo.py`, el formato que diseñó el usuario), del periodo y la
cámara elegidos en el dashboard. Este informe clásico queda de reserva: sale
si no hay navegador para imprimir el nuevo (p. ej. el servidor de Linux) o
si falla, y con `formato=clasico`. `/api/informes/tienda.html` da la página
del nuevo, para verla o imprimirla a mano.

## Diseño

- **reportlab** (ya en el venv) para el documento; **matplotlib** (Agg)
  solo para las barras de demografía. Nada nuevo que instalar.
- Cada sección se arma en su propio try/except: una fuente caída (sin
  capturas, sin anaqueles ese día) degrada a «sin datos», JAMÁS tumba el
  informe entero.
- Los datos salen de las MISMAS funciones que alimentan el dashboard
  (api_lectura, anaqueles_registro, personal_galeria, heatmap render):
  el PDF nunca puede contradecir a la página.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/informes', tags=['informes'])

_AZUL = '#005f87'
_NARANJA = '#b35a00'
_GRIS = '#555555'


_FECHA = r'^\d{4}-\d{2}-\d{2}$'
#: `Annotated` y no `= Query(None)`: así el valor por defecto es None también
#: al llamar a la función directamente (las pruebas lo hacen).
_Fecha = Annotated[Optional[str], Query(pattern=_FECHA)]
_Camara = Annotated[Optional[str], Query(max_length=200)]


def _periodo(fecha: Optional[str], desde: Optional[str],
             hasta: Optional[str]) -> Tuple[str, str]:
    """(desde, hasta). Sin nada, la jornada en curso; `fecha` = un día.

    «Hoy» es la JORNADA en curso, no el día natural (5-sep-2026): con
    ELDE_TIENDA_CORTE_JORNADA_H=22, a la 01:30 la tienda sigue en la jornada
    de ayer, y el informe por defecto salía vacío (el del día natural, que
    aún no tiene datos)."""
    from src.analityc.core import tienda_almacen as _ta
    hasta = hasta or fecha or desde or _ta.jornada_de()
    desde = desde or fecha or hasta
    return (hasta, desde) if hasta < desde else (desde, hasta)


@router.get('/tienda.pdf')
def informe_tienda(fecha: _Fecha = None, desde: _Fecha = None,
                   hasta: _Fecha = None, camara: _Camara = None,
                   formato: Annotated[str, Query(pattern=r'^(pasillo|clasico)$')] = 'pasillo'):
    """El informe del periodo (hoy por defecto), listo para imprimir/compartir.

    El del pasillo (13-oct-2026); si no se puede, el clásico del último día
    del periodo, como antes (ver el docstring del módulo)."""
    desde, hasta = _periodo(fecha, desde, hasta)
    if formato == 'pasillo':
        pdf = _pdf_pasillo(desde, hasta, camara or None)
        if pdf is not None:
            nombre = (f'Informe_Pasillo_{desde}.pdf' if desde == hasta
                      else f'Informe_Pasillo_{desde}_a_{hasta}.pdf')
            return Response(content=pdf, media_type='application/pdf',
                            headers={'Content-Disposition':
                                     f'inline; filename="{nombre}"'})
    fecha = hasta
    buf = io.BytesIO()
    try:
        construir_informe_tienda(buf, fecha)
    except Exception as exc:                                # noqa: BLE001
        logger.error("informe: fallo generando el PDF: %s", exc,
                     exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f'no se pudo generar el informe: {exc}')
    return Response(
        content=buf.getvalue(), media_type='application/pdf',
        headers={'Content-Disposition':
                 f'inline; filename="informe_tienda_{fecha}.pdf"'})


@router.get('/tienda.html')
def informe_tienda_html(fecha: _Fecha = None, desde: _Fecha = None,
                        hasta: _Fecha = None, camara: _Camara = None):
    """El informe del pasillo como página (lo mismo que se imprime a PDF)."""
    from . import informe_pasillo
    desde, hasta = _periodo(fecha, desde, hasta)
    try:
        return HTMLResponse(informe_pasillo.construir(desde, hasta, camara or None))
    except Exception as exc:                                # noqa: BLE001
        logger.error("informe del pasillo: falló la página: %s", exc,
                     exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f'no se pudo generar el informe: {exc}')


def _pdf_pasillo(desde: str, hasta: str,
                 camara: Optional[str]) -> Optional[bytes]:
    """El informe del pasillo en PDF, o None si no se pudo (sin navegador
    para imprimirlo o un fallo): entonces sale el clásico."""
    try:
        from . import informe_pasillo
        t0 = time.time()
        pdf = informe_pasillo.a_pdf(
            informe_pasillo.construir(desde, hasta, camara))
    except Exception as exc:                                # noqa: BLE001
        logger.error("informe del pasillo: falló (%s); sale el clásico",
                     exc, exc_info=True)
        return None
    if pdf is None:
        logger.warning("informe del pasillo: no se pudo imprimir (¿sin "
                       "Edge/Chrome?); sale el clásico")
    else:
        logger.info("informe del pasillo %s a %s: %d KB en %.1f s", desde,
                    hasta, len(pdf) // 1024, time.time() - t0)
    return pdf


# ── Construcción ─────────────────────────────────────────────────────────
def construir_informe_tienda(destino, fecha: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer)

    base = getSampleStyleSheet()
    st = {
        'titulo': ParagraphStyle('t', parent=base['Title'], fontSize=22,
                                 textColor=colors.HexColor(_AZUL)),
        'h2': ParagraphStyle('h2', parent=base['Heading2'], fontSize=13,
                             textColor=colors.HexColor(_AZUL),
                             spaceBefore=14, spaceAfter=4),
        'h3': ParagraphStyle('h3', parent=base['Heading3'], fontSize=11,
                             spaceBefore=8, spaceAfter=2),
        'cuerpo': ParagraphStyle('c', parent=base['BodyText'], fontSize=9.5,
                                 leading=13),
        'nota': ParagraphStyle('n', parent=base['BodyText'], fontSize=8.5,
                               leading=11.5,
                               textColor=colors.HexColor(_GRIS)),
    }

    historia: List[Any] = []
    historia.append(Paragraph('Informe del día — Tienda', st['titulo']))
    historia.append(Paragraph(
        f"Fecha del informe: <b>{fecha}</b> &nbsp;·&nbsp; generado el "
        f"{time.strftime('%Y-%m-%d a las %H:%M')}", st['cuerpo']))
    historia.append(Paragraph(
        'Todas las cifras son SOLO de clientes: el personal enrolado '
        '(seguridad y pasilleros) está excluido de visitas, agarres, '
        'mapas de calor y demografía.', st['nota']))
    historia.append(Spacer(1, 8))

    secciones = (
        ('Resumen del día', _sec_resumen),
        ('Anaqueles — qué se visitó y qué se agarró', _sec_anaqueles),
        ('Quién visitó — demografía', _sec_demografia),
        ('Mapas de calor por cámara', _sec_mapas),
        ('Personas detectadas (registro del día)', _sec_visitantes),
        ('Personal de la tienda', _sec_personal),
        ('Cómo leer este informe (metodología)', _sec_metodologia),
    )
    for titulo, seccion in secciones:
        historia.append(Paragraph(titulo, st['h2']))
        try:
            historia.extend(seccion(fecha, st) or [])
        except Exception as exc:                            # noqa: BLE001
            logger.warning("informe: sección %r sin datos: %s", titulo, exc)
            historia.append(Paragraph(
                f'Sin datos disponibles para esta sección ({exc}).',
                st['nota']))

    def _pie(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(colors.HexColor(_GRIS))
        canvas.drawString(2 * cm, 1.1 * cm,
                          f'ELDE · Informe de tienda · {fecha}')
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm,
                               f'página {doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(destino, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                            title=f'Informe tienda {fecha}')
    doc.build(historia, onFirstPage=_pie, onLaterPages=_pie)


# ── Secciones ────────────────────────────────────────────────────────────
def _sec_resumen(fecha: str, st) -> List[Any]:
    """El día pedido, comparado con el anterior.

    Antes esta sección llamaba a `/api/v1/resumen`, que NO sabe de fechas:
    devolvía lo que llevara viva la sesión de los procesadores. El informe
    decía «lunes» en la portada y traía las cifras de todo lo que el servidor
    llevase encendido. Ahora sale de `tienda_almacen`, que guarda por jornada.
    """
    from reportlab.platypus import Paragraph

    from ..analityc.core import tienda_almacen as _almacen
    from .api_tienda import _aforo_vivo

    alm = _almacen.get_almacen()
    r = alm.resumen(fecha, fecha)
    hubo_datos = bool(alm.jornadas_con_datos(fecha, fecha))
    ant_desde, ant_hasta = _almacen.ventana_anterior(fecha, fecha)
    previo = alm.resumen(ant_desde, ant_hasta)

    def _delta(clave: str) -> str:
        antes = previo.get(clave) or 0
        if not antes:
            return '— sin base'          # dividir entre cero no da 0 %, da nada
        pct = 100.0 * ((r.get(clave) or 0) - antes) / antes
        return f"{pct:+.1f} %"

    # «Personas ahora» solo significa algo si el informe es de hoy.
    es_hoy = fecha == _almacen.jornada_de()
    filas = [
        ('Visitantes únicos', r.get('visitantes_unicos'),
         _delta('visitantes_unicos'),
         'personas DISTINTAS: el rostro evita contar dos veces a la misma'),
        ('Entradas totales', r.get('entradas'), _delta('entradas'),
         'cada cruce del área de conteo suma, aunque repita la persona'),
        ('Salidas', r.get('salidas'), _delta('salidas'),
         'cruces confirmados hacia fuera del área'),
        ('Permanencia media',
         (f"{float(r['permanencia_media_s']):.1f} s"
          if r.get('permanencia_media_s') is not None else None),
         _delta('permanencia_media_s'),
         'cuánto se queda la gente: más tiempo, más oportunidad de venta'),
        ('Personas ahora', (_aforo_vivo(None) if es_hoy else None), '—',
         'dentro del área al generar el informe (solo en el informe de hoy)'),
    ]
    salida = [_tabla(
        [('Indicador', 'Valor', f'vs {ant_desde}', 'Qué significa')]
        + [(n, _fmt(v), d, q) for n, v, d, q in filas],
        anchos=[3.8, 2.0, 2.2, 8.0])]
    if not hubo_datos:
        salida.append(Paragraph(
            f'<b>No hay ningún dato registrado el {fecha}.</b> O no había '
            'ninguna cámara de tienda transmitiendo, o el servidor estuvo '
            'apagado. No es un día de cero visitas: es un día sin medir.',
            st['nota']))
    salida.append(Paragraph(
        'Los indicadores del área de conteo dependen de que cada cámara '
        'tenga su área dibujada en el cliente; una raya (—) significa '
        '«sin datos», no cero. La comparación es contra el día anterior '
        f'({ant_desde}).', st['nota']))
    return salida


def _sec_anaqueles(fecha: str, st) -> List[Any]:
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, Spacer

    from ..analityc.core.anaqueles_registro import (camaras_registradas,
                                                    ref_ruta,
                                                    ultimo_registro)

    salida: List[Any] = [Paragraph(
        '<b>Visita</b> = alguien se acercó al nivel y se quedó unos '
        'segundos (el producto llama la atención). <b>Agarre</b> = metió '
        'la mano con un gesto real de entrada, no basta pararse cerca '
        '(el producto se tocó). <b>Conversión</b> = agarres ÷ visitas.',
        st['cuerpo']), Spacer(1, 4)]

    con_datos = False
    for cam in camaras_registradas():
        datos = ultimo_registro(cam, fecha)
        if not datos or not datos.get('areas'):
            continue
        con_datos = True
        areas = sorted(datos['areas'].items(), key=lambda kv: int(kv[0]))
        total_v = sum(int(a.get('visitas') or 0) for _i, a in areas)
        total_a = sum(int(a.get('agarres') or 0) for _i, a in areas)
        salida.append(Paragraph(
            f'Cámara <b>{cam}</b> — {total_a} agarres · {total_v} visitas '
            f'en el día', st['h3']))

        filas = [('Nivel', 'Visitas', 'Agarres', 'Conversión',
                  'Quién más (género)')]
        ranking = sorted(areas, key=lambda kv: (
            -int(kv[1].get('agarres') or 0), -int(kv[1].get('visitas') or 0)))
        for idx, a in ranking:
            v = int(a.get('visitas') or 0)
            g = int(a.get('agarres') or 0)
            conv = f'{100 * g / v:.0f}%' if v else '—'
            genero = a.get('por_genero') or {}
            quien = max(genero, key=genero.get) if genero else '—'
            quien = {'male': 'hombres', 'female': 'mujeres'}.get(quien, quien)
            filas.append((a.get('nombre') or f'Área {int(idx) + 1}',
                          str(v), str(g), conv, str(quien)))
        salida.append(_tabla(filas, anchos=[5.2, 2.2, 2.2, 2.6, 3.8]))

        # El calor del nivel MÁS agarrado, sobre la foto real del estante.
        imagen = _imagen_area(cam, datos, ranking)
        if imagen is not None:
            salida.append(Spacer(1, 4))
            salida.append(RLImage(imagen, width=15 * cm, height=8 * cm,
                                  kind='proportional'))
            salida.append(Paragraph(
                'Calor del nivel más agarrado, compuesto sobre la foto '
                'real del estante (tomada con el área vacía): las zonas '
                'cálidas son donde más metieron la mano.', st['nota']))
        salida.append(Spacer(1, 6))

    if not con_datos:
        salida.append(Paragraph(
            'Sin datos de anaqueles ese día: se llenan cuando una cámara '
            'transmite en modo Anaqueles con las áreas dibujadas.',
            st['nota']))
    return salida


def _imagen_area(cam: str, datos: Dict[str, Any],
                 ranking) -> Optional[io.BytesIO]:
    """PNG del área top compuesto sobre su foto de referencia, o None."""
    try:
        import cv2
        import numpy as np

        from ..analityc.core.analytics.heatmap import render_calor_sobre
        from ..analityc.core.anaqueles_registro import ref_ruta
        for idx, a in ranking:
            grid = np.asarray(a.get('grid') or [], dtype=np.float32)
            if grid.size == 0 or float(grid.sum()) <= 0:
                continue
            ruta = ref_ruta(cam, int(idx))
            if ruta is None:
                continue
            img = cv2.imread(str(ruta))
            if img is None or img.size == 0:
                continue
            if img.shape[1] < 560:
                esc = 560.0 / img.shape[1]
                img = cv2.resize(img, (560, max(1, int(img.shape[0] * esc))))
            out = render_calor_sobre(
                img, grid, alpha_max=0.6, con_leyenda=True,
                titulo=a.get('nombre') or f'Área {int(idx) + 1}',
                subtitulo=(f"{int(a.get('agarres') or 0)} agarres · "
                           f"{int(a.get('visitas') or 0)} visitas"))
            ok, buf = cv2.imencode('.png', out)
            if ok:
                return io.BytesIO(buf.tobytes())
        return None
    except Exception:                                       # noqa: BLE001
        return None


def _sec_demografia(fecha: str, st) -> List[Any]:
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, Table

    from ..analityc.core import tienda_almacen as _almacen

    # Del DÍA pedido (antes venía de /api/v1/resumen, que no filtra por fecha
    # y traía la demografía de toda la galería histórica).
    datos = _almacen.get_almacen().demografia(fecha, fecha)
    genero = datos.get('genero') or {}
    edad = datos.get('edad') or {}
    if not genero and not edad:
        return [Paragraph(f'Sin demografía registrada el {fecha}: se llena a '
                          'medida que las cámaras ven rostros con claridad.',
                          st['nota'])]
    celdas = []
    for titulo, dist in (('Género', genero), ('Edad', edad)):
        img = _barras(titulo, dist)
        if img is not None:
            celdas.append(RLImage(img, width=7.6 * cm, height=6 * cm,
                                  kind='proportional'))
    salida: List[Any] = []
    if celdas:
        salida.append(Table([celdas]))

    # Desglose POR PASILLO: imposible hasta el HITO 1, porque la captura solo
    # guardaba el nombre visible de la ventana y cuatro cámaras pueden
    # llamarse «iVMS-4200». Ahora la demografía va con el device_id.
    por_camara = datos.get('por_camara') or {}
    if len(por_camara) > 1:
        def _top(dist):
            if not dist:
                return '—'
            k, n = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[0]
            return f'{k} ({n})'

        filas = [('Pasillo / cámara', 'Personas', 'Género dominante',
                  'Franja dominante')]
        for cam, rec in sorted(por_camara.items()):
            filas.append((cam, _fmt(sum((rec.get('genero') or {}).values())),
                          _top(rec.get('genero')), _top(rec.get('edad'))))
        salida.append(Paragraph('Perfil por pasillo', st['h3']))
        salida.append(_tabla(filas, anchos=[6.0, 2.2, 3.9, 3.9]))

    salida.append(Paragraph(
        'Estimado por el rostro de las capturas; solo entran caras vistas '
        'con claridad (precisión antes que cantidad). Se cuentan personas '
        'DISTINTAS, no fotos.', st['nota']))
    return salida


def _barras(titulo: str, dist: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Barras horizontales chicas (matplotlib Agg) para el PDF."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pares = sorted(((str(k), float(v or 0)) for k, v in dist.items()),
                       key=lambda kv: kv[1])
        if not pares:
            return None
        etiquetas = [{'male': 'Hombres', 'female': 'Mujeres',
                      'Male': 'Hombres', 'Female': 'Mujeres'}.get(k, k)
                     for k, _v in pares]
        valores = [v for _k, v in pares]
        fig, ax = plt.subplots(figsize=(3.6, 0.5 + 0.42 * len(pares)),
                               dpi=150)
        ax.barh(etiquetas, valores, color=_AZUL, height=0.62)
        for i, v in enumerate(valores):
            ax.text(v, i, f' {int(v)}', va='center', fontsize=8,
                    color='#333333')
        ax.set_title(titulo, fontsize=10, loc='left', color=_AZUL)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.tick_params(labelsize=8, length=0)
        ax.set_xticks([])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:                                       # noqa: BLE001
        return None


def _sec_mapas(fecha: str, st) -> List[Any]:
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, Spacer

    from ..analityc.core.analytics.config import AnalyticsConfig
    from .api_lectura import listar_heatmaps

    filas = (listar_heatmaps() or {}).get('heatmaps') or []
    if not filas:
        return [Paragraph('Sin mapas de calor todavía: se generan solos '
                          'mientras las cámaras transmiten.', st['nota'])]
    salida: List[Any] = [Paragraph(
        'Por dónde se movió la gente en cada cámara (los pies acumulados '
        'del día; escala grabada en cada imagen: colores cálidos = más '
        'tránsito). El personal enrolado no calienta estos mapas.',
        st['cuerpo']), Spacer(1, 4)]
    raiz = Path(AnalyticsConfig.HEATMAP_DIR)
    if not raiz.is_absolute():
        raiz = Path(__file__).resolve().parents[2] / raiz
    from ..analityc.core.analytics import heatmap_bajo_pedido as _bajo_pedido
    for h in filas[:6]:
        # Pintado ahora, en JPEG (5-oct-2026: el mapa vigente ya no deja
        # PNG en disco); si no se puede, el PNG de antes si existe.
        datos = _bajo_pedido.imagen(h['device_id'], str(raiz))
        ruta = raiz / f"{h['device_id']}.png"
        if datos is not None:
            fuente: Any = io.BytesIO(datos)
        elif ruta.is_file():
            fuente = str(ruta)
        else:
            continue
        salida.append(Paragraph(
            f"<b>{h.get('camera_name') or h['device_id']}</b>"
            + (' · sin registrar' if h.get('huerfano') else ''), st['h3']))
        salida.append(RLImage(fuente, width=15 * cm, height=9 * cm,
                              kind='proportional'))
        salida.append(Spacer(1, 6))
    if len(filas) > 6:
        salida.append(Paragraph(
            f'(y {len(filas) - 6} cámaras más en el dashboard)', st['nota']))
    return salida


def _sec_visitantes(fecha: str, st) -> List[Any]:
    from reportlab.platypus import Paragraph

    # Sin el tope de 500 que traía antes: era un límite SILENCIOSO. Un día con
    # más de 500 capturas, o cualquier fecha lo bastante antigua, salía con la
    # tabla corta o vacía sin decir por qué. El índice completo cuesta un
    # recorrido de disco, y este informe se genera a demanda, no en bucle.
    from .dashboard import _captures_index
    compacta = fecha.replace('-', '')
    del_dia = [c for c in _captures_index()
               if str(c.get('timestamp') or '').startswith(compacta)]
    if not del_dia:
        return [Paragraph('Sin personas registradas ese día (las capturas '
                          'aparecen cuando una cámara transmite con '
                          'demografía).', st['nota'])]
    filas = [('Hora', 'Cámara', 'Género', 'Edad', 'Visitas')]
    tope = 60
    for c in del_dia[:tope]:
        ts = str(c.get('timestamp') or '')
        hora = (f'{ts[9:11]}:{ts[11:13]}:{ts[13:15]}'
                if len(ts) >= 15 else ts)
        filas.append((hora, str(c.get('camera') or '—'),
                      str(c.get('gender') or '—'),
                      str(c.get('age_range') or '—'),
                      str(c.get('visitas') or '—')))
    salida = [Paragraph(
        f'<b>{len(del_dia)}</b> personas registradas con foto ese día. '
        'Cada fila es una persona detectada (una foto por persona y '
        'visita; el rostro evita duplicados).', st['cuerpo']),
        _tabla(filas, anchos=[2.4, 5.2, 2.6, 3.0, 2.0])]
    if len(del_dia) > tope:
        salida.append(Paragraph(
            f'(se muestran las {tope} más recientes; las {len(del_dia)} '
            'completas están en el dashboard, sección Capturas)',
            st['nota']))
    return salida


def _sec_personal(fecha: str, st) -> List[Any]:
    from reportlab.platypus import Paragraph

    from ..analityc.core.personal_galeria import get_galeria
    fichas = get_galeria().listar()
    if not fichas:
        return [Paragraph(
            'Sin personal enrolado: cualquier empleado que camine por la '
            'tienda está contando como cliente. Se enrola con su foto (o '
            'la del uniforme) en el dashboard, sección Personal.',
            st['nota'])]
    filas = [('Nombre', 'Rol', 'Identificación', 'Estado', 'Fotos')]
    for f in fichas:
        filas.append((str(f.get('nombre') or '—'),
                      str(f.get('rol') or '—'),
                      'uniforme (vestimenta)'
                      if f.get('tipo') == 'uniforme' else 'rostro',
                      'activo' if f.get('activo') else 'inactivo',
                      str(f.get('n_fotos') or 0)))
    return [Paragraph(
        'Los enrolados NO cuentan en ninguna métrica de este informe; en '
        'su lugar generan avisos operativos («seguridad en el pasillo», '
        '«empleado surte el anaquel»).', st['cuerpo']),
        _tabla(filas, anchos=[5.0, 2.6, 4.0, 2.2, 1.6])]


def _sec_metodologia(fecha: str, st) -> List[Any]:
    from reportlab.platypus import Paragraph
    puntos = (
        ('Visitante único', 'una persona DISTINTA, reconocida por el '
         'rostro: entrar y salir tres veces cuenta una sola vez.'),
        ('Visita a un anaquel', 'DETENERSE frente a un nivel del estante '
         'al menos ~1.5 s. Pasar caminando por el pasillo no cuenta, y la '
         'misma persona no re-cuenta en el mismo nivel hasta pasados ~2 '
         'minutos (aunque la cámara le pierda el rastro un instante).'),
        ('Agarre', 'un GESTO real con el cuerpo QUIETO: la mano se ve '
         'fuera, viaja y entra profundo al nivel. Estar parado con la '
         'mano colgando, el braceo al caminar o el temblor de la muñeca '
         'NO cuentan, y tras un agarre hay que sacar la mano para poder '
         'contar otro.'),
        ('Conversión', 'agarres ÷ visitas. Un nivel muy visitado con baja '
         'conversión atrae pero no vende: candidato a cambiar producto, '
         'precio o altura.'),
        ('Mapas de calor', 'acumulan posiciones reales (pies en los mapas '
         'de cámara; manos en los de anaquel). La escala va grabada en '
         'cada imagen. Si el operador cambia el canal de la cámara, el '
         'sistema lo detecta y NO mezcla escenas.'),
        ('Exclusión del personal', 'seguridad y pasilleros enrolados (por '
         'rostro o por uniforme) quedan fuera de todas las métricas.'),
        ('Límites honestos', 'de noche las cámaras infrarrojas pierden '
         'color (el reconocimiento por uniforme rinde menos); un empleado '
         'de espaldas sin uniforme enrolado cuenta como cliente hasta '
         'confirmar su rostro; «—» significa sin datos, no cero.'),
    )
    return [Paragraph(f'<b>{t}:</b> {d}', st['cuerpo']) for t, d in puntos]


# ── Utilidades ───────────────────────────────────────────────────────────
def _fmt(v: Any) -> str:
    if v is None:
        return '—'
    try:
        return f'{int(v):,}'.replace(',', '.')
    except (TypeError, ValueError):
        return str(v)


def _tabla(filas: List[tuple], anchos: List[float]):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle
    t = Table([[str(c) for c in fila] for fila in filas],
              colWidths=[a * cm for a in anchos], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(_AZUL)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f2f6f8')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c8d4da')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t
