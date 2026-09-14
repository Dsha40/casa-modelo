"""src/analityc/core/tienda_capturas.py — el índice de capturas (HITO 3).

## El problema que resuelve

`dashboard._captures_index()` abría y parseaba **todos** los sidecars JSON de
`output/captures/persons/` en CADA petición. Y se le llamaba:

  - una vez, sin límite, en `dashboard_summary` (solo para contar);
  - **dos** veces en `dashboard_captures` (una para el total, otra para la
    página);

y el dashboard de tienda pedía `/resumen` + `/capturas` + `/capturas?limite=500`
cada 15 segundos. Total: ~5 recorridos completos del disco por refresco y por
pestaña abierta, cuatro veces por minuto, **en el mismo proceso que corre la
inferencia**. Con 20.000 capturas eso son 100.000 aperturas de archivo por
minuto para dibujar unas barras que casi nunca cambian.

## Cómo lo resuelve, sin dejar de ser honesto

El disco SIGUE siendo la fuente de verdad. Esto es una **caché reconciliada**:

  - `os.scandir` enumera el directorio una vez. En Windows eso trae ya el
    `mtime` de cada archivo sin un `stat()` por foto, que es lo que hace viable
    comparar 20.000 entradas en milisegundos.
  - Solo se **parsean** los sidecars nuevos o cuyo `mtime` cambió (la
    demografía se escribe más tarde, cuando converge, y el reanálisis con VLM
    también los reescribe: por eso no basta con «está o no está»).
  - Lo que desapareció del disco se borra del índice.
  - La reconciliación va **amortiguada** (`PERIODO_SYNC_S`): una ráfaga de
    peticiones no vuelve a enumerar el directorio.

## Y si la base falla

Se cae con elegancia al recorrido de disco de siempre
(`indice_desde_disco`), que se conserva justo para eso. Un índice roto no
puede dejar la galería en blanco: el dashboard seguirá lento, pero completo.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from . import tienda_almacen as _almacen
from .analytics.config import AnalyticsConfig

logger = logging.getLogger(__name__)

#: Mínimo entre reconciliaciones con el disco. Las capturas nacen de a poco
#: (una por persona), así que 3 s no deja ver nada «viejo» a un humano y corta
#: en seco la enumeración repetida cuando hay varias pestañas abiertas.
PERIODO_SYNC_S: float = float(os.getenv('ELDE_TIENDA_SYNC_CAPTURAS_S', '3.0'))

_ultimo_sync: float = 0.0

#: `{stem: mtime}` de lo indexado, en memoria. Se llena de la base UNA vez y
#: luego lo mantiene la propia reconciliación: releer 20.000 filas cada 3 s
#: para descubrir que no cambió nada es justo el coste que este módulo existe
#: para quitar. Este proceso es el único que escribe el índice; si otro lo
#: tocara, lo peor que pasa es que se reparsee algún sidecar de más (la
#: reconciliación es idempotente).
_indexadas: Optional[Dict[str, float]] = None


def _raiz() -> str:
    """`server/` — las rutas de `AnalyticsConfig` son relativas a ahí."""
    return str(_almacen._RAIZ_SERVIDOR)


def _dir_personas() -> str:
    base = AnalyticsConfig.CAPTURE_DIR
    if not os.path.isabs(base):
        base = os.path.join(_raiz(), base)
    return os.path.join(base, 'persons')


def _dir_rostros() -> str:
    base = AnalyticsConfig.CAPTURE_DIR
    if not os.path.isabs(base):
        base = os.path.join(_raiz(), base)
    return os.path.join(base, 'faces')


def ts_de_stem(sello: Any) -> float:
    """`AAAAMMDD_HHMMSS` -> época. 0.0 si no hay forma de leerlo.

    Es el mismo formato que escribe `tiendas._capture_detection`. Se ordena y
    se agrupa por esto, así que un sello ilegible va al final en vez de
    reventar la lista.
    """
    texto = str(sello or '')
    try:
        return time.mktime(time.strptime(texto[:15], '%Y%m%d_%H%M%S'))
    except (ValueError, TypeError):
        return 0.0


def _fila_de_sidecar(stem: str, meta: Dict[str, Any], mtime: float,
                     tiene_rostro: bool) -> Dict[str, Any]:
    ts = ts_de_stem(meta.get('timestamp') or stem)
    return {
        'stem': stem,
        'ts': ts,
        'jornada': _almacen.jornada_de(ts) if ts else '',
        'camera': meta.get('camera'),
        'device_id': meta.get('device_id'),
        'genero': meta.get('gender'),
        'edad': meta.get('age_range'),
        'person_uuid': meta.get('person_uuid'),
        'visitas': (int(meta['visitas'])
                    if str(meta.get('visitas') or '').isdigit() else None),
        'tiene_rostro': 1 if tiene_rostro else 0,
        'no_es_persona': 1 if meta.get('no_es_persona') else 0,
        'revisado_vlm': 1 if meta.get('revisado_por_vlm') else 0,
        'origen_demografia': meta.get('origen_demografia'),
        'motivo': meta.get('motivo_sin_demografia'),
        'mtime': float(mtime or 0.0),
    }


def sincronizar(force: bool = False) -> Dict[str, int]:
    """Pone el índice al día con el disco. Devuelve qué hizo.

    Barata por diseño: una enumeración del directorio y parseo SOLO de lo
    nuevo o modificado. Nunca lanza.
    """
    global _ultimo_sync, _indexadas
    ahora = time.time()
    if not force and (ahora - _ultimo_sync) < PERIODO_SYNC_S:
        return {'nuevas': 0, 'actualizadas': 0, 'borradas': 0, 'omitido': 1}
    _ultimo_sync = ahora

    pdir = _dir_personas()
    if not os.path.isdir(pdir):
        return {'nuevas': 0, 'actualizadas': 0, 'borradas': 0, 'omitido': 0}

    alm = _almacen.get_almacen()
    if _indexadas is None:
        _indexadas = alm.capturas_indexadas()
        # La jornada de cada captura se calculó al indexarla con el corte
        # (ELDE_TIENDA_CORTE_JORNADA_H) vigente ENTONCES; si el operador lo
        # cambió y reinició, las filas viejas conservan la jornada vieja y
        # la galería las muestra en el día equivocado (5-sep-2026). Se mira
        # la captura más reciente: si su jornada ya no cuadra, se reindexa
        # todo (un solo parseo extra por captura, una vez por arranque).
        reciente = alm.captura_mas_reciente()
        if (reciente and reciente['ts']
                and _almacen.jornada_de(reciente['ts']) != reciente['jornada']):
            logger.info('índice de capturas: el corte de jornada cambió '
                        '(%s -> %s); se reindexan %d capturas',
                        reciente['jornada'], _almacen.jornada_de(reciente['ts']),
                        len(_indexadas))
            _indexadas = {}
    indexadas = _indexadas

    en_disco: Dict[str, float] = {}
    try:
        # scandir en vez de listdir+stat: en Windows el mtime viene con la
        # entrada del directorio, sin una llamada al sistema por archivo.
        with os.scandir(pdir) as entradas:
            for e in entradas:
                if not e.name.endswith('.json'):
                    continue
                try:
                    en_disco[e.name[:-5]] = e.stat().st_mtime
                except OSError:
                    continue
    except OSError as exc:
        logger.warning('índice de capturas: no se pudo leer %s: %s', pdir, exc)
        return {'nuevas': 0, 'actualizadas': 0, 'borradas': 0, 'omitido': 0}

    fdir = _dir_rostros()
    nuevas: List[Dict[str, Any]] = []
    actualizadas = 0
    for stem, mtime in en_disco.items():
        anterior = indexadas.get(stem)
        if anterior is not None and abs(anterior - mtime) < 0.001:
            continue                    # sin cambios: no se vuelve a abrir
        # La foto manda: un sidecar sin su JPG no es una captura.
        if not os.path.isfile(os.path.join(pdir, f'{stem}.jpg')):
            continue
        meta: Dict[str, Any] = {}
        try:
            with open(os.path.join(pdir, f'{stem}.json'), encoding='utf-8') as f:
                meta = json.load(f) or {}
        except (OSError, ValueError):
            meta = {}
        nuevas.append(_fila_de_sidecar(
            stem, meta, mtime,
            os.path.isfile(os.path.join(fdir, f'{stem}.jpg'))))
        if anterior is not None:
            actualizadas += 1

    sobran = [s for s in indexadas if s not in en_disco]
    guardadas = alm.guardar_capturas(nuevas)
    borradas = alm.borrar_capturas(sobran)
    if guardadas:
        for fila in nuevas:
            indexadas[fila['stem']] = fila['mtime']
    for stem in sobran:
        indexadas.pop(stem, None)
    if guardadas or borradas:
        logger.info('índice de capturas: %d al día (%d nuevas, %d '
                    'actualizadas, %d borradas)',
                    len(en_disco), guardadas - actualizadas, actualizadas,
                    borradas)
    return {'nuevas': guardadas - actualizadas, 'actualizadas': actualizadas,
            'borradas': borradas, 'omitido': 0}


def reiniciar_para_pruebas() -> None:
    """Olvida la caché en memoria y el throttle.

    Lo llama `tienda_almacen.reiniciar_para_pruebas`: una prueba que cambia de
    base tendría, si no, el índice de la anterior en memoria — y el fallo
    aparecería en OTRA prueba, que es la peor forma de encontrarlo.
    """
    global _indexadas, _ultimo_sync
    _indexadas = None
    _ultimo_sync = 0.0


def _a_forma_de_siempre(fila: Dict[str, Any]) -> Dict[str, Any]:
    """Traduce una fila del índice a la forma que ya consume el dashboard.

    Las claves son las de `_captures_index()` de toda la vida: cambiarlas
    rompería la galería, el buscador VLM y el PDF a la vez.
    """
    ts = float(fila.get('ts') or 0.0)
    return {
        'stem': fila['stem'],
        'gender': fila.get('genero'),
        'age_range': fila.get('edad'),
        'camera': fila.get('camera'),
        'device_id': fila.get('device_id'),
        'timestamp': (time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))
                      if ts else fila['stem'][:15]),
        'person_uuid': fila.get('person_uuid'),
        'visitas': fila.get('visitas'),
        'has_face': bool(fila.get('tiene_rostro')),
        'no_es_persona': bool(fila.get('no_es_persona')),
        'revisado_por_vlm': bool(fila.get('revisado_vlm')),
        'origen_demografia': fila.get('origen_demografia'),
        'motivo': fila.get('motivo'),
    }


def listar(limite: int = 60, offset: int = 0,
           desde: Optional[str] = None, hasta: Optional[str] = None,
           camara: Optional[str] = None) -> Dict[str, Any]:
    """Capturas más recientes primero, con su total. Coste independiente de
    cuántas haya acumuladas."""
    sincronizar()
    datos = _almacen.get_almacen().capturas(
        desde=desde, hasta=hasta, camera_id=camara,
        limite=limite, offset=offset)
    return {'total': datos['total'],
            'capturas': [_a_forma_de_siempre(f) for f in datos['capturas']]}


def total() -> int:
    """Cuántas capturas hay. Antes esto costaba abrir todos los sidecars."""
    sincronizar()
    return _almacen.get_almacen().total_capturas()


def indice_desde_disco(limite: int = 0) -> List[Dict[str, Any]]:
    """El recorrido de disco de SIEMPRE, tal cual era.

    Se conserva por dos motivos: es el respaldo si la base falla (una galería
    lenta es mejor que una galería en blanco) y es la vara con la que se mide
    la mejora del índice.
    """
    pdir = _dir_personas()
    if not os.path.isdir(pdir):
        return []
    try:
        archivos = [f for f in os.listdir(pdir) if f.endswith('.json')]
    except OSError:
        return []
    archivos.sort(reverse=True)         # el stem empieza por AAAAMMDD_HHMMSS
    if limite > 0:
        archivos = archivos[:limite]
    fdir = _dir_rostros()
    salida = []
    for nombre in archivos:
        stem = nombre[:-5]
        if not os.path.isfile(os.path.join(pdir, f'{stem}.jpg')):
            continue
        meta: Dict[str, Any] = {}
        try:
            with open(os.path.join(pdir, nombre), encoding='utf-8') as f:
                meta = json.load(f) or {}
        except (OSError, ValueError):
            pass
        salida.append({
            'stem': stem,
            'gender': meta.get('gender'),
            'age_range': meta.get('age_range'),
            'camera': meta.get('camera'),
            'device_id': meta.get('device_id'),
            'timestamp': meta.get('timestamp'),
            'person_uuid': meta.get('person_uuid'),
            'visitas': meta.get('visitas'),
            'has_face': os.path.isfile(os.path.join(fdir, f'{stem}.jpg')),
            'no_es_persona': bool(meta.get('no_es_persona')),
            'revisado_por_vlm': bool(meta.get('revisado_por_vlm')),
            'origen_demografia': meta.get('origen_demografia'),
            'motivo': meta.get('motivo_sin_demografia'),
        })
    return salida
