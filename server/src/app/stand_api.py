"""src/app/stand_api.py — API REST del stand de degustación (HITO 10).

## Prefijo propio y por qué

`api_lectura.py` reserva `/api/v1` para lectura pura, sin un solo POST. El
stand necesita **una** escritura (enrolar personal), así que vive bajo
`/api/stand` en vez de partirse entre dos prefijos por un único endpoint. Es
un módulo autocontenido: su propio almacén, su propio namespace.

## Precalculado, no on-the-fly

Los KPIs se calculan sobre `stand_visitantes`, que tiene una fila por persona
única. Para sesiones largas el dashboard debe leer de `stand_metricas` (job de
15 min); estos endpoints recalculan al vuelo porque una jornada de stand cabe
de sobra en memoria y así el dato nunca va desfasado respecto a la BD.

## Reconexión del cliente

`/kpis` devuelve totales ABSOLUTOS, no incrementos. El cliente que se
reconecta pide este endpoint y se pone al día de golpe, sin arrastrar
contadores desfasados.
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from ..analityc.core import stand_metricas as _m
from ..analityc.core.stand_almacen import AlmacenStand
from ..analityc.core.stand_config import cargar as cargar_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/stand', tags=['stand'])

_almacen: Optional[AlmacenStand] = None


def almacen() -> AlmacenStand:
    """Almacén perezoso: no se crea el .db hasta que alguien pregunta."""
    global _almacen
    if _almacen is None:
        _almacen = AlmacenStand()
    return _almacen


def _visitantes(sesion_id: int) -> List[Dict[str, Any]]:
    """Visitantes que CUENTAN: sin personal y sin descartados."""
    return almacen().visitantes(sesion_id)


def _existe(sesion_id: int) -> None:
    if not any(s['id'] == sesion_id for s in almacen().sesiones()):
        raise HTTPException(status_code=404,
                            detail=f'sesión {sesion_id} no encontrada')


@router.get('/frame')
def frame_actual():
    """PNG del último frame que el pipeline procesó. Para calibrar zonas.

    ## Por qué hace falta

    iVMS-4200 corre elevado: `PrintWindow` sobre su ventana devuelve `None` y
    `SetWindowPos` da «Acceso denegado», así que la herramienta de calibración
    no siempre puede capturarla. Pero el **cliente sí** la captura y se la
    manda al servidor, que la tiene en memoria.

    Este frame es además mejor que una captura de pantalla: es literalmente el
    que el pipeline analiza, así que las zonas dibujadas sobre él coinciden
    píxel a píxel con lo que se va a medir.

    Requiere que el cliente esté conectado y enviando.

    Devuelve además, en las cabeceras, lo que el pipeline está viendo AHORA:
    `X-Detecciones` (personas seguidas) y `X-Aforo` (cuántas tienen el pie
    dentro del ROI). Con eso se distingue de un vistazo el «no detecta a
    nadie» del «detecta pero el ROI está mal puesto», que es la pregunta que
    de verdad se hace cuando algo no cuenta.
    """
    from fastapi.responses import Response

    from .app import _iter_person_processors

    for _cid, cam_id, cam_proc in _iter_person_processors():
        img = _ultimo_frame(cam_proc)
        if img is None:
            continue
        try:
            import cv2
            ok, buf = cv2.imencode('.png', img)
            if not ok:
                continue
            return Response(
                content=buf.tobytes(), media_type='image/png',
                headers={'X-Camara': str(cam_id),
                         'X-Tamano': f'{img.shape[1]}x{img.shape[0]}',
                         'X-Detecciones': str(_cuantas_personas(cam_proc)),
                         'X-Aforo': str(int(getattr(cam_proc, '_zone_count',
                                                    0) or 0))})
        except Exception as exc:                            # noqa: BLE001
            logger.error("stand: no se pudo codificar el frame: %s", exc)

    raise HTTPException(
        status_code=503,
        detail='ningún cliente está enviando frames ahora mismo. Arranca el '
               'cliente Amazonas y pon la cámara en marcha.')


def _ultimo_frame(cam_proc):
    """El último frame procesado, esté donde esté.

    `process_frame` lo escribe en `self.last_processed_frame` con el estado de
    la cámara intercambiado, y al salir `_pop_state` **restaura el valor
    anterior**: el atributo vuelve a quedar en `None` y el frame se queda en
    `camera_states[camera_id]`. Leyendo solo el atributo, este endpoint
    devolvía 503 siempre aunque el cliente estuviera enviando vídeo.
    """
    img = getattr(cam_proc, 'last_processed_frame', None)
    if img is not None:
        return img
    for estado in (getattr(cam_proc, 'camera_states', None) or {}).values():
        img = (estado or {}).get('last_processed_frame')
        if img is not None:
            return img
    return None


def _cuantas_personas(cam_proc) -> int:
    """Tracks de PERSONA vivos (clase interna -1), estén o no en el ROI."""
    tracks = getattr(cam_proc, 'active_tracks', None) or {}
    if not tracks:
        for estado in (getattr(cam_proc, 'camera_states', None) or {}).values():
            tracks = (estado or {}).get('active_tracks') or {}
            if tracks:
                break
    return sum(1 for t in tracks.values() if t.get('class', -1) == -1)


@router.get('/sesiones')
def sesiones() -> Dict[str, Any]:
    filas = almacen().sesiones()
    return {'total': len(filas), 'sesiones': filas}


@router.get('/{sesion_id}/kpis')
def kpis(sesion_id: int) -> Dict[str, Any]:
    _existe(sesion_id)
    vis = _visitantes(sesion_id)
    cfg = cargar_config()
    listo, motivo = cfg.listo_para_contar()
    return {
        'sesion_id': sesion_id,
        **_m.kpis(vis),
        # El estado de la configuración viaja con los números: una cifra
        # calculada sobre zonas sin calibrar no es comparable con otra.
        'configuracion': {'listo_para_contar': listo, 'motivo': motivo,
                          'avisos': cfg.avisos()},
    }


@router.get('/{sesion_id}/embudo')
def embudo(sesion_id: int) -> Dict[str, Any]:
    _existe(sesion_id)
    return {'sesion_id': sesion_id, **_m.embudo(_visitantes(sesion_id))}


@router.get('/{sesion_id}/demografia')
def demografia(sesion_id: int) -> Dict[str, Any]:
    _existe(sesion_id)
    return {'sesion_id': sesion_id, **_m.demografia(_visitantes(sesion_id))}


@router.get('/{sesion_id}/temporal')
def temporal(sesion_id: int,
             bucket: str = Query('15m', pattern=r'^\d+m$')) -> Dict[str, Any]:
    _existe(sesion_id)
    minutos = max(1, int(bucket[:-1]))
    return {'sesion_id': sesion_id, 'bucket': bucket,
            'serie': _m.temporal(_visitantes(sesion_id), minutos)}


@router.get('/{sesion_id}/heatmap')
def heatmap(sesion_id: int,
            tipo: str = Query('permanencia')) -> Dict[str, Any]:
    _existe(sesion_id)
    if tipo not in ('trafico', 'permanencia', 'detenciones'):
        raise HTTPException(status_code=400, detail=f'tipo inválido: {tipo}')
    snap = almacen().ultimo_heatmap(sesion_id, tipo)
    if snap is None:
        # 200 con datos vacíos, no 404: «todavía no hay mapa» es un estado
        # normal al principio de la sesión, no un error del cliente.
        return {'sesion_id': sesion_id, 'tipo': tipo, 'grid': None,
                'matriz': [], 'vacio': True}
    return {'sesion_id': sesion_id, 'tipo': tipo, 'grid': snap.get('grid'),
            'matriz': snap.get('matriz'), 'ts': snap.get('ts'),
            'vacio': False}


@router.get('/{sesion_id}/trafico-perdido')
def trafico_perdido(sesion_id: int) -> Dict[str, Any]:
    _existe(sesion_id)
    return {'sesion_id': sesion_id,
            **_m.trafico_perdido(_visitantes(sesion_id))}


@router.get('/{sesion_id}/calidad')
def calidad(sesion_id: int) -> Dict[str, Any]:
    """Panel de calidad del dato. Sin esto ninguna cifra es definitiva."""
    _existe(sesion_id)
    alm = almacen()
    todos = alm.visitantes(sesion_id, incluir_personal=True,
                           incluir_descartados=True)
    descartados = [v for v in todos if v.get('descartado')]
    personal = [v for v in todos if v.get('es_personal')]
    cfg = cargar_config()
    listo, motivo = cfg.listo_para_contar()
    return {
        'sesion_id': sesion_id,
        **_m.calidad_dato(_visitantes(sesion_id), descartados,
                          {'personal_excluido': len(personal),
                           'zonas_calibradas': cfg.zonas_calibradas(),
                           'listo_para_contar': listo,
                           'motivo_configuracion': motivo,
                           'avisos_configuracion': cfg.avisos()}),
    }


_COLUMNAS_CSV = [
    'persistent_id', 'primera_deteccion', 'ultima_deteccion',
    'duracion_total_s', 'genero', 'genero_confianza', 'rango_etario',
    'edad_estimada', 'demografia_valida', 'motivo_sin_demografia',
    'se_detuvo', 'dwell_max_s', 'fue_atendido', 'duracion_atencion_s',
    'degusto', 'confianza_degustacion', 'metodo_deteccion_degustacion',
    'clasificacion',
]


@router.get('/{sesion_id}/visitantes')
def visitantes(sesion_id: int,
               export: Optional[str] = Query(None),
               clasificacion: Optional[str] = Query(None),
               genero: Optional[str] = Query(None)):
    _existe(sesion_id)
    filas = _visitantes(sesion_id)
    if clasificacion:
        filas = [v for v in filas if v.get('clasificacion') == clasificacion]
    if genero:
        filas = [v for v in filas if v.get('genero') == genero]

    if (export or '').lower() == 'csv':
        buf = io.StringIO()
        escritor = csv.DictWriter(buf, fieldnames=_COLUMNAS_CSV,
                                  extrasaction='ignore')
        escritor.writeheader()
        for v in filas:
            escritor.writerow(v)
        return PlainTextResponse(
            buf.getvalue(), media_type='text/csv',
            headers={'Content-Disposition':
                     f'attachment; filename="stand_{sesion_id}.csv"'})

    return {'sesion_id': sesion_id, 'total': len(filas), 'visitantes': filas}


@router.post('/{sesion_id}/personal/enrolar')
def enrolar_personal(sesion_id: int, cuerpo: Dict[str, Any]) -> Dict[str, Any]:
    """Marca a una persona como personal. La saca de todas las métricas.

    Único POST del módulo. Se limita a escribir el flag en la BD: la galería
    de embeddings vive en el proceso del pipeline (`stand_personal.py`), que
    la recarga al ver el cambio.
    """
    _existe(sesion_id)
    pid = str(cuerpo.get('persistent_id') or '').strip()
    if not pid:
        raise HTTPException(status_code=400,
                            detail='falta persistent_id')
    es_personal = bool(cuerpo.get('es_personal', True))
    alm = almacen()
    if alm.visitante(sesion_id, pid) is None:
        raise HTTPException(status_code=404,
                            detail=f'visitante {pid} no existe en la sesión')
    alm.upsert_visitante(sesion_id, pid, es_personal=es_personal)
    logger.info("stand: %s marcado es_personal=%s en sesión %s",
                pid, es_personal, sesion_id)
    return {'ok': True, 'persistent_id': pid, 'es_personal': es_personal,
            'contadores': alm.contadores(sesion_id)}
