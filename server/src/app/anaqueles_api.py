"""src/app/anaqueles_api.py — API REST del modo Anaqueles.

## Prefijo propio y por qué

`/api/v1` es solo-lectura por regla y `dashboards/shared/api.js` solo
prefija esa ruta, pero el patrón que ya funciona para un modo autocontenido
es el del stand: router propio (`/api/stand`) con su almacén y su namespace.
Anaqueles hace lo mismo bajo `/api/anaqueles`.

## Fuente de verdad: el DISCO, no el pipeline

Los endpoints leen los JSON diarios de `anaqueles_registro` (escritura
amortiguada cada 5 s desde el pipeline). Leer los objetos vivos del
procesador desde otro hilo sería una carrera; 5 s de desfase no le importan
a un dashboard de marketing. La excepción es `/frame`, que sí mira el último
frame en memoria porque existe exactamente para calibrar las áreas contra lo
que el pipeline está viendo AHORA (patrón `/api/stand/frame`, la fuente de
verdad del diagnóstico del stand).

## Totales absolutos

`/resumen` devuelve totales del día, no incrementos: el cliente o dashboard
que se reconecta se pone al día de golpe (regla del emisor del stand).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ..analityc.core.anaqueles_registro import (camaras_registradas,
                                                ref_ruta, ultimo_registro)

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/anaqueles', tags=['anaqueles'])


@router.get('/frame')
def frame_actual():
    """PNG del último frame que el pipeline procesó. Para calibrar áreas.

    Mismo mecanismo que `/api/stand/frame` (iVMS corre elevado y no se deja
    capturar, pero el cliente ya manda el frame): devuelve LITERALMENTE lo
    que el pipeline analiza, con `X-Detecciones` en las cabeceras para
    distinguir «no detecta a nadie» de «las áreas están mal puestas».
    """
    import cv2
    from fastapi.responses import Response

    from ..analityc.core.anaqueles import Anaqueles
    from .app import _iter_person_processors

    for _cid, cam_id, cam_proc in _iter_person_processors():
        # Solo procesadores del MODO Anaqueles: sin este filtro, con una
        # cámara en modo Tiendas conectada a la vez, este endpoint podía
        # devolver el frame de la cámara equivocada y el operador calibraría
        # las áreas sobre una escena que el modo jamás va a medir.
        if not isinstance(cam_proc, Anaqueles):
            continue
        img = _ultimo_frame(cam_proc)
        if img is None:
            continue
        try:
            ok, buf = cv2.imencode('.png', img)
            if not ok:
                continue
            return Response(
                content=buf.tobytes(), media_type='image/png',
                headers={'X-Camara': str(cam_id),
                         'X-Tamano': f'{img.shape[1]}x{img.shape[0]}',
                         'X-Detecciones': str(_cuantas_personas(cam_proc))})
        except Exception as exc:                            # noqa: BLE001
            logger.error("anaqueles: no se pudo codificar el frame: %s", exc)

    raise HTTPException(
        status_code=503,
        detail='ninguna cámara está en modo Anaqueles enviando frames ahora '
               'mismo. Arranca el cliente de Tienda, elige el modo Anaqueles '
               'y pon la cámara en marcha.')


@router.get('/camaras')
def camaras() -> Dict[str, Any]:
    """Cámaras con al menos un día persistido."""
    lista = camaras_registradas()
    return {'total': len(lista), 'camaras': lista}


@router.get('/resumen')
def resumen_de_todas(
        fecha: Optional[str] = Query(
            None, pattern=r'^\d{4}-\d{2}-\d{2}$')) -> Dict[str, Any]:
    """TODAS las cámaras de una vez (HITO 3).

    El dashboard hacía 1 petición para la lista de cámaras y otra por cada
    una, cada 30 segundos. Con una tienda de ocho pasillos eso son nueve
    viajes para pintar una sección. Aquí van todas juntas, con la rejilla del
    heatmap incluida (que ya viajaba en el resumen por cámara).

    Una cámara sin datos ese día NO rompe la respuesta ni desaparece en
    silencio: sale en `sin_datos`, que es información de verdad («ese pasillo
    no midió hoy»), distinta de «midió cero».

    Los endpoints por cámara siguen existiendo: los usa quien consulta una
    sola y son los que ya conoce cualquier integración.
    """
    filas = []
    sin_datos = []
    for cam in camaras_registradas():
        datos = ultimo_registro(cam, fecha)
        if datos is None:
            sin_datos.append(cam)
            continue
        filas.append(resumen(cam, fecha))
    return {'fecha': fecha or (filas[0]['fecha'] if filas else None),
            'total': len(filas), 'camaras': filas, 'sin_datos': sin_datos}


@router.get('/{camera_id}/resumen')
def resumen(camera_id: str,
            fecha: Optional[str] = Query(
                None, pattern=r'^\d{4}-\d{2}-\d{2}$')) -> Dict[str, Any]:
    """Totales ABSOLUTOS por área del día (hoy por defecto) + ranking.

    El ranking va precalculado para que el dashboard no reimplemente el
    criterio: por AGARRES y, a igualdad, por visitas — el agarre es la señal
    de marketing fuerte («lo tocó»), la visita es la débil («se acercó»).
    """
    datos = _registro_o_404(camera_id, fecha)
    areas = []
    for clave, a in sorted((datos.get('areas') or {}).items(),
                           key=lambda kv: int(kv[0])):
        grid = a.get('grid') or []
        areas.append({
            'idx': int(clave),
            'nombre': a.get('nombre') or f'Área {int(clave) + 1}',
            'visitas': int(a.get('visitas') or 0),
            'agarres': int(a.get('agarres') or 0),
            'por_genero': a.get('por_genero') or {},
            'por_edad': a.get('por_edad') or {},
            # La rejilla viaja AQUÍ para que el dashboard no tenga que hacer
            # 1+N peticiones por cámara (el JSON del día ya está leído);
            # /heatmap/{idx} sigue existiendo para consultas sueltas.
            'grid': grid,
            'grid_dim': datos.get('grid_dim'),
            # Productos dibujados dentro del área (25-sep-2026). Lista vacía
            # en las áreas sin productos y en los días anteriores.
            'productos': _productos_de(a),
            'agarres_sin_producto': int(a.get('agarres_sin_producto') or 0),
        })
    ranking = sorted(areas, key=lambda x: (-x['agarres'], -x['visitas']))
    # Mismo motivo que `ranking`: que el dashboard no reimplemente el
    # criterio. Entre áreas, porque «el producto más agarrado del pasillo»
    # es la pregunta de marketing, no «el más agarrado de la balda 2».
    ranking_productos = sorted(
        ({'area_idx': a['idx'], 'area': a['nombre'], **p}
         for a in areas for p in a['productos']),
        key=lambda x: (-x['agarres'], x['area_idx'], x['idx']))
    return {
        'camera_id': camera_id,
        'fecha': datos.get('fecha'),
        'areas': areas,
        'ranking': [{'idx': a['idx'], 'nombre': a['nombre'],
                     'agarres': a['agarres'], 'visitas': a['visitas']}
                    for a in ranking],
        'ranking_productos': ranking_productos,
        'total_visitas': sum(a['visitas'] for a in areas),
        'total_agarres': sum(a['agarres'] for a in areas),
    }


def _productos_de(area: Dict[str, Any]) -> list:
    """Los productos de un área del JSON diario, en orden de posición."""
    salida = []
    for clave, p in (area.get('productos') or {}).items():
        try:
            idx = int(clave)
        except (TypeError, ValueError):
            continue                       # clave ilegible: no se inventa
        salida.append({'idx': idx,
                       'nombre': (p or {}).get('nombre')
                       or f'Producto {idx + 1}',
                       'agarres': int((p or {}).get('agarres') or 0)})
    return sorted(salida, key=lambda p: p['idx'])


@router.get('/{camera_id}/heatmap/{area_idx}')
def heatmap(camera_id: str, area_idx: int,
            fecha: Optional[str] = Query(
                None, pattern=r'^\d{4}-\d{2}-\d{2}$')) -> Dict[str, Any]:
    """La rejilla de calor de UN área (dónde exactamente tocan ese nivel)."""
    datos = _registro_o_404(camera_id, fecha)
    a = (datos.get('areas') or {}).get(str(int(area_idx)))
    if a is None:
        # 200 vacío, no 404: «todavía nadie tocó esa área» es un estado
        # normal al empezar el día, no un error del cliente.
        return {'camera_id': camera_id, 'area_idx': int(area_idx),
                'grid': None, 'vacio': True}
    return {
        'camera_id': camera_id,
        'area_idx': int(area_idx),
        'nombre': a.get('nombre'),
        'dim': datos.get('grid_dim'),
        'grid': a.get('grid') or [],
        'vacio': not any(any(fila) for fila in (a.get('grid') or [])),
    }


@router.get('/{camera_id}/heatmap/{area_idx}/imagen')
def heatmap_imagen(camera_id: str, area_idx: int,
                   fecha: Optional[str] = Query(
                       None, pattern=r'^\d{4}-\d{2}-\d{2}$')):
    """El calor del área COMPUESTO sobre la foto real del estante.

    La base es la foto de referencia que el pipeline toma solo cuando el
    área está vacía (ni clientes ni personal delante): se ven los productos
    de verdad debajo del calor, con la leyenda grabada en la imagen. 404
    con mensaje claro si aún no hay foto (se toma sola, cada ~10 min).
    """
    import cv2
    import numpy as np
    from fastapi.responses import Response

    from ..analityc.core.analytics.heatmap import render_calor_sobre

    datos = _registro_o_404(camera_id, fecha)
    a = (datos.get('areas') or {}).get(str(int(area_idx)))
    if a is None:
        raise HTTPException(status_code=404,
                            detail=f'el área {area_idx} no tiene datos ese día')
    ruta = ref_ruta(camera_id, int(area_idx))
    if ruta is None:
        raise HTTPException(
            status_code=404,
            detail='aún no hay foto de referencia de esta área: el sistema '
                   'la toma solo cuando el área está vacía (unos minutos '
                   'con la cámara transmitiendo).')
    img = cv2.imread(str(ruta))
    if img is None or img.size == 0:
        raise HTTPException(status_code=404,
                            detail='la foto de referencia no se pudo leer')
    # La leyenda necesita ancho: los recortes de un nivel suelen ser
    # apaisados y chicos — se reescala a un ancho cómodo de ver.
    if img.shape[1] < 560:
        escala = 560.0 / img.shape[1]
        img = cv2.resize(img, (560, max(1, int(img.shape[0] * escala))),
                         interpolation=cv2.INTER_CUBIC)
    grid = np.asarray(a.get('grid') or [], dtype=np.float32)
    nombre = a.get('nombre') or f'Área {int(area_idx) + 1}'
    sub = (f"{int(a.get('agarres') or 0)} agarres · "
           f"{int(a.get('visitas') or 0)} visitas · {datos.get('fecha')}")
    out = render_calor_sobre(img, grid, alpha_max=0.6, con_leyenda=True,
                             titulo=nombre, subtitulo=sub)
    ok, buf = cv2.imencode('.png', out)
    if not ok:
        raise HTTPException(status_code=500,
                            detail='no se pudo codificar la imagen')
    return Response(content=buf.tobytes(), media_type='image/png')


# ── Interno ──────────────────────────────────────────────────────────────
def _registro_o_404(camera_id: str, fecha: Optional[str]) -> Dict[str, Any]:
    datos = ultimo_registro(camera_id, fecha)
    if datos is None:
        raise HTTPException(
            status_code=404,
            detail=f'sin datos de anaqueles para la cámara {camera_id!r} '
                   f'en {fecha or "hoy"}. ¿El modo Anaqueles estuvo activo?')
    return datos


def _ultimo_frame(cam_proc):
    """El último frame procesado, esté donde esté (ver stand_api._ultimo_frame:
    tras `_pop_state` el atributo queda en None y el frame vive en
    `camera_states`)."""
    img = getattr(cam_proc, 'last_processed_frame', None)
    if img is not None:
        return img
    for estado in (getattr(cam_proc, 'camera_states', None) or {}).values():
        img = (estado or {}).get('last_processed_frame')
        if img is not None:
            return img
    return None


def _cuantas_personas(cam_proc) -> int:
    tracks = getattr(cam_proc, 'active_tracks', None) or {}
    if not tracks:
        for estado in (getattr(cam_proc, 'camera_states', None) or {}).values():
            tracks = (estado or {}).get('active_tracks') or {}
            if tracks:
                break
    return sum(1 for t in tracks.values() if t.get('class', -1) == -1)
