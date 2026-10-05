"""Mapa de calor vigente pintado BAJO PEDIDO y en JPEG (5-oct-2026).

## Por qué

Hasta hoy, cada 5 s y DENTRO del procesado de una imagen, el acumulador
pintaba el mapa sobre la foto a tamaño completo y lo guardaba en PNG
(`maybe_save_snapshot`): ~174 ms de pintar + ~258 ms del PNG, medidos el
2-oct en el PC de la tienda con la CPU cargada. Esa imagen esperaba ~430 ms,
y con py-spy (5-oct, modo G) el mapa de calor era el 9,6 % del hilo de las
imágenes. Para un PNG que solo mira alguien cuando abre el dashboard.

Ahora el procesado solo guarda lo barato (JSON de zonas, fondo cada 60 s,
estado cada 30 s), y la imagen se pinta AQUÍ cuando alguien la pide
(dashboard, API, informe PDF), como ya hacía la API de anaqueles con el calor
de cada área. En JPEG: guardar en PNG costaba 258 ms; en JPEG, 8.

## Qué conserva

- El mismo dibujo de siempre (`render_calor_sobre` con leyenda, sobre la
  cuadrícula TOTAL), sobre el último fondo guardado (`bg/<cámara>.jpg`, de
  hace como mucho 60 s, preferiblemente sin gente).
- La dirección de siempre (`/dashboard/img/heatmap/<cámara>.png`): por
  dentro va JPEG y el navegador lo muestra igual.
- El histórico por horas sigue en PNG: se pinta una vez por hora.
- `HEATMAP_PNG_PERIODICO = True` vuelve al PNG cada 5 s, por si algo de
  fuera de este repositorio lee el fichero del disco.

Si varias pantallas piden la misma cámara, se reutiliza la imagen durante
`CACHE_S` segundos en vez de pintarla cada vez.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import heatmap_registro
from .config import AnalyticsConfig
from .heatmap import _safe_camera_name

logger = logging.getLogger(__name__)

#: Segundos que se reutiliza una imagen ya pintada (varias pantallas a la vez).
CACHE_S = 5.0
#: Calidad del JPEG entregado.
CALIDAD_JPEG = 85

_candado = threading.Lock()
# clave de cámara -> (momento en que se pintó, bytes JPEG)
_cache: Dict[str, Tuple[float, bytes]] = {}


def _carpeta(out_dir: Optional[str]) -> str:
    return out_dir or AnalyticsConfig.HEATMAP_DIR


def _meta(clave: str, out_dir: Optional[str]) -> Dict[str, Any]:
    try:
        with open(os.path.join(_carpeta(out_dir), f'{clave}.json'),
                  encoding='utf-8') as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def listado(out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cámaras con mapa vigente: las que tienen su `<cámara>.json` (lo
    escribe el acumulador cada 5 s), más las que solo tienen el `.png` de
    antes de este cambio. `mtime` = última actualización del mapa."""
    carpeta = _carpeta(out_dir)
    filas: Dict[str, Dict[str, Any]] = {}
    try:
        nombres = sorted(os.listdir(carpeta))
    except OSError:
        return []
    for fn in nombres:
        ruta = os.path.join(carpeta, fn)
        if not os.path.isfile(ruta) or '.tmp' in fn:
            continue
        base, ext = os.path.splitext(fn)
        if ext not in ('.json', '.png'):
            continue
        fila = filas.setdefault(base, {'camera': base, 'mtime': 0.0})
        fila['mtime'] = max(fila['mtime'], os.path.getmtime(ruta))
    return [filas[k] for k in sorted(filas)]


def existe(camera_id: Any, out_dir: Optional[str] = None) -> bool:
    clave = _safe_camera_name(camera_id)
    carpeta = _carpeta(out_dir)
    return any(os.path.isfile(os.path.join(carpeta, f'{clave}{ext}'))
               for ext in ('.json', '.png'))


def imagen(camera_id: Any, out_dir: Optional[str] = None) -> Optional[bytes]:
    """JPEG del mapa vigente de esa cámara, pintado ahora (o hace menos de
    `CACHE_S` s). None si la cámara no tiene mapa. Nunca lanza."""
    try:
        clave = _safe_camera_name(camera_id)
        ahora = time.time()
        with _candado:
            guardada = _cache.get(clave)
        if guardada is not None and ahora - guardada[0] < CACHE_S:
            return guardada[1]
        if not existe(clave, out_dir):
            return None            # no crear acumuladores para nombres ajenos
        acumulador = heatmap_registro.obtener(clave)   # en memoria o del disco
        if acumulador._samples == 0:
            return None
        fondo = acumulador._bg_guardado(clave, out_dir)
        if fondo is None:
            fondo = np.zeros((acumulador.grid_h * 10, acumulador.grid_w * 10, 3),
                             np.uint8)
        meta = _meta(clave, out_dir)
        titulo = str(heatmap_registro.nombre(clave) or meta.get('camera_name')
                     or clave)
        img = acumulador._render_grid(
            fondo, acumulador._total.copy(), con_leyenda=True, titulo=titulo,
            subtitulo=f'{acumulador._samples} registros · acumulado')
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPEG])
        if not ok:
            return None
        datos = buf.tobytes()
        with _candado:
            _cache[clave] = (ahora, datos)
        return datos
    except Exception as exc:                                # noqa: BLE001
        logger.debug('heatmap bajo pedido fallo para %s: %s', camera_id, exc)
        return None
