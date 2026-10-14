"""Áreas a ignorar: quien queda ENTERO dentro no se cuenta (12-oct-2026).

## Para qué

Lo pidió el usuario. En la cámara 3 se cuela gente del fondo que no está en
el pasillo, y el filtro por alto (`filtro_fondo.py`, 220 px) no la quita
toda: hay quien se ve grande pero está detrás. Al otro lado, delante de la
cámara, pasa gente que se ve cortada y con la caja pequeña, y un filtro por
alto más duro la perdería. Un área dibujada a mano sobre el fondo resuelve
lo primero sin tocar lo segundo.

## La regla

Se descarta la persona cuya caja está ENTERA dentro de un área (sus cuatro
esquinas dentro del polígono). Con un trozo fuera, cuenta: quien pasa por
delante del fondo y lo tapa en parte no se pierde.

## Dónde

Justo detrás de `filtro_fondo` y ANTES de ByteTrack (`tiendas.py`): lo que
se quita no llega a nada (seguimiento, ficha, entradas, demografía,
anaqueles, mapa de calor). Las áreas las dibuja el operador en el recuadro
(«Añadir área a ignorar») y llegan en cada frame en `zonas_ignorar`, en
píxeles de la imagen enviada, igual que las áreas de anaquel.

## Cómo se vigila

Cada minuto y por cámara, si hay áreas, una línea en el log:

    IGNORAR [cam] descartadas=40 de 900 detecciones (2 áreas) en 60 s
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Cada cuánto se escribe la línea del log por cámara.
PERIODO_S = 60.0

# camera_id -> [descartadas, total, inicio del periodo, n áreas]
_cuentas: Dict[Any, List[float]] = {}


def poligonos(crudo: Any) -> List[np.ndarray]:
    """Las áreas válidas (3+ puntos numéricos) como arrays float32."""
    salida = []
    for zona in crudo or []:
        try:
            p = np.asarray(zona, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if len(p) >= 3:
            salida.append(p)
    return salida


def dentro_entera(caja: Any, poligono: np.ndarray) -> bool:
    """¿Las cuatro esquinas de la caja `[x1, y1, x2, y2]` están dentro (o en
    el borde) del polígono?"""
    x1, y1, x2, y2 = (float(v) for v in caja[:4])
    return all(cv2.pointPolygonTest(poligono, (x, y), False) >= 0
               for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


def mascara(detecciones: Any, zonas: Any, camera_id: Any = None) -> Any:
    """True por cada detección que NO queda entera dentro de ningún área
    (la que «vale»). Sin áreas o sin detecciones, todas valen. Nunca lanza:
    si algo falla, todas valen. `tiendas.py` usa esto con
    `filtro_memoria.py` (13-oct-2026): ver `filtro_fondo.mascara`."""
    try:
        n = 0 if detecciones is None else len(detecciones)
    except Exception:                                       # noqa: BLE001
        n = 0
    todas = np.ones(n, dtype=bool)
    try:
        polis = poligonos(zonas)
        if not polis or n == 0:
            return todas
        valen = np.array([not any(dentro_entera(c, p) for p in polis)
                          for c in detecciones.xyxy], dtype=bool)
        _anotar(camera_id, int((~valen).sum()), n, len(polis))
        return valen
    except Exception as exc:                                # noqa: BLE001
        logger.debug("zonas_ignorar: no se marcó (%s)", exc)
        return todas


def filtrar(detecciones: Any, zonas: Any, camera_id: Any = None) -> Any:
    """Las detecciones que NO quedan enteras dentro de ninguna área. Sin
    áreas o sin detecciones, devuelve lo mismo que recibe. Nunca lanza: si
    algo falla no se descarta nada (mejor contar de más que perder
    clientes)."""
    try:
        if not poligonos(zonas) or detecciones is None or len(detecciones) == 0:
            return detecciones
        quedan = mascara(detecciones, zonas, camera_id)
        if quedan.all():
            return detecciones
        return detecciones[quedan]
    except Exception as exc:                                # noqa: BLE001
        logger.debug("zonas_ignorar: no se filtró (%s)", exc)
        return detecciones


def _anotar(camera_id: Any, descartadas: int, total: int, n_zonas: int) -> None:
    ahora = time.monotonic()
    c = _cuentas.setdefault(camera_id, [0, 0, ahora, n_zonas])
    c[0] += descartadas
    c[1] += total
    c[3] = n_zonas
    if ahora - c[2] >= PERIODO_S:
        logger.info("IGNORAR [%s] descartadas=%d de %d detecciones (%d área(s)) "
                    "en %.0f s", camera_id, c[0], c[1], c[3], ahora - c[2])
        _cuentas[camera_id] = [0, 0, ahora, n_zonas]
