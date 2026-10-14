"""Descartar a la gente del FONDO por su alto en píxeles (2-oct-2026).

## Para qué

En el pasillo de la cámara 3 se ve, al fondo, gente que no está en el
pasillo: pasa por detrás, en otra zona de la tienda. Se veía pequeña, pero
contaba igual: ficha, entradas, intentos de demografía (los «Persona muy
lejana» del log) y seguimientos. El usuario midió con la regla del cliente
que esa gente queda por debajo de ~235 px de alto, y pidió no tenerla en
cuenta para nada.

## Dónde y por qué ahí

Justo después del detector y ANTES de ByteTrack (`tiendas.py`). Lo que se
quita aquí no llega a nada de lo de detrás: ni seguimiento, ni ficha, ni
conteo de zona, ni demografía, ni anaqueles, ni mapa de calor. Quitarlo más
tarde dejaría seguimientos y calor de gente que no se quiere contar.

## En qué píxeles

En los de la imagen que manda el cliente: el detector trabaja sobre ella sin
redimensionar (`inference_image = image` en `tiendas.py`), y la regla del
cliente mide en esos mismos píxeles. El umbral es POR CÁMARA
(`PERSON_MIN_HEIGHT_PX` en su perfil), porque depende de a qué distancia
está cada cámara del pasillo. En la config global vale 0: apagado.

## Cómo se vigila

Cada minuto y por cámara, una línea en el log con cuántas detecciones se
tiraron, para comprobar que el corte no se come a gente del pasillo:

    FONDO [cam] descartadas=12 de 140 detecciones (alto < 235 px) en 60 s
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

#: Cada cuánto se escribe la línea del log por cámara.
PERIODO_S = 60.0

# camera_id -> [descartadas, total, inicio del periodo]
_cuentas: Dict[Any, List[float]] = {}


def mascara(detecciones: Any, alto_min: Any, camera_id: Any = None) -> Any:
    """True por cada detección con alto >= `alto_min` px (la que «vale»).
    Sin umbral (0/None) o sin detecciones, todas valen. Nunca lanza: si algo
    falla, todas valen (mejor contar de más que perder clientes).

    Desde el 13-oct-2026 `tiendas.py` usa esto y NO `filtrar`: marca, deja
    que ByteTrack siga a todos y quita detrás a quien NUNCA valió
    (`filtro_memoria.py`). Borrar antes partía en dos a quien rondaba el
    umbral (al agacharse o alejarse, su caja bajaba un instante)."""
    try:
        n = 0 if detecciones is None else len(detecciones)
    except Exception:                                       # noqa: BLE001
        n = 0
    todas = np.ones(n, dtype=bool)
    try:
        alto_min = float(alto_min or 0)
        if alto_min <= 0 or n == 0:
            return todas
        xyxy = detecciones.xyxy
        valen = (xyxy[:, 3] - xyxy[:, 1]) >= alto_min
        _anotar(camera_id, int((~valen).sum()), n, alto_min)
        return valen
    except Exception as exc:                                # noqa: BLE001
        logger.debug("filtro_fondo: no se marcó (%s)", exc)
        return todas


def filtrar(detecciones: Any, alto_min: Any, camera_id: Any = None) -> Any:
    """Las detecciones con alto >= `alto_min` px. Con `alto_min` 0/None, o
    sin detecciones, devuelve lo mismo que recibe. Nunca lanza: si algo
    falla, no se descarta nada (mejor contar de más que perder clientes)."""
    try:
        alto_min = float(alto_min or 0)
        if alto_min <= 0 or detecciones is None or len(detecciones) == 0:
            return detecciones
        quedan = mascara(detecciones, alto_min, camera_id)
        if quedan.all():
            return detecciones
        return detecciones[quedan]
    except Exception as exc:                                # noqa: BLE001
        logger.debug("filtro_fondo: no se filtró (%s)", exc)
        return detecciones


def _anotar(camera_id: Any, descartadas: int, total: int, alto_min: float) -> None:
    ahora = time.monotonic()
    c = _cuentas.setdefault(camera_id, [0, 0, ahora])
    c[0] += descartadas
    c[1] += total
    if ahora - c[2] >= PERIODO_S:
        logger.info("FONDO [%s] descartadas=%d de %d detecciones (alto < %.0f px) en %.0f s",
                    camera_id, c[0], c[1], alto_min, ahora - c[2])
        _cuentas[camera_id] = [0, 0, ahora]
