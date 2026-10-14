"""Filtro del fondo y áreas a ignorar CON MEMORIA (13-oct-2026).

## El fallo

`filtro_fondo` (alto mínimo en px) y `zonas_ignorar` (áreas dibujadas)
BORRABAN las cajas antes de ByteTrack. Quien rondaba el umbral (al
agacharse, quedar medio tapado o alejarse, su caja baja un momento) se
perdía unos instantes, ByteTrack seguía «imaginándolo» en movimiento y al
volver le daba un NÚMERO NUEVO: otra ficha, otra entrada al área, el género
otra vez desde cero y a veces otra identidad. En la grabación del 8-oct
(filtro a 200 px), 57 de 405 números morían y renacían al lado en menos de
5 s (la mitad en 2,2 s); al perderse, la mitad de esas cajas medía 222 px o
menos (personas normales, 282). Con las etiquetas a mano, 20 de 22 eran la
misma persona. El usuario lo vio en vivo: dos fichas casi iguales, «la
continuación de un movimiento».

## El arreglo

Los filtros ya no borran: MARCAN cada caja (`filtro_fondo.mascara`,
`zonas_ignorar.mascara`) y ByteTrack sigue a todos. Justo después,
`filtrar` deja solo a los números que han sido válidos ALGUNA VEZ: quien
nunca lo fue (gente del fondo) sigue sin contar para nada, como antes; quien
lo fue no se pierde porque su caja encoja un momento.

Coste medido (13-oct): ByteTrack tarda ~0,25 ms más por caja; esta mañana
el filtro y el área quitaban 0,89 cajas por imagen → ~0,2 ms de ~53 (0,4 %).

La memoria vive en el propio ByteTrack de cada cámara (`tracker`), que
`TiendasProcess._push_state` ya intercambia por cámara.

Cada minuto y por cámara, si rescató algo, una línea en el log:

    MEMORIA [cam] conservadas=N cajas de personas ya válidas que el filtro habría quitado en 60 s
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

#: Clave de la marca en `Detections.data` (ByteTrack la conserva).
CLAVE = 'elde_valida'
#: Imágenes sin ver un número antes de olvidar que fue válido (ByteTrack lo
#: olvida mucho antes: 9 s); solo acota la memoria.
OLVIDAR_TRAS = 20000
PERIODO_S = 60.0

# camera_id -> [conservadas, inicio del periodo]
_cuentas: Dict[Any, List[float]] = {}


def marcar(detecciones: Any, valida: Any) -> Any:
    """Apunta en cada detección si vale (array de bool). Nunca lanza."""
    try:
        if detecciones is not None and len(detecciones) > 0:
            detecciones.data[CLAVE] = np.asarray(valida, dtype=bool)
    except Exception as exc:                                # noqa: BLE001
        logger.debug("filtro_memoria: no se marcó (%s)", exc)
    return detecciones


def filtrar(tracker: Any, detecciones: Any, camera_id: Any = None) -> Any:
    """Tras ByteTrack: deja las detecciones cuyo número ha sido válido
    alguna vez. Sin marca, todas pasan (como sin filtros). Nunca lanza: si
    algo falla se cae a lo de antes (solo las válidas de esta imagen)."""
    if detecciones is None or len(detecciones) == 0:
        return detecciones
    valida = detecciones.data.get(CLAVE) if hasattr(detecciones, 'data') else None
    if valida is None:
        return detecciones
    try:
        ids = detecciones.tracker_id
        if ids is None:
            return detecciones[np.asarray(valida, dtype=bool)]
        memoria = getattr(tracker, '_elde_validos', None)
        if memoria is None:
            memoria = {}
            tracker._elde_validos = memoria
        n = getattr(tracker, '_elde_n', 0) + 1
        tracker._elde_n = n
        quedan = np.zeros(len(detecciones), dtype=bool)
        for i, (tid, v) in enumerate(zip(ids, valida)):
            tid = int(tid)
            if v or tid in memoria:
                memoria[tid] = n
                quedan[i] = True
        if len(memoria) > 500:
            for tid in [t for t, ultimo in memoria.items() if n - ultimo > OLVIDAR_TRAS]:
                memoria.pop(tid, None)
        _anotar(camera_id, int((quedan & ~np.asarray(valida, dtype=bool)).sum()))
        if quedan.all():
            return detecciones
        return detecciones[quedan]
    except Exception as exc:                                # noqa: BLE001
        logger.debug("filtro_memoria: sin memoria esta imagen (%s)", exc)
        try:
            return detecciones[np.asarray(valida, dtype=bool)]
        except Exception:                                   # noqa: BLE001
            return detecciones


def _anotar(camera_id: Any, conservadas: int) -> None:
    ahora = time.monotonic()
    c = _cuentas.setdefault(camera_id, [0, ahora])
    c[0] += conservadas
    if ahora - c[1] >= PERIODO_S:
        if c[0]:
            logger.info("MEMORIA [%s] conservadas=%d cajas de personas ya válidas "
                        "que el filtro habría quitado en %.0f s",
                        camera_id, c[0], ahora - c[1])
        _cuentas[camera_id] = [0, ahora]
