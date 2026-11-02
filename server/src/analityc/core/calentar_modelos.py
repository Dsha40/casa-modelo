"""src/analityc/core/calentar_modelos.py — calentar las sesiones ONNX nada
más crearlas, para que la primera persona no congele el servidor.

## Por qué (30-oct-2026, rama contructora)

El usuario: «se queda pegado cuando llega gente». Medido:

  * En vivo, la PRIMERA persona tras cada reinicio dejó el servidor sin
    responder ~20-25 s (avisos del cliente a las 13:14:36, 13:29:35 y
    13:56:44; tramos grabados con huecos de 4,2-4,8 s justo después). Mientras
    tanto la gente cruzaba el ROI sin verse, y sus identidades nacían fuera:
    no contaban como visita.
  * Reproducido en frío con cProfile (`scratchpad/reproducir_tramos/
    perfil_primera.py`, tramo de las 13:56): la primera imagen con una
    persona tardó 38 s. De ellos, la PRIMERA llamada de OSNet (huella del
    cuerpo) 29,2 s y la de YuNet (caras) 6,3 s; las siguientes, ~30 y ~12 ms.
  * Banco aparte (`banco_dos_sesiones.py`): con torch cargado ANTES que
    onnxruntime —como en el servidor— la primera llamada de una sesión nueva
    de OSNet tarda ~26 s y la de YuNet ~4,4 s; sin torch, 1,3 s y 0,2 s. No
    es la búsqueda de cuDNN (con HEURISTIC, igual) ni la caché de CUDA en
    disco (se queda vacía). Y es POR SESIÓN: una segunda sesión del mismo
    modelo vuelve a tardar lo mismo. La causa exacta dentro de las librerías
    no se ha averiguado.

Así que se paga al crear el procesador (arranque o reconexión del cliente),
en un hilo aparte, cuando todavía no hay nadie: una llamada con ceros a cada
sesión ONNX que vaya a usar.

## Lo que queda en el log

`CALENTAR: <modelo> lista en N ms` por sesión y `CALENTAR: listo en N s`.

## Interruptor

`CALENTAR_MODELOS` en `config.py`, o `ELDE_CALENTAR=0`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from typing import Any, Dict, List, Tuple

import numpy as np

from .analytics.config import AnalyticsConfig

logger = logging.getLogger(__name__)

#: Las sesiones ya calentadas (o calentándose) en este proceso: las cámaras
#: de una misma conexión comparten sesiones y no se calientan dos veces.
#: Débil, y no por id(): la sesión de una conexión cerrada se libera, y una
#: nueva podría caer en la misma dirección y saltarse el calentamiento.
_hechas: 'weakref.WeakSet' = weakref.WeakSet()
_candado = threading.Lock()


def activo() -> bool:
    v = os.environ.get('ELDE_CALENTAR', '').strip().lower()
    if v:
        return v not in ('0', 'false', 'no', 'off')
    return bool(getattr(AnalyticsConfig, 'CALENTAR_MODELOS', True))


def sesiones_de(proc: Any) -> List[Tuple[str, Any]]:
    """Las sesiones ONNX que usará el procesador, con un nombre para el log."""
    yunet = getattr(proc, '_yunet', None)
    candidatas = [
        ('OSNet', getattr(proc, '_osnet_session', None)),
        ('YuNet', getattr(yunet, '_sess', None)),
        ('ArcFace', getattr(proc, '_face_embedding_session', None)),
        ('InsightFace', getattr(proc, '_insightface_session', None)),
        ('MiVOLO-onnx', getattr(proc, '_mivolo_session', None)),
        ('PAR', getattr(proc, '_par_session', None)),
    ]
    return [(n, s) for n, s in candidatas
            if s is not None and hasattr(s, 'run') and hasattr(s, 'get_inputs')]


def entrada_de_ceros(sesion: Any) -> Dict[str, np.ndarray]:
    """Una entrada de ceros con la forma que pide la sesión (las dimensiones
    libres, como el lote, a 1)."""
    out = {}
    for i in sesion.get_inputs():
        forma = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        tipo = np.float32 if 'float' in str(i.type) else np.int64
        out[i.name] = np.zeros(forma, tipo)
    return out


def calentar(proc: Any, en_hilo: bool = True) -> List[str]:
    """Calienta las sesiones del procesador que aún no lo estén. Devuelve
    sus nombres. Nunca lanza."""
    if not activo():
        return []
    try:
        pendientes = []
        with _candado:
            for nombre, s in sesiones_de(proc):
                if s not in _hechas:
                    _hechas.add(s)
                    pendientes.append((nombre, s))
        if not pendientes:
            return []
        if en_hilo:
            threading.Thread(target=_correr, args=(pendientes,),
                             name='calentar_modelos', daemon=True).start()
        else:
            _correr(pendientes)
        return [n for n, _s in pendientes]
    except Exception as exc:                                # noqa: BLE001
        logger.warning("CALENTAR: no se pudo empezar: %s", exc)
        return []


def _correr(pendientes: List[Tuple[str, Any]]) -> None:
    t0 = time.perf_counter()
    for nombre, s in pendientes:
        t = time.perf_counter()
        try:
            s.run(None, entrada_de_ceros(s))
            logger.info("CALENTAR: %s lista en %.0f ms", nombre,
                        (time.perf_counter() - t) * 1000)
        except Exception as exc:                            # noqa: BLE001
            logger.warning("CALENTAR: %s falló (%s); se calentará con la "
                           "primera persona", nombre, exc)
    logger.info("CALENTAR: listo en %.1f s", time.perf_counter() - t0)
