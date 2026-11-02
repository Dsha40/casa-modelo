"""analytics/aprender_cara.py — la galería aprende la cara durante la visita.

## Por qué (30-oct-2026, rama contructora)

Hasta este día, `FaceReidentifier.identify_or_register` guardaba una
plantilla (el embedding ArcFace de una cara) solo al crear a la persona o al
reconocerla en un seguimiento NUEVO. Mientras el seguimiento seguía vivo, sus
caras siguientes no se guardaban: si una persona tenía identidad, la galería
se quedaba con la cara de la primera vez (a menudo de lejos o de lado).

El usuario pidió reconocer a la gente durante 24 h y «que tome varias
imágenes de esa persona y aprenda». No hace falta entrenar un modelo: ArcFace
ya sabe comparar caras. Lo que aprende es la COLECCIÓN de plantillas de cada
persona: de frente, algo girada, con otra luz.

## Cómo

Con cada cara de calidad de un seguimiento que ya tiene identidad:

  * como mucho una vez por seguimiento cada `REID_APRENDER_CADA_S`;
  * solo con la pose que se exige para registrar a alguien nuevo
    (`REID_MAX_YAW_FOR_REGISTRATION` y `..._PITCH_...`);
  * solo si la cara se parece a las plantillas que ya tiene la persona por
    encima de `REID_APRENDER_SIM_MIN`. Si el número de seguimiento saltó a
    otra persona (pasa en el 11-15 % de los etiquetados, CLAUDE.md 15.6),
    su cara no se parece y no se mezcla.

La plantilla entra con `_add_embedding_diverse`, que al llenarse sustituye a
la más parecida a la nueva y así conserva la variedad.

## Lo que queda en el log

Cada 60 s, una línea `CARA APRENDE` con cuántas plantillas se añadieron,
cuántas se rechazaron por no parecerse y el parecido mínimo, mediano y máximo
de las aceptadas y de las rechazadas. Sirve para revisar el 0,40, que se
eligió sin calibrar.

## Interruptor

`REID_APRENDER_EN_VISITA` en `config.py`, o `ELDE_REID_APRENDER=0`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .config import AnalyticsConfig

logger = logging.getLogger(__name__)

#: Cada cuánto se escribe la línea `CARA APRENDE` en el log, como mucho.
RESUMEN_CADA_S = 60.0

_candado = threading.Lock()
_aceptadas: List[float] = []
_rechazadas: List[float] = []
_ultimo_resumen = time.time()


def activo() -> bool:
    """`ELDE_REID_APRENDER` manda sobre la configuración."""
    v = os.environ.get('ELDE_REID_APRENDER', '').strip().lower()
    if v:
        return v not in ('0', 'false', 'no', 'off')
    return bool(getattr(AnalyticsConfig, 'REID_APRENDER_EN_VISITA', True))


def parecido_con(plantillas: List[np.ndarray], emb: np.ndarray) -> float:
    """Máximo coseno entre `emb` y las plantillas de UNA persona."""
    if not plantillas:
        return -1.0
    m = np.asarray(plantillas, dtype=np.float32).reshape(len(plantillas), -1)
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    q = np.asarray(emb, dtype=np.float32).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    return float(np.max(m @ q))


def _pose_valida(yaw: Optional[float], pitch: Optional[float]) -> bool:
    if yaw is not None and abs(yaw) > AnalyticsConfig.REID_MAX_YAW_FOR_REGISTRATION:
        return False
    if (pitch is not None
            and abs(pitch) > AnalyticsConfig.REID_MAX_PITCH_FOR_REGISTRATION):
        return False
    return True


def aprender_en_visita(reid: Any, track_id: int, rec: Dict[str, Any],
                       face_bgr: np.ndarray, yaw: Optional[float],
                       pitch: Optional[float],
                       ahora: Optional[float] = None) -> Optional[float]:
    """Quizá añade la cara a las plantillas de `rec`.

    El llamador (`FaceReidentifier.identify_or_register`) tiene el lock de la
    galería. Devuelve el parecido medido, o None si no tocaba medirlo (otro
    intento hace menos de `REID_APRENDER_CADA_S`, pose lateral, apagado o
    sin embedding)."""
    # `demographics.py` llama sin cara (array vacío) solo para actualizar el
    # género de quien ya tiene identidad: eso no gasta el turno.
    if not activo() or face_bgr is None or getattr(face_bgr, 'size', 0) == 0:
        return None
    ahora = time.time() if ahora is None else float(ahora)
    tiempos = reid._aprendido_ts
    ultimo = tiempos.get(int(track_id))
    if ultimo is not None and ahora - ultimo < float(
            AnalyticsConfig.REID_APRENDER_CADA_S):
        return None
    if not _pose_valida(yaw, pitch):
        return None
    # Se anota antes de calcular: la espera también ahorra el embedding
    # (~ms de ArcFace) cuando la cara no se va a aceptar.
    tiempos[int(track_id)] = ahora
    if len(tiempos) > 5000:                     # números de hace rato
        for t in [t for t, ts in tiempos.items() if ahora - ts > 60.0]:
            del tiempos[t]
    emb = reid.compute_embedding(face_bgr)
    if emb is None:
        return None
    from .face_reidentifier import _add_embedding_diverse, _get_embeddings
    sim = parecido_con(_get_embeddings(rec), emb)
    if sim >= float(AnalyticsConfig.REID_APRENDER_SIM_MIN):
        _add_embedding_diverse(
            rec, emb, int(AnalyticsConfig.REID_MAX_EMBEDDINGS_PER_PERSON))
        rec['embedding_count'] = rec.get('embedding_count', 1) + 1
        _anotar(sim, True, ahora)
    else:
        _anotar(sim, False, ahora)
    return sim


def _anotar(sim: float, aceptada: bool, ahora: float) -> None:
    global _ultimo_resumen
    with _candado:
        (_aceptadas if aceptada else _rechazadas).append(float(sim))
        if ahora - _ultimo_resumen < RESUMEN_CADA_S:
            return
        logger.info("CARA APRENDE: añadidas=%d %s | rechazadas=%d %s "
                    "(umbral %.2f, en %.0f s)",
                    len(_aceptadas), _rango(_aceptadas),
                    len(_rechazadas), _rango(_rechazadas),
                    float(AnalyticsConfig.REID_APRENDER_SIM_MIN),
                    ahora - _ultimo_resumen)
        _aceptadas.clear()
        _rechazadas.clear()
        _ultimo_resumen = ahora


def _rango(sims: List[float]) -> str:
    if not sims:
        return ''
    return 'parecido min/mediana/max=%.2f/%.2f/%.2f' % (
        min(sims), float(np.median(sims)), max(sims))
