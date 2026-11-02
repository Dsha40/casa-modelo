"""src/analityc/core/identidad_cuerpo.py — identidad por el cuerpo cuando no
se ve la cara.

## Por qué (30-oct-2026, rama contructora)

En la cámara 4 (casa modelo, cámara alta y lejos) la cara casi nunca sirve:
de 10:15 a 10:23, 2 caras aceptadas frente a ~175 rechazos, 154 de ellos por
persona demasiado pequeña. Sin cara, cada entrada era un `prov:<número>`: la
misma persona que entraba y salía contaba como otra cada vez, y su género se
decidía por separado en cada entrada. El usuario: «no se va a ver bien la
cara, vamos a usar el cuerpo».

Hasta este día la huella del cuerpo (OSNet, `BodyReidentifier`) NUNCA creaba
identidades: solo volvía a enlazar a quien ya tenía cara. Aquí sí crea una.

## Cómo

Un seguimiento sin cara y sin parecido con nadie de la galería del cuerpo va
juntando sus huellas (una cada `BODY_REID_EVERY_N` imágenes). Con
`BODY_ID_MIN_HUELLAS` huellas que se parecen entre sí (coseno mínimo entre
ellas ≥ `BODY_ID_COHERENCIA_MIN`: si el número saltó de una persona a otra a
mitad, no se crea nada), recibe una identidad nueva y sus huellas entran en
la galería del cuerpo con ella. La próxima vez que entre, `body.match` la
encuentra como a cualquier otra.

Si más tarde se le ve la cara, la galería del cuerpo pasa a llamarse con la
identidad de la cara (`unir_con_cara`): desde ahí cuenta como esa persona.
Las visitas ya guardadas con la identidad del cuerpo se quedan como estaban.

## Riesgos

  * Dos personas con ropa parecida (el personal con el mismo uniforme) se
    juntan. El uniforme de personal se marca aparte (`clasificador_uniforme`).
  * La ropa cambia de un día a otro: la galería del cuerpo caduca a las
    `BODY_REID_TTL_S`.
  * Los umbrales no están calibrados: cada 60 s el log escribe una línea
    `CUERPO ID` con las identidades creadas y los parecidos medidos.

## Interruptor

`BODY_ID_CREAR` en `config.py`, o `ELDE_BODY_ID=0`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .analytics.config import AnalyticsConfig

logger = logging.getLogger(__name__)

#: Huellas guardadas por seguimiento mientras espera identidad.
MAX_HUELLAS_TRACK = 8
RESUMEN_CADA_S = 60.0

_candado = threading.Lock()
_creadas: List[float] = []        # coherencia de las identidades creadas
_incoherentes: List[float] = []   # coherencia de las que no se crearon
_enlaces: List[float] = []        # parecido de las huellas que enlazaron
_sin_enlace: List[float] = []     # mejor parecido de las que no enlazaron
_ultimo_resumen = time.time()
#: Identidades que solo tienen cuerpo (aún no se les vio la cara).
_solo_cuerpo: set = set()


def activo() -> bool:
    v = os.environ.get('ELDE_BODY_ID', '').strip().lower()
    if v:
        return v not in ('0', 'false', 'no', 'off')
    return bool(getattr(AnalyticsConfig, 'BODY_ID_CREAR', False))


def guardar_huella(track: Dict[str, Any], emb: np.ndarray) -> None:
    """Apunta una huella del seguimiento (las últimas MAX_HUELLAS_TRACK)."""
    if emb is None:
        return
    lista = track.setdefault('_huellas_cuerpo', [])
    lista.append(np.asarray(emb, dtype=np.float32))
    del lista[:-MAX_HUELLAS_TRACK]


def coherencia(huellas: List[np.ndarray]) -> float:
    """Coseno MÍNIMO entre cualquier par de huellas (1.0 con una sola)."""
    if len(huellas) < 2:
        return 1.0
    m = np.stack(huellas)
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    sims = m @ m.T
    return float(np.min(sims))


def buscar(body: Any, emb: np.ndarray, ahora: Optional[float] = None,
           ocupadas: Any = (), info: Optional[Dict[str, Any]] = None
           ) -> Tuple[Optional[str], Optional[float]]:
    """La identidad del cuerpo que corresponde a `emb`, con umbral según
    cuándo se vio a esa persona por última vez (30-oct-2026, idea del
    usuario: quien acaba de salir se recupera con menos exigencia).

    Devuelve (identidad o None, mejor parecido o None).

    Simulado sobre 18 tramos de UNA persona (cámara 4, 10:15-10:39): con
    0,90 fijo salían 11 identidades o más; con 0,70 durante 10 min y 0,80
    después, 2 (y 47 de 54 pasadas reconocidas). Lo que no mide: si junta
    a dos personas distintas (en esos tramos solo había una).

    `ocupadas`: identidades que ahora mismo tiene OTRO seguimiento visible.
    Una persona no puede estar dos veces en la imagen, así que no se le
    pueden dar a este (30-oct-2026: a las 12:22, con tres personas a la vez,
    el cuerpo juntó a dos distintas con un parecido de 0,73).

    `info`: si se pasa, recibe `info['ocupada']` = la identidad OCUPADA que
    habría pasado el umbral (o None). Con ella el llamador espera en vez de
    crear otra identidad: a las 13:35, una mujer medio tapada se partió en
    dos seguimientos, el segundo no podía tener su identidad y se creó otra
    (`4f2abe01`) con su ficha sobrante."""
    ocupadas = set(ocupadas or ())
    ahora = time.time() if ahora is None else float(ahora)
    libre = (None, -1.0, 0.0)          # (identidad, parecido, última vez)
    ocupada = (None, -1.0, 0.0)
    with body._lock:
        body._prune_locked(ahora)
        for p, lst in body._gallery.items():
            if not lst:
                continue
            s = max(float(np.dot(e, emb)) for e, _t in lst)
            visto = max(t for _e, t in lst)
            if p in ocupadas:
                if s > ocupada[1]:
                    ocupada = (p, s, visto)
            elif s > libre[1]:
                libre = (p, s, visto)
    if info is not None:
        info['ocupada'] = (ocupada[0] if ocupada[0] is not None and
                           ocupada[1] >= _umbral(ocupada[2], ahora) else None)
    pid, sim, ultimo = libre
    if pid is None:
        return None, None
    return (pid if sim >= _umbral(ultimo, ahora) else None), float(sim)


def _umbral(ultimo: float, ahora: float) -> float:
    """0,70 si la identidad se vio hace menos de la ventana; 0,80 si no."""
    reciente = ahora - ultimo < float(
        AnalyticsConfig.BODY_REID_VENTANA_RECIENTE_S)
    return float(AnalyticsConfig.BODY_REID_UMBRAL_RECIENTE if reciente
                 else AnalyticsConfig.BODY_REID_SIMILARITY_THRESHOLD)


def quizas_crear(body: Any, track: Dict[str, Any],
                 ahora: Optional[float] = None) -> Optional[str]:
    """Crea la identidad del cuerpo de este seguimiento si ya toca.

    El llamador ya comprobó que no tiene cara ni parecido en la galería.
    Se crea dentro o fuera del ROI: si cuenta como visita lo decide
    `origen_identidad.py` (30-oct-2026). Devuelve la identidad nueva o
    None."""
    if not activo() or body is None:
        return None
    huellas = track.get('_huellas_cuerpo') or []
    if len(huellas) < int(AnalyticsConfig.BODY_ID_MIN_HUELLAS):
        return None
    ahora = time.time() if ahora is None else float(ahora)
    coh = coherencia(huellas)
    if coh < float(AnalyticsConfig.BODY_ID_COHERENCIA_MIN):
        # Se vuelve a intentar con las siguientes: se tira la más vieja.
        del huellas[0]
        _anotar(coh, False, ahora)
        return None
    pid = str(uuid.uuid4())
    for h in huellas:
        body.enroll(pid, h)
    with _candado:
        _solo_cuerpo.add(pid)
    _anotar(coh, True, ahora)
    logger.info("CUERPO NUEVA persona=%s (%d huellas, coherencia %.2f)",
                pid[:8], len(huellas), coh)
    return pid


def es_solo_cuerpo(pid: Any) -> bool:
    with _candado:
        return pid in _solo_cuerpo


def unir_con_cara(body: Any, track: Dict[str, Any], uid: str) -> None:
    """Un seguimiento con identidad del cuerpo (creada por él o por otro)
    acaba de ver su cara: su galería del cuerpo pasa a ser la de la cara."""
    viejo = track.get('_body_match')
    if not viejo or not uid or viejo == uid or body is None \
            or not es_solo_cuerpo(viejo):
        return
    with _candado:
        _solo_cuerpo.discard(viejo)
    track['_body_match'] = uid
    try:
        with body._lock:
            lst = body._gallery.pop(viejo, [])
            body._gallery.setdefault(uid, []).extend(lst)
            maximo = int(getattr(body, '_max', len(body._gallery[uid])))
            del body._gallery[uid][:-maximo]
        logger.info("CUERPO %s pasa a ser la cara %s", viejo[:8], uid[:8])
    except Exception as exc:                                # noqa: BLE001
        logger.debug("identidad_cuerpo: unir falló: %s", exc)


def anotar_busqueda(sim: Optional[float], enlazado: bool,
                    ahora: Optional[float] = None) -> None:
    """Apunta el MEJOR parecido de una huella con la galería, haya enlazado
    o no: es lo que dirá si los umbrales del cuerpo están bien puestos."""
    if sim is None:
        return
    with _candado:
        (_enlaces if enlazado else _sin_enlace).append(float(sim))
    _resumen(time.time() if ahora is None else float(ahora))


def _anotar(coh: float, creada: bool, ahora: float) -> None:
    with _candado:
        (_creadas if creada else _incoherentes).append(float(coh))
    _resumen(ahora)


def _resumen(ahora: float) -> None:
    global _ultimo_resumen
    with _candado:
        if ahora - _ultimo_resumen < RESUMEN_CADA_S:
            return
        if _creadas or _incoherentes or _enlaces or _sin_enlace:
            logger.info(
                "CUERPO ID: creadas=%d %s | no creadas por incoherentes=%d %s"
                " (umbral %.2f) | enlaces=%d %s | sin enlace=%d mejor %s"
                " (umbral %.2f reciente / %.2f) | en %.0f s",
                len(_creadas), _rango(_creadas),
                len(_incoherentes), _rango(_incoherentes),
                float(AnalyticsConfig.BODY_ID_COHERENCIA_MIN),
                len(_enlaces), _rango(_enlaces),
                len(_sin_enlace), _rango(_sin_enlace),
                float(AnalyticsConfig.BODY_REID_UMBRAL_RECIENTE),
                float(AnalyticsConfig.BODY_REID_SIMILARITY_THRESHOLD),
                ahora - _ultimo_resumen)
        for lista in (_creadas, _incoherentes, _enlaces, _sin_enlace):
            lista.clear()
        _ultimo_resumen = ahora


def _rango(v: List[float]) -> str:
    if not v:
        return ''
    return 'min/mediana/max=%.2f/%.2f/%.2f' % (
        min(v), float(np.median(v)), max(v))
