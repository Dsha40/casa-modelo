"""Pausar durante los cortes de vídeo de iVMS (13-oct-2026). OPCIONAL.

## El fallo

iVMS pierde el vídeo de la cámara 3 cada pocos minutos durante ~30 s (118
cortes en horario de tienda del 12-oct 17h al 13-oct 13h, mediana 29 s;
10-15 por hora). Mientras, el programa de la tienda manda SIEMPRE la misma
imagen fija (~39 KB; la misma de las noches). Nadie sale en ella: pasados
9 s ByteTrack olvida a quien estaba en imagen y, al volver el vídeo, la
misma persona recibe un número nuevo (otra ficha, otra entrada, el género
desde cero, a veces otra identidad).

## El parche (apagado por defecto)

Con `ELDE_CONGELAR_CORTES=1` (o `AnalyticsConfig.CONGELAR_EN_CORTES`): si
llega la MISMA imagen exacta durante `MIN_S` o más, el servidor responde
sin procesarla (la última respuesta, sin cajas) y no corre el reloj de
nadie: ByteTrack no avanza, nadie se da por perdido. Al volver el vídeo,
quien siguió en el mismo sitio conserva su número. El vídeo normal nunca
repite la imagen exacta más de un instante (iVMS refresca a ~9 img/s y el
cliente copia a ~18: como mucho 1 repetida seguida).

  * `MIN_S` = 0,4 s: antes de los 0,65 s en que `tiendas.py` da a alguien
    por perdido.
  * Al volver, los géneros pendientes de borrar (`demografia_aplazada`)
    se alargan lo que duró la pausa.
  * Si la pausa duró más de `MAX_S` (p. ej. la noche), al volver se
    reinicia el ByteTrack: no se le pega a alguien de mañana el número de
    alguien de anoche.

Cada corte deja dos líneas en el log:

    CORTE [cam] imagen congelada: seguimiento en pausa
    CORTE [cam] vuelve el vídeo tras 29 s (430 imágenes sin procesar)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

from . import demografia_aplazada

logger = logging.getLogger(__name__)

MIN_S = 0.4
MAX_S = 120.0
_CLAVE = '_corte_video'
_ULTIMO = '_ultimo_metadata'


def activo() -> bool:
    """Apagado salvo ELDE_CONGELAR_CORTES=1 o CONGELAR_EN_CORTES = True."""
    valor = os.environ.get('ELDE_CONGELAR_CORTES', '').strip().lower()
    if valor:
        return valor in ('1', 'true', 'si', 'sí', 'on')
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'CONGELAR_EN_CORTES', False))
    except Exception:                                       # noqa: BLE001
        return False


def _huella(imagen: np.ndarray) -> bytes:
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen
    chica = cv2.resize(gris, (64, 36), interpolation=cv2.INTER_AREA)
    return hashlib.md5(chica.tobytes()).digest()


def congelada(estado: Dict[str, Any], imagen: np.ndarray, ahora: float,
              demografia: Any = None, tracker: Any = None) -> bool:
    """True si esta imagen es parte de un corte (la misma exacta desde hace
    `MIN_S` o más) y no debe procesarse. `estado` es el de la cámara. Nunca
    lanza: ante un fallo, se procesa (False)."""
    try:
        est = estado.setdefault(_CLAVE, {'huella': None, 'desde': 0.0,
                                         'pausa_desde': None, 'imagenes': 0})
        h = _huella(imagen)
        if h == est['huella']:
            if ahora - est['desde'] < MIN_S:
                return False
            if est['pausa_desde'] is None:
                est['pausa_desde'] = ahora
                est['imagenes'] = 0
                logger.info("CORTE [%s] imagen congelada: seguimiento en pausa",
                            estado.get('_camera_id', '?'))
            est['imagenes'] += 1
            return True
        if est['pausa_desde'] is not None:
            dur = ahora - est['pausa_desde']
            demografia_aplazada.desplazar(demografia, dur)
            if dur > MAX_S and tracker is not None and hasattr(tracker, 'reset'):
                tracker.reset()
            logger.info("CORTE [%s] vuelve el vídeo tras %.0f s (%d imágenes sin "
                        "procesar)%s", estado.get('_camera_id', '?'), dur,
                        est['imagenes'],
                        '; seguimiento reiniciado' if dur > MAX_S else '')
            est['pausa_desde'] = None
        est['huella'] = h
        est['desde'] = ahora
        return False
    except Exception as exc:                                # noqa: BLE001
        logger.debug("cortes_video: sin comprobar (%s)", exc)
        return False


def recordar(estado: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> None:
    """Guarda la última respuesta de la cámara (para repetirla en un corte)."""
    if isinstance(metadata, dict):
        estado[_ULTIMO] = metadata


def metadata_congelada(estado: Dict[str, Any]) -> Dict[str, Any]:
    """La última respuesta, sin cajas ni alertas (ya se enviaron)."""
    m = dict(estado.get(_ULTIMO) or {})
    m['alerts'] = []
    m['detections'] = []
    m['congelado'] = True
    return m
