"""src/analityc/core/origen_identidad.py — dónde nació cada identidad.

## Por qué (30-oct-2026, rama contructora)

Casa modelo, cámara 4. El usuario dibujó el ROI grande, cubriendo la zona por
la que entra la gente, para que dé tiempo a crear bien su identidad. Después
la gente recorre la casa y entra en cuartos que la cámara no ve; al volver,
el reconocimiento puede fallar y crearle OTRA identidad. Esa segunda nace
fuera del ROI y no es una visita nueva. En palabras del usuario: «si la
identidad se creó fuera del ROI […] esta no cuenta como visita».

Así que las identidades se crean en cualquier sitio (cara o cuerpo) y cada
una apunta aquí si nació DENTRO del ROI. Solo esas cuentan para las visitas:
entradas y salidas (`tienda_visitas`), visitantes únicos y el aviso de
«visitante». Las nacidas fuera se siguen detectando, siguiendo y
clasificando.

## Lo que no se sabe

Una identidad sin origen apuntado (la galería de caras viene del disco tras
un reinicio y este registro vive en RAM) CUENTA: es lo que pasaba antes de
este cambio, y preferimos eso a dejar de contar a alguien que sí entró.

Sin ROI activo todas nacen «dentro»: todo cuenta, como antes.

## Interruptor

`VISITAS_SOLO_NACIDAS_EN_ROI` en `config.py`, o `ELDE_VISITAS_SOLO_ROI=0`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from .analytics.config import AnalyticsConfig

logger = logging.getLogger(__name__)

_candado = threading.Lock()
#: pid -> (nació dentro del ROI, cuándo)
_origen: Dict[str, Tuple[bool, float]] = {}


def activo() -> bool:
    v = os.environ.get('ELDE_VISITAS_SOLO_ROI', '').strip().lower()
    if v:
        return v not in ('0', 'false', 'no', 'off')
    return bool(getattr(AnalyticsConfig, 'VISITAS_SOLO_NACIDAS_EN_ROI', False))


def anotar(pid: Any, dentro: bool, como: str = '',
           ahora: Optional[float] = None) -> None:
    """Apunta dónde nació `pid`. Solo la primera vez: nacer es una vez."""
    if not pid or str(pid).startswith('prov:'):
        return
    ahora = time.time() if ahora is None else float(ahora)
    with _candado:
        _caducar(ahora)
        if str(pid) in _origen:
            return
        _origen[str(pid)] = (bool(dentro), ahora)
    logger.info("ORIGEN persona=%s nace %s del ROI (%s)", str(pid)[:8],
                'DENTRO' if dentro else 'FUERA', como)


def heredar(de: Any, a: Any) -> None:
    """`a` es la misma persona que `de` (el cuerpo pasa a la cara): si `a`
    no tiene origen, se queda con el de `de`."""
    with _candado:
        if de and a and str(a) not in _origen and str(de) in _origen:
            _origen[str(a)] = _origen[str(de)]


def nacio_dentro(pid: Any) -> Optional[bool]:
    with _candado:
        o = _origen.get(str(pid))
        return None if o is None else o[0]


def cuenta(pid: Any) -> bool:
    """¿Cuenta `pid` para las visitas? Las nacidas fuera del ROI, no; las
    de origen desconocido, sí."""
    if not activo() or not pid:
        return True
    return nacio_dentro(pid) is not False


def _caducar(ahora: float) -> None:
    ttl = float(getattr(AnalyticsConfig, 'REID_TTL_S', 86400.0))
    for pid in [p for p, (_d, t) in _origen.items() if ahora - t > ttl]:
        del _origen[pid]
