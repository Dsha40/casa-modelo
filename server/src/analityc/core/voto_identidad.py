"""src/analityc/core/voto_identidad.py — un solo género por persona.

## Por qué (30-oct-2026, rama contructora)

El clasificador B decide el género de cada SEGUIMIENTO por separado y, a
propósito, no hereda el de la identidad (`clasificador_b.py`: 8 de los 9
errores del clásico eran géneros heredados de un reconocimiento equivocado).
Pero la fila de la visita (`tienda_almacen.anotar_demografia`) se queda con
el ÚLTIMO género que llega: si una persona entra tres veces y una sale
«Mujer» y dos «Hombre», su visita cambia de género cada vez. Lo vio el
usuario en la cámara 4.

Aquí cada seguimiento decidido es UN voto para su identidad (cara o cuerpo,
`identidad_cuerpo.py`), y lo que se guarda y se avisa es la mayoría. No se
cambia lo que el clasificador decidió para el seguimiento: solo lo que se
escribe de la persona.

En empate gana el voto más reciente (es lo que había antes).

## Memoria

En RAM, con la misma vida que la galería de caras (`REID_TTL_S`, 24 h desde
que se vio por última vez). Un reinicio la vacía: la persona vuelve a
empezar a votar.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

from .analytics.config import AnalyticsConfig

_candado = threading.Lock()
#: pid -> {'votos': {stem: (genero, ts)}, 'visto': ts}
_personas: Dict[str, Dict[str, Any]] = {}


def votar(pid: Any, stem: Any, genero: Optional[str],
          ahora: Optional[float] = None) -> Optional[str]:
    """Apunta el voto de un seguimiento (`stem`, uno por ficha) y devuelve el
    género de la MAYORÍA de esa persona. Sin `pid` devuelve el voto tal cual."""
    g = (genero or '').strip()
    if not pid or not g or g.lower() == 'desconocido':
        return genero
    ahora = time.time() if ahora is None else float(ahora)
    with _candado:
        _caducar(ahora)
        rec = _personas.setdefault(str(pid), {'votos': {}, 'visto': ahora})
        rec['votos'][str(stem)] = (g, ahora)
        rec['visto'] = ahora
        return _mayoria(rec['votos'])


def genero_de(pid: Any) -> Optional[str]:
    with _candado:
        rec = _personas.get(str(pid))
        return _mayoria(rec['votos']) if rec else None


def votos_de(pid: Any) -> Dict[str, int]:
    with _candado:
        rec = _personas.get(str(pid)) or {'votos': {}}
        cuenta: Dict[str, int] = {}
        for g, _t in rec['votos'].values():
            cuenta[g] = cuenta.get(g, 0) + 1
        return cuenta


def _mayoria(votos: Dict[str, Tuple[str, float]]) -> Optional[str]:
    cuenta: Dict[str, Tuple[int, float]] = {}
    for g, t in votos.values():
        n, ultimo = cuenta.get(g, (0, 0.0))
        cuenta[g] = (n + 1, max(ultimo, t))
    if not cuenta:
        return None
    return max(cuenta.items(), key=lambda kv: kv[1])[0]


def _caducar(ahora: float) -> None:
    ttl = float(getattr(AnalyticsConfig, 'REID_TTL_S', 86400.0))
    for pid in [p for p, r in _personas.items() if ahora - r['visto'] > ttl]:
        del _personas[pid]
