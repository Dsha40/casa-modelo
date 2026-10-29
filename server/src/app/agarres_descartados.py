"""Agarres descartados a mano tras revisarlos en imagen (15-oct-2026).

## Por qué

Un agarre solo cuenta si lo confirma una bajada de stock, y la bajada mide
«cuánto cambió la imagen», no cuántos productos faltan. El 15-oct, de 25
bajadas, 22 eran falsas revisando sus fotos (el empleado recolocando o
reponiendo, algo tapando el producto, imagen borrosa), y con ellas 21 de los
24 agarres del día. El usuario pidió sacarlos del informe del pasillo.

Los datos del servidor (`<fecha>.json` y el registro de eventos) NO se tocan:
el servidor los reescribe mientras corre y son la medida original. Esta lista
es una corrección aparte, que se puede deshacer borrando su línea.

## Formato

`output/anaqueles/<cámara>/agarres_descartados.json`:

    [{"hora": "2026-10-15 10:25:04", "area": 2, "motivo": "empleado recolocando"}]

`hora` es la del GESTO, la del registro de eventos (`anaqueles_eventos`,
hora local del servidor), y `area` el índice del área (Anaquel 3 = 2). OJO:
el log apunta el agarre cuando lo confirma la bajada, de 2 a 35 s después
del gesto («confirmado por bajada N s después del gesto»): esa hora no
casa. Se admite ±3 s.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

NOMBRE = 'agarres_descartados.json'
MARGEN_S = 3.0


def leer(cam: str) -> List[Dict[str, Any]]:
    """La lista de la cámara, cada uno con su `t` (epoch). Vacía si no hay."""
    from ..analityc.core.anaqueles_registro import _carpeta_nombre, raiz_salida
    ruta = raiz_salida() / _carpeta_nombre(cam) / NOMBRE
    if not ruta.exists():
        return []
    try:
        datos = json.loads(ruta.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning("agarres descartados de %s ilegibles: %s", cam, exc)
        return []
    salida = []
    for d in datos if isinstance(datos, list) else []:
        try:
            t = time.mktime(time.strptime(str(d['hora']), '%Y-%m-%d %H:%M:%S'))
            salida.append(dict(d, t=t, area=int(d['area'])))
        except (KeyError, TypeError, ValueError):
            logger.warning("agarre descartado mal escrito en %s: %r", cam, d)
    return salida


def descartado(t: float, area: int,
               lista: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """(¿está en la lista?, motivo)."""
    for d in lista:
        if d['area'] == int(area) and abs(d['t'] - float(t)) <= MARGEN_S:
            return True, str(d.get('motivo') or '')
    return False, ''
