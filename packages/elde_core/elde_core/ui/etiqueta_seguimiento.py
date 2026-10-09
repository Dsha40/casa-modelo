"""elde_core/ui/etiqueta_seguimiento.py — el número de seguimiento en la caja.

## Para qué (6-oct-2026)

La clasificación (edad, género, cara) junta muestras de la MISMA persona a
lo largo de varias imágenes: si ByteTrack le cambia el número a mitad de
camino, empieza de cero. Para comprobar a ojo que la sigue bien, el
recuadro puede pintar `#<tracker_id>` delante de la etiqueta de cada caja
(menú Vista → «IDs de seguimiento»). El servidor ya mandaba ese número en
el modo directo (`tracker_id`, `tiendas_dibujo._build_detections`); solo
faltaba pintarlo.

Qué mirar: una persona que camina debería conservar su número. Si cambia
de número al cruzarse con otra o al taparse un momento, ByteTrack la ha
perdido; si dos personas intercambian números, las ha confundido.
"""
from __future__ import annotations

from typing import Any, Dict, List


def con_id(detecciones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copia de `detecciones` con `#<tracker_id>` delante de cada etiqueta.

    No toca la lista original (la usan otras partes del recuadro). Una
    detección sin `tracker_id` queda igual.
    """
    salida: List[Dict[str, Any]] = []
    for d in detecciones or []:
        tid = d.get('tracker_id') if isinstance(d, dict) else None
        if tid is None:
            salida.append(d)
            continue
        nueva = dict(d)
        etiqueta = str(d.get('label') or '')
        nueva['label'] = f'#{tid} {etiqueta}'.rstrip()
        salida.append(nueva)
    return salida
