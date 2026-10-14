"""Conservar el género mientras el seguimiento recuerde el número
(12-oct-2026).

## El fallo

`tiendas.py` da por perdida a una persona tras 0,65 s sin verla (5 imágenes
a 7,7 fps) y, con ella, BORRABA su jurado de género y sellaba su ficha «sin
demografía». Pero ByteTrack recuerda el número 9 s (`BYTETRACK_LOST_S`): si
la persona vuelve antes, vuelve con el MISMO número y se la trataba como
nueva. En la grabación del 8-oct, 27 de cada 100 personas desaparecían más
de 0,65 s y volvían (la mitad, ~1,2 s). Consecuencias:

  * en pantalla, la etiqueta desaparecía («Personas») y se decidía otra vez
    desde cero (26 de 251 personas etiquetadas);
  * la ficha quedaba sellada sin género aunque luego se decidiera (estimado:
    1 de cada 9 personas decididas);
  * se repetían lecturas del modelo.

## El arreglo

El conteo de la zona sigue igual (la persona sale de `active_tracks` a los
0,65 s, como siempre). Solo se APLAZA lo de la demografía: borrar el jurado
y sellar la ficha esperan a que pasen `BYTETRACK_LOST_S` sin verla. Si
vuelve antes con el mismo número, se cancela y conserva su género al
instante. En el banco (grabación del 8-oct, etiquetas a mano): etiquetas que
desaparecen 26 → 1, acierto 93,3 → 94,3 %, 18 % menos de lecturas.

La lista de pendientes vive DENTRO del clasificador de cada cámara
(`_demographics`), que `TiendasProcess._push_state` ya intercambia por
cámara: así no hay que tocar esa lista de estado.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

_ATRIBUTO = '_aplazados_demografia'


def _pendientes(demo: Any) -> Dict[Any, float]:
    p = getattr(demo, _ATRIBUTO, None)
    if p is None:
        p = {}
        try:
            setattr(demo, _ATRIBUTO, p)
        except Exception:                                   # noqa: BLE001
            pass
    return p


def aplazar(demo: Any, track_id: Any, ahora: float) -> None:
    """La persona dejó de verse: su demografía se borrará más tarde."""
    if demo is not None:
        _pendientes(demo)[track_id] = float(ahora)


def cancelar(demo: Any, track_id: Any) -> bool:
    """Volvió con el mismo número: conserva su jurado y su género."""
    if demo is None:
        return False
    return _pendientes(demo).pop(track_id, None) is not None


def desplazar(demo: Any, segundos: float) -> None:
    """Un corte de vídeo pausó el reloj (`cortes_video.py`): los pendientes
    esperan esos segundos de más."""
    if demo is None or not segundos or segundos <= 0:
        return
    p = _pendientes(demo)
    for t in list(p):
        p[t] += float(segundos)


def purgar(demo: Any, ahora: float, espera_s: float,
           borrar: Callable[[Any], None]) -> int:
    """Borra (con `borrar(track_id)`) a quien lleva más de `espera_s` sin
    volver. Devuelve cuántos. Nunca lanza."""
    if demo is None:
        return 0
    p = _pendientes(demo)
    vencidos = [t for t, desde in p.items() if ahora - desde >= espera_s]
    for t in vencidos:
        p.pop(t, None)
        try:
            borrar(t)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("demografia_aplazada: no se borró %s: %s", t, exc)
    return len(vencidos)
