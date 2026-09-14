"""
El payload del frame acepta `establecimiento` (3-ago-2026) sin aflojar el
modo estricto: es el local de la camara que encabeza su mensaje de WhatsApp.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'elde_core')) if (RAIZ / 'elde_core').is_dir() \
    else None
sys.path.insert(0, str(RAIZ))

from elde_core.contracts.eventos import FrameInference  # noqa: E402


def test_el_frame_acepta_establecimiento():
    frame = FrameInference(image=b'\xff\xd8x', establecimiento='La Comarca')
    assert frame.establecimiento == 'La Comarca'
    # Opcional: sin el, sigue valiendo (clientes que no lo mandan).
    assert FrameInference(image=b'\xff\xd8x').establecimiento is None


def test_estricto_sigue_rechazando_lo_desconocido():
    try:
        FrameInference(image=b'\xff\xd8x', local_de_la_camara='X')
        assert False, 'una clave desconocida debe rechazarse (extra=forbid)'
    except Exception:
        pass


def test_el_frame_acepta_los_segundos_de_permanencia():
    """Los contadores del recuadro (19-ago-2026) viajan en el mismo payload.

    Sin declararlos, `extra=forbid` tiraria el frame ENTERO — el modo de
    fallo que dejo sin procesar el 100% de los frames de permanencia el
    13-ago-2026.
    """
    frame = FrameInference(image=b'\xff\xd8x',
                           min_permanencia_personas=45,
                           min_permanencia_vehiculos=60)
    assert frame.min_permanencia_personas == 45
    assert frame.min_permanencia_vehiculos == 60
    # Opcionales: los clientes sin los contadores siguen valiendo.
    vacio = FrameInference(image=b'\xff\xd8x')
    assert vacio.min_permanencia_personas is None
    # 0 es un valor LEGITIMO (alertar de inmediato), no un "sin valor".
    assert FrameInference(image=b'\xff\xd8x',
                          min_permanencia_personas=0).min_permanencia_personas == 0
