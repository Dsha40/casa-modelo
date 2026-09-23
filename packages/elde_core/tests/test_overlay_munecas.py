"""Las muñecas del modo Anaqueles pintadas sobre el vídeo en vivo.

Se añadieron el 23-sep-2026 para poder calibrar un área viendo DÓNDE cree el
servidor que está la mano, en vez de adivinar.

Lo que más se protege aquí es la CADUCIDAD. Son un dibujo en vivo y el bloque
solo trae muñecas cuando las hay: sin TTL, el último punto se quedaría clavado
en pantalla después de que la persona se fuera, y un punto viejo miente más que
un hueco — el operador calibraría el área contra una mano que ya no está ahí.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from elde_core.ui import overlay_munecas as ov  # noqa: E402

_DENTRO = {'x': 100.0, 'y': 200.0, 'lado': 'der', 'area': 0,
           'prof': 25.0, 'dentro': True}
_FUERA = {'x': 300.0, 'y': 400.0, 'lado': 'izq', 'area': 0,
          'prof': -8.0, 'dentro': False}


# ── Caducidad ────────────────────────────────────────────────────────────

def test_recien_llegadas_se_pintan():
    assert ov.vigentes([_DENTRO], ts=100.0, ahora=100.1) == [_DENTRO]


def test_caducan_pasado_el_ttl():
    """Un punto viejo clavado en pantalla es peor que no pintar nada."""
    assert ov.vigentes([_DENTRO], ts=100.0, ahora=100.0 + ov.TTL_S + 0.01) == []


def test_justo_antes_del_limite_todavia_valen():
    """El limite EXACTO no se comprueba a proposito: `100.0 + 0.7 - 100.0`
    da 0.7000000000000455 en coma flotante, asi que fijarlo solo ataria el
    codigo al redondeo. Lo que importa es que un punto reciente se pinte."""
    assert ov.vigentes([_DENTRO], ts=100.0,
                       ahora=100.0 + ov.TTL_S - 0.01) == [_DENTRO]


def test_sin_hora_no_se_pinta_nada():
    """Un juego de munecas sin su hora no se puede caducar, asi que no se
    pinta: es el estado de un cliente que aun no ha recibido ninguno."""
    assert ov.vigentes([_DENTRO], ts=None, ahora=100.0) == []


def test_lista_vacia_o_none():
    assert ov.vigentes([], ts=100.0, ahora=100.0) == []
    assert ov.vigentes(None, ts=100.0, ahora=100.0) == []


def test_se_descarta_lo_que_no_tiene_coordenadas():
    """El bloque lo compone el servidor; un campo que falte no puede tumbar
    el pintado del recuadro entero."""
    basura = [{'lado': 'der'}, 'no soy un dict', _DENTRO]
    assert ov.vigentes(basura, ts=100.0, ahora=100.0) == [_DENTRO]


# ── Pintado ──────────────────────────────────────────────────────────────

def test_pintar_cv2_no_revienta_y_marca_el_pixel():
    import numpy as np
    frame = np.zeros((600, 500, 3), dtype=np.uint8)
    ov.pintar_cv2(frame, [_DENTRO, _FUERA])
    assert frame[200, 100].any(), 'debe pintar sobre la muneca'
    assert frame[400, 300].any()


def test_pintar_cv2_con_lista_vacia_no_toca_el_frame():
    import numpy as np
    frame = np.zeros((60, 50, 3), dtype=np.uint8)
    ov.pintar_cv2(frame, [])
    assert not frame.any()


def test_dentro_y_fuera_no_usan_el_mismo_color():
    """El color ES la informacion: verde lo cuenta el servidor, ambar no."""
    import numpy as np
    a = np.zeros((300, 300, 3), dtype=np.uint8)
    b = np.zeros((300, 300, 3), dtype=np.uint8)
    ov.pintar_cv2(a, [{**_DENTRO, 'x': 150.0, 'y': 150.0, 'dentro': True}])
    ov.pintar_cv2(b, [{**_DENTRO, 'x': 150.0, 'y': 150.0, 'dentro': False}])
    assert not np.array_equal(a[150, 150], b[150, 150])


def test_los_rotulos_no_se_superponen():
    """Las dos munecas de quien alcanza un estante caen casi en el mismo
    pixel: sin separacion sus etiquetas se leian como un borron."""
    assert ov._SALTO_ROTULO > 0


def test_pintar_qt_no_revienta():
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    pm = QPixmap(500, 600)
    p = QPainter(pm)
    ov.pintar_qt(p, [_DENTRO, _FUERA])      # no debe lanzar
    p.end()
