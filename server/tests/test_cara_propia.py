"""«Cara propia» (14-oct-2026, `clasificador_b._EspiaYuNet`): de las caras
de la ventana de búsqueda solo vale la que cae en la cabeza de ESA persona,
no la más nítida (que a veces era la del vecino). Banco con etiquetas a mano
(8-oct): la identidad que más personas juntaba, de 10 a 5 números."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analityc.core.analytics import clasificador_b as cb  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core' / 'analytics'


# ── «Cara propia» ────────────────────────────────────────────────────────
class _YuNetFalso:
    def __init__(self, caras):
        self.caras = caras

    def detect(self, img):
        return 1, self.caras


def _cara(cx, cy, puntos):
    """Fila de YuNet: x, y, ancho, alto, 10 coordenadas y la puntuación."""
    return [cx - 10, cy - 10, 20, 20] + [0.0] * 10 + [puntos]


def test_la_cara_del_vecino_no_vale_aunque_sea_la_mas_nitida():
    propia, vecino = _cara(100, 40, 0.70), _cara(180, 40, 0.95)
    esp = cb._EspiaYuNet(_YuNetFalso(np.array([propia, vecino])))
    img = np.zeros((200, 200, 3), np.uint8)
    assert len(esp.detect(img)[1]) == 2                    # sin región, las dos
    esp.region = (0, 0, 200, 200, (50, 20, 150, 200))      # persona: x 50-150
    caras = esp.detect(img)[1]
    assert len(caras) == 1 and caras[0][0] == propia[0]
    esp.region = (0, 0, 200, 200, (110, 20, 150, 200))     # nadie en su cabeza
    assert esp.detect(img)[1] is None


def test_la_cara_propia_es_la_de_la_cabeza_no_la_de_las_manos():
    """Entre un 10 % por encima de la caja y el 40 % de su alto."""
    cabeza, abajo = _cara(100, 40, 0.70), _cara(100, 150, 0.90)
    esp = cb._EspiaYuNet(_YuNetFalso(np.array([cabeza, abajo])))
    esp.region = (0, 0, 200, 200, (50, 20, 150, 200))
    assert len(esp.detect(np.zeros((200, 200, 3), np.uint8))[1]) == 1


def test_la_cara_propia_esta_encendida_y_se_puede_apagar(monkeypatch):
    monkeypatch.delenv('ELDE_CARA_PROPIA', raising=False)
    assert cb._cara_propia() is True
    monkeypatch.setenv('ELDE_CARA_PROPIA', '0')
    assert cb._cara_propia() is False
    src = (RAIZ / 'clasificador_b.py').read_text(encoding='utf-8')
    i = src.index('    def _detect_face_with_pose(self, person_crop):')
    assert src.index('_cara_propia()', i) < src.index('esp.region = (', i)
