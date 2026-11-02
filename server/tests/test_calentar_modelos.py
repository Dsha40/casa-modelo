"""
Calentar las sesiones ONNX al crear el procesador (30-oct-2026,
`calentar_modelos.py`).

La PRIMERA llamada a una sesión nueva de OSNet tardaba ~26 s y la de YuNet
~4,4 s (con torch cargado antes, como en el servidor), y la pagaba la
primera persona tras arrancar: el servidor se quedaba ~20-25 s sin
responder y la gente cruzaba el ROI sin verse. Reproducido en frío: la
primera imagen con persona pasó de 38 s a 1,2 s con esto. Aquí, sesiones
falsas: sin GPU.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import calentar_modelos as cm  # noqa: E402


class _Sesion:
    def __init__(self, forma=('batch', 3, 256, 128), falla=False):
        self.forma, self.falla, self.llamadas = forma, falla, []

    def get_inputs(self):
        return [types.SimpleNamespace(name='x', shape=list(self.forma),
                                      type='tensor(float)')]

    def run(self, _salidas, entradas):
        if self.falla:
            raise RuntimeError('sin GPU')
        self.llamadas.append({k: v.shape for k, v in entradas.items()})
        return [np.zeros(1)]


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    monkeypatch.delenv('ELDE_CALENTAR', raising=False)
    cm._hechas.clear()


def _proc(**sesiones):
    p = types.SimpleNamespace(**sesiones)
    return p


def test_calienta_cada_sesion_una_vez_con_ceros_de_su_forma():
    osnet, yunet = _Sesion(), _Sesion(forma=(1, 3, 640, 640))
    p = _proc(_osnet_session=osnet, _yunet=types.SimpleNamespace(_sess=yunet))
    assert cm.calentar(p, en_hilo=False) == ['OSNet', 'YuNet']
    assert osnet.llamadas == [{'x': (1, 3, 256, 128)}]   # el lote libre, a 1
    assert yunet.llamadas == [{'x': (1, 3, 640, 640)}]
    # Otra cámara de la misma conexión comparte sesiones: no se repite.
    assert cm.calentar(p, en_hilo=False) == []
    assert len(osnet.llamadas) == 1


def test_una_sesion_que_falla_no_rompe_nada():
    buena, mala = _Sesion(), _Sesion(falla=True)
    p = _proc(_osnet_session=mala, _face_embedding_session=buena)
    assert cm.calentar(p, en_hilo=False) == ['OSNet', 'ArcFace']
    assert len(buena.llamadas) == 1


def test_sin_sesiones_no_hace_nada():
    assert cm.calentar(_proc(), en_hilo=False) == []


def test_apagado(monkeypatch):
    monkeypatch.setenv('ELDE_CALENTAR', '0')
    s = _Sesion()
    assert cm.calentar(_proc(_osnet_session=s), en_hilo=False) == []
    assert s.llamadas == []


def test_tiendas_calienta_al_crear_el_procesador():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    i = src.index('calentar_modelos.calentar(self)')
    # Después de crear las sesiones y la demografía que las usa.
    assert src.index('self._demographics = clasificador_b.crear(') < i
