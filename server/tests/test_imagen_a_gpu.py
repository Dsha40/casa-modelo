"""
Preparar en la GPU la imagen del detector RF-DETR (`imagen_a_gpu.py`,
2-oct-2026).

Lo que fijan estas pruebas:

- que el tensor preparado es IDÉNTICO, número a número, al de siempre
  (`to_tensor` sobre el RGB), en la CPU y en la GPU. Con `div(255)` en CUDA
  salían distintos 126 de los 256 valores de un píxel (en el último bit), y
  el detector habría visto otra imagen;
- que el detector le pasa a RF-DETR un tensor cuando prepara en la GPU, y
  el numpy de siempre cuando no;
- que el interruptor `ELDE_DETECTOR_PREPARAR_EN_GPU=0` lo apaga;
- que un fallo al preparar vuelve al camino de siempre, en vez de dejar al
  pipeline sin detecciones.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torchvision.transforms import functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analityc.core import deteccion_compartida as dc  # noqa: E402
from src.analityc.core import imagen_a_gpu as ig  # noqa: E402

CUDA = torch.cuda.is_available()


def _imagen():
    """BGR uint8 con los 256 valores posibles en los tres canales, más ruido,
    y con canales distintos entre sí (para cazar un BGR/RGB cambiado)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    plano = img.reshape(-1, 3)
    plano[:256, 0] = np.arange(256)
    plano[:256, 1] = np.arange(256)[::-1]
    plano[:256, 2] = (np.arange(256) * 7) % 256
    return img


def _de_siempre(bgr):
    return F.to_tensor(np.ascontiguousarray(bgr[:, :, ::-1]))


@pytest.fixture(autouse=True)
def _sin_fallos_previos(monkeypatch):
    # `preparar` se desactiva para siempre tras un fallo: cada prueba empieza
    # limpia y deja el módulo como estaba.
    monkeypatch.setattr(ig, '_desactivado', False)
    monkeypatch.delenv(ig.VARIABLE, raising=False)


def test_identica_a_la_de_siempre_en_cpu():
    bgr = _imagen()
    nueva = ig.preparar(bgr, 'cpu')
    assert nueva is not None and nueva.dtype == torch.float32
    assert torch.equal(nueva, _de_siempre(bgr))


@pytest.mark.skipif(not CUDA, reason='sin CUDA')
def test_identica_a_la_de_siempre_en_gpu():
    bgr = _imagen()
    nueva = ig.preparar(bgr, 'cuda')
    assert nueva is not None and nueva.is_cuda
    assert torch.equal(nueva.cpu(), _de_siempre(bgr))


def test_el_interruptor_lo_apaga(monkeypatch):
    assert ig.usar_gpu('cuda') and ig.usar_gpu('cuda:0')
    assert not ig.usar_gpu('cpu')
    for apagado in ('0', 'false', 'no', 'off'):
        monkeypatch.setenv(ig.VARIABLE, apagado)
        assert not ig.usar_gpu('cuda')


def _detector_doble(preparar_en):
    """Como en test_deteccion_compartida: sin __init__, con un modelo que
    apunta lo que le llega."""
    d = dc._DetectorCompartido.__new__(dc._DetectorCompartido)
    d._id91_a_pseudo = {1: 0}
    d._candado = threading.Lock()
    d._preparar_en = preparar_en
    recibido = []

    class _Modelo:
        def predict(self, imagen, **kwargs):
            recibido.append(imagen)
            return None

    d._modelo = _Modelo()
    return d, recibido


def test_preparando_en_gpu_el_detector_pasa_un_tensor():
    destino = 'cuda' if CUDA else 'cpu'
    d, recibido = _detector_doble(destino)
    bgr = _imagen()
    d.detectar(bgr)
    assert isinstance(recibido[0], torch.Tensor)
    assert recibido[0].shape == (3, 37, 53)
    assert torch.equal(recibido[0].cpu(), _de_siempre(bgr))


def test_sin_preparar_en_gpu_el_detector_pasa_el_numpy_de_siempre():
    d, recibido = _detector_doble(None)
    bgr = _imagen()
    d.detectar(bgr)
    assert isinstance(recibido[0], np.ndarray)
    assert np.array_equal(recibido[0], bgr[:, :, ::-1])


def test_si_preparar_falla_vuelve_al_camino_de_siempre():
    d, recibido = _detector_doble('dispositivo-que-no-existe')
    bgr = _imagen()
    d.detectar(bgr)
    d.detectar(bgr)
    assert all(isinstance(r, np.ndarray) for r in recibido)
    assert ig._desactivado        # avisó una vez y no lo reintenta
