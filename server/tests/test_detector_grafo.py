"""
El detector con grafo de CUDA (`detector_grafo.py`, 29-oct-2026).

Lo que fijan estas pruebas: apagado por defecto (no toca el modelo); con
`ELDE_GRAFO_DETECTOR=1` el modelo de inferencia queda envuelto, da LO MISMO
que sin grafo y no se envuelve dos veces; sin CUDA en el modelo no se
instala; y si el grafo no se puede usar, se llama al modelo de siempre.
Con un modelo pequeño: el RF-DETR real se midió aparte (93 de 93
detecciones iguales, 32,1 → 24,1 ms el modelo).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('sin CUDA', allow_module_level=True)

from src.analityc.core import detector_grafo as dg  # noqa: E402
from src.analityc.core import grafo_cuda as gc  # noqa: E402


class _Red(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        y = self.c(x)
        return y.mean((2, 3)), y.amax((2, 3))       # dos salidas, como RF-DETR


def _rfdetr(dispositivo='cuda'):
    red = _Red().to(dispositivo).eval()
    nucleo = types.SimpleNamespace(inference_model=red, device=torch.device(dispositivo))
    return types.SimpleNamespace(model=nucleo), red


def test_apagado_por_defecto_no_toca_el_modelo(monkeypatch):
    monkeypatch.delenv(dg.VARIABLE, raising=False)
    rf, red = _rfdetr()
    assert not dg.instalar(rf)
    assert rf.model.inference_model is red


def test_encendido_da_lo_mismo_y_no_se_envuelve_dos_veces(monkeypatch):
    monkeypatch.setenv(dg.VARIABLE, '1')
    monkeypatch.delenv(gc.VARIABLE, raising=False)
    rf, red = _rfdetr()
    assert dg.instalar(rf)
    assert not dg.instalar(rf)
    for semilla in range(3):
        torch.manual_seed(semilla)
        x = torch.rand(1, 3, 16, 16, device='cuda')
        with torch.no_grad():
            esperado = red(x)
            salida = rf.model.inference_model(x)
        assert all(torch.equal(p, q) for p, q in zip(salida, esperado))


def test_sin_cuda_no_se_instala(monkeypatch):
    monkeypatch.setenv(dg.VARIABLE, '1')
    rf, red = _rfdetr('cpu')
    assert not dg.instalar(rf)
    assert rf.model.inference_model is red


def test_si_el_grafo_no_se_puede_usar_llama_al_modelo(monkeypatch):
    monkeypatch.setenv(dg.VARIABLE, '1')
    monkeypatch.setenv(gc.VARIABLE, '0')            # grafos apagados
    rf, red = _rfdetr()
    assert dg.instalar(rf)
    x = torch.rand(1, 3, 16, 16, device='cuda')
    with torch.no_grad():
        esperado = red(x)
        salida = rf.model.inference_model(x)
    assert all(torch.equal(p, q) for p, q in zip(salida, esperado))
