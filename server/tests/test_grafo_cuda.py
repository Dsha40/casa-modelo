"""
Grafo de CUDA para repetir un modelo (`grafo_cuda.py`, 6-oct-2026).

Lo que fijan estas pruebas: la salida con grafo es la MISMA que sin él; se
graba un grafo por forma de entrada y se reutiliza; la salida devuelta no
la pisa la siguiente repetición; `ELDE_GRAFOS_CUDA=0`, sin CUDA o con un lote
demasiado grande devuelve None (camino de siempre); y tras un fallo al
grabar no se vuelve a intentar.

Medido aparte con el MiVOLO real (no es prueba: necesita los pesos):
30,1 -> 10,5 ms por llamada; 38 recortes con las mismas muestras.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('sin CUDA', allow_module_level=True)

from src.analityc.core import grafo_cuda as gc  # noqa: E402


def _modelo():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(8, 3)).cuda().eval()


def test_da_lo_mismo_que_sin_grafo(monkeypatch):
    monkeypatch.delenv(gc.VARIABLE, raising=False)
    m = _modelo()
    grafos = gc.GrafosPorLote(m)
    for semilla in range(4):
        torch.manual_seed(semilla)
        x = torch.rand(1, 3, 32, 32, device='cuda')
        with torch.no_grad():
            esperado = m(x)
        assert torch.equal(grafos.ejecutar(x), esperado)


def test_un_grafo_por_forma_y_la_salida_no_se_pisa(monkeypatch):
    monkeypatch.delenv(gc.VARIABLE, raising=False)
    grafos = gc.GrafosPorLote(_modelo())
    a = grafos.ejecutar(torch.rand(1, 3, 32, 32, device='cuda'))
    a_copia = a.clone()
    grafos.ejecutar(torch.rand(1, 3, 32, 32, device='cuda'))
    grafos.ejecutar(torch.rand(2, 3, 32, 32, device='cuda'))
    assert len(grafos._grafos) == 2               # (1,...) y (2,...)
    assert torch.equal(a, a_copia)                # la 1.ª salida sigue igual


def test_cuando_no_se_puede_devuelve_none(monkeypatch):
    grafos = gc.GrafosPorLote(_modelo())
    monkeypatch.setenv(gc.VARIABLE, '0')
    assert grafos.ejecutar(torch.rand(1, 3, 8, 8, device='cuda')) is None
    monkeypatch.delenv(gc.VARIABLE, raising=False)
    assert grafos.ejecutar(torch.rand(1, 3, 8, 8)) is None            # CPU
    assert grafos.ejecutar(
        torch.rand(gc.LOTE_MAX + 1, 3, 8, 8, device='cuda')) is None


def test_tras_un_fallo_no_se_reintenta(monkeypatch):
    monkeypatch.delenv(gc.VARIABLE, raising=False)
    grafos = gc.GrafosPorLote(_modelo())
    llamadas = []

    def falla(entrada):
        llamadas.append(1)
        raise RuntimeError('grabación imposible')
    monkeypatch.setattr(grafos, '_grabar', falla)
    x = torch.rand(1, 3, 8, 8, device='cuda')
    assert grafos.ejecutar(x) is None
    assert grafos.ejecutar(x) is None
    assert llamadas == [1]


class _DosSalidas(torch.nn.Module):
    """Como RF-DETR: devuelve una tupla (cajas, puntuaciones)."""

    def __init__(self):
        super().__init__()
        self.m = _modelo()

    def forward(self, x):
        y = self.m(x)
        return y * 2, y + 1


def test_varias_salidas_se_copian_todas(monkeypatch):
    """29-oct-2026 (detector con grafo): una tupla de salidas, cada una
    igual que sin grafo y sin que la pise la siguiente repetición."""
    monkeypatch.delenv(gc.VARIABLE, raising=False)
    m = _DosSalidas().cuda().eval()
    grafos = gc.GrafosPorLote(m)
    x = torch.rand(1, 3, 32, 32, device='cuda')
    with torch.no_grad():
        esperado = m(x)
    a = grafos.ejecutar(x)
    assert isinstance(a, tuple) and len(a) == 2
    assert all(torch.equal(p, q) for p, q in zip(a, esperado))
    copia = tuple(t.clone() for t in a)
    grafos.ejecutar(torch.rand(1, 3, 32, 32, device='cuda'))
    assert all(torch.equal(p, q) for p, q in zip(a, copia))
