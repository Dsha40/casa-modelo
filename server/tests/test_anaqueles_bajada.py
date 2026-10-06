"""
Agarres falsos en el área de ABAJO al bajar la mano de la de arriba
(6-oct-2026).

Vídeo de la cámara 3, grabado con `ELDE_ANAQUELES_GRABAR` y reproducido:
la clienta coge de la balda alta y el sistema contaba además un agarre en
la de abajo, donde (lo revisó el usuario en el vídeo) no cogió nada. Dos
causas, y hacían falta las dos reglas (con una sola seguía saliendo):

1. **Misma mano:** la pose vio una sola mano y le puso las dos etiquetas;
   la «izquierda» saltó desde el costado (y ≈ 800) a encima de la derecha
   (1-3 px): 443 px de recorrido hacia el área de abajo.
2. **Bajada:** la derecha volvía de DENTRO del área de arriba y se quedaba
   a la cintura, que con el ángulo de la cámara cae dentro del área de
   abajo. Para esa área venía «de fuera».

Subir no cambia: los agarres buenos de arriba salen de la cintura, dentro
del área de abajo, y esa no está encima.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.anaqueles import (  # noqa: E402
    _DEFECTOS, MedidorAnaqueles, _una_mano_si_coinciden)
from src.analityc.core.anaqueles_registro import RegistroAnaqueles  # noqa: E402

FRAME = np.zeros((900, 800, 3), dtype=np.uint8)
MEDIODIA = time.mktime((2026, 10, 6, 12, 0, 0, 0, 0, -1))
# Como en el vídeo: la balda alta (y 150-350) y la baja (y 390-510).
ARRIBA = [[20, 150], [700, 150], [700, 350], [20, 350]]
ABAJO = [[20, 390], [700, 390], [700, 510], [20, 510]]
BOX = [100, 200, 380, 750]                    # la clienta, delante

COSTADO = ('izq', 70, 800, 0.8)               # mano colgando, bajo las dos
CINTURA = ('der', 250, 440, 0.9)              # dentro de la baja
EN_ALTA = ('der', 230, 260, 0.9)              # dentro de la alta


class _Pose:
    def __init__(self, guion):
        self.guion, self.i, self.disponible = list(guion), 0, True

    def munecas(self, frame, box):
        paso = self.guion[min(self.i, len(self.guion) - 1)]
        self.i += 1
        return [(x, y, c, lado) for lado, x, y, c in paso]


class _Escena:
    def evaluar(self, camara, frame, ahora=None):
        return None


class _SinPersonal:
    def ficha_de(self, pid, reidentifier):
        return None


@pytest.fixture(autouse=True)
def _reloj(monkeypatch):
    strftime_real, localtime_real = time.strftime, time.localtime
    monkeypatch.setattr(time, 'time', lambda: MEDIODIA)
    monkeypatch.setattr(time, 'strftime', lambda f, t=None: strftime_real(
        f, localtime_real(MEDIODIA) if t is None else t))


def _agarres(tmp_path, guion, fps=7.0):
    cfg = dict(_DEFECTOS, pose_cercania=True, agarre_amplitud_desde='extremo',
               stock_activo=False)
    m = MedidorAnaqueles('cam', cfg, pose=_Pose(guion),
                         registro=RegistroAnaqueles('cam', raiz=tmp_path),
                         escena=_Escena(), identificador=_SinPersonal())
    m.poner_zonas([ARRIBA, ABAJO], ['Alta', 'Baja'])
    for i in range(len(guion)):
        m.procesar(FRAME, {1: {'class': -1, 'box': list(BOX),
                               'persistent_id': 'p1'}},
                   ahora=MEDIODIA + i / fps)
    return m.registro.area(0)['agarres'], m.registro.area(1)['agarres']


# ── Misma mano ──────────────────────────────────────────────────────────
def test_dos_muñecas_en_el_mismo_punto_son_una():
    buena, mala = (251, 413, 0.92, 'der'), (252, 412, 0.38, 'izq')
    assert _una_mano_si_coinciden([mala, buena], 285) == [buena]


def test_dos_manos_separadas_siguen_siendo_dos():
    a, b = (200, 413, 0.9, 'der'), (260, 412, 0.8, 'izq')
    assert _una_mano_si_coinciden([a, b], 285) == [a, b]


def test_la_izquierda_que_salta_encima_de_la_derecha_no_agarra(tmp_path):
    """La izquierda colgando al costado y, de pronto, sobre la derecha
    (que está quieta a la cintura, dentro de la baja)."""
    guion = ([[COSTADO, CINTURA]] * 20
             + [[('izq', 251, 441, 0.4), CINTURA]] * 15)
    assert _agarres(tmp_path, guion) == (0, 0)


# ── Bajada ──────────────────────────────────────────────────────────────
def test_bajar_de_la_alta_a_la_cintura_no_agarra_abajo(tmp_path):
    guion = ([[COSTADO, CINTURA]] * 15         # quieta a la cintura
             + [[COSTADO, EN_ALTA]] * 15       # sube y coge arriba
             + [[COSTADO, CINTURA]] * 20)      # baja y se queda
    alta, baja = _agarres(tmp_path, guion)
    assert alta == 1
    assert baja == 0


def test_subir_desde_la_cintura_sigue_agarrando_arriba(tmp_path):
    """La cintura cae dentro de la baja: esa área está DEBAJO, no encima."""
    guion = [[COSTADO, CINTURA]] * 15 + [[COSTADO, EN_ALTA]] * 15
    assert _agarres(tmp_path, guion) == (1, 0)


def test_un_agarre_de_verdad_abajo_sigue_contando(tmp_path):
    """La mano viene de FUERA por debajo (el costado) y entra en la baja."""
    guion = ([[('der', 250, 820, 0.9)]] * 15
             + [[('der', 250, 450, 0.9)]] * 15)
    assert _agarres(tmp_path, guion) == (0, 1)
