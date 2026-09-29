"""
Agarres en baldas altas: mirar la muñeca de quien está CERCA (28/29-sep-2026).

El caso real (grabación del 23-sep de la cámara 3): una clienta coge botes de
Coffee-mate de una balda que le queda POR ENCIMA de la cabeza. Con el brazo
abajo su caja no pisa el área (compuerta cerrada) y la pose solo se calculaba
para quien está «en compuerta y quieto»: la máquina no veía la mano fuera ni
la entrada, y no contaba nada. Lo que se fija aquí:

  - `pose_cercania`: con ella, quien está DEBAJO del área recibe pose y su
    agarre cuenta; sin ella, ni se le mira (el coste de siempre);
  - `agarre_amplitud_desde='extremo'`: el viaje de la mano se mide desde el
    punto MÁS alejado en que se la vio fuera, no desde la última muestra,
    que puede ir rozando el borde;
  - la geometría de «cerca»;
  - el grabador de diagnóstico escribe lo que el medidor recibe y nunca
    lanza.

Mismos dobles que `test_anaqueles.py`: pose con guion, escena controlable,
registro en `tmp_path`, reloj de pared congelado.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.anaqueles import (  # noqa: E402
    MedidorAnaqueles, _DEFECTOS)
from src.analityc.core.anaqueles_grabador import Grabador  # noqa: E402
from src.analityc.core.anaqueles_registro import (  # noqa: E402
    RegistroAnaqueles)

FRAME = np.zeros((700, 640, 3), dtype=np.uint8)
MEDIODIA = time.mktime((2026, 9, 29, 12, 0, 0, 0, 0, -1))

#: Una balda ALTA: x 100..300, y 100..200.
AREA = [[100, 100], [300, 100], [300, 200], [100, 200]]
#: Persona DEBAJO de la balda: su franja superior (y 260..464) no la toca,
#: así que la compuerta está cerrada. Es la clienta con el brazo abajo.
BOX_DEBAJO = [150, 260, 250, 600]
#: Persona pegada a la balda (la de `test_anaqueles.py`): compuerta abierta.
BOX_EN_AREA = [120, 120, 220, 460]


class PoseSecuencia:
    def __init__(self, guion):
        self.guion, self.llamadas, self.disponible = list(guion), 0, True

    def munecas(self, frame, box):
        i = min(self.llamadas, len(self.guion) - 1)
        self.llamadas += 1
        return list(self.guion[i])


class EscenaEstable:
    def evaluar(self, camara, frame, ahora=None):
        return None


class SinPersonal:
    def ficha_de(self, pid, reidentifier):
        return None


@pytest.fixture(autouse=True)
def reloj(monkeypatch):
    """Reloj de pared congelado: el motivo, en `test_anaqueles.reloj`."""
    strftime_real, localtime_real = time.strftime, time.localtime

    def strftime_congelado(formato, tupla=None):
        return strftime_real(formato, localtime_real(MEDIODIA)
                             if tupla is None else tupla)

    monkeypatch.setattr(time, 'time', lambda: MEDIODIA)
    monkeypatch.setattr(time, 'strftime', strftime_congelado)


def _medidor(tmp_path, guion, **cfg_extra):
    cfg = dict(_DEFECTOS)
    cfg.update(cfg_extra)
    pose = PoseSecuencia(guion)
    m = MedidorAnaqueles('cam', cfg, pose=pose,
                         registro=RegistroAnaqueles('cam', raiz=tmp_path),
                         escena=EscenaEstable(), identificador=SinPersonal())
    m.poner_zonas([AREA], ['Anaquel 1'])
    return m, pose


def _correr(m, box, frames):
    for i in range(frames):
        m.procesar(FRAME, {1: {'class': -1, 'box': list(box),
                               'persistent_id': None}},
                   ahora=MEDIODIA + i * 0.5)
    return m.registro.area(0)['agarres']


# ── pose_cercania ────────────────────────────────────────────────────────
#: Mano abajo (fuera, muy lejos del área) y luego dentro de la balda alta.
ABAJO = [(200, 300, 0.9)]
ARRIBA = [(200, 150, 0.9)]


def test_sin_cercania_a_quien_esta_debajo_ni_se_le_mira(tmp_path):
    """La regla de siempre: compuerta cerrada = sin pose = sin agarre. Es el
    fallo del vídeo, y también el contrato de coste de siempre."""
    m, pose = _medidor(tmp_path, [ABAJO] * 4 + [ARRIBA] * 6)
    assert _correr(m, BOX_DEBAJO, 10) == 0
    assert pose.llamadas == 0


def test_con_cercania_el_agarre_desde_debajo_cuenta(tmp_path):
    """Mano fuera (abajo) unos frames y luego dentro de la balda alta:
    UN agarre, sin que el cuerpo llegue a pisar el área."""
    m, pose = _medidor(tmp_path, [ABAJO] * 4 + [ARRIBA] * 6,
                       pose_cercania=True)
    assert _correr(m, BOX_DEBAJO, 10) == 1
    assert pose.llamadas > 0


def test_con_cercania_la_mano_que_nunca_sale_no_cuenta(tmp_path):
    """La protección del 14-ago sigue en pie: sin ver la mano FUERA no hay
    gesto, aunque ahora se la mire aunque el cuerpo no pise el área."""
    m, _pose = _medidor(tmp_path, [ARRIBA] * 10, pose_cercania=True)
    assert _correr(m, BOX_DEBAJO, 10) == 0


def test_cerca_de_area_geometria(tmp_path):
    m, _pose = _medidor(tmp_path, [ABAJO])
    assert m._cerca_de_area(BOX_DEBAJO, 0)                  # justo debajo
    assert m._cerca_de_area(BOX_EN_AREA, 0)                 # delante
    assert not m._cerca_de_area([400, 260, 500, 600], 0)    # a un lado
    assert not m._cerca_de_area([150, 500, 250, 700], 0)    # muy abajo
    assert not m._cerca_de_area([150, 0, 250, 90], 0)       # entera encima


# ── agarre_amplitud_desde ────────────────────────────────────────────────
#: Mano lejos del área, luego rozando su borde de abajo (y=200) por fuera, y
#: por fin dentro. Persona de 100 px de ancho: exige 30 px de viaje.
LEJOS = [(170, 400, 0.9)]
ROZANDO = [(170, 202, 0.9)]
DENTRO = [(170, 185, 0.9)]


def test_desde_la_ultima_muestra_el_roce_del_borde_se_come_el_gesto(tmp_path):
    """La regla de siempre: el viaje se mide desde (170,202), la última
    muestra fuera: 17 px < 30. Es lo que se vio en el vídeo."""
    m, _pose = _medidor(tmp_path, [LEJOS] * 3 + [ROZANDO] + [DENTRO] * 6)
    assert _correr(m, BOX_EN_AREA, 11) == 0
    assert m._descartes.get('agarre_amplitud', 0) >= 1


def test_desde_el_extremo_cuenta_el_viaje_real(tmp_path):
    """Con 'extremo' el origen es (170,400), el punto más alejado de esa
    racha fuera: 215 px de viaje, un agarre."""
    m, _pose = _medidor(tmp_path, [LEJOS] * 3 + [ROZANDO] + [DENTRO] * 6,
                        agarre_amplitud_desde='extremo')
    assert _correr(m, BOX_EN_AREA, 11) == 1


# ── Grabador de diagnóstico ──────────────────────────────────────────────
def test_el_grabador_guarda_lo_que_recibe_el_medidor(tmp_path):
    g = Grabador(tmp_path / 'g', maximo=2)
    tracks = {np.int64(100002): {'class': -1, 'box': np.array([1., 2, 3, 4]),
                                 'persistent_id': None},
              7: {'class': 3, 'box': [1, 2, 3, 4]}}        # no es persona
    for i in range(3):                                       # 3 > maximo
        g.grabar('cam', FRAME, tracks, [np.array(AREA)], ['Anaquel 1'],
                 [{'anaquel': 0, 'nombre': 'X', 'poligono': AREA}], 10.0 + i)
    lineas = [json.loads(l) for l in
              (tmp_path / 'g' / 'eventos.jsonl').read_text().splitlines()]
    assert len(lineas) == 2                                  # respeta el tope
    assert len(list((tmp_path / 'g' / 'frames').iterdir())) == 2
    assert list(lineas[0]['tracks']) == ['100002']           # solo personas
    assert lineas[0]['zonas'] == [AREA]                      # numpy -> JSON


def test_el_grabador_nunca_lanza(tmp_path):
    g = Grabador(tmp_path / 'g', maximo=5)
    g.grabar('cam', None, None, None, None, None, 0.0)       # frame ilegible
