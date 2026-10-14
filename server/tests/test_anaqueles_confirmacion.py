"""Agarre confirmado por una bajada y visita por la muñeca (12-oct-2026,
`anaqueles.py`). Lo pidió el usuario:

  - el gesto de agarre deja el agarre POR CONFIRMAR; cuenta cuando el stock
    ve una bajada (o un posible agotado) en esa área, y se apunta al
    producto que bajó, no al de debajo de la muñeca (hallazgo 25);
  - si la vigilancia del área termina sin bajada, no cuenta: queda como
    `agarres_sin_confirmar` para revisarlo;
  - la visita la da la muñeca dentro del área medio segundo, no el cuerpo.

Mismo arnés que `test_anaqueles_cercania.py` (balda alta, persona debajo,
pose de guion). El stock se sustituye en sus dos puntos de contacto
(`vigilando` y `drenar_sucesos`): aquí se prueba la confirmación, no la
medida de la foto (eso es `test_anaqueles_stock*.py`).
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
    MedidorAnaqueles, _DEFECTOS)
from src.analityc.core.anaqueles_registro import (  # noqa: E402
    RegistroAnaqueles)

FRAME = np.zeros((700, 640, 3), dtype=np.uint8)
MEDIODIA = time.mktime((2026, 10, 12, 12, 0, 0, 0, 0, -1))
AREA = [[100, 100], [300, 100], [300, 200], [100, 200]]
BOX_DEBAJO = [150, 260, 250, 600]
BOX_EN_AREA = [120, 120, 220, 460]
ABAJO = [(200, 300, 0.9)]
ARRIBA = [(200, 150, 0.9)]


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
    strftime_real, localtime_real = time.strftime, time.localtime

    def strftime_congelado(formato, tupla=None):
        return strftime_real(formato, localtime_real(MEDIODIA)
                             if tupla is None else tupla)

    monkeypatch.setattr(time, 'time', lambda: MEDIODIA)
    monkeypatch.setattr(time, 'strftime', strftime_congelado)


def _medidor(tmp_path, guion, vigilando=True, **cfg_extra):
    """Con los DEFECTOS de hoy (visita por muñeca, agarre con bajada) y la
    pose de cercanía del YAML. `vigilando` = lo que diría el stock."""
    cfg = dict(_DEFECTOS, pose_cercania=True, agarre_amplitud_desde='extremo')
    cfg.update(cfg_extra)
    m = MedidorAnaqueles('cam', cfg, pose=PoseSecuencia(guion),
                         registro=RegistroAnaqueles('cam', raiz=tmp_path),
                         escena=EscenaEstable(), identificador=SinPersonal())
    m.poner_zonas([AREA], ['Anaquel 1'])
    m._sucesos_falsos = []
    m._stock.vigilando = lambda area: vigilando
    m._stock.drenar_sucesos = lambda: [m._sucesos_falsos.pop(0)] if m._sucesos_falsos else []
    return m


def _correr(m, box, frames, desde=0):
    for i in range(desde, desde + frames):
        m.procesar(FRAME, {1: {'class': -1, 'box': list(box),
                               'persistent_id': None}},
                   ahora=MEDIODIA + i * 0.5)
    return m.registro.area(0)


def _bajada(producto_idx=1, nombre='coffemate', tipo='bajada'):
    return {'tipo': tipo, 'area': 0, 'nombre': 'Anaquel 1',
            'producto': nombre, 'producto_idx': producto_idx, 'ts': '12:00:10',
            'genero': None, 'rango_etario': None,
            'id_corto': f'stock-0-{producto_idx}'}


# ── Agarre confirmado por la bajada ──────────────────────────────────────
def test_el_gesto_solo_no_cuenta_hasta_la_bajada(tmp_path):
    m = _medidor(tmp_path, [ABAJO] * 4 + [ARRIBA] * 6)
    area = _correr(m, BOX_DEBAJO, 10)
    assert int(area.get('agarres') or 0) == 0
    assert len(m._agarres_por_confirmar) == 1


@pytest.mark.parametrize('tipo', ['bajada', 'agotado'])
def test_la_bajada_confirma_el_agarre_y_lo_apunta_al_producto_que_bajo(tmp_path, tipo):
    m = _medidor(tmp_path, [ABAJO] * 4 + [ARRIBA] * 6)
    _correr(m, BOX_DEBAJO, 10)
    m._sucesos_falsos.append(_bajada(producto_idx=1, nombre='coffemate', tipo=tipo))
    area = _correr(m, BOX_DEBAJO, 1, desde=10)
    assert area['agarres'] == 1
    assert area['productos']['1']['agarres'] == 1
    assert m._agarres_por_confirmar == []
    assert any(e.get('tipo') == 'agarre' and e.get('producto') == 'coffemate'
               for e in m._feed)


def test_sin_bajada_queda_sin_confirmar(tmp_path):
    """La vigilancia del área terminó sin bajada: no suma, se aparta."""
    m = _medidor(tmp_path, [ABAJO] * 4 + [ARRIBA] * 6, vigilando=False)
    area = _correr(m, BOX_DEBAJO, 14)
    assert int(area.get('agarres') or 0) == 0
    assert area['agarres_sin_confirmar'] == 1
    assert m._agarres_por_confirmar == []


def test_una_bajada_sin_gesto_no_inventa_un_agarre(tmp_path):
    m = _medidor(tmp_path, [ABAJO])
    m._sucesos_falsos.append(_bajada())
    area = _correr(m, BOX_DEBAJO, 2)
    assert int(area.get('agarres') or 0) == 0


def test_con_la_regla_apagada_el_gesto_cuenta_solo(tmp_path):
    m = _medidor(tmp_path, [ABAJO] * 4 + [ARRIBA] * 6,
                 agarre_requiere_bajada=False)
    assert _correr(m, BOX_DEBAJO, 10)['agarres'] == 1


# ── Visita por la muñeca ─────────────────────────────────────────────────
def test_la_muneca_dentro_medio_segundo_es_visita_aunque_el_cuerpo_no_pise(tmp_path):
    """La balda alta: el cuerpo queda debajo (con la regla del cuerpo no
    había visita) y la mano entra."""
    m = _medidor(tmp_path, [ABAJO] * 2 + [ARRIBA] * 4)
    assert _correr(m, BOX_DEBAJO, 6)['visitas'] == 1


def test_un_instante_dentro_no_es_visita(tmp_path):
    m = _medidor(tmp_path, [ABAJO] * 2 + [ARRIBA] + [ABAJO] * 5)
    assert int(_correr(m, BOX_DEBAJO, 8).get('visitas') or 0) == 0


def test_pararse_delante_sin_meter_la_mano_no_es_visita(tmp_path):
    """Con la regla del cuerpo esto era visita (cuerpo quieto en el área)."""
    m = _medidor(tmp_path, [ABAJO] * 12)
    assert int(_correr(m, BOX_EN_AREA, 12).get('visitas') or 0) == 0


def test_una_visita_por_estancia_de_la_mano(tmp_path):
    m = _medidor(tmp_path, [ARRIBA] * 12)
    assert _correr(m, BOX_DEBAJO, 12)['visitas'] == 1
