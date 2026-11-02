"""
Permanencia en toda la imagen (30-oct-2026, `PERMANENCIA_TODA_LA_IMAGEN`).

Casa modelo, cámara 4: el ROI cubre la entrada. Con la permanencia medida
dentro del ROI, el tiempo era el de la entrada, no el de la visita. Ahora
DENTRO = se le ve en cualquier parte de la imagen. Se ejercita la máquina de
`_process_zone_transitions` con un procesador sin modelos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import origen_identidad  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.tiendas import TiendasProcess  # noqa: E402


def _proc(tracks):
    p = object.__new__(TiendasProcess)
    p.active_tracks = tracks
    p._zone_state = {}
    p._total_entries = 0
    p._total_exits = 0
    p._unique_entered = set()
    p._anotados = []
    p._anotar_zona = lambda tipo, pid, ts, perm=0.0: p._anotados.append(
        (tipo, pid))
    p._imagenes = lambda n: 1                      # histéresis de 1 imagen
    return p


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    monkeypatch.delenv('ELDE_VISITAS_SOLO_ROI', raising=False)
    origen_identidad._origen.clear()


def _fuera_del_roi():
    return {1: {'persistent_id': 'p1', 'persistent_confirmed': True,
                'in_zone': False, 'class': -1}}


def test_visible_fuera_del_roi_cuenta_como_dentro(monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'PERMANENCIA_TODA_LA_IMAGEN', True)
    p = _proc(_fuera_del_roi())
    p._process_zone_transitions()
    assert p._zone_state['p1']['estado'] == 'DENTRO'
    assert p._total_entries == 1
    # Deja de verse: sale.
    p.active_tracks = {}
    p._process_zone_transitions()
    assert p._zone_state['p1']['estado'] == 'FUERA'
    assert p._total_exits == 1


def test_con_la_opcion_apagada_manda_el_roi(monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'PERMANENCIA_TODA_LA_IMAGEN', False)
    p = _proc(_fuera_del_roi())
    p._process_zone_transitions()
    assert p._total_entries == 0


def test_nacida_fuera_del_roi_no_entra_aunque_se_la_vea(monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'PERMANENCIA_TODA_LA_IMAGEN', True)
    origen_identidad.anotar('p1', False)
    p = _proc(_fuera_del_roi())
    p._process_zone_transitions()
    assert p._total_entries == 0


def test_permanencia_area_mide_su_zona():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'permanencia_area.py').read_text(encoding='utf-8')
    assert 'self._zona_toda_la_imagen = False' in src
