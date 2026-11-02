"""
Una foto por persona, no por zona de la imagen (2-nov-2026,
`CAPTURE_UNA_POR_PERSONA`). En la recámara todos entran por la misma puerta:
con el dedup espacial de 60 s, quien entraba detrás del primero se quedaba
sin foto (sábado 31-oct, 5 de 7 visitas sin foto en c4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.tiendas_fotos import FotosMixin  # noqa: E402

PUERTA = [1100, 100, 1200, 700]          # todos entran por la misma caja


class _Proc(FotosMixin):
    def __init__(self):
        self.active_tracks = {}
        self._captured_ids = set()
        self._captured_persons = {}
        self._cell_last_capture = {}
        self.fotos = []

    def _ficha_personal_tienda(self, pid):
        return None

    def _capture_detection(self, tid, frame, box):
        self._captured_ids.add(tid)
        self.fotos.append(tid)
        return f'stem_{tid}'


def _track(pid=None, confirmado=False):
    return {'class': -1, 'box': list(PUERTA), 'seen_frames': 50,
            'persistent_id': pid, 'persistent_confirmed': confirmado}


FRAME = np.zeros((868, 1456, 3), np.uint8)


@pytest.fixture
def proc(monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_UNA_POR_PERSONA', True)
    return _Proc()


def test_dos_personas_por_la_misma_puerta_tienen_cada_una_su_foto(proc):
    proc.active_tracks = {1: _track('aaaa', True)}
    proc._capture_established_tracks(FRAME)
    proc.active_tracks = {2: _track('bbbb', True)}      # 5 s después, misma caja
    proc._capture_established_tracks(FRAME)
    assert proc.fotos == [1, 2]


def test_la_misma_persona_con_otro_numero_no_repite_foto(proc):
    proc.active_tracks = {1: _track('aaaa', True)}
    proc._capture_established_tracks(FRAME)
    proc.active_tracks = {7: _track('aaaa', True)}
    proc._capture_established_tracks(FRAME)
    assert proc.fotos == [1]


def test_sin_identidad_espera_a_tenerla_en_vez_de_perder_la_foto(proc):
    proc.active_tracks = {1: _track('aaaa', True)}
    proc._capture_established_tracks(FRAME)
    proc.active_tracks = {2: _track('prov:2')}            # aún sin identidad
    proc._capture_established_tracks(FRAME)
    assert proc.fotos == [1]                               # la zona la frena…
    proc.active_tracks = {2: _track('bbbb', True)}        # …hasta que la tiene
    proc._capture_established_tracks(FRAME)
    assert proc.fotos == [1, 2]


def test_quien_nunca_tiene_identidad_sigue_con_el_dedup_de_zona(proc):
    """Lo que el dedup espacial protegía: alguien quieto sin cara al que el
    seguimiento le cambia el número no recibe una foto por número."""
    for tid in range(1, 6):
        proc.active_tracks = {tid: _track(f'prov:{tid}')}
        proc._capture_established_tracks(FRAME)
    assert proc.fotos == [1]


def test_apagado_vuelve_a_lo_de_antes(proc, monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_UNA_POR_PERSONA', False)
    proc.active_tracks = {1: _track('aaaa', True)}
    proc._capture_established_tracks(FRAME)
    proc.active_tracks = {2: _track('bbbb', True)}
    proc._capture_established_tracks(FRAME)
    assert proc.fotos == [1]
