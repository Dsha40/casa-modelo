"""
Permanencia por cuarto en las cámaras de solo permanencia (30-oct-2026,
`permanencia_cuarto.py`). Tiempos inyectados, sin servidor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import permanencia_cuarto as pc  # noqa: E402

T0 = 1793390000.0          # 30-oct-2026 por la tarde (hora local)


def _p(x, alto=300):
    return {'class': -1, 'box': [x - 50, 400 - alto, x + 50, 400]}


def _pasar(c, tramos, paso=0.5, hasta=40.0):
    """tramos: lista de (desde_s, hasta_s, tid, x0, vx) -> un seguimiento."""
    t = 0.0
    while t <= hasta:
        tracks = {}
        for a, b, tid, x0, vx in tramos:
            if a <= t < b:
                tracks[tid] = _p(x0 + vx * (t - a))
        c.observar(tracks, T0 + t)
        t += paso


def _dia(c):
    return json.loads(next(c.carpeta.glob('*.json')).read_text(encoding='utf-8'))


def test_una_estancia_y_su_promedio(tmp_path):
    c = pc.PermanenciaCuarto('cam', carpeta=tmp_path)
    _pasar(c, [(1.0, 11.0, 1, 100, 10)])          # 10 s andando
    d = _dia(c)
    assert len(d['estancias']) == 1
    assert d['estancias'][0]['segundos'] == pytest.approx(9.5, abs=0.6)
    assert d['resumen']['promedio_s'] == pytest.approx(9.5, abs=0.6)


def test_un_corte_corto_cerca_es_la_misma_estancia(tmp_path):
    """Detrás de la cama la caja se pierde 3 s y vuelve con otro número."""
    c = pc.PermanenciaCuarto('cam', carpeta=tmp_path)
    _pasar(c, [(1.0, 6.0, 1, 100, 5), (9.0, 15.0, 2, 140, 5)])
    d = _dia(c)
    assert len(d['estancias']) == 1
    assert d['estancias'][0]['segundos'] == pytest.approx(13.5, abs=0.6)


def test_lejos_o_mucho_despues_es_otra_estancia(tmp_path):
    c = pc.PermanenciaCuarto('cam', carpeta=tmp_path)
    _pasar(c, [(1.0, 6.0, 1, 100, 5), (9.0, 15.0, 2, 900, 5)])   # lejos
    assert len(_dia(c)['estancias']) == 2
    c2 = pc.PermanenciaCuarto('cam2', carpeta=tmp_path / 'b')
    _pasar(c2, [(1.0, 6.0, 1, 100, 5), (16.0, 22.0, 2, 130, 5)])  # tarde
    assert len(_dia(c2)['estancias']) == 2


def test_lo_muy_corto_no_cuenta(tmp_path):
    c = pc.PermanenciaCuarto('cam', carpeta=tmp_path)
    _pasar(c, [(1.0, 2.0, 1, 100, 5), (20.0, 30.0, 2, 300, 5)])
    assert len(_dia(c)['estancias']) == 1


def test_un_objeto_quieto_no_es_gente(tmp_path):
    """30-oct: un cuadro de la pared contó como visitante."""
    c = pc.PermanenciaCuarto('cam', carpeta=tmp_path)
    _pasar(c, [(0.0, 70.0, 1, 500, 0)], hasta=80.0)
    d = _dia(c)
    assert d['estancias'] == [] and d['quietas_descartadas'] == 1


def test_leer_dia_para_la_api(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, '_RAIZ', tmp_path)
    c = pc.PermanenciaCuarto('cocina', carpeta=tmp_path / 'output'
                             / 'permanencia_cuartos' / 'cocina')
    _pasar(c, [(1.0, 11.0, 1, 100, 10)])
    dia = next(c.carpeta.glob('*.json')).stem
    r = pc.leer_dia(dia)
    assert r[0]['camera_id'] == 'cocina' and r[0]['estancias'] == 1


def test_tiendas_salta_todo_en_las_camaras_de_solo_permanencia():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    i = src.index("if getattr(self, '_solo_permanencia', False):\n"
                  "                    permanencia_cuarto.de(camera_id)")
    resto = src[i:i + 4000]
    # Lo caro va en la rama `else:` (sangrado de 20 espacios).
    assert '\n                    self._resolve_track_identities(' in resto
    assert '\n                    self._process_zone_transitions()' in resto
    assert "SOLO_PERMANENCIA', False))" in src
    from src.analityc.core.analytics.config import AnalyticsConfig
    assert AnalyticsConfig.SOLO_PERMANENCIA is False
