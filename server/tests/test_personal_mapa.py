"""Mapa de calor y permanencia propios del personal de camiseta (19-oct-2026,
`personal_mapa.py`). Se fija: solo cuenta el rol «pasillero» dentro del área;
dos a la vez no duplican el tiempo «con personal» pero sí los segundos-persona;
los huecos largos no se cuentan enteros; se guarda por día y hora; el mapa
propio solo se guarda en cámaras donde ya hubo personal.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import personal_mapa as pm  # noqa: E402

T0 = time.mktime(time.strptime('2026-10-19 10:00:00', '%Y-%m-%d %H:%M:%S'))


def _tr(rol, dentro=True):
    return {'es_personal': True, 'personal_rol': rol, 'in_zone': dentro}


def test_solo_camiseta_dentro_del_area(tmp_path):
    p = pm.PermanenciaPersonal('cam', carpeta=tmp_path)
    tracks = {1: _tr('pasillero'), 2: _tr('pasillero'), 3: _tr('vigilante'),
              4: _tr('pasillero', dentro=False), 5: {'in_zone': True}}
    for i in range(11):                                   # 10 s, a 1 imagen/s
        p.observar(tracks, T0 + i)
    p.observar({1: _tr('pasillero')}, T0 + 100)           # hueco: cuenta 1 s como mucho
    p.guardar()
    d = json.loads((tmp_path / '2026-10-19.json').read_text(encoding='utf-8'))
    assert d['horas']['10'] == {'con_personal_s': 11.0, 'persona_s': 21.0}
    assert d['con_personal_s'] == 11.0


def test_guarda_y_sigue_tras_reiniciar(tmp_path):
    p = pm.PermanenciaPersonal('cam', carpeta=tmp_path)
    for i in range(6):
        p.observar({1: _tr('pasillero')}, T0 + i)
    p.guardar()
    otro = pm.PermanenciaPersonal('cam', carpeta=tmp_path)
    for i in range(4):
        otro.observar({1: _tr('pasillero')}, T0 + 50 + i)
    otro.guardar()
    d = json.loads((tmp_path / '2026-10-19.json').read_text(encoding='utf-8'))
    assert d['horas']['10']['con_personal_s'] == 8.0


def test_el_mapa_propio_solo_se_guarda_si_hubo_personal(monkeypatch):
    guardados = []

    class Hm:
        def maybe_save_snapshot(self, clave, **k):
            guardados.append(clave)
    from src.analityc.core.analytics import heatmap_registro
    monkeypatch.setattr(heatmap_registro, 'obtener', lambda *a, **k: Hm())
    monkeypatch.setattr(pm, '_USADOS', set())
    pm.guardar_mapa('cam-x', None)
    assert guardados == []
    pm.mapa('cam-x')
    pm.guardar_mapa('cam-x', None)
    assert guardados == ['cam-x__personal']
