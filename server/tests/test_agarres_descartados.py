"""Agarres descartados a mano (15-oct-2026, `agarres_descartados.py`).

De 25 bajadas del 15-oct, 22 eran falsas al revisarlas en imagen, y con ellas
21 agarres. El informe del pasillo deja de contarlos sin tocar los datos del
servidor. Se fija: casa por área y hora del GESTO (±3 s), no por la hora de
confirmación del log, y sin lista todo sigue igual.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app import agarres_descartados as ad  # noqa: E402
from src.analityc.core import anaqueles_registro  # noqa: E402

CAM = 'cam-descartes-test'


def _t(hora: str) -> float:
    return time.mktime(time.strptime(f'2026-10-15 {hora}', '%Y-%m-%d %H:%M:%S'))


def test_casa_por_area_y_hora_del_gesto(tmp_path, monkeypatch):
    monkeypatch.setattr(anaqueles_registro, 'raiz_salida', lambda: tmp_path)
    (tmp_path / CAM).mkdir()
    (tmp_path / CAM / ad.NOMBRE).write_text(json.dumps([
        {'hora': '2026-10-15 10:24:55', 'area': 2, 'motivo': 'empleado recolocando'},
        {'hora': 'mal escrita', 'area': 2},
    ]), encoding='utf-8')
    lista = ad.leer(CAM)
    assert len(lista) == 1                                   # la mal escrita se ignora
    assert ad.descartado(_t('10:24:57'), 2, lista) == (True, 'empleado recolocando')
    assert ad.descartado(_t('10:24:55'), 1, lista)[0] is False   # otra área
    # La hora del log es la de la CONFIRMACIÓN, 9 s después: no casa.
    assert ad.descartado(_t('10:25:04'), 2, lista)[0] is False


def test_sin_lista_no_se_descarta_nada(tmp_path, monkeypatch):
    monkeypatch.setattr(anaqueles_registro, 'raiz_salida', lambda: tmp_path)
    assert ad.leer(CAM) == []
    assert ad.descartado(_t('10:24:55'), 2, [])[0] is False
