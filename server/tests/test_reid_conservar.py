"""Un numero que ya tenia identidad no estrena otra si su cara no coincide
(14-oct-2026, `face_reidentifier.anotar_identidad`). El 14-oct, de 123
identidades nuevas, al menos 5 cayeron sobre un numero que ya tenia una (la
clienta de negro: 5af8954d y luego 14ef269d)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_reid_ttl import _reid, _rostro, db_path  # noqa: E402,F401


def test_con_identidad_por_cuerpo_una_cara_nueva_no_crea_otra(db_path, monkeypatch):
    monkeypatch.delenv('ELDE_REID_CONSERVAR', raising=False)
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(1))[0]
    reid.anotar_identidad(100002, uid)                 # el cuerpo la enlazo
    uid2, nueva, _ = reid.identify_or_register(100002, _rostro(2))
    assert uid2 == uid and not nueva
    assert reid.total_unique_persons == 1


def test_al_perder_el_mapeo_conserva_la_suya(db_path, monkeypatch):
    monkeypatch.delenv('ELDE_REID_CONSERVAR', raising=False)
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(3))[0]
    reid.remove_track(100001)                          # se perdio un momento
    assert reid.identify_or_register(100001, _rostro(4))[0] == uid


def test_sin_identidad_previa_si_se_crea_y_el_interruptor_apaga(db_path, monkeypatch):
    reid = _reid(db_path)
    reid.identify_or_register(100001, _rostro(5))
    assert reid.identify_or_register(100002, _rostro(6))[1] is True   # nadie previo
    monkeypatch.setenv('ELDE_REID_CONSERVAR', '0')
    reid.anotar_identidad(100003, reid.get_persistent_id(100001))
    assert reid.identify_or_register(100003, _rostro(7))[1] is True


def test_una_cara_que_si_coincide_con_otra_se_reasigna(db_path, monkeypatch):
    monkeypatch.delenv('ELDE_REID_CONSERVAR', raising=False)
    reid = _reid(db_path)
    a = reid.identify_or_register(100001, _rostro(8))[0]
    b = reid.identify_or_register(100002, _rostro(9))[0]
    reid.anotar_identidad(100003, a)
    assert reid.identify_or_register(100003, _rostro(9))[0] == b
