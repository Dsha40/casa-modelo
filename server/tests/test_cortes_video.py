"""Pausar durante los cortes de vídeo de iVMS (13-oct-2026,
`cortes_video.py`). OPCIONAL: apagado por defecto.

Lo que fijan: está apagado salvo que se pida; la imagen repetida solo se
salta pasados 0,4 s (el vídeo normal repite como mucho una); al volver el
vídeo se alarga la espera de los géneros pendientes y, tras una pausa
larga, se reinicia el seguimiento; la respuesta en pausa no lleva cajas ni
repite alertas; y `tiendas.py`/`anaqueles.py` lo usan donde deben.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analityc.core import cortes_video as cv_  # noqa: E402
from src.analityc.core import demografia_aplazada as da  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core'


def _img(valor):
    im = np.zeros((90, 160, 3), np.uint8)
    im[10:40, 10:60] = valor
    return im


def test_apagado_por_defecto_y_se_enciende_con_la_variable(monkeypatch):
    monkeypatch.delenv('ELDE_CONGELAR_CORTES', raising=False)
    assert cv_.activo() is False
    monkeypatch.setenv('ELDE_CONGELAR_CORTES', '1')
    assert cv_.activo() is True


def test_solo_salta_la_imagen_repetida_pasados_04_s():
    est = {}
    assert cv_.congelada(est, _img(100), 0.0) is False       # nueva
    assert cv_.congelada(est, _img(100), 0.1) is False       # una repetida (normal)
    assert cv_.congelada(est, _img(120), 0.2) is False       # el vídeo sigue
    assert cv_.congelada(est, _img(120), 0.5) is False       # 0,3 s igual: aún no
    assert cv_.congelada(est, _img(120), 0.7) is True        # 0,5 s igual: corte
    assert cv_.congelada(est, _img(120), 30.0) is True
    assert cv_.congelada(est, _img(140), 30.1) is False      # vuelve el vídeo


class _Demo:
    pass


class _Tracker:
    def __init__(self):
        self.reinicios = 0

    def reset(self):
        self.reinicios += 1


def test_al_volver_alarga_los_generos_pendientes_y_no_reinicia_si_fue_corto():
    est, demo, bt = {}, _Demo(), _Tracker()
    da.aplazar(demo, 7, 100.0)
    cv_.congelada(est, _img(1), 100.0, demo, bt)
    cv_.congelada(est, _img(1), 100.5, demo, bt)            # pausa desde 100,5
    cv_.congelada(est, _img(1), 129.5, demo, bt)
    cv_.congelada(est, _img(2), 130.5, demo, bt)            # vuelve: 30 s de pausa
    assert bt.reinicios == 0
    assert da.purgar(demo, 130.5, 9.0, lambda t: None) == 0  # sin el parche, vencía
    assert da.purgar(demo, 140.0, 9.0, lambda t: None) == 1


def test_tras_una_pausa_larga_se_reinicia_el_seguimiento():
    est, bt = {}, _Tracker()
    cv_.congelada(est, _img(1), 0.0, None, bt)
    cv_.congelada(est, _img(1), 1.0, None, bt)
    cv_.congelada(est, _img(2), 1000.0, None, bt)
    assert bt.reinicios == 1


def test_la_respuesta_en_pausa_no_lleva_cajas_ni_repite_alertas():
    est = {}
    cv_.recordar(est, {'alerts': [{'x': 1}], 'detections': [{'box': [0, 0, 1, 1]}], 'zone_count': 2})
    m = cv_.metadata_congelada(est)
    assert m['alerts'] == [] and m['detections'] == [] and m['congelado'] is True
    assert m['zone_count'] == 2
    assert cv_.metadata_congelada({})['congelado'] is True


def test_un_fallo_procesa_la_imagen():
    assert cv_.congelada({}, None, 0.0) is False


def test_tiendas_y_anaqueles_lo_usan_donde_deben():
    src = (RAIZ / 'tiendas.py').read_text(encoding='utf-8')
    i = src.index('if cortes_video.activo():')
    assert i < src.index('self.frame_counter += 1', i)
    assert src.index('try:', i - 400) < i                    # dentro del try (se restaura el estado)
    assert 'cortes_video.recordar(cam_state, metadata)' in src
    anq = (RAIZ / 'anaqueles.py').read_text(encoding='utf-8')
    assert "metadata.get('congelado')" in anq
