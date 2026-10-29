"""Género v2 en las fichas cerradas (27-oct-2026, `genero_v2.py`).

Se fija: la regla (2ª opinión solo de Hombre a Mujer; indecisas solo con
mucha seguridad); que solo revisa fichas quietas, una vez, y que la marca
`genero_v2` evita repetir; que escribe el origen y el género anterior y anota
en la base; que no recrea la ficha que ya se movió; y el interruptor.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import genero_v2 as g2  # noqa: E402


class ModeloFijo:
    def __init__(self, p):
        self.p, self.llamadas = p, 0

    def p_mujer(self, img):
        self.llamadas += 1
        return self.p


class Almacen:
    def __init__(self):
        self.clas, self.demo = [], []

    def anotar_clasificacion(self, cam, stem, genero, edad, person_uuid=None, solo_cuerpo=False):
        self.clas.append((cam, stem, genero, person_uuid))

    def anotar_demografia(self, cam, puid, genero, edad):
        self.demo.append((cam, puid, genero))


def _ficha(carpeta, stem, meta, hace_s=1000.0):
    cv2.imwrite(str(carpeta / f'{stem}.jpg'), np.full((256, 128, 3), 120, np.uint8))
    jp = carpeta / f'{stem}.json'
    jp.write_text(json.dumps(meta), encoding='utf-8')
    t = time.time() - hace_s
    os.utime(jp, (t, t))
    return jp


def _hoy(n):
    return time.strftime('%Y%m%d_120000') + f'_cam_t{n}'


def test_regla():
    assert g2.decidir('Hombre', 0.95) == 'Mujer'
    assert g2.decidir('Hombre', 0.85) is None
    assert g2.decidir('Mujer', 0.01) is None              # nunca de Mujer a Hombre
    assert g2.decidir(None, 0.96) == 'Mujer'
    assert g2.decidir(None, 0.03) == 'Hombre'
    assert g2.decidir(None, 0.92) is None                 # indecisa: hace falta >= 0,95


def test_revisa_fichas_quietas_una_vez(tmp_path):
    _ficha(tmp_path, _hoy(1), {'gender': 'Hombre'})
    _ficha(tmp_path, _hoy(2), {'gender': None}, hace_s=10)        # aún en su espera
    _ficha(tmp_path, _hoy(3), {'gender': None, 'genero_v2': 0.5})  # ya revisada
    _ficha(tmp_path, '20200101_120000_cam_t4', {'gender': None})  # de otro día
    m = ModeloFijo(0.97)
    r = g2.Revisor(tmp_path, modelo=m)
    assert r.buscar() == 1 and r.buscar() == 0
    assert r.resultados.get_nowait() == (_hoy(1), 0.97) and m.llamadas == 1


def test_aplica_segunda_opinion_y_anota(tmp_path):
    jp = _ficha(tmp_path, _hoy(1), {'gender': 'Hombre', 'age_range': '26-35',
                                    'device_id': 'cam3', 'person_uuid': 'u1'})
    a = Almacen()
    assert g2.aplicar_uno(tmp_path, _hoy(1), 0.97, a) == 'Mujer'
    meta = json.loads(jp.read_text(encoding='utf-8'))
    assert meta['gender'] == 'Mujer' and meta['genero_sistema'] == 'Hombre'
    assert meta['origen_demografia'] == g2.ORIGEN and meta['genero_v2'] == 0.97
    assert a.clas == [('cam3', _hoy(1), 'Mujer', 'u1')] and a.demo == [('cam3', 'u1', 'Mujer')]
    assert g2.aplicar_uno(tmp_path, _hoy(1), 0.97, a) is None    # no se repite


def test_indecisa_y_sin_cambio_solo_marca(tmp_path):
    jp = _ficha(tmp_path, _hoy(1), {'gender': None, 'motivo_sin_demografia': 'muestras_insuficientes'})
    assert g2.aplicar_uno(tmp_path, _hoy(1), 0.02, Almacen()) == 'Hombre'
    meta = json.loads(jp.read_text(encoding='utf-8'))
    assert meta['gender'] == 'Hombre' and 'motivo_sin_demografia' not in meta
    jp2 = _ficha(tmp_path, _hoy(2), {'gender': 'Mujer'})
    a = Almacen()
    assert g2.aplicar_uno(tmp_path, _hoy(2), 0.10, a) is None and a.clas == []
    assert json.loads(jp2.read_text(encoding='utf-8')) == {'gender': 'Mujer', 'genero_v2': 0.1}


def test_no_recrea_la_ficha_movida(tmp_path):
    jp = _ficha(tmp_path, _hoy(1), {'gender': 'Hombre'})
    (tmp_path / f'{_hoy(1)}.jpg').unlink()                 # el personal se aparta:
    a = Almacen()                                          # la foto ya se fue
    assert g2.aplicar_uno(tmp_path, _hoy(1), 0.99, a) is None
    assert json.loads(jp.read_text(encoding='utf-8')) == {'gender': 'Hombre'} and a.clas == []
    jp.unlink()                                            # y luego su JSON
    assert g2.aplicar_uno(tmp_path, _hoy(1), 0.99, a) is None
    assert not jp.exists()


def test_interruptor(monkeypatch):
    monkeypatch.setenv('ELDE_GENERO_V2', '0')
    assert g2.activo() is False
    g2.aplicar(None)                                       # apagado: ni arranca
    monkeypatch.setenv('ELDE_GENERO_V2', '1')
    assert g2.activo() is True
