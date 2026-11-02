"""
Pruebas del grabador de tramos con gente (30-oct-2026,
`src/app/grabador_personas.py`).

Lo pidió el usuario para comparar versiones sobre las MISMAS imágenes: graba
solo cuando hay seguimientos activos, con 2 s de antes y 3 s de después, y
guarda el JPEG tal como llegó. Tiempos inyectados, sin servidor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app import grabador_personas as gp  # noqa: E402

CAM = 'win-iVMS-4200-c4-i58bf7b'


def _jpeg(n: int) -> bytes:
    return b'\xff\xd8' + n.to_bytes(4, 'big') + b'\xff\xd9'


def _pasar(g, segundos_con_gente, t0=1000.0, paso=0.5, hasta=20.0):
    """Una imagen cada `paso` s; hay gente en [ini, fin) de cada tramo."""
    t, n = t0, 0
    while t < t0 + hasta:
        gente = any(a <= t - t0 < b for a, b in segundos_con_gente)
        g.ver(CAM, _jpeg(n), {'active_tracks': 1 if gente else 0},
              {'roi': [1, 2], 'camera_id': CAM}, ahora=t)
        t += paso
        n += 1


def _tramos(carpeta: Path):
    return sorted((carpeta / CAM).iterdir()) if (carpeta / CAM).exists() else []


def _lineas(tramo: Path):
    return [json.loads(x) for x in (tramo / 'imagenes.jsonl').read_text(
        encoding='utf-8').splitlines()]


def test_sin_gente_no_graba_nada(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 100)
    _pasar(g, [])
    assert _tramos(tmp_path) == []


def test_un_tramo_con_antes_y_despues(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 100)
    _pasar(g, [(5.0, 8.0)])                     # gente de 5 a 8 s
    tramos = _tramos(tmp_path)
    assert len(tramos) == 1
    lineas = _lineas(tramos[0])
    ts = [x['t'] - 1000.0 for x in lineas]
    # 2 s antes (3,0-4,5), con gente (5,0-7,5) y 3 s después (8,0-11,0).
    assert ts[0] == pytest.approx(3.0)
    assert ts[-1] == pytest.approx(11.0)
    assert len(lineas) == len(list(tramos[0].glob('*.jpg')))


def test_el_jpeg_se_guarda_tal_como_llego(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 100)
    _pasar(g, [(5.0, 6.0)])
    tramo = _tramos(tmp_path)[0]
    primera = _lineas(tramo)[0]
    assert (tramo / primera['imagen']).read_bytes() == _jpeg(6)   # t=3,0 s
    assert primera['parametros'] == {'roi': [1, 2], 'camera_id': CAM}


def test_dos_visitas_separadas_son_dos_tramos(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 100)
    _pasar(g, [(2.0, 3.0), (12.0, 13.0)], hasta=25.0)
    assert len(_tramos(tmp_path)) == 2


def test_para_al_llegar_al_maximo(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 30 / 1024 / 1024)      # 30 bytes
    _pasar(g, [(0.0, 20.0)])
    assert g.parado
    assert g.bytes <= 30


def test_sin_la_variable_no_hay_grabador(monkeypatch):
    monkeypatch.delenv('ELDE_GRABAR_PERSONAS', raising=False)
    monkeypatch.setattr(gp, '_INTENTADO', False)
    monkeypatch.setattr(gp, '_GRABADOR', None)
    assert gp.grabador() is None
    gp.ver(CAM, _jpeg(1), {'active_tracks': 1}, {})       # no lanza


def test_app_le_pasa_cada_imagen():
    src = (RAIZ / 'server' / 'src' / 'app' / 'app.py').read_text(
        encoding='utf-8')
    assert '_grabador_personas.ver(camera_id, image_data, metadata, ' \
           '_parametros)' in src


# ── Parar y reanudar sin reiniciar (2-nov-2026, `api_grabador_personas.py`) ──

def test_parar_cierra_el_tramo_y_no_graba_mas(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 100)
    _pasar(g, [(0.0, 20.0)], hasta=4.0)          # tramo abierto
    antes = len(_lineas(_tramos(tmp_path)[0]))
    g.parar()
    assert g.estado()['grabando'] is False
    _pasar(g, [(0.0, 20.0)], t0=1010.0)
    assert len(_tramos(tmp_path)) == 1
    assert len(_lineas(_tramos(tmp_path)[0])) == antes


def test_reanudar_vuelve_a_grabar_en_otro_tramo(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 100)
    g.parar()
    _pasar(g, [(0.0, 20.0)], hasta=4.0)
    assert _tramos(tmp_path) == []
    assert g.reanudar() is True
    _pasar(g, [(0.0, 20.0)], t0=1010.0, hasta=4.0)
    assert len(_tramos(tmp_path)) == 1


def test_no_reanuda_si_llego_al_maximo(tmp_path):
    g = gp.GrabadorPersonas(tmp_path, 30 / 1024 / 1024)
    _pasar(g, [(0.0, 20.0)])
    assert g.parado and g.reanudar() is False


def test_la_api_para_y_reanuda(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.app import api_grabador_personas as api
    monkeypatch.setattr(gp, '_INTENTADO', True)
    monkeypatch.setattr(gp, '_GRABADOR', gp.GrabadorPersonas(tmp_path, 100))
    app = FastAPI()
    app.include_router(api.router)
    c = TestClient(app)
    assert c.get('/api/v1/tienda/grabador').json()['grabando'] is True
    assert c.post('/api/v1/tienda/grabador/parar').json()['grabando'] is False
    assert c.post('/api/v1/tienda/grabador/reanudar').json()['grabando'] is True


def test_la_api_sin_grabador_configurado(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.app import api_grabador_personas as api
    monkeypatch.setattr(gp, '_INTENTADO', True)
    monkeypatch.setattr(gp, '_GRABADOR', None)
    app = FastAPI()
    app.include_router(api.router)
    r = TestClient(app).post('/api/v1/tienda/grabador/parar').json()
    assert r['configurado'] is False


def test_app_monta_la_api_del_grabador():
    src = (RAIZ / 'server' / 'src' / 'app' / 'app.py').read_text(
        encoding='utf-8')
    assert 'app.include_router(api_grabador_personas_router)' in src
