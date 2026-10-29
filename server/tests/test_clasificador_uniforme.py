"""Personal por el uniforme, en sombra (16-oct-2026, `clasificador_uniforme.py`).

Con un modelo de mentira (el de verdad, entrenado con fotos de clientes, no
viaja con el repositorio). Se fija: vota a quien se ve en la imagen, con su
cadencia y su alto mínimo; al terminar el seguimiento apunta UNA línea con
la decisión; la decisión pide votos mínimos y media >= umbral; nunca toca el
seguimiento; sin modelo o en «apagado» no hace nada.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import clasificador_uniforme as cu  # noqa: E402

FRAME = np.zeros((300, 300, 3), np.uint8)
ALTA = [10, 10, 60, 210]                    # 200 px de alto
BAJA = [10, 10, 30, 50]                     # 40 px: no se mira


@pytest.fixture(autouse=True)
def modo_sombra(monkeypatch):
    """Cada prueba fija su modo; por defecto «sombra» (el config dice
    «activo» desde el 16-oct)."""
    monkeypatch.setenv('ELDE_UNIFORME', 'sombra')


def _modelo(p):
    llamadas = []

    def fn(recortes):
        llamadas.append(len(recortes))
        return np.full(len(recortes), p)
    fn.llamadas = llamadas
    return fn


def _pasar(voto, cajas, desde, hasta, pid='uuid-1'):
    for fc in range(desde, hasta):
        tracks = {tid: {'box': b, 'last_seen': fc, 'persistent_id': pid} for tid, b in cajas.items()}
        voto.observar(FRAME, tracks, fc, cada=1, t=float(fc))
    return tracks


def _lineas(carpeta):
    return [json.loads(l) for f in sorted(Path(carpeta).glob('*.jsonl'))
            for l in f.read_text(encoding='utf-8').splitlines()]


def test_vota_y_apunta_al_terminar_el_seguimiento(tmp_path):
    voto = cu.VotoUniforme('cam', modelo=_modelo(0.95), carpeta=tmp_path)
    tracks = _pasar(voto, {7: ALTA}, 1, 6)
    assert _lineas(tmp_path) == []                          # sigue a la vista
    voto.observar(FRAME, {}, 20, cada=1, t=20.0)            # 2 s sin verlo
    (fila,) = _lineas(tmp_path)
    assert fila['tid'] == 7 and fila['votos'] == 5 and fila['personal'] is True
    assert fila['pid'] == 'uuid-1' and abs(fila['media'] - 0.95) < 1e-6
    assert fila['serie'] == [0.95] * 5                      # cada voto, en orden
    assert 'es_personal' not in tracks[7]                   # sombra: no lo toca


def _modelo_serie(serie):
    """Devuelve, llamada a llamada, los valores de `serie` (uno por recorte)."""
    it = iter(serie)
    return lambda recortes: np.array([next(it) for _ in recortes])


def test_la_decision_es_la_mayoria_de_votos(tmp_path):
    """16-oct-2026: con «media >= 0,9» se escapaban 7 de 12 empleados que
    trabajaban largo rato; decide la MAYORÍA de votos >= 0,5."""
    casos = {'mayoria': ([0.3, 0.7, 0.8, 0.9, 0.2], True),    # 3 de 5, media 0,58
             'minoria': ([0.9, 0.9, 0.1, 0.2, 0.3], False),   # 2 de 5, media 0,48
             'pocos': ([0.99, 0.99], False)}                   # 2 votos
    for nombre, (serie, esperado) in casos.items():
        voto = cu.VotoUniforme('cam', modelo=_modelo_serie(serie), carpeta=tmp_path / nombre)
        _pasar(voto, {1: ALTA}, 1, 1 + len(serie))
        voto.observar(FRAME, {}, 20, cada=1, t=20.0)
        assert _lineas(tmp_path / nombre)[0]['personal'] is esperado, nombre


def test_cadencia_y_alto_minimo(tmp_path):
    m = _modelo(0.95)
    voto = cu.VotoUniforme('cam', modelo=m, carpeta=tmp_path)
    for fc in range(1, 9):
        voto.observar(FRAME, {4: {'box': ALTA, 'last_seen': fc}, 5: {'box': BAJA, 'last_seen': fc}},
                      fc, cada=4, t=float(fc))
    assert sum(m.llamadas) == 2                  # 8 imágenes, cada 4; la pequeña nunca
    voto.observar(FRAME, {}, 30, cada=4, t=30.0)
    assert [f['tid'] for f in _lineas(tmp_path)] == [4]


def test_solo_quien_se_ve_en_esta_imagen(tmp_path):
    m = _modelo(0.95)
    voto = cu.VotoUniforme('cam', modelo=m, carpeta=tmp_path)
    voto.observar(FRAME, {3: {'box': ALTA, 'last_seen': 1}}, 2, cada=1, t=2.0)   # caja vieja
    assert m.llamadas == []


def test_activo_marca_personal_al_cumplir_la_regla_y_no_lo_quita(tmp_path, monkeypatch):
    """16-oct-2026 (lo pidió el usuario): en «activo» se pone la MISMA marca
    del personal dado de alta, en cuanto hay 3 votos y mayoría."""
    monkeypatch.setenv('ELDE_UNIFORME', 'activo')
    voto = cu.VotoUniforme('cam', modelo=_modelo_serie([0.9, 0.95, 0.97, 0.1, 0.1, 0.1]),
                           carpeta=tmp_path)
    tracks = {8: {'box': ALTA, 'persistent_id': 'prov:8'}}
    devueltos = []
    for fc in range(1, 7):
        tracks[8]['last_seen'] = fc
        devueltos.append(voto.observar(FRAME, tracks, fc, cada=1, t=float(fc)))
    assert devueltos == [[], [], [8], [], [], []]           # al 3.er voto, una vez
    assert tracks[8]['es_personal'] is True                 # y no se quita
    assert tracks[8]['personal_via'] == 'modelo_uniforme'
    voto.observar(FRAME, {}, 20, cada=1, t=20.0)
    assert _lineas(tmp_path)[0]['marcado'] == cu.time.strftime('%H:%M:%S', cu.time.localtime(3.0))


def test_activo_no_marca_a_quien_no_cumple(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_UNIFORME', 'activo')
    voto = cu.VotoUniforme('cam', modelo=_modelo(0.2), carpeta=tmp_path)
    tracks = {2: {'box': ALTA}}
    for fc in range(1, 8):
        tracks[2]['last_seen'] = fc
        assert voto.observar(FRAME, tracks, fc, cada=1, t=float(fc)) == []
    assert 'es_personal' not in tracks[2]


def test_sin_modelo_o_apagado_no_hace_nada(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, '_modelo', None)
    monkeypatch.setattr(cu, '_RAIZ', tmp_path)              # models/ vacío
    assert cu._obtener_modelo() is None
    voto = cu.VotoUniforme('cam', carpeta=tmp_path / 'out')
    _pasar(voto, {1: ALTA}, 1, 6)
    voto.observar(FRAME, {}, 20, cada=1, t=20.0)
    assert not (tmp_path / 'out').exists()
    monkeypatch.setenv('ELDE_UNIFORME', 'apagado')
    assert cu.modo() == 'apagado'
    llamado = []
    monkeypatch.setattr(cu, 'VotoUniforme', lambda *a, **k: llamado.append(1))
    cu.observar('cam', FRAME, {1: {'box': ALTA, 'last_seen': 1}}, 1, 1)
    assert llamado == []


def test_modelo_de_tipos_marca_al_vigilante_con_su_rol(tmp_path, monkeypatch):
    """19-oct-2026: el modelo de 4 salidas (sin uniforme, camiseta,
    vigilante, mantenimiento) pone el rol del tipo que más votó."""
    monkeypatch.setenv('ELDE_UNIFORME', 'activo')
    vig = np.array([0.1, 0.1, 0.75, 0.05])                  # «es personal» 0,9
    voto = cu.VotoUniforme('cam', modelo=lambda r: np.tile(vig, (len(r), 1)),
                           carpeta=tmp_path)
    tracks = {3: {'box': ALTA}}
    for fc in range(1, 4):
        tracks[3]['last_seen'] = fc
        voto.observar(FRAME, tracks, fc, cada=1, t=float(fc))
    assert tracks[3]['es_personal'] is True
    assert tracks[3]['personal_rol'] == 'vigilante'
    voto.observar(FRAME, {}, 20, cada=1, t=20.0)
    assert _lineas(tmp_path)[0]['tipo'] == 'vigilante'


def test_modelo_viejo_sigue_siendo_camiseta(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_UNIFORME', 'activo')
    voto = cu.VotoUniforme('cam', modelo=_modelo(0.95), carpeta=tmp_path)
    tracks = {4: {'box': ALTA}}
    for fc in range(1, 4):
        tracks[4]['last_seen'] = fc
        voto.observar(FRAME, tracks, fc, cada=1, t=float(fc))
    assert tracks[4]['personal_rol'] == 'pasillero'


def test_ficha_protegida_por_sus_primeros_votos_en_sombra(tmp_path, monkeypatch):
    """23-oct-2026: la ficha de un cliente se apartaba cuando su número (o su
    identidad) pasaba a un empleado. En sombra: ¿la protegerían sus 2
    primeros votos «sin uniforme»? Se apunta y se sigue funcionando igual."""
    casos = {'cliente_luego_empleado': ([0.0, 0.1, 0.99, 0.99, 0.99], True),
             'empleado_desde_el_principio': ([0.9, 0.95, 0.99], False),
             'un_voto_bajo': ([0.1, 0.9, 0.99], False)}
    for nombre, (serie, esperado) in casos.items():
        voto = cu.VotoUniforme('cam', modelo=_modelo_serie(serie), carpeta=tmp_path / nombre)
        monkeypatch.setitem(cu._por_camara, 'cam', voto)
        for fc in range(1, 1 + len(serie)):
            voto.observar(FRAME, {1: {'box': ALTA, 'last_seen': fc}}, fc, cada=1, t=float(fc),
                          fichas={1: 'ficha_1'})
        voto.observar(FRAME, {}, 50, cada=1, t=50.0)        # el seguimiento ya terminó
        assert cu.ficha_protegida('cam', 'ficha_1', 1, 'por_identidad') is esperado, nombre
        (fila,) = [json.loads(l) for l in next((tmp_path / nombre).glob('fichas_protegidas_*.jsonl'))
                   .read_text(encoding='utf-8').splitlines()]
        assert fila['protegeria'] is esperado and fila['primeros'] == [round(x, 2) for x in serie[:2]]
    monkeypatch.delitem(cu._por_camara, 'cam')
    assert cu.ficha_protegida('cam', 'ficha_1') is False                # cámara sin votos: nada
