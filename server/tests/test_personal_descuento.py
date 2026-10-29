"""Quien resulta ser personal a mitad de su seguimiento se descuenta (19-oct-2026).

Medido el 16-oct: las 8 personas marcadas por el uniforme con identidad
tenían su visita en la base, y 40 fichas eran de personal. Se fija: al
marcar, su visita y eventos de la jornada se borran (también los aún sin
volcar), sale de la máquina DENTRO/FUERA sin emitir salida, su ficha se
APARTA (no se borra) y su clasificación se borra; un `prov:` solo aparta ficha.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import personal_descuento as pd  # noqa: E402
from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402


class Proc:
    def __init__(self, pid, stem=None):
        self.active_tracks = {5: {'persistent_id': pid, 'es_personal': True}}
        self._confirmed_unique = {pid}
        self._unique_entered = {pid}
        self._zone_state = {pid: {'estado': 'DENTRO'}}
        self._capture_meta = {5: stem} if stem else {}
        self._captured_demo_done = set()
        self._captured_persons = {}


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_DIR', str(tmp_path / 'cap'), raising=False)
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_CLIENT_DIR', '', raising=False)
    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db', periodo_volcado_s=3600)
    (tmp_path / 'cap' / 'persons').mkdir(parents=True)
    return tmp_path, alm or ta.get_almacen()


def _filas(tmp_path, sql):
    con = sqlite3.connect(tmp_path / 'tienda.db')
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_borra_visita_y_eventos_tambien_los_sin_volcar(entorno):
    tmp_path, alm = entorno
    t = time.time()
    alm.anotar_entrada('cam', 'uuid-emp', t - 20)
    alm.anotar_entrada('cam', 'uuid-cli', t - 20)
    alm.volcar(force=True)
    alm.anotar_salida('cam', 'uuid-emp', 5.0, t - 10)        # aún en la cola
    proc = Proc('uuid-emp')
    pd.descontar(proc, 5, 'cam', t)
    alm.volcar(force=True)
    assert _filas(tmp_path, 'SELECT persistent_id FROM tienda_visitas') == [('uuid-cli',)]
    assert _filas(tmp_path, "SELECT COUNT(*) FROM tienda_eventos WHERE persistent_id='uuid-emp'") == [(0,)]
    assert 'uuid-emp' not in proc._zone_state                 # sin salida pendiente
    assert not proc._confirmed_unique and not proc._unique_entered
    assert pd.es_personal(proc, 'uuid-emp') and not pd.es_personal(proc, 'uuid-cli')


def test_aparta_la_ficha_y_borra_su_clasificacion(entorno):
    tmp_path, alm = entorno
    stem = '20261019_100000_cam_t5'
    for ext in ('.jpg', '.json'):
        (tmp_path / 'cap' / 'persons' / f'{stem}{ext}').write_text('x')
    alm.anotar_clasificacion('cam', stem, 'Hombre', '26-35', ts=time.time())
    alm.volcar(force=True)
    proc = Proc('prov:5', stem)
    pd.descontar(proc, 5, 'cam')
    alm.volcar(force=True)
    assert list((tmp_path / 'cap' / 'persons').iterdir()) == []
    assert sorted(p.name for p in (tmp_path / 'cap' / pd.CARPETA).iterdir()) == [f'{stem}.jpg', f'{stem}.json']
    assert _filas(tmp_path, 'SELECT COUNT(*) FROM tienda_clasificaciones') == [(0,)]
    assert 5 in proc._captured_demo_done and 5 not in proc._capture_meta
    assert not pd.es_personal(proc, 'prov:5')                 # prov: no se recuerda
    assert not proc._zone_state                               # ni su permanencia en vivo


def test_no_se_clasifica_al_personal_y_se_tira_lo_acumulado(entorno):
    class Demo:
        quitados = []

        def remove_track(self, tid):
            self.quitados.append(tid)
    proc = Proc('uuid-emp')
    proc._demographics = Demo()
    proc.active_tracks[6] = {'persistent_id': 'uuid-emp'}      # otro seguimiento suyo
    proc.active_tracks[7] = {'persistent_id': 'uuid-cli'}
    assert pd.no_clasificar(proc, 5)                           # marcado
    assert not pd.no_clasificar(proc, 6)                       # aún no se sabe
    pd.descontar(proc, 5, 'cam')
    assert Demo.quitados == [5]
    assert pd.no_clasificar(proc, 6) and not pd.no_clasificar(proc, 7)


def test_nunca_lanza(entorno):
    pd.descontar(object(), 5, 'cam')


def test_heredar_la_marca_por_costura_aparta_la_ficha_nueva(entorno, tmp_path, monkeypatch):
    """22-oct-2026, 12:41:22 en vivo: el número nuevo de un empleado abrió
    ficha de cliente, la costura le pasó la marca y la ficha se quedó.
    Desde el 23-oct la marca espera a 2 votos propios del uniforme."""
    from src.analityc.core import clasificador_uniforme as cu
    from src.analityc.core.costura_sombra import CosturaSombra
    monkeypatch.setattr(cu, 'votos', lambda cam, tid: [0.97, 0.99])
    tmp, _alm = entorno
    stem = '20261022_124123_cam_t111'
    for ext in ('.jpg', '.json'):
        (tmp / 'cap' / 'persons' / f'{stem}{ext}').write_text('x')
    proc = Proc('prov:111', stem)
    proc.active_tracks = {111: {'persistent_id': 'prov:111'}}
    proc._capture_meta = {111: stem}
    cos = CosturaSombra('cam', carpeta=tmp_path / 'costura')
    cos._vistos = proc.active_tracks
    assert cos._heredar(109, {'pid': 'prov:109', 'confirmado': False,
                              'personal': True, 'rol': 'pasillero'}, 111)
    assert not pd.tras_costura(proc, 111, 'cam')         # aún sin votos propios
    cos._resolver_personal(proc.active_tracks)
    assert cos.personal_confirmado == [111]
    assert proc.active_tracks[111]['es_personal']
    assert pd.tras_costura(proc, 111, 'cam')
    assert list((tmp / 'cap' / 'persons').iterdir()) == []
    assert 111 not in proc._capture_meta


def test_tras_costura_no_toca_a_quien_no_hereda_la_marca(entorno):
    tmp, _alm = entorno
    stem = '20261022_124203_cam_t113'
    (tmp / 'cap' / 'persons' / f'{stem}.json').write_text('x')
    proc = Proc('uuid-cli', stem)
    proc.active_tracks = {113: {'persistent_id': 'uuid-cli'}}       # cliente
    proc._capture_meta = {113: stem}
    assert not pd.tras_costura(proc, 113, 'cam')
    proc.active_tracks[113].update(es_personal=True)                 # marcado por el uniforme
    assert not pd.tras_costura(proc, 113, 'cam')                     # eso ya lo descuenta su camino
    assert [p.name for p in (tmp / 'cap' / 'persons').iterdir()] == [f'{stem}.json']
