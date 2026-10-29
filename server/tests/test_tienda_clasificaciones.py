"""Clasificados que se perdían por cerrar su ficha demasiado pronto.

Medido el 24-sep-2026 sobre el vídeo de prueba: la ficha de cada persona se
cerraba en cuanto se conocía el género O la identidad, lo primero que llegara,
y ya no se volvía a mirar. Dos pérdidas, simétricas:

  - **género primero**: sin identidad, el género no puede ir a
    `tienda_visitas`. Ese día 3 de 3 clasificados no llegaron a la base.
  - **identidad primero**: la ficha se sellaba con «muestras_insuficientes»
    mientras el clasificador seguía trabajando. El track 100004 se cerró a
    las 21:02:53 y a los 34 s se clasificó como Mujer, 36-50: perdido.

Las pruebas de `FotosMixin` reproducen esos dos casos sobre el código REAL
de las fichas, con la demografía y el reidentificador de mentira. Son las que
tienen que caer si alguien vuelve a cerrar la ficha sin esperar.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.tienda_clasificaciones import (  # noqa: E402
    EsperaDemografia, MAX_ESPERAS)
from src.analityc.core.tiendas_fotos import FotosMixin  # noqa: E402

MUJER = {'gender': 'Mujer', 'age_range': '36-50', 'confidence': 0.91,
         'n_face_samples': 4}
DESCONOCIDO = {'gender': 'Desconocido', 'age_range': 'Desconocido'}


# ── Base de pruebas ──────────────────────────────────────────────────────

@pytest.fixture
def almacen(tmp_path, monkeypatch):
    """Base recién hecha, escritura inmediata y jornada al día natural."""
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    return ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')


def _visita(almacen, uuid):
    almacen.volcar(force=True)
    with almacen._conectar() as con:
        fila = con.execute(
            'SELECT genero, edad FROM tienda_visitas WHERE persistent_id=?',
            (uuid,)).fetchone()
    # El almacén usa `sqlite3.Row`, que no se compara igual que una tupla.
    return tuple(fila) if fila is not None else None


# ── 1. La espera, sola ───────────────────────────────────────────────────

def _nada(_tid):
    return None


def test_la_identidad_que_llega_tarde_se_entrega():
    e = EsperaDemografia()
    e.esperar_identidad(7, 's7', 'cam-1', 'Mujer', 'Joven', ahora=100.0)
    ids, gens = e.revisar('cam-1', _nada, _nada, ahora=101.0)
    assert (ids, gens) == ([], []), 'sin identidad todavía: sigue esperando'
    assert len(e) == 1
    ids, _ = e.revisar('cam-1', lambda t: 'uuid-7', _nada, ahora=102.0)
    assert ids == [(7, 's7', 'uuid-7', 'Mujer', 'Joven')]
    assert len(e) == 0, 'entregada una vez, olvidada'


def test_el_genero_que_llega_tarde_se_entrega():
    e = EsperaDemografia()
    e.esperar_genero(4, 's4', 'cam-1', 'uuid-4', ahora=100.0)
    _, gens = e.revisar('cam-1', _nada, lambda t: DESCONOCIDO, ahora=101.0)
    assert gens == [], '«Desconocido» no es un género: sigue esperando'
    _, gens = e.revisar('cam-1', _nada, lambda t: MUJER, ahora=134.0)
    assert gens == [(4, 's4', 'uuid-4', MUJER)]


def test_caduca_a_los_cinco_minutos():
    """Si no llegó en 5 min, la persona ya se fue: no se espera para siempre."""
    e = EsperaDemografia(ttl_s=300)
    e.esperar_genero(1, 's1', 'cam-1', 'u1', ahora=0.0)
    e.revisar('cam-1', _nada, _nada, ahora=301.0)
    assert len(e) == 0
    _, gens = e.revisar('cam-1', _nada, lambda t: MUJER, ahora=302.0)
    assert gens == []


def test_solo_se_resuelve_en_su_camara():
    """Demografía y seguimiento son por cámara: el track 5 de la cámara A no
    se puede resolver con el clasificador de la B."""
    e = EsperaDemografia()
    e.esperar_identidad(5, 's5', 'cam-A', 'Hombre', 'Adulto', ahora=0.0)
    ids, _ = e.revisar('cam-B', lambda t: 'uuid-ajeno', _nada, ahora=1.0)
    assert ids == []
    ids, _ = e.revisar('cam-A', lambda t: 'uuid-5', _nada, ahora=2.0)
    assert ids and ids[0][2] == 'uuid-5'


def test_una_consulta_que_falla_cuenta_como_todavia_no():
    """Corre dentro del bucle de inferencia: no puede lanzar."""
    def revienta(_tid):
        raise RuntimeError('reidentificador caído')
    e = EsperaDemografia()
    e.esperar_identidad(1, 's1', 'c', 'Mujer', None, ahora=0.0)
    e.esperar_genero(2, 's2', 'c', 'u2', ahora=0.0)
    assert e.revisar('c', revienta, revienta, ahora=1.0) == ([], [])
    assert len(e) == 2


def test_la_lista_tiene_tope():
    e = EsperaDemografia()
    for i in range(MAX_ESPERAS + 50):
        e.esperar_genero(i, f's{i}', 'c', f'u{i}', ahora=0.0)
    assert len(e) == MAX_ESPERAS


# ── 2. La tabla nueva ────────────────────────────────────────────────────

def test_se_guarda_al_clasificado_sin_identidad(almacen):
    almacen.anotar_clasificacion('cam-1', 'st1', 'Mujer', 'Joven')
    c = almacen.clasificaciones()
    assert c['sin_identidad'] == {'total': 1, 'genero': {'Mujer': 1},
                                  'edad': {'Joven': 1}}
    assert c['con_identidad']['total'] == 0


def test_la_identidad_tardia_lo_mueve_de_grupo(almacen):
    almacen.anotar_clasificacion('cam-1', 'st1', 'Hombre', 'Adulto')
    almacen.identificar_clasificacion('st1', 'uuid-1')
    c = almacen.clasificaciones()
    assert c['sin_identidad']['total'] == 0
    assert c['con_identidad']['total'] == 1
    assert c['con_identidad']['genero'] == {'Hombre': 1}


def test_una_identidad_escrita_no_se_pisa(almacen):
    almacen.anotar_clasificacion('cam-1', 'st1', 'Mujer', None,
                                 person_uuid='uuid-bueno')
    almacen.identificar_clasificacion('st1', 'uuid-otro')
    almacen.volcar(force=True)
    with almacen._conectar() as con:
        uuid = con.execute('SELECT person_uuid FROM tienda_clasificaciones '
                           "WHERE stem='st1'").fetchone()[0]
    assert uuid == 'uuid-bueno'


def test_con_identidad_cuenta_personas_no_capturas(almacen):
    """La misma persona reconocida en dos capturas es UNA persona."""
    almacen.anotar_clasificacion('cam-1', 'st1', 'Mujer', 'Joven',
                                 person_uuid='uuid-1')
    almacen.anotar_clasificacion('cam-1', 'st2', 'Mujer', 'Joven',
                                 person_uuid='uuid-1')
    c = almacen.clasificaciones()
    assert c['con_identidad']['total'] == 1
    assert c['con_identidad']['genero'] == {'Mujer': 1}


def test_desconocido_no_es_una_clasificacion(almacen):
    almacen.anotar_clasificacion('cam-1', 'st1', 'Desconocido', 'Desconocido')
    almacen.anotar_clasificacion('cam-1', 'st2', '', None)
    assert almacen.clasificaciones()['total'] == 0


def test_la_purga_tambien_la_limpia(almacen):
    viejo = time.time() - 500 * 86400
    almacen.anotar_clasificacion('cam-1', 'viejo', 'Mujer', None, ts=viejo)
    almacen.anotar_clasificacion('cam-1', 'nuevo', 'Mujer', None)
    almacen.volcar(force=True)
    almacen.purgar(retencion_dias=400)
    with almacen._conectar() as con:
        quedan = [r[0] for r in con.execute(
            'SELECT stem FROM tienda_clasificaciones')]
    assert quedan == ['nuevo'], 'sin purga, la tabla crecería sin fin'


def test_las_visitas_siguen_igual(almacen):
    """El volcado aprendió operaciones nuevas: las de siempre no cambian."""
    almacen.anotar_entrada('cam-1', 'p1')
    almacen.anotar_demografia('cam-1', 'p1', 'Hombre', 'Adulto')
    assert _visita(almacen, 'p1') == ('Hombre', 'Adulto')


# ── 3. Los dos casos del 24-sep, sobre el código real de las fichas ─────

class _Demo:
    def __init__(self):
        self.datos = {}

    def get_cached(self, tid):
        return self.datos.get(tid)


class _Reid:
    def __init__(self):
        self.ids = {}

    def get_persistent_id(self, tid):
        return self.ids.get(tid)

    def get_all_persons(self):
        return []


class _Camara(FotosMixin):
    """Lo mínimo que usa el cierre de fichas; el resto es el código real."""

    def __init__(self, cam='cam-1'):
        self._capture_meta = {}
        self._captured_demo_done = set()
        self._captured_persons = {}
        self._camera_id_actual = cam
        self.active_tracks = {}
        self._demographics = _Demo()
        self._reidentifier = _Reid()
        self.copias_cliente = []

    def _ficha_personal_tienda(self, pid):
        return None

    def _write_client_capture(self, stem, meta, *a, **k):
        self.copias_cliente.append((stem, dict(meta)))

    def _motivo_sin_demografia(self, tid):
        return 'muestras_insuficientes'


@pytest.fixture
def camara(tmp_path, monkeypatch, almacen):
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_DIR', str(tmp_path / 'cap'))
    (tmp_path / 'cap' / 'persons').mkdir(parents=True)
    return _Camara()


def _ficha(tmp_path, cam, tid):
    stem = f'20260924_210253_cam_t{tid}'
    (tmp_path / 'cap' / 'persons' / f'{stem}.json').write_text(
        json.dumps({'track_id': tid}), encoding='utf-8')
    cam._capture_meta[tid] = stem
    return stem


def _leer(tmp_path, stem):
    return json.loads((tmp_path / 'cap' / 'persons' / f'{stem}.json')
                      .read_text(encoding='utf-8'))


def test_genero_primero_la_identidad_tardia_llega_a_la_base(
        tmp_path, camara, almacen):
    """El caso de los 3 clasificados perdidos del 24-sep."""
    stem = _ficha(tmp_path, camara, 100041)
    camara._demographics.datos[100041] = dict(MUJER, age_range='Adulto')
    camara._update_captures_demographics()

    c = almacen.clasificaciones()
    assert c['sin_identidad']['genero'] == {'Mujer': 1}, 'ANTES se perdía'
    assert _visita(almacen, 'uuid-A') is None

    camara._reidentifier.ids[100041] = 'uuid-A'       # la identidad, tarde
    camara._update_captures_demographics()

    assert _visita(almacen, 'uuid-A') == ('Mujer', 'Adulto'), \
        'la identidad llegó después y el género SÍ tiene que ir a la base'
    c = almacen.clasificaciones()
    assert c['con_identidad']['total'] == 1
    assert c['sin_identidad']['total'] == 0
    assert _leer(tmp_path, stem)['person_uuid'] == 'uuid-A'


def test_identidad_primero_el_genero_tardio_no_se_pierde(
        tmp_path, camara, almacen):
    """El track 100004 del 24-sep: cerrado con «muestras_insuficientes» y
    clasificado 34 s después."""
    stem = _ficha(tmp_path, camara, 100004)
    camara._reidentifier.ids[100004] = 'uuid-B'
    camara._demographics.datos[100004] = DESCONOCIDO
    camara._update_captures_demographics()

    assert _leer(tmp_path, stem)['motivo_sin_demografia'] == \
        'muestras_insuficientes'
    assert not getattr(camara, '_alertas_pendientes', []), \
        'sin género todavía no hay visitante que avisar'

    camara._demographics.datos[100004] = MUJER        # 34 s después
    camara._update_captures_demographics()

    ficha = _leer(tmp_path, stem)
    assert ficha['gender'] == 'Mujer' and ficha['age_range'] == '36-50'
    assert 'motivo_sin_demografia' not in ficha, 'ya no es cierto'
    assert _visita(almacen, 'uuid-B') == ('Mujer', '36-50')
    # El aviso espera a que pase la marca de personal (20-oct-2026,
    # `_soltar_visitantes`): se suelta como si ya hubiera pasado la espera.
    import time as _time
    camara._soltar_visitantes(_time.time() + 3600)
    alertas = camara._alertas_pendientes
    assert len(alertas) == 1 and alertas[0]['gender'] == 'Mujer', \
        'el aviso de visitante que no salió en su momento'
    assert almacen.clasificaciones()['con_identidad']['total'] == 1


def test_si_nunca_llega_la_ficha_queda_como_estaba(
        tmp_path, camara, almacen, monkeypatch):
    stem = _ficha(tmp_path, camara, 100027)
    camara._reidentifier.ids[100027] = 'uuid-C'
    camara._demographics.datos[100027] = DESCONOCIDO
    camara._update_captures_demographics()
    for _ in range(3):
        camara._update_captures_demographics()
    ficha = _leer(tmp_path, stem)
    assert 'gender' not in ficha
    assert ficha['motivo_sin_demografia'] == 'muestras_insuficientes'


def test_la_ficha_completa_de_una_vez_no_espera_nada(
        tmp_path, camara, almacen):
    """Con género E identidad a la vez, el camino de siempre: sin esperas."""
    _ficha(tmp_path, camara, 100009)
    camara._reidentifier.ids[100009] = 'uuid-D'
    camara._demographics.datos[100009] = MUJER
    camara._update_captures_demographics()
    assert len(camara._espera_demografia()) == 0
    assert _visita(almacen, 'uuid-D') == ('Mujer', '36-50')
    assert almacen.clasificaciones()['con_identidad']['total'] == 1


# ── 4. La API ────────────────────────────────────────────────────────────

def test_el_endpoint_separa_con_y_sin_identidad(almacen):
    almacen.anotar_clasificacion('cam-1', 'a', 'Mujer', 'Joven')
    almacen.anotar_clasificacion('cam-1', 'b', 'Hombre', 'Adulto',
                                 person_uuid='u1')
    app = FastAPI()
    from src.app.api_tienda import router
    app.include_router(router)
    r = TestClient(app).get('/api/v1/tienda/clasificaciones')
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['sin_identidad']['genero'] == {'Mujer': 1}
    assert d['con_identidad']['genero'] == {'Hombre': 1}
    assert d['total'] == 2
