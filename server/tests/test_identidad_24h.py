"""
Pruebas de la identidad de 24 h (30-oct-2026, rama contructora).

## Por qué existen

Sin anaqueles ni pose, el usuario quiere sacar más de la clasificación y de
la identidad:

  * reconocer a la gente durante 24 h (antes 30 min), con la galería en disco
    para que un reinicio no la olvide (`REID_PERSISTIR_CON_TTL`);
  * que la galería «aprenda» la cara de cada persona con varias imágenes
    durante la visita (`aprender_cara.py`): hasta ese día, un seguimiento con
    identidad no guardaba ninguna plantilla más;
  * clasificar y tomar la huella del cuerpo más seguido, en imágenes reales
    (`TiendasProcess._cada`).

Con 24 h y 15 plantillas por persona la búsqueda en la galería pasa a una
multiplicación de matriz: aquí se comprueba que da lo mismo que el bucle.

Embeddings sintéticos, sin GPU, como `test_reid_ttl.py`.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics import aprender_cara  # noqa: E402
from src.analityc.core.analytics import face_reidentifier as fr  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402

DIA = 86400.0


def _unitario(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


_RNG = np.random.default_rng(7)
#: Persona A, otra vista de A (coseno ~0,8 con la primera) y persona B.
A = _unitario(_RNG.standard_normal(512))
A_LADO = _unitario(A + 0.75 * _unitario(_RNG.standard_normal(512)))
B = _unitario(_RNG.standard_normal(512))
#: Valor de gris del parche -> embedding que devuelve la sesión falsa.
_CARAS = {10: A, 20: A_LADO, 30: B}


class _SesionFalsa:
    """ArcFace de mentira: el parche de gris N da el vector `_CARAS[N]`."""

    def __init__(self):
        self.llamadas = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="in")]

    def get_outputs(self):
        return [types.SimpleNamespace(name="out")]

    def run(self, _salidas, entradas):
        self.llamadas += 1
        blob = next(iter(entradas.values()))
        gris = int(round((float(np.mean(blob)) * 127.5) + 127.5))
        return [_CARAS[gris][None, :]]


def _cara(gris: int) -> np.ndarray:
    return np.full((120, 120, 3), gris, dtype=np.uint8)


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "person_db" / "persons.pkl")


@pytest.fixture(autouse=True)
def _sin_variable(monkeypatch):
    monkeypatch.delenv('ELDE_REID_APRENDER', raising=False)


def _reid(db_path, persistir=False):
    return fr.FaceReidentifier(
        embedding_session=_SesionFalsa(), db_path=db_path,
        similarity_threshold=0.35, reset_policy="ttl", ttl_seconds=DIA,
        persistir=persistir)


def _plantillas(reid, uid):
    return len(fr._get_embeddings(reid._db[uid]))


def test_los_vectores_de_prueba_son_lo_que_dicen():
    """Sin esto, las pruebas de abajo podrían pasar por casualidad."""
    assert 0.6 < float(A @ A_LADO) < 0.9
    assert abs(float(A @ B)) < 0.2


# ── Aprender la cara durante la visita ──────────────────────────────────

def test_un_seguimiento_con_identidad_aprende_otra_vista(db_path):
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    assert _plantillas(reid, uid) == 1

    reid._aprendido_ts.clear()
    otra = reid.identify_or_register(1, _cara(20), face_yaw=5.0)
    assert otra[0] == uid
    assert _plantillas(reid, uid) == 2


def test_sin_aprender_la_rama_rapida_no_guardaba_nada(db_path, monkeypatch):
    """Control negativo: el comportamiento de antes del 30-oct-2026."""
    monkeypatch.setenv('ELDE_REID_APRENDER', '0')
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    reid.identify_or_register(1, _cara(20))
    assert _plantillas(reid, uid) == 1


def test_si_el_numero_salta_a_otra_persona_no_se_mezcla(db_path):
    """El seguimiento 1 era A; su número pasa a B (11-15 % en vivo)."""
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    reid._aprendido_ts.clear()
    reid.identify_or_register(1, _cara(30))
    assert _plantillas(reid, uid) == 1
    assert aprender_cara.parecido_con(fr._get_embeddings(reid._db[uid]),
                                      B) < AnalyticsConfig.REID_APRENDER_SIM_MIN


def test_como_mucho_una_vez_por_segundo_y_seguimiento(db_path):
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    reid._aprendido_ts.clear()
    reid.identify_or_register(1, _cara(20))
    llamadas = reid._session.llamadas
    reid.identify_or_register(1, _cara(20))       # en el mismo segundo
    assert reid._session.llamadas == llamadas     # ni siquiera calcula
    assert _plantillas(reid, uid) == 2


def test_de_lado_no_aprende(db_path):
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    reid._aprendido_ts.clear()
    reid.identify_or_register(1, _cara(20), face_yaw=40.0)
    assert _plantillas(reid, uid) == 1


def test_la_llamada_sin_cara_no_gasta_el_turno(db_path):
    """`demographics.py` llama con un array vacío para actualizar el género."""
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    reid._aprendido_ts.clear()
    reid.identify_or_register(1, np.zeros((0,), dtype=np.uint8))
    assert 1 not in reid._aprendido_ts
    reid.identify_or_register(1, _cara(20))
    assert _plantillas(reid, uid) == 2


def test_no_pasa_del_maximo_de_plantillas(db_path, monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'REID_MAX_EMBEDDINGS_PER_PERSON', 2)
    reid = _reid(db_path)
    uid = reid.identify_or_register(1, _cara(10))[0]
    for _ in range(4):
        reid._aprendido_ts.clear()
        reid.identify_or_register(1, _cara(20))
    assert _plantillas(reid, uid) == 2


# ── 24 h y en disco ─────────────────────────────────────────────────────

def test_el_defecto_es_24_h_en_disco_con_15_plantillas():
    if 'ELDE_REID_TTL_S' in os.environ:
        pytest.skip('el entorno sobreescribe el TTL')
    assert AnalyticsConfig.REID_TTL_S == DIA
    assert AnalyticsConfig.REID_PERSISTIR_CON_TTL is True
    assert AnalyticsConfig.REID_MAX_EMBEDDINGS_PER_PERSON == 15


def test_con_ttl_y_disco_sobrevive_al_reinicio(db_path):
    uno = _reid(db_path, persistir=True)
    uid = uno.identify_or_register(1, _cara(10))[0]
    uno.force_save()

    dos = _reid(db_path, persistir=True)
    assert dos.total_unique_persons == 1
    assert dos.identify_or_register(2, _cara(10))[0] == uid


def test_lo_aprendido_tambien_se_guarda(db_path):
    uno = _reid(db_path, persistir=True)
    uid = uno.identify_or_register(1, _cara(10))[0]
    uno._aprendido_ts.clear()
    uno.identify_or_register(1, _cara(20))
    uno.force_save()

    dos = _reid(db_path, persistir=True)
    assert _plantillas(dos, uid) == 2


def test_tras_un_reinicio_la_caducada_sale_tambien_del_disco(db_path):
    uno = _reid(db_path, persistir=True)
    uid = uno.identify_or_register(1, _cara(10))[0]
    uno._db[uid]['last_seen'] -= DIA + 1
    uno.force_save()

    dos = _reid(db_path, persistir=True)
    assert dos.total_unique_persons == 0      # caduca al cargar
    dos.force_save()
    tres = _reid(db_path, persistir=True)
    assert uid not in tres._db


def test_tiendas_guarda_en_disco_con_ttl_y_sin_atexit():
    """Sobre el código, como `test_tiendas_cablea_el_ttl_y_deja_atexit_solo_
    con_disco`: instanciar TiendasProcess cargaría los modelos."""
    import inspect
    from src.analityc.core.tiendas import TiendasProcess
    fuente = inspect.getsource(TiendasProcess.__init__)
    assert "'REID_PERSISTIR_CON_TTL'" in fuente
    assert 'persistir=_persistir' in fuente
    rama_ttl = fuente.split('if _con_ttl:', 1)[1].split('else:', 1)[0]
    assert 'atexit' not in rama_ttl.replace('Sin `atexit`', '')


# ── Búsqueda de una sola vez ────────────────────────────────────────────

def _bucle_de_antes(reid, emb):
    mejor, uid_mejor = -1.0, None
    for uid, rec in reid._db.items():
        sim = fr._best_match_similarity(emb, rec)
        if sim > mejor:
            mejor, uid_mejor = sim, uid
    return uid_mejor, mejor


def test_la_busqueda_da_lo_mismo_que_el_bucle(db_path):
    reid = _reid(db_path)
    rng = np.random.default_rng(3)
    for i in range(120):
        reid._db[f'p{i}'] = {'embeddings': [
            _unitario(rng.standard_normal(512)) for _ in range(1 + i % 15)],
            'last_seen': 1e12}
    for _ in range(50):
        q = _unitario(rng.standard_normal(512))
        uid, sim = reid._best_in_gallery(q)
        uid2, sim2 = _bucle_de_antes(reid, q)
        assert uid == uid2
        assert sim == pytest.approx(sim2, abs=1e-5)
    # Y con una que sí está:
    objetivo = reid._db['p88']['embeddings'][3]          # p88 tiene 14
    assert reid._best_in_gallery(objetivo)[0] == 'p88'


def test_la_busqueda_con_galeria_vacia(db_path):
    assert _reid(db_path)._best_in_gallery(A) == (None, -1.0)


# ── Cadencias en imágenes reales ────────────────────────────────────────

def test_cada_cuenta_imagenes_reales_o_tiempo_real(monkeypatch):
    from src.analityc.core.tiendas import TiendasProcess
    falso = types.SimpleNamespace(_imagenes=lambda n: n * 3)
    monkeypatch.setattr(AnalyticsConfig, 'DEMO_CADENCIA_EN_IMAGENES', True,
                        raising=False)
    assert TiendasProcess._cada(falso, 5, 'DEMO_CADENCIA_EN_IMAGENES') == 5
    monkeypatch.setattr(AnalyticsConfig, 'DEMO_CADENCIA_EN_IMAGENES', False)
    assert TiendasProcess._cada(falso, 5, 'DEMO_CADENCIA_EN_IMAGENES') == 15


def test_clasificacion_y_huella_usan_la_cadencia_nueva():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    assert src.count("'DEMO_CADENCIA_EN_IMAGENES'") == 2
    assert "'BODY_REID_CADENCIA_EN_IMAGENES'" in src
    assert AnalyticsConfig.BODY_REID_EVERY_N == 6
    assert AnalyticsConfig.BODY_REID_MAX_EMB_PER_PERSON == 10
