"""
Pruebas de la identidad por el cuerpo y del género por persona (30-oct-2026,
`identidad_cuerpo.py` y `voto_identidad.py`).

En la cámara 4 la cara casi nunca sirve: sin esto, quien entraba y salía era
otra persona cada vez, y su visita cambiaba de género con cada entrada.
Huellas sintéticas y la galería del cuerpo real (`BodyReidentifier`) sin
modelo: `enroll` y `match` no lo necesitan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import identidad_cuerpo as ic  # noqa: E402
from src.analityc.core import voto_identidad as vi  # noqa: E402
from src.analityc.core.analytics.body_reidentifier import (  # noqa: E402
    BodyReidentifier)


def _u(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


_R = np.random.default_rng(11)
PERSONA = _u(_R.standard_normal(512))
OTRA = _u(_R.standard_normal(512))


def _vista(base, ruido=0.15):
    """Otra huella de la misma persona (coseno ~0,99 con `base`)."""
    return _u(base + ruido * _u(_R.standard_normal(512)))


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    monkeypatch.delenv('ELDE_BODY_ID', raising=False)
    ic._solo_cuerpo.clear()
    vi._personas.clear()


def _galeria():
    return BodyReidentifier(session=None, similarity_threshold=0.90)


# ── Crear la identidad ──────────────────────────────────────────────────

def test_con_tres_huellas_coherentes_crea_una_identidad():
    body, track = _galeria(), {}
    for _ in range(2):
        ic.guardar_huella(track, _vista(PERSONA))
        assert ic.quizas_crear(body, track) is None       # aún pocas
    ic.guardar_huella(track, _vista(PERSONA))
    pid = ic.quizas_crear(body, track)
    assert pid is not None and ic.es_solo_cuerpo(pid)
    # Y la próxima vez que entre, la galería la encuentra.
    assert body.match(_vista(PERSONA))[0] == pid


def test_si_el_numero_salto_de_persona_no_crea():
    body, track = _galeria(), {}
    for emb in (_vista(PERSONA), _vista(PERSONA), _vista(OTRA)):
        ic.guardar_huella(track, emb)
    assert ic.quizas_crear(body, track) is None
    assert body.stats()['personas_con_cuerpo'] == 0


def test_otra_persona_no_se_enlaza_con_ella():
    body, track = _galeria(), {}
    for _ in range(3):
        ic.guardar_huella(track, _vista(PERSONA))
    ic.quizas_crear(body, track)
    assert body.match(_vista(OTRA)) is None


def test_apagado_no_crea(monkeypatch):
    monkeypatch.setenv('ELDE_BODY_ID', '0')
    body, track = _galeria(), {}
    for _ in range(3):
        ic.guardar_huella(track, _vista(PERSONA))
    assert ic.quizas_crear(body, track) is None


def test_al_verle_la_cara_su_cuerpo_pasa_a_la_cara():
    body, track = _galeria(), {}
    for _ in range(3):
        ic.guardar_huella(track, _vista(PERSONA))
    pid = ic.quizas_crear(body, track)
    track['_body_match'] = pid
    ic.unir_con_cara(body, track, 'cara-123')
    assert body.match(_vista(PERSONA))[0] == 'cara-123'
    assert not ic.es_solo_cuerpo(pid)
    assert track['_body_match'] == 'cara-123'


def test_una_identidad_de_cara_no_se_renombra():
    body = _galeria()
    body.enroll('cara-a', PERSONA)
    track = {'_body_match': 'cara-a'}
    ic.unir_con_cara(body, track, 'cara-b')
    assert body.match(_vista(PERSONA))[0] == 'cara-a'


def test_nacida_fuera_del_roi_no_cuenta_como_visita(monkeypatch):
    """Quien vuelve de un cuarto sin ser reconocido recibe otra identidad
    fuera del ROI: existe, pero no es otra visita (`origen_identidad`)."""
    from src.analityc.core import origen_identidad as oi
    monkeypatch.delenv('ELDE_VISITAS_SOLO_ROI', raising=False)
    oi._origen.clear()
    oi.anotar('entra', True, ahora=1.0)
    oi.anotar('vuelve', False, ahora=2.0)
    assert oi.cuenta('entra') and not oi.cuenta('vuelve')
    assert oi.cuenta('sin-origen')                  # desconocido: cuenta
    oi.anotar('vuelve', True, ahora=3.0)            # nacer es una vez
    assert not oi.cuenta('vuelve')
    oi.heredar('entra', 'cara-de-entra')            # el cuerpo pasa a la cara
    assert oi.cuenta('cara-de-entra')
    monkeypatch.setenv('ELDE_VISITAS_SOLO_ROI', '0')
    assert oi.cuenta('vuelve')                      # apagado: todo cuenta


def test_tiendas_y_fotos_solo_cuentan_las_nacidas_en_el_roi():
    core = RAIZ / 'server' / 'src' / 'analityc' / 'core'
    t = (core / 'tiendas.py').read_text(encoding='utf-8')
    f = (core / 'tiendas_fotos.py').read_text(encoding='utf-8')
    assert "origen_identidad.anotar(\n                        matched, " \
           "self._dentro_del_roi(track), 'cuerpo')" in t
    assert t.count('if origen_identidad.cuenta(') == 2       # únicos
    assert 'if not origen_identidad.cuenta(pid):' in t       # entradas
    assert 'if not es_visita:\n                return' in f  # aviso
    assert 'if origen_identidad.cuenta(puid):' in f          # fila de visita


# ── Recuperar a quien acaba de salir ────────────────────────────────────

def _parecida(base, coseno):
    """Una huella con coseno EXACTO `coseno` respecto a `base`."""
    otra = _R.standard_normal(512)
    otra = _u(otra - (otra @ base) * base)                 # perpendicular
    return _u(coseno * base + np.sqrt(1 - coseno ** 2) * otra)


def test_quien_acaba_de_salir_se_recupera_con_menos_parecido():
    body = _galeria()
    body.enroll('p', PERSONA)                             # visto ahora
    ahora = body._gallery['p'][0][1]
    huella = _parecida(PERSONA, 0.75)
    # 0,75: pasa el 0,70 de «reciente»...
    assert ic.buscar(body, huella, ahora=ahora + 60)[0] == 'p'
    # ...pero no el 0,80 pasados 10 min.
    pid, sim = ic.buscar(body, huella, ahora=ahora + 601)
    assert pid is None and sim == pytest.approx(0.75, abs=1e-4)


def test_por_debajo_del_reciente_no_se_recupera():
    body = _galeria()
    body.enroll('p', PERSONA)
    ahora = body._gallery['p'][0][1]
    assert ic.buscar(body, _parecida(PERSONA, 0.65), ahora=ahora + 1)[0] is None


def test_una_persona_no_puede_estar_dos_veces_en_la_imagen():
    """12:22 del 30-oct: tres personas a la vez y el cuerpo dio a una la
    identidad de otra que seguía en la imagen (parecido 0,73)."""
    body = _galeria()
    body.enroll('p', PERSONA)
    ahora = body._gallery['p'][0][1]
    huella = _parecida(PERSONA, 0.75)
    assert ic.buscar(body, huella, ahora=ahora + 1)[0] == 'p'
    # La tiene otro seguimiento visible: no se puede dar.
    assert ic.buscar(body, huella, ahora=ahora + 1, ocupadas={'p'})[0] is None
    # Pero si hay otra identidad libre que se parece, esa sí.
    body.enroll('q', _parecida(huella, 0.95))
    assert ic.buscar(body, huella, ahora=ahora + 1, ocupadas={'p'})[0] == 'q'


def test_avisa_si_se_parece_a_una_ocupada_para_esperar():
    """13:35 del 30-oct: una mujer medio tapada se partió en dos
    seguimientos y el segundo recibió otra identidad (`4f2abe01`). Ahora
    `buscar` avisa y el seguimiento espera en vez de crear otra."""
    body = _galeria()
    body.enroll('p', PERSONA)
    ahora = body._gallery['p'][0][1]
    info = {}
    pid, _sim = ic.buscar(body, _parecida(PERSONA, 0.75), ahora=ahora + 1,
                          ocupadas={'p'}, info=info)
    assert pid is None and info['ocupada'] == 'p'
    # Si no se parece lo bastante, no hay a quién esperar: puede crear.
    info = {}
    ic.buscar(body, _parecida(PERSONA, 0.5), ahora=ahora + 1,
              ocupadas={'p'}, info=info)
    assert info['ocupada'] is None
    # Y si la identidad está libre, se enlaza directamente.
    info = {}
    assert ic.buscar(body, _parecida(PERSONA, 0.75), ahora=ahora + 1,
                     info=info)[0] == 'p'
    assert info['ocupada'] is None


def test_tiendas_espera_en_vez_de_crear():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    assert "track['_espera_ocupada'] = bool(" in src
    assert "and not track.get('_espera_ocupada')):" in src


def test_tiendas_pasa_las_identidades_de_los_demas():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    assert 'body, emb, ocupadas={' in src
    assert "if t2_id != tid and t2.get('persistent_id')}" in src


def test_galeria_vacia():
    assert ic.buscar(_galeria(), PERSONA) == (None, None)


def test_tiendas_crea_la_identidad_y_la_une_con_la_cara():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    assert 'identidad_cuerpo.quizas_crear(body, track)' in src
    assert 'identidad_cuerpo.unir_con_cara(body, track, uid)' in src
    assert 'identidad_cuerpo.guardar_huella(track, emb)' in src
    assert 'identidad_cuerpo.buscar(\n' in src
    assert "body.enroll(track['_body_match'], emb)" in src
    assert 'self._roi_activo = bool(roi_active)' in src


# ── Un solo género por persona ──────────────────────────────────────────

def test_la_visita_se_queda_con_la_mayoria():
    assert vi.votar('p1', 'f1', 'Hombre', ahora=1.0) == 'Hombre'
    assert vi.votar('p1', 'f2', 'Mujer', ahora=2.0) == 'Mujer'    # empate
    assert vi.votar('p1', 'f3', 'Hombre', ahora=3.0) == 'Hombre'
    assert vi.votar('p1', 'f4', 'Mujer', ahora=4.0) == 'Mujer'    # empate
    assert vi.votar('p1', 'f5', 'Hombre', ahora=5.0) == 'Hombre'
    assert vi.votos_de('p1') == {'Hombre': 3, 'Mujer': 2}
    # El último voto es «Mujer», pero manda la mayoría (antes ganaba él).
    vi.votar('p2', 'g1', 'Hombre', ahora=1.0)
    vi.votar('p2', 'g2', 'Hombre', ahora=2.0)
    assert vi.votar('p2', 'g3', 'Mujer', ahora=3.0) == 'Hombre'



def test_una_ficha_que_cambia_de_genero_cambia_su_voto():
    """genero_v2 corrige el género de una ficha ya votada."""
    vi.votar('p1', 'f1', 'Hombre', ahora=1.0)
    vi.votar('p1', 'f2', 'Hombre', ahora=2.0)
    vi.votar('p1', 'f2', 'Mujer', ahora=3.0)
    assert vi.votos_de('p1') == {'Hombre': 1, 'Mujer': 1}


def test_sin_identidad_o_sin_genero_no_vota():
    assert vi.votar(None, 'f1', 'Mujer') == 'Mujer'
    assert vi.votar('p1', 'f1', 'Desconocido') == 'Desconocido'
    assert vi.votos_de('p1') == {}


def test_caduca_como_la_galeria():
    vi.votar('p1', 'f1', 'Mujer', ahora=0.0)
    vi.votar('p2', 'f9', 'Hombre', ahora=86400.0 + 10)
    assert vi.genero_de('p1') is None


def test_la_visita_y_el_aviso_usan_la_mayoria():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas_fotos.py').read_text(encoding='utf-8')
    assert 'voto_identidad.votar(_pid_demo, stem, genero)' in src
    assert "'gender': genero_persona" in src
    assert 'voto_identidad.votar(puid, stem, genero)' in src
