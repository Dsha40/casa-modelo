"""Revisión del género tras un cruce, en sombra (`revision_cruces.py`, 22-oct-2026)."""
import json

import numpy as np

from src.analityc.core import revision_cruces as rc
from src.analityc.core.analytics.config import AnalyticsConfig
from src.analityc.core.analytics.estimador_edad_genero import MuestraDemografica


class _Acc:
    def __init__(self, genero):
        self.committed = True
        self.result = {'gender': genero}


class _Estimador:
    def __init__(self, genero, conf=0.97):
        self.genero, self.conf, self.llamadas = genero, conf, 0

    def estimar(self, track_id, cuerpo, rostro=None):
        self.llamadas += 1
        return MuestraDemografica(track_id=-1, rango_edad='36-50', genero=self.genero,
                                  conf_genero=self.conf, conf_edad=0.55,
                                  solo_cuerpo=rostro is None, edad_anios=40.0)


class _Clf:
    def __init__(self, ve):
        self._accum = {1: _Acc('Mujer')}
        self._estimador_cuerpo = _Estimador(ve)
        self._yunet = None
        self._cfg = AnalyticsConfig


_FRAME = np.zeros((600, 900, 3), np.uint8)
_A = [100, 100, 200, 400]
_JUNTO = [150, 100, 250, 400]         # solapa la mitad de A
_LEJOS = [600, 100, 700, 400]


def _pasar(r, clf, pasos, t0=0.0, dt=0.1):
    """t0: 1 y 2 cruzados; después, separados."""
    r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _JUNTO}}, clf, t=t0)
    for i in range(1, pasos + 1):
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _LEJOS}}, clf, t=t0 + i * dt)


def _lineas(carpeta):
    fs = list(carpeta.glob('*.jsonl'))
    return [json.loads(x) for f in fs for x in f.read_text(encoding='utf-8').splitlines()]


def test_tras_el_cruce_apunta_que_cambiaria(tmp_path):
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    _pasar(r, clf, 60)
    filas = _lineas(tmp_path)
    assert len(filas) == 1
    f = filas[0]
    assert (f['tid'], f['con'], f['etiqueta'], f['resultado'], f['nueva']) == \
        ('1', '2', 'Mujer', 'cambiaria', 'Hombre')
    assert f['foto'] and (tmp_path / f['foto']).is_file()
    # En sombra: la etiqueta del seguimiento no se toca.
    assert clf._accum[1].result == {'gender': 'Mujer'}


def test_si_sigue_siendo_la_misma_confirma(tmp_path):
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Mujer')
    _pasar(r, clf, 60)
    assert [f['resultado'] for f in _lineas(tmp_path)] == ['confirma']


def test_no_lee_mientras_se_solapan_ni_antes_de_separarse(tmp_path):
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    for i in range(30):                       # 3 s cruzados
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _JUNTO}}, clf, t=i * 0.1)
    assert clf._estimador_cuerpo.llamadas == 0
    for i in range(1, 10):                    # 0,9 s separados (< SEPARADO_S)
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _LEJOS}}, clf, t=2.9 + i * 0.1)
    assert clf._estimador_cuerpo.llamadas == 0


def test_sin_cruce_no_revisa(tmp_path):
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    for i in range(60):
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _LEJOS}}, clf, t=i * 0.1)
    assert clf._estimador_cuerpo.llamadas == 0 and _lineas(tmp_path) == []


def test_sin_genero_decidido_o_personal_no_revisa(tmp_path):
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    clf._accum[1].committed = False
    _pasar(r, clf, 60)
    clf._accum[1].committed = True
    tracks_cruce = {1: {'box': _A, 'es_personal': True}, 2: {'box': _JUNTO}}
    r.observar(_FRAME, tracks_cruce, clf, t=10.0)
    for i in range(1, 60):
        r.observar(_FRAME, {1: {'box': _A, 'es_personal': True}, 2: {'box': _LEJOS}},
                   clf, t=10.0 + i * 0.1)
    assert clf._estimador_cuerpo.llamadas == 0 and _lineas(tmp_path) == []


def test_lecturas_dudosas_acaban_sin_decidir(tmp_path):
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    clf._estimador_cuerpo.conf = 0.55         # nunca llega al umbral
    clf._cfg = type('C', (), {k: getattr(AnalyticsConfig, k) for k in dir(AnalyticsConfig)
                              if not k.startswith('__')})
    clf._cfg.DEMO_BODY_MIN_GENDER_CONF = 0.0
    _pasar(r, clf, 150)
    filas = _lineas(tmp_path)
    assert [f['resultado'] for f in filas] == ['sin_decidir']
    assert len(filas[0]['lecturas']) <= rc.MAX_LECTURAS


_ROZA = [195, 100, 295, 400]          # toca a A (solape ~5 %), no llega a cruce


def test_no_mira_mientras_el_vecino_aun_roza_la_caja(tmp_path):
    """22-oct, 14:43:43: el vecino seguía pegado y se leyó su cara."""
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _JUNTO}}, clf, t=0.0)
    for i in range(1, 40):                    # 4 s rozándose
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _ROZA}}, clf, t=i * 0.1)
    assert clf._estimador_cuerpo.llamadas == 0
    for i in range(1, 60):                    # se aleja: ahora sí
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _LEJOS}}, clf, t=3.9 + i * 0.1)
    assert [f['resultado'] for f in _lineas(tmp_path)] == ['cambiaria']


def test_cambiar_pide_3_lecturas_y_confirmar_no(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, '_cara', lambda clf, frame, box: np.zeros((40, 40, 3), np.uint8))
    for ve, esperado, lecturas in (('Hombre', 'cambiaria', 3), ('Mujer', 'confirma', 2)):
        carpeta = tmp_path / ve
        r, clf = rc.RevisionCruces('cam', carpeta=carpeta), _Clf(ve)
        clf._estimador_cuerpo.conf = 1.0      # con cara, 2 lecturas ya pasan el umbral de B
        _pasar(r, clf, 60)
        f = _lineas(carpeta)[0]
        assert (f['resultado'], len(f['lecturas'])) == (esperado, lecturas)


def test_guarda_foto_de_antes_del_cruce_aunque_no_haya_ficha(tmp_path):
    """23-oct: 2 de 11 casos sin ficha propia no tenían con qué comparar."""
    import cv2
    r, clf = rc.RevisionCruces('cam', carpeta=tmp_path), _Clf('Hombre')
    antes = _FRAME.copy()
    antes[100:400, 100:200] = 255                       # la persona de antes: blanca
    for i in range(10):                                  # 1 s sola, nadie la toca
        r.observar(antes, {1: {'box': _A}, 2: {'box': _LEJOS}}, clf, t=i * 0.1)
    r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _JUNTO}}, clf, t=1.0)
    for i in range(1, 60):
        r.observar(_FRAME, {1: {'box': _A}, 2: {'box': _LEJOS}}, clf, t=1.0 + i * 0.1)
    f = _lineas(tmp_path)[0]
    assert f['con_antes'] is True and f['ficha'] is None
    hoja = cv2.imread(str(tmp_path / f['foto']))
    # «antes» (blanca) a la izquierda, «ahora» (negra) a la derecha.
    assert hoja[150, 50].mean() > 200 and hoja[150, -30].mean() < 50


def test_apagada_no_hace_nada(monkeypatch):
    monkeypatch.setenv('ELDE_REVISION_CRUCES', '0')
    clf = _Clf('Hombre')
    rc._POR_CAMARA.clear()
    rc.observar('cam', _FRAME, {1: {'box': _A}, 2: {'box': _JUNTO}}, clf, t=0.0)
    assert rc._POR_CAMARA == {}
