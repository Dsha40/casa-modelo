"""El personal surte (19-oct-2026, `VigilanteStock.tras_personal`). Lo pidió
el usuario: cuando el personal toca un área y, al apartarse, un producto
cambia mucho (>= `stock_umbral_surtido`, 0,80 en producción) frente a su
foto, se da por SURTIDO: suceso, foto de evidencia y foto de balda surtida
nueva (cuenta de bajadas a cero). Si solo acomodó, nada.

Imágenes sintéticas de `test_anaqueles_stock_bajadas.py`; umbral propio.
"""
from __future__ import annotations

import numpy as np

from test_anaqueles_stock_bajadas import (  # noqa: F401  (fixture autouse)
    LEJOS, NOMBRES, PRODUCTOS, SURTIDO, FALTA_ALGO, T0, _frame, _pasar, _vig,
    limpiar_puente, st)

LLENA = _frame(100)                 # el área entera cambia: balda rehecha


def _con_ref(tmp_path):
    v = _vig(tmp_path, stock_umbral_surtido=0.45)
    st.pedir_referencia('cam-bajadas-test')
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0)
    return v


def test_el_cambio_de_la_balda_rehecha_pasa_el_umbral():
    caja = st._caja(PRODUCTOS[0][0][2], (480, 640))
    x1, y1, x2, y2 = caja
    d = st.diferencia(SURTIDO[y1:y2, x1:x2], LLENA[y1:y2, x1:x2],
                      st._mascara(PRODUCTOS[0][0][2], caja))
    assert d >= 0.45


def test_surte_si_cambia_mucho_tras_tocar_el_personal(tmp_path):
    v = _con_ref(tmp_path)
    v._bajadas[(0, 0)] = 5
    v.tras_personal(0, T0 + 10)
    _pasar(v, LLENA, T0 + 10, T0 + 45)
    assert 'surtido' in [s['tipo'] for s in v.drenar_sucesos()]
    assert list((tmp_path / 'stock_alertas').glob('*_surtido.jpg'))
    x1, y1, x2, y2 = v._refs[(0, 0)]['caja']
    assert np.array_equal(v._refs[(0, 0)]['img'], LLENA[y1:y2, x1:x2])  # foto nueva
    assert v._bajadas[(0, 0)] == 0                                       # cuenta a cero
    assert not v._surtiendo


def test_si_solo_acomoda_no_hay_surtido(tmp_path):
    v = _con_ref(tmp_path)
    v.tras_personal(0, T0 + 10)
    _pasar(v, SURTIDO, T0 + 10, T0 + 45)
    assert 'surtido' not in [s['tipo'] for s in v.drenar_sucesos()]
    assert not v._surtiendo                                  # se acabó de mirar
    assert not list(tmp_path.glob('stock_alertas/*_surtido.jpg'))


def test_el_personal_anula_la_vigilancia_del_cliente_en_esa_area(tmp_path):
    v = _con_ref(tmp_path)
    v.tras_agarre(0, T0 + 5)
    assert v.vigilando(0)
    v.tras_personal(0, T0 + 6)
    assert not v.vigilando(0)
    _pasar(v, FALTA_ALGO, T0 + 6, T0 + 16)                   # ni bajada ni surtido
    assert not [s for s in v.drenar_sucesos() if s['tipo'] in ('bajada', 'surtido')]


def test_un_gesto_con_el_personal_surtiendo_no_vigila_bajada(tmp_path):
    """19-oct-2026, 16:36: la reposición en curso no se mide como bajada."""
    v = _con_ref(tmp_path)
    v.tras_personal(0, T0 + 5)
    v.tras_agarre(0, T0 + 6)
    assert not v.vigilando(0)
    _pasar(v, FALTA_ALGO, T0 + 6, T0 + 16)
    assert 'bajada' not in [s['tipo'] for s in v.drenar_sucesos()]


def test_tras_un_acomodo_se_rehace_la_foto_sin_tocar_la_cuenta(tmp_path):
    """19-oct-2026, 17:20: sin esto la foto se quedaba de antes del personal
    y el siguiente agarre de un cliente daba bajadas falsas."""
    v = _con_ref(tmp_path)
    v._bajadas[(0, 0)] = 2
    v.tras_personal(0, T0 + 10)
    _pasar(v, FALTA_ALGO, T0 + 10, T0 + 45)                  # cambia, pero < umbral de surtido
    assert not v._surtiendo
    x1, y1, x2, y2 = v._refs[(0, 0)]['caja']
    assert np.array_equal(v._refs[(0, 0)]['img'], FALTA_ALGO[y1:y2, x1:x2])  # foto nueva
    assert v._bajadas[(0, 0)] == 2
    assert 'surtido' not in [s['tipo'] for s in v.drenar_sucesos()]


def test_prueba_foto_del_agarre_sin_confirmar_aparte(tmp_path):
    """22-oct-2026 (prueba pedida por el usuario): el agarre que no confirma
    deja su foto en `stock_pruebas/`, NO en `stock_alertas/`."""
    v = _con_ref(tmp_path)
    v.cfg['stock_pruebas_sin_confirmar'] = True
    v.tras_agarre(0, T0 + 5)
    _pasar(v, SURTIDO, T0 + 5, T0 + 15)                      # no cambia nada
    assert not v.vigilando(0)
    assert list((tmp_path / 'stock_pruebas').glob('*_sin_confirmar.jpg'))
    assert not list(tmp_path.glob('stock_alertas/*'))


def test_prueba_foto_del_acomodo_que_no_llega_a_surtido(tmp_path):
    """22-oct-2026 (prueba): el personal toca y el cambio no llega al surtido:
    queda su antes/después en `stock_pruebas/`, con el cambio máximo."""
    v = _con_ref(tmp_path)
    v.cfg['stock_pruebas_sin_confirmar'] = True
    v.tras_personal(0, T0 + 10)
    _pasar(v, FALTA_ALGO, T0 + 10, T0 + 45)
    fotos = list((tmp_path / 'stock_pruebas').glob('*_acomodo.jpg'))
    assert len(fotos) == 1
    import json as _json
    assert _json.loads(fotos[0].with_suffix('.json').read_text(encoding='utf-8'))['cambio'] > 0
    assert not list(tmp_path.glob('stock_alertas/*_acomodo*'))


def test_la_reposicion_a_ratos_no_se_trocea(tmp_path):
    """22-oct-2026: con el personal reponiendo a ratos, no se rehace la foto
    a mitad: el área sigue vigilada hasta 30 s sin tocarla."""
    v = _con_ref(tmp_path)
    v.tras_personal(0, T0 + 10)
    _pasar(v, SURTIDO, T0 + 10, T0 + 25)                     # pausa de 15 s
    assert v._surtiendo                                      # no se acabó
    v.tras_personal(0, T0 + 26)                              # vuelve a tocar
    _pasar(v, LLENA, T0 + 26, T0 + 40)
    assert 'surtido' in [s['tipo'] for s in v.drenar_sucesos()]
