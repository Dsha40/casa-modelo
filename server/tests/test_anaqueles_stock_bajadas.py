"""Foto de referencia por producto, diaria y tras cada bajada; aviso «cerca
de vaciarse» (9-oct-2026, `anaqueles_stock.py`). Lo pidió el usuario:

  - poder pedir la foto de UN producto, no solo de todos;
  - que a las 7:30 se rehagan solas (una vez al día, aunque se reinicie);
  - que tras un agarre con bajada confirmada se cuente, se rehaga la foto de
    ese producto, y cada 8 bajadas avise de que está cerca de vaciarse;
  - la foto de balda surtida (manual o diaria) pone la cuenta a cero.

Imágenes sintéticas como en `test_anaqueles_stock.py`: un tablero hace de
balda surtida y un tramo gris de «falta algo». Los umbrales no están
calibrados: se fija el mecanismo, no la cifra.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import anaqueles_stock as st  # noqa: E402
from src.analityc.core.anaqueles import _DEFECTOS  # noqa: E402

CAM = 'cam-bajadas-test'
T0 = 1_000_000.0
POLI = np.array([[100, 100], [200, 100], [200, 180], [100, 180]], dtype=np.float32)
POLI2 = POLI + np.float32([250, 0])
PRODUCTOS = {0: [(0, 'Colcafe', POLI)]}
DOS = {0: [(0, 'Colcafe', POLI), (1, 'Juan Valdez', POLI2)]}
NOMBRES = {0: 'Anaquel 2'}
LEJOS = (520, 300, 620, 470)


def _frame(gris_ancho: int = 0) -> np.ndarray:
    """Balda surtida (tablero) con los primeros `gris_ancho` px «vacíos»."""
    f = np.full((480, 640, 3), 90, dtype=np.uint8)
    for x0 in (100, 350):
        yy, xx = np.mgrid[100:181, x0:x0 + 101]
        f[100:181, x0:x0 + 101] = ((((yy // 10) + (xx // 10)) % 2) * 150 + 40)[..., None]
    if gris_ancho:
        f[100:181, 100:100 + gris_ancho] = 120
    return f


SURTIDO, FALTA_ALGO = _frame(), _frame(25)


@pytest.fixture(autouse=True)
def limpiar_puente():
    st._peticiones.clear()
    st._estados.clear()
    yield
    st._peticiones.clear()
    st._estados.clear()


#: Umbrales PROPIOS de estas pruebas (las imágenes son sintéticas): así no
#: dependen de los de producción, que el usuario va ajustando en vivo.
UMBRAL_BAJADA, UMBRAL_AGOTADO = 0.25, 0.45


def _vig(tmp_path, **cfg):
    cfg.setdefault('stock_reinicio_diario', '')
    cfg.setdefault('stock_umbral_bajada', UMBRAL_BAJADA)
    cfg.setdefault('stock_umbral', UMBRAL_AGOTADO)
    return st.VigilanteStock(CAM, tmp_path, dict(_DEFECTOS, **cfg))


def _pasar(v, frame, desde, hasta, productos=PRODUCTOS):
    t = desde
    while t <= hasta:
        v.vigilar(frame, productos, NOMBRES, [LEJOS], t)
        t += 1.0


def _con_ref(tmp_path, **cfg):
    v = _vig(tmp_path, **cfg)
    st.pedir_referencia(CAM)
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0)
    assert v.bloque()['referencias'] == 1
    return v


def test_el_cambio_de_falta_algo_queda_entre_los_dos_umbrales():
    caja = st._caja(POLI, (480, 640))
    x1, y1, x2, y2 = caja
    d = st.diferencia(SURTIDO[y1:y2, x1:x2], FALTA_ALGO[y1:y2, x1:x2], st._mascara(POLI, caja))
    assert UMBRAL_BAJADA <= d < UMBRAL_AGOTADO
    # Y en producción la bajada pide menos cambio que el agotado.
    assert _DEFECTOS['stock_umbral_bajada'] < _DEFECTOS['stock_umbral']


# ── Por producto ────────────────────────────────────────────────────────
def test_se_puede_pedir_la_foto_de_un_solo_producto(tmp_path):
    v = _vig(tmp_path)
    v.vigilar(SURTIDO, DOS, NOMBRES, [LEJOS], T0)          # cámara viva
    assert st.pedir_referencia(CAM, 0, 1)
    v.vigilar(SURTIDO, DOS, NOMBRES, [LEJOS], T0 + 1)
    assert [r['nombre'] for r in v._refs.values()] == ['Juan Valdez']


# ── Diaria ──────────────────────────────────────────────────────────────
def _hora(hhmm: str) -> float:
    return time.mktime(time.strptime(f'2026-10-09 {hhmm}', '%Y-%m-%d %H:%M'))


def test_la_foto_diaria_sale_una_vez_desde_la_hora_y_sobrevive_al_reinicio(tmp_path):
    v = _vig(tmp_path, stock_reinicio_diario='07:30')
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], _hora('07:00'))
    assert v.bloque()['referencias'] == 0
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], _hora('07:31'))
    assert v.bloque()['referencias'] == 1
    assert v._toca_diaria(_hora('09:00')) is False          # ya hecha hoy
    otro = _vig(tmp_path, stock_reinicio_diario='07:30')    # reinicio
    assert otro._toca_diaria(_hora('10:00')) is False


# ── Bajadas ─────────────────────────────────────────────────────────────
def test_una_bajada_se_cuenta_y_rehace_la_foto(tmp_path):
    v = _con_ref(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, FALTA_ALGO, T0 + 10, T0 + 20)
    b = v.bloque()
    assert b['bajadas'] == [{'area': 0, 'producto_idx': 0, 'bajadas': 1}]
    assert b['alertas'] == [] and b['cerca'] == []           # no es agotado
    assert 'bajada' in [s['tipo'] for s in v.drenar_sucesos()]
    # La foto nueva es la de ahora (con el hueco): la siguiente se mide aquí.
    x1, y1, x2, y2 = v._refs[(0, 0)]['caja']
    assert np.array_equal(v._refs[(0, 0)]['img'], FALTA_ALGO[y1:y2, x1:x2])
    # Y la cuenta sobrevive al reinicio.
    otro = _vig(tmp_path)
    otro.vigilar(FALTA_ALGO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 30)
    assert otro.bloque()['bajadas'][0]['bajadas'] == 1


def test_sin_cambio_no_hay_bajada(tmp_path):
    v = _con_ref(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, SURTIDO, T0 + 10, T0 + 20)
    assert v.bloque()['bajadas'] == []


def test_cada_n_bajadas_avisa_cerca_de_vaciarse_y_la_foto_surtida_lo_cierra(tmp_path):
    v = _con_ref(tmp_path, stock_bajadas_aviso=2)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, FALTA_ALGO, T0 + 10, T0 + 20)                 # bajada 1
    v.tras_agarre(0, T0 + 30)
    _pasar(v, SURTIDO, T0 + 30, T0 + 40)                    # bajada 2 -> aviso
    cerca = v.bloque()['cerca']
    assert len(cerca) == 1 and cerca[0]['bajadas'] == 2 and cerca[0]['producto'] == 'Colcafe'
    salida = v.drenar_alertas()
    assert [a['tipo'] for a in salida] == ['cerca'] and salida[0]['bajadas'] == 2
    assert list((tmp_path / 'stock_alertas').glob('*_0_0_cerca.jpg'))
    # Foto de balda surtida: cuenta a cero y aviso cerrado.
    st.pedir_referencia(CAM, 0, 0)
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 50)
    assert v.bloque()['cerca'] == [] and v.bloque()['bajadas'] == []


def test_cada_bajada_guarda_su_foto_antes_y_despues(tmp_path):
    """12-oct-2026, lo pidió el usuario: con el posible agotado a 0,90 casi
    no salta ninguna alerta; cada bajada deja su foto para revisarla en el
    dashboard, con el cambio y la cuenta al lado. No es una alerta: el panel
    del cliente no recibe nada."""
    import json
    v = _con_ref(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, FALTA_ALGO, T0 + 10, T0 + 20)
    fotos = list((tmp_path / 'stock_alertas').glob('*_0_0_bajada.jpg'))
    assert len(fotos) == 1 and fotos[0].stat().st_size > 1000
    meta = json.loads(fotos[0].with_suffix('.json').read_text(encoding='utf-8'))
    assert meta['tipo'] == 'bajada' and meta['bajadas'] == 1
    assert meta['producto'] == 'Colcafe'
    assert UMBRAL_BAJADA <= meta['cambio'] < UMBRAL_AGOTADO
    assert v.drenar_alertas() == []

