"""
«Posible agotado» tras un agarre (5-oct-2026, `anaqueles_stock.py`).

El operador pide una foto «surtido» de cada producto; tras un agarre en el
área, el servidor compara el recuadro con esa foto cuando nadie lo tapa y
avisa si cambió mucho, varias veces seguidas. Lo que más se protege:

  - **un cambio de luz no es un agotado** (la comparación normaliza el
    brillo), y un recuadro que pasa de tener dibujo a liso sí lo es;
  - **tapado no se mide ni se fotografía**: el cuerpo delante no puede
    pasar por «balda vacía» ni quedar como referencia;
  - **sin agarre no se vigila**, y una sola medida alta no basta;
  - **la alerta se cierra sola** al reponer, con histéresis;
  - **un producto redibujado pierde su referencia** (era de otro recuadro);
  - **la referencia sobrevive al reinicio** (disco), y la API y el bloque
    en vivo dicen lo mismo.

Las imágenes son sintéticas: un tablero de ajedrez hace de «balda con
productos» (mucho dibujo) y un gris liso de «balda vacía». El UMBRAL real no
está calibrado (no hay una balda vaciada de verdad): estas pruebas fijan el
mecanismo, no la cifra.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

from src.analityc.core import anaqueles_stock as st  # noqa: E402
from src.analityc.core.anaqueles import _DEFECTOS  # noqa: E402

CAM = 'cam-stock-test'
T0 = 1_000_000.0

#: Un producto: x 100..200, y 100..180 dentro de un frame de 640x480.
POLI = np.array([[100, 100], [200, 100], [200, 180], [100, 180]],
                dtype=np.float32)
PRODUCTOS = {0: [(0, 'Colcafé', POLI)]}
NOMBRES = {0: 'Anaquel 2'}
#: Un cuerpo plantado delante del producto, y otro lejos.
DELANTE = (120, 60, 220, 470)
LEJOS = (450, 60, 550, 470)


#: Suelo a rayas, abajo y lejos del producto: una imagen real nunca es lisa,
#: y una lisa ENTERA es la de un corte de vídeo, que el vigilante no mide
#: (13-oct-2026, `cortes_video.sin_video`).
SUELO = (((np.arange(640) // 16) % 2) * 120 + 40).astype(np.uint8)[None, :, None]


def _frame(relleno: str, brillo: float = 1.0) -> np.ndarray:
    """Frame con el recuadro del producto «surtido» (ajedrez) o «vacío»."""
    f = np.full((480, 640, 3), 90, dtype=np.uint8)
    f[300:460] = SUELO
    if relleno == 'surtido':
        yy, xx = np.mgrid[100:181, 100:201]
        tablero = (((yy // 10) + (xx // 10)) % 2) * 150 + 40
        f[100:181, 100:201] = tablero[..., None].astype(np.uint8)
    else:
        f[100:181, 100:201] = 120
    return np.clip(f.astype(np.float32) * brillo, 0, 255).astype(np.uint8)


SURTIDO = _frame('surtido')
VACIO = _frame('vacio')


@pytest.fixture(autouse=True)
def limpiar_puente():
    """El puente con la API es de módulo: cada prueba empieza de cero."""
    st._peticiones.clear()
    st._estados.clear()
    yield
    st._peticiones.clear()
    st._estados.clear()


def _vigilante(tmp_path, **cfg):
    return st.VigilanteStock(CAM, tmp_path, dict(_DEFECTOS, **cfg))


def _con_referencia(tmp_path, **cfg):
    v = _vigilante(tmp_path, **cfg)
    assert st.pedir_referencia(CAM)
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0)
    assert v.bloque()['referencias'] == 1
    return v


def _pasar(v, frame, desde, hasta, paso=1.0, cuerpos=(LEJOS,)):
    t = desde
    while t <= hasta:
        v.vigilar(frame, PRODUCTOS, NOMBRES, list(cuerpos), t)
        t += paso


# ── La comparación ──────────────────────────────────────────────────────
def _mascara():
    caja = st._caja(POLI, (480, 640))
    return caja, st._mascara(POLI, caja)


def _recorte(frame):
    (x1, y1, x2, y2), _m = _mascara()
    return frame[y1:y2, x1:x2]


def test_identicos_dan_cero_y_vaciar_da_mucho():
    _caja, m = _mascara()
    assert st.diferencia(_recorte(SURTIDO), _recorte(SURTIDO), m) < 0.01
    # Cifra fija y no `stock_umbral` (que el usuario ajusta en vivo: 0,80
    # desde el 9-oct): la balda vaciada sintética da ~0,9.
    assert st.diferencia(_recorte(SURTIDO), _recorte(VACIO), m) > 0.8


def test_un_cambio_de_luz_no_cuenta_como_cambio():
    """La tarde oscurece todo un 40 %: el dibujo sigue ahí."""
    _caja, m = _mascara()
    oscuro = _frame('surtido', brillo=0.6)
    assert st.diferencia(_recorte(SURTIDO), _recorte(oscuro), m) < \
        _DEFECTOS['stock_umbral'] * 0.3


def test_tamanos_distintos_no_se_comparan():
    _caja, m = _mascara()
    assert st.diferencia(_recorte(SURTIDO), _recorte(SURTIDO)[:-1], m) \
        is None


# ── La foto de referencia ───────────────────────────────────────────────
def test_sin_camara_viva_la_peticion_se_rechaza():
    assert st.pedir_referencia('nadie') is False


def test_tapado_no_se_fotografia_y_se_toma_al_apartarse(tmp_path):
    v = _vigilante(tmp_path)
    st.pedir_referencia(CAM)
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [DELANTE], T0)
    assert v.bloque()['referencias'] == 0
    assert v.bloque()['esperando_foto'] == 1
    # A 1 s aún «tapa» (memoria de 1,5 s, 15-oct-2026); a los 2 s, no.
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 1)
    assert v.bloque()['referencias'] == 0
    v.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 2)
    assert v.bloque()['referencias'] == 1
    assert v.bloque()['esperando_foto'] == 0


def test_la_peticion_caduca_si_siguen_tapando(tmp_path):
    v = _vigilante(tmp_path, stock_peticion_max_s=10.0)
    st.pedir_referencia(CAM)
    _pasar(v, SURTIDO, T0, T0 + 11, cuerpos=(DELANTE,))
    assert v.bloque()['esperando_foto'] == 0
    assert v.bloque()['referencias'] == 0


def test_la_referencia_sobrevive_al_reinicio(tmp_path):
    _con_referencia(tmp_path)
    otro = _vigilante(tmp_path)          # servidor reiniciado
    otro.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 5)
    assert otro.bloque()['referencias'] == 1


def test_redibujar_el_producto_tira_la_referencia(tmp_path):
    v = _con_referencia(tmp_path)
    movido = {0: [(0, 'Colcafé', POLI + np.float32(20))]}
    v.vigilar(SURTIDO, movido, NOMBRES, [LEJOS], T0 + 1)
    assert v.bloque()['referencias'] == 0
    # Y tampoco revive desde el disco: era de otro recuadro.
    otro = _vigilante(tmp_path)
    otro.vigilar(SURTIDO, movido, NOMBRES, [LEJOS], T0 + 2)
    assert otro.bloque()['referencias'] == 0


# ── La alerta ───────────────────────────────────────────────────────────
def test_sin_agarre_no_se_vigila(tmp_path):
    v = _con_referencia(tmp_path)
    _pasar(v, VACIO, T0 + 1, T0 + 30)
    assert v.bloque()['alertas'] == []


def test_tras_un_agarre_la_balda_vacia_avisa(tmp_path):
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, VACIO, T0 + 10, T0 + 20)
    alertas = v.bloque()['alertas']
    assert len(alertas) == 1
    assert alertas[0]['producto'] == 'Colcafé'
    assert alertas[0]['area_nombre'] == 'Anaquel 2'
    assert alertas[0]['cambio'] >= _DEFECTOS['stock_umbral']
    sucesos = v.drenar_sucesos()
    assert [s['tipo'] for s in sucesos] == ['agotado']
    assert sucesos[0]['producto'] == 'Colcafé'


def test_tras_un_agarre_sin_cambio_no_avisa_y_deja_de_vigilar(tmp_path):
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, SURTIDO, T0 + 10, T0 + 20)
    assert v.bloque()['alertas'] == []
    assert v.bloque()['vigilando'] == 0


def test_una_sola_medida_alta_no_basta(tmp_path):
    """Un reflejo de un frame: la racha se rompe y no hay alerta."""
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    v.vigilar(VACIO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 13)
    _pasar(v, SURTIDO, T0 + 14, T0 + 20)
    assert v.bloque()['alertas'] == []
    # Ni siquiera un aviso que luego se cerrara solo.
    assert v.drenar_sucesos() == []


def test_tapado_no_se_mide(tmp_path):
    """El cuerpo delante tras el agarre no puede pasar por balda vacía."""
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, VACIO, T0 + 10, T0 + 40, cuerpos=(DELANTE,))
    assert v.bloque()['alertas'] == []
    assert v.bloque()['vigilando'] == 1


def test_la_vigilancia_caduca_si_nunca_se_ve(tmp_path):
    v = _con_referencia(tmp_path, stock_ventana_s=20.0)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, VACIO, T0 + 10, T0 + 31, cuerpos=(DELANTE,))
    assert v.bloque()['vigilando'] == 0


def test_al_reponer_la_alerta_se_cierra_sola(tmp_path):
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, VACIO, T0 + 10, T0 + 20)
    assert v.bloque()['alertas']
    v.drenar_sucesos()
    _pasar(v, SURTIDO, T0 + 30, T0 + 60, paso=5.0)
    assert v.bloque()['alertas'] == []
    sucesos = v.drenar_sucesos()
    assert [s['tipo'] for s in sucesos] == ['repuesto']
    # Detectado a T0+14, repuesto a T0+40 (tres medidas, una cada 5 s).
    assert sucesos[0]['duracion_s'] == 26


# ── Cuánto lleva (5-oct, lo pidió el usuario) ───────────────────────────
def test_el_bloque_dice_cuanto_lleva(tmp_path):
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, VACIO, T0 + 10, T0 + 20)
    # Espera de 2 s y tres medidas de 1 s: detectado a T0+14.
    assert v.bloque()['alertas'][0]['desde'] == st._hora(T0 + 14)
    v.vigilar(VACIO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 74)
    assert v.bloque()['alertas'][0]['vacio_s'] == 60


def test_la_duracion_se_lee_bien():
    assert st._duracion(45) == '45 s'
    assert st._duracion(125) == '2 min 05 s'
    assert st._duracion(3780) == '1 h 03 min'


def test_la_api_ve_lo_mismo_que_el_bloque(tmp_path):
    from src.app import anaqueles_api
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)
    _pasar(v, VACIO, T0 + 10, T0 + 20)
    r = anaqueles_api.stock(CAM)
    assert r['alertas'] == v.bloque()['alertas']
    assert r['productos_con_referencia'][0]['producto'] == 'Colcafé'
    assert anaqueles_api.stock_referencia(CAM)['status'] == 'ok'
    with pytest.raises(Exception):
        anaqueles_api.stock_referencia('nadie')


# ── Cableado en el medidor ──────────────────────────────────────────────
def test_el_agarre_del_medidor_arranca_la_vigilancia(tmp_path, monkeypatch):
    """De punta a punta: foto pedida, gesto de agarre (el de
    `test_anaqueles_productos`), y el bloque en vivo lleva `stock`."""
    import test_anaqueles_productos as tp
    medio = time.mktime((2026, 10, 5, 12, 0, 0, 0, 0, -1))
    monkeypatch.setattr(time, 'time', lambda: medio)
    m = tp._medidor(tmp_path, [tp.FUERA] * 4 + [[(250, 150, 0.9)]] * 4,
                    [tp._prod(0, 'Colcafé', tp.DER)])
    assert st.pedir_referencia(m.camera_id)
    avisos = []
    personas = []
    real = m._stock.tras_agarre

    def espia(area, t, persona=None):
        avisos.append(area)
        personas.append(persona)
        return real(area, t, persona=persona)
    monkeypatch.setattr(m._stock, 'tras_agarre', espia)
    # El FRAME de esas pruebas es negro entero: para el stock, un corte de
    # vídeo (no lo mira). Con suelo a rayas es una imagen con vídeo.
    frame = tp.FRAME.copy()
    frame[300:460] = SUELO
    bloques = []
    for i in range(8):
        bloques.append(m.procesar(
            frame, {1: {'class': -1, 'box': list(tp.BOX_EN_AREA),
                           'persistent_id': None}}, ahora=medio + i * 0.5))
    assert m.registro.area(0)['agarres'] == 1
    assert avisos == [0]                  # el agarre llegó al vigilante
    # Con el recorte de quien agarró: su caja entera (BOX_EN_AREA).
    x1, y1, x2, y2 = tp.BOX_EN_AREA
    assert personas[0].shape[:2] == (y2 - y1, x2 - x1)
    # La persona tapaba el producto: la foto sigue esperando, y el agarre
    # no vigila nada porque aún no hay referencia.
    stock = [b['stock'] for b in bloques if b and 'stock' in b]
    assert stock and stock[-1]['esperando_foto'] == 1
    assert stock[-1]['referencias'] == 0


def test_con_el_stock_apagado_ni_se_vigila_ni_se_arma(tmp_path, monkeypatch):
    """`stock_activo: false` (6-oct, para la A/B de fps): el agarre cuenta
    igual, pero el vigilante no se llama nunca."""
    import test_anaqueles_productos as tp
    medio = time.mktime((2026, 10, 5, 12, 0, 0, 0, 0, -1))
    monkeypatch.setattr(time, 'time', lambda: medio)
    m = tp._medidor(tmp_path, [tp.FUERA] * 4 + [[(250, 150, 0.9)]] * 4,
                    [tp._prod(0, 'Colcafé', tp.DER)])
    m.cfg['stock_activo'] = False
    llamadas = []
    monkeypatch.setattr(m._stock, 'tras_agarre',
                        lambda *a, **k: llamadas.append('agarre'))
    monkeypatch.setattr(m._stock, 'vigilar',
                        lambda *a, **k: llamadas.append('vigilar'))
    for i in range(8):
        m.procesar(tp.FRAME, {1: {'class': -1, 'box': list(tp.BOX_EN_AREA),
                                  'persistent_id': None}},
                   ahora=medio + i * 0.5)
    assert m.registro.area(0)['agarres'] == 1
    assert llamadas == []


# ── La alerta con foto: el ANTES y el DESPUÉS (5-oct) ───────────────────
#: La persona que agarró: un recorte reconocible (rojo puro). Ya NO es la
#: foto de la alerta (el usuario prefirió el antes y el después); sigue en
#: la evidencia en disco.
PERSONA = np.zeros((300, 120, 3), dtype=np.uint8)
PERSONA[..., 2] = 255


def _b64_a_imagen(b64):
    import base64
    import cv2
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                        cv2.IMREAD_COLOR)


def _agotar(tmp_path, persona=PERSONA):
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10, persona=persona)
    _pasar(v, VACIO, T0 + 10, T0 + 20)
    return v


def test_la_foto_de_la_alerta_es_el_antes_y_el_despues(tmp_path):
    alertas = _agotar(tmp_path).drenar_alertas()
    assert len(alertas) == 1
    a = alertas[0]
    assert a['producto'] == 'Colcafé' and a['area_nombre'] == 'Anaquel 2'
    assert a['hora_agarre'] == st._hora(T0 + 10)
    foto = _b64_a_imagen(a['imagen_b64'])
    # Cuadrada: la tarjeta la pinta en 56×56 estirándola.
    lado = st._LADO_MINIATURA
    assert foto.shape[:2] == (lado, lado)
    # Recuadro apaisado: ANTES arriba (el tablero, mucho dibujo) y AHORA
    # abajo (liso). Se mira el centro de cada mitad, lejos de los rótulos.
    gris = foto.mean(axis=2)
    arriba = gris[lado // 8:lado // 2 - lado // 8, lado // 4:3 * lado // 4]
    abajo = gris[lado // 2 + lado // 8:lado - lado // 8,
                 lado // 4:3 * lado // 4]
    assert arriba.std() > 30 and abajo.std() < 10
    # Y no es la persona (roja entera).
    assert not (foto[..., 2].mean() > 200 and foto[..., 0].mean() < 50)


def test_un_producto_alto_tampoco_se_deforma():
    alto = np.zeros((200, 50, 3), dtype=np.uint8)
    foto = st.componer_miniatura(alto, alto)
    assert foto.shape[:2] == (st._LADO_MINIATURA, st._LADO_MINIATURA)


def test_la_evidencia_en_disco_sigue_con_quien_agarro(tmp_path):
    import cv2
    a = _agotar(tmp_path).drenar_alertas()[0]
    ruta = Path(a['ruta'])
    assert ruta.parent == tmp_path / 'stock_alertas'
    img = cv2.imread(str(ruta))
    # Tres paneles al alto común (+ rótulo arriba y pie abajo).
    assert img.shape[0] == st._ALTO_EVIDENCIA + 28 + 30
    sin_persona = st.componer_evidencia(_recorte(SURTIDO), _recorte(VACIO),
                                        None, 'x')
    assert img.shape[1] > sin_persona.shape[1]


def test_sin_persona_la_evidencia_lleva_solo_antes_y_ahora(tmp_path):
    import cv2
    v = _con_referencia(tmp_path)
    v.tras_agarre(0, T0 + 10)            # sin recorte
    _pasar(v, VACIO, T0 + 10, T0 + 20)
    a = v.drenar_alertas()[0]
    assert a['hora_agarre'] is None
    # La foto de la alerta no depende de la persona.
    assert _b64_a_imagen(a['imagen_b64']).shape[:2] == \
        (st._LADO_MINIATURA, st._LADO_MINIATURA)
    assert cv2.imread(a['ruta']).shape == st.componer_evidencia(
        _recorte(SURTIDO), _recorte(VACIO), None, 'x').shape


def test_la_alerta_sale_una_vez(tmp_path):
    v = _agotar(tmp_path)
    assert len(v.drenar_alertas()) == 1
    _pasar(v, VACIO, T0 + 21, T0 + 60)
    assert v.drenar_alertas() == []


def test_el_pipeline_la_convierte_en_alerta_del_panel():
    """La misma forma que las alertas del personal: la foto (antes y
    después) en `image_base64`/`crop_image` y la evidencia en
    `screenshot_path`."""
    from src.analityc.core.anaqueles import Anaqueles

    class MedidorFalso:
        def drenar_alertas_stock(self):
            return [{'area': 0, 'producto_idx': 1, 'area_nombre': 'Anaquel 1',
                     'producto': 'coffemate', 'cambio': 0.586,
                     'hora': '17:14:25', 'hora_agarre': '17:14:08',
                     'imagen_b64': 'QUJD', 'ruta': 'C:/x/evidencia.jpg'}]

    pipeline = object.__new__(Anaqueles)
    pipeline._medidor = MedidorFalso()
    pipeline._camera_display_name = 'Cámara 3'
    pipeline._alertar_stock('cam-3')
    alerta = pipeline._alertas_pendientes[-1]
    assert alerta['event_type'] == 'Posible agotado'
    assert alerta['class_name'] == 'coffemate'
    assert alerta['image_base64'] == alerta['crop_image'] == 'QUJD'
    assert alerta['screenshot_path'] == 'C:/x/evidencia.jpg'
    assert 'Anaquel 1' in alerta['description']
    assert 'detectado a las 17:14:25' in alerta['description']
    assert '17:14:08' in alerta['description']


# ── Reintentar la referencia de disco (5-oct, 18:06) ────────────────────
def test_si_el_primer_frame_no_cuadra_se_reintenta(tmp_path):
    """Al reconectar el cliente, el primer frame trajo otros píxeles y,
    con un solo intento, las 8 referencias no se cargaron NUNCA. Ahora se
    reintenta cuando el recuadro vuelve a cuadrar."""
    _con_referencia(tmp_path)
    otro = _vigilante(tmp_path)               # el servidor tras reconectar
    raro = {0: [(0, 'Colcafé', POLI * np.float32(0.5))]}
    otro.vigilar(SURTIDO, raro, NOMBRES, [LEJOS], T0 + 5)
    assert otro.bloque()['referencias'] == 0
    otro.vigilar(SURTIDO, PRODUCTOS, NOMBRES, [LEJOS], T0 + 6)
    assert otro.bloque()['referencias'] == 1


def test_no_relee_el_disco_en_cada_frame(tmp_path, monkeypatch):
    _con_referencia(tmp_path)
    otro = _vigilante(tmp_path)
    raro = {0: [(0, 'Colcafé', POLI * np.float32(0.5))]}
    lecturas = []
    real = Path.read_text
    monkeypatch.setattr(Path, 'read_text',
                        lambda s, *a, **k: (lecturas.append(s),
                                            real(s, *a, **k))[1])
    for i in range(5):
        otro.vigilar(SURTIDO, raro, NOMBRES, [LEJOS], T0 + 5 + i)
    assert len(lecturas) == 1                 # mismos píxeles: una vez
