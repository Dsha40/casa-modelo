"""
Productos dibujados DENTRO de las áreas del anaquel (25-sep-2026).

El operador dibuja, dentro de una balda, un cuadrilátero por marca («Nescafé
Gold», «Colcafé»…) y el servidor reparte los AGARRES del área entre ellos
según dónde entró la muñeca. Lo que más se protege:

  - **el agarre va al producto bajo la muñeca**, y el que no cae en ninguno
    va a `agarres_sin_producto`, que es la señal de productos mal dibujados;
  - **un área sin productos cuenta EXACTAMENTE como antes** — ni un contador
    nuevo que se mueva, ni una clave nueva en el feed;
  - **repartir no decide agarres**: el total del área es el mismo con y sin
    productos (los productos solo reparten lo que ya se contaba);
  - **la identidad es la POSICIÓN** dentro del área, no el nombre: corregir
    una errata no parte los contadores, y un producto roto no corre a los
    que van detrás;
  - **los totales sobreviven al renacer** y los JSON de antes se siguen
    leyendo;
  - **el contrato lo declara** (un campo mal formado sí tira el frame en
    modo estricto, así que se prueba que la validación es solo de forma).

Mismo método que `test_anaqueles.py`: `MedidorAnaqueles` con dobles de pose y
escena, registro en `tmp_path`, reloj de pared congelado.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

from src.analityc.core import anaqueles_productos as ap  # noqa: E402
from src.analityc.core.anaqueles import (  # noqa: E402
    Anaqueles, MedidorAnaqueles, _DEFECTOS)
# Estas pruebas fijan la MÁQUINA del gesto y la visita del cuerpo: desde
# el 12-oct-2026 los defectos cuentan la visita con la muñeca y esperan
# la bajada para contar el agarre (test_anaqueles_confirmacion.py).
_REGLAS_DEL_GESTO = {'visita_por_muneca': False, 'agarre_requiere_bajada': False}
from src.analityc.core.anaqueles_registro import (  # noqa: E402
    RegistroAnaqueles)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
MEDIODIA = time.mktime((2026, 9, 25, 12, 0, 0, 0, 0, -1))

#: Una balda: x 100..300, y 100..200 (la misma de `test_anaqueles.py`).
AREA = [[100, 100], [300, 100], [300, 200], [100, 200]]
#: Persona quieta pegada a la balda: su franja superior pisa el área.
BOX_EN_AREA = [120, 120, 220, 460]

#: Dos marcas lado a lado que cubren la balda entera.
IZQ = [[100, 100], [200, 100], [200, 200], [100, 200]]
DER = [[200, 100], [300, 100], [300, 200], [200, 200]]
#: Dos marcas que dejan un HUECO en medio (x 160..240 sin producto).
IZQ_ESTRECHO = [[100, 100], [160, 100], [160, 200], [100, 200]]
DER_ESTRECHO = [[240, 100], [300, 100], [300, 200], [240, 200]]

FUERA = [(400, 300, 0.9)]


def _prod(anaquel, nombre, poligono):
    """Un producto tal como lo manda el recuadro."""
    return {'anaquel': anaquel, 'nombre': nombre, 'poligono': poligono}


class PoseSecuencia:
    """Guion de muñecas: una lista por llamada; la última se repite."""

    def __init__(self, guion):
        self.guion = list(guion)
        self.llamadas = 0
        self.disponible = True

    def munecas(self, frame, box):
        i = min(self.llamadas, len(self.guion) - 1)
        self.llamadas += 1
        return list(self.guion[i])


class EscenaEstable:
    def evaluar(self, camara, frame, ahora=None):
        return None


class SinPersonal:
    """Nadie es empleado: sin esto el medidor cargaría la galería real."""

    def ficha_de(self, pid, reidentifier):
        return None


@pytest.fixture(autouse=True)
def reloj(monkeypatch):
    """Reloj de pared congelado (el motivo, en `test_anaqueles.reloj`): la
    jornada del registro la decide `time.strftime` real y un test que
    cruzara la medianoche vería las áreas reabrirse a cero."""
    strftime_real = time.strftime
    localtime_real = time.localtime

    def strftime_congelado(formato, tupla=None):
        if tupla is None:
            tupla = localtime_real(MEDIODIA)
        return strftime_real(formato, tupla)

    monkeypatch.setattr(time, 'time', lambda: MEDIODIA)
    monkeypatch.setattr(time, 'strftime', strftime_congelado)


def _medidor(tmp_path, guion, productos=None):
    registro = RegistroAnaqueles('cam-test', raiz=tmp_path)
    m = MedidorAnaqueles('cam-test', dict(_DEFECTOS, **_REGLAS_DEL_GESTO),
                         pose=PoseSecuencia(guion), registro=registro,
                         escena=EscenaEstable(), identificador=SinPersonal())
    m.poner_zonas([AREA], ['Anaquel 2'])
    m.poner_productos(productos)
    return m


def _gesto_en(tmp_path, x, y, productos=None):
    """Mano fuera 4 frames (asienta y arma), luego DENTRO en (x, y): el
    gesto que `test_anaqueles` ya fija como UN agarre."""
    m = _medidor(tmp_path, [FUERA] * 4 + [[(x, y, 0.9)]] * 4, productos)
    for i in range(8):
        m.procesar(FRAME, {1: {'class': -1, 'box': list(BOX_EN_AREA),
                               'persistent_id': None}},
                   ahora=MEDIODIA + i * 0.5)
    return m


# ── El reparto ───────────────────────────────────────────────────────────
def test_el_agarre_va_al_producto_bajo_la_muneca(tmp_path):
    m = _gesto_en(tmp_path, 250, 150, [_prod(0, 'Nescafé Gold', IZQ),
                                       _prod(0, 'Colcafé', DER)])
    area = m.registro.area(0)
    assert area['agarres'] == 1
    assert area['productos']['1'] == {'nombre': 'Colcafé', 'agarres': 1}
    assert area['productos']['0']['agarres'] == 0
    assert area['agarres_sin_producto'] == 0


def test_el_agarre_fuera_de_todo_producto_va_a_sin_producto(tmp_path):
    """La mano entra por el hueco entre las dos marcas dibujadas."""
    m = _gesto_en(tmp_path, 200, 150,
                  [_prod(0, 'Nescafé Gold', IZQ_ESTRECHO),
                   _prod(0, 'Colcafé', DER_ESTRECHO)])
    area = m.registro.area(0)
    assert area['agarres'] == 1                    # el área cuenta igual
    assert area['agarres_sin_producto'] == 1
    assert all(p['agarres'] == 0 for p in area['productos'].values())


def test_sin_productos_el_area_cuenta_EXACTAMENTE_como_antes(tmp_path):
    """Ni `agarres_sin_producto` se mueve (sería el total repetido), ni el
    feed estrena la clave `producto`."""
    m = _gesto_en(tmp_path, 250, 150, productos=None)
    area = m.registro.area(0)
    assert area['agarres'] == 1
    assert area['productos'] == {}
    assert area['agarres_sin_producto'] == 0
    agarres = [e for e in m._feed if e['tipo'] == 'agarre']
    assert agarres and all('producto' not in e for e in agarres)


def test_repartir_no_cambia_cuantos_agarres_hay(tmp_path):
    """Los productos reparten lo que YA se contaba: el total del área es
    el mismo con productos, sin ellos y con la mano en el hueco."""
    sin = _gesto_en(tmp_path / 'a', 250, 150).registro.area(0)['agarres']
    con = _gesto_en(tmp_path / 'b', 250, 150,
                    [_prod(0, 'Colcafé', DER)]).registro.area(0)['agarres']
    hueco = _gesto_en(tmp_path / 'c', 200, 150,
                      [_prod(0, 'X', IZQ_ESTRECHO)]).registro.area(0)[
                          'agarres']
    assert sin == con == hueco == 1


def test_el_evento_y_el_log_dicen_el_producto(tmp_path, caplog):
    """El log es la forma de verificarlo en producción sin panel todavía."""
    with caplog.at_level(logging.INFO):
        m = _gesto_en(tmp_path, 150, 150, [_prod(0, 'Nescafé Gold', IZQ)])
    agarre = [e for e in m._feed if e['tipo'] == 'agarre'][-1]
    assert agarre['producto'] == 'Nescafé Gold'
    assert any('AGARRE #1 en Anaquel 2 → Nescafé Gold' in r.getMessage()
               for r in caplog.records)


# ── Qué producto hay bajo un punto ───────────────────────────────────────
def _productos_de(*poligonos):
    return ap.productos_por_area(
        [_prod(0, f'P{i}', p) for i, p in enumerate(poligonos)], 1)[0]


def test_el_borde_compartido_tiene_dueno():
    """Dos marcas pegadas comparten lado: el punto justo encima no puede
    quedarse sin producto (sería un agarre «sin producto» falso)."""
    assert ap.producto_en(_productos_de(IZQ, DER), 200, 150) is not None


def test_con_solape_gana_donde_la_muneca_esta_mas_adentro():
    ancho = [[100, 100], [300, 100], [300, 200], [100, 200]]
    estrecho = [[180, 120], [220, 120], [220, 180], [180, 180]]
    # (200,150) está a 50 px del borde del ancho y a 20 del estrecho.
    assert ap.producto_en(_productos_de(ancho, estrecho), 200, 150) == \
        (0, 'P0')
    # (110,150): solo dentro del ancho.
    assert ap.producto_en(_productos_de(estrecho, ancho), 110, 150) == \
        (1, 'P1')


def test_fuera_de_todos_es_None():
    assert ap.producto_en(_productos_de(IZQ_ESTRECHO), 250, 150) is None


# ── Identidad y forma de lo que llega ────────────────────────────────────
def test_se_agrupan_por_area_y_numeran_dentro_de_cada_una():
    crudo = [_prod(0, 'A', IZQ), _prod(1, 'B', IZQ), _prod(0, '', DER)]
    por_area = ap.productos_por_area(crudo, 2)
    assert [(p, n) for p, n, _ in por_area[0]] == [(0, 'A'),
                                                   (1, 'Producto 2')]
    assert [(p, n) for p, n, _ in por_area[1]] == [(0, 'B')]


def test_un_producto_roto_no_corre_la_identidad_de_los_siguientes():
    """Mismo principio que `poner_zonas`: la posición se reserva ANTES de
    descartar. Si no, «Colcafé» heredaría los agarres de «Nescafé»."""
    crudo = [_prod(0, 'Nescafé', [[1, 1], [2, 2]]),       # 2 puntos: roto
             _prod(0, 'Colcafé', DER)]
    assert [(p, n) for p, n, _ in ap.productos_por_area(crudo, 1)[0]] == \
        [(1, 'Colcafé')]


def test_los_huerfanos_y_la_basura_se_ignoran_sin_lanzar():
    crudo = [_prod(3, 'De un área que no existe', IZQ),
             {'anaquel': 'no', 'poligono': IZQ},
             {'anaquel': 0, 'poligono': 'tampoco'},
             'ni siquiera un dict',
             _prod(0, 'Bueno', DER)]
    por_area = ap.productos_por_area(crudo, 1)
    assert [n for _p, n, _ in por_area[0]] == ['Bueno']
    assert ap.firma(crudo, 1) is not None          # la huella tampoco lanza


def test_se_dan_de_alta_al_dibujarlos_y_renombrar_conserva(tmp_path):
    """La API tiene que listar un producto dibujado aunque nadie lo haya
    tocado; y corregir una errata no puede poner su contador a cero."""
    m = _gesto_en(tmp_path, 150, 150, [_prod(0, 'Nescafe Gol', IZQ)])
    assert m.registro.area(0)['productos']['0'] == {
        'nombre': 'Nescafe Gol', 'agarres': 1}
    m.poner_productos([_prod(0, 'Nescafé Gold', IZQ),
                       _prod(0, 'Colcafé', DER)])
    assert m.registro.area(0)['productos'] == {
        '0': {'nombre': 'Nescafé Gold', 'agarres': 1},
        '1': {'nombre': 'Colcafé', 'agarres': 0}}


def test_un_producto_de_un_area_borrada_deja_de_repartir(tmp_path):
    """Si el operador quita el área, su producto queda huérfano: ya no
    reparte nada (la huella lleva el número de áreas por esto)."""
    m = _medidor(tmp_path, [FUERA], [_prod(0, 'X', IZQ)])
    assert m._productos
    m.poner_zonas([], [])
    m.poner_productos([_prod(0, 'X', IZQ)])
    assert m._productos == {}


# ── Disco ────────────────────────────────────────────────────────────────
def test_los_productos_sobreviven_al_renacer(tmp_path):
    m = _gesto_en(tmp_path, 250, 150, [_prod(0, 'Nescafé Gold', IZQ),
                                       _prod(0, 'Colcafé', DER)])
    m.registro.volcar(force=True)
    renacido = RegistroAnaqueles('cam-test', raiz=tmp_path)
    area = renacido.area(0)
    assert area['productos']['1'] == {'nombre': 'Colcafé', 'agarres': 1}
    assert area['agarres_sin_producto'] == 0


def test_un_json_de_antes_del_25_sep_se_sigue_leyendo(tmp_path):
    carpeta = tmp_path / 'cam-test'
    carpeta.mkdir()
    (carpeta / f"{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps({
        'fecha': time.strftime('%Y-%m-%d'), 'camera_id': 'cam-test',
        'grid_dim': [12, 24],
        'areas': {'0': {'nombre': 'Anaquel 2', 'visitas': 5,
                        'agarres': 3}}}), encoding='utf-8')
    area = RegistroAnaqueles('cam-test', raiz=tmp_path).area(0)
    assert area['agarres'] == 3                    # lo de antes, intacto
    assert area['productos'] == {} and area['agarres_sin_producto'] == 0


# ── Del procesador al medidor ────────────────────────────────────────────
def _procesador(medidor, activado=True):
    """`Anaqueles` sin su `__init__` (que cargaría los modelos): solo los
    atributos que `app.py` cuelga en cada frame."""
    proc = Anaqueles.__new__(Anaqueles)
    proc._medidor, proc._medidor_camara = medidor, 'cam-test'
    proc._anaqueles_zonas = [AREA]
    proc._anaqueles_nombres = ['Anaquel 2']
    proc._anaqueles_activate = activado
    proc._anaqueles_productos = [_prod(0, 'Colcafé', DER)]
    proc.active_tracks = {}
    return proc


def test_el_procesador_entrega_los_productos_que_cuelga_app(tmp_path):
    m = _medidor(tmp_path, [FUERA])
    _procesador(m)._medir_anaqueles(FRAME, 'cam-test')
    assert [n for _p, n, _ in m._productos[0]] == ['Colcafé']


def test_con_las_areas_apagadas_tampoco_hay_productos(tmp_path):
    m = _medidor(tmp_path, [FUERA], [_prod(0, 'Previo', IZQ)])
    _procesador(m, activado=False)._medir_anaqueles(FRAME, 'cam-test')
    assert m._productos == {}


# ── Contrato ─────────────────────────────────────────────────────────────
def _jpeg():
    import cv2
    return cv2.imencode('.jpg', np.zeros((64, 64, 3), np.uint8))[1].tobytes()


def test_el_contrato_declara_los_productos():
    from elde_core.contracts.eventos import FrameInference
    frame = FrameInference(image=_jpeg(), anaqueles_zonas=[AREA],
                           anaqueles_productos=[_prod(0, 'Colcafé', DER)])
    assert frame.anaqueles_productos[0].nombre == 'Colcafé'
    # Un cliente sin productos sigue valiendo, con la lista vacía o sin ella.
    FrameInference(image=_jpeg(), anaqueles_productos=[])
    FrameInference(image=_jpeg())


def test_el_contrato_valida_la_forma_y_nada_mas():
    """Un campo DECLARADO y mal formado tira el frame en modo estricto, así
    que se valida lo imprescindible. Un producto que apunta a un área que
    no existe NO es error: el servidor lo ignora (ver `ProductoAnaquel`)."""
    import pydantic
    from elde_core.contracts.eventos import FrameInference
    FrameInference(image=_jpeg(), anaqueles_zonas=[AREA],
                   anaqueles_productos=[_prod(7, 'Huérfano', DER)])
    with pytest.raises(pydantic.ValidationError):
        FrameInference(image=_jpeg(),
                       anaqueles_productos=[_prod(0, 'X', [[1, 1], [2, 2]])])


def test_el_servidor_en_estricto_acepta_el_frame_con_productos(monkeypatch):
    """El camino de producción: `validacion_contrato.revisar` en estricto.
    Y la prueba de que el campo está DECLARADO y no solo tolerado: uno mal
    formado se RECHAZA, cosa que no pasaría con un campo desconocido."""
    from src.app import validacion_contrato as vc
    monkeypatch.setattr(vc, 'MODO', 'estricto')

    def mensaje(productos):
        return {'event': 'frame', 'id_connection': 'c-1',
                'type_inference': 'Anaqueles', 'component_key': 'k',
                'data': {'image': _jpeg(), 'roi_coordinates': [],
                         'roi_activate': False, 'anaqueles_activate': True,
                         'anaqueles_zonas': [AREA],
                         'anaqueles_nombres': ['Anaquel 2'],
                         'anaqueles_productos': productos}}

    assert vc.revisar(mensaje([_prod(0, 'Colcafé', DER)])) == (True, None)
    ok, _motivo = vc.revisar(mensaje([_prod(0, 'X', [[1, 1], [2, 2]])]))
    assert ok is False


def test_lo_que_empaqueta_el_cliente_lo_reparte_el_servidor():
    """Las dos puntas juntas: `ProductosAnaquel` (cliente, sin Qt) arma el
    campo y `productos_por_area` (servidor) lo lee. Si una cambiara la forma
    sin la otra, esto se pone rojo en vez de dejar de contar en silencio."""
    from elde_core.ui.productos_anaquel import ProductosAnaquel
    cliente = ProductosAnaquel()
    area_visor = [[100, 200], [500, 200], [500, 400], [100, 400]]
    cliente.agregar(0, area_visor, 'Nescafé Gold')
    cliente.agregar(0, area_visor, 'Colcafé')
    enviado = cliente.coordenadas_pixel(1000, 1000, n_anaqueles=1)

    por_area = ap.productos_por_area(enviado, 1)
    assert [(p, n) for p, n, _ in por_area[0]] == [(0, 'Nescafé Gold'),
                                                   (1, 'Colcafé')]
    # El centro de cada producto dibujado cae en ESE producto.
    for i, prod in enumerate(enviado):
        xs = [pt[0] for pt in prod['poligono']]
        ys = [pt[1] for pt in prod['poligono']]
        centro = (sum(xs) / 4, sum(ys) / 4)
        assert ap.producto_en(por_area[0], *centro) == (i, prod['nombre'])


# ── API ──────────────────────────────────────────────────────────────────
def test_la_api_devuelve_productos_y_su_ranking(monkeypatch):
    from src.app import anaqueles_api
    datos = {'fecha': '2026-09-25', 'grid_dim': [12, 24], 'areas': {
        '0': {'nombre': 'Anaquel 1', 'visitas': 11, 'agarres': 8,
              'productos': {'0': {'nombre': 'Illy', 'agarres': 2},
                            '1': {'nombre': 'Pilón', 'agarres': 5}},
              'agarres_sin_producto': 1},
        '1': {'nombre': 'Anaquel 2', 'visitas': 54, 'agarres': 1,
              'productos': {'0': {'nombre': 'Colcafé', 'agarres': 1}},
              'agarres_sin_producto': 0}}}
    monkeypatch.setattr(anaqueles_api, 'ultimo_registro',
                        lambda cam, fecha=None: datos)

    r = anaqueles_api.resumen('cam-test')

    assert r['areas'][0]['productos'] == [
        {'idx': 0, 'nombre': 'Illy', 'agarres': 2},
        {'idx': 1, 'nombre': 'Pilón', 'agarres': 5}]
    assert r['areas'][0]['agarres_sin_producto'] == 1
    assert [(p['area'], p['nombre'], p['agarres'])
            for p in r['ranking_productos']] == [
        ('Anaquel 1', 'Pilón', 5), ('Anaquel 1', 'Illy', 2),
        ('Anaquel 2', 'Colcafé', 1)]


def test_la_api_con_un_dia_sin_productos_no_cambia_nada(monkeypatch):
    from src.app import anaqueles_api
    datos = {'fecha': '2026-09-24', 'areas': {
        '0': {'nombre': 'Anaquel 1', 'visitas': 17, 'agarres': 3}}}
    monkeypatch.setattr(anaqueles_api, 'ultimo_registro',
                        lambda cam, fecha=None: datos)
    r = anaqueles_api.resumen('cam-test')
    assert r['areas'][0]['productos'] == []
    assert r['areas'][0]['agarres_sin_producto'] == 0
    assert r['ranking_productos'] == []
    assert r['total_agarres'] == 3
