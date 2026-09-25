"""
Productos dibujados dentro de las áreas de anaquel (25-sep-2026), lado cliente.

`ProductosAnaquel` guarda los puntos como enteros e importa Qt solo al pintar,
así que todo esto corre SIN PySide6 (el servidor no lo tiene). Lo que se fija:

  - **nace DENTRO de su área**, siguiendo su perspectiva, y los siguientes a
    su derecha, como van las marcas en la balda;
  - **solo se quita el último de cada área**: el servidor identifica cada
    producto por su posición y quitar uno del medio le daría a cada vecino
    los agarres del de al lado;
  - **quitar el área se lleva sus productos**;
  - **lo que viaja** son píxeles del frame, solo de áreas que existen, y con
    la forma que declara el contrato (`ProductoAnaquel`);
  - **un nombre largo no puede tirar frames**: el contrato admite 120 y un
    campo declarado que se pasa descarta el frame entero en modo estricto.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.ui.productos_anaquel import (  # noqa: E402
    MAX_NOMBRE, ProductosAnaquel)

#: Un área tal como la crea `agregar_anaquel` (0..1000): sup-izq, sup-der,
#: inf-der, inf-izq.
AREA = [[150, 120], [850, 120], [850, 250], [150, 250]]
#: La misma balda vista en perspectiva: el lado derecho más bajo.
AREA_INCLINADA = [[150, 120], [850, 200], [850, 330], [150, 250]]


class Punto:
    """Lo mínimo de un QPoint: `.x()` y `.y()`."""

    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


def _identidad(x, y):
    """Visor de 1000×1000: 0..1000 == píxeles del visor."""
    return Punto(x, y)


def _dentro(punto, area):
    """¿`punto` dentro del cuadrilátero convexo `area`? (producto cruz)"""
    signos = []
    for (x1, y1), (x2, y2) in zip(area, area[1:] + area[:1]):
        signos.append((x2 - x1) * (punto[1] - y1)
                      - (y2 - y1) * (punto[0] - x1) >= 0)
    return all(signos) or not any(signos)


# ── Dónde nace ───────────────────────────────────────────────────────────
def test_nace_dentro_de_su_area_y_los_siguientes_a_la_derecha():
    p = ProductosAnaquel()
    for _ in range(3):
        p.agregar(0, AREA)
    for prod in p.productos:
        assert all(_dentro(pt, AREA) for pt in prod['puntos'])
    izquierdas = [prod['puntos'][0][0] for prod in p.productos]
    assert izquierdas == sorted(izquierdas) and len(set(izquierdas)) == 3


def test_nace_con_la_perspectiva_de_la_balda():
    """Si la balda baja hacia la derecha, el producto también: nacer como
    rectángulo recto lo sacaría del área en cuanto la cámara está ladeada."""
    p = ProductosAnaquel()
    for _ in range(4):
        p.agregar(0, AREA_INCLINADA)
    for prod in p.productos:
        assert all(_dentro(pt, AREA_INCLINADA) for pt in prod['puntos'])
    ultimo = p.productos[-1]['puntos']
    assert ultimo[0][1] > p.productos[0]['puntos'][0][1]


def test_cada_area_numera_sus_productos_por_separado():
    p = ProductosAnaquel()
    p.agregar(0, AREA, 'Nescafé Gold')
    p.agregar(1, AREA, '')
    p.agregar(0, AREA, '')
    assert p.de_anaquel(0) == [0, 2]
    assert [p.posicion(i) for i in range(3)] == [0, 0, 1]
    # Sin nombre, el mismo «Producto N» que pondrá el servidor.
    assert [p.nombre_visible(i) for i in range(3)] == [
        'Nescafé Gold', 'Producto 1', 'Producto 2']


# ── Quitar y renombrar ───────────────────────────────────────────────────
def test_solo_se_quita_el_ultimo_de_esa_area():
    p = ProductosAnaquel()
    p.agregar(0, AREA, 'A')
    p.agregar(1, AREA, 'X')
    p.agregar(0, AREA, 'B')
    assert p.quitar_ultimo(0) is True
    assert [q['nombre'] for q in p.productos] == ['A', 'X']
    assert p.quitar_ultimo(5) is False


def test_quitar_el_area_se_lleva_sus_productos():
    p = ProductosAnaquel()
    p.agregar(0, AREA, 'A')
    p.agregar(1, AREA, 'X')
    assert p.quitar_de_anaquel(1) is True
    assert [q['nombre'] for q in p.productos] == ['A']
    assert p.quitar_de_anaquel(1) is False        # nada que quitar: no avisa


def test_el_nombre_se_recorta_muy_por_debajo_del_tope_del_contrato():
    p = ProductosAnaquel()
    p.agregar(0, AREA, 'x' * 500)
    p.renombrar(0, '  y' * 300)
    assert len(p.productos[0]['nombre']) == MAX_NOMBRE < 120


# ── Ratón ────────────────────────────────────────────────────────────────
def test_se_coge_la_esquina_del_que_esta_encima():
    """Dos productos con una esquina en el mismo sitio: gana el ÚLTIMO, que
    es el que se pinta encima y el que el operador ve."""
    p = ProductosAnaquel([
        {'anaquel': 0, 'nombre': 'A',
         'puntos': [[100, 100], [200, 100], [200, 200], [100, 200]]},
        {'anaquel': 0, 'nombre': 'B',
         'puntos': [[200, 100], [300, 100], [300, 200], [200, 200]]}])
    assert p.vertice_en(Punto(201, 101), _identidad) == (1, 0)
    assert p.vertice_en(Punto(150, 150), _identidad) is None


def test_arrastrar_no_saca_la_esquina_del_visor():
    p = ProductosAnaquel()
    p.agregar(0, AREA)
    p.mover(0, 2, 1500, -30)
    assert p.productos[0]['puntos'][2] == [1000, 0]
    p.mover(9, 0, 10, 10)                         # índice viejo: no revienta


# ── Lo que viaja al servidor ─────────────────────────────────────────────
def test_viajan_en_pixeles_del_frame_y_solo_de_areas_que_existen():
    p = ProductosAnaquel([
        {'anaquel': 0, 'nombre': 'A',
         'puntos': [[100, 100], [200, 100], [200, 200], [100, 200]]},
        {'anaquel': 3, 'nombre': 'Huérfano',
         'puntos': [[100, 100], [200, 100], [200, 200], [100, 200]]}])
    salida = p.coordenadas_pixel(1280, 720, n_anaqueles=1)
    assert salida == [{'anaquel': 0, 'nombre': 'A',
                       'poligono': [[128, 72], [256, 72],
                                    [256, 144], [128, 144]]}]
    assert p.coordenadas_pixel(0, 720, n_anaqueles=1) == []


def test_lo_que_viaja_lo_acepta_el_contrato():
    from elde_core.contracts.eventos import FrameInference
    p = ProductosAnaquel()
    p.agregar(0, AREA, 'Nescafé Gold')
    p.agregar(0, AREA, 'x' * 500)
    frame = FrameInference(
        image=b'\xff\xd8', anaqueles_activate=True,
        anaqueles_zonas=[[[0, 0], [100, 0], [100, 50], [0, 50]]],
        anaqueles_productos=p.coordenadas_pixel(1280, 720, n_anaqueles=1))
    assert [q.nombre for q in frame.anaqueles_productos] == [
        'Nescafé Gold', 'x' * MAX_NOMBRE]


def test_lo_guardado_vuelve_igual_y_lo_ilegible_se_descarta():
    """La config del recuadro sobrevive a versiones: lo que no se entiende
    se tira, no rompe el recuadro al abrirlo."""
    p = ProductosAnaquel()
    p.agregar(0, AREA, 'A')
    p.agregar(1, AREA, 'B')
    assert ProductosAnaquel(p.a_lista()).a_lista() == p.a_lista()
    basura = [None, 'texto', {'anaquel': 'no'}, {'anaquel': -1, 'puntos':
              [[1, 1], [2, 2], [3, 3]]}, {'anaquel': 0, 'puntos': [[1, 1]]}]
    assert ProductosAnaquel(basura + p.a_lista()).a_lista() == p.a_lista()
