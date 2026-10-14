"""Áreas a ignorar del recuadro (12-oct-2026, `areas_ignorar.py`).

Sin Qt para la lógica (nacer, arrastrar, quitar, qué viaja) y con la
etiqueta real para el payload de la ruta de ventana, que es la que usan iVMS
y el vídeo.
"""
from __future__ import annotations

from elde_core.ui.areas_ignorar import AreasIgnorar


class _P:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


def _escalar(x, y):          # visor de 1000×1000: mismo espacio
    return _P(x, y)


def test_nace_encendida_y_viaja_en_pixeles():
    a = AreasIgnorar()
    assert a.coordenadas_pixel(1456, 868) == []
    a.agregar()
    assert a.activa
    zona = a.coordenadas_pixel(1000, 500)
    assert len(zona) == 1 and zona[0][0] == [100, 30]


def test_apagada_no_viaja_nada():
    a = AreasIgnorar([[[0, 0], [500, 0], [500, 500], [0, 500]]], activa=False)
    assert a.coordenadas_pixel(1000, 1000) == []
    a.activa = True
    assert a.coordenadas_pixel(1000, 1000) == [[[0, 0], [500, 0], [500, 500], [0, 500]]]


def test_se_arrastra_una_esquina_y_se_queda_en_la_imagen():
    a = AreasIgnorar()
    a.agregar()
    esquina = a.vertice_en(_P(100, 60), _escalar)
    assert esquina == (0, 0)
    a.mover(0, 0, -50, 2000)
    assert a.areas[0][0] == [0, 1000]


def test_se_quita_la_ultima_y_se_restaura_lo_guardado():
    a = AreasIgnorar()
    a.agregar()
    a.agregar()
    guardado = a.a_lista()
    assert a.quitar_ultima() and len(a.areas) == 1
    b = AreasIgnorar(guardado + [[[1, 2]], 'basura'], activa=True)
    assert b.a_lista() == guardado


def test_la_ruta_de_ventana_las_manda():
    """`render_box_captura` arma el payload de iVMS y del vídeo: tiene que
    llevar `zonas_ignorar` también fuera del modo Anaqueles."""
    from elde_core.ui import render_box_captura as rbc
    src = open(rbc.__file__, encoding='utf-8').read()
    i = src.index('salida["zonas_ignorar"]')
    assert 'get_ignorar_coordinates' in src[i - 300:i]


def test_la_etiqueta_avisa_y_entrega_las_areas():
    """Con la etiqueta real de Qt: añadir avisa por `ignorar_change` (lo
    guarda el recuadro) y `get_ignorar_coordinates` entrega píxeles."""
    from PySide6.QtWidgets import QApplication
    from elde_core.ui.interactive_image_label import Interactive_imageLabel
    _app = QApplication.instance() or QApplication([])
    etiqueta = Interactive_imageLabel("visor")
    avisos = []
    etiqueta.ignorar_change.connect(lambda areas, activo: avisos.append((areas, activo)))
    etiqueta.agregar_ignorar()
    assert avisos and avisos[-1][1] is True and len(avisos[-1][0]) == 1
    assert len(etiqueta.get_ignorar_coordinates(1456, 868)) == 1
    etiqueta.toggle_ignorar(False)
    assert etiqueta.get_ignorar_coordinates(1456, 868) == []
    assert avisos[-1][1] is False
    etiqueta.quitar_ignorar()
    assert avisos[-1][0] == []
