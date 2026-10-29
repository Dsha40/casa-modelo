"""`jpeg_a_medida.abrir`: la imagen sale ya al tamaño del recuadro (28-oct-2026).

Lo que se protege:
  - con el recuadro menor que la foto, sale reducida a su tamaño exacto
    (sin guardar la proporción, como la escala el recuadro);
  - el ancho y el alto REALES vuelven aparte: de ellos dependen las zonas y
    la regla;
  - con el recuadro mayor, o sin tamaño todavía, sale entera, como antes;
  - unos bytes que no son JPEG no rompen nada.
"""
from __future__ import annotations

from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QColor, QGuiApplication, QImage

from elde_core.ui import jpeg_a_medida

_app = QGuiApplication.instance() or QGuiApplication([])


def _jpeg(ancho, alto):
    img = QImage(ancho, alto, QImage.Format_RGB32)
    img.fill(QColor(200, 30, 30))
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, 'JPEG', 90)
    return bytes(buf.data())


def test_sale_reducida_al_tamano_del_recuadro_y_dice_el_real():
    imagen, ancho, alto = jpeg_a_medida.abrir(_jpeg(800, 600), 300, 260)
    assert (imagen.width(), imagen.height()) == (300, 260)
    assert (ancho, alto) == (800, 600)
    # Es la misma foto, no una en blanco.
    color = imagen.pixelColor(150, 130)
    assert color.red() > 150 and color.green() < 80


def test_con_el_recuadro_mayor_o_sin_tamano_sale_entera():
    jpeg = _jpeg(800, 600)
    for ancho, alto in ((1000, 700), (0, 0), (500, 900)):
        imagen, real_w, real_h = jpeg_a_medida.abrir(jpeg, ancho, alto)
        assert (imagen.width(), imagen.height()) == (800, 600)
        assert (real_w, real_h) == (800, 600)


def test_bytes_que_no_son_jpeg():
    assert jpeg_a_medida.abrir(b'no es un jpeg', 300, 200) == (None, 0, 0)
    assert jpeg_a_medida.abrir(b'', 300, 200) == (None, 0, 0)
