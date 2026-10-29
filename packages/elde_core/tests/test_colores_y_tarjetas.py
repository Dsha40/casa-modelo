"""
Colores de los anaqueles y tarjetas del panel de visitantes (28-oct-2026).

Lo pidió el usuario: áreas de anaquel en verde, cada producto de un color y
uno nuevo de un color distinto; y que en la columna de Visitantes la foto y
el texto no se solapen ni se corten. Lo que se fija:

  - **los productos no repiten color** entre los 40 primeros, **ninguno es
    verde** (se confundiría con su área) y el primero sigue violeta;
  - la miniatura de Visitantes **no deforma**: mantiene la proporción de la
    foto (antes, un cuadrado de 56 px que aplastaba el recorte);
  - el estilo de la tarjeta **no se cuela** en sus textos (una QLabel es un
    QFrame y heredaba la barra de color).
"""
from __future__ import annotations

import base64
import colorsys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.ui.productos_anaquel import color_de  # noqa: E402


def test_cada_producto_su_color_y_ninguno_verde():
    colores = [color_de(i) for i in range(40)]
    assert len(set(colores)) == len(colores)
    assert colores[0] == (190, 100, 255)
    for c in colores:
        tono, sat, _ = colorsys.rgb_to_hsv(*[x / 255 for x in c])
        assert sat < 0.15 or not (60 <= tono * 360 < 165), c


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _foto_b64(ancho, alto):
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QPixmap
    pix = QPixmap(ancho, alto)
    pix.fill(QColor(200, 30, 30))
    datos = QByteArray()
    buf = QBuffer(datos)
    buf.open(QIODevice.WriteOnly)
    pix.save(buf, 'JPEG')
    return base64.b64encode(bytes(datos)).decode()


def test_miniatura_arriba_no_deforma_la_foto():
    _app()
    from elde_core.ui.panel_alertas import _MINIATURA_ALTO, TarjetaAlerta
    from PySide6.QtWidgets import QLabel
    tarjeta = TarjetaAlerta({'event_type': 'Visitante',
                             'description': 'Visitante Mujer, 26-35 años.',
                             'crop_image': _foto_b64(60, 150)},
                            '#00a8e8', miniatura_arriba=True)
    fotos = [l.pixmap() for l in tarjeta.findChildren(QLabel)
             if l.pixmap() is not None and not l.pixmap().isNull()]
    assert len(fotos) == 1
    assert fotos[0].height() == _MINIATURA_ALTO
    assert abs(fotos[0].width() / fotos[0].height() - 60 / 150) < 0.02


def test_el_estilo_de_la_tarjeta_no_se_cuela_en_los_textos():
    _app()
    from elde_core.ui.panel_alertas import TarjetaAlerta
    tarjeta = TarjetaAlerta({'event_type': 'Visitante'}, '#00a8e8',
                            miniatura_arriba=True)
    assert 'QFrame {' not in tarjeta.styleSheet()
    assert '#tarjetaAlerta' in tarjeta.styleSheet()
