"""El recuadro pinta las cajas sobre la imagen ya reducida (28-oct-2026).

Desde el 28-oct `_draw_detections_and_show` abre el JPEG al tamaño del
recuadro (`elde_core.ui.jpeg_a_medida`) y escala el pintor. Se instancia el
`Render_box` de verdad (Qt offscreen), como en test_envio_antes_de_pintar.py.
Lo que se protege:

  - la imagen se abre ya al tamaño del recuadro, no entera;
  - las cajas, que llegan en píxeles de la imagen ENVIADA, caen en su sitio
    sobre la reducida;
  - `image_w`/`image_h` siguen siendo los de la imagen enviada (las zonas y
    la regla pasan con ellos de 0-1000 a píxeles);
  - con el recuadro mayor que la imagen, todo como antes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))

from PySide6.QtCore import QBuffer, QIODevice, QObject, Signal  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

CAJA = [{'box': [100, 100, 300, 300], 'color': [0, 255, 0], 'label': ''}]


class _SocketFalso(QObject):
    connected_signal = Signal(dict)
    disconnected_signal = Signal(dict)
    signal_inference = Signal(dict)

    def is_connected(self):
        return True

    def send_binary_frame(self, clave, data):
        pass


def _jpeg_negro(ancho=800, alto=600):
    img = QImage(ancho, alto, QImage.Format_RGB32)
    img.fill(QColor(0, 0, 0))
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, 'JPEG', 95)
    return bytes(buf.data())


def _recuadro(ancho, alto):
    from gui.components.render_box.render_box import Render_box
    caja = Render_box(index=0, socket_services=_SocketFalso(),
                      callback_save_data=lambda i, k, v: None)
    caja._try_supervision_overlay = lambda det, jpeg=None: False
    caja.imagen_label.resize(ancho, alto)
    return caja


def _verde(pixmap, x, y):
    c = pixmap.toImage().pixelColor(x, y)
    return c.green() > 60 and c.red() < 60 and c.blue() < 60


def test_se_abre_ya_al_tamano_del_recuadro(monkeypatch):
    from gui.components.render_box import render_box as rb
    abiertas = []
    original = rb._jpeg_a_medida.abrir

    def espia(jpeg, ancho, alto):
        imagen, real_w, real_h = original(jpeg, ancho, alto)
        abiertas.append((imagen.width(), imagen.height()))
        return imagen, real_w, real_h

    monkeypatch.setattr(rb._jpeg_a_medida, 'abrir', espia)
    caja = _recuadro(400, 300)
    caja._draw_detections_and_show(CAJA, _jpeg_negro())
    assert abiertas == [(400, 300)]


def test_la_caja_cae_en_su_sitio_sobre_la_imagen_reducida():
    caja = _recuadro(400, 300)
    caja._draw_detections_and_show(CAJA, _jpeg_negro())
    pix = caja.imagen_label.pixmap()
    assert (pix.width(), pix.height()) == (400, 300)
    # La caja va de 100 a 300 en la imagen de 800x600: de 50 a 150 aquí.
    assert _verde(pix, 50, 100)            # borde izquierdo
    assert _verde(pix, 100, 150)           # borde de abajo
    assert not _verde(pix, 100, 100)       # dentro, sin pintar
    assert not _verde(pix, 100, 50 + 110)  # fuera, por debajo
    # Lo que usan las zonas y la regla: el tamaño de la imagen ENVIADA.
    assert (caja.image_w, caja.image_h) == (800, 600)


def test_con_el_recuadro_mayor_que_la_imagen_todo_como_antes():
    caja = _recuadro(1000, 750)
    caja._draw_detections_and_show(CAJA, _jpeg_negro())
    pix = caja.imagen_label.pixmap()
    assert (pix.width(), pix.height()) == (1000, 750)
    assert _verde(pix, 125, 250)           # 100 x 1,25 = 125
    assert (caja.image_w, caja.image_h) == (800, 600)
