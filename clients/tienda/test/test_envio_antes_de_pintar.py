"""Tarea 2 en el recuadro de este cliente: al llegar la respuesta, la captura
guardada se ENVÍA antes de pintar, y se pinta la imagen de ESA respuesta
(5-oct-2026).

Se instancia el `Render_box` de verdad (Qt offscreen), como en
test_puerta_direccional.py. Lo que se protege:

  - el orden: primero sale la siguiente imagen, después se pinta;
  - las cajas van sobre la imagen respondida, no sobre la recién enviada
    (enviar sobrescribe `_pending_frame_bytes`);
  - queda UNA imagen en vuelo: si salió la siguiente, la respuesta no
    libera el envío (serían dos, y la cola del servidor no lo espera);
  - sin captura guardada, todo como antes: se libera el envío;
  - un error del servidor no bloquea el recuadro.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _SocketFalso(QObject):
    connected_signal = Signal(dict)
    disconnected_signal = Signal(dict)
    signal_inference = Signal(dict)

    def __init__(self, diario):
        super().__init__()
        self._diario = diario

    def is_connected(self):
        return True

    def send_binary_frame(self, clave, data):
        self._diario.append(('enviar', data.get('image')))


def _recuadro():
    from gui.components.render_box.render_box import Render_box
    diario = []
    caja = Render_box(index=0, socket_services=_SocketFalso(diario),
                      callback_save_data=lambda i, k, v: None)
    caja.socket = _SocketFalso(diario)
    caja.smart_mode = True
    caja._draw_detections_and_show = (
        lambda det, jpeg=None: diario.append(('pintar', jpeg)))
    return caja, diario


def _respuesta(caja, status='success'):
    return {'component_key': caja.component_key,
            'data': {'status': status, 'camera_id': caja._device_id(),
                     'message': 'fallo de prueba',
                     'metadata': {'detections': [{'box': [1, 2, 3, 4]}]},
                     'processed_image': None}}


def _en_vuelo_A_con_B_guardada(caja):
    caja._pending_frame_bytes = b'A'          # A está en el servidor
    caja.can_send_next_frame = False
    caja._captura_guardada = ({'image': b'B'}, b'B', {}, None)


def test_envia_la_siguiente_antes_de_pintar_y_pinta_la_respondida():
    caja, diario = _recuadro()
    _en_vuelo_A_con_B_guardada(caja)
    caja.on_text_message_received(_respuesta(caja))
    assert diario == [('enviar', b'B'), ('pintar', b'A')]
    assert caja._pending_frame_bytes == b'B'
    assert caja.can_send_next_frame is False      # B en vuelo: una sola


def test_sin_guardada_libera_el_envio_como_antes():
    caja, diario = _recuadro()
    caja._pending_frame_bytes = b'A'
    caja.can_send_next_frame = False
    caja._captura_guardada = None
    caja.on_text_message_received(_respuesta(caja))
    assert diario == [('pintar', b'A')]
    assert caja.can_send_next_frame is True


def test_un_error_del_servidor_no_bloquea_el_recuadro():
    caja, diario = _recuadro()
    _en_vuelo_A_con_B_guardada(caja)
    caja.on_text_message_received(_respuesta(caja, status='error'))
    assert diario == [('enviar', b'B')]           # B sigue su camino
    assert caja.can_send_next_frame is False      # y está en vuelo
    caja.on_text_message_received(_respuesta(caja))   # llega la de B
    assert caja.can_send_next_frame is True


def test_la_respuesta_de_otra_camara_no_hace_enviar():
    """Varias cámaras (6-oct-2026): cada recuadro recibe TODAS las
    respuestas. Si la de la cámara A hiciera enviar la guardada de B, B
    tendría dos imágenes en el servidor, la cola tiraría una sin respuesta
    y B se quedaría sin enviar hasta reconectar (hallazgo 30)."""
    a, _ = _recuadro()
    b, diario_b = _recuadro()
    a.component_key, b.component_key = 'recuadro-A', 'recuadro-B'
    a._device_id = lambda: 'win-camA'             # cámaras distintas
    b._device_id = lambda: 'win-camB'
    _en_vuelo_A_con_B_guardada(b)
    b.on_text_message_received(_respuesta(a))     # la respuesta es de A
    assert diario_b == []                          # B ni envía ni pinta
    assert b.can_send_next_frame is False          # sigue esperando la suya
    assert b._captura_guardada is not None
    b.on_text_message_received(_respuesta(b))     # llega la suya
    assert diario_b[0] == ('enviar', b'B')


def test_apagar_la_ia_olvida_la_guardada():
    caja, _ = _recuadro()
    _en_vuelo_A_con_B_guardada(caja)
    caja.activate_modesmart()                     # la apaga
    assert caja._captura_guardada is None
