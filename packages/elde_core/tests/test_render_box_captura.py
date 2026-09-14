"""
Pruebas de la captura compartida (`elde_core.ui.render_box_captura`).

`loop_show_result` envuelve todo su cuerpo en un `try/except`. Eso esta bien
para que un frame corrupto no tumbe el cliente, pero convierte cualquier
`AttributeError` en un fallo **mudo**: el recuadro deja de enviar frames y no
dice por que. Estas pruebas cubren justo los casos que provocarian eso.

No hace falta un widget de verdad: los dos metodos solo leen atributos y llaman
metodos, asi que se invocan sobre un doble. Corre sin pantalla y sin servidor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.ui.render_box_captura import CapturaDVRMixin  # noqa: E402


class EtiquetaConZonas:
    """La `Interactive_imageLabel` del nucleo: ROI y las TRES zonas del
    comedor (pedido, entrega y personal)."""

    def get_coordinates(self, w, h):
        return [[0, 0], [w, 0], [w, h], [0, h]]

    # Cuadrilateros de verdad: el contrato descarta cualquier poligono de
    # menos de 3 puntos (`_valida_poligono`), asi que un doble que devolviera
    # [[1, 1]] afirmaria un payload que en produccion se tiraria entero.
    def get_order_zone_coordinates(self, w, h):
        return [[10, 10], [20, 10], [20, 20], [10, 20]]

    def get_delivery_zone_coordinates(self, w, h):
        return [[30, 10], [40, 10], [40, 20], [30, 20]]

    def get_personal_zone_coordinates(self, w, h):
        return [[50, 10], [60, 10], [60, 20], [50, 20]]


class EtiquetaSinZonas:
    """Una etiqueta que solo tiene ROI.

    Fue la de Amazonas hasta que se reconcilio con el nucleo (26-ago-2026);
    hoy no es la de ningun cliente vivo, pero el caso sigue valiendo: el
    mixin no puede exigir getters a ciegas, o un cliente sin ellos dejaria de
    enviar frames sin decir por que.
    """

    def get_coordinates(self, w, h):
        return [[0, 0], [w, 0], [w, h], [0, h]]


class EtiquetaSoloPedidoYEntrega:
    """La etiqueta ANTERIOR a la zona del personal (25-ago-2026).

    Existe para fijar que anadir la tercera zona no obliga a nadie: una
    etiqueta con las dos de siempre debe seguir enviando frames, sin la clave
    nueva y sin `AttributeError`.
    """

    def get_coordinates(self, w, h):
        return [[0, 0], [w, 0], [w, h], [0, h]]

    def get_order_zone_coordinates(self, w, h):
        return [[10, 10], [20, 10], [20, 20], [10, 20]]

    def get_delivery_zone_coordinates(self, w, h):
        return [[30, 10], [40, 10], [40, 20], [30, 20]]


class ProcesoFalso:
    def __init__(self, datos: bytes):
        self._datos = datos

    def readAllStandardOutput(self):
        class _B:
            def __init__(self, d):
                self._d = d

            def data(self):
                return self._d
        return _B(self._datos)


class SocketFalso:
    def __init__(self):
        self.enviado = []

    def send_binary_frame(self, clave, data):
        self.enviado.append((clave, data))


class TextoFalso:
    def setText(self, t):
        self.ultimo = t


class RecuadroFalso(CapturaDVRMixin):
    """Lo minimo que la captura compartida necesita leer."""

    def __init__(self, etiqueta):
        mensaje = msgpack.packb(
            {'header': {'x': 1}, 'image_bytes': b'\xff\xd8jpeg'}, use_bin_type=True)
        self.process = ProcesoFalso(mensaje)
        self.socket = SocketFalso()
        self.imagen_label = etiqueta
        self.text_fps = TextoFalso()
        self._unpacker = None
        self.frame_count = 0
        self.last_fps_time = 0.0
        self.current_fps = 0
        self.smart_mode = True
        self.can_send_next_frame = True
        # Distintos de cero: asi no se construye un QPixmap (necesitaria
        # QApplication) y la prueba corre sin entorno grafico.
        self.image_w, self.image_h = 640, 480
        self.roi_boolean = True
        self.order_zone_boolean = False
        self.delivery_zone_boolean = False
        self.vlm_enabled_boolean = False
        self.camera_angle = 'auto'
        self.whatsapp_boolean = False
        self.heatmap_boolean = False
        self._selected_classes = [0]
        self._direct_mode = False
        self._pending_frame_bytes = None
        self.component_key = 'clave-de-enrutado'

    def _device_id(self):
        return 'dvr-J12345678-2'

    def _camera_display_name(self):
        return 'Pasillo 3'

    def update_streaming_frame(self, *a, **k):
        pass


def _enviado(etiqueta):
    r = RecuadroFalso(etiqueta)
    r.loop_show_result()
    assert r.socket.enviado, 'no se envio ningun frame'
    return r.socket.enviado[0][1], r


def test_el_payload_lleva_el_device_id_estable_y_no_el_uuid():
    """H-11 visto desde el envio: lo que viaja es la identidad, no la clave
    de enrutado."""
    data, r = _enviado(EtiquetaConZonas())
    assert data['camera_id'] == 'dvr-J12345678-2'
    assert data['camera_id'] != r.component_key
    # La clave de enrutado si sigue siendo component_key.
    assert r.socket.enviado[0][0] == 'clave-de-enrutado'


def test_una_etiqueta_sin_zonas_no_rompe_el_envio():
    """El fallo mudo que habria roto Amazonas.

    Su `Interactive_imageLabel` no tiene zonas de pedido/entrega. Llamarlas a
    ciegas lanzaria `AttributeError`, el `except` se lo tragaria y el cliente
    dejaria de enviar frames sin decir nada.
    """
    data, _ = _enviado(EtiquetaSinZonas())
    assert 'roi_coordinates' in data
    assert 'order_zone_coordinates' not in data
    assert 'delivery_zone_coordinates' not in data
    assert 'personal_zone_coordinates' not in data
    assert 'personal_zone_activate' not in data


def test_con_etiqueta_completa_si_van_las_zonas():
    data, _ = _enviado(EtiquetaConZonas())
    assert data['order_zone_coordinates'] == [[10, 10], [20, 10], [20, 20], [10, 20]]
    assert data['delivery_zone_coordinates'] == [[30, 10], [40, 10], [40, 20], [30, 20]]


def test_la_zona_del_personal_viaja_cuando_la_etiqueta_la_tiene():
    """La tercera zona del comedor (25-ago-2026).

    Es la que EXCLUYE del conteo de clientes a quien trabaja en el local. Si
    la clave no viaja, el servidor no excluye a nadie y las ventas por
    visitante salen falseadas por los propios empleados.
    """
    data, _ = _enviado(EtiquetaConZonas())
    assert data['personal_zone_coordinates'] == [[50, 10], [60, 10], [60, 20], [50, 20]]
    # El recuadro falso NO tiene `personal_zone_boolean`: el getattr
    # defensivo debe resolverlo a False en vez de romper el envio.
    assert data['personal_zone_activate'] is False


def test_el_interruptor_del_personal_sale_del_recuadro():
    r = RecuadroFalso(EtiquetaConZonas())
    r.personal_zone_boolean = True
    r.loop_show_result()
    assert r.socket.enviado[0][1]['personal_zone_activate'] is True


def test_una_etiqueta_sin_zona_de_personal_sigue_enviando():
    """El mismo fallo mudo de Amazonas, pero para la zona nueva.

    Las etiquetas que solo tienen pedido y entrega no pueden quedarse sin
    enviar frames porque el mixin llame a ciegas a un getter que no existe.
    """
    data, _ = _enviado(EtiquetaSoloPedidoYEntrega())
    assert data['order_zone_coordinates'] == [[10, 10], [20, 10], [20, 20], [10, 20]]
    assert data['delivery_zone_coordinates'] == [[30, 10], [40, 10], [40, 20], [30, 20]]
    assert 'personal_zone_coordinates' not in data
    assert 'personal_zone_activate' not in data


def test_draw_server_es_lo_contrario_de_modo_directo():
    """Si esto se invierte, un cliente que no sabe dibujar se queda sin cajas.

    Amazonas no tiene overlay de Supervision: necesita `draw_server=True`, que
    es ademas el valor por defecto del servidor.
    """
    data, _ = _enviado(EtiquetaSinZonas())
    assert data['draw_server'] is True

    r = RecuadroFalso(EtiquetaConZonas())
    r._direct_mode = True
    r.loop_show_result()
    assert r.socket.enviado[0][1]['draw_server'] is False


def test_el_unpacker_junta_un_mensaje_partido_en_dos_lecturas():
    """El arreglo que a Amazonas le faltaba.

    Su version desempaquetaba el chunk entero de una vez, asi que un frame
    partido entre dos lecturas del pipe se perdia.
    """
    crudo = msgpack.packb(
        {'header': {'x': 1}, 'image_bytes': b'\xff\xd8jpeg'}, use_bin_type=True)
    corte = len(crudo) // 2

    r = RecuadroFalso(EtiquetaConZonas())
    r.process = ProcesoFalso(crudo[:corte])
    r.loop_show_result()
    assert not r.socket.enviado, 'con media trama no debe enviar nada todavia'

    r.process = ProcesoFalso(crudo[corte:])
    r.can_send_next_frame = True
    r.loop_show_result()
    assert r.socket.enviado, ('la segunda mitad debe completar el mensaje: '
                             'para eso el Unpacker es persistente')


def test_sin_proceso_no_hace_nada():
    r = RecuadroFalso(EtiquetaConZonas())
    r.process = None
    r.loop_show_result()
    assert not r.socket.enviado


def test_el_payload_del_mixin_valida_contra_el_contrato():
    """Lo que arma el mixin tiene que pasar el validador del servidor.

    Sin esta prueba, los dobles podian devolver poligonos de un punto y las
    demas seguian en verde mientras el servidor, en `estricto`, habria tirado
    el frame entero. Es el fallo del 13-ago: el cliente deja de contar y no
    da un solo error.
    """
    from elde_core.contracts.eventos import FrameInference

    data, _ = _enviado(EtiquetaConZonas())
    modelo = FrameInference(**data)   # el payload TAL CUAL sale al socket
    assert modelo.personal_zone_coordinates == [[50, 10], [60, 10], [60, 20], [50, 20]]
    assert modelo.order_zone_coordinates == [[10, 10], [20, 10], [20, 20], [10, 20]]
