"""Modo F: el envío de cada imagen se empuja al socket en el acto
(6-oct-2026; explicación, cómo apagarlo y cómo llevarlo a otro proyecto en
la cabecera de `elde_core/transport/socket_client.py`).

`QWebSocket.sendBinaryMessage` deja el mensaje en el búfer y sale cuando el
hilo de Qt vuelve a su bucle; el cliente pinta justo después de enviar, así
que la imagen esperaba ese pintado (~18 ms) antes de salir. Se fija:

- que tras cada `sendBinaryMessage` hay un `flush()`;
- que está ENCENDIDO por defecto y se apaga con `ELDE_MODO_F=0` (o
  false / no / off), por si un día da un problema que hoy no conocemos;
- que apagado, el envío es el de antes (sin `flush()`).
"""
from elde_core.transport import socket_client as sc
from elde_core.transport.socket_client import Socket_services


class _QWebSocketFalso:
    def __init__(self):
        self.llamadas = []

    def sendBinaryMessage(self, datos):          # noqa: N802 (API de Qt)
        self.llamadas.append('send')

    def flush(self):
        self.llamadas.append('flush')
        return True


def _cliente(modo_f=None):
    c = Socket_services.__new__(Socket_services)   # sin abrir conexión real
    c.client = _QWebSocketFalso()
    c.id_connection = 1
    c.type_inference = 'Anaqueles'
    if modo_f is not None:
        c.modo_f = modo_f
    return c


def test_cada_imagen_se_empuja_al_socket():
    c = _cliente(modo_f=True)
    c.send_binary_frame('k', {'image': b'x' * 10})
    c.send_binary_frame('k', {'image': b'y' * 10})
    assert c.client.llamadas == ['send', 'flush', 'send', 'flush']


def test_apagado_envia_como_antes():
    c = _cliente(modo_f=False)
    c.send_binary_frame('k', {'image': b'x' * 10})
    assert c.client.llamadas == ['send']


def test_sin_estado_sigue_encendido():
    """Una conexión creada sin __init__ (sin `modo_f`): el defecto, encendido."""
    c = _cliente()
    c.send_binary_frame('k', {'image': b'x' * 10})
    assert c.client.llamadas == ['send', 'flush']


def test_encendido_por_defecto_y_se_apaga_con_la_variable(monkeypatch):
    monkeypatch.delenv(sc.VARIABLE_MODO_F, raising=False)
    assert sc.modo_f_activado() is True
    for apagado in ('0', 'false', 'no', 'off', 'OFF', ' 0 '):
        monkeypatch.setenv(sc.VARIABLE_MODO_F, apagado)
        assert sc.modo_f_activado() is False, apagado
    for encendido in ('1', 'true', 'si', ''):
        monkeypatch.setenv(sc.VARIABLE_MODO_F, encendido)
        assert sc.modo_f_activado() is True, encendido


def test_la_conexion_lee_el_interruptor_al_crearse(monkeypatch):
    monkeypatch.setenv(sc.VARIABLE_MODO_F, '0')
    assert Socket_services().modo_f is False
    monkeypatch.delenv(sc.VARIABLE_MODO_F, raising=False)
    assert Socket_services().modo_f is True
