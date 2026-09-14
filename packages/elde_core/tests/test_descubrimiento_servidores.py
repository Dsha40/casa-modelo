"""
Pruebas del descubrimiento de servidores ELDE (12-ago-2026).

Cubren lo que de verdad puede salir mal cuando el cliente ya no lleva la IP
del servidor escrita en el codigo:

  - Un servicio cualquiera escuchando en el 9000 NO es un servidor ELDE. Sin
    esa comprobacion, el selector ofreceria maquinas ajenas y el operador se
    conectaria a la nada.
  - Un servidor con VARIAS tarjetas de red contesta con una direccion por
    interfaz. Aparecer tres veces en la lista obliga a adivinar cual es la
    buena, asi que se unifican por la identidad que anuncia.
  - Las tarjetas desconectadas (169.254.x.x) y las punto a punto (/32 de una
    VPN) no se escanean: son 1.270 sondas a subredes donde no hay nada.
  - La baliza habla de verdad con el respondedor del servidor.

Sin GUI. La unica red que se usa es el bucle local.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402

from elde_core.red import descubrimiento_servidores as D  # noqa: E402


# ---------------------------------------------------------------------------
#  Dobles: un /health que responde como ELDE y otro que no
# ---------------------------------------------------------------------------
def _servidor_http(cuerpo: dict | None, ruta_ok: str = "/health"):
    """Levanta un HTTP de mentira en un puerto libre. Devuelve (puerto, parar)."""

    class Manejador(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802  (API de http.server)
            if cuerpo is None or self.path != ruta_ok:
                self.send_response(404)
                self.end_headers()
                return
            crudo = json.dumps(cuerpo).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(crudo)))
            self.end_headers()
            self.wfile.write(crudo)

        def log_message(self, *_):      # silencio en la salida de pytest
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Manejador)
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()
    return httpd.server_port, httpd.shutdown


HEALTH_ELDE = {
    "status": "active",
    "total_connections": 2,
    "servidor_id": "EQUIPO-A:9000",
    "servidor_nombre": "EQUIPO-A",
}


# ---------------------------------------------------------------------------
#  consultar(): confirmar que al otro lado hay un ELDE
# ---------------------------------------------------------------------------
def test_consultar_reconoce_un_servidor_elde():
    puerto, parar = _servidor_http(HEALTH_ELDE)
    try:
        srv = D.consultar("127.0.0.1", puerto, timeout=3.0)
    finally:
        parar()
    assert srv is not None
    assert srv.clientes == 2
    assert srv.nombre == "EQUIPO-A"
    assert srv.ident == "EQUIPO-A:9000"
    assert srv.local is True                 # 127.0.0.1 es esta maquina
    assert srv.ws_url == f"ws://127.0.0.1:{puerto}/ws"


def test_consultar_descarta_un_servicio_ajeno():
    """Un HTTP cualquiera en el puerto no puede colarse como servidor ELDE.

    Es el caso real que justifica confirmar por /health en vez de fiarse de
    que el puerto este abierto: en una LAN de oficina el 9000 lo usa mas de
    un programa."""
    puerto, parar = _servidor_http({"status": "ok", "app": "otra cosa"})
    try:
        assert D.consultar("127.0.0.1", puerto, timeout=3.0) is None
    finally:
        parar()


def test_consultar_sin_nadie_al_otro_lado():
    # Puerto cerrado: no debe lanzar, solo devolver None.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        puerto = s.getsockname()[1]
    assert D.consultar("127.0.0.1", puerto, timeout=1.0) is None


def test_consultar_tolera_un_servidor_viejo():
    """Un servidor anterior a esto no manda `servidor_id` ni nombre: se
    descubre igual, solo que sin agrupar por identidad."""
    puerto, parar = _servidor_http({"status": "active", "total_connections": 0})
    try:
        srv = D.consultar("127.0.0.1", puerto, timeout=3.0)
    finally:
        parar()
    assert srv is not None
    assert srv.ident == ""
    assert srv.nombre == ""


# ---------------------------------------------------------------------------
#  Un servidor con varias tarjetas es UN servidor
# ---------------------------------------------------------------------------
def test_se_unifican_las_direcciones_del_mismo_servidor():
    vistas = [
        D.ServidorELDE(host="10.8.0.5", puerto=9000, ident="EQ:9000",
                       origen="baliza", nombre="EQ", gpu="RTX 5060 Ti"),
        D.ServidorELDE(host="192.168.1.10", puerto=9000, ident="EQ:9000",
                       origen="escaneo", clientes=4, latencia_ms=12),
    ]
    unico = D._mejor_direccion(vistas)
    # Los datos de las dos vistas se conservan: la baliza sabe el nombre y la
    # GPU, el escaneo midio la latencia y vio los clientes.
    assert unico.nombre == "EQ"
    assert unico.gpu == "RTX 5060 Ti"
    assert unico.clientes == 4
    assert unico.latencia_ms == 12


def test_los_sin_identidad_no_se_agrupan():
    """Agrupar sin dato seria inventarselo: dos servidores viejos distintos
    acabarian fundidos en uno y el operador perderia uno de los dos."""
    sueltos = [
        D.ServidorELDE(host="10.0.0.1", ident=""),
        D.ServidorELDE(host="10.0.0.2", ident=""),
    ]
    assert len(D._unificar_por_identidad(sueltos)) == 2


def test_un_suelto_que_repite_direccion_de_uno_unificado_se_descarta():
    entradas = [
        D.ServidorELDE(host="10.0.0.1", ident="EQ:9000", origen="baliza"),
        D.ServidorELDE(host="10.0.0.1", ident="", origen="escaneo"),
    ]
    assert len(D._unificar_por_identidad(entradas)) == 1


# ---------------------------------------------------------------------------
#  Que subredes se escanean
# ---------------------------------------------------------------------------
def test_no_se_escanean_las_tarjetas_desconectadas(monkeypatch):
    """169.254.x.x es la autoconfiguracion de una tarjeta SIN DHCP. En esta
    maquina hay cinco: escanearlas son 1.270 sondas a la nada."""
    monkeypatch.setattr(D, "ips_locales",
                        lambda: ["192.168.1.10", "169.254.5.5",
                                 "169.254.99.1"])
    redes = [str(r) for r in D.redes_locales()]
    assert redes == ["192.168.1.0/24"]


def test_una_red_enorme_se_recorta_al_24(monkeypatch):
    """Un /16 son 65.534 sondas: minutos de espera y un pico de sockets que
    Windows corta. Se escanea el /24 de nuestra propia IP."""
    monkeypatch.setattr(D, "ips_locales", lambda: ["10.1.2.3"])
    monkeypatch.setattr("psutil.net_if_addrs", lambda: {})
    redes = D.redes_locales()
    assert all(r.num_addresses <= D.MAX_HOSTS_POR_RED for r in redes)


def test_escanear_sin_redes_no_lanza():
    assert D.escanear(redes=[]) == []


# ---------------------------------------------------------------------------
#  Utilidades de URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entrada,esperado", [
    ("ws://10.0.0.5:9000/ws", ("10.0.0.5", 9000)),
    ("wss://servidor.local:8443/ws", ("servidor.local", 8443)),
    ("http://10.0.0.5:9000/health", ("10.0.0.5", 9000)),
    ("10.0.0.5", ("10.0.0.5", 9000)),
    ("10.0.0.5:1234", ("10.0.0.5", 1234)),
    ("10.0.0.5:puerto-malo", ("10.0.0.5", 9000)),   # tolerante, no revienta
])
def test_partes_de_ws_url(entrada, esperado):
    assert D.partes_de_ws_url(entrada) == esperado


def test_orden_local_primero_y_caidos_al_final():
    lista = [
        D.ServidorELDE(host="10.0.0.9", en_linea=False),
        D.ServidorELDE(host="10.0.0.2"),
        D.ServidorELDE(host="10.0.0.7", local=True),
    ]
    assert [s.host for s in D.ordenar(lista)] == ["10.0.0.7", "10.0.0.2",
                                                  "10.0.0.9"]


def test_etiqueta_dice_lo_que_pasa():
    caido = D.ServidorELDE(host="10.0.0.9", en_linea=False)
    assert "sin respuesta" in caido.etiqueta
    vivo = D.ServidorELDE(host="10.0.0.2", nombre="EQ", clientes=1,
                          local=True)
    assert "este equipo" in vivo.etiqueta
    assert "1 cliente" in vivo.etiqueta      # singular, no "1 clientes"


# ---------------------------------------------------------------------------
#  La baliza, de punta a punta
# ---------------------------------------------------------------------------
def test_baliza_responde_a_la_sonda():
    """El respondedor del servidor y el buscador del cliente se entienden.

    Vive en `server/`, asi que se importa por ruta: es la unica prueba del
    nucleo que mira al servidor, y lo hace porque el protocolo es de los dos.
    """
    servidor = Path(__file__).resolve().parents[3] / "server"
    if not (servidor / "src" / "app" / "baliza_descubrimiento.py").exists():
        pytest.skip("el servidor no esta en este arbol")
    sys.path.insert(0, str(servidor))
    from src.app.baliza_descubrimiento import BalizaDescubrimiento

    baliza = BalizaDescubrimiento(puerto_servidor=9123, nombre="EQUIPO-PRUEBA",
                                  contar_clientes=lambda: 7)
    if not baliza.iniciar():
        pytest.skip("el puerto de la baliza esta ocupado en esta maquina")
    try:
        # Directo al bucle local: el broadcast depende del cortafuegos y de
        # que la maquina tenga red, y eso no es lo que se esta probando.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(3.0)
            s.sendto(D.SONDA_BALIZA, ("127.0.0.1", D.PUERTO_BALIZA))
            crudo, _ = s.recvfrom(4096)
        datos = json.loads(crudo)
    finally:
        baliza.detener()

    assert datos["elde"] == D.MARCA_BALIZA
    assert datos["nombre"] == "EQUIPO-PRUEBA"
    assert datos["puerto"] == 9123
    assert datos["clientes"] == 7
    assert datos["id"].endswith(":9123")


def test_baliza_ignora_trafico_ajeno():
    servidor = Path(__file__).resolve().parents[3] / "server"
    if not (servidor / "src" / "app" / "baliza_descubrimiento.py").exists():
        pytest.skip("el servidor no esta en este arbol")
    sys.path.insert(0, str(servidor))
    from src.app.baliza_descubrimiento import BalizaDescubrimiento

    baliza = BalizaDescubrimiento(puerto_servidor=9124)
    if not baliza.iniciar():
        pytest.skip("el puerto de la baliza esta ocupado en esta maquina")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(1.0)
            s.sendto(b"hola que tal", ("127.0.0.1", D.PUERTO_BALIZA))
            with pytest.raises((socket.timeout, TimeoutError)):
                s.recvfrom(4096)
    finally:
        baliza.detener()
