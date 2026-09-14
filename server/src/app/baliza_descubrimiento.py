"""
Baliza de descubrimiento: el servidor contesta cuando le preguntan quien es.

Los clientes ELDE ya no llevan la IP del servidor escrita en el codigo: la
buscan en la red (`elde_core.red.descubrimiento_servidores`). Ese buscador
tiene dos vias y esta es la buena:

    cliente  --UDP broadcast-->  "ELDE_DESCUBRIR_v1"
    servidor --UDP unicast--->   {"elde":"elde-servidor","nombre":...}

Responde en ~1 ms y de paso se presenta: nombre del equipo, puerto real,
cuantos clientes tiene conectados ahora mismo y que GPU usa. La otra via —el
escaneo del puerto 9000 en toda la subred— sigue existiendo como respaldo para
servidores que aun no tengan este modulo, pero tarda segundos y no puede
averiguar nada de eso.

Es un hilo daemon con un socket UDP. No toca el bucle de uvicorn, no comparte
estado con el websocket (solo LEE un contador) y si el puerto esta ocupado se
queda callado: un servidor que no puede anunciarse debe seguir sirviendo
frames igual, simplemente lo encontraran por escaneo.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from typing import Callable, Optional

from elde_core.red.descubrimiento_servidores import (
    MARCA_BALIZA, PUERTO_BALIZA, PUERTO_SERVIDOR, SONDA_BALIZA)

logger = logging.getLogger(__name__)

# El nombre de la GPU no cambia en caliente, pero averiguarlo cuesta un
# `import torch` y una consulta al driver CUDA: entre 5 y 10 segundos la
# primera vez si torch no esta ya cargado. Hacerlo DENTRO del manejador
# retrasaba la primera respuesta muchisimo mas que el timeout del cliente
# (0.8 s), asi que la baliza parecia muerta justo en el primer arranque, que
# es cuando mas se la necesita. Se resuelve en un hilo aparte al iniciar y
# hasta entonces se responde sin ese dato.
_gpu_cacheada: Optional[str] = None


def _resolver_gpu() -> str:
    global _gpu_cacheada
    if _gpu_cacheada is not None:
        return _gpu_cacheada
    nombre = ""
    try:
        import torch

        if torch.cuda.is_available():
            nombre = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001  sin torch, sin CUDA o driver a medias
        nombre = ""
    _gpu_cacheada = nombre
    return nombre


def _gpu_conocida() -> str:
    """Lo que se sepa AHORA, sin bloquear. '' mientras se esta resolviendo."""
    return _gpu_cacheada or ""


def nombre_servidor() -> str:
    """Como se llama este servidor para los clientes."""
    return (os.getenv("ELDE_SERVIDOR_NOMBRE", "").strip()
            or socket.gethostname())


def identidad(puerto_servidor: int = PUERTO_SERVIDOR) -> str:
    """Identidad ESTABLE del servidor, la misma por la baliza y por /health.

    Es lo que permite al cliente saber que la respuesta que llego por la
    tarjeta Ethernet y la que llego por la VPN son el MISMO servidor. Vive
    aqui, y no duplicada en `app.py`, para que las dos vias no puedan
    divergir: si divergieran, el cliente listaria el mismo servidor dos veces
    justo cuando las dos vias funcionan.
    """
    return f"{socket.gethostname()}:{int(puerto_servidor)}"


def _ip_hacia(destino: str) -> str:
    """IP local por la que se llega a `destino`.

    Importa en esta maquina: tiene Ethernet (72.68.60.171), VirtualBox
    (192.168.56.1) y VPN (10.8.0.32). Contestar con la IP equivocada le daria
    al cliente una direccion a la que no puede conectarse. UDP no hace
    handshake, asi que `connect` aqui solo consulta la tabla de rutas.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((destino, 9))
            return s.getsockname()[0]
    except OSError:
        return ""


class BalizaDescubrimiento:
    """Respondedor UDP. `iniciar()` y `detener()` son idempotentes."""

    def __init__(self,
                 puerto_servidor: int = PUERTO_SERVIDOR,
                 puerto_baliza: int = PUERTO_BALIZA,
                 nombre: str = "",
                 sitio: str = "",
                 contar_clientes: Optional[Callable[[], int]] = None) -> None:
        self.puerto_servidor = int(puerto_servidor)
        self.puerto_baliza = int(puerto_baliza)
        self.nombre = nombre or nombre_servidor()
        # SITE_ID (mayusculas) primero: en Linux las variables distinguen
        # mayusculas y `site_id` solo vale si se escribe exactamente asi.
        self.sitio = sitio or (os.getenv("SITE_ID") or os.getenv("site_id") or "").strip()
        self._contar = contar_clientes
        self._sock: Optional[socket.socket] = None
        self._hilo: Optional[threading.Thread] = None
        self._parar = threading.Event()

    # -- ciclo de vida ----------------------------------------------------
    def iniciar(self) -> bool:
        if self._hilo is not None and self._hilo.is_alive():
            return True
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Sin SO_REUSEADDR a proposito: en Windows permitiria que DOS
            # servidores en la misma maquina se apropiaran del puerto y
            # contestaran a medias. Que el segundo falle al enlazar y se
            # quede mudo es el comportamiento correcto.
            sock.bind(("0.0.0.0", self.puerto_baliza))
            sock.settimeout(0.5)          # para que detener() responda pronto
        except OSError as exc:
            logger.warning(
                "baliza de descubrimiento no disponible en UDP %d (%s); los "
                "clientes tendran que encontrar este servidor por escaneo",
                self.puerto_baliza, exc)
            return False
        self._sock = sock
        self._parar.clear()
        # El nombre de la GPU, fuera del camino critico (ver _resolver_gpu).
        threading.Thread(target=_resolver_gpu, name="elde-baliza-gpu",
                         daemon=True).start()
        self._hilo = threading.Thread(
            target=self._servir, name="elde-baliza", daemon=True)
        self._hilo.start()
        logger.info("baliza de descubrimiento escuchando en UDP %d "
                    "(anuncia el puerto %d como '%s')",
                    self.puerto_baliza, self.puerto_servidor, self.nombre)
        return True

    def detener(self) -> None:
        self._parar.set()
        hilo, sock = self._hilo, self._sock
        self._hilo, self._sock = None, None
        if hilo is not None:
            hilo.join(timeout=2.0)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # -- interior ---------------------------------------------------------
    def _clientes(self) -> int:
        if self._contar is None:
            return -1
        try:
            return int(self._contar())
        except Exception:  # noqa: BLE001  nunca por un contador
            return -1

    def _respuesta(self, destino: str) -> bytes:
        datos = {
            "elde": MARCA_BALIZA,
            "version": 1,
            # Identidad ESTABLE del servidor, independiente de por que
            # tarjeta le llegue la pregunta. Sin esto, un servidor con varias
            # interfaces (aqui: Ethernet + VirtualBox + VPN) contesta a cada
            # broadcast con una IP distinta y el cliente lo lista tres veces
            # como si fueran tres servidores.
            "id": identidad(self.puerto_servidor),
            "nombre": self.nombre,
            "host": _ip_hacia(destino),
            "puerto": self.puerto_servidor,
            "clientes": self._clientes(),
            "gpu": _gpu_conocida(),
            "sitio": self.sitio,
        }
        return json.dumps(datos).encode("utf-8")

    def _servir(self) -> None:
        sock = self._sock
        while not self._parar.is_set() and sock is not None:
            try:
                crudo, remitente = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break                     # socket cerrado por detener()
            if not crudo.startswith(SONDA_BALIZA):
                continue                  # trafico ajeno en el puerto
            try:
                sock.sendto(self._respuesta(remitente[0]), remitente)
            except OSError as exc:
                logger.debug("no se pudo responder a %s: %s", remitente, exc)


# Instancia unica del proceso: el servidor solo tiene una identidad que
# anunciar y solo puede enlazar el puerto una vez.
_baliza: Optional[BalizaDescubrimiento] = None


def iniciar(puerto_servidor: int = PUERTO_SERVIDOR,
            contar_clientes: Optional[Callable[[], int]] = None) -> bool:
    """Arranca la baliza del proceso. Se puede desactivar con
    ELDE_BALIZA=0 (redes donde el broadcast este prohibido por politica)."""
    global _baliza
    if (os.getenv("ELDE_BALIZA", "1").strip().lower()
            in ("0", "no", "false", "off")):
        logger.info("baliza de descubrimiento desactivada por ELDE_BALIZA")
        return False
    if _baliza is None:
        _baliza = BalizaDescubrimiento(
            puerto_servidor=puerto_servidor, contar_clientes=contar_clientes)
    return _baliza.iniciar()


def detener() -> None:
    global _baliza
    if _baliza is not None:
        _baliza.detener()
        _baliza = None
