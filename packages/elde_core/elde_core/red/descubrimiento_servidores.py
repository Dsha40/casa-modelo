"""
Descubrimiento de servidores ELDE en la red local.

Hasta ahora la lista de servidores era un literal en `selector.py`:

    SERVERS = [{"host": "72.68.60.171", ...}, {"host": "72.68.60.141", ...}]

Cada instalacion nueva obligaba a editar codigo, y un servidor que cambiaba de
IP desaparecia del menu sin decir nada. Aqui se averigua QUIEN hay realmente en
la red, por dos vias que se complementan:

  1. BALIZA UDP (rapida, ~0.6 s, y trae metadatos).
     Se manda un datagrama a los broadcast de cada subred; los servidores que
     corren `baliza_descubrimiento` responden con su nombre, puerto, numero de
     clientes conectados y GPU. Es la via buena, pero solo la contestan los
     servidores ya actualizados.

  2. ESCANEO TCP (~2 s, funciona con CUALQUIER servidor).
     Barrido del puerto 9000 sobre las subredes locales y, para cada puerto
     abierto, un GET /health que CONFIRMA que es un ELDE y no otro servicio
     cualquiera escuchando en ese puerto. Sin esa confirmacion el selector
     ofreceria maquinas que no son servidores nuestros.

Las dos se fusionan por `host:puerto`, prevaleciendo la informacion de la
baliza. A eso se le suman las semillas (servidores guardados a mano, p. ej. el
que llega por VPN, que no es escaneable porque su interfaz es /32).

Solo stdlib + `psutil` OPCIONAL para las mascaras de red: este fichero viaja
dentro de los .exe distribuibles.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from ..logging import obtener

# Del registro unificado (`elde.<modulo>`): un logger que no cuelga de `elde`
# se queda sin manejadores y no escribe en ningun sitio.
logger = obtener(__name__)

# Puerto del servidor de inferencia (FastAPI: /ws, /health, /dashboards).
PUERTO_SERVIDOR = 9000
# Puerto del respondedor UDP. NO es el mismo que el de HTTP a proposito: el
# 9000 lo ocupa uvicorn en TCP y mezclar responsabilidades en un puerto obliga
# a tocar el servidor web para algo que no es suyo.
PUERTO_BALIZA = 9010

# Lo que se manda y lo que se espera de vuelta. Versionado desde el dia uno:
# el dia que el formato cambie, un cliente viejo debe poder ignorar lo que no
# entiende en vez de romperse.
SONDA_BALIZA = b"ELDE_DESCUBRIR_v1"
MARCA_BALIZA = "elde-servidor"

# El escaneo se corta en redes grandes. Un /16 son 65.534 sondas: minutos de
# espera y un pico de sockets que el propio Windows corta. Por encima de este
# tamano se escanea solo el /24 alrededor de la IP local.
MAX_HOSTS_POR_RED = 1024


# ---------------------------------------------------------------------------
#  El dato
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServidorELDE:
    """Un servidor de inferencia encontrado (o recordado)."""

    host: str
    puerto: int = PUERTO_SERVIDOR
    nombre: str = ""
    # Identidad estable que anuncia la baliza ('equipo:puerto'). Un servidor
    # con varias tarjetas de red contesta con una IP distinta por cada una;
    # esto es lo que permite saber que son EL MISMO y no listarlo tres veces.
    # Vacio en los descubiertos por escaneo, que no tienen como saberlo.
    ident: str = ""
    # -1 = desconocido. Se distingue de 0 ("esta arriba y sin clientes"), que
    # es informacion util para el operador.
    clientes: int = -1
    gpu: str = ""
    sitio: str = ""
    # 'baliza' | 'escaneo' | 'manual' | 'guardado' | 'semilla'
    origen: str = "escaneo"
    latencia_ms: int = -1
    local: bool = False
    # False = esta en la lista pero ahora mismo no responde (un guardado que
    # esta apagado). Se muestra igual, en gris: que desaparezca de la lista
    # deja al operador sin saber si lo borro alguien o si esta caido.
    en_linea: bool = True

    @property
    def clave(self) -> str:
        return f"{self.host}:{self.puerto}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.puerto}/ws"

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.puerto}"

    @property
    def etiqueta(self) -> str:
        """Texto de una linea para el combo del selector y el dialogo."""
        partes = [self.nombre or self.host]
        if self.nombre and self.nombre != self.host:
            partes.append(f"({self.host})")
        if self.local:
            partes.append("· este equipo")
        if not self.en_linea:
            partes.append("· sin respuesta")
        elif self.clientes >= 0:
            plural = "cliente" if self.clientes == 1 else "clientes"
            partes.append(f"· {self.clientes} {plural}")
        if self.gpu:
            partes.append(f"· {self.gpu}")
        return "  ".join(partes)


# ---------------------------------------------------------------------------
#  Interfaces locales
# ---------------------------------------------------------------------------
def ips_locales() -> list[str]:
    """IPv4 de esta maquina, sin loopback.

    `gethostbyname_ex` se queda corto en equipos multi-tarjeta (aqui hay
    Ethernet, VirtualBox y VPN), asi que se prueba psutil primero.
    """
    encontradas: list[str] = []
    try:
        import psutil  # opcional

        for direcciones in psutil.net_if_addrs().values():
            for d in direcciones:
                if d.family == socket.AF_INET and d.address:
                    encontradas.append(d.address)
    except Exception:  # noqa: BLE001  (psutil ausente o interfaz rara)
        try:
            _, _, ips = socket.gethostbyname_ex(socket.gethostname())
            encontradas.extend(ips)
        except OSError:
            pass
    # Ultimo recurso: la IP con la que saldriamos a la red. No abre conexion
    # (UDP no hace handshake), solo consulta la tabla de rutas.
    if not encontradas:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                encontradas.append(s.getsockname()[0])
        except OSError:
            pass
    return [ip for ip in dict.fromkeys(encontradas)
            if ip and not ip.startswith("127.")]


def es_ip_local(host: str) -> bool:
    """El host apunta a ESTA maquina."""
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    return host in ips_locales()


def redes_locales() -> list[ipaddress.IPv4Network]:
    """Subredes a escanear, deducidas de las interfaces de esta maquina.

    La mascara sale de psutil cuando esta; si no, se asume /24, que es lo que
    usa cualquier LAN de oficina. Las interfaces punto a punto (VPN con
    mascara 255.255.255.255) quedan fuera a proposito: escanear un /32 es
    escanearte a ti mismo, y al otro extremo del tunel no se llega por
    broadcast. Esos servidores se anaden a mano y viven en la lista guardada.
    """
    redes: list[ipaddress.IPv4Network] = []
    mascaras: dict[str, str] = {}
    try:
        import psutil

        for direcciones in psutil.net_if_addrs().values():
            for d in direcciones:
                if d.family == socket.AF_INET and d.address and d.netmask:
                    mascaras[d.address] = d.netmask
    except Exception:  # noqa: BLE001
        pass

    for ip in ips_locales():
        # 169.254.x.x es la auto-configuracion de Windows: la pone una tarjeta
        # DESCONECTADA que no encontro DHCP. En esta maquina hay cinco, y
        # escanearlas serian 1.270 sondas a subredes donde no hay ni puede
        # haber nada.
        if ip.startswith("169.254."):
            continue
        mascara = mascaras.get(ip, "255.255.255.0")
        if mascara == "255.255.255.255":
            continue                      # punto a punto: nada que escanear
        try:
            red = ipaddress.ip_network(f"{ip}/{mascara}", strict=False)
        except ValueError:
            continue
        if not isinstance(red, ipaddress.IPv4Network):
            continue
        if red.num_addresses > MAX_HOSTS_POR_RED:
            # Demasiado grande: nos quedamos con el /24 de esta IP.
            red = ipaddress.ip_network(f"{ip}/24", strict=False)
        if red not in redes:
            redes.append(red)
    return redes


def _broadcasts() -> list[str]:
    """Direcciones a las que mandar la sonda UDP.

    El 255.255.255.255 global no siempre sale por la tarjeta correcta en
    equipos multi-tarjeta como este (Ethernet + VirtualBox + VPN); mandar
    ademas al broadcast de cada subred es lo que hace que el servidor de la
    LAN conteste de verdad.
    """
    destinos = [str(r.broadcast_address) for r in redes_locales()]
    destinos.append("255.255.255.255")
    return list(dict.fromkeys(destinos))


# ---------------------------------------------------------------------------
#  Via 1: baliza UDP
# ---------------------------------------------------------------------------
def _servidor_desde_baliza(datos: dict, host: str,
                           latencia_ms: int) -> ServidorELDE | None:
    if datos.get("elde") != MARCA_BALIZA:
        return None
    puerto = int(datos.get("puerto") or PUERTO_SERVIDOR)
    return ServidorELDE(
        host=str(datos.get("host") or host),
        puerto=puerto,
        nombre=str(datos.get("nombre") or ""),
        ident=str(datos.get("id") or ""),
        clientes=int(datos.get("clientes", -1)),
        gpu=str(datos.get("gpu") or ""),
        sitio=str(datos.get("sitio") or ""),
        origen="baliza",
        latencia_ms=latencia_ms,
        local=es_ip_local(str(datos.get("host") or host)),
        en_linea=True,
    )


def _en_red_local(host: str) -> bool:
    """La direccion cae en una de MIS subredes. Se usa para quedarse con la
    mas util cuando un servidor multi-tarjeta contesta con varias."""
    try:
        direccion = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(direccion in red for red in redes_locales())


def _mejor_direccion(candidatos: list[ServidorELDE]) -> ServidorELDE:
    """De varias vistas del MISMO servidor, una sola con lo mejor de cada una.

    La DIRECCION se elige: la que esta en nuestra propia subred (ruta
    directa) antes que la de una VPN o una red virtual, a la que quiza no
    lleguemos igual de bien. Los DATOS se fusionan, porque cada via sabe
    cosas distintas: la baliza trae nombre y GPU, el escaneo mide la latencia
    real sobre HTTP. Quedarse con una sola vista a secas perderia la mitad de
    lo que se sabe del servidor.
    """
    elegido = next((c for c in candidatos if _en_red_local(c.host)),
                   candidatos[0])
    cambios = {}
    for campo in ("nombre", "gpu", "sitio", "ident"):
        if not getattr(elegido, campo):
            valor = next((getattr(c, campo) for c in candidatos
                          if getattr(c, campo)), "")
            if valor:
                cambios[campo] = valor
    if elegido.clientes < 0:
        otros = [c.clientes for c in candidatos if c.clientes >= 0]
        if otros:
            cambios["clientes"] = otros[0]
    if elegido.latencia_ms < 0:
        otras = [c.latencia_ms for c in candidatos if c.latencia_ms >= 0]
        if otras:
            cambios["latencia_ms"] = min(otras)
    if any(c.en_linea for c in candidatos):
        cambios["en_linea"] = True
    return replace(elegido, **cambios) if cambios else elegido


def preguntar_baliza(espera: float = 0.8,
                     puerto: int = PUERTO_BALIZA) -> list[ServidorELDE]:
    """Grita '¿quien es ELDE?' por UDP y recoge lo que conteste en `espera`.

    Nunca lanza: una red que no deja hacer broadcast (algunas VPN, algunas
    politicas de dominio) debe degradar al escaneo, no tumbar el arranque.
    """
    por_identidad: dict[str, list[ServidorELDE]] = {}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.settimeout(espera)
            for destino in _broadcasts():
                try:
                    s.sendto(SONDA_BALIZA, (destino, puerto))
                except OSError:
                    continue          # esa tarjeta no admite broadcast
            limite = time.monotonic() + espera
            while time.monotonic() < limite:
                s.settimeout(max(0.05, limite - time.monotonic()))
                try:
                    crudo, remitente = s.recvfrom(4096)
                except (socket.timeout, OSError):
                    break
                try:
                    datos = json.loads(crudo.decode("utf-8", "replace"))
                except (ValueError, UnicodeDecodeError):
                    continue
                srv = _servidor_desde_baliza(datos, remitente[0], 0)
                if srv is None:
                    continue
                # Agrupar por identidad anunciada: cada broadcast que sale por
                # una tarjeta distinta trae una respuesta con una IP distinta
                # del MISMO servidor. Sin `id` (baliza vieja) se agrupa por
                # direccion, que es el comportamiento de antes.
                grupo = srv.ident or srv.clave
                por_identidad.setdefault(grupo, [])
                if not any(x.clave == srv.clave
                           for x in por_identidad[grupo]):
                    por_identidad[grupo].append(srv)
    except OSError as exc:
        logger.debug("baliza UDP no disponible: %s", exc)
    return [_mejor_direccion(c) for c in por_identidad.values() if c]


# ---------------------------------------------------------------------------
#  Via 2: escaneo TCP + confirmacion por /health
# ---------------------------------------------------------------------------
def _puerto_abierto(host: str, puerto: int, timeout: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, puerto)) == 0


def consultar(host: str, puerto: int = PUERTO_SERVIDOR,
              timeout: float = 1.5,
              origen: str = "manual") -> ServidorELDE | None:
    """Pregunta a /health y devuelve el servidor si responde como ELDE.

    Devuelve None si no contesta o si contesta algo que no es un ELDE: es lo
    que impide que el selector liste una impresora o un servicio ajeno que
    resulte estar escuchando en el 9000.
    """
    inicio = time.monotonic()
    url = f"http://{host}:{puerto}/health"
    try:
        peticion = urllib.request.Request(url, headers={"User-Agent": "ELDE"})
        with urllib.request.urlopen(peticion, timeout=timeout) as r:
            datos = json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    # La firma de /health del servidor ELDE. 'total_connections' es el campo
    # que ningun otro servicio va a tener por casualidad.
    if not isinstance(datos, dict) or "total_connections" not in datos:
        return None
    return ServidorELDE(
        host=host,
        puerto=puerto,
        # Un servidor anterior al 12-ago-2026 no manda estos dos campos: se
        # queda sin nombre y sin identidad, y se agrupa por direccion como
        # antes. Se sigue descubriendo igual.
        nombre=str(datos.get("servidor_nombre") or ""),
        ident=str(datos.get("servidor_id") or ""),
        clientes=int(datos.get("total_connections", -1)),
        origen=origen,
        latencia_ms=int((time.monotonic() - inicio) * 1000),
        local=es_ip_local(host),
        en_linea=True,
    )


def escanear(puerto: int = PUERTO_SERVIDOR,
             redes: Sequence[ipaddress.IPv4Network] | None = None,
             timeout: float = 0.35,
             hilos: int = 256,
             progreso: Callable[[str, int, int], None] | None = None,
             cancelado: Callable[[], bool] | None = None,
             ) -> list[ServidorELDE]:
    """Barrido del puerto en las subredes locales, con confirmacion.

    Dos fases porque son dos costes muy distintos: el `connect_ex` a 254
    maquinas es barato y paralelo; el GET /health solo se hace sobre los pocos
    que tenian el puerto abierto.

    `progreso(texto, hechos, total)` y `cancelado()` existen para el dialogo
    de la GUI: sin ellos, cerrar la ventana durante una busqueda dejaria el
    hilo vivo barriendo 254 direcciones (el crash 0xc0000409 al salir).
    """
    def aviso(texto: str, hechos: int = 0, total: int = 0) -> None:
        if progreso is not None:
            progreso(texto, hechos, total)

    def abortado() -> bool:
        return cancelado is not None and cancelado()

    redes = list(redes) if redes is not None else redes_locales()
    if not redes:
        return []

    candidatos: list[str] = []
    for red in redes:
        # `hosts()` ya excluye red y broadcast. La IP propia SI entra: el
        # servidor local es un servidor mas y debe aparecer en la lista.
        candidatos.extend(str(ip) for ip in red.hosts())
    candidatos = list(dict.fromkeys(candidatos))
    if not candidatos:
        return []

    total = len(candidatos)
    aviso(f"Explorando {total} direcciones en {len(redes)} subred(es)…",
          0, total)
    abiertos: list[str] = []
    hechos = 0
    with ThreadPoolExecutor(max_workers=min(hilos, total)) as pool:
        futuros = {pool.submit(_puerto_abierto, ip, puerto, timeout): ip
                   for ip in candidatos}
        for f in as_completed(futuros):
            hechos += 1
            if abortado():
                for pendiente in futuros:
                    pendiente.cancel()
                break
            try:
                if f.result():
                    abiertos.append(futuros[f])
            except Exception:  # noqa: BLE001  un socket suelto no corta el barrido
                continue
            if hechos % 32 == 0:
                aviso(f"Explorando la red… ({hechos}/{total})", hechos, total)

    if not abiertos or abortado():
        return []

    aviso(f"Confirmando {len(abiertos)} candidato(s)…", total, total)
    servidores: list[ServidorELDE] = []
    with ThreadPoolExecutor(max_workers=min(32, len(abiertos))) as pool:
        futuros = [pool.submit(consultar, ip, puerto, 1.5, "escaneo")
                   for ip in abiertos]
        for f in as_completed(futuros):
            try:
                srv = f.result()
            except Exception:  # noqa: BLE001
                continue
            if srv is not None:
                servidores.append(srv)
    return servidores


# ---------------------------------------------------------------------------
#  La union de todo
# ---------------------------------------------------------------------------
def ordenar(servidores: Iterable[ServidorELDE]) -> list[ServidorELDE]:
    """Local primero, luego los que responden, luego por IP numerica."""
    def clave(s: ServidorELDE):
        try:
            octetos = tuple(int(p) for p in s.host.split("."))
        except ValueError:
            octetos = (999, 999, 999, 999)   # nombres DNS al final
        return (not s.local, not s.en_linea, octetos, s.puerto)

    return sorted(servidores, key=clave)


def descubrir(puerto: int = PUERTO_SERVIDOR,
              semillas: Sequence[str] = (),
              usar_baliza: bool = True,
              usar_escaneo: bool = True,
              espera_baliza: float = 0.8,
              timeout_escaneo: float = 0.35,
              incluir_caidos: bool = True,
              progreso: Callable[[str, int, int], None] | None = None,
              cancelado: Callable[[], bool] | None = None,
              ) -> list[ServidorELDE]:
    """Lista de servidores ELDE alcanzables, fusionando las tres fuentes.

    `semillas` son 'host' o 'host:puerto' que hay que probar aunque no salgan
    en el escaneo (el servidor por VPN, uno en otra sede detras de un router).
    Con `incluir_caidos` una semilla que no responde se devuelve igualmente
    marcada `en_linea=False`, para que el operador vea que existe y esta
    apagada en vez de creer que la borro alguien.
    """
    def aviso(texto: str, hechos: int = 0, total: int = 0) -> None:
        if progreso is not None:
            progreso(texto, hechos, total)

    def abortado() -> bool:
        return cancelado is not None and cancelado()

    encontrados: dict[str, ServidorELDE] = {}

    if usar_baliza and not abortado():
        aviso("Preguntando por la red…", 0, 0)
        for srv in preguntar_baliza(espera_baliza):
            encontrados[srv.clave] = srv

    if usar_escaneo and not abortado():
        for srv in escanear(puerto, timeout=timeout_escaneo,
                            progreso=progreso, cancelado=cancelado):
            previo = encontrados.get(srv.clave)
            if previo is None:
                encontrados[srv.clave] = srv
            else:
                # La baliza manda (trae nombre y GPU), pero el escaneo aporta
                # la latencia real medida sobre HTTP.
                encontrados[srv.clave] = replace(
                    previo, latencia_ms=srv.latencia_ms)

    for semilla in semillas:
        if abortado():
            break
        aviso(f"Probando {semilla}…", 0, 0)
        host, _, texto_puerto = str(semilla).strip().partition(":")
        host = host.strip()
        if not host:
            continue
        try:
            p = int(texto_puerto) if texto_puerto else puerto
        except ValueError:
            p = puerto
        clave = f"{host}:{p}"
        if clave in encontrados:
            continue
        srv = consultar(host, p, origen="semilla")
        if srv is not None:
            encontrados[clave] = srv
        elif incluir_caidos:
            encontrados[clave] = ServidorELDE(
                host=host, puerto=p, origen="semilla",
                local=es_ip_local(host), en_linea=False)

    return ordenar(_unificar_por_identidad(encontrados.values()))


def _unificar_por_identidad(
        servidores: Iterable[ServidorELDE]) -> list[ServidorELDE]:
    """Un servidor por maquina, no uno por direccion suya.

    Un equipo con varias tarjetas (aqui: Ethernet, red virtual y VPN) tiene
    el puerto 9000 abierto en TODAS, asi que el escaneo lo encuentra una vez
    por subred y la baliza contesta una vez por interfaz. Listarlo tres veces
    obliga al operador a adivinar cual de las tres direcciones es "la buena".
    Con la identidad que anuncian /health y la baliza se sabe que son el
    mismo y se conserva la direccion mas util (la de nuestra propia subred).

    Los que no traen identidad —un servidor anterior a esto— no se agrupan:
    sin dato, agrupar seria inventarselo.
    """
    grupos: dict[str, list[ServidorELDE]] = {}
    sueltos: list[ServidorELDE] = []
    for s in servidores:
        if s.ident:
            grupos.setdefault(s.ident, []).append(s)
        else:
            sueltos.append(s)
    unificados = [_mejor_direccion(c) for c in grupos.values()]
    # Un servidor sin identidad que comparte direccion con uno ya unificado
    # seria un duplicado evidente: fuera.
    claves = {s.clave for s in unificados}
    return unificados + [s for s in sueltos if s.clave not in claves]


def alcanzable(host: str, puerto: int = PUERTO_SERVIDOR,
               timeout: float = 1.0) -> bool:
    """El servidor esta escuchando? Para el local se prueba tambien por
    loopback: desde la propia maquina es mas fiable que su IP de tarjeta."""
    if es_ip_local(host) and _puerto_abierto("127.0.0.1", puerto, timeout):
        return True
    return _puerto_abierto(host, puerto, timeout)


def ws_url(host: str, puerto: int = PUERTO_SERVIDOR) -> str:
    return f"ws://{host}:{puerto}/ws"


def partes_de_ws_url(url: str) -> tuple[str, int]:
    """'ws://1.2.3.4:9000/ws' -> ('1.2.3.4', 9000). Tolerante: lo que no se
    entienda cae al puerto por defecto en vez de reventar el arranque."""
    texto = (url or "").strip()
    for prefijo in ("ws://", "wss://", "http://", "https://"):
        if texto.lower().startswith(prefijo):
            texto = texto[len(prefijo):]
            break
    texto = texto.split("/", 1)[0]
    host, _, texto_puerto = texto.partition(":")
    try:
        return host, int(texto_puerto) if texto_puerto else PUERTO_SERVIDOR
    except ValueError:
        return host, PUERTO_SERVIDOR
