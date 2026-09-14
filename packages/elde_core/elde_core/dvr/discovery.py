"""
src/core/dvr/discovery.py
Descubrimiento de dispositivos (DVR / NVR / cámaras IP) en la red local.

Tres sondas complementarias, ejecutadas en paralelo y fusionadas por IP:

  1. SADP      — protocolo propietario de Hikvision (UDP multicast
                 239.255.255.250:37020). Es el que usa la herramienta SADP
                 oficial: devuelve modelo, serie, MAC, firmware y puertos
                 SIN credenciales. Es la vía más rica para Hikvision.
  2. WS-Discovery (ONVIF) — UDP multicast 239.255.255.250:3702. Estándar de
                 la industria: encuentra cámaras/NVR de CUALQUIER marca que
                 hable ONVIF (Dahua, Hikvision, genéricas...).
  3. Barrido TCP — conecta a los puertos típicos (80/8000/554/37777...) de
                 cada IP del rango. Cubre equipos con multicast bloqueado
                 (VLAN/switch) y confirma qué servicios están abiertos.

Sin dependencias externas: solo sockets de la stdlib. Nada bloquea la GUI si
se llama desde un hilo (ver DiscoveryWorker en workers/).
"""
from __future__ import annotations

import re
import socket
import struct
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Multicast estándar SSDP/WS-Discovery (Hikvision reutiliza el grupo con su
# propio puerto 37020).
_MCAST_GRP = "239.255.255.250"
_SADP_PORT = 37020
_WSD_PORT = 3702

# Puertos que delatan un equipo de videovigilancia. Lista AMPLIA a propósito:
# un DVR puede estar en cualquier puerto (el instalador los cambia a gusto), así
# que se cubren los conocidos de todas las marcas. No es exhaustiva —para un
# puerto raro, el operador puede pasar su propia lista a `barrer_rango`—, pero
# barrer los 65535 en todo un /24 sería lentísimo y, en una red con eco de
# middlebox (ver _es_middlebox), inútil.
PUERTOS_SONDA: dict[int, str] = {
    # HTTP y variantes de administración web
    80:    "HTTP",
    81:    "HTTP alt",
    82:    "HTTP alt",
    83:    "HTTP alt",
    88:    "HTTP alt",
    8000:  "Hikvision SDK",     # servicio Hikvision (también web en muchos)
    8001:  "HTTP alt",
    8080:  "HTTP alt",
    8081:  "HTTP alt",
    8082:  "HTTP alt",          # un puerto por equipo: 8080/8081/8082… (medido)
    8083:  "HTTP alt",
    8084:  "HTTP alt",
    8085:  "HTTP alt",
    8090:  "HTTP alt",
    7001:  "HTTP alt",
    9000:  "HTTP alt",
    # HTTPS
    443:   "HTTPS",
    8443:  "HTTPS alt",
    # RTSP (stream directo)
    554:   "RTSP",
    555:   "RTSP alt",
    8554:  "RTSP alt",
    10554: "RTSP alt",
    # SDK / protocolos de fabricante
    37777: "Dahua SDK",
    37778: "Dahua SDK",
    34567: "XiongMai/DVR",      # DVR genéricos chinos (NetSurveillance)
    34599: "XiongMai/DVR",
    # ONVIF y descubrimiento sobre TCP
    2020:  "ONVIF",
    8899:  "ONVIF alt",
    5000:  "ONVIF alt",
}

# Puerto SEÑUELO: nada legítimo escucha aquí. Si un host lo "abre", no es un
# equipo: es un firewall/proxy/router (a menudo del ISP, o un middlebox de la
# red) que contesta el SYN de CUALQUIER puerto por cortesía. Sin este filtro,
# en la red 72.68.60.x aparecían 150 "equipos" idénticos, ninguno real
# (medido el 8-ago-2026). Se elige un puerto alto improbable.
_PUERTO_SENUELO = 47123

# Puertos donde probar el reto ISAPI para CONFIRMAR un Hikvision cuando el
# barrido TCP no es fiable (red con eco) o cuando el equipo está en un puerto
# no estándar. El instalador cambia el puerto web a gusto —los 3 equipos de
# Richard estaban en 8080/8081/8082, uno por aparato—, así que la lista es
# amplia y ordenada por probabilidad. La búsqueda para en el primer acierto.
_PUERTOS_WEB_ISAPI = [
    80, 8080, 8081, 8082, 8083, 8084, 8085, 8000, 8001, 8090,
    81, 82, 83, 84, 85, 88, 7001, 9000, 8899, 443, 8443,
]

# Prefijos MAC (OUI) conocidos -> marca. Solo para etiquetar, no es
# exhaustivo: si no coincide se deja la marca que reporte el protocolo.
_OUI_MARCAS: dict[str, str] = {
    "44:19:b6": "Hikvision", "c0:56:e3": "Hikvision", "bc:ad:28": "Hikvision",
    "4c:bd:8f": "Hikvision", "58:03:fb": "Hikvision", "a4:14:37": "Hikvision",
    "e0:ca:3c": "Hikvision", "18:68:cb": "Hikvision", "c4:2f:90": "Hikvision",
    "3c:ef:8c": "Dahua",     "90:02:a9": "Dahua",     "e0:50:8b": "Dahua",
    "bc:32:5f": "Dahua",     "14:a7:8b": "Dahua",
}


@dataclass
class DispositivoRed:
    """Un equipo encontrado en la red. Los campos vacíos son 'no reportado'."""
    ip: str
    marca: str = ""
    modelo: str = ""
    tipo: str = ""              # "camara" | "dvr" | "" (desconocido)
    nombre: str = ""
    mac: str = ""
    serie: str = ""
    firmware: str = ""
    puerto_http: int = 0
    puerto_sdk: int = 0
    puertos_abiertos: list[int] = field(default_factory=list)
    origen: set[str] = field(default_factory=set)   # {"SADP", "ONVIF", "TCP"}

    @property
    def descripcion(self) -> str:
        etiqueta = {"camara": "Cámara IP", "dvr": "DVR/NVR"}.get(self.tipo, "")
        partes = [p for p in (self.marca, self.modelo, etiqueta) if p]
        return " ".join(partes) or "Dispositivo desconocido"

    @property
    def marca_ui(self) -> str:
        """Marca tal como la espera el ComboBox del formulario DVR."""
        m = (self.marca or "").lower()
        if "hikvision" in m or "hiwatch" in m or "hilook" in m:
            return "Hikvision HTTP"
        if "dahua" in m or "amcrest" in m or "lorex" in m:
            return "Dahua HTTP"
        # Sin marca clara: deducir por los puertos abiertos.
        if 37777 in self.puertos_abiertos:
            return "Dahua HTTP"
        if 8000 in self.puertos_abiertos:
            return "Hikvision HTTP"
        return "Hikvision HTTP"

    @property
    def es_equipo_video(self) -> bool:
        """¿Es (casi seguro) un DVR/NVR/cámara y no un PC o router?

        Lo es si respondió a SADP/ONVIF (protocolos de videovigilancia) o si
        expone un puerto inequívoco: RTSP (554) o un SDK de fabricante. Un
        equipo con solo 80/443/8080 puede ser cualquier cosa, así que no se
        da por bueno.

        Y al revés también: un "equipo" con TODOS los puertos sonda abiertos
        —Hikvision SDK y Dahua SDK a la vez— no es un equipo, es un firewall
        o proxy del ISP contestando SYN por cortesía. 150 de esos aparecieron
        en el barrido del 8-ago-2026; ninguno era real.
        """
        if self.origen & {"SADP", "ONVIF", "ISAPI"}:
            return True                     # protocolo de vídeo confirmado
        abiertos = set(self.puertos_abiertos)
        if {8000, 37777} <= abiertos:
            return False                    # nadie habla ambos SDK a la vez
        return bool({554, 8000, 37777} & abiertos)

    @property
    def puerto_ui(self) -> int:
        if self.puerto_http:
            return self.puerto_http
        for p in (80, 8080, 443):
            if p in self.puertos_abiertos:
                return p
        return 80


# ─────────────────────────────────────────────────────────────────────
# Utilidades de red
# ─────────────────────────────────────────────────────────────────────

def ips_locales() -> list[str]:
    """IPv4 de las interfaces activas (sin loopback). Se usan para enviar el
    multicast por CADA interfaz: con varias NIC (Wi-Fi + Ethernet) el sistema
    elige una sola y el descubrimiento fallaría en la otra red."""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    if not ips:
        # Truco del socket UDP: no envía nada, solo resuelve la ruta de salida.
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return sorted(ips)


def rango_de_ip(ip: str) -> str:
    """'192.168.1.37' -> '192.168.1' (asume /24, el caso normal en LAN)."""
    partes = ip.split(".")
    return ".".join(partes[:3]) if len(partes) == 4 else ""


def es_ip_privada(ip: str) -> bool:
    """¿Es una IP de red local (RFC1918)? 10/8, 172.16/12, 192.168/16.

    El barrido TCP SOLO debe recorrer rangos privados. En una PC con interfaz
    pública (IP del ISP), barrer ese /24 recorre direcciones ajenas y, peor,
    muchos routers/ISP responden SYN en todos los puertos: cada IP parece un
    equipo y el resultado es 150 falsos positivos (medido el 8-ago-2026 en una
    PC con 72.68.60.x pública). Solo la LAN del cliente tiene sus DVR.
    """
    try:
        a, b, *_ = (int(x) for x in ip.split("."))
    except (ValueError, TypeError):
        return False
    if a == 10:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 192 and b == 168:
        return True
    return False


def _marca_por_mac(mac: str) -> str:
    if not mac:
        return ""
    clave = mac.lower().replace("-", ":")[:8]
    return _OUI_MARCAS.get(clave, "")


# ─────────────────────────────────────────────────────────────────────
# 1) SADP — Hikvision
# ─────────────────────────────────────────────────────────────────────

_SADP_INQUIRY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    "<Probe><Uuid>{uuid}</Uuid><Types>inquiry</Types></Probe>"
)


def _texto(nodo, *nombres: str) -> str:
    """Primer hijo que exista con alguno de esos nombres (case-insensitive)."""
    for hijo in nodo.iter():
        etiqueta = hijo.tag.split("}")[-1].lower()
        for n in nombres:
            if etiqueta == n.lower() and hijo.text:
                return hijo.text.strip()
    return ""


def sondear_sadp(timeout: float = 4.0) -> list[DispositivoRed]:
    """Descubre equipos Hikvision por SADP. Devuelve [] si nada responde."""
    encontrados: dict[str, DispositivoRed] = {}
    lock = threading.Lock()

    def _en_interfaz(ip_local: str) -> None:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind((ip_local, 0))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                            socket.inet_aton(ip_local))
            sock.settimeout(0.5)
            mensaje = _SADP_INQUIRY.format(uuid=str(uuid.uuid4()).upper()).encode()
            for destino in ((_MCAST_GRP, _SADP_PORT),
                            ("255.255.255.255", _SADP_PORT)):
                try:
                    sock.sendto(mensaje, destino)
                except Exception:
                    pass
            fin = time.time() + timeout
            while time.time() < fin:
                try:
                    datos, origen = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                except Exception:
                    break
                d = _parsear_sadp(datos, origen[0])
                if d:
                    with lock:
                        if d.ip not in encontrados:
                            encontrados[d.ip] = d
        except Exception:
            pass
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    hilos = [threading.Thread(target=_en_interfaz, args=(ip,), daemon=True)
             for ip in ips_locales()]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout + 2)
    return list(encontrados.values())


def _parsear_sadp(datos: bytes, ip_origen: str) -> DispositivoRed | None:
    try:
        texto = datos.decode("utf-8", errors="ignore")
        if "<" not in texto:
            return None
        raiz = ET.fromstring(texto[texto.index("<"):])
    except Exception:
        return None
    ip = _texto(raiz, "IPv4Address", "IPv4address") or ip_origen
    if not ip:
        return None
    mac = _texto(raiz, "MAC", "MACAddress")
    marca = _marca_por_mac(mac) or "Hikvision"
    def _int(v: str) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    serie = _texto(raiz, "DeviceSN", "SerialNumber")
    modelo = _texto(raiz, "DeviceType", "DeviceDescription")
    # SADP suele devolver DeviceType como código numérico (44497). El modelo
    # legible va al principio del nº de serie: "DVR-216Q-F1162018..." o
    # "DS-2CV1023G2-LIDWF20...". Se extrae el prefijo hasta el bloque de dígitos.
    if modelo.isdigit() and serie:
        m = re.match(r"^([A-Za-z]{2,}[\w\-/]*?)(?=\d{6,})", serie)
        if m:
            modelo = m.group(1).rstrip("-")
    return DispositivoRed(
        ip=ip,
        marca=marca,
        modelo=modelo,
        nombre=_texto(raiz, "DeviceName", "DeviceDescription"),
        mac=mac,
        serie=serie,
        firmware=_texto(raiz, "SoftwareVersion", "FirmwareVersion"),
        puerto_http=_int(_texto(raiz, "HttpPort")) or 80,
        puerto_sdk=_int(_texto(raiz, "CommandPort", "Port")) or 8000,
        origen={"SADP"},
    )


# ─────────────────────────────────────────────────────────────────────
# 2) WS-Discovery (ONVIF) — cualquier marca
# ─────────────────────────────────────────────────────────────────────

_WSD_PROBE = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
    'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
    'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
    'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    "<e:Header>"
    "<w:MessageID>uuid:{uuid}</w:MessageID>"
    "<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
    "<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>"
    "</e:Header>"
    "<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>"
    "</e:Envelope>"
)


def sondear_onvif(timeout: float = 4.0) -> list[DispositivoRed]:
    """Descubre cámaras/NVR ONVIF de cualquier marca."""
    encontrados: dict[str, DispositivoRed] = {}
    lock = threading.Lock()

    def _en_interfaz(ip_local: str) -> None:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((ip_local, 0))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                            socket.inet_aton(ip_local))
            sock.settimeout(0.5)
            mensaje = _WSD_PROBE.format(uuid=str(uuid.uuid4())).encode()
            try:
                sock.sendto(mensaje, (_MCAST_GRP, _WSD_PORT))
            except Exception:
                pass
            fin = time.time() + timeout
            while time.time() < fin:
                try:
                    datos, origen = sock.recvfrom(16384)
                except socket.timeout:
                    continue
                except Exception:
                    break
                d = _parsear_onvif(datos, origen[0])
                if d:
                    with lock:
                        if d.ip not in encontrados:
                            encontrados[d.ip] = d
        except Exception:
            pass
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    hilos = [threading.Thread(target=_en_interfaz, args=(ip,), daemon=True)
             for ip in ips_locales()]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout + 2)
    return list(encontrados.values())


def _parsear_onvif(datos: bytes, ip_origen: str) -> DispositivoRed | None:
    try:
        texto = datos.decode("utf-8", errors="ignore")
    except Exception:
        return None
    if "ProbeMatch" not in texto and "XAddrs" not in texto:
        return None
    # La IP real va en XAddrs (http://192.168.1.64/onvif/device_service).
    ip = ip_origen
    m = re.search(r"https?://([\d.]+)[:/]", texto)
    if m:
        ip = m.group(1)
    # Los Scopes traen marca/modelo/nombre en texto plano.
    marca = modelo = nombre = ""
    for campo, patron in (("marca", r"onvif://www\.onvif\.org/manufacturer/([^\s<]+)"),
                          ("modelo", r"onvif://www\.onvif\.org/hardware/([^\s<]+)"),
                          ("nombre", r"onvif://www\.onvif\.org/name/([^\s<]+)")):
        mm = re.search(patron, texto)
        if mm:
            valor = mm.group(1).replace("%20", " ").strip()
            if campo == "marca":
                marca = valor
            elif campo == "modelo":
                modelo = valor
            else:
                nombre = valor
    return DispositivoRed(ip=ip, marca=marca, modelo=modelo, nombre=nombre,
                          origen={"ONVIF"})


# ─────────────────────────────────────────────────────────────────────
# 3) Barrido TCP
# ─────────────────────────────────────────────────────────────────────

def _puerto_abierto(ip: str, puerto: int, timeout: float) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((ip, puerto)) == 0
    except Exception:
        return False


def _huella_http(ip: str, puerto: int, timeout: float = 1.2) -> dict:
    """Identifica un equipo por su respuesta HTTP, SIN credenciales.

    El juez UNIVERSAL es el reto de ISAPI: `GET /ISAPI/System/deviceInfo`
    devuelve `401 WWW-Authenticate: Digest` en CUALQUIER equipo Hikvision,
    esté en el puerto que esté, y el eco de un middlebox NO lo hace. Medido el
    8-ago-2026 contra 3 equipos reales:
      - realm="DVRNVRDVS"        -> DVR/NVR
      - realm="IP Camera(FF877)" -> cámara IP
      - realm="8b39ba3f..."      -> hash (equipo real, modelo no revelado)
      - realm="DS-7208..."       -> el modelo va en el propio realm
    El campo `isapi` marca que respondió el reto (device Hikvision confirmado),
    aunque el realm no diga el modelo. Nunca lanza.
    """
    datos: dict = {}
    try:
        import urllib.error
        import urllib.request
        base = f"http://{ip}:{puerto}"
        try:
            req = urllib.request.Request(f"{base}/ISAPI/System/deviceInfo")
            urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as he:
            if he.code == 401:
                auth = he.headers.get("WWW-Authenticate", "") or ""
                server = he.headers.get("Server", "") or ""
                if "Digest" in auth or "ISAPI" in (server or ""):
                    # ISAPI + Digest = Hikvision, sin lugar a dudas.
                    datos["marca"] = "Hikvision"
                    datos["isapi"] = True
                    m = re.search(r'realm="([^"]+)"', auth)
                    if m:
                        realm = m.group(1).strip()
                        datos["tipo"] = _tipo_por_realm(realm)
                        # DS-XXXX es un modelo real; los realm genéricos
                        # (DVRNVRDVS, "IP Camera(..)", hash) no lo son.
                        mod = re.search(r'(DS-[\w-]+)', realm)
                        if mod:
                            datos["modelo"] = mod.group(1)
        except Exception:
            pass
        # Respaldo por cabecera Server (cámaras viejas sin ISAPI abierto).
        if "marca" not in datos:
            try:
                r = urllib.request.urlopen(base, timeout=timeout)
                server = (r.headers.get("Server", "") or "")
            except urllib.error.HTTPError as he:
                server = (he.headers.get("Server", "") or "")
            except Exception:
                server = ""
            s = server.lower()
            # "DNVRS-Webs"/"DVRDVS-Webs"/"App-webs"/"webserver" son Hikvision;
            # NO confundir con Dahua (que se identifica por 37777/ONVIF).
            if any(t in s for t in ("hikvision", "dnvrs", "dvrdvs",
                                    "app-webs", "webserver", "dvs-webs")):
                datos["marca"] = "Hikvision"
    except Exception:
        pass
    return datos


def _tipo_por_realm(realm: str) -> str:
    """Deduce si el realm de ISAPI es de una cámara o de un DVR/NVR."""
    r = realm.lower()
    if "ip camera" in r or "network camera" in r:
        return "camara"
    if "nvr" in r or "dvr" in r or "dvs" in r:
        return "dvr"
    return ""


def barrer_rango(rango: str, puertos: list[int] | None = None,
                 timeout: float = 0.35, hilos: int = 128,
                 progreso=None, cancelado=None) -> list[DispositivoRed]:
    """Escanea 'x.y.z.1-254' probando los puertos típicos de videovigilancia.

    `rango`: los tres primeros octetos ('192.168.1').
    `progreso(hechos, total)`: callback opcional para la barra de la UI.
    `cancelado()`: si devuelve True, aborta cuanto antes.
    """
    puertos = puertos or list(PUERTOS_SONDA.keys())
    if not rango:
        return []
    objetivos = [f"{rango}.{i}" for i in range(1, 255)]
    total = len(objetivos)
    hechos = 0
    resultados: list[DispositivoRed] = []
    lock = threading.Lock()

    def _avanzar() -> None:
        nonlocal hechos
        with lock:
            hechos += 1
            n = hechos
        if progreso:
            try:
                progreso(n, total)
            except Exception:
                pass

    def _revisar(ip: str) -> None:
        # Toda la E/S de red va FUERA del lock; el lock solo protege el
        # contador y la lista (si no, un solo hilo escanearía a la vez).
        if cancelado and cancelado():
            _avanzar()
            return

        # 1) Señuelo primero: si un puerto donde nada escucha "abre", este host
        #    es un middlebox que contesta a todo. En esa red el barrido TCP no
        #    distingue equipos, así que NO se confía en la lista de puertos: se
        #    va directo a confirmar identidad por HTTP (un DVR real responde
        #    ISAPI con su realm "DS-..."; el eco del middlebox, no).
        if _puerto_abierto(ip, _PUERTO_SENUELO, timeout):
            huella = {}
            web = 0
            for p in _PUERTOS_WEB_ISAPI:
                if cancelado and cancelado():
                    break
                h = _huella_http(ip, p, min(timeout, 1.0))
                if h.get("marca"):
                    huella, web = h, p
                    break
            if not huella.get("marca"):
                _avanzar()          # eco sin identidad real -> descartado
                return
            with lock:
                resultados.append(DispositivoRed(
                    ip=ip, marca=huella["marca"],
                    modelo=huella.get("modelo", ""),
                    tipo=huella.get("tipo", ""),
                    puerto_http=web, puertos_abiertos=[web],
                    origen={"TCP", "ECO", "ISAPI"}))
            _avanzar()
            return

        # 2) Red limpia: la lista de puertos abiertos es fiable.
        abiertos = [p for p in puertos if _puerto_abierto(ip, p, timeout)]
        if not abiertos:
            _avanzar()
            return
        d = DispositivoRed(
            ip=ip,
            puerto_http=(80 if 80 in abiertos else
                         (8080 if 8080 in abiertos else 0)),
            puerto_sdk=(8000 if 8000 in abiertos else
                        (37777 if 37777 in abiertos else 0)),
            puertos_abiertos=abiertos,
            origen={"TCP"},
        )
        # HTTP para identidad sin credenciales: nombra los equipos que
        # SADP/ONVIF no alcanzaron (switch que bloquea multicast). Se prueba
        # el reto ISAPI en los puertos web abiertos —un Hikvision en 8082 se
        # confirma igual que uno en 80—, parando en el primer acierto.
        candidatos = [p for p in _PUERTOS_WEB_ISAPI if p in abiertos]
        for p in (candidatos or ([puerto_web] if (puerto_web := d.puerto_http) else [])):
            huella = _huella_http(ip, p, timeout)
            if huella.get("marca"):
                d.marca = huella["marca"]
                d.modelo = huella.get("modelo", "") or d.modelo
                d.tipo = huella.get("tipo", "") or d.tipo
                if huella.get("isapi"):
                    d.origen.add("ISAPI")
                    d.puerto_http = p       # el puerto real de administración
                break

        with lock:
            resultados.append(d)
        _avanzar()

    with ThreadPoolExecutor(max_workers=hilos) as pool:
        list(pool.map(_revisar, objetivos))
    return resultados


# ─────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────

def _fusionar(destino: dict[str, DispositivoRed], nuevos: list[DispositivoRed]) -> None:
    """Une por IP: los datos ricos (SADP/ONVIF) ganan sobre los del barrido."""
    for d in nuevos:
        actual = destino.get(d.ip)
        if actual is None:
            destino[d.ip] = d
            continue
        actual.origen |= d.origen
        for campo in ("marca", "modelo", "nombre", "mac", "serie", "firmware"):
            if not getattr(actual, campo) and getattr(d, campo):
                setattr(actual, campo, getattr(d, campo))
        if not actual.puerto_http and d.puerto_http:
            actual.puerto_http = d.puerto_http
        if not actual.puerto_sdk and d.puerto_sdk:
            actual.puerto_sdk = d.puerto_sdk
        actual.puertos_abiertos = sorted(
            set(actual.puertos_abiertos) | set(d.puertos_abiertos))


def descubrir(incluir_barrido: bool = True, rango: str = "",
              timeout_multicast: float = 4.0,
              progreso=None, cancelado=None) -> list[DispositivoRed]:
    """Descubrimiento completo: SADP + ONVIF (+ barrido TCP opcional).

    Devuelve la lista fusionada y ordenada por IP. Nunca lanza: una sonda que
    falle (firewall, permisos) simplemente no aporta resultados.
    """
    hallazgos: dict[str, DispositivoRed] = {}

    def _aviso(texto: str, hechos: int = 0, total: int = 0) -> None:
        if progreso:
            try:
                progreso(texto, hechos, total)
            except Exception:
                pass

    _aviso("Buscando equipos Hikvision (SADP)…")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_sadp = pool.submit(sondear_sadp, timeout_multicast)
        f_onvif = pool.submit(sondear_onvif, timeout_multicast)
        try:
            _fusionar(hallazgos, f_sadp.result())
        except Exception:
            pass
        _aviso("Buscando cámaras ONVIF…")
        try:
            _fusionar(hallazgos, f_onvif.result())
        except Exception:
            pass

    if incluir_barrido and not (cancelado and cancelado()):
        # Se barren los rangos de las interfaces DIRECTAMENTE conectadas (la
        # máquina tiene una IP en cada uno), sean privadas o no: hay redes de
        # cliente que usan direccionamiento de rango público internamente
        # (p.ej. 72.68.60.x). El eco de un middlebox que responde a todo se
        # filtra por otra vía (puerto señuelo), no descartando el rango.
        rangos = [rango] if rango else sorted(
            {r for r in (rango_de_ip(ip) for ip in ips_locales()) if r})
        for r in rangos:
            _aviso(f"Escaneando la red {r}.0/24…")
            try:
                _fusionar(hallazgos, barrer_rango(
                    r,
                    progreso=(lambda h, t, _r=r: _aviso(
                        f"Escaneando {_r}.0/24…", h, t)),
                    cancelado=cancelado))
            except Exception:
                pass

    def _clave(d: DispositivoRed) -> tuple:
        try:
            return tuple(int(p) for p in d.ip.split("."))
        except Exception:
            return (0, 0, 0, 0)

    return sorted(hallazgos.values(), key=_clave)
