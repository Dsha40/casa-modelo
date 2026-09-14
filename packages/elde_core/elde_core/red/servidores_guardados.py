"""
Lo que recordamos de los servidores entre arranques.

El descubrimiento encuentra lo que esta en la LAN, pero hay dos cosas que la
red no puede contar sola:

  * Los servidores que NO son escaneables. El de la VPN (10.8.0.x) llega por
    una interfaz punto a punto: no hay subred que barrer ni broadcast que
    alcance al otro extremo. Alguien tiene que escribirlo una vez.
  * Cual eligio el operador la ultima vez. Sin esto, cada arranque volveria a
    preguntar aunque la respuesta sea siempre la misma.

Un JSON en la carpeta de configuracion del usuario, COMPARTIDO por los cuatro
clientes y el selector: elegir servidor en uno deja el resto apuntando ahi,
que es lo que espera quien instala la suite en una maquina. Cada cliente puede
ademas fijar el suyo propio (`cliente=`) y esa preferencia gana sobre la
comun.

Deliberadamente en texto plano y sin cifrar, al contrario que `SettingsModel`:
aqui no hay credenciales, solo direcciones de red, y este fichero tiene que
poder leerlo el selector (que corre con el Python global, sin el venv de
ningun cliente ni `cryptography`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..logging import obtener
from .descubrimiento_servidores import PUERTO_SERVIDOR

# Del registro unificado (`elde.<modulo>`): un logger que no cuelga de `elde`
# se queda sin manejadores y no escribe en ningun sitio.
logger = obtener(__name__)

VERSION = 1
NOMBRE_ARCHIVO = "servidores.json"


def carpeta_config() -> Path:
    """Carpeta de configuracion compartida del ecosistema.

    ELDE_CONFIG_DIR la fuerza (util en pruebas y en instalaciones portables
    donde todo debe quedar junto al .exe).
    """
    forzada = (os.getenv("ELDE_CONFIG_DIR") or "").strip()
    if forzada:
        return Path(forzada)
    if os.name == "nt":
        base = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "ELDE"
    return Path.home() / ".config" / "elde"


def ruta_archivo() -> Path:
    return carpeta_config() / NOMBRE_ARCHIVO


def _vacio() -> dict:
    return {"version": VERSION, "manuales": [], "ultimo": None,
            "por_cliente": {}}


def cargar() -> dict:
    """Lee el archivo. Un JSON corrupto NO tumba el arranque: se ignora y se
    empieza de cero, que es preferible a dejar al operador sin cliente."""
    ruta = ruta_archivo()
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        return _vacio()
    except (OSError, ValueError) as exc:
        logger.warning("servidores guardados ilegibles (%s): %s", ruta, exc)
        return _vacio()
    if not isinstance(datos, dict):
        return _vacio()
    base = _vacio()
    base.update({k: v for k, v in datos.items() if k in base})
    if not isinstance(base.get("manuales"), list):
        base["manuales"] = []
    if not isinstance(base.get("por_cliente"), dict):
        base["por_cliente"] = {}
    return base


def guardar(datos: dict) -> bool:
    """Escritura atomica: primero un temporal y luego `replace`. Un corte de
    luz a mitad de escritura dejaria el JSON truncado y el arranque siguiente
    sin lista de servidores."""
    ruta = ruta_archivo()
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_suffix(".tmp")
        with open(temporal, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        os.replace(temporal, ruta)
        return True
    except OSError as exc:
        logger.warning("no se pudieron guardar los servidores en %s: %s",
                       ruta, exc)
        return False


# ---------------------------------------------------------------------------
#  Servidores anadidos a mano
# ---------------------------------------------------------------------------
def _clave(host: str, puerto: int = PUERTO_SERVIDOR) -> str:
    return f"{str(host).strip()}:{int(puerto)}"


def manuales() -> list[str]:
    """'host:puerto' de los servidores escritos a mano, mas los que venga
    en la variable ELDE_SERVIDORES (lista separada por comas).

    La variable existe para el despliegue: el instalador puede sembrar la
    direccion del servidor de esa sucursal sin tocar codigo ni obligar al
    operador a escribirla."""
    lista = [str(x) for x in cargar().get("manuales", []) if str(x).strip()]
    del_entorno = (os.getenv("ELDE_SERVIDORES") or "").replace(";", ",")
    for pieza in del_entorno.split(","):
        pieza = pieza.strip()
        if not pieza:
            continue
        host, _, puerto = pieza.partition(":")
        clave = _clave(host, int(puerto) if puerto.isdigit()
                       else PUERTO_SERVIDOR)
        if clave not in lista:
            lista.append(clave)
    return lista


def anadir_manual(host: str, puerto: int = PUERTO_SERVIDOR) -> bool:
    host = str(host).strip()
    if not host:
        return False
    datos = cargar()
    clave = _clave(host, puerto)
    if clave in datos["manuales"]:
        return True
    datos["manuales"].append(clave)
    return guardar(datos)


def quitar_manual(host: str, puerto: int = PUERTO_SERVIDOR) -> bool:
    datos = cargar()
    clave = _clave(host, puerto)
    if clave not in datos["manuales"]:
        return False
    datos["manuales"] = [c for c in datos["manuales"] if c != clave]
    return guardar(datos)


# ---------------------------------------------------------------------------
#  El ultimo elegido
# ---------------------------------------------------------------------------
def recordar(host: str, puerto: int = PUERTO_SERVIDOR,
             cliente: Optional[str] = None) -> bool:
    """Fija el servidor elegido. Sin `cliente` es el de todos; con `cliente`
    solo el de ese cliente (que gana sobre el comun)."""
    host = str(host).strip()
    if not host:
        return False
    datos = cargar()
    entrada = {"host": host, "puerto": int(puerto)}
    if cliente:
        datos["por_cliente"][str(cliente)] = entrada
    else:
        datos["ultimo"] = entrada
    return guardar(datos)


def ultimo(cliente: Optional[str] = None) -> Optional[tuple[str, int]]:
    """(host, puerto) del ultimo elegido, o None si nunca se eligio."""
    datos = cargar()
    entrada = None
    if cliente:
        entrada = datos.get("por_cliente", {}).get(str(cliente))
    if not entrada:
        entrada = datos.get("ultimo")
    if not isinstance(entrada, dict):
        return None
    host = str(entrada.get("host") or "").strip()
    if not host:
        return None
    try:
        puerto = int(entrada.get("puerto") or PUERTO_SERVIDOR)
    except (TypeError, ValueError):
        puerto = PUERTO_SERVIDOR
    return host, puerto


def olvidar(cliente: Optional[str] = None) -> bool:
    """Borra la eleccion guardada: el proximo arranque volvera a preguntar."""
    datos = cargar()
    if cliente:
        datos.get("por_cliente", {}).pop(str(cliente), None)
    else:
        datos["ultimo"] = None
    return guardar(datos)
