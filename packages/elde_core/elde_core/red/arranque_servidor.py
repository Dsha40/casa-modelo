"""
A que servidor se conecta ESTE cliente en ESTE arranque.

Los cuatro clientes leen `server_ws_url` del entorno y `config/ajustes.py` los
para en seco si falta. Eso funciona mientras exista una carpeta del proyecto
con un `.env` que alguien haya editado a mano; en un .exe instalado en la
maquina de un cliente, no. Este modulo decide la URL ANTES de que
`ajustes.cargar()` la mire, y la deja puesta en `os.environ` para que ni los
clientes ni el nucleo tengan que cambiar como la leen.

El orden, de mas explicito a menos:

  1. `--servidor <host|url>` en la linea de ordenes  -> manda siempre.
     `--elegir-servidor`                            -> fuerza la ventana.
  2. El SELECTOR lo fijo (ELDE_SERVIDOR_FIJADO=1)   -> ya se pregunto alli.
  3. Lo que el operador eligio la ultima vez        -> gana sobre el .env:
     es una decision posterior y mas informada.
  4. `server_ws_url` del .env / del entorno         -> instalaciones actuales.
  5. La ventana de conexion.

Con una excepcion que atraviesa 3 y 4: si el servidor decidido NO responde, se
pregunta igualmente (con el mismo preseleccionado y un clic para seguir). Es
lo que hace que un servidor que cambia de IP se arregle solo en vez de dejar
al cliente reintentando en silencio contra una direccion muerta. Para
instalaciones desatendidas, ELDE_NO_PREGUNTAR=1 lo desactiva.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

from ..logging import obtener
from . import descubrimiento_servidores as D
from . import servidores_guardados as G

# Del registro unificado (`elde.<modulo>`), no `logging.getLogger` a secas: un
# logger que no cuelga de `elde` se queda sin manejadores y sus mensajes se
# pierden. Y aqui sale la linea que dice A QUE SERVIDOR se conecto el cliente
# y POR QUE, que es lo primero que se mira cuando alguien reporta que "no
# llega nada".
logger = obtener(__name__)

# "El servidor ya se eligio, no vuelvas a preguntar". Sin ella no habria forma
# de distinguir un `server_ws_url` puesto por el selector de uno que venga del
# .env, y el cliente volveria a preguntar justo despues de que el operador
# eligiera en el hub.
#
# El VALOR dice para quien vale:
#   "1"        el selector, y por tanto para cualquier cliente que lance.
#   "<nombre>" un cliente que ya resolvio LO SUYO.
# La distincion importa porque las variables de entorno se HEREDAN: un proceso
# hijo de otro cliente veria la marca del padre y se saltaria su propia
# eleccion, conectandose al servidor de otro.
MARCA_FIJADO = "ELDE_SERVIDOR_FIJADO"

_VERDADEROS = ("1", "si", "sí", "yes", "true", "on")


def _de_argv(argv: Sequence[str]) -> tuple[Optional[str], bool]:
    """(url pedida por linea de ordenes, hay que forzar la ventana)."""
    url, forzar = None, False
    lista = list(argv)
    for i, arg in enumerate(lista):
        if arg in ("--elegir-servidor", "--seleccionar-servidor"):
            forzar = True
        elif arg == "--servidor" and i + 1 < len(lista):
            url = lista[i + 1]
        elif arg.startswith("--servidor="):
            url = arg.split("=", 1)[1]
    return url, forzar


def _normalizar(texto: str) -> str:
    """'10.0.0.5' | '10.0.0.5:9000' | 'ws://10.0.0.5:9000/ws' -> ws url."""
    texto = (texto or "").strip().strip('"').strip("'")
    if not texto:
        return ""
    if texto.lower().startswith(("ws://", "wss://")):
        return texto
    host, puerto = D.partes_de_ws_url(texto)
    return D.ws_url(host, puerto) if host else ""


def _no_preguntar() -> bool:
    return (os.getenv("ELDE_NO_PREGUNTAR", "0").strip().lower()
            in ("1", "si", "sí", "yes", "true", "on"))


def _hay_pantalla() -> bool:
    """Se puede abrir una ventana? En un servicio sin sesion grafica, no."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def preguntar(cliente: str = "", parent=None) -> Optional[str]:
    """Abre la ventana de conexion. Devuelve la ws url o None si se cancelo.

    Reutiliza la QApplication que ya exista (los clientes crean la suya con
    AppSingleton); si no hay ninguna crea una, porque este dialogo puede
    tener que salir antes que la ventana principal.
    """
    if not _hay_pantalla():
        return None
    from PySide6.QtWidgets import QApplication

    from ..ui.dialogo_servidor import DialogoServidor

    creada_aqui = QApplication.instance() is None
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        dlg = DialogoServidor(cliente=cliente, parent=parent)
        if dlg.exec() and dlg.seleccionado is not None:
            return dlg.seleccionado.ws_url
        return None
    finally:
        # Si la QApplication se creo solo para esto, se deja procesar la cola
        # pendiente antes de que el cliente cree la suya de verdad.
        if creada_aqui:
            app.processEvents()


def resolver(cliente: str = "", argv: Optional[Sequence[str]] = None,
             permitir_dialogo: bool = True) -> Optional[str]:
    """Decide el servidor y lo deja en `os.environ['server_ws_url']`.

    Devuelve la URL, o None si el operador cerro la ventana sin elegir (en
    cuyo caso quien llama debe salir sin abrir el cliente: conectarse a
    ninguna parte no le sirve a nadie).
    """
    argv = list(argv if argv is not None else sys.argv)
    pedida, forzar = _de_argv(argv)

    # 1) Linea de ordenes.
    if pedida and not forzar:
        url = _normalizar(pedida)
        if url:
            return _fijar(url, cliente, recordar=True, motivo="linea de ordenes")

    # 2) El selector ya eligio (o este mismo cliente, en otra pasada).
    marca = os.getenv(MARCA_FIJADO, "").strip()
    fijado = bool(marca) and (marca.lower() in _VERDADEROS
                              or marca == (cliente or ""))
    del_entorno = _normalizar(os.getenv("server_ws_url", ""))
    if fijado and del_entorno and not forzar:
        return _fijar(del_entorno, cliente, recordar=False, motivo="selector")

    # 3) y 4) candidato: lo guardado gana sobre el .env.
    candidato = ""
    procedencia = ""
    guardado = G.ultimo(cliente or None)
    if guardado:
        candidato = D.ws_url(*guardado)
        procedencia = "eleccion guardada"
    elif del_entorno:
        candidato = del_entorno
        procedencia = ".env"

    if candidato and not forzar:
        host, puerto = D.partes_de_ws_url(candidato)
        if _no_preguntar() or D.alcanzable(host, puerto):
            return _fijar(candidato, cliente, recordar=False,
                          motivo=procedencia)
        logger.warning(
            "el servidor %s (%s) no responde; se pregunta cual usar",
            candidato, procedencia)

    # 5) Preguntar.
    if not permitir_dialogo:
        return _fijar(candidato, cliente, recordar=False,
                      motivo=procedencia) if candidato else None
    elegido = preguntar(cliente)
    if elegido:
        return _fijar(elegido, cliente, recordar=False, motivo="ventana")
    # Cancelado: si habia un candidato razonable se sigue con el en vez de
    # cerrar el cliente de golpe.
    if candidato:
        return _fijar(candidato, cliente, recordar=False,
                      motivo=f"{procedencia} (ventana cancelada)")
    return None


def _fijar(url: str, cliente: str, recordar: bool, motivo: str) -> str:
    os.environ["server_ws_url"] = url
    # El NOMBRE del cliente, no un "1": la marca la heredan los procesos hijo
    # y solo debe valer para quien la puso (ver MARCA_FIJADO).
    os.environ[MARCA_FIJADO] = cliente or "1"
    if recordar:
        host, puerto = D.partes_de_ws_url(url)
        G.recordar(host, puerto, cliente or None)
    logger.info("servidor de inferencia: %s (%s)", url, motivo)
    return url
