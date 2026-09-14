"""
Arranca el servidor de inferencia (websocket :9000) SIN interfaz gráfica.

Equivalente a abrir main.py y pulsar "Iniciar servidor", pero apto para
lanzarse desatendido (scripts, PM2, inicio automático). Detener: Ctrl+C o
matar el proceso.

Uso:
    venv\\Scripts\\python.exe iniciar_servidor_headless.py [puerto]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _asegurar_venv() -> None:
    """Si se lanzó con OTRO Python (p. ej. el global de Program Files), se
    re-lanza con el del venv. GOTCHA conocido de esta máquina: dos servidores
    (venv y global) compitiendo por el puerto 9000 — el global no tiene el
    stack completo (supervision/tensorrt/socketio) y rompe el modo
    VigilanteAmazonas en silencio."""
    base = Path(__file__).resolve().parent
    # Windows: venv\Scripts\python.exe — Linux/Mac: venv/bin/python
    if os.name == "nt":
        venv_py = base / "venv" / "Scripts" / "python.exe"
    else:
        venv_py = base / "venv" / "bin" / "python"
    # ¿YA corremos dentro del venv? Se decide por sys.prefix, no comparando
    # ejecutables: en Linux venv/bin/python es un SYMLINK al python del
    # sistema y resolver ambos lados los colapsa al mismo archivo — el guard
    # nunca re-lanzaria (17-ago-2026). sys.prefix apunta al venv activo en
    # los dos sistemas.
    en_venv = Path(sys.prefix).resolve() == (base / "venv").resolve()
    if venv_py.exists() and not en_venv:
        print(f"[AVISO] lanzado con {sys.executable}; "
              f"re-lanzando con el venv: {venv_py}")
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)


_asegurar_venv()

# Registro unificado (nucleo), ANTES de importar nada del servidor: sus modulos
# hacen `logging.getLogger(__name__)` al importarse y `app.py` llama a
# `logging.basicConfig()`. Configurando primero, esa llamada queda en nada (no
# hace nada si la raiz ya tiene manejadores) y el servidor entero escribe con el
# mismo formato que los cuatro clientes, con `server/<site_id>` delante.
from elde_core.logging import configurar as _iniciar_registro  # noqa: E402

_iniciar_registro(
    'server',
    # SITE_ID en MAYUSCULAS: en Windows las variables de entorno no distinguen
    # mayusculas y `site_id` valia; en Linux si distinguen y los contenedores
    # se identificaban todos como 'sitio-unico'. Se acepta la forma vieja.
    (os.getenv('SITE_ID') or os.getenv('site_id') or '').strip() or 'sitio-unico',
    raiz_proyecto=Path(__file__).resolve().parent,
    tambien_raiz=True)

import uvicorn  # noqa: E402  (tras asegurar el venv)


def _iniciar_dashboard_vigilante() -> None:
    """Levanta el dashboard de galería (:8090) desde el arranque, en ESTE
    mismo proceso.

    Así un solo proceso ofrece todo (websocket 9000 + dashboard 8090 +
    alertas 8091) y no hace falta lanzar `vigilante_amazonas/main.py` aparte
    (que duplicaría los modelos en la GPU y chocaría de puertos). Crear la app
    NO carga modelos: se resuelven de forma perezosa al usarlos."""
    try:
        from vigilante_amazonas.web.lanzador import iniciar_dashboard
        from vigilante_amazonas import config as vig_config
        iniciar_dashboard()
        print(f"Dashboard de galería:  http://localhost:{vig_config.PUERTO_API}")
    except Exception as exc:
        print(f"[AVISO] dashboard de VIGILANTE no disponible: {exc}")


# El dashboard de TIENDA del 9030 se retiro el 31-jul-2026 y su modulo se
# BORRO el 19-ago-2026: lo sustituye /dashboards/tienda/, que ademas hace
# cosas que aquel no sabia hacer (ventana temporal, comparativas, curva por
# hora real). Se conservaba «por si el rollback», pero el rollback dejo de
# tener sentido el dia que la sustituta fue mejor; y 686 lineas muertas con su
# propia copia del HTML y su propia idea de «pasillo» solo sirven para que
# alguien las lea creyendo que estan vivas. El original esta en el historico
# de git, que es donde vive el codigo retirado.


def main() -> int:
    puerto: int = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    # La baliza de descubrimiento (UDP) anuncia a los clientes en que puerto
    # escucha ESTE servidor. El puerto solo se conoce aqui (llega por argv),
    # y `app.py` arranca la baliza dentro de su lifespan, asi que viaja por
    # entorno. Sin esto anunciaria siempre el 9000 y un servidor arrancado en
    # otro puerto mandaria a los clientes a una puerta cerrada.
    # setdefault (no asignacion): un contenedor puede escuchar en 9101 y, con
    # ELDE_PUERTO_SERVIDOR=9000 desde fuera, anunciar en la baliza el 9000
    # del proxy, que es a donde deben ir los clientes.
    os.environ.setdefault("ELDE_PUERTO_SERVIDOR", str(puerto))
    # Import tardío: carga modelos/GPU (tarda ~1 min la primera vez).
    from src.app.app import app
    # El panel del vigilante (:5333) solo lo levanta el proceso que sirve ese
    # vertical: con varios contenedores en red del host, si todos lo abrieran
    # pelearian por el puerto. ELDE_PANEL_VIGILANTE=1 lo enciende (apagado
    # por defecto; el compose lo fija en el servicio que corresponde).
    if os.getenv("ELDE_PANEL_VIGILANTE", "0") == "1":
        _iniciar_dashboard_vigilante()
    print(f"Dashboards por dominio: http://localhost:{puerto}/dashboards/")
    print(f"Servidor de inferencia SIN GUI en ws://0.0.0.0:{puerto}/ws "
          f"(Ctrl+C para detener)")
    # Keepalive del websocket (H-26). Uvicorn manda un PING de protocolo y
    # si el PONG no vuelve a tiempo cierra con 1011 "keepalive ping timeout".
    # Con los valores por defecto (20s/20s) mataba conexiones VIVAS cada
    # 40-100s: el pong se retrasa cuando la GUI del cliente se congela unos
    # segundos (AppHangB1) o cuando hay contrapresion (el cliente escribe mas
    # rapido de lo que el servidor drena y el pong queda en cola tras los
    # frames). Reproducido sinteticamente: closeCode=1011 a los 51s con
    # 427 frames enviados y 3 respuestas. Un timeout de 60s tolera esos
    # baches y sigue detectando peers realmente muertos; el cliente ademas
    # se reconecta solo a los 5s si aun asi se corta.
    ping_cada = float(os.getenv("ELDE_WS_PING_INTERVALO_SEG", "20"))
    ping_timeout = float(os.getenv("ELDE_WS_PING_TIMEOUT_SEG", "60"))
    # timeout_graceful_shutdown: al recibir SIGINT (Ctrl+C, docker stop con
    # STOPSIGNAL SIGINT) uvicorn espera hasta 60 s a que cierren las
    # conexiones y corre el lifespan/atexit que vuelcan el registro de
    # dispositivos y los mapas de calor, en vez de cortar a la primera.
    uvicorn.run(app, host="0.0.0.0", port=puerto, log_level="info",
                ws_ping_interval=ping_cada, ws_ping_timeout=ping_timeout,
                timeout_graceful_shutdown=60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
