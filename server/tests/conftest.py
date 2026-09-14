"""Ajustes compartidos de las pruebas del servidor.

Dos cosas viven aqui:

1. La permanencia minima antes de confirmar la llegada (30 s en produccion,
   personas 20 s; 17-ago-2026) se NEUTRALIZA para toda prueba que no la
   este probando: las suites de histeresis, escena, TTL y merodeo fijan SUS
   contratos con relojes simulados de 1-2 s, y obligarlas a "esperar" medio
   minuto falso en cada paso las convertiria en pruebas del umbral, no de lo
   suyo. El contrato del umbral vive en test_llegada_min_permanencia.py, que
   lo re-activa explicito.

2. Los DATOS que genera la suite (log del vigilante, su base y sus capturas,
   la base de la tienda y la bitacora CSV del estacionamiento) van a un
   directorio temporal de la sesion, nunca al repositorio (4-sep-2026). En
   el servidor Linux esas carpetas estan montadas dentro de los contenedores
   de produccion: `vigilante_amazonas/{db,logs,galeria,snapshots}` en
   elde-perimetral y `output/` en elde-tiendas. Cada corrida de la suite
   mezclaba lineas de prueba en `vigilante.jsonl`, abria `output/tienda.db`
   y dejaba filas «Pernocta VIG-c1-T3» falsas en la bitacora del dia, que el
   reporte de WhatsApp contaba como vehiculos pernoctados de verdad. La
   redireccion se aplica al cargar este modulo y se RE-APLICA alrededor de
   cada prueba, porque un `importlib.reload(config)` la deshace (ver
   `_redirigir_rutas_del_vigilante`).
"""
import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

try:
    from vigilante_amazonas import config
except ModuleNotFoundError:
    # Este repositorio es el recorte de TIENDAS: el paquete del vigilante no
    # viaja con el. Todo lo que este conftest hacia por el —redirigir su base,
    # su galeria, sus capturas y su log a un temporal— deja de tener objeto, y
    # las pruebas de tiendas no lo necesitan. Se anula en vez de borrarse para
    # que el fichero siga siendo el mismo a los dos lados y traer cambios de
    # un repositorio al otro no genere un conflicto en cada linea.
    config = None

# ── Datos de la suite a un directorio temporal de SESION (4-sep-2026) ────
# A NIVEL DE MODULO y no en un fixture, ni siquiera de sesion: los modulos del
# vigilante crean su logger al importarse (`logger = configurar_registro(
# __name__)`), y ese primer logger llama a `config.crear_directorios()` y abre
# `config.LOG_ARCHIVO` con un RotatingFileHandler. Un fixture llega tarde: la
# coleccion ya importo los tests y con ellos al vigilante, y la primera linea
# ya esta en `vigilante_amazonas/logs/vigilante.jsonl`. pytest carga este
# conftest antes de importar ningun test, asi que aqui todavia estamos a
# tiempo. Todo el vigilante lee las rutas como `config.RUTA_X` en el momento
# de usarlas (nadie hace `from config import RUTA_X`), por eso basta con
# reasignar los atributos del modulo.
_RAIZ_TMP: Path = Path(tempfile.mkdtemp(prefix="elde-tests-"))
_VIGILANTE_TMP: Path = _RAIZ_TMP / "vigilante"


def _redirigir_rutas_del_vigilante() -> None:
    """Apunta las rutas de datos del vigilante al directorio temporal.

    Es una funcion, y no cinco asignaciones sueltas a nivel de modulo, porque
    hay que RE-APLICARLA (4-sep-2026): `test_los_valores_de_produccion_son_
    45_personas_60_resto` (test_llegada_min_permanencia.py) hace
    `importlib.reload(config)` para leer los umbrales de produccion, y el
    reload re-ejecuta config.py entero: `RUTA_DB`, `RUTA_SNAPSHOTS`,
    `RUTA_GALERIA`, `RUTA_LOGS` y `LOG_ARCHIVO` volvian al repositorio para
    todo lo que corria despues (el resto de ese fichero y ~33 ficheros mas
    de la suite). El log no se resentia porque su handler ya estaba abierto
    en /tmp, pero cualquier prueba posterior que instanciara
    `web.base_datos.BaseDatos()` (usa `config.RUTA_DB`) o guardara una
    captura con `emisor_alertas._guardar_snapshot` (`config.RUTA_SNAPSHOTS`)
    habria escrito en `vigilante_amazonas/{db,snapshots}`, montadas en
    elde-perimetral. El fixture autouse `_rutas_del_vigilante_en_tmp` la
    llama antes y despues de CADA prueba, asi que un reload queda deshecho
    en cuanto termina la prueba que lo hizo. (Dentro de esa misma prueba,
    tras el reload, las rutas son las del repositorio: es responsabilidad
    de la prueba no tocar disco ahi; hoy solo hace asserts.)

    Idempotente: `crear_directorios` es `mkdir(exist_ok=True)`.
    """
    if config is None:
        return
    config.RUTA_GALERIA = _VIGILANTE_TMP / "galeria"
    config.RUTA_SNAPSHOTS = _VIGILANTE_TMP / "snapshots"
    config.RUTA_DB = _VIGILANTE_TMP / "db" / "vigilante.db"
    config.RUTA_LOGS = _VIGILANTE_TMP / "logs"
    config.LOG_ARCHIVO = config.RUTA_LOGS / "vigilante.jsonl"
    config.crear_directorios()          # lee los atributos recien cambiados


_redirigir_rutas_del_vigilante()

# `setdefault`, no asignacion: quien exporte la variable a proposito (una
# corrida contra una base concreta) manda; sin ella, la suite no toca
# `output/`. Los dos modulos leen la variable en CADA llamada
# (`tienda_almacen.ruta_bd()`, `estacionamiento_registro.carpeta()`).
os.environ.setdefault("ELDE_TIENDA_DB", str(_RAIZ_TMP / "tienda" / "tienda.db"))
os.environ.setdefault("ELDE_ESTACIONAMIENTO_DIR",
                      str(_RAIZ_TMP / "estacionamiento"))


def _limpiar_datos_de_sesion() -> None:
    """Cierra el log del vigilante y borra el directorio temporal.

    Va por `atexit` y no por `pytest_sessionfinish` para correr DESPUES del
    resto de ganchos de salida (atexit es LIFO y este se registra el
    primero): `tienda_almacen.get_almacen()` registra un volcado final, y si
    el directorio ya no existiera lo re-crearia (`_preparar` hace mkdir) y
    quedaria un residuo en /tmp. Cerrar el handler antes de borrar es por
    Windows, donde un fichero abierto impide borrar su carpeta.
    """
    raiz = logging.getLogger("vigilante")
    for manejador in list(raiz.handlers):
        if isinstance(manejador, logging.FileHandler):
            raiz.removeHandler(manejador)
            manejador.close()
    shutil.rmtree(_RAIZ_TMP, ignore_errors=True)


atexit.register(_limpiar_datos_de_sesion)


@pytest.fixture(autouse=True)
def _rutas_del_vigilante_en_tmp():
    """Re-aplica la redireccion de rutas antes y despues de cada prueba.

    Antes: la prueba arranca con las rutas en /tmp aunque la anterior
    hiciera `importlib.reload(config)`. Despues: los ganchos de fin de
    sesion (y una prueba de reload que fuese la ultima) tampoco ven el
    repositorio. Asignacion directa y no `monkeypatch.setattr`: el `undo` de
    monkeypatch repondria «lo que habia al entrar», que coincide con /tmp
    solo por el orden en que corren las cosas; asi queda dicho que la
    redireccion es la unica verdad, sin importar que hiciera la prueba.
    """
    _redirigir_rutas_del_vigilante()
    yield
    _redirigir_rutas_del_vigilante()


@pytest.fixture(autouse=True)
def _sin_minimo_de_permanencia(monkeypatch):
    if config is None:
        return
    monkeypatch.setattr(config, "AREA_LLEGADA_MIN_PERMANENCIA_SEG", 0.0)
    monkeypatch.setattr(config, "AREA_LLEGADA_MIN_PERMANENCIA_POR_CLASE", {})


@pytest.fixture(autouse=True)
def _bitacora_de_estacionamiento_en_tmp(tmp_path, monkeypatch):
    """La bitacora CSV del estacionamiento se escribe en el `tmp_path` de CADA
    prueba, nunca en `server/output/estacionamiento/` (4-sep-2026).

    `estacionamiento_registro` es un singleton de proceso con un buffer que
    se vuelca a la carpeta que diga `ELDE_ESTACIONAMIENTO_DIR` EN EL MOMENTO
    del volcado. Sin la variable, cae en `server/output/estacionamiento/`,
    que en el servidor Linux esta montada dentro del contenedor de
    produccion: cada corrida de `test_solo_pernocta_el_vehiculo_...` dejaba
    una fila «Pernocta VIG-c1-T3» falsa en el CSV real del dia, y el reporte
    de WhatsApp la contaba como un vehiculo pernoctado de verdad.

    El valor de sesion de arriba es la red de seguridad (un volcado fuera de
    una prueba, o una prueba que haga `os.environ.pop`, como las de
    test_estacionamiento_reporte.py: monkeypatch lo restaura al salir); este
    fixture ademas aisla cada prueba en su propio `tmp_path`.

    Se fuerza ademas el volcado inmediato (buffer de 1) para que la fila
    aterrice DENTRO de la prueba que la anota, y al salir se vacia el buffer
    del singleton: una fila que quedara encolada la escribiria la siguiente
    prueba (o ninguna) en una carpeta que ya no es esta.
    """
    monkeypatch.setenv("ELDE_ESTACIONAMIENTO_DIR", str(tmp_path))
    monkeypatch.setenv("ELDE_ESTACIONAMIENTO_BUFFER", "1")
    yield
    try:
        from src.analityc.core import estacionamiento_registro as _reg
    except Exception:                                       # noqa: BLE001
        return                      # suites que no cargan ese modulo
    with _reg._REGISTRO._lock:
        _reg._REGISTRO._buffer.clear()


@pytest.fixture(autouse=True)
def _sin_descargas_de_modelos(monkeypatch):
    """La suite NUNCA baja un modelo de internet (8-sep-2026).

    `AnalyticsConfig.EVENT_DETECTION_ENABLED` viene en True, y
    `Hummus.process_frame` levanta con ella un `EventMonitor` en cuanto hay
    una persona dentro del ROI. Ese monitor pide el router multimodal, que
    carga **Qwen2.5-VL-7B-Instruct**: cinco safetensors, unos 16 GB, tirados
    desde HuggingFace en un pool de hilos de fondo.

    Hasta el 4-sep-2026 ninguna prueba llegaba a ese punto, porque el modo
    Hummus no tenia pruebas que empujaran frames de verdad. Las del comedor si
    los empujan, y el resultado era el peor de los sintomas: la suite pasaba
    ("1245 passed in 30s") y despues el proceso NO TERMINABA — los hilos de
    descarga son de un ThreadPoolExecutor, o sea NO demonios, y el `atexit` de
    `concurrent.futures` los espera. Sin mensaje, sin error, sin nada: pytest
    en verde y colgado para siempre. En un CI eso es un trabajo muerto por
    tiempo de espera, y en una maquina de desarrollo, 16 GB de descarga que
    nadie pidio.

    Se apaga aqui y no en cada prueba porque el fallo aparece por la puerta de
    atras (proceso -> ROI -> monitor), no porque nadie lo invoque a proposito.
    Una prueba que quiera de verdad el monitor puede volver a encenderlo con
    su propio monkeypatch; hoy no hay ninguna.
    """
    try:
        from src.analityc.core.analytics.config import AnalyticsConfig
    except Exception:                                       # noqa: BLE001
        return                      # suites que no cargan analytics
    monkeypatch.setattr(AnalyticsConfig, "EVENT_DETECTION_ENABLED", False,
                        raising=False)


# ── Pruebas que no aplican a ESTE repositorio ────────────────────────────────
# Este es el recorte de TIENDAS: viaja `clients/tienda`, pero no los otros tres
# clientes ni los verticales del vigilante, el estacionamiento y el comedor.
# Estas cinco pruebas afirman cosas SOBRE esas piezas —leen el fuente de
# `clients/amazonas` o `clients/detecciones` para comprobar que ofrecen un modo,
# o importan `estacionamiento`— asi que aqui fallarian por ausencia, no por un
# fallo de verdad. Se saltan por nombre, y NO se borran ni se editan sus
# ficheros: asi cada test_*.py es byte a byte el mismo que en el monorepo y
# traer un cambio de alla no genera conflicto. En el monorepo corren todas.
_NO_APLICAN = {
    "test_amazonas_stand.py::test_el_cliente_amazonas_ofrece_el_modo",
    "test_deteccion_por_camara.py::test_el_resumen_publica_el_aviso_para_el_panel",
    "test_permanencia_area.py::test_los_clientes_ofrecen_el_modo",
    "test_payload_a_frame_sync.py::test_la_cadena_completa_acepta_leer_placas",
    "test_payload_a_frame_sync.py::test_la_cadena_completa_acepta_min_permanencia",
}


def pytest_collection_modifyitems(items):
    marca = pytest.mark.skip(
        reason="afirma sobre otro cliente o vertical que no vive en el "
               "repositorio de tiendas (ver conftest._NO_APLICAN)")
    for item in items:
        corto = f"{Path(str(item.fspath)).name}::{item.originalname or item.name}"
        if corto in _NO_APLICAN:
            item.add_marker(marca)
