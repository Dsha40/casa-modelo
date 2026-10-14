import asyncio
import base64
import importlib.util
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import msgpack
import numpy as np
from fastapi import (FastAPI, WebSocket, WebSocketDisconnect,
                     UploadFile, File, Form)
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from . import baliza_descubrimiento as _baliza
from . import captura_contrato as _captura
from . import fallo_imagen_IA_pipeline as _fallo
from . import registro_dispositivos as _registro
from . import validacion_contrato as _validacion
from . import whatsapp_alertas as _whatsapp
from .ws_drenaje import DrenajeCliente, ItemFrame
from ..analityc.core.hardware_available import device_hardware
from ..analityc.core.analytics import heatmap_registro as _heatmaps
from ..analityc.config.config import get_config

# Pipelines: imports PEREZOSOS. Cada clase se importa dentro de su rama de
# _build_processor la primera vez que un cliente pide ese modo, y queda
# enlazada aqui (global) para el despacho por isinstance de mas abajo. Asi
# importar app.py no carga las 8 clases (ni el .pyc de perimetrales) en un
# contenedor que solo sirve un vertical; las ramas isinstance van detras de
# `if X is not None`, el mismo patron que VigilanteWS y PerimetralNocturnaWS.
BasePerimeter = None
BoTSORTWrapper = None
VehicleProcessor = None
HummusProcess = None
MistersProcess = None
PerimetralesMultiCam = None
TiendasProcess = None

# Motor perimetral NUEVO (paquete perimetral_nocturna en la raiz del proyecto):
# 8 clases finales con genero (HOMBRE/MUJER/NIÑO + CARRO/MOTO/CAMIONETA/CAMIÓN),
# Re-ID global entre camaras y alertas llegada/salida con tiempos. Reemplaza al
# PerimetralesMultiCam legado para el cliente perimetrales-view.
try:
    from perimetral_nocturna.adaptador_websocket import PerimetralNocturnaWS
except Exception as _exc:  # el server sigue arrancando si el paquete falta
    PerimetralNocturnaWS = None
    logging.getLogger(__name__).warning(
        "perimetral_nocturna no disponible (%s); el modo PerimetralesMultiCam "
        "usara el procesador legado", _exc)

# VIGILANTE-AMAZONAS (paquete vigilante_amazonas): deteccion 6 clases
# (persona/seguridad/moto/carro/camioneta/camion; "objeto" apagado) +
# personal_seguridad (CLIP) + Re-ID de personas ESPECIFICAS (galeria de
# rostros/vestimenta administrada en el dashboard :8090) + alertas Socket.IO.
# Las camaras se VEN en perimetrales-view: el cliente envia los frames por
# este websocket (modo "VigilanteAmazonas").
try:
    from vigilante_amazonas.adaptador_websocket import VigilanteWS
except Exception as _exc:
    VigilanteWS = None
    logging.getLogger(__name__).warning(
        "vigilante_amazonas no disponible (%s); modo VigilanteAmazonas "
        "desactivado", _exc)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constantes de configuracion
# -----------------------------------------------------------------------------
WEBSOCKET_TIMEOUT: float = 30.0      # segundos sin mensaje antes de ping
JPEG_QUALITY: int = 70               # calidad JPEG para imagen de salida
EXECUTOR_WORKERS: int = 8


# -----------------------------------------------------------------------------
# Ciclo de vida: ThreadPoolExecutor gestionado correctamente
# -----------------------------------------------------------------------------
executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)


def _get_live_executor() -> ThreadPoolExecutor:
    """Devuelve el executor activo; si fue apagado, lo recrea."""
    global executor
    if getattr(executor, "_shutdown", False):
        logger.info("Recreando ThreadPoolExecutor (anterior estaba apagado)")
        executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
    return executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    if getattr(executor, "_shutdown", False):
        executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
    logger.info("Servidor iniciando - ThreadPoolExecutor con %d workers", EXECUTOR_WORKERS)
    # El registro de dispositivos se acumula entre arranques: sin esto, cada
    # reinicio del servidor perderia el historico de que camaras existen.
    _registro.cargar()
    # Cronómetro por etapas, imagen a imagen (6-oct-2026, diagnóstico): sin
    # ELDE_TIEMPOS_ETAPAS no hace nada. Aquí y no al importar: tiene que
    # envolver `_full_frame_sync` ANTES de que una conexión la tome.
    try:
        from ..analityc.core import tiempos_etapas
        tiempos_etapas.instalar()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("tiempos_etapas no disponible: %s", exc)
    # Clasificación en sombra (8-oct-2026, prueba): sin
    # ELDE_CLASIFICACION_SOMBRA no hace nada; con ella, una segunda forma de
    # clasificar apunta aparte lo que habría decidido, sin tocar la actual.
    try:
        from ..analityc.core import clasificacion_sombra
        clasificacion_sombra.instalar()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("clasificacion_sombra no disponible: %s", exc)
    # Baliza de descubrimiento (UDP): responde a los clientes que preguntan
    # "¿que servidores ELDE hay en esta red?". El puerto real lo pone quien
    # arranca uvicorn (iniciar_servidor_headless.py exporta ELDE_PUERTO_
    # SERVIDOR); si nadie lo dice, el de siempre. Que falle no impide servir:
    # los clientes tienen el escaneo del puerto como respaldo.
    _baliza.iniciar(
        puerto_servidor=int(os.getenv('ELDE_PUERTO_SERVIDOR', '9000') or 9000),
        contar_clientes=lambda: len(active_connections))
    # Reporte diario del estacionamiento: arranca con el SERVIDOR y no con el
    # primer cliente (9-sep-2026). Antes el hilo nacia dentro de
    # `EstacionamientoWS.__init__`, asi que un dia sin nadie conectado en ese
    # modo era un dia sin reporte — y sin un solo aviso de que faltaba. Al
    # arrancar recupera el envio de hoy si la hora ya paso y no salio, que es
    # lo que cubre el otro agujero: un despliegue cerca de la hora se lo
    # comia. Cuando un cliente se conecte, el singleton le toma el contexto
    # (local, foto, ocupacion) y el reporte sale mas completo.
    _arrancar_reporte_estacionamiento()
    yield
    logger.info("Servidor apagandose - cerrando ThreadPoolExecutor")
    _baliza.detener()
    # El volcado esta amortiguado (cada 30 s), asi que al cerrar puede quedar
    # medio minuto de altas sin escribir.
    _registro.volcar()
    # Mapas de calor: el acumulado y la hora en curso a disco. Sin esto, un
    # apagado pierde hasta 30 s de estado y la hora parcial del historico.
    _heatmaps.volcar_todos()
    executor.shutdown(wait=False)


def _arrancar_reporte_estacionamiento() -> None:
    """Programa el reporte diario si ESTA vertical sirve Estacionamiento.

    Se comprueba contra `ELDE_PIPELINES` porque la misma imagen levanta los
    tres servidores: sin el filtro, los tres armarian el temporizador y el
    grupo recibiria el mismo PDF por triplicado.
    """
    permitidos = _pipelines_permitidos()
    if permitidos is not None and 'Estacionamiento' not in permitidos:
        return
    # El corte de tiendas NO trae `estacionamiento_planificador`. Hasta el
    # 22-sep-2026 su ausencia caia en el `except` de abajo y salia por
    # `logger.exception`: un ERROR en rojo con traceback completo en CADA
    # arranque del servidor, por algo que es exactamente lo mismo que
    # `vigilante_amazonas` y `perimetral_nocturna` reportan como un WARNING de
    # una linea. Asustaba sin motivo y tapaba los errores de verdad.
    #
    # Se mira ANTES si el modulo existe, en vez de reblandecer el `except`, para
    # no perder el caso que si importa: en el monorepo el planificador SI esta,
    # y si revienta al importarse (una dependencia suya rota, por ejemplo) el
    # reporte diario del estacionamiento se pierde en silencio. Ese caso tiene
    # que seguir saliendo como ERROR con su traza.
    modulo = '..analityc.core.estacionamiento_planificador'
    try:
        presente = importlib.util.find_spec(modulo, package=__package__) is not None
    except (ImportError, ValueError):
        # find_spec no siempre devuelve None cuando algo falta, de ahi los dos:
        #   ImportError -> tiene que importar los paquetes PADRE para mirar
        #                  dentro, y `analityc.core` tampoco estuviera.
        #   ValueError  -> el nombre ya esta en sys.modules pero sin `__spec__`
        #                  (un modulo postizo, tipico de una prueba).
        # Los dos significan lo mismo aqui: no hay planificador que arrancar.
        presente = False
    if not presente:
        logger.warning(
            'estacionamiento_planificador no disponible; el reporte diario del '
            'estacionamiento no se programara')
        return
    try:
        from ..analityc.core import estacionamiento_planificador as _plan
        _plan.planificador().iniciar()
    except Exception:                                       # noqa: BLE001
        # Que falle no puede impedir servir frames; pero que se vea. El modulo
        # EXISTE y aun asi no arranco: el reporte se perdio de verdad.
        logger.exception('no se pudo programar el reporte del estacionamiento')


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dashboard web de analitica de visitantes (GET /dashboard + API JSON).
# Import al final del modulo NO es posible (app debe existir); el router
# usa imports perezosos de este modulo para evitar el ciclo.
from .dashboard import router as dashboard_router  # noqa: E402
app.include_router(dashboard_router)
# Apartado «Stock» del dashboard (9-oct-2026): alertas de «posible agotado»
# activas y su historial con foto (`dashboard_stock.py`).
from .dashboard_stock import router as dashboard_stock_router  # noqa: E402
app.include_router(dashboard_stock_router)

# API REST de LECTURA (HITO 8). Es la puerta por la que los dashboards leeran
# cuando dejen de vivir dentro de este proceso (HITO 9). Solo GET: lo que
# modifica algo se queda en /dashboard/api/.
from .api_lectura import router as api_lectura_router  # noqa: E402
app.include_router(api_lectura_router)

# Tienda CON VENTANA TEMPORAL (HITO 2, 18-ago-2026). Prefijo propio
# /api/v1/tienda: lee de tienda_almacen (SQLite con jornada), no de los
# procesadores vivos, asi que responde igual con el cliente conectado o no y
# suma TODAS las sesiones del rango. Los endpoints de siempre no se tocan
# (compatibilidad aditiva): sus campos los leen otras tres paginas y el PDF.
from .api_tienda import router as api_tienda_router  # noqa: E402
app.include_router(api_tienda_router)

# Busqueda VLM de los dashboards de producto ("buscame el carro rojo").
# Son ACCIONES (encender el buscador, lanzar una busqueda), asi que viven en
# /dashboard/api y no en /api/v1, que es solo lectura por regla.
from .busqueda_vlm import router as busqueda_vlm_router  # noqa: E402
app.include_router(busqueda_vlm_router)

# Stand de degustacion (HITO 10). Prefijo propio /api/stand porque el modulo
# necesita UN post (enrolar personal) y /api/v1 es solo lectura por regla.
# El router es autocontenido: su propio SQLite, su propio namespace, y no
# comparte estado con el resto del servidor. Si el modulo esta deshabilitado
# (config/stand_config.yaml -> habilitado: false) los endpoints responden
# igual, con las cifras a cero: no hay nada que romper.
from .stand_api import router as stand_router  # noqa: E402
app.include_router(stand_router)

# Anaqueles (13-ago-2026). Prefijo propio /api/anaqueles, patron stand_api:
# modulo autocontenido que lee los JSON diarios de anaqueles_registro (el
# disco es la fuente de verdad; el pipeline escribe amortiguado cada 5 s).
# GET /api/anaqueles/frame es LA herramienta para calibrar las areas.
from .anaqueles_api import router as anaqueles_router  # noqa: E402
app.include_router(anaqueles_router)

# Personal de tienda (14-ago-2026). Prefijo propio /api/personal (escribe:
# altas/bajas/fotos, y /api/v1 es solo lectura por regla). Enrolamiento con
# FOTO desde el dashboard y el cliente; el pipeline excluye a los enrolados
# de las metricas de marketing y emite las alertas de seguridad/pasillero.
from .personal_api import router as personal_router  # noqa: E402
app.include_router(personal_router)

# Informe PDF del dia (14-ago-2026). Boton "Exportar PDF" del dashboard de
# tienda: documento autocontenido con resumen, anaqueles, demografia, mapas
# de calor, registro de personas, personal y metodologia.
from .informe_pdf import router as informes_router  # noqa: E402
app.include_router(informes_router)

# Dashboards del HITO 9: paginas ESTATICAS en `dashboards/` (hermana de
# `server/` en la raiz del proyecto) que leen exclusivamente de /api/v1 por
# HTTP. El servidor solo les hace de fichero: no las importa, no comparte
# estado con ellas, y el dia que vivan en otro proceso basta con servir la
# misma carpeta desde alli. Si la carpeta no existe, el servidor arranca igual.
from fastapi.staticfiles import StaticFiles  # noqa: E402

_dashboards_dir = (os.getenv('ELDE_DASHBOARDS_DIR', '').strip()
                   or str(Path(__file__).resolve().parents[3] / 'dashboards'))
if os.path.isdir(_dashboards_dir):
    app.mount('/dashboards',
              StaticFiles(directory=_dashboards_dir, html=True),
              name='dashboards')
else:
    logger.warning('carpeta de dashboards no encontrada (%s); /dashboards '
                   'queda sin montar', _dashboards_dir)


# -----------------------------------------------------------------------------
# Estado de conexiones
# -----------------------------------------------------------------------------
active_connections: Dict[str, Dict[str, Any]] = {}
connection_lock = asyncio.Lock()


# -----------------------------------------------------------------------------
# Helpers de configuracion / hardware
# -----------------------------------------------------------------------------

def _get_device_default() -> str:
    dev = getattr(device_hardware, "device_default", "cpu")
    if isinstance(dev, dict):
        return dev.get("gpu_use", "cpu")
    return dev if isinstance(dev, str) else "cpu"


def _get_first_gpu() -> str:
    gpus = getattr(device_hardware, "gpu_tuple", [])
    if isinstance(gpus, (list, tuple)) and gpus:
        gpu0 = gpus[0]
        if isinstance(gpu0, dict):
            return gpu0.get("gpu_use", _get_device_default())
    return _get_device_default()


#: Raiz del servidor (`server/`), deducida de la posicion de este archivo:
#: `server/src/app/app.py` -> parents[2].
_RAIZ_SERVIDOR = Path(__file__).resolve().parents[2]


def _modelo(relativa: str) -> str:
    """Ruta a un modelo, anclada a la carpeta del servidor.

    Cinco de estos valores por defecto eran rutas ABSOLUTAS a esta maquina
    (`C:\\Users\\...\\SERVER-IA PERIMETRALES\\models\\...`), lo que rompia la
    regla 6 y ademas dejo de existir al mover el servidor a `server/` el
    30-jul-2026. Anclar a `__file__` en vez de a rutas relativas evita tambien
    depender del directorio de trabajo desde el que se arranque.
    """
    return str(_RAIZ_SERVIDOR / 'models' / relativa)


def _get_model_path(config: Dict[str, Any], inference_name: str, default_path: str) -> str:
    model_paths = config.get("model_paths")
    if isinstance(model_paths, dict):
        path = model_paths.get(inference_name)
        if path:
            return path
    return config.get("model_path", default_path)


def _get_person_model_path(config: Dict[str, Any], inference_name: str, default_path: str) -> str:
    person_model_paths = config.get("person_model_paths")
    if isinstance(person_model_paths, dict):
        path = person_model_paths.get(inference_name)
        if path:
            return path
    return default_path


# -----------------------------------------------------------------------------
# Creacion de procesadores
# -----------------------------------------------------------------------------

def _pipelines_permitidos() -> Optional[set]:
    """Que modos sirve ESTE proceso, segun ELDE_PIPELINES (lista separada por
    comas con los valores exactos de type_inference). Vacia o ausente = todos.
    Es lo que permite repartir los verticales en contenedores distintos con
    la misma imagen: la vertical es configuracion, no codigo."""
    crudo = (os.getenv("ELDE_PIPELINES") or "").strip()
    if not crudo:
        return None
    return {p.strip() for p in crudo.split(",") if p.strip()}


def _build_processor(type_inference: str, client_id: str, config: Dict[str, Any]) -> Any:
    """Instancia y devuelve el procesador correcto segun el tipo de inferencia."""
    # Las clases de pipeline se importan aqui (perezoso) y se publican como
    # globales del modulo para el despacho por isinstance.
    global BasePerimeter, BoTSORTWrapper, VehicleProcessor, HummusProcess, \
        MistersProcess, PerimetralesMultiCam, TiendasProcess
    permitidos = _pipelines_permitidos()
    if permitidos is not None and type_inference not in permitidos:
        # Un modo que este proceso no sirve se trata como desconocido: el
        # mismo ValueError de siempre, que websocket_endpoint ya convierte
        # en {"status":"error"} + close(1008). Sin mecanismo nuevo.
        logger.warning("modo '%s' no servido por este proceso (ELDE_PIPELINES=%s)",
                       type_inference, ",".join(sorted(permitidos)))
        raise ValueError(f"Tipo de inferencia desconocido: '{type_inference}' "
                         f"(no lo sirve este servidor; ELDE_PIPELINES)")
    gpu = _get_first_gpu()

    if type_inference == "Misters":
        from ..analityc.core.Misters import MistersProcess
        return MistersProcess(
            client_id=client_id,
            model_path=_get_model_path(config, "Misters", "models/base/misters.pt"),
            person_model_path=_get_person_model_path(config, "Misters", _modelo("base/yolo26m.pt")),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            iou_threshold=config.get("iou_threshold", 0.5),
            device=gpu,
        )

    if type_inference == "Autolavado":
        from ..analityc.core.car_washed import VehicleProcessor
        return VehicleProcessor(
            client_id=client_id,
            model_path=_get_model_path(config, "Lavado", "models/base/best.pt"),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    if type_inference == "Hummus":
        # Piso suave a 0.20 (descartar conf<0.20 del modelo evita la basura).
        _hummus_conf = max(0.20, float(config.get("confidence_threshold", 0.25) or 0.25))
        _hummus_iou  = float(config.get("iou_threshold", 0.45) or 0.45)
        # ETAPA 2 — VLM verificador (opcional, activable desde config)
        _hummus_enable_vlm = bool(config.get("enable_vlm", False))
        _hummus_vlm_model  = str(config.get("vlm_model_id", "Qwen/Qwen2-VL-2B-Instruct"))
        _hummus_dwell_order = float(config.get("order_dwell_seconds", 1.5))
        _hummus_dwell_deliv = float(config.get("delivery_dwell_seconds", 1.5))
        from ..analityc.core.Hummus import HummusProcess
        return HummusProcess(
            client_id=client_id,
            model_path=_get_model_path(config, "Hummus", "models/base/1080.pt"),
            person_model_path=_get_person_model_path(config, "Hummus", "models/base/yolo26m.pt"),
            confidence_threshold=_hummus_conf,
            iou_threshold=_hummus_iou,
            device=gpu,
            enable_vlm=_hummus_enable_vlm,
            vlm_model_id=_hummus_vlm_model,
            order_dwell_seconds=_hummus_dwell_order,
            delivery_dwell_seconds=_hummus_dwell_deliv,
        )

    if type_inference == "Perimetrales":
        from ..analityc.core.base_perimeter import BasePerimeter
        return BasePerimeter(
            client_id=client_id,
            model_path=_get_model_path(config, "Perimetrales", _modelo("base/yolo26m.pt")),
            device=gpu,
        )

    if type_inference == "PerimetralesMultiCam":
        # Motor perimetral nuevo (singleton: modelos cargados una sola vez y
        # la identidad global sobrevive reconexiones). Si el paquete no esta
        # disponible se cae al procesador legado.
        if PerimetralNocturnaWS is not None:
            from perimetral_nocturna.adaptador_websocket import get_perimetral_ws
            return get_perimetral_ws()
        from ..analityc.core.perimetrales_multicam import PerimetralesMultiCam
        return PerimetralesMultiCam(
            model_path=_get_model_path(config, "PerimetralesMultiCam", _modelo("base/yolo26m.pt")),
            device=gpu,
            debug=False,
        )

    if type_inference == "VigilanteAmazonas":
        # VIGILANTE-AMAZONAS (singleton: modelos y galeria compartidos entre
        # conexiones; dashboard :8090 y Socket.IO :8091 quedan activos).
        if VigilanteWS is None:
            raise ValueError("vigilante_amazonas no esta disponible en este "
                             "servidor (ver log de arranque)")
        from vigilante_amazonas.adaptador_websocket import get_vigilante_ws
        return get_vigilante_ws()

    if type_inference == "Estacionamiento":
        # ESTACIONAMIENTO (4-ago-2026): la optica vehicular del motor
        # vigilante — solo alertan vehiculos, "estacionado" al superar el
        # umbral, salida con tiempo total y ocupacion en el metadata. Es
        # subclase de VigilanteWS: el despacho de process_image_sync le
        # sirve tal cual (isinstance). La logica vive con el resto de
        # procesadores en analityc/core (un archivo por funcionalidad).
        if VigilanteWS is None:
            raise ValueError("vigilante_amazonas no esta disponible en este "
                             "servidor (ver log de arranque)")
        from ..analityc.core.estacionamiento import get_estacionamiento_ws
        return get_estacionamiento_ws()

    if type_inference == "PerimetralesBoTSORT":
        from ..analityc.core.botsort_wrapper import BoTSORTWrapper
        return BoTSORTWrapper(
            model_path=_get_model_path(config, "PerimetralesBoTSORT", _modelo("base/yolo26m.pt")),
            device=gpu,
            match_thresh=0.45,
            debug=False,
        )

    # "Tiendas" (6-ago-2026) sustituye a "Personal de Amazonas", que describia
    # el pipeline y no lo que el operador cree estar eligiendo. Es el modo del
    # cliente de TIENDA y solo lo ofrece ese cliente.
    #
    # El nombre viejo se retiro por completo: amazonas y managers ya no lo
    # ofrecen, y quien lo tenga guardado en `last_inference` no reconecta con
    # el (los clientes descartan modos que su selector no lista, ver
    # windows_main). Los modos derivados AmazonasStand y PermanenciaArea son
    # aparte y no dependen de este nombre.
    if type_inference == "Tiendas":
        from ..analityc.core.tiendas import TiendasProcess
        return TiendasProcess(
            client_id=client_id,
            # Cosmeticos eliminado: solo personas + genero + edad.
            model_path=None,
            person_model_path=_get_person_model_path(config, type_inference, _modelo("base/yolo26m.pt")),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    if type_inference == "PermanenciaArea":
        # PERMANENCIA EN AREA (6-ago-2026): cronometro por persona dentro del
        # ROI del recuadro, con alertas al entrar, al superar el umbral y al
        # salir (con el tiempo total). Va por WhatsApp y Jarvis reusando la
        # cola de alertas de siempre.
        #
        # Subclase de TiendasProcess: hereda deteccion, BoT-SORT, Re-ID y la
        # maquina DENTRO/FUERA por persistent_id que ya existe y esta
        # calibrada. Solo lee esas transiciones y cuenta el tiempo.
        from ..analityc.core.tiendas import TiendasProcess  # base del despacho
        from ..analityc.core.permanencia_area import PermanenciaArea
        return PermanenciaArea(
            client_id=client_id,
            model_path=None,
            person_model_path=_get_person_model_path(config, "PermanenciaArea", _modelo("base/yolo26m.pt")),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    if type_inference == "Anaqueles":
        # ANAQUELES (13-ago-2026): interacción con los niveles del estante
        # para marketing — visitas y AGARRES por área (muñeca de la pose
        # dentro del polígono), con heatmap y demografía propios por área.
        # Las áreas las dibuja el operador en el cliente de tienda y viajan
        # POR FRAME (patrón PermanenciaArea, no el YAML del stand).
        #
        # Subclase de TiendasProcess: el despacho por isinstance de este
        # archivo le sirve tal cual.
        from ..analityc.core.tiendas import TiendasProcess  # base del despacho
        from ..analityc.core.anaqueles import Anaqueles
        return Anaqueles(
            client_id=client_id,
            model_path=None,
            person_model_path=_get_person_model_path(config, "Anaqueles", _modelo("base/yolo26m.pt")),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    if type_inference == "AmazonasStand":
        # STAND DE DEGUSTACION (5-ago-2026): personas + genero + edad, mas
        # el embudo del stand (paso -> detencion -> atencion -> degustacion),
        # chef excluido y mapa de calor de tres capas.
        #
        # Es SUBCLASE de TiendasProcess, asi que todo el despacho por
        # isinstance de este archivo le sirve tal cual — mismo patron que
        # Estacionamiento sobre VigilanteWS. Modo propio y no un flag dentro
        # de "Personal de Amazonas" porque ese pipeline lo comparte el
        # cliente de Tienda, que no mide ningun stand.
        from ..analityc.core.tiendas import TiendasProcess  # base del despacho
        from ..analityc.core.amazonas_stand import AmazonasStand
        return AmazonasStand(
            client_id=client_id,
            model_path=None,
            person_model_path=_get_person_model_path(config, "AmazonasStand", _modelo("base/yolo26m.pt")),
            confidence_threshold=config["confidence_threshold"],
            iou_threshold=config["iou_threshold"],
            device=gpu,
        )

    raise ValueError(f"Tipo de inferencia desconocido: '{type_inference}'")


# -----------------------------------------------------------------------------
# Procesamiento sincronico de frames (ejecutado en ThreadPoolExecutor)
# -----------------------------------------------------------------------------

def _decode_image(image_data: Any) -> np.ndarray:
    """Decodifica la imagen desde base64-str, bytes o data-URI."""
    if isinstance(image_data, (bytes, bytearray)):
        image_bytes = bytes(image_data)
    elif isinstance(image_data, str):
        if image_data.startswith("data:") and "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        raise ValueError(f"Tipo de imagen no soportado: {type(image_data)}")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen corrupta o formato invalido")
    return img


def _encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Error en la codificacion JPEG de salida")
    return buf.tobytes()


def _apply_track_classes(processor: Any, track_classes: list) -> None:
    """Aplica dinámicamente las clases a detectar en el procesador."""
    if not track_classes:
        return
    classes = [int(c) for c in track_classes if isinstance(c, (int, float))]
    if not classes:
        return

    # BasePerimeter: usa class_ids
    if BasePerimeter is not None and isinstance(processor, BasePerimeter):
        processor.class_ids = classes

    # TiendasProcess: solo personas + demografia (cosmeticos eliminado).
    # processor.model es None, asi que se usa directamente la lista de
    # clases COCO que envia el cliente (por defecto [0] = Persona).
    elif TiendasProcess is not None and isinstance(processor, TiendasProcess):
        if hasattr(processor, 'model') and processor.model and hasattr(processor.model, 'names'):
            full = list(processor.model.names.keys())
            processor.all_classes = full
            processor.staff_names = {
                cid: processor.model.names[cid].replace('_', ' ').title()
                for cid in full
            }
            processor.staff_names[-1] = 'Persona'
        else:
            processor.all_classes = classes

    # BoTSORTWrapper / PerimetralesMultiCam: detectan solo personas por defecto
    elif hasattr(processor, 'detect_classes'):
        processor.detect_classes = classes


def segundos_de_permanencia(crudo: Any) -> Optional[float]:
    """Sanea los segundos que manda el recuadro antes de alertar.

    Viene de la red, asi que un texto o un negativo no pueden llegar al
    rastreador. Devuelve None cuando el cliente NO maneja el campo (misma
    convencion que el local: None = no tocar nada), y acota a [0, 3600] —
    el 0 es legitimo (alertar de inmediato) y por eso se compara con None,
    nunca por falsy.
    """
    if crudo is None:
        return None
    try:
        return max(0.0, min(3600.0, float(crudo)))
    except (TypeError, ValueError):
        # Un cliente manipulado o una version futura con otro formato no
        # deben tumbar el frame: se ignora el ajuste y rige config.
        return None


def process_image_sync(
    processor: Any,
    img: np.ndarray,
    roi: Any,
    roi_activate: bool,
    door_roi: Any = None,
    door_activate: bool = False,
    door_direction: Any = None,
    door_direction_activate: bool = False,
    pay_roi: Any = None,
    withdraw_roi: Any = None,
    pay_roi_activate: bool = True,
    withdraw_roi_activate: bool = True,
    camera_id: Any = 1,
    track_classes: Any = None,
    order_zone: Any = None,
    order_zone_activate: bool = False,
    delivery_zone: Any = None,
    delivery_zone_activate: bool = False,
    personal_zone: Any = None,
    personal_zone_activate: bool = False,
    # Zona del PERSONAL del comedor (gris): quien la ocupa queda fuera del
    # conteo de clientes. Va al final del bloque de zonas para no correr los
    # argumentos posicionales de `frame_payload`.
    enable_vlm: Optional[bool] = None,
    camera_angle: Optional[str] = None,
    heatmap_activate: bool = False,
    camera_name: Optional[str] = None,
    draw_server: bool = True,
    establecimiento: Optional[str] = None,
    leer_placas: bool = False,
    flecha_direccion: Any = None,
    linea_cruce: Any = None,
    area_nombre: Optional[str] = None,
    anaqueles_zonas: Any = None,
    anaqueles_nombres: Any = None,
    anaqueles_activate: bool = False,
    # Segundos DENTRO del area antes de alertar, elegidos en el recuadro
    # (19-ago-2026). None = el cliente no maneja el campo: manda el config
    # del servidor. Van AL FINAL porque `frame_payload` es una tupla
    # POSICIONAL: insertar en medio corre todos los argumentos siguientes.
    min_permanencia_personas: Optional[float] = None,
    min_permanencia_vehiculos: Optional[float] = None,
    # Con que herramienta dibujo el operador el umbral: 'linea' o
    # 'puerta' (la flecha doble entrada/salida del recuadro, 5-sep-2026).
    # NO entra en ninguna cuenta: viaja hasta la alerta como
    # `umbral_dibujado_con` para que el registro conteste que veia el
    # operador si algun dia hay que revisar una calibracion al reves.
    linea_cruce_estilo: Optional[str] = None,
    # Puertas del recuadro (7-sep-2026): hasta 8 umbrales
    # `[[x1,y1],[x2,y2]]` en pixeles del frame, cada uno un sitio con su
    # nombre (Almacen, Patio...), mas la lista PARALELA de rotulos y el
    # interruptor. Tambien AL FINAL, y por lo mismo que los de arriba:
    # `frame_payload` viaja SIN nombres hasta `_full_frame_sync`, asi que
    # colar uno en medio corre todos los siguientes y el servidor pasa a
    # leer los rotulos de las puertas como si fueran las lineas.
    # `puertas_lineas=None` —cliente viejo, nada dibujado o interruptor
    # apagado— es EXACTAMENTE el camino de siempre: aforo de la camara.
    puertas_lineas: Any = None,
    puertas_nombres: Any = None,
    puertas_activate: bool = False,
    # Productos dibujados DENTRO de las areas del anaquel (25-sep-2026):
    # lista de `{anaquel, nombre, poligono}`. AL FINAL por lo mismo que las
    # puertas: `frame_payload` es posicional. Solo lo lee Anaqueles.
    anaqueles_productos: Any = None,
    # Áreas a ignorar (12-oct-2026): quien queda entero dentro no se
    # cuenta (`zonas_ignorar.py`). AL FINAL: `frame_payload` es posicional.
    zonas_ignorar: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Despacha el frame al procesador correcto.
    Se ejecuta en el ThreadPoolExecutor, nunca en el event loop.
    """
    # Aplicar clases dinámicas al procesador si las envía el cliente
    if track_classes is not None and isinstance(track_classes, list):
        _apply_track_classes(processor, track_classes)

    # VIGILANTE-AMAZONAS: deteccion 6 clases + Re-ID + permanencia en el area.
    if VigilanteWS is not None and isinstance(processor, VigilanteWS):
        return processor.process_frame(
            img, camera_id,
            camera_name=camera_name,
            track_classes=track_classes if isinstance(track_classes, list) else None,
            draw=bool(draw_server),
            roi=roi,                       # poligono del area (contador/alertas)
            roi_activate=bool(roi_activate),
            establecimiento=establecimiento,   # local de la camara (WhatsApp)
            # Umbral de alerta de ESTA camara (contadores del recuadro).
            min_permanencia_personas=min_permanencia_personas,
            min_permanencia_vehiculos=min_permanencia_vehiculos,
            leer_placas=leer_placas,           # boton "Placas" del recuadro
            # La PUERTA del recuadro (9-sep-2026). Hasta hoy esta llamada no
            # los pasaba: `umbrales_del_payload` los sacaba del payload unas
            # lineas mas arriba y aqui morian como variables locales, asi que
            # el operador dibujaba el porton y no pasaba NADA — sin un error
            # en el log ni un campo rechazado por el contrato. Los demas
            # procesadores si los recibian; este era el unico agujero.
            linea_cruce=linea_cruce,
            # El interruptor del recuadro YA se aplico en
            # `umbrales_del_payload`: con la puerta apagada, `linea_cruce`
            # llega None. Aqui llegar con geometria significa activa.
            linea_cruce_activate=True,
            linea_cruce_estilo=linea_cruce_estilo,
            puertas_lineas=puertas_lineas,
            puertas_nombres=puertas_nombres,
            puertas_activate=puertas_activate,
        )

    # Motor perimetral NUEVO: detections (modo directo) + alerts con el
    # contrato llegada/salida que renderiza el cliente perimetrales-view.
    if PerimetralNocturnaWS is not None and isinstance(processor, PerimetralNocturnaWS):
        return processor.process_frame(
            img, camera_id,
            camera_name=camera_name,
            track_classes=track_classes if isinstance(track_classes, list) else None,
            draw=bool(draw_server),
        )

    # H-21 (gemelo de H-20, salio al verificar aquel): VehicleProcessor
    # tampoco tenia rama propia y la generica lo llamaba con 8 argumentos
    # aceptando (image, roi, send_to_server). El modo Autolavado fallaba en
    # cada frame igual. Devuelve (frame, metadata) directamente.
    if VehicleProcessor is not None and isinstance(processor, VehicleProcessor):
        # UNA conexion websocket transporta los 16 recuadros del cliente: sin
        # aislar, este procesador unico veria las escenas ENTRELAZADAS (H-39).
        cam_proc = processor.get_camera_processor(camera_id)
        if track_classes is not None and isinstance(track_classes, list):
            _apply_track_classes(cam_proc, track_classes)
        # Como las ramas hermanas (VigilanteWS/TiendasProcess): el LOCAL y el
        # nombre de la camara viajan al procesador para que el mensaje del
        # lavado diga el establecimiento, no el titulo de la ventana de iVMS.
        # Van al SUB-procesador: en la raiz saldria el local de otra camara.
        cam_proc._establecimiento = establecimiento
        cam_proc._camera_name = camera_name
        return cam_proc.process_frame(img, roi=roi if roi_activate else None)

    # H-20: BoTSORTWrapper no tenia rama propia y caia en la generica, que lo
    # llamaba con 5 argumentos cuando su firma acepta (frame, camera_id). El
    # modo PerimetralesBoTSORT fallaba en CADA frame desde que las firmas
    # divergieron. Devuelve pistas con la misma forma que PerimetralesMultiCam,
    # asi que se dibujan igual — salvo `conf`, que aqui puede ser None.
    if BoTSORTWrapper is not None and isinstance(processor, BoTSORTWrapper):
        tracks = processor.process_frame(img, camera_id)
        for t in tracks:
            x1, y1, x2, y2 = t.get("bbox", (0, 0, 0, 0))
            gid = t.get("global_id", 0)
            conf = t.get("conf")
            cam = t.get("camera_id", camera_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = (f"ID:{gid} CAM:{cam} {conf:.2f}" if conf is not None
                     else f"ID:{gid} CAM:{cam}")
            y_text = max(10, y1 - 6)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y_text - th - bl)), (x1 + tw, y_text + bl), (0, 255, 255), -1)
            cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img, {"tracks": tracks}

    if PerimetralesMultiCam is not None and isinstance(processor, PerimetralesMultiCam):
        tracks = processor.process_frame(
            img, camera_id,
            pay_roi=pay_roi, withdraw_roi=withdraw_roi,
            pay_roi_activate=pay_roi_activate, withdraw_roi_activate=withdraw_roi_activate,
        )
        for t in tracks:
            x1, y1, x2, y2 = t.get("bbox", (0, 0, 0, 0))
            gid = t.get("global_id", 0)
            conf = t.get("conf", 0.0)
            cam = t.get("camera_id", camera_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{gid} CAM:{cam} {conf:.2f}"
            y_text = max(10, y1 - 6)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y_text - th - bl)), (x1 + tw, y_text + bl), (0, 255, 255), -1)
            cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img, {"tracks": tracks}

    if TiendasProcess is not None and isinstance(processor, TiendasProcess):
        try:
            cam_proc = processor.get_camera_processor(camera_id)
            # Perfil de angulo por camara enviado por el cliente (opcional):
            # frontal|lateral|cenital. Ajusta el filtro de aspecto en cenital.
            if camera_angle:
                cam_proc.set_camera_angle(camera_angle)
            # Si el cliente no manda nombre, se usa el camera_id: asi las
            # capturas quedan etiquetadas con la camara REAL y no con "cam"
            # generico, que era lo que impedia el desglose por pasillo en el
            # dashboard de tienda (los heatmaps ya usaban el camera_id).
            cam_proc._camera_display_name = camera_name or str(camera_id)
            cam_proc._draw_server = bool(draw_server)
            # Flecha "hacia adentro" del recuadro. Va como atributo, igual que
            # el nombre de la camara, para no cambiar la firma de
            # process_frame, que comparten todos los modos de esta familia.
            # Solo la lee PermanenciaArea; a los demas les sobra sin molestar.
            cam_proc._flecha_direccion = flecha_direccion
            # Línea de cruce (10-ago-2026): [[x1,y1],[x2,y2]] en pixeles.
            # DENTRO queda a la DERECHA caminando de P1 a P2. Igual que la
            # flecha, va como atributo y solo la lee PermanenciaArea.
            cam_proc._linea_cruce = linea_cruce
            cam_proc._linea_cruce_estilo = linea_cruce_estilo
            # Local y nombre del area de ESTA camara: encabezan el texto de
            # la alerta de permanencia («Cámaras {local}» / «Permanencia en
            # {area}»). El nombre del area lo escribe el operador; sin el,
            # NO se usa el titulo de la ventana capturada (iVMS).
            cam_proc._establecimiento = establecimiento
            cam_proc._area_nombre = area_nombre
            # Áreas del anaquel (13-ago-2026): N polígonos en píxeles del
            # frame + nombres paralelos. Como atributos, igual que la flecha:
            # solo los lee el pipeline Anaqueles; al resto le sobran sin
            # molestar.
            cam_proc._anaqueles_zonas = anaqueles_zonas
            cam_proc._anaqueles_nombres = anaqueles_nombres
            cam_proc._anaqueles_activate = bool(anaqueles_activate)
            # Puertas (7-sep-2026): N umbrales con nombre, uno por sitio.
            # Como la linea suelta, viajan de atributo y solo los lee
            # PermanenciaArea. Se cuelgan en LAS DOS ramas de este try a
            # proposito: esta es la MULTICAMARA y la del `except` es la
            # simple. Colgarlos en una sola dejaria a la otra mitad de las
            # camaras sin ver ninguna puerta, cayendo al aforo unico de la
            # camara sin un solo error que lo dijera.
            cam_proc._puertas_lineas = puertas_lineas
            cam_proc._puertas_nombres = puertas_nombres
            cam_proc._puertas_activate = bool(puertas_activate)
            # Productos del anaquel (25-sep-2026): en LAS DOS ramas, por el
            # mismo motivo que las puertas.
            cam_proc._anaqueles_productos = anaqueles_productos
            cam_proc._zonas_ignorar = zonas_ignorar
            return cam_proc.process_frame(img, roi, roi_activate, camera_id, heatmap_activate=heatmap_activate)
        except Exception:
            if camera_angle:
                processor.set_camera_angle(camera_angle)
            processor._camera_display_name = camera_name or str(camera_id)
            processor._draw_server = bool(draw_server)
            processor._flecha_direccion = flecha_direccion
            processor._linea_cruce = linea_cruce
            processor._linea_cruce_estilo = linea_cruce_estilo
            processor._establecimiento = establecimiento
            processor._area_nombre = area_nombre
            processor._anaqueles_zonas = anaqueles_zonas
            processor._anaqueles_nombres = anaqueles_nombres
            processor._anaqueles_activate = bool(anaqueles_activate)
            # Las MISMAS puertas que en la rama de arriba (7-sep-2026).
            # Esta es la rama SIMPLE: el procesador de una sola camara.
            # Si solo una de las dos las cuelga, la otra pierde el modo
            # entero con el video perfecto y nadie se entera.
            processor._puertas_lineas = puertas_lineas
            processor._puertas_nombres = puertas_nombres
            processor._puertas_activate = bool(puertas_activate)
            processor._anaqueles_productos = anaqueles_productos
            processor._zonas_ignorar = zonas_ignorar
            return processor.process_frame(img, roi, roi_activate, camera_id, heatmap_activate=heatmap_activate)

    # HummusProcess: ROI personas (amarillo) + zonas Toma_De_Orden (azul) y Entrega_De_Plato (rojo)
    if HummusProcess is not None and isinstance(processor, HummusProcess):
        effective_roi = roi if (isinstance(roi, list) and len(roi) >= 3) else None
        eff_order   = order_zone    if (isinstance(order_zone, list)    and len(order_zone)    >= 3) else None
        eff_deliv   = delivery_zone if (isinstance(delivery_zone, list) and len(delivery_zone) >= 3) else None
        eff_personal = personal_zone if (isinstance(personal_zone, list) and len(personal_zone) >= 3) else None
        # UNA conexion websocket transporta los 16 recuadros del cliente, asi
        # que este `processor` unico veria las escenas ENTRELAZADAS en un solo
        # tracker (medido en el repo: 0/40 frames con tracks, sin un error).
        # Cada camara recibe el suyo, compartiendo los modelos pesados —el
        # mismo patron que TiendasProcess (H-39).
        cam_proc = processor.get_camera_processor(camera_id)
        if track_classes is not None and isinstance(track_classes, list):
            _apply_track_classes(cam_proc, track_classes)
        return cam_proc.process_frame(
            img,
            roi=effective_roi, roi_active=bool(roi_activate),
            order_zone=eff_order,       order_zone_active=bool(order_zone_activate),
            delivery_zone=eff_deliv,    delivery_zone_active=bool(delivery_zone_activate),
            personal_zone=eff_personal, personal_zone_active=bool(personal_zone_activate),
            enable_vlm=enable_vlm,
            send_to_server=False,
        )

    # MistersProcess: misma lógica de ROI
    if MistersProcess is not None and isinstance(processor, MistersProcess):
        effective_roi = roi if (roi_activate and isinstance(roi, list) and len(roi) >= 3) else None
        # Un procesador por camara (H-39): ver el comentario de Hummus arriba.
        cam_proc = processor.get_camera_processor(camera_id)
        if track_classes is not None and isinstance(track_classes, list):
            _apply_track_classes(cam_proc, track_classes)
        return cam_proc.process_frame(img, effective_roi, send_to_server=False)

    # BasePerimeter (modo "Perimetrales"): hasta hoy caia en la rama generica
    # de abajo, que NI SIQUIERA le pasaba camera_id. Con un solo procesador
    # para los 16 recuadros, la geometria (ROI y puerta) se reescribia en
    # CADA frame con la del ultimo recuadro que llegara, y los contadores
    # sumaban los vehiculos de todas las camaras en un unico marcador.
    # La guarda `is not None` no es cosmetica: los procesadores se importan
    # PEREZOSOS (BasePerimeter nace en None y solo se rellena cuando alguien
    # construye un Perimetrales), y `isinstance(x, None)` es TypeError. Hoy
    # no se alcanza por poco —PermanenciaArea, AmazonasStand y Anaqueles
    # heredan de TiendasProcess y retornan antes—, pero es una trampa armada.
    if BasePerimeter is not None and isinstance(processor, BasePerimeter):
        cam_proc = processor.get_camera_processor(camera_id)
        if track_classes is not None and isinstance(track_classes, list):
            _apply_track_classes(cam_proc, track_classes)
        return cam_proc.process_frame(
            img, roi, roi_activate,
            door_roi, door_activate, door_direction, door_direction_activate,
            camera_id=camera_id,
        )

    # Procesadores legacy genericos
    return processor.process_frame(
        img, roi, roi_activate,
        door_roi, door_activate, door_direction, door_direction_activate,
    )


def umbral_del_payload(data: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """El umbral de cruce que trae el recuadro: `(linea, estilo)`.

    La linea es `[[x1,y1],[x2,y2]]` en pixeles del frame (DENTRO a la
    DERECHA caminando de P1 a P2; invertirla = intercambiar los puntos).
    `estilo` dice con QUE la dibujo el operador —"linea" o "puerta", la
    flecha doble entrada/salida del recuadro (5-sep-2026)— y no pesa en
    ninguna decision: solo acompana a la alerta.

    Dos reglas que no son obvias:

    - `linea_cruce_activate` AUSENTE cuenta como True. Los clientes
      anteriores a la puerta direccional mandan la linea sin ese campo;
      exigirlo apagaria la lectura que ya usan. Con el campo a False el
      umbral esta dibujado pero desactivado: vuelve el area clasica.
    - Una linea vacia tambien apaga la lectura (el operador la borro).
    """
    linea = (data or {}).get("linea_cruce") or None
    if not bool((data or {}).get("linea_cruce_activate", True)):
        linea = None
    estilo = (data or {}).get("linea_cruce_estilo") or None
    return linea, (str(estilo) if estilo is not None else None)


def puertas_del_payload(data: Dict[str, Any]) -> Tuple[Any, Any, bool]:
    """Las puertas dibujadas en el recuadro: `(lineas, nombres, activado)`.

    `lineas` es la lista de umbrales `[[x1,y1],[x2,y2]]` en pixeles del
    frame (7-sep-2026). Cada POSICION de la lista es un sitio distinto
    —«Almacen», «Patio»— y esa posicion es su identidad; `nombres` solo
    rotula. Misma convencion que `linea_cruce`: DENTRO queda a la DERECHA
    caminando de P1 a P2, e invertirla es intercambiar los dos puntos.

    Las reglas que no son obvias, con el fallo que evita cada una:

    - Campo AUSENTE y lista VACIA valen lo mismo: `None`, o sea NINGUNA
      puerta, que es el camino de siempre (el aforo de la camara). La
      lista vacia llega en CADA frame de todo recuadro que no dibujo
      puertas, que hoy son casi todos: tratarla como un error los dejaria
      sin video.
    - `puertas_activate` AUSENTE cuenta como True, igual que
      `linea_cruce_activate`: un cliente que manda las lineas sin el
      interruptor las quiere. Con el interruptor a False las puertas
      siguen dibujadas pero no rigen, y vuelve el aforo de la camara.
    - El tercer valor es ese interruptor YA RESUELTO contra lo dibujado:
      True solo si queda al menos una puerta rigiendo. Devolver el True
      crudo del ausente le pondria `_puertas_activate = True` a TODOS los
      clientes viejos —que no mandan ni el campo ni las lineas— y quien
      mirase solo el interruptor cambiaria de modo sin una sola puerta.
    - Lo que no sea una lista se ignora: pasarle una cadena al gestor lo
      dejaria iterando letras y montando una puerta por caracter.
    - Aqui NO se revisa la forma de cada segmento ni el tope de 8. Eso lo
      hace `GestorPuertas`, que sabe de que camara habla y puede saltarse
      UN umbral ilegible conservando el indice de los demas; descartarlo
      aqui colapsaria la lista y correria las identidades, que es
      exactamente mover de sitio a la gente ya contada dentro.
    - `nombres` no se cruza con `lineas`: un desfase de rotulos es
      cosmetico, y convertirlo en frame descartado seria mucho peor.
    """
    crudas = (data or {}).get("puertas_lineas")
    lineas = (list(crudas)
              if isinstance(crudas, (list, tuple)) and crudas else None)
    if not bool((data or {}).get("puertas_activate", True)):
        lineas = None
    if lineas is None:
        return None, None, False
    crudos = (data or {}).get("puertas_nombres")
    nombres = list(crudos) if isinstance(crudos, (list, tuple)) else None
    return lineas, nombres, True


def umbrales_del_payload(
        data: Dict[str, Any]) -> Tuple[Any, Optional[str], Any, Any, bool]:
    """Todos los umbrales del frame con la precedencia YA aplicada:
    `(linea, estilo, puertas_lineas, puertas_nombres, puertas_activate)`.

    `puertas_lineas` MANDA sobre `linea_cruce`: en cuanto llega una puerta,
    el umbral suelto se ignora. No es una preferencia estetica — el
    operador dibuja las puertas ENCIMA de la linea que ya tenia, y medir el
    mismo umbral dos veces (una en el aforo de la camara y otra en la
    puerta) emitiria DOS entradas por cada persona que pasa UNA vez.

    La decision vive aqui y no en el contrato (7-sep-2026) porque los dos
    campos son validos y pueden llegar en el MISMO payload: el contrato no
    puede rechazar ninguno de los dos; quien elige es quien reparte el
    frame. `estilo` no se toca al ignorar la linea: es un rotulo cosmetico
    que solo llega a la alerta cuando rige una lectura de umbral.
    """
    linea, estilo = umbral_del_payload(data)
    puertas, nombres, activado = puertas_del_payload(data)
    if puertas:
        linea = None
    return linea, estilo, puertas, nombres, activado


def _full_frame_sync(
    processor: Any,
    image_data: Any,
    roi: Any,
    roi_activate: bool,
    door_roi: Any,
    door_activate: bool,
    door_direction: Any,
    door_direction_activate: bool,
    pay_roi: Any,
    withdraw_roi: Any,
    pay_roi_activate: bool,
    withdraw_roi_activate: bool,
    camera_id: Any,
    track_classes: Any = None,
    order_zone: Any = None,
    order_zone_activate: bool = False,
    delivery_zone: Any = None,
    delivery_zone_activate: bool = False,
    personal_zone: Any = None,
    personal_zone_activate: bool = False,
    enable_vlm: Optional[bool] = None,
    camera_angle: Optional[str] = None,
    heatmap_activate: bool = False,
    camera_name: Optional[str] = None,
    draw_server: bool = True,
    establecimiento: Optional[str] = None,
    leer_placas: bool = False,
    flecha_direccion: Any = None,
    linea_cruce: Any = None,
    area_nombre: Optional[str] = None,
    anaqueles_zonas: Any = None,
    anaqueles_nombres: Any = None,
    anaqueles_activate: bool = False,
    # Segundos DENTRO del area antes de alertar, elegidos en el recuadro
    # (19-ago-2026). None = el cliente no maneja el campo: manda el config
    # del servidor. Van AL FINAL porque `frame_payload` es una tupla
    # POSICIONAL: insertar en medio corre todos los argumentos siguientes.
    min_permanencia_personas: Optional[float] = None,
    min_permanencia_vehiculos: Optional[float] = None,
    # Con que herramienta dibujo el operador el umbral: 'linea' o
    # 'puerta' (la flecha doble entrada/salida del recuadro, 5-sep-2026).
    # NO entra en ninguna cuenta: viaja hasta la alerta como
    # `umbral_dibujado_con` para que el registro conteste que veia el
    # operador si algun dia hay que revisar una calibracion al reves.
    linea_cruce_estilo: Optional[str] = None,
    # Puertas del recuadro (7-sep-2026). AL FINAL y en el MISMO orden
    # que en la tupla y que en `process_image_sync`: este es el salto
    # que se hace SIN nombres (`_full_frame_sync(*payload)`), y ahi una
    # permuta no falla, solo empieza a leer mal en silencio.
    puertas_lineas: Any = None,
    puertas_nombres: Any = None,
    puertas_activate: bool = False,
    # Productos del anaquel (25-sep-2026). AL FINAL, y en el mismo orden que
    # en la tupla y en `process_image_sync`.
    anaqueles_productos: Any = None,
    # Áreas a ignorar (12-oct-2026). AL FINAL, como los productos.
    zonas_ignorar: Any = None,
) -> Dict[str, Any]:
    """
    Combina decode + inferencia + encode JPEG en una sola llamada al executor.
    El event loop no se bloquea en ningun paso de CPU.

    draw_server=False -> MODO DIRECTO: el servidor NO dibuja ni codifica la
    imagen; devuelve solo las detecciones en metadata y el cliente las pinta
    sobre su propio frame (mucha menos latencia/ancho de banda).
    """
    t0 = time.time()
    img = _decode_image(image_data)
    processed_img, metadata = process_image_sync(
        processor, img, roi, roi_activate,
        door_roi, door_activate, door_direction, door_direction_activate,
        pay_roi, withdraw_roi, pay_roi_activate, withdraw_roi_activate, camera_id,
        track_classes=track_classes,
        order_zone=order_zone, order_zone_activate=order_zone_activate,
        delivery_zone=delivery_zone, delivery_zone_activate=delivery_zone_activate,
        personal_zone=personal_zone, personal_zone_activate=personal_zone_activate,
        enable_vlm=enable_vlm,
        camera_angle=camera_angle,
        heatmap_activate=heatmap_activate,
        camera_name=camera_name,
        draw_server=draw_server,
        establecimiento=establecimiento,
        leer_placas=leer_placas,
        flecha_direccion=flecha_direccion,
        linea_cruce=linea_cruce,
        area_nombre=area_nombre,
        anaqueles_zonas=anaqueles_zonas,
        anaqueles_nombres=anaqueles_nombres,
        anaqueles_activate=anaqueles_activate,
        min_permanencia_personas=min_permanencia_personas,
        min_permanencia_vehiculos=min_permanencia_vehiculos,
        linea_cruce_estilo=linea_cruce_estilo,
        puertas_lineas=puertas_lineas,
        puertas_nombres=puertas_nombres,
        puertas_activate=puertas_activate,
        anaqueles_productos=anaqueles_productos,
        zonas_ignorar=zonas_ignorar,
    )
    # Mapa de calor GLOBAL (1-ago-2026): TODAS las camaras acumulan, no solo
    # el pipeline de visitantes. Come las cajas del MISMO metadata que viaja
    # al cliente; si el pipeline ya acumulo por dentro (trae `heatmap` en el
    # metadata, caso TiendasProcess), aqui no se estampa de nuevo. En modo
    # clasico algunas ramas dibujan sobre `img`, asi que el fondo del PNG
    # puede llevar cajas pintadas; en modo directo (el real hoy) va limpio.
    _heatmaps.acumular_desde_metadata(camera_id, camera_name, img, metadata)
    processing_time = round(time.time() - t0, 3)

    # Un fallo ATRAPADO por el pipeline sale como `error`, no como un frame
    # bueno sin detecciones: eso congelaba al cliente en silencio (17-sep-2026,
    # ver fallo_imagen_IA_pipeline.py).
    fallo = _fallo.respuesta_si_fallo(metadata, camera_id, processing_time)
    if fallo is not None:
        return fallo


    # MODO DIRECTO: no se codifica ni envia la imagen (el cliente la dibuja).
    if not draw_server:
        return {
            "camera_id": camera_id,
            "status": "success",
            "metadata": metadata,
            "processed_image": None,
            "processing_time": processing_time,
        }

    jpeg_bytes = _encode_jpeg(processed_img)
    return {
        "camera_id": camera_id,
        "status": "success",
        "metadata": metadata,
        "processed_image": f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode()}",
        "processing_time": processing_time,
    }


# -----------------------------------------------------------------------------
# Worker por cliente
# -----------------------------------------------------------------------------
# El worker de futures que vivia aqui (`ClientWorker`) se sustituyo el
# 7-ago-2026 por `ws_drenaje.DrenajeCliente`: el bucle del endpoint esperaba
# cada resultado antes de volver a leer el socket, la contrapresion llenaba el
# buffer del websocket y uvicorn cerraba conexiones VIVAS con 1011 cada ~3
# minutos (y cada reconexion reiniciaba los contadores de los modos). La
# historia completa esta en el docstring de ws_drenaje.py.

#: Construir un procesador carga modelos (segundos, y ~1 min en frio). Se hace
#: FUERA del event loop para no congelar a los demas clientes en cada
#: (re)conexion, y de uno en uno: las libs de modelos comparten caches que no
#: estan pensadas para cargas concurrentes.
_candado_construccion = asyncio.Lock()


# -----------------------------------------------------------------------------
# Endpoint WebSocket
# -----------------------------------------------------------------------------

@app.websocket("/ws/{type_inference}")
async def websocket_endpoint(websocket: WebSocket, type_inference: str):
    await websocket.accept()
    client_id = f"client_{id(websocket)}"
    worker: Optional[DrenajeCliente] = None
    processor: Any = None
    #: Lector y drenador escriben en el MISMO websocket; sin candado, un ping
    #: del lector podria intercalarse con una respuesta del drenador.
    candado_envio = asyncio.Lock()

    try:
        # -- 1. Inicializacion del procesador ----------------------------------
        config = get_config()
        try:
            # En el executor, no en el event loop: cargar modelos tarda
            # segundos y congelaba a TODOS los clientes en cada (re)conexion.
            async with _candado_construccion:
                processor = await asyncio.get_running_loop().run_in_executor(
                    None, _build_processor, type_inference, client_id, config)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            # RuntimeError: crear_detector no pudo cargar RF-DETR (pesos o
            # paquete). Sin esto caia al except generico y el cliente veia
            # una desconexion muda + bucle de reconexion (3-sep-2026).
            # FileNotFoundError: un procesador cuyo peso no esta en disco. Es
            # el caso del comedor —models/base/1080.pt no existe en ningun
            # repositorio (8-sep-2026)—: sin esto el cliente se queda otra vez
            # sin mensaje de error y solo ve la desconexion.
            await websocket.send_text(json.dumps({"status": "error", "message": str(exc)}))
            await websocket.close(code=1008)
            return

        worker = DrenajeCliente(
            client_id, websocket,
            procesar=_full_frame_sync,
            executor_getter=_get_live_executor,
            candado_envio=candado_envio,
            reenviar_whatsapp=_whatsapp.reenviar,
            registrar_saliente=_captura.registrar,
        )
        worker.start()

        async with connection_lock:
            active_connections[client_id] = {
                "websocket": websocket,
                "processor": processor,
                "worker": worker,
                "type_inference": type_inference,
                "connected_at": time.time(),
                "last_active": time.time(),
            }

        logger.info("Cliente conectado: %s tipo=%s", client_id, type_inference)

        await websocket.send_text(json.dumps({
            "id_connection": id(websocket),
            "event": "connection_init",
            "type_inference": type_inference,
            "data": {"roi": False},
        }))

        # -- 2. Bucle principal ------------------------------------------------
        while True:
            try:
                recv = await asyncio.wait_for(websocket.receive(), timeout=WEBSOCKET_TIMEOUT)
            except asyncio.TimeoutError:
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        async with candado_envio:
                            await websocket.send_text(json.dumps({"status": "ping"}))
                    except Exception:
                        break
                continue

            # Deserializar (msgpack o JSON)
            incoming_is_binary = recv.get("bytes") is not None
            try:
                if incoming_is_binary:
                    request = msgpack.unpackb(recv["bytes"], raw=False, strict_map_key=False)
                else:
                    request = json.loads(recv.get("text") or "")
            except Exception as exc:
                logger.warning("Error deserializando mensaje client=%s: %s", client_id, exc)
                async with candado_envio:
                    await _send_error(websocket, incoming_is_binary, str(exc))
                continue

            # Captura de payloads para el contrato (HITO 3). Apagada salvo que
            # ELDE_CAPTURA_PAYLOADS=1; guarda una muestra por forma distinta de
            # mensaje, sin binarios. Nunca lanza.
            _captura.registrar("entrante", type_inference, request)

            # Validacion del contrato (HITO 3), detras de la capa de
            # compatibilidad: entiende el formato antiguo y el nuevo. Arranca
            # en modo 'observar' (registra lo que rechazaria y deja pasar), asi
            # que activarla NO cambia el comportamiento hasta que se ponga
            # ELDE_VALIDAR_CONTRATO=estricto.
            # Registro de dispositivos (HITO 8): apunta que camaras existen.
            # Va aqui, junto a la validacion, porque mira el MISMO mensaje. No
            # se pudo construir antes de cerrar H-11: con un `camera_id`
            # aleatorio por sesion habria acumulado un dispositivo nuevo cada
            # vez que se abria un cliente. Nunca lanza.
            _registro.anotar(request, type_inference)

            _ok, _motivo = _validacion.revisar(request)
            if not _ok:
                logger.warning("contrato rechaza el mensaje client=%s: %s",
                               client_id, _motivo)
                async with candado_envio:
                    await _send_error(websocket, incoming_is_binary,
                                      f"contrato: {_motivo}")
                continue

            data = request.get("data", {})
            if not isinstance(data, dict):
                async with candado_envio:
                    await _send_error(websocket, incoming_is_binary, "Campo 'data' invalido o ausente")
                continue

            if "image" not in data:
                async with candado_envio:
                    await _send_error(websocket, incoming_is_binary, "Campo 'image' requerido")
                continue

            # Extraer y normalizar campos
            image_data = data["image"]
            roi = data.get("roi_coordinates", "")
            roi_activate = bool(data.get("roi_activate", False))
            door_roi = data.get("door_roi_coordinates")
            door_activate = bool(data.get("door_roi_activate", False))
            door_direction = data.get("door_direction")
            door_direction_activate = data.get("door_direction_activate", False)
            # Flecha "hacia adentro" de PermanenciaArea (modo puerta):
            # [[x_cola,y_cola],[x_punta,y_punta]] en pixeles del frame. Lista
            # vacia o ausente = el operador no la marco, y el detector cae a
            # su heuristica de nacimiento/muerte del track. No se inventa una.
            flecha_direccion = data.get("flecha_direccion") or None
            # Línea de cruce de PermanenciaArea (lectura 'linea'):
            # [[x1,y1],[x2,y2]] en pixeles del frame; DENTRO a la derecha de
            # P1→P2. Lista vacía o ausente = no dibujada.
            # Y las PUERTAS del recuadro (7-sep-2026), con la precedencia
            # ya resuelta: si llegan puertas, la línea suelta se IGNORA
            # —medir el mismo umbral dos veces emitiría DOS entradas por
            # cada paso—. El porqué completo, en `umbrales_del_payload`.
            (linea_cruce, linea_cruce_estilo,
             puertas_lineas, puertas_nombres,
             puertas_activate) = umbrales_del_payload(data)
            # Nombre del area escrito por el operador (p.ej. "el pasillo de
            # cooler"): encabeza la alerta de permanencia. Sin el, el texto
            # cae a "el área" — nunca al titulo de la ventana de iVMS.
            area_nombre = data.get("area_nombre") or None
            # Áreas del anaquel (modo Anaqueles): lista de N polígonos
            # [[x,y],...] en píxeles del frame + nombres paralelos. Lista
            # vacía o ausente = el operador no dibujó áreas; no se inventan.
            anaqueles_zonas = data.get("anaqueles_zonas") or None
            anaqueles_nombres = data.get("anaqueles_nombres") or None
            anaqueles_activate = bool(data.get("anaqueles_activate", False))
            # Productos dibujados dentro de esas áreas (25-sep-2026): solo
            # reparten los agarres del área. Ausente = conteo por área de
            # siempre.
            anaqueles_productos = data.get("anaqueles_productos") or None
            # Áreas a ignorar (12-oct-2026). Ausente = no se ignora nada.
            zonas_ignorar = data.get("zonas_ignorar") or None

            # Compatibilidad: door_direction_activate como lista de puntos
            if isinstance(door_direction_activate, (list, tuple)) and door_direction is None:
                door_direction = door_direction_activate
                door_direction_activate = True

            pay_roi = data.get("pay_roi")
            withdraw_roi = data.get("withdraw_roi")
            pay_roi_activate = bool(data.get("pay_roi_activate", True))
            withdraw_roi_activate = bool(data.get("withdraw_roi_activate", True))

            # Zonas Hummus específicas por clase
            order_zone = data.get("order_zone_coordinates")
            order_zone_activate = bool(data.get("order_zone_activate", False))
            delivery_zone = data.get("delivery_zone_coordinates")
            delivery_zone_activate = bool(data.get("delivery_zone_activate", False))
            # Zona del personal del comedor (la dibuja el operador igual que
            # las otras dos). Ausente en clientes que no la ofrecen.
            personal_zone = data.get("personal_zone_coordinates")
            personal_zone_activate = bool(data.get("personal_zone_activate", False))

            # ETAPA 2: toggle de VLM desde el cliente (None = no cambiar)
            _raw_enable_vlm = data.get("enable_vlm", None)
            enable_vlm = bool(_raw_enable_vlm) if _raw_enable_vlm is not None else None

            # Perfil de angulo por camara (opcional): frontal|lateral|cenital.
            # Solo aplica a TiendasProcess; ajusta el filtro de aspecto.
            camera_angle = data.get("camera_angle") or None

            # Mapa de calor: overlay sobre el video (TiendasProcess).
            heatmap_activate = bool(data.get("heatmap_activate", False))

            # Nombre legible de la camara (para dashboard/heatmap).
            camera_name = str(data.get("camera_name") or "").strip()[:48] or None

            # Local (establecimiento) de ESTA camara: encabeza su mensaje de
            # WhatsApp en VigilanteWS. Va POR FRAME (no como atributo del
            # procesador) porque una misma conexion lleva camaras de locales
            # distintos. OJO: la cadena VACIA se conserva distinta de None —
            # el cliente arranca sin seleccion y manda '' para LIMPIAR el
            # local recordado; None significa "cliente que no maneja el
            # campo" y no toca nada.
            _est_crudo = data.get("establecimiento")
            establecimiento = (None if _est_crudo is None
                               else str(_est_crudo).strip()[:120])

            # Lectura de placas (ALPR) de ESTA camara: boton del recuadro.
            # Va por frame, como el local, porque cada camara decide.
            leer_placas = bool(data.get("leer_placas", False))

            # Segundos DENTRO del area antes de alertar, de ESTA camara
            # (19-ago-2026): antes solo se cambiaban editando el config del
            # servidor y reiniciandolo. Misma convencion que el local: None
            # = el cliente no maneja el campo y no se toca nada.
            min_permanencia_personas = segundos_de_permanencia(
                data.get("min_permanencia_personas"))
            min_permanencia_vehiculos = segundos_de_permanencia(
                data.get("min_permanencia_vehiculos"))

            # MODO DIRECTO: si el cliente pide draw_server=False, el servidor
            # NO dibuja ni envia la imagen; devuelve solo detecciones y el
            # cliente las pinta (menos latencia). Default True (compatibilidad).
            draw_server = bool(data.get("draw_server", True))

            # Alertas por WhatsApp (bot 'ava'): lo activa el cliente
            # perimetrales-view con un toggle. Se fija como atributo del
            # procesador (mismo patrón que _draw_server); solo VigilanteWS lo
            # consume — inofensivo para el resto de procesadores.
            try:
                _wa = bool(data.get("enviar_whatsapp", False))
                # Se registra cada CAMBIO del interruptor, una sola vez. Sin
                # esto el reenvio es una caja negra: si el flag no llega, no
                # hay forma de distinguirlo de "no hubo alertas" ni de un fallo
                # del bot, que es justo donde se atasco el diagnostico del
                # 31-jul (H-25).
                if getattr(processor, "_enviar_whatsapp", None) != _wa:
                    logger.info("WhatsApp: interruptor %s para client=%s "
                                "(pipeline=%s)",
                                "ACTIVADO" if _wa else "desactivado",
                                client_id, type_inference)
                processor._enviar_whatsapp = _wa
            except Exception:
                pass

            camera_id_raw = data.get("camera_id", 1)
            try:
                camera_id = int(camera_id_raw)
            except (TypeError, ValueError):
                camera_id = camera_id_raw

            # Clases a trackear (lista de class_ids COCO enviadas por el cliente)
            track_classes = data.get("track_classes", None)

            # Payload para _full_frame_sync (orden fijo de argumentos)
            frame_payload = (
                processor, image_data,
                roi, roi_activate,
                door_roi, door_activate,
                door_direction, door_direction_activate,
                pay_roi, withdraw_roi,
                pay_roi_activate, withdraw_roi_activate,
                camera_id,
                track_classes,
                order_zone, order_zone_activate,
                delivery_zone, delivery_zone_activate,
                personal_zone, personal_zone_activate,
                enable_vlm,
                camera_angle,
                heatmap_activate,
                camera_name,
                draw_server,
                establecimiento,
                leer_placas,
                flecha_direccion,
                linea_cruce,
                area_nombre,
                anaqueles_zonas,
                anaqueles_nombres,
                anaqueles_activate,
                min_permanencia_personas,
                min_permanencia_vehiculos,
                linea_cruce_estilo,
                puertas_lineas,
                puertas_nombres,
                puertas_activate,
                anaqueles_productos,
                zonas_ignorar,
            )

            # El lector SOLO encola y vuelve a leer. Antes esperaba aqui el
            # resultado y esa espera era el origen de los cierres 1011 cada
            # ~3 minutos: con el socket sin leer, el buffer del websocket se
            # llenaba, uvicorn pausaba el TCP y el PONG del keepalive del
            # cliente quedaba atascado (ver ws_drenaje.py). La respuesta, el
            # reenvio a WhatsApp y la captura saliente los hace el drenador.
            worker.encolar(ItemFrame(
                request=request,
                binario=incoming_is_binary,
                payload=frame_payload,
                camera_id=camera_id,
                camera_id_raw=camera_id_raw,
                camera_name=camera_name or '',
                establecimiento=establecimiento or '',
                enviar_whatsapp=bool(data.get("enviar_whatsapp")),
                type_inference=type_inference,
            ))

            async with connection_lock:
                if client_id in active_connections:
                    active_connections[client_id]["last_active"] = time.time()

    except WebSocketDisconnect:
        logger.info("Cliente desconectado: %s", client_id)
    except RuntimeError as exc:
        # Starlette NO lanza WebSocketDisconnect cuando el bucle intenta leer
        # (o escribir) una vez mas despues del mensaje de cierre: lanza un
        # RuntimeError con un texto concreto. Caia en el `except Exception` de
        # abajo y se registraba como "Error critico" cada vez que alguien
        # cerraba su cliente. El ruido no era el problema: enmascara. Un fallo
        # de verdad al desconectar se confundia con esto.
        if _es_cierre_normal(exc):
            logger.info("Cliente desconectado: %s", client_id)
        else:
            logger.error("Error critico client=%s: %s", client_id, exc)
    except Exception as exc:
        logger.error("Error critico client=%s: %s", client_id, exc)
    finally:
        # -- 3. Limpieza completa ----------------------------------------------
        if worker:
            await worker.stop()
        async with connection_lock:
            active_connections.pop(client_id, None)
        if processor:
            _cerrar_procesadores(processor)
        # Mapas de calor a disco AHORA: "cerrar la camara" no puede costar el
        # acumulado ni la hora parcial del historico. Volcar todos es barato
        # (pocas camaras, PNG pequeno) y cubre tambien las de otros clientes.
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _heatmaps.volcar_todos)
        except Exception:
            pass
        logger.info("Limpieza completa para client=%s", client_id)


#: Textos con los que Starlette avisa de que el otro extremo YA se fue. No
#: son fallos: son la desconexion vista desde el lado equivocado del bucle.
#: Se comparan por fragmento porque el mensaje exacto ha cambiado entre
#: versiones y no hay una clase de excepcion propia que atrapar.
_CIERRES_NORMALES = (
    'once a disconnect message has been received',
    'once a close message has been sent',
)


def _es_cierre_normal(exc: BaseException) -> bool:
    """¿Este RuntimeError es solo que el cliente ya cerro?"""
    texto = str(exc).lower()
    return any(fragmento in texto for fragmento in _CIERRES_NORMALES)


async def _send_error(websocket: WebSocket, is_binary: bool, message: str) -> None:
    """Envia un error estructurado al cliente."""
    payload = {"data": {"status": "error", "message": message}}
    try:
        if is_binary:
            await websocket.send_bytes(msgpack.packb(payload, use_bin_type=True))
        else:
            await websocket.send_json(payload)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Endpoints HTTP
# -----------------------------------------------------------------------------

@app.get("/")
def init_server():
    return {"status": "active", "connections": len(active_connections)}


@app.get("/health")
def health_check():
    now = time.time()
    clients = [
        {
            "client_id": cid,
            "type_inference": info.get("type_inference", "unknown"),
            "idle_seconds": round(now - info.get("last_active", now), 1),
            "connected_seconds": round(now - info.get("connected_at", now), 1),
        }
        for cid, info in active_connections.items()
    ]
    _puerto = int(os.getenv('ELDE_PUERTO_SERVIDOR', '9000') or 9000)
    return {
        "status": "active",
        "total_connections": len(active_connections),
        # Identidad y nombre para el descubrimiento de los clientes. Los
        # mismos que anuncia la baliza UDP (salen de la misma funcion), para
        # que un servidor encontrado por las DOS vias no se liste dos veces.
        "servidor_id": _baliza.identidad(_puerto),
        "servidor_nombre": _baliza.nombre_servidor(),
        "executor_max_workers": EXECUTOR_WORKERS,
        "clients": clients,
        # Estado del contrato: sirve para decidir cuando se puede pasar de
        # 'observar' a 'estricto' sin romper ningun cliente.
        "contrato": _validacion.resumen(),
        "registro": _registro.resumen(),
        "captura_payloads": _captura.resumen() if _captura.activa() else None,
    }


# ──────────────────────────────────────────────────────────────────────────
#  Comedor (Hummus): horario y reporte
#    GET  /hummus/status   horario, jornada y conteos de cada motor conectado
#    POST /hummus/reporte  manda el PDF del dia al grupo, bajo demanda
# ──────────────────────────────────────────────────────────────────────────

def _motores_hummus(client_id: Optional[str] = None):
    """Los HummusProcess conectados, opcionalmente uno solo.

    Se itera sobre una COPIA: un cliente puede desconectarse a mitad del
    recorrido y `active_connections` cambiaria de tamano bajo los pies.

    La salida temprana NO es cosmetica: los procesadores se importan
    PEREZOSAMENTE, asi que `HummusProcess` vale None hasta que alguien
    construye un Hummus, e `isinstance(x, None)` es TypeError -> 500. El
    resto del fichero se guarda con `X is not None and isinstance(...)`; aqui
    faltaba. Y el fallo era dificil de ver: con CERO conexiones el bucle no
    entra y el endpoint devuelve 200 con lista vacia, asi que un curl contra
    un contenedor recien levantado lo daba por bueno. Hace falta al menos un
    cliente conectado —de cualquier modo— para que aparezca (8-sep-2026).
    """
    if HummusProcess is None:
        return
    for cid, info in list(active_connections.items()):
        proc = info.get("processor")
        if not isinstance(proc, HummusProcess):
            continue
        if client_id and cid != client_id:
            continue
        yield cid, proc


@app.get("/hummus/status")
def hummus_status():
    """Horario del local, jornada en curso y conteos de cada motor Hummus."""
    motores = []
    for cid, proc in _motores_hummus():
        horario = getattr(proc, "_horario", None)
        gestor = getattr(proc, "_reportes", None)
        motores.append({
            "client_id": cid,
            # Copia: el hilo de inferencia registra camaras en ese dict
            # mientras esto lo recorre.
            "camaras": sorted(str(c) for c in list(proc._camera_processors)),
            "hora_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "horario": horario.describe() if horario is not None else "n/d",
            "en_pausa": bool(horario.en_pausa()) if horario is not None else False,
            "jornada": horario.jornada() if horario is not None else None,
            "reporte_activo": bool(getattr(gestor, "activo", False)),
            "turnos": [v.etiqueta() for v in getattr(gestor, "turnos", [])],
            "comedor": proc.contadores_comedor(),
        })
    return {"motores": motores, "total": len(motores)}


@app.post("/hummus/reporte")
def hummus_reporte(client_id: Optional[str] = None,
                   jornada: Optional[str] = None):
    """Genera y manda el reporte de la jornada al grupo de WhatsApp.

    Sin `client_id` lo hace para todos los motores Hummus conectados. No pasa
    por el libro de jornada: si un humano lo pide, lo quiere ahora, aunque el
    automatico de ese dia ya haya salido.

    **Un reporte por (grupo, jornada), no por conexion.** Tres ventanas del
    cliente abiertas son tres HummusProcess con el MISMO grupo y el MISMO
    almacen: sin deduplicar, el grupo de WhatsApp recibiria tres PDF identicos
    y la peticion se quedaria bloqueada hasta 90 s por cada uno.
    """
    enviados = []
    omitidos = []
    ya_hecho = set()
    for cid, proc in _motores_hummus(client_id):
        gestor = getattr(proc, "_reportes", None)
        if gestor is None:
            continue
        destino = (getattr(gestor, "grupo", ""), jornada or "")
        if destino in ya_hecho:
            omitidos.append({"client_id": cid,
                             "detalle": "mismo grupo y jornada: ya enviado"})
            continue
        ya_hecho.add(destino)
        enviados.append({"client_id": cid, **gestor.enviar_ahora(jornada)})
    if not enviados:
        return {"status": "sin_motores",
                "detalle": "Ningun motor Hummus conectado"}
    return {"status": "ok", "reportes": enviados, "omitidos": omitidos}


# ──────────────────────────────────────────────────────────────────────────
#  Router multimodal de percepcion (bajo demanda)
#    POST /vlm/ask     imagen + question  -> VQA (Qwen2.5-VL)  "¿que hay?"
#    POST /vlm/detect  imagen + classes   -> grounding (YOLO-World)  "detecta X"
#    POST /vlm/route   imagen + query     -> decide VQA vs detect por la consulta
#  Son `def` (no async) -> FastAPI los corre en threadpool y NO bloquean el
#  WebSocket de tiempo real. Los modelos se cargan perezosamente la 1a vez.
# ──────────────────────────────────────────────────────────────────────────
def _decode_upload(image: UploadFile):
    raw = image.file.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return frame


def _get_router():
    from ..analityc.core.multimodal_router import get_multimodal_router
    return get_multimodal_router(device=0)


@app.post("/vlm/ask")
def vlm_ask(image: UploadFile = File(...), question: str = Form(...),
            max_new_tokens: int = Form(160), context: str = Form("")):
    try:
        frame = _decode_upload(image)
        if frame is None:
            return {"status": "error", "message": "imagen invalida"}
        ans = _get_router().vqa(frame, question, max_new_tokens=max_new_tokens,
                                context=context)
        return {"status": "ok", "answer": ans}
    except Exception as exc:
        logger.exception("vlm_ask error")
        return {"status": "error", "message": str(exc)}


@app.post("/vlm/detect")
def vlm_detect(image: UploadFile = File(...), classes: str = Form(...),
               conf: float = Form(0.05)):
    try:
        frame = _decode_upload(image)
        if frame is None:
            return {"status": "error", "message": "imagen invalida"}
        terms = [t.strip() for t in classes.split(",") if t.strip()] or [classes]
        dets = _get_router().detect(frame, terms, conf=conf)
        return {"status": "ok", "terms": terms, "detections": dets}
    except Exception as exc:
        logger.exception("vlm_detect error")
        return {"status": "error", "message": str(exc)}


@app.post("/vlm/route")
def vlm_route(image: UploadFile = File(...), query: str = Form(...),
              context: str = Form("")):
    try:
        frame = _decode_upload(image)
        if frame is None:
            return {"status": "error", "message": "imagen invalida"}
        result = _get_router().route(frame, query, context=context)
        return {"status": "ok", **result}
    except Exception as exc:
        logger.exception("vlm_route error")
        return {"status": "error", "message": str(exc)}


@app.post("/vlm/events")
def vlm_events(image: UploadFile = File(...), context: str = Form("")):
    """Detecta eventos en el frame: 'compra' y 'entrega de bandeja de comida'.
    'context' (opcional) se suma al contexto global del negocio."""
    try:
        frame = _decode_upload(image)
        if frame is None:
            return {"status": "error", "message": "imagen invalida"}
        result = _get_router().detect_events(frame, context=context)
        return {"status": "ok", **result}
    except Exception as exc:
        logger.exception("vlm_events error")
        return {"status": "error", "message": str(exc)}


@app.post("/vlm/model")
def vlm_set_model(model: str = Form(...)):
    """Selecciona el modelo VQA del servidor ('3b' rapido | '7b' calidad).
    Afecta a TODO el servidor (manual, route y deteccion continua)."""
    try:
        r = _get_router()
        ok = r.set_vqa_model(model)
        return {"status": "ok" if ok else "error", "current": r.current_vqa()}
    except Exception as exc:
        logger.exception("vlm_set_model error")
        return {"status": "error", "message": str(exc)}


@app.get("/vlm/model")
def vlm_get_model():
    try:
        return {"status": "ok", "current": _get_router().current_vqa()}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/vlm/context")
def vlm_set_context(context: str = Form("")):
    """Fija el contexto GLOBAL del negocio/escena (se inyecta en todas las
    consultas: VQA, route y deteccion continua). Persiste."""
    try:
        r = _get_router()
        r.set_context(context)
        return {"status": "ok", "context": r.context}
    except Exception as exc:
        logger.exception("vlm_set_context error")
        return {"status": "error", "message": str(exc)}


@app.get("/vlm/context")
def vlm_get_context():
    try:
        return {"status": "ok", "context": _get_router().context}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ───────────────────────────────────────────────────────────────────────
#  Camaras activas de deteccion de personas (lo usa el dashboard)
# ───────────────────────────────────────────────────────────────────────
#: Metodos de cierre ordenado, en el orden en que se llaman. Un procesador
#: que no los tenga simplemente se salta.
#:
#: `cerrar_sesion_stand` NO esta aqui a proposito: cerrar la jornada del stand
#: en cada desconexion partiria el dia en trozos cada vez que el cliente
#: reconecta por un corte de red, que es justo el fallo de las 4 sesiones
#: colgadas que ya se arreglo con `sesion_abierta_de`. Ver amazonas_stand.py.
#: `cerrar_anaqueles` solo VUELCA a disco (la jornada diaria se reanuda sola,
#: como la del stand): es seguro llamarlo en cada desconexion.
_METODOS_DE_CIERRE = ('cerrar_permanencia', 'cerrar_anaqueles',
                      'cerrar_almacen_tienda', 'cerrar_comedor', 'cleanup')


def _cerrar_procesadores(processor: Any) -> None:
    """Cierre ordenado del procesador Y de sus procesadores POR CAMARA.

    Los frames NO los procesa el objeto que construye `_build_processor`,
    sino el que devuelve `get_camera_processor(camera_id)`. Es ESE el que
    tiene el estado vivo —el libro de estancias de permanencia, la sesion del
    stand—, asi que cerrar solo el de arriba no cerraba nada: el de arriba
    nunca vio un frame.
    """
    objetivos = [processor]
    objetivos.extend(
        (getattr(processor, "_camera_processors", None) or {}).values())
    for objetivo in objetivos:
        for nombre in _METODOS_DE_CIERRE:
            metodo = getattr(objetivo, nombre, None)
            if not callable(metodo):
                continue
            try:
                metodo()
            except Exception as exc:                        # noqa: BLE001
                # Un cierre que falla no puede impedir los demas ni dejar la
                # conexion sin limpiar.
                logger.debug("cierre %s fallo en %s: %s", nombre,
                             type(objetivo).__name__, exc)


def _iter_person_processors():
    """(client_id, camera_id, camera_processor) de todas las camaras activas
    del tipo 'Personal de Amazonas'."""
    for cid, info in list(active_connections.items()):
        proc = info.get("processor")
        if TiendasProcess is None or not isinstance(proc, TiendasProcess):
            continue
        cams = getattr(proc, "_camera_processors", None) or {}
        if cams:
            for cam_id, cam_proc in list(cams.items()):
                yield cid, cam_id, cam_proc
        else:
            # Instancia unica (sin multi-camara): primera camara conocida.
            cam_ids = list(getattr(proc, "camera_states", None) or {})
            yield cid, (cam_ids[0] if cam_ids else None), proc
