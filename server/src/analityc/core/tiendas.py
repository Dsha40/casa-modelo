import cv2
import numpy as np
import torch
import time
import supervision as sv
from collections import defaultdict, deque, Counter
import logging
from typing import Tuple, Dict, Any, List, Optional
import asyncio
import threading
import base64
import httpx
import datetime
import os
import traceback
from ..config.config import (
    DEFAULT_ROI, get_config, CAMERA_ANGLE_PRESETS, DEFAULT_CAMERA_ANGLE,
    AUTO_ANGLE_DEFAULT, AUTO_ANGLE_MIN_SAMPLES, AUTO_ANGLE_CENITAL_HW_MAX,
    AUTO_ANGLE_WINDOW,
)

# Deteccion de personas SIN ultralytics (AGPL, 2-sep-2026): el RF-DETR
# COMPARTIDO del proceso. `detectar_sv` devuelve sv.Detections con class_id
# en el espacio de siempre (0 = persona), asi que ByteTrack, PolygonZone,
# demografia y heatmap siguen intactos aguas abajo.
from . import deteccion_compartida
from . import demografia_aplazada
from . import filtro_fondo
from . import zonas_ignorar
from . import ritmo_imagenes
# ── Modulos de analitica de personas (deteccion + genero/edad) ──
from .analytics.config import AnalyticsConfig
from .analytics.demographics import (DemographicsClassifier,
                                     _load_camera_profile)
# Clasificador oficial desde el 9-oct-2026 («B sin herencia»; el clásico,
# con ELDE_CLASIFICADOR=clasico). Ver la cabecera de clasificador_b.py.
from .analytics import clasificador_b
from .analytics.face_reidentifier import FaceReidentifier
from .analytics.body_reidentifier import BodyReidentifier
from .analytics.people_counter import PeopleCounter
from .analytics.heatmap import HeatmapAccumulator  # noqa: F401 (compat)
from .analytics import heatmap_registro as _heatmaps
from .utils.overlay import draw_demographics_panel, draw_people_total
from .utils.logger import AnalyticsLogger
# Memoria con FECHA de la analitica de tienda (HITO 1, 18-ago-2026): sin esto
# entradas/salidas/unicos solo viven en RAM y una reconexion los pone a cero.
from . import tienda_almacen as _almacen_tienda
from .tienda_alertas import AlertasNegocioMixin
from .tiendas_fotos import FotosMixin
from .tiendas_dibujo import DibujoMixin
# La carga de modelos vive en su propio archivo (HITO 5). Estos nombres se
# RE-EXPORTAN a proposito: `personal_galeria` y `estimador_edad_genero` los
# importan desde `.tiendas` y mover el codigo no puede romperlos.
from .tiendas_modelos import (  # noqa: F401  (re-export de compatibilidad)
    ModelosMixin, _PROJECT_ROOT, _MODELS_DIR, _GENDER_PROTO, _GENDER_MODEL,
    _AGE_PROTO, _AGE_MODEL, _INSIGHTFACE_GENDERAGE, _MIVOLO_MODEL,
    _FACE_EMBEDDING_MODEL, _YUNET_MODEL, _PAR_MODEL, _GpuYuNet,
    _build_onnx_providers, _create_yunet, _tensorrt_enabled,
    registrar_pipeline, sesiones_compartidas_globales)
from .tiendas_reporte import ReporteMixin

logger = logging.getLogger(__name__)

class _AutoAngleDetector:
    """Deduce automaticamente el angulo de la camara (frontal vs cenital).

    Se basa en la relacion alto/ancho (h/w) de los bboxes de personas:
      - frontal/lateral: personas mas altas que anchas -> h/w alto (~2-3).
      - cenital (techo) : personas vistas desde arriba -> bbox ~cuadrado o
        mas ancho -> h/w bajo (~0.7-1.2).

    Acumula muestras en una ventana y, al superar `min_samples`, clasifica
    por la mediana. Mantiene el resultado estable hasta tener evidencia de
    cambio (se re-evalua continuamente con la ventana deslizante).
    """

    def __init__(self, min_samples: int = 40,
                 cenital_hw_max: float = 1.25, window: int = 250):
        self._ratios = deque(maxlen=window)
        self._min_samples = int(min_samples)
        self._cenital_hw_max = float(cenital_hw_max)
        self._angle: str = None  # type: ignore  # None hasta tener confianza

    def add_person(self, bw: float, bh: float) -> None:
        """Registra el bbox de una persona (ancho, alto en px)."""
        if bw <= 0 or bh <= 0 or bh < 20 or bw < 10:
            return
        self._ratios.append(float(bh) / float(bw))

    def evaluate(self) -> str:
        """Re-clasifica y devuelve 'frontal'|'cenital' o None si no hay datos.

        Usa una banda muerta (histeresis) alrededor del umbral para no
        oscilar en camaras borderline (mediana ~ umbral).
        """
        if len(self._ratios) < self._min_samples:
            return self._angle
        med = float(np.median(self._ratios))
        margin = 0.15
        if self._angle == "cenital":
            if med > self._cenital_hw_max + margin:
                self._angle = "frontal"
        elif self._angle == "frontal":
            if med < self._cenital_hw_max - margin:
                self._angle = "cenital"
        else:  # primera decision
            self._angle = ("cenital" if med < self._cenital_hw_max
                           else "frontal")
        return self._angle

    @property
    def angle(self):
        return self._angle

    @property
    def median_hw(self) -> float:
        return float(np.median(self._ratios)) if self._ratios else 0.0

    @property
    def n_samples(self) -> int:
        return len(self._ratios)


class TiendasProcess(ReporteMixin, DibujoMixin, FotosMixin, ModelosMixin,
                     AlertasNegocioMixin):
    """Procesador para reconocimiento de personal específico"""
    
    # Ruta LEGADO del modelo de personas. Ya no se carga nada de aqui (la
    # deteccion es el RF-DETR compartido, pesos en RFDETR_PESOS); se conserva
    # solo porque app.py y get_camera_processor siguen pasando
    # person_model_path y romper esa firma tumbaria el factory.
    _DEFAULT_PERSON_MODEL = os.path.join(
        _PROJECT_ROOT, 'models', 'base', 'yolo26m.pt')

    def __init__(self,
                client_id: None = None,
                model_path: str = None,  # (sin uso) cosmeticos eliminado
                person_model_path: str = None,  # (sin uso) ruta legado
                confidence_threshold: float = 0.7,
                iou_threshold: float = 0.4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                log_file: str = "output/detection_log.txt",
                image_quality: int = 70,
                min_time_in_roi: int = 10,
                max_frames_out: int = 5,
                min_track_frames: int = 3,
                show_minimal_info: bool = True,
                exit_frames_threshold: int = 1,
                max_frames_without_detection: int = 5,
                max_image_size: tuple = (640, 480),
                staff_names_file: str = None,
                shared_model: Any = None,
                shared_person_model: Any = None,
                camera_angle: str = None,
                shared_sessions: dict = None):

        try:
            # Sesiones ONNX + reidentifier COMPARTIDOS entre cámaras (los pasa
            # get_camera_processor). Evita recargar ArcFace/MiVOLO/genderage/PAR
            # por cada cámara (ahorra VRAM) y unifica la DB de re-identificación.
            self._shared_sessions = shared_sessions
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            # Cosmeticos eliminado: no hay modelo de productos.
            self.model_path = None
            self.person_model_path = person_model_path or self._DEFAULT_PERSON_MODEL
            self.log_file = log_file
            self.image_quality = image_quality
            self.show_minimal_info = show_minimal_info
            self.exit_frames_threshold = exit_frames_threshold
            self.max_image_size = max_image_size
            
            # Configuración de tiempos
            self.min_time_in_roi = min_time_in_roi
            self.max_frames_out = max_frames_out
            self.min_track_frames = min_track_frames
            self.max_frames_without_detection = max_frames_without_detection
            
            # Estado interno
            self.frame_counter = 0
            self.personas_en_area = 0
            self.active_tracks = {}
            self.next_id = 1

            # Tracks ya capturados (foto persona/rostro) para no duplicar.
            self._captured_ids = set()
            # track_id -> stem de su captura, y tracks cuya demografia ya se
            # escribio en el sidecar JSON (para mostrar genero/edad en el dash).
            self._capture_meta = {}
            self._captured_demo_done = set()
            # person_uuid (Re-ID) ya capturados -> stem de SU foto. Dict (no
            # set) a proposito: al deduplicar hay que saber que archivo es el
            # DUENO del uuid para no borrar la foto original como si fuera
            # su propio duplicado.
            self._captured_persons = {}
            # Dedup espacial por celda de grilla: ultima captura por zona de la
            # imagen (cell -> timestamp). Acota a ~1 foto por zona por ventana.
            self._cell_last_capture = {}

            self.track_history = defaultdict(lambda: deque(maxlen=30))
            
            # Contadores por empleado
            self.employee_counters = defaultdict(int)

            # Contadores de entradas por categoría
            self.entry_counts = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
            
            # ROI por defecto
            self.roi_polygon = np.array(DEFAULT_ROI if DEFAULT_ROI else [(0, 0), (640, 0), (640, 480), (0, 480)], np.int32)
            
            # Para evitar reconteo
            self.counted_tracks = set()
            self.recent_counted_persons = deque(maxlen=30)
            self.person_cooldown = defaultdict(int)
            
            # Estados de seguimiento
            self.last_counted_frame = 0
            self.last_counted_id = 0
            self.debug_mode = True
            
            # Historial de posiciones para validar movimiento
            self.movement_history = defaultdict(lambda: deque(maxlen=10))

            # Historial de clases por track para estabilizar identidad
            self.class_history_len = 8
            self.class_stability_threshold = 0.6  # proporción para considerar estable
            self.track_class_history = defaultdict(lambda: deque(maxlen=self.class_history_len))
            
            # Nombres del personal (se cargan desde archivo o modelo)
            self.staff_names = {}
            self.load_staff_names(staff_names_file)
            
            # Clases a detectar - SOLO PERSONAL
            self.all_classes = []  # Se determinará después de cargar el modelo
            
            # Para controlar alertas periódicas
            self.alert_minutes_sent = defaultdict(list)
            
            # Para controlar que no se envíen múltiples fotos del mismo evento
            self.sent_entry_photos = defaultdict(lambda: deque(maxlen=2))
            self.sent_exit_photos = defaultdict(lambda: deque(maxlen=2))
            
            # Control de frecuencia de envío
            self.last_sent_time = defaultdict(float)
            self.send_cooldown = 1.0
            
            self.model = None
            self.person_model = None
            self.device = device
            # Registro del pipeline VIVO a nivel de modulo (14-ago-2026):
            # la galeria de personal reutiliza su sesion ArcFace via
            # `sesiones_compartidas_globales()` en vez de duplicarla en la
            # GPU. «El ultimo gana» es inocuo: los clones por camara
            # comparten las mismas sesiones. El accesor lee con getattr en
            # el momento de la consulta, asi que registrarse aqui (antes de
            # crear las sesiones) es correcto.
            registrar_pipeline(self)
            # Tracker de corto plazo (sv.ByteTrack), POR-INSTANCIA = por-camara.
            # Sustituye al BoT-SORT interno de YOLO (que guardaba estado dentro
            # del modelo COMPARTIDO y se mezclaba entre camaras). Cada
            # camera_processor tiene su propio ByteTrack -> tracking aislado.
            self._byte_tracker = self._make_byte_tracker()
            # Zona de conteo (sv.PolygonZone) — Hito 2. Se construye perezosa a
            # partir de roi_polygon y se reconstruye si el poligono cambia.
            self._zone = None
            self._zone_roi_sig = None       # firma del roi_polygon actual
            self._zone_count = 0            # aforo instantaneo (personas dentro)
            # ── Identidad persistente (Hito 4) ──
            # Conjunto de persistent_id (person_uuid) CONFIRMADOS por rostro en
            # esta sesion -> conteo de visitantes unicos SIN inflar por
            # reapariciones (un mismo uuid que reaparece no suma de nuevo).
            # Los tracks sin rostro reciben un id PROVISIONAL y NO entran aqui.
            self._confirmed_unique: set = set()
            # ── Entradas/salidas por persona (Hito 5) ──
            # Maquina de estados de zona POR persistent_id:
            #   _zone_state[pid] = {estado, pending, pending_frames,
            #     conteo_entradas, conteo_salidas, tiempo_total_dentro,
            #     entry_time, last_ts}
            self._zone_state: dict = {}
            self._total_entries: int = 0   # global FUERA->DENTRO confirmadas
            self._total_exits: int = 0     # global DENTRO->FUERA confirmadas
            # persistent_id distintos que ENTRARON al menos una vez (visitantes
            # unicos del AREA; subconjunto de los confirmados que pisaron dentro).
            self._unique_entered: set = set()
            # Reporte consolidado (Hito 8): cacheado + persistido throttled.
            self._last_report_ts: float = 0.0
            self._cached_report = None
            # Identificador del cliente/instancia
            self.client_id = client_id
            logger.info("TiendasProcess device: %s", device)
            # Si se proporciona un modelo compartido, usarlo y evitar
            # re-inicializar. shared_model (productos) es siempre None.
            if shared_person_model is not None:
                # Caso compartido: la instancia padre ya cargo los modelos
                self.model = shared_model  # puede ser None
                self.person_model = shared_person_model
                if (self.model is not None and
                        hasattr(self.model, 'names') and self.model.names):
                    self.all_classes = list(self.model.names.keys())
                    self.staff_names = {
                        cid: self.model.names[cid]
                            .replace('_', ' ').title()
                        for cid in self.all_classes
                    }
                else:
                    self.all_classes = []
                    self.staff_names = {}
                self.staff_names[-1] = 'Persona'
            else:
                self._initialize_model()

            # Cache de procesadores por cámara (si se usa una instancia compartida)
            self._camera_processors = {}

            # Estado por cámara (separado)
            # Cada camera_id tendrá su propio conjunto de tracks y contadores
            # camera_states[camera_id] = {
            #    'active_tracks': {}, 'next_id': int, 'track_history': defaultdict(deque),
            #    'movement_history': defaultdict(deque), 'employee_counters': defaultdict(int), ...
            # }
            self.camera_states = {}
            self._state_swap_stack = []
            # Lock to prevent concurrent process_frame calls interfering with state swapping
            self._process_lock = threading.RLock()
            # atributo auxiliar para referencia al último frame procesado
            self.last_processed_frame = None

            # Perfiles de cámara por ángulo (detección multi-ángulo): definen
            # el filtro de aspecto que se aplica en process_frame.
            self._init_camera_profiles(camera_angle)

            # Estas carpetas solo las llena la ruta LEGADO de fotos de
            # entrada/salida, apagada por defecto (ver LEGACY_STAFF_PHOTOS).
            # Crearlas siempre dejaba tres carpetas vacias en cada
            # instalacion nueva, sugiriendo que algo deberia estar
            # escribiendo ahi.
            if AnalyticsConfig.LEGACY_STAFF_PHOTOS:
                os.makedirs("output/Personal", exist_ok=True)
                os.makedirs("output/Personal/Entradas", exist_ok=True)
                os.makedirs("output/Personal/Salidas", exist_ok=True)
                os.makedirs("output/Personal/Alertas", exist_ok=True)
            os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
            
            self._log_buffer = []
            self.setup_log_file()

            # ── Clasificador de género / edad (OpenCV DNN) ──
            self._init_gender_age_classifier()
            # Cadencia de re-clasificacion de un track NO convergido. Antes
            # era 1 (cada frame), lo que con muchas personas sin cara visible
            # disparaba el pipeline facial completo por persona/frame y tumbaba
            # el FPS (~2 fps con 7 personas). Ahora se reintenta cada N frames
            # y ESCALONADO por track_id (ver _track_match_and_classify) para
            # repartir la carga. Configurable en AnalyticsConfig.
            self._classify_every_n = max(
                1, int(getattr(AnalyticsConfig, 'DEMO_RECLASSIFY_EVERY', 6)))
            # El clasificador B se validó mirando cada ~0,6 s (9-oct-2026):
            # 5 imágenes a 7,7 fps, en tiempo real con `_imagenes`.
            if clasificador_b.usar_b():
                self._classify_every_n = max(1, int(getattr(
                    AnalyticsConfig, 'DEMO_RECLASSIFY_EVERY_B', 5)))
            # Umbral de altura relativa para niño (fallback sin modelo de edad)
            self.child_height_ratio = 0.30

            # ── Modulos de analitica retail ──
            self._analytics_config = AnalyticsConfig
            # Face Re-Identification. Si viene COMPARTIDO (multi-cámara), se
            # reusa: una sola DB en memoria -> conteo único correcto ENTRE
            # cámaras y una sola sesión ArcFace. Es thread-safe (RLock).
            _shared_reid = (self._shared_sessions or {}).get('reidentifier') \
                if getattr(self, '_shared_sessions', None) else None
            if _shared_reid is not None:
                self._reidentifier = _shared_reid
                logger.info("Re-identificador COMPARTIDO reusado (DB única "
                            "entre cámaras)")
            else:
                # Politica "ttl": la galeria es memoria de trabajo (caduca por
                # persona y no toca disco). Ver REID_RESET_POLICY en config.
                _politica = AnalyticsConfig.REID_RESET_POLICY
                _con_ttl = _politica == "ttl"
                self._reidentifier = FaceReidentifier(
                    embedding_session=getattr(self, '_face_embedding_session',
                                              None),
                    db_path=AnalyticsConfig.REID_DB_PATH,
                    similarity_threshold=(
                        AnalyticsConfig.REID_SIMILARITY_THRESHOLD),
                    reset_policy=_politica,
                    ttl_seconds=(AnalyticsConfig.REID_TTL_S if _con_ttl
                                 else 0.0),
                    persistir=not _con_ttl,
                )
                if _con_ttl:
                    logger.info(
                        "Re-identificador con TTL de %.0f min (galeria en RAM)",
                        AnalyticsConfig.REID_TTL_S / 60.0)
                else:
                    # Guardar la DB al cerrar el proceso (Ctrl+C / shutdown).
                    # Solo con galeria persistente: `atexit` guarda una
                    # referencia FUERTE al metodo, asi que cada conexion
                    # websocket dejaria vivo su FaceReidentifier —con su
                    # galeria entera— hasta que muera el proceso, y al apagar
                    # se ejecutarian todos en orden inverso, sobrescribiendo
                    # el mismo fichero uno tras otro.
                    try:
                        import atexit
                        atexit.register(self._reidentifier.force_save)
                    except Exception:
                        pass

            # Re-ID CORPORAL (OSNet) — 2a senal para espaldas. COMPARTIDO entre
            # camaras (galeria efimera unica). NUNCA se persiste (la ropa cambia).
            _shared_body = (self._shared_sessions or {}).get('body_reid') \
                if getattr(self, '_shared_sessions', None) else None
            if _shared_body is not None:
                self._body_reidentifier = _shared_body
            elif (AnalyticsConfig.BODY_REID_ENABLED
                  and getattr(self, '_osnet_session', None) is not None):
                self._body_reidentifier = BodyReidentifier(
                    session=self._osnet_session,
                    similarity_threshold=(
                        AnalyticsConfig.BODY_REID_SIMILARITY_THRESHOLD),
                    ttl_seconds=AnalyticsConfig.BODY_REID_TTL_S,
                    max_emb_per_person=AnalyticsConfig.BODY_REID_MAX_EMB_PER_PERSON,
                )
                logger.info("Re-ID corporal (OSNet) ACTIVO")
            else:
                self._body_reidentifier = None

            self._demographics = clasificador_b.crear(
                gender_net=self._gender_net,
                age_net=self._age_net,
                face_cascade=self._face_cascade,
                face_detector_dnn=getattr(self, '_face_dnn', None),
                insightface_session=getattr(self, '_insightface_session', None),
                yunet=getattr(self, '_yunet', None),
                mivolo_session=getattr(self, '_mivolo_session', None),
                reidentifier=self._reidentifier,
                par_session=getattr(self, '_par_session', None),
            )
            self._people_counter = PeopleCounter()
            self._analytics_logger = AnalyticsLogger()
            # Mapa de calor: se toma del registro GLOBAL al conocer el
            # camera_id (primer frame). Asi el acumulado PERSISTE entre
            # sesiones y el volcado por desconexion tambien lo cubre.
            self._heatmap = None

            print(f'✅ Modelo de personal inicializado para {client_id}')
            print(f'🖥️  Dispositivo: {self.device}')
            print(f'🎯 Umbral de confianza: {confidence_threshold}')
            print(f'👥 Personal registrado: {len(self.staff_names)} personas')
            print(f'📍 Modo debug: {"ACTIVADO" if self.debug_mode else "DESACTIVADO"}')
        except Exception as e:
            # NO tragar la excepcion: imprimir traza completa y re-lanzar
            # para que el factory en app.py sepa que la instancia no quedo usable.
            print(f"❌ Error fatal en TiendasProcess.__init__: {e}")
            print(traceback.format_exc())
            logger.error("TiendasProcess.__init__ fallo: %s", e, exc_info=True)
            raise



    def load_staff_names(self, staff_names_file: str = None):
        """Carga los nombres del personal desde archivo o modelo"""
        try:
            if staff_names_file and os.path.exists(staff_names_file):
                with open(staff_names_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            class_id, name = line.strip().split(':')
                            self.staff_names[int(class_id)] = name
                print(f"📋 Nombres cargados desde archivo: {len(self.staff_names)} empleados")
            else:
                print("ℹ️  No se encontró archivo de nombres, se usarán los del modelo")
        except Exception as e:
            print(f"⚠️  Error cargando nombres: {e}")

    # Compat: algunos caminos de codigo antiguos siguen consultando
    # _ALLOWED_CLASS_IDS. Se calcula dinamicamente tras cargar el modelo.
    _ALLOWED_CLASS_IDS = []

    def _perfil_camara(self, camera_id: Any):
        """Vista de AnalyticsConfig con los overrides del perfil de ESTA camara.

        Es el mismo config/camera_profiles/<camera_id>.json que ya ajusta la
        demografia, ampliado a las claves de DETECCION (8-ago-2026): una
        camara alta o exterior con gente pequena puede bajar su
        PERSON_DETECT_CONF o BYTETRACK_TRACK_ACTIVATION_THRESH sin tocar al
        resto de camaras ni a la config global. Cache por camara: el json se
        lee UNA vez, no en cada frame.
        """
        cache = getattr(self, '_perfiles_camara', None)
        if cache is None:
            cache = self._perfiles_camara = {}
        if camera_id not in cache:
            cache[camera_id] = _load_camera_profile(camera_id)
        return cache[camera_id]

    def _imagenes(self, imagenes: int) -> int:
        """Un ajuste contado en IMAGENES, en tiempo real (5-oct-2026,
        `ritmo_imagenes.py`): las mismas que hoy a ~7,7 fps, mas si esta
        camara va mas rapido. Al subir de 7,7 a 9,9 fps las entradas al ROI
        pasaron de 11 a 29 en el mismo trozo del video: borrar a quien sale
        1 imagen del ROI, confirmar con 3, no recontar en 30... eran cada vez
        menos tiempo. Los fps salen del ByteTrack de la camara en curso."""
        return ritmo_imagenes.imagenes_ref(
            imagenes, ritmo_imagenes.fps_de(getattr(self, '_byte_tracker', None)))

    def _make_byte_tracker(self, camera_id: Any = None):
        """Crea un sv.ByteTrack nuevo con los parametros de AnalyticsConfig.
        Uno por camara (stateful). minimum_consecutive_frames = confirmacion
        de track (anti falso-positivo). Con `camera_id`, el perfil de esa
        camara puede ajustar la activacion (gente pequena = confianza baja =
        tracks que jamas nacian con el umbral global)."""
        cfg = (self._perfil_camara(camera_id) if camera_id is not None
               else AnalyticsConfig)
        return sv.ByteTrack(
            track_activation_threshold=float(
                cfg.BYTETRACK_TRACK_ACTIVATION_THRESH),
            lost_track_buffer=int(cfg.BYTETRACK_LOST_BUFFER),
            minimum_matching_threshold=float(
                cfg.BYTETRACK_MATCH_THRESH),
            frame_rate=float(cfg.BYTETRACK_FRAME_RATE),
            minimum_consecutive_frames=int(
                cfg.BYTETRACK_MIN_CONSECUTIVE_FRAMES),
        )

    def _get_zone(self):
        """Devuelve el sv.PolygonZone del area de conteo (roi_polygon),
        reconstruyendolo si el poligono cambio (set_roi / cambio en frame).
        El ancla de disparo (pie/centro) es configurable via
        AnalyticsConfig.ZONE_TRIGGER_ANCHOR.

        Retorna:
            sv.PolygonZone | None: la zona, o None si no hay roi_polygon.
        """
        roi = getattr(self, 'roi_polygon', None)
        if roi is None:
            return None
        sig = roi.tobytes()
        if self._zone is None or sig != self._zone_roi_sig:
            anchor = getattr(sv.Position,
                             str(AnalyticsConfig.ZONE_TRIGGER_ANCHOR),
                             sv.Position.BOTTOM_CENTER)
            # sv 0.28.0: PolygonZone(polygon, triggering_anchors=...) — SIN
            # frame_resolution_wh (param eliminado; la mascara se dimensiona
            # del propio poligono). Verificado contra el codigo instalado.
            self._zone = sv.PolygonZone(
                polygon=roi.astype(int),
                triggering_anchors=(anchor,),
            )
            self._zone_roi_sig = sig
        return self._zone

    def _initialize_model(self):
        try:
            # ── Cosmeticos ELIMINADO ──
            # El sistema es exclusivamente deteccion de personas + genero +
            # edad. No se carga ningun modelo de productos.
            self.model = None
            self.all_classes = []
            self.staff_names = {}
            type(self)._ALLOWED_CLASS_IDS = []

            # ── Modelo de personas: RF-DETR COMPARTIDO del proceso ──
            # Aqui vivia la maquinaria YOLO/TensorRT (.engine por imgsz, half,
            # warmup, fallback al .pt). Fuera de servicio: el singleton de
            # `deteccion_compartida` ya es UNO por proceso y se carga PEREZOSO
            # (el primer detectar_sv lo carga y calienta), asi que instanciar
            # el pipeline no toca la GPU. `person_model` queda apuntando al
            # modulo para que los chequeos de aguas abajo
            # (`if self.person_model is None`) sigan siendo validos.
            self.person_model = deteccion_compartida
            print("✅ Detector personas: RF-DETR compartido (carga perezosa)")
            # Usamos class_id -1 internamente para personas
            self.staff_names[-1] = 'Persona'
        except Exception as e:
            print(f"❌ Error inicializando modelos: {e}")
            raise

    def setup_log_file(self):
        try:
            with open(self.log_file, 'w', encoding="utf-8") as f:
                f.write("Timestamp,Frame,Empleado_ID,Empleado_Nombre,Evento,Tiempo_Area,Confianza\n")
        except Exception as e:
            print(f"❌ Error creando archivo de log: {e}")

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x11, y11, x21, y21 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x11, x1_2)
        yi1 = max(y11, y1_2)
        xi2 = min(x21, x2_2)
        yi2 = min(y21, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def center_of(self, box: Tuple) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def is_inside_polygon(self, point: Tuple, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) >= 0

    def compress_image(self, image: np.ndarray) -> np.ndarray:
        try:
            height, width = image.shape[:2]
            max_width, max_height = self.max_image_size
            
            if width > max_width or height > max_height:
                aspect_ratio = width / height
                if width > max_width:
                    width = max_width
                    height = int(width / aspect_ratio)
                if height > max_height:
                    height = max_height
                    width = int(height * aspect_ratio)
                
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            logger.error(f"Error al comprimir imagen: {e}")
            return image

    def get_staff_name(self, class_id: int, confidence: float = None) -> str:
        """Obtiene el nombre legible del empleado"""
        if class_id in self.staff_names:
            name = self.staff_names[class_id]
            if confidence:
                return f"{name} ({confidence:.2f})"
            return name
        else:
            if confidence:
                return f"Empleado_{class_id} ({confidence:.2f})"
            return f"Empleado_{class_id}"

    def get_staff_display_name(self, class_id: int) -> str:
        """Nombre para mostrar en pantalla"""
        if class_id in self.staff_names:
            return self.staff_names[class_id]
        return f"ID_{class_id}"

    # ── Clasificación por categoría (Hombre / Mujer / Niño) ──────
    _CATEGORY_KEYWORDS = {
        'Hombres': ['hombre', 'hombres', 'man', 'male', 'boy', 'chico'],
        'Mujeres': ['mujer', 'mujeres', 'woman', 'female', 'girl', 'chica'],
        'Niños':   ['niño', 'niña', 'niños', 'child', 'kid', 'infant', 'bebe', 'baby'],
    }

    def _classify_category(self, class_id: int) -> str:
        """Clasifica un class_id del modelo en Hombres/Mujeres/Niños/Personas."""
        name = self.get_staff_display_name(class_id).lower().strip()
        for category, keywords in self._CATEGORY_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                return category
        return 'Personas'

    # ── Clasificador de género / edad con OpenCV DNN ──────────────

    def _init_camera_profiles(self, camera_angle: str = None):
        """Carga los perfiles de cámara por ángulo desde la config.

        Determinan el filtro de aspecto del detector (ver `_aspect_ratio_min_for`):
        en cenital se desactiva, porque una persona vista desde arriba se ve
        más ancha que alta y el filtro clásico la descartaría.

        Precedencia del ángulo: override manual del cliente (`camera_angle`)
        > ángulo AUTO-DETECTADO > mapa por camera_id en config > por defecto.

        Si `camera_angle` es None o 'auto', se activa la auto-detección: el
        ángulo se deduce solo de la forma de los bboxes de personas.
        """
        # Fallbacks defensivos (si la config no estuviera disponible)
        self._camera_angle_presets = dict(CAMERA_ANGLE_PRESETS)
        self._camera_angle_by_id: Dict[str, str] = {}
        self._default_camera_angle = DEFAULT_CAMERA_ANGLE
        try:
            cfg = get_config(validate=False)
            self._camera_angle_presets = cfg.get(
                "camera_angle_presets", self._camera_angle_presets)
            self._default_camera_angle = cfg.get(
                "default_camera_angle", DEFAULT_CAMERA_ANGLE)
            self._camera_angle_by_id = {
                str(k): str(v)
                for k, v in cfg.get("camera_angle_by_id", {}).items()
            }
        except Exception as e:  # noqa: BLE001 - degradar con warning, no crashear
            logger.warning(
                "No se pudieron leer perfiles de cámara de config (%s). "
                "Uso defaults.", e)

        # ── Auto-detección de ángulo ──
        self._auto_angle_detector = _AutoAngleDetector(
            min_samples=AUTO_ANGLE_MIN_SAMPLES,
            cenital_hw_max=AUTO_ANGLE_CENITAL_HW_MAX,
            window=AUTO_ANGLE_WINDOW)
        self._auto_angle: Optional[str] = None
        # Aplicar el ángulo inicial (puede ser 'auto' -> activa detección)
        self._camera_angle_override = None
        self._auto_angle_enabled = bool(AUTO_ANGLE_DEFAULT)
        self.set_camera_angle(camera_angle if camera_angle else "auto")
        logger.info(
            "Perfiles de cámara: default=%s, override=%s, auto=%s, por_id=%s",
            self._default_camera_angle, self._camera_angle_override,
            self._auto_angle_enabled, self._camera_angle_by_id or "{}")

    def _aspect_ratio_min_for(self, camera_id: Any) -> float:
        """Devuelve el aspect_ratio_min (alto/ancho) del perfil de la cámara.

        0.0 => filtro de aspecto desactivado (cenital). Un bbox de persona se
        descarta solo si `aspect_ratio_min > 0` y es más ancho que alto según
        ese umbral.
        """
        angle = (
            self._camera_angle_override                       # manual cliente
            or (self._auto_angle if self._auto_angle_enabled else None)  # auto
            or self._camera_angle_by_id.get(str(camera_id))   # mapa config
            or self._default_camera_angle
            or DEFAULT_CAMERA_ANGLE
        )
        preset = self._camera_angle_presets.get(angle.lower().strip())
        if preset is None:
            preset = self._camera_angle_presets.get(
                DEFAULT_CAMERA_ANGLE, {"aspect_ratio_min": 0.80})
        return float(preset.get("aspect_ratio_min", 0.80))

    def set_camera_angle(self, angle: str):
        """Fija el ángulo de esta cámara en runtime.

        - 'auto' o vacío: activa la AUTO-detección (el ángulo se deduce de la
          forma de los bboxes de personas).
        - 'frontal'|'lateral'|'cenital': override MANUAL (gana sobre la
          auto-detección hasta que se vuelva a poner 'auto').
        """
        a = (angle or "").lower().strip()
        if a in ("", "auto"):
            self._camera_angle_override = None
            self._auto_angle_enabled = True
        else:
            self._camera_angle_override = a
            self._auto_angle_enabled = False
            logger.info("Ángulo de cámara override MANUAL -> %s", a)

    def _classify_person(self, frame: np.ndarray, bbox, track_id: int = None) -> str:
        """Clasifica persona por genero/edad usando DemographicsClassifier.

        Retorna categoria legacy ('Hombres', 'Mujeres', 'Ninos') para
        compatibilidad con el sistema existente de contadores.
        """
        if track_id is not None and hasattr(self, '_demographics'):
            result = self._demographics.classify(frame, bbox, track_id)
            # Log de clasificacion
            if hasattr(self, '_analytics_logger'):
                self._analytics_logger.log_demographic(
                    track_id, result['gender'], result['age_range']
                )
            return result.get('category', 'Hombres')
        return 'Hombres'

    async def send_alert(self, base64_img: str, text: str):
        """Envía una alerta con imagen"""
        payload = {
            "my-text": text, 
            "my-file": base64_img, 
            "type": "image/jpg",
            "compressed": "true",
            "quality": str(self.image_quality)
        }
        
        payload_size = len(base64_img) / 1024
        if self.debug_mode:
            print(f"📦 Tamaño del payload: {payload_size:.2f} KB")
        
        if payload_size > 500:
            logger.warning(f"Payload demasiado grande ({payload_size:.2f} KB)")
            payload = {
                "my-text": f"{text} (Imagen muy grande: {payload_size:.2f} KB)",
                "type": "text"
            }
        
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            try: 
                respuesta = await client.post(
                    "https://72.68.60.254:4000/bot/imgV2/number=120363167230826480@g.us",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                respuesta.raise_for_status()
                logger.info(f"✅ Alerta enviada: {respuesta.status_code}")
                return respuesta.json()
            except Exception as e:
                logger.error(f"❌ Error en envío: {e}")
                raise

    def send_alert_wrapper(self, base64_img: str, text: str, object_id: int):
        current_time = time.time()
        last_time = self.last_sent_time.get(object_id, 0)

        if current_time - last_time < self.send_cooldown:
            if self.debug_mode:
                print(f"⏳ Cooldown para objeto {object_id}")
            return

        # Se sella la marca AL INTENTAR, no al lograr. Sellando solo en el
        # exito, con el endpoint caido se reintentaba CADA frame: dos
        # timeouts de 30 s por frame en hilos daemon que se acumulaban a
        # cientos y hundian los FPS del pipeline (inundacion del 7-ago-2026).
        # Perder un reintento dentro del cooldown es el mal menor.
        self.last_sent_time[object_id] = current_time

        def send_async():
            try:
                asyncio.run(self.send_alert(base64_img, text))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.send_alert(base64_img, text))
                    finally:
                        loop.close()
                else:
                    logger.error(f"Envío falló: {e}")
            except Exception as e:
                logger.error(f"Envío falló: {e}")
        
        thread = threading.Thread(target=send_async)
        thread.daemon = True
        thread.start()

    def create_annotated_image(self, frame: np.ndarray, staff_id: int, object_id: int, confidence: float = None) -> np.ndarray:
        annotated_frame = frame.copy()
        
        if object_id in self.active_tracks:
            track = self.active_tracks[object_id]
            x1, y1, x2, y2 = [int(v) for v in track['box']]
            
            # Color diferente por empleado (basado en ID)
            color_map = [
                (0, 255, 255),   # Amarillo - ID 0
                (255, 0, 0),     # Azul - ID 1
                (0, 255, 0),     # Verde - ID 2
                (255, 0, 255),   # Magenta - ID 3
                (0, 165, 255),   # Naranja - ID 4
                (255, 255, 0),   # Cyan - ID 5
                (128, 0, 128),   # Púrpura - ID 6
                (255, 192, 203)  # Rosa - ID 7
            ]
            
            color_idx = staff_id % len(color_map)
            box_color = color_map[color_idx]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Etiqueta con nombre y confianza
            staff_name = self.get_staff_name(staff_id, confidence)
            label = f"{staff_name}"
            
            # Añadir tiempo en área si está dentro del ROI
            if 'entry_time' in track:
                current_time = time.time()
                time_in_roi = int(current_time - track['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                label += f" - {minutes}m {seconds}s"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fondo para el texto
            cv2.rectangle(annotated_frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        (0, 0, 0), 
                        -1)
            
            # Texto
            cv2.putText(annotated_frame, label,
                      (x1, y1 - 5), font, font_scale,
                      (255, 255, 255), thickness)
        
        return annotated_frame

    def get_action_message(self, staff_id: int, event: str, time_in_roi: int = 0, confidence: float = None) -> str:
        staff_name = self.get_staff_display_name(staff_id)
        
        if event == 'entrada':
            return f"👤 {staff_name} entró en el área de oficina"
        elif event == 'salida':
            minutes = time_in_roi // 60
            if minutes == 1:
                return f"👤 {staff_name} salió después de {minutes} minuto"
            else:
                return f"👤 {staff_name} salió después de {minutes} minutos"
        elif event == 'alerta_periodica':
            minutes = time_in_roi // 60
            if minutes == 1:
                return f"⏰ {staff_name} lleva {minutes} minuto en el área"
            else:
                return f"⏰ {staff_name} lleva {minutes} minutos en el área"
        else:
            return f"{staff_name} - {event.upper()}"

    def save_staff_photo(self, frame: np.ndarray, staff_id: int, object_id: int, event: str,
                         confidence: float = None, time_in_roi: int = 0):
        """Foto de entrada/salida del LEGADO. Apagada por defecto.

        Ver `AnalyticsConfig.LEGACY_STAFF_PHOTOS`: esto avisaba de entradas y
        salidas del PERSONAL, una funcion que dejo de existir el 27-jul-2026.
        Lo que quedaba era el coste —anotar, comprimir, JPEG, base64, un POST
        HTTP y escribir a disco, por cada entrada y cada salida de cada
        persona, dentro del bucle de inferencia— sin que nadie leyera el
        resultado. La contabilidad en memoria (contadores por categoria y el
        cronometro del overlay) NO depende de esto y sigue igual.
        """
        if not AnalyticsConfig.LEGACY_STAFF_PHOTOS:
            return False
        # Candado por (track, evento): una persona parada JUSTO en el borde
        # del ROI parpadea dentro/fuera en cada frame y disparaba una foto de
        # entrada/salida ~5 veces por segundo del MISMO track (inundacion de
        # 'Foto de salida guardada' del 7-ago-2026), llenando output/Personal
        # y encolando un envio por foto. El mismo evento real nunca ocurre
        # dos veces en menos de 5 s.
        ahora = time.time()
        if not hasattr(self, '_ultimo_evento_staff'):
            self._ultimo_evento_staff = {}
        clave = (object_id, event)
        if ahora - self._ultimo_evento_staff.get(clave, 0.0) < 5.0:
            return False
        self._ultimo_evento_staff[clave] = ahora
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            staff_name_safe = self.get_staff_display_name(staff_id).replace(' ', '_')
            
            event_dir = os.path.join("output/Personal", event.capitalize())
            os.makedirs(event_dir, exist_ok=True)
            
            filename = f"{staff_name_safe}_{event}_{object_id}_{timestamp}.jpg"
            filepath = os.path.join(event_dir, filename)
            
            annotated_frame = self.create_annotated_image(frame, staff_id, object_id, confidence)
            compressed_frame = self.compress_image(annotated_frame)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
            
            if success:
                imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                base64_size_kb = len(imagen_base64) / 1024
                
                if base64_size_kb > 500:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
                    success, buffer = cv2.imencode('.jpg', compressed_frame, encode_params)
                    if success:
                        imagen_base64 = base64.b64encode(buffer).decode('utf-8')
                        base64_size_kb = len(imagen_base64) / 1024
                
                message = self.get_action_message(staff_id, event, time_in_roi, confidence)
                self.send_alert_wrapper(imagen_base64, message, object_id)
                
                with open(filepath, 'wb') as f:
                    f.write(buffer)
                
                # Registrar en log
                self._log_staff_event(staff_id, event, time_in_roi, confidence)
                
                logger.info(f"✅ Foto de {event} guardada: {filename}")
                if self.debug_mode:
                    minutes = time_in_roi // 60
                    seconds = time_in_roi % 60
                    time_info = f" - Tiempo: {minutes}m {seconds}s" if time_in_roi > 0 else ""
                    print(f"📸 {message}{time_info} ({base64_size_kb:.2f} KB)")
                
                return True
            return False
        except Exception as e:
            logger.error(f"No se pudo guardar la foto: {e}")
            return False

    def _log_staff_event(self, staff_id: int, event: str, time_in_roi: int = 0, confidence: float = None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        staff_name = self.get_staff_display_name(staff_id)
        minutes = time_in_roi // 60
        seconds = time_in_roi % 60
        
        log_entry = f"{ts},{self.frame_counter},{staff_id},{staff_name},{event},{minutes}:{seconds}"
        if confidence:
            log_entry += f",{confidence:.2f}"
        else:
            log_entry += ",N/A"
        
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(f"{log_entry}\n")

    def _get_cam_state(self, camera_id: Any) -> Dict[str, Any]:
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = {
                'active_tracks': {},
                'next_id': 1,
                'track_history': defaultdict(lambda: deque(maxlen=30)),
                'movement_history': defaultdict(lambda: deque(maxlen=10)),
                'employee_counters': defaultdict(int),
                'counted_tracks': set(),
                'recent_counted_persons': deque(maxlen=30),
                'person_cooldown': defaultdict(int),
                'alert_minutes_sent': defaultdict(list),
                'sent_entry_photos': defaultdict(lambda: deque(maxlen=2)),
                'sent_exit_photos': defaultdict(lambda: deque(maxlen=2)),
                'last_sent_time': defaultdict(float),
                'last_processed_frame': None,
                'personas_en_area': 0,
                'entry_counts': {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0},
            }
        # Migrar estados que no tengan entry_counts
        state = self.camera_states[camera_id]
        if 'entry_counts' not in state:
            state['entry_counts'] = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
        # Migrar analytics por camara
        if '_demographics' not in state:
            state['_demographics'] = clasificador_b.crear(
                gender_net=self._gender_net,
                age_net=self._age_net,
                face_cascade=self._face_cascade,
                face_detector_dnn=getattr(self, '_face_dnn', None),
                insightface_session=getattr(self, '_insightface_session', None),
                yunet=getattr(self, '_yunet', None),
                mivolo_session=getattr(self, '_mivolo_session', None),
                reidentifier=getattr(self, '_reidentifier', None),
                par_session=getattr(self, '_par_session', None),
                # Perfil por camara: aplica los overrides de
                # config/camera_profiles/<camera_id>.json si existe.
                camera_id=camera_id,
            )
            state['_people_counter'] = PeopleCounter()
            state['_analytics_logger'] = AnalyticsLogger(
                log_path=f"output/analytics_log_{camera_id}.jsonl"
            )
            # Acumulador del registro GLOBAL: restaura lo persistido de esta
            # camara y queda cubierto por el volcado a disco de app.py.
            state['_heatmap'] = _heatmaps.obtener(camera_id)
            # ByteTrack propio por camara (stateful) -> tracking aislado en la
            # ruta de swap de estado (igual que active_tracks).
            state['_byte_tracker'] = self._make_byte_tracker(camera_id)
            # Set de unicos confirmados por camara (Hito 4).
            state['_confirmed_unique'] = set()
            # Maquina entrada/salida por persona y globales (Hito 5), sembrada
            # con lo que esta camara ya lleva contado HOY (HITO 1): una
            # reconexion ya no pone el marcador a cero.
            self._sembrar_desde_almacen(state, camera_id)
        return state

    def get_shared_sessions(self) -> dict:
        """Devuelve las sesiones ONNX + reidentifier para compartir entre
        cámaras (las usa get_camera_processor). Solo se comparten objetos
        thread-safe: sesiones ORT y el reidentifier (con RLock)."""
        return {
            'insightface': getattr(self, '_insightface_session', None),
            'embedding': getattr(self, '_face_embedding_session', None),
            'mivolo': getattr(self, '_mivolo_session', None),
            'par': getattr(self, '_par_session', None),
            'reidentifier': getattr(self, '_reidentifier', None),
            # YuNet GPU (onnxruntime) ES thread-safe -> se comparte entre camaras
            # (no se duplica por camara como el cv2.FaceDetectorYN de CPU).
            'yunet': getattr(self, '_yunet', None),
            # Re-ID corporal (OSNet) — sesion + galeria efimera compartidas.
            'osnet': getattr(self, '_osnet_session', None),
            'body_reid': getattr(self, '_body_reidentifier', None),
        }

    def get_camera_processor(self, camera_id: Any):
        """Return or create a per-camera processor instance that shares the heavy model.

        This keeps tracking state isolated per camera while sharing the detection model.

        `type(self)`, NO `TiendasProcess`: los modos derivados (AmazonasStand,
        PermanenciaArea) heredan de esta clase, y app.py despacha por
        `isinstance(processor, TiendasProcess)` y luego procesa el frame con
        ESTE objeto. Con la clase incrustada, el procesador por camara salia
        siendo el padre y toda la logica del modo derivado —el embudo del
        stand, el cronometro de permanencia— no llegaba a ejecutarse nunca:
        el video y las detecciones se veian bien, y no se emitia un solo
        evento del modo. Sus __init__ son (*args, **kwargs) sobre este mismo,
        asi que aceptan estos parametros tal cual.
        """
        if camera_id in self._camera_processors:
            return self._camera_processors[camera_id]

        # Crear nueva instancia ligera que comparte ambos modelos
        cam_proc = type(self)(
            client_id=f"{self.client_id}_{camera_id}",
            model_path=self.model_path,
            person_model_path=self.person_model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
            log_file=self.log_file,
            image_quality=self.image_quality,
            min_time_in_roi=self.min_time_in_roi,
            max_frames_out=self.max_frames_out,
            min_track_frames=self.min_track_frames,
            show_minimal_info=self.show_minimal_info,
            exit_frames_threshold=self.exit_frames_threshold,
            max_frames_without_detection=self.max_frames_without_detection,
            max_image_size=self.max_image_size,
            staff_names_file=None,
            shared_model=self.model,
            shared_person_model=self.person_model,
            shared_sessions=self.get_shared_sessions(),
        )

        # Keep camera-specific state separate
        # Copy model-related metadata so camera instance can use class names and filters
        try:
            cam_proc.staff_names = dict(self.staff_names)
            cam_proc.all_classes = list(self.all_classes)
            cam_proc.roi_polygon = np.array(self.roi_polygon)
            # Los modulos de analytics ya se inicializan en __init__ con sus propias
            # instancias independientes, asi que cada camera_processor tiene su propio estado
        except Exception:
            pass

        self._camera_processors[camera_id] = cam_proc
        return cam_proc

    def _push_state(self, cam_state: Dict[str, Any]):
        # Guardar punteros actuales
        names = [
            'active_tracks', 'next_id', 'track_history', 'movement_history', 'employee_counters',
            'counted_tracks', 'recent_counted_persons', 'person_cooldown', 'alert_minutes_sent',
            'sent_entry_photos', 'sent_exit_photos', 'last_sent_time', 'last_processed_frame', 'personas_en_area',
            'entry_counts',
            '_demographics', '_people_counter', '_analytics_logger',
            '_heatmap', '_byte_tracker', '_confirmed_unique',
            '_zone_state', '_total_entries', '_total_exits', '_unique_entered',
        ]
        saved = {}
        for n in names:
            saved[n] = getattr(self, n, None)
            # asignar el estado de la cámara, si existe la clave
            setattr(self, n, cam_state.get(n))
        self._state_swap_stack.append(saved)

    def _pop_state(self):
        if not self._state_swap_stack:
            return
        saved = self._state_swap_stack.pop()
        for n, v in saved.items():
            setattr(self, n, v)

    def check_periodic_alerts(self, frame: np.ndarray):
        current_time = time.time()
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)
            
            if is_inside and track['class'] in self.staff_names:
                if 'entry_time' not in track:
                    track['entry_time'] = current_time
                    track['last_alert_minute'] = 0
                
                time_in_roi = int(current_time - track['entry_time'])
                current_minute = time_in_roi // 60
                
                # Alertas a los 1, 3, 6, 9... minutos
                target_minutes = []
                minute = 1
                while minute <= current_minute:
                    target_minutes.append(minute)
                    minute += 3
                
                sent_minutes = self.alert_minutes_sent.get(track_id, [])
                
                for target_minute in target_minutes:
                    if target_minute not in sent_minutes:
                        success = self.save_staff_photo(
                            frame,
                            track['class'],
                            track_id,
                            'alerta_periodica',
                            track.get('confidence'),
                            time_in_roi
                        )
                        
                        if success:
                            if track_id not in self.alert_minutes_sent:
                                self.alert_minutes_sent[track_id] = []
                            
                            if target_minute not in self.alert_minutes_sent[track_id]:
                                self.alert_minutes_sent[track_id].append(target_minute)
                            
                            track['last_alert_minute'] = target_minute
                            
                            if self.debug_mode:
                                staff_name = self.get_staff_display_name(track['class'])
                                print(f"⏰ ALERTA: {staff_name} lleva {target_minute} minuto{'s' if target_minute > 1 else ''}")
            else:
                if 'entry_time' in track:
                    del track['entry_time']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

    def is_near_recent_counted(self, center: Tuple, threshold: int = 50) -> bool:
        for counted_id, counted_center, counted_frame in self.recent_counted_persons:
            distance = np.sqrt((center[0] - counted_center[0])**2 + (center[1] - counted_center[1])**2)
            if distance < threshold and (self.frame_counter - counted_frame) < self._imagenes(30):
                return True
        return False

    def validate_movement(self, track_id: int, current_pos: Tuple) -> bool:
        if track_id not in self.movement_history:
            self.movement_history[track_id].append(current_pos)
            return True
        
        positions = list(self.movement_history[track_id])
        if len(positions) < 3:
            self.movement_history[track_id].append(current_pos)
            return True
        
        first_pos = positions[0]
        distance = np.sqrt((current_pos[0] - first_pos[0])**2 + (current_pos[1] - first_pos[1])**2)
        
        self.movement_history[track_id].append(current_pos)
        return distance > 10

    def cleanup_undetected_tracks(self, current_detections: list):
        if not current_detections:
            if self.active_tracks and self.debug_mode:
                print(f"⚠️ No hay detecciones, limpiando tracks")
            self.active_tracks.clear()
            return
        
        track_ids = list(self.active_tracks.keys())
        unmatched_tracks = []
        
        for track_id in track_ids:
            track = self.active_tracks[track_id]
            track_box = track['box']
            
            has_match = False
            for det in current_detections:
                iou = self.calculate_iou(track_box, det['box'])
                if iou > 0.3:
                    has_match = True
                    break
            
            if not has_match:
                track['frames_without_detection'] = track.get('frames_without_detection', 0) + 1
                
                if track['frames_without_detection'] >= self._imagenes(self.max_frames_without_detection):
                    unmatched_tracks.append(track_id)
            else:
                track['frames_without_detection'] = 0
        
        for track_id in unmatched_tracks:
            if self.debug_mode:
                track_info = self.active_tracks[track_id]
                staff_name = self.get_staff_display_name(track_info['class'])
                print(f"🗑️ {staff_name} eliminado - {self.max_frames_without_detection} frames sin detección")
            self._remove_track(track_id)

    def process_entry_exit_logic(self, frame: np.ndarray):
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        counted_in_frame = []
        tracks_to_remove = []

        self.person_count_inside = 0
        self.active_by_category = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}

        for track_id, track in list(self.active_tracks.items()):
            current_pos = track['center']
            is_inside = self.is_inside_polygon(current_pos, roi_polygon_points)

            previous_inside = track.get('is_inside', False)
            track['is_inside'] = is_inside

            if is_inside and not previous_inside:
                # ENTRADA del empleado
                track['entry_time'] = time.time()
                track['last_alert_minute'] = 0
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

                if hasattr(self, 'last_processed_frame'):
                    self.save_staff_photo(
                        self.last_processed_frame,
                        track['class'],
                        track_id,
                        'entrada',
                        track.get('confidence')
                    )

                if self.debug_mode:
                    staff_name = self.get_staff_display_name(track['class'])
                    confidence = track.get('confidence', 0)
                    print(f"🚪 {staff_name} ({confidence:.2f}) entró al área")

                # Incrementar contador del empleado
                self.employee_counters[track['class']] += 1

                # Incrementar contador por categoría (usa la clasificación del track)
                category = track.get('category', 'Personas')
                self.entry_counts[category] = self.entry_counts.get(category, 0) + 1
            
            elif not is_inside and previous_inside:
                # SALIDA del empleado
                total_time = 0
                total_minutes = 0
                if 'entry_time' in track:
                    total_time = int(time.time() - track['entry_time'])
                    total_minutes = total_time // 60

                # Registrar visita en CSV.
                if total_time > 0:
                    gender = "Desconocido"
                    age_range = "Desconocido"
                    if hasattr(self, '_demographics'):
                        demo = self._demographics.get_cached(track_id)
                        if demo:
                            gender = demo.get('gender', 'Desconocido')
                            age_range = demo.get('age_range', 'Desconocido')
                    if hasattr(self, '_analytics_logger'):
                        try:
                            self._analytics_logger.log_visit_csv(
                                gender, age_range, total_time
                            )
                        except Exception as _e:
                            logger.error(f"Error CSV visita: {_e}")

                if 'entry_time' in track:
                    del track['entry_time']
                if track_id in self.alert_minutes_sent:
                    del self.alert_minutes_sent[track_id]

                if hasattr(self, 'last_processed_frame'):
                    self.save_staff_photo(
                        self.last_processed_frame,
                        track['class'],
                        track_id,
                        'salida',
                        track.get('confidence'),
                        total_time
                    )

                if self.debug_mode:
                    staff_name = self.get_staff_display_name(track['class'])
                    print(f"🚪 {staff_name} salió del área - Duró {total_minutes} minutos")
            
            if is_inside:
                track['has_been_inside'] = True
                track['frames_out_roi'] = 0
                track['frames_in_roi'] = track.get('frames_in_roi', 0) + 1
                self.person_count_inside += 1
                cat = track.get('category', 'Personas')
                self.active_by_category[cat] = self.active_by_category.get(cat, 0) + 1
            else:
                track['frames_out_roi'] = track.get('frames_out_roi', 0) + 1
                track['frames_in_roi'] = 0
                
                frames_in_roi = track.get('total_frames_in_roi', 0)
                frames_out_roi = track.get('frames_out_roi', 0)
                
                # En tiempo real, no en imagenes (`_imagenes`, 5-oct-2026).
                if (frames_in_roi >= self._imagenes(self.min_time_in_roi) and
                    frames_out_roi >= self._imagenes(2) and
                    track['seen_frames'] >= self._imagenes(self.min_track_frames) and
                    not track.get('counted', False)):
                    
                    if not self.is_near_recent_counted(current_pos):
                        counted_in_frame.append(track_id)
                
                if track.get('has_been_inside', False) and track['frames_out_roi'] >= self._imagenes(self.exit_frames_threshold):
                    tracks_to_remove.append(track_id)

                elif not track.get('has_been_inside', False) and track['frames_out_roi'] > self._imagenes(self.max_frames_out):
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self._remove_track(track_id)
        
        for track_id in counted_in_frame:
            if self._count_staff_safe(track_id):
                if track_id in self.active_tracks:
                    self._remove_track(track_id)
        
        return self.person_count_inside

    def _count_staff_safe(self, track_id: int) -> bool:
        if track_id not in self.active_tracks:
            return False
        
        track = self.active_tracks[track_id]
        
        if track.get('counted', False) or track_id in self.counted_tracks:
            return False
        
        if not self.validate_movement(track_id, track['center']):
            if self.debug_mode:
                print(f"⚠️ Movimiento no válido - ignorando")
            return False
        
        self.counted_tracks.add(track_id)
        track['counted'] = True
        track['counted_at_frame'] = self.frame_counter
        
        self.recent_counted_persons.append((track_id, track['center'], self.frame_counter))
        
        self.personas_en_area += 1
        self.last_counted_frame = self.frame_counter
        self.last_counted_id = track_id
        
        staff_name = self.get_staff_display_name(track['class'])
        confidence = track.get('confidence', 0)
        
        print(f"\n{'='*60}")
        print(f"🎉 {staff_name} ({confidence:.2f}) EN ÁREA!")
        print(f"   Total personal en área: {self.personas_en_area}")
        print(f"   Contador de {staff_name}: {self.employee_counters[track['class']]}")
        print(f"{'='*60}\n")
        
        return True

    def _remove_track(self, track_id: int):
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
            staff_id = track['class']
            staff_name = self.get_staff_display_name(staff_id)

            # Si es persona (class == -1) y tiene entry_time, registrar
            # visita en CSV aunque no haya habido SALIDA formal (el track
            # se puede limpiar por timeout antes de cruzar el ROI).
            if staff_id == -1 and 'entry_time' in track:
                try:
                    total_time = int(time.time() - track['entry_time'])
                    if total_time > 0:
                        gender = "Desconocido"
                        age_range = "Desconocido"
                        if hasattr(self, '_demographics'):
                            demo = self._demographics.get_cached(track_id)
                            if demo:
                                gender = demo.get('gender', 'Desconocido')
                                age_range = demo.get('age_range',
                                                     'Desconocido')
                        if hasattr(self, '_analytics_logger'):
                            self._analytics_logger.log_visit_csv(
                                gender, age_range, total_time
                            )
                except Exception as _e:
                    logger.error(f"Error CSV visita en remove: {_e}")

            if self.debug_mode:
                print(f"✅ {staff_name} eliminado del seguimiento")
            del self.active_tracks[track_id]
        
        if track_id in self.track_history:
            del self.track_history[track_id]
        
        if track_id in self.movement_history:
            del self.movement_history[track_id]
        
        if track_id in self.person_cooldown:
            del self.person_cooldown[track_id]
        
        if track_id in self.alert_minutes_sent:
            del self.alert_minutes_sent[track_id]
        # clean up class history
        if track_id in self.track_class_history:
            try:
                del self.track_class_history[track_id]
            except Exception:
                pass

        # La demografia NO se borra aqui (12-oct-2026,
        # `demografia_aplazada.py`): ByteTrack recuerda el numero 9 s y quien
        # vuelve antes conserva su genero. Se borra (y se sella la ficha)
        # en `_purgar_demografia` cuando pasa ese tiempo sin volver.
        if getattr(self, '_demographics', None) is not None:
            demografia_aplazada.aplazar(self._demographics, track_id,
                                        time.time())
        else:
            self._borrar_demografia(track_id)

    def _borrar_demografia(self, track_id: int):
        # El track muere: si tenia una foto y nunca llego a tener genero,
        # se sella el JSON con el MOTIVO (Hito 7). Va ANTES de liberar el
        # estado, que es de donde sale el diagnostico.
        try:
            self._sellar_captura_sin_demografia(track_id)
        except Exception:  # noqa: BLE001
            pass

        # CRITICO (fuga de memoria): liberar el estado de demografia del track
        # (accumulator + contadores). Sin esto, _accum crecia sin limite con
        # cada track_id de BoT-SORT -> RAM al alza -> OOM -> crash del servidor.
        if hasattr(self, '_demographics') and self._demographics is not None:
            try:
                self._demographics.remove_track(track_id)
            except Exception:
                pass

    def _purgar_demografia(self):
        """Borra la demografia de quien lleva `BYTETRACK_LOST_S` sin volver
        (`demografia_aplazada.py`). Una vez por imagen de esta camara."""
        demografia_aplazada.purgar(
            getattr(self, '_demographics', None), time.time(),
            float(getattr(AnalyticsConfig, 'BYTETRACK_LOST_S', 9.0) or 9.0),
            self._borrar_demografia)

    def cleanup_stale_tracks(self):
        current_frame = self.frame_counter
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last = current_frame - track['last_seen']
            
            if (frames_since_last > 30 or 
                (track.get('frames_out_roi', 0) > 50 and not track.get('has_been_inside', False))):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if self.debug_mode:
                staff_id = self.active_tracks[track_id]['class'] if track_id in self.active_tracks else None
                staff_name = self.get_staff_display_name(staff_id) if staff_id else "desconocido"
                print(f"🗑️ {staff_name} eliminado (inactivo)")
            self._remove_track(track_id)

    def match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.active_tracks or not detections:
            return [], list(range(len(detections))), list(self.active_tracks.keys())
        
        similarity_matrix = np.zeros((len(self.active_tracks), len(detections)))
        track_ids = list(self.active_tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track_box = self.active_tracks[track_id]['box']
            for j, det in enumerate(detections):
                similarity_matrix[i, j] = self.calculate_iou(track_box, det['box'])
        
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        iou_threshold = 0.3
        
        while True:
            if similarity_matrix.size == 0:
                break
                
            max_iou = np.max(similarity_matrix)
            if max_iou < iou_threshold:
                break
            
            i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            
            matched_pairs.append((track_ids[i], j))
            unmatched_tracks.remove(i)
            unmatched_detections.remove(j)
            
            similarity_matrix[i, :] = 0
            similarity_matrix[:, j] = 0
        
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_track_ids

    def update_tracks(self, detections: list):
        self.cleanup_undetected_tracks(detections)
        self.cleanup_stale_tracks()
        
        filtered_detections = []
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        
        for det in detections:
            center = det['center']
            distance_to_roi = cv2.pointPolygonTest(roi_polygon_points, (int(center[0]), int(center[1])), True)
            
            if distance_to_roi > -50:
                filtered_detections.append(det)
        
        current_detections = []
        for det in filtered_detections:
            current_detections.append({
                'box': det['box'],
                'class': det['class'],
                'center': det['center'],
                'confidence': det.get('confidence', 0.5)
            })
        
        if self.active_tracks and current_detections:
            matched_pairs, unmatched_detections, unmatched_tracks = self.match_detections_to_tracks(current_detections)
            
            for track_id, det_idx in matched_pairs:
                det = current_detections[det_idx]
                self.active_tracks[track_id].update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': self.active_tracks[track_id]['seen_frames'] + 1,
                    'confidence': max(self.active_tracks[track_id].get('confidence', 0), det['confidence'])
                })
                # Re-clasificar mientras la categoria no sea definitiva.
                # Si la HISTERESIS esta activa, seguimos llamando a classify
                # tambien DESPUES del commit: classify devuelve el resultado
                # cacheado (O(1)) y solo re-muestrea la cara a baja cadencia
                # para poder corregir un genero mal fijado.
                _cur_cat = self.active_tracks[track_id].get('category', 'Personas')
                _needs_classify = (
                    _cur_cat in ('Personas', 'Desconocido')
                    or AnalyticsConfig.DEMO_HYSTERESIS_ENABLED
                )
                if (_needs_classify
                        and (self.frame_counter + track_id) % self._imagenes(self._classify_every_n) == 0
                        and hasattr(self, 'last_processed_frame')
                        and self.last_processed_frame is not None):
                    new_cat = self._classify_person(self.last_processed_frame, det['box'], track_id)
                    # Solo guardamos categorias firmes; 'Desconocido' es estado
                    # transitorio del classifier mientras acumula muestras.
                    if new_cat in ('Hombres', 'Mujeres', 'Ninos'):
                        self.active_tracks[track_id]['category'] = new_cat
                # Append class observation for stability voting
                self.track_history[track_id].append(det['center'])
                try:
                    self.track_class_history[track_id].append(det['class'])
                except Exception:
                    pass
                # Update stable class if threshold reached
                try:
                    self._update_track_class_stability(track_id)
                except Exception:
                    pass
            
            for track_id in unmatched_tracks:
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['last_seen'] = self.frame_counter
            
            for det_idx in unmatched_detections:
                det = current_detections[det_idx]
                self._create_new_track(det)
        
        elif current_detections:
            for det in current_detections:
                self._create_new_track(det)

    def _create_new_track(self, detection: Dict):
        new_id = self.next_id
        self.next_id += 1

        center = detection['center']
        roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
        is_inside = self.is_inside_polygon(center, roi_polygon_points)

        # Clasificar persona (género/edad). Si el classifier devuelve
        # "Desconocido" (no convergio aun), dejamos category='Personas'
        # como estado pendiente para que se siga reclasificando en frames
        # siguientes (en _update_tracks_botsort).
        category = 'Personas'
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
            cls_result = self._classify_person(
                self.last_processed_frame, detection['box'], new_id
            )
            if cls_result in ('Hombres', 'Mujeres', 'Ninos'):
                category = cls_result

        # Registrar la persona en el contador.
        if hasattr(self, '_people_counter'):
            is_new = self._people_counter.register_track(new_id)
            if is_new and hasattr(self, '_analytics_logger'):
                demo = self._demographics.get_cached(new_id) if hasattr(self, '_demographics') else None
                self._analytics_logger.log_person_detected(
                    new_id,
                    gender=demo.get('gender', '') if demo else '',
                    age_range=demo.get('age_range', '') if demo else '',
                )

        track_data = {
            'class': detection['class'],
            'box': detection['box'],
            'center': center,
            'last_seen': self.frame_counter,
            'seen_frames': 1,
            'counted': False,
            'is_inside': is_inside,
            'has_been_inside': is_inside,
            'frames_in_roi': 1 if is_inside else 0,
            'total_frames_in_roi': 0,
            'frames_out_roi': 0 if is_inside else 1,
            'entry_frame': self.frame_counter if is_inside else None,
            'frames_without_detection': 0,
            'confidence': detection.get('confidence', 0.5),
            'category': category,
            # Pertenencia a la zona (Hito 2/5) tambien en la ruta fallback.
            'in_zone': detection.get('in_zone'),
        }
        
        if is_inside:
            track_data['entry_time'] = time.time()
            track_data['last_alert_minute'] = 0
        
        self.active_tracks[new_id] = track_data
        self.track_history[new_id].append(center)
        # Initialize class history for stabilization
        try:
            self.track_class_history[new_id].append(detection['class'])
            self._update_track_class_stability(new_id)
        except Exception:
            pass
        
        if self.debug_mode and is_inside:
            staff_name = self.get_staff_display_name(detection['class'])
            confidence = detection.get('confidence', 0)
            print(f"🆕 {staff_name} ({confidence:.2f}) detectado en ROI")

    def _update_tracks_botsort(self, detections: list):
        """Actualiza tracks usando los IDs estables de BoTSORT.

        A diferencia del matching manual por IoU, BoTSORT ya asigna IDs
        consistentes usando Kalman filter + IoU + apariencia. Solo necesitamos
        sincronizar nuestro dict active_tracks con esos IDs.
        """
        current_botsort_ids = set()
        self._purgar_demografia()

        for det in detections:
            tid = det.get('track_id')
            if tid is None:
                # Sin track ID (frame sin tracking), usar fallback
                self._create_new_track(det)
                continue

            current_botsort_ids.add(tid)

            if tid in self.active_tracks:
                # Track existente: actualizar posición y metadata
                track = self.active_tracks[tid]
                track.update({
                    'box': det['box'],
                    'center': det['center'],
                    'last_seen': self.frame_counter,
                    'seen_frames': track['seen_frames'] + 1,
                    'confidence': det['confidence'],
                    'frames_without_detection': 0,
                    # Pertenencia a la zona (Hito 2, ancla pie). La usa la
                    # maquina de entrada/salida por persona (Hito 5). Durante
                    # oclusion el track no se actualiza -> conserva el ultimo
                    # valor (no dispara salida espuria).
                    'in_zone': det.get('in_zone'),
                })
                self.track_history[tid].append(det['center'])

                # Re-clasificar mientras la categoria no sea definitiva.
                # Con HISTERESIS activa tambien post-commit (cacheado O(1);
                # re-muestrea a baja cadencia para corregir generos mal fijados).
                _cur_cat = track.get('category', 'Personas')
                _needs_classify = (
                    _cur_cat in ('Personas', 'Desconocido')
                    or AnalyticsConfig.DEMO_HYSTERESIS_ENABLED
                )
                if (_needs_classify
                        and (self.frame_counter + tid) % self._imagenes(self._classify_every_n) == 0
                        and self.last_processed_frame is not None):
                    new_cat = self._classify_person(self.last_processed_frame, det['box'], tid)
                    if new_cat in ('Hombres', 'Mujeres', 'Ninos'):
                        track['category'] = new_cat

                # Registrar la persona en el contador.
                if hasattr(self, '_people_counter'):
                    self._people_counter.register_track(tid)
            else:
                # Nuevo track de BoTSORT: crear entrada. Si es un numero que
                # volvio antes de 9 s, conserva su genero (12-oct-2026).
                demografia_aplazada.cancelar(
                    getattr(self, '_demographics', None), tid)
                center = det['center']
                roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
                is_inside = self.is_inside_polygon(center, roi_polygon_points)

                category = 'Personas'
                if self.last_processed_frame is not None:
                    cls_result = self._classify_person(
                        self.last_processed_frame, det['box'], tid
                    )
                    if cls_result in ('Hombres', 'Mujeres', 'Ninos'):
                        category = cls_result

                self.active_tracks[tid] = {
                    'class': det['class'],
                    'box': det['box'],
                    'center': center,
                    'last_seen': self.frame_counter,
                    'seen_frames': 1,
                    'counted': False,
                    'is_inside': is_inside,
                    'has_been_inside': is_inside,
                    'frames_in_roi': 1 if is_inside else 0,
                    'total_frames_in_roi': 0,
                    'frames_out_roi': 0 if is_inside else 1,
                    'entry_frame': self.frame_counter if is_inside else None,
                    'frames_without_detection': 0,
                    'confidence': det.get('confidence', 0.5),
                    'category': category,
                    # Pertenencia a la zona (Hito 2, ancla pie) -> maquina de
                    # entrada/salida por persona (Hito 5).
                    'in_zone': det.get('in_zone'),
                }
                if is_inside:
                    self.active_tracks[tid]['entry_time'] = time.time()
                    self.active_tracks[tid]['last_alert_minute'] = 0

                self.track_history[tid].append(center)

                # Registrar la persona en el contador.
                if hasattr(self, '_people_counter'):
                    is_new = self._people_counter.register_track(tid)
                    if is_new and hasattr(self, '_analytics_logger'):
                        demo = self._demographics.get_cached(tid) if hasattr(self, '_demographics') else None
                        self._analytics_logger.log_person_detected(
                            tid,
                            gender=demo.get('gender', '') if demo else '',
                            age_range=demo.get('age_range', '') if demo else '',
                        )

                if self.debug_mode and is_inside:
                    print(f"🆕 [{category}] track_id={tid} ({det.get('confidence', 0):.2f}) en ROI")

        # Limpiar tracks (personas) que BoTSORT ya no reporta.
        stale_ids = []
        for tid in list(self.active_tracks.keys()):
            if tid not in current_botsort_ids:
                track = self.active_tracks[tid]
                track['frames_without_detection'] = track.get('frames_without_detection', 0) + 1
                if track['frames_without_detection'] >= self._imagenes(self.max_frames_without_detection):
                    stale_ids.append(tid)

        for tid in stale_ids:
            self._remove_track(tid)

    def _resolve_track_identities(self, frame: Optional[np.ndarray] = None) -> None:
        """Resuelve el persistent_id de cada track activo (Hito 4 + Hito 9).

        Capa 2 de identidad: mapea tracker_id (corto plazo, ByteTrack) ->
        persistent_id (largo plazo, galeria facial). Logica de la seccion 2
        del plan:
          - Si el reidentificador YA mapeo este track a un person_uuid (su
            rostro paso los filtros de calidad y matcheo o se registro), el
            track queda CONFIRMADO con ese persistent_id y el uuid entra al
            conteo de unicos confirmados (que NO se infla: un uuid que
            reaparece con otro tracker_id no vuelve a sumar).
          - Si aun no hay rostro confirmado, el track recibe un persistent_id
            PROVISIONAL ('prov:<tid>') y NO se cuenta como visitante unico
            definitivo (se resolvera en frames siguientes al verse la cara).

        FUSION: si dos tracker_id distintos matchean el MISMO rostro, el
        reidentificador los mapea al mismo person_uuid -> ambos tracks
        comparten persistent_id y el set de unicos (indexado por uuid) los
        deduplica automaticamente.

        Rendimiento: NO recomputa embeddings; solo consulta el mapeo cacheado
        del reidentificador (O(1) por track). El embedding ya se calculo (o no)
        en la rama de demografia, throttled.
        """
        reid = getattr(self, '_reidentifier', None)
        reid_ok = reid is not None and reid.is_available
        # Re-ID CORPORAL (Hito 9): 2a senal cuando NO hay rostro (espaldas).
        # El ROSTRO MANDA (se evalua primero); el cuerpo solo ENLAZA a una
        # identidad YA confirmada por cara (nunca crea nuevas -> no infla).
        body = getattr(self, '_body_reidentifier', None)
        body_ok = (body is not None and body.is_available
                   and frame is not None and getattr(frame, 'size', 0) > 0)
        # Cadencia y antiguedad en tiempo real, no en imagenes (`_imagenes`).
        _every = max(1, self._imagenes(int(AnalyticsConfig.BODY_REID_EVERY_N)))
        _min_h = int(AnalyticsConfig.BODY_REID_MIN_CROP_H)
        _min_seen = self._imagenes(int(AnalyticsConfig.BODY_REID_MIN_SEEN_FRAMES))
        _links = 0
        for tid, track in self.active_tracks.items():
            uid = reid.get_persistent_id(tid) if reid_ok else None
            _due = (body_ok and track.get('seen_frames', 0) >= _min_seen
                    and (self.frame_counter + tid) % _every == 0)
            if uid:
                track['persistent_id'] = uid
                track['persistent_confirmed'] = True
                track['identity_source'] = 'face'
                # PERSONAL enrolado: se marca en el track (para que los
                # modos y el heatmap lo salten) y NO cuenta como visitante.
                # OJO: si el UNIFORME ya lo marco (racha corporal), el que
                # su rostro no este enrolado NO lo desmarca.
                _ficha_per = self._ficha_personal_tienda(uid)
                if _ficha_per is not None:
                    track['es_personal'] = True
                    track['personal_rol'] = _ficha_per.get('rol')
                elif not track.get('es_personal'):
                    track['es_personal'] = False
                    self._confirmed_unique.add(uid)
                # Enrolar su cuerpo en la galeria corporal (throttled) -> poder
                # reconocerlo si reaparece de espaldas dentro de la sesion.
                if _due:
                    crop = self._body_crop(frame, track.get('box'), _min_h)
                    if crop is not None:
                        emb = body.compute_embedding(crop)
                        if emb is not None:
                            body.enroll(uid, emb)
                            # ¿Viste un uniforme enrolado? (identifica de
                            # espaldas; el rostro no puede)
                            self._marcar_si_uniforme(track, emb)
                continue
            # Sin rostro -> intentar enlazar por CUERPO (entra de espaldas).
            matched = None
            if _due:
                crop = self._body_crop(frame, track.get('box'), _min_h)
                if crop is not None:
                    emb = body.compute_embedding(crop)
                    if emb is not None:
                        # ¿Viste un uniforme enrolado? Funciona TAMBIEN para
                        # los prov: (de espaldas, sin rostro que confirmar).
                        self._marcar_si_uniforme(track, emb)
                        m = body.match(emb)
                        if m is not None:
                            matched = m[0]
                            track['_body_match'] = matched  # cache para el track
            # Reusar el ultimo match corporal del track (el cuerpo no cambia
            # mientras el track vive) -> no recomputa cada frame.
            if matched is None:
                matched = track.get('_body_match')
            if matched:
                track['persistent_id'] = matched
                track['persistent_confirmed'] = True
                track['identity_source'] = 'body'
                _ficha_per = self._ficha_personal_tienda(matched)
                if _ficha_per is not None:
                    track['es_personal'] = True
                    track['personal_rol'] = _ficha_per.get('rol')
                elif not track.get('es_personal'):
                    track['es_personal'] = False
                    self._confirmed_unique.add(matched)
                _links += 1
            else:
                # Sin rostro ni match corporal -> provisional (no cuenta como
                # unico definitivo hasta confirmarse).
                track['persistent_id'] = f"prov:{tid}"
                track['persistent_confirmed'] = False
        self._body_links = _links  # tracks enlazados por cuerpo este frame

    def _ficha_personal_tienda(self, pid) -> Optional[dict]:
        """La ficha si `pid` es PERSONAL de tienda enrolado (14-ago-2026).

        El personal (seguridad/pasilleros con foto en `/api/personal`) NO es
        un cliente: no cuenta como visitante, no se fotografia, no calienta
        el heatmap ni entra a la demografia — el marketing es solo de
        clientes. Perezoso y a prueba de todo: sin galeria o con galeria
        vacia responde None en microsegundos y jamas lanza (esto corre
        dentro del bucle de inferencia).
        """
        try:
            ident = getattr(self, '_identificador_personal', None)
            if ident is None:
                from .personal_galeria import get_identificador
                ident = get_identificador()
                self._identificador_personal = ident
            return ident.ficha_de(pid, getattr(self, '_reidentifier', None))
        except Exception:                                   # noqa: BLE001
            return None

    def _marcar_si_uniforme(self, track: dict, emb) -> None:
        """Marca el track como PERSONAL si viste un uniforme enrolado.

        El uniforme (bordado/colores, ficha tipo 'uniforme') identifica de
        ESPALDAS, que es como más trabaja un pasillero — el rostro no puede.
        Exige RACHA de 2 coincidencias en cómputos sucesivos del embedding
        corporal: una coincidencia suelta puede ser un cliente con ropa
        parecida, y el falso positivo caro es borrarlo del marketing.
        Nunca lanza (bucle de inferencia).
        """
        try:
            if emb is None:
                return
            ident = getattr(self, '_identificador_personal', None)
            if ident is None:
                from .personal_galeria import get_identificador
                ident = get_identificador()
                self._identificador_personal = ident
            ficha = ident.ficha_por_cuerpo(emb)
        except Exception:                                   # noqa: BLE001
            return
        if ficha is None:
            track['_uniforme_racha'] = 0
            return
        racha = int(track.get('_uniforme_racha') or 0) + 1
        track['_uniforme_racha'] = racha
        if racha < 2 or track.get('es_personal'):
            return
        track['es_personal'] = True
        track['personal_rol'] = ficha.get('rol')
        track['personal_nombre'] = ficha.get('nombre')
        uid = track.get('persistent_id')
        if uid and not str(uid).startswith('prov:'):
            # Pudo contarse como visitante en los frames previos a la
            # confirmacion del uniforme: se descuenta.
            self._confirmed_unique.discard(uid)

    def _body_crop(self, frame: np.ndarray, box, min_h: int):
        """Recorta el cuerpo de una persona del frame para el Re-ID corporal.
        Devuelve None si la caja es invalida o el recorte es mas bajo que
        min_h px (cuerpos diminutos/lejanos dan embeddings basura)."""
        if frame is None or box is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except (TypeError, ValueError):
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (y2 - y1) < int(min_h) or (x2 - x1) < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if getattr(crop, 'size', 0) > 0 else None

    # ── Memoria con FECHA (HITO 1, 18-ago-2026) ─────────────────────────
    def _anotar_zona(self, tipo: str, pid: Any, ts: float,
                     permanencia_s: float = 0.0) -> None:
        """Deja constancia EN DISCO de una transicion de zona confirmada.

        Se llama solo desde `_process_zone_transitions`, que es la maquina
        AUTORITATIVA por `persistent_id`. La contabilidad legacy por track_id
        (`process_entry_exit_logic`) no anota nada: duplicaria los numeros.

        Los ids PROVISIONALES (`prov:`, sin rostro) se descartan a proposito.
        Una oclusion que parta el track da un pid nuevo, asi que guardarlos
        inflaria los visitantes unicos con la misma persona varias veces.
        Preferimos contar de menos a inventar gente.

        Nunca lanza ni frena el frame: el almacen encola en memoria, vuelca
        amortiguado y se traga sus propios errores.
        """
        try:
            cam = getattr(self, '_camera_id_actual', None)
            if cam is None or not pid or str(pid).startswith('prov:'):
                return
            alm = _almacen_tienda.get_almacen()
            if tipo == _almacen_tienda.EVENTO_ENTRADA:
                alm.anotar_entrada(cam, pid, ts)
            else:
                alm.anotar_salida(cam, pid, permanencia_s, ts)
        except Exception:                                   # noqa: BLE001
            pass

    def cerrar_almacen_tienda(self) -> None:
        """Vuelca lo pendiente al desconectarse el cliente.

        La lista de `_METODOS_DE_CIERRE` de `app.py` llama a esto en cada
        desconexion. Sin ello, los ultimos segundos de conteo (hasta
        `PERIODO_VOLCADO_S`) se quedarian en memoria y se irian con el
        procesador, que es justo el momento en el que mas falta hacen: la
        reconexion siguiente los va a leer para reanudar.
        """
        try:
            _almacen_tienda.get_almacen().volcar(force=True)
        except Exception:                                   # noqa: BLE001
            pass

    def _sembrar_desde_almacen(self, state: Dict[str, Any],
                               camera_id: Any) -> None:
        """Rehidrata los contadores de zona de ESTA camara con lo ya contado
        hoy, en vez de arrancar de cero.

        Por que: el procesador RENACE en cada reconexion del cliente (~5 s tras
        cualquier corte de red). Sin esto, cada parpadeo reiniciaba entradas,
        salidas, unicos y permanencia, y el dashboard enseñaba «lo que lleve
        viva esta sesion» creyendo que enseñaba el dia.

        Quien tenga mas entradas que salidas se restaura DENTRO: es la unica
        pista fiable de que seguia en el area cuando se corto la conexion. Su
        cronometro abierto vuelve a empezar (el tramo interrumpido no hay forma
        de saberlo), pero el tiempo YA acumulado se conserva. Y si en realidad
        se habia ido, la histeresis lo saca en unos frames y cuenta su salida.
        """
        state['_zone_state'] = {}
        state['_total_entries'] = 0
        state['_total_exits'] = 0
        state['_unique_entered'] = set()
        try:
            datos = _almacen_tienda.get_almacen().contadores_de(camera_id)
        except Exception:                                   # noqa: BLE001
            return                       # sin memoria se empieza en cero
        ahora = time.time()
        for pid, d in (datos.get('personas') or {}).items():
            entradas = int(d.get('entradas', 0) or 0)
            salidas = int(d.get('salidas', 0) or 0)
            dentro = entradas > salidas
            state['_zone_state'][pid] = {
                'estado': 'DENTRO' if dentro else 'FUERA',
                'pending': None, 'pending_frames': 0,
                'conteo_entradas': entradas, 'conteo_salidas': salidas,
                'tiempo_total_dentro': float(d.get('permanencia_s', 0.0) or 0.0),
                'entry_time': ahora if dentro else None,
                'last_ts': ahora,
            }
            if entradas > 0:
                state['_unique_entered'].add(pid)
        state['_total_entries'] = int(datos.get('entradas', 0) or 0)
        state['_total_exits'] = int(datos.get('salidas', 0) or 0)
        if state['_total_entries'] or state['_total_exits']:
            logger.info(
                'tienda: camara %s reanuda la jornada con %d entradas, '
                '%d salidas y %d visitantes unicos ya contados',
                camera_id, state['_total_entries'], state['_total_exits'],
                len(state['_unique_entered']))

    def _process_zone_transitions(self) -> None:
        """Maquina de estados de zona POR persistent_id (Hito 5).

        Integra Hito 2 (in_zone via sv.PolygonZone, ancla pie) con Hito 4
        (persistent_id) para contar entradas/salidas POR PERSONA con
        anti-rebote, y derivar los visitantes unicos del area sin inflar.

        - Estado por persona: FUERA / DENTRO.
        - Una transicion se CONFIRMA solo tras ZONE_HYSTERESIS_FRAMES frames
          consecutivos en el nuevo estado (evita entradas/salidas fantasma de
          alguien parado en el borde del poligono).
        - FUERA->DENTRO = entrada: visita +1 de esa persona, _total_entries +1
          y la persona entra al set de visitantes unicos del area.
          DENTRO->FUERA = salida: +1 y acumula tiempo_total_dentro.
        - Solo personas con persistent_id CONFIRMADO disparan transiciones
          (ZONE_ENTRY_REQUIRES_CONFIRMED); las provisionales esperan a tener
          rostro -> el conteo de unicos es exacto.
        - Una persona cuyo track desaparecio (ya no esta en active_tracks) se
          observa como FUERA -> tras histeresis cuenta su salida.
        """
        now = time.time()
        # Override POR INSTANCIA sobre la politica global. El defecto no
        # cambia: sin fijar el atributo se comporta exactamente igual que
        # antes. Lo usa PermanenciaArea, que CRONOMETRA una estancia y no
        # necesita identidad unica global: en una camara que no saca rostros
        # (picada alta, contraluz, gente de espaldas) exigirla dejaba la
        # maquina de zonas VACIA y el modo no medía nada, sin un solo error.
        require_conf = bool(getattr(self, '_zone_requiere_confirmado',
                                    AnalyticsConfig.ZONE_ENTRY_REQUIRES_CONFIRMED))
        # En tiempo real (`_imagenes`, 5-oct-2026): a 10 fps, 3 imagenes eran
        # 0,3 s y quien estaba parado en el borde entraba y salia varias veces.
        hyst = max(1, self._imagenes(int(AnalyticsConfig.ZONE_HYSTERESIS_FRAMES)))

        # 1) Observacion por persona desde los tracks activos (DENTRO si
        #    cualquiera de sus tracks tiene el pie en la zona).
        observed: Dict[str, bool] = {}
        for _tid, track in self.active_tracks.items():
            if require_conf and not track.get('persistent_confirmed', False):
                continue
            # El PERSONAL enrolado no entra a la maquina DENTRO/FUERA: sus
            # entradas/salidas inflarian el aforo y la demografia del area.
            if track.get('es_personal'):
                continue
            pid = track.get('persistent_id')
            if not pid:
                continue
            # Los provisionales solo entran cuando NO se exige confirmacion.
            # Con ellos se mide PRESENCIA, no visitantes unicos: una oclusion
            # que parta el track da un pid nuevo (ver el aviso del YAML).
            if require_conf and str(pid).startswith('prov:'):
                continue
            observed[pid] = observed.get(pid, False) or bool(
                track.get('in_zone'))

        # 2) Personas ya conocidas SIN track activo este frame -> FUERA
        #    (salieron del encuadre; tras histeresis contara salida).
        for pid in self._zone_state.keys():
            observed.setdefault(pid, False)

        # 3) Histeresis + confirmacion de transiciones.
        for pid, dentro in observed.items():
            nuevo = 'DENTRO' if dentro else 'FUERA'
            st = self._zone_state.get(pid)
            if st is None:
                # Primera vez: estado inicial FUERA. Si ya esta DENTRO, la
                # transicion de abajo contara su primera entrada tras histeresis.
                st = {
                    'estado': 'FUERA', 'pending': None, 'pending_frames': 0,
                    'conteo_entradas': 0, 'conteo_salidas': 0,
                    'tiempo_total_dentro': 0.0, 'entry_time': None,
                    'last_ts': now,
                }
                self._zone_state[pid] = st

            if nuevo == st['estado']:
                st['pending'] = None
                st['pending_frames'] = 0
            else:
                if st['pending'] == nuevo:
                    st['pending_frames'] += 1
                else:
                    st['pending'] = nuevo
                    st['pending_frames'] = 1
                if st['pending_frames'] >= hyst:
                    if nuevo == 'DENTRO':
                        st['estado'] = 'DENTRO'
                        st['conteo_entradas'] += 1
                        st['entry_time'] = now
                        self._total_entries += 1
                        self._unique_entered.add(pid)
                        self._anotar_zona(
                            _almacen_tienda.EVENTO_ENTRADA, pid, now)
                    else:
                        st['estado'] = 'FUERA'
                        st['conteo_salidas'] += 1
                        _dentro_s = 0.0
                        if st.get('entry_time'):
                            _dentro_s = max(0.0, now - st['entry_time'])
                            st['tiempo_total_dentro'] += _dentro_s
                        st['entry_time'] = None
                        self._total_exits += 1
                        self._anotar_zona(
                            _almacen_tienda.EVENTO_SALIDA, pid, now,
                            _dentro_s)
                    st['pending'] = None
                    st['pending_frames'] = 0
            st['last_ts'] = now

    def _drenar_alertas(self) -> list:
        """Devuelve las alertas acumuladas y vacia la cola.

        Se drena al construir la metadata de cada frame: asi cada alerta viaja
        exactamente una vez."""
        pendientes = getattr(self, '_alertas_pendientes', None)
        if not pendientes:
            return []
        self._alertas_pendientes = []
        return pendientes

    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False, camera_id: Any = 1, heatmap_activate: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Solo el modelo de personas es necesario.
        if self.person_model is None:
            raise RuntimeError("Modelo de personas no inicializado")

        if roi is not None and isinstance(roi, list) and len(roi) >= 3:
            self.roi_polygon = np.array(roi, np.int32)
            if self.debug_mode:
                print(f"📍 ROI actualizado: {len(roi)} puntos")

        # Usar estado por cámara para mantener separación de tracks
        cam_state = self._get_cam_state(camera_id)

        # Auto-detección de ángulo: re-evaluar con las muestras acumuladas
        # (de frames previos). Actualiza self._auto_angle, que usa el filtro.
        if self._auto_angle_enabled:
            prev = self._auto_angle
            self._auto_angle = self._auto_angle_detector.evaluate()
            if self._auto_angle != prev and self._auto_angle is not None:
                logger.info(
                    "Auto-ángulo cámara=%s -> %s (mediana h/w=%.2f, n=%d)",
                    camera_id, self._auto_angle,
                    self._auto_angle_detector.median_hw,
                    self._auto_angle_detector.n_samples)

        # Umbral del filtro de aspecto segun el perfil/angulo de esta camara
        # (0.0 = desactivado, p.ej. en cenital).
        _aspect_min = self._aspect_ratio_min_for(camera_id)

        # Evitar solapamiento concurrente entre llamadas a process_frame
        with self._process_lock:
            self._push_state(cam_state)
            # Camara EN CURSO. La necesitan el almacen de tienda y el sidecar
            # de la captura, que llavean por camera_id (= device_id) y NUNCA
            # por el nombre visible: cuatro ventanas «iVMS-4200» son cuatro
            # camaras distintas (H-39). El lock cubre el frame entero, asi
            # que dos camaras no pueden pisarse este atributo.
            self._camera_id_actual = camera_id

            try:
                self.frame_counter += 1
                self.last_processed_frame = image.copy()

                # Ruta sin estado por camara (fallback): el acumulador se
                # toma del registro global la primera vez que se conoce el
                # camera_id, con su estado restaurado de disco.
                if (AnalyticsConfig.HEATMAP_ENABLED
                        and getattr(self, '_heatmap', None) is None):
                    self._heatmap = _heatmaps.obtener(
                        camera_id,
                        getattr(self, '_camera_display_name', None))

                # Detectar sobre la imagen COMPLETA (sin máscara) para evitar
                # artefactos en los bordes del ROI que generan falsos positivos.
                inference_image = image

                # Preparar polígono ROI para filtrado espacial
                roi_active = activate_roi and hasattr(self, 'roi_polygon') and self.roi_polygon is not None
                roi_poly = self.roi_polygon.reshape((-1, 1, 2)) if roi_active else None

                detections = []
                staff_detected = 0

                # ── Deteccion de personas (RF-DETR compartido, clase 0) ──
                if self.person_model is not None:
                    # Conf personas: el piso global (PERSON_DETECT_CONF) o el
                    # del perfil de ESTA camara si lo define. Una camara alta
                    # o exterior con gente pequena/parcial necesita un piso
                    # mas bajo que el pasillo del super, y bajarlo global
                    # subiria falsos positivos en todas las demas.
                    _cfg_cam = self._perfil_camara(camera_id)
                    _per_conf = max(
                        float(_cfg_cam.PERSON_DETECT_CONF),
                        float(self.confidence_threshold))
                    # detectar_sv entrega sv.Detections DIRECTO (adios al
                    # Results de ultralytics y a from_ultralytics). Sin iou=
                    # ni max_det=: RF-DETR es end-to-end (no hay NMS que
                    # ajustar); resolucion/dtype los gobiernan los knobs
                    # RFDETR_* del modulo compartido, no PERSON_DETECT_IMGSZ.
                    sv_det = deteccion_compartida.detectar_sv(
                        inference_image, conf_min=_per_conf, clases=[0])
                    # Gente del FONDO fuera, ANTES de ByteTrack: lo que se
                    # quita aqui no genera seguimiento, ficha, entrada,
                    # demografia, anaqueles ni calor (`filtro_fondo.py`).
                    # El umbral es del perfil de ESTA camara; global = 0.
                    sv_det = filtro_fondo.filtrar(
                        sv_det, getattr(_cfg_cam, 'PERSON_MIN_HEIGHT_PX', 0),
                        camera_id)
                    # Areas a ignorar del recuadro (12-oct-2026,
                    # `zonas_ignorar.py`): quien queda con la caja ENTERA
                    # dentro tampoco llega a ByteTrack.
                    sv_det = zonas_ignorar.filtrar(
                        sv_det, getattr(self, '_zonas_ignorar', None),
                        camera_id)

                    # ── Tracking de corto plazo con sv.ByteTrack ──
                    # El tracking lo hace ByteTrack (no un tracker interno del
                    # detector), POR-CAMARA. update_with_detections DEVUELVE
                    # solo las detecciones con track CONFIRMADO (tracker_id
                    # valido, ya filtradas: minimum_consecutive_frames
                    # descarta las efimeras antes de emitirlas -> anti
                    # falso-positivo).
                    if self._byte_tracker is None:
                        self._byte_tracker = self._make_byte_tracker(camera_id)
                    # Memoria y confirmacion en SEGUNDOS con los fps reales
                    # de esta camara, nunca menos que en imagenes
                    # (`ritmo_imagenes.py`, 5-oct-2026).
                    ritmo_imagenes.ajustar_bytetrack(
                        self._byte_tracker, time.time(), _cfg_cam)
                    sv_det = self._byte_tracker.update_with_detections(sv_det)

                    # ── Conteo de zona (sv.PolygonZone) — Hito 2 ──
                    # Aforo INSTANTANEO: cuantas personas estan DENTRO del area
                    # (ancla configurable, por defecto el pie). Se calcula sobre
                    # TODAS las detecciones confirmadas, ANTES del filtro ROI de
                    # abajo -> conteo correcto este o no activo el filtro ROI.
                    # El indice de _inside_mask se alinea con sv_det (= boxes_p).
                    _zone = self._get_zone()
                    if _zone is not None and len(sv_det) > 0:
                        _inside_mask = _zone.trigger(sv_det)  # setea current_count
                        self._zone_count = int(_zone.current_count)
                    else:
                        _inside_mask = None
                        self._zone_count = 0

                    if len(sv_det) > 0 and sv_det.tracker_id is not None:
                        boxes_p = sv_det.xyxy
                        confs_p = (sv_det.confidence
                                   if sv_det.confidence is not None
                                   else np.full(len(sv_det), 0.5))
                        track_ids_p = sv_det.tracker_id

                        # Offset de track_id UNICO por camara: el reidentificador
                        # es compartido y los track_id de ByteTrack son locales a
                        # cada camara -> sin namespace colisionarian (camara A
                        # track 1 == camara B track 1). 1a camara=100000 (compat).
                        _cam_off = (
                            self._reidentifier.camera_track_namespace(camera_id)
                            if getattr(self, '_reidentifier', None) is not None
                            else 100000)

                        for i in range(boxes_p.shape[0]):
                            box = boxes_p[i]
                            center = self.center_of(box)
                            confidence = confs_p[i] if i < len(confs_p) else 0.5

                            # Filtro ROI
                            if roi_poly is not None:
                                if cv2.pointPolygonTest(roi_poly, (int(center[0]), int(center[1])), False) < 0:
                                    continue

                            bw = box[2] - box[0]
                            bh = box[3] - box[1]

                            # Auto-detección de ángulo: alimentamos el detector
                            # con TODAS las personas (antes del filtro) para que
                            # vea la forma real, incluidas las anchas del cenital.
                            if self._auto_angle_enabled:
                                self._auto_angle_detector.add_person(bw, bh)

                            # Filtro de aspecto configurable por perfil de
                            # cámara. En cenital `_aspect_min`=0.0 lo desactiva,
                            # porque una persona vista desde arriba se ve más
                            # ancha que alta y aquí se descartaría.
                            if _aspect_min > 0.0 and bh < bw * _aspect_min:
                                continue

                            # Offset de track_id unico por camara (anti-colision
                            # en el reidentificador compartido).
                            tid_p = int(track_ids_p[i]) + _cam_off if track_ids_p is not None else None
                            # Pertenencia a la zona segun sv.PolygonZone (ancla
                            # pie/centro). Disponible para el Hito 5 (entradas/
                            # salidas por persistent_id). None si no hay zona.
                            _in_zone = (bool(_inside_mask[i])
                                        if _inside_mask is not None else None)
                            detections.append({
                                'class': -1,  # -1 = persona genérica
                                'box': box,
                                'center': center,
                                'confidence': confidence,
                                'track_id': tid_p,
                                'in_zone': _in_zone,
                            })
                            staff_detected += 1

                            # Mapa de calor: acumular posicion de los pies.
                            # El PERSONAL enrolado no calienta el mapa (el
                            # marketing es de clientes); la identidad es la
                            # del frame ANTERIOR del track — la fuga son los
                            # primeros frames hasta confirmar el rostro.
                            # `observar_escena` (throttle interno 1 s) ancla
                            # el mapa a SU escena: si el operador cambio el
                            # canal de iVMS, add_person no acumula.
                            if (AnalyticsConfig.HEATMAP_ENABLED
                                    and hasattr(self, '_heatmap')
                                    and self._heatmap is not None):
                                self._heatmap.observar_escena(inference_image)
                                if not (self.active_tracks.get(tid_p)
                                        or {}).get('es_personal'):
                                    self._heatmap.add_person(
                                        box,
                                        inference_image.shape[1],
                                        inference_image.shape[0])

                if self.debug_mode and detections:
                    print(f"📊 Frame {self.frame_counter}: {len(detections)} detectados ({staff_detected} con track)")

                # Actualizar tracks usando IDs de BoTSORT (estables con Kalman)
                self._update_tracks_botsort(detections)

                # Capturas: foto por persona (tracks ESTABLECIDOS) + dedup Re-ID
                # + relleno de genero/edad cuando converge la demografia.
                if AnalyticsConfig.CAPTURE_DETECTIONS:
                    self._capture_established_tracks(inference_image)
                    self._update_captures_demographics()

                # Resolucion de identidad (Hito 4): tracker_id -> persistent_id
                # (confirmado por rostro) o id provisional. Va DESPUES de la
                # demografia (que es quien mapea el rostro a la galeria) y ANTES
                # de entrada/salida, segun el orden del plan. Se le pasa el
                # frame para el Re-ID corporal (Hito 9): enrolar/matchear cuerpo.
                self._resolve_track_identities(inference_image)

                # Entradas/salidas POR PERSONA (Hito 5): maquina FUERA/DENTRO
                # por persistent_id con anti-rebote -> visitantes unicos del
                # area + entradas/salidas exactas (no infladas).
                self._process_zone_transitions()

                # Avisos de NEGOCIO (aforo alto, pasillo dormido, caida de
                # trafico). Va DESPUES de las transiciones porque lee el aforo
                # que acaban de dejar, y sale por el primer `if` en 1.499 de
                # cada 1.500 frames: se evalua como mucho cada EVAL_S.
                self._evaluar_alertas_negocio()

                # Procesar entrada/salida (LEGACY por track_id: alimenta fotos
                # de entrada/salida y conteos por categoria del dashboard; el
                # conteo AUTORITATIVO de unicos/entradas es el de arriba).
                persons_inside = self.process_entry_exit_logic(image)

                # Verificar alertas periódicas
                self.check_periodic_alerts(image)

                # ── Analitica de personas (conteo + calidad demografica) ──
                self._run_analytics(image)

                # Log periódico
                if self.frame_counter % 30 == 0:
                    if self.debug_mode:
                        print(f"\n📈 Resumen Frame {self.frame_counter}:")
                        print(f"   Empleados en ROI: {persons_inside}")
                        print(f"   Total en área: {self.personas_en_area}")
                        print(f"   Tracks activos: {len(self.active_tracks)}")

                        # Mostrar contadores por empleado
                        for staff_id, count in self.employee_counters.items():
                            if count > 0:
                                staff_name = self.get_staff_display_name(staff_id)
                                print(f"   {staff_name}: {count} veces")

                # Dibujado: modo CLASICO (servidor dibuja y envia la imagen) o
                # modo DIRECTO (no dibuja ni codifica; devuelve las detecciones
                # y el CLIENTE las pinta -> mucha menos latencia/ancho de banda).
                _draw_server = getattr(self, '_draw_server', True)
                _detections_payload = None
                if _draw_server:
                    processed_image = self.draw_detections(
                        image.copy(), persons_inside)
                else:
                    _detections_payload = self._build_detections()
                    processed_image = image  # frame limpio (solo para snapshot)

                # Mapa de calor: snapshot periodico PNG/JSON para el dashboard
                # (sobre la imagen limpia) y overlay en vivo SOLO en modo clasico
                # (en modo directo el cliente no recibe imagen del servidor).
                if (AnalyticsConfig.HEATMAP_ENABLED
                        and hasattr(self, '_heatmap')
                        and self._heatmap is not None):
                    self._heatmap.maybe_save_snapshot(
                        camera_id, background=processed_image,
                        camera_name=getattr(
                            self, '_camera_display_name', None),
                        # El fondo se prefiere SIN personas (antes quedaban
                        # «quemadas» en la base de todos los renders).
                        personas=len(detections)
                        if isinstance(detections, list) else None)
                    if heatmap_activate and _draw_server:
                        processed_image = self._heatmap.render_overlay(
                            processed_image)

                # Metadatos.
                #
                # AQUI CONVIVEN DOS CONTABILIDADES y conviene no confundirlas:
                #
                #   AUTORITATIVA, por `persistent_id` (rostro):
                #     unique_visitors_entered, zone_entries_total,
                #     zone_exits_total, zone_count, analytics_report.
                #     Es la que persiste `tienda_almacen` y la que alimenta el
                #     dashboard y los informes. Una persona que entra tres
                #     veces cuenta UNA como visitante y TRES como entradas.
                #
                #   LEGADO, por `track_id` (`process_entry_exit_logic`):
                #     entry_counts, active_by_category, employee_counters,
                #     total_entries, persons_inside.
                #     Se conserva porque alimenta el panel de categorias del
                #     overlay y el cronometro por persona, pero NO es la
                #     cuenta buena: un track partido por una oclusion cuenta
                #     dos veces. `total_entries` (legado) no es lo mismo que
                #     `zone_entries_total` (autoritativo), aunque los nombres
                #     se parezcan demasiado — quien lea esto que use el
                #     segundo. Sus efectos en disco y por red ya estan
                #     apagados (LEGACY_STAFF_PHOTOS).
                ec = getattr(self, 'entry_counts', {})
                abc = getattr(self, 'active_by_category', {})
                metadata = {
                    # Alertas de este pipeline. Antes iba SIEMPRE vacia: el
                    # cliente no tenia nada que mostrar en su panel lateral y
                    # el reenvio por WhatsApp no tenia nada que enviar. Ahora
                    # lleva un evento por visitante, en el instante en que su
                    # genero y edad convergen (ver _emitir_visitante).
                    'alerts': self._drenar_alertas(),
                    'frame_number': self.frame_counter,
                    'roi_active': activate_roi,
                    'staff_detected': staff_detected,
                    'persons_inside': persons_inside,
                    'persons_in_area': self.personas_en_area,
                    # Aforo instantaneo via sv.PolygonZone (Hito 2): personas
                    # DENTRO del area ahora mismo (ancla pie/centro segun config).
                    'zone_count': int(getattr(self, '_zone_count', 0)),
                    # Identidad persistente (Hito 4): visitantes unicos
                    # CONFIRMADOS por rostro en la sesion (NO inflado por
                    # reapariciones, los provisionales no cuentan). Y cuantos
                    # tracks activos aun son provisionales (sin rostro).
                    'unique_visitors_confirmed': len(
                        getattr(self, '_confirmed_unique', set())),
                    'provisional_tracks': sum(
                        1 for t in self.active_tracks.values()
                        if not t.get('persistent_confirmed', False)),
                    # Total de personas distintas en la galeria persistente
                    # (acumulado historico, DB nunca se borra).
                    'total_unique_persons_db': (
                        self._reidentifier.total_unique_persons
                        if getattr(self, '_reidentifier', None) is not None
                        else 0),
                    # Entradas/salidas por persona (Hito 5): visitantes unicos
                    # del AREA (persistent_id distintos que ENTRARON) + totales
                    # de transiciones confirmadas. Aforo instantaneo = zone_count.
                    'unique_visitors_entered': len(
                        getattr(self, '_unique_entered', set())),
                    'zone_entries_total': int(getattr(self, '_total_entries', 0)),
                    'zone_exits_total': int(getattr(self, '_total_exits', 0)),
                    # Re-ID corporal (Hito 9): tracks re-enlazados por cuerpo
                    # este frame (sin rostro, p.ej. de espaldas).
                    'body_relinked_tracks': int(getattr(self, '_body_links', 0)),
                    'active_tracks': len(self.active_tracks),
                    'employee_counters': dict(self.employee_counters),
                    'entry_counts': dict(ec),
                    'active_by_category': dict(abc),
                    'total_entries': sum(ec.values()),
                }
                # Modo DIRECTO: detecciones para que el cliente las dibuje.
                if _detections_payload is not None:
                    metadata['detections'] = _detections_payload


                # Ángulo de cámara (auto-detectado o manual) para que el
                # cliente sepa qué perfil se está aplicando.
                if hasattr(self, '_auto_angle_detector'):
                    _eff = (self._camera_angle_override
                            or (self._auto_angle if self._auto_angle_enabled
                                else None)
                            or self._default_camera_angle)
                    metadata['camera_angle'] = {
                        'efectivo': _eff,
                        'auto_activo': bool(self._auto_angle_enabled),
                        'auto_detectado': self._auto_angle,
                        'auto_muestras': self._auto_angle_detector.n_samples,
                        'mediana_hw': round(
                            self._auto_angle_detector.median_hw, 2),
                    }

                # Enriquecer metadata con analitica de personas
                if hasattr(self, '_people_counter'):
                    metadata['people_counter'] = self._people_counter.get_stats()
                if hasattr(self, '_demographics'):
                    metadata['demographics'] = self._demographics.get_counts()
                if (hasattr(self, '_heatmap')
                        and self._heatmap is not None):
                    _hm = self._heatmap.get_stats()
                    _hm['activo'] = bool(heatmap_activate)
                    metadata['heatmap'] = _hm

                # Reporte consolidado (Hito 8): throttled (la distribucion
                # genero/edad recorre la galeria -> no cada frame). Se cachea,
                # se mete en metadata y se persiste a JSON (sobrevive reinicios).
                _now_r = time.time()
                if (_now_r - getattr(self, '_last_report_ts', 0.0)
                        >= AnalyticsConfig.REPORT_EVERY_S):
                    self._last_report_ts = _now_r
                    try:
                        self._cached_report = self.get_analytics_report()
                        self.save_analytics_report()
                    except Exception as _re:
                        logger.debug("reporte consolidado fallo: %s", _re)
                if getattr(self, '_cached_report', None) is not None:
                    metadata['analytics_report'] = self._cached_report

                # NOTA: el stand de degustacion NO se engancha aqui. Vive en
                # su propio pipeline, `amazonas_stand.AmazonasStand`, que
                # hereda de esta clase y anade la medicion despues de este
                # metadata.
                #
                # Estuvo aqui detras de un flag global y fue un error: este
                # pipeline lo usa TAMBIEN el cliente de Tienda, asi que al
                # encender el flag Tienda empezo a abrir sesiones y a contar
                # visitantes de un stand que no esta midiendo. Un modo propio
                # hace la eleccion explicita.

                return processed_image, metadata
            except Exception as e:
                logger.error(f"Error procesando frame: {e}")
                import traceback
                traceback.print_exc()
                return image, {'error': str(e)}
            finally:
                # Guardar último frame procesado en el estado de la cámara y restaurar estado global
                try:
                    cam_state['last_processed_frame'] = self.last_processed_frame
                    cam_state['personas_en_area'] = getattr(self, 'personas_en_area', cam_state.get('personas_en_area', 0))
                    cam_state['entry_counts'] = getattr(self, 'entry_counts', cam_state.get('entry_counts', {}))
                    # Contadores ENTEROS (inmutables): hay que reescribirlos al
                    # estado de la camara o el incremento del frame se pierde al
                    # restaurar (los sets/dicts persisten solos por referencia).
                    cam_state['_total_entries'] = int(getattr(self, '_total_entries', 0))
                    cam_state['_total_exits'] = int(getattr(self, '_total_exits', 0))
                except Exception:
                    pass
                # Restaurar estado global
                self._pop_state()

    def set_roi(self, roi_points: List[Tuple[int, int]]):
        self.roi_polygon = np.array(roi_points, np.int32)
        print(f"✅ ROI actualizado a {len(roi_points)} puntos")

    def reset_counter(self):
        self.personas_en_area = 0
        self.employee_counters.clear()
        self.entry_counts = {'Hombres': 0, 'Mujeres': 0, 'Niños': 0, 'Personas': 0}
        self.last_counted_frame = 0
        self.last_counted_id = 0
        self.counted_tracks.clear()
        self.recent_counted_persons.clear()
        self.person_cooldown.clear()
        self.active_tracks.clear()
        # Reiniciar el tracker de corto plazo (IDs de ByteTrack desde cero).
        self._byte_tracker = self._make_byte_tracker()
        # Reiniciar el set de unicos confirmados de la sesion (Hito 4). NO
        # toca la galeria persistente (esa nunca se borra automaticamente).
        self._confirmed_unique = set()
        # Reiniciar la maquina de entrada/salida por persona (Hito 5).
        self._zone_state = {}
        self._total_entries = 0
        self._total_exits = 0
        self._unique_entered = set()
        self.track_history.clear()
        self.movement_history.clear()
        self.sent_entry_photos.clear()
        self.sent_exit_photos.clear()
        self.last_sent_time.clear()
        self.alert_minutes_sent.clear()
        # Reiniciar modulos de analitica
        if hasattr(self, '_demographics'):
            self._demographics.reset()
        if hasattr(self, '_people_counter'):
            self._people_counter.reset()
        print("🔄 Contadores reiniciados")

    def toggle_minimal_info(self):
        self.show_minimal_info = not self.show_minimal_info
        status = "MÍNIMA" if self.show_minimal_info else "COMPLETA"
        print(f"🔧 Información: {status}")

    def get_stats(self) -> Dict[str, Any]:
        persons_inside = 0
        
        for track in self.active_tracks.values():
            roi_polygon_points = self.roi_polygon.reshape((-1, 1, 2))
            is_inside = self.is_inside_polygon(track['center'], roi_polygon_points)
            
            if is_inside:
                persons_inside += 1
        
        ec = getattr(self, 'entry_counts', {})
        stats = {
            'total_persons_in_area': self.personas_en_area,
            'persons_inside': persons_inside,
            'frame_counter': self.frame_counter,
            'active_tracks': len(self.active_tracks),
            'employee_counters': dict(self.employee_counters),
            'entry_counts': dict(ec),
            'total_entries': sum(ec.values()),
            'staff_names': dict(self.staff_names),
            'roi_points': self.roi_polygon.tolist(),
        }
        # Estado de la auto-detección de ángulo (visibilidad para el cliente)
        if hasattr(self, '_auto_angle_detector'):
            stats['camera_angle'] = {
                'override_manual': self._camera_angle_override,
                'auto_activo': bool(getattr(self, '_auto_angle_enabled', False)),
                'auto_detectado': getattr(self, '_auto_angle', None),
                'auto_muestras': self._auto_angle_detector.n_samples,
                'auto_mediana_hw': round(
                    self._auto_angle_detector.median_hw, 2),
            }
        # Agregar stats de analitica de personas
        if hasattr(self, '_people_counter'):
            stats['people_counter'] = self._people_counter.get_stats()
        if hasattr(self, '_demographics'):
            stats['demographics'] = self._demographics.get_counts()
        return stats

    # ── Orquestacion de la analitica de personas ─────────────────

    def _run_analytics(self, frame: np.ndarray):
        """Actualiza el conteo de personas y el log de calidad demografica."""
        try:
            # 1. Actualizar contador de personas activas
            if hasattr(self, '_people_counter'):
                self._people_counter.update_active(set(self.active_tracks.keys()))

            # 2. Log periodico del reporte de calidad facial (Hito 3): deja
            #    constancia en el log de que NO entran muestras borrosas.
            if (hasattr(self, '_demographics')
                    and self.frame_counter % 300 == 0):
                self._demographics.log_quality_report()

        except Exception as e:
            logger.error(f"Error en analytics: {e}")

    def get_analytics_stats(self) -> Dict[str, Any]:
        """Retorna estadisticas de los modulos de analitica de personas."""
        stats = {}
        if hasattr(self, '_people_counter'):
            stats['people_counter'] = self._people_counter.get_stats()
        if hasattr(self, '_demographics'):
            stats['demographics'] = self._demographics.get_counts()
        return stats

    def _update_track_class_stability(self, track_id: int):
        """Compute majority vote over recent class observations and update track class if stable."""
        history = self.track_class_history.get(track_id)
        if not history:
            return

        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        if len(history) >= 1:
            proportion = count / len(history)
            # require minimum number of observations or proportion
            min_obs = min(3, self.class_history_len)
            if len(history) >= min_obs and proportion >= self.class_stability_threshold:
                # update track's class to stable one
                if track_id in self.active_tracks:
                    self.active_tracks[track_id]['class'] = most_common
                    self.active_tracks[track_id]['stable_class'] = most_common


def create_staff_processor(**kwargs) -> TiendasProcess:
    return TiendasProcess(**kwargs)