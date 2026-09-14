"""
config.py - v2

Cambios respecto a v1
---------------------
1. VARIABLES DE ENTORNO: todas las rutas y parametros clave se pueden
   sobreescribir via env vars (compatible con Docker / CI / multi-servidor).
   Los defaults siguen apuntando a las rutas originales de produccion.

2. COPIA PROFUNDA en get_config(): evita corrupcion del dict global cuando
   algun modulo modifica el resultado de get_config() en runtime.

3. VALIDACION AL ARRANCAR: _validate_config() comprueba que los modelos
   existen en disco y lanza errores claros antes de aceptar conexiones WS.

4. DEVICE REAL: la clave "device" se expone para que app.py pueda usarla.

5. RECARGA EN CALIENTE opcional: get_config() puede leer ENV vars en cada
   llamada (util para cambiar umbrales sin reiniciar, via CONFIG_RELOAD=true).
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas base (relativas a la raiz del proyecto)
# ---------------------------------------------------------------------------
# Raiz del repo, derivada de la ubicacion de este archivo:
#   src/analityc/config/config.py -> parents[3] = raiz del proyecto.
# Sobreescribibles con las env vars MODELS_DIR / OUTPUT_DIR.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_BASE = str(_PROJECT_ROOT / "models" / "base")
_OUTPUT_BASE = str(_PROJECT_ROOT / "output")


# ---------------------------------------------------------------------------
# Perfiles de camara por angulo (deteccion multi-angulo, Hito 2)
# ---------------------------------------------------------------------------
# El filtro de aspecto del detector descarta cajas mas anchas que altas
# (asume que las personas son mas altas que anchas). Eso ROMPE el angulo
# cenital, donde una persona vista desde arriba se ve mas ancha que alta.
# Por eso el umbral es configurable por angulo:
#   aspect_ratio_min = alto/ancho minimo de la caja para aceptarla.
#   0.0 => filtro DESACTIVADO (recomendado para cenital).
CAMERA_ANGLE_PRESETS: Dict[str, Dict[str, float]] = {
    "frontal": {"aspect_ratio_min": 0.80},
    "lateral": {"aspect_ratio_min": 0.50},
    "cenital": {"aspect_ratio_min": 0.00},
}
DEFAULT_CAMERA_ANGLE = "frontal"

# ── Auto-deteccion del angulo de camara (frontal vs cenital) ──
# Se deduce de la relacion alto/ancho (h/w) de los bboxes de personas:
#   frontal/lateral -> personas mas altas que anchas (h/w alto).
#   cenital (techo)  -> personas vistas desde arriba, bbox ~cuadrado o mas
#                       ancho (h/w bajo).
# Si la mediana de h/w cae por debajo de AUTO_ANGLE_CENITAL_HW_MAX, la
# camara se clasifica como cenital (y se desactiva el filtro de aspecto).
AUTO_ANGLE_DEFAULT: bool = True
AUTO_ANGLE_MIN_SAMPLES: int = 40       # personas observadas antes de decidir
AUTO_ANGLE_CENITAL_HW_MAX: float = 1.25  # mediana h/w < esto => cenital
AUTO_ANGLE_WINDOW: int = 250           # tamano de la ventana de muestras


def aspect_ratio_min_for_angle(angle: Optional[str]) -> float:
    """Devuelve el aspect_ratio_min del preset del angulo dado.

    Si el angulo es desconocido o None, cae al preset 'frontal'.
    """
    preset = CAMERA_ANGLE_PRESETS.get(
        (angle or DEFAULT_CAMERA_ANGLE).lower().strip(),
        CAMERA_ANGLE_PRESETS[DEFAULT_CAMERA_ANGLE],
    )
    return float(preset.get("aspect_ratio_min", 0.80))


def _parse_camera_angle_map(raw: str) -> Dict[str, str]:
    """Parsea 'cam1:cenital,cam2:lateral' -> {'cam1':'cenital', ...}.

    Las claves se dejan como string; el caller compara con str(camera_id).
    """
    out: Dict[str, str] = {}
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        cam, ang = chunk.split(":", 1)
        cam = cam.strip()
        ang = ang.strip().lower()
        if cam and ang in CAMERA_ANGLE_PRESETS:
            out[cam] = ang
    return out


# ---------------------------------------------------------------------------
# Helpers para leer variables de entorno con tipos
# ---------------------------------------------------------------------------

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key, "").strip()
    if val:
        try:
            return float(val)
        except ValueError:
            logger.warning("ENV %s='%s' no es un float valido; usando default=%.3f", key, val, default)
    return default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "").strip()
    if val:
        try:
            return int(val)
        except ValueError:
            logger.warning("ENV %s='%s' no es un int valido; usando default=%d", key, val, default)
    return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------------------
# Construccion del diccionario de configuracion
# ---------------------------------------------------------------------------

def _build_config() -> Dict[str, Any]:
    """
    Construye el dict de config combinando defaults con ENV vars.
    Llamada una vez al arrancar (o en cada get_config() si CONFIG_RELOAD=true).
    """
    models_dir = _env_str("MODELS_DIR", _MODELS_BASE)
    output_dir = _env_str("OUTPUT_DIR", _OUTPUT_BASE)

    def model(filename: str) -> str:
        """Ruta absoluta a un modelo dentro de models_dir."""
        return str(Path(models_dir) / filename)

    def output(filename: str) -> str:
        """Ruta absoluta a un archivo dentro de output_dir."""
        return str(Path(output_dir) / filename)

    return {
        # -- Rutas de modelos: LEGADO, NADIE LAS CARGA ------------------------
        # Desde la retirada de YOLO (3-sep-2026) la deteccion la hace el
        # RF-DETR COMPARTIDO del proceso (core/deteccion_compartida.py, pesos
        # en RFDETR_PESOS), y los pesos custom de eventos quedaron retirados
        # (core/Hummus.py). NINGUN `.pt` de los de abajo se abre: si el archivo
        # no existe en disco no pasa nada.
        # Se conservan estos valores porque app.py sigue pasandolos a los
        # constructores (app.py 307, 350, 363, 393, 414, 434, 454, 475) y
        # quitarlos romperia esas firmas; los consumidores ya los declaran
        # inertes (tiendas.py 120-125, perimetrales_multicam.py 110-114,
        # base_perimeter.py 37-38, car_washed.py 31).
        # Que digan `yolo26m.pt` es memoria del modelo viejo, NO significa que
        # el servidor cargue YOLO hoy: no lo hace, y ultralytics ni siquiera se
        # puede importar.
        "model_path": _env_str("MODEL_PATH_DEFAULT", model("yolo26m.pt")),
        "model_paths": {
            "Lavado":               _env_str("MODEL_PATH_LAVADO",                model("yolo26m.pt")),
            "Hummus":               _env_str("MODEL_PATH_HUMMUS",                model("1080.pt")),
            "Perimetrales":         _env_str("MODEL_PATH_PERIMETRALES",          model("yolo26m.pt")),
            "PerimetralesMultiCam": _env_str("MODEL_PATH_PERIMETRALES_MULTICAM", model("yolo26m.pt")),
            "PerimetralesBoTSORT":  _env_str("MODEL_PATH_PERIMETRALES_BOTSORT",  model("yolo26m.pt")),
            # "Personal de Amazonas": cosmeticos ELIMINADO. Solo personas +
            # genero + edad; no usa modelo de productos (sin entrada aqui).
        },
        "person_model_paths": {
            "Hummus": _env_str("PERSON_MODEL_PATH_HUMMUS", model("yolo26m.pt")),
        },

        # -- Directorios de salida -------------------------------------------
        "output_dir":                     output_dir,
        "hummus_order_screenshot_dir":    _env_str("HUMMUS_ORDER_DIR",    output("Toma_de_orden_hummus")),
        "hummus_delivery_screenshot_dir": _env_str("HUMMUS_DELIVERY_DIR", output("Entrega_de_plato_hummus")),
        "hummus_alert_csv":               _env_str("HUMMUS_ALERT_CSV",    output("hummus_alertas.csv")),

        # -- Parametros de inferencia ----------------------------------------
        # Umbral global por defecto (lo usan Perimetrales/Hummus directamente).
        # TiendasProcess aplica su propio piso de seguridad max(0.30, conf)
        # para deteccion de personas, asi que este valor no lo afecta.
        "confidence_threshold": _env_float("CONFIDENCE_THRESHOLD", 0.12),
        "iou_threshold":        _env_float("IOU_THRESHOLD",        0.30),

        # -- Hardware --------------------------------------------------------
        # "auto"   -> app.py usa _get_first_gpu() / _get_device_default()
        # "cpu"    -> fuerza CPU aunque haya GPU
        # "cuda:0" -> fuerza esa GPU especifica
        "device": _env_str("INFERENCE_DEVICE", "auto"),

        # -- Red -------------------------------------------------------------
        "host":           _env_str("SERVER_HOST", "0.0.0.0"),
        "websocket_port": _env_int("SERVER_PORT", 9000),

        # -- Comportamiento --------------------------------------------------
        # CONFIG_RELOAD=true -> get_config() reconstruye en cada llamada
        # (util para ajustar umbrales sin reiniciar el servidor)
        "config_reload": _env_bool("CONFIG_RELOAD", False),

        # -- Perfiles de camara por angulo (deteccion multi-angulo) ----------
        # Presets de umbrales por angulo y mapa camera_id -> angulo. El mapa
        # se puede setear por env: CAMERA_ANGLE_BY_ID="1:cenital,2:lateral".
        "camera_angle_presets": copy.deepcopy(CAMERA_ANGLE_PRESETS),
        "default_camera_angle": _env_str("DEFAULT_CAMERA_ANGLE",
                                         DEFAULT_CAMERA_ANGLE),
        "camera_angle_by_id": _parse_camera_angle_map(
            _env_str("CAMERA_ANGLE_BY_ID", "")),
    }


# ---------------------------------------------------------------------------
# Validacion de modelos al arrancar
# ---------------------------------------------------------------------------

def _pesos_vivos() -> List[Any]:
    """Los pesos que el servidor SI abre hoy: el detector RF-DETR.

    Se lee de `vigilante_amazonas.config` de forma perezosa y tolerante: este
    modulo es de configuracion y no puede tumbar el arranque porque otro
    paquete no este a mano (los tests lo importan suelto).
    """
    try:
        from vigilante_amazonas import config as _vig
        return [getattr(_vig, 'RFDETR_PESOS', '')]
    except Exception:                                   # noqa: BLE001
        return []


def _validate_config(cfg: Dict[str, Any], *, strict: bool = False) -> List[str]:
    """
    Verifica que los pesos que el servidor CARGA DE VERDAD existan en disco.

    Parametros
    ----------
    strict : bool
        False (default) -> solo warnings, el servidor arranca igual.
        True            -> lanza FileNotFoundError si falta alguno.

    Retorna
    -------
    Lista de paths que no se encontraron.
    """
    # Las rutas LEGADO no se comprueban (8-sep-2026). `model_path`,
    # `model_paths` y `person_model_paths` siguen en el config porque app.py
    # los pasa a las firmas de los pipelines, pero NINGUNO abre ya un `.pt`:
    # detectan todos por el RF-DETR compartido. Mirarlos en disco hacia que
    # el servidor escupiera en CADA arranque siete avisos de «Modelo no
    # encontrado» rematados por «el servidor iniciara pero fallara al
    # intentar usarlos» — una frase falsa que manda al operador a buscar
    # pesos de YOLO que ya no existen ni hacen falta. Se comprueba lo que
    # SI se carga; si algun dia vuelve a haber un modelo propio por
    # pipeline, se anade aqui y el aviso vuelve a ser cierto.
    paths_to_check: List[str] = [
        str(ruta) for ruta in _pesos_vivos() if str(ruta or '').strip()]

    missing: List[str] = []
    for path in paths_to_check:
        if not Path(path).exists():
            logger.warning("Modelo no encontrado en disco: %s", path)
            missing.append(path)

    if strict and missing:
        raise FileNotFoundError(
            "Los siguientes modelos no existen (strict=True):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    return missing


# ---------------------------------------------------------------------------
# Cache del config
# ---------------------------------------------------------------------------

_cached_config: Optional[Dict[str, Any]] = None


def get_config(*, validate: bool = True, strict: bool = False) -> Dict[str, Any]:
    """
    Retorna una copia profunda de la configuracion activa.

    Parametros
    ----------
    validate : bool
        Verifica existencia de modelos al primer uso (y cuando CONFIG_RELOAD=true).
    strict : bool
        Si True, lanza FileNotFoundError cuando faltan modelos.
    """
    global _cached_config

    # Reconstruir si no hay cache o si CONFIG_RELOAD esta activo
    if _cached_config is None or _env_bool("CONFIG_RELOAD", False):
        _cached_config = _build_config()
        if validate:
            missing = _validate_config(_cached_config, strict=strict)
            if missing:
                logger.warning(
                    "%d modelo(s) no encontrado(s) en disco. "
                    "El servidor iniciara pero fallara al intentar usarlos.",
                    len(missing),
                )

    # Siempre retornar copia profunda para evitar corrupcion del cache global
    return copy.deepcopy(_cached_config)


def reload_config() -> Dict[str, Any]:
    """Fuerza reconstruccion del config desde ENV vars. Util en tests o hot-reload."""
    global _cached_config
    _cached_config = None
    return get_config()


# ---------------------------------------------------------------------------
# DEFAULT_ROI
#
# Mantenido aqui por retrocompatibilidad con los modulos que lo importan.
# Corresponde al dominio visual (Hummus), no es config de servidor.
# ---------------------------------------------------------------------------

DEFAULT_ROI = [
    [500, 250],   # Esquina superior izquierda
    [900, 250],   # Esquina superior derecha
    [1040, 560],  # Esquina inferior derecha
    [600, 560],   # Esquina inferior izquierda
]