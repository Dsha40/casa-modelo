"""src/analityc/core/tiendas_modelos.py — cargar los modelos, una sola vez.

Extraído de `tiendas.py` (HITO 5). Aquí vive todo lo que tiene que ver con
PONER LOS MODELOS EN MEMORIA, que es un problema distinto del de analizar un
frame y estaba mezclado con él:

  - qué execution providers de onnxruntime usar según el hardware, con
    TensorRT detrás de un flag (`ENABLE_TENSORRT`, apagado por defecto);
  - `_GpuYuNet`, que corre el detector de rostros YuNet sobre onnxruntime-GPU
    imitando la interfaz de `cv2.FaceDetectorYN` — así el resto del pipeline
    no sabe si está en GPU o en CPU;
  - dónde está cada `.onnx` / `.caffemodel`;
  - `_init_gender_age_classifier`, que decide qué se carga y con qué
    prioridad, y deja los que falten en `None` sin tumbar nada: el pipeline
    tiene que arrancar aunque el operador no haya bajado un modelo opcional.

## El pipeline vivo, y por qué es un global

`sesiones_compartidas_globales()` deja que la galería de personal REUSE la
sesión ArcFace del pipeline en vez de cargar otra copia en la GPU. El registro
es «el último gana», y es inocuo: los clones por cámara comparten las mismas
sesiones.

`personal_galeria` y `estimador_edad_genero` importan varios de estos nombres
desde `.tiendas`; ese módulo los re-exporta, así que sus imports siguen
funcionando exactamente igual.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

from .analytics.config import AnalyticsConfig

logger = logging.getLogger(__name__)

# ── Raiz del proyecto (relativa a este archivo) ──────────────────────
# src/analityc/core/tiendas.py -> ../../.. = raiz del repo.
# Evita rutas hardcodeadas a C:\Users\... (portabilidad entre equipos).
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
)


# ── Constantes para el clasificador de género / edad (Caffe DNN) ──────
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models', 'classifiers')
_GENDER_PROTO = os.path.join(_MODELS_DIR, 'gender_deploy.prototxt')
_GENDER_MODEL = os.path.join(_MODELS_DIR, 'gender_net.caffemodel')
_AGE_PROTO    = os.path.join(_MODELS_DIR, 'age_deploy.prototxt')
_AGE_MODEL    = os.path.join(_MODELS_DIR, 'age_net.caffemodel')
# ── InsightFace genderage.onnx (primario, mas preciso) ──────────────────
_INSIGHTFACE_GENDERAGE = os.path.join(_MODELS_DIR, 'genderage.onnx')
# ── MiVOLO ONNX (modelo SOTA 2024 opcional para ensemble) ─────────────
# Si esta presente activa ensemble con acuerdo de genero. Descargar de
# https://github.com/WildChlamydia/MiVOLO, convertir a ONNX y copiar a
# models/classifiers/mivolo.onnx
_MIVOLO_MODEL = os.path.join(_MODELS_DIR, 'mivolo.onnx')
# ── ArcFace w600k_r50 ONNX (Face Re-Identification) ──────────────────
# Modelo de embeddings 512-dim para evitar contar la misma persona dos
# veces. Se descarga con scripts/setup_face_embedding.py
_FACE_EMBEDDING_MODEL = os.path.join(_MODELS_DIR, 'w600k_r50.onnx')
# YuNet face detector (2023 mar): detector moderno con keypoints
# (ojos, nariz, boca) que permite alineacion facial. SOTA para
# caras pequenas/ladeadas en CCTV.
_YUNET_MODEL = os.path.join(_MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
# ── PAR (rama corporal E3B): genero desde el cuerpo SIN cara ─────────
# Modelo de atributos de persona (PA-100K/PETA) exportado a ONNX. Solo
# genero. Fallback de la rama corporal cuando MiVOLO body-only no esta.
# Descargar/exportar con scripts/setup_modelos.py
_PAR_MODEL = os.path.join(_MODELS_DIR, 'par_gender.onnx')


# ── Providers ONNX (Hito 6): GPU + flag TensorRT ─────────────────────
_ONNX_DLLS_PRELOADED = False


def _tensorrt_enabled() -> bool:
    """Flag de config (env ENABLE_TENSORRT, default OFF) para TensorRT."""
    return os.environ.get("ENABLE_TENSORRT", "").strip().lower() in (
        "1", "true", "yes", "on")


def _build_onnx_providers() -> list:
    """Lista de execution providers para onnxruntime segun el hardware.

    - Llama `ort.preload_dlls()` una sola vez para que ORT encuentre las
      DLLs de CUDA 12 / cuDNN 9 que trae torch (cinturon de seguridad).
    - Usa CUDA si esta disponible; si no, CPU.
    - Antepone TensorRT solo si ENABLE_TENSORRT esta activo (default OFF).
    """
    global _ONNX_DLLS_PRELOADED
    try:
        import onnxruntime as ort
        if not _ONNX_DLLS_PRELOADED:
            try:
                ort.preload_dlls()
            except Exception:
                pass
            _ONNX_DLLS_PRELOADED = True
        available = ort.get_available_providers()
    except Exception:
        return ['CPUExecutionProvider']
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if _tensorrt_enabled() and 'TensorrtExecutionProvider' in available:
        providers = ['TensorrtExecutionProvider'] + providers
        logger.info("TensorRT habilitado para ONNX (ENABLE_TENSORRT)")
    return providers


class _GpuYuNet:
    """YuNet (face_detection_yunet_2023mar.onnx) sobre onnxruntime-GPU con la
    MISMA interfaz que cv2.FaceDetectorYN: ``setInputSize((w,h))`` +
    ``detect(img) -> (retval, faces)`` con ``faces`` Nx15
    ``[x, y, w, h, 5*(kx, ky), score]`` en pixeles del ``img`` de entrada.

    El modelo onnx tiene input fijo 640x640: se hace letterbox del crop a 640
    (preservando aspecto, pad abajo/derecha) y se mapean las coords de vuelta.
    El decode (priors/bbox/kps) replica EXACTO a cv2 (validado: dbox<0.01px).
    Una sola sesion se comparte entre camaras (ORT run() es thread-safe).
    """

    _INPUT = 640
    _STRIDES = (8, 16, 32)
    is_gpu = True  # marcador para que demographics le pase el crop nativo

    def __init__(self, model_path, score_thr=0.55, nms_thr=0.30, top_k=5000,
                 providers=None):
        import onnxruntime as ort
        self._score_thr = float(score_thr)
        self._nms_thr = float(nms_thr)
        self._top_k = int(top_k)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            model_path, sess_options=so, providers=providers)
        self._in_name = self._sess.get_inputs()[0].name
        self._out_names = [o.name for o in self._sess.get_outputs()]
        self._grid = {s: self._INPUT // s for s in self._STRIDES}
        self.providers = self._sess.get_providers()

    def setInputSize(self, size):  # noqa: N802 (compat API cv2)
        # El modelo es 640 fijo; el tamano real se maneja por letterbox interno.
        return None

    def detect(self, img):
        if img is None or getattr(img, 'size', 0) == 0:
            return 0, None
        h, w = img.shape[:2]
        n = self._INPUT
        scale = n / float(max(h, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        canvas = np.zeros((n, n, 3), np.uint8)
        canvas[:nh, :nw] = resized
        blob = np.ascontiguousarray(
            canvas.astype(np.float32).transpose(2, 0, 1)[None])
        try:
            outs = self._sess.run(self._out_names, {self._in_name: blob})
        except Exception as e:
            logger.debug("GpuYuNet run error: %s", e)
            return 0, None
        omap = {k: v for k, v in zip(self._out_names, outs)}
        dets = self._decode(omap)
        if dets.shape[0] == 0:
            return 0, None
        dets[:, :14] /= scale  # de 640-letterbox de vuelta al img de entrada
        return 1, dets

    def _decode(self, o):
        rows = []
        st = self._score_thr
        for s in self._STRIDES:
            cls = o['cls_%d' % s][0, :, 0]
            obj = o['obj_%d' % s][0, :, 0]
            bbox = o['bbox_%d' % s][0]
            kps = o['kps_%d' % s][0]
            g = self._grid[s]
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            idx = np.where(score >= st)[0]
            if idx.size == 0:
                continue
            r = (idx // g).astype(np.float32)
            c = (idx % g).astype(np.float32)
            cx = (c + bbox[idx, 0]) * s
            cy = (r + bbox[idx, 1]) * s
            ww = np.exp(bbox[idx, 2]) * s
            hh = np.exp(bbox[idx, 3]) * s
            blk = np.stack([cx - ww / 2.0, cy - hh / 2.0, ww, hh], 1)
            kp = np.empty((idx.size, 10), np.float32)
            for k in range(5):
                kp[:, 2 * k] = (c + kps[idx, 2 * k]) * s
                kp[:, 2 * k + 1] = (r + kps[idx, 2 * k + 1]) * s
            rows.append(np.concatenate(
                [blk, kp, score[idx, None]], 1).astype(np.float32))
        if not rows:
            return np.empty((0, 15), np.float32)
        dets = np.concatenate(rows, 0)
        keep = cv2.dnn.NMSBoxes(
            dets[:, :4].tolist(), dets[:, 14].tolist(),
            st, self._nms_thr, top_k=self._top_k)
        if len(keep) == 0:
            return np.empty((0, 15), np.float32)
        keep = np.array(keep).flatten()
        d = dets[keep]
        return d[d[:, 14].argsort()[::-1]]


#: El ultimo TiendasProcess construido (ver el registro en __init__). Es la
#: via para que consumidores EXTERNOS a la familia (galeria de personal)
#: reutilicen las sesiones ONNX sin duplicarlas.
_PIPELINE_ACTIVO = None


def registrar_pipeline(proc: Any) -> None:
    """Anota el pipeline vivo. Lo llama `TiendasProcess.__init__`.

    Estaba como `global _PIPELINE_ACTIVO` dentro del constructor; ahora el
    global vive en el módulo que lo lee, que es donde se entiende.
    """
    global _PIPELINE_ACTIVO
    _PIPELINE_ACTIVO = proc


def sesiones_compartidas_globales() -> Optional[dict]:
    """Las sesiones ONNX/reidentifier del pipeline vivo, o None si ninguno
    ha arrancado. Nunca lanza."""
    p = _PIPELINE_ACTIVO
    try:
        return p.get_shared_sessions() if p is not None else None
    except Exception:                                       # noqa: BLE001
        return None


def _create_yunet(device):
    """Crea el detector YuNet: GPU (onnxruntime CUDA) si hay GPU, con fallback
    automatico a cv2.FaceDetectorYN (CPU). Devuelve un objeto con la interfaz
    setInputSize/detect, o None si no se pudo cargar."""
    if not os.path.exists(_YUNET_MODEL):
        return None
    try:
        import torch
        cuda_ok = torch.cuda.is_available() and str(device) != 'cpu'
    except Exception:
        cuda_ok = False
    if cuda_ok:
        try:
            gy = _GpuYuNet(_YUNET_MODEL, 0.55, 0.30, 5000,
                           providers=_build_onnx_providers())
            if 'CUDAExecutionProvider' in gy.providers:
                logger.info("YuNet en GPU (onnxruntime CUDA, sesion compartida)")
                return gy
            logger.warning(
                "YuNet GPU pedido pero ORT no activo CUDA; uso cv2 CPU")
        except Exception as e:
            logger.warning("No se pudo crear YuNet GPU (%s); uso cv2 CPU", e)
    try:
        y = cv2.FaceDetectorYN_create(
            _YUNET_MODEL, "", (320, 320), 0.55, 0.30, 5000)
        logger.info("YuNet en CPU (cv2.FaceDetectorYN)")
        return y
    except Exception as e:
        logger.warning("No se pudo cargar YuNet: %s", e)
        return None


class ModelosMixin:
    """La carga perezosa y defensiva de los clasificadores de género y edad."""

    def _init_gender_age_classifier(self):
        """Carga los modelos Caffe de género y edad. Si faltan, desactiva.

        Adicionalmente, intenta cargar el modelo InsightFace genderage.onnx
        como PRIMARIO (mucho mas preciso que el Caffe Levi-Hassner 2015).
        Si ONNX no esta disponible, cae a Caffe automaticamente.
        """
        self._gender_net = None
        self._age_net = None
        self._face_cascade = None
        self._face_dnn = None
        self._yunet = None
        self._insightface_session = None
        self._mivolo_session = None
        self._face_embedding_session = None
        self._par_session = None
        self._osnet_session = None   # Re-ID corporal (OSNet ONNX) — Hito 9

        # Reuso de sesiones ONNX COMPARTIDAS (multi-cámara): si las hay, se
        # asignan aquí y NO se recargan abajo (ahorra VRAM). Los detectores
        # OpenCV (YuNet/Caffe/Haar/Res10) SÍ se cargan por cámara (pequeños y
        # potencialmente no thread-safe).
        _use_shared = bool(getattr(self, '_shared_sessions', None))
        if _use_shared:
            _ss = self._shared_sessions
            self._insightface_session = _ss.get('insightface')
            self._face_embedding_session = _ss.get('embedding')
            self._mivolo_session = _ss.get('mivolo')
            self._par_session = _ss.get('par')
            self._osnet_session = _ss.get('osnet')
            # YuNet GPU compartido (thread-safe). Si es None se crea abajo.
            self._yunet = _ss.get('yunet')
            logger.info("Clasificador ONNX: sesiones COMPARTIDAS reusadas "
                        "(sin recargar; ahorra VRAM en multi-cámara)")

        try:
            if os.path.exists(_GENDER_MODEL) and os.path.exists(_GENDER_PROTO):
                self._gender_net = cv2.dnn.readNet(_GENDER_MODEL, _GENDER_PROTO)
                logger.info("Gender classifier loaded")
            else:
                logger.warning(f"Gender model not found in {_MODELS_DIR}")

            if os.path.exists(_AGE_MODEL) and os.path.exists(_AGE_PROTO):
                self._age_net = cv2.dnn.readNet(_AGE_MODEL, _AGE_PROTO)
                logger.info("Age classifier loaded")
            else:
                logger.warning(f"Age model not found in {_MODELS_DIR}")

            # YuNet face detector (primario moderno, devuelve 5 keypoints).
            # Ahora corre en GPU via onnxruntime (sesion compartida thread-safe);
            # cae a cv2.FaceDetectorYN (CPU) si no hay GPU. Antes iba en CPU por
            # el OOM de OpenCV-CUDA en 8GB (ver _restore_opencv_cuda); con
            # onnxruntime-GPU no hay workspace cuDNN por camara -> VRAM plana.
            if self._yunet is None:
                self._yunet = _create_yunet(self.device)

            # Face detector DNN (Res10 SSD) como fallback secundario
            face_proto = os.path.join(_MODELS_DIR, 'opencv_face_detector.pbtxt')
            face_model = os.path.join(_MODELS_DIR, 'opencv_face_detector_uint8.pb')
            if os.path.exists(face_proto) and os.path.exists(face_model):
                try:
                    self._face_dnn = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
                    logger.info("Face detector DNN loaded (Res10 SSD)")
                except Exception as e:
                    logger.warning(f"No se pudo cargar face DNN: {e}")
                    self._face_dnn = None

            # InsightFace genderage.onnx (primario, preciso >95%)
            if not _use_shared and os.path.exists(_INSIGHTFACE_GENDERAGE):
                try:
                    import onnxruntime as ort
                    providers = _build_onnx_providers()
                    self._insightface_session = ort.InferenceSession(
                        _INSIGHTFACE_GENDERAGE, providers=providers
                    )
                    logger.info("InsightFace genderage ONNX loaded")
                except Exception as e:
                    logger.warning(f"No se pudo cargar InsightFace ONNX: {e}")
                    self._insightface_session = None
            elif not _use_shared:
                logger.info("InsightFace genderage.onnx no presente, "
                            "usando Caffe como primario")

            # ArcFace w600k_r50 ONNX (Face Re-Identification)
            if (AnalyticsConfig.REID_ENABLED and not _use_shared and
                    os.path.exists(_FACE_EMBEDDING_MODEL)):
                try:
                    import onnxruntime as ort
                    providers = _build_onnx_providers()
                    self._face_embedding_session = ort.InferenceSession(
                        _FACE_EMBEDDING_MODEL, providers=providers
                    )
                    logger.info(
                        "ArcFace w600k_r50 loaded -> "
                        "Face Re-Identification activa"
                    )
                except Exception as e:
                    logger.warning(
                        f"No se pudo cargar ArcFace embedding: {e}"
                    )
                    self._face_embedding_session = None
            elif not AnalyticsConfig.REID_ENABLED:
                logger.info("Face Re-Identification deshabilitada en config")
            elif not _use_shared:
                logger.info(
                    "ArcFace no presente. Descargar con: "
                    "python scripts/setup_face_embedding.py"
                )

            # MiVOLO ONNX (SOTA 2024, opcional para ensemble)
            if not _use_shared and os.path.exists(_MIVOLO_MODEL):
                try:
                    import onnxruntime as ort
                    providers = _build_onnx_providers()
                    self._mivolo_session = ort.InferenceSession(
                        _MIVOLO_MODEL, providers=providers
                    )
                    logger.info("MiVOLO ONNX loaded -> ensemble activado")
                except Exception as e:
                    logger.warning(f"No se pudo cargar MiVOLO ONNX: {e}")
                    self._mivolo_session = None
            elif not _use_shared:
                logger.info(
                    "MiVOLO no presente en %s. Para activar ensemble "
                    "SOTA descargar de https://github.com/WildChlamydia/MiVOLO",
                    _MIVOLO_MODEL
                )

            # PAR ONNX (rama corporal E3B: genero desde el cuerpo sin cara)
            if not _use_shared and os.path.exists(_PAR_MODEL):
                try:
                    import onnxruntime as ort
                    providers = _build_onnx_providers()
                    self._par_session = ort.InferenceSession(
                        _PAR_MODEL, providers=providers
                    )
                    logger.info("PAR ONNX loaded -> rama corporal (genero) activa")
                except Exception as e:
                    logger.warning(f"No se pudo cargar PAR ONNX: {e}")
                    self._par_session = None
            elif not _use_shared:
                logger.info(
                    "PAR no presente en %s. La rama corporal usara MiVOLO "
                    "body-only si esta, o quedara inactiva. Generar con "
                    "scripts/setup_modelos.py", _PAR_MODEL
                )

            # OSNet ONNX (Re-ID corporal/vestimenta — 2a senal para espaldas)
            _osnet_path = os.path.join(
                _PROJECT_ROOT, AnalyticsConfig.BODY_REID_MODEL)
            if (AnalyticsConfig.BODY_REID_ENABLED and not _use_shared
                    and os.path.exists(_osnet_path)):
                try:
                    import onnxruntime as ort
                    providers = _build_onnx_providers()
                    self._osnet_session = ort.InferenceSession(
                        _osnet_path, providers=providers)
                    logger.info("OSNet ONNX loaded -> Re-ID corporal activo")
                except Exception as e:
                    logger.warning(f"No se pudo cargar OSNet ONNX: {e}")
                    self._osnet_session = None
            elif AnalyticsConfig.BODY_REID_ENABLED and not _use_shared:
                logger.info(
                    "OSNet no presente en %s. Re-ID corporal inactivo "
                    "(exportar con torchreid).", _osnet_path)

            # Haar cascade (fallback y default)
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            has_gender = self._gender_net is not None
            has_age = self._age_net is not None
            has_face_dnn = self._face_dnn is not None
            has_yunet = self._yunet is not None
            has_insight = self._insightface_session is not None
            has_mivolo = self._mivolo_session is not None
            ensemble_n = sum([has_insight, has_mivolo, has_gender])
            primary = "MiVOLO-ONNX" if has_mivolo else (
                "InsightFace-ONNX" if has_insight else (
                    "Caffe" if has_gender else "NONE"
                )
            )
            detector = "YuNet+keypoints+pose" if has_yunet else (
                "Res10-SSD" if has_face_dnn else "Haar"
            )
            has_reid = self._face_embedding_session is not None
            print(f"🧑 Clasificador primario: {primary} | "
                  f"Ensemble: {ensemble_n} modelos | "
                  f"Detector: {detector} | "
                  f"MiVOLO: {'OK' if has_mivolo else 'NO'} | "
                  f"InsightFace: {'OK' if has_insight else 'NO'} | "
                  f"Caffe género: {'OK' if has_gender else 'NO'} | "
                  f"Caffe edad: {'OK' if has_age else 'NO'} | "
                  f"Re-ID biométrico: {'OK' if has_reid else 'NO'}")
        except Exception as e:
            logger.error(f"Error loading gender/age classifier: {e}")
