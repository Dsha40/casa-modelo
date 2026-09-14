"""Detección RF-DETR COMPARTIDA para los pipelines fuera del vigilante (2-sep-2026).

Salida de ultralytics (AGPL): tiendas, perimetrales clásicos, autolavado,
Hummus y Misters detectaban con YOLO — cada uno con su copia del modelo y
varios con el tracker INTERNO de ultralytics (.track). Este módulo les da:

  - UN solo modelo RF-DETR-M por proceso (los mismos pesos/knobs del
    vigilante: RFDETR_PESOS/RESOLUCION/DTYPE de vigilante_amazonas.config),
    con carga perezosa — un contenedor que no lo usa no paga nada.
  - `detectar_sv()`: sv.Detections listo para ByteTrack/PolygonZone, con
    class_id en el espacio ultralytics-80 que todo el código ya entiende.
  - `RastreadorPersistente`: el sustituto de `.track(persist=True)` —
    sv.ByteTrack POR INSTANCIA (cada pipeline/cámara conserva el suyo).

La traducción de ids es POR NOMBRE (COCO-91 de rfdetr -> pseudo-80): en el
espacio 91 person=1, car=3, motorcycle=4, bus=6, truck=8 y train=7 — por
número los trenes serían trucks. Tabla más ancha que la del vigilante
porque estos pipelines también piden bicycle/dog (base_perimeter).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

#: nombre COCO -> id del espacio ultralytics-80 (el contrato de siempre).
NOMBRE_A_PSEUDO: Dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5,
    "truck": 7, "dog": 16, "backpack": 24, "handbag": 26, "suitcase": 28,
}

_lock = threading.Lock()
_instancia: Optional["_DetectorCompartido"] = None
#: (ts, mensaje) del ultimo fallo de construccion: durante 60 s se re-lanza
#: el error cacheado sin reconstruir (antes cada frame de cada pipeline
#: pagaba import + intento de carga completos bajo el candado).
_fallo_construccion: Optional[tuple] = None


class _DetectorCompartido:
    """RF-DETR-M único del proceso; mismos knobs que el vigilante."""

    def __init__(self) -> None:
        from vigilante_amazonas import config as cfg
        from rfdetr import RFDETRMedium
        from rfdetr.detr import COCO_CLASSES

        if not cfg.RFDETR_PESOS.exists():
            raise RuntimeError(f"pesos RF-DETR ausentes: {cfg.RFDETR_PESOS}")
        self.dispositivo: str = cfg.resolver_dispositivo(cfg.DEVICE_DETECCION)
        self._modelo = RFDETRMedium(
            pretrain_weights=str(cfg.RFDETR_PESOS),
            resolution=cfg.RFDETR_RESOLUCION,
            device=self.dispositivo)
        try:
            self._modelo.inference(compile=True, batch_size=1,
                                   dtype=cfg.RFDETR_DTYPE)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("RF-DETR compartido sin compilar (eager): %s", exc)
        self._id91_a_pseudo: Dict[int, int] = {
            i: NOMBRE_A_PSEUDO[n]
            for i, n in COCO_CLASSES.items() if n in NOMBRE_A_PSEUDO}
        # Varios pipelines (y varios hilos de drenaje) comparten ESTA
        # instancia: el candado serializa el predict — la GPU igual lo
        # serializaria, y sin el dos trazas JIT concurrentes pueden chocar.
        self._candado = threading.Lock()
        self.detectar(np.zeros((600, 960, 3), dtype=np.uint8))   # warmup
        logger.info(
            "detector compartido cargado: %s (RF-DETR %dpx %s) en %s",
            cfg.RFDETR_PESOS.name, cfg.RFDETR_RESOLUCION, cfg.RFDETR_DTYPE,
            self.dispositivo)

    def detectar(self, frame_bgr: np.ndarray, conf_min: float = 0.3,
                 clases: Optional[Sequence[int]] = None) -> sv.Detections:
        """Un frame BGR -> sv.Detections (xyxy, confidence, class_id pseudo-80).

        `clases`: ids pseudo-80 a conservar (como el classes= de ultralytics);
        None = todas las de NOMBRE_A_PSEUDO. `conf_min` es el umbral del
        predict (piso 0.05: por debajo un DETR solo escupe basura).
        """
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        with self._candado:
            det = self._modelo.predict(rgb,
                                       threshold=max(0.05, float(conf_min)),
                                       include_source_image=False)
        if det is None or len(det) == 0:
            return sv.Detections.empty()
        filas = []
        for (x1, y1, x2, y2), conf, cid in zip(det.xyxy, det.confidence,
                                               det.class_id):
            pseudo = self._id91_a_pseudo.get(int(cid))
            if pseudo is None:
                continue
            if clases is not None and pseudo not in clases:
                continue
            filas.append((float(x1), float(y1), float(x2), float(y2),
                          float(conf), pseudo))
        if not filas:
            return sv.Detections.empty()
        arr = np.array(filas, dtype=np.float32)
        return sv.Detections(
            xyxy=arr[:, :4],
            confidence=arr[:, 4],
            class_id=arr[:, 5].astype(int),
        )


def detector_compartido() -> _DetectorCompartido:
    """Singleton perezoso del proceso (doble candado, como get_clip)."""
    global _instancia, _fallo_construccion
    if _instancia is None:
        with _lock:
            if _instancia is None:
                import time as _t
                if (_fallo_construccion is not None
                        and _t.time() - _fallo_construccion[0] < 60.0):
                    raise RuntimeError(_fallo_construccion[1])
                try:
                    _instancia = _DetectorCompartido()
                    _fallo_construccion = None
                except Exception as exc:
                    _fallo_construccion = (_t.time(), f"detector compartido no disponible: {exc}")
                    raise
    return _instancia


def detectar_sv(frame_bgr: np.ndarray, conf_min: float = 0.3,
                clases: Optional[Sequence[int]] = None) -> sv.Detections:
    return detector_compartido().detectar(frame_bgr, conf_min, clases)


class RastreadorPersistente:
    """El sustituto de `.track(persist=True)` de ultralytics.

    sv.ByteTrack POR INSTANCIA: cada pipeline (o cámara) crea el suyo y el
    estado persiste entre llamadas — la semántica de persist=True. Devuelve
    dicts planos porque eso es lo que los pipelines consumían del Results.
    """

    def __init__(self, frame_rate: int = 25,
                 activacion: float = 0.25,
                 buffer_seg: float = 2.0) -> None:
        self._bt = sv.ByteTrack(
            track_activation_threshold=activacion,
            lost_track_buffer=max(1, int(buffer_seg * frame_rate)),
            frame_rate=frame_rate)
        # GOTCHA de supervision (el mismo que fija el rastreador del
        # vigilante): det_thresh interno = activacion + 0.1 para NACER
        # tracks; sin esto una detección válida no inicia track.
        try:
            self._bt.det_thresh = float(activacion)
        except Exception:                            # noqa: BLE001
            pass

    def seguir(self, frame_bgr: np.ndarray, conf_min: float = 0.3,
               clases: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
        det = detectar_sv(frame_bgr, conf_min, clases)
        det = self._bt.update_with_detections(det)
        salida: List[Dict[str, Any]] = []
        for i in range(len(det)):
            tid = det.tracker_id[i] if det.tracker_id is not None else None
            if tid is None:
                continue
            x1, y1, x2, y2 = det.xyxy[i]
            salida.append({
                "track_id": int(tid),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(det.confidence[i]),
                "cls_coco": int(det.class_id[i]),
            })
        return salida
