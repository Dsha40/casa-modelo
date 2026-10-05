"""Detección RF-DETR COMPARTIDA para los pipelines fuera del vigilante (2-sep-2026).

Salida de ultralytics (AGPL): tiendas, perimetrales clásicos, autolavado,
Hummus y Misters detectaban con YOLO — cada uno con su copia del modelo y
varios con el tracker INTERNO de ultralytics (.track). Este módulo les da:

  - UN solo modelo RF-DETR-M por proceso (los mismos pesos/knobs del
    vigilante cuando está; sin él, valores propios — ver `ajustes_rfdetr`),
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
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import supervision as sv

from . import imagen_a_gpu

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

#: Carpeta de los pesos sin vigilante: la que documenta
#: deploy/requirements_servidor_ubuntu.txt. MODELS_DIR la mueve igual que al
#: resto de modelos base (ver analityc/config/config.py).
_CARPETA_PESOS = Path(__file__).resolve().parents[3] / 'models' / 'base'
_NOMBRE_PESOS = 'rf-detr-medium.pth'


def ajustes_rfdetr() -> Dict[str, Any]:
    """Pesos, resolución, dtype y dispositivo del RF-DETR compartido.

    Con el vigilante a mano (monorepo) salen de `vigilante_amazonas.config`,
    como siempre: los dos detectores del proceso van con los mismos knobs.

    En el corte de TIENDAS ese paquete no viaja, y hasta el 16-sep-2026 este
    import tumbaba el detector: `process_frame` fallaba en CADA frame con «No
    module named 'vigilante_amazonas'», devolvia metadata sin `detections`,
    app.py lo respondia como `success` y el cliente en modo directo no tenia
    nada que repintar. Sin vigilante, resolución y dtype quedan en None = los
    de la propia librería para esos pesos: los valores afinados viven en un
    config que este repositorio no tiene, y aquí no se inventan.
    """
    try:
        from vigilante_amazonas import config as cfg
    except ModuleNotFoundError as exc:
        # Solo la AUSENCIA del paquete activa los valores propios. Si el
        # vigilante está pero le falta algo por dentro, el error debe verse.
        if exc.name != 'vigilante_amazonas':
            raise
    else:
        return {
            'pesos': Path(cfg.RFDETR_PESOS),
            'resolucion': cfg.RFDETR_RESOLUCION,
            'dtype': cfg.RFDETR_DTYPE,
            'dispositivo': cfg.resolver_dispositivo(cfg.DEVICE_DETECCION),
        }
    import torch
    carpeta = (os.getenv('MODELS_DIR') or '').strip() or str(_CARPETA_PESOS)
    return {
        'pesos': Path(carpeta) / _NOMBRE_PESOS,
        'resolucion': None,
        'dtype': None,
        'dispositivo': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


class _DetectorCompartido:
    """RF-DETR-M único del proceso; mismos knobs que el vigilante si está."""

    def __init__(self) -> None:
        ajustes = ajustes_rfdetr()
        pesos: Path = ajustes['pesos']
        if not pesos.exists():
            raise RuntimeError(f"pesos RF-DETR ausentes: {pesos}")

        from rfdetr import RFDETRMedium
        from rfdetr.detr import COCO_CLASSES

        self.dispositivo: str = ajustes['dispositivo']
        # None = no se pasa: que la librería use el suyo para estos pesos.
        opciones: Dict[str, Any] = {'pretrain_weights': str(pesos),
                                    'device': self.dispositivo}
        if ajustes['resolucion'] is not None:
            opciones['resolution'] = ajustes['resolucion']
        self._modelo = RFDETRMedium(**opciones)
        compilar: Dict[str, Any] = {'compile': True, 'batch_size': 1}
        if ajustes['dtype'] is not None:
            compilar['dtype'] = ajustes['dtype']
        try:
            self._modelo.inference(**compilar)
        except Exception as exc:                     # noqa: BLE001
            logger.warning("RF-DETR compartido sin compilar (eager): %s", exc)
        self._id91_a_pseudo: Dict[int, int] = {
            i: NOMBRE_A_PSEUDO[n]
            for i, n in COCO_CLASSES.items() if n in NOMBRE_A_PSEUDO}
        # Varios pipelines (y varios hilos de drenaje) comparten ESTA
        # instancia: el candado serializa el predict — la GPU igual lo
        # serializaria, y sin el dos trazas JIT concurrentes pueden chocar.
        self._candado = threading.Lock()
        # Imagen preparada en la GPU (2-oct-2026, `imagen_a_gpu.py`): en el PC
        # de la tienda, prepararla en la CPU era el 31 % del hilo de las
        # imagenes. None = el camino de siempre (numpy). Se decide UNA vez.
        self._preparar_en: Optional[str] = (
            self.dispositivo if imagen_a_gpu.usar_gpu(self.dispositivo)
            else None)
        self.detectar(np.zeros((600, 960, 3), dtype=np.uint8))   # warmup
        logger.info(
            "detector compartido cargado: %s (RF-DETR %s px, dtype %s) en %s; "
            "imagen preparada en la %s",
            pesos.name, ajustes['resolucion'] or 'por defecto',
            ajustes['dtype'] or 'por defecto', self.dispositivo,
            'GPU' if self._preparar_en else 'CPU')

    def detectar(self, frame_bgr: np.ndarray, conf_min: float = 0.3,
                 clases: Optional[Sequence[int]] = None) -> sv.Detections:
        """Un frame BGR -> sv.Detections (xyxy, confidence, class_id pseudo-80).

        `clases`: ids pseudo-80 a conservar (como el classes= de ultralytics);
        None = todas las de NOMBRE_A_PSEUDO. `conf_min` es el umbral del
        predict (piso 0.05: por debajo un DETR solo escupe basura).
        """
        # En la GPU si se puede (`imagen_a_gpu.py`); si no, o si falla, el RGB
        # en numpy de siempre. Los dobles de las pruebas se crean sin
        # __init__ y no tienen el atributo: camino de siempre.
        entrada = None
        destino = getattr(self, '_preparar_en', None)
        if destino is not None:
            entrada = imagen_a_gpu.preparar(frame_bgr, destino)
        if entrada is None:
            entrada = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        with self._candado:
            det = self._modelo.predict(entrada,
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
