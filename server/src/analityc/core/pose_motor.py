"""src/analityc/core/pose_motor.py — el motor de pose (ViTPose, único).

## Por qué existe

La pose era el último consumidor de ultralytics (AGPL) en el contenedor de
tiendas: la degustación del stand (`stand_puente._keypoints`) y las muñecas
de Anaqueles (`anaqueles_pose.PoseMunecas`) cargaban cada uno su copia de
`models/pose/yolo11m-pose.pt`. Este módulo les da UNA fábrica con el mismo
contrato que ya consumían — 17 keypoints COCO `[x, y, conf]` en coordenadas
de la imagen dada — sobre ViTPose-base (transformers, Apache-2.0) con pesos
LOCALES: los contenedores no tienen red, así que `local_files_only=True`
SIEMPRE (un timeout de HuggingFace dentro del bucle de inferencia sería
letal).

## Sin rollback (3-sep-2026)

El 2-sep nació con `POSE_MOTOR=yolo` como red de seguridad; el 3-sep se
retiró junto con el resto de ultralytics de la imagen. ViTPose es el ÚNICO
motor y la fábrica ya no elige nada. Si faltan los pesos locales (descarga a
medias, volumen sin montar) NO se cae a nada: sale UN error en el log, el
motor se crea igual y `keypoints_17` devuelve None — los consumidores ya
toleran quedarse sin pose (el stand devuelve None, anaqueles lista vacía) y
el contenedor arranca aunque la migración esté a medias, el mismo criterio
a diferencia de `crear_detector` del vigilante, que lanza. `POSE_MOTOR` en el entorno solo produce
un aviso para que nadie crea que sigue eligiendo motor.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Carpeta de pesos, resuelta relativa al repo (la misma técnica que
#: stand_puente/anaqueles_pose: parents[3] desde src/analityc/core es server/).
_DIR_POSE = Path(__file__).resolve().parents[3] / 'models' / 'pose'
RUTA_VITPOSE = _DIR_POSE / 'vitpose-base-simple'

#: Los avisos de este módulo salen UNA vez por proceso: la fábrica se llama
#: desde el stand Y desde anaqueles, y repetirlos solo ensucia el log.
_avisos_emitidos: set = set()


def _avisar_una_vez(clave: str, nivel: int, mensaje: str, *args: Any) -> None:
    if clave in _avisos_emitidos:
        return
    _avisos_emitidos.add(clave)
    logger.log(nivel, mensaje, *args)


def _pesos_en_disco() -> bool:
    """Los dos ficheros sin los que `from_pretrained` no arranca."""
    return ((RUTA_VITPOSE / 'model.safetensors').is_file()
            and (RUTA_VITPOSE / 'config.json').is_file())


def _avisar_pesos_ausentes() -> None:
    # ERROR y no warning: sin pose no hay degustación en el stand ni agarres
    # en anaqueles, y ya no existe un rollback que lo disimule.
    _avisar_una_vez(
        'pesos', logging.ERROR,
        "pose: faltan los pesos de ViTPose en %s (model.safetensors + "
        "config.json): SIN pose — degustacion del stand y agarres de "
        "anaqueles apagados (el rollback yolo11m-pose se retiro el "
        "3-sep-2026)", RUTA_VITPOSE)


class _MotorViTPose:
    """ViTPose-base sobre la imagen ENTERA: quien llama ya recortó a la
    persona, así que la caja del top-down es `[0, 0, w, h]`.

    Carga perezosa con candado (la fábrica se llama dentro del bucle de
    inferencia y construir no puede costar segundos — el mismo patrón que
    `PoseMunecas`). fp16 solo en cuda: en CPU half es más lento.
    """

    def __init__(self, device: Any = None) -> None:
        self._device_pedido = device
        self._dispositivo: Any = 'cpu'
        self._fp16 = False
        self._modelo = None
        self._procesador = None
        self._intentado = False
        self._lock = threading.Lock()

    @property
    def disponible(self) -> bool:
        """True si el modelo cargó (solo significativo tras el primer uso)."""
        return self._modelo is not None

    def _cargar_si_falta(self) -> None:
        if self._intentado:
            return
        with self._lock:
            if self._intentado:
                return
            # Sin pesos no se importa ni transformers: el error ya salió (o
            # sale ahora) una sola vez y el motor queda «sin modelo» para
            # siempre en vez de reintentar la carga en cada frame.
            if not _pesos_en_disco():
                _avisar_pesos_ausentes()
                self._intentado = True
                return
            try:
                import torch
                from transformers import (VitPoseForPoseEstimation,
                                          VitPoseImageProcessor)
                self._procesador = VitPoseImageProcessor.from_pretrained(
                    str(RUTA_VITPOSE), local_files_only=True)
                modelo = VitPoseForPoseEstimation.from_pretrained(
                    str(RUTA_VITPOSE), local_files_only=True)
                disp = self._device_pedido
                if disp is None:
                    disp = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._dispositivo = disp
                self._fp16 = 'cuda' in str(disp)
                if self._fp16:
                    modelo = modelo.half()
                self._modelo = modelo.to(disp).eval()
                logger.info("pose: ViTPose cargado (%s, %s)", disp,
                            'fp16' if self._fp16 else 'fp32')
            except Exception as exc:                        # noqa: BLE001
                logger.error("pose: no se pudo cargar ViTPose: %s", exc)
            finally:
                # Con o sin éxito, no se reintenta: un peso corrupto que
                # relanzara la carga en cada frame congelaría el bucle.
                self._intentado = True

    def keypoints_17(self,
                     imagen_bgr: np.ndarray) -> Optional[List[List[float]]]:
        """17 keypoints COCO `[x, y, conf]` en coordenadas de `imagen_bgr`,
        o None. Nunca lanza."""
        self._cargar_si_falta()
        if self._modelo is None or self._procesador is None:
            return None
        try:
            import torch
            alto, ancho = imagen_bgr.shape[:2]
            # OpenCV entrega BGR; el procesador espera RGB. La caja va en
            # formato COCO (x, y, w, h) y cubre la imagen entera.
            rgb = np.ascontiguousarray(imagen_bgr[:, :, ::-1])
            cajas = [[[0.0, 0.0, float(ancho), float(alto)]]]
            entradas = self._procesador(images=rgb, boxes=cajas,
                                        return_tensors='pt')
            pv = entradas['pixel_values'].to(self._dispositivo)
            if self._fp16:
                pv = pv.half()
            with torch.inference_mode():
                salida = self._modelo(pixel_values=pv)
            if self._fp16:
                # El post-proceso pasa los heatmaps por scipy en numpy;
                # en float16 el filtro gaussiano no está garantizado.
                salida.heatmaps = salida.heatmaps.float()
            res = self._procesador.post_process_pose_estimation(
                salida, boxes=cajas)
            pose = res[0][0]
            kps = pose['keypoints'].detach().float().cpu().numpy()
            conf = pose['scores'].detach().float().cpu().numpy().reshape(-1)
            if kps.shape[0] != 17 or conf.shape[0] != 17:
                return None
            return [[float(x), float(y), float(c)]
                    for (x, y), c in zip(kps, conf)]
        except Exception as exc:                            # noqa: BLE001
            logger.debug("pose: ViTPose falló: %s", exc)
            return None


    def keypoints_lote(self, frame_bgr: np.ndarray,
                       cajas: List[Tuple[int, int, int, int]]
                       ) -> Optional[List[Optional[List[List[float]]]]]:
        """Los 17 keypoints de VARIAS personas del mismo frame en UNA
        llamada al modelo, con los recortes preparados en la GPU
        (`pose_gpu.py`, 6-oct-2026). `cajas` = (x1, y1, x2, y2) enteros ya
        dentro del frame; cada resultado va en coordenadas de SU recorte,
        igual que `keypoints_17(recorte)`.

        None si no se puede (sin CUDA, interruptor apagado o fallo): quien
        llama vuelve entonces a `keypoints_17` persona a persona.
        """
        from . import pose_gpu
        self._cargar_si_falta()
        if (self._modelo is None or self._procesador is None or not cajas
                or 'cuda' not in str(self._dispositivo)
                or not pose_gpu.activado()):
            return None
        try:
            import torch
            pv = pose_gpu.preparar(frame_bgr, cajas,
                                   self._procesador.image_mean,
                                   self._procesador.image_std,
                                   self._dispositivo, self._fp16)
            with torch.inference_mode():
                salida = self._modelo(pixel_values=pv)
            if self._fp16:
                salida.heatmaps = salida.heatmaps.float()
            # Una «imagen» por recorte, con su caja = el recorte entero: así
            # el posproceso devuelve coordenadas del recorte, como antes.
            coco = [[[0.0, 0.0, float(x2 - x1), float(y2 - y1)]]
                    for x1, y1, x2, y2 in cajas]
            res = self._procesador.post_process_pose_estimation(
                salida, boxes=coco)
            todos: List[Optional[List[List[float]]]] = []
            for r in res:
                kps = r[0]['keypoints'].detach().float().cpu().numpy()
                conf = r[0]['scores'].detach().float().cpu().numpy() \
                    .reshape(-1)
                if kps.shape[0] != 17 or conf.shape[0] != 17:
                    todos.append(None)
                    continue
                todos.append([[float(x), float(y), float(c)]
                              for (x, y), c in zip(kps, conf)])
            return todos
        except Exception as exc:                            # noqa: BLE001
            _avisar_una_vez(
                'lote', logging.WARNING,
                "pose: la preparación en la GPU falló (%s); se vuelve a la "
                "CPU persona a persona", exc)
            return None


def crear_motor_pose(device: Any = None) -> _MotorViTPose:
    """Fábrica del motor de pose: SIEMPRE ViTPose (único motor desde el
    3-sep-2026).

    Devuelve un objeto con `keypoints_17(imagen_bgr)`. Aquí no se carga nada
    (el motor es perezoso): solo se mira si los pesos están en disco para
    dejar el error en el log al arrancar y no en mitad del primer frame.
    """
    motor_env = os.environ.get('POSE_MOTOR')
    if motor_env is not None and motor_env.strip().lower() != 'vitpose':
        # Un compose viejo con ELDE_POSE_MOTOR=yolo no debe dejar a nadie
        # creyendo que corre yolo: aquí ya no se elige motor.
        if motor_env.strip().lower() == 'yolo':
            _avisar_una_vez(
                'env', logging.WARNING,
                "POSE_MOTOR=yolo ignorado: el rollback yolo11m-pose se "
                "retiro el 3-sep-2026 (ViTPose es el unico motor)")
        else:
            _avisar_una_vez(
                'env', logging.WARNING,
                "POSE_MOTOR=%r ignorado: ViTPose es el unico motor desde el "
                "3-sep-2026", motor_env)
    if not _pesos_en_disco():
        _avisar_pesos_ausentes()
    return _MotorViTPose(device)
