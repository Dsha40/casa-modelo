"""Preparar en la GPU la imagen que entra al detector RF-DETR (2-oct-2026).

## Por qué

En el PC de la tienda (i5-6500 de 4 hilos, Quadro RTX 4000) la CPU va al
100 %: la comparten el servidor, el cliente, Media Player e iVMS. La GPU, al
25 %. Medido con py-spy SIN pausar al servidor (2-oct, vídeo de la cámara 3),
el 31 % del tiempo del hilo que procesa las imágenes se iba en PREPARAR cada
imagen para RF-DETR, y en la CPU:

  - dar la vuelta a los colores, BGR -> RGB (una copia)     6,6 %
  - pasarla a decimales (`to_tensor` de torchvision)       13,3 %
  - revisarla entera dos veces (rango 0-1, en rfdetr)       9,3 %
  - copiarla a memoria fijada antes de subirla              2,2 %

RF-DETR hace todo eso porque le llega un array de numpy. Si le llega un
tensor YA en la GPU, en [0, 1] y con forma (3, H, W), se salta la conversión
y la copia, y las dos revisiones corren en la GPU (docstring de
`RFDETR.predict`: «passing a tensor already on the model's accelerator skips
this image transfer entirely»).

## Qué hace

Sube la imagen TAL CUAL llega (uint8: 4 veces menos bytes que en decimales)
y, ya en la GPU: BGR -> RGB, (H, W, 3) -> (3, H, W) y uint8 -> float32 / 255.

## Idéntica, no «casi igual»

Se divide por un 255 que YA está en la GPU, no por el número 255 de Python.
Comprobado el 2-oct-2026 (torch 2.8.0+cu128, Quadro RTX 4000) con los 256
valores posibles de un píxel: con `div(255)` en CUDA, 126 de los 256 salen
distintos de los de `to_tensor` (en el último bit, 6e-8); con el 255 en la
GPU, ninguno. Así el detector ve EXACTAMENTE la misma imagen que antes.
Lo fijan las pruebas de `tests/test_imagen_a_gpu.py`.

## Cómo apagarlo

`ELDE_DETECTOR_PREPARAR_EN_GPU=0` vuelve al camino de siempre (numpy) sin
tocar código: sirve para comparar A/B. Se lee al cargar el detector. Sin
CUDA, o si algo falla al preparar, también se vuelve al camino de siempre:
un fallo aquí no puede dejar sin detecciones al pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Variable de entorno para apagarlo ('0', 'false', 'no' u 'off').
VARIABLE = 'ELDE_DETECTOR_PREPARAR_EN_GPU'

# dispositivo -> tensor 0-dim con 255.0 en ese dispositivo (ver «Idéntica»).
_divisores: Dict[str, Any] = {}

# True tras el primer fallo: se avisa UNA vez y, desde ahí, el camino de
# siempre. Un error que se repitiera en cada imagen llenaría el log.
_desactivado = False


def usar_gpu(dispositivo: Any) -> bool:
    """True si hay que preparar en `dispositivo`: es CUDA y el interruptor
    no lo apaga."""
    valor = os.getenv(VARIABLE, '1').strip().lower()
    if valor in ('0', 'false', 'no', 'off'):
        return False
    return 'cuda' in str(dispositivo).lower()


def preparar(frame_bgr: np.ndarray, dispositivo: Any) -> Optional[Any]:
    """(H, W, 3) uint8 BGR de OpenCV -> tensor float32 (3, H, W) RGB en
    [0, 1] en `dispositivo`, idéntico a `to_tensor(frame_bgr[:, :, ::-1])`.
    None si falla (o tras un fallo anterior): el que llama usa el camino de
    siempre. Nunca lanza."""
    global _desactivado
    if _desactivado:
        return None
    try:
        import torch
        clave = str(dispositivo)
        divisor = _divisores.get(clave)
        if divisor is None:
            divisor = _divisores[clave] = torch.tensor(
                255.0, dtype=torch.float32, device=dispositivo)
        t = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(dispositivo)
        t = t.permute(2, 0, 1).flip(0).contiguous()     # HWC BGR -> CHW RGB
        return t.to(torch.float32).div_(divisor)
    except Exception as exc:                                # noqa: BLE001
        _desactivado = True
        logger.warning("imagen del detector: no se pudo preparar en %s (%s); "
                       "se vuelve a prepararla en la CPU", dispositivo, exc)
        return None
