"""src/analityc/core/pose_gpu.py — preparar en la GPU los recortes de ViTPose.

## Por qué (6-oct-2026)

Con gente junto al estante, la pose era lo que más bajaba los fps: en la
pasada del 6-oct el agarre iba a ~7 fps con 2-3 personas junto al estante,
y a 11-13 fps con 3 personas lejos de él. Cada persona costaba 25,5 ms en
el PC de la tienda (Quadro RTX 4000, i5-6500): **8,6 ms preparando el
recorte en la CPU** (el preparador de Hugging Face estira la imagen con
scipy), 13,4 ms en el modelo y 3,3 ms de posproceso. Y cada persona iba en
su propia llamada al modelo.

Aquí se hace esa preparación en la GPU, **con la misma fórmula** que
`VitPoseImageProcessor` (transformers 4.x, `image_processing_vitpose.py`):

1. `box_to_center_and_scale`: el recuadro se lleva a la proporción 192/256
   y se agranda ×1,25;
2. `get_warp_matrix` + `scipy_warp_affine`: muestreo bilineal (`order=1`)
   en coordenadas de píxel, con 0 fuera de la imagen;
3. de vuelta a `uint8` (scipy devuelve el tipo de entrada), /255 y
   normalización con la media y desviación del preparador.

El recorte se sigue tomando como antes (`PoseMunecas._recorte`): lo que el
recuadro agrandado deja fuera de la caja de la persona sale NEGRO, no con
los píxeles vecinos. Así el modelo ve exactamente lo mismo que ayer.

El posproceso (`post_process_pose_estimation`, con su refinado DARK en
scipy) NO se toca: sigue siendo el de Hugging Face, ahora en una sola
llamada para todas las personas.

Interruptor: `ELDE_POSE_PREPARAR_EN_GPU=0` vuelve al camino de antes.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

#: Tamaño de entrada de ViTPose (alto, ancho): `size` del preparador.
ALTO, ANCHO = 256, 192
#: `normalize_factor` × `padding_factor` del preparador: solo importa el 1,25.
_PADDING = 1.25


def activado() -> bool:
    """False con `ELDE_POSE_PREPARAR_EN_GPU=0` (vuelta al camino de la CPU)."""
    return os.environ.get('ELDE_POSE_PREPARAR_EN_GPU', '1').strip() != '0'


def _tam_destino(w: float, h: float) -> Tuple[float, float]:
    """Ancho y alto del recuadro agrandado, en píxeles del recorte
    (`box_to_center_and_scale` con la caja = el recorte entero)."""
    proporcion = ANCHO / ALTO
    if w > proporcion * h:
        h = w / proporcion
    elif w < proporcion * h:
        w = h * proporcion
    return w * _PADDING, h * _PADDING


def _coordenadas(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dónde lee cada píxel de salida, en píxeles del recorte (float64).

    Se calcula IGUAL que scipy en el preparador: la matriz float32 de
    `get_warp_matrix`, invertida en float64. Con una fórmula equivalente
    pero escrita a mano, la última cifra cambiaba y una fila entera del
    borde caía dentro en un lado y fuera en el otro (diferencias de hasta
    250 niveles en esa fila).
    """
    tw, th = _tam_destino(float(w), float(h))
    centro = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    escala = np.array([tw / _PADDING / 200.0, th / _PADDING / 200.0],
                      dtype=np.float32) * _PADDING
    M = np.zeros((2, 3), dtype=np.float32)
    tam = escala * 200.0
    sx = (ANCHO - 1.0) / tam[0]
    sy = (ALTO - 1.0) / tam[1]
    entrada = centro * 2.0
    M[0, 0], M[0, 2] = sx, sx * (-0.5 * entrada[0] + 0.5 * tam[0])
    M[1, 1], M[1, 2] = sy, sy * (-0.5 * entrada[1] + 0.5 * tam[1])
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    j = np.arange(ANCHO, dtype=np.float64)
    i = np.arange(ALTO, dtype=np.float64)
    return inv[0, 0] * j + inv[0, 2], inv[1, 1] * i + inv[1, 2]


def rejilla(w: int, h: int, device) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """Rejilla (1, ALTO, ANCHO, 2) para `grid_sample` con
    `align_corners=True` (cuenta en centros de píxel, como scipy) y la
    máscara (ALTO, ANCHO) de los puntos que caen dentro del recorte."""
    import torch
    x_in, y_in = _coordenadas(w, h)
    # scipy (`mode='constant'`) da 0 a TODO punto fuera de [0, n-1];
    # `grid_sample` mezcla el borde con 0 hasta un píxel fuera. Sin la
    # máscara, la franja de borde salía con 35-60 niveles de más.
    mx = (x_in >= 0) & (x_in <= w - 1)
    my = (y_in >= 0) & (y_in <= h - 1)
    # Píxel -> [-1, 1]. Con w o h = 1 no hay recorte posible (lo filtra
    # `_recorte` con su mínimo), pero no dividir entre 0.
    gx = 2.0 * x_in / max(1, w - 1) - 1.0
    gy = 2.0 * y_in / max(1, h - 1) - 1.0
    gx_t = torch.tensor(gx, device=device, dtype=torch.float32)
    gy_t = torch.tensor(gy, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(gy_t, gx_t, indexing='ij')
    mascara = torch.tensor(np.outer(my, mx), device=device)
    return torch.stack((xx, yy), dim=-1).unsqueeze(0), mascara


def preparar(frame_bgr: np.ndarray,
             cajas: Sequence[Tuple[int, int, int, int]],
             media: Sequence[float], desv: Sequence[float],
             device, fp16: bool) -> 'torch.Tensor':
    """`pixel_values` (N, 3, ALTO, ANCHO) de los recortes `cajas`
    (x1, y1, x2, y2 enteros y ya dentro del frame) de `frame_bgr`.

    El frame sube UNA vez a la GPU; cada recorte es una vista de él.
    """
    import torch
    import torch.nn.functional as F
    frame = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(
        device, non_blocking=True)
    m = torch.tensor(media, device=device, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(desv, device=device, dtype=torch.float32).view(3, 1, 1)
    salida: List['torch.Tensor'] = []
    for x1, y1, x2, y2 in cajas:
        # BGR -> RGB (el preparador espera RGB) y HWC -> 1CHW.
        rec = frame[y1:y2, x1:x2, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
        g, dentro = rejilla(x2 - x1, y2 - y1, device)
        est = F.grid_sample(rec.float(), g, mode='bilinear',
                            padding_mode='zeros', align_corners=True)[0]
        est = est * dentro
        # scipy escribe en uint8 (el tipo de la imagen): redondea y recorta.
        est = est.round().clamp(0, 255)
        salida.append((est / 255.0 - m) / s)
    pv = torch.stack(salida)
    return pv.half() if fp16 else pv
