"""Pasar a SEGUNDOS los ajustes que se contaban en IMÁGENES (5-oct-2026).

## Por qué

Varios ajustes se cuentan en imágenes seguidas y se fijaron pensando en
pocos fps:

  - anaqueles: `agarre_frames_min: 2` y `agarre_rearme_frames: 2` («a 2-3
    FPS efectivos, un agarre entero dura 2-6 frames», anaqueles_config.yaml);
  - ByteTrack: `BYTETRACK_LOST_BUFFER: 90` (memoria de una persona tapada,
    pensada como «~9 s a ~10 FPS») y `BYTETRACK_MIN_CONSECUTIVE_FRAMES: 3`.

El 5-oct la imagen del detector pasó a la GPU (5,22 -> 7,71 fps) y hay más
cambios en camino para subir los fps. En imágenes, a 15 fps un agarre se
contaría con 0,13 s de mano dentro (más agarres falsos) y una persona tapada
perdería su seguimiento a los 6 s (más gente contada dos veces).

## La regla: nunca menos exigente que hoy

Cada ajuste exige LO MAYOR de dos cosas: lo de siempre en imágenes (el
suelo) y lo equivalente a unos segundos con los fps MEDIDOS de esa cámara.
A los fps ya validados (~5-8) da lo mismo que antes; solo se endurece
cuando los fps suben. Sin fps medidos todavía (primeras imágenes), el suelo.

    ajuste                         suelo   segundos   7,7 fps   15 fps
    mano dentro para el agarre       2      0,25 s       2         4
    mano fuera para rearmar          2      0,25 s       2         4
    ByteTrack: memoria de tapado    90      9 s         90       135
    ByteTrack: confirmar persona     3      0,35 s       3         6

Los segundos a 0 vuelven a la regla de solo imágenes.
"""

from __future__ import annotations

import math
from typing import Any, Optional

#: Peso de cada intervalo nuevo en la media móvil (0..1).
ALFA = 0.2
#: Intervalos más largos no son el ritmo: son una pausa (vídeo parado,
#: cámara sin señal, reconexión) y no se promedian.
MAX_DT_S = 2.0


class Ritmo:
    """Media móvil del tiempo entre imágenes de UNA cámara."""

    def __init__(self) -> None:
        self.dt_medio: Optional[float] = None
        self._ultimo: Optional[float] = None

    def anotar(self, t: float) -> None:
        """Una imagen en el instante `t` (segundos)."""
        if self._ultimo is not None:
            self.anotar_dt(t - self._ultimo)
        self._ultimo = t

    def anotar_dt(self, dt: float) -> None:
        """Un intervalo ya medido entre dos imágenes."""
        if not 0.0 < dt <= MAX_DT_S:
            return
        self.dt_medio = (dt if self.dt_medio is None
                         else (1.0 - ALFA) * self.dt_medio + ALFA * dt)

    @property
    def fps(self) -> Optional[float]:
        return 1.0 / self.dt_medio if self.dt_medio else None


def imagenes_para(segundos: float, minimo: int,
                  fps: Optional[float]) -> int:
    """Imágenes que cubren `segundos` a `fps`, nunca menos que `minimo`."""
    minimo = int(minimo)
    if not fps or segundos is None or float(segundos) <= 0:
        return minimo
    # El -1e-9 evita que 0,25 s × 8 fps (=2,0000000001) pida 3.
    return max(minimo, int(math.ceil(float(segundos) * fps - 1e-9)))


#: Fps a los que están validados los ajustes contados en imágenes: la pasada
#: G del 5-oct-2026 (mismo tramo del vídeo, agarres y visitas revisados).
FPS_REFERENCIA = 7.7


def imagenes_ref(imagenes: int, fps: Optional[float]) -> int:
    """`imagenes` validadas a FPS_REFERENCIA, convertidas al MISMO tiempo
    real a `fps`, y nunca menos que `imagenes`. Es la regla de la tabla de
    arriba sin inventar unos segundos por ajuste: 3 imágenes = 0,39 s, que a
    10 fps son 4 y a 15 fps, 6. Sin fps medidos, las de siempre."""
    imagenes = int(imagenes)
    return imagenes_para(imagenes / FPS_REFERENCIA, imagenes, fps)


def fps_de(tracker: Any) -> Optional[float]:
    """Los fps medidos de la cámara cuyo ByteTrack es `tracker` (los anota
    `ajustar_bytetrack` en cada imagen). None si aún no se sabe."""
    ritmo = getattr(tracker, '_elde_ritmo', None)
    return ritmo.fps if ritmo is not None else None


def ajustar_bytetrack(tracker: Any, t: float, cfg: Any) -> None:
    """Antes de cada `update_with_detections`: anota la imagen y pone en el
    ByteTrack la memoria de tapado y la confirmación EN SEGUNDOS (con el
    suelo de siempre en imágenes). supervision lee `max_time_lost` y
    `minimum_consecutive_frames` en cada imagen. Nunca lanza."""
    try:
        ritmo = getattr(tracker, '_elde_ritmo', None)
        if ritmo is None:
            ritmo = tracker._elde_ritmo = Ritmo()
            # El suelo = lo que el ByteTrack calculó al nacer (90 con los
            # valores de siempre: frame_rate/30 * lost_track_buffer).
            tracker._elde_perdido_min = int(tracker.max_time_lost)
            tracker._elde_confirmar_min = int(tracker.minimum_consecutive_frames)
        ritmo.anotar(t)
        tracker.max_time_lost = imagenes_para(
            float(getattr(cfg, 'BYTETRACK_LOST_S', 0.0)),
            tracker._elde_perdido_min, ritmo.fps)
        tracker.minimum_consecutive_frames = imagenes_para(
            float(getattr(cfg, 'BYTETRACK_MIN_CONSECUTIVE_S', 0.0)),
            tracker._elde_confirmar_min, ritmo.fps)
    except Exception:                                       # noqa: BLE001
        pass
