"""src/analityc/core/anaqueles_pose.py — muñecas para el modo Anaqueles.

## Por qué muñecas y no manos

No hay ningún modelo de manos en este servidor (MediaPipe está vetado; ver
PLAN.md) y el único motor de keypoints instalado y probado es el de
`pose_motor.crear_motor_pose` (ViTPose, único motor desde el 3-sep-2026),
el mismo que usa la degustación del stand
(`stand_puente._keypoints`). De los 17 keypoints COCO, los que responden a
«metió la mano al anaquel» son las muñecas: índice 9 (izquierda) y 10
(derecha).

## Por qué sobre el RECORTE de la persona

Correr pose sobre el frame completo paga por toda la escena en cada frame.
Sobre el recorte del bbox solo se paga por quien está pegado al anaquel
(medido en la 5060 Ti: ~41 fps por recorte), y el que llama decide a quiénes
(`anaqueles.py`, en `_medir`): por defecto, solo a quien está QUIETO en la
compuerta de un área; con `pose_cercania` —encendida en
`anaqueles_config.yaml` desde el 29-sep-2026— también a quien está en la
compuerta aunque se mueva y a quien está CERCA del área sin pisarla. Como
mucho `pose_max_personas` por frame. Los keypoints vuelven en coordenadas
del recorte; aquí se les suma el origen para devolverlos en coordenadas del
FRAME, que es donde viven los polígonos de las áreas.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Índices COCO de las muñecas en los 17 keypoints de la pose.
MUNECA_IZQ = 9
MUNECA_DER = 10

#: Confianza mínima del keypoint, la misma que calibró la degustación del
#: stand (CONF_KP_MIN). Por debajo, la muñeca «detectada» es ruido que
#: contaría agarres fantasma.
CONF_MUNECA_MIN = 0.30

#: Recorte mínimo utilizable (el mismo umbral que stand_puente._recorte):
#: por debajo la pose devuelve keypoints inventados.
_MIN_ANCHO = 20
_MIN_ALTO = 40


class PoseMunecas:
    """Detector perezoso de muñecas sobre recortes de persona.

    El modelo se carga la PRIMERA vez que hace falta, nunca al construir:
    instanciar esta clase no puede costar segundos dentro del bucle de
    inferencia (mismo patrón que `stand_puente`).
    """

    def __init__(self, device: Any = None,
                 conf_min: float = CONF_MUNECA_MIN) -> None:
        self._device = device
        self._conf_min = float(conf_min)
        self._modelo = None
        self._intentado = False

    @property
    def disponible(self) -> bool:
        """True si el motor cargó (solo significativo tras el primer uso)."""
        return bool(self._modelo is not None and
                    getattr(self._modelo, 'disponible', True))

    def munecas(self, frame: np.ndarray,
                box: Sequence[float]
                ) -> List[Tuple[float, float, float, str]]:
        """Muñecas de la persona del `box`, en coordenadas del FRAME.

        Devuelve `[(x, y, conf, lado), ...]` (0, 1 o 2) con conf >= el
        mínimo y `lado` en ('izq', 'der'). LA IDENTIDAD IMPORTA (v3): sin
        ella, cuando la pose perdía la muñeca interior unos frames, la
        OTRA mano «armaba» el gesto y regalaba la amplitud — un agarre
        falso cada cooldown para cualquiera parado (lo cazó la revisión).
        Lista vacía si no hay modelo, el recorte es muy chico o la pose no
        vio muñecas fiables. Nunca lanza.
        """
        self._cargar_si_falta()
        if self._modelo is None:
            return []
        recorte, origen = self._recorte(frame, box)
        if recorte is None:
            return []
        try:
            # 17 keypoints [x, y, conf] en coordenadas del RECORTE; el motor
            # ya corre en el device que se le propagó al crearlo.
            kps = self._modelo.keypoints_17(recorte)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("anaqueles: pose falló: %s", exc)
            return []
        return self._de_kps(kps, origen)

    def munecas_lote(self, frame: np.ndarray,
                     boxes: Sequence[Sequence[float]]
                     ) -> List[List[Tuple[float, float, float, str]]]:
        """`munecas` de varias personas del mismo frame, en UNA llamada al
        modelo y con los recortes preparados en la GPU (6-oct-2026,
        `pose_gpu.py`). Mismo resultado que llamar a `munecas` caja a caja
        (medido sobre 298 fotos de fichas: muñecas a ≤ 0,23 px). Si el
        motor no puede (sin CUDA, interruptor `ELDE_POSE_PREPARAR_EN_GPU=0`
        o fallo), se hace caja a caja como siempre. Nunca lanza."""
        boxes = list(boxes)
        self._cargar_si_falta()
        if self._modelo is None:
            return [[] for _ in boxes]
        cajas: List[Tuple[int, int, int, int]] = []
        indices: List[int] = []
        for i, box in enumerate(boxes):
            c = self._caja_recorte(frame, box)
            if c is not None:
                cajas.append(c)
                indices.append(i)
        lote_fn = getattr(self._modelo, 'keypoints_lote', None)
        todos = None
        if cajas and lote_fn is not None:
            try:
                todos = lote_fn(frame, cajas)
            except Exception as exc:                        # noqa: BLE001
                logger.debug("anaqueles: pose en lote falló: %s", exc)
        if todos is None or len(todos) != len(cajas):
            return [self.munecas(frame, b) for b in boxes]
        salida: List[List[Tuple[float, float, float, str]]] = [
            [] for _ in boxes]
        for i, c, kps in zip(indices, cajas, todos):
            salida[i] = self._de_kps(kps, (c[0], c[1]))
        return salida

    def _de_kps(self, kps, origen: Tuple[int, int]
                ) -> List[Tuple[float, float, float, str]]:
        """Muñecas fiables de 17 keypoints del recorte, en coords del frame."""
        if not kps:
            return []
        ox, oy = origen
        salida: List[Tuple[float, float, float, str]] = []
        for idx, lado in ((MUNECA_IZQ, 'izq'), (MUNECA_DER, 'der')):
            if idx >= len(kps):
                continue
            x, y, conf = kps[idx][:3]
            if float(conf) < self._conf_min:
                continue
            salida.append((float(x) + ox, float(y) + oy, float(conf), lado))
        return salida

    # ── Interno ─────────────────────────────────────────────────────────
    def _cargar_si_falta(self) -> None:
        if self._intentado:
            return
        self._intentado = True
        try:
            # ViTPose es el unico motor de pose desde el 3-sep-2026: la
            # salida de ultralytics (AGPL) de anaqueles. El
            # device del detector principal se propaga para que la pose
            # corra donde ya viven los demás modelos.
            from .pose_motor import crear_motor_pose
            self._modelo = crear_motor_pose(self._device)
            if self._modelo is None:
                logger.error("anaqueles: sin motor de pose — los AGARRES no "
                             "se detectarán (las visitas sí)")
        except Exception as exc:                            # noqa: BLE001
            logger.error("anaqueles: no se pudo crear el motor de pose: %s",
                         exc)

    @staticmethod
    def _caja_recorte(frame: np.ndarray, box: Sequence[float]
                      ) -> Optional[Tuple[int, int, int, int]]:
        """La caja ya recortada al frame, o None si queda muy chica. La
        misma regla para el camino de siempre y el del lote."""
        try:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < _MIN_ANCHO or y2 - y1 < _MIN_ALTO:
                return None
            return x1, y1, x2, y2
        except Exception:                                   # noqa: BLE001
            return None

    @classmethod
    def _recorte(cls, frame: np.ndarray, box: Sequence[float]
                 ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        c = cls._caja_recorte(frame, box)
        if c is None:
            return None, (0, 0)
        x1, y1, x2, y2 = c
        return frame[y1:y2, x1:x2].copy(), (x1, y1)
