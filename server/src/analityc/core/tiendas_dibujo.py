"""src/analityc/core/tiendas_dibujo.py — lo que se PINTA de cada frame.

Extraído de `tiendas.py` (HITO 5) con el cuerpo verbatim. Aquí viven los dos
caminos que tiene el pipeline para que el operador vea las cajas:

  - **modo clásico**: `draw_detections` dibuja sobre el frame en el servidor y
    la imagen anotada viaja al cliente;
  - **modo directo**: `_build_detections` NO dibuja nada — devuelve la lista de
    cajas, etiquetas y colores para que las pinte el cliente, que es mucha
    menos latencia y mucho menos ancho de banda.

Los dos tienen que decir LO MISMO (mismos colores, mismas etiquetas): que
estén juntos en un archivo es justamente para que quien toque uno vea el otro.
`PermanenciaArea` extiende `_build_detections` con `super()`, y sigue
funcionando igual porque esto entra como mixin de `TiendasProcess`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .analytics.config import AnalyticsConfig
from .utils.overlay import draw_demographics_panel, draw_people_total

logger = logging.getLogger(__name__)


class DibujoMixin:
    """El overlay del servidor y su equivalente para el cliente."""

    def _build_detections(self) -> list:
        """Lista de detecciones para que el CLIENTE las dibuje (modo directo,
        baja latencia): [{box:[x1,y1,x2,y2], label:str, color:[r,g,b],
        tracker_id:int, class_id:int, confidence:float}].
        Replica la misma logica/colores/etiquetas que draw_detections, pero
        sin tocar la imagen -> el servidor no codifica ni envia el frame.
        tracker_id/class_id/confidence permiten al cliente armar un
        sv.Detections (Supervision) para estelas, zonas y heatmap.
        """
        # Mismos colores que draw_detections (BGR). Se convierten a RGB para Qt.
        category_colors = {
            'Hombres': (255, 180, 0),
            'Mujeres': (180, 0, 255),
            'Niños':   (0, 255, 180),
            'Personas': (0, 255, 255),
        }
        # class_id estable por categoria (para color_lookup CLASS en Supervision).
        category_ids = {'Hombres': 0, 'Mujeres': 1, 'Niños': 2, 'Personas': 3}
        # MODO DIRECTO: se mandan TODAS las personas (sin filtrar por ROI ni por
        # 'counted'). Asi el cliente (Supervision) ve a todos y PolygonZone puede
        # contar dentro/fuera de las zonas. El filtro de ROI lo aplica el cliente
        # con sus zonas, no el servidor. (El modo clasico draw_detections SI sigue
        # filtrando por ROI; esto solo afecta al modo directo.)
        out = []
        for _tid, obj in self.active_tracks.items():
            if obj.get('class', -1) != -1:   # solo personas (class interno -1)
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in obj['box']]
            except (KeyError, TypeError, ValueError):
                continue
            category = obj.get('category', 'Personas')
            confidence = obj.get('confidence', 0)
            demo = (self._demographics.get_cached(_tid)
                    if hasattr(self, '_demographics') else None)
            suffix = ""
            if demo:
                g = demo.get('gender', 'Desconocido')
                a = demo.get('age_range', 'Desconocido')
                suffix = (" [?]" if (g == 'Desconocido' and a == 'Desconocido')
                          else f" [{g[:1]}|{a}]")
            text = f"{category} {confidence:.2f}{suffix}"
            if 'entry_time' in obj:
                t = int(time.time() - obj['entry_time'])
                text += f" - {t // 60}m {t % 60}s"
            b, g_, r = category_colors.get(category, (0, 255, 0))
            out.append({"box": [x1, y1, x2, y2], "label": text,
                        "color": [r, g_, b],
                        "tracker_id": int(_tid),
                        "class_id": int(category_ids.get(category, 3)),
                        "confidence": float(confidence)})
        return out

    def draw_detections(self, image: np.ndarray, persons_inside: int) -> np.ndarray:
        # Dibujar zona de conteo (estilo configurable — Hito 2).
        _zcol = tuple(AnalyticsConfig.ZONE_DRAW_COLOR_BGR)
        _zop = float(AnalyticsConfig.ZONE_FILL_OPACITY)
        _zth = int(AnalyticsConfig.ZONE_DRAW_THICKNESS)
        roi_overlay = image.copy()
        cv2.fillPoly(roi_overlay, [self.roi_polygon], _zcol)
        cv2.addWeighted(roi_overlay, _zop, image, 1.0 - _zop, 0, image)
        cv2.polylines(image, [self.roi_polygon], isClosed=True,
                      color=_zcol, thickness=_zth)
        # Numero de aforo instantaneo (personas dentro de la zona) — Hito 2.
        if AnalyticsConfig.ZONE_SHOW_COUNT:
            _zc = int(getattr(self, '_zone_count', 0))
            _px, _py = int(self.roi_polygon[:, 0].min()), int(self.roi_polygon[:, 1].min())
            _lbl = f"Dentro: {_zc}"
            cv2.putText(image, _lbl, (_px + 6, max(_py - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, _lbl, (_px + 6, max(_py - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _zcol, 2, cv2.LINE_AA)

        for x, y in self.roi_polygon:
            cv2.circle(image, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(image, (x, y), 8, (255, 255, 255), 2)

        # Colores por categoría (personas)
        category_colors = {
            'Hombres': (255, 180, 0),    # Azul claro
            'Mujeres': (180, 0, 255),    # Rosa/Magenta
            'Niños':   (0, 255, 180),    # Verde/Cyan
            'Personas': (0, 255, 255),   # Amarillo
        }

        # Preparar polígono ROI para filtrar dibujo
        roi_pts = self.roi_polygon.reshape((-1, 1, 2)) if self.roi_polygon is not None else None

        # Solo personas (class == -1); cosmeticos eliminado.
        ordered_items = [
            (tid, obj) for tid, obj in self.active_tracks.items()
            if not obj.get('counted', False)
            and obj.get('class', -1) == -1
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        for _tid, obj in ordered_items:
            staff_id_pre = obj.get('class', -1)
            # Filtrar por ROI amarillo SOLO a personas (-1).
            if staff_id_pre == -1 and roi_pts is not None:
                cx, cy = obj.get('center', (0, 0))
                if cv2.pointPolygonTest(roi_pts, (int(cx), int(cy)), False) < 0:
                    continue

            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            staff_id = obj.get('class', -1)
            category = obj.get('category', 'Personas')
            confidence = obj.get('confidence', 0)

            demo = None
            if staff_id == -1 and hasattr(self, '_demographics'):
                demo = self._demographics.get_cached(_tid)

            def _demo_suffix(d):
                if not d:
                    return ""
                g = d.get('gender', 'Desconocido')
                a = d.get('age_range', 'Desconocido')
                if g == 'Desconocido' and a == 'Desconocido':
                    return " [?]"
                return f" [{g[:1]}|{a}]"

            # Solo personas: color por categoria demografica.
            color = category_colors.get(category, (0, 255, 0))
            text = f"{category} {confidence:.2f}{_demo_suffix(demo)}"
            thickness = 2

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            if 'entry_time' in obj:
                current_time = time.time()
                time_in_roi = int(current_time - obj['entry_time'])
                minutes = time_in_roi // 60
                seconds = time_in_roi % 60
                text += f" - {minutes}m {seconds}s"

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )

            cv2.rectangle(
                image, (x1, y1 - text_height - 10),
                (x1 + text_width, y1), (0, 0, 0), -1
            )
            cv2.putText(
                image, text, (x1, y1 - 5),
                font, font_scale, (255, 255, 255),
                font_thickness
            )
        
        # ── Overlay de analitica de personas ──
        if self.show_minimal_info:
            # 1. Panel demografico (izquierda)
            if hasattr(self, '_demographics'):
                demo_counts = self._demographics.get_counts()
                image = draw_demographics_panel(image, demo_counts, x=10, y=10)

            # 2. Personas totales (superior central). Si no hay active_tracks
            # usamos el contador heredado persons_inside como fallback.
            total_unique = 0
            active_persons = sum(
                1 for t in self.active_tracks.values()
                if t.get('class', -1) == -1 and t.get('is_inside', False)
            )
            active_now = active_persons if self.active_tracks else persons_inside
            if hasattr(self, '_people_counter'):
                total_unique = self._people_counter.total_unique
            image = draw_people_total(image, total_unique, active_now)

        return image
