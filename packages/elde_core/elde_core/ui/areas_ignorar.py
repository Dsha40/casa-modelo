"""Áreas a ignorar dibujadas en el recuadro (12-oct-2026).

## Qué es

Cuadriláteros de cuatro esquinas arrastrables que el operador pone sobre el
fondo de la imagen. El servidor descarta a toda persona cuya caja quede
ENTERA dentro de una de ellas (`zonas_ignorar.py` del servidor), antes del
seguimiento: no cuenta para nada. Lo pidió el usuario: en la cámara 3 se
cuela gente del fondo que no está en el pasillo.

Viven en el mismo espacio 0..1000 que las demás zonas y viajan en el campo
`zonas_ignorar` del contrato, en píxeles del frame, solo con la herramienta
encendida (apagada = lista vacía = no se ignora nada).

## Por qué un fichero aparte y sin Qt arriba

Como `productos_anaquel.py`: `interactive_image_label.py` ya carga muchas
herramientas y lo usan varios clientes. Aquí los puntos son listas de
enteros y Qt solo se importa al pintar, así que dónde nace un área, qué
esquina se arrastra y qué viaja al servidor se prueban sin PySide6.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Tuple

#: Radio de las esquinas, el mismo que el de las áreas de anaquel.
RADIO = 10

#: Rojo oscuro: ninguna otra herramienta lo usa (ROI amarillo, pedido azul,
#: entrega rojo vivo, personal gris, anaqueles naranja, productos violeta).
_COLOR_RGB = (200, 40, 70)


class AreasIgnorar:
    def __init__(self, datos: Any = None, activa: bool = False) -> None:
        self.areas: List[List[List[int]]] = []
        self.activa = bool(activa)
        self.poner(datos)

    # ── Estado ──────────────────────────────────────────────────────────
    def poner(self, datos: Any) -> None:
        """Repone lo guardado (lista de áreas de 3+ puntos en 0..1000)."""
        self.areas = []
        for area in datos or []:
            try:
                puntos = [[max(0, min(1000, int(x))), max(0, min(1000, int(y)))]
                          for x, y in area]
            except (TypeError, ValueError):
                continue
            if len(puntos) >= 3:
                self.areas.append(puntos)

    def a_lista(self) -> List[List[List[int]]]:
        return [[list(p) for p in a] for a in self.areas]

    def agregar(self) -> None:
        """Un área nueva, arriba (donde suele estar el fondo) y desplazada
        respecto a la anterior para que no nazcan una encima de otra."""
        n = len(self.areas)
        x0 = 100 + (n * 120) % 600
        self.areas.append([[x0, 60], [x0 + 300, 60],
                           [x0 + 300, 360], [x0, 360]])
        self.activa = True

    def quitar_ultima(self) -> bool:
        if not self.areas:
            return False
        self.areas.pop()
        return True

    # ── Ratón ───────────────────────────────────────────────────────────
    def vertice_en(self, pos: Any, escalar: Callable[[int, int], Any],
                   radio: int = RADIO) -> Optional[Tuple[int, int]]:
        """`(área, esquina)` bajo `pos` (píxeles del visor), o `None`. Del
        último al primero: el último se pinta encima."""
        if not self.activa:
            return None
        for i in range(len(self.areas) - 1, -1, -1):
            for j, (x, y) in enumerate(self.areas[i]):
                p = escalar(x, y)
                if math.hypot(pos.x() - p.x(), pos.y() - p.y()) < radio:
                    return i, j
        return None

    def mover(self, indice: int, esquina: int, x: int, y: int) -> None:
        try:
            self.areas[indice][esquina] = [max(0, min(1000, int(x))),
                                           max(0, min(1000, int(y)))]
        except (IndexError, TypeError, ValueError):
            pass

    # ── Hacia el servidor ───────────────────────────────────────────────
    def coordenadas_pixel(self, ancho: int, alto: int) -> List[List[List[int]]]:
        """Lo que viaja en `zonas_ignorar`. VACÍA si la herramienta está
        apagada: apagarla tiene que dejar de ignorar."""
        if not self.activa or ancho <= 0 or alto <= 0:
            return []
        return [[[int(x * ancho / 1000), int(y * alto / 1000)] for x, y in a]
                for a in self.areas]

    # ── Dibujo ──────────────────────────────────────────────────────────
    def pintar(self, painter: Any, escalar: Callable[[int, int], Any],
               radio: int = RADIO,
               activo: Optional[Tuple[int, int]] = None) -> None:
        """Contorno discontinuo, relleno tenue y rótulo «Ignorar N»."""
        if not self.activa:
            return
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QBrush, QColor, QPen, QPolygon

        color = QColor(*_COLOR_RGB)
        relleno = QColor(*_COLOR_RGB, 45)
        for i, area in enumerate(self.areas):
            pix = [escalar(x, y) for x, y in area]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(relleno))
            painter.drawPolygon(QPolygon(pix))
            painter.setPen(QPen(color, 2, Qt.DashLine))
            for a, b in zip(pix, pix[1:] + pix[:1]):
                painter.drawLine(a, b)
            for j, q in enumerate(pix):
                painter.setPen(QPen(Qt.black, 2))
                painter.setBrush(QBrush(Qt.white if activo == (i, j) else color))
                painter.drawEllipse(q, radio, radio)
            etiqueta = f"Ignorar {i + 1}"
            origen = pix[0]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.black, 4))
            painter.drawText(origen.x() + 14, origen.y() - 9, etiqueta)
            painter.setPen(QPen(color, 2))
            painter.drawText(origen.x() + 13, origen.y() - 10, etiqueta)
