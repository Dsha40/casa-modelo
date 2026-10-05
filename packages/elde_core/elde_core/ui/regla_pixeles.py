"""Regla manual para medir en píxeles sobre el vídeo (2-oct-2026).

## Para qué

Para decidir a partir de qué tamaño se descarta a la gente del FONDO del
pasillo hace falta saber cuánto mide en pantalla. El servidor ya aparta de
la demografía a quien mide menos de 60×140 px («Persona muy lejana», en
`demographics.py`), pero desde el cliente no había forma de ver cuánto mide
una persona concreta. Con la regla: se activa en el menú Vista, se hace clic
y se arrastra, por ejemplo de la cabeza a los pies, y se pinta la línea con
su longitud.

## En qué píxeles mide

En los de la IMAGEN ORIGINAL que se manda al servidor, no en los de la
pantalla. Así la cifra se compara tal cual con los umbrales del servidor,
mida lo que mida el recuadro. Se usa la misma conversión que las zonas: los
puntos viven en el espacio 0..1000 del visor y pasan a la imagen como
`x * ancho / 1000` (igual que `get_coordinates`). Sin tamaño de imagen
(aún no ha llegado ningún frame) no hay medida.

Además de la longitud se dan el ALTO y el ANCHO por separado: a una persona
se la mide de pie, y lo que compara el servidor es el alto de su caja.

## Por qué un fichero aparte y sin Qt arriba

Como `productos_anaquel.py`: la lógica se prueba sin PySide6 y Qt solo se
importa al pintar.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

#: Turquesa: no lo usa ninguna otra herramienta del recuadro (ROI amarillo,
#: pedido azul, entrega rojo, personal gris, áreas naranja, productos
#: violeta, muñecas verde y ámbar).
_COLOR_RGB = (0, 220, 255)
_RADIO = 5


class ReglaPixeles:
    """Una línea de medida en el espacio 0..1000 del visor."""

    def __init__(self) -> None:
        self.activa = False
        self.arrastrando = False
        self.p0: Optional[Tuple[float, float]] = None
        self.p1: Optional[Tuple[float, float]] = None
        #: Devuelve (ancho, alto) de la imagen original en píxeles. Lo pone
        #: el recuadro, que es quien sabe qué frame se está viendo.
        self.tamano_frame: Callable[[], Tuple[int, int]] = lambda: (0, 0)

    # ── Ratón ───────────────────────────────────────────────────────────
    def empezar(self, x: float, y: float) -> None:
        self.p0 = self.p1 = (float(x), float(y))
        self.arrastrando = True

    def mover(self, x: float, y: float) -> None:
        if self.arrastrando:
            self.p1 = (float(x), float(y))

    def terminar(self) -> None:
        self.arrastrando = False

    def borrar(self) -> None:
        self.p0 = self.p1 = None
        self.arrastrando = False

    # ── Medida ──────────────────────────────────────────────────────────
    def medida(self) -> Optional[Dict[str, float]]:
        """Longitud, alto y ancho en píxeles de la imagen original, o
        `None` si no hay línea o todavía no se sabe el tamaño de la imagen."""
        if self.p0 is None or self.p1 is None:
            return None
        try:
            ancho_img, alto_img = self.tamano_frame()
        except Exception:                                   # noqa: BLE001
            return None
        if not ancho_img or not alto_img:
            return None
        dx = abs(self.p1[0] - self.p0[0]) * ancho_img / 1000.0
        dy = abs(self.p1[1] - self.p0[1]) * alto_img / 1000.0
        return {'px': math.hypot(dx, dy), 'alto': dy, 'ancho': dx}

    def texto(self) -> str:
        m = self.medida()
        if m is None:
            return 'sin imagen todavía'
        return f"{m['px']:.0f} px  (alto {m['alto']:.0f} · ancho {m['ancho']:.0f})"

    # ── Dibujo ──────────────────────────────────────────────────────────
    def pintar(self, painter: Any, escalar: Callable[[float, float], Any]) -> None:
        """Línea turquesa con sus extremos y la medida junto al final.
        `escalar(x, y)` pasa de 0..1000 a píxeles del visor (`QPoint`)."""
        if self.p0 is None or self.p1 is None:
            return
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QBrush, QColor, QPen

        color = QColor(*_COLOR_RGB)
        a, b = escalar(*self.p0), escalar(*self.p1)
        painter.setPen(QPen(Qt.black, 5))
        painter.drawLine(a, b)
        painter.setPen(QPen(color, 2))
        painter.drawLine(a, b)
        for q in (a, b):
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(q, _RADIO, _RADIO)
        texto = self.texto()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(Qt.black, 4))
        painter.drawText(b.x() + 11, b.y() - 7, texto)
        painter.setPen(QPen(color, 2))
        painter.drawText(b.x() + 10, b.y() - 8, texto)
