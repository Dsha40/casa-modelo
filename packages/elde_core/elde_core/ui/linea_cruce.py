"""Línea de cruce: se dibuja SOBRE el umbral y rotula qué lado es DENTRO.

## En qué se diferencia de la flecha

La flecha (`flecha_direccion.py`) se dibuja A LO LARGO de la dirección de
caminata y el servidor proyecta recorridos sobre ella. Esta línea se dibuja
A LO ANCHO del vano —donde la puerta ES, una referencia física visible— y el
servidor solo mira de qué lado a qué lado se pasó. Para cámaras gran angular
donde la gente entra curvándose, que es donde la proyección confunde.

## Convención de lados (la misma que el servidor)

Caminando de P1 hacia P2, **DENTRO queda a la DERECHA** (con la `y` de
imagen hacia abajo). El operador no necesita saberlo: los rótulos
DENTRO/FUERA y la flechita perpendicular lo pintan. Doble clic sobre la
línea = intercambiar P1 y P2 = voltear los lados.

## Qué hereda

Estado, geometría, arrastre (extremos, cuerpo, rombo central), `invertir()`
y `trasladar()` vienen de `FlechaDireccion`: mismos dos puntos en el espacio
0..1000 del visor, mismo `LARGO_MINIMO` para no aceptar un segmento que no
define nada. Solo cambian el pintado y la semántica del lado.

Nace **inactiva**: los clientes que no la enciendan no ven ningún cambio.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen

from .flecha_direccion import (LARGO_MINIMO, ROJO, VERDE, _GRIS, _SOMBRA,
                               FlechaDireccion)

#: El cuerpo de la línea: cian, que no choca con el ROI amarillo, las zonas
#: azul/roja ni la flecha verde/roja.
CIAN = QColor(41, 182, 246)


class LineaCruce(FlechaDireccion):
    """Dos puntos sobre el umbral, con su lado DENTRO rotulado."""

    # ── Semántica del lado ───────────────────────────────────────────────
    def normal_dentro(self) -> Optional[Tuple[float, float]]:
        """Unitario perpendicular apuntando al lado DENTRO, o `None`.

        Rotación de P1→P2: `n = (-dy, dx)`. Con la `y` de imagen hacia
        abajo es la derecha del sentido de dibujo — la MISMA cuenta que hace
        `permanencia_linea.py` en el servidor.
        """
        dx = self.punta.x() - self.cola.x()
        dy = self.punta.y() - self.cola.y()
        largo = math.hypot(dx, dy)
        if largo < 1e-6:
            return None
        return (-dy / largo, dx / largo)

    def lado(self, punto: QPoint) -> int:
        """+1 si `punto` cae del lado DENTRO, -1 FUERA, 0 sobre la línea."""
        dx = self.punta.x() - self.cola.x()
        dy = self.punta.y() - self.cola.y()
        cruz = dx * (punto.y() - self.cola.y()) - dy * (punto.x() - self.cola.x())
        if cruz > 0:
            return 1
        if cruz < 0:
            return -1
        return 0

    # ── Pintado ──────────────────────────────────────────────────────────
    def pintar(self, painter: QPainter, escalar, radio: int,
               indice_activo: int = -1) -> None:
        """Línea, flechita hacia DENTRO, rombo y rótulos. Nunca lanza."""
        if not self.activa:
            return
        try:
            from .flecha_direccion import INDICE_MEDIO
            p1, p2 = escalar(self.cola), escalar(self.punta)
            corto = self.largo() < LARGO_MINIMO

            estilo = Qt.DashLine if corto else Qt.SolidLine
            painter.setPen(QPen(_SOMBRA, 7, estilo, Qt.RoundCap))
            painter.drawLine(p1, p2)
            painter.setPen(QPen(_GRIS if corto else CIAN, 4, estilo,
                                Qt.RoundCap))
            painter.drawLine(p1, p2)

            medio = escalar(self.punto_medio())
            if not corto:
                # La flechita perpendicular: nace junto al rombo y apunta al
                # lado DENTRO. Se calcula en PÍXELES porque el visor no
                # conserva la proporción del espacio 0..1000.
                ndx = -(p2.y() - p1.y())
                ndy = (p2.x() - p1.x())
                nlargo = math.hypot(ndx, ndy)
                if nlargo > 1e-6:
                    ndx, ndy = ndx / nlargo, ndy / nlargo
                    base = QPoint(int(medio.x() + ndx * (radio + 4)),
                                  int(medio.y() + ndy * (radio + 4)))
                    apex = QPoint(int(medio.x() + ndx * (radio + 34)),
                                  int(medio.y() + ndy * (radio + 34)))
                    painter.setPen(QPen(_SOMBRA, 5, Qt.SolidLine, Qt.RoundCap))
                    painter.drawLine(base, apex)
                    painter.setPen(QPen(VERDE, 3, Qt.SolidLine, Qt.RoundCap))
                    painter.drawLine(base, apex)
                    self._pintar_cabeza(painter, base, apex, VERDE)
                    # Rótulos de lado, anclados sobre la normal para que
                    # queden cada uno en su medio plano.
                    dentro = QPoint(int(medio.x() + ndx * (radio + 44)),
                                    int(medio.y() + ndy * (radio + 44)))
                    fuera = QPoint(int(medio.x() - ndx * (radio + 24)),
                                   int(medio.y() - ndy * (radio + 24)))
                    self._pintar_etiqueta(painter, dentro, 'DENTRO', VERDE, 0)
                    self._pintar_etiqueta(painter, fuera, 'FUERA', ROJO, 0)
                self._pintar_rombo(painter, medio, radio,
                                   indice_activo == INDICE_MEDIO)
            else:
                self._pintar_etiqueta(painter, medio, 'LÍNEA MUY CORTA',
                                      _GRIS, 0)

            # Extremos agarrables: cian neutro — aquí los extremos no
            # significan entrada ni salida, solo dónde acaba el umbral.
            for i, pix in enumerate((p1, p2)):
                painter.setPen(QPen(Qt.black, 2))
                painter.setBrush(QBrush(Qt.white if i == indice_activo
                                        else CIAN))
                painter.drawEllipse(pix, radio, radio)
        except Exception:                                   # noqa: BLE001
            # Un fallo pintando no puede tumbar el visor de video.
            pass
