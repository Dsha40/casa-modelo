"""Puerta direccional: la línea del umbral con la flecha doble ENTRADA/SALIDA.

## Qué problema resuelve

Los dos utensilios anteriores le hablan al operador en dos idiomas distintos,
y ninguno es el suyo:

  - `FlechaDireccion` se dibuja A LO LARGO del sentido de caminata. El
    operador tiene que imaginar por dónde pasa la gente, no señalar dónde
    está la puerta.
  - `LineaCruce` sí se dibuja sobre el umbral —que es una referencia física
    visible— pero rotula los lados como DENTRO y FUERA. Quien calibra una
    puerta no piensa en lados: piensa en «por aquí entran y por aquí salen».

Esta es la misma línea del umbral con el rótulo que el operador sí usa: una
flecha DOBLE que la atraviesa por el centro, con la cabeza verde ENTRADA
apuntando hacia donde se entra y la roja SALIDA hacia donde se sale. Dibujas
la puerta, miras la flecha y sabes de un vistazo si quedó al derecho.

## Por qué NO estrena un detector

El eje de la flecha doble es EXACTAMENTE la normal de la línea, y la cabeza
verde cae en el lado que `LineaCruce` llama DENTRO. Es decir: la geometría es
la misma que ya se manda al servidor, punto por punto.

Así que este utensilio viaja por el mismo campo (`linea_cruce`) y lo lee el
mismo `permanencia_linea.DetectorLinea`, que lleva desde el 10-ago-2026 en
producción con su histéresis, sus desempates de vano y su cooldown. Estrenar
un detector paralelo para volver a medir lo mismo habría duplicado esos tres
mecanismos, y con ellos los errores que costó cazar.

Lo único que cambia es lo que ve el operador. Eso no es poco: la calibración
al revés —entradas contadas como salidas— es el fallo más frecuente de este
modo, y se corrige mirando, no midiendo.

## Convención de lados (idéntica al servidor)

Caminando de P1 hacia P2, **DENTRO queda a la DERECHA** (con la `y` de imagen
hacia abajo), y ahí es donde se pinta la cabeza verde. El operador no razona
sobre la convención: la ve. Doble clic sobre el utensilio intercambia P1 y P2
y con ello ENTRADA y SALIDA.

## Qué hereda

Todo el estado y toda la geometría: los mismos dos puntos en el espacio
0..1000 del visor, `LARGO_MINIMO`, el arrastre de extremos, cuerpo y rombo
central, `invertir()`, `trasladar()`, `normal_dentro()` y `lado()`.

Lo único propio es el DIBUJO: `pintar()` y sus dos ayudantes `_brazo()` y
`_pintar_extremos()`. Ni una línea de geometría, a propósito — si esta clase
calculara algo por su cuenta, podría divergir de lo que mide el servidor, y
ese desacuerdo es invisible hasta que alguien reporta entradas contadas como
salidas. Lo fija `test_puerta_extremo_a_extremo.py`.

Nace **inactiva**: quien no la encienda no ve ningún cambio.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QPainter, QPen

from .flecha_direccion import (INDICE_MEDIO, LARGO_MINIMO, ROJO, VERDE, _GRIS,
                               _SOMBRA)
from .linea_cruce import CIAN, LineaCruce

#: Largo de cada brazo de la flecha doble, en PÍXELES del visor. Se calcula
#: como una fracción del largo de la línea pero acotado por los dos lados:
#: sin tope, una línea que cruza la pantalla dispara una flecha fuera del
#: encuadre; sin suelo, una puerta estrecha la deja del tamaño de un punto.
_BRAZO_FRACCION = 0.38
_BRAZO_MINIMO = 40.0
_BRAZO_MAXIMO = 95.0


class PuertaDireccional(LineaCruce):
    """La línea del umbral, rotulada con una flecha doble ENTRADA/SALIDA."""

    def _brazo(self, largo_pixeles: float) -> float:
        """Cuánto sobresale cada cabeza a partir del centro de la línea."""
        return max(_BRAZO_MINIMO,
                   min(largo_pixeles * _BRAZO_FRACCION, _BRAZO_MAXIMO))

    # ── Pintado ──────────────────────────────────────────────────────────
    def pintar(self, painter: QPainter, escalar, radio: int,
               indice_activo: int = -1) -> None:
        """Umbral en cian + flecha doble atravesándolo. Nunca lanza."""
        if not self.activa:
            return
        try:
            p1, p2 = escalar(self.cola), escalar(self.punta)
            corto = self.largo() < LARGO_MINIMO

            # ── El umbral ────────────────────────────────────────────────
            estilo = Qt.DashLine if corto else Qt.SolidLine
            painter.setPen(QPen(_SOMBRA, 7, estilo, Qt.RoundCap))
            painter.drawLine(p1, p2)
            painter.setPen(QPen(_GRIS if corto else CIAN, 4, estilo,
                                Qt.RoundCap))
            painter.drawLine(p1, p2)

            medio = escalar(self.punto_medio())
            if corto:
                self._pintar_etiqueta(painter, medio, 'PUERTA MUY CORTA',
                                      _GRIS, 0)
                self._pintar_extremos(painter, p1, p2, radio, indice_activo)
                return

            # ── La flecha doble ──────────────────────────────────────────
            # La normal se calcula en PÍXELES y no en el espacio 0..1000: el
            # visor no conserva la proporción, y una perpendicular calculada
            # en el espacio lógico se vería torcida sobre el vídeo — justo en
            # el dibujo cuyo único trabajo es que se vea derecho.
            ndx = -(p2.y() - p1.y())
            ndy = (p2.x() - p1.x())
            nlargo = math.hypot(ndx, ndy)
            if nlargo > 1e-6:
                ndx, ndy = ndx / nlargo, ndy / nlargo
                brazo = self._brazo(nlargo)

                # `apex_dentro` cae del lado que el servidor llama DENTRO:
                # la MISMA cuenta que `normal_dentro()`, para que lo pintado
                # y lo medido no puedan divergir.
                apex_dentro = QPoint(int(medio.x() + ndx * brazo),
                                     int(medio.y() + ndy * brazo))
                apex_fuera = QPoint(int(medio.x() - ndx * brazo),
                                    int(medio.y() - ndy * brazo))
                # Con la puerta pegada al borde del encuadre —el caso más
                # común— un apex fuera del visor haría que QPainter recortara
                # la cabeza ENTERA, que es justo la señal de si quedó al
                # revés. Se mete dentro en vez de desaparecer.
                dispositivo = painter.device()
                if dispositivo is not None:
                    for apex in (apex_dentro, apex_fuera):
                        apex.setX(max(8, min(apex.x(),
                                             dispositivo.width() - 8)))
                        apex.setY(max(8, min(apex.y(),
                                             dispositivo.height() - 8)))

                # Cada brazo de su color: el sentido se lee sin acercarse a
                # mirar las etiquetas, y sigue leyéndose si una persona tapa
                # una de las dos cabezas.
                painter.setPen(QPen(_SOMBRA, 7, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(apex_fuera, apex_dentro)
                painter.setPen(QPen(VERDE, 4, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(medio, apex_dentro)
                painter.setPen(QPen(ROJO, 4, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(medio, apex_fuera)

                self._pintar_cabeza(painter, medio, apex_dentro, VERDE)
                self._pintar_cabeza(painter, medio, apex_fuera, ROJO)
                self._pintar_etiqueta(painter, apex_dentro, 'ENTRADA',
                                      VERDE, 0)
                self._pintar_etiqueta(painter, apex_fuera, 'SALIDA', ROJO, 0)

            # El rombo, DESPUÉS de la flecha: es el agarre de «mover entera»
            # y cae en el centro, donde se cruzan los dos brazos; pintado
            # antes quedaría sepultado bajo ellos y no se podría ver dónde
            # coger la puerta.
            self._pintar_rombo(painter, medio, radio,
                               indice_activo == INDICE_MEDIO)
            self._pintar_extremos(painter, p1, p2, radio, indice_activo)
        except Exception:                                   # noqa: BLE001
            # Un fallo pintando no puede tumbar el visor de video.
            pass

    @staticmethod
    def _pintar_extremos(painter: QPainter, p1, p2, radio: int,
                         indice_activo: int) -> None:
        """Los dos agarres del umbral, en cian.

        Neutros a propósito: aquí los extremos dicen dónde ACABA la puerta,
        no por dónde se entra. Quien quiera cambiar el sentido tiene la
        flecha doble y el doble clic; que los extremos fueran verde y rojo
        sugeriría que arrastrarlos es lo que decide la dirección.
        """
        for i, pix in enumerate((p1, p2)):
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(Qt.white if i == indice_activo else CIAN))
            painter.drawEllipse(pix, radio, radio)
