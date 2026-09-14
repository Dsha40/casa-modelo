"""Flecha de direccion: el operador dibuja «hacia adentro», una sola vez.

## Por que UNA flecha y no dos

Una puerta tiene **un umbral**. Si marcas el sentido de entrada, el de salida
es el contrario por definicion; no es informacion nueva, es la misma dicha dos
veces. Dos flechas independientes se pueden dibujar casi paralelas, y entonces
un mismo movimiento encaja en las dos: el sistema tendria que elegir una y
estaria contradiciendo al operador sin decirselo.

Aqui se dibuja un solo gesto —arrastrar de fuera hacia dentro— pero se PINTAN
los dos sentidos: cabeza verde «ENTRADA» en la punta y cabeza roja «SALIDA»
saliendo por la cola, cada una con su etiqueta de color. El dato sigue siendo
un unico vector; lo doble es solo la presentacion, que es lo que el operador
necesita para comprobar de un vistazo que no la dejo al reves.

## Como se maneja

  - **Extremos** (circulos): arrastran cola o punta, como siempre.
  - **Rombo central o cuerpo**: arrastran la flecha ENTERA sin cambiar su
    direccion, para posarla sobre la puerta sin redibujarla.
  - **Doble clic** sobre cualquier parte de la flecha: la INVIERTE. Es el
    arreglo de un segundo cuando las entradas salen como salidas: el error
    mas frecuente es dejarla apuntando al reves, no dibujarla torcida.

## Por que geometria y no un modelo de vision

La direccion no esta en la imagen: es una propiedad del movimiento **entre**
fotogramas. Preguntarle a un VLM «¿esta persona entra o sale?» ensenandole un
fotograma es pedirle que adivine. Y ademas no llega a tiempo: una consulta
tarda decenas de segundos y un cruce de puerta dura uno.

Con la flecha, la decision es una proyeccion: el producto escalar entre el
desplazamiento del track y este vector. Positivo = entro, negativo = salio. Es
determinista, cuesta nanosegundos y se puede probar.

## Sistema de coordenadas

Puntos en el espacio 0..1000 del visor, igual que el ROI, para que sobrevivan
a que se redimensione la ventana. `coordenadas_pixel()` los traduce al tamano
real del frame antes de mandarlos al servidor.

## Compartido

Vive en el nucleo porque el widget que la aloja lo comparten los cuatro
clientes. Nace **inactiva**: quien no la encienda no ve ningun cambio.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                           QPen, QPolygon)

#: Punta de la flecha: hacia donde se ENTRA.
VERDE = QColor(60, 200, 90)
#: Cola de la flecha: por donde se SALE.
ROJO = QColor(230, 70, 70)
_SOMBRA = QColor(0, 0, 0, 170)
_GRIS = QColor(150, 150, 150)

#: Indices de agarre. Cola y punta son posiciones reales; MEDIO es el rombo
#: central Y el cuerpo: cualquiera de los dos arrastra la flecha entera.
INDICE_COLA = 0
INDICE_PUNTA = 1
INDICE_MEDIO = 2

#: Posicion por defecto (centro del visor, horizontal). Nace visible y
#: agarrable: una flecha de longitud cero no se puede coger con el raton.
_POR_DEFECTO = ((350, 500), (650, 500))

#: Longitud minima (en unidades de 0..1000) para que el vector signifique
#: algo. Por debajo, dos puntos casi pegados dan una direccion aleatoria.
LARGO_MINIMO = 40.0


class FlechaDireccion:
    """Dos puntos —cola y punta— con su pintado y su vector unitario."""

    def __init__(self, puntos: Any = None, activa: bool = False) -> None:
        self.activa = bool(activa)
        self.puntos: List[QPoint] = self._desde_lista(puntos)

    # ── Estado ───────────────────────────────────────────────────────────
    @staticmethod
    def _desde_lista(datos: Any) -> List[QPoint]:
        """Acepta lo guardado en los ajustes; cae a la posicion por defecto."""
        try:
            if datos and len(datos) >= 2:
                return [QPoint(int(datos[0][0]), int(datos[0][1])),
                        QPoint(int(datos[1][0]), int(datos[1][1]))]
        except (TypeError, ValueError, IndexError, KeyError):
            pass
        return [QPoint(*_POR_DEFECTO[0]), QPoint(*_POR_DEFECTO[1])]

    def a_lista(self) -> List[List[int]]:
        return [[p.x(), p.y()] for p in self.puntos]

    def poner(self, datos: Any) -> None:
        self.puntos = self._desde_lista(datos)

    @property
    def cola(self) -> QPoint:
        """Por donde se SALE."""
        return self.puntos[0]

    @property
    def punta(self) -> QPoint:
        """Hacia donde se ENTRA."""
        return self.puntos[1]

    def punto_medio(self) -> QPoint:
        return QPoint((self.cola.x() + self.punta.x()) // 2,
                      (self.cola.y() + self.punta.y()) // 2)

    # ── Geometria ────────────────────────────────────────────────────────
    def largo(self) -> float:
        return math.hypot(self.punta.x() - self.cola.x(),
                          self.punta.y() - self.cola.y())

    def es_utilizable(self) -> bool:
        """Una flecha demasiado corta no define ninguna direccion."""
        return self.activa and self.largo() >= LARGO_MINIMO

    def vector_unitario(self) -> Optional[Tuple[float, float]]:
        """`(dx, dy)` normalizado hacia ADENTRO. `None` si no es utilizable.

        `None` es una respuesta valida y hay que tratarla: significa «el
        operador no ha marcado la direccion», no «la direccion es cero».
        """
        largo = self.largo()
        if not self.activa or largo < LARGO_MINIMO:
            return None
        return ((self.punta.x() - self.cola.x()) / largo,
                (self.punta.y() - self.cola.y()) / largo)

    def coordenadas_pixel(self, ancho: int, alto: int) -> List[List[int]]:
        """Al tamano real del frame, que es lo que entiende el servidor."""
        if ancho <= 0 or alto <= 0:
            return []
        return [[int(p.x() * ancho / 1000), int(p.y() * alto / 1000)]
                for p in self.puntos]

    def invertir(self) -> None:
        """Intercambia entrada y salida.

        Es EL arreglo cuando las entradas se cuentan como salidas: el error
        tipico no es una flecha torcida, es una flecha apuntando al reves.
        Invertirla conserva la posicion exacta que ya estaba calibrada.
        """
        self.puntos.reverse()

    def trasladar(self, dx: int, dy: int) -> None:
        """Mueve la flecha ENTERA, recortando el paso en los bordes.

        Se recorta el DESPLAZAMIENTO y no cada punto por separado: recortar
        los puntos aplastaria la flecha contra el borde y cambiaria su
        direccion, que es justo lo que un arrastre de cuerpo promete no tocar.
        """
        xs = [p.x() for p in self.puntos]
        ys = [p.y() for p in self.puntos]
        dx = max(-min(xs), min(int(dx), 1000 - max(xs)))
        dy = max(-min(ys), min(int(dy), 1000 - max(ys)))
        self.puntos = [QPoint(p.x() + dx, p.y() + dy) for p in self.puntos]

    # ── Raton ────────────────────────────────────────────────────────────
    def indice_en(self, pos, escalar, radio: int) -> int:
        """Que agarre hay bajo el cursor. `escalar` mapea 0..1000 a pixeles.

        `INDICE_COLA`/`INDICE_PUNTA` para los extremos, `INDICE_MEDIO` para
        el rombo central o el cuerpo (los dos mueven la flecha entera), -1
        si no toca nada. Los extremos ganan al cuerpo: en el solape junto a
        un extremo, el operador casi siempre quiere reorientar, no mover.
        """
        if not self.activa:
            return -1
        for i, punto in enumerate(self.puntos):
            pix = escalar(punto)
            if math.hypot(pos.x() - pix.x(), pos.y() - pix.y()) < radio:
                return i
        cola, punta = escalar(self.cola), escalar(self.punta)
        if self._dist_a_segmento(pos, cola, punta) < max(radio, 8):
            return INDICE_MEDIO
        return -1

    @staticmethod
    def _dist_a_segmento(pos, a, b) -> float:
        ax, ay = a.x(), a.y()
        dx, dy = b.x() - ax, b.y() - ay
        largo2 = dx * dx + dy * dy
        if largo2 <= 0:
            return math.hypot(pos.x() - ax, pos.y() - ay)
        t = ((pos.x() - ax) * dx + (pos.y() - ay) * dy) / largo2
        t = max(0.0, min(1.0, t))
        return math.hypot(pos.x() - (ax + t * dx), pos.y() - (ay + t * dy))

    def mover(self, indice: int, punto: QPoint) -> None:
        if 0 <= indice < len(self.puntos):
            self.puntos[indice] = punto

    # ── Pintado ──────────────────────────────────────────────────────────
    def pintar(self, painter: QPainter, escalar, radio: int,
               indice_activo: int = -1) -> None:
        """Cuerpo, dos cabezas, rombo central y etiquetas. Nunca lanza."""
        if not self.activa:
            return
        try:
            cola, punta = escalar(self.cola), escalar(self.punta)
            corto = self.largo() < LARGO_MINIMO

            # Cuerpo sobre una sombra que lo despega del video. Punteado y
            # gris cuando es demasiado corta: se ve que esta dibujada pero
            # que NO va a decidir nada.
            estilo = Qt.DashLine if corto else Qt.SolidLine
            painter.setPen(QPen(_SOMBRA, 7, estilo, Qt.RoundCap))
            painter.drawLine(cola, punta)
            if corto:
                painter.setPen(QPen(_GRIS, 3, estilo, Qt.RoundCap))
            else:
                # Degradado rojo→verde: el sentido se lee en el propio
                # cuerpo aunque las cabezas queden tapadas por una persona.
                grad = QLinearGradient(QPointF(cola), QPointF(punta))
                grad.setColorAt(0.0, ROJO)
                grad.setColorAt(1.0, VERDE)
                painter.setPen(QPen(QBrush(grad), 4, estilo, Qt.RoundCap))
            painter.drawLine(cola, punta)

            if not corto:
                # Dos cabezas: verde ENTRANDO por la punta, roja SALIENDO
                # por la cola. Nacen un poco mas alla del circulo de agarre
                # para que ni ellas ni el circulo se tapen entre si.
                dist = math.hypot(punta.x() - cola.x(), punta.y() - cola.y())
                if dist > 0:
                    ux = (punta.x() - cola.x()) / dist
                    uy = (punta.y() - cola.y()) / dist
                    # El vertice se aleja lo bastante para que el triangulo
                    # ENTERO flote mas alla del circulo de agarre: pegado al
                    # extremo quedaba sepultado debajo y no se veia.
                    fuera = radio + 24
                    apex_verde = QPoint(int(punta.x() + ux * fuera),
                                        int(punta.y() + uy * fuera))
                    apex_rojo = QPoint(int(cola.x() - ux * fuera),
                                       int(cola.y() - uy * fuera))
                    # Con un extremo pegado al borde del visor el apex caeria
                    # fuera y QPainter recortaria la cabeza ENTERA: se mete
                    # dentro, como las etiquetas. La cabeza es justo la senal
                    # que dice si la flecha quedo al reves; no puede
                    # desaparecer en el caso mas comun (la puerta al borde
                    # del encuadre).
                    dispositivo = painter.device()
                    if dispositivo is not None:
                        for apex in (apex_verde, apex_rojo):
                            apex.setX(max(8, min(apex.x(),
                                                 dispositivo.width() - 8)))
                            apex.setY(max(8, min(apex.y(),
                                                 dispositivo.height() - 8)))
                    self._pintar_cabeza(painter, cola, apex_verde, VERDE)
                    self._pintar_cabeza(painter, punta, apex_rojo, ROJO)
                self._pintar_rombo(painter, escalar(self.punto_medio()),
                                   radio, indice_activo == INDICE_MEDIO)

            # Extremos agarrables.
            for i, (pix, relleno) in enumerate(((cola, ROJO), (punta, VERDE))):
                painter.setPen(QPen(Qt.black, 2))
                painter.setBrush(QBrush(Qt.white if i == indice_activo
                                        else relleno))
                painter.drawEllipse(pix, radio, radio)

            # Etiquetas al final, encima de todo. En gris cuando la flecha
            # es demasiado corta: unos carteles a todo color prometerian
            # una decision que no se esta tomando.
            self._pintar_etiqueta(painter, cola, 'SALIDA',
                                  _GRIS if corto else ROJO, radio)
            self._pintar_etiqueta(painter, punta, 'ENTRADA',
                                  _GRIS if corto else VERDE, radio)
        except Exception:                                   # noqa: BLE001
            # Un fallo pintando no puede tumbar el visor de video.
            pass

    @staticmethod
    def _pintar_cabeza(painter: QPainter, origen, apex,
                       color: QColor) -> None:
        """Triangulo con el vertice en `apex`, mirando desde `origen`."""
        ang = math.atan2(apex.y() - origen.y(), apex.x() - origen.x())
        largo, abre = 26.0, math.radians(30)
        alas = [
            QPoint(int(apex.x() - largo * math.cos(ang - abre)),
                   int(apex.y() - largo * math.sin(ang - abre))),
            QPoint(int(apex.x() - largo * math.cos(ang + abre)),
                   int(apex.y() - largo * math.sin(ang + abre))),
        ]
        painter.setPen(QPen(_SOMBRA, 1))
        painter.setBrush(QBrush(color))
        painter.drawPolygon(QPolygon([apex, alas[0], alas[1]]))

    @staticmethod
    def _pintar_rombo(painter: QPainter, centro, radio: int,
                      activo: bool) -> None:
        """El agarre de «mover entera»: distinto de los circulos a proposito."""
        lado = max(6, int(radio * 1.2))
        painter.save()
        painter.translate(centro)
        painter.rotate(45)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(Qt.white if activo
                                else QColor(210, 210, 210)))
        mitad = lado // 2
        painter.drawRect(-mitad, -mitad, lado, lado)
        painter.restore()

    @staticmethod
    def _pintar_etiqueta(painter: QPainter, pos, texto: str,
                         color: QColor, radio: int = 10) -> None:
        """Cartel solido con texto blanco: legible sobre cualquier video."""
        painter.save()
        fuente = QFont(painter.font())
        fuente.setFamilies(['Segoe UI', 'Arial'])
        fuente.setBold(True)
        fuente.setPointSize(max(10, fuente.pointSize()))
        painter.setFont(fuente)
        fm = painter.fontMetrics()
        ancho = fm.horizontalAdvance(texto) + 14
        alto = fm.height() + 4
        x = pos.x() - ancho // 2
        y = pos.y() + radio + 8
        # Dentro del visor: pegada a un borde, el cartel se mete, no se corta.
        dispositivo = painter.device()
        if dispositivo is not None:
            x = max(2, min(x, dispositivo.width() - ancho - 2))
            y = max(2, min(y, dispositivo.height() - alto - 2))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(_SOMBRA))
        painter.drawRoundedRect(x - 1, y - 1, ancho + 2, alto + 2, 4, 4)
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(x, y, ancho, alto, 4, 4)
        painter.setPen(QPen(Qt.white))
        painter.drawText(x + 7, y + 2 + fm.ascent(), texto)
        painter.restore()
