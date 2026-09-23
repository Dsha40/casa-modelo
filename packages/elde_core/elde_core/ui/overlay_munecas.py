"""Las muñecas del modo Anaqueles, pintadas sobre el vídeo en vivo.

## Para qué

Calibrar un área de anaquel es adivinar hasta que ves DÓNDE cree el servidor
que está la mano. El bloque `metadata['anaqueles']['munecas']` trae, por
fotograma, cada muñeca con su profundidad respecto al área (positiva =
dentro). Esto lo dibuja: un punto por muñeca, verde si el servidor la cuenta
como dentro y ámbar si no, con los píxeles de margen al lado.

Con eso el operador ve, sin tocar nada, por qué un agarre no cuenta: la mano
sale ámbar, o el punto ni aparece porque la persona no entró en compuerta.

## Por qué caducan

Son un dibujo EN VIVO. El bloque solo trae muñecas cuando las hay, así que
sin caducidad el último punto se quedaría clavado en pantalla después de que
la persona se fuera — y un punto viejo miente más que un hueco. Pasado
`TTL_S` no se pinta nada.

## Por qué dos funciones

El recuadro tiene dos caminos de pintado: Supervision (numpy BGR, el normal)
y QPainter (el respaldo cuando Supervision no está). Las muñecas tienen que
salir en los dos o el operador vería un comportamiento distinto según qué
tenga instalado.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: Segundos que un punto sigue siendo creíble. Corto a propósito: el
#: pipeline va a 2-3 fps, así que medio segundo son 1-2 fotogramas.
TTL_S = 0.7

#: BGR para cv2. Verde = el servidor la cuenta dentro del área; ámbar = no.
_VERDE_BGR = (80, 220, 80)
_AMBAR_BGR = (40, 160, 255)
#: RGB para Qt (los mismos colores, al revés).
_VERDE_RGB = (80, 220, 80)
_AMBAR_RGB = (255, 160, 40)

_RADIO = 9

#: Separacion vertical entre rotulos. Las dos munecas de una persona que
#: alcanza un estante caen casi en el mismo pixel, y sin esto sus etiquetas
#: se superponen en un borron ilegible (visto sobre el video real).
_SALTO_ROTULO = 22


def vigentes(munecas: Optional[Sequence[Dict[str, Any]]],
             ts: Optional[float], ahora: float) -> List[Dict[str, Any]]:
    """Las muñecas que todavía se pueden pintar, o lista vacía."""
    if not munecas or ts is None:
        return []
    if (ahora - float(ts)) > TTL_S:
        return []
    return [m for m in munecas if isinstance(m, dict)
            and m.get('x') is not None and m.get('y') is not None]


def _etiqueta(m: Dict[str, Any]) -> str:
    prof = m.get('prof')
    lado = str(m.get('lado') or '?')
    if prof is None:
        return lado
    return f'{lado} {float(prof):+.0f}px'


def pintar_cv2(frame_bgr, munecas: Sequence[Dict[str, Any]]):
    """Dibuja sobre un frame BGR de numpy. Devuelve el mismo frame."""
    try:
        import cv2
    except Exception:                                       # noqa: BLE001
        return frame_bgr
    for n, m in enumerate(munecas or []):
        x, y = int(float(m['x'])), int(float(m['y']))
        color = _VERDE_BGR if m.get('dentro') else _AMBAR_BGR
        y_texto = y - 10 - n * _SALTO_ROTULO
        # Aro negro debajo: sobre una caja de cereales de color vivo, un
        # punto de un solo color se pierde.
        cv2.circle(frame_bgr, (x, y), _RADIO + 3, (0, 0, 0), -1)
        cv2.circle(frame_bgr, (x, y), _RADIO, color, -1)
        texto = _etiqueta(m)
        cv2.putText(frame_bgr, texto, (x + 14, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(frame_bgr, texto, (x + 14, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame_bgr


def pintar_qt(painter, munecas: Sequence[Dict[str, Any]]) -> None:
    """Dibuja con un QPainter ya abierto sobre el pixmap del frame."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QBrush, QColor, QPen
    except Exception:                                       # noqa: BLE001
        return
    for n, m in enumerate(munecas or []):
        x, y = int(float(m['x'])), int(float(m['y']))
        r, g, b = _VERDE_RGB if m.get('dentro') else _AMBAR_RGB
        y_texto = y - 9 - n * _SALTO_ROTULO
        color = QColor(r, g, b)
        painter.setPen(QPen(Qt.black, 3))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(x - _RADIO, y - _RADIO, _RADIO * 2, _RADIO * 2)
        texto = _etiqueta(m)
        painter.setPen(QPen(Qt.black, 4))
        painter.drawText(x + 15, y_texto, texto)
        painter.setPen(QPen(color, 1))
        painter.drawText(x + 14, y_texto - 1, texto)
