"""Abrir la imagen de la respuesta ya reducida al tamaño del recuadro (28-oct-2026).

## Por qué

En modo directo el cliente pinta las cajas del servidor sobre la imagen que
envió. Hasta el 28-oct la abría a tamaño completo (1456x868 con el vídeo),
pintaba encima y la encogía al recuadro: dos pasadas por todos los píxeles,
en el hilo de la ventana, con cada respuesta. Con `QImageReader.setScaledSize`
libjpeg la descomprime ya reducida (a 1/2, 5/8...: se salta parte del
trabajo) y Qt hace el ajuste fino, suavizado.

Medido en el PC de la tienda con una imagen sintética del mismo tamaño y
calidad que las del vídeo (316 KB), abrir + pintar + encoger:

    recuadro   antes    ahora
    700x420    20,5 ms  11,1 ms
    900x520    21,5 ms  13,0 ms
    1300x780   24,3 ms  24,6 ms   (casi la imagen entera: no hay qué ahorrar)

libjpeg solo reduce de verdad cuando el recuadro es menor que ~7/8 de la
imagen; con el recuadro maximizado todo queda como antes.

## Lo que no cambia

Quien pinta lo hace en píxeles de la imagen ENVIADA, con el pintor escalado
(ver `render_box._draw_detections_and_show`): cajas, muñecas y rótulos salen
donde y como antes. El ancho y el alto REALES (los de la cabecera del JPEG)
se devuelven aparte: son los que usan las zonas, la regla y el rótulo del
tamaño.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize
from PySide6.QtGui import QImage, QImageReader


def abrir(jpeg: bytes, ancho: int,
          alto: int) -> Tuple[Optional[QImage], int, int]:
    """(imagen, ancho real, alto real) del JPEG.

    La imagen sale ya a `ancho` x `alto` (sin guardar la proporción, como la
    escala el recuadro) si los dos son menores que los de la foto; si no, o
    si el recuadro aún no tiene tamaño, sale entera, como antes, y la escala
    quien la muestra. (None, 0, 0) si no se puede abrir."""
    buf = QBuffer()
    buf.setData(QByteArray(bytes(jpeg)))
    if not buf.open(QIODevice.ReadOnly):
        return None, 0, 0
    lector = QImageReader(buf, b'JPEG')
    real = lector.size()
    if not real.isValid():
        return None, 0, 0
    if 0 < ancho < real.width() and 0 < alto < real.height():
        lector.setScaledSize(QSize(ancho, alto))
    imagen = lector.read()
    if imagen.isNull():
        return None, 0, 0
    return imagen, real.width(), real.height()
