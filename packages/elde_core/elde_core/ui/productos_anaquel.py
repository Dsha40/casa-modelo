"""Productos dibujados DENTRO de las áreas de anaquel (25-sep-2026).

## Qué es

Un área de anaquel es una balda entera; un producto es un cuadrilátero de
cuatro esquinas arrastrables DENTRO de ella, con el nombre que escribe el
operador («Nescafé Gold»). El servidor reparte los agarres del área entre sus
productos según dónde entró la muñeca (`anaqueles_productos.py` del servidor).
Aquí solo se dibujan, se editan y se empaquetan para el payload.

Viven en el mismo espacio 0..1000 que las áreas y pertenecen a UNA por su
índice (`anaquel`), que es también como viajan: `{anaquel, nombre,
poligono}` en píxeles del frame, en el campo `anaqueles_productos` del
contrato.

## Identidad: la POSICIÓN dentro de su área

El servidor identifica cada producto por su posición entre los de su área
(el 1.º, el 2.º…), no por su nombre, para que corregir una errata no parta
los contadores. Por eso aquí solo se QUITA EL ÚLTIMO de cada área, igual que
con las áreas: quitar uno del medio renumeraría los de detrás y cada uno
heredaría los agarres del vecino.

## Por qué un fichero aparte y sin Qt arriba

`interactive_image_label.py` ya carga siete herramientas y lo usan varios
clientes. Esta clase guarda los puntos como listas de enteros (no `QPoint`)
e importa Qt SOLO al pintar: así dónde nace un producto, qué esquina se
arrastra y qué viaja al servidor se prueban sin PySide6, que el servidor no
tiene instalado.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Tope del nombre en el cliente. El contrato admite 120 y un campo
#: declarado que se pasa tira el frame ENTERO en modo estricto: el cliente
#: recorta mucho antes para no llegar nunca ahí.
MAX_NOMBRE = 60

#: Radio de las esquinas. Más chico que el de las áreas (10) a propósito:
#: los productos son estrechos, y con el mismo radio las esquinas de dos
#: marcas vecinas se pisan y no hay forma de coger la que uno quiere.
RADIO = 7

#: Los primeros productos, con colores elegidos a mano para que se distingan
#: bien entre sí (28-oct-2026). Ninguno verde: el verde es de las áreas.
_PALETA = (
    (190, 100, 255),   # violeta (el de siempre)
    (255, 210, 60),    # amarillo
    (255, 90, 170),    # rosa
    (60, 210, 255),    # cian
    (255, 140, 40),    # naranja
    (240, 240, 240),   # blanco
    (70, 110, 255),    # azul
    (255, 80, 80),     # rojo claro
    (200, 160, 110),   # marrón claro
    (230, 170, 255),   # lila
    (255, 175, 150),   # salmón
    (150, 190, 255),   # azul pálido
)


def color_de(indice: int) -> Tuple[int, int, int]:
    """Color RGB del producto nº `indice` (28-oct-2026): cada producto el
    suyo, y uno nuevo, uno distinto de los anteriores.

    Los 12 primeros, de `_PALETA`. Del 13 en adelante, el tono avanza con el
    ángulo áureo (137,5°), que no repite, más claro para no confundirse con
    los de la paleta. Se saltan los verdes (60-165°): son de las áreas de
    anaquel, y un producto verde no se distinguiría del área donde está.
    Como la identidad del producto es su posición, el color también: si se
    quita el último, los demás no cambian."""
    if 0 <= indice < len(_PALETA):
        return _PALETA[indice]
    import colorsys
    tono = (270.0 + indice * 137.508) % 360.0
    if 60.0 <= tono < 165.0:
        tono = (tono + 100.0) % 360.0
    r, g, b = colorsys.hsv_to_rgb(tono / 360.0, 0.4, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


#: Dónde nace el n-ésimo producto de un área, como fracción del ancho de la
#: balda: de izquierda a derecha, como van las marcas en el estante, y a
#: partir del quinto vuelve a empezar (se solapa y el operador lo arrastra).
_ANCHO_NACER = 0.18
_PASO_NACER = 0.23
_ALTO_NACER = (0.12, 0.88)


def _en_cuadrilatero(area: Sequence[Sequence[int]], u: float,
                     v: float) -> List[int]:
    """El punto (u, v) ∈ [0,1]² DENTRO del área, siguiendo su perspectiva.

    Con cuatro esquinas (las que crea `agregar_anaquel`: sup-izq, sup-der,
    inf-der, inf-izq) se interpola bilinealmente, así un producto nace
    inclinado igual que la balda. Con otra forma, dentro de su caja.
    """
    if len(area) == 4:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = [
            (float(p[0]), float(p[1])) for p in area]
        arriba = (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
        abajo = (x3 + (x2 - x3) * u, y3 + (y2 - y3) * u)
        x = arriba[0] + (abajo[0] - arriba[0]) * v
        y = arriba[1] + (abajo[1] - arriba[1]) * v
    else:
        xs = [float(p[0]) for p in area] or [0.0]
        ys = [float(p[1]) for p in area] or [0.0]
        x = min(xs) + (max(xs) - min(xs)) * u
        y = min(ys) + (max(ys) - min(ys)) * v
    return [max(0, min(1000, int(round(x)))),
            max(0, min(1000, int(round(y))))]


class ProductosAnaquel:
    """Los productos de UN recuadro (de todas sus áreas)."""

    def __init__(self, datos: Any = None) -> None:
        #: [{'anaquel': int, 'nombre': str, 'puntos': [[x, y] ×4]}], 0..1000.
        self.productos: List[Dict[str, Any]] = []
        self.poner(datos)

    # ── Datos ───────────────────────────────────────────────────────────
    @staticmethod
    def _limpio(item: Any) -> Optional[Dict[str, Any]]:
        """Un producto guardado, o `None` si no se puede leer.

        La configuración del cliente es un fichero que sobrevive a versiones:
        lo ilegible se descarta en vez de romper el recuadro al abrirlo.
        """
        try:
            anaquel = int(item.get('anaquel'))
            puntos = [[int(p[0]), int(p[1])]
                      for p in (item.get('puntos') or [])]
        except (AttributeError, TypeError, ValueError, IndexError):
            return None
        if anaquel < 0 or len(puntos) < 3:
            return None
        return {'anaquel': anaquel,
                'nombre': str(item.get('nombre') or '').strip()[:MAX_NOMBRE],
                'puntos': puntos}

    def poner(self, datos: Any) -> None:
        self.productos = [p for p in (self._limpio(i) for i in (datos or []))
                          if p is not None]

    def a_lista(self) -> List[Dict[str, Any]]:
        """Copia para persistir (la config del recuadro)."""
        return [{'anaquel': p['anaquel'], 'nombre': p['nombre'],
                 'puntos': [list(pt) for pt in p['puntos']]}
                for p in self.productos]

    def de_anaquel(self, anaquel: int) -> List[int]:
        """Índices (en `self.productos`) de los productos de un área, en su
        orden: el orden ES la identidad que ve el servidor."""
        return [i for i, p in enumerate(self.productos)
                if p['anaquel'] == int(anaquel)]

    def posicion(self, indice: int) -> int:
        """Posición del producto entre los de su área (0, 1, 2…)."""
        return self.de_anaquel(self.productos[indice]['anaquel']).index(
            indice)

    def nombre_visible(self, indice: int) -> str:
        """El nombre, o el mismo «Producto N» que pondrá el servidor."""
        return (self.productos[indice]['nombre']
                or f'Producto {self.posicion(indice) + 1}')

    # ── Edición ─────────────────────────────────────────────────────────
    def agregar(self, anaquel: int, area: Sequence[Sequence[int]],
                nombre: str = '') -> int:
        """Añade un producto que nace DENTRO del área. Devuelve su índice."""
        n = len(self.de_anaquel(anaquel))
        u0 = 0.05 + (n % 4) * _PASO_NACER
        u1 = u0 + _ANCHO_NACER
        v0, v1 = _ALTO_NACER
        puntos = [_en_cuadrilatero(area, u, v)
                  for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
        self.productos.append({'anaquel': int(anaquel),
                               'nombre': str(nombre or '').strip()[:MAX_NOMBRE],
                               'puntos': puntos})
        return len(self.productos) - 1

    def quitar_ultimo(self, anaquel: int) -> bool:
        """Quita el ÚLTIMO producto del área (ver «Identidad» arriba)."""
        indices = self.de_anaquel(anaquel)
        if not indices:
            return False
        del self.productos[indices[-1]]
        return True

    def quitar_de_anaquel(self, anaquel: int) -> bool:
        """Al quitar un área se van sus productos: sin ella quedarían
        huérfanos, pintados sobre nada y sin contar nada."""
        antes = len(self.productos)
        self.productos = [p for p in self.productos
                          if p['anaquel'] != int(anaquel)]
        return len(self.productos) != antes

    def renombrar(self, indice: int, nombre: str) -> bool:
        if not 0 <= indice < len(self.productos):
            return False
        self.productos[indice]['nombre'] = \
            str(nombre or '').strip()[:MAX_NOMBRE]
        return True

    # ── Ratón ───────────────────────────────────────────────────────────
    def vertice_en(self, pos: Any, escalar: Callable[[int, int], Any],
                   radio: int = RADIO) -> Optional[Tuple[int, int]]:
        """`(producto, esquina)` bajo `pos` (píxeles del visor), o `None`.

        Se recorre del ÚLTIMO al primero: el último se pinta encima, así que
        es el que el operador ve y el que espera coger. `escalar(x, y)`
        pasa de 0..1000 a píxeles del visor y devuelve algo con `.x()` e
        `.y()` (un `QPoint` en el recuadro).
        """
        for i in range(len(self.productos) - 1, -1, -1):
            for j, (x, y) in enumerate(self.productos[i]['puntos']):
                p = escalar(x, y)
                if math.hypot(pos.x() - p.x(), pos.y() - p.y()) < radio:
                    return i, j
        return None

    def mover(self, indice: int, esquina: int, x: int, y: int) -> None:
        try:
            self.productos[indice]['puntos'][esquina] = [
                max(0, min(1000, int(x))), max(0, min(1000, int(y)))]
        except (IndexError, TypeError, ValueError):
            pass

    # ── Hacia el servidor ───────────────────────────────────────────────
    def coordenadas_pixel(self, ancho: int, alto: int,
                          n_anaqueles: int) -> List[Dict[str, Any]]:
        """Lo que viaja en `anaqueles_productos`, en píxeles del frame.

        Solo los productos de áreas que existen: uno huérfano el servidor lo
        ignoraría igual, pero no hay por qué mandarlo en cada frame.
        """
        if ancho <= 0 or alto <= 0:
            return []
        salida = []
        for p in self.productos:
            if not 0 <= p['anaquel'] < int(n_anaqueles):
                continue
            salida.append({
                'anaquel': p['anaquel'],
                'nombre': p['nombre'],
                'poligono': [[int(x * ancho / 1000), int(y * alto / 1000)]
                             for x, y in p['puntos']],
            })
        return salida

    # ── Dibujo ──────────────────────────────────────────────────────────
    def pintar(self, painter: Any, escalar: Callable[[int, int], Any],
               radio: float = RADIO,
               activo: Optional[Tuple[int, int]] = None) -> None:
        """Contorno violeta, esquinas y nombre de cada producto.

        El nombre va DENTRO del producto, junto a su esquina superior
        izquierda: encima de ella está el rótulo del área, y los dos
        rótulos se pisarían.
        """
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QBrush, QColor, QPen

        for i, p in enumerate(self.productos):
            color = QColor(*color_de(i))
            pix = [escalar(x, y) for x, y in p['puntos']]
            if len(pix) < 2:
                continue
            painter.setPen(QPen(color, 2, Qt.SolidLine))
            for a, b in zip(pix, pix[1:] + pix[:1]):
                painter.drawLine(a, b)
            for j, q in enumerate(pix):
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(QBrush(Qt.white if activo == (i, j)
                                        else color))
                # QPointF: el radio puede ser decimal (2,5 px, 28-oct-2026).
                painter.drawEllipse(QPointF(q), radio, radio)
            etiqueta = self.nombre_visible(i)
            origen = pix[0]
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.black, 4))
            painter.drawText(origen.x() + 9, origen.y() + 17, etiqueta)
            painter.setPen(QPen(color, 2))
            painter.drawText(origen.x() + 8, origen.y() + 16, etiqueta)
