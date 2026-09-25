"""src/analityc/core/anaqueles_productos.py — productos DENTRO de un área.

## Qué resuelve (25-sep-2026)

Un área de anaquel es una balda entera, y en una balda hay varias marcas una
al lado de otra: en la cámara 3 del pasillo de café, el Anaquel 2 lleva
Nescafé Gold, Nescafé, Colcafé… (visto en `fotos_ref/1.jpg`). El conteo por
área dice «agarraron del Anaquel 2»; marketing pregunta «¿de cuál?».

El medidor YA sabía en qué píxel entró la muñeca al confirmar un agarre
(`pend['x'], pend['y']` en `_medir_agarres`) y solo lo usaba para el mapa de
calor. Aquí se cruza ese punto con los productos que el operador dibujó
dentro del área. Nada más: el agarre sigue decidiéndolo la máquina del gesto,
que no cambia. Un producto no añade agarres, reparte los que ya había.

## Identidad de un producto

Es su POSICIÓN entre los productos de su área (el 0.º, el 1.º… de ese
anaquel), no su nombre: el mismo criterio que las áreas, cuya identidad es su
índice. Así, corregir una errata en el nombre no parte en dos los contadores
del día. La posición se asigna ANTES de descartar los mal formados, por la
misma razón que `poner_zonas` empareja nombres antes de filtrar: uno roto no
puede correr la identidad de los que vienen detrás.

## Lo que NO resuelve (sin medir)

La muñeca queda unos centímetros DETRÁS de los dedos. En productos estrechos
un agarre podría caer en el vecino. La pose ya da el codo (keypoints 7 y 8)
en la misma pasada y con él se podría estimar la mano, pero eso hay que
MEDIRLO con agarres reales antes de meterlo.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: (posición dentro del área, nombre, polígono en píxeles del frame).
Producto = Tuple[int, str, np.ndarray]


def _campo(item: Any, clave: str, defecto: Any = None) -> Any:
    """Lee `clave` de un dict (lo normal: el payload crudo) o de un objeto
    con atributos (el modelo del contrato, si alguien lo pasa validado)."""
    if isinstance(item, dict):
        return item.get(clave, defecto)
    return getattr(item, clave, defecto)


def nombre_por_defecto(posicion: int) -> str:
    return f'Producto {int(posicion) + 1}'


def firma(crudo: Optional[Sequence[Any]], n_areas: int) -> Any:
    """Huella de lo recibido, para no reconstruir en cada frame.

    Lleva `n_areas` porque un producto se vuelve válido o huérfano cuando
    cambian las áreas aunque él no se haya movido. Nunca lanza: un item
    ilegible entra en la huella como su `repr`, y ya lo descartará
    `productos_por_area`.
    """
    partes = []
    for item in (crudo or []):
        try:
            puntos = tuple((int(p[0]), int(p[1]))
                           for p in (_campo(item, 'poligono') or []))
            partes.append((_campo(item, 'anaquel'),
                           str(_campo(item, 'nombre', '') or ''), puntos))
        except (TypeError, ValueError, IndexError, KeyError):
            partes.append(repr(item))
    return (int(n_areas), tuple(partes))


def productos_por_area(crudo: Optional[Sequence[Any]],
                       n_areas: int) -> Dict[int, List[Producto]]:
    """Agrupa los productos del payload por el área a la que pertenecen.

    Se descartan, sin lanzar, los que apuntan a un área que no existe en
    este frame y los de polígono inservible (menos de 3 puntos o puntos que
    no son `[x, y]`). El contrato ya filtra lo peor, pero `app.py` lee el
    payload CRUDO y un fallo aquí no puede tumbar la medición del área.
    """
    salida: Dict[int, List[Producto]] = {}
    posiciones: Dict[int, int] = {}
    for item in (crudo or []):
        try:
            area = int(_campo(item, 'anaquel'))
        except (TypeError, ValueError):
            continue
        # La posición se reserva ANTES de validar el polígono: ver la
        # cabecera («Identidad de un producto»).
        posicion = posiciones.get(area, 0)
        posiciones[area] = posicion + 1
        if not 0 <= area < int(n_areas):
            continue
        try:
            puntos = [[float(p[0]), float(p[1])]
                      for p in (_campo(item, 'poligono') or [])
                      if len(p) == 2]
        except (TypeError, ValueError):
            continue
        if len(puntos) < 3:
            continue
        nombre = (str(_campo(item, 'nombre', '') or '').strip()
                  or nombre_por_defecto(posicion))
        salida.setdefault(area, []).append(
            (posicion, nombre, np.array(puntos, dtype=np.float32)))
    return salida


def producto_en(productos: Sequence[Producto], x: float,
                y: float) -> Optional[Tuple[int, str]]:
    """`(posición, nombre)` del producto bajo el punto, o `None`.

    El borde cuenta como dentro: dos productos pegados comparten lado y el
    punto que cae justo encima no puede quedarse sin dueño. Si el operador
    dibujó dos productos solapados, gana aquel en el que el punto está MÁS
    ADENTRO (la misma regla que ya usa el medidor para decidir el área de
    una muñeca); a igualdad, el primero de la lista.
    """
    mejor: Optional[Tuple[int, str]] = None
    mejor_prof = -1.0
    for posicion, nombre, poligono in productos:
        prof = float(cv2.pointPolygonTest(poligono, (float(x), float(y)),
                                          True))
        if prof >= 0.0 and prof > mejor_prof:
            mejor, mejor_prof = (posicion, nombre), prof
    return mejor
