"""src/analityc/core/anaqueles_registro.py — persistencia del modo Anaqueles.

## Por qué existe

El procesador RENACE en cada reconexión del cliente (~5 s tras cualquier
corte; ver `app.py`): todo contador en RAM vuelve a cero. Las cifras de
marketing («cuántas veces agarraron del nivel 2 hoy») no pueden depender de
que la red no parpadee, así que visitas, agarres y el mapa de calor de cada
área viven aquí, en disco, con jornada DIARIA reanudable — el mismo motivo
por el que el stand tiene `sesion_abierta_de`.

## Formato

Un JSON por cámara y día: `server/output/anaqueles/<camara>/<AAAA-MM-DD>.json`.
JSON y no SQLite porque el volumen es minúsculo (decenas de áreas, una rejilla
chica por área) y el patrón atómico tmp+`os.replace` de
`permanencia_estancias` ya demostró que aguanta cortes a mitad de escritura.
La escritura va AMORTIGUADA (mínimo cada `PERIODO_VOLCADO_S` y solo si hay
cambios): esto corre dentro del bucle de inferencia.

Los totales son ABSOLUTOS: el cliente y el dashboard leen y se ponen al día
de golpe tras reconectar, sin protocolo de resincronización (regla del
emisor del stand).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Mínimo entre escrituras a disco. 5 s pierde como mucho 5 s de conteo en un
#: corte duro, y no convierte el disco en el cuello del bucle.
PERIODO_VOLCADO_S = 5.0

#: Dimensión por defecto de la rejilla del heatmap POR ÁREA (filas, columnas).
#: Un anaquel es apaisado: más columnas que filas.
GRID_DIM = (12, 24)

#: En Windows los `:` de un camera_id romperían la ruta (el envelope los
#: permite en el id). Todo lo que no sea seguro para un nombre de archivo se
#: sustituye por `_`.
_SEGURO = re.compile(r'[^A-Za-z0-9_.-]+')


def _hoy() -> str:
    return time.strftime('%Y-%m-%d')


def raiz_salida() -> Path:
    """`server/output/anaqueles/`. Junto al resto de salidas del servidor."""
    return Path(__file__).resolve().parents[3] / 'output' / 'anaqueles'


def _carpeta_nombre(camera_id: Any) -> str:
    """Nombre de carpeta SEGURO y sin colisiones para un camera_id.

    Un id ya limpio se usa tal cual (así `camaras_registradas()` devuelve
    nombres que la API puede recibir de vuelta y resolver a la MISMA
    carpeta). Uno con caracteres sustituidos o solo-puntos (p. ej. `..`, que
    escaparía de la raíz) se desambigua con un hash corto del original: dos
    ids distintos nunca comparten carpeta, y el traversal muere aquí.
    """
    crudo = str(camera_id)
    slug = (_SEGURO.sub('_', crudo)).strip('.')
    if not slug or slug != crudo:
        slug = (slug or 'cam') + '_' + \
            hashlib.sha1(crudo.encode('utf-8')).hexdigest()[:8]
    return slug


class RegistroAnaqueles:
    """Contadores y heatmaps por área de UNA cámara, reanudables por día."""

    def __init__(self, camera_id: Any,
                 raiz: Optional[Path] = None,
                 grid_dim: tuple = GRID_DIM) -> None:
        self.camera_id = str(camera_id)
        self.grid_dim = (int(grid_dim[0]), int(grid_dim[1]))
        self._carpeta = (raiz or raiz_salida()) / _carpeta_nombre(camera_id)
        self._ruta = self._carpeta / f'{_hoy()}.json'
        self._fecha = _hoy()
        #: idx (str) -> {'nombre', 'visitas', 'agarres', 'por_genero',
        #: 'por_edad', '_grid' (np.ndarray), 'productos' (posición (str) ->
        #: {'nombre', 'agarres'}), 'agarres_sin_producto'}
        self.areas: Dict[str, Dict[str, Any]] = {}
        self._sucio = False
        self._ultimo_volcado = 0.0
        self._cargar()

    # ── Acceso ──────────────────────────────────────────────────────────
    def area(self, idx: int, nombre: str = '') -> Dict[str, Any]:
        """El registro del área `idx`; lo crea si es nueva.

        El nombre se refresca siempre: si el operador renombra el área a
        mitad de jornada, los totales se conservan (la identidad es el
        ÍNDICE) y el reporte sale con el nombre nuevo.
        """
        clave = str(int(idx))
        a = self.areas.get(clave)
        if a is None:
            a = {
                'nombre': nombre or f'Área {int(idx) + 1}',
                'visitas': 0,
                'agarres': 0,
                'por_genero': {},
                'por_edad': {},
                '_grid': np.zeros(self.grid_dim, dtype=np.float32),
                'productos': {},
                'agarres_sin_producto': 0,
            }
            self.areas[clave] = a
            self._sucio = True
        elif nombre and a.get('nombre') != nombre:
            a['nombre'] = nombre
            self._sucio = True
        return a

    def sumar(self, idx: int, campo: str, nombre: str = '') -> int:
        """`visitas`/`agarres` += 1. Devuelve el total ya incrementado."""
        a = self.area(idx, nombre)
        a[campo] = int(a.get(campo) or 0) + 1
        self._sucio = True
        return a[campo]

    # ── Productos dentro del área (25-sep-2026) ─────────────────────────
    def producto(self, idx: int, posicion: int,
                 nombre: str = '') -> Dict[str, Any]:
        """El registro del producto `posicion` del área `idx`; lo crea.

        Mismo trato que `area()`: la identidad es la POSICIÓN y el nombre se
        refresca, así que corregir una errata a media jornada conserva lo
        contado. Se crea al DIBUJARLO, no al primer agarre, para que la API
        distinga «dibujado y nadie lo tocó» de «no está dibujado».
        """
        a = self.area(idx)
        clave = str(int(posicion))
        p = a['productos'].get(clave)
        if p is None:
            p = {'nombre': nombre or f'Producto {int(posicion) + 1}',
                 'agarres': 0}
            a['productos'][clave] = p
            self._sucio = True
        elif nombre and p.get('nombre') != nombre:
            p['nombre'] = nombre
            self._sucio = True
        return p

    def sumar_producto(self, idx: int, posicion: int,
                       nombre: str = '') -> int:
        """Un agarre más para ese producto. Devuelve su total."""
        p = self.producto(idx, posicion, nombre)
        p['agarres'] = int(p.get('agarres') or 0) + 1
        self._sucio = True
        return p['agarres']

    def sumar_sin_producto(self, idx: int) -> int:
        """Un agarre del área que no cayó en ningún producto dibujado.

        Es la cifra que dice si los productos están BIEN dibujados: si crece
        a la par que los del área, las manos entran por donde no hay ningún
        producto marcado. Solo se cuenta en áreas que tienen productos; en
        las demás sería simplemente el total del área repetido.
        """
        a = self.area(idx)
        a['agarres_sin_producto'] = int(a.get('agarres_sin_producto')
                                        or 0) + 1
        self._sucio = True
        return a['agarres_sin_producto']

    def sumar_segmento(self, idx: int, genero: Optional[str],
                       rango_edad: Optional[str]) -> None:
        """Acumula la demografía de una visita (para el reporte de marketing)."""
        a = self.area(idx)
        if genero:
            a['por_genero'][str(genero)] = \
                int(a['por_genero'].get(str(genero)) or 0) + 1
        if rango_edad:
            a['por_edad'][str(rango_edad)] = \
                int(a['por_edad'].get(str(rango_edad)) or 0) + 1
        self._sucio = True

    def calor(self, idx: int, fila: int, col: int, peso: float = 1.0) -> None:
        """Suma `peso` a la celda del heatmap del área."""
        a = self.area(idx)
        g: np.ndarray = a['_grid']
        f = max(0, min(self.grid_dim[0] - 1, int(fila)))
        c = max(0, min(self.grid_dim[1] - 1, int(col)))
        g[f, c] += float(peso)
        self._sucio = True

    def resumen(self) -> List[Dict[str, Any]]:
        """Totales por área, SIN la rejilla (para el bloque de 2 s)."""
        salida = []
        for clave in sorted(self.areas, key=int):
            a = self.areas[clave]
            salida.append({
                'idx': int(clave),
                'nombre': a.get('nombre') or f'Área {int(clave) + 1}',
                'visitas': int(a.get('visitas') or 0),
                'agarres': int(a.get('agarres') or 0),
            })
        return salida

    def heatmaps(self) -> Dict[str, Dict[str, Any]]:
        """Las rejillas serializadas (para el bloque de 30 s y la API)."""
        return {
            clave: {
                'nombre': a.get('nombre'),
                'dim': list(self.grid_dim),
                'grid': [[round(float(v), 2) for v in fila]
                         for fila in a['_grid']],
            }
            for clave, a in self.areas.items()
        }

    def demografia(self) -> Dict[str, Dict[str, Any]]:
        return {
            clave: {'nombre': a.get('nombre'),
                    'por_genero': dict(a.get('por_genero') or {}),
                    'por_edad': dict(a.get('por_edad') or {})}
            for clave, a in self.areas.items()
        }

    @property
    def carpeta(self) -> Path:
        """Carpeta de esta cámara (la usa `anaqueles_stock` para sus fotos
        de referencia, 5-oct-2026)."""
        return self._carpeta

    # ── Foto de referencia del área (14-ago) ────────────────────────────
    def guardar_ref(self, idx: int, imagen_bgr) -> None:
        """Foto del área VACÍA (el recorte exacto del bbox del polígono):
        es la base sobre la que la API compone el mapa de calor — se ve el
        estante y los productos REALES debajo del calor. Nunca lanza."""
        try:
            import cv2
            carpeta = self._carpeta / 'fotos_ref'
            carpeta.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode('.jpg', imagen_bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return
            ruta = carpeta / f'{int(idx)}.jpg'
            tmp = carpeta / f'{int(idx)}.tmp.jpg'
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, ruta)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("anaqueles: ref del área %s falló: %s", idx, exc)

    # ── Disco ───────────────────────────────────────────────────────────
    def rollover_si_cambio_dia(self) -> bool:
        """Si cruzamos la medianoche con la conexión viva, cierra el archivo
        de ayer y abre el de hoy. True si hubo cambio de día.

        Sin esto (lo cazó la revisión del 13-ago): un cliente conectado de
        23:50 a 08:00 seguía sumando los eventos de la mañana al JSON de
        AYER (histórico inflado) mientras la API buscaba el de HOY y daba
        404 aunque el pipeline estuviera contando.
        """
        hoy = _hoy()
        if hoy == self._fecha:
            return False
        try:
            self.volcar(force=True)     # lo pendiente pertenece al día viejo
        except Exception:               # noqa: BLE001
            pass
        self._fecha = hoy
        self._ruta = self._carpeta / f'{hoy}.json'
        self.areas = {}
        self._sucio = False
        self._ultimo_volcado = 0.0
        logger.info("anaqueles[%s]: cambio de día — jornada nueva %s",
                    self.camera_id, hoy)
        return True

    def volcar(self, force: bool = False) -> None:
        """Escribe si hay cambios y pasó el periodo (o `force`). Nunca lanza."""
        ahora = time.time()
        if not self._sucio:
            return
        if not force and (ahora - self._ultimo_volcado) < PERIODO_VOLCADO_S:
            return
        try:
            self._ruta.parent.mkdir(parents=True, exist_ok=True)
            datos = {
                'fecha': self._fecha,
                'camera_id': self.camera_id,
                'grid_dim': list(self.grid_dim),
                'areas': {
                    clave: {
                        'nombre': a.get('nombre'),
                        'visitas': int(a.get('visitas') or 0),
                        'agarres': int(a.get('agarres') or 0),
                        'por_genero': dict(a.get('por_genero') or {}),
                        'por_edad': dict(a.get('por_edad') or {}),
                        'grid': [[round(float(v), 3) for v in fila]
                                 for fila in a['_grid']],
                        'productos': {
                            k: {'nombre': p.get('nombre'),
                                'agarres': int(p.get('agarres') or 0)}
                            for k, p in (a.get('productos') or {}).items()},
                        'agarres_sin_producto': int(
                            a.get('agarres_sin_producto') or 0),
                    }
                    for clave, a in self.areas.items()
                },
            }
            tmp = self._ruta.with_suffix('.tmp')
            tmp.write_text(json.dumps(datos, ensure_ascii=False),
                           encoding='utf-8')
            os.replace(tmp, self._ruta)   # atómico: nunca queda a medias
            self._sucio = False
            self._ultimo_volcado = ahora
        except Exception as exc:                            # noqa: BLE001
            logger.error("anaqueles: no se pudo persistir %s: %s",
                         self._ruta, exc)

    def _cargar(self) -> None:
        """Reanuda la jornada de HOY si el archivo existe. Un archivo de otro
        día no se toca: cada fecha tiene el suyo y el histórico se conserva."""
        try:
            if not self._ruta.is_file():
                return
            datos = json.loads(self._ruta.read_text(encoding='utf-8'))
            if datos.get('fecha') != self._fecha:
                return
            dim = datos.get('grid_dim') or list(self.grid_dim)
            for clave, a in (datos.get('areas') or {}).items():
                grid = np.array(a.get('grid') or [], dtype=np.float32)
                if grid.shape != tuple(self.grid_dim):
                    g = np.zeros(self.grid_dim, dtype=np.float32)
                    # Si la dimensión coincide con la guardada, se respeta lo
                    # acumulado; si alguien cambió la rejilla a media jornada,
                    # se empieza limpia (mezclar rejillas de tamaños distintos
                    # inventaría posiciones).
                    if grid.size and list(grid.shape) == dim \
                            and tuple(dim) == self.grid_dim:
                        g = grid
                    grid = g
                self.areas[str(clave)] = {
                    'nombre': a.get('nombre') or f'Área {int(clave) + 1}',
                    'visitas': int(a.get('visitas') or 0),
                    'agarres': int(a.get('agarres') or 0),
                    'por_genero': dict(a.get('por_genero') or {}),
                    'por_edad': dict(a.get('por_edad') or {}),
                    '_grid': grid,
                    # Los JSON anteriores al 25-sep no traen productos: se
                    # reanudan vacíos, sin inventar nada. El nombre NO se
                    # deriva de la clave aquí: un `int()` que fallara lo
                    # atraparía el `except` de abajo y se perdería la
                    # reanudación del día ENTERO por un producto.
                    'productos': {
                        str(k): {'nombre': p.get('nombre') or '',
                                 'agarres': int(p.get('agarres') or 0)}
                        for k, p in (a.get('productos') or {}).items()},
                    'agarres_sin_producto': int(
                        a.get('agarres_sin_producto') or 0),
                }
            if self.areas:
                logger.info(
                    "anaqueles[%s]: jornada de hoy reanudada (%d áreas, "
                    "%d visitas, %d agarres)", self.camera_id,
                    len(self.areas),
                    sum(int(a.get('visitas') or 0)
                        for a in self.areas.values()),
                    sum(int(a.get('agarres') or 0)
                        for a in self.areas.values()))
        except Exception as exc:                            # noqa: BLE001
            logger.error("anaqueles: no se pudo reanudar %s: %s",
                         self._ruta, exc)


def ultimo_registro(camera_id: Any,
                    fecha: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Lee el JSON de una cámara y fecha (hoy por defecto), para la API.

    Lectura directa de disco: la API corre en otro contexto que el pipeline y
    no debe tocar sus objetos vivos.
    """
    try:
        carpeta = raiz_salida() / _carpeta_nombre(camera_id)
        ruta = carpeta / f'{fecha or _hoy()}.json'
        if not ruta.is_file():
            return None
        return json.loads(ruta.read_text(encoding='utf-8'))
    except Exception as exc:                                # noqa: BLE001
        logger.error("anaqueles: no se pudo leer %s/%s: %s",
                     camera_id, fecha, exc)
        return None


def ref_ruta(camera_id: Any, idx: int) -> Optional[Path]:
    """La foto de referencia de un área, si existe (para la API)."""
    try:
        ruta = (raiz_salida() / _carpeta_nombre(camera_id) / 'fotos_ref'
                / f'{int(idx)}.jpg')
        return ruta if ruta.is_file() else None
    except Exception:                                       # noqa: BLE001
        return None


def camaras_registradas() -> List[str]:
    """Carpetas de cámara que tienen al menos un día persistido."""
    try:
        base = raiz_salida()
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())
    except Exception:                                       # noqa: BLE001
        return []
