"""Mapa de calor del stand: tres capas con filtros (HITO 9).

## Por qué tres capas y no una

- **`trafico`** — dónde pasa la gente. Una muestra por frame y persona.
- **`permanencia`** — ponderada por tiempo. Es la que revela dónde la gente
  *se queda*, que no es lo mismo que por dónde camina: un pasillo estrecho
  concentra tráfico sin que nadie se detenga.
- **`detenciones`** — solo los puntos donde se disparó el evento `detencion`.
  Son los anclajes reales, sin la cola de la gente que va llegando.

La comparación entre capas es el diagnóstico: mucho tráfico y poca permanencia
frente al stand significa que la gente pasa por delante sin mirarlo.

## Por qué se acumula por SEGMENTO y no se guardan las muestras

Los filtros del dashboard (hora, género, rango etario, clasificación) exigen
poder recomponer el mapa de cualquier subconjunto. Guardar cada muestra sería
lo directo, pero una jornada de 8 h a 25 fps con 5 personas son millones de
puntos.

En su lugar se acumula directamente sobre una rejilla **por combinación de
atributos**. La memoria queda acotada por el número de combinaciones (no por
la duración de la sesión), y un filtro se resuelve sumando las rejillas que
encajan. Una jornada entera ocupa unos pocos MB.

## Coordenadas

Entran normalizadas 0..1, igual que las zonas (ver `stand_config.py`), así que
el mapa sobrevive a que la ventana de iVMS-4200 cambie de tamaño.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TIPO_TRAFICO = "trafico"
TIPO_PERMANENCIA = "permanencia"
TIPO_DETENCIONES = "detenciones"
TIPOS = (TIPO_TRAFICO, TIPO_PERMANENCIA, TIPO_DETENCIONES)

# Clave de segmento: (tipo, genero, rango_etario, clasificacion, hora)
_Clave = Tuple[str, str, str, str, int]

_SIN_DATO = "?"


class MapaCalorStand:
    """Tres rejillas acumulativas, segmentadas para poder filtrarlas."""

    def __init__(self, cfg: Any, ancho_frame: int, alto_frame: int) -> None:
        self._celda = max(1, int(cfg.get('heatmap.celda_px', 20)))
        self._sigma = float(cfg.get('heatmap.sigma_celdas', 2))
        self.ancho_frame = int(ancho_frame)
        self.alto_frame = int(alto_frame)
        self.cols = max(1, self.ancho_frame // self._celda)
        self.filas = max(1, self.alto_frame // self._celda)
        self._rejillas: Dict[_Clave, np.ndarray] = defaultdict(
            lambda: np.zeros((self.filas, self.cols), dtype=np.float32))
        self._muestras = 0

    # ── Acumulación ─────────────────────────────────────────────────────
    def _celda_de(self, punto: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Punto normalizado -> (fila, columna). None si cae fuera."""
        x, y = float(punto[0]), float(punto[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        col = min(self.cols - 1, int(x * self.cols))
        fila = min(self.filas - 1, int(y * self.filas))
        return fila, col

    def _clave(self, tipo: str, genero: Optional[str],
               rango: Optional[str], clasificacion: Optional[str],
               hora: Optional[int]) -> _Clave:
        return (tipo,
                genero or _SIN_DATO,
                rango or _SIN_DATO,
                clasificacion or _SIN_DATO,
                int(hora) if hora is not None else -1)

    def registrar_posicion(self, punto: Tuple[float, float], *,
                           dt: float = 0.0,
                           genero: Optional[str] = None,
                           rango_etario: Optional[str] = None,
                           clasificacion: Optional[str] = None,
                           hora: Optional[int] = None) -> bool:
        """Una observación del punto de pie. Alimenta tráfico y permanencia.

        `punto` es el **centro inferior del bbox** proyectado, no el centro
        de la caja: el centro flota a la altura del pecho y dibujaría el mapa
        medio metro por delante de donde la persona pisa.
        """
        celda = self._celda_de(punto)
        if celda is None:
            return False
        f, c = celda
        k = (genero, rango_etario, clasificacion, hora)
        self._rejillas[self._clave(TIPO_TRAFICO, *k)][f, c] += 1.0
        if dt > 0:
            self._rejillas[self._clave(TIPO_PERMANENCIA, *k)][f, c] += float(dt)
        self._muestras += 1
        return True

    def registrar_detencion(self, punto: Tuple[float, float], *,
                            genero: Optional[str] = None,
                            rango_etario: Optional[str] = None,
                            clasificacion: Optional[str] = None,
                            hora: Optional[int] = None) -> bool:
        celda = self._celda_de(punto)
        if celda is None:
            return False
        f, c = celda
        self._rejillas[self._clave(TIPO_DETENCIONES, genero, rango_etario,
                                   clasificacion, hora)][f, c] += 1.0
        return True

    # ── Consulta ────────────────────────────────────────────────────────
    def matriz(self, tipo: str, *,
               genero: Optional[str] = None,
               rango_etario: Optional[str] = None,
               clasificacion: Optional[str] = None,
               horas: Optional[List[int]] = None,
               suavizar: bool = True,
               normalizar: bool = True) -> np.ndarray:
        """Rejilla del subconjunto pedido. Cada filtro en None = «todos»."""
        if tipo not in TIPOS:
            raise ValueError(f"tipo de heatmap desconocido: {tipo!r}")

        acumulada = np.zeros((self.filas, self.cols), dtype=np.float32)
        for (t, g, r, cl, h), rejilla in self._rejillas.items():
            if t != tipo:
                continue
            if genero is not None and g != genero:
                continue
            if rango_etario is not None and r != rango_etario:
                continue
            if clasificacion is not None and cl != clasificacion:
                continue
            if horas is not None and h not in horas:
                continue
            acumulada += rejilla

        if suavizar and self._sigma > 0:
            acumulada = self._suavizar(acumulada)
        if normalizar:
            pico = float(acumulada.max())
            if pico > 0:
                acumulada = acumulada / pico
        return acumulada

    def _suavizar(self, m: np.ndarray) -> np.ndarray:
        """Gaussiana de sigma celdas. Cae a numpy si no hay OpenCV."""
        try:
            import cv2
            k = int(self._sigma * 6) | 1        # impar, ~3 sigma por lado
            return cv2.GaussianBlur(m, (k, k), self._sigma)
        except Exception:                                   # noqa: BLE001
            return m

    def grid(self) -> Dict[str, int]:
        return {'ancho': self.cols, 'alto': self.filas,
                'celda_px': self._celda}

    def zonas_calientes(self, tipo: str, k: int = 5) -> List[Dict[str, Any]]:
        """Las k celdas con más carga, en coordenadas normalizadas."""
        m = self.matriz(tipo)
        if m.max() <= 0:
            return []
        planos = np.argsort(m, axis=None)[::-1][:k]
        salida = []
        for idx in planos:
            f, c = divmod(int(idx), self.cols)
            salida.append({
                'x': round((c + 0.5) / self.cols, 4),
                'y': round((f + 0.5) / self.filas, 4),
                'intensidad': round(float(m[f, c]), 4),
            })
        return salida

    # ── Exportación ─────────────────────────────────────────────────────
    def overlay(self, tipo: str, fondo: Optional[np.ndarray] = None,
                alpha: float = 0.5, **filtros: Any) -> Optional[np.ndarray]:
        """Imagen BGR con el colormap superpuesto sobre un frame."""
        try:
            import cv2
        except ImportError:
            logger.error("stand: sin OpenCV no se puede renderizar el mapa")
            return None

        m = self.matriz(tipo, **filtros)
        mapa = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if fondo is None:
            return cv2.resize(mapa, (self.ancho_frame, self.alto_frame),
                              interpolation=cv2.INTER_LINEAR)
        mapa = cv2.resize(mapa, (fondo.shape[1], fondo.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(mapa, alpha, fondo, 1.0 - alpha, 0)

    def como_payload(self, tipo: str, **filtros: Any) -> Dict[str, Any]:
        """Forma que viaja por WebSocket y se guarda en `stand_heatmap`."""
        m = self.matriz(tipo, **filtros)
        return {
            'tipo': tipo,
            'grid': self.grid(),
            # Se redondea a 4 decimales: el payload va por WebSocket cada
            # 30 s y la precisión extra solo engorda el mensaje.
            'matriz': [[round(float(v), 4) for v in fila] for fila in m],
        }

    def snapshot(self, almacen: Any, sesion_id: int) -> int:
        """Persiste las tres capas. Devuelve cuántas guardó."""
        n = 0
        for tipo in TIPOS:
            p = self.como_payload(tipo)
            if not any(any(f) for f in p['matriz']):
                continue                    # capa vacía: no ensucia la BD
            almacen.guardar_heatmap(sesion_id, tipo, p['grid'], p['matriz'])
            n += 1
        return n

    def salud(self) -> Dict[str, Any]:
        return {
            'muestras': self._muestras,
            'segmentos': len(self._rejillas),
            'grid': self.grid(),
        }
