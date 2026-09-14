"""Configuración y zonas del stand de degustación (HITO 3).

## Por qué las zonas se guardan NORMALIZADAS

Los polígonos van en coordenadas 0..1 relativas al frame, no en píxeles. El
cliente Amazonas captura una **ventana de Windows** (iVMS-4200), cuyo tamaño
cambia si el operador la redimensiona, la maximiza o cambia de monitor. Unas
coordenadas en píxeles calibradas a 1280×720 quedarían silenciosamente
desplazadas al día siguiente, y el sistema seguiría contando — mal, y sin
avisar. Normalizadas sobreviven al cambio de tamaño.

## Por qué hay DOS archivos

- `stand_config.yaml` — escrito a mano, comentado umbral por umbral. Nunca lo
  toca una máquina: `yaml.dump` borraría todos los comentarios, que son la
  mitad del valor del archivo.
- `stand_zonas.yaml` — **generado** por `tools/calibrar_stand.py`. Se
  sobreescribe entero en cada calibración sin perder nada.

El loader los mezcla. Si `stand_zonas.yaml` no existe o tiene zonas vacías, el
módulo del stand **no cuenta y lo dice en el log**: nunca asume una zona por
defecto, porque un polígono inventado produce cifras creíbles y falsas, que es
el peor resultado posible.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_RAIZ_SERVIDOR = Path(__file__).resolve().parents[3]

# Las cinco zonas del HITO 3. `linea_conteo` es un segmento (2 puntos), el
# resto son polígonos (>=3 puntos).
ZONAS_POLIGONO = ('zona_paso', 'zona_stand', 'zona_atencion', 'zona_personal')
ZONA_LINEA = 'linea_conteo'
TODAS_LAS_ZONAS = ZONAS_POLIGONO + (ZONA_LINEA,)


def ruta_config() -> Path:
    propia = (os.getenv('STAND_CONFIG') or '').strip()
    return (Path(propia) if propia
            else _RAIZ_SERVIDOR / 'config' / 'stand_config.yaml')


def ruta_zonas() -> Path:
    propia = (os.getenv('STAND_ZONAS') or '').strip()
    return (Path(propia) if propia
            else _RAIZ_SERVIDOR / 'config' / 'stand_zonas.yaml')


def _leer_yaml(ruta: Path) -> Dict[str, Any]:
    if not ruta.is_file():
        return {}
    try:
        with open(ruta, encoding='utf-8') as f:
            datos = yaml.safe_load(f)
        return datos if isinstance(datos, dict) else {}
    except Exception as exc:                                # noqa: BLE001
        logger.error("stand: no se pudo leer %s: %s", ruta, exc)
        return {}


class ConfigStand:
    """Umbrales + zonas, con las zonas ya validadas."""

    def __init__(self, datos: Dict[str, Any],
                 zonas: Optional[Dict[str, Any]] = None) -> None:
        self._d = datos or {}
        z = (zonas or {}).get('zonas') or self._d.get('zonas') or {}
        self._zonas: Dict[str, List[List[float]]] = {}
        for nombre in TODAS_LAS_ZONAS:
            puntos = z.get(nombre) or []
            self._zonas[nombre] = self._normalizar_puntos(nombre, puntos)
        self._meta_zonas = {k: v for k, v in (zonas or {}).items()
                            if k != 'zonas'}

    # ── Acceso a umbrales ───────────────────────────────────────────────
    def get(self, ruta: str, defecto: Any = None) -> Any:
        """Lee con ruta por puntos: `get('detencion.dwell_min_s', 3.0)`."""
        nodo: Any = self._d
        for parte in ruta.split('.'):
            if not isinstance(nodo, dict) or parte not in nodo:
                return defecto
            nodo = nodo[parte]
        return nodo

    @property
    def habilitado(self) -> bool:
        return bool(self._d.get('habilitado', False))

    @property
    def stand_id(self) -> str:
        return str(self.get('stand.stand_id', 'stand-desconocido'))

    @property
    def camara_id(self) -> str:
        return str(self.get('stand.camara_id', ''))

    # ── Zonas ───────────────────────────────────────────────────────────
    @staticmethod
    def _normalizar_puntos(nombre: str,
                           puntos: Any) -> List[List[float]]:
        """Valida y deja los puntos en 0..1.

        Acepta píxeles por comodidad de quien edite a mano: si algún valor
        supera 1.0 se asume que son píxeles y se avisa, porque sin saber el
        tamaño del frame no se pueden convertir aquí.
        """
        if not isinstance(puntos, (list, tuple)) or not puntos:
            return []
        limpios: List[List[float]] = []
        for p in puntos:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                logger.warning("stand: punto inválido en %s: %r", nombre, p)
                return []
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                logger.warning("stand: punto no numérico en %s: %r", nombre, p)
                return []
            limpios.append([x, y])

        fuera = [p for p in limpios if p[0] > 1.0 or p[1] > 1.0]
        if fuera:
            logger.error(
                "stand: %s parece estar en PÍXELES (%d puntos > 1.0). Las "
                "zonas deben ir normalizadas 0..1; recalibra con "
                "tools/calibrar_stand.py", nombre, len(fuera))
            return []

        minimo = 2 if nombre == ZONA_LINEA else 3
        if len(limpios) < minimo:
            logger.warning("stand: %s tiene %d puntos, necesita >= %d",
                           nombre, len(limpios), minimo)
            return []
        return limpios

    def zona_normalizada(self, nombre: str) -> List[List[float]]:
        return list(self._zonas.get(nombre) or [])

    def zona_px(self, nombre: str, ancho: int,
                alto: int) -> Optional[np.ndarray]:
        """Polígono en píxeles para un frame de `ancho`×`alto`.

        Devuelve None si la zona no está calibrada — quien llame debe tratar
        el None, nunca sustituirlo por el frame entero.
        """
        puntos = self._zonas.get(nombre) or []
        if not puntos:
            return None
        return np.array([[int(round(x * ancho)), int(round(y * alto))]
                         for x, y in puntos], dtype=np.int32)

    def zonas_calibradas(self) -> Dict[str, bool]:
        return {n: bool(self._zonas.get(n)) for n in TODAS_LAS_ZONAS}

    def listo_para_contar(self) -> Tuple[bool, str]:
        """¿Se puede medir con esta configuración?

        Mínimo imprescindible: `zona_paso` (denominador de "pasaron") y
        `zona_stand` (dónde ocurre todo). Sin ellas no hay métrica posible.
        `zona_atencion`, `zona_personal` y `linea_conteo` degradan funciones
        concretas pero no impiden contar.
        """
        if not self.habilitado:
            return False, "el módulo está deshabilitado (habilitado: false)"
        faltan = [n for n in ('zona_paso', 'zona_stand')
                  if not self._zonas.get(n)]
        if faltan:
            return False, (f"zonas sin calibrar: {', '.join(faltan)}. "
                           "Ejecuta tools/calibrar_stand.py")
        return True, "ok"

    def avisos(self) -> List[str]:
        """Degradaciones conocidas, para el panel de calidad del dato."""
        fuera = []
        if not self._zonas.get('zona_atencion'):
            fuera.append("sin zona_atencion: no se puede medir 'atendido'")
        if not self._zonas.get('zona_personal'):
            fuera.append("sin zona_personal: el chef no se detecta por zona, "
                         "solo por enrolamiento manual")
        if not self._zonas.get(ZONA_LINEA):
            fuera.append("sin linea_conteo: no hay conteo direccional")
        if not self.homografia_activa:
            fuera.append("sin homografía: los umbrales de distancia y "
                         "velocidad van en píxeles y dependen de la "
                         "perspectiva (limitación documentada)")
        return fuera

    # ── Homografía ──────────────────────────────────────────────────────
    @property
    def homografia_activa(self) -> bool:
        h = self._meta_zonas.get('homografia') or self._d.get('homografia') \
            or {}
        if not h.get('activa'):
            return False
        return (len(h.get('puntos_imagen') or []) == 4 and
                len(h.get('puntos_mundo_m') or []) == 4)

    def matriz_homografia(self) -> Optional[np.ndarray]:
        """Matriz 3×3 imagen→mundo, o None si no hay homografía."""
        if not self.homografia_activa:
            return None
        h = self._meta_zonas.get('homografia') or self._d.get('homografia')
        try:
            import cv2
            origen = np.array(h['puntos_imagen'], dtype=np.float32)
            destino = np.array(h['puntos_mundo_m'], dtype=np.float32)
            matriz, _ = cv2.findHomography(origen, destino)
            return matriz
        except Exception as exc:                            # noqa: BLE001
            logger.error("stand: homografía inválida: %s", exc)
            return None


def cargar(ruta_cfg: Optional[Path] = None,
           ruta_z: Optional[Path] = None) -> ConfigStand:
    """Carga umbrales + zonas. Nunca lanza: devuelve config deshabilitada."""
    datos = _leer_yaml(Path(ruta_cfg or ruta_config()))
    zonas = _leer_yaml(Path(ruta_z or ruta_zonas()))
    return ConfigStand(datos, zonas)
