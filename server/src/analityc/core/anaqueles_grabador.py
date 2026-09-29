"""src/analityc/core/anaqueles_grabador.py — grabación de diagnóstico.

## Para qué (28-sep-2026)

Un agarre que el operador VE (la muñeca pasa de ámbar a verde) y que el
servidor no cuenta no se puede arreglar a ciegas: la máquina del gesto se
endureció para cortar falsos agarres (14-ago: la muñeca colgando dentro) y
aflojarla sin medir los trae de vuelta. Con esto se graba exactamente lo que
vio el modo Anaqueles —el frame, los tracks de persona, las áreas y los
productos— para reproducirlo fuera de línea con la pose real y comparar la
regla vieja con la nueva sobre los MISMOS datos.

No se re-ejecutan detección ni tracking al reproducir: se guardan los tracks
tal como los tenía el pipeline, porque otra pasada de ByteTrack daría otros
ids y otras cajas, y ya no sería la misma prueba.

## Cómo se enciende

`ELDE_ANAQUELES_GRABAR=<carpeta absoluta>` al arrancar el servidor. Apagado
por defecto. Escribe `frames/<n>.jpg` y una línea JSON por frame en
`eventos.jsonl`, y se detiene solo al llegar a `ELDE_ANAQUELES_GRABAR_MAX`
frames (4000 por defecto, ~10 min a 6 fps): es diagnóstico, no un archivo.
Las fotos son de clientes: la carpeta debe quedar fuera del repo y borrarse
al terminar.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Grabador:
    """Graba frames + estado del modo Anaqueles en una carpeta."""

    def __init__(self, carpeta: Path, maximo: int = 4000) -> None:
        self.carpeta = Path(carpeta)
        self.maximo = int(maximo)
        self.n = 0
        self._lock = threading.Lock()
        (self.carpeta / 'frames').mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.carpeta / 'eventos.jsonl', 'a',
                           encoding='utf-8')
        logger.warning("anaqueles: GRABANDO frames de diagnóstico en %s "
                       "(máximo %d)", self.carpeta, self.maximo)

    def grabar(self, camera_id: Any, frame: np.ndarray,
               tracks: Dict[Any, Dict], zonas: Any, nombres: Any,
               productos: Any, t: float) -> None:
        """Un frame. Nunca lanza: grabar no puede costar la medición."""
        try:
            with self._lock:
                if self.n >= self.maximo:
                    return
                n = self.n
                self.n += 1
            nombre = f'{n:06d}.jpg'
            cv2.imwrite(str(self.carpeta / 'frames' / nombre), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            personas = {}
            for tid, tr in (tracks or {}).items():
                if tr.get('class', -1) != -1 or tr.get('box') is None:
                    continue
                personas[str(int(tid))] = {
                    'box': [float(v) for v in list(tr['box'])[:4]],
                    'persistent_id': (str(tr['persistent_id'])
                                      if tr.get('persistent_id') else None),
                    'es_personal': bool(tr.get('es_personal')),
                }
            linea = {'n': n, 'frame': nombre, 't': float(t),
                     'camera_id': str(camera_id), 'tracks': personas,
                     'zonas': zonas, 'nombres': nombres,
                     'productos': productos}
            with self._lock:
                self._jsonl.write(json.dumps(linea, ensure_ascii=False,
                                             default=_a_json) + '\n')
                self._jsonl.flush()
                if n + 1 == self.maximo:
                    logger.warning("anaqueles: grabación de diagnóstico "
                                   "COMPLETA (%d frames)", self.maximo)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("anaqueles: grabar falló: %s", exc)


def _a_json(v: Any) -> Any:
    """numpy → tipos de JSON (las zonas pueden llegar como arrays)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


_GRABADOR: Optional[Grabador] = None
_INTENTADO = False


def grabador() -> Optional[Grabador]:
    """El grabador si `ELDE_ANAQUELES_GRABAR` está puesto; si no, None."""
    global _GRABADOR, _INTENTADO
    if not _INTENTADO:
        _INTENTADO = True
        carpeta = os.getenv('ELDE_ANAQUELES_GRABAR', '').strip()
        if carpeta:
            try:
                _GRABADOR = Grabador(
                    Path(carpeta),
                    int(os.getenv('ELDE_ANAQUELES_GRABAR_MAX', '4000')))
            except Exception as exc:                        # noqa: BLE001
                logger.error("anaqueles: no se pudo abrir la grabación en "
                             "%s: %s", carpeta, exc)
    return _GRABADOR
