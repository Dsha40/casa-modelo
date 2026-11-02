"""src/app/grabador_personas.py — graba las imágenes cuando hay gente.

## Para qué (30-oct-2026, rama contructora)

Para comparar dos versiones de la identidad y la clasificación hace falta
pasarles las MISMAS imágenes: dos cámaras, o dos ratos con gente distinta,
no se pueden comparar (lo señaló el usuario). En la tienda pasa poca gente,
así que grabar todo el día llenaría el disco de imágenes vacías. Este
grabador solo guarda los tramos con personas, para reproducirlos después
por el servidor con un cliente de mentira, idénticos byte a byte.

El de `anaqueles_grabador.py` no sirve: solo graba en modo Anaqueles, vuelve
a comprimir cada imagen (~15-20 ms de CPU en el PC de la tienda) y no guarda
lo que mandó el cliente con ella.

## Qué guarda

Por cámara, una carpeta por tramo (`<cámara>/<AAAA-MM-DD_HHMMSS>/`):

  * `NNNNNN.jpg`: el JPEG TAL COMO LLEGÓ del cliente, sin recomprimir (~1 ms
    de disco por imagen; ~125 KB a 1456 px);
  * `imagenes.jsonl`: una línea por imagen con la hora y los parámetros del
    mensaje (ROI, zonas a ignorar, ángulo…) para reproducirla igual.

Un tramo empieza cuando el pipeline tiene algún seguimiento activo y lleva
dentro los `ANTES_S` segundos previos (ByteTrack necesita ver llegar a la
persona). Acaba `DESPUES_S` segundos después de que no quede ninguno.

## Cómo se enciende

`ELDE_GRABAR_PERSONAS=<carpeta absoluta>` al arrancar el servidor. Apagado
por defecto. Se detiene solo al llegar a `ELDE_GRABAR_PERSONAS_MAX_MB`
(5000 por defecto, ~40 min de gente a 17 imágenes/s) o si el disco baja de
`DISCO_MINIMO_GB`. Son fotos de clientes: la carpeta va fuera de git
(`server/output/` lo está) y se borra con permiso del usuario.

## Pararlo y reanudarlo sin reiniciar (2-nov-2026)

El usuario lo pidió para probar con vídeos sin grabarlos
(`api_grabador_personas.py`): `POST /api/v1/tienda/grabador/parar` y
`.../reanudar`; `GET /api/v1/tienda/grabador` dice cómo está. Parar cierra
el tramo abierto. Un reinicio del servidor vuelve a lo que diga la
variable. Reanudar no enciende un grabador que no se configuró al arrancar.
"""
from __future__ import annotations

import collections
import datetime
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: Segundos que se guardan ANTES de que aparezca la primera persona.
ANTES_S = 2.0
#: Segundos que se sigue grabando cuando ya no queda nadie.
DESPUES_S = 3.0
#: Por debajo de este espacio libre se deja de grabar.
DISCO_MINIMO_GB = 6.0


def _a_json(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
    except Exception:                                       # noqa: BLE001
        pass
    return str(v)


class _Camara:
    """Estado de una cámara: imágenes recientes y tramo abierto."""

    def __init__(self) -> None:
        self.recientes: Deque[Tuple[float, bytes, Dict[str, Any]]] = \
            collections.deque()
        self.carpeta: Optional[Path] = None
        self.jsonl: Any = None
        self.n = 0
        self.ultima_con_gente = 0.0


class GrabadorPersonas:
    """Graba los tramos con gente de cada cámara."""

    def __init__(self, carpeta: Path, max_mb: float) -> None:
        self.carpeta = Path(carpeta)
        self.max_bytes = int(float(max_mb) * 1024 * 1024)
        self.bytes = 0
        self.parado = False
        self.motivo = ''
        self.lleno = False
        self._camaras: Dict[str, _Camara] = {}
        self._lock = threading.Lock()
        self.carpeta.mkdir(parents=True, exist_ok=True)
        logger.warning("GRABADOR PERSONAS: encendido en %s (máximo %.0f MB)",
                       self.carpeta, max_mb)

    def ver(self, camera_id: Any, jpeg: Any, metadata: Dict[str, Any],
            parametros: Dict[str, Any], ahora: Optional[float] = None) -> None:
        """Una imagen ya procesada. Nunca lanza: grabar no puede costar la
        imagen."""
        try:
            if self.parado or not isinstance(jpeg, (bytes, bytearray)):
                return
            ahora = time.time() if ahora is None else float(ahora)
            gente = int((metadata or {}).get('active_tracks') or 0)
            with self._lock:
                cam = self._camaras.setdefault(str(camera_id), _Camara())
                if gente > 0:
                    cam.ultima_con_gente = ahora
                if cam.carpeta is None:
                    cam.recientes.append((ahora, bytes(jpeg), parametros))
                    while cam.recientes and ahora - cam.recientes[0][0] > ANTES_S:
                        cam.recientes.popleft()
                    if gente > 0:
                        self._abrir(str(camera_id), cam, ahora)
                    return
                self._escribir(cam, ahora, bytes(jpeg), parametros, gente)
                if ahora - cam.ultima_con_gente > DESPUES_S:
                    self._cerrar(str(camera_id), cam)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("GRABADOR PERSONAS: falló: %s", exc)

    def _abrir(self, camera_id: str, cam: _Camara, ahora: float) -> None:
        libre_gb = shutil.disk_usage(self.carpeta).free / 1024 ** 3
        if libre_gb < DISCO_MINIMO_GB:
            self._parar("quedan %.1f GB libres" % libre_gb)
            return
        nombre = datetime.datetime.fromtimestamp(ahora).strftime(
            '%Y-%m-%d_%H%M%S')
        seguro = ''.join(c if c.isalnum() or c in '-_.' else '_'
                         for c in camera_id)
        cam.carpeta = self.carpeta / seguro / nombre
        cam.carpeta.mkdir(parents=True, exist_ok=True)
        cam.jsonl = open(cam.carpeta / 'imagenes.jsonl', 'a', encoding='utf-8')
        cam.n = 0
        logger.info("GRABADOR PERSONAS [%s]: empieza tramo %s", camera_id,
                    nombre)
        previas = list(cam.recientes)
        cam.recientes.clear()
        for t, jpeg, parametros in previas:
            self._escribir(cam, t, jpeg, parametros, None)

    def _escribir(self, cam: _Camara, t: float, jpeg: bytes,
                  parametros: Dict[str, Any], gente: Optional[int]) -> None:
        if self.bytes + len(jpeg) > self.max_bytes:
            self.lleno = True
            self._parar("llegó al máximo de %.0f MB"
                        % (self.max_bytes / 1024 / 1024))
            return
        nombre = f'{cam.n:06d}.jpg'
        (cam.carpeta / nombre).write_bytes(jpeg)
        self.bytes += len(jpeg)
        cam.n += 1
        cam.jsonl.write(json.dumps(
            {'imagen': nombre, 't': t, 'seguimientos': gente,
             'parametros': parametros},
            ensure_ascii=False, default=_a_json) + '\n')
        cam.jsonl.flush()

    def _cerrar(self, camera_id: str, cam: _Camara) -> None:
        logger.info("GRABADOR PERSONAS [%s]: acaba tramo %s (%d imágenes; "
                    "%.0f MB en total)", camera_id, cam.carpeta.name, cam.n,
                    self.bytes / 1024 / 1024)
        if cam.jsonl is not None:
            cam.jsonl.close()
        cam.carpeta, cam.jsonl, cam.n = None, None, 0

    def _parar(self, motivo: str) -> None:
        self.parado = True
        self.motivo = motivo
        for camera_id, cam in self._camaras.items():
            if cam.carpeta is not None:
                self._cerrar(camera_id, cam)
            cam.recientes.clear()
        logger.warning("GRABADOR PERSONAS: PARADO, %s", motivo)

    def parar(self) -> None:
        """Lo pide el usuario: cierra lo abierto y deja de grabar."""
        with self._lock:
            if not self.parado:
                self._parar("lo pidió el usuario")

    def reanudar(self) -> bool:
        """Vuelve a grabar, salvo si falta disco o se llegó al máximo."""
        with self._lock:
            if self.lleno:
                return False
            if shutil.disk_usage(self.carpeta).free / 1024 ** 3 < DISCO_MINIMO_GB:
                return False
            self.parado, self.motivo = False, ''
            logger.warning("GRABADOR PERSONAS: REANUDADO (lo pidió el usuario)")
            return True

    def estado(self) -> Dict[str, Any]:
        return {'configurado': True, 'grabando': not self.parado,
                'motivo_parado': self.motivo or None,
                'carpeta': str(self.carpeta),
                'mb_grabados': round(self.bytes / 1024 / 1024),
                'mb_maximo': round(self.max_bytes / 1024 / 1024)}


_GRABADOR: Optional[GrabadorPersonas] = None
_INTENTADO = False


def grabador() -> Optional[GrabadorPersonas]:
    """El grabador si `ELDE_GRABAR_PERSONAS` está puesto; si no, None."""
    global _GRABADOR, _INTENTADO
    if not _INTENTADO:
        _INTENTADO = True
        carpeta = os.getenv('ELDE_GRABAR_PERSONAS', '').strip()
        if carpeta:
            try:
                _GRABADOR = GrabadorPersonas(
                    Path(carpeta),
                    float(os.getenv('ELDE_GRABAR_PERSONAS_MAX_MB', '5000')))
            except Exception as exc:                        # noqa: BLE001
                logger.error("GRABADOR PERSONAS: no se pudo abrir %s: %s",
                             carpeta, exc)
    return _GRABADOR


def ver(camera_id: Any, jpeg: Any, metadata: Dict[str, Any],
        parametros: Dict[str, Any]) -> None:
    """Lo que llama `app.py` con cada imagen. Sin la variable no hace nada."""
    g = grabador()
    if g is not None:
        g.ver(camera_id, jpeg, metadata, parametros)
