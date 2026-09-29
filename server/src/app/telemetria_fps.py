"""src/app/telemetria_fps.py — cuántos frames recibe y procesa cada cámara.

## Por qué existe (28-sep-2026)

El servidor no medía su propio ritmo: el «2-3 fps» de los comentarios nunca
estuvo medido, y el 24-sep hubo que deducir los fps del contador `Frame N`
del log. El 28-sep, al probar la pose de Anaqueles en más frames, la cifra
salió de una grabación de diagnóstico y quedó mezclada con otra carga ajena
en la misma GPU. Sin una medida propia no se puede comparar una GPU con otra
ni saber si el cuello está en el servidor o en el ciclo con el cliente.

Una línea por cámara cada `PERIODO_S` en el log:

    FPS [cam] recibidos=5.3/s procesados=5.3/s descartados=0 proceso=38/61 ms
        entrada=1650 KB/s (311 KB/frame)

- `recibidos`: frames que llegaron por el websocket (lo que manda el cliente).
- `entrada`: lo que pesan esas imágenes (29-sep-2026). El cliente manda un
  frame y espera la respuesta, y de un ciclo de ~164 ms el servidor solo
  ocupa ~63: el resto pasa FUERA, y cada frame de iVMS pesa ~300 KB por un
  túnel SSH. Si los KB/s se quedan planos aunque cambien el tamaño o los
  fps, el techo es la conexión, no la GPU.
- `procesados`: frames que pasaron por el pipeline (la cola se queda solo
  con el último por cámara; la diferencia son los `descartados`).
- `proceso`: mediana/p90 del tiempo en el servidor por frame (decodificar,
  inferir y armar la respuesta), sin contar la red ni al cliente. Si se
  acerca al intervalo entre frames, el cuello es el servidor; si es mucho
  menor, el cliente está esperando por otra cosa.

Nunca lanza: medir no puede costar un frame.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

#: Cada cuánto se escribe el resumen. 30 s da ~150 frames por muestra a
#: 5 fps: suficiente para una mediana estable, corto para ver un cambio.
PERIODO_S = 30.0

_lock = threading.Lock()
_por_camara: Dict[str, Dict[str, Any]] = {}
_desde = time.monotonic()


def _cam(camera_id: Any) -> Dict[str, Any]:
    clave = str(camera_id)
    c = _por_camara.get(clave)
    if c is None:
        c = _por_camara[clave] = {'recibidos': 0, 'procesados': 0,
                                  'ms': [], 'bytes': 0}
    return c


def tamano_imagen(imagen: Any) -> int:
    """Bytes de la imagen tal como llegó: binaria (msgpack) o en texto
    base64 (formato antiguo; 4 caracteres = 3 bytes). 0 si no se sabe."""
    try:
        if isinstance(imagen, (bytes, bytearray, memoryview)):
            return len(imagen)
        if isinstance(imagen, str):
            return len(imagen.split(',', 1)[-1]) * 3 // 4
    except Exception:                                       # noqa: BLE001
        pass
    return 0


def tamano_de_payload(payload: Any) -> int:
    """Bytes de la imagen de un `ItemFrame.payload`: es el 2.º elemento de
    la tupla `frame_payload` de app.py (`processor, image_data, ...`).
    Nunca lanza: lo llama `encolar`, que tampoco puede lanzar."""
    try:
        return tamano_imagen(payload[1])
    except Exception:                                       # noqa: BLE001
        return 0


def anotar_recibido(camera_id: Any, tamano: int = 0) -> None:
    try:
        with _lock:
            c = _cam(camera_id)
            c['recibidos'] += 1
            c['bytes'] += max(0, int(tamano))
        _volcar_si_toca()
    except Exception:                                       # noqa: BLE001
        pass


def anotar_procesado(camera_id: Any, ms: float) -> None:
    try:
        with _lock:
            c = _cam(camera_id)
            c['procesados'] += 1
            c['ms'].append(float(ms))
        _volcar_si_toca()
    except Exception:                                       # noqa: BLE001
        pass


def resumen(reiniciar: bool = False) -> List[Dict[str, Any]]:
    """Las cifras del periodo en curso, por cámara."""
    global _desde
    with _lock:
        ahora = time.monotonic()
        lapso = max(1e-6, ahora - _desde)
        salida = []
        for cam, c in sorted(_por_camara.items()):
            ms = sorted(c['ms'])
            salida.append({
                'camera_id': cam,
                'segundos': round(lapso, 1),
                'recibidos_s': round(c['recibidos'] / lapso, 2),
                'procesados_s': round(c['procesados'] / lapso, 2),
                'descartados': max(0, c['recibidos'] - c['procesados']),
                'proceso_ms_mediana': (round(ms[len(ms) // 2], 1)
                                       if ms else None),
                'proceso_ms_p90': (round(ms[int(len(ms) * 0.9)], 1)
                                   if ms else None),
                'entrada_kb_s': round(c['bytes'] / 1024 / lapso, 1),
                'kb_por_frame': (round(c['bytes'] / 1024 / c['recibidos'])
                                 if c['recibidos'] else None),
            })
        if reiniciar:
            _por_camara.clear()
            _desde = ahora
        return salida


def _volcar_si_toca() -> None:
    if (time.monotonic() - _desde) < PERIODO_S:
        return
    for r in resumen(reiniciar=True):
        if not (r['recibidos_s'] or r['procesados_s']):
            continue
        logger.info(
            "FPS [%s] recibidos=%.1f/s procesados=%.1f/s descartados=%d "
            "proceso=%s/%s ms entrada=%.0f KB/s (%s KB/frame) (en %.0f s)",
            r['camera_id'], r['recibidos_s'], r['procesados_s'],
            r['descartados'], r['proceso_ms_mediana'], r['proceso_ms_p90'],
            r['entrada_kb_s'], r['kb_por_frame'], r['segundos'])
