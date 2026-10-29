"""src/app/api_en_vivo.py — lo que pasa AHORA, para la página «En vivo» (22-oct-2026).

## Por qué

El dashboard de tienda (`dashboards/tienda/`) cuenta un periodo: cuánta
gente entró, qué anaquel se agarra. Para vigilar el sistema MIENTRAS corre
había que leer a mano el log (`FPS [...]`, `SIN CONFIRMAR`) y
`logs/gpu_temp.csv`. El 22-oct el usuario pidió verlo en tiempo real: la
velocidad, los agarres confirmados y sin confirmar, si la cámara manda vídeo
o la imagen gris de un corte de iVMS, la gente y la temperatura de la GPU.
Lo pinta `dashboards/tienda/en_vivo.html`, que pide `GET /api/v1/en-vivo`
cada 3 s.

## De dónde sale cada cosa

| Qué | De dónde | Retraso |
|---|---|---|
| Velocidad | `telemetria_fps` (memoria) | el periodo de 30 s en curso no se ve hasta que se cierra |
| Vídeo o imagen gris | `cortes_video.sin_video` sobre la última imagen procesada | ninguno |
| Gente ahora | seguimientos y área de conteo del procesador vivo | ninguno |
| Gente de hoy | `tienda.db` | hasta `CACHE_HOY_S` |
| Agarres de hoy | `output/anaqueles/<cámara>/<fecha>.json` | hasta 5 s (el registro vuelca cada 5 s) |
| Agarres esperando su bajada | memoria del medidor de Anaqueles (`proc._medidor`) | ninguno; NO se guardan en disco |
| GPU | `logs/gpu_temp.csv`, que escribe cada minuto `vigilante_gpu.ps1` | hasta 1 min |
| CPU y RAM del PC, por programa | `recursos_pc` (un hilo que mide cada 5 s) | hasta 5 s |

Los objetos vivos del procesador se LEEN, nunca se tocan (copias de listas y
diccionarios, como `_aforo_vivo` de `api_tienda.py`): una lectura a medio
cambiar da, como mucho, la cifra de una imagen antes.

## Imagen gris o captura vacía

Las dos son una imagen lisa (`sin_video`) y en ninguna se cuenta nada, pero
se arreglan distinto, así que se separan por lo que pesa la última imagen:

  * `gris`: la pantalla gris de un corte de iVMS (~39 KB). Se arregla en
    iVMS o en el grabador de la tienda.
  * `vacia`: menos de `VACIA_KB`. Es la captura de una ventana MINIMIZADA:
    el 22-oct a las 17:40 la ventana principal de iVMS estaba minimizada
    (en −32000, −32000) y llegaban imágenes de 1 KB a 29 por segundo; el
    30-sep pasó lo mismo de 16:28 a 16:30. Se arregla restaurando la
    ventana.

## «Desde cuándo»

El servidor no apunta cuándo empieza un corte (el aviso `CORTE [...]` solo
sale con el parche opcional `ELDE_CONGELAR_CORTES`). Aquí se recuerda el
estado de la última consulta: si cambia entre dos consultas, la hora del
cambio es buena (±3 s, lo que tarda la página en volver a preguntar). Si es
la primera vez que se mira esa cámara, no se sabe cuándo empezó, y la
respuesta lo dice (`cambio_visto: false`) para que la página escriba «al
menos desde» en vez de inventarse una hora.

Nunca lanza por un dato que falte: el bloque que no se puede leer sale como
`null` y la página escribe «sin datos» en ese bloque, sin romper los demás.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from . import telemetria_fps

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v1', tags=['lectura'])

#: Sin imágenes desde hace más de esto, la cámara está «parada»: el
#: recuadro tiene la IA apagada, el cliente se cerró o se perdió la
#: conexión. Con el cliente sano llegan varias por segundo, y la red de
#: seguridad del recuadro reenvía a los 5 s (`ESPERA_RESPUESTA_MAX_S`).
PARADA_S = 15.0

#: Una imagen lisa de menos de esto es una captura vacía (ventana
#: minimizada, ~1 KB), no la pantalla gris de un corte (~39 KB).
VACIA_KB = 5.0

#: Lo de `tienda.db` se reutiliza durante esto: la página pregunta cada 3 s
#: y cada `resumen` del almacén fuerza un volcado a la base.
CACHE_HOY_S = 10.0

#: El CSV del vigilante: `server/logs/gpu_temp.csv`.
RUTA_GPU = Path(__file__).resolve().parents[2] / 'logs' / 'gpu_temp.csv'

#: Filas del CSV que se devuelven: una por minuto, la última hora.
FILAS_GPU = 60

#: Bits de `clocks_event_reasons.active` de nvidia-smi (los
#: `nvmlClocksEventReason*` de NVML): por qué la tarjeta baja su reloj. Los
#: que faltan (ajustes de aplicación, sincronía, pantalla) no son un aviso.
MOTIVOS_GPU = {
    0x1: 'en reposo',
    0x4: 'tope de consumo',
    0x8: 'freno de hardware',
    0x20: 'calor',
    0x40: 'calor (hardware)',
    0x80: 'freno de consumo (hardware)',
}
CALOR = 0x20 | 0x40

_lock = threading.Lock()
_video: Dict[str, Dict[str, Any]] = {}
_cache_hoy: Dict[str, Any] = {}


# ── Procesadores vivos ───────────────────────────────────────────────────

def _procesadores() -> Dict[str, Any]:
    """`camera_id -> procesador` de las cámaras conectadas ahora.

    Se mira en `sys.modules` en vez de importar `app.py`, por lo mismo que
    `_aforo_vivo` de `api_tienda.py`: importarlo arrastra torch, y en las
    pruebas no existe."""
    servidor = sys.modules.get(f'{__package__}.app')
    if servidor is None:
        return {}
    try:
        return {str(cam): proc
                for _cid, cam, proc in servidor._iter_person_processors()
                if cam is not None}
    except Exception:                                       # noqa: BLE001
        return {}


def _modo(proc: Any) -> Optional[str]:
    if proc is None:
        return None
    nombre = type(proc).__name__
    return 'Tiendas' if nombre == 'TiendasProcess' else nombre


# ── Velocidad y vídeo ────────────────────────────────────────────────────

def _velocidad(cam: str) -> Optional[Dict[str, Any]]:
    historial = telemetria_fps.historial(cam)
    if not historial:
        return None
    return {'ultimo': historial[-1], 'historial': historial}


def _seguir_video(cam: str, estado: str, ahora: float) -> Dict[str, Any]:
    """Recuerda el estado de la cámara y desde cuándo (ver la cabecera)."""
    with _lock:
        previo = _video.get(cam)
        if previo is None:
            previo = _video[cam] = {'estado': estado, 'desde': ahora,
                                    'cambio_visto': False}
        elif previo['estado'] != estado:
            previo.update(estado=estado, desde=ahora, cambio_visto=True)
        return dict(previo)


def _video_de(cam: str, proc: Any, ahora: float) -> Dict[str, Any]:
    """`video`, `gris` (pantalla de un corte), `vacia` (ventana minimizada)
    o `parada` (no llegan imágenes), con desde cuándo."""
    ultima = telemetria_fps.ultima_imagen(cam)
    tamano = telemetria_fps.ultimo_tamano(cam)
    kb = round(tamano / 1024, 1) if tamano else None
    if ultima is None:
        # Ninguna imagen desde que arrancó el servidor: no se recuerda como
        # estado. Si no, el primer estado de verdad salía como un cambio
        # visto a la hora del arranque: el 22-oct decía «vacía desde las
        # 17:47:22», el reinicio, con iVMS minimizado desde antes.
        return {'estado': 'parada', 'desde': None, 'cambio_visto': False,
                'ultima_imagen': None, 'ultima_imagen_kb': kb}
    if (ahora - ultima) > PARADA_S or proc is None:
        # Aquí la hora sí se sabe: la de la última imagen.
        salida = _seguir_video(cam, 'parada', ahora)
        salida.update(desde=ultima, cambio_visto=True)
    else:
        from ..analityc.core import cortes_video
        from .anaqueles_api import _ultimo_frame
        estado = 'video'
        if cortes_video.sin_video(_ultimo_frame(proc)):
            estado = 'vacia' if kb is not None and kb < VACIA_KB else 'gris'
        salida = _seguir_video(cam, estado, ahora)
    salida.update(ultima_imagen=ultima, ultima_imagen_kb=kb)
    return salida


# ── Gente ────────────────────────────────────────────────────────────────

def _gente_ahora(cam: str, proc: Any) -> Optional[Dict[str, Any]]:
    if proc is None:
        return None
    try:
        from .anaqueles_api import _cuantas_personas
        from .api_tienda import _aforo_vivo
        return {'en_imagen': _cuantas_personas(proc),
                'dentro_area': _aforo_vivo(cam)}
    except Exception as exc:                                # noqa: BLE001
        logger.debug("en vivo: gente de %s ilegible: %s", cam, exc)
        return None


def _clasificadas(alm: Any, jornada: str, cam: str) -> Dict[str, Any]:
    """Fichas clasificadas de la jornada, por género. Una fila por ficha
    (persona seguida): aquí sí suman, a diferencia de los cubos de
    `clasificaciones()`, donde quien tiene identidad cuenta una vez por
    género visto."""
    with closing(sqlite3.connect(str(alm.ruta), timeout=5)) as con:
        filas = con.execute(
            'SELECT genero, COUNT(*) FROM tienda_clasificaciones '
            'WHERE jornada = ? AND camera_id = ? GROUP BY genero',
            (jornada, cam)).fetchall()
    genero = {str(g): int(n) for g, n in filas}
    return {'total': sum(genero.values()), 'genero': genero}


def _gente_hoy(cam: str, ahora: float) -> Optional[Dict[str, Any]]:
    with _lock:
        guardado = _cache_hoy.get(cam)
        if guardado and (ahora - guardado[0]) < CACHE_HOY_S:
            return guardado[1]
    try:
        from ..analityc.core import tienda_almacen
        alm = tienda_almacen.get_almacen()
        jornada = tienda_almacen.jornada_de()
        datos = dict(alm.resumen(jornada, jornada, camera_id=cam),
                     jornada=jornada,
                     clasificadas=_clasificadas(alm, jornada, cam))
    except Exception as exc:                                # noqa: BLE001
        logger.debug("en vivo: gente de hoy de %s ilegible: %s", cam, exc)
        return None
    with _lock:
        _cache_hoy[cam] = (ahora, datos)
    return datos


# ── Agarres ──────────────────────────────────────────────────────────────

def _agarres(cam: str, proc: Any, ahora: float) -> Optional[Dict[str, Any]]:
    """Los de hoy del disco y los que esperan su bajada en memoria.

    Confirmado = el stock vio una bajada tras el gesto; sin confirmar = la
    vigilancia de su área acabó sin bajada (`anaqueles.py`,
    `_confirmar_por_stock`). Los que esperan aún no son ni lo uno ni lo
    otro, y un reinicio del servidor los pierde.

    Los que esperan viven en el MEDIDOR de la cámara (`MedidorAnaqueles`,
    `proc._medidor`), no en el procesador: la primera versión de este
    módulo los buscaba en el procesador y en vivo siempre daba «nada»
    (22-oct-2026). Hasta la primera imagen medida, el medidor es None."""
    if proc is None or not hasattr(proc, '_medidor'):
        return None                       # modo Tiendas: no hay anaqueles
    from ..analityc.core import anaqueles_eventos
    from ..analityc.core.anaqueles_registro import ultimo_registro

    medidor = getattr(proc, '_medidor', None)
    cfg = getattr(medidor, 'cfg', None) or {}
    ventana = float(cfg.get('stock_ventana_s', 180.0))
    esperando: Dict[int, List[float]] = {}
    for ag in list(getattr(medidor, '_agarres_por_confirmar', None) or []):
        try:
            esperando.setdefault(int(ag['idx']), []).append(
                max(0.0, ahora - float(ag['t'])))
        except (KeyError, TypeError, ValueError):
            continue

    registro = ultimo_registro(cam) or {}
    areas: Dict[int, Dict[str, Any]] = {}
    for clave, a in (registro.get('areas') or {}).items():
        try:
            idx = int(clave)
        except (TypeError, ValueError):
            continue
        areas[idx] = {'idx': idx, 'nombre': a.get('nombre') or f'Área {idx + 1}',
                      'visitas': int(a.get('visitas') or 0),
                      'confirmados': int(a.get('agarres') or 0),
                      'sin_confirmar': int(a.get('agarres_sin_confirmar') or 0)}
    for idx in esperando:
        # Un gesto en un área que aún no llegó al disco (volcado cada 5 s).
        areas.setdefault(idx, {'idx': idx, 'nombre': f'Área {idx + 1}',
                               'visitas': 0, 'confirmados': 0,
                               'sin_confirmar': 0})
    for idx, a in areas.items():
        esperas = esperando.get(idx, [])
        a['esperando'] = len(esperas)
        a['esperando_max_s'] = round(max(esperas), 1) if esperas else None

    fecha = registro.get('fecha') or time.strftime('%Y-%m-%d')
    ultimos = []
    for e in reversed(anaqueles_eventos.leer(cam, fecha) or []):
        if e.get('tipo') != 'agarre':
            continue
        ultimos.append({'hora': e.get('hora'), 'area': e.get('area_nombre'),
                        'producto': e.get('producto')})
        if len(ultimos) == 5:
            break
    return {'fecha': fecha, 'ventana_s': ventana,
            'areas': [areas[i] for i in sorted(areas)],
            'ultimos_confirmados': ultimos}


# ── GPU ──────────────────────────────────────────────────────────────────

def _fila_gpu(linea: str) -> Optional[Dict[str, Any]]:
    partes = [p.strip() for p in linea.lstrip('﻿').split(',')]
    if len(partes) < 8 or not partes[0][:1].isdigit():
        return None                       # la cabecera o una fila cortada

    def numero(texto: str) -> Optional[float]:
        try:
            return float(texto)
        except ValueError:
            return None                   # «[N/A]» de nvidia-smi
    try:
        t = time.mktime(time.strptime(partes[0], '%Y-%m-%d %H:%M:%S'))
    except ValueError:
        return None
    try:
        bits = int(partes[7], 16)
    except ValueError:
        bits = 0
    return {'t': t, 'temp_c': numero(partes[1]), 'uso_pct': numero(partes[2]),
            'memoria_mb': numero(partes[3]), 'consumo_w': numero(partes[4]),
            'reloj_mhz': numero(partes[5]), 'ventilador_pct': numero(partes[6]),
            'limitacion': partes[7],
            'motivos': [texto for bit, texto in MOTIVOS_GPU.items() if bits & bit],
            'por_calor': bool(bits & CALOR)}


def _gpu() -> Optional[Dict[str, Any]]:
    """Las últimas filas del CSV del vigilante. Se lee solo la cola del
    archivo: crece ~3 KB por hora y no rota."""
    try:
        with open(RUTA_GPU, 'rb') as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16 * 1024))
            cola = fh.read().decode('utf-8', errors='replace')
    except OSError:
        return None                       # el vigilante no corre (o Linux)
    filas = [f for f in (_fila_gpu(l) for l in cola.splitlines()[1:]) if f]
    if not filas:
        return None
    filas = filas[-FILAS_GPU:]
    return {'ultima': filas[-1],
            'historial': [{'t': f['t'], 'temp_c': f['temp_c'],
                           'uso_pct': f['uso_pct']} for f in filas]}


# ── El endpoint ──────────────────────────────────────────────────────────

def _camara(cam: str, proc: Any, ahora: float) -> Dict[str, Any]:
    salida: Dict[str, Any] = {'camera_id': cam, 'modo': _modo(proc),
                              'conectada': proc is not None}
    for clave, leer in (('video', lambda: _video_de(cam, proc, ahora)),
                        ('velocidad', lambda: _velocidad(cam)),
                        ('gente_ahora', lambda: _gente_ahora(cam, proc)),
                        ('gente_hoy', lambda: _gente_hoy(cam, ahora)),
                        ('agarres', lambda: _agarres(cam, proc, ahora))):
        try:
            salida[clave] = leer()
        except Exception as exc:                            # noqa: BLE001
            logger.debug("en vivo: %s de %s ilegible: %s", clave, cam, exc)
            salida[clave] = None
    return salida


@router.get('/en-vivo')
def en_vivo() -> Dict[str, Any]:
    """Todo lo de la página «En vivo» en una respuesta. `ahora` es la hora
    del servidor: la página calcula con ella los «hace X s», para que no
    dependan del reloj de quien la mira."""
    ahora = time.time()
    procs = _procesadores()
    camaras = sorted(set(procs) | set(telemetria_fps.camaras_vistas()),
                     key=lambda c: (c not in procs, c))
    try:
        gpu = _gpu()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("en vivo: GPU ilegible: %s", exc)
        gpu = None
    try:
        from . import recursos_pc
        pc = recursos_pc.resumen()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("en vivo: CPU y RAM ilegibles: %s", exc)
        pc = None
    return {'ahora': round(ahora, 1),
            'camaras': [_camara(c, procs.get(c), ahora) for c in camaras],
            'gpu': gpu, 'pc': pc}
