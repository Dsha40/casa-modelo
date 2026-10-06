"""src/analityc/core/tiempos_etapas.py — cronómetro por etapas, imagen a imagen.

## Para qué (6-oct-2026)

py-spy reparte el tiempo por MUESTRAS: da porcentajes de todo un minuto, y
pasarlos a ms por imagen es una cuenta con ±2 ms que no distingue la imagen
con la clienta en el estante de la imagen vacía. Para decir con seguridad
qué frena cuando hay gente junto al estante, esto cronometra cada etapa en
CADA imagen y lo guarda con cuántas personas había y a cuántas se les
calculó la pose.

## Cómo

`ELDE_TIEMPOS_ETAPAS=<ruta absoluta de un .jsonl>` al arrancar el servidor.
Sin la variable no se toca nada. Con ella, se envuelven (en memoria, al
arrancar) las funciones de cada etapa con un cronómetro. Cada etapa cuenta
su tiempo PROPIO: si una etapa cronometrada llama a otra, el tiempo de la
de dentro se le resta a la de fuera, así la suma de etapas = el total y
nada se cuenta dos veces. Lo que no cae en ninguna etapa va a «resto».

Una línea por imagen: `{"t", "total_ms", "etapas": {...}, "personas",
"pose_personas"}`. Diagnóstico: no dejarlo puesto (una línea por imagen).
"""
from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VARIABLE = 'ELDE_TIEMPOS_ETAPAS'

#: (módulo, clase o None, función, etapa). El orden no importa: cada etapa
#: cuenta solo su tiempo propio.
_OBJETIVOS = [
    ('src.app.app', None, '_full_frame_sync', 'TOTAL'),
    ('src.app.app', None, '_decode_image', 'abrir JPEG'),
    ('src.analityc.core.deteccion_compartida', '_DetectorCompartido',
     'detectar', 'detector'),
    ('supervision', 'ByteTrack', 'update_with_detections', 'ByteTrack'),
    ('src.analityc.core.anaqueles_pose', 'PoseMunecas', 'munecas_lote',
     'pose'),
    ('src.analityc.core.anaqueles_pose', 'PoseMunecas', 'munecas', 'pose'),
    ('src.analityc.core.anaqueles', 'MedidorAnaqueles', 'procesar',
     'anaqueles (resto)'),
    ('src.analityc.core.estacionamiento_escena', 'DetectorEscena', 'evaluar',
     'comprobar escena'),
    ('src.analityc.core.tiendas_modelos', '_GpuYuNet', 'detect',
     'caras: detectar (YuNet)'),
    ('src.analityc.core.analytics.face_reidentifier', 'FaceReidentifier',
     'compute_embedding', 'caras: huella (embedding)'),
    ('src.analityc.core.analytics.face_reidentifier', 'FaceReidentifier',
     'identify_or_register', 'caras: identidad (resto)'),
    ('src.analityc.core.analytics.body_reidentifier', 'BodyReidentifier',
     'compute_embedding', 'cuerpo: huella (OSNet)'),
    ('src.analityc.core.analytics.estimador_edad_genero',
     'EstimadorEdadGenero', 'estimar_lote', 'edad/género (MiVOLO)'),
    ('src.analityc.core.analytics.estimador_edad_genero',
     'EstimadorEdadGenero', 'estimar', 'edad/género (MiVOLO)'),
    ('src.analityc.core.analytics.demographics', 'DemographicsClassifier',
     'classify', 'demografía (resto)'),
    ('src.analityc.core.analytics.heatmap', 'HeatmapAccumulator',
     'observar_escena', 'mapa de calor'),
    ('src.analityc.core.analytics.heatmap', 'HeatmapAccumulator',
     'add_person', 'mapa de calor'),
    ('src.analityc.core.analytics.heatmap', 'HeatmapAccumulator',
     'maybe_save_snapshot', 'mapa de calor'),
    ('src.analityc.core.tiendas_fotos', 'FotosMixin',
     '_capture_established_tracks', 'fichas y fotos'),
    ('src.analityc.core.tiendas_fotos', 'FotosMixin',
     '_update_captures_demographics', 'fichas y fotos'),
    ('src.analityc.core.anaqueles_stock', 'VigilanteStock', 'vigilar',
     'stock'),
]

_local = threading.local()
_fichero = None
_candado = threading.Lock()


def _pila() -> List[List[Any]]:
    p = getattr(_local, 'pila', None)
    if p is None:
        p = _local.pila = []
    return p


def _imagen() -> Optional[Dict[str, Any]]:
    return getattr(_local, 'imagen', None)


def anotar(clave: str, n: int) -> None:
    """Suma `n` a un contador de la imagen en curso (personas, poses)."""
    img = _imagen()
    if img is not None:
        img[clave] = img.get(clave, 0) + int(n)


def _envolver(funcion, etapa: str):
    @functools.wraps(funcion)
    def cronometrada(*args, **kwargs):
        pila = _pila()
        es_total = etapa == 'TOTAL' and not pila
        if es_total:
            _local.imagen = {'etapas': {}}
        if etapa == 'pose' and funcion.__name__ == 'munecas_lote' \
                and len(args) > 2:
            anotar('pose_personas', len(args[2]))
        elif etapa == 'anaqueles (resto)' and len(args) > 2 \
                and isinstance(args[2], dict):
            # Personas que ve el anaquel en esta imagen (tracks de persona).
            anotar('personas', sum(1 for tr in args[2].values()
                                   if tr.get('class', -1) == -1))
        marco = [etapa, time.perf_counter(), 0.0]   # etapa, inicio, hijos
        pila.append(marco)
        try:
            return funcion(*args, **kwargs)
        finally:
            pila.pop()
            dur = time.perf_counter() - marco[1]
            img = _imagen()
            if img is not None:
                propio = (dur - marco[2]) * 1000.0
                if etapa != 'TOTAL':
                    e = img['etapas']
                    e[etapa] = e.get(etapa, 0.0) + propio
                else:
                    img['etapas']['resto'] = \
                        img['etapas'].get('resto', 0.0) + propio
            if pila:
                pila[-1][2] += dur
            if es_total and img is not None:
                img['total_ms'] = dur * 1000.0
                img['t'] = time.time()
                _escribir(img)
                _local.imagen = None
    cronometrada._elde_cronometro = True
    return cronometrada


def _escribir(img: Dict[str, Any]) -> None:
    try:
        img['etapas'] = {k: round(v, 2) for k, v in img['etapas'].items()}
        img['total_ms'] = round(img['total_ms'], 2)
        with _candado:
            _fichero.write(json.dumps(img, ensure_ascii=False) + '\n')
            _fichero.flush()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("tiempos_etapas: no se pudo escribir: %s", exc)


def instalar() -> bool:
    """Envuelve las etapas si `ELDE_TIEMPOS_ETAPAS` está puesta. Nunca
    lanza; True si quedó instalado."""
    global _fichero
    ruta = os.environ.get(VARIABLE, '').strip()
    if not ruta or _fichero is not None:
        return False
    try:
        _fichero = open(ruta, 'a', encoding='utf-8')
    except Exception as exc:                                # noqa: BLE001
        logger.error("tiempos_etapas: no se pudo abrir %s: %s", ruta, exc)
        return False
    puestas = []
    # El paquete raíz con el que corre el servidor ('src' al arrancar con
    # iniciar_servidor_headless.py): los módulos se buscan con ese nombre,
    # o se envolvería una copia que el servidor no usa.
    raiz = __name__.split('.analityc')[0]
    for modulo, clase, nombre, etapa in _OBJETIVOS:
        if modulo.startswith('src.'):
            modulo = raiz + modulo[3:]
        try:
            m = importlib.import_module(modulo)
            dueno = getattr(m, clase) if clase else m
            f = getattr(dueno, nombre)
            if getattr(f, '_elde_cronometro', False):
                continue
            setattr(dueno, nombre, _envolver(f, etapa))
            puestas.append(f'{clase + "." if clase else ""}{nombre}')
        except Exception as exc:                            # noqa: BLE001
            logger.warning("tiempos_etapas: sin cronómetro en %s.%s.%s: %s",
                           modulo, clase, nombre, exc)
    logger.warning("tiempos_etapas: CRONOMETRANDO %d funciones en %s",
                   len(puestas), ruta)
    return True
