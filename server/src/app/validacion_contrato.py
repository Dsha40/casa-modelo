"""
src/app/validacion_contrato.py — Validacion del contrato en el servidor.

Paso 5 del plan de migracion del HITO 2. Conecta los esquemas de
`elde_core.contracts` al bucle del websocket, **detras de la capa de
compatibilidad**, de modo que el servidor entiende a la vez el formato antiguo
y el nuevo.

## Por que arranca en modo observacion

Rechazar de entrada seria romper los 4 clientes en produccion el mismo dia. El
comportamiento se controla con `ELDE_VALIDAR_CONTRATO`:

- `observar` (**por defecto**) — valida, registra lo que rechazaria y **deja
  pasar el mensaje**. Coste: una validacion Pydantic por frame. No cambia el
  comportamiento del servidor en absoluto.
- `estricto` — descarta el mensaje invalido y responde con un error claro. Es
  el objetivo del HITO 8, cuando los registros de `observar` esten limpios y
  los 4 clientes migrados.
- `apagado` — ni valida. Para descartar la validacion como sospechosa si algun
  dia hay que diagnosticar un problema de rendimiento.

Esa progresion es la que permite activar la validacion sin una ventana de
caida: primero se mira, luego se corta.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Por defecto `estricto` desde el 31-jul-2026 (decision del usuario, tras
# 7/8 pipelines validados con 0% de fallos en `observar` y los payloads
# reales capturados pasando con `extra='forbid'`). `observar` y `apagado`
# siguen disponibles por entorno para diagnosticar.
MODO = (os.getenv('ELDE_VALIDAR_CONTRATO', 'estricto') or 'estricto').strip().lower()
if MODO not in ('observar', 'estricto', 'apagado'):
    logger.warning("ELDE_VALIDAR_CONTRATO=%r no reconocido; se usa 'estricto'",
                   MODO)
    MODO = 'estricto'

SITE_ID = os.getenv('ELDE_SITE_ID', 'sitio-unico').strip() or 'sitio-unico'

_lock = threading.Lock()
_stats: Counter = Counter()
# Pipelines que se han visto de verdad. Sin esto, "sin errores" se confundia
# con "seguro para cortar": un solo cliente conectado puede dar 0% de fallos
# mientras los otros siete pipelines no se han probado nunca.
_pipelines: set = set()
# Un mismo error se repite 25 veces por segundo: se registra el primero de cada
# clase y despues solo se cuenta.
_ya_registrado: set = set()

try:
    from elde_core.contracts import (POR_EVENTO, Envelope, desde_antiguo,
                                     es_formato_antiguo)
    DISPONIBLE = True
except Exception as exc:                      # el servidor debe arrancar igual
    logger.warning("contrato no disponible (%s); la validacion queda apagada",
                   exc)
    DISPONIBLE = False


def activa() -> bool:
    return DISPONIBLE and MODO != 'apagado'


def _anota(clave: str, detalle: str = '') -> None:
    with _lock:
        _stats[clave] += 1
        nuevo = clave not in _ya_registrado
        if nuevo:
            _ya_registrado.add(clave)
    if nuevo and detalle:
        # Solo la primera vez de cada clase de error, para no inundar el log.
        logger.warning("contrato [%s] %s: %s", MODO, clave, detalle[:400])


def _solo_campos_desconocidos(exc: Exception) -> Optional[list]:
    """Los campos sobrantes, si el ÚNICO problema del payload es que trae
    claves que el contrato no declara. `None` si hay algún otro error.

    Se mira el detalle de Pydantic (`extra_forbidden`) y no el texto, que
    cambia entre versiones.
    """
    errores = getattr(exc, 'errors', None)
    if not callable(errores):
        return None
    try:
        detalle = exc.errors()
    except Exception:                             # noqa: BLE001
        return None
    if not detalle or any(e.get('type') != 'extra_forbidden' for e in detalle):
        return None
    return sorted({'.'.join(str(x) for x in e.get('loc', ())) for e in detalle})


def revisar(mensaje: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Valida un mensaje entrante.

    Devuelve `(aceptar, motivo)`. En modo `observar`, `aceptar` es siempre
    True: el motivo se registra pero el mensaje sigue su curso. Nunca lanza —
    un fallo de la propia validacion no puede tumbar una conexion.
    """
    if not activa():
        return True, None
    try:
        # ── 1. Traducir el formato antiguo, si lo es ──
        if es_formato_antiguo(mensaje):
            env, err = desde_antiguo(mensaje, site_id=SITE_ID)
            if env is None:
                _anota('antiguo_intraducible', err or '')
                return (MODO != 'estricto'), err
            _anota('traducido_del_formato_antiguo')
        else:
            try:
                env = Envelope(**mensaje)
            except Exception as exc:
                _anota('envelope_invalido', str(exc))
                return (MODO != 'estricto'), f"envelope invalido: {exc}"
            _anota('envelope_nuevo')

        # ── 2. Validar el payload contra el modelo de su evento ──
        modelo = POR_EVENTO.get(getattr(env.event_type, 'value',
                                        str(env.event_type)))
        if modelo is None:
            _anota('evento_sin_modelo', str(env.event_type))
            return True, None
        try:
            modelo(**env.payload)
        except Exception as exc:
            sobrantes = _solo_campos_desconocidos(exc)
            if sobrantes:
                # Un campo que el contrato no conoce NO ciega la camara
                # (7-sep-2026). `extra='forbid'` esta para que una clave
                # nueva o mal escrita se VEA, y se sigue viendo: queda en el
                # log y en los contadores. Pero descartar el frame entero
                # convertia cualquier herramienta nueva del recuadro en un
                # cliente muerto sin causa evidente — paso dos veces el mismo
                # dia, con `linea_cruce_estilo` y con `personal_zone_*`, y el
                # sintoma que llega es «le doy play y no detecta». El resto
                # de errores (tipo equivocado, campo obligatorio ausente,
                # imagen ilegible) siguen descartando el mensaje.
                _anota(f'campos_desconocidos:{env.pipeline}', ','.join(sobrantes))
                return True, None
            clave = f'payload_invalido:{env.pipeline}'
            _anota(clave, str(exc))
            return (MODO != 'estricto'), f"payload invalido: {exc}"

        _anota('valido')
        with _lock:
            _pipelines.add(getattr(env.pipeline, 'value', str(env.pipeline)))
        return True, None
    except Exception as exc:                  # la validacion jamas rompe nada
        _anota('error_interno_validacion', str(exc))
        return True, None


def resumen() -> Dict[str, Any]:
    """Contadores para exponer por `/health` y decidir cuando pasar a
    `estricto`."""
    with _lock:
        datos = dict(_stats)
        vistos = sorted(_pipelines)
    total = sum(datos.values()) or 1
    problemas = sum(v for k, v in datos.items()
                    if k.startswith(('payload_invalido', 'envelope_invalido',
                                     'antiguo_intraducible')))
    sin_errores = problemas == 0 and total > 1
    return {
        'modo': MODO,
        'disponible': DISPONIBLE,
        'site_id': SITE_ID,
        'contadores': datos,
        'mensajes_con_problema_pct': round(100.0 * problemas / total, 2),
        'sin_errores': sin_errores,
        # Que pipelines se han ejercitado de verdad. "Sin errores" NO basta
        # para cortar: un solo cliente conectado da 0% de fallos dejando el
        # resto sin probar. El total se DERIVA del contrato (antes era un 8
        # clavado a mano que se pudrio al nacer Estacionamiento).
        'pipelines_observados': vistos,
        'pipelines_totales': _total_pipelines(),
        'listo_para_estricto': sin_errores and len(vistos) >= _total_pipelines(),
        'para_cortar_falta': (
            'nada' if (sin_errores and len(vistos) >= _total_pipelines())
            else ('resolver los errores' if not sin_errores
                  else (f'ejercitar los {_total_pipelines() - len(vistos)} '
                        'pipelines restantes'))),
    }


def _total_pipelines() -> int:
    """Cuantos pipelines conoce el contrato (fuente de verdad: el enum)."""
    try:
        from elde_core.contracts.envelope import Pipeline
        return len(Pipeline)
    except Exception:
        return 9
