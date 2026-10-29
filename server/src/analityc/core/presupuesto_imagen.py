"""Presupuesto por imagen: como mucho N tareas pesadas por persona en cada
imagen (26-oct-2026). APAGADO por defecto: es una PRUEBA.

## Por qué

Cronómetro por etapas del 26-oct (vídeo, 35 min, 24.218 imágenes): la imagen
vacía tarda ~63-72 ms y cada persona añade trabajo, pero no siempre. Con 4
personas, las imágenes RÁPIDAS (<= 77 ms) no tenían ninguna tarea por persona
y las LENTAS (> 100 ms) sumaban pose 18 ms, género (MiVOLO) 12,5, huella del
cuerpo (OSNet) 12 y caras (YuNet) 9: las tareas van por turnos fijos (cada X
imágenes por persona) y a veces COINCIDEN en la misma imagen. Eso son las
caídas de imágenes por segundo, no el número de personas (el usuario lo vio:
con 4 personas a veces iba a 13-14).

## Qué hace

Cada tarea pesada pide permiso antes de hacerse (`toca`). En cada imagen se
conceden como mucho `MAX_POR_IMAGEN`; las demás quedan PENDIENTES y se hacen
en las imágenes siguientes, por orden de llegada (mientras haya pendientes,
una tarea nueva espera su turno). No se quita trabajo: se reparte.

Tareas que pasan por aquí: «genero» (buscar cara + MiVOLO, `_classify_person`)
y «huella» (OSNet del reidentificador corporal). La POSE NO: retrasarla podría
perder un agarre.

Coste: el género y la huella llegan algo más tarde (con 5 personas, unas
pocas imágenes); quien pasa muy deprisa puede tener menos lecturas.

## Interruptor

`ELDE_PRESUPUESTO_IMAGEN=1` (o `AnalyticsConfig.PRESUPUESTO_IMAGEN`) lo
enciende; apagado, `toca` devuelve lo mismo que pedía el turno de siempre.
Cada 60 s, si está encendido, una línea en el log:

    PRESUPUESTO tareas hechas=120 aplazadas=35 espera media 1,4 imágenes (máx 6)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

#: Tareas pesadas concedidas por imagen.
MAX_POR_IMAGEN = 1
#: Una tarea pendiente que no se vuelve a pedir en estas imágenes (la persona
#: ya no está) se olvida.
OLVIDAR_TRAS = 3
RESUMEN_S = 60.0
_ESTADO = '_presupuesto_imagen'


def activo() -> bool:
    v = (os.getenv('ELDE_PRESUPUESTO_IMAGEN') or '').strip().lower()
    if v:
        return v in ('1', 'true', 'si', 'sí', 'on')
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'PRESUPUESTO_IMAGEN', False))
    except Exception:                                       # noqa: BLE001
        return False


def _estado(proc: Any) -> Dict[str, Any]:
    est = getattr(proc, _ESTADO, None)
    if est is None:
        est = {'n': 0, 'libres': MAX_POR_IMAGEN, 'pend': {}, 'hechas': 0,
               'aplazadas': 0, 'esperas': [], 'desde': time.monotonic()}
        try:
            setattr(proc, _ESTADO, est)
        except Exception:                                   # noqa: BLE001
            pass
    return est


def nueva_imagen(proc: Any) -> None:
    """Al empezar cada imagen. Nunca lanza."""
    if not activo():
        return
    try:
        est = _estado(proc)
        est['n'] += 1
        est['libres'] = MAX_POR_IMAGEN
        pend: Dict[Tuple[Any, str], Dict[str, int]] = est['pend']
        for k in [k for k, v in pend.items() if est['n'] - v['pedida'] > OLVIDAR_TRAS]:
            pend.pop(k, None)
        ahora = time.monotonic()
        if ahora - est['desde'] >= RESUMEN_S:
            esp = est['esperas']
            logger.info("PRESUPUESTO tareas hechas=%d aplazadas=%d espera media %.1f "
                        "imágenes (máx %d)", est['hechas'], est['aplazadas'],
                        (sum(esp) / len(esp)) if esp else 0.0, max(esp) if esp else 0)
            est.update(hechas=0, aplazadas=0, esperas=[], desde=ahora)
    except Exception as exc:                                # noqa: BLE001
        logger.debug("presupuesto: %s", exc)


def toca(proc: Any, tid: Any, tarea: str, debida: bool) -> bool:
    """¿Se hace AHORA la tarea `tarea` de la persona `tid`? `debida` es lo
    que decidía el turno de siempre. Apagado, devuelve `debida`."""
    if not activo():
        return bool(debida)
    try:
        est = _estado(proc)
        clave = (tid, tarea)
        pend = est['pend']
        pendiente = clave in pend
        if not (debida or pendiente):
            return False
        # Mientras haya pendientes, las tareas nuevas esperan su turno.
        if est['libres'] > 0 and (pendiente or not pend):
            est['libres'] -= 1
            est['hechas'] += 1
            if pendiente:
                est['esperas'].append(est['n'] - pend.pop(clave)['desde'])
            return True
        if pendiente:
            pend[clave]['pedida'] = est['n']
        else:
            pend[clave] = {'desde': est['n'], 'pedida': est['n']}
            est['aplazadas'] += 1
        return False
    except Exception as exc:                                # noqa: BLE001
        logger.debug("presupuesto: %s", exc)
        return bool(debida)
