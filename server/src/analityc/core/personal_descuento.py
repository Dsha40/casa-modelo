"""Descontar a quien resulta ser PERSONAL a mitad de su seguimiento (19-oct-2026).

Por qué: el modelo del uniforme (`clasificador_uniforme.py`) marca a un
empleado tras 3 votos, y para entonces ya pudo dejar rastro de cliente.
Medido con los datos del 16-oct (17:06-18:34, modo activo): las 8 personas
marcadas con identidad confirmada tenían su fila en `tienda_visitas`, y 40
fichas eran de personal marcado (17 contaban como identificadas).

Qué hace, al marcar el seguimiento `tid`:
  1. su identidad confirmada pasa a `_pids_personal`: sale de la máquina
     DENTRO/FUERA sin emitir salida, de los únicos, y se borran su visita y
     sus eventos de la jornada en la base (por la misma cola: también lo aún
     sin volcar). Otro seguimiento suyo, más tarde, ya no entra al ROI;
  2. su ficha se APARTA (no se borra: el modelo se equivoca a veces) a
     `output/captures/personal_uniforme/`; el índice de capturas la quita
     solo al no verla en `persons/`, y su clasificación se borra;
  3. se tira su género/edad acumulado y, desde entonces, `no_clasificar`
     evita gastar caras y MiVOLO en él (`_classify_person`).
Los `prov:` no llegan nunca a la base (`_anotar_zona`): no hay nada que borrar.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

from .analytics.config import AnalyticsConfig
from . import tienda_almacen as _almacen

logger = logging.getLogger(__name__)

CARPETA = 'personal_uniforme'


def es_personal(proc: Any, pid: Any) -> bool:
    return bool(pid) and pid in (getattr(proc, '_pids_personal', None) or ())


def no_clasificar(proc: Any, tid: Any) -> bool:
    """El seguimiento es personal (marcado, o con una identidad ya marcada):
    no se le calcula género ni edad."""
    tr = (getattr(proc, 'active_tracks', None) or {}).get(tid) or {}
    return bool(tr.get('es_personal')) or es_personal(proc, tr.get('persistent_id'))


def _apartar_ficha(stem: str) -> int:
    base = AnalyticsConfig.CAPTURE_DIR
    destino = os.path.join(base, CARPETA)
    movidos = 0
    for sub, ext in (('persons', '.jpg'), ('persons', '.json'), ('faces', '.jpg')):
        src = os.path.join(base, sub, f'{stem}{ext}')
        if not os.path.isfile(src):
            continue
        try:
            os.makedirs(destino, exist_ok=True)
            nombre = f'{stem}{ext}' if sub == 'persons' else f'{stem}_cara{ext}'
            shutil.move(src, os.path.join(destino, nombre))
            movidos += 1
        except OSError as exc:
            logger.warning('personal: no se pudo apartar %s: %s', src, exc)
    cdir = getattr(AnalyticsConfig, 'CAPTURE_CLIENT_DIR', '') or ''
    for ext in ('.jpg', '.json'):
        p = os.path.join(cdir, f'{stem}{ext}') if cdir else ''
        try:
            if p and os.path.isfile(p):
                os.remove(p)            # copia anotada del cliente: sobra
        except OSError:
            pass
    return movidos


def descontar(proc: Any, tid: Any, camera_id: Any, ts: float | None = None) -> None:
    """Nunca lanza: un fallo aquí no puede tumbar el frame."""
    try:
        ts = time.time() if ts is None else ts
        tr = (getattr(proc, 'active_tracks', None) or {}).get(tid) or {}
        pid = tr.get('persistent_id')
        confirmado = bool(pid) and not str(pid).startswith('prov:')
        alm = _almacen.get_almacen()
        if confirmado:
            if not hasattr(proc, '_pids_personal'):
                proc._pids_personal = set()
            proc._pids_personal.add(pid)
            alm.quitar_persona(camera_id, pid, ts)
        if pid:
            # También un `prov:`: con `contar_sin_rostro` entra a la máquina
            # de zonas, y su tiempo dentro iría a la permanencia del panel en
            # vivo (`dashboard._dwell_by_person`) aunque nunca llegue a la base.
            for nombre in ('_confirmed_unique', '_unique_entered'):
                conj = getattr(proc, nombre, None)
                if conj is not None:
                    conj.discard(pid)
            zona = getattr(proc, '_zone_state', None)
            if zona is not None:
                zona.pop(pid, None)
        # Lo que llevaba acumulado de género/edad se tira: no se clasifica.
        demo = getattr(proc, '_demographics', None)
        if demo is not None:
            demo.remove_track(tid)
        meta = getattr(proc, '_capture_meta', None) or {}
        stem = meta.get(tid)
        via = 'propia'
        if not stem and confirmado:
            stem = (getattr(proc, '_captured_persons', None) or {}).get(pid)
            via = 'por_identidad'
        if stem:
            # EN SOMBRA (23-oct-2026): ¿la protegería la regla de los 2
            # primeros votos? Solo se apunta; la ficha se aparta igual.
            try:
                from . import clasificador_uniforme
                clasificador_uniforme.ficha_protegida(camera_id, stem, tid, via,
                                                      str(tr.get('personal_via') or ''))
            except Exception:                               # noqa: BLE001
                pass
            meta.pop(tid, None)
            hechos = getattr(proc, '_captured_demo_done', None)
            if hechos is not None:
                hechos.add(tid)
            n = _apartar_ficha(stem)
            alm.quitar_clasificacion(stem)
            logger.info('personal: ficha %s apartada (%d archivos)', stem, n)
        if confirmado:
            logger.info('personal: %s sin visita ni conteos de hoy', pid)
    except Exception as exc:                                # noqa: BLE001
        logger.warning('personal: no se pudo descontar %s: %s', tid, exc)


def tras_costura(proc: Any, tid: Any, camera_id: Any, ts: float | None = None) -> bool:
    """Tras una unión de `costura_sombra.py` (22-oct-2026): si el seguimiento
    nuevo HEREDÓ la marca de personal (`personal_via == 'costura'`), la misma
    limpieza que si lo marcara el uniforme. Medido en vivo el 22-oct a las
    12:41:22: el número nuevo de un empleado abrió ficha de cliente, heredó
    la marca y la ficha se quedó en `persons/`. Devuelve si descontó."""
    tr = (getattr(proc, 'active_tracks', None) or {}).get(tid) or {}
    if tr.get('personal_via') != 'costura':
        return False
    descontar(proc, tid, camera_id, ts)
    return True
