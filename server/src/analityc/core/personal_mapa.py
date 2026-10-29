"""Mapa de calor y permanencia PROPIOS del personal de camiseta (19-oct-2026).

Lo pidió el usuario: el empleado con camiseta (rol «pasillero» del modelo del
uniforme) no calienta el mapa de los clientes —eso ya era así— y ahora tiene
SU mapa y SU permanencia. Vigilante y mantenimiento no entran en ninguno.

- Mapa: un acumulador más del registro global, con la clave
  `<cámara>__personal`. Sale como una «cámara» más en el dashboard y en
  `/api/v1/heatmaps`, con su histórico por horas.
- Permanencia: por hora, segundos con AL MENOS un empleado dentro del área
  (`in_zone`; dos a la vez no suman doble) y segundos-persona (sí suman).
  Se guarda en `output/personal_permanencia/<cámara>/<AAAA-MM-DD>.json`
  como mucho cada `GUARDAR_S`. Hora local del servidor.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SUFIJO = '__personal'
ROLES = ('pasillero',)
GUARDAR_S = 30.0
#: Un hueco mayor entre imágenes (vídeo parado, corte) no se cuenta entero.
DT_MAX_S = 1.0
_RAIZ = Path(__file__).resolve().parents[3]


def es_de_camiseta(track: Optional[Dict[str, Any]]) -> bool:
    return bool(track and track.get('es_personal')
                and track.get('personal_rol') in ROLES)


def clave_mapa(camera_id: Any) -> str:
    return f'{camera_id}{SUFIJO}'


class PermanenciaPersonal:
    """Una por cámara. `observar` en cada imagen; nunca lanza."""

    def __init__(self, camera_id: Any, carpeta: Optional[Path] = None) -> None:
        from .anaqueles_registro import _carpeta_nombre
        self.camera_id = str(camera_id)
        self._carpeta = Path(carpeta) if carpeta else (
            _RAIZ / 'output' / 'personal_permanencia' / _carpeta_nombre(camera_id))
        self._t_prev: Optional[float] = None
        self._dia = ''
        self._horas: Dict[str, Dict[str, float]] = {}
        self._guardado = 0.0

    def observar(self, tracks: Dict[Any, Dict[str, Any]], t: Optional[float] = None) -> None:
        try:
            t = time.time() if t is None else float(t)
            dt = 0.0 if self._t_prev is None else min(max(0.0, t - self._t_prev), DT_MAX_S)
            self._t_prev = t
            dentro = sum(1 for tr in (tracks or {}).values()
                         if es_de_camiseta(tr) and tr.get('in_zone'))
            dia = time.strftime('%Y-%m-%d', time.localtime(t))
            if dia != self._dia:
                self.guardar()
                self._dia, self._horas = dia, self._leer(dia)
            if dentro and dt:
                h = self._horas.setdefault(time.strftime('%H', time.localtime(t)),
                                           {'con_personal_s': 0.0, 'persona_s': 0.0})
                h['con_personal_s'] += dt
                h['persona_s'] += dt * dentro
            if (t - self._guardado) >= GUARDAR_S:
                self.guardar(t)
        except Exception as exc:                            # noqa: BLE001
            logger.debug('permanencia del personal: %s', exc)

    def _ruta(self, dia: str) -> Path:
        return self._carpeta / f'{dia}.json'

    def _leer(self, dia: str) -> Dict[str, Dict[str, float]]:
        try:
            return json.loads(self._ruta(dia).read_text(encoding='utf-8'))['horas']
        except Exception:                                   # noqa: BLE001
            return {}

    def guardar(self, t: Optional[float] = None) -> None:
        if not self._dia:
            return
        self._guardado = time.time() if t is None else t
        try:
            self._carpeta.mkdir(parents=True, exist_ok=True)
            horas = {h: {k: round(v, 1) for k, v in d.items()} for h, d in sorted(self._horas.items())}
            tmp = self._ruta(self._dia).with_suffix('.tmp')
            tmp.write_text(json.dumps({
                'camera_id': self.camera_id, 'fecha': self._dia, 'horas': horas,
                'con_personal_s': round(sum(d['con_personal_s'] for d in self._horas.values()), 1),
                'persona_s': round(sum(d['persona_s'] for d in self._horas.values()), 1),
            }, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, self._ruta(self._dia))
        except Exception as exc:                            # noqa: BLE001
            logger.warning('permanencia del personal: no se pudo guardar: %s', exc)


_POR_CAMARA: Dict[str, PermanenciaPersonal] = {}
#: Cámaras en las que ya se vio personal de camiseta (solo esas guardan mapa).
_USADOS: set = set()


def permanencia(camera_id: Any) -> PermanenciaPersonal:
    clave = str(camera_id)
    if clave not in _POR_CAMARA:
        _POR_CAMARA[clave] = PermanenciaPersonal(camera_id)
    return _POR_CAMARA[clave]


def mapa(camera_id: Any, nombre: Optional[str] = None):
    """El acumulador del mapa del personal de esa cámara (registro global:
    se restaura de disco y lo vuelca el apagado como los demás)."""
    from .analytics import heatmap_registro
    _USADOS.add(str(camera_id))
    return heatmap_registro.obtener(clave_mapa(camera_id),
                                    f'{nombre or camera_id} (personal)')


def guardar_mapa(camera_id: Any, fondo: Any, nombre: Optional[str] = None,
                 personas: Optional[int] = None) -> None:
    """Instantánea periódica del mapa del personal, con el MISMO fondo que el
    de los clientes. Solo si en esa cámara ya hubo personal. Nunca lanza."""
    if str(camera_id) not in _USADOS:
        return
    try:
        mapa(camera_id, nombre).maybe_save_snapshot(
            clave_mapa(camera_id), background=fondo,
            camera_name=f'{nombre or camera_id} (personal)', personas=personas)
    except Exception as exc:                                # noqa: BLE001
        logger.debug('mapa del personal: %s', exc)
