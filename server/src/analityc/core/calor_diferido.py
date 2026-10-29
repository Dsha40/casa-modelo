"""Mapa de calor DIFERIDO por seguimiento (22-oct-2026).

Por qué: el empleado solo dejaba de calentar el mapa de CLIENTES desde que el
modelo del uniforme lo marcaba (3 votos), y lo de antes se quedaba en el de
clientes. Medido el 22-oct en el vídeo de la cámara 3: 1.439 de 5.658 s del
personal marcado (25 %) fueron ANTES de la marca, un ~20 % de calor de más en
el mapa de clientes, justo en los anaqueles donde trabaja.

Cómo: el calor de cada seguimiento se guarda aparte y se decide su destino
cuando el seguimiento termina (o cada `ESPERA_MAX_S`, para no retrasar
demasiado el mapa en vivo ni acumular memoria):
  - marcado como personal de camiseta -> mapa del personal (`personal_mapa`);
  - marcado como vigilante o mantenimiento -> ninguno;
  - no marcado -> mapa de clientes.
Coste: el mapa se actualiza con unos segundos de retraso.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Se reparte lo pendiente de un seguimiento aunque siga vivo, cada tanto.
ESPERA_MAX_S = 60.0
#: Sin verlo este tiempo, el seguimiento se da por terminado.
TERMINADO_S = 2.0

Caja = Tuple[Any, int, int]                       # (box, ancho, alto)


class CalorDiferido:
    """Uno por cámara. Nunca lanza."""

    def __init__(self) -> None:
        #: tid -> {'desde': t, 'visto': t, 'cajas': [...]}
        self._pendiente: Dict[Any, Dict[str, Any]] = {}

    def anotar(self, tid: Any, box: Any, ancho: int, alto: int,
               t: Optional[float] = None) -> None:
        t = time.time() if t is None else float(t)
        p = self._pendiente.setdefault(tid, {'desde': t, 'visto': t, 'cajas': []})
        p['visto'] = t
        p['cajas'].append((box, int(ancho), int(alto)))

    def repartir(self, tracks: Dict[Any, Dict[str, Any]], cliente: Any,
                 personal: Callable[[], Any], es_de_camiseta: Callable[[dict], bool],
                 t: Optional[float] = None, todo: bool = False) -> int:
        """Reparte lo de los seguimientos terminados o esperando de más.
        `personal()` da el acumulador del mapa del personal (perezoso).
        Devuelve cuántas cajas se repartieron."""
        t = time.time() if t is None else float(t)
        n = 0
        for tid in list(self._pendiente):
            p = self._pendiente[tid]
            # Se recuerda lo que era mientras se le ve: al terminar, el
            # seguimiento ya no está en `tracks` y se perdería la marca.
            vivo = (tracks or {}).get(tid)
            if vivo is not None and vivo.get('es_personal'):
                p['personal'] = (True, bool(es_de_camiseta(vivo)))
            terminado = tid not in tracks or (t - p['visto']) > TERMINADO_S
            if not (todo or terminado or (t - p['desde']) >= ESPERA_MAX_S):
                continue
            self._pendiente.pop(tid)
            es_pers, camiseta = p.get('personal', (False, False))
            try:
                if not es_pers:
                    destino = cliente
                elif camiseta:
                    destino = personal()
                else:
                    destino = None                 # vigilante, mantenimiento
                if destino is not None:
                    for box, ancho, alto in p['cajas']:
                        destino.add_person(box, ancho, alto)
                n += len(p['cajas'])
            except Exception as exc:               # noqa: BLE001
                logger.debug('calor diferido: %s', exc)
            if not terminado and not todo:
                # Sigue vivo: lo que venga después se decide aparte.
                self._pendiente[tid] = {'desde': t, 'visto': p['visto'], 'cajas': [],
                                        'personal': p.get('personal', (False, False))}
        return n


_POR_CAMARA: Dict[str, CalorDiferido] = {}


def de(camera_id: Any) -> CalorDiferido:
    clave = str(camera_id)
    if clave not in _POR_CAMARA:
        _POR_CAMARA[clave] = CalorDiferido()
    return _POR_CAMARA[clave]
