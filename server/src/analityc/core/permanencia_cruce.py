"""Detector de cruces VISIBLES del ROI, con la dirección medida en la flecha.

## El hueco que llena

`PermanenciaArea` tenía dos lecturas del ROI y ninguna respondía «¿en qué
dirección pasó?» cuando la persona se ve todo el tiempo:

  - en `area`, entrar al ROI avisa, pero da igual hacia dónde caminara: un
    ROI en la entrada del local avisa lo mismo para quien entra que para
    quien sale;
  - en `puerta`, el cruce solo existe si el track NACE o MUERE dentro del
    vano. Quien atraviesa el ROI y sigue a la vista no genera nada.

Esta lectura es para el caso pedido el 8-ago-2026: el ROI es un umbral que la
gente atraviesa SIN desaparecer, y la flecha «hacia adentro» decide si el paso
es «Persona entrando al área» o «Persona saliendo del área».

## Cómo decide

Igual que el desempate de `puerta`, pero como señal PRINCIPAL: el
desplazamiento del punto-pie proyectado sobre la flecha, medido en ALTURAS DE
LA PROPIA PERSONA (el mismo umbral vale en 720p y en 4K, de cerca y de
lejos). Proyección >= umbral con el ROI tocado = entró; <= -umbral = salió.

Dos detalles que evitan falsos cruces:

  - **La base del recorrido se congela al tocar el ROI.** Mientras la persona
    no lo pisa, la base la sigue (ventana deslizante): así el paseo previo por
    el encuadre no envenena la proyección. Quien camina 3 alturas en contra de
    la flecha por el pasillo y LUEGO gira y entra, se mide desde el borde del
    ROI, no desde donde apareció.
  - **Cada emisión rearma la base en el punto actual.** Sin esto, quien entra
    (+0.8 alturas) y sale más tarde tendría que RETROCEDER más allá de su
    origen para acumular la proyección negativa; con el rearme, la salida se
    mide desde donde se le dio por entrado.

## Lo que este detector se NIEGA a hacer

  - **Sin flecha no emite nada.** La dirección aquí no se infiere: se mide.
    Se avisa una sola vez en el log para que el operador la dibuje.
  - Quien merodea DENTRO del ROI con pasitos cortos no supera el umbral en
    ninguna dirección y no genera cruces: el silencio es la respuesta honesta.

Emite los mismos `Cruce` que `permanencia_puerta`, así que aguas abajo
(estancias, fotos, alertas, WhatsApp/Jarvis, panel del cliente) todo funciona
sin tocar una línea.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .permanencia_puerta import (ENTRADA, POR_FLECHA, SALIDA, Cruce, _caja,
                                 _unitario, centro_pie)

logger = logging.getLogger(__name__)

DEFECTOS_CRUCE: Dict[str, Any] = {
    # Frames mínimos de vida antes de creerse nada de un track. Filtra los
    # parpadeos del detector, que nacen y mueren en el mismo sitio.
    'frames_minimos': 4,
    # Desplazamiento mínimo a lo largo de la flecha para creerse la dirección,
    # en ALTURAS DE LA PROPIA PERSONA. 0.5 = medio cuerpo.
    'desplazamiento_min_altura': 0.5,
    # La MISMA persona no repite el MISMO sentido antes de esto. Cubre el
    # churn del tracker (la pierde y la recupera en el umbral) sin ocultar
    # un reingreso real, que casi nunca baja de este tiempo.
    'no_reemitir_s': 10.0,
    # Un track que reaparece después de esto cuenta como paso nuevo.
    'reinicio_ausencia_s': 30.0,
}


@dataclass
class _Rastro:
    """Recorrido de un track mientras interesa."""
    clave: str
    primer_ts: float
    ultimo_ts: float
    frames_vistos: int = 0
    toco_zona: bool = False
    confirmado: bool = False
    box: Optional[Tuple[float, float, float, float]] = None
    #: Origen del tramo que se está midiendo. Sigue a la persona mientras no
    #: pise el ROI; se congela al tocarlo; se rearma tras cada emisión.
    base: Optional[Tuple[float, float]] = None
    actual: Optional[Tuple[float, float]] = None
    #: Altura MÁXIMA vista: una caja recortada por el borde no debe inflar el
    #: desplazamiento medido en alturas.
    alto_max: float = 0.0
    ultimo_tipo: str = ''
    ultimo_emitido_ts: float = 0.0


class DetectorCruce:
    """Convierte observaciones por frame en cruces direccionales del ROI."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        base = dict(DEFECTOS_CRUCE)
        base.update({k: v for k, v in (cfg or {}).items() if k in base})
        self.cfg = base
        self._rastros: Dict[str, _Rastro] = {}
        self._aviso_sin_flecha = 0.0

    # -- API ---------------------------------------------------------------
    def observar(self, ahora: float,
                 visibles: Dict[str, Dict[str, Any]],
                 flecha: Optional[Sequence[float]] = None) -> List[Cruce]:
        """Procesa un frame y devuelve los cruces confirmados en él.

        Args:
            ahora: marca de tiempo del frame.
            visibles: `clave -> {'en_zona': bool, 'box': (x1,y1,x2,y2),
                'confirmado': bool}` de los tracks vistos en ESTE frame.
            flecha: la flecha «hacia adentro» del recuadro (par de puntos o
                vector). Sin ella NO se emite nada: aquí la dirección se mide.

        Returns:
            Lista de `Cruce` (normalmente vacía). Nunca lanza.
        """
        try:
            vector = _unitario(flecha)
            if vector is None:
                # Cada 5 min, no cada frame: el log no es un metrónomo.
                if (ahora - self._aviso_sin_flecha) > 300:
                    self._aviso_sin_flecha = ahora
                    logger.warning(
                        "DetectorCruce: sin flecha de dirección dibujada no "
                        "se clasifica ningún paso. Dibújala en el recuadro "
                        "del cliente (menú ROI → Flecha de dirección).")
                return []
            cruces = self._avanzar(ahora, visibles, vector)
            self._olvidar(ahora, set(visibles))
            return cruces
        except Exception as exc:                            # noqa: BLE001
            logger.error("DetectorCruce: fallo observando (%s)", exc)
            return []

    def reset(self) -> None:
        self._rastros.clear()

    @property
    def rastreados(self) -> int:
        return len(self._rastros)

    # -- Piezas ------------------------------------------------------------
    def _avanzar(self, ahora: float, visibles: Dict[str, Dict[str, Any]],
                 vector: Tuple[float, float]) -> List[Cruce]:
        minimos = max(1, int(self.cfg['frames_minimos']))
        umbral = float(self.cfg['desplazamiento_min_altura'])
        no_reemitir = float(self.cfg['no_reemitir_s'])
        reinicio = float(self.cfg['reinicio_ausencia_s'])
        cruces: List[Cruce] = []

        for clave, obs in visibles.items():
            en_zona = bool(obs.get('en_zona'))
            box = _caja(obs.get('box'))
            centro = centro_pie(box)
            r = self._rastros.get(clave)

            if r is None or (ahora - r.ultimo_ts) > reinicio:
                r = _Rastro(clave=clave, primer_ts=ahora, ultimo_ts=ahora,
                            base=centro)
                self._rastros[clave] = r

            r.ultimo_ts = ahora
            r.frames_vistos += 1
            if bool(obs.get('confirmado')):
                r.confirmado = True
            if box is not None:
                r.box = box
                r.actual = centro
                r.alto_max = max(r.alto_max, abs(box[3] - box[1]))
                if not r.toco_zona:
                    # Ventana deslizante hasta pisar el ROI: el tramo se mide
                    # desde el borde, no desde donde apareció la persona.
                    r.base = centro
            if en_zona:
                r.toco_zona = True

            if not r.toco_zona or r.frames_vistos < minimos:
                continue
            proyeccion = self._proyeccion(r, vector)
            if proyeccion is None:
                continue
            if proyeccion >= umbral:
                tipo = ENTRADA
            elif proyeccion <= -umbral:
                tipo = SALIDA
            else:
                continue

            if (tipo == r.ultimo_tipo
                    and (ahora - r.ultimo_emitido_ts) < no_reemitir):
                r.base = r.actual        # rearmar sin duplicar el aviso
                continue

            r.ultimo_tipo = tipo
            r.ultimo_emitido_ts = ahora
            r.base = r.actual            # el siguiente tramo empieza aquí
            cruces.append(Cruce(
                clave=clave, tipo=tipo, ts=ahora, box=r.box,
                frames=r.frames_vistos, confirmado=r.confirmado,
                via=POR_FLECHA, proyeccion=proyeccion))
        return cruces

    def _proyeccion(self, r: _Rastro,
                    vector: Tuple[float, float]) -> Optional[float]:
        """Avance a lo largo de la flecha, en alturas de la persona.

        `None` = sin recorrido o persona tan pequeña que su altura no sirve
        de escala. La respuesta honesta es «no lo sé», no un cero.
        """
        if r.base is None or r.actual is None or r.alto_max < 8.0:
            return None
        dx = r.actual[0] - r.base[0]
        dy = r.actual[1] - r.base[1]
        return (dx * vector[0] + dy * vector[1]) / r.alto_max

    def _olvidar(self, ahora: float, visibles_ahora: set) -> None:
        """Tira los rastros viejos (cota de memoria)."""
        reinicio = float(self.cfg['reinicio_ausencia_s'])
        muertos = [c for c, r in self._rastros.items()
                   if c not in visibles_ahora
                   and (ahora - r.ultimo_ts) > reinicio]
        for clave in muertos:
            self._rastros.pop(clave, None)
        if len(self._rastros) > 500:
            viejos = sorted(self._rastros.values(),
                            key=lambda r: r.ultimo_ts)[:len(self._rastros) - 500]
            for r in viejos:
                self._rastros.pop(r.clave, None)
