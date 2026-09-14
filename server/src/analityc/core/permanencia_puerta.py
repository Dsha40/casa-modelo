"""Detector de cruces de puerta: quién entró y quién salió, viendo solo el marco.

## El problema que resuelve

En un área normal la persona está a la vista mientras dure la permanencia, y
basta con leer el estado DENTRO/FUERA del pipeline. En una puerta no se ve el
interior: quien entra **desaparece del encuadre**, y para la máquina de zonas
eso es indistinguible de haberse marchado por donde vino. Cronometrar con ese
estado daría siempre cero.

## La señal que sí funciona: dónde nace y dónde muere el track

Con la cámara mirando al marco, el sentido del cruce está en el ciclo de vida
del seguimiento, no en la geometría:

  - **Entró** = el track venía de fuera de la zona, sus últimas apariciones
    son *dentro* del marco, y ahí **se acaba**. No se fue: se metió.
  - **Salió** = el track **empieza** dentro del marco (nació ahí, no venía de
    ningún lado del encuadre) y después se aleja hacia fuera de la zona.

Es simétrico y no necesita saber qué lado es «adentro», que es justo lo que la
cámara no puede ver.

## La flecha, cuando el operador la dibuja

El ciclo de vida del track dice *que* hubo un cruce; la flecha «hacia adentro»
dice *hacia dónde*. Con ella, la decisión deja de ser una inferencia sobre la
ausencia y pasa a ser una medida: el producto escalar entre el desplazamiento
de la persona y el vector. Positivo = se metió, negativo = se fue.

La flecha **confirma y desempata**, no sustituye:

  - un cruce que el ciclo de vida da por entrada, pero cuyo desplazamiento
    apunta hacia afuera, se descarta (antes se contaba);
  - el caso que sin flecha había que callar —nace y muere dentro del vano— con
    ella sí se decide, si el desplazamiento es lo bastante grande.

El umbral no va en píxeles sino en **alturas de la propia persona**: así el
mismo ajuste vale para una cámara de 720p y para una de 4K, y para alguien que
cruza al fondo y alguien que cruza en primer plano.

## Lo que este detector se NIEGA a decidir

Sin flecha, quien se queda parado en el vano sin llegar a salir de la zona no
genera ni entrada ni salida. Nació dentro (parecería salida) y muere dentro
(parecería entrada): las dos lecturas son igual de defendibles, así que no se
reporta ninguna. Un dato ambiguo silenciado es mejor que dos alertas
contradictorias.

Con `emitir_ambiguos`, en vez de callar se emite un cruce de tipo `AMBIGUO`
para que alguien de fuera lo arbitre —el VLM, fuera del bucle—. Sigue sin
inventarse una dirección aquí dentro.

## Ajustes cuando el sitio no se porta

  - **Se pierden entradas**: probablemente el ROI cubre casi todo el pasillo y
    la gente nunca llega a verse *fuera* de la zona. Pon
    `exigir_fuera_de_zona: false`.
  - **Sobran cruces**: sube `frames_confirmacion` (más evidencia por cruce) y
    `segundos_ausencia` (más paciencia antes de dar un track por muerto).

Ninguno de los dos umbrales está incrustado: salen de `permanencia_config.yaml`.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENTRADA = 'entrada'
SALIDA = 'salida'
#: Hubo cruce, pero ni el ciclo de vida ni la flecha alcanzan para decir hacia
#: dónde. Solo se emite con `emitir_ambiguos`; lo resuelve el árbitro.
AMBIGUO = 'ambiguo'

#: Cómo se tomó la decisión. Viaja hasta la alerta: quien la lee debe poder
#: saber si el dato es una medida o una inferencia.
POR_FLECHA = 'flecha'
POR_CICLO = 'ciclo_de_vida'

DEFECTOS_PUERTA: Dict[str, Any] = {
    # Frames seguidos en el nuevo estado antes de dar un cruce por bueno.
    'frames_confirmacion': 3,
    # Sin ver un track este tiempo, se da por muerto (se metió o se fue).
    'segundos_ausencia': 2.0,
    # Un track que reaparece después de esto cuenta como nuevo, no como
    # continuación. Cubre la oclusión corta sin fabricar cruces.
    'reinicio_ausencia_s': 30.0,
    # Frames mínimos de vida antes de creerse nada de un track. Filtra los
    # parpadeos del detector, que nacen y mueren en el mismo sitio.
    'frames_minimos': 4,
    # Exigir haber estado FUERA de la zona en algún momento. Ver el docstring.
    'exigir_fuera_de_zona': True,
    # Desplazamiento mínimo a lo largo de la flecha para creerse la dirección,
    # medido en ALTURAS DE LA PROPIA PERSONA. 0.5 = medio cuerpo. En alturas y
    # no en píxeles para que el mismo ajuste valga en 720p y en 4K, de cerca
    # y de lejos.
    'desplazamiento_min_altura': 0.5,
    # Emitir cruces AMBIGUO en vez de callarlos (para el árbitro externo).
    'emitir_ambiguos': False,
}


@dataclass
class Cruce:
    """Un paso confirmado por el marco de la puerta."""
    clave: str                       # persistent_id o 'prov:<track>'
    tipo: str                        # ENTRADA | SALIDA | AMBIGUO
    ts: float
    box: Optional[Tuple[float, float, float, float]] = None
    frames: int = 0
    confirmado: bool = False         # ¿la identidad venía de rostro/cuerpo?
    via: str = POR_CICLO             # POR_FLECHA | POR_CICLO
    # Desplazamiento a lo largo de la flecha, en alturas de la persona.
    # `None` cuando no había flecha que proyectar.
    proyeccion: Optional[float] = None


@dataclass
class _Huella:
    """Rastro de un track mientras vive. Se tira cuando deja de importar."""
    clave: str
    nacio_en_zona: bool
    primer_ts: float
    ultimo_ts: float
    box: Optional[Tuple[float, float, float, float]] = None
    frames_vistos: int = 0
    estuvo_fuera_zona: bool = False
    ultimas: Deque[bool] = field(default_factory=lambda: deque(maxlen=16))
    fuera_seguidos: int = 0
    emitio_salida: bool = False
    emitio_entrada: bool = False
    confirmado: bool = False
    # Recorrido, para proyectar sobre la flecha.
    centro_inicial: Optional[Tuple[float, float]] = None
    centro_actual: Optional[Tuple[float, float]] = None
    # Altura típica del cuerpo: la escala con la que se mide «cuánto se movió».
    # Se usa la MÁXIMA vista para que una caja momentáneamente recortada por
    # el borde del encuadre no infle artificialmente el desplazamiento.
    alto_max: float = 0.0

    def desplazamiento(self) -> Optional[Tuple[float, float]]:
        if self.centro_inicial is None or self.centro_actual is None:
            return None
        return (self.centro_actual[0] - self.centro_inicial[0],
                self.centro_actual[1] - self.centro_inicial[1])


class DetectorPuerta:
    """Convierte observaciones por frame en cruces de puerta."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        base = dict(DEFECTOS_PUERTA)
        base.update({k: v for k, v in (cfg or {}).items() if k in base})
        self.cfg = base
        self._huellas: Dict[str, _Huella] = {}

    # -- API ---------------------------------------------------------------
    def observar(self, ahora: float,
                 visibles: Dict[str, Dict[str, Any]],
                 flecha: Optional[Sequence[float]] = None) -> List[Cruce]:
        """Procesa un frame y devuelve los cruces confirmados en él.

        Args:
            ahora: marca de tiempo del frame.
            visibles: `clave -> {'en_zona': bool, 'box': (x1,y1,x2,y2),
                'confirmado': bool}` de los tracks vistos en ESTE frame.
            flecha: `(dx, dy)` unitario apuntando HACIA ADENTRO, o `None` si
                el operador no la dibujó (entonces decide el ciclo de vida).

        Returns:
            Lista de `Cruce` (normalmente vacía). Nunca lanza.
        """
        try:
            vector = _unitario(flecha)
            cruces = self._salidas(ahora, visibles, vector)
            cruces.extend(self._entradas(ahora, set(visibles), vector))
            self._olvidar(ahora, set(visibles))
            return cruces
        except Exception as exc:                            # noqa: BLE001
            logger.error("DetectorPuerta: fallo observando (%s)", exc)
            return []

    # -- Proyección sobre la flecha ---------------------------------------
    def _proyeccion(self, huella: _Huella,
                    vector: Optional[Tuple[float, float]]) -> Optional[float]:
        """Cuánto avanzó la persona a lo largo de la flecha, en alturas suyas.

        `None` = no hay flecha, o no hay recorrido, o la persona es tan
        pequeña en la imagen que su altura no sirve de escala. En los tres
        casos la respuesta honesta es «no lo sé», no un cero.
        """
        if vector is None:
            return None
        desplazamiento = huella.desplazamiento()
        if desplazamiento is None or huella.alto_max < 8.0:
            return None
        return ((desplazamiento[0] * vector[0]
                 + desplazamiento[1] * vector[1]) / huella.alto_max)

    def _veredicto(self, proyeccion: Optional[float]) -> Optional[str]:
        """`ENTRADA`, `SALIDA` o `None` si no llega al umbral."""
        if proyeccion is None:
            return None
        umbral = float(self.cfg['desplazamiento_min_altura'])
        if proyeccion >= umbral:
            return ENTRADA
        if proyeccion <= -umbral:
            return SALIDA
        return None

    def reset(self) -> None:
        self._huellas.clear()

    @property
    def rastreados(self) -> int:
        return len(self._huellas)

    # -- Piezas ------------------------------------------------------------
    def _salidas(self, ahora: float, visibles: Dict[str, Dict[str, Any]],
                 vector: Optional[Tuple[float, float]]) -> List[Cruce]:
        """Nació dentro del marco y ya se alejó -> salió."""
        confirmacion = max(1, int(self.cfg['frames_confirmacion']))
        minimos = max(1, int(self.cfg['frames_minimos']))
        reinicio = float(self.cfg['reinicio_ausencia_s'])
        cruces: List[Cruce] = []

        for clave, obs in visibles.items():
            en_zona = bool(obs.get('en_zona'))
            box = _caja(obs.get('box'))
            h = self._huellas.get(clave)

            if h is None or (ahora - h.ultimo_ts) > reinicio:
                # Track nuevo (o tan viejo que ya no es el mismo paso).
                h = _Huella(clave=clave, nacio_en_zona=en_zona,
                            primer_ts=ahora, ultimo_ts=ahora,
                            centro_inicial=centro_pie(box))
                self._huellas[clave] = h

            h.ultimo_ts = ahora
            h.frames_vistos += 1
            h.ultimas.append(en_zona)
            if box is not None:
                h.box = box
                centro = centro_pie(box)
                if h.centro_inicial is None:
                    h.centro_inicial = centro
                h.centro_actual = centro
                h.alto_max = max(h.alto_max, abs(box[3] - box[1]))
            if bool(obs.get('confirmado')):
                h.confirmado = True
            if not en_zona:
                h.estuvo_fuera_zona = True
                h.fuera_seguidos += 1
            else:
                h.fuera_seguidos = 0

            if (h.nacio_en_zona and not h.emitio_salida
                    and h.frames_vistos >= minimos
                    and h.fuera_seguidos >= confirmacion):
                proyeccion = self._proyeccion(h, vector)
                veredicto = self._veredicto(proyeccion)
                if veredicto == ENTRADA:
                    # La flecha contradice al ciclo de vida: nació en el vano
                    # pero se ha ido HACIA ADENTRO. No es una salida; se deja
                    # vivo para que su muerte se evalúe como entrada.
                    continue
                h.emitio_salida = True
                # Ya salió: no puede además «entrar» cuando el track muera
                # fuera del encuadre.
                h.emitio_entrada = True
                cruces.append(Cruce(
                    clave=clave, tipo=SALIDA, ts=ahora, box=h.box,
                    frames=h.frames_vistos, confirmado=h.confirmado,
                    via=POR_FLECHA if veredicto == SALIDA else POR_CICLO,
                    proyeccion=proyeccion))
        return cruces

    def _entradas(self, ahora: float, visibles_ahora: set,
                  vector: Optional[Tuple[float, float]]) -> List[Cruce]:
        """Última vez visto dentro del marco y desapareció -> entró."""
        ausencia = float(self.cfg['segundos_ausencia'])
        confirmacion = max(1, int(self.cfg['frames_confirmacion']))
        minimos = max(1, int(self.cfg['frames_minimos']))
        exigir_fuera = bool(self.cfg['exigir_fuera_de_zona'])
        emitir_ambiguos = bool(self.cfg['emitir_ambiguos'])
        cruces: List[Cruce] = []

        for clave, h in self._huellas.items():
            if clave in visibles_ahora or h.emitio_entrada:
                continue
            if (ahora - h.ultimo_ts) < ausencia:
                continue                    # todavía puede reaparecer
            if h.frames_vistos < minimos:
                h.emitio_entrada = True     # parpadeo del detector: se ignora
                continue
            ultimas = list(h.ultimas)[-confirmacion:]
            if len(ultimas) < confirmacion or not all(ultimas):
                h.emitio_entrada = True     # murió fuera del marco: se fue
                continue

            h.emitio_entrada = True
            proyeccion = self._proyeccion(h, vector)
            veredicto = self._veredicto(proyeccion)

            if veredicto == SALIDA:
                # Murió dentro del vano, pero venía alejándose: no se metió.
                continue

            if veredicto == ENTRADA:
                # La flecha lo dice: entró. Vale incluso sin haberse visto
                # nunca fuera de la zona, que es el caso que antes se callaba.
                cruces.append(Cruce(
                    clave=clave, tipo=ENTRADA, ts=h.ultimo_ts, box=h.box,
                    frames=h.frames_vistos, confirmado=h.confirmado,
                    via=POR_FLECHA, proyeccion=proyeccion))
                continue

            if exigir_fuera and not h.estuvo_fuera_zona:
                # Nunca se le vio fuera y la flecha no desempata (o no hay).
                if emitir_ambiguos:
                    cruces.append(Cruce(
                        clave=clave, tipo=AMBIGUO, ts=h.ultimo_ts, box=h.box,
                        frames=h.frames_vistos, confirmado=h.confirmado,
                        proyeccion=proyeccion))
                continue

            cruces.append(Cruce(clave=clave, tipo=ENTRADA, ts=h.ultimo_ts,
                                box=h.box, frames=h.frames_vistos,
                                confirmado=h.confirmado, via=POR_CICLO,
                                proyeccion=proyeccion))
        return cruces

    def _olvidar(self, ahora: float, visibles_ahora: set) -> None:
        """Tira las huellas ya resueltas y viejas (cota de memoria)."""
        reinicio = float(self.cfg['reinicio_ausencia_s'])
        muertas = [c for c, h in self._huellas.items()
                   if c not in visibles_ahora
                   and (ahora - h.ultimo_ts) > reinicio
                   and h.emitio_entrada]
        for clave in muertas:
            self._huellas.pop(clave, None)
        if len(self._huellas) > 500:
            viejas = sorted(self._huellas.values(),
                            key=lambda h: h.ultimo_ts)[:len(self._huellas) - 500]
            for h in viejas:
                self._huellas.pop(h.clave, None)


# ── Geometría de apoyo ──────────────────────────────────────────────────
def _unitario(flecha: Any) -> Optional[Tuple[float, float]]:
    """Normaliza lo que llegue del cliente. `None` si no sirve.

    Acepta las dos formas: el par de puntos `[[x1,y1],[x2,y2]]` que manda el
    recuadro, o un vector `(dx, dy)` ya calculado. Una flecha de longitud cero
    devuelve `None`: apuntar a ninguna parte no es una dirección.
    """
    if flecha is None:
        return None
    try:
        datos = list(flecha)
        if len(datos) < 2:
            return None
        if isinstance(datos[0], (list, tuple)):
            dx = float(datos[1][0]) - float(datos[0][0])
            dy = float(datos[1][1]) - float(datos[0][1])
        else:
            dx, dy = float(datos[0]), float(datos[1])
        largo = math.hypot(dx, dy)
        if largo < 1e-6:
            return None
        return (dx / largo, dy / largo)
    except (TypeError, ValueError, IndexError):
        return None


def _caja(valor: Any) -> Optional[Tuple[float, float, float, float]]:
    if valor is None:
        return None
    try:
        return (float(valor[0]), float(valor[1]),
                float(valor[2]), float(valor[3]))
    except (TypeError, ValueError, IndexError):
        return None


def centro_pie(box: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    """Punto que representa a la persona: el centro de sus pies.

    Es el mismo ancla que usa el pipeline para decidir `in_zone`; usar el
    centroide del cuerpo pondría a una persona alta «dentro» de la puerta un
    metro antes de pisarla.
    """
    caja = _caja(box)
    if caja is None:
        return None
    x1, y1, x2, y2 = caja
    return ((x1 + x2) / 2.0, y2)


def diagonal(box: Optional[Sequence[float]]) -> float:
    caja = _caja(box)
    if caja is None:
        return 0.0
    x1, y1, x2, y2 = caja
    return math.hypot(x2 - x1, y2 - y1)
