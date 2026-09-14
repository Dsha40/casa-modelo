"""Máquina de estados del visitante del stand (HITO 7).

```
DETECTADO
   └─> EN_ZONA_PASO
          ├─> PASO_DE_LARGO
          └─> SE_DETUVO
                 ├─> SE_RETIRO_SIN_ATENCION
                 └─> ATENDIDO
                        ├─> ATENDIDO_SIN_DEGUSTAR
                        └─> DEGUSTO   (terminal)
```

## Por qué máquina de estados explícita y no condicionales sueltos

Con `if` dispersos por el pipeline, la pregunta «¿por qué esta persona cuenta
como atendida?» no tiene respuesta: hay que reconstruirla leyendo código. Aquí
cada transición es un método, emite su evento con marca de tiempo, y el camino
completo queda en `stand_eventos`. Auditable persona a persona.

## Los flags son ACUMULATIVOS y MONÓTONOS

Quien degustó cuenta también como atendido, como detenido y como pasante. No
es un detalle cosmético: es lo que impide que el embudo se invierta. Por eso
`marcar_degustacion()` enciende los tres flags anteriores en vez de confiar en
que el recorrido los haya encendido — si el detector de degustación se dispara
sobre alguien a quien no vimos ser atendido, la cifra sigue siendo coherente.

Los flags **nunca se apagan**. Alguien que fue atendido y se va sigue siendo
«atendido» al cerrar: el embudo mide lo que ocurrió, no el estado final.

## Unidades

Trabaja en el espacio que le den. Si hay homografía, en metros; si no, en
píxeles normalizados 0..1, con los umbrales `*_px` convertidos por el llamador.
El módulo no decide eso — solo compara contra el umbral que recibe.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Estado:
    DETECTADO = "DETECTADO"
    EN_ZONA_PASO = "EN_ZONA_PASO"
    PASO_DE_LARGO = "PASO_DE_LARGO"
    SE_DETUVO = "SE_DETUVO"
    SE_RETIRO_SIN_ATENCION = "SE_RETIRO_SIN_ATENCION"
    ATENDIDO = "ATENDIDO"
    ATENDIDO_SIN_DEGUSTAR = "ATENDIDO_SIN_DEGUSTAR"
    DEGUSTO = "DEGUSTO"


# Estado terminal -> clasificación persistida (stand_almacen.Clasificacion).
CLASIFICACION_DE = {
    Estado.PASO_DE_LARGO: "paso_de_largo",
    Estado.SE_RETIRO_SIN_ATENCION: "se_detuvo_sin_atencion",
    Estado.ATENDIDO_SIN_DEGUSTAR: "atendido_sin_degustar",
    Estado.DEGUSTO: "degusto",
}


@dataclass
class _Punto:
    ts: float
    xy: Tuple[float, float]


@dataclass
class ResumenVisitante:
    """Lo que la máquina sabe al cerrar. Encaja con `stand_visitantes`."""
    persistent_id: str
    estado: str
    clasificacion: str
    se_detuvo: bool
    fue_atendido: bool
    degusto: bool
    dwell_max_s: float
    duracion_total_s: float
    ts_atencion: Optional[float]
    duracion_atencion_s: float
    ts_degustacion: Optional[float]
    confianza_degustacion: Optional[float]
    metodo_deteccion_degustacion: Optional[str]
    velocidad_media: Optional[float]
    distancia_recorrida: float
    puntos_detencion: List[Tuple[float, float]] = field(default_factory=list)


class MaquinaVisitante:
    """Una instancia por `persistent_id`."""

    def __init__(self, persistent_id: str, cfg: Any,
                 al_evento: Optional[Callable[[str, str, Dict], None]] = None,
                 usar_metros: bool = False) -> None:
        g = cfg.get
        self.persistent_id = persistent_id
        self._al_evento = al_evento

        self._dwell_min_s = float(g('detencion.dwell_min_s', 3.0))
        self._desplaz_max = (float(g('detencion.desplazamiento_max_m', 0.8))
                             if usar_metros
                             else float(g('detencion.desplazamiento_max_px', 60)))
        self._vel_paso_largo = float(g('paso_largo.velocidad_min_ms', 0.6))
        self._atencion_dist_max = float(g('atencion.distancia_max_m', 1.2))
        self._atencion_min_s = float(g('atencion.duracion_min_s', 2.0))

        self.estado = Estado.DETECTADO
        # Flags acumulativos: una vez encendidos, no se apagan.
        self.paso = False
        self.se_detuvo = False
        self.fue_atendido = False
        self.degusto = False

        self._historial: Deque[_Punto] = deque(maxlen=600)
        self._primera_ts: Optional[float] = None
        self._ultima_ts: Optional[float] = None
        self._dwell_stand_s = 0.0
        self._dwell_max_s = 0.0
        self._atencion_acum_s = 0.0
        self._ts_primera_atencion: Optional[float] = None
        self._ts_degustacion: Optional[float] = None
        self._conf_degustacion: Optional[float] = None
        self._metodo_degustacion: Optional[str] = None
        self._distancia = 0.0
        self._puntos_detencion: List[Tuple[float, float]] = []

    # ── Eventos ─────────────────────────────────────────────────────────
    def _emitir(self, tipo: str, zona: str = "", **payload: Any) -> None:
        if self._al_evento is None:
            return
        try:
            self._al_evento(tipo, zona, payload)
        except Exception:                                   # noqa: BLE001
            logger.debug("stand: callback de evento falló para %s",
                         self.persistent_id)

    # ── Actualización por frame ─────────────────────────────────────────
    def actualizar(self, *, ts: float,
                   punto_pie: Optional[Tuple[float, float]] = None,
                   en_zona_paso: bool = False,
                   en_zona_stand: bool = False,
                   en_zona_atencion: bool = False,
                   distancia_chef: Optional[float] = None,
                   chef_disponible: bool = False) -> str:
        """Un frame. Devuelve el estado tras procesarlo."""
        if self._primera_ts is None:
            self._primera_ts = ts
            self._emitir('entrada_escena')

        dt = 0.0 if self._ultima_ts is None else max(0.0, ts - self._ultima_ts)
        self._ultima_ts = ts

        if punto_pie is not None:
            if self._historial:
                previo = self._historial[-1].xy
                self._distancia += _dist(previo, punto_pie)
            self._historial.append(_Punto(ts, (float(punto_pie[0]),
                                                float(punto_pie[1]))))

        # ── DETECTADO -> EN_ZONA_PASO ──
        if en_zona_paso and not self.paso:
            self.paso = True
            if self.estado == Estado.DETECTADO:
                self.estado = Estado.EN_ZONA_PASO
            self._emitir('entrada_zona_stand', zona='zona_paso')

        # ── Permanencia en zona_stand -> SE_DETUVO ──
        if en_zona_stand:
            self._dwell_stand_s += dt
            self._dwell_max_s = max(self._dwell_max_s, self._dwell_stand_s)
            if (not self.se_detuvo
                    and self._dwell_stand_s >= self._dwell_min_s
                    and self._quieto()):
                self._marcar_detenido(ts, punto_pie)
        else:
            # Salir de la zona reinicia el contador de permanencia continua,
            # pero NO el flag: quien ya se detuvo, se detuvo.
            if self._dwell_stand_s > 0:
                self._emitir('salida_zona_stand', zona='zona_stand',
                             dwell_s=round(self._dwell_stand_s, 2))
            self._dwell_stand_s = 0.0

        # ── ATENDIDO ──
        cerca = (distancia_chef is not None
                 and distancia_chef <= self._atencion_dist_max)
        if cerca and chef_disponible and en_zona_atencion:
            self._atencion_acum_s += dt
            if (not self.fue_atendido
                    and self._atencion_acum_s >= self._atencion_min_s):
                self._marcar_atendido(ts)

        return self.estado

    def _quieto(self) -> bool:
        """¿Se movió poco durante la ventana de detención?

        Sin esta comprobación, alguien que camina despacio a través de
        `zona_stand` acumula 3 s dentro y se contaría como interesado.
        """
        if not self._historial:
            return True
        corte = self._historial[-1].ts - self._dwell_min_s
        ventana = [p.xy for p in self._historial if p.ts >= corte]
        if len(ventana) < 2:
            return True
        xs = [p[0] for p in ventana]
        ys = [p[1] for p in ventana]
        extension = max(max(xs) - min(xs), max(ys) - min(ys))
        return extension < self._desplaz_max

    # ── Transiciones ────────────────────────────────────────────────────
    def _marcar_detenido(self, ts: float,
                         punto: Optional[Tuple[float, float]]) -> None:
        self.se_detuvo = True
        self.paso = True                    # monotonía
        self.estado = Estado.SE_DETUVO
        if punto is not None:
            self._puntos_detencion.append(punto)
        self._emitir('detencion', zona='zona_stand',
                     dwell_s=round(self._dwell_stand_s, 2))

    def _marcar_atendido(self, ts: float) -> None:
        self.fue_atendido = True
        self.se_detuvo = True               # monotonía
        self.paso = True
        self.estado = Estado.ATENDIDO
        self._ts_primera_atencion = ts
        self._emitir('inicio_atencion', zona='zona_atencion',
                     acumulado_s=round(self._atencion_acum_s, 2))

    def marcar_degustacion(self, ts: float, confianza: float,
                           metodo: str) -> None:
        """Lo dispara el HITO 8 tras confirmar con pose + VLM."""
        if self.degusto:
            return
        self.degusto = True
        self.fue_atendido = True            # monotonía: los tres, sí o sí
        self.se_detuvo = True
        self.paso = True
        self.estado = Estado.DEGUSTO
        self._ts_degustacion = ts
        self._conf_degustacion = float(confianza)
        self._metodo_degustacion = metodo
        self._emitir('degustacion', zona='zona_atencion',
                     confianza=round(float(confianza), 3), metodo=metodo)

    # ── Cierre ──────────────────────────────────────────────────────────
    def cerrar(self, ts: Optional[float] = None) -> ResumenVisitante:
        """Fija el estado terminal. Idempotente."""
        # GOTCHA: `self._primera_ts or fin` estaba mal. Un timestamp de 0.0 es
        # FALSY, así que con relojes relativos (que arrancan en 0) el `or`
        # devolvía `fin` y la duración salía 0 para todos los visitantes. Lo
        # cazó test_terminal_paso_de_largo. Comparar contra None, siempre.
        if ts is not None:
            fin = float(ts)
        elif self._ultima_ts is not None:
            fin = float(self._ultima_ts)
        else:
            fin = 0.0
        inicio = self._primera_ts if self._primera_ts is not None else fin
        duracion = max(0.0, fin - inicio)

        if self.degusto:
            self.estado = Estado.DEGUSTO
        elif self.fue_atendido:
            self.estado = Estado.ATENDIDO_SIN_DEGUSTAR
        elif self.se_detuvo:
            self.estado = Estado.SE_RETIRO_SIN_ATENCION
        else:
            self.estado = Estado.PASO_DE_LARGO

        velocidad = (self._distancia / duracion) if duracion > 0 else None
        self._emitir('salida_escena', clasificacion=CLASIFICACION_DE[self.estado])

        return ResumenVisitante(
            persistent_id=self.persistent_id,
            estado=self.estado,
            clasificacion=CLASIFICACION_DE[self.estado],
            se_detuvo=self.se_detuvo,
            fue_atendido=self.fue_atendido,
            degusto=self.degusto,
            dwell_max_s=round(self._dwell_max_s, 2),
            duracion_total_s=round(duracion, 2),
            ts_atencion=self._ts_primera_atencion,
            duracion_atencion_s=round(self._atencion_acum_s, 2),
            ts_degustacion=self._ts_degustacion,
            confianza_degustacion=self._conf_degustacion,
            metodo_deteccion_degustacion=self._metodo_degustacion,
            velocidad_media=(round(velocidad, 3)
                             if velocidad is not None else None),
            distancia_recorrida=round(self._distancia, 3),
            puntos_detencion=list(self._puntos_detencion),
        )

    def campos_visitante(self, resumen: ResumenVisitante) -> Dict[str, Any]:
        """Traducción a los campos de `stand_visitantes`."""
        return {
            'se_detuvo': resumen.se_detuvo,
            'dwell_max_s': resumen.dwell_max_s,
            'fue_atendido': resumen.fue_atendido,
            'duracion_atencion_s': resumen.duracion_atencion_s,
            'degusto': resumen.degusto,
            'confianza_degustacion': resumen.confianza_degustacion,
            'metodo_deteccion_degustacion':
                resumen.metodo_deteccion_degustacion,
            'duracion_total_s': resumen.duracion_total_s,
            'clasificacion': resumen.clasificacion,
            'trayectoria_resumen': {
                'distancia_recorrida': resumen.distancia_recorrida,
                'velocidad_media': resumen.velocidad_media,
                'puntos_detencion': [list(p) for p in resumen.puntos_detencion],
            },
        }


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
