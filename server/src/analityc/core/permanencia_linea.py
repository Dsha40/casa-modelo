"""Detector de LÍNEA de cruce: entra o sale según de qué lado a qué lado pasó.

## Por qué una línea y no la flecha

La lectura `cruce` mide el desplazamiento ACUMULADO proyectado sobre la
flecha: exige medio cuerpo de avance en una dirección recta. En una cámara
gran angular apuntando a una esquina (el caso que la rompió, 10-ago-2026:
la puerta de Kitchen Productions) la gente entra CURVÁNDOSE alrededor del
muro; parte del recorrido va en contra de la flecha aunque estén entrando, y
el tracker suele perderlos en el marco antes de acumular el umbral.

Una puerta ES una línea: su umbral en el suelo. Aquí la decisión es binaria y
local — ¿de qué lado estabas y de qué lado estás? — y se toma EN el cruce,
sin acumular recorrido y sin que importe hacia dónde curve la trayectoria
después.

## La «puerta» del recuadro (5-sep-2026)

El cliente tiene un botón «⇅ Puerta»: se traza el umbral igual que la línea
y encima aparece una flecha doble que lo atraviesa por el centro —cabeza
verde ENTRADA hacia donde se entra, roja SALIDA hacia donde se sale—, y el
doble clic la invierte sin moverla del sitio. Es la MISMA geometría: el eje
de esa flecha es la normal de esta línea, e invertirla es intercambiar P1 y
P2, que es justo lo que ya significa aquí. Por eso no hay detector nuevo ni
modo nuevo: del recuadro solo llegan dos campos más, `linea_cruce_activate`
(el interruptor) y `linea_cruce_estilo` ('linea' o 'puerta'), y el estilo no
entra en ninguna cuenta — viaja hasta la alerta como `umbral_dibujado_con`
para que el registro conteste qué veía el operador si hay que revisar una
calibración al revés. Ver `app.umbral_del_payload`.

## Convención de lados

La línea llega como `[[x1,y1],[x2,y2]]` en píxeles del frame. Caminando de
P1 hacia P2, **DENTRO queda a la DERECHA** (en coordenadas de imagen, con la
`y` hacia abajo). El cliente pinta los rótulos DENTRO/FUERA con esta misma
regla, así que el operador nunca razona sobre la convención: la ve.
Invertir la línea = intercambiar P1 y P2.

## Cómo decide

  - **Histéresis**: solo un lado FIRME cuenta — estar a más de
    `margen_lado_altura` alturas de la persona de la línea. El jitter del
    tracker justo sobre el umbral no fabrica cruces.
  - **Cruce geométrico**: lado firme A → lado firme B, con el trayecto
    cortando el SEGMENTO dibujado (± `extension_segmento` de propina en las
    puntas). Cruzar la prolongación imaginaria de la línea, lejos de la
    puerta, no cuenta.
  - **Desempate de vano** (la herencia de `puerta`): el tracker pierde gente
    justo en el marco. Un track que MUERE a menos de `umbral_vano_altura`
    alturas del segmento viniendo firme de un lado, cruzó hacia el otro; uno
    que NACE junto al segmento y se afirma en un lado, cruzó hacia ese lado.
    Cada desempate tiene su interruptor en la config.
  - **Cooldown**: la misma persona no repite el mismo sentido antes de
    `no_reemitir_s` (churn del tracker en el umbral).

## Lo que este detector se NIEGA a hacer

  - **Sin línea no emite nada.** Se avisa una vez en el log para que el
    operador la dibuje.
  - Quien merodea pegado al umbral sin afirmarse en el otro lado no genera
    cruces: el silencio es la respuesta honesta.

Emite los mismos `Cruce` que `permanencia_puerta`, así que aguas abajo
(estancias, fotos, alertas, WhatsApp/Jarvis, panel del cliente) todo funciona
sin tocar una línea.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .permanencia_puerta import ENTRADA, SALIDA, Cruce, _caja, centro_pie

logger = logging.getLogger(__name__)

#: Cómo se decidió: cruzó la línea (o murió/nació en su vano).
POR_LINEA = 'linea'

DEFECTOS_LINEA: Dict[str, Any] = {
    # Frames mínimos de vida antes de creerse nada de un track. Filtra los
    # parpadeos del detector, que nacen y mueren en el mismo sitio.
    'frames_minimos': 4,
    # Histéresis: distancia mínima a la línea para dar un lado por FIRME,
    # en ALTURAS DE LA PROPIA PERSONA (vale igual en 720p y en 4K).
    'margen_lado_altura': 0.15,
    # Propina en las puntas del segmento, como fracción de su largo: cruzar
    # justo al filo del marco también cuenta.
    'extension_segmento': 0.15,
    # La MISMA persona no repite el MISMO sentido antes de esto.
    'no_reemitir_s': 10.0,
    # Un track que reaparece después de esto cuenta como paso nuevo.
    'reinicio_ausencia_s': 30.0,
    # Sin ver un track este tiempo, se da por muerto (entró o se marchó).
    'segundos_ausencia': 2.0,
    # Desempates tipo `puerta` para el track que el tracker pierde o
    # encuentra JUSTO en el marco. Distancia máxima al segmento (en alturas)
    # para considerar que murió/nació «en el vano».
    'umbral_vano_altura': 0.5,
    'desempate_muerte': True,
    'desempate_nacimiento': True,
}


@dataclass
class _Paso:
    """Estado de un track respecto a la línea, mientras interesa."""
    clave: str
    primer_ts: float
    ultimo_ts: float
    frames_vistos: int = 0
    confirmado: bool = False
    box: Optional[Tuple[float, float, float, float]] = None
    alto_max: float = 0.0
    #: Último lado FIRME (+1 dentro, -1 fuera). `None` = aún sin afirmarse.
    lado_firme: Optional[int] = None
    #: Posición del pie la última vez que estuvo firme (ancla del cruce).
    pos_firme: Optional[Tuple[float, float]] = None
    #: Dónde y a qué distancia (en alturas) NACIÓ el track: el origen contra
    #: el que se miden los dos desempates.
    pos_nacimiento: Optional[Tuple[float, float]] = None
    dist_nacimiento_alturas: Optional[float] = None
    #: Cruce visto ANTES de `frames_minimos`: queda pendiente y se emite en
    #: cuanto el track acumula la vida mínima, si sigue existiendo. Un
    #: parpadeo de verdad muere antes y se filtra solo.
    pendiente: Optional[str] = None
    #: Última posición y distancia vistas (para el desempate de muerte).
    ultima_pos: Optional[Tuple[float, float]] = None
    ultima_dist_alturas: Optional[float] = None
    ultimo_tipo: str = ''
    ultimo_emitido_ts: float = 0.0
    resuelto_muerte: bool = False


class DetectorLinea:
    """Convierte observaciones por frame en cruces de la línea."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        base = dict(DEFECTOS_LINEA)
        base.update({k: v for k, v in (cfg or {}).items() if k in base})
        self.cfg = base
        self._pasos: Dict[str, _Paso] = {}
        self._aviso_sin_linea = 0.0

    # -- API ---------------------------------------------------------------
    def observar(self, ahora: float,
                 visibles: Dict[str, Dict[str, Any]],
                 linea: Optional[Sequence[Any]] = None) -> List[Cruce]:
        """Procesa un frame y devuelve los cruces confirmados en él.

        Args:
            ahora: marca de tiempo del frame.
            visibles: `clave -> {'box': (x1,y1,x2,y2), 'confirmado': bool}`
                de los tracks vistos en ESTE frame (`en_zona` se ignora: la
                línea no necesita ROI).
            linea: `[[x1,y1],[x2,y2]]` en píxeles del frame. Sin ella NO se
                emite nada.

        Returns:
            Lista de `Cruce` (normalmente vacía). Nunca lanza.
        """
        try:
            geo = _geometria(linea)
            if geo is None:
                if (ahora - self._aviso_sin_linea) > 300:
                    self._aviso_sin_linea = ahora
                    logger.warning(
                        "DetectorLinea: sin línea de cruce dibujada no se "
                        "clasifica ningún paso. Dibújala en el recuadro del "
                        "cliente (botón «Línea» / menú ROI).")
                return []
            cruces = self._avanzar(ahora, visibles, geo)
            cruces.extend(self._muertes(ahora, set(visibles), geo))
            self._olvidar(ahora, set(visibles))
            return cruces
        except Exception as exc:                            # noqa: BLE001
            logger.error("DetectorLinea: fallo observando (%s)", exc)
            return []

    def reset(self) -> None:
        self._pasos.clear()

    @property
    def rastreados(self) -> int:
        return len(self._pasos)

    # -- Piezas ------------------------------------------------------------
    def _avanzar(self, ahora: float, visibles: Dict[str, Dict[str, Any]],
                 geo: '_Linea') -> List[Cruce]:
        minimos = max(1, int(self.cfg['frames_minimos']))
        margen = float(self.cfg['margen_lado_altura'])
        reinicio = float(self.cfg['reinicio_ausencia_s'])
        nacimiento_on = bool(self.cfg['desempate_nacimiento'])
        ext = float(self.cfg['extension_segmento'])
        cruces: List[Cruce] = []

        for clave, obs in visibles.items():
            box = _caja(obs.get('box'))
            pie = centro_pie(box)
            p = self._pasos.get(clave)

            if p is None or (ahora - p.ultimo_ts) > reinicio:
                p = _Paso(clave=clave, primer_ts=ahora, ultimo_ts=ahora)
                self._pasos[clave] = p

            p.ultimo_ts = ahora
            p.frames_vistos += 1
            p.resuelto_muerte = False
            if bool(obs.get('confirmado')):
                p.confirmado = True
            if box is None or pie is None:
                continue
            p.box = box
            p.ultima_pos = pie
            # Altura MÁXIMA vista: una caja recortada por el borde no debe
            # encoger la escala y convertir jitter en «lado firme».
            p.alto_max = max(p.alto_max, abs(box[3] - box[1]))
            if p.alto_max < 8.0:
                continue                    # tan pequeña que no hay escala

            dist = geo.distancia_firmada(pie)
            p.ultima_dist_alturas = abs(dist) / p.alto_max
            if p.pos_nacimiento is None:
                p.pos_nacimiento = pie
                p.dist_nacimiento_alturas = p.ultima_dist_alturas

            # Cruce visto de recién nacido: se emite en cuanto el track
            # junta la vida mínima. Descartarlo sin más perdía el paso para
            # siempre aunque la persona siguiera a la vista.
            if p.pendiente and p.frames_vistos >= minimos:
                cruce = self._emitir(p, p.pendiente, ahora, dist)
                p.pendiente = None
                if cruce is not None:
                    cruces.append(cruce)

            if abs(dist) < margen * p.alto_max:
                continue                    # zona de histéresis: sin lado
            lado = 1 if dist > 0 else -1

            if p.lado_firme is None:
                # Desempate de nacimiento: nació SIN lado firme (en la banda
                # del umbral) y se afirma por primera vez. Solo es un cruce
                # si el trayecto desde el nacimiento CORTA el segmento —
                # afirmarse del mismo lado en el que nació es jitter, no un
                # paso, y emitirlo contaba salidas de gente parada.
                if nacimiento_on and geo.trayecto_cruza(
                        p.pos_nacimiento, pie, ext):
                    self._cruce_o_pendiente(p, lado, ahora, dist, minimos,
                                            cruces)
                p.lado_firme = lado
                p.pos_firme = pie
                continue

            if lado == p.lado_firme:
                # El ancla se desliza: el próximo cruce se mide desde la
                # última posición firme, no desde una de hace una hora.
                p.pos_firme = pie
                continue

            # Cambio de lado firme: ¿el trayecto corta el SEGMENTO?
            if geo.trayecto_cruza(p.pos_firme, pie, ext):
                self._cruce_o_pendiente(p, lado, ahora, dist, minimos,
                                        cruces)
            # Cruzara el segmento o su prolongación, el lado nuevo es un
            # hecho: sin esto, quien rodea la línea por fuera emitiría un
            # cruce fantasma al volver.
            p.lado_firme = lado
            p.pos_firme = pie
        return cruces

    def _cruce_o_pendiente(self, p: _Paso, lado: int, ahora: float,
                           dist: float, minimos: int,
                           cruces: List[Cruce]) -> None:
        """Emite el cruce, o lo deja pendiente si el track es muy joven."""
        tipo = ENTRADA if lado > 0 else SALIDA
        if p.pendiente and p.pendiente != tipo:
            # Ida y vuelta antes de la vida mínima: un parpadeo que se
            # cancela solo. Emitir cualquiera de los dos sería inventar.
            p.pendiente = None
            return
        if p.frames_vistos >= minimos:
            cruce = self._emitir(p, tipo, ahora, dist)
            if cruce is not None:
                cruces.append(cruce)
        else:
            p.pendiente = tipo

    def _muertes(self, ahora: float, visibles_ahora: set,
                 geo: '_Linea') -> List[Cruce]:
        """El tracker lo perdió JUNTO al segmento: cruzó hacia el otro lado.

        SOLO para pasos que no emitieron nada NI tienen un cruce pendiente:
        quien acaba de cruzar geométricamente y muere al otro lado (el muro
        lo oculta enseguida, el caso normal de la esquina) NO está
        volviendo — inferirle el regreso cancelaría el cruce real con uno
        fantasma.

        Y SOLO con evidencia de aproximación: la distancia final tiene que
        ser claramente menor que la del nacimiento. La persona PARADA junto
        a la puerta a la que el churn del tracker le renueva la clave nace y
        muere a la misma distancia, y sin esta guarda cada re-track emitía
        un cruce fantasma que ningún cooldown podía frenar.
        """
        if not bool(self.cfg['desempate_muerte']):
            return []
        ausencia = float(self.cfg['segundos_ausencia'])
        minimos = max(1, int(self.cfg['frames_minimos']))
        margen = float(self.cfg['margen_lado_altura'])
        vano = float(self.cfg['umbral_vano_altura'])
        ext = float(self.cfg['extension_segmento'])
        cruces: List[Cruce] = []

        for clave, p in self._pasos.items():
            if clave in visibles_ahora or p.resuelto_muerte:
                continue
            if (ahora - p.ultimo_ts) < ausencia:
                continue                    # todavía puede reaparecer
            p.resuelto_muerte = True
            if (p.ultimo_tipo or p.pendiente is not None
                    or p.lado_firme is None
                    or p.frames_vistos < minimos
                    or p.ultima_pos is None
                    or p.ultima_dist_alturas is None
                    or p.dist_nacimiento_alturas is None
                    or p.ultima_dist_alturas > vano
                    or (p.ultima_dist_alturas + margen
                        > p.dist_nacimiento_alturas)
                    or not geo.dentro_del_segmento(p.ultima_pos, ext)):
                continue
            # Venía firme de un lado, se ACERCÓ y se esfumó en el umbral:
            # pasó al otro lado.
            tipo = ENTRADA if -p.lado_firme > 0 else SALIDA
            cruce = self._emitir(p, tipo, p.ultimo_ts, None)
            if cruce is not None:
                cruces.append(cruce)
        return cruces

    def _emitir(self, p: _Paso, tipo: str, ts: float,
                dist: Optional[float]) -> Optional[Cruce]:
        """Aplica el cooldown y construye el `Cruce`. `None` = suprimido."""
        if (tipo == p.ultimo_tipo
                and (ts - p.ultimo_emitido_ts)
                < float(self.cfg['no_reemitir_s'])):
            return None
        p.ultimo_tipo = tipo
        p.ultimo_emitido_ts = ts
        proyeccion = None
        if dist is not None and p.alto_max >= 8.0:
            proyeccion = dist / p.alto_max
        return Cruce(clave=p.clave, tipo=tipo, ts=ts, box=p.box,
                     frames=p.frames_vistos, confirmado=p.confirmado,
                     via=POR_LINEA, proyeccion=proyeccion)

    def _olvidar(self, ahora: float, visibles_ahora: set) -> None:
        """Tira los pasos viejos (cota de memoria)."""
        reinicio = float(self.cfg['reinicio_ausencia_s'])
        muertos = [c for c, p in self._pasos.items()
                   if c not in visibles_ahora
                   and (ahora - p.ultimo_ts) > reinicio]
        for clave in muertos:
            self._pasos.pop(clave, None)
        if len(self._pasos) > 500:
            viejos = sorted(self._pasos.values(),
                            key=lambda p: p.ultimo_ts)[:len(self._pasos) - 500]
            for p in viejos:
                self._pasos.pop(p.clave, None)


# ── Geometría ───────────────────────────────────────────────────────────
@dataclass
class _Linea:
    """El segmento parseado: origen, dirección unitaria y largo."""
    x1: float
    y1: float
    ux: float
    uy: float
    largo: float

    def distancia_firmada(self, p: Tuple[float, float]) -> float:
        """Perpendicular a la línea, en píxeles. POSITIVA = lado DENTRO.

        DENTRO es la derecha caminando de P1 a P2 (y de imagen hacia abajo):
        el producto cruzado `u × (p - P1)`.
        """
        return (self.ux * (p[1] - self.y1) - self.uy * (p[0] - self.x1))

    def _t(self, p: Tuple[float, float]) -> float:
        """Posición de la proyección de `p` a lo largo del segmento (0..1)."""
        return ((p[0] - self.x1) * self.ux
                + (p[1] - self.y1) * self.uy) / self.largo

    def dentro_del_segmento(self, p: Tuple[float, float],
                            extension: float) -> bool:
        return -extension <= self._t(p) <= 1.0 + extension

    def trayecto_cruza(self, a: Optional[Tuple[float, float]],
                       b: Optional[Tuple[float, float]],
                       extension: float) -> bool:
        """El tramo a→b corta el SEGMENTO (con la propina de las puntas)."""
        if a is None or b is None:
            return False
        da = self.distancia_firmada(a)
        db = self.distancia_firmada(b)
        if da == db:
            return False
        # Punto de corte con la recta, interpolando por la distancia firmada.
        f = da / (da - db)
        if not 0.0 <= f <= 1.0:
            return False
        corte = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        return self.dentro_del_segmento(corte, extension)


def _geometria(linea: Any) -> Optional[_Linea]:
    """Parsea lo que llegue del cliente. `None` si no define un segmento."""
    if linea is None:
        return None
    try:
        datos = list(linea)
        if len(datos) < 2:
            return None
        x1, y1 = float(datos[0][0]), float(datos[0][1])
        x2, y2 = float(datos[1][0]), float(datos[1][1])
        largo = math.hypot(x2 - x1, y2 - y1)
        if largo < 1e-6:
            return None
        return _Linea(x1=x1, y1=y1, ux=(x2 - x1) / largo,
                      uy=(y2 - y1) / largo, largo=largo)
    except (TypeError, ValueError, IndexError):
        return None
