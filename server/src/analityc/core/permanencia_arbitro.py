"""Árbitro de los cruces que la geometría no puede decidir.

## Qué arbitra, y qué NO

**No decide la dirección.** Eso lo hace la flecha, y lo hace bien: es una
proyección, cuesta nanosegundos y es reproducible. Preguntarle la dirección a
un modelo de visión sería sustituir una medida por una opinión, y además una
que tarda decenas de segundos en llegar cuando el cruce dura uno.

Lo que arbitra es **el residuo**: la persona que aparece y desaparece en el
vano sin desplazarse lo suficiente como para que la proyección signifique
nada. Ahí la geometría no tiene respuesta —hoy se calla— y una segunda lectura
sí puede aportar algo.

## Por qué fuera del bucle

Encolar no bloquea y la cola está acotada. Si se llena, la petición **se
descarta y se cuenta**: nunca se acumula trabajo por detrás del vídeo. El
veredicto llega segundos después y se recoge cuando esté; mientras tanto el
pipeline sigue a su ritmo.

Desactivado por defecto. Sin activarlo, el modelo ni se carga: no ocupa VRAM
ni cambia nada.

## Por qué una TIRA de tres imágenes y no una

Un fotograma no contiene el movimiento. Al modelo se le enseñan tres momentos
del mismo track en orden temporal, con la persona recuadrada, y se le pregunta
qué hizo **entre** ellos. Eso sí está en lo que ve.

Se guarda el frame reducido —no el recorte de la persona— porque la respuesta
depende de dónde está ella *respecto a la puerta*, y un recorte pegado al
cuerpo se lleva justo esa referencia por delante.

## El veredicto no se disfraza de medida

Lo que resuelve el árbitro sale marcado (`direccion_por: 'vlm'`) y con su
confianza. Quien lea la alerta tiene que poder distinguir un cruce medido de
uno opinado.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Cómo se marca en la alerta lo que decidió el árbitro.
POR_VLM = 'vlm'

#: Momentos que se recuerdan de cada track: primero, uno intermedio y último.
RECUERDOS = 3
#: Ancho al que se reduce cada frame recordado. Suficiente para ver a una
#: persona ante una puerta, y barato de guardar y de mandar al modelo.
ANCHO_RECUERDO = 360

_PREGUNTA = (
    "Las tres imagenes, de izquierda a derecha, son tres momentos "
    "CONSECUTIVOS de la misma escena, grabados por una camara de seguridad "
    "que enfoca una puerta. La persona relevante esta marcada con un "
    "rectangulo. Fijate en como cambia su posicion respecto a la puerta "
    "entre la primera y la ultima imagen.\n"
    "Responde UNICAMENTE con un objeto JSON valido, sin texto adicional, "
    "con esta forma exacta:\n"
    '{"accion": "entro|salio|ninguno", "seguridad": "alta|media|baja"}\n'
    'Usa "entro" si la persona atraviesa la puerta hacia el interior; '
    '"salio" si viene del interior hacia afuera; y "ninguno" si se queda '
    "junto a la puerta, pasa de largo o no se distingue. No inventes: ante "
    'la duda responde "ninguno".'
)

_ACCIONES = {'entro': 'entrada', 'entró': 'entrada', 'entra': 'entrada',
             'salio': 'salida', 'salió': 'salida', 'sale': 'salida'}


@dataclass
class Veredicto:
    """Lo que el árbitro concluyó sobre un cruce ambiguo."""
    clave: str
    #: 'entrada' | 'salida' | None (None = sigue sin saberse, y así se queda)
    tipo: Optional[str]
    #: Marca de tiempo ORIGINAL del cruce, no la de la respuesta. El
    #: cronómetro tiene que arrancar cuando la persona cruzó, no cuando el
    #: modelo termino de pensar.
    ts: float
    seguridad: str = 'baja'
    box: Optional[Tuple[float, float, float, float]] = None
    crudo: str = ''

    @property
    def resuelto(self) -> bool:
        return self.tipo in ('entrada', 'salida')


def _parsear(respuesta: str, clave: str, ts: float,
             box) -> Optional[Veredicto]:
    """Extrae el JSON de la respuesta. `None` si no hay nada aprovechable.

    Los VLM envuelven el JSON en prosa o en un bloque markdown, así que se
    busca el primer objeto `{...}` en vez de exigir una respuesta limpia. Una
    respuesta rara se descarta: no puede tumbar el hilo ni ensuciar los datos.
    """
    if not respuesta:
        return None
    trozo = re.search(r'\{.*?\}', respuesta, re.DOTALL)
    if not trozo:
        return None
    try:
        datos = json.loads(trozo.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(datos, dict):
        return None

    accion = str(datos.get('accion', '')).strip().lower()
    seguridad = str(datos.get('seguridad', 'baja')).strip().lower()
    if seguridad not in ('alta', 'media', 'baja'):
        seguridad = 'baja'
    return Veredicto(clave=clave, tipo=_ACCIONES.get(accion), ts=ts,
                     seguridad=seguridad, box=box, crudo=respuesta[:300])


class MemoriaVisual:
    """Tres momentos de cada track vivo, listos por si hay que preguntar."""

    def __init__(self, max_tracks: int = 60,
                 cada_n_frames: int = 5) -> None:
        self.max_tracks = max(1, int(max_tracks))
        self.cada_n = max(1, int(cada_n_frames))
        self._por_clave: Dict[str, List[np.ndarray]] = {}
        self._contador: Dict[str, int] = {}
        self._lock = threading.Lock()

    def recordar(self, clave: str, frame: Optional[np.ndarray],
                 box: Optional[Sequence[float]]) -> None:
        """Guarda un momento (limitado en frecuencia). Nunca lanza."""
        if frame is None or getattr(frame, 'size', 0) == 0:
            return
        clave = str(clave)
        with self._lock:
            n = self._contador.get(clave, 0)
            self._contador[clave] = n + 1
            if n % self.cada_n != 0:
                return
        try:
            reducido = _reducir(frame, box)
            if reducido is None:
                return
            with self._lock:
                momentos = self._por_clave.setdefault(clave, [])
                if len(momentos) < RECUERDOS:
                    momentos.append(reducido)
                else:
                    # Se conservan el PRIMERO y el ÚLTIMO —que son los que
                    # marcan el recorrido— y se renueva el de en medio.
                    momentos[1] = momentos[2]
                    momentos[2] = reducido
                self._podar()
        except Exception as exc:                            # noqa: BLE001
            logger.debug("Árbitro: no se pudo recordar %s (%s)", clave, exc)

    def tira(self, clave: str) -> Optional[np.ndarray]:
        """Los momentos pegados en horizontal, en orden. `None` si no llegan."""
        import cv2
        with self._lock:
            momentos = list(self._por_clave.get(str(clave)) or [])
        if len(momentos) < 2:
            return None                 # con una sola imagen no hay movimiento
        try:
            alto = max(m.shape[0] for m in momentos)
            iguales = []
            for m in momentos:
                if m.shape[0] != alto:
                    fondo = np.zeros((alto, m.shape[1], 3), dtype=np.uint8)
                    fondo[0:m.shape[0], :] = m
                    m = fondo
                iguales.append(m)
                iguales.append(np.full((alto, 3, 3), 90, dtype=np.uint8))
            return cv2.hconcat(iguales[:-1])
        except Exception as exc:                            # noqa: BLE001
            logger.debug("Árbitro: no se pudo montar la tira (%s)", exc)
            return None

    def olvidar(self, clave: str) -> None:
        with self._lock:
            self._por_clave.pop(str(clave), None)
            self._contador.pop(str(clave), None)

    def _podar(self) -> None:
        """Cota de memoria. Se llama con el lock tomado."""
        while len(self._por_clave) > self.max_tracks:
            self._por_clave.pop(next(iter(self._por_clave)), None)
        while len(self._contador) > self.max_tracks * 4:
            self._contador.pop(next(iter(self._contador)), None)


def _reducir(frame: np.ndarray,
             box: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Frame reducido con la persona recuadrada. `None` si no se puede."""
    import cv2
    alto, ancho = frame.shape[:2]
    if ancho <= 0 or alto <= 0:
        return None
    escala = min(ANCHO_RECUERDO / float(ancho), 1.0)
    reducido = (frame if escala >= 0.999 else
                cv2.resize(frame, (max(1, int(ancho * escala)),
                                   max(1, int(alto * escala))),
                           interpolation=cv2.INTER_AREA)).copy()
    if box is not None:
        try:
            x1, y1, x2, y2 = (int(float(box[i]) * escala) for i in range(4))
            cv2.rectangle(reducido, (x1, y1), (x2, y2), (0, 230, 255), 2)
        except (TypeError, ValueError, IndexError):
            pass
    return reducido


class ArbitroPuerta:
    """Cola asíncrona de arbitraje. El pipeline nunca la espera."""

    def __init__(self, activo: bool = False, tam_cola: int = 4,
                 max_tokens: int = 60, cada_n_frames: int = 5) -> None:
        self._activo = bool(activo)
        self._max_tokens = int(max_tokens)
        self._cola: "queue.Queue[Optional[tuple]]" = queue.Queue(
            maxsize=max(1, int(tam_cola)))
        self.memoria = MemoriaVisual(cada_n_frames=cada_n_frames)
        self._veredictos: List[Veredicto] = []
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._hilo: Optional[threading.Thread] = None
        self.stats = {'encolados': 0, 'descartados': 0, 'procesados': 0,
                      'resueltos': 0, 'no_parseables': 0, 'errores': 0}
        if self._activo:
            self._arrancar()

    @property
    def activo(self) -> bool:
        return self._activo

    def _arrancar(self) -> None:
        if self._hilo is not None and self._hilo.is_alive():
            return
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, name='arbitro-puerta',
                                      daemon=True)
        self._hilo.start()
        logger.info("Árbitro de puerta ACTIVO: los cruces que la geometría no "
                    "resuelve se consultarán al VLM fuera del bucle")

    def detener(self, esperar: float = 2.0) -> None:
        if self._hilo is None:
            return
        self._parar.set()
        try:
            self._cola.put_nowait(None)
        except queue.Full:
            pass
        self._hilo.join(timeout=esperar)
        self._hilo = None

    # -- Alta y recogida ---------------------------------------------------
    def recordar(self, clave: str, frame: Optional[np.ndarray],
                 box: Optional[Sequence[float]]) -> None:
        """Alimenta la memoria visual. No hace nada si está desactivado."""
        if self._activo:
            self.memoria.recordar(clave, frame, box)

    def encolar(self, clave: str, ts: float,
                box: Optional[Sequence[float]] = None) -> bool:
        """Pide un veredicto. No bloquea; `False` si no se pudo encolar."""
        if not self._activo:
            return False
        tira = self.memoria.tira(clave)
        if tira is None:
            # Sin al menos dos momentos no hay nada que comparar. Se dice en
            # las estadísticas en vez de mandar una consulta que no puede
            # responder nada.
            with self._lock:
                self.stats['descartados'] += 1
            return False
        try:
            self._cola.put_nowait((str(clave), tira, float(ts), box))
        except queue.Full:
            with self._lock:
                self.stats['descartados'] += 1
            return False
        with self._lock:
            self.stats['encolados'] += 1
        self.memoria.olvidar(clave)
        return True

    def recoger(self) -> List[Veredicto]:
        """Veredictos listos desde la última llamada. Vacía la bandeja."""
        with self._lock:
            listos, self._veredictos = self._veredictos, []
        return listos

    # -- Hilo --------------------------------------------------------------
    def _bucle(self) -> None:
        while not self._parar.is_set():
            try:
                item = self._cola.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            clave, tira, ts, box = item
            try:
                self._procesar(clave, tira, ts, box)
            except Exception as exc:                        # noqa: BLE001
                # Una consulta fallida jamás debe matar el hilo: si muere, el
                # árbitro queda mudo sin que nadie se entere.
                with self._lock:
                    self.stats['errores'] += 1
                logger.warning("Árbitro: fallo con %s: %s: %s", clave,
                               type(exc).__name__, exc)
            finally:
                self._cola.task_done()

    def _procesar(self, clave: str, tira: np.ndarray, ts: float,
                  box) -> None:
        from .multimodal_router import get_multimodal_router
        respuesta = get_multimodal_router().vqa(
            tira, _PREGUNTA, max_new_tokens=self._max_tokens)
        with self._lock:
            self.stats['procesados'] += 1
        veredicto = _parsear(respuesta, clave, ts, box)
        if veredicto is None:
            with self._lock:
                self.stats['no_parseables'] += 1
            logger.debug("Árbitro: respuesta no parseable para %s: %r",
                         clave, (respuesta or '')[:120])
            return
        with self._lock:
            if veredicto.resuelto:
                self.stats['resueltos'] += 1
                self._veredictos.append(veredicto)
            # Un 'ninguno' NO se encola: no hay nada que corregir. Se queda
            # en las estadísticas y el cruce sigue sin contarse, que es la
            # respuesta honesta.
            if len(self._veredictos) > 200:
                self._veredictos = self._veredictos[-100:]
