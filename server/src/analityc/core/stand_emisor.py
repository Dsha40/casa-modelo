"""Emisión en vivo del stand por el WebSocket existente (HITO 10).

## Por qué no hay Socket.IO

La spec pedía un namespace `/stand`. El HITO 0 verificó que el transporte real
cliente↔servidor es el **WebSocket de FastAPI** (`/ws/{type_inference}`), y que
montar Socket.IO habría añadido un segundo canal en paralelo con su propia
reconexión y su propio puerto. Los eventos viajan dentro del `metadata` que ya
vuelve por ese WebSocket, bajo la clave `stand`.

Los clientes que no conocen esa clave la ignoran (leen con `.get()`), así que
Tienda y Perimetrales no se enteran de nada.

## Totales absolutos, nunca incrementos

Cada payload lleva el total, no el delta. Un cliente que se reconecta se pone
al día con el siguiente mensaje sin arrastrar contadores desfasados, y no hace
falta protocolo de resincronización. Es la razón por la que el bug clásico de
«los contadores se quedan pegados tras un corte de red» no puede ocurrir aquí.

## Throttling

Cada bloque tiene su ritmo (contadores 2 s, salud 10 s, heatmap 30 s). Sin
esto, a 25 fps se enviaría el mapa de calor 25 veces por segundo y el
WebSocket se convertiría en el cuello de botella del pipeline.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ritmos del contrato (CONTRATO_STAND.md §2).
PERIODO_CONTADORES_S = 2.0
PERIODO_SALUD_S = 10.0
PERIODO_HEATMAP_S = 30.0
PERIODO_TRACKS_S = 2.0

MAX_EVENTOS_FEED = 10

# Tope de tracks en el payload. El cliente los usa para el selector de
# "Marcar como personal": con mas de 20 en pantalla el selector deja de ser
# usable, y en este encuadre nunca hay tantas personas a la vez.
MAX_TRACKS_PAYLOAD = 20


class EmisorStand:
    """Construye el bloque `metadata['stand']` respetando los ritmos."""

    def __init__(self,
                 periodo_contadores: float = PERIODO_CONTADORES_S,
                 periodo_salud: float = PERIODO_SALUD_S,
                 periodo_heatmap: float = PERIODO_HEATMAP_S,
                 periodo_tracks: float = PERIODO_TRACKS_S,
                 sesion_id: Optional[int] = None) -> None:
        self._periodos = {
            'contadores': float(periodo_contadores),
            'salud': float(periodo_salud),
            'heatmap': float(periodo_heatmap),
            'tracks': float(periodo_tracks),
        }
        self._ultimo: Dict[str, float] = {}
        self._feed: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENTOS_FEED)
        self._feed_sucio = False
        self.sesion_id = sesion_id

    # ── Feed de eventos ─────────────────────────────────────────────────
    def anotar_evento(self, tipo: str, *, genero: Optional[str] = None,
                      rango_etario: Optional[str] = None,
                      ts: Optional[float] = None,
                      **extra: Any) -> None:
        """Añade un evento al feed del cliente (atención / degustación).

        Sin fotos ni recortes: el panel muestra hora, género y rango. Es la
        regla 7 de privacidad, y además hace el payload trivialmente ligero.
        """
        self._feed.append({
            'tipo': tipo,
            'ts': _hora(ts),
            'genero': genero,
            'rango_etario': rango_etario,
            **extra,
        })
        self._feed_sucio = True

    # ── Construcción del payload ────────────────────────────────────────
    def _toca(self, bloque: str, ahora: float) -> bool:
        anterior = self._ultimo.get(bloque)
        if anterior is None or (ahora - anterior) >= self._periodos[bloque]:
            self._ultimo[bloque] = ahora
            return True
        return False

    def construir(self, *,
                  contadores: Optional[Dict[str, int]] = None,
                  salud: Optional[Dict[str, Any]] = None,
                  heatmap: Optional[Callable[[], Dict[str, Any]]] = None,
                  tracks: Optional[List[Dict[str, Any]]] = None,
                  ahora: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Bloque `stand` para este frame, o None si no toca emitir nada.

        `heatmap` es un callable y no un dict a propósito: serializar la
        matriz cuesta, y así solo se paga cuando toca enviarla.
        """
        t = float(ahora if ahora is not None else time.time())
        bloque: Dict[str, Any] = {}

        if contadores is not None and self._toca('contadores', t):
            bloque['contadores'] = {**contadores,
                                    'tasas': _tasas(contadores)}

        if tracks is not None and self._toca('tracks', t):
            bloque['tracks_activos'] = _limpiar_tracks(tracks)

        if self._feed_sucio:
            bloque['ultimos_eventos'] = list(self._feed)
            self._feed_sucio = False

        if salud is not None and self._toca('salud', t):
            bloque['salud'] = salud

        if heatmap is not None and self._toca('heatmap', t):
            try:
                bloque['heatmap'] = heatmap()
            except Exception as exc:                        # noqa: BLE001
                logger.error("stand: no se pudo construir el heatmap: %s", exc)

        # El id de sesión viaja SIEMPRE que haya algo que mandar: es lo que
        # permite al cliente construir la URL del POST de enrolamiento y del
        # GET de resincronización sin tener que preguntarlo aparte.
        if bloque and self.sesion_id is not None:
            bloque['sesion_id'] = int(self.sesion_id)

        return bloque or None

    def adjuntar(self, metadata: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Inserta el bloque en un metadata existente, sin tocar el resto.

        Aditivo por diseño (regla 2): `detections`, `alerts` y todo lo demás
        se quedan exactamente como estaban.
        """
        bloque = self.construir(**kwargs)
        if bloque:
            metadata['stand'] = bloque
        return metadata


def _tasas(c: Dict[str, int]) -> Dict[str, Optional[float]]:
    """Las cuatro tasas. Denominador cero -> None, nunca 0."""
    def t(num: str, den: str) -> Optional[float]:
        d = c.get(den) or 0
        return round((c.get(num) or 0) / d, 4) if d else None
    return {
        'captacion': t('se_detuvieron', 'pasaron'),
        'atencion': t('atendidos', 'se_detuvieron'),
        'degustacion': t('degustaron', 'atendidos'),
        'global': t('degustaron', 'pasaron'),
    }


def _limpiar_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deja solo los campos que el selector del cliente necesita.

    Nada de embeddings ni recortes: el cliente solo tiene que poder señalar a
    una persona en una lista. Enviar el resto engordaría el payload cada 2 s y
    sacaría datos biométricos del servidor sin motivo (regla 7).

    Los `prov:` se descartan: una identidad todavía sin confirmar no debe
    poder enrolarse como personal, porque el id cambiará al confirmarse y el
    enrolamiento apuntaría a nadie.
    """
    salida: List[Dict[str, Any]] = []
    for t in tracks or []:
        pid = str(t.get('persistent_id') or '')
        if not pid or pid.startswith('prov:'):
            continue
        salida.append({
            'persistent_id': pid,
            'genero': t.get('genero'),
            'rango_etario': t.get('rango_etario'),
            'zona': t.get('zona'),
            'es_personal': bool(t.get('es_personal')),
            'dwell_s': round(float(t.get('dwell_s') or 0.0), 1),
        })
        if len(salida) >= MAX_TRACKS_PAYLOAD:
            break
    return salida


def _hora(ts: Optional[float]) -> str:
    return time.strftime('%H:%M:%S',
                         time.localtime(ts if ts is not None else time.time()))
