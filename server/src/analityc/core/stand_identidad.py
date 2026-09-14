"""Identidad única y anti-doble-conteo del stand (HITO 4).

Esta es la pieza de la que depende la confiabilidad de todo lo demás: si una
persona se cuenta dos veces, el embudo entero miente y las decisiones sobre el
stand se toman sobre cifras infladas.

## Las tres reglas que lo garantizan

1. **Se cuenta por `persistent_id`, jamás por detección ni por `tracker_id`.**
   Un `tracker_id` se pierde con cada oclusión —y en este encuadre el
   mostrador ocluye constantemente—, así que contarlos multiplica personas.
2. **Un track solo cuenta si se lo gana:** `min_frames_track` frames vistos
   Y (cruzar `linea_conteo` O permanecer en `zona_paso` O permanecer en
   `zona_stand`). Un track de tres frames casi siempre es un falso positivo
   del detector. `zona_stand` entró el 7-ago-2026: quien llega al stand por
   el frente sin pisar el pasillo se descartaba «nunca_en_zona» aunque
   llevara 3 s detenido frente al mostrador — que es justo el visitante que
   este módulo existe para medir.
3. **Toda fusión decrementa el agregado y queda registrada.** Si dos tracks
   resultan ser la misma persona, el conteo baja y se emite `merge_identidad`.

## Por qué es un módulo aparte, sin GPU

Recibe *embeddings ya calculados* (el pipeline los saca de OSNet/ArcFace) en
vez de calcularlos. Así toda la lógica de decisión —que es donde viven los
errores de conteo— se prueba con vectores sintéticos, sin cámara ni GPU, y en
milisegundos. Ver `tests/test_stand_identidad.py`.

Nada aquí escribe en disco: el que persiste es `stand_almacen.py`.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Motivos de descarte (van a stand_visitantes.motivo_descarte) ─────────
class MotivoDescarte:
    TRACK_CORTO = "track_corto"              # menos de min_frames_track
    NUNCA_EN_ZONA = "nunca_en_zona"          # ni línea, ni zona_paso, ni stand
    SIN_EMBEDDING = "sin_embedding"          # nunca hubo crop utilizable


def _unitario(v: np.ndarray) -> np.ndarray:
    """Vector normalizado. Con norma 0 devuelve el propio vector."""
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno de dos embeddings ya unitarios."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _cruza(p1: Sequence[float], p2: Sequence[float],
           q1: Sequence[float], q2: Sequence[float]) -> bool:
    """¿Se cortan los segmentos p1-p2 y q1-q2?"""
    def orient(a, b, c) -> float:
        return ((b[1] - a[1]) * (c[0] - b[0]) -
                (b[0] - a[0]) * (c[1] - b[1]))

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass
class _Muestra:
    """Un crop bueno de un track: su embedding y su calidad."""
    embedding: np.ndarray
    calidad: float


@dataclass
class _TrackVivo:
    """Lo que se sabe de un `tracker_id` mientras existe."""
    tracker_id: int
    primera_ts: float
    ultima_ts: float
    frames: int = 0
    muestras: List[_Muestra] = field(default_factory=list)
    dwell_zona_paso_s: float = 0.0
    dwell_zona_stand_s: float = 0.0
    cruzo_linea: bool = False
    ultimo_pie: Optional[Tuple[float, float]] = None
    persistent_id: Optional[str] = None
    ultima_zona: str = ""

    @property
    def provisional(self) -> str:
        return f"prov:{self.tracker_id}"


@dataclass
class _Identidad:
    """Una persona en la galería de la sesión."""
    persistent_id: str
    embedding: np.ndarray
    ultima_vez: float
    tracker_ids: List[int] = field(default_factory=list)
    n_fusiones: int = 0
    ultima_zona: str = ""


@dataclass
class ResultadoTrack:
    """Veredicto al cerrar un track."""
    tracker_id: int
    persistent_id: Optional[str]
    contable: bool
    descartado: bool
    motivo_descarte: Optional[str]
    frames: int
    fusionado_con: Optional[str] = None


class GestorIdentidad:
    """Galería de identidades de la sesión, con TTL y anti-doble-conteo."""

    def __init__(self, cfg: Any,
                 al_fusionar: Optional[Callable[[str, str, float], None]] = None
                 ) -> None:
        """`cfg` es un ConfigStand (o cualquier objeto con .get por ruta)."""
        g = cfg.get
        self._umbral_merge = float(g('reid.umbral_merge', 0.78))
        self._umbral_ventana = float(g('reid.umbral_merge_ventana', 0.65))
        self._ventana_reentrada_s = float(g('reid.ventana_reentrada_s', 120))
        self._ttl_s = float(g('reid.ttl_galeria_min', 45)) * 60.0
        self._min_frames = int(g('conteo.min_frames_track', 15))
        self._min_dwell_paso = float(g('conteo.min_dwell_zona_paso_s', 1.0))
        # Sin clave propia hereda el umbral del pasillo: mismo criterio de
        # «permanencia mínima que gana el conteo» en las dos zonas.
        self._min_dwell_stand = float(
            g('conteo.min_dwell_zona_stand_s', self._min_dwell_paso))
        self._max_muestras = int(g('demografia.buffer.max_muestras', 10))
        # Mínimo de crops buenos antes de comprometer identidad. Con uno solo
        # el embedding es ruidoso y provoca fusiones erróneas.
        self._min_muestras = 3

        self._al_fusionar = al_fusionar
        self._vivos: Dict[int, _TrackVivo] = {}
        self._galeria: Dict[str, _Identidad] = {}
        self._contables: set[str] = set()
        self._descartes: List[Dict[str, Any]] = []
        self._fusiones: List[Dict[str, Any]] = []
        self._linea: Optional[Tuple[Sequence[float], Sequence[float]]] = None

        linea = cfg.zona_normalizada('linea_conteo') \
            if hasattr(cfg, 'zona_normalizada') else []
        if len(linea) == 2:
            self._linea = (linea[0], linea[1])

    # ── Observación ─────────────────────────────────────────────────────
    def observar(self, tracker_id: int, *,
                 embedding: Optional[np.ndarray] = None,
                 calidad: float = 0.0,
                 punto_pie: Optional[Tuple[float, float]] = None,
                 en_zona_paso: bool = False,
                 en_zona_stand: bool = False,
                 zona: str = "",
                 identidad_externa: Optional[str] = None,
                 ts: Optional[float] = None) -> str:
        """Registra un frame de un track y devuelve su identidad actual.

        Devuelve `prov:<tracker_id>` mientras no haya evidencia suficiente
        para comprometer una identidad. El pipeline NUNCA debe contar un
        `prov:` — es exactamente la salvaguarda que ya usa
        `_process_zone_transitions` en el pipeline de tienda.
        """
        ahora = float(ts if ts is not None else time.time())
        t = self._vivos.get(tracker_id)
        if t is None:
            t = _TrackVivo(tracker_id=tracker_id, primera_ts=ahora,
                           ultima_ts=ahora)
            self._vivos[tracker_id] = t

        dt = max(0.0, ahora - t.ultima_ts)
        t.frames += 1
        t.ultima_ts = ahora
        if zona:
            t.ultima_zona = zona
        if en_zona_paso:
            t.dwell_zona_paso_s += dt
        if en_zona_stand:
            t.dwell_zona_stand_s += dt

        if punto_pie is not None:
            if (self._linea is not None and t.ultimo_pie is not None
                    and not t.cruzo_linea):
                if _cruza(t.ultimo_pie, punto_pie,
                          self._linea[0], self._linea[1]):
                    t.cruzo_linea = True
            t.ultimo_pie = (float(punto_pie[0]), float(punto_pie[1]))

        if embedding is not None:
            emb = _unitario(np.asarray(embedding, dtype=np.float32).ravel())
            t.muestras.append(_Muestra(emb, float(calidad)))
            # Solo se guardan las mejores: promediar crops borrosos o
            # diminutos ensucia el embedding y provoca fusiones erróneas.
            if len(t.muestras) > self._max_muestras:
                t.muestras.sort(key=lambda m: m.calidad, reverse=True)
                t.muestras = t.muestras[:self._max_muestras]

        # ── Identidad EXTERNA ──
        # `tiendas.py` ya resuelve Re-ID en produccion y
        # asigna `persistent_id`. Cuando el llamador la pasa, se adopta en vez
        # de calcular una segunda identidad en paralelo: dos galerias sobre
        # los mismos rostros acabarian discrepando, y la del pipeline esta
        # calibrada y en uso. Aqui lo que aporta este modulo son las REGLAS
        # DE CONTEO (min_frames, cruce de linea, dwell), no el Re-ID.
        if identidad_externa and not str(identidad_externa).startswith('prov:'):
            if t.persistent_id != identidad_externa:
                if t.persistent_id is not None:
                    # El pipeline fusiono este track con otra identidad.
                    self._fusiones.append({
                        'persistent_id': identidad_externa,
                        'tracker_id': t.tracker_id,
                        'similitud': None,          # la decidio el pipeline
                        'ts': ahora,
                    })
                t.persistent_id = str(identidad_externa)
            ident = self._galeria.get(t.persistent_id)
            if ident is None:
                self._galeria[t.persistent_id] = _Identidad(
                    persistent_id=t.persistent_id,
                    embedding=self._embedding_de(t),
                    ultima_vez=ahora, tracker_ids=[t.tracker_id],
                    ultima_zona=t.ultima_zona)
            else:
                ident.ultima_vez = ahora
                if t.tracker_id not in ident.tracker_ids:
                    ident.tracker_ids.append(t.tracker_id)
        elif t.persistent_id is None and len(t.muestras) >= self._min_muestras:
            self._resolver(t, ahora)

        self._caducar(ahora)
        return t.persistent_id or t.provisional

    def _embedding_de(self, t: _TrackVivo) -> Optional[np.ndarray]:
        """Media de las mejores muestras, renormalizada."""
        if not t.muestras:
            return None
        mejores = sorted(t.muestras, key=lambda m: m.calidad,
                         reverse=True)[:self._max_muestras]
        return _unitario(np.mean([m.embedding for m in mejores], axis=0))

    def _resolver(self, t: _TrackVivo, ahora: float) -> None:
        """Asigna `persistent_id`: fusiona con la galería o crea identidad."""
        emb = self._embedding_de(t)
        if emb is None:
            return

        mejor_pid, mejor_sim = None, -1.0
        for pid, ident in self._galeria.items():
            # Una identidad adoptada del pipeline puede no traer embedding
            # (no hubo crop utilizable). No es comparable: se salta en vez de
            # reventar el bucle de inferencia.
            if ident.embedding is None:
                continue
            sim = coseno(emb, ident.embedding)
            if sim > mejor_sim:
                mejor_pid, mejor_sim = pid, sim

        fusiona = False
        if mejor_pid is not None:
            if mejor_sim >= self._umbral_merge:
                fusiona = True
            elif mejor_sim >= self._umbral_ventana:
                # Zona gris: solo se fusiona si la reaparición es plausible
                # en el tiempo. Sin esta ventana, dos personas parecidas a lo
                # largo de la jornada se colapsarían en una (subconteo).
                ident = self._galeria[mejor_pid]
                if (ahora - ident.ultima_vez) <= self._ventana_reentrada_s:
                    fusiona = True

        if fusiona and mejor_pid is not None:
            ident = self._galeria[mejor_pid]
            # Media incremental: la identidad se refina con cada reaparición
            # sin que un solo crop malo la desplace.
            ident.embedding = _unitario((ident.embedding + emb) / 2.0)
            ident.ultima_vez = ahora
            ident.tracker_ids.append(t.tracker_id)
            ident.n_fusiones += 1
            t.persistent_id = mejor_pid
            self._fusiones.append({
                'persistent_id': mejor_pid,
                'tracker_id': t.tracker_id,
                'similitud': round(mejor_sim, 4),
                'ts': ahora,
            })
            if self._al_fusionar is not None:
                try:
                    self._al_fusionar(mejor_pid, str(t.tracker_id), mejor_sim)
                except Exception:                           # noqa: BLE001
                    logger.debug("stand: callback de fusión falló")
            return

        pid = f"p-{uuid.uuid4().hex[:12]}"
        self._galeria[pid] = _Identidad(
            persistent_id=pid, embedding=emb, ultima_vez=ahora,
            tracker_ids=[t.tracker_id], ultima_zona=t.ultima_zona)
        t.persistent_id = pid

    def _caducar(self, ahora: float) -> None:
        """TTL de la galería: es efímera, no un registro biométrico.

        Regla 7 del proyecto (privacidad): los embeddings viven lo que dura
        la sesión y caducan solos.
        """
        if self._ttl_s <= 0:
            return
        vencidas = [pid for pid, i in self._galeria.items()
                    if (ahora - i.ultima_vez) > self._ttl_s]
        for pid in vencidas:
            del self._galeria[pid]

    # ── Cierre ──────────────────────────────────────────────────────────
    def cerrar(self, tracker_id: int,
               ts: Optional[float] = None) -> ResultadoTrack:
        """Cierra un track y dictamina si su persona cuenta."""
        ahora = float(ts if ts is not None else time.time())
        t = self._vivos.pop(tracker_id, None)
        if t is None:
            return ResultadoTrack(tracker_id, None, False, True,
                                  MotivoDescarte.TRACK_CORTO, 0)

        if t.persistent_id is None and t.muestras:
            self._resolver(t, ahora)

        motivo: Optional[str] = None
        if t.frames < self._min_frames:
            motivo = MotivoDescarte.TRACK_CORTO
        elif t.persistent_id is None:
            motivo = MotivoDescarte.SIN_EMBEDDING
        elif not (t.cruzo_linea
                  or t.dwell_zona_paso_s >= self._min_dwell_paso
                  or t.dwell_zona_stand_s >= self._min_dwell_stand):
            motivo = MotivoDescarte.NUNCA_EN_ZONA

        if motivo is not None:
            self._descartes.append({
                'tracker_id': tracker_id, 'motivo': motivo,
                'frames': t.frames, 'ts': ahora,
                'persistent_id': t.persistent_id,
            })
            return ResultadoTrack(tracker_id, t.persistent_id, False, True,
                                  motivo, t.frames)

        ya_contada = t.persistent_id in self._contables
        self._contables.add(t.persistent_id)
        return ResultadoTrack(
            tracker_id, t.persistent_id, True, False, None, t.frames,
            fusionado_con=t.persistent_id if ya_contada else None)

    # ── Consulta ────────────────────────────────────────────────────────
    def total_contable(self) -> int:
        """Personas únicas contables. ESTE es el "pasaron" del embudo.

        Es `len` de un set de `persistent_id`: una fusión no lo incrementa,
        y por eso el conteo no se infla con las reentradas — a diferencia de
        `total_entradas` del pipeline de tienda (ver AUDITORIA_STAND.md §4).
        """
        return len(self._contables)

    def contables(self) -> List[str]:
        return sorted(self._contables)

    def descartes(self) -> List[Dict[str, Any]]:
        return list(self._descartes)

    def fusiones(self) -> List[Dict[str, Any]]:
        return list(self._fusiones)

    def tracks_activos(self) -> int:
        return len(self._vivos)

    def calidad_dato(self) -> Dict[str, Any]:
        """Resumen para el panel de calidad del HITO 12."""
        por_motivo: Dict[str, int] = {}
        for d in self._descartes:
            por_motivo[d['motivo']] = por_motivo.get(d['motivo'], 0) + 1
        return {
            'personas_contables': self.total_contable(),
            'identidades_en_galeria': len(self._galeria),
            'tracks_activos': len(self._vivos),
            'tracks_descartados': len(self._descartes),
            'descartes_por_motivo': por_motivo,
            'fusiones_realizadas': len(self._fusiones),
        }
