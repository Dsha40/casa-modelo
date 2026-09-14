"""Identificación del chef / personal del stand (HITO 6).

## Por qué este módulo es prerrequisito de todo lo demás

El chef está en el encuadre toda la jornada: es el track más persistente y
mejor iluminado de la escena. Sin excluirlo, aparece como un visitante que
«pasó», «se detuvo» y probablemente «degustó» — y contamina las cuatro cifras
del embudo a la vez. La auditoría (§4, trampa 3) lo dejó anotado: hoy no existe
ningún mecanismo que lo excluya.

## Tres mecanismos, y por qué el automático solo PROPONE

1. **Enrolamiento manual** (primario): el operador marca un track desde el
   cliente. Es la única vía que da de alta directamente.
2. **Auto-enrolamiento por permanencia**: quien acumule más de N minutos
   dentro de `zona_personal` se **propone**, y queda esperando confirmación.
3. **Verificación continua por Re-ID**: en cada reaparición se compara contra
   la galería de personal.

El punto 2 nunca confirma solo, y esa es una decisión deliberada. El criterio
de aceptación del hito exige **cero falsos positivos**: marcar por error a un
cliente como personal lo borra de las métricas para siempre y hunde el
denominador sin dejar rastro visible. Un cliente que se queda 12 minutos
charlando existe; un chef mal detectado, no. Ante la duda, se pregunta.

## Efectos de `es_personal = True`

- Excluido de todos los conteos y del embudo (lo aplica `stand_almacen`, cuyo
  `visitantes()` filtra `es_personal = 0` por defecto).
- **Sí** participa como actor en la detección de atención: sin chef no hay
  atención que detectar.
- Se le calculan métricas propias: atenciones/hora, tiempo activo, tiempo
  muerto.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .stand_identidad import coseno, _unitario

logger = logging.getLogger(__name__)


class OrigenPersonal:
    MANUAL = "manual"                  # el operador lo marcó
    CONFIRMADO = "confirmado"          # propuesta automática, aceptada
    REID = "reid"                      # reconocido contra la galería


@dataclass
class _FichaPersonal:
    persistent_id: str
    embedding: Optional[np.ndarray]
    nombre: str = ""
    origen: str = OrigenPersonal.MANUAL
    desde_ts: float = 0.0


@dataclass
class _Actividad:
    """Actividad de una persona dentro de zona_personal."""
    dwell_s: float = 0.0
    ultima_ts: Optional[float] = None
    atenciones: List[Dict[str, Any]] = field(default_factory=list)
    primera_ts: Optional[float] = None


class GaleriaPersonal:
    """Quién es personal, quién lo parece, y qué rinde el chef."""

    def __init__(self, cfg: Any) -> None:
        g = cfg.get
        self._umbral_reid = float(g('personal.umbral_reid', 0.75))
        self._dwell_auto_s = float(
            g('personal.dwell_auto_enrolamiento_min', 10)) * 60.0

        self._galeria: Dict[str, _FichaPersonal] = {}
        self._actividad: Dict[str, _Actividad] = {}
        self._propuestos: Dict[str, float] = {}      # pid -> dwell al proponer
        self._rechazados: set[str] = set()

    # ── Alta y baja ─────────────────────────────────────────────────────
    def enrolar(self, persistent_id: str,
                embedding: Optional[np.ndarray] = None,
                nombre: str = "",
                origen: str = OrigenPersonal.MANUAL,
                ts: Optional[float] = None) -> None:
        """Da de alta a alguien como personal."""
        emb = None
        if embedding is not None:
            emb = _unitario(np.asarray(embedding, dtype=np.float32).ravel())
        self._galeria[persistent_id] = _FichaPersonal(
            persistent_id=persistent_id, embedding=emb, nombre=nombre,
            origen=origen, desde_ts=float(ts if ts is not None else time.time()))
        self._propuestos.pop(persistent_id, None)
        self._rechazados.discard(persistent_id)
        logger.info("stand: %s dado de alta como personal (%s)",
                    persistent_id, origen)

    def dar_de_baja(self, persistent_id: str) -> None:
        """Corrige un enrolamiento erróneo. La persona vuelve a contar."""
        self._galeria.pop(persistent_id, None)

    # ── Observación ─────────────────────────────────────────────────────
    def observar(self, persistent_id: str, *,
                 en_zona_personal: bool = False,
                 embedding: Optional[np.ndarray] = None,
                 ts: Optional[float] = None) -> None:
        """Acumula permanencia en `zona_personal` y propone si procede."""
        ahora = float(ts if ts is not None else time.time())
        act = self._actividad.setdefault(persistent_id, _Actividad())
        if act.primera_ts is None:
            act.primera_ts = ahora

        if en_zona_personal:
            if act.ultima_ts is not None:
                act.dwell_s += max(0.0, ahora - act.ultima_ts)
            act.ultima_ts = ahora
        else:
            act.ultima_ts = None

        # Propuesta automática: solo si no está ya en galería ni rechazado.
        if (act.dwell_s >= self._dwell_auto_s
                and persistent_id not in self._galeria
                and persistent_id not in self._rechazados
                and persistent_id not in self._propuestos):
            self._propuestos[persistent_id] = act.dwell_s
            logger.info(
                "stand: %s propuesto como personal (%.0f s en zona_personal). "
                "Pendiente de confirmación en el cliente.",
                persistent_id, act.dwell_s)

    # ── Consulta ────────────────────────────────────────────────────────
    def es_personal(self, persistent_id: str,
                    embedding: Optional[np.ndarray] = None) -> bool:
        """¿Está esta persona en la galería, directa o por Re-ID?"""
        if persistent_id in self._galeria:
            return True
        if embedding is None:
            return False
        emb = _unitario(np.asarray(embedding, dtype=np.float32).ravel())
        for ficha in self._galeria.values():
            if ficha.embedding is None:
                continue
            if coseno(emb, ficha.embedding) >= self._umbral_reid:
                # Se ancla la identidad nueva al personal ya conocido: el
                # chef cambia de persistent_id cada vez que el tracker lo
                # pierde, y sin esto habría que re-enrolarlo a cada rato.
                self._galeria[persistent_id] = _FichaPersonal(
                    persistent_id=persistent_id, embedding=emb,
                    nombre=ficha.nombre, origen=OrigenPersonal.REID,
                    desde_ts=time.time())
                return True
        return False

    def propuestas(self) -> List[Dict[str, Any]]:
        """Candidatos a personal, esperando confirmación humana."""
        return [{'persistent_id': pid, 'dwell_s': round(d, 1),
                 'motivo': 'permanencia en zona_personal'}
                for pid, d in sorted(self._propuestos.items())]

    def confirmar_propuesta(self, persistent_id: str,
                            embedding: Optional[np.ndarray] = None,
                            nombre: str = "") -> bool:
        if persistent_id not in self._propuestos:
            return False
        self.enrolar(persistent_id, embedding, nombre,
                     origen=OrigenPersonal.CONFIRMADO)
        return True

    def rechazar_propuesta(self, persistent_id: str) -> bool:
        """El operador dice que NO es personal. No se vuelve a proponer."""
        if persistent_id not in self._propuestos:
            return False
        del self._propuestos[persistent_id]
        self._rechazados.add(persistent_id)
        return True

    def personal(self) -> List[Dict[str, Any]]:
        return [{'persistent_id': f.persistent_id, 'nombre': f.nombre,
                 'origen': f.origen, 'desde_ts': f.desde_ts}
                for f in self._galeria.values()]

    # ── Rendimiento del chef (HITO 12 §6) ───────────────────────────────
    def registrar_atencion(self, persistent_id: str, duracion_s: float,
                           ts: Optional[float] = None) -> None:
        act = self._actividad.setdefault(persistent_id, _Actividad())
        act.atenciones.append({
            'ts': float(ts if ts is not None else time.time()),
            'duracion_s': float(duracion_s),
        })

    def metricas(self, persistent_id: str,
                 ahora: Optional[float] = None) -> Dict[str, Any]:
        """Rendimiento de un miembro del personal.

        `tiempo_muerto_s` es jornada menos tiempo atendiendo: es la cifra que
        revela si el stand está infrautilizado o si el chef no da abasto.
        """
        act = self._actividad.get(persistent_id)
        if act is None or act.primera_ts is None:
            return {'atenciones': 0, 'atenciones_por_hora': None,
                    'tiempo_medio_atencion_s': None, 'tiempo_activo_s': 0.0,
                    'tiempo_muerto_s': 0.0, 'jornada_s': 0.0}

        t = float(ahora if ahora is not None else time.time())
        jornada = max(0.0, t - act.primera_ts)
        activo = sum(a['duracion_s'] for a in act.atenciones)
        n = len(act.atenciones)
        return {
            'atenciones': n,
            'atenciones_por_hora': (round(n / (jornada / 3600.0), 2)
                                    if jornada >= 60 else None),
            'tiempo_medio_atencion_s': (round(activo / n, 1) if n else None),
            'tiempo_activo_s': round(activo, 1),
            'tiempo_muerto_s': round(max(0.0, jornada - activo), 1),
            'jornada_s': round(jornada, 1),
        }

    def calidad_dato(self) -> Dict[str, Any]:
        return {
            'personal_enrolado': len(self._galeria),
            'propuestas_pendientes': len(self._propuestos),
            'propuestas_rechazadas': len(self._rechazados),
            'por_origen': {
                o: sum(1 for f in self._galeria.values() if f.origen == o)
                for o in (OrigenPersonal.MANUAL, OrigenPersonal.CONFIRMADO,
                          OrigenPersonal.REID)},
        }
