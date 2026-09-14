"""Demografía del stand: agregación por persona y control del null-rate (HITO 5).

## Qué se corrige exactamente

La auditoría (§3) encontró que el null-rate alto **no es un bug**: el módulo
demográfico global está calibrado a precisión máxima a propósito
(`DEMO_MIN_BLUR_VAR` ×8.3 sobre el valor permisivo, acuerdo de ensemble
obligatorio, nada de resultados provisionales). Para auditar quién entra a una
tienda eso es defendible. Para medir un stand **no**: aquí importa más saber
que pasaron 100 personas con género de 80, que 40 con género de 40.

Este módulo NO toca la configuración global —Tienda y Perimetrales siguen
igual— sino que agrega las muestras con los umbrales del stand
(`stand_config.yaml → demografia`).

## Los dos cambios de fondo

1. **Se agrega por persona, no por frame.** Un voto por frame deja que 25 fps
   de una pose mala aplasten a 3 frames buenos. Aquí cada `persistent_id`
   tiene un buffer de hasta N muestras espaciadas, y el veredicto sale de
   votar sobre ellas.
2. **Un `null` siempre lleva motivo.** `sin_muestras`, `baja_confianza` o
   `track_corto`. Un null mudo no distingue «no se pudo» de «no se intentó», y
   esa ambigüedad es la que hizo invisible el problema original.

## Lo que este módulo NO hace

No infiere. Recibe muestras ya calculadas por las dos ramas existentes
(InsightFace facial / MiVOLO corporal) y decide. Igual que
`stand_identidad.py`, así la lógica se prueba sin GPU.

**Contar no depende de esto.** Un visitante sin demografía se cuenta igual, con
su motivo: es el desacople que arregla el hallazgo §3.2 de la auditoría, donde
`_emitir_visitante:2894` hacía desaparecer a quien no tuviera género.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Vocabulario del stand (la spec pide minúsculas; el pipeline global usa
# "Hombre"/"Mujer" capitalizado — la traducción se hace en el adaptador).
HOMBRE = "hombre"
MUJER = "mujer"
GENEROS = (HOMBRE, MUJER)

# Los MISMOS rangos que ya usa AnalyticsConfig.AGE_RANGES. No se inventan
# otros: el dashboard y las capturas históricas hablan este vocabulario.
RANGOS_ETARIOS = [
    (0, 12, "0-12"), (13, 17, "13-17"), (18, 25, "18-25"),
    (26, 35, "26-35"), (36, 50, "36-50"), (51, 65, "51-65"),
    (66, 200, "65+"),
]

FUENTE_FACIAL = "facial"
FUENTE_CORPORAL = "corporal"


class MotivoSinDemografia:
    """Coincide con el vocabulario de stand_almacen.MotivoSinDemografia."""
    SIN_MUESTRAS = "sin_muestras"
    BAJA_CONFIANZA = "baja_confianza"
    TRACK_CORTO = "track_corto"


# Condiciones de descarte de un crop, para la instrumentación obligatoria.
class Descarte:
    BBOX_PEQUENO = "bbox_pequeno"
    BORROSO = "borroso"
    SIN_ROSTRO = "sin_rostro"
    ROSTRO_PEQUENO = "rostro_pequeno"
    SCORE_BAJO = "score_bajo"
    ESPACIADO = "espaciado"          # llegó demasiado pegada a la anterior
    BUFFER_LLENO = "buffer_lleno"


def rango_de_edad(edad: Optional[float]) -> Optional[str]:
    if edad is None:
        return None
    try:
        e = float(edad)
    except (TypeError, ValueError):
        return None
    for lo, hi, etiqueta in RANGOS_ETARIOS:
        if lo <= e <= hi:
            return etiqueta
    return None


@dataclass
class _Muestra:
    genero: Optional[str]
    genero_conf: float
    edad: Optional[float]
    fuente: str
    frame_idx: int


@dataclass
class Veredicto:
    """Demografía consolidada de una persona."""
    genero: Optional[str] = None
    genero_confianza: float = 0.0
    genero_muestras: int = 0
    edad_estimada: Optional[float] = None
    rango_etario: Optional[str] = None
    edad_muestras: int = 0
    valida: bool = False
    motivo: Optional[str] = None
    fuente: Optional[str] = None

    def como_campos_visitante(self) -> Dict[str, Any]:
        """Traducción directa a los campos de `stand_visitantes`."""
        return {
            'genero': self.genero,
            'genero_confianza': round(self.genero_confianza, 4),
            'genero_muestras': self.genero_muestras,
            'edad_estimada': self.edad_estimada,
            'rango_etario': self.rango_etario,
            'edad_muestras': self.edad_muestras,
            'demografia_valida': self.valida,
            'motivo_sin_demografia': self.motivo,
        }


class AgregadorDemografia:
    """Buffer por persona + voto ponderado + instrumentación."""

    def __init__(self, cfg: Any) -> None:
        g = cfg.get
        self._max_muestras = int(g('demografia.buffer.max_muestras', 10))
        self._espaciado = int(g('demografia.buffer.espaciado_frames', 5))
        self._min_validas = int(g('demografia.buffer.min_muestras_validas', 3))
        self._conf_min = float(g('demografia.genero.confianza_min', 0.65))
        # Una muestra corporal pesa menos que una facial: es el mismo criterio
        # que ya aplica DEMO_BODY_SAMPLE_WEIGHT en el pipeline global.
        self._peso_corporal = float(g('demografia.peso_muestra_corporal', 0.5))

        self._buffers: Dict[str, List[_Muestra]] = {}
        self._ultimo_frame: Dict[str, int] = {}
        self._descartes: Dict[str, int] = {}
        self._crops_recibidos = 0
        self._crops_aceptados = 0
        self._por_rama: Dict[str, int] = {FUENTE_FACIAL: 0,
                                          FUENTE_CORPORAL: 0}

    # ── Ingesta ─────────────────────────────────────────────────────────
    def muestra(self, persistent_id: str, *,
                genero: Optional[str] = None,
                genero_conf: float = 0.0,
                edad: Optional[float] = None,
                fuente: str = FUENTE_CORPORAL,
                frame_idx: int = 0) -> bool:
        """Añade una muestra. Devuelve False si se descartó (y por qué, en
        los contadores)."""
        self._crops_recibidos += 1

        anterior = self._ultimo_frame.get(persistent_id)
        if anterior is not None and (frame_idx - anterior) < self._espaciado:
            # Muestras pegadas son casi idénticas: inflan el voto sin aportar
            # información y hacen que una pose mala domine el resultado.
            self._descartes[Descarte.ESPACIADO] = \
                self._descartes.get(Descarte.ESPACIADO, 0) + 1
            return False

        buf = self._buffers.setdefault(persistent_id, [])
        if len(buf) >= self._max_muestras:
            # Buffer circular: la más antigua sale.
            buf.pop(0)
            self._descartes[Descarte.BUFFER_LLENO] = \
                self._descartes.get(Descarte.BUFFER_LLENO, 0) + 1

        g = (genero or '').strip().lower() or None
        if g is not None and g not in GENEROS:
            g = None
        buf.append(_Muestra(g, float(genero_conf or 0.0), edad, fuente,
                            frame_idx))
        self._ultimo_frame[persistent_id] = frame_idx
        self._crops_aceptados += 1
        self._por_rama[fuente] = self._por_rama.get(fuente, 0) + 1
        return True

    def descartar_crop(self, motivo: str) -> None:
        """Registra un crop que ni llegó a inferencia. Instrumentación
        obligatoria del HITO 5: sin esto no se puede saber DÓNDE se pierde
        la demografía."""
        self._crops_recibidos += 1
        self._descartes[motivo] = self._descartes.get(motivo, 0) + 1

    def olvidar(self, persistent_id: str) -> None:
        self._buffers.pop(persistent_id, None)
        self._ultimo_frame.pop(persistent_id, None)

    # ── Veredicto ───────────────────────────────────────────────────────
    def veredicto(self, persistent_id: str) -> Veredicto:
        """Consolida la demografía de una persona.

        Género por voto PONDERADO por confianza (facial pesa el doble que
        corporal); edad por MEDIANA, que es robusta a un outlier — con media,
        una sola estimación de 80 años arrastra a todo el track.
        """
        buf = self._buffers.get(persistent_id) or []
        if not buf:
            return Veredicto(motivo=MotivoSinDemografia.SIN_MUESTRAS)

        con_genero = [m for m in buf if m.genero is not None]
        if len(con_genero) < self._min_validas:
            return Veredicto(
                genero_muestras=len(con_genero),
                motivo=(MotivoSinDemografia.TRACK_CORTO if len(buf) < self._min_validas
                        else MotivoSinDemografia.SIN_MUESTRAS))

        pesos: Dict[str, float] = {}
        for m in con_genero:
            peso = m.genero_conf * (1.0 if m.fuente == FUENTE_FACIAL
                                    else self._peso_corporal)
            pesos[m.genero] = pesos.get(m.genero, 0.0) + peso

        total = sum(pesos.values())
        if total <= 0:
            return Veredicto(genero_muestras=len(con_genero),
                             motivo=MotivoSinDemografia.BAJA_CONFIANZA)

        ganador = max(pesos, key=lambda k: pesos[k])
        ratio = pesos[ganador] / total
        if ratio < self._conf_min:
            # Las muestras se contradicen: mejor null honesto que una
            # moneda al aire presentada como dato.
            return Veredicto(genero_muestras=len(con_genero),
                             genero_confianza=ratio,
                             motivo=MotivoSinDemografia.BAJA_CONFIANZA)

        edades = [float(m.edad) for m in buf if m.edad is not None]
        edad_final = round(median(edades), 1) if edades else None

        hay_facial = any(m.fuente == FUENTE_FACIAL for m in con_genero)
        hay_corporal = any(m.fuente == FUENTE_CORPORAL for m in con_genero)
        fuente = ('mixto' if hay_facial and hay_corporal
                  else (FUENTE_FACIAL if hay_facial else FUENTE_CORPORAL))

        return Veredicto(
            genero=ganador, genero_confianza=ratio,
            genero_muestras=len(con_genero),
            edad_estimada=edad_final, rango_etario=rango_de_edad(edad_final),
            edad_muestras=len(edades), valida=True, motivo=None,
            fuente=fuente)

    # ── Instrumentación ─────────────────────────────────────────────────
    def null_rate(self, persistent_ids: Iterable[str]) -> Dict[str, Any]:
        """Null-rate sobre un conjunto de personas.

        El criterio del HITO 5 se mide sobre **visitantes que se detuvieron**,
        no sobre todos: a quien cruza el encuadre en dos frames no se le puede
        exigir demografía, y meterlo en el denominador maquilla el número
        hacia arriba en dificultad.
        """
        pids = list(persistent_ids)
        if not pids:
            return {'total': 0, 'con_demografia': 0, 'null_rate': None,
                    'por_motivo': {}, 'por_rama': {}}

        con, motivos, ramas = 0, {}, {}
        for pid in pids:
            v = self.veredicto(pid)
            if v.valida:
                con += 1
                ramas[v.fuente] = ramas.get(v.fuente, 0) + 1
            else:
                motivos[v.motivo] = motivos.get(v.motivo, 0) + 1
        return {
            'total': len(pids),
            'con_demografia': con,
            'null_rate': round(1.0 - (con / len(pids)), 4),
            'por_motivo': motivos,
            'por_rama': ramas,
        }

    def salud(self) -> Dict[str, Any]:
        """Contadores en vivo para `stand:salud` y el panel de calidad."""
        return {
            'crops_recibidos': self._crops_recibidos,
            'crops_aceptados': self._crops_aceptados,
            'crops_descartados': self._crops_recibidos - self._crops_aceptados,
            'descartes_por_condicion': dict(self._descartes),
            'muestras_por_rama': dict(self._por_rama),
            'personas_con_buffer': len(self._buffers),
        }
