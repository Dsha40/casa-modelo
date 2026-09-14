"""Detección de degustación: pose + verificador VLM (HITO 8).

La métrica más difícil del sistema y la más importante: es la que dice si el
stand cumple su función. Diseño en dos etapas para no saturar la GPU.

## Etapa 1 — disparador barato (pose)

ViTPose-base (antes `yolo11m-pose.pt`, que se midió a 41 fps / 0.11 GiB en la
RTX 5060 Ti) corre **solo sobre visitantes ya atendidos**, dentro de una
ventana tras el fin de la atención. Condición candidata: muñeca cerca de la
boca, con el codo flexionado, sostenido N frames dentro de una ventana corta.

No se exige que los frames sean **consecutivos**: llevarse algo a la boca es
un gesto con oclusiones (la propia mano tapa la cara, otra persona cruza), y
exigir continuidad perdía degustaciones reales.

## Etapa 2 — verificador VLM

La pose sola confunde «comer» con «rascarse la nariz», «hablar por teléfono» o
«toser». Qwen2.5-VL recibe una ráfaga de recortes y responde JSON cerrado. Solo
se confirma con `degustando: true` **y** confianza sobre el umbral.

## Control de coste

El VLM es caro y va en la misma GPU que el pipeline. Hay un tope por minuto y
una **cola**: al saturar se encola, nunca se descarta. Descartar sesgaría la
métrica justo en los momentos de más tráfico —cuando más degustaciones hay—,
que es exactamente cuando el dato importa.

## Sin GPU en las pruebas

La geometría de pose trabaja sobre arrays de keypoints y el cliente VLM se
inyecta, así que toda la lógica de decisión se prueba con datos sintéticos.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Índices de keypoints COCO (los de yolo11*-pose) ──────────────────────
NARIZ = 0
OJO_IZQ, OJO_DER = 1, 2
OREJA_IZQ, OREJA_DER = 3, 4
HOMBRO_IZQ, HOMBRO_DER = 5, 6
CODO_IZQ, CODO_DER = 7, 8
MUNECA_IZQ, MUNECA_DER = 9, 10

CONF_KP_MIN = 0.30          # por debajo, el keypoint es ruido

METODO_POSE_VLM = "pose+vlm"
METODO_POSE = "pose"


def _kp(keypoints: Sequence[Sequence[float]],
        idx: int) -> Optional[Tuple[float, float]]:
    """Keypoint (x, y) si su confianza es suficiente."""
    try:
        k = keypoints[idx]
    except (IndexError, TypeError):
        return None
    if len(k) >= 3 and float(k[2]) < CONF_KP_MIN:
        return None
    return (float(k[0]), float(k[1]))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ancho_cabeza(keypoints: Sequence[Sequence[float]]) -> Optional[float]:
    """Escala de referencia de la persona, en píxeles.

    Normalizar por el ancho de cabeza hace el umbral independiente de la
    distancia a la cámara: sin esto, alguien al fondo nunca dispararía y
    alguien cerca dispararía siempre.
    """
    oi, od = _kp(keypoints, OREJA_IZQ), _kp(keypoints, OREJA_DER)
    if oi and od:
        d = _dist(oi, od)
        if d > 1.0:
            return d
    pi, pd = _kp(keypoints, OJO_IZQ), _kp(keypoints, OJO_DER)
    if pi and pd:
        d = _dist(pi, pd)
        if d > 1.0:
            return d * 2.0          # ojos ≈ mitad del ancho de cabeza
    hi, hd = _kp(keypoints, HOMBRO_IZQ), _kp(keypoints, HOMBRO_DER)
    if hi and hd:
        d = _dist(hi, hd)
        if d > 1.0:
            return d * 0.45         # cabeza ≈ 45% del ancho de hombros
    return None


def angulo_codo(hombro: Tuple[float, float], codo: Tuple[float, float],
                muneca: Tuple[float, float]) -> float:
    """Ángulo en el codo, en grados. 180 = brazo estirado."""
    v1 = (hombro[0] - codo[0], hombro[1] - codo[1])
    v2 = (muneca[0] - codo[0], muneca[1] - codo[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


@dataclass
class GestoPose:
    """Resultado de evaluar un frame."""
    candidato: bool
    ratio: Optional[float] = None          # distancia muñeca-boca / cabeza
    angulo: Optional[float] = None
    lado: Optional[str] = None
    motivo: Optional[str] = None


class DisparadorPose:
    """Etapa 1: geometría pura sobre keypoints."""

    def __init__(self, cfg: Any) -> None:
        g = cfg.get
        self._ratio_max = float(g('degustacion.pose.ratio_muneca_boca', 0.35))
        self._frames_min = int(g('degustacion.pose.frames_min', 8))
        self._ventana_s = float(g('degustacion.pose.ventana_s', 2.0))
        # Codo flexionado: un brazo estirado no lleva nada a la boca.
        self._angulo_max = float(g('degustacion.pose.angulo_codo_max', 120.0))
        self._hits: Dict[str, Deque[float]] = {}

    def evaluar_frame(self,
                      keypoints: Sequence[Sequence[float]]) -> GestoPose:
        """¿Este frame muestra una mano cerca de la boca?"""
        boca = _kp(keypoints, NARIZ)
        if boca is None:
            return GestoPose(False, motivo='sin_cara')
        cabeza = ancho_cabeza(keypoints)
        if cabeza is None or cabeza <= 0:
            return GestoPose(False, motivo='sin_escala')

        mejor = GestoPose(False, motivo='mano_lejos')
        for lado, i_h, i_c, i_m in (('izq', HOMBRO_IZQ, CODO_IZQ, MUNECA_IZQ),
                                    ('der', HOMBRO_DER, CODO_DER, MUNECA_DER)):
            muneca = _kp(keypoints, i_m)
            if muneca is None:
                continue
            ratio = _dist(muneca, boca) / cabeza
            hombro, codo = _kp(keypoints, i_h), _kp(keypoints, i_c)
            ang = (angulo_codo(hombro, codo, muneca)
                   if (hombro and codo) else None)

            if ratio <= self._ratio_max:
                if ang is not None and ang > self._angulo_max:
                    if mejor.motivo != 'mano_cerca':
                        mejor = GestoPose(False, ratio, ang, lado,
                                          'codo_estirado')
                    continue
                return GestoPose(True, ratio, ang, lado)
            if mejor.ratio is None or ratio < mejor.ratio:
                mejor = GestoPose(False, ratio, ang, lado, 'mano_lejos')
        return mejor

    def observar(self, persistent_id: str,
                 keypoints: Sequence[Sequence[float]],
                 ts: float) -> Tuple[bool, GestoPose]:
        """Acumula el gesto. Devuelve (dispara, detalle del frame)."""
        gesto = self.evaluar_frame(keypoints)
        cola = self._hits.setdefault(persistent_id, deque())
        if gesto.candidato:
            cola.append(ts)
        # Solo cuentan los hits dentro de la ventana: no consecutivos, pero
        # sí concentrados en el tiempo.
        while cola and (ts - cola[0]) > self._ventana_s:
            cola.popleft()
        return (len(cola) >= self._frames_min, gesto)

    def reiniciar(self, persistent_id: str) -> None:
        self._hits.pop(persistent_id, None)


# ── Etapa 2: verificador VLM ─────────────────────────────────────────────
PROMPT_DEGUSTACION = (
    "¿La persona en estas imágenes se está llevando comida o bebida a la "
    "boca, o está comiendo/probando algo? Responde únicamente en JSON: "
    '{"degustando": true|false, "confianza": 0.0-1.0, '
    '"evidencia": "descripción breve"}'
)


@dataclass
class Confirmacion:
    persistent_id: str
    degustando: bool
    confianza: float
    evidencia: str
    metodo: str
    ts: float


@dataclass
class _Pendiente:
    persistent_id: str
    ts: float
    intentos: int = 0


class PlanificadorVLM:
    """Cola con tope por minuto. Encola, nunca descarta.

    Descartar en saturación sesgaría la métrica justo en las horas punta,
    que es cuando más degustaciones ocurren y cuando el dato más importa.
    """

    def __init__(self, cfg: Any) -> None:
        self._max_por_min = int(cfg.get('degustacion.vlm.max_por_minuto', 20))
        self._cola: Deque[_Pendiente] = deque()
        self._encolados: set[str] = set()
        self._marcas: Deque[float] = deque()

    def encolar(self, persistent_id: str, ts: float) -> bool:
        """False si esa persona ya estaba en cola (no se duplica)."""
        if persistent_id in self._encolados:
            return False
        self._cola.append(_Pendiente(persistent_id, ts))
        self._encolados.add(persistent_id)
        return True

    def _hay_cupo(self, ahora: float) -> bool:
        while self._marcas and (ahora - self._marcas[0]) > 60.0:
            self._marcas.popleft()
        return len(self._marcas) < self._max_por_min

    def siguiente(self, ahora: float) -> Optional[str]:
        """Siguiente persona a verificar, o None si no hay cupo/cola."""
        if not self._cola or not self._hay_cupo(ahora):
            return None
        p = self._cola.popleft()
        self._encolados.discard(p.persistent_id)
        self._marcas.append(ahora)
        return p.persistent_id

    @property
    def en_cola(self) -> int:
        return len(self._cola)

    def consumidas_ultimo_minuto(self, ahora: float) -> int:
        while self._marcas and (ahora - self._marcas[0]) > 60.0:
            self._marcas.popleft()
        return len(self._marcas)


def parsear_respuesta_vlm(texto: str) -> Optional[Dict[str, Any]]:
    """Extrae el JSON de la respuesta del VLM.

    Los VLM envuelven el JSON en prosa o en ```json ... ``` pese a pedirles
    lo contrario. Se busca el primer objeto balanceado en vez de confiar en
    que la respuesta sea JSON puro.
    """
    if not texto:
        return None
    ini = texto.find('{')
    if ini < 0:
        return None
    profundidad = 0
    for i in range(ini, len(texto)):
        if texto[i] == '{':
            profundidad += 1
        elif texto[i] == '}':
            profundidad -= 1
            if profundidad == 0:
                try:
                    d = json.loads(texto[ini:i + 1])
                except ValueError:
                    return None
                return d if isinstance(d, dict) else None
    return None


class DetectorDegustacion:
    """Orquesta las dos etapas y aplica la ventana post-atención."""

    def __init__(self, cfg: Any,
                 cliente_vlm: Optional[Callable[[List[Any], str], str]] = None
                 ) -> None:
        self._cfg = cfg
        self._pose = DisparadorPose(cfg)
        self._plan = PlanificadorVLM(cfg)
        self._cliente_vlm = cliente_vlm
        self._conf_min = float(cfg.get('degustacion.vlm.confianza_min', 0.7))
        self._ventana_post = float(
            cfg.get('degustacion.ventana_post_atencion_s', 90))
        self._confirmados: Dict[str, Confirmacion] = {}
        self._stats = {'frames_pose': 0, 'disparos_pose': 0,
                       'verificaciones_vlm': 0, 'confirmadas': 0,
                       'rechazadas_vlm': 0, 'errores_vlm': 0}

    def elegible(self, *, fue_atendido: bool, ts: float,
                 ts_fin_atencion: Optional[float]) -> bool:
        """¿Merece la pena correr pose sobre esta persona?

        Solo sobre atendidos y dentro de la ventana posterior. Correr pose
        sobre todo el mundo multiplicaría el coste sin aportar: nadie degusta
        sin que le hayan dado la muestra.
        """
        if not fue_atendido:
            return False
        if ts_fin_atencion is None:
            return True                     # atención todavía en curso
        return (ts - ts_fin_atencion) <= self._ventana_post

    def observar(self, persistent_id: str,
                 keypoints: Sequence[Sequence[float]], ts: float,
                 *, fue_atendido: bool = True,
                 ts_fin_atencion: Optional[float] = None) -> bool:
        """Alimenta la etapa 1. True si acaba de encolarse para el VLM."""
        if persistent_id in self._confirmados:
            return False
        if not self.elegible(fue_atendido=fue_atendido, ts=ts,
                             ts_fin_atencion=ts_fin_atencion):
            return False

        self._stats['frames_pose'] += 1
        dispara, _gesto = self._pose.observar(persistent_id, keypoints, ts)
        if not dispara:
            return False
        self._stats['disparos_pose'] += 1
        return self._plan.encolar(persistent_id, ts)

    def procesar_cola(self, obtener_frames: Callable[[str], List[Any]],
                      ahora: Optional[float] = None
                      ) -> List[Confirmacion]:
        """Consume la cola dentro del cupo. Llamar una vez por ciclo."""
        t = float(ahora if ahora is not None else time.time())
        salida: List[Confirmacion] = []

        while True:
            pid = self._plan.siguiente(t)
            if pid is None:
                break
            frames = obtener_frames(pid) or []
            if not frames:
                continue
            conf = self._verificar(pid, frames, t)
            if conf is not None:
                salida.append(conf)
        return salida

    def _verificar(self, persistent_id: str, frames: List[Any],
                   ts: float) -> Optional[Confirmacion]:
        if self._cliente_vlm is None:
            # Sin VLM disponible NO se confirma por pose sola: la pose
            # confunde comer con rascarse la nariz o hablar por teléfono, y
            # una precisión baja aquí arruina la métrica principal.
            logger.warning("stand: sin cliente VLM; degustación de %s "
                           "queda sin confirmar", persistent_id)
            return None

        self._stats['verificaciones_vlm'] += 1
        try:
            crudo = self._cliente_vlm(frames, PROMPT_DEGUSTACION)
        except Exception as exc:                            # noqa: BLE001
            self._stats['errores_vlm'] += 1
            logger.error("stand: el VLM falló para %s: %s",
                         persistent_id, exc)
            return None

        datos = parsear_respuesta_vlm(crudo if isinstance(crudo, str)
                                      else json.dumps(crudo))
        if datos is None:
            self._stats['errores_vlm'] += 1
            logger.error("stand: respuesta VLM no parseable para %s",
                         persistent_id)
            return None

        degustando = bool(datos.get('degustando'))
        try:
            confianza = float(datos.get('confianza', 0.0))
        except (TypeError, ValueError):
            confianza = 0.0
        evidencia = str(datos.get('evidencia', ''))[:300]

        if not degustando or confianza < self._conf_min:
            self._stats['rechazadas_vlm'] += 1
            return None

        conf = Confirmacion(persistent_id, True, confianza, evidencia,
                            METODO_POSE_VLM, ts)
        self._confirmados[persistent_id] = conf
        self._pose.reiniciar(persistent_id)
        self._stats['confirmadas'] += 1
        return conf

    # ── Salud ───────────────────────────────────────────────────────────
    def salud(self, ahora: Optional[float] = None) -> Dict[str, Any]:
        t = float(ahora if ahora is not None else time.time())
        return {
            **self._stats,
            'vlm_en_cola': self._plan.en_cola,
            'vlm_ultimo_minuto': self._plan.consumidas_ultimo_minuto(t),
        }
