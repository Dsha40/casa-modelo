"""Libro de estancias abiertas: quién está dentro y desde cuándo.

El cronómetro del área se podía llevar en memoria porque la persona está
**a la vista** todo el rato. En una puerta no: quien entra desaparece del
encuadre y puede tardar horas en volver a salir. Ese hueco es lo que resuelve
este módulo.

## Qué guarda de cada persona

Lo mínimo para poder cerrar la estancia y decir «esta es la misma que entró»:

  - el `persistent_id` (identidad de rostro del pipeline, la que manda);
  - un **embedding corporal** de respaldo, para reconocerla si sale de
    espaldas y el rostro no confirma;
  - sus **rasgos** consolidados (género y rango de edad) como tercera señal
    de descarte;
  - la **foto de entrada** en base64, que viaja luego en la alerta de salida.

## Por qué se persiste a disco

Un reinicio del servidor a media tarde no puede «vaciar» un almacén: la gente
que estaba dentro seguiría dentro y sus salidas quedarían sin pareja. El libro
se escribe en JSON por cámara y se relee al arrancar.

La escritura es **atómica** (archivo temporal + reemplazo) y está limitada en
frecuencia: escribir en cada frame costaría más que la inferencia.

## Por qué no vale reusar la galería corporal del pipeline

`BodyReidentifier` es efímera a propósito (TTL de 4 h) porque la ropa cambia de
un día para otro, y es **compartida** entre modos. Aquí hace falta lo contrario:
una copia privada, con TTL propio y largo, que sobreviva a la purga de esa
galería y no la contamine con identidades de puertas.

## El TTL, y qué pasa al vencer

Una estancia sin cerrar más vieja que `ttl_horas` se cierra como **`sin_salida`**.
No se inventa una hora de salida: la alerta dice explícitamente que la salida
nunca se vio y da la duración como cota inferior. Sin esto, alguien que se fue
por otra puerta se quedaría «dentro» para siempre e inflaría el aforo.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Motivos por los que una estancia deja de estar abierta.
CIERRE_SALIDA = 'salida'
CIERRE_SIN_SALIDA = 'sin_salida'

#: Cómo se reconoció a quien sale. Se reporta tal cual en la alerta: si fue
#: por cuerpo y no por rostro, quien la lee debe poder saberlo.
POR_ROSTRO = 'rostro'
POR_CUERPO = 'cuerpo'
POR_ZONA = 'zona'
SIN_ATRIBUIR = 'no atribuida'


def coseno(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Similitud coseno robusta. `-1.0` cuando alguno falta o es nulo."""
    if a is None or b is None:
        return -1.0
    try:
        va = np.asarray(a, dtype=np.float32).ravel()
        vb = np.asarray(b, dtype=np.float32).ravel()
        if va.size == 0 or va.size != vb.size:
            return -1.0
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na <= 0.0 or nb <= 0.0:
            return -1.0
        return float(np.dot(va, vb) / (na * nb))
    except Exception:                                       # noqa: BLE001
        return -1.0


@dataclass
class Estancia:
    """Una permanencia en curso. Vive desde la entrada hasta la salida."""

    persistent_id: str
    camara: str = ''
    zona: str = 'area'                      # 'area' | 'puerta'
    entrada_ts: float = 0.0
    salida_ts: Optional[float] = None
    # Evidencia de entrada (ruta en disco + base64 para la alerta de salida).
    entrada_foto: str = ''
    entrada_b64: str = ''
    # La ESCENA completa del instante de entrada (sin recorte): es el cuadro
    # «Inicio» de la foto apilada que viaja en la alerta de salida.
    entrada_escena_b64: str = ''
    # Última caja conocida: la foto de salida sale de aquí si en el frame de
    # cierre la persona ya no tiene track.
    box: Optional[Tuple[float, float, float, float]] = None
    # Respaldo de identidad. `None` = entró sin que se le pudiera medir el
    # cuerpo (demasiado lejos o demasiado pequeña).
    emb_cuerpo: Optional[np.ndarray] = None
    rasgos: Dict[str, Any] = field(default_factory=dict)
    # Avisos de permanencia prolongada ya emitidos.
    avisos_permanencia: int = 0
    ultimo_aviso_ts: float = 0.0
    # Última vez que se la vio en el encuadre (en puerta puede ser antiquísima).
    ultima_vista_ts: float = 0.0
    identidad_por: str = POR_ROSTRO
    # Confianza del emparejamiento de SALIDA. `None` mientras siga dentro.
    score_salida: Optional[float] = None
    # -- Por qué puerta (7-sep-2026) --------------------------------------
    # `-1` es el aforo de la CÁMARA: el camino de siempre (modo área, cruce
    # y línea suelta), en el que no hay ninguna puerta dibujada. Con varias
    # puertas en la misma cámara la estancia tiene que saber por cuál entró:
    # sin este dato, una salida por la trasera se emparejaría con la entrada
    # por la principal y las fundiría en una visita a un sitio por el que esa
    # persona no pasó.
    puerta: int = -1
    # El nombre que le puso el operador («Almacén», «Patio») es SOLO para
    # mostrar: la identidad de la puerta es el ÍNDICE (misma regla que
    # `RegistroAnaqueles.area`). Si la identidad fuese el nombre, renombrarla
    # a media tarde bifurcaría el aforo en silencio y la gente que ya estaba
    # dentro quedaría en una sección inalcanzable.
    puerta_nombre: str = ''
    # Los dos van AL FINAL de la lista de campos a propósito: `Estancia` se
    # construye por palabra clave, pero añadirlos en medio cambiaría el orden
    # posicional del dataclass y rompería a cualquiera que la construya así.

    def duracion(self, ahora: float) -> float:
        fin = self.salida_ts if self.salida_ts is not None else ahora
        return max(0.0, fin - self.entrada_ts)

    # -- Serialización ----------------------------------------------------
    def a_dict(self, con_foto: bool = True) -> Dict[str, Any]:
        datos: Dict[str, Any] = {
            'persistent_id': self.persistent_id,
            'camara': self.camara,
            'zona': self.zona,
            'entrada_ts': float(self.entrada_ts),
            'salida_ts': self.salida_ts,
            'entrada_foto': self.entrada_foto,
            'box': list(self.box) if self.box else None,
            'emb_cuerpo': (np.asarray(self.emb_cuerpo,
                                      dtype=np.float32).ravel().tolist()
                           if self.emb_cuerpo is not None else None),
            'rasgos': dict(self.rasgos or {}),
            'avisos_permanencia': int(self.avisos_permanencia),
            'ultimo_aviso_ts': float(self.ultimo_aviso_ts),
            'ultima_vista_ts': float(self.ultima_vista_ts),
            'identidad_por': self.identidad_por,
            # 7-sep-2026: la puerta se persiste porque el libro se relee tras
            # un reinicio y cada estancia tiene que volver a SU sección. Sin
            # este par, al recargar todo el mundo caería en el aforo único
            # (-1) y los aforos por puerta quedarían a cero con el vídeo
            # perfecto. Se escribe también con `con_foto=False` (el volcado
            # ligero del panel): es identidad, no evidencia.
            'puerta': int(self.puerta),
            'puerta_nombre': self.puerta_nombre,
        }
        if con_foto:
            datos['entrada_b64'] = self.entrada_b64
            datos['entrada_escena_b64'] = self.entrada_escena_b64
        return datos

    @classmethod
    def desde_dict(cls, datos: Dict[str, Any]) -> Optional['Estancia']:
        """`None` si el registro está corrupto: un JSON a medias no puede
        tumbar el arranque del pipeline."""
        try:
            pid = str(datos['persistent_id'])
            emb = datos.get('emb_cuerpo')
            caja = datos.get('box')
            return cls(
                persistent_id=pid,
                camara=str(datos.get('camara') or ''),
                zona=str(datos.get('zona') or 'area'),
                entrada_ts=float(datos.get('entrada_ts') or 0.0),
                salida_ts=(float(datos['salida_ts'])
                           if datos.get('salida_ts') is not None else None),
                entrada_foto=str(datos.get('entrada_foto') or ''),
                entrada_b64=str(datos.get('entrada_b64') or ''),
                entrada_escena_b64=str(datos.get('entrada_escena_b64') or ''),
                box=(tuple(float(v) for v in caja[:4]) if caja else None),
                emb_cuerpo=(np.asarray(emb, dtype=np.float32)
                            if emb else None),
                rasgos=dict(datos.get('rasgos') or {}),
                avisos_permanencia=int(datos.get('avisos_permanencia') or 0),
                ultimo_aviso_ts=float(datos.get('ultimo_aviso_ts') or 0.0),
                ultima_vista_ts=float(datos.get('ultima_vista_ts') or 0.0),
                identidad_por=str(datos.get('identidad_por') or POR_ROSTRO),
                # Un libro escrito ANTES del 7-sep-2026 no trae estas dos
                # claves: se leen como el aforo único (-1, ''), que es
                # exactamente lo que esas estancias eran. Así el fichero que
                # ya está en disco se sigue leyendo entero.
                #
                # Y `int(datos.get('puerta') or -1)` NO vale: la puerta 0 es
                # la PRIMERA (es un índice, no un contador) y ese `or` la
                # convertiría en -1, mandando a toda su gente al aforo de la
                # cámara en cada reinicio. Por eso se compara con `None`.
                puerta=(int(datos['puerta'])
                        if datos.get('puerta') is not None else -1),
                puerta_nombre=str(datos.get('puerta_nombre') or ''),
            )
        except Exception as exc:                            # noqa: BLE001
            logger.warning("Estancia ilegible en el libro, se descarta: %s",
                           exc)
            return None


class RegistroEstancias:
    """Las estancias abiertas de UNA cámara, con respaldo en disco."""

    def __init__(self, ruta: Optional[Path] = None,
                 ttl_horas: float = 24.0,
                 umbral_cuerpo: float = 0.86,
                 max_abiertas: int = 200,
                 persistir: bool = True) -> None:
        self.ruta = Path(ruta) if ruta else None
        self.ttl_s = max(60.0, float(ttl_horas) * 3600.0)
        self.umbral_cuerpo = float(umbral_cuerpo)
        self.max_abiertas = max(1, int(max_abiertas))
        self.persistir = bool(persistir) and self.ruta is not None
        #: `persistent_id` -> Estancia. Es un dict normal a propósito: el
        #: pipeline lo lee directamente para pintar el panel.
        self.abiertas: Dict[str, Estancia] = {}
        self._sucio = False
        self._ultimo_guardado = 0.0
        self._intervalo_guardado = 5.0

    # -- Ciclo de vida de una estancia ------------------------------------
    def abrir(self, estancia: Estancia) -> Estancia:
        """Registra una entrada. Si ya estaba abierta, NO reinicia el reloj.

        Reabrir sería el bug clásico: una persona que parpadea en el borde de
        la zona acabaría con el cronómetro a cero cada pocos segundos y no
        superaría nunca el umbral de permanencia.
        """
        previa = self.abiertas.get(estancia.persistent_id)
        if previa is not None:
            previa.ultima_vista_ts = max(previa.ultima_vista_ts,
                                         estancia.ultima_vista_ts)
            if estancia.box is not None:
                previa.box = estancia.box
            if estancia.emb_cuerpo is not None and previa.emb_cuerpo is None:
                previa.emb_cuerpo = estancia.emb_cuerpo
            if estancia.rasgos:
                previa.rasgos.update(estancia.rasgos)
            self._sucio = True
            return previa

        self.abiertas[estancia.persistent_id] = estancia
        self._recortar()
        self._sucio = True
        return estancia

    def cerrar(self, persistent_id: str, ahora: Optional[float] = None,
               motivo: str = CIERRE_SALIDA,
               score: Optional[float] = None) -> Optional[Estancia]:
        """Saca la estancia del libro y le pone hora de salida."""
        est = self.abiertas.pop(str(persistent_id), None)
        if est is None:
            return None
        est.salida_ts = time.time() if ahora is None else float(ahora)
        est.score_salida = score
        if motivo == CIERRE_SIN_SALIDA:
            est.identidad_por = SIN_ATRIBUIR
        self._sucio = True
        return est

    def get(self, persistent_id: Any) -> Optional[Estancia]:
        return self.abiertas.get(str(persistent_id))

    def __len__(self) -> int:
        return len(self.abiertas)

    def __contains__(self, persistent_id: Any) -> bool:
        return str(persistent_id) in self.abiertas

    # -- Emparejar una salida ---------------------------------------------
    def emparejar_por_cuerpo(self, embedding: Optional[np.ndarray],
                             umbral: Optional[float] = None
                             ) -> Optional[Tuple[Estancia, float]]:
        """Mejor estancia abierta cuyo cuerpo se parezca al que sale.

        Devuelve `None` —y hay que decirlo en la alerta— cuando ninguna llega
        al umbral. Emparejar «con la más probable» sin umbral daría siempre
        una respuesta, y estaría mal la mitad de las veces.
        """
        if embedding is None:
            return None
        thr = self.umbral_cuerpo if umbral is None else float(umbral)
        mejor: Optional[Estancia] = None
        mejor_score = -1.0
        for est in self.abiertas.values():
            if est.emb_cuerpo is None:
                continue
            score = coseno(est.emb_cuerpo, embedding)
            if score > mejor_score:
                mejor_score, mejor = score, est
        if mejor is not None and mejor_score >= thr:
            return mejor, mejor_score
        return None

    def unica_abierta(self) -> Optional[Estancia]:
        """La única persona dentro, si es que hay exactamente una.

        Es el desempate honesto de la puerta: con una sola estancia abierta,
        quien sale solo puede ser esa persona. Con dos o más, sin Re-ID no se
        puede saber y se reporta como no atribuida.
        """
        if len(self.abiertas) == 1:
            return next(iter(self.abiertas.values()))
        return None

    # -- Caducidad ---------------------------------------------------------
    def caducadas(self, ahora: Optional[float] = None) -> List[Estancia]:
        """Cierra y devuelve las estancias que pasaron del TTL."""
        ahora = time.time() if ahora is None else float(ahora)
        vencidas = [pid for pid, est in self.abiertas.items()
                    if (ahora - est.entrada_ts) > self.ttl_s]
        return [e for e in (self.cerrar(pid, ahora, CIERRE_SIN_SALIDA)
                            for pid in vencidas) if e is not None]

    def _recortar(self) -> None:
        """Cota dura de memoria: fuera las más antiguas si se desborda."""
        if len(self.abiertas) <= self.max_abiertas:
            return
        sobran = len(self.abiertas) - self.max_abiertas
        viejas = sorted(self.abiertas.values(),
                        key=lambda e: e.entrada_ts)[:sobran]
        for est in viejas:
            self.abiertas.pop(est.persistent_id, None)
        logger.warning("Registro de estancias lleno (%d): se descartaron las "
                       "%d entradas más antiguas", self.max_abiertas, sobran)

    # -- Disco -------------------------------------------------------------
    def cargar(self) -> int:
        """Relee el libro tras un reinicio. Devuelve cuántas recuperó."""
        if not self.persistir or self.ruta is None or not self.ruta.is_file():
            return 0
        try:
            datos = json.loads(self.ruta.read_text(encoding='utf-8'))
        except Exception as exc:                            # noqa: BLE001
            logger.warning("Libro de estancias ilegible (%s): se empieza "
                           "vacío", exc)
            return 0
        ahora = time.time()
        recuperadas = 0
        for crudo in (datos.get('abiertas') or []):
            est = Estancia.desde_dict(crudo)
            if est is None:
                continue
            if (ahora - est.entrada_ts) > self.ttl_s:
                continue                    # ya vencida: no resucita
            self.abiertas[est.persistent_id] = est
            recuperadas += 1
        if recuperadas:
            logger.info("Permanencia: %d estancia(s) seguían abiertas al "
                        "reiniciar y se recuperaron de %s",
                        recuperadas, self.ruta.name)
        return recuperadas

    def guardar(self, forzar: bool = False) -> bool:
        """Escritura atómica y limitada en frecuencia. Nunca lanza."""
        if not self.persistir or self.ruta is None:
            return False
        ahora = time.time()
        if not forzar:
            if not self._sucio:
                return False
            if (ahora - self._ultimo_guardado) < self._intervalo_guardado:
                return False
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            cuerpo = json.dumps(
                {'version': 1, 'guardado_ts': ahora,
                 'abiertas': [e.a_dict() for e in self.abiertas.values()]},
                ensure_ascii=False)
            tmp = self.ruta.with_suffix('.tmp')
            tmp.write_text(cuerpo, encoding='utf-8')
            os.replace(tmp, self.ruta)      # atómico en NTFS y en POSIX
            self._sucio = False
            self._ultimo_guardado = ahora
            return True
        except Exception as exc:                            # noqa: BLE001
            logger.debug("No se pudo guardar el libro de estancias: %s", exc)
            return False
