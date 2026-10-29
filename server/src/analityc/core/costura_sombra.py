"""Unir seguimientos cortados, EN SOMBRA (20-oct-2026).

Cuando ByteTrack pierde a alguien y le da un número nuevo, la persona sin
cara se pierde (`prov:`) y vuelve como visitante nuevo. Medido con 47 parejas
etiquetadas por el usuario (grabaciones del 16-oct): la regla «reaparece en
<= 3 s, a menos de un alto de persona y con el cuerpo parecido (OSNet >= 0,80)»
recuperó 7 de 16 cortes y unió 1 de 31 parejas distintas. Solo por el cuerpo
no sirve (13 de 16 a cambio de 12 de 31 errores); con huecos largos, tampoco.

En sombra NO cambia nada: apunta en `output/costura_sombra/<cámara>/<fecha>
.jsonl` cada unión que haría, con una foto «antes | después» para que el
usuario la revise. Usa el embedding de cuerpo que el pipeline ya calcula
(`track['_emb_cuerpo']`): no añade llamadas al modelo.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DT_MAX_S = 3.0
DIST_MAX = 1.0           # en altos de persona, entre los pies
SIM_MIN = 0.80
#: Un seguimiento terminado se ofrece como candidato este tiempo (> DT_MAX_S
#: para dar margen a que el nuevo tenga su primer embedding).
VIGENCIA_S = 8.0
#: Sin verlo este tiempo, el seguimiento se da por terminado.
PERDIDO_S = 0.3
#: Tantos seguimientos perdidos en la misma imagen = salto de vídeo. 21-oct:
#: con 2 saltaba de más en el vídeo con más gente (5 de 7 sin otra señal);
#: el salto confirmado por el «cambio de escena» de las 14:28 fue de 3.
SALTO_MIN = 3
#: Votos propios del uniforme que espera una marca de personal heredada
#: (~1 s: se vota cada ~0,5 s).
VOTOS_PERSONALES_MIN = 2
_RAIZ = Path(__file__).resolve().parents[3]


def _pies(box) -> np.ndarray:
    return np.array([(float(box[0]) + float(box[2])) / 2.0, float(box[3])])


def _recorte(frame, box) -> Optional[np.ndarray]:
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()


class CosturaSombra:
    """Una por cámara. `observar` en cada imagen; nunca lanza."""

    def __init__(self, camera_id: Any, carpeta: Optional[Path] = None) -> None:
        from .anaqueles_registro import _carpeta_nombre
        self.camera_id = str(camera_id)
        self._carpeta = Path(carpeta) if carpeta else (
            _RAIZ / 'output' / 'costura_sombra' / _carpeta_nombre(camera_id))
        #: tid vivo -> {'t0', 'box0', 't1', 'box1', 'emb', 'emb0', 'pid', 'rec0', 'rec1', 'personal'}
        self._vivos: Dict[Any, Dict[str, Any]] = {}
        #: tid terminado -> lo mismo (candidatos a «A»)
        self._terminados: Dict[Any, Dict[str, Any]] = {}
        self.uniones = 0
        self.saltos = 0
        #: (tid_a, tid_b) unidos DE VERDAD en la última imagen (modo activo).
        self.unidos: List[Tuple[Any, Any]] = []
        #: En la última imagen: seguimientos cuya marca de personal heredada
        #: se confirmó con sus propios votos, y uniones deshechas por ellos.
        #: Parejas ya unidas (en cualquier orden): ver `_sin_rebote`.
        self._parejas: set = set()
        self.personal_confirmado: List[Any] = []
        self.deshechos: List[Any] = []
        self._vistos: Dict[Any, Dict[str, Any]] = {}

    def observar(self, frame, tracks: Dict[Any, Dict[str, Any]],
                 t: Optional[float] = None,
                 frame_counter: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Devuelve los (tid_a, tid_b) unidos de verdad en esta imagen."""
        self.unidos, self.personal_confirmado, self.deshechos = [], [], []
        try:
            t = time.time() if t is None else float(t)
            # Solo quien se ve en ESTA imagen: los perdidos siguen un rato en
            # el diccionario con su última caja.
            vistos = {k: v for k, v in (tracks or {}).items()
                      if frame_counter is None
                      or v.get('last_seen', frame_counter) == frame_counter}
            self._vistos = vistos
            self._observar(frame, vistos, t)
            self._resolver_personal(vistos)
        except Exception as exc:                            # noqa: BLE001
            logger.debug('costura en sombra: %s', exc)
        return self.unidos

    def _observar(self, frame, tracks, t) -> None:
        # Primero se dan por terminados los que ya no se ven: el nuevo de
        # ESTA imagen tiene que poder emparejarse con ellos.
        nuevos_fin = [k for k, v in self._vivos.items()
                      if k not in tracks and t - v['t1'] > PERDIDO_S]
        for tid in nuevos_fin:
            v = self._terminados[tid] = self._vivos.pop(tid)
            # Huella del ÚLTIMO instante en que se le vio (26-oct-2026): es
            # la que más se parece a cómo vuelve. Ver `_huella_inmediata`.
            e = _calcular_huella(v.pop('rec_ult', None))
            if e is not None:
                v['emb'] = e
        # SALTO de vídeo (21-oct-2026): el usuario cierra, reabre o mueve el
        # vídeo, y todos desaparecen en la MISMA imagen. En ~23.000 imágenes
        # de 5 grabaciones normales, dos a la vez pasó 4 veces y tres nunca.
        # Al otro lado del salto hay otras personas en el mismo sitio: nada
        # de lo anterior se ofrece para unir.
        ultimas = collections.Counter(self._terminados[k]['t1'] for k in nuevos_fin)
        if _detectar_saltos() and any(n >= SALTO_MIN for n in ultimas.values()):
            n = len(self._terminados)
            self._terminados.clear()
            self.saltos += 1
            logger.info('COSTURA [%s] sombra: posible salto de vídeo (%d seguimientos '
                        'perdidos a la vez); no se une nada a través de él',
                        self.camera_id, n)
        for tid in [k for k, v in self._terminados.items() if t - v['t1'] > VIGENCIA_S]:
            self._terminados.pop(tid, None)
        for tid, tr in tracks.items():
            box = tr.get('box')
            if box is None:
                continue
            v = self._vivos.get(tid)
            if v is None and tid in self._terminados:      # volvió con su número
                v = self._vivos[tid] = self._terminados.pop(tid)
            if v is None:
                v = self._vivos[tid] = {'t0': t, 'box0': list(box), 'emb0': None,
                                        'rec0': _recorte(frame, box), 'emparejado': False}
                # Huella en la PRIMERA imagen (26-oct-2026): sin esperar a la
                # cadencia del pipeline (cada 12 imágenes, tras 4 visto).
                v['emb0'] = _calcular_huella(v['rec0'])
            if _huella_inmediata():
                v['rec_ult'] = _recorte(frame, box)
            v.update(t1=t, box1=list(box), pid=tr.get('persistent_id'),
                     confirmado=bool(tr.get('persistent_confirmed')),
                     personal=bool(tr.get('es_personal')),
                     rol=tr.get('personal_rol'), nombre=tr.get('personal_nombre'))
            emb = tr.get('_emb_cuerpo')
            if emb is not None:
                v['emb'] = emb
                if v['emb0'] is None:
                    v['emb0'] = emb
            if t - v.get('t_rec1', -1e9) >= 0.5:                # último recorte, cada 0,5 s
                v['rec1'], v['t_rec1'] = _recorte(frame, box), t
            # Se reintenta un rato: el anterior puede tardar en darse por
            # terminado (PERDIDO_S) o el nuevo en tener su cuerpo calculado.
            if (not v['emparejado'] and v['emb0'] is not None
                    and t - v['t0'] <= VIGENCIA_S):
                v['emparejado'] = self._buscar_pareja(tid, v)

    def _buscar_pareja(self, tid_b, b) -> bool:
        mejor = None
        sin_rebote = _sin_rebote()
        for tid_a, a in self._terminados.items():
            if tid_a == tid_b or a.get('emb') is None:
                continue
            # SIN REBOTE (26-oct-2026): nunca la unión contraria de una ya
            # hecha (A->B y luego B->A). En vivo, una madre y su hija pegadas
            # se pasaron la identidad de una a otra y de vuelta (13:25). Se
            # probó también no unir con hueco 0 s (dos números en la misma
            # imagen) y se QUITÓ: en la grabación «empleados» las 2 uniones que
            # quitaba eran la misma persona con un recuadro duplicado.
            if sin_rebote and frozenset((tid_a, tid_b)) in self._parejas:
                continue
            dt = b['t0'] - a['t1']
            alto = max(1.0, float(a['box1'][3]) - float(a['box1'][1]))
            dist = float(np.linalg.norm(_pies(a['box1']) - _pies(b['box0']))) / alto
            sim = float(np.dot(a['emb'], b['emb0']))
            if 0.0 <= dt <= DT_MAX_S and dist <= DIST_MAX and sim >= SIM_MIN:
                if mejor is None or sim > mejor[3]:
                    mejor = (tid_a, dt, dist, sim)
        if mejor is None:
            return False
        tid_a, dt, dist, sim = mejor
        a = self._terminados.pop(tid_a)
        self._parejas.add(frozenset((tid_a, tid_b)))
        if len(self._parejas) > 5000:                    # no crecer sin fin
            self._parejas.clear()
        self.uniones += 1
        hecho = unir() and self._heredar(tid_a, a, tid_b)
        # HACIA ATRÁS, A LA SOMBRA (27-oct-2026): si no se pudo heredar porque
        # el nuevo ya tiene identidad por su cara y el de antes era
        # provisional, el tramo de antes podría tomar la del nuevo. Revisado
        # por el usuario en el vídeo del 26-oct: 7 de 8 eran la misma persona
        # (sin eso, ese tramo cuenta como otra persona). Solo se apunta.
        atras = (not hecho and str(a.get('pid') or 'prov:').startswith('prov:')
                 and bool(b.get('pid')) and not str(b.get('pid')).startswith('prov:'))
        self._apuntar(tid_a, a, tid_b, b, dt, dist, sim, hecho, atras)
        return True

    def _heredar(self, tid_a, a, tid_b) -> bool:
        """MODO ACTIVO (22-oct-2026, tras 58 de 61 uniones bien revisadas por
        el usuario; clientes 41 de 44): el seguimiento nuevo hereda la
        identidad del anterior. La cara manda: si el nuevo ya tiene una
        identidad confirmada por su propia cara, no se toca."""
        tr = self._vistos.get(tid_b)
        pid = a.get('pid')
        if tr is None or not pid or (tr.get('persistent_confirmed')
                                     and tr.get('identity_source') == 'face'):
            return False
        tr['_heredado'] = (pid, bool(a.get('confirmado')))
        tr['persistent_id'] = pid
        tr['persistent_confirmed'] = bool(a.get('confirmado'))
        if a.get('personal') and not tr.get('es_personal'):
            # La marca de personal NO se pasa en el acto (23-oct-2026): espera
            # a los primeros votos PROPIOS del uniforme (`_resolver_personal`).
            # Medido en vivo el 23-oct a las 08:56:13: un cliente heredó la
            # marca de un empleado, sus 2 votos decían «sin uniforme» (0,0) y
            # su ficha se apartó como de personal.
            tr['_personal_heredado'] = {'rol': a.get('rol'), 'nombre': a.get('nombre'),
                                        'pid_propio': f'prov:{tid_b}'}
        self.unidos.append((tid_a, tid_b))
        return True

    def _resolver_personal(self, tracks) -> None:
        """Marca de personal heredada y pendiente: con VOTOS_PERSONALES_MIN
        votos propios, se confirma si la mayoría dice «uniforme»; si no, se
        deshace la unión (la identidad heredada también: era otra persona)."""
        from . import clasificador_uniforme
        for tid, tr in tracks.items():
            pend = tr.get('_personal_heredado')
            if not pend:
                continue
            if tr.get('es_personal'):              # ya lo marcó su propio uniforme
                tr.pop('_personal_heredado', None)
                continue
            p = clasificador_uniforme.votos(self.camera_id, tid)
            if len(p) < VOTOS_PERSONALES_MIN:
                continue
            tr.pop('_personal_heredado', None)
            if float(np.mean(np.asarray(p) >= 0.5)) >= 0.5:
                tr['es_personal'] = True
                tr['personal_rol'] = pend.get('rol')
                tr['personal_nombre'] = pend.get('nombre')
                tr['personal_via'] = 'costura'
                self.personal_confirmado.append(tid)
            else:
                tr.pop('_heredado', None)
                tr['persistent_id'] = pend.get('pid_propio') or f'prov:{tid}'
                tr['persistent_confirmed'] = False
                logger.info('COSTURA [%s] deshecha la unión de %s: heredaba la marca de '
                            'personal y sus votos dicen que no lleva uniforme (%s)',
                            self.camera_id, tid, [round(x, 2) for x in p])
                self.deshechos.append(tid)

    def _apuntar(self, tid_a, a, tid_b, b, dt, dist, sim, hecho: bool = False,
                 atras: bool = False) -> None:
        try:
            dia = time.strftime('%Y-%m-%d', time.localtime(b['t0']))
            hora = time.strftime('%H%M%S', time.localtime(b['t0']))
            self._carpeta.mkdir(parents=True, exist_ok=True)
            foto = ''
            if a.get('rec1') is not None and b.get('rec0') is not None:
                ims = [cv2.resize(r, (max(1, int(r.shape[1] * 300 / r.shape[0])), 300))
                       for r in (a['rec1'], b['rec0'])]
                hoja = np.hstack([ims[0], np.full((300, 12, 3), 255, np.uint8), ims[1]])
                ruta = self._carpeta / f'{dia}_{hora}_{tid_a}_{tid_b}.jpg'
                cv2.imwrite(str(ruta), hoja, [cv2.IMWRITE_JPEG_QUALITY, 85])
                foto = ruta.name
            with open(self._carpeta / f'{dia}.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'hora': time.strftime('%H:%M:%S', time.localtime(b['t0'])),
                    'a': str(tid_a), 'pid_a': a.get('pid'), 'b': str(tid_b), 'pid_b': b.get('pid'),
                    'dt': round(dt, 2), 'dist': round(dist, 2), 'sim': round(sim, 3),
                    'personal': bool(a.get('personal') or b.get('personal')), 'foto': foto,
                    'unido': bool(hecho),
                    'hacia_atras': bool(atras),
                }, ensure_ascii=False) + '\n')
            logger.info('COSTURA [%s] %s %s -> %s (%.1f s, %.2f altos, cuerpo %.2f)',
                        self.camera_id, 'UNIDO' if hecho else ('sombra: hacia atrás' if atras else 'sombra: uniría'),
                        tid_a, tid_b, dt, dist, sim)
        except Exception as exc:                            # noqa: BLE001
            logger.warning('costura en sombra: no se pudo apuntar: %s', exc)


_POR_CAMARA: Dict[str, CosturaSombra] = {}

#: Función recorte BGR -> huella OSNet (la pone `tiendas.py` con su
#: reidentificador corporal). Sin ella no hay huella inmediata.
_huella_fn: Any = None
#: Recortes más bajos no dan huella fiable (el agachado del 26-oct, solo la
#: cabeza y la espalda, medía 50-100 px).
HUELLA_ALTO_MIN = 50


def poner_huella(fn: Any) -> None:
    global _huella_fn
    _huella_fn = fn


def _huella_inmediata() -> bool:
    """Huella al APARECER y al PERDERSE cada número (26-oct-2026).

    Por qué: el pipeline calcula la huella del cuerpo cada 12 imágenes y tras
    4 visto (`BODY_REID_EVERY_N`, `BODY_REID_MIN_SEEN_FRAMES`: ahorro de
    procesador). Quien aparece y desaparece en ~1 s no llega a tenerla y la
    costura no lo puede unir: un empleado agachado en la balda de abajo tuvo
    5 números en 25 s (grabación del 26-oct). Probado fuera de línea en 3
    grabaciones (~17 min): 9 -> 13 uniones; las 8 nuevas, revisadas en foto
    por el usuario, eran la misma persona. Coste: ~5-10 ms por número que
    aparece o se pierde. `ELDE_COSTURA_HUELLA_INMEDIATA=0` la apaga."""
    v = (os.getenv('ELDE_COSTURA_HUELLA_INMEDIATA') or '').strip()
    if v in ('0', '1'):
        return v == '1'
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'COSTURA_HUELLA_INMEDIATA', False))
    except Exception:                                       # noqa: BLE001
        return False


def _sin_rebote() -> bool:
    """Sin uniones de ida y vuelta (26-oct-2026).
    `ELDE_COSTURA_SIN_REBOTE` (1/0) manda sobre `COSTURA_SIN_REBOTE`."""
    v = (os.getenv('ELDE_COSTURA_SIN_REBOTE') or '').strip()
    if v in ('0', '1'):
        return v == '1'
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'COSTURA_SIN_REBOTE', False))
    except Exception:                                       # noqa: BLE001
        return False


def _calcular_huella(recorte: Any) -> Any:
    """Huella normalizada del recorte, o None. Nunca lanza."""
    if (not _huella_inmediata() or _huella_fn is None or recorte is None
            or recorte.shape[0] < HUELLA_ALTO_MIN):
        return None
    try:
        e = _huella_fn(recorte)
        if e is None:
            return None
        e = np.asarray(e, dtype=np.float32)
        n = float(np.linalg.norm(e))
        return e / n if n > 0 else None
    except Exception as exc:                                # noqa: BLE001
        logger.debug('costura: sin huella inmediata (%s)', exc)
        return None


def _detectar_saltos() -> bool:
    """`COSTURA_DETECTAR_SALTOS` (apagada desde el 21-oct a petición del usuario)."""
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'COSTURA_DETECTAR_SALTOS', True))
    except Exception:                                       # noqa: BLE001
        return True


def unir() -> bool:
    """¿Modo ACTIVO? `ELDE_COSTURA_UNIR` (1/0) manda sobre `COSTURA_UNIR`.
    Apagado = solo apunta (sombra)."""
    import os
    v = (os.getenv('ELDE_COSTURA_UNIR') or '').strip()
    if v in ('0', '1'):
        return v == '1'
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'COSTURA_UNIR', False))
    except Exception:                                       # noqa: BLE001
        return False


def activa() -> bool:
    """`ELDE_COSTURA_SOMBRA` (1/0) manda sobre `COSTURA_SOMBRA_ACTIVA`."""
    import os
    v = (os.getenv('ELDE_COSTURA_SOMBRA') or '').strip()
    if v in ('0', '1'):
        return v == '1'
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'COSTURA_SOMBRA_ACTIVA', False))
    except Exception:                                       # noqa: BLE001
        return False


def resueltos(camera_id: Any) -> Tuple[List[Any], List[Any]]:
    """De la última imagen de esa cámara: (seguimientos cuya marca de
    personal heredada se confirmó, uniones deshechas por sus votos)."""
    c = _POR_CAMARA.get(str(camera_id))
    return (list(c.personal_confirmado), list(c.deshechos)) if c else ([], [])


def observar(camera_id: Any, frame, tracks, frame_counter: Optional[int] = None,
             t: Optional[float] = None) -> List[Tuple[Any, Any]]:
    if not activa():
        return []
    clave = str(camera_id)
    if clave not in _POR_CAMARA:
        _POR_CAMARA[clave] = CosturaSombra(camera_id)
    return _POR_CAMARA[clave].observar(frame, tracks, t, frame_counter)
