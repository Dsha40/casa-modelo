"""Revisar el género tras un CRUCE, EN SOMBRA (22-oct-2026).

Por qué: el género se guarda por NÚMERO de seguimiento y, una vez decidido,
no se vuelve a mirar (`clasificador_b.py`, «sin histéresis»). Si ByteTrack
pasa el número a otra persona, la etiqueta vieja se queda pegada. Visto en
vivo el 22-oct (cámara 3): el número 300100066 se decidió «Mujer» a las
12:37:08 con una clienta (su ficha) y luego llevó esa etiqueta 1 min 40 s
encima de un hombre calvo. El número cambia de persona en el 11-15 % de los
etiquetados (CLAUDE.md, 15.6), sobre todo cuando dos personas se cruzan.

Idea del usuario: cuando dos cajas se solapan, volver a revisar. Cómo:
  - se apunta el cruce de quien ya tiene género decidido;
  - cuando las cajas llevan `SEPARADO_S` sin solaparse (durante el cruce el
    recorte mezcla a las dos personas), se toman lecturas NUEVAS con el mismo
    modelo que vota en B (MiVOLO, cuerpo + cara si se encuentra), aparte del
    acumulador del seguimiento, y se deciden con la misma regla de evidencia
    (`_AcumuladorEvidencia`);
  - resultado: «confirma», «cambiaria» o «sin_decidir».

En sombra NO cambia nada (ni la etiqueta, ni la identidad, ni la galería de
caras, ni los contadores): apunta en `output/revision_cruces/<cámara>/<fecha>
.jsonl` y guarda una foto «ficha | ahora» para que el usuario la revise. La
identidad queda fuera a propósito (decisión del usuario, 22-oct). Coste: como
mucho UNA lectura de MiVOLO (+ YuNet en la cabeza) por imagen y cámara, solo
tras un cruce.

Diferencias con una lectura del clasificador (aproximación, no idéntica): la
cara se busca una vez en la ventana de la cabeza sin reescalar, y una lectura
con cara pesa 1,0 en vez del peso por calidad de B.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Solape mínimo para que cuente como CRUCE: intersección / área de la caja MENOR.
SOLAPE_MIN = 0.15
#: Tiempo sin que NINGUNA otra caja toque la suya antes de mirar (22-oct-2026,
#: tras el único error revisado por el usuario: a las 14:43:43 el vecino
#: seguía pegado, con el hombro dentro del recuadro, y se leyó su cara).
SEPARADO_S = 1.0
#: Lecturas válidas mínimas para decir «cambiaria» (confirmar sigue con la
#: regla de B). El error de las 14:43 se decidió con 2; los 5 aciertos
#: tenían 2-4.
MIN_LECTURAS_CAMBIO = 3
LECTURA_CADA_S = 0.5
MAX_LECTURAS = 10
#: Desde la primera lectura; pasado esto sin decidir, «sin_decidir».
VENTANA_S = 8.0
#: Sin verlo este tiempo, se abandona la revisión («perdido», sin foto).
PERDIDO_S = 2.0
_RAIZ = Path(__file__).resolve().parents[3]


def _solape(a, b) -> float:
    ix = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    iy = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    if ix <= 0 or iy <= 0:
        return 0.0
    area = lambda c: max(1.0, (float(c[2]) - float(c[0])) * (float(c[3]) - float(c[1])))
    return ix * iy / min(area(a), area(b))


def _recorte(frame, box) -> Optional[np.ndarray]:
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _cara(clf, frame, box) -> Optional[np.ndarray]:
    """La cara de ESA persona (regla de «cara propia»: centro en su cabeza)
    dentro de la ventana que usa B; None si no hay."""
    yn = getattr(clf, '_yunet', None)
    yn = getattr(yn, 'real', yn)
    if yn is None:
        return None
    hh, ww = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    pw, ph = x2 - x1, max(1, y2 - y1)
    mx = int(pw * 0.25)
    t_x1, t_y1 = max(0, x1 - mx), max(0, y1 - int(ph * 0.20))
    t_x2, t_y2 = min(ww, x2 + mx), min(hh, y1 + int(ph * 0.55))
    if t_x2 - t_x1 < 40 or t_y2 - t_y1 < 40:
        return None
    ventana = frame[t_y1:t_y2, t_x1:t_x2]
    if not getattr(yn, 'is_gpu', False):
        yn.setInputSize((ventana.shape[1], ventana.shape[0]))
    _, caras = yn.detect(ventana)
    if caras is None or len(caras) == 0:
        return None
    propias = [f for f in caras
               if x1 <= t_x1 + float(f[0]) + float(f[2]) / 2 <= x2
               and y1 - 0.10 * ph <= t_y1 + float(f[1]) + float(f[3]) / 2 <= y1 + 0.40 * ph
               and float(f[-1]) >= 0.60]
    if not propias:
        return None
    f = max(propias, key=lambda f: float(f[-1]))
    x, y, fw, fh = [float(v) for v in f[:4]]
    m = 0.15 * max(fw, fh)
    cx1, cy1 = int(max(0, x - m)), int(max(0, y - m))
    cx2 = int(min(ventana.shape[1], x + fw + m))
    cy2 = int(min(ventana.shape[0], y + fh + m))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return ventana[cy1:cy2, cx1:cx2].copy()


class RevisionCruces:
    """Una por cámara. `observar` en cada imagen; nunca lanza."""

    def __init__(self, camera_id: Any, carpeta: Optional[Path] = None) -> None:
        from .anaqueles_registro import _carpeta_nombre
        self.camera_id = str(camera_id)
        self._carpeta = Path(carpeta) if carpeta else (
            _RAIZ / 'output' / 'revision_cruces' / _carpeta_nombre(camera_id))
        #: tid -> {'t_cruce', 'con', 'etiqueta', 'acc', 'lecturas', 't0', 't_lect', 'visto', 'recorte'}
        self._pend: Dict[Any, Dict[str, Any]] = {}
        #: tid -> (t, recorte) de la última vez que NADIE tocaba su caja, para
        #: la foto «antes» (23-oct-2026: 2 de 11 casos no tenían ficha propia
        #: —número sin ficha o heredado por la costura— y no había con qué
        #: comparar). Solo de quien ya tiene género decidido.
        self._limpio: Dict[Any, Tuple[float, np.ndarray]] = {}
        self.cruces = 0
        self.resultados: Dict[str, int] = {}

    def observar(self, frame, tracks, clf, capture_meta=None,
                 t: Optional[float] = None, frame_counter: Optional[int] = None) -> None:
        try:
            t = time.time() if t is None else float(t)
            vistos = {k: v for k, v in (tracks or {}).items()
                      if v.get('box') is not None and not v.get('es_personal')
                      and (frame_counter is None
                           or v.get('last_seen', frame_counter) == frame_counter)}
            self._observar(frame, vistos, clf, capture_meta or {}, t)
        except Exception as exc:                            # noqa: BLE001
            logger.debug('revisión de cruces: %s', exc)

    def _observar(self, frame, tracks, clf, capture_meta, t) -> None:
        accs = getattr(clf, '_accum', {}) or {}
        ids = list(tracks)
        cruzando, tocando = {}, set()
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                s = _solape(tracks[a]['box'], tracks[b]['box'])
                if s > 0:
                    tocando.update((a, b))
                if s >= SOLAPE_MIN:
                    cruzando.setdefault(a, b)
                    cruzando.setdefault(b, a)
        for tid in ids:
            acc = accs.get(tid)
            if (tid not in tocando and tid not in self._pend and acc is not None
                    and getattr(acc, 'committed', False)
                    and t - self._limpio.get(tid, (-1e9, None))[0] >= LECTURA_CADA_S):
                rec = _recorte(frame, tracks[tid]['box'])
                if rec is not None:
                    self._limpio[tid] = (t, rec.copy())
        for tid in [k for k, v in self._limpio.items() if k not in tracks and t - v[0] > PERDIDO_S]:
            self._limpio.pop(tid, None)
        for tid, otro in cruzando.items():
            acc = accs.get(tid)
            if acc is None or not getattr(acc, 'committed', False) or not acc.result:
                continue
            p = self._pend.get(tid)
            if p is None:
                self.cruces += 1
            # Un cruce nuevo (o el mismo que sigue) reinicia la revisión; la
            # foto «antes» es la de antes del PRIMER roce.
            antes = p.get('antes') if p else self._limpio.get(tid, (None, None))[1]
            self._pend[tid] = {'t_cruce': t, 'con': otro, 'visto': t, 't_toque': t,
                               'etiqueta': acc.result.get('gender'), 'antes': antes,
                               'acc': None, 'lecturas': [], 't0': None, 't_lect': -1e9}
        for tid in list(self._pend):
            p = self._pend[tid]
            if tid in tracks:
                p['visto'] = t
                if tid in tocando:
                    p['t_toque'] = t       # alguien aún roza su caja: no mirar
            elif t - p['visto'] > PERDIDO_S:
                self._terminar(tid, p, 'perdido', None, capture_meta, frame=None)
        # Como mucho una lectura por imagen: la revisión que más espera.
        listos = [k for k, p in self._pend.items()
                  if k in tracks and k not in tocando
                  and t - p['t_toque'] >= SEPARADO_S
                  and t - p['t_lect'] >= LECTURA_CADA_S]
        if not listos:
            return
        tid = min(listos, key=lambda k: self._pend[k]['t_lect'])
        p = self._pend[tid]
        if p['acc'] is None:
            from .analytics.clasificador_b import _AcumuladorEvidencia
            p['acc'], p['t0'] = _AcumuladorEvidencia(getattr(clf, '_cfg', None)), t
        p['t_lect'] = t
        self._leer(clf, frame, tracks[tid]['box'], p)
        dec = p['acc'].try_commit(0, 0.0, 0.0)
        if (dec is not None and dec.get('gender') != p['etiqueta']
                and len(p['acc'].gender_hist) < MIN_LECTURAS_CAMBIO):
            # Para cambiar hace falta más prueba: se sigue mirando.
            p['acc'].committed, p['acc'].result, dec = False, None, None
        if dec is not None:
            res = 'confirma' if dec.get('gender') == p['etiqueta'] else 'cambiaria'
            self._terminar(tid, p, res, dec, capture_meta, frame, tracks[tid])
        elif len(p['lecturas']) >= MAX_LECTURAS or t - p['t0'] >= VENTANA_S:
            self._terminar(tid, p, 'sin_decidir', None, capture_meta, frame, tracks[tid])

    def _leer(self, clf, frame, box, p) -> None:
        est = getattr(clf, '_estimador_cuerpo', None)
        cuerpo = _recorte(frame, box)
        if est is None or cuerpo is None:
            return
        cara = None
        try:
            cara = _cara(clf, frame, box)
        except Exception as exc:                            # noqa: BLE001
            logger.debug('revisión de cruces: cara: %s', exc)
        m = est.estimar(-1, cuerpo, cara)
        if m is None or not m.es_valida():
            p['lecturas'].append('-')
            return
        cfg = getattr(clf, '_cfg', None)
        if cara is None and m.conf_genero < float(getattr(cfg, 'DEMO_BODY_MIN_GENDER_CONF', 0.0)):
            p['lecturas'].append('-')
            return
        g = np.zeros(2)
        i = 0 if m.genero == 'Hombre' else 1
        g[i], g[1 - i] = m.conf_genero, 1.0 - m.conf_genero
        a = np.full(8, 1.0 / 8.0)
        peso = 1.0 if cara is not None else float(getattr(cfg, 'DEMO_BODY_SAMPLE_WEIGHT', 0.5))
        p['acc'].add(g, a, float(m.edad_anios), face_width=0.0, weight=peso,
                     decay=float(getattr(cfg, 'DEMO_DECAY', 1.0)),
                     source='face' if cara is not None else 'body')
        p['lecturas'].append(f"{'H' if i == 0 else 'M'}{m.conf_genero:.2f}{'c' if cara is not None else ''}")

    def _terminar(self, tid, p, res, dec, capture_meta, frame=None, track=None) -> None:
        self._pend.pop(tid, None)
        self.resultados[res] = self.resultados.get(res, 0) + 1
        if res == 'perdido' and not p['lecturas']:
            return                       # nunca se llegó a mirar: nada que apuntar
        try:
            ahora = time.time()
            dia = time.strftime('%Y-%m-%d', time.localtime(ahora))
            hora = time.strftime('%H%M%S', time.localtime(ahora))
            self._carpeta.mkdir(parents=True, exist_ok=True)
            ficha = capture_meta.get(tid)
            foto = ''
            rec = _recorte(frame, track.get('box')) if (frame is not None and track) else None
            if rec is not None:
                # Hoja «ficha | antes del cruce | ahora», cada una rotulada.
                ims = []
                if ficha:
                    from .analytics.config import AnalyticsConfig
                    fj = os.path.join(AnalyticsConfig.CAPTURE_DIR, 'persons', f'{ficha}.jpg')
                    im = cv2.imread(fj) if os.path.isfile(fj) else None
                    if im is not None:
                        ims.append(('ficha', im))
                if p.get('antes') is not None:
                    ims.append(('antes', p['antes']))
                ims.append(('ahora', rec))
                partes = []
                for nombre, r in ims:
                    r = cv2.resize(r, (max(60, int(r.shape[1] * 300 / r.shape[0])), 300))
                    cv2.rectangle(r, (0, 0), (62, 20), (0, 0, 0), -1)
                    cv2.putText(r, nombre, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    partes += [r, np.full((300, 12, 3), 255, np.uint8)]
                hoja = np.hstack(partes[:-1])
                ruta = self._carpeta / f'{dia}_{hora}_{tid}_{res}.jpg'
                cv2.imwrite(str(ruta), hoja, [cv2.IMWRITE_JPEG_QUALITY, 85])
                foto = ruta.name
            with open(self._carpeta / f'{dia}.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'hora': time.strftime('%H:%M:%S', time.localtime(ahora)),
                    'cruce': time.strftime('%H:%M:%S', time.localtime(p['t_cruce'])),
                    'tid': str(tid), 'con': str(p['con']),
                    'pid': (track or {}).get('persistent_id'),
                    'etiqueta': p['etiqueta'], 'resultado': res,
                    'nueva': (dec or {}).get('gender'),
                    'evidencia': round(float(p['acc'].evidencia()), 2) if p['acc'] else None,
                    'lecturas': p['lecturas'], 'ficha': ficha, 'foto': foto,
                    'con_antes': p.get('antes') is not None,
                }, ensure_ascii=False) + '\n')
            if res == 'cambiaria':
                logger.info('CRUCES [%s] sombra: %s es «%s» y cambiaría a «%s» (cruce con %s)',
                            self.camera_id, tid, p['etiqueta'], (dec or {}).get('gender'), p['con'])
        except Exception as exc:                            # noqa: BLE001
            logger.warning('revisión de cruces: no se pudo apuntar: %s', exc)


_POR_CAMARA: Dict[str, RevisionCruces] = {}


def activa() -> bool:
    """`ELDE_REVISION_CRUCES` (1/0) manda sobre `REVISION_CRUCES_SOMBRA`."""
    v = (os.getenv('ELDE_REVISION_CRUCES') or '').strip()
    if v in ('0', '1'):
        return v == '1'
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'REVISION_CRUCES_SOMBRA', False))
    except Exception:                                       # noqa: BLE001
        return False


def observar(camera_id: Any, frame, tracks, clf, capture_meta=None,
             frame_counter: Optional[int] = None, t: Optional[float] = None) -> None:
    if not activa() or clf is None:
        return
    clave = str(camera_id)
    if clave not in _POR_CAMARA:
        _POR_CAMARA[clave] = RevisionCruces(camera_id)
    _POR_CAMARA[clave].observar(frame, tracks, clf, capture_meta, t, frame_counter)
