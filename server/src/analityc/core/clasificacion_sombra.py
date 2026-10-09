"""src/analityc/core/clasificacion_sombra.py — clasificación «en sombra».

## Para qué (8-oct-2026)

En el banco aislado (grabación del 7-oct, 201 personas etiquetadas a mano)
una combinación de cambios clasificó el doble de personas con el mismo
acierto (~90 %), decidiendo en 2,3 s en vez de 7 y con un 40 % menos de
cálculo. Antes de cambiar nada, se prueba EN VIVO a la sombra: la forma
actual sigue mandando (fichas, dashboard, informes) y esta, en paralelo y
sobre las mismas personas, solo apunta lo que habría decidido.

## La combinación («B sin herencia»)

  * mira a cada persona cada `CADA_S` (0,6 s) en vez de cada ~1,3 s;
  * busca la cara UNA vez por intento (hoy hasta 3 recortes);
  * vota solo MiVOLO (cara sin «estirar» + cuerpo); no InsightFace;
  * acepta más proporciones de cuerpo (alto/ancho 1,0-5,0; hoy 1,2-4,0);
  * decide sumando la evidencia de cada lectura en vez de exigir 5;
  * no hereda el género del reconocimiento de caras (5 de los 7 errores de
    la forma actual eran mujeres que heredaban «Hombre» así);
  * el jurado dura mientras la persona no lleve `FIN_S` (9 s) sin verse.

## Aislamiento

Clasificador propio, SIN reidentificador (no registra caras ni toca la
galería de identidades) y sin la telemetría de siempre. Las sesiones de los
modelos (YuNet, MiVOLO) son las mismas. Nunca lanza hacia fuera.

## Cómo se enciende

`ELDE_CLASIFICACION_SOMBRA=<ruta absoluta .jsonl>` al arrancar el servidor.
Una línea por persona al decidir y otra al cerrarse su número. Cada 60 s,
`SOMBRA [cam] ...` en el log con intentos, decididas y coste.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

VARIABLE = 'ELDE_CLASIFICACION_SOMBRA'
CADA_S = 0.6
FIN_S = 9.0

_fichero = None
_candado = threading.Lock()


def _escribir(fila: Dict[str, Any]) -> None:
    try:
        with _candado:
            _fichero.write(json.dumps(fila, ensure_ascii=False) + '\n')
            _fichero.flush()
    except Exception as exc:                                # noqa: BLE001
        logger.debug("sombra: no se pudo escribir: %s", exc)


def _clase_sombra():
    """La clase se crea al instalar: importa el clasificador del servidor."""
    from .analytics import demographics as dm
    from .analytics import estimador_edad_genero as eg

    class _EspiaYuNet:
        """YuNet sin cambiar su resultado: recuerda la última imagen y sus
        caras para recortar la cara «natural» sin buscarla otra vez."""

        def __init__(self, real):
            self.real = real
            self.is_gpu = getattr(real, 'is_gpu', False)
            self.ultimo = None

        def setInputSize(self, size):  # noqa: N802
            return self.real.setInputSize(size)

        def detect(self, img):
            r = self.real.detect(img)
            self.ultimo = (img, r[1])
            return r

    class _Acumulador(dm._TrackAccumulator):
        """Decide cuando la evidencia sumada (log-odds recortados, por su
        peso y con descuento por parecido entre lecturas seguidas) pasa el
        umbral, con al menos 2 lecturas y sin desacuerdo fuerte."""
        umbral = math.log(0.95 / 0.05)
        descuento = 0.5
        tope = 3.0
        min_lecturas = 2
        max_desacuerdo = 0.25
        __slots__ = ()

        def try_commit(self, min_samples, min_conf, min_margin):
            if self.committed:
                return self.result
            if len(self.gender_hist) < self.min_lecturas:
                return None
            s, votos = 0.0, []
            for g, w in zip(self.gender_hist, self.weight_hist):
                ph, pm = max(float(g[0]), 1e-6), max(float(g[1]), 1e-6)
                s += self.descuento * float(w) * max(-self.tope, min(self.tope, math.log(ph / pm)))
                votos.append(0 if ph >= pm else 1)
            if abs(s) < self.umbral:
                return None
            idx = 0 if s > 0 else 1
            if sum(1 for x in votos if x != idx) / len(votos) > self.max_desacuerdo:
                return None
            a = self._robust_age()
            edad = float(np.sum(a * dm._AGE_BIN_CENTERS))
            if len(self.age_value_hist) >= 4 and float(np.std(self.age_value_hist)) > 18.0:
                rango = 'Desconocido'
            elif self.is_body_only:
                rango = self._cfg.coarse_age_from_value(edad)
            else:
                rango = self._cfg.age_range_from_value(int(round(edad)))
            genero = dm._GENDER_LIST[idx]
            conf = 1.0 / (1.0 + math.exp(-abs(s)))
            self.committed = True
            self.committed_gender_idx = idx
            self.result = {'gender': genero, 'age_range': rango,
                           'category': 'Hombres' if genero == 'Hombre' else 'Mujeres',
                           'confidence': round(conf, 3), 'samples': self.samples,
                           'bracket': 'evidencia', 'age_value': round(edad, 1)}
            return self.result

    class _SinTelemetria:
        def registrar_track(self, *a, **k):
            return None

        def registrar_excepcion(self, *a, **k):
            return None

    class ClasificadorSombra(dm.DemographicsClassifier):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._telemetria = _SinTelemetria()
            if self._yunet is not None:
                self._yunet = _EspiaYuNet(self._yunet)

        def classify(self, frame, bbox, track_id):
            self._ctx_frame, self._ctx_bbox, self._ctx_busquedas = frame, bbox, 0
            if isinstance(self._yunet, _EspiaYuNet):
                self._yunet.ultimo = None
            if track_id not in self._accum:
                self._accum[track_id] = _Acumulador(self._cfg)
            return super().classify(frame, bbox, track_id)

        # Una búsqueda de cara por intento (el primer recorte).
        def _detect_face_with_pose(self, person_crop):
            self._ctx_busquedas += 1
            if self._ctx_busquedas > 1:
                return None, None, 0.0
            return super()._detect_face_with_pose(person_crop)

        def _cara_natural(self, score_min):
            esp = self._yunet
            if not isinstance(esp, _EspiaYuNet) or esp.ultimo is None:
                return None
            img, faces = esp.ultimo
            if faces is None or len(faces) == 0:
                return None
            f = max(faces, key=lambda f: float(f[-1]))
            if float(f[-1]) < score_min:
                return None
            h, w = img.shape[:2]
            x, y, fw, fh = [float(v) for v in f[:4]]
            m = 0.15 * max(fw, fh)
            x1, y1 = int(max(0, x - m)), int(max(0, y - m))
            x2, y2 = int(min(w, x + fw + m)), int(min(h, y + fh + m))
            if x2 - x1 < 8 or y2 - y1 < 8:
                return None
            return img[y1:y2, x1:x2].copy()

        # Solo MiVOLO vota, con la cara natural + el cuerpo.
        def _infer_ensemble(self, face, track_id=None):
            x1, y1, x2, y2 = [int(v) for v in self._ctx_bbox]
            hh, ww = self._ctx_frame.shape[:2]
            cuerpo = self._ctx_frame[max(0, y1):min(hh, y2), max(0, x1):min(ww, x2)]
            cara = self._cara_natural(self._cfg.DEMO_YUNET_MIN_SCORE)
            m = self._estimador_cuerpo.estimar(-1, cuerpo, cara if cara is not None else face)
            if m is None or not m.es_valida():
                return None, None, 0.0
            g = np.zeros(2)
            i = 0 if m.genero == 'Hombre' else 1
            g[i], g[1 - i] = m.conf_genero, 1.0 - m.conf_genero
            a = np.exp(-((dm._AGE_BIN_CENTERS - m.edad_anios) ** 2) / 32.0)
            a = a / a.sum() if a.sum() > 0 else np.full(8, 1 / 8)
            return g, a, float(m.edad_anios)

        def _try_body_sample(self, frame, bbox, acc, track_id, cara=None):
            nat = self._cara_natural(0.60)
            if nat is not None:
                cara = nat
            return super()._try_body_sample(frame, bbox, acc, track_id, cara=cara)

        # Más proporciones de cuerpo.
        def _infer_body_only(self, person_crop, cara=None):
            viejo = (eg._RATIO_MIN, eg._RATIO_MAX)
            try:
                eg._RATIO_MIN, eg._RATIO_MAX = 1.0, 5.0
                return super()._infer_body_only(person_crop, cara=cara)
            finally:
                eg._RATIO_MIN, eg._RATIO_MAX = viejo

    return ClasificadorSombra


class _Estado:
    """Lo de UNA cámara: su clasificador sombra y las personas a la vista."""

    def __init__(self, clase, base: Any) -> None:
        cam = getattr(base, '_camara_id', None)
        self.cam = cam
        self.clf = clase(insightface_session=getattr(base, '_insight', None),
                         yunet=getattr(base, '_yunet', None),
                         reidentifier=None, camera_id=cam)
        self.pistas: Dict[int, Dict[str, Any]] = {}
        self.minuto = {'t': time.time(), 'intentos': 0, 'ms': 0.0, 'decididas': 0}


def instalar() -> bool:
    """Engancha la sombra si `ELDE_CLASIFICACION_SOMBRA` está puesta.
    Nunca lanza; True si quedó instalada."""
    global _fichero
    ruta = os.environ.get(VARIABLE, '').strip()
    if not ruta or _fichero is not None:
        return False
    try:
        _fichero = open(ruta, 'a', encoding='utf-8')
        clase = _clase_sombra()
        raiz = __name__.split('.analityc')[0]
        import importlib
        tiendas = importlib.import_module(raiz + '.analityc.core.tiendas')
    except Exception as exc:                                # noqa: BLE001
        logger.error("sombra: no se pudo instalar: %s", exc)
        return False

    estados: Dict[int, _Estado] = {}
    original = tiendas.TiendasProcess._update_tracks_botsort

    def sombra(self, detections) -> None:
        base = getattr(self, '_demographics', None)
        frame = getattr(self, 'last_processed_frame', None)
        if base is None or frame is None:
            return
        est = estados.get(id(base))
        if est is None:
            est = estados[id(base)] = _Estado(clase, base)
        ahora = time.time()
        for d in detections or ():
            tid = d.get('track_id')
            if tid is None or d.get('class', -1) != -1 or d.get('box') is None:
                continue
            p = est.pistas.get(tid)
            if p is None:
                p = est.pistas[tid] = {'t0': ahora, 'ult': ahora, 'prox': ahora,
                                       'decidida': False, 'intentos': 0}
            p['ult'] = ahora
            if p['decidida'] or ahora < p['prox']:
                continue
            p['prox'] = ahora + CADA_S
            c0 = time.perf_counter()
            est.clf.classify(frame, tuple(d['box']), tid)
            ms = (time.perf_counter() - c0) * 1000.0
            p['intentos'] += 1
            est.minuto['intentos'] += 1
            est.minuto['ms'] += ms
            acc = est.clf._accum.get(tid)
            if acc is not None and acc.committed and acc.result:
                p['decidida'] = True
                est.minuto['decididas'] += 1
                r = acc.result
                _escribir({'t': round(ahora, 3), 'cam': est.cam, 'tid': int(tid),
                           'evento': 'decide', 'genero': r.get('gender'),
                           'edad': r.get('age_range'), 'conf': r.get('confidence'),
                           's_decidir': round(ahora - p['t0'], 2), 'intentos': p['intentos'],
                           'cara': acc.n_face_samples, 'cuerpo': acc.n_body_samples})
        for tid in [t for t, p in est.pistas.items() if ahora - p['ult'] > FIN_S]:
            p = est.pistas.pop(tid)
            acc = est.clf._accum.get(tid)
            _escribir({'t': round(ahora, 3), 'cam': est.cam, 'tid': int(tid), 'evento': 'cierra',
                       'decidida': p['decidida'], 'intentos': p['intentos'],
                       'muestras': getattr(acc, 'samples', 0) if acc is not None else 0,
                       'visto_s': round(p['ult'] - p['t0'], 2)})
            try:
                est.clf.remove_track(tid)
            except Exception:                               # noqa: BLE001
                pass
        if ahora - est.minuto['t'] >= 60.0:
            m = est.minuto
            logger.info("SOMBRA [%s] intentos=%d decididas=%d coste=%.0f ms en %.0f s (%.1f ms/intento)",
                        est.cam, m['intentos'], m['decididas'], m['ms'], ahora - m['t'],
                        m['ms'] / max(1, m['intentos']))
            est.minuto = {'t': ahora, 'intentos': 0, 'ms': 0.0, 'decididas': 0}

    def envoltura(self, detections, *args, **kwargs):
        resultado = original(self, detections, *args, **kwargs)
        try:
            sombra(self, detections)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("sombra: fallo (sigue sin sombra en este frame): %s", exc)
        return resultado

    tiendas.TiendasProcess._update_tracks_botsort = envoltura
    logger.warning("sombra: CLASIFICACION EN SOMBRA activa (B sin herencia), escribe en %s", ruta)
    return True
