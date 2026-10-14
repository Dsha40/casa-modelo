"""src/analityc/core/analytics/clasificador_b.py — clasificación «B sin herencia».

## Por qué es la forma oficial desde el 9-oct-2026

Medido en la cámara 3 contra etiquetas puestas a mano por el usuario:

| | Forma clásica | B sin herencia |
|---|---:|---:|
| Banco, grabación del 7-oct (201 personas) | 41 % decididas, 88,5 % acierto | 80 %, 90,6 % |
| EN VIVO, 8-oct 18:00-20:30 (251 personas) | 27 % decididas, 87 % acierto | 63 %, 94,9 % |
| Mujeres bien clasificadas en vivo (de 111) | 15 | 62 |

8 de los 9 errores en vivo de la forma clásica eran géneros HEREDADOS del
reconocimiento de caras (una mujer confundida con un hombre visto antes).
Detalle en `informes/informe_2026-10-08_dia.md` (local) y en CLAUDE.md.

## Qué cambia frente a `DemographicsClassifier`

  * busca la cara UNA vez por intento (antes, hasta 3 recortes);
  * vota solo MiVOLO v2 con la cara SIN alinear (+ el cuerpo); no InsightFace;
  * acepta más proporciones de cuerpo (alto/ancho 1,0-5,0; antes 1,2-4,0);
  * decide sumando la evidencia de cada lectura en vez de exigir 5 lecturas
    con 90 % de acuerdo (`_AcumuladorEvidencia`);
  * NO hereda el género del reconocimiento de caras (sigue reconociendo y
    registrando caras: la identidad no cambia);
  * NO vuelve a mirar a quien ya decidió (la «histéresis»): así se validó;
  * 12-oct-2026: decide con el 90 % de seguridad si alguna lectura tuvo
    CARA y con el 95 % si solo hubo cuerpo (`_AcumuladorEvidencia.umbral`).
    En vivo (8-oct) con cara acertaba el 97 % y solo con cuerpo el 93 %
    (mujeres de cuerpo, 88 %). En el banco, junto con conservar el género
    9 s (`demografia_aplazada.py`): 69 % decididas frente a 59 %, acierto
    93,0 % frente a 93,3 % (dentro del margen) y un 24 % menos de lecturas;
  * la cadencia (mirar cada ~0,6 s) la pone `tiendas.py` con
    `DEMO_RECLASSIFY_EVERY_B`.

## Interruptor

`ELDE_CLASIFICADOR=clasico` (o `AnalyticsConfig.DEMO_CLASIFICADOR`) vuelve
a la forma anterior. Por defecto, `b`.
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np

from . import demographics as dm
from . import estimador_edad_genero as eg


def usar_b() -> bool:
    """True salvo que se pida la forma clásica."""
    valor = os.environ.get('ELDE_CLASIFICADOR', '').strip().lower()
    if not valor:
        valor = str(getattr(dm.AnalyticsConfig, 'DEMO_CLASIFICADOR', 'b')).lower()
    return valor not in ('clasico', 'clásico', 'classic', 'a', '0', 'no')


def _cara_propia() -> bool:
    """«Cara propia» (14-oct-2026): solo vale la cara que cae donde va la
    cabeza de ESA persona. Encendida por defecto (AnalyticsConfig.
    DEMO_B_CARA_PROPIA); apagar con ELDE_CARA_PROPIA=0."""
    v = os.environ.get('ELDE_CARA_PROPIA', '').strip().lower()
    if v:
        return v in ('1', 'true', 'si', 'sí', 'on')
    return bool(getattr(dm.AnalyticsConfig, 'DEMO_B_CARA_PROPIA', False))


class _EspiaYuNet:
    """YuNet sin cambiar su resultado: recuerda la última imagen y sus caras
    para recortar la cara «natural» sin buscarla otra vez.

    Con `region` puesta («cara propia», 14-oct-2026) se quitan las caras
    cuyo centro no cae en la cabeza de la persona: dentro de su caja a lo
    ancho y entre un 10 % por encima de su borde de arriba y el 40 % de su
    altura. La ventana de búsqueda es la caja ±25 % de ancho y, de las caras
    que hay, se elegía la más nítida aunque fuera la del VECINO: en la
    grabación del 8-oct, dos números a la vez con parecido de cara 0,96-0,99
    (la misma cara). En el banco con etiquetas a mano (13-oct): identidades
    que juntaban a un hombre y una mujer, de 10 a 5."""

    region = None   # (x1, y1, ancho, alto de la ventana, caja de la persona)

    def __init__(self, real: Any) -> None:
        self.real = real
        self.is_gpu = getattr(real, 'is_gpu', False)
        self.ultimo = None

    def setInputSize(self, size):  # noqa: N802 (API de cv2)
        return self.real.setInputSize(size)

    def detect(self, img):
        r = self.real.detect(img)
        if self.region is not None and r[1] is not None and len(r[1]):
            x1c, y1c, wc, hc, (bx1, by1, bx2, by2) = self.region
            sx, sy = wc / max(1, img.shape[1]), hc / max(1, img.shape[0])
            ph = max(1.0, by2 - by1)
            propias = [f for f in r[1]
                       if bx1 <= x1c + (float(f[0]) + float(f[2]) / 2) * sx <= bx2
                       and by1 - 0.10 * ph <= y1c + (float(f[1]) + float(f[3]) / 2) * sy
                       <= by1 + 0.40 * ph]
            r = (r[0], np.array(propias) if propias else None)
        self.ultimo = (img, r[1])
        return r


class _ReidSinDemografia:
    """El reidentificador de siempre (registra y reconoce caras), pero sin
    prestar el género guardado de otra persona."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, nombre: str) -> Any:
        return getattr(self._real, nombre)

    def get_demographics_for_person(self, track_id: int) -> None:
        return None


def _log_odds(seguridad: float) -> float:
    return math.log(seguridad / (1.0 - seguridad))


class _AcumuladorEvidencia(dm._TrackAccumulator):
    """Decide cuando la evidencia sumada pasa el umbral: log-odds de cada
    lectura recortados a ±`tope` (una lectura no vale más que p≈0,95), por
    su peso, con un `descuento` porque lecturas seguidas se parecen; al
    menos 2 lecturas y sin más de un 25 % de votos en contra. El umbral
    depende de si hubo cara (ver la cabecera del módulo)."""
    descuento = 0.5
    tope = 3.0
    min_lecturas = 2
    max_desacuerdo = 0.25
    __slots__ = ()

    @property
    def umbral(self) -> float:
        cfg = dm.AnalyticsConfig
        if self.n_face_samples > 0:
            return _log_odds(float(getattr(cfg, 'DEMO_B_SEGURIDAD_CON_CARA', 0.90)))
        return _log_odds(float(getattr(cfg, 'DEMO_B_SEGURIDAD', 0.95)))

    def evidencia(self) -> float:
        s = 0.0
        for g, w in zip(self.gender_hist, self.weight_hist):
            ph, pm = max(float(g[0]), 1e-6), max(float(g[1]), 1e-6)
            s += self.descuento * float(w) * max(-self.tope, min(self.tope, math.log(ph / pm)))
        return s

    def try_commit(self, min_samples, min_conf, min_margin):
        if self.committed:
            return self.result
        if len(self.gender_hist) < self.min_lecturas:
            return None
        s = self.evidencia()
        if abs(s) < self.umbral:
            return None
        idx = 0 if s > 0 else 1
        votos = [0 if float(g[0]) >= float(g[1]) else 1 for g in self.gender_hist]
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
        categoria = 'Hombres' if genero == 'Hombre' else 'Mujeres'
        if rango in ('0-12', 'Nino'):
            categoria = 'Ninos'
        conf = 1.0 / (1.0 + math.exp(-abs(s)))
        self.committed = True
        self.committed_gender_idx = idx
        self.result = {'gender': genero, 'age_range': rango, 'category': categoria,
                       'confidence': round(conf, 3), 'gender_confidence': round(conf, 3),
                       'age_value': round(edad, 1), 'samples': self.samples,
                       'bracket': 'evidencia',
                       'age_source': 'body' if self.is_body_only else 'face',
                       'n_face_samples': self.n_face_samples,
                       'n_body_samples': self.n_body_samples}
        return self.result


class ClasificadorB(dm.DemographicsClassifier):
    """La forma oficial desde el 9-oct (ver la cabecera del módulo)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        if self._yunet is not None and not isinstance(self._yunet, _EspiaYuNet):
            self._yunet = _EspiaYuNet(self._yunet)
        if self._reidentifier is not None:
            self._reidentifier = _ReidSinDemografia(self._reidentifier)
        self._ctx_frame = None
        self._ctx_bbox = None
        self._ctx_busquedas = 0

    def classify(self, frame, bbox, track_id):
        acc = self._accum.get(track_id)
        if acc is not None and acc.committed and acc.result is not None:
            return acc.result                      # sin histéresis
        self._ctx_frame, self._ctx_bbox, self._ctx_busquedas = frame, bbox, 0
        if isinstance(self._yunet, _EspiaYuNet):
            self._yunet.ultimo = None
        if acc is None:
            self._accum[track_id] = _AcumuladorEvidencia(self._cfg)
        return super().classify(frame, bbox, track_id)

    def _commit_from_reid(self, acc, reid_demo, track_id):
        return None                                # sin herencia

    def _detect_face_with_pose(self, person_crop):
        self._ctx_busquedas += 1
        if self._ctx_busquedas > 1:
            return None, None, 0.0
        esp = self._yunet
        if (_cara_propia() and isinstance(esp, _EspiaYuNet)
                and self._ctx_frame is not None and self._ctx_bbox is not None):
            # La ventana que monta `_sample_once` para su primer recorte:
            # caja ±25 % de ancho, de -20 % a +55 % de alto. Si el recorte no
            # es esa ventana, no se filtra.
            hh, ww = self._ctx_frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in self._ctx_bbox]
            pw, ph = x2 - x1, max(1, y2 - y1)
            mx = int(pw * 0.25)
            t_x1, t_y1 = max(0, x1 - mx), max(0, y1 - int(ph * 0.20))
            t_x2, t_y2 = min(ww, x2 + mx), min(hh, y1 + int(ph * 0.55))
            if (t_x2 - t_x1, t_y2 - t_y1) == (person_crop.shape[1], person_crop.shape[0]):
                esp.region = (t_x1, t_y1, t_x2 - t_x1, t_y2 - t_y1, (x1, y1, x2, y2))
        try:
            return super()._detect_face_with_pose(person_crop)
        finally:
            if isinstance(esp, _EspiaYuNet):
                esp.region = None

    def _cara_natural(self, score_min: float) -> Optional[np.ndarray]:
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

    def _infer_ensemble(self, face, track_id=None):
        if self._ctx_frame is None or self._estimador_cuerpo is None:
            return None, None, 0.0
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

    def _infer_body_only(self, person_crop, cara=None):
        viejo = (eg._RATIO_MIN, eg._RATIO_MAX)
        try:
            eg._RATIO_MIN, eg._RATIO_MAX = 1.0, 5.0
            return super()._infer_body_only(person_crop, cara=cara)
        finally:
            eg._RATIO_MIN, eg._RATIO_MAX = viejo


def crear(**kwargs: Any) -> dm.DemographicsClassifier:
    """El clasificador oficial (B) o el clásico si así se pide."""
    clase = ClasificadorB if usar_b() else dm.DemographicsClassifier
    return clase(**kwargs)
