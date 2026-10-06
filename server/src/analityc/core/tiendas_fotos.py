"""src/analityc/core/tiendas_fotos.py — la foto de cada persona, y su ficha.

Extraído de `tiendas.py` (HITO 5) con el cuerpo verbatim. Es la parte más
delicada del pipeline fuera del tracking, y la que más reglas acumula:

  - **una foto por PERSONA, no por track.** Doble deduplicación: espacial por
    celda de la imagen (atrapa a la persona quieta a la que el tracker le
    cambia el id) y por identidad Re-ID (si el rostro dice que ya está
    fotografiada, la nueva se borra).
  - **el sidecar se rellena en dos tiempos.** Al capturar no se sabe el género
    ni la edad: convergen frames después. Por eso se escribe primero la ficha
    en blanco y se re-escribe cuando la demografía se decide, momento en el
    que además se RE-anota la copia del cliente y se emite el evento del
    visitante (la única alerta que produce este pipeline).
  - **nunca un null mudo.** Una captura sin demografía guarda POR QUÉ no la
    tiene: «no se pudo» y «no se intentó» son cosas distintas, y confundirlas
    fue lo que hizo invisible un fallo entero en su día.
  - **el personal enrolado no es un visitante**: ni evento, ni alerta.

Entra como mixin de `TiendasProcess` para no cambiar ni una llamada.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from . import tienda_almacen as _almacen_tienda
from .tienda_clasificaciones import EsperaDemografia
from .analytics.config import AnalyticsConfig

logger = logging.getLogger(__name__)


class FotosMixin:
    """Capturas por persona, sus sidecars y el evento del visitante."""

    def _capture_established_tracks(self, frame):
        """Captura UNA foto por persona. Doble dedup, robusto al churn de track:
        (a) ESPACIAL por celda de grilla -> a lo sumo 1 foto por zona de la
        imagen cada CAPTURE_DEDUP_SECONDS (asi una persona quieta a la que el
        tracker le cambia el id no se vuelve a fotografiar); y (b) por IDENTIDAD
        Re-ID despues (agrupa la misma cara aunque cambie de zona)."""
        if frame is None or getattr(frame, 'size', 0) == 0:
            return
        kmin = int(AnalyticsConfig.CAPTURE_MIN_SEEN_FRAMES)
        # En tiempo real, no en imagenes (5-oct-2026, `TiendasProcess._imagenes`).
        _imagenes = getattr(self, '_imagenes', None)
        if _imagenes is not None:
            kmin = _imagenes(kmin)
        ddt = float(AnalyticsConfig.CAPTURE_DEDUP_SECONDS)
        now = time.time()
        h, w = frame.shape[:2]
        cell_w = max(1.0, float(w) / float(AnalyticsConfig.CAPTURE_GRID_COLS))
        cell_h = max(1.0, float(h) / float(AnalyticsConfig.CAPTURE_GRID_ROWS))
        for tid, obj in list(self.active_tracks.items()):
            if tid in self._captured_ids or obj.get('class', -1) != -1:
                continue
            if int(obj.get('seen_frames', 0)) < kmin:
                continue
            box = obj.get('box')
            if box is None:
                continue
            # ── Dedup por IDENTIDAD *ANTES* de fotografiar ──
            # Si el track ya tiene persistent_id CONFIRMADO (rostro o cuerpo)
            # y esa persona YA fue fotografiada, NO se escribe un duplicado
            # que luego habria que borrar (y que quedaba huerfano si la
            # demografia no convergia). Una persona = una foto por sesion.
            pid = obj.get('persistent_id')
            confirmado = bool(
                obj.get('persistent_confirmed')
                and pid and not str(pid).startswith('prov:'))
            if confirmado and pid in self._captured_persons:
                self._captured_ids.add(tid)   # persona ya fotografiada
                continue
            # El PERSONAL enrolado no se fotografia como visitante: sus
            # fotos ya estan en la galeria de personal, y una captura suya
            # contaminaria la demografia y la galeria del cliente.
            if obj.get('es_personal') or (
                    confirmado
                    and self._ficha_personal_tienda(pid) is not None):
                self._captured_ids.add(tid)
                continue
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            cell = (int(cx / cell_w), int(cy / cell_h))
            if now - self._cell_last_capture.get(cell, 0.0) <= ddt:
                self._captured_ids.add(tid)  # zona ya cubierta: no reintentar
                continue
            stem = self._capture_detection(tid, frame, box)
            if stem and confirmado:
                # Foto escrita con identidad conocida: sellar a la persona
                # YA (no esperar a que la demografia converja).
                self._captured_persons[pid] = stem
            self._cell_last_capture[cell] = now
        # acotar el dict para que no crezca sin limite
        if len(self._cell_last_capture) > 5000:
            self._cell_last_capture = {
                k: v for k, v in self._cell_last_capture.items()
                if now - v <= ddt}

    def _capture_detection(self, track_id, frame, box):
        """Guarda UNA foto del recorte de la persona detectada (por track nuevo)
        y, si se le detecta cara, tambien el recorte del rostro. En CAPTURE_DIR
        (output/captures/persons y .../faces). Devuelve el `stem` del archivo
        escrito o None. Defensivo: nunca lanza."""
        if track_id in self._captured_ids:
            return None
        self._captured_ids.add(track_id)
        if len(self._captured_ids) > 20000:  # acotar memoria en sesiones largas
            self._captured_ids = set(list(self._captured_ids)[-10000:])
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 10 or (y2 - y1) < int(
                    AnalyticsConfig.CAPTURE_MIN_PERSON_H):
                return
            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                return
            base = AnalyticsConfig.CAPTURE_DIR
            ts = time.strftime("%Y%m%d_%H%M%S")
            cam = getattr(self, '_camera_display_name', None) or "cam"
            safe_cam = "".join(
                c for c in str(cam) if c.isalnum() or c in "_-")[:24] or "cam"
            stem = f"{ts}_{safe_cam}_t{track_id}"
            # Foto de la persona
            pdir = os.path.join(base, "persons")
            os.makedirs(pdir, exist_ok=True)
            cv2.imwrite(os.path.join(pdir, f"{stem}.jpg"), person,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Sidecar con metadata (genero/edad se rellenan al converger).
            self._capture_meta[track_id] = stem
            # `camera` es el nombre VISIBLE y no identifica nada: cuatro
            # ventanas pueden llamarse «iVMS-4200» (H-39), y entonces todas
            # las capturas de la tienda caian en un unico cubo. `device_id`
            # es la llave real, la misma del registro de dispositivos, los
            # heatmaps y el ranking de pasillos: con ella SI se puede cruzar
            # la demografia con el pasillo. `camera` se conserva tal cual
            # porque la galeria y el buscador VLM lo leen.
            meta0 = {"track_id": int(track_id), "camera": str(cam),
                     "device_id": str(
                         getattr(self, '_camera_id_actual', '') or ''),
                     "timestamp": ts, "gender": None, "age_range": None}
            try:
                import json as _json
                with open(os.path.join(pdir, f"{stem}.json"), "w",
                          encoding="utf-8") as _f:
                    _json.dump(meta0, _f, ensure_ascii=False)
            except Exception:
                pass
            # Foto del ROSTRO con ZOOM (recorte natural ampliado): es la
            # que deja ver la cara en la que se baso el genero/edad.
            face_zoom = None
            if hasattr(self, '_demographics') and self._demographics is not None:
                try:
                    face_zoom = self._demographics.extract_face_zoom(person)
                    if face_zoom is None:
                        # Sin rostro detectable: guardar al menos el crop
                        # alineado si la demografia logro uno.
                        face_zoom, _pose, _nw = \
                            self._demographics._detect_face_with_pose(person)
                    if face_zoom is not None and getattr(face_zoom, 'size', 0):
                        fdir = os.path.join(base, "faces")
                        os.makedirs(fdir, exist_ok=True)
                        cv2.imwrite(os.path.join(fdir, f"{stem}.jpg"),
                                    face_zoom,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                except Exception:
                    face_zoom = None
            # Copia ANOTADA inmediata para la galeria del cliente, con el
            # rostro ampliado al lado (se re-anota con genero/edad cuando
            # la demografia converge).
            self._write_client_capture(stem, meta0, person_img=person,
                                       face_zoom=face_zoom)
            return stem
        except Exception as exc:  # noqa: BLE001
            logger.debug("Captura de deteccion fallo: %s", exc)
        return None

    def _delete_capture(self, stem):
        """Borra los archivos de una captura (persona + rostro + sidecar),
        incluida la copia anotada de la carpeta del cliente."""
        base = AnalyticsConfig.CAPTURE_DIR
        for sub, ext in (("persons", ".jpg"), ("persons", ".json"),
                         ("faces", ".jpg")):
            try:
                p = os.path.join(base, sub, f"{stem}{ext}")
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        cdir = getattr(AnalyticsConfig, 'CAPTURE_CLIENT_DIR', '') or ''
        if cdir:
            for ext in (".jpg", ".json"):
                try:
                    p = os.path.join(cdir, f"{stem}{ext}")
                    if os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass

    # ── Capturas ANOTADAS para la galeria del cliente (Amazonas View) ──

    @staticmethod
    def _capture_ts_legible(ts: str) -> str:
        """'20260727_154210' -> '27/07/2026 15:42:10' (defensivo)."""
        try:
            dt = datetime.datetime.strptime(str(ts), "%Y%m%d_%H%M%S")
            return dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            return str(ts)

    @staticmethod
    def _capture_ts_epoch(ts) -> float:
        """'20260817_144210' (sello del sidecar) -> epoch.

        La alerta de visitante viaja con epoch como todas las demas: el sello
        compacto rompia la linea 🕒 del WhatsApp (float('20260817_144210')
        parsea por PEP 515 a un epoch absurdo y el respaldo lo devolvia
        crudo) y el pie de la tarjeta mostraba '7_144210' (17-ago-2026)."""
        try:
            return time.mktime(
                time.strptime(str(ts), "%Y%m%d_%H%M%S"))
        except Exception:
            return time.time()

    @staticmethod
    def _compose_face_zoom_panel(person_img, face_zoom):
        """Pega el ROSTRO AMPLIADO a la derecha del recorte de la persona.

        El rostro se escala a la altura del cuerpo (con tope) y se separa
        con una franja, de modo que la foto guardada muestre a la vez a
        quien se detecto y el primer plano de su cara.
        """
        ph, pw = person_img.shape[:2]
        fh, fw = face_zoom.shape[:2]
        if fh <= 0 or fw <= 0:
            return person_img
        # El rostro ocupa la altura del cuerpo, sin pasarse de ancho.
        escala = ph / float(fh)
        max_face_w = max(80, int(pw * 1.10))
        if fw * escala > max_face_w:
            escala = max_face_w / float(fw)
        nw = max(1, int(round(fw * escala)))
        nh = max(1, int(round(fh * escala)))
        interp = cv2.INTER_CUBIC if escala > 1.0 else cv2.INTER_AREA
        rostro = cv2.resize(face_zoom, (nw, nh), interpolation=interp)

        sep = 4
        alto = max(ph, nh)
        lienzo = np.full((alto, pw + sep + nw, 3), 18, dtype=np.uint8)
        lienzo[:ph, :pw] = person_img
        y0 = (alto - nh) // 2          # rostro centrado verticalmente
        lienzo[y0:y0 + nh, pw + sep:pw + sep + nw] = rostro
        # Marco del panel de zoom para que se lea como un primer plano.
        cv2.rectangle(lienzo, (pw + sep, y0),
                      (pw + sep + nw - 1, y0 + nh - 1), (90, 90, 90), 1)
        return lienzo

    def _annotate_capture_image(self, person_img, meta: dict,
                                face_zoom=None):
        """Copia del recorte de la persona con un banner inferior legible:
        linea 1 = genero + rango de edad (o 'Analizando...'), linea 2 =
        camara + fecha/hora (+ visitas si el Re-ID las conoce).

        Si se pasa `face_zoom`, se compone a la derecha un panel con el
        ROSTRO AMPLIADO: es la cara concreta sobre la que se decidio el
        genero/edad, para poder verificarlo de un vistazo.
        """
        img = person_img
        h, w = img.shape[:2]

        # ── Panel de ZOOM: el rostro ampliado, a la derecha del cuerpo ──
        if face_zoom is not None and getattr(face_zoom, 'size', 0) > 0:
            img = self._compose_face_zoom_panel(img, face_zoom)
            h, w = img.shape[:2]

        # Ancho minimo para que el texto quepa sin encimarse.
        min_w = 240
        if w < min_w:
            scale = min_w / float(w)
            img = cv2.resize(img, (min_w, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        gender = meta.get('gender')
        age = meta.get('age_range')
        if gender in (None, '', 'Desconocido'):
            linea1 = ("Desconocido" if meta.get('demo_final')
                      else "Analizando...")
            color = (160, 160, 160)                  # gris
        else:
            linea1 = f"{gender} | {age or 'Desconocido'}"
            color = ((200, 120, 40) if gender == 'Hombre'
                     else (180, 60, 200))            # azul / magenta (BGR)
        partes2 = [str(meta.get('camera') or 'cam'),
                   self._capture_ts_legible(meta.get('timestamp', ''))]
        visitas = meta.get('visitas')
        if visitas and int(visitas) > 1:
            partes2.append(f"{int(visitas)}a visita")
        linea2 = "  |  ".join(p for p in partes2 if p)
        footer = 52
        canvas = np.zeros((h + footer, w, 3), dtype=np.uint8)
        canvas[:h] = img
        canvas[h:] = (18, 18, 18)
        cv2.rectangle(canvas, (0, h), (w, h + 3), color, -1)

        def _fit(text, base_scale, max_w):
            # Reduce la escala hasta que el texto quepa (no se recorta).
            scale = base_scale
            while scale > 0.28:
                tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                     scale, 1)[0][0]
                if tw <= max_w:
                    break
                scale -= 0.04
            return scale
        s1 = _fit(linea1, 0.60, w - 16)
        s2 = _fit(linea2, 0.42, w - 16)
        cv2.putText(canvas, linea1, (8, h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, s1, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, linea2, (8, h + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, s2, (210, 210, 210), 1,
                    cv2.LINE_AA)
        return canvas

    def _write_client_capture(self, stem: str, meta: dict, person_img=None,
                              face_zoom=None):
        """Escribe/actualiza la foto ANOTADA (+ sidecar JSON) de una captura
        en la carpeta del cliente (CAPTURE_CLIENT_DIR). Se llama al capturar
        (etiqueta 'Analizando...') y de nuevo cuando la demografia converge
        (etiqueta final con genero/edad; MISMO archivo -> sin duplicados).
        Defensivo: nunca lanza."""
        cdir = getattr(AnalyticsConfig, 'CAPTURE_CLIENT_DIR', '') or ''
        if not cdir or not stem:
            return
        try:
            if person_img is None:
                src = os.path.join(AnalyticsConfig.CAPTURE_DIR, "persons",
                                   f"{stem}.jpg")
                person_img = cv2.imread(src)
            if person_img is None or getattr(person_img, 'size', 0) == 0:
                return
            if face_zoom is None:
                # Re-anotacion: recuperar el rostro ampliado ya guardado.
                fsrc = os.path.join(AnalyticsConfig.CAPTURE_DIR, "faces",
                                    f"{stem}.jpg")
                if os.path.isfile(fsrc):
                    face_zoom = cv2.imread(fsrc)
            os.makedirs(cdir, exist_ok=True)
            annotated = self._annotate_capture_image(person_img, meta,
                                                     face_zoom=face_zoom)
            # Escritura atomica: el watcher del cliente no debe leer a medias.
            dst = os.path.join(cdir, f"{stem}.jpg")
            tmp = os.path.join(cdir, f"{stem}.tmp.jpg")
            if cv2.imwrite(tmp, annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                os.replace(tmp, dst)
            import json as _json
            jtmp = os.path.join(cdir, f"{stem}.json.tmp")
            with open(jtmp, "w", encoding="utf-8") as f:
                _json.dump({**meta, "file": f"{stem}.jpg"}, f,
                           ensure_ascii=False)
            os.replace(jtmp, os.path.join(cdir, f"{stem}.json"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Captura cliente fallo (%s): %s", stem, exc)

    def _sellar_captura_sin_demografia(self, track_id) -> None:
        """Escribe `motivo_sin_demografia` en la captura de un track que
        muere sin genero. Sin esto el JSON se queda con el `null` inicial
        para siempre y no hay forma de saber que paso."""
        stem = self._capture_meta.get(track_id)
        if not stem or track_id in self._captured_demo_done:
            return
        jp = os.path.join(AnalyticsConfig.CAPTURE_DIR, "persons",
                          f"{stem}.json")
        if not os.path.isfile(jp):
            return
        import json as _json
        try:
            meta = _json.loads(open(jp, encoding='utf-8').read())
        except Exception:  # noqa: BLE001
            return
        if meta.get('gender'):
            return                      # ya tenia demografia: nada que sellar
        meta['motivo_sin_demografia'] = self._motivo_sin_demografia(track_id)
        meta['demo_final'] = True
        try:
            with open(jp, "w", encoding="utf-8") as fichero:
                _json.dump(meta, fichero, ensure_ascii=False)
            self._write_client_capture(stem, meta)
            self._captured_demo_done.add(track_id)
        except Exception:  # noqa: BLE001
            pass

    def _motivo_sin_demografia(self, track_id) -> str:
        """Por que este track no tiene genero, en una palabra.

        Reusa el diagnostico que ya calcula la telemetria (Hito 2) en vez
        de duplicar la logica: alli estan los contadores por track
        (intentos sin rostro, rechazos de pose/calidad, tamano visto).
        """
        try:
            demo = getattr(self, '_demographics', None)
            if demo is None:
                return "sin_clasificador"
            acc = demo._accum.get(track_id)
            motivo, _detalles = demo._veredicto_final(track_id, acc)
            return str(motivo)
        except Exception:  # noqa: BLE001
            return "desconocido"

    def _update_captures_demographics(self):
        """Para cada captura cuyo track ya convergio en demografia:
        (1) DEDUP: si el Re-ID la identifica como una persona YA capturada,
            BORRA esta foto (es la misma persona en otro track de BoT-SORT);
        (2) si es nueva, escribe genero/edad/identidad en el sidecar JSON para
            el dashboard. Barato: solo toca tracks aun no procesados."""
        self._resolver_esperas_demografia()
        if not self._capture_meta or not hasattr(self, '_demographics'):
            return
        base = AnalyticsConfig.CAPTURE_DIR
        import json as _json
        for tid in list(self._capture_meta.keys()):
            if tid in self._captured_demo_done:
                continue
            try:
                demo = self._demographics.get_cached(tid)
            except Exception:
                demo = None
            if not demo:
                continue
            g = demo.get('gender')
            puid = self._identidad_de_track(tid, demo)
            # Esperar a que haya algo firme (genero o identidad Re-ID).
            if g in (None, 'Desconocido') and not puid:
                continue
            stem = self._capture_meta[tid]
            # ── DEDUP por identidad Re-ID: una sola foto por persona ──
            # Solo es duplicado si el uuid ya tiene OTRA foto (dueno con
            # stem distinto); la foto propia jamas se borra a si misma.
            if puid:
                dueno = self._captured_persons.get(puid)
                if dueno is not None and dueno != stem:
                    self._delete_capture(stem)            # duplicado -> fuera
                    self._captured_demo_done.add(tid)
                    self._capture_meta.pop(tid, None)
                    continue
                self._captured_persons[puid] = stem
            # ── Escribir metadata en el sidecar ──
            try:
                jp = os.path.join(base, "persons", f"{stem}.json")
                meta = {}
                if os.path.isfile(jp):
                    try:
                        meta = _json.loads(open(jp, encoding='utf-8').read())
                    except Exception:
                        meta = {}
                if g not in (None, 'Desconocido'):
                    meta['gender'] = g
                    meta['age_range'] = demo.get('age_range', 'Desconocido')
                    # ── Campos ADICIONALES (Hito 7) ──
                    # `gender` y `age_range` NO cambian de nombre ni de
                    # tipo: el cliente mapea colores sobre esos literales.
                    # Estos son extra y el panel los ignora sin romperse
                    # (lee con .get()).
                    conf = demo.get('confidence')
                    if conf is not None:
                        meta['conf_genero'] = round(float(conf), 3)
                    meta['solo_cuerpo'] = (
                        demo.get('age_source') == 'body'
                        or int(demo.get('n_face_samples', 0) or 0) == 0)
                else:
                    # Sin demografia: dejar constancia de POR QUE. Un null
                    # mudo no distingue "no se pudo" de "no se intento", y
                    # esa ambiguedad hizo invisible el fallo original.
                    meta.setdefault('motivo_sin_demografia',
                                    self._motivo_sin_demografia(tid))
                if puid:
                    meta['person_uuid'] = puid
                    # Visitas conocidas por el Re-ID (para el banner/galeria).
                    try:
                        rec = next(
                            (p for p in self._reidentifier.get_all_persons()
                             if p.get('uuid') == puid), None)
                        if rec:
                            meta['visitas'] = int(rec.get('visit_count', 1))
                    except Exception:
                        pass
                with open(jp, "w", encoding="utf-8") as _f:
                    _json.dump(meta, _f, ensure_ascii=False)
                # La ficha se cierra con lo PRIMERO que llegue (género o
                # identidad) y ya no se vuelve a mirar: sin esto, lo que
                # llegaba después se perdía (medido el 24-sep, ver
                # tienda_clasificaciones.py). Se deja esperando lo que falte.
                _cam = getattr(self, '_camera_id_actual', '') or ''
                if g not in (None, 'Desconocido') and not puid:
                    self._espera_demografia().esperar_identidad(
                        tid, stem, _cam, g, demo.get('age_range'))
                elif puid and g in (None, 'Desconocido'):
                    self._espera_demografia().esperar_genero(
                        tid, stem, _cam, puid)
                self._captured_demo_done.add(tid)
                # RE-anotar la copia del cliente con la demografia final
                # (sobreescribe el mismo archivo -> sin duplicados).
                meta['demo_final'] = True
                self._write_client_capture(stem, meta)
                # Y avisar: este es EL evento de tienda. Ver _emitir_visitante.
                self._emitir_visitante(stem, meta)
            except Exception:
                pass
        if len(self._capture_meta) > 20000:  # acotar memoria
            self._capture_meta = dict(
                list(self._capture_meta.items())[-10000:])

    # ── Fichas cerradas a medias (24-sep-2026) ─────────────────────────
    def _espera_demografia(self) -> EsperaDemografia:
        # Perezosa y en el propio procesador: así no hay que tocar el
        # __init__ de la clase base, que comparten cuatro modos.
        espera = getattr(self, '_espera_demo', None)
        if espera is None:
            espera = self._espera_demo = EsperaDemografia()
        return espera

    def _identidad_de_track(self, tid, demo: Optional[dict] = None) -> str:
        """La identidad firme de un track, o `''`.

        La demografía no siempre adjunta el uuid: se pregunta también al
        reidentificador y al track vivo. Sin esto, un duplicado sin uuid en
        caché quedaba en disco PARA SIEMPRE. Los `prov:` no cuentan: son
        provisionales y no identifican a nadie.
        """
        puid = str((demo or {}).get('person_uuid') or '')
        if not puid and getattr(self, '_reidentifier', None) is not None:
            try:
                puid = self._reidentifier.get_persistent_id(tid) or ''
            except Exception:
                puid = ''
        if not puid:
            t = (getattr(self, 'active_tracks', None) or {}).get(tid)
            if t is not None and t.get('persistent_confirmed'):
                p = str(t.get('persistent_id') or '')
                if p and not p.startswith('prov:'):
                    puid = p
        return puid

    def _demografia_de_track(self, tid) -> Optional[dict]:
        try:
            return self._demographics.get_cached(tid)
        except Exception:                                   # noqa: BLE001
            return None

    def _completar_sidecar(self, stem: str, cambios: dict,
                           quitar: tuple = ()) -> Optional[dict]:
        """Añade `cambios` a la ficha en disco. `None` si ya no existe."""
        jp = os.path.join(AnalyticsConfig.CAPTURE_DIR, "persons",
                          f"{stem}.json")
        if not os.path.isfile(jp):
            return None
        try:
            meta = json.loads(open(jp, encoding='utf-8').read())
        except Exception:                                   # noqa: BLE001
            return None
        meta.update(cambios)
        for clave in quitar:
            meta.pop(clave, None)
        with open(jp, "w", encoding="utf-8") as fichero:
            json.dump(meta, fichero, ensure_ascii=False)
        return meta

    def _resolver_esperas_demografia(self) -> None:
        """Completa las fichas a las que les llegó lo que les faltaba.

        Nunca lanza: corre dentro del bucle de inferencia.
        """
        espera = getattr(self, '_espera_demo', None)
        if not espera or not hasattr(self, '_demographics'):
            return
        cam = getattr(self, '_camera_id_actual', '') or ''
        try:
            identidades, generos = espera.revisar(
                cam, self._identidad_de_track, self._demografia_de_track)
        except Exception:                                   # noqa: BLE001
            return
        almacen = _almacen_tienda.get_almacen()

        # Tenían género y acaban de recibir identidad: ahora sí puede ir a
        # la fila de la visita, y la clasificación deja de ser anónima.
        for _tid, stem, puid, genero, edad in identidades:
            try:
                if self._ficha_personal_tienda(puid) is not None:
                    # Resultó ser personal enrolado: no es un visitante, así
                    # que no se le escribe demografía. Su clasificación queda
                    # como «sin identidad»; es un caso raro y se prefiere eso
                    # a borrar a ciegas.
                    continue
                almacen.anotar_demografia(cam, puid, genero, edad)
                almacen.identificar_clasificacion(stem, puid)
                self._completar_sidecar(stem, {'person_uuid': puid})
                self._captured_persons.setdefault(puid, stem)
            except Exception:                               # noqa: BLE001
                pass

        # Tenían identidad y acaban de recibir género: la ficha se había
        # cerrado con «muestras_insuficientes» y nunca se avisó. Se completa
        # y se emite el aviso de visitante que no salió en su momento.
        for _tid, stem, puid, demo in generos:
            try:
                cambios = {
                    'gender': demo.get('gender'),
                    'age_range': demo.get('age_range', 'Desconocido'),
                    'person_uuid': puid,
                    'solo_cuerpo': (
                        demo.get('age_source') == 'body'
                        or int(demo.get('n_face_samples', 0) or 0) == 0),
                }
                conf = demo.get('confidence')
                if conf is not None:
                    cambios['conf_genero'] = round(float(conf), 3)
                meta = self._completar_sidecar(
                    stem, cambios, quitar=('motivo_sin_demografia',))
                if meta is None:
                    continue
                meta['demo_final'] = True
                self._write_client_capture(stem, meta)
                self._emitir_visitante(stem, meta)
            except Exception:                               # noqa: BLE001
                pass

    def _emitir_visitante(self, stem: str, meta: dict) -> None:
        """Encola el evento «visitante identificado» de este pipeline.

        ## Por que este momento y no otro

        El pipeline de tienda no producia NINGUNA alerta: su metadata solo
        traia `detections`, `demographics`, `heatmap` y `analytics_report`. Eso
        dejaba dos cosas vacias: el panel lateral del cliente y el envio por
        WhatsApp, que reenvia lo que haya en `metadata['alerts']`.

        Se emite justo cuando **el genero y la edad convergen** para un track,
        que ocurre UNA sola vez por persona. Emitir por deteccion inundaria a
        25 fps; emitir al capturar seria prematuro, porque en ese instante la
        demografia todavia dice «Analizando...».

        Nunca lanza: esto corre dentro del bucle de inferencia.
        """
        try:
            # El PERSONAL enrolado no es un visitante: ni alerta ni novedad.
            if self._ficha_personal_tienda(
                    meta.get('person_uuid')) is not None:
                return
            genero = str(meta.get('gender') or '').strip()
            edad = str(meta.get('age_range') or '').strip()
            if not genero or genero.lower() == 'desconocido':
                return              # sin demografia util no hay nada que contar

            # Memoria con FECHA (HITO 1): la demografia consolidada se guarda
            # en la fila de la visita, atada a la jornada y a la CAMARA. Es lo
            # que hace posible el desglose por pasillo, imposible hasta ahora
            # porque la captura solo llevaba el nombre visible de la ventana.
            _pid_demo = meta.get('person_uuid')
            if _pid_demo:
                _almacen_tienda.get_almacen().anotar_demografia(
                    getattr(self, '_camera_id_actual', '') or '',
                    _pid_demo, genero, edad)
            # Y SIEMPRE, tenga identidad o no (24-sep-2026): `anotar_demografia`
            # solo puede escribir a quien tiene identidad, y ese día 3 de 3
            # clasificados se quedaron fuera. Esta tabla los guarda a todos
            # sin tocar `tienda_visitas` ni el conteo de personas únicas.
            _almacen_tienda.get_almacen().anotar_clasificacion(
                getattr(self, '_camera_id_actual', '') or '', stem,
                genero, edad, person_uuid=_pid_demo,
                solo_cuerpo=bool(meta.get('solo_cuerpo')))

            visitas = meta.get('visitas')
            recurrente = isinstance(visitas, int) and visitas > 1
            desc = f"Visitante {genero}"
            if edad:
                desc += f", {edad} años"
            desc += (f". Visita nº {visitas} (recurrente)." if recurrente
                     else ". Primera visita registrada.")

            alerta = {
                'event_type': 'Visitante',
                'class_name': f"{genero} {edad}".strip(),
                'description': desc,
                'timestamp': self._capture_ts_epoch(meta.get('timestamp')),
                'camera_name': str(
                    getattr(self, '_camera_display_name', '') or ''),
                'gender': genero,
                'age_range': edad,
                'visitas': visitas,
                'person_uuid': meta.get('person_uuid'),
                'crop_image': self._captura_en_b64(stem),
            }
            if not hasattr(self, '_alertas_pendientes'):
                self._alertas_pendientes = []
            self._alertas_pendientes.append(alerta)
            # Cota de memoria: si el cliente no recoge, no se acumula sin fin.
            if len(self._alertas_pendientes) > 200:
                self._alertas_pendientes = self._alertas_pendientes[-100:]
        except Exception:
            pass

    def _captura_en_b64(self, stem: str) -> str:
        """Recorte de la persona en base64, para que la alerta lleve imagen.

        Sin imagen, el emisor de WhatsApp descarta la alerta y la tarjeta del
        panel sale sin miniatura."""
        try:
            import base64 as _b64
            ruta = os.path.join(AnalyticsConfig.CAPTURE_DIR, "persons",
                                f"{stem}.jpg")
            if not os.path.isfile(ruta):
                return ''
            with open(ruta, 'rb') as f:
                return _b64.b64encode(f.read()).decode('ascii')
        except Exception:
            return ''
