"""Puente entre el pipeline de inferencia y los módulos del stand.

## Por qué existe este archivo

Los diez módulos del stand deciden bien, pero hasta ahora **nadie los
llamaba**: `tiendas.py` procesaba frames y no alimentaba
nada. Este puente es lo que los conecta.

Vive aparte a propósito (regla 2 del proyecto: cambios quirúrgicos y
aditivos). El archivo de inferencia son 3620 líneas en producción sirviendo
también a Tienda y Perimetrales; meterle 400 líneas de lógica del stand lo
haría más frágil para todos. Aquí el enganche son ~15 líneas allí y todo lo
demás en este módulo.

## Qué NO hace: su propio Re-ID

El pipeline ya resuelve identidad persistente y la tiene calibrada en
producción. Este puente **adopta ese `persistent_id`** en vez de calcular uno
en paralelo: dos galerías sobre los mismos rostros acabarían discrepando, y
entonces «cuántas personas pasaron» tendría dos respuestas. Lo que sí aporta
`stand_identidad` son las reglas de conteo (mínimo de frames, cruce de línea,
permanencia en zona) que el pipeline no tiene.

## Coste

Todo lo caro está condicionado:
  - pose solo sobre visitantes ya atendidos y dentro de la ventana posterior;
  - VLM en cola con tope por minuto;
  - heatmap y persistencia con su propio ritmo, no cada frame.

Si algo falla, se registra y se sigue: **el stand nunca debe tumbar el bucle
de inferencia** del que dependen los otros clientes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import stand_config as _cfgmod
from .stand_almacen import AlmacenStand, TipoEvento
from .stand_degustacion import DetectorDegustacion
from .stand_demografia import (FUENTE_CORPORAL, FUENTE_FACIAL,
                               AgregadorDemografia)
from .stand_emisor import EmisorStand
from .stand_estados import MaquinaVisitante
from .stand_heatmap import MapaCalorStand
from .stand_identidad import GestorIdentidad
from .stand_personal import GaleriaPersonal, OrigenPersonal

logger = logging.getLogger(__name__)

PERIODO_SNAPSHOT_HEATMAP_S = 30.0
PERIODO_PERSISTENCIA_S = 5.0
PERIODO_SYNC_PERSONAL_S = 5.0


def _punto_pie(box) -> Tuple[float, float]:
    """Centro INFERIOR del bbox: donde la persona pisa.

    El centro de la caja flota a la altura del pecho y dibujaría el mapa y las
    zonas medio metro por delante de donde la persona está realmente.
    """
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return ((x1 + x2) / 2.0, y2)


class PuenteStand:
    """Una instancia por cámara con el módulo habilitado."""

    def __init__(self, ancho: int, alto: int,
                 camara_id: Any = '', cfg=None) -> None:
        self.cfg = cfg if cfg is not None else _cfgmod.cargar()
        self.ancho, self.alto = int(ancho), int(alto)
        self.camara_id = str(camara_id)

        self.almacen = AlmacenStand()
        self.identidad = GestorIdentidad(self.cfg)
        self.personal = GaleriaPersonal(self.cfg)
        self.demografia = AgregadorDemografia(self.cfg)
        self.degustacion = DetectorDegustacion(self.cfg)
        self.heatmap = MapaCalorStand(self.cfg, self.ancho, self.alto)
        self.emisor = EmisorStand()

        self._maquinas: Dict[str, MaquinaVisitante] = {}
        self._vistos: Dict[int, float] = {}          # tracker_id -> ultimo ts
        self._pendientes: set[str] = set()           # pids por persistir
        self._personal_sync: set = set()             # pids enrolados desde BD
        self._ultimo_snapshot = 0.0
        self._ultima_persistencia = 0.0
        self._ultimo_sync_personal = 0.0
        self._frame_idx = 0
        self._pose = None
        self._pose_intentada = False

        # Polígonos en píxeles, resueltos una vez.
        self._zonas_px = {
            n: self.cfg.zona_px(n, self.ancho, self.alto)
            for n in ('zona_paso', 'zona_stand', 'zona_atencion',
                      'zona_personal')
        }

        listo, motivo = self.cfg.listo_para_contar()
        self.activo = listo
        if not listo:
            logger.warning("stand: NO se contará nada — %s", motivo)
            self.sesion_id = None
            return

        # REUTILIZAR la jornada ya abierta de hoy en vez de crear otra. Cada
        # reinicio del servidor (o reconexión del cliente) instancia este
        # puente de nuevo; abrir sesión cada vez partía la jornada en trozos
        # inconexos y hacía que una misma persona contase en varios.
        self.sesion_id = self.almacen.sesion_abierta_de(self.cfg.stand_id)
        if self.sesion_id is not None:
            logger.info("stand: reanudando la sesión %s de hoy (%s)",
                        self.sesion_id, self.cfg.stand_id)
        else:
            self.sesion_id = self.almacen.abrir_sesion(
                stand_id=self.cfg.stand_id,
                nombre=str(self.cfg.get('stand.nombre', '')),
                producto=str(self.cfg.get('stand.producto', '')),
                camara_id=self.camara_id or self.cfg.camara_id,
                config_snapshot=self._snapshot_config())
            logger.info("stand: sesión %s abierta (%s)", self.sesion_id,
                        self.cfg.stand_id)
        self.emisor.sesion_id = self.sesion_id
        for aviso in self.cfg.avisos():
            logger.warning("stand: %s", aviso)

    def _snapshot_config(self) -> Dict[str, Any]:
        """Umbrales vigentes al abrir. Sin esto, recalibrar a media jornada
        hace incomparables las cifras y nadie sabe con qué reglas se midió."""
        return {
            'detencion': self.cfg.get('detencion'),
            'atencion': self.cfg.get('atencion'),
            'conteo': self.cfg.get('conteo'),
            'reid': self.cfg.get('reid'),
            'degustacion': self.cfg.get('degustacion'),
            'zonas_calibradas': self.cfg.zonas_calibradas(),
            'homografia_activa': self.cfg.homografia_activa,
        }

    # ── Zonas ───────────────────────────────────────────────────────────
    def _en_zona(self, nombre: str, punto: Tuple[float, float]) -> bool:
        poli = self._zonas_px.get(nombre)
        if poli is None:
            return False
        try:
            import cv2
            return cv2.pointPolygonTest(poli, (float(punto[0]),
                                               float(punto[1])), False) >= 0
        except Exception:                                   # noqa: BLE001
            return False

    def _zona_de(self, punto: Tuple[float, float]) -> str:
        """La zona MÁS específica que contiene el punto."""
        for nombre in ('zona_atencion', 'zona_personal', 'zona_stand',
                       'zona_paso'):
            if self._en_zona(nombre, punto):
                return nombre
        return ''

    # ── Bucle ───────────────────────────────────────────────────────────
    def procesar(self, frame: np.ndarray, tracks: Dict[Any, Dict],
                 demografia_cache=None,
                 ts: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Un frame. Devuelve el bloque `metadata['stand']` o None.

        `tracks` es `self.active_tracks` del pipeline; `demografia_cache` es
        `self._demographics` (para leer las muestras que ya calcula).
        """
        if not self.activo or self.sesion_id is None:
            return None
        ahora = float(ts if ts is not None else time.time())
        self._frame_idx += 1
        self._sincronizar_personal(ahora)

        detalle_tracks: List[Dict[str, Any]] = []
        pid_chef, pie_chef = self._localizar_chef(tracks)

        for tid, tr in list(tracks.items()):
            try:
                self._procesar_track(frame, tid, tr, pid_chef, pie_chef,
                                     demografia_cache, ahora, detalle_tracks)
            except Exception as exc:                        # noqa: BLE001
                # Un track corrupto no puede llevarse por delante a los
                # demas NI al bucle de inferencia del que dependen Tienda y
                # Perimetrales. Lo cazo la prueba con un box de basura.
                logger.debug("stand: track %s ignorado: %s", tid, exc)

        self._cerrar_ausentes(tracks, ahora)
        self._persistir(ahora)
        self._snapshot(ahora)

        return self.emisor.construir(
            contadores=self.almacen.contadores(self.sesion_id),
            salud=self.salud(),
            tracks=detalle_tracks,
            heatmap=lambda: self.heatmap.como_payload('permanencia'),
            ahora=ahora)

    def _procesar_track(self, frame, tid, tr, pid_chef, pie_chef,
                        demografia_cache, ahora: float,
                        detalle_tracks: List[Dict[str, Any]]) -> None:
        """Un track. Lo llama `procesar` dentro de su propio try."""
        box = tr.get('box')
        if box is None:
            return
        pie = _punto_pie(box)
        pie_norm = (pie[0] / self.ancho, pie[1] / self.alto)
        zona = self._zona_de(pie)

        pid_externo = tr.get('persistent_id') if tr.get(
            'persistent_confirmed') else None
        pid = self.identidad.observar(
            tid, punto_pie=pie_norm, zona=zona,
            en_zona_paso=(zona == 'zona_paso'),
            en_zona_stand=(zona in ('zona_stand', 'zona_atencion')),
            identidad_externa=pid_externo, ts=ahora)
        self._vistos[tid] = ahora
        if pid.startswith('prov:'):
            return                       # sin identidad firme no se cuenta

        # ── Personal ──
        self.personal.observar(pid, en_zona_personal=(zona == 'zona_personal'),
                               ts=ahora)
        es_personal = self.personal.es_personal(pid)

        # ── Demografía (de las muestras que el pipeline ya calculó) ──
        self._absorber_demografia(pid, tid, demografia_cache)

        if es_personal:
            detalle_tracks.append(self._detalle(pid, zona, ahora, True))
            return

        # ── Máquina de estados ──
        maq = self._maquinas.get(pid)
        if maq is None:
            maq = MaquinaVisitante(
                pid, self.cfg, al_evento=self._registrar_evento(pid),
                usar_metros=self.cfg.homografia_activa)
            self._maquinas[pid] = maq

        dist_chef = None
        if pie_chef is not None:
            dist_chef = float(np.hypot(pie[0] - pie_chef[0],
                                       pie[1] - pie_chef[1]))
            if not self.cfg.homografia_activa:
                # Sin homografía el umbral va en píxeles: se normaliza por la
                # altura del bbox, que es el proxy de escala acordado (una
                # persona mide ~1.7 m).
                alto_box = max(1.0, float(box[3]) - float(box[1]))
                dist_chef = dist_chef / alto_box * 1.7

        maq.actualizar(
            ts=ahora, punto_pie=pie_norm,
            en_zona_paso=(zona == 'zona_paso'),
            en_zona_stand=(zona in ('zona_stand', 'zona_atencion')),
            en_zona_atencion=(zona == 'zona_atencion'),
            distancia_chef=dist_chef,
            chef_disponible=pid_chef is not None)

        # ── Heatmap ──
        v = self.demografia.veredicto(pid)
        self.heatmap.registrar_posicion(
            pie_norm, dt=1.0 / 25.0, genero=v.genero,
            rango_etario=v.rango_etario,
            clasificacion=None, hora=time.localtime(ahora).tm_hour)
        if maq.se_detuvo and not getattr(maq, '_pintado_detencion', False):
            self.heatmap.registrar_detencion(
                pie_norm, genero=v.genero, rango_etario=v.rango_etario,
                hora=time.localtime(ahora).tm_hour)
            maq._pintado_detencion = True

        # ── Degustación (pose + VLM) ──
        if maq.fue_atendido and not maq.degusto:
            self._buscar_degustacion(frame, box, pid, maq, ahora)

        self._pendientes.add(pid)
        detalle_tracks.append(self._detalle(pid, zona, ahora, False))

    def _sincronizar_personal(self, ahora: float) -> None:
        """Refleja en la galería VIVA los `es_personal` que hay en la BD.

        El botón «Marcar como personal» del cliente hace POST a
        `/api/stand/{sesion}/personal/enrolar`, que solo escribe el flag en
        `stand_visitantes`. La galería que consulta `_localizar_chef`
        (`GaleriaPersonal`) vive en memoria y nace vacía con cada reinicio
        del servidor: sin esta relectura el marcado del operador no llegaba
        nunca al pipeline, `chef_disponible` era False para siempre y no se
        detectaba una sola atención (7-ago-2026: chef marcado el día 5 en la
        BD y cero eventos de atención después).

        También da de baja: desmarcar en el cliente (es_personal=False) saca
        a la persona de la galería en el siguiente ciclo, pero solo si fue
        esta sincronización quien la metió — los enrolados por Re-ID o por
        propuesta confirmada no se tocan.
        """
        if (ahora - self._ultimo_sync_personal) < PERIODO_SYNC_PERSONAL_S:
            return
        self._ultimo_sync_personal = ahora
        try:
            en_bd = set(self.almacen.personal_de(self.sesion_id))
        except Exception as exc:                            # noqa: BLE001
            logger.debug("stand: no se pudo leer el personal de la BD: %s",
                         exc)
            return
        for pid in en_bd - self._personal_sync:
            if not self.personal.es_personal(pid):
                self.personal.enrolar(pid, origen=OrigenPersonal.MANUAL,
                                      ts=ahora)
        for pid in self._personal_sync - en_bd:
            self.personal.dar_de_baja(pid)
        self._personal_sync = en_bd

    # ── Piezas ──────────────────────────────────────────────────────────
    def _localizar_chef(self, tracks: Dict[Any, Dict]):
        """Chef = personal enrolado visible ahora. Sin él no hay atención."""
        for tr in tracks.values():
            pid = tr.get('persistent_id')
            box = tr.get('box')
            if not pid or box is None or not tr.get('persistent_confirmed'):
                continue
            if self.personal.es_personal(str(pid)):
                return str(pid), _punto_pie(box)
        return None, None

    def _absorber_demografia(self, pid: str, tid: Any, cache) -> None:
        """Traduce la demografía del pipeline a una muestra del agregador.

        No se re-infiere nada: `DemographicsClassifier` ya corrió sobre este
        track. Aquí solo se reagrupa POR PERSONA con los umbrales del stand.
        """
        if cache is None:
            return
        try:
            demo = cache.get_cached(tid)
        except Exception:                                   # noqa: BLE001
            return
        if not demo:
            return
        genero = (demo.get('gender') or '').strip().lower()
        if genero in ('', 'desconocido'):
            return
        genero = {'masculino': 'hombre', 'femenino': 'mujer'}.get(genero,
                                                                  genero)
        edad = demo.get('age')
        if edad is None:
            rango = str(demo.get('age_range') or '')
            if '-' in rango:
                try:
                    lo, hi = rango.split('-')[:2]
                    edad = (float(lo) + float(hi)) / 2.0
                except ValueError:
                    edad = None
        fuente = (FUENTE_CORPORAL if demo.get('age_source') == 'body'
                  else FUENTE_FACIAL)
        self.demografia.muestra(
            pid, genero=genero,
            genero_conf=float(demo.get('confidence') or 0.0),
            edad=edad, fuente=fuente, frame_idx=self._frame_idx)

    def _buscar_degustacion(self, frame, box, pid: str,
                            maq: MaquinaVisitante, ahora: float) -> None:
        kps = self._keypoints(frame, box)
        if kps is None:
            return
        encolado = self.degustacion.observar(
            pid, kps, ahora, fue_atendido=True,
            ts_fin_atencion=getattr(maq, '_ts_primera_atencion', None))
        if not encolado:
            return
        recorte = self._recorte(frame, box)
        confirmadas = self.degustacion.procesar_cola(
            lambda _p: [recorte] * 3 if recorte is not None else [],
            ahora=ahora)
        for c in confirmadas:
            m = self._maquinas.get(c.persistent_id)
            if m is not None:
                m.marcar_degustacion(c.ts, c.confianza, c.metodo)
                self._pendientes.add(c.persistent_id)
                v = self.demografia.veredicto(c.persistent_id)
                self.emisor.anotar_evento(
                    'degustacion', genero=v.genero,
                    rango_etario=v.rango_etario, ts=c.ts)

    def _keypoints(self, frame, box):
        """Pose del recorte. Crea el motor la primera vez que hace falta."""
        if not self._pose_intentada:
            self._pose_intentada = True
            try:
                # ViTPose (transformers, Apache) es el unico motor de pose
                # desde el 3-sep-2026: la salida de ultralytics (AGPL) del stand.
                # El contrato no cambia: 17 keypoints COCO [x, y, conf] en
                # coordenadas del recorte.
                from .pose_motor import crear_motor_pose
                self._pose = crear_motor_pose()
                if self._pose is None:
                    logger.error("stand: sin motor de pose — no habrá "
                                 "detección de degustación")
            except Exception as exc:                        # noqa: BLE001
                logger.error("stand: no se pudo crear el motor de pose: %s",
                             exc)
        if self._pose is None:
            return None
        recorte = self._recorte(frame, box)
        if recorte is None:
            return None
        try:
            return self._pose.keypoints_17(recorte)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("stand: pose falló: %s", exc)
            return None

    @staticmethod
    def _recorte(frame, box):
        try:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 40:
                return None
            return frame[y1:y2, x1:x2].copy()
        except Exception:                                   # noqa: BLE001
            return None

    def _registrar_evento(self, pid: str):
        def _cb(tipo: str, zona: str, payload: Dict[str, Any]) -> None:
            try:
                self.almacen.registrar_evento(
                    self.sesion_id, tipo, persistent_id=pid,
                    zona=zona or None, payload=payload or None,
                    confianza=payload.get('confianza'))
                if tipo == TipoEvento.INICIO_ATENCION:
                    v = self.demografia.veredicto(pid)
                    self.emisor.anotar_evento('atencion', genero=v.genero,
                                              rango_etario=v.rango_etario)
            except Exception as exc:                        # noqa: BLE001
                logger.debug("stand: no se registró %s: %s", tipo, exc)
        return _cb

    def _detalle(self, pid: str, zona: str, ahora: float,
                 es_personal: bool) -> Dict[str, Any]:
        v = self.demografia.veredicto(pid)
        maq = self._maquinas.get(pid)
        return {
            'persistent_id': pid, 'genero': v.genero,
            'rango_etario': v.rango_etario, 'zona': zona,
            'es_personal': es_personal,
            'dwell_s': getattr(maq, '_dwell_max_s', 0.0) if maq else 0.0,
        }

    def _cerrar_ausentes(self, tracks: Dict[Any, Dict], ahora: float) -> None:
        """Cierra los tracks que el pipeline ya no reporta."""
        vivos = set(tracks.keys())
        for tid in [t for t in self._vistos if t not in vivos]:
            del self._vistos[tid]
            try:
                r = self.identidad.cerrar(tid, ts=ahora)
            except Exception:                               # noqa: BLE001
                continue
            if r.persistent_id:
                self._pendientes.add(r.persistent_id)
                if r.descartado:
                    self.almacen.upsert_visitante(
                        self.sesion_id, r.persistent_id,
                        descartado=True, motivo_descarte=r.motivo_descarte)

    def _persistir(self, ahora: float) -> None:
        if (ahora - self._ultima_persistencia) < PERIODO_PERSISTENCIA_S:
            return
        self._ultima_persistencia = ahora
        for pid in list(self._pendientes):
            maq = self._maquinas.get(pid)
            if maq is None:
                continue
            try:
                resumen = maq.cerrar(ts=ahora)
                campos = maq.campos_visitante(resumen)
                campos.update(
                    self.demografia.veredicto(pid).como_campos_visitante())
                campos['ultima_deteccion'] = time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(ahora))
                self.almacen.upsert_visitante(self.sesion_id, pid, **campos)
            except Exception as exc:                        # noqa: BLE001
                logger.debug("stand: no se persistió %s: %s", pid, exc)
        self._pendientes.clear()

    def _snapshot(self, ahora: float) -> None:
        if (ahora - self._ultimo_snapshot) < PERIODO_SNAPSHOT_HEATMAP_S:
            return
        self._ultimo_snapshot = ahora
        try:
            self.heatmap.snapshot(self.almacen, self.sesion_id)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("stand: snapshot de heatmap falló: %s", exc)

    def salud(self) -> Dict[str, Any]:
        pids = [p for p, m in self._maquinas.items() if m.se_detuvo]
        return {
            'tracks_activos': self.identidad.tracks_activos(),
            'null_rate_demografia': self.demografia.null_rate(pids).get(
                'null_rate'),
            **self.degustacion.salud(),
            **self.identidad.calidad_dato(),
            **self.demografia.salud(),
        }

    def cerrar(self) -> None:
        """Cierra la jornada. Idempotente."""
        if self.sesion_id is None:
            return
        try:
            self._persistir(time.time() + 1e6)   # fuerza el volcado final
            self.heatmap.snapshot(self.almacen, self.sesion_id)
            self.almacen.cerrar_sesion(self.sesion_id)
            logger.info("stand: sesión %s cerrada", self.sesion_id)
        except Exception as exc:                            # noqa: BLE001
            logger.error("stand: fallo al cerrar la sesión: %s", exc)
        finally:
            self.sesion_id = None


# ── Fábrica perezosa ────────────────────────────────────────────────────
def crear_si_habilitado(ancho: int, alto: int,
                        camara_id: Any = '') -> Optional[PuenteStand]:
    """Devuelve el puente, o None si el módulo está apagado.

    Nunca lanza: si la configuración está rota, Tienda y Perimetrales deben
    seguir funcionando igual que antes de que existiera el stand.
    """
    try:
        cfg = _cfgmod.cargar()
        if not cfg.habilitado:
            return None
        return PuenteStand(ancho, alto, camara_id, cfg)
    except Exception as exc:                                # noqa: BLE001
        logger.error("stand: no se pudo iniciar el módulo: %s", exc)
        return None
