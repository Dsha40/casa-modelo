"""Pipeline `Anaqueles` — interacción con los niveles del estante (modo propio).

## Qué mide y para quién

Marketing de tienda: el operador dibuja N áreas sobre la cámara (una por
nivel del anaquel, separados por la lámina metálica del soporte) y el modo
responde dos preguntas por área:

  - **visitas** — cuántas personas se acercaron a ESE nivel (la franja
    superior de su cuerpo tocó el área un mínimo de tiempo);
  - **agarres** — cuántas veces alguien METIÓ LA MANO en ese nivel (muñeca
    de la pose dentro del polígono, confirmada en frames consecutivos).

Cada área acumula además su PROPIO mapa de calor (dónde exactamente tocan)
y la demografía de sus visitas (género/edad del pipeline compartido). Con
eso el dashboard responde «cuál es el anaquel más visitado y más agarrado».

## Por qué es un pipeline aparte (mismo argumento que AmazonasStand)

Elegir el modo ES habilitarlo: un flag dentro de `Tiendas` habría hecho
medir estantes a cámaras que no ven ninguno. Hereda TODO el pipeline de
personas (detección, ByteTrack, Re-ID, demografía) y añade su medición
DESPUÉS del `metadata`, sin tocar la firma de `process_frame` que comparte
la familia.

## Las zonas viajan POR FRAME desde el cliente

Patrón PermanenciaArea, no el YAML calibrado del stand: el operador dibuja
las áreas en el recuadro y llegan en `data.anaqueles_zonas` (píxeles del
frame enviado). `app.py` las estampa como atributos del procesador de la
cámara (`_anaqueles_zonas`, `_anaqueles_nombres`, `_anaqueles_activate`)
igual que hace con la flecha de permanencia.

## Las cinco lecciones del stand, aplicadas de nacimiento

1. Estado por cámara: los tracks se leen de `camera_states[camera_id]` a
   pelo, NUNCA de `self.active_tracks` (vacío tras `_pop_state`).
2. Guardia de escena (`DetectorEscena`): el operador cambia el canal de
   iVMS y las áreas quedan sobre OTRA escena; se suspende el conteo y se
   avisa, en vez de fabricar cifras creíbles y falsas.
3. `__init__(*args, **kwargs)` sobre `super()`: `get_camera_processor`
   clona con `type(self)` y cualquier otra firma deja el modo MUDO.
4. Nada después del `super()` puede propagar excepciones: el mismo
   websocket sirve el vídeo.
5. Totales ABSOLUTOS persistidos en disco (`anaqueles_registro`): el
   procesador renace a cero en cada reconexión.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import (anaqueles_grabador, anaqueles_productos, anaqueles_stock,
               ritmo_imagenes)
from .anaqueles_pose import PoseMunecas
from .anaqueles_registro import GRID_DIM, RegistroAnaqueles, raiz_salida
from .estacionamiento_escena import DetectorEscena
from .tiendas import TiendasProcess

logger = logging.getLogger(__name__)

#: Diagnóstico del gesto (28-sep-2026): con `ELDE_ANAQUELES_DIAG=1` el log
#: cuenta, frame a frame, qué ve la máquina del agarre de CADA muñeca
#: (profundidad, quietud, armado) y por qué descarta. Existe porque el
#: 28-sep un vídeo que ya había contado agarres dejó de contarlos con la
#: mano visiblemente dentro, y los `descartes` no se ven en ningún sitio.
#: Apagado por defecto: a 6 fps son varias líneas por frame.
_DIAG = os.getenv('ELDE_ANAQUELES_DIAG', '').strip() not in ('', '0')

#: Nombre del modo tal como lo pide el cliente por WebSocket.
MODO = "Anaqueles"

# ── Configuración ────────────────────────────────────────────────────────
#: Defaults en código. Igual que en permanencia: SOLO se copian del YAML las
#: claves pre-declaradas aquí — una clave nueva sin declarar se descarta.
_DEFECTOS: Dict[str, Any] = {
    # Visita: la persona se DETIENE frente al nivel este tiempo seguido.
    # (v2, 14-ago: pasar CAMINANDO junto al estante ya no cuenta — el dwell
    # solo corre con el cuerpo quieto.)
    'visita_dwell_s': 1.5,
    # La misma persona en el mismo nivel no re-cuenta visita hasta pasar esto.
    'visita_cooldown_s': 120.0,
    # Fracción mínima de la FRANJA superior del cuerpo que debe pisar el
    # área: el roce de quien va por el medio del pasillo no abre compuerta.
    'visita_solape_min': 0.15,
    # ── Quietud (v2): el discriminador de «estar» vs «pasar» ──
    # Velocidad máxima del cuerpo para considerarlo QUIETO, en ANCHOS DE SU
    # PROPIO BBOX por segundo (independiente de la distancia a la cámara).
    # Caminando ≈ 1.5-3 anchos/s; parado con vaivén ≈ <0.3.
    'quieto_max_vel': 0.45,
    # Ventana (s) sobre la que se estima la velocidad.
    'quieto_ventana_s': 1.0,
    # ── Dedup ESPACIAL anti-fragmentos (v2) ──
    # A 2-3 FPS los tracks se parten con cada oclusión y la MISMA persona
    # parada recibe un id `prov:` nuevo: el cooldown por persona no la
    # reconoce. Regla: mismo sitio (este radio, en anchos de bbox) + dentro
    # de la ventana = la misma persona, no se re-cuenta. Dos personas cuyos
    # tracks COEXISTEN en el mismo frame jamás son fragmentos (v3): el
    # acompañante hombro con hombro sí cuenta.
    'dedup_espacial_frac': 0.8,
    # Ventana del dedup espacial para un sitio VACANTE (v3): el horizonte
    # real en que un track partido renace (~segundos), no el cooldown de
    # 120 s — con 120 s, el segundo cliente que llegaba al mismo punto
    # popular no contaba. Mientras la persona SIGUE en el sitio, su entrada
    # se refresca sola y no envejece.
    'dedup_espacial_ventana_s': 20.0,
    # Agarre: un GESTO de la muñeca, no una presencia (corregido 14-ago: una
    # persona PARADA frente al estante se proyecta sobre el polígono y su
    # muñeca colgando quedaba «dentro» — contaba un agarre falso por
    # cooldown, para siempre). Las cuatro defensas:
    'agarre_frames_min': 2,
    'agarre_cooldown_s': 10.0,
    # La muñeca debe PENETRAR el polígono. v3: la profundidad ESCALA con la
    # persona — `frac` del ancho de su bbox, acotada entre 4 px y el tope
    # `px` (y nunca más del 40% del alto del nivel dibujado, para que
    # ningún área angosta quede matemáticamente imposible). Histéresis a la
    # mitad.
    'agarre_profundidad_frac': 0.08,
    'agarre_profundidad_px': 12.0,
    # Frames seguidos con la muñeca fuera/superficial para RE-ARMAR: solo
    # una transición fuera→dentro puede contar, y tras contar hay que salir
    # antes de poder contar otra. Parado quieto = cero agarres.
    'agarre_rearme_frames': 2,
    # Las dos de arriba (frames_min y rearme) también en SEGUNDOS (5-oct-
    # 2026, `ritmo_imagenes.py`): se exige lo MAYOR entre esas imágenes y
    # estos segundos a los fps medidos. A ~5-8 fps da 2, como siempre; a
    # 15 fps, 4. 0 = solo imágenes.
    'agarre_min_s': 0.25,
    'agarre_rearme_s': 0.25,
    # Segundos QUIETO antes de poder contar (v2: era «en compuerta», pero
    # quien camina a lo largo del estante pasa >1.2 s en compuerta y su
    # braceo hacía el patrón completo — un agarre real ocurre parado).
    'agarre_asentamiento_s': 1.2,
    # Amplitud mínima del gesto (fracción del ancho del bbox): la mano debe
    # VIAJAR desde donde estaba fuera hasta dentro del nivel. El temblor de
    # la muñeca colgando (20-50 px alrededor del borde) no llega.
    'agarre_amplitud_min': 0.30,
    # ── Correcciones del 28-sep-2026, APAGADAS por defecto ──
    # Salen de un vídeo real (la grabación del 23-sep de la cámara 3, 482
    # frames reproducidos con el grabador de diagnóstico) en el que una
    # clienta coge botes de Coffee-mate de la balda ALTA y la máquina no
    # contaba nada. Resultado de reproducirlo con cada regla, mismo área:
    #   regla de siempre ................ 0 agarres
    #   solo `pose_cercania` ............ 1
    #   `pose_cercania` + `extremo` ..... 2 (los dos reales, revisados en
    #                                      imagen; repetido en vivo 2 veces)
    # El defecto del CÓDIGO sigue apagado (así las pruebas fijan la regla de
    # siempre), pero desde el 29-sep `anaqueles_config.yaml` las ENCIENDE
    # como norma general para todas las cámaras, con los valores anteriores
    # comentados al lado. Aún no medidas en las cámaras en vivo.
    #
    #  - `agarre_amplitud_desde`: 'ultimo' mide el viaje desde la ÚLTIMA
    #    muestra de mano fuera; si esa fue rozando el borde (-3 px), el
    #    viaje medido se queda en ~25 px aunque la mano viniera de -30.
    #    'extremo' lo mide desde el punto MÁS alejado de esa racha fuera.
    #    Sola no cambió nada; junto a `pose_cercania` aportó el 2.º agarre.
    'agarre_amplitud_desde': 'ultimo',
    #
    # PROBADAS Y QUITADAS el 29-sep-2026 (no las vuelvas a añadir sin datos
    # nuevos que las justifiquen):
    #  - `agarre_quietud='antes_de_entrar'`: exigir la quietud con la mano
    #    aún FUERA en vez de al contar. 0 agarres en el mismo vídeo, igual
    #    que la regla de siempre: la clienta alargó el brazo nada más
    #    llegar y no había estado 1,2 s quieta en ningún momento, ni antes
    #    ni después de entrar.
    #  - `pose_en_movimiento`: pose también para quien se mueve DENTRO de
    #    la compuerta. 0 agarres: la mano entra con el brazo en alto, y con
    #    el brazo abajo la compuerta está CERRADA (la balda queda por encima
    #    de la cabeza), así que ni siquiera así se veía la entrada. Subió
    #    la pose de 40 a 54 cálculos sin cambiar un solo conteo. Lo que sí
    #    resolvió el caso es `pose_cercania`, que la cubre: mira también a
    #    quien se mueve, y además a quien está debajo del área.
    #
    #  - `pose_cercania` (28-sep-2026): mirar la muñeca de quien está CERCA
    #    del área (su caja la solapa en horizontal y su borde de arriba no
    #    queda más de `cercania_frac_alto` de su propio alto por debajo del
    #    área), aunque se mueva o aún no pise la compuerta. Medido sobre un
    #    vídeo real: con pose solo «en compuerta y quieto», la máquina vio la
    #    muñeca a -34 px y 2 s después a +92 px, y se perdió la entrada
    #    entera (-158 → +49 → +74): con el brazo abajo la compuerta está
    #    cerrada y al subirlo el cuerpo cuenta como «en movimiento». Contar
    #    sigue exigiendo lo mismo; esto solo deja de mirar a saltos.
    'pose_cercania': False,
    'cercania_frac_alto': 0.5,
    'conf_muneca_min': 0.30,
    # Pose solo para quienes están en compuerta y quietos (o también cerca
    # del área, con `pose_cercania`), y como mucho estos por frame (los más
    # cercanos al anaquel): N personas = N inferencias.
    'pose_max_personas': 4,
    # Qué fracción SUPERIOR del bbox cuenta como «alcanza el estante»: el
    # ancla de pie del padre es inútil a la altura de un anaquel.
    'franja_superior': 0.60,
    # Salud: aviso si lleva tanto sin ver NI UNA persona (patrón permanencia).
    'aviso_sin_personas_s': 900.0,
    # Tras un cambio de escena confirmado, cuánto NO contar mientras el
    # operador termina de conmutar el canal.
    'escena_estabilizacion_s': 3.0,
    # Ritmos del bloque en vivo (patrón stand_emisor).
    'periodo_contadores_s': 2.0,
    'periodo_salud_s': 10.0,
    'periodo_heatmap_s': 30.0,
    # Personal enrolado (14-ago): cooldown de las alertas operativas para no
    # inundar — el guardia patrulla el pasillo cada pocos minutos y el
    # pasillero pasa un rato largo surtiendo el mismo nivel.
    'alerta_seguridad_cooldown_s': 300.0,
    'alerta_surte_cooldown_s': 180.0,
    # Dwell para «está surtiendo» (más largo que la visita de un cliente:
    # rozar el área caminando no es surtir).
    'surte_dwell_s': 3.0,
    # Aviso de «posible agotado» tras un agarre (5-oct-2026,
    # `anaqueles_stock.py`). El UMBRAL NO ESTÁ CALIBRADO: no hay todavía ni
    # una balda vaciada de verdad, solo el vídeo. Cada medida sale en el log
    # (`STOCK [...] cambio=...`) para ajustarlo.
    # Interruptor (6-oct-2026): para la A/B de fps con y sin stock. Apagado,
    # no se vigila ni se arma nada tras un agarre.
    'stock_activo': True,
    'stock_umbral': 0.45,
    # Mediciones seguidas por encima (alerta) o por debajo (sigue surtido).
    'stock_confirmaciones': 3,
    # Tras el agarre, cuánto esperar antes de mirar (la mano aún sale).
    'stock_espera_s': 2.0,
    # Si en este tiempo nadie deja ver el producto, se deja de vigilar.
    'stock_ventana_s': 180.0,
    'stock_muestreo_s': 1.0,
    # Con la alerta puesta se mira menos a menudo: solo para ver si reponen.
    'stock_muestreo_alerta_s': 5.0,
    # Repuesto = cambio por debajo de umbral × esto (histéresis).
    'stock_repuesto_frac': 0.6,
    # La petición de foto de referencia caduca si siguen tapando el producto.
    'stock_peticion_max_s': 120.0,
}


def ruta_config() -> Path:
    return (Path(__file__).resolve().parents[3] / 'config' /
            'anaqueles_config.yaml')


def cargar_config(camara: str = '') -> Dict[str, Any]:
    """Defaults + YAML global + bloque `camaras.<camera_id>` si existe."""
    cfg = dict(_DEFECTOS)
    try:
        ruta = ruta_config()
        if ruta.is_file():
            import yaml
            crudo = yaml.safe_load(ruta.read_text(encoding='utf-8')) or {}
            for clave in _DEFECTOS:
                if clave in crudo:
                    cfg[clave] = crudo[clave]
            por_camara = (crudo.get('camaras') or {}).get(str(camara)) or {}
            for clave in _DEFECTOS:
                if clave in por_camara:
                    cfg[clave] = por_camara[clave]
    except Exception as exc:                                # noqa: BLE001
        logger.error("anaqueles: config ilegible (%s); se usan defaults", exc)
    return cfg


# ── Medición ─────────────────────────────────────────────────────────────
class MedidorAnaqueles:
    """Visitas, agarres y calor de N áreas de UNA cámara.

    Vive fuera del mecanismo de swap de `TiendasProcess`: todo su estado es
    suyo, indexado por (área, persona). Los totales duros están en
    `RegistroAnaqueles` (disco); aquí solo el estado efímero (compuertas,
    rachas, cooldowns).
    """

    def __init__(self, camera_id: Any, cfg: Dict[str, Any],
                 device: Any = None,
                 pose: Optional[PoseMunecas] = None,
                 registro: Optional[RegistroAnaqueles] = None,
                 escena: Optional[DetectorEscena] = None,
                 identificador: Any = None) -> None:
        self.camera_id = str(camera_id)
        self.cfg = cfg
        self.registro = registro or RegistroAnaqueles(camera_id)
        self.pose = pose or PoseMunecas(
            device, float(cfg.get('conf_muneca_min', 0.30)))
        self.escena = escena or DetectorEscena()
        #: Identificador de PERSONAL enrolado (inyectable en pruebas).
        #: Perezoso: la galería solo se carga si de verdad se procesa.
        self._identificador = identificador

        self._poligonos: List[np.ndarray] = []
        self._cajas_area: List[Tuple[float, float, float, float]] = []
        self._nombres: List[str] = []
        self._firma_zonas: Any = None
        #: área -> productos dibujados dentro (25-sep-2026). Solo reparten
        #: los agarres del área; no deciden si hubo agarre.
        self._productos: Dict[int, List[anaqueles_productos.Producto]] = {}
        self._firma_productos: Any = None
        #: «Posible agotado» tras un agarre (5-oct-2026). Sus fotos de
        #: referencia van junto a los JSON del día de esta cámara.
        # `getattr`: las pruebas inyectan registros de mentira sin carpeta.
        self._stock = anaqueles_stock.VigilanteStock(
            camera_id, getattr(self.registro, 'carpeta', None)
            or Path(raiz_salida()) / str(camera_id), cfg)

        #: (área, pid) -> ts en que entró a la compuerta (estancia en curso).
        self._compuerta_desde: Dict[Tuple[int, str], float] = {}
        #: (área, pid) -> ts desde el que está QUIETO dentro de la
        #: compuerta: es el reloj del dwell de visita (v2 — caminar junto
        #: al estante no lo hace avanzar).
        self._quieto_en_area: Dict[Tuple[int, str], float] = {}
        #: (área, pid) con visita YA contada en la estancia en curso.
        self._visita_abierta: set = set()
        #: (área, pid) -> ts de la última visita CONTADA (cooldown).
        self._visita_ts: Dict[Tuple[int, str], float] = {}

        # ── Movimiento por track (v2/v3): quieto vs caminando ──
        #: tid -> deque[(t, cx, cy_sup, ancho_bbox)] (ventana corta). El
        #: centro es de la MITAD SUPERIOR (inmune al truncamiento de
        #: piernas por carritos/oclusiones).
        self._trayectoria: Dict[Any, deque] = {}
        #: tid -> ts desde el que el cuerpo está quieto.
        self._quieto_desde: Dict[Any, float] = {}
        #: tid -> frames NO-quietos consecutivos (histéresis: 1-2 frames
        #: malos pausan el reloj, no lo borran).
        self._no_quieto_racha: Dict[Any, int] = {}
        #: tid -> última vez visto (la poda con gracia: un miss del
        #: detector de 1 frame no derriba la quietud acumulada).
        self._tid_visto: Dict[Any, float] = {}
        #: (área, pid) -> ts desde el que NO está en compuerta (gracia de
        #: estancia: 1 frame ausente no destruye el gesto armado).
        self._fuera_compuerta: Dict[Tuple[int, str], float] = {}
        #: (pid, lado) -> conteo de agarre PENDIENTE de arbitraje (v3): un
        #: reach que atraviesa áreas apiladas solo cuenta donde SE ASIENTA.
        self._agarre_pendiente: Dict[Tuple[str, str], Dict[str, Any]] = {}
        #: Huella 24x18 de la escena EN QUE SE DIBUJARON las zonas: tras un
        #: cambio de canal, el conteo no reanuda por timer sino cuando la
        #: cámara VUELVE a esta escena (v3).
        self._huella_zonas: Optional[np.ndarray] = None
        self._huella_pendiente = False
        #: Áreas de los polígonos (px², shoelace) para el umbral de solape.
        self._areas_poligono: List[float] = []
        # ── Dedup espacial anti-fragmentos (v2) ──
        #: área -> deque[(t, cx, cy, ancho)] de visitas CONTADAS.
        self._visitas_recientes: Dict[int, deque] = {}
        #: área -> deque[(t, x, y, ancho)] de agarres CONTADOS.
        self._agarres_recientes: Dict[int, deque] = {}
        #: Por qué NO se contó (transparencia para calibrar): viaja en salud.
        self._descartes: Dict[str, int] = {}
        #: (área, pid) -> máquina del gesto de agarre:
        #: {'armado': bool, 'dentro': frames seguidos con la muñeca
        #:  PROFUNDA, 'fuera': frames seguidos fuera/superficial}. Solo la
        #: transición fuera→dentro puede contar (ver _DEFECTOS).
        self._agarre_estado: Dict[Tuple[int, str], Dict[str, Any]] = {}
        #: (área, pid) -> ts del último agarre CONTADO (cooldown).
        self._agarre_ts: Dict[Tuple[int, str], float] = {}

        # ── Personal enrolado (excluido del marketing) ──
        #: tid -> ficha del empleado detectado en el ÚLTIMO frame (para que
        #: el pipeline etiquete su caja y el cliente vea por qué no cuenta).
        self.personal_tids: Dict[Any, Dict[str, Any]] = {}
        #: (clave, pid) -> ts de la última alerta emitida (cooldowns).
        self._alerta_personal_ts: Dict[Tuple[str, Any], float] = {}
        #: (área, pid) -> ts en que el pasillero empezó a tocar el área.
        self._personal_desde: Dict[Tuple[int, str], float] = {}
        #: Alertas operativas pendientes (las drena la clase Anaqueles, que
        #: es quien tiene acceso a la cola del pipeline y al frame).
        self._alertas_personal: List[Dict[str, Any]] = []
        #: True mientras el último bloque emitido llevó personal: permite
        #: emitir UNA vez la lista vacía cuando el último empleado sale (sin
        #: esto, «En cámara ahora» se quedaba pegado en el panel para
        #: siempre — lo cazó la revisión del 14-ago).
        self._personal_emitido = False

        self._suspendido_hasta = 0.0
        #: idx -> ts de la última foto de referencia del área (vacía).
        self._ref_ts: Dict[int, float] = {}
        self._aviso_escena: Optional[Dict[str, Any]] = None
        self._ultimo_frame_ts: Optional[float] = None
        # Fps medidos de esta cámara, para pasar a segundos los ajustes del
        # agarre contados en imágenes (`ritmo_imagenes.py`, 5-oct-2026).
        self._ritmo = ritmo_imagenes.Ritmo()
        self._ultima_persona_ts = time.time()
        self._sin_personas_s: Optional[float] = None

        self._feed: deque = deque(maxlen=10)
        self._feed_sucio = False
        self._ultimo_emitido: Dict[str, float] = {}

        #: Munecas vistas en ESTE frame, para que el operador las vea en el
        #: recuadro mientras calibra (23-sep-2026). Es lo unico del bloque
        #: que NO lleva throttle: con 2 s de retraso el punto se pinta donde
        #: la mano YA no esta, que es peor que no pintarlo. Se repone en
        #: cada frame — nunca acumula.
        self._munecas_frame: List[Dict[str, Any]] = []

    # ── Zonas ───────────────────────────────────────────────────────────
    def poner_zonas(self, zonas: Sequence[Sequence[Sequence[float]]],
                    nombres: Sequence[str]) -> None:
        """Adopta los polígonos del frame actual (píxeles del frame).

        La identidad de cada área es su ÍNDICE en la lista: mover un vértice
        no borra contadores. Se recalcula solo si algo cambió, y la firma
        lleva TODOS los vértices: una versión anterior solo miraba el primero
        y el último, así que arrastrar un vértice interior del cuadrilátero
        dejaba al servidor midiendo el polígono viejo en silencio (lo cazó la
        revisión del 13-ago). El costo de reconstruir es trivial: N áreas de
        ~4 vértices.
        """
        # Los nombres se emparejan con su polígono ANTES de filtrar, para que
        # un polígono degenerado (que el cliente y el contrato ya impiden,
        # pero por si acaso) no desalinee los nombres de los demás.
        pares = [(p, str(nombres[i]) if i < len(nombres) else '')
                 for i, p in enumerate(zonas) if len(p) >= 3]
        firma = tuple(
            (tuple((int(pt[0]), int(pt[1])) for pt in p), nombre)
            for p, nombre in pares)
        if firma == self._firma_zonas:
            return
        self._firma_zonas = firma
        self._poligonos = []
        self._cajas_area = []
        self._areas_poligono = []
        self._nombres = [nombre for _p, nombre in pares]
        # Zonas nuevas/movidas: capturar la huella de SU escena en el
        # próximo frame (es el ancla del conteo tras un cambio de canal).
        self._huella_pendiente = True
        for i, (p, _nombre) in enumerate(pares):
            arr = np.array([[float(x), float(y)] for x, y in p],
                           dtype=np.float32)
            self._poligonos.append(arr)
            xs, ys = arr[:, 0], arr[:, 1]
            self._cajas_area.append((float(xs.min()), float(ys.min()),
                                     float(xs.max()), float(ys.max())))
            # Área del polígono (shoelace): el umbral de compuerta se acota
            # por el MENOR de (franja, polígono) — un nivel angosto frente a
            # un bbox gigante era matemáticamente imposible de visitar.
            area_p = 0.0
            n = len(arr)
            for j in range(n):
                xa, ya = arr[j]
                xb, yb = arr[(j + 1) % n]
                area_p += float(xa) * float(yb) - float(xb) * float(ya)
            self._areas_poligono.append(max(1.0, abs(area_p) / 2.0))
            # Asegura el área en el registro con su nombre vigente.
            self.registro.area(i, self._nombre(i))

    def poner_productos(self, crudo: Optional[Sequence[Any]]) -> None:
        """Adopta los productos del frame actual. Llamar DESPUÉS de
        `poner_zonas`: un producto solo vale si su área existe.

        Mismo patrón que las zonas: se reconstruye solo si algo cambió, y
        cada producto nuevo se da de alta en el registro para que la API lo
        liste aunque nadie lo haya tocado todavía.
        """
        firma = anaqueles_productos.firma(crudo, len(self._poligonos))
        if firma == self._firma_productos:
            return
        self._firma_productos = firma
        self._productos = anaqueles_productos.productos_por_area(
            crudo, len(self._poligonos))
        self._registrar_productos()

    def _registrar_productos(self) -> None:
        for idx, productos in self._productos.items():
            for posicion, nombre, _poligono in productos:
                self.registro.producto(idx, posicion, nombre)

    def _nombre(self, idx: int) -> str:
        if idx < len(self._nombres) and str(self._nombres[idx]).strip():
            return str(self._nombres[idx]).strip()
        return f'Área {idx + 1}'

    # ── Bucle ───────────────────────────────────────────────────────────
    def procesar(self, frame: np.ndarray, tracks: Dict[Any, Dict],
                 demografia: Any = None,
                 ahora: Optional[float] = None,
                 reidentifier: Any = None) -> Optional[Dict[str, Any]]:
        """Un frame. Devuelve el bloque `metadata['anaqueles']` o None."""
        t = float(ahora if ahora is not None else time.time())
        dt = 0.0
        if self._ultimo_frame_ts is not None:
            # dt REAL entre frames (el pipeline va a 2-3 FPS efectivos);
            # acotado para que una pausa larga no pese como permanencia.
            dt = max(0.0, min(2.0, t - self._ultimo_frame_ts))
        self._ritmo.anotar(t)
        self._ultimo_frame_ts = t

        # Cruce de medianoche con la conexión viva: el registro cierra el
        # archivo de ayer y abre el de hoy. Sin esto, una sesión nocturna
        # inflaba el histórico de ayer y la API daba 404 toda la mañana.
        if self.registro.rollover_si_cambio_dia():
            self._limpiar_efimero()
            self._visita_ts.clear()
            self._agarre_ts.clear()
            self._visitas_recientes.clear()
            self._agarres_recientes.clear()
            self._descartes.clear()
            for i in range(len(self._poligonos)):
                self.registro.area(i, self._nombre(i))
            self._registrar_productos()

        # 1. Guardia de escena: el canal de iVMS cambió → las áreas están
        #    sobre OTRA escena. Se limpia lo efímero y se SUSPENDE. v3: la
        #    reanudación NO es por timer (a los 3 s seguía contando sobre
        #    el canal equivocado, indefinidamente) — se reanuda cuando la
        #    cámara VUELVE a la escena en que se dibujaron las zonas.
        self._vigilar_huella_zonas(frame, t)
        veredicto = self.escena.evaluar(self.camera_id, frame, t)
        if veredicto:
            self._limpiar_efimero()
            self._suspendido_hasta = (float('inf')
                                      if self._huella_zonas is not None
                                      else t + float(self.cfg.get(
                                          'escena_estabilizacion_s', 3.0)))
            self._aviso_escena = {'tipo': veredicto, 'ts': _hora(t)}
            logger.warning(
                "anaqueles[%s]: escena %s — conteo suspendido hasta que la "
                "cámara vuelva a la escena de las áreas dibujadas",
                self.camera_id, veredicto)

        self._munecas_frame = []
        personas = self._personas_de(tracks)
        # Para recortar a quien agarra (la foto de la alerta de stock,
        # 5-oct-2026): el agarre se confirma más abajo sin el frame a mano.
        self._frame_actual = frame
        self._cajas_tid = {tid: box for _p, tid, box in personas}
        # El PERSONAL enrolado se separa ANTES de medir: el marketing es
        # solo de clientes. Sus visitas/agarres no cuentan, no calientan el
        # mapa ni suman demografía; en su lugar generan alertas operativas.
        personas, personal = self._separar_personal(personas, reidentifier,
                                                    tracks)
        self.personal_tids = {tid: ficha for _p, tid, _b, ficha in personal}
        self._vigilar_silencio(t, bool(personas) or bool(personal))

        personas_por_area: Dict[int, int] = {}
        medir = t >= self._suspendido_hasta
        if self._poligonos and medir:
            personas_por_area = self._medir(frame, personas, demografia, t, dt)
            # Foto de referencia de cada área VACÍA (cada 10 min, escena
            # estable): la base real sobre la que se compone el calor.
            self._guardar_refs(frame, personas_por_area, personas,
                               personal, t)
            self._vigilar_stock(frame, personas, personal, t)
        if personal:
            self._atender_personal(personal, t, medir_areas=medir)

        self.registro.volcar()
        return self._bloque(t, personas_por_area)

    def _vigilar_huella_zonas(self, frame: np.ndarray, t: float) -> None:
        """Captura/compara la huella de la escena de LAS ZONAS. Nunca lanza.

        - Zonas recién dibujadas → la escena actual es su casa (captura).
        - Suspendido por cambio de canal → si la escena actual vuelve a
          parecerse a la casa, programa la reanudación (+ estabilización).
        """
        try:
            from .analytics.heatmap import _huella_escena
            if self._huella_pendiente and t >= self._suspendido_hasta:
                huella = _huella_escena(frame)
                if huella is not None:
                    self._huella_zonas = huella
                    self._huella_pendiente = False
                return
            if (self._suspendido_hasta == float('inf')
                    and self._huella_zonas is not None):
                huella = _huella_escena(frame)
                if huella is None:
                    return
                diff = float(np.mean(np.abs(huella - self._huella_zonas)))
                if diff < 0.65:          # el mismo umbral de DetectorEscena
                    self._suspendido_hasta = t + float(
                        self.cfg.get('escena_estabilizacion_s', 3.0))
                    logger.info(
                        "anaqueles[%s]: la cámara volvió a la escena de "
                        "las áreas — el conteo se reanuda", self.camera_id)
        except Exception:                                   # noqa: BLE001
            pass

    # ── Personal enrolado ───────────────────────────────────────────────
    def _separar_personal(self, personas: List, reidentifier: Any,
                          tracks: Optional[Dict[Any, Dict]] = None
                          ) -> Tuple[List, List]:
        """Parte la lista en (clientes, personal). Nunca lanza.

        DOS vías: (a) el flag `es_personal` que el pipeline base ya marcó en
        el track — cubre el UNIFORME (funciona de espaldas, también para los
        `prov:`); (b) el rostro contra la galería (`ficha_de`). Un `prov:`
        sin uniforme cuenta como cliente: el error barato. El caro —borrar a
        un cliente del marketing— lo evitan los umbrales estrictos.
        """
        if not personas:
            return personas, []
        ident = self._identificador
        if ident is None:
            try:
                from .personal_galeria import get_identificador
                ident = self._identificador = get_identificador()
            except Exception:                               # noqa: BLE001
                ident = None
        clientes, personal = [], []
        for pid, tid, box in personas:
            tr = (tracks or {}).get(tid) or {}
            if tr.get('es_personal'):
                personal.append((pid, tid, box, {
                    'nombre': tr.get('personal_nombre') or 'empleado',
                    'rol': tr.get('personal_rol') or '',
                    'via': 'uniforme' if tr.get('personal_nombre')
                    else 'rostro',
                }))
                continue
            ficha = None
            if ident is not None:
                try:
                    ficha = ident.ficha_de(pid, reidentifier)
                except Exception:                           # noqa: BLE001
                    ficha = None
            if ficha:
                personal.append((pid, tid, box, ficha))
            else:
                clientes.append((pid, tid, box))
        return clientes, personal

    def _atender_personal(self, personal: List, t: float,
                          medir_areas: bool) -> None:
        """Alertas operativas del personal, con cooldown por persona.

        - seguridad: «está en el pasillo» al verlo (no depende de áreas,
          así que funciona incluso con la escena en estabilización);
        - pasillero: «surte el área N» cuando su torso toca un área un
          dwell sostenido (solo con áreas fiables: `medir_areas`).
        """
        franja = float(self.cfg.get('franja_superior', 0.60))
        dwell_surte = float(self.cfg.get('surte_dwell_s', 3.0))
        cd_seg = float(self.cfg.get('alerta_seguridad_cooldown_s', 300.0))
        cd_surte = float(self.cfg.get('alerta_surte_cooldown_s', 180.0))
        tocadas: set = set()

        for pid, tid, box, ficha in personal:
            rol = str(ficha.get('rol') or '')
            nombre = str(ficha.get('nombre') or 'empleado')
            if rol == 'seguridad':
                clave = ('seg', pid)
                if (t - self._alerta_personal_ts.get(clave, -1e12)) >= cd_seg:
                    self._alerta_personal_ts[clave] = t
                    self._evento_personal('seguridad', nombre, rol, None, t)
                    self._alertas_personal.append({
                        'tipo': 'seguridad', 'nombre': nombre, 'rol': rol,
                        'pid': pid, 'box': box, 'area': None})
                continue

            if not (medir_areas and self._poligonos):
                continue
            for idx, poli in enumerate(self._poligonos):
                if not self._toca_area(box, idx, poli, franja):
                    continue
                clave_dwell = (idx, pid)
                tocadas.add(clave_dwell)
                desde = self._personal_desde.setdefault(clave_dwell, t)
                clave_cd = (f'surte:{idx}', pid)
                if ((t - desde) >= dwell_surte
                        and (t - self._alerta_personal_ts.get(
                            clave_cd, -1e12)) >= cd_surte):
                    self._alerta_personal_ts[clave_cd] = t
                    area = self._nombre(idx)
                    self._evento_personal('surte', nombre, rol, area, t)
                    self._alertas_personal.append({
                        'tipo': 'surte', 'nombre': nombre, 'rol': rol,
                        'pid': pid, 'box': box, 'area': area})

        for clave in list(self._personal_desde):
            if clave not in tocadas:
                self._personal_desde.pop(clave, None)

    def _evento_personal(self, tipo: str, nombre: str, rol: str,
                         area: Optional[str], t: float) -> None:
        self._feed.append({
            'tipo': tipo,
            'nombre': area or '',
            # `area` también como campo propio: la clave de dedup del panel
            # lo incluye — sin él, dos «surte» de áreas distintas en el
            # mismo segundo se tragaban uno.
            'area': area or '',
            'ts': _hora(t),
            'genero': None,
            'rango_etario': None,
            'id_corto': nombre,        # el feed muestra AL EMPLEADO, no un id
            'rol': rol,
        })
        self._feed_sucio = True

    def drenar_alertas_personal(self) -> List[Dict[str, Any]]:
        """Las alertas operativas pendientes (la clase Anaqueles las
        convierte en alertas del pipeline con foto). Vacía la lista."""
        pendientes, self._alertas_personal = self._alertas_personal, []
        return pendientes

    # ── Interno: medición ───────────────────────────────────────────────
    @staticmethod
    def _personas_de(tracks: Dict[Any, Dict]
                     ) -> List[Tuple[str, Any, Tuple[float, float, float, float]]]:
        """`(pid, tid, box)` de los tracks de persona con caja válida.

        Los `prov:` CUENTAN: frente a un anaquel la gente suele estar de
        espaldas a la cámara y el rostro nunca confirma (misma decisión que
        permanencia en puerta/cruce). El cooldown por persona+área es lo que
        evita el doble conteo si la oclusión parte el track.
        """
        salida = []
        for tid, tr in (tracks or {}).items():
            if tr.get('class', -1) != -1:       # -1 = persona en este pipeline
                continue
            box = tr.get('box')
            if box is None:
                continue
            try:
                b = tuple(float(v) for v in box[:4])
            except (TypeError, ValueError):
                continue
            pid = str(tr.get('persistent_id') or f'prov:{tid}')
            salida.append((pid, tid, b))
        return salida

    # ── Movimiento (v3): quieto vs caminando ────────────────────────────
    def _actualizar_movimiento(self, tid: Any, box: Tuple, t: float) -> None:
        """Alimenta la trayectoria del track y su reloj de quietud.

        v3: el punto seguido es el centro del TORSO (mitad superior del
        bbox) — el truncamiento de piernas por un carrito u oclusión ya no
        «mueve» a la persona; y la quietud tiene histéresis: 1-2 frames
        malos PAUSAN el reloj en vez de borrarlo (un outlier del tracker
        reseteaba el dwell a cero eternamente).
        """
        cx = (float(box[0]) + float(box[2])) / 2.0
        cy_sup = float(box[1]) + (float(box[3]) - float(box[1])) * 0.3
        ancho = max(1.0, float(box[2]) - float(box[0]))
        tray = self._trayectoria.get(tid)
        if tray is None:
            tray = self._trayectoria[tid] = deque(maxlen=8)
        tray.append((t, cx, cy_sup, ancho))
        self._tid_visto[tid] = t
        if self._es_quieto(tid, t):
            self._no_quieto_racha[tid] = 0
            self._quieto_desde.setdefault(tid, t)
        else:
            racha = self._no_quieto_racha.get(tid, 0) + 1
            self._no_quieto_racha[tid] = racha
            if racha > 2:
                self._quieto_desde.pop(tid, None)

    def _es_quieto(self, tid: Any, t: float) -> bool:
        """¿El CUERPO está quieto? v3 mide DOS velocidades, en anchos de su
        propio bbox por segundo:

        - PLANAR: la LONGITUD DEL CAMINO del torso (no la distancia entre
          extremos — la deriva lenta oscilante daba cero);
        - RADIAL: el cambio del ANCHO del bbox (medianas de las mitades de
          la ventana). Quien camina HACIA la cámara apenas desplaza su
          centro en píxeles, pero su bbox crece — antes pasaba por quieto
          y renacían visitas y agarres en el eje principal del pasillo
          (lo cazó la revisión).

        Sin datos suficientes responde False: el error barato es no contar.
        """
        tray = self._trayectoria.get(tid)
        if not tray or len(tray) < 2:
            return False
        ventana = float(self.cfg.get('quieto_ventana_s', 1.0))
        recorte = [p for p in tray if (t - p[0]) <= ventana]
        if len(recorte) < 2:
            recorte = list(tray)[-2:]
        lapso = max(0.2, recorte[-1][0] - recorte[0][0])
        if lapso < 0.35:
            return False                 # aún no hay ventana que mida nada
        anchos = [p[3] for p in recorte]
        ancho_medio = max(1.0, sum(anchos) / len(anchos))
        camino = sum(
            ((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5
            for a, b in zip(recorte, recorte[1:]))
        vel_planar = camino / lapso / ancho_medio
        mitad = max(1, len(anchos) // 2)
        w_ini = float(np.median(anchos[:mitad]))
        w_fin = float(np.median(anchos[mitad:]))
        vel_radial = abs(w_fin - w_ini) / lapso / ancho_medio
        return (vel_planar + vel_radial) <= float(
            self.cfg.get('quieto_max_vel', 0.45))

    def _quieto_efectivo(self, tid: Any) -> bool:
        """Quieto CON la histéresis: el reloj sigue vivo aunque este frame
        haya sido un outlier (racha <= 2)."""
        return tid in self._quieto_desde

    def _quieto_dur(self, tid: Any, t: float) -> float:
        desde = self._quieto_desde.get(tid)
        return (t - desde) if desde is not None else 0.0

    def _dedup_espacial(self, registro: Dict[int, deque], idx: int,
                        x: float, y: float, ancho: float, t: float,
                        ventana_s: float, tid: Any = None,
                        vivos: Optional[set] = None) -> bool:
        """True si en este SITIO ya se contó hace poco (anti-fragmentos:
        un track partido recibe un pid nuevo, pero no se teletransporta).

        v3: una entrada cuyo tid COEXISTE vivo en este frame junto al
        candidato NO es un fragmento — es OTRA persona (el acompañante
        hombro con hombro cuenta su propia visita)."""
        cola = registro.get(idx)
        if cola is None:
            cola = registro[idx] = deque(maxlen=40)
        radio_frac = float(self.cfg.get('dedup_espacial_frac', 0.8))
        while cola and (t - cola[0][0]) > ventana_s:
            cola.popleft()
        for (ts, px, py, pw, ptid) in cola:
            if (vivos is not None and ptid is not None and ptid in vivos
                    and ptid != tid):
                continue                 # dos cuerpos a la vez: no es fragmento
            radio = radio_frac * max(ancho, pw)
            if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 <= radio:
                return True
        return False

    def _refrescar_reciente(self, registro: Dict[int, deque], idx: int,
                            x: float, y: float, ancho: float, t: float,
                            tid: Any, vivos: Optional[set] = None) -> None:
        """Mantiene «ocupado» el sitio de una estancia viva: la entrada del
        dedup solo envejece cuando el sitio queda libre (sin esto, una
        estancia de >2 min con el track partido re-contaba).

        v3.1 (lo cazó la verificación de punta a punta): se refresca LA
        ENTRADA PROPIA (mismo tid) y, si no existe, se ADOPTA solo una
        entrada MUERTA cercana (mi track partido). Por radio a secas, el
        acompañante adoptaba la entrada del comprador de al lado, le
        cambiaba el dueño y el fragmento del comprador quedaba exento del
        dedup — re-contaba.
        """
        cola = registro.get(idx)
        if not cola:
            return
        radio_frac = float(self.cfg.get('dedup_espacial_frac', 0.8))
        muerta = None
        for i, (ts, px, py, pw, ptid) in enumerate(cola):
            if ptid == tid:
                cola[i] = (t, px, py, pw, tid)     # mi entrada: refrescar
                return
            if muerta is None and (vivos is None or ptid not in vivos):
                radio = radio_frac * max(ancho, pw)
                if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 <= radio:
                    muerta = i
        if muerta is not None:
            ts, px, py, pw, _vieja = cola[muerta]
            cola[muerta] = (t, px, py, pw, tid)    # adoptar mi fragmento

    def _anotar_descarte(self, motivo: str) -> None:
        self._descartes[motivo] = self._descartes.get(motivo, 0) + 1
        if _DIAG and motivo.startswith('agarre'):
            logger.info("anaqueles-diag[%s] DESCARTE %s (acumulado %d)",
                        self.camera_id, motivo, self._descartes[motivo])

    def _medir(self, frame: np.ndarray, personas: List,
               demografia: Any, t: float, dt: float) -> Dict[int, int]:
        cooldown_visita = float(self.cfg.get('visita_cooldown_s', 120.0))
        dwell_min = float(self.cfg.get('visita_dwell_s', 1.5))
        franja = float(self.cfg.get('franja_superior', 0.60))
        solape_min = float(self.cfg.get('visita_solape_min', 0.15))

        ventana_dedup = float(self.cfg.get('dedup_espacial_ventana_s', 20.0))
        en_compuerta: set = set()
        personas_por_area: Dict[int, int] = {}
        candidatos_pose: Dict[str, Tuple[Any, Tuple, List[int]]] = {}
        cercania = bool(self.cfg.get('pose_cercania', False))
        #: (área, pid) de quien está CERCA este frame (solo con cercania).
        cerca: set = set()
        vivos = {tid for _p, tid, _b in personas}

        for pid, tid, box in personas:
            self._actualizar_movimiento(tid, box, t)
            quieto = self._quieto_efectivo(tid)
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            ancho = max(1.0, float(box[2]) - float(box[0]))
            areas_de_pid: List[int] = []
            for idx, poli in enumerate(self._poligonos):
                solape, area_franja = self._solape_franja(box, idx, poli,
                                                         franja)
                # Compuerta: la franja debe PISAR el área de verdad. v3: el
                # umbral se acota por el MENOR de (franja, polígono) — un
                # nivel angosto frente a un bbox gigante era imposible.
                base = min(area_franja, self._areas_poligono[idx]
                           if idx < len(self._areas_poligono)
                           else area_franja)
                if solape < max(1.0, solape_min * base):
                    if solape > 1.0:
                        self._anotar_descarte('visita_solape')
                    continue
                areas_de_pid.append(idx)
                clave = (idx, pid)
                en_compuerta.add(clave)
                personas_por_area[idx] = personas_por_area.get(idx, 0) + 1
                self._compuerta_desde.setdefault(clave, t)
                # El dwell SOLO corre con el cuerpo quieto: caminar junto
                # al estante (lateral O hacia la cámara) es tránsito.
                if not quieto:
                    self._quieto_en_area.pop(clave, None)
                    self._anotar_descarte('visita_movimiento')
                    continue
                if clave in self._visita_abierta:
                    # Estancia viva: el cooldown y el sitio del dedup se
                    # refrescan — cuentan desde la ÚLTIMA vez vista, no
                    # desde el conteo (una estancia de >2 min con un
                    # parpadeo del track re-contaba).
                    self._visita_ts[clave] = t
                    self._refrescar_reciente(self._visitas_recientes, idx,
                                             cx, cy, ancho, t, tid,
                                             vivos=vivos)
                    continue
                desde = self._quieto_en_area.setdefault(clave, t)
                if ((t - desde) < dwell_min
                        or (t - self._visita_ts.get(clave, -1e12))
                        < cooldown_visita):
                    continue
                # Anti-fragmentos: si en ESTE sitio ya se contó hace poco,
                # es la misma persona con el track partido — salvo que el
                # otro track siga VIVO (dos cuerpos = dos visitas).
                if self._dedup_espacial(self._visitas_recientes, idx,
                                        cx, cy, ancho, t, ventana_dedup,
                                        tid=tid, vivos=vivos):
                    self._visita_abierta.add(clave)   # no reintentar la estancia
                    self._refrescar_reciente(self._visitas_recientes, idx,
                                             cx, cy, ancho, t, tid,
                                             vivos=vivos)
                    self._anotar_descarte('visita_dedup_espacial')
                    continue
                self._visita_abierta.add(clave)
                self._visita_ts[clave] = t
                self._visitas_recientes.setdefault(
                    idx, deque(maxlen=40)).append((t, cx, cy, ancho, tid))
                self._contar_visita(idx, pid, tid, demografia, t)
            # Pose SOLO para quien está QUIETO en compuerta: correrla sobre
            # los que caminan gasta GPU y su braceo no puede contar. Con
            # `pose_cercania` (ver _DEFECTOS), también para quien está cerca
            # del área aunque se mueva o no pise la compuerta: contar sigue
            # exigiendo la quietud.
            if cercania:
                areas_pose = list(areas_de_pid) + [
                    i for i in range(len(self._poligonos))
                    if i not in areas_de_pid and self._cerca_de_area(box, i)]
                if areas_pose:
                    candidatos_pose[pid] = (tid, box, areas_pose)
                    cerca.update((i, pid) for i in areas_pose)
            elif areas_de_pid and quieto:
                candidatos_pose[pid] = (tid, box, areas_de_pid)

        # Cierre de estancias CON GRACIA (v3): un solo miss del detector
        # borraba el gesto armado y el dwell — ahora la ausencia debe
        # sostenerse.
        for clave in list(self._compuerta_desde):
            if clave in en_compuerta:
                self._fuera_compuerta.pop(clave, None)
                continue
            ausente = self._fuera_compuerta.setdefault(clave, t)
            if (t - ausente) < 1.2:
                continue                 # gracia: el estado sobrevive
            self._fuera_compuerta.pop(clave, None)
            self._compuerta_desde.pop(clave, None)
            self._quieto_en_area.pop(clave, None)
            self._visita_abierta.discard(clave)
            # Con `pose_cercania`, bajar el brazo (compuerta cerrada) no
            # borra el gesto a medias mientras la persona siga cerca.
            if clave not in cerca:
                self._agarre_estado.pop(clave, None)

        # Poda de trayectorias con la misma gracia (tid no visto > 2 s).
        for tid in list(self._trayectoria):
            if (t - self._tid_visto.get(tid, -1e12)) > 2.0:
                self._trayectoria.pop(tid, None)
                self._quieto_desde.pop(tid, None)
                self._no_quieto_racha.pop(tid, None)
                self._tid_visto.pop(tid, None)

        self._medir_agarres(frame, candidatos_pose, demografia, t, dt,
                            vivos=vivos)

        # Poda de cooldowns vencidos: los pid `prov:<tid>` de ByteTrack (con
        # offset por cámara) son infinitos y estos dicts crecerían sin tope
        # en una jornada larga. Una entrada vencida es inerte por definición
        # (la lectura con default -1e12 decide igual sin ella).
        for claves, ventana in ((self._visita_ts, cooldown_visita),
                                (self._agarre_ts,
                                 float(self.cfg.get('agarre_cooldown_s',
                                                    10.0)))):
            for clave, ts in list(claves.items()):
                if (t - ts) >= ventana:
                    claves.pop(clave, None)

        return personas_por_area

    def _solape_franja(self, box: Tuple[float, float, float, float],
                       idx: int, poli: np.ndarray, franja: float
                       ) -> Tuple[float, float]:
        """(solape_px², área_franja_px²) de la franja SUPERIOR del bbox con
        el polígono del área.

        Intersección REAL rectángulo-polígono (recorte de Sutherland-Hodgman
        y área del resultado), no un muestreo de puntos: la primera versión
        sondeaba 9 puntos en 3 filas y un nivel de anaquel de ~100 px que
        cayera ENTRE dos filas de sondas jamás contaba nada. Devolver el
        ÁREA (y no un booleano) permite exigir una fracción mínima de
        solape (v2): el roce de quien va por el medio del pasillo no abre
        compuerta. A la altura de un estante lo que importa es el torso y
        los brazos, no dónde pisa la persona; por eso la franja superior.
        """
        x1, y1, x2, y2 = box
        ym = y1 + (y2 - y1) * franja
        area_franja = max(1.0, (x2 - x1) * (ym - y1))
        ax1, ay1, ax2, ay2 = self._cajas_area[idx]
        # Descarte rápido por cajas: sin solape no recortamos.
        if x2 < ax1 or x1 > ax2 or ym < ay1 or y1 > ay2:
            return 0.0, area_franja
        return _solape_rect_poligono((x1, y1, x2, ym), poli), area_franja

    def _cerca_de_area(self, box: Tuple[float, float, float, float],
                       idx: int) -> bool:
        """¿La persona está debajo o delante del área, al alcance del brazo?

        Su caja solapa el área en horizontal, no queda entera por encima, y
        su borde de arriba no está más abajo del área que `cercania_frac_alto`
        de su propio alto (una balda alta queda por encima de la cabeza: la
        compuerta no se abre hasta que el brazo sube, y esto sí).
        """
        if idx >= len(self._cajas_area):
            return False
        ax1, ay1, ax2, ay2 = self._cajas_area[idx]
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        frac = float(self.cfg.get('cercania_frac_alto', 0.5))
        return (x2 > ax1 and x1 < ax2 and y2 > ay1
                and y1 <= ay2 + frac * (y2 - y1))

    def _toca_area(self, box: Tuple[float, float, float, float], idx: int,
                   poli: np.ndarray, franja: float) -> bool:
        """Versión booleana (la usa la vía de PERSONAL para el «surte»)."""
        solape, area_franja = self._solape_franja(box, idx, poli, franja)
        solape_min = float(self.cfg.get('visita_solape_min', 0.15))
        base = min(area_franja, self._areas_poligono[idx]
                   if idx < len(self._areas_poligono) else area_franja)
        return solape >= max(1.0, solape_min * base)

    def _profundidad_efectiva(self, ancho_bbox: float, idx: int) -> float:
        """La penetración exigida a la muñeca ESCALA con la persona (v3):
        `frac` de su ancho de bbox, entre 4 px y el tope configurado, y
        nunca más del 40% del alto del nivel dibujado (un área angosta no
        puede quedar matemáticamente imposible de contar)."""
        frac = float(self.cfg.get('agarre_profundidad_frac', 0.08))
        tope = float(self.cfg.get('agarre_profundidad_px', 12.0))
        prof = min(max(4.0, frac * ancho_bbox), tope)
        if idx < len(self._cajas_area):
            alto = self._cajas_area[idx][3] - self._cajas_area[idx][1]
            prof = min(prof, max(2.0, 0.4 * alto))
        return prof

    def _medir_agarres(self, frame: np.ndarray,
                       candidatos: Dict[str, Tuple[Any, Tuple, List[int]]],
                       demografia: Any, t: float, dt: float,
                       vivos: Optional[set] = None) -> None:
        """Un agarre es un GESTO VERIFICADO (v3), no una presencia.

        La máquina corre POR MUÑECA (izq/der): sin identidad, cuando la
        pose perdía la muñeca interior unos frames, la OTRA mano «armaba»
        el gesto y regalaba la amplitud — un agarre falso cada cooldown
        para cualquiera parado (lo cazó la revisión). Además: el conteo
        queda PENDIENTE un frame para arbitrar áreas apiladas (un reach que
        atraviesa el nivel de abajo camino al de arriba solo cuenta donde
        la mano SE ASIENTA), la amplitud se mide en el MARCO DEL CUERPO
        (la deriva lenta no la fabrica), el armado caduca a los 4 s sin
        evidencia fresca de «fuera», y la profundidad escala con la
        persona.
        """
        maximo = int(self.cfg.get('pose_max_personas', 4))
        # En imágenes, pero nunca menos que sus segundos a los fps medidos
        # (5-oct-2026): a 15 fps, 2 imágenes eran 0,13 s de mano dentro.
        fps = self._ritmo.fps
        frames_min = ritmo_imagenes.imagenes_para(
            float(self.cfg.get('agarre_min_s', 0.0)),
            int(self.cfg.get('agarre_frames_min', 2)), fps)
        cooldown = float(self.cfg.get('agarre_cooldown_s', 10.0))
        rearme = ritmo_imagenes.imagenes_para(
            float(self.cfg.get('agarre_rearme_s', 0.0)),
            int(self.cfg.get('agarre_rearme_frames', 2)), fps)
        asentamiento = float(self.cfg.get('agarre_asentamiento_s', 1.2))
        amplitud_frac = float(self.cfg.get('agarre_amplitud_min', 0.30))
        desde_extremo = (self.cfg.get('agarre_amplitud_desde', 'ultimo')
                         == 'extremo')

        # Los bbox más grandes primero: son los más cercanos al anaquel y a
        # la cámara, donde la pose es fiable. El resto espera su frame.
        orden = sorted(
            candidatos.items(),
            key=lambda kv: -((kv[1][1][2] - kv[1][1][0])
                             * (kv[1][1][3] - kv[1][1][1])))[:maximo]

        # Todas las personas en UNA llamada al modelo, con los recortes
        # preparados en la GPU (6-oct-2026, `pose_gpu.py`): con 2-3 personas
        # junto al estante el agarre bajaba a ~7 fps. Los dobles de prueba
        # sin `munecas_lote` siguen persona a persona.
        lote = getattr(self.pose, 'munecas_lote', None)
        cajas_orden = [box for _pid, (_tid, box, _a) in orden]
        todas = (lote(frame, cajas_orden) if lote is not None and orden
                 else [self.pose.munecas(frame, b) for b in cajas_orden])

        for (pid, (tid, box, areas_de_pid)), munecas in zip(orden, todas):
            if _DIAG:
                logger.info(
                    "anaqueles-diag[%s] t=%.2f pid=%s tid=%s areas=%s "
                    "quieto=%.1fs munecas=%s", self.camera_id, t,
                    _corto(pid), tid, areas_de_pid, self._quieto_dur(tid, t),
                    [(m[3] if len(m) > 3 else '?', round(float(m[0])),
                      round(float(m[1])), round(float(m[2]), 2))
                     for m in munecas])
            ancho_bbox = max(1.0, float(box[2]) - float(box[0]))
            cx = (float(box[0]) + float(box[2])) / 2.0
            cy = (float(box[1]) + float(box[3])) / 2.0
            if not munecas:
                # Pose perdida: estado CONGELADO, pero con TTL — un estado
                # de hace minutos ya no describe nada (fuera_pos rancio).
                for clave in [c for c in self._agarre_estado
                              if c[1] == pid]:
                    ent = self._agarre_estado[clave]
                    if (t - ent.get('visto_ts', t)) > 2.5:
                        ent['lados'] = {}
                        self._anotar_descarte('agarre_pose_perdida')
                continue
            # lado: 'izq'/'der' de la pose v3; '?' si un doble de prueba
            # viejo devuelve tuplas de 3 (una sola máquina, como antes).
            vistos: Dict[str, Tuple[float, float]] = {}
            munecas = _una_mano_si_coinciden(munecas, ancho_bbox)
            for m in munecas:
                lado = m[3] if len(m) > 3 else '?'
                vistos[lado] = (float(m[0]), float(m[1]))
            quieto = self._quieto_efectivo(tid)
            asentado = self._quieto_dur(tid, t) >= asentamiento

            # Profundidad de CADA muñeca en CADA área (se usa dos veces:
            # arbitraje de pendientes y máquinas).
            prof_de: Dict[Tuple[int, str], float] = {}
            for idx in areas_de_pid:
                poli = self._poligonos[idx]
                for lado, (mx, my) in vistos.items():
                    prof_de[(idx, lado)] = cv2.pointPolygonTest(
                        poli, (mx, my), True)

            # Para el recuadro del operador: la muneca con su profundidad
            # respecto al area (positiva = dentro). Se toma AQUI porque es
            # donde ya esta calculada: pedir la pose otra vez para dibujar
            # duplicaria el coste del modelo.
            for lado, (mx, my) in vistos.items():
                mejor = max(
                    ((idx, prof_de.get((idx, lado), -1e9))
                     for idx in areas_de_pid),
                    key=lambda par: par[1], default=(None, None))
                self._munecas_frame.append({
                    'x': round(float(mx), 1), 'y': round(float(my), 1),
                    'lado': lado,
                    'area': mejor[0],
                    'prof': (round(float(mejor[1]), 1)
                             if mejor[1] is not None else None),
                    'dentro': bool(
                        mejor[1] is not None
                        and mejor[0] is not None
                        and mejor[1] >= self._profundidad_efectiva(
                            ancho_bbox, mejor[0])),
                })

            # 1) Arbitraje de PENDIENTES de este pid (áreas apiladas): el
            #    conteo del frame anterior se confirma salvo que la mano
            #    haya seguido de largo hacia OTRO nivel.
            for lado in list(vistos):
                pend = self._agarre_pendiente.pop((pid, lado), None)
                if pend is None:
                    continue
                idx_p = pend['idx']
                prof_ef_p = self._profundidad_efectiva(ancho_bbox, idx_p)
                sigue_aqui = prof_de.get((idx_p, lado), -1e9) >= \
                    prof_ef_p * 0.5
                en_otra = any(
                    j != idx_p and prof_de.get((j, lado), -1e9)
                    >= self._profundidad_efectiva(ancho_bbox, j)
                    for j in areas_de_pid)
                if not sigue_aqui and en_otra:
                    self._anotar_descarte('agarre_transito')
                    continue                  # era tránsito entre niveles
                self._confirmar_agarre(pend, demografia)

            # 2) Máquinas por (área, muñeca).
            for idx in areas_de_pid:
                clave = (idx, pid)
                ent = self._agarre_estado.setdefault(
                    clave, {'lados': {}, 'visto_ts': t})
                ent['visto_ts'] = t
                prof_ef = self._profundidad_efectiva(ancho_bbox, idx)
                for lado, (mx, my) in vistos.items():
                    prof = prof_de[(idx, lado)]
                    est = ent['lados'].setdefault(lado, {
                        'armado': False, 'dentro': 0, 'fuera': 0,
                        'fuera_pos': None, 'fuera_centro': None,
                        'fuera_ts': 0.0})
                    if _DIAG:
                        # Estado ANTES de este frame: con esto se ve si la
                        # muñeca llegó a armarse, y cuál de las dos.
                        logger.info(
                            "anaqueles-diag[%s]   area=%d lado=%s prof=%.1f "
                            "(exige %.1f, fuera si <%.1f) armado=%s "
                            "dentro=%d fuera=%d", self.camera_id, idx, lado,
                            prof, prof_ef, prof_ef * 0.5, est['armado'],
                            est['dentro'], est['fuera'])
                    if prof >= prof_ef:
                        est['dentro'] += 1
                        est['fuera'] = 0
                        # Calor SOLO con la mano dentro Y el cuerpo quieto.
                        if quieto:
                            self._calor(idx, mx, my, max(dt, 0.3))
                        # El armado caduca sin evidencia fresca de «fuera».
                        if (est['armado']
                                and (t - est['fuera_ts']) > 4.0):
                            est['armado'] = False
                            self._anotar_descarte('agarre_armado_caducado')
                        if not (est['armado']
                                and est['dentro'] >= frames_min):
                            if 0 < prof < prof_ef:
                                self._anotar_descarte('agarre_superficial')
                            continue
                        if not (quieto and asentado):
                            self._anotar_descarte('agarre_movimiento')
                            continue
                        # Amplitud en el MARCO DEL CUERPO: la muñeca debe
                        # moverse respecto al TORSO — la deriva del cuerpo
                        # entero no fabrica gesto.
                        origen = est.get('fuera_pos')
                        centro0 = est.get('fuera_centro')
                        if origen is not None and centro0 is not None:
                            viaje = (
                                ((mx - cx) - (origen[0] - centro0[0])) ** 2
                                + ((my - cy)
                                   - (origen[1] - centro0[1])) ** 2) ** 0.5
                            if _DIAG:
                                # TODAS las evaluaciones, pasen o no: para
                                # ver cuánto sobra o falta, y cuánto del
                                # «viaje» se come la caja al estirar el
                                # brazo (se ensancha y su centro se mueve
                                # hacia la mano).
                                fb = est.get('fuera_box') or tuple(box[:4])
                                logger.info(
                                    "anaqueles-diag[%s]   AMPLITUD ok=%s "
                                    "viaje=%.0f bruto=%.0f ancho=%.0f "
                                    "ancho_fuera=%.0f alto=%.0f "
                                    "alto_fuera=%.0f exige=%.0f "
                                    "dcx=%.0f dcy=%.0f", self.camera_id,
                                    viaje >= amplitud_frac * ancho_bbox,
                                    viaje, ((mx - origen[0]) ** 2
                                            + (my - origen[1]) ** 2) ** 0.5,
                                    ancho_bbox, fb[2] - fb[0],
                                    float(box[3]) - float(box[1]),
                                    fb[3] - fb[1],
                                    amplitud_frac * ancho_bbox,
                                    cx - centro0[0], cy - centro0[1])
                            if viaje < amplitud_frac * ancho_bbox:
                                self._anotar_descarte('agarre_amplitud')
                                continue
                        if est.get('fuera_arriba'):
                            # La mano partió de DENTRO de un área de arriba:
                            # es la vuelta de ese agarre, no uno aquí. Hay
                            # que salir de verdad y volver a entrar.
                            est['armado'] = False
                            est['dentro'] = 0
                            self._anotar_descarte('agarre_bajada')
                            continue
                        if (t - self._agarre_ts.get(clave, -1e12)) \
                                < cooldown:
                            continue
                        # Anti-fragmentos (sin castigar dos veces: el
                        # descarte NO hereda cooldown).
                        if self._dedup_espacial(self._agarres_recientes,
                                                idx, mx, my, ancho_bbox,
                                                t, cooldown, tid=tid,
                                                vivos=vivos):
                            est['armado'] = False
                            est['dentro'] = 0
                            self._anotar_descarte('agarre_dedup_espacial')
                            continue
                        est['armado'] = False   # exigir salir y volver
                        est['dentro'] = 0
                        # PENDIENTE: se confirma al frame siguiente salvo
                        # que la mano siga de largo a otro nivel apilado.
                        previo = self._agarre_pendiente.get((pid, lado))
                        if previo is not None and previo['t'] == t \
                                and previo['prof'] >= prof:
                            continue      # ya hay uno más profundo este frame
                        self._agarre_pendiente[(pid, lado)] = {
                            'idx': idx, 'clave': clave, 'pid': pid,
                            'tid': tid, 'x': mx, 'y': my, 't': t,
                            'ancho': ancho_bbox, 'prof': prof}
                    elif prof < prof_ef * 0.5:
                        # Fuera o superficial (histéresis): SOLO esta misma
                        # muñeca puede armarse con esta evidencia.
                        racha_nueva = est['fuera'] == 0
                        est['fuera'] += 1
                        est['dentro'] = 0
                        # 'extremo': el origen del viaje es el punto MÁS
                        # fuera de esta racha, no la última muestra (que
                        # puede ir rozando el borde).
                        if (not desde_extremo or racha_nueva
                                or prof <= est.get('fuera_prof', 1e9)):
                            est['fuera_pos'] = (mx, my)
                            est['fuera_centro'] = (cx, cy)
                            est['fuera_prof'] = prof
                            est['fuera_box'] = tuple(float(v)
                                                     for v in box[:4])
                            est['fuera_arriba'] = self._en_area_de_arriba(
                                idx, lado, prof_de, ancho_bbox)
                        est['fuera_ts'] = t
                        if est['fuera'] >= rearme:
                            est['armado'] = True
                    # Zona gris: la muñeca parpadeando en el borde no arma
                    # ni desarma nada.

        # 3) Pendientes huérfanos (la pose se perdió justo después del
        #    gesto — p. ej. la mano quedó tras el producto): a los 2 s se
        #    confirman con sus datos originales.
        for llave, pend in list(self._agarre_pendiente.items()):
            if (t - pend['t']) > 2.0:
                del self._agarre_pendiente[llave]
                self._confirmar_agarre(pend, demografia)

    def _en_area_de_arriba(self, idx: int, lado: str,
                           prof_de: Dict[Tuple[int, str], float],
                           ancho_bbox: float) -> bool:
        """¿Está la muñeca `lado` DENTRO de un área que queda por ENCIMA de
        `idx` (centro vertical de su caja más arriba en la imagen)?

        Para `idx` esa mano está «fuera», y el gesto de BAJAR de ahí a la
        cintura parecía un agarre en `idx` (6-oct-2026, vídeo de la cámara
        3: la clienta coge de la balda alta y deja las manos delante de la
        de abajo; el usuario revisó el vídeo y no cogió nada abajo). Subir
        no cambia: el origen de un agarre de arriba es la cintura, que cae
        en el área de ABAJO, no en una de encima."""
        if idx >= len(self._cajas_area):
            return False
        cy = (self._cajas_area[idx][1] + self._cajas_area[idx][3]) / 2.0
        for (j, l), prof in prof_de.items():
            if l != lado or j == idx or j >= len(self._cajas_area):
                continue
            caja = self._cajas_area[j]
            if (caja[1] + caja[3]) / 2.0 < cy and \
                    prof >= self._profundidad_efectiva(ancho_bbox, j):
                return True
        return False

    def _confirmar_agarre(self, pend: Dict[str, Any],
                          demografia: Any) -> None:
        """Consuma un agarre arbitrado (re-chequeo de dedup y conteo).

        El dedup se RE-CHEQUEA aquí (v3.1, lo cazó la verificación de punta
        a punta): dos gestos que completaban en el MISMO frame y el mismo
        punto pasaban ambos, porque al chequear el primero aún no estaba
        confirmado. Estricto a propósito (sin exención por vivos): dos
        muñecas en el mismo punto en <10 s es una oclusión/confusión de la
        pose, no dos agarres — el error caro es inflar.
        """
        clave = pend['clave']
        idx = pend['idx']
        cooldown = float(self.cfg.get('agarre_cooldown_s', 10.0))
        if self._dedup_espacial(self._agarres_recientes, idx,
                                pend['x'], pend['y'], pend['ancho'],
                                pend['t'], cooldown, tid=pend['tid'],
                                vivos=None):
            self._anotar_descarte('agarre_dedup_espacial')
            return
        self._agarre_ts[clave] = pend['t']
        self._agarres_recientes.setdefault(idx, deque(maxlen=40)).append(
            (pend['t'], pend['x'], pend['y'], pend['ancho'], pend['tid']))
        self._contar_agarre(idx, pend['pid'], pend['tid'], demografia,
                            pend['t'], x=pend['x'], y=pend['y'])

    def _guardar_refs(self, frame: np.ndarray,
                      personas_por_area: Dict[int, int],
                      clientes: List, personal: List, t: float) -> None:
        """Foto de referencia de cada área SIN NADIE delante, cada 10 min.

        El recorte es el bbox EXACTO del polígono (sin margen): así la
        rejilla de calor calza 1:1 sobre la foto al componer. v3: la
        exclusión es por INTERSECCIÓN DE BBOX (cliente o personal) — la
        compuerta no bastaba: un cliente cruzando la mitad baja de la caja
        sin abrir compuerta salía «quemado» en la base. Nunca lanza.
        """
        try:
            cuerpos = ([b for _p, _tid, b in clientes]
                       + [b for _p, _tid, b, _f in personal])
            for idx, caja in enumerate(self._cajas_area):
                if personas_por_area.get(idx, 0):
                    continue                  # hay un cliente en el área
                if (t - self._ref_ts.get(idx, -1e12)) < 600.0:
                    continue
                x1, y1, x2, y2 = [int(v) for v in caja]
                # NADIE cruzando la caja del área, ni de refilón.
                if any(not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2)
                       for (bx1, by1, bx2, by2) in cuerpos):
                    continue
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 40 or y2 - y1 < 24:
                    continue
                self._ref_ts[idx] = t
                self.registro.guardar_ref(idx, frame[y1:y2, x1:x2])
        except Exception as exc:                            # noqa: BLE001
            logger.debug("anaqueles: refs fallaron: %s", exc)

    def _vigilar_stock(self, frame: np.ndarray, clientes: List,
                       personal: List, t: float) -> None:
        """«Posible agotado» (5-oct-2026, `anaqueles_stock.py`). Tapan el
        producto tanto los clientes como el personal: los dos cuerpos."""
        if not self.cfg.get('stock_activo', True):
            return
        cuerpos =([b for _p, _tid, b in clientes]
                   + [b for _p, _tid, b, _f in personal])
        nombres = {i: self._nombre(i) for i in range(len(self._poligonos))}
        self._stock.vigilar(frame, self._productos, nombres, cuerpos, t)
        sucesos = self._stock.drenar_sucesos()
        if sucesos:
            self._feed.extend(sucesos)
            self._feed_sucio = True

    def _calor(self, idx: int, x: float, y: float, peso: float) -> None:
        ax1, ay1, ax2, ay2 = self._cajas_area[idx]
        ancho = max(1.0, ax2 - ax1)
        alto = max(1.0, ay2 - ay1)
        filas, cols = self.registro.grid_dim
        col = int((x - ax1) / ancho * cols)
        fila = int((y - ay1) / alto * filas)
        self.registro.calor(idx, fila, col, peso)

    def _contar_visita(self, idx: int, pid: str, tid: Any,
                       demografia: Any, t: float) -> None:
        total = self.registro.sumar(idx, 'visitas', self._nombre(idx))
        genero, rango = self._rasgos(demografia, tid)
        self.registro.sumar_segmento(idx, genero, rango)
        self._evento('visita', idx, pid, genero, rango, t)
        logger.info("anaqueles[%s] visita #%d a %s (%s)",
                    self.camera_id, total, self._nombre(idx), _corto(pid))

    def _contar_agarre(self, idx: int, pid: str, tid: Any,
                       demografia: Any, t: float,
                       x: Optional[float] = None,
                       y: Optional[float] = None) -> None:
        total = self.registro.sumar(idx, 'agarres', self._nombre(idx))
        genero, rango = self._rasgos(demografia, tid)
        producto = self._repartir_a_producto(idx, x, y)
        self._evento('agarre', idx, pid, genero, rango, t, producto=producto)
        # TODOS los productos del área, no solo `producto`: la muñeca no es
        # la mano y el asignado cambia de una pasada a otra (hallazgo 25).
        if self.cfg.get('stock_activo', True):
            self._stock.tras_agarre(idx, t, persona=self._recorte_de(tid))
        logger.info("anaqueles[%s] AGARRE #%d en %s%s (%s)",
                    self.camera_id, total, self._nombre(idx),
                    f' → {producto}' if producto else '', _corto(pid))

    def _recorte_de(self, tid: Any) -> Optional[np.ndarray]:
        """Recorte (copia) de la persona `tid` en el frame actual, o None."""
        try:
            frame = getattr(self, '_frame_actual', None)
            box = (getattr(self, '_cajas_tid', None) or {}).get(tid)
            if frame is None or box is None:
                return None
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
            if x2 - x1 < 8 or y2 - y1 < 8:
                return None
            return frame[y1:y2, x1:x2].copy()
        except Exception:                                   # noqa: BLE001
            return None

    def drenar_alertas_stock(self) -> List[Dict[str, Any]]:
        """Alertas «posible agotado» con su foto (las convierte en alertas
        del pipeline la clase Anaqueles, como las del personal)."""
        return self._stock.drenar_alertas()

    def _repartir_a_producto(self, idx: int, x: Optional[float],
                             y: Optional[float]) -> Optional[str]:
        """Suma el agarre al producto bajo la muñeca. Devuelve su nombre.

        Sin productos en el área no se toca nada: el conteo por área es
        exactamente el de antes del 25-sep. Con productos, lo que no cae en
        ninguno va a `agarres_sin_producto`, que es la señal de que están
        mal dibujados. Nunca lanza: repartir es un extra y no puede costar
        el agarre, que ya está contado arriba.
        """
        productos = self._productos.get(idx)
        if not productos or x is None or y is None:
            return None
        try:
            encontrado = anaqueles_productos.producto_en(productos, x, y)
            if encontrado is None:
                self.registro.sumar_sin_producto(idx)
                return None
            posicion, nombre = encontrado
            self.registro.sumar_producto(idx, posicion, nombre)
            return nombre
        except Exception as exc:                            # noqa: BLE001
            logger.error("anaqueles[%s]: no se pudo repartir el agarre de "
                         "%s entre sus productos: %s", self.camera_id,
                         self._nombre(idx), exc)
            return None

    @staticmethod
    def _rasgos(demografia: Any, tid: Any
                ) -> Tuple[Optional[str], Optional[str]]:
        """Género/edad del cache demográfico del pipeline, si los tiene ya."""
        try:
            if demografia is None:
                return None, None
            demo = demografia.get_cached(tid)
            if not demo:
                return None, None
            return demo.get('gender'), demo.get('age_range')
        except Exception:                                   # noqa: BLE001
            return None, None

    def _evento(self, tipo: str, idx: int, pid: str,
                genero: Optional[str], rango: Optional[str],
                t: float, producto: Optional[str] = None) -> None:
        evento = {
            'tipo': tipo,
            'area': idx,
            'nombre': self._nombre(idx),
            'ts': _hora(t),
            'genero': genero,
            'rango_etario': rango,
            'id_corto': _corto(pid),
        }
        # Solo cuando hay producto: el panel lee el feed con .get y la clave
        # de dedup no la incluye, así que un cliente viejo no se entera.
        if producto:
            evento['producto'] = producto
        self._feed.append(evento)
        self._feed_sucio = True

    def _limpiar_efimero(self) -> None:
        """Tras un cambio de escena: nada de lo «en curso» sigue siendo real."""
        self._compuerta_desde.clear()
        self._quieto_en_area.clear()
        self._visita_abierta.clear()
        self._agarre_estado.clear()
        self._agarre_pendiente.clear()
        self._trayectoria.clear()
        self._quieto_desde.clear()
        self._no_quieto_racha.clear()
        self._fuera_compuerta.clear()
        # Las posiciones contadas eran de la escena vieja: en la nueva no
        # deben bloquear nada.
        self._visitas_recientes.clear()
        self._agarres_recientes.clear()

    def _vigilar_silencio(self, t: float, hay_personas: bool) -> None:
        if hay_personas:
            self._ultima_persona_ts = t
            self._sin_personas_s = None
            return
        sin = t - self._ultima_persona_ts
        umbral = float(self.cfg.get('aviso_sin_personas_s', 900.0))
        self._sin_personas_s = sin if sin >= umbral else None

    # ── Bloque en vivo (patrón stand_emisor: throttle + absolutos) ──────
    def _toca(self, bloque: str, t: float) -> bool:
        clave = f'periodo_{bloque}_s'
        anterior = self._ultimo_emitido.get(bloque)
        if anterior is None or (t - anterior) >= float(
                self.cfg.get(clave, 2.0)):
            self._ultimo_emitido[bloque] = t
            return True
        return False

    def _bloque(self, t: float,
                personas_por_area: Dict[int, int]) -> Optional[Dict[str, Any]]:
        bloque: Dict[str, Any] = {}

        if self._toca('contadores', t):
            resumen = self.registro.resumen()
            for fila in resumen:
                fila['personas_ahora'] = int(
                    personas_por_area.get(fila['idx'], 0))
            bloque['areas'] = resumen
            bloque['sin_areas'] = not self._poligonos

        # Las munecas van SIN throttle y solo cuando las hay: es un dibujo
        # en vivo. El panel de anaqueles ignora las claves que no conoce
        # (`actualizar` lo lee todo con guardas), asi que un cliente viejo no
        # se entera de que existen.
        if self._munecas_frame:
            bloque['munecas'] = list(self._munecas_frame)

        if self._feed_sucio:
            bloque['eventos'] = list(self._feed)
            self._feed_sucio = False

        # Stock: al cambiar algo y, además, con los contadores (un panel que
        # se conecta tarde se pone al día).
        if self._stock.cambio or 'areas' in bloque:
            bloque['stock'] = self._stock.bloque()
            self._stock.cambio = False

        if self._aviso_escena is not None:
            bloque['escena'] = self._aviso_escena
            self._aviso_escena = None

        if ((self.personal_tids or self._personal_emitido)
                and self._toca('contadores_personal', t)):
            # Quién del personal está a la vista AHORA (el panel lo muestra
            # aparte de los clientes; comparte el ritmo de contadores). La
            # lista VACÍA también viaja — una sola vez, al salir el último
            # empleado — para que el panel limpie «En cámara ahora».
            bloque['personal'] = sorted(
                ({'nombre': f.get('nombre'), 'rol': f.get('rol')}
                 for f in self.personal_tids.values()),
                key=lambda x: (x['rol'] or '', x['nombre'] or ''))
            self._personal_emitido = bool(self.personal_tids)

        if self._toca('salud', t):
            bloque['salud'] = {
                'areas_definidas': len(self._poligonos),
                'pose_disponible': self.pose.disponible,
                'suspendido': t < self._suspendido_hasta,
                'sin_personas_s': (round(self._sin_personas_s)
                                   if self._sin_personas_s else None),
                # La versión de la lógica de conteo que ESTÁ corriendo: si
                # los falsos positivos «siguen», lo primero es mirar esto
                # (un servidor sin reiniciar sigue con la lógica vieja).
                'logica': 'gesto-v3-munecas',
                # Por qué NO se contó (acumulado): la transparencia para
                # calibrar umbrales sin adivinar.
                'descartes': dict(self._descartes),
            }

        if self._poligonos and self._toca('heatmap', t):
            mapas = self.registro.heatmaps()
            for i, caja in enumerate(self._cajas_area):
                if str(i) in mapas:
                    mapas[str(i)]['bbox'] = [round(v, 1) for v in caja]
            bloque['heatmap'] = mapas

        if bloque:
            bloque['camera'] = self.camera_id
        return bloque or None

    def cerrar(self) -> None:
        """Volcado final a disco. La JORNADA no se cierra: cada día tiene su
        archivo y una reconexión no debe partir el día (lección del stand)."""
        try:
            self.registro.volcar(force=True)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("anaqueles: volcado de cierre falló: %s", exc)


# ── Pipeline ─────────────────────────────────────────────────────────────
class Anaqueles(TiendasProcess):
    """`TiendasProcess` + medición de interacción con los anaqueles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._medidor: Optional[MedidorAnaqueles] = None
        self._medidor_camara: Any = None
        logger.info("Anaqueles: pipeline de interacción con estantes activo")

    # ── Bucle ───────────────────────────────────────────────────────────
    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False,
                      camera_id: Any = 1, heatmap_activate: bool = False
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """El frame de `TiendasProcess` + el bloque `anaqueles` del metadata."""
        imagen, metadata = super().process_frame(
            image, roi=roi, activate_roi=activate_roi, camera_id=camera_id,
            heatmap_activate=heatmap_activate)

        try:
            bloque = self._medir_anaqueles(image, camera_id)
            if bloque:
                metadata['anaqueles'] = bloque
            if self._medidor is not None:
                self._alertar_personal(image, camera_id)
                self._alertar_stock(camera_id)
                self._etiquetar_personal(metadata)
        except Exception as exc:                            # noqa: BLE001
            # Nunca propaga: si la medición falla, el cliente debe seguir
            # viendo vídeo y detecciones.
            logger.error("Anaqueles: fallo midiendo: %s", exc, exc_info=True)

        return imagen, metadata

    def _alertar_personal(self, image: np.ndarray, camera_id: Any) -> None:
        """Convierte las alertas operativas del medidor en alertas del
        pipeline (panel lateral + WhatsApp/Jarvis si están activos).

        Se encolan DESPUÉS del super(): viajan en el frame siguiente, que a
        2-3 FPS son ~400 ms — irrelevante para «seguridad en el pasillo».
        """
        camara = (getattr(self, '_camera_display_name', None)
                  or str(camera_id))
        for a in self._medidor.drenar_alertas_personal():
            if a['tipo'] == 'surte':
                titulo = 'Empleado surte anaquel'
                descripcion = (f"Empleado {a['nombre']} surte "
                               f"{a['area']} ({camara}).")
            else:
                titulo = 'Seguridad en pasillo'
                descripcion = (f"Seguridad {a['nombre']} presente en "
                               f"{camara}.")
            foto = _recorte_b64(image, a.get('box'))
            alerta = {
                'event_type': titulo,
                'class_name': titulo,
                'description': descripcion,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'camera_name': camara,
                'person_uuid': str(a.get('pid') or ''),
                # WhatsApp DESCARTA alertas sin imagen: el recorte de la
                # persona es la evidencia.
                'crop_image': foto,
                'image_base64': foto,
                'clase_gruesa': 'persona',
                # Sin la persona+área en la clave, dos empleados surtiendo a
                # la vez se tragarían una alerta (dedup de Jarvis/WhatsApp).
                'dedup_clave': (f"personal|{camara}|{a['tipo']}|"
                                f"{a.get('area') or '-'}|{a.get('pid')}"),
                'rol_personal': a.get('rol'),
            }
            if not hasattr(self, '_alertas_pendientes'):
                self._alertas_pendientes = []
            self._alertas_pendientes.append(alerta)
            if len(self._alertas_pendientes) > 200:
                self._alertas_pendientes = self._alertas_pendientes[-100:]
            logger.info("Anaqueles[%s] %s", camara, descripcion)

    def _alertar_stock(self, camera_id: Any) -> None:
        """«Posible agotado» al panel de alertas del cliente (5-oct-2026),
        por la misma vía que las del personal. La foto es el ANTES y el AHORA
        del producto (el usuario la prefirió a la de la persona), y
        `screenshot_path` = la evidencia antes | ahora | quién agarró, que la
        tarjeta abre al pulsarla. El «lleva X» en vivo lo pinta el panel de
        Anaqueles: la tarjeta es fija."""
        camara = (getattr(self, '_camera_display_name', None)
                  or str(camera_id))
        for a in self._medidor.drenar_alertas_stock():
            agarre = (f" Último agarre: {a['hora_agarre']}."
                      if a.get('hora_agarre') else '')
            alerta = {
                'event_type': 'Posible agotado',
                'class_name': str(a['producto']),
                'description': (f"{a['producto']} en {a['area_nombre']} "
                                f"({camara}): detectado a las {a['hora']} "
                                f"(cambio {a['cambio']:.2f}).{agarre}"),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'camera_name': camara,
                'crop_image': a['imagen_b64'],
                'image_base64': a['imagen_b64'],
                'screenshot_path': a['ruta'],
                'clase_gruesa': 'stock',
                'dedup_clave': (f"stock|{camara}|{a['area']}|"
                                f"{a['producto_idx']}|{a['hora']}"),
            }
            if not hasattr(self, '_alertas_pendientes'):
                self._alertas_pendientes = []
            self._alertas_pendientes.append(alerta)
            if len(self._alertas_pendientes) > 200:
                self._alertas_pendientes = self._alertas_pendientes[-100:]

    def _etiquetar_personal(self, metadata: Dict[str, Any]) -> None:
        """Marca las cajas del personal en el modo directo.

        Sin esto el operador ve a un empleado sin contar y cree que el modo
        está roto; con la etiqueta se ve POR QUÉ no suma. Campos extra +
        label: el overlay del cliente lee con .get y no se rompe.
        """
        tids = self._medidor.personal_tids
        if not tids:
            return
        for d in (metadata.get('detections') or []):
            ficha = tids.get(d.get('tracker_id'))
            if not ficha:
                continue
            rol = str(ficha.get('rol') or '')
            d['personal'] = True
            d['rol'] = rol
            d['label'] = (('SEGURIDAD · ' if rol == 'seguridad'
                           else 'PERSONAL · ')
                          + str(ficha.get('nombre') or ''))

    def _medir_anaqueles(self, image: np.ndarray,
                         camera_id: Any) -> Optional[Dict[str, Any]]:
        if self._medidor is None or self._medidor_camara != camera_id:
            cfg = cargar_config(str(camera_id))
            self._medidor = MedidorAnaqueles(
                camera_id, cfg, device=getattr(self, 'device', None))
            self._medidor_camara = camera_id

        zonas = getattr(self, '_anaqueles_zonas', None) or []
        nombres = getattr(self, '_anaqueles_nombres', None) or []
        productos = getattr(self, '_anaqueles_productos', None) or []
        if not bool(getattr(self, '_anaqueles_activate', True)):
            zonas, nombres, productos = [], [], []
        self._medidor.poner_zonas(zonas, nombres)
        # Después de las zonas: un producto solo vale si su área existe.
        self._medidor.poner_productos(productos)

        tracks = self._tracks_de(camera_id)
        ahora = time.time()
        # Grabación de diagnóstico (28-sep-2026), apagada salvo con
        # ELDE_ANAQUELES_GRABAR: lo MISMO que recibe el medidor, con el mismo
        # reloj, para poder reproducirlo fuera de línea.
        grabador = anaqueles_grabador.grabador()
        if grabador is not None:
            grabador.grabar(camera_id, image, tracks, zonas, nombres,
                            productos, ahora)

        return self._medidor.procesar(
            image, tracks,
            self._demografia_de(camera_id),
            ahora=ahora,
            # El reidentificador NO participa del swap por cámara: aquí
            # sigue siendo consultable, y es de donde salen los embeddings
            # para reconocer al personal enrolado.
            reidentifier=getattr(self, '_reidentifier', None))

    def _tracks_de(self, camera_id: Any) -> Dict[Any, Any]:
        """Los tracks de ESTA cámara, no los que quedan tras `_pop_state`.

        Mismo patrón (y mismo motivo) que `AmazonasStand._tracks_de`: el
        estado real vive en `camera_states[camera_id]`; `self.active_tracks`
        es un diccionario vacío que no es de nadie. Se lee `camera_states` a
        pelo — `_get_cam_state` crea y MIGRA estado, que no es lo que quiere
        quien solo consulta.
        """
        estado = (getattr(self, 'camera_states', None) or {}).get(camera_id)
        if isinstance(estado, dict):
            tracks = estado.get('active_tracks')
            if isinstance(tracks, dict):
                return tracks
        return getattr(self, 'active_tracks', {}) or {}

    def _demografia_de(self, camera_id: Any) -> Any:
        """El clasificador demográfico de ESTA cámara (no el del `__init__`)."""
        estado = (getattr(self, 'camera_states', None) or {}).get(camera_id)
        if isinstance(estado, dict):
            demo = estado.get('_demographics')
            if demo is not None:
                return demo
        return getattr(self, '_demographics', None)

    # ── Cierre ──────────────────────────────────────────────────────────
    def cerrar_anaqueles(self) -> None:
        """Volcado final. Lo llama `_cerrar_procesadores` en la desconexión;
        el archivo del día queda listo para reanudarse en la siguiente."""
        if self._medidor is not None:
            self._medidor.cerrar()


# ── Utilidades ───────────────────────────────────────────────────────────
def _solape_rect_poligono(rect: Tuple[float, float, float, float],
                          poli: np.ndarray) -> float:
    """Área (px²) de la intersección entre un rectángulo y un polígono.

    Recorte de Sutherland-Hodgman del polígono contra los 4 semiplanos del
    rectángulo + área por la fórmula del cordón (shoelace). Sin dependencias
    y exacto: es lo que hace honesta a la compuerta de visita — un muestreo
    de puntos deja niveles enteros sin contar según dónde caigan las sondas.
    """
    rx1, ry1, rx2, ry2 = rect
    pts: List[Tuple[float, float]] = [(float(p[0]), float(p[1]))
                                      for p in poli]

    def _recortar(puntos, dentro, cruce):
        salida = []
        n = len(puntos)
        for i in range(n):
            a, b = puntos[i], puntos[(i + 1) % n]
            da, db = dentro(a), dentro(b)
            if da:
                salida.append(a)
            if da != db:
                salida.append(cruce(a, b))
        return salida

    def _cruce_x(x0):
        def _c(a, b):
            t = (x0 - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
            return (x0, a[1] + t * (b[1] - a[1]))
        return _c

    def _cruce_y(y0):
        def _c(a, b):
            t = (y0 - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
            return (a[0] + t * (b[0] - a[0]), y0)
        return _c

    for dentro, cruce in (
            (lambda p: p[0] >= rx1, _cruce_x(rx1)),
            (lambda p: p[0] <= rx2, _cruce_x(rx2)),
            (lambda p: p[1] >= ry1, _cruce_y(ry1)),
            (lambda p: p[1] <= ry2, _cruce_y(ry2))):
        pts = _recortar(pts, dentro, cruce)
        if len(pts) < 3:
            return 0.0

    area = 0.0
    n = len(pts)
    for i in range(n):
        (xa, ya), (xb, yb) = pts[i], pts[(i + 1) % n]
        area += xa * yb - xb * ya
    return abs(area) / 2.0


def _recorte_b64(frame: np.ndarray, box: Any) -> str:
    """Recorte JPEG en base64 de la persona (evidencia de la alerta)."""
    try:
        import base64
        if frame is None or box is None:
            return ''
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return ''
        ok, buf = cv2.imencode('.jpg', frame[y1:y2, x1:x2],
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''
    except Exception:                                       # noqa: BLE001
        return ''


def _hora(ts: float) -> str:
    return time.strftime('%H:%M:%S', time.localtime(ts))


def _corto(pid: str) -> str:
    pid = str(pid)
    return pid[-4:] if len(pid) > 4 else pid


#: Dos muñecas a menos de esto (fracción del ancho de la persona, nunca
#: menos de 8 px) son la MISMA mano. Ver `_una_mano_si_coinciden`.
_MISMA_MANO_FRAC = 0.05


def _una_mano_si_coinciden(munecas: List, ancho_bbox: float) -> List:
    """Si la izquierda y la derecha caen en el mismo punto, se queda solo
    la de más confianza (6-oct-2026).

    Cuando ViTPose ve UNA sola mano, a veces le pone las DOS etiquetas: en
    el vídeo de la cámara 3 (11:07:16 del 6-oct, grabado y reproducido) la
    muñeca izquierda colgaba al costado (y ≈ 800) y, de una imagen a la
    siguiente, apareció encima de la derecha (a 1-3 px, confianza 0,30-0,63
    frente a 0,9). Para la máquina del gesto eso era una mano que subía
    443 px y entraba en el área de abajo: un agarre falso (el usuario
    revisó el vídeo: la clienta solo cogió de la balda de arriba). Dos
    manos juntas cogiendo algo siguen dando el agarre: cuenta la buena.
    """
    if len(munecas) != 2:
        return munecas
    a, b = munecas
    if len(a) < 4 or len(b) < 4 or a[3] == b[3]:
        return munecas
    umbral = max(8.0, _MISMA_MANO_FRAC * float(ancho_bbox))
    if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 >= umbral:
        return munecas
    return [a if a[2] >= b[2] else b]


def crear(**kwargs: Any) -> Anaqueles:
    """Fábrica, para que `app.py` no importe la clase directamente."""
    return Anaqueles(**kwargs)
