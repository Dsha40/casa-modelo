"""
src/gui/components/render_box/render_box.py
RenderBox original + soporte de drag & drop DVR (RTSP) con detección automática de Hik-Connect e IP.

Cambios vs original:
  • dragEnterEvent / dropEvent aceptan también CAMERA_MIME
  • start_dvr_stream() lanza RTSPWorker con detección automática de tipo (Hik-Connect/IP)
  • _stop_dvr_stream() detiene el RTSPWorker
  • Botón "⏹ DVR" en la barra flotante cuando hay stream activo
  • Compatible con datos cifrados de Hik-Connect y URLs de IP
"""
import os, json, base64, sys
import re
import time
import uuid
import msgpack

# cv2/numpy/Supervision para el overlay de detecciones (modo directo). Con
# guardas: si faltan, el cliente cae al dibujo con QPainter sin romperse.
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None
try:
    from .sv_overlay import SupervisionOverlay
except Exception:
    SupervisionOverlay = None

from PySide6.QtWidgets import (
    QFrame, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QGridLayout, QSizePolicy, QMenu, QWidgetAction, QCheckBox,
    QLineEdit, QMessageBox, QToolButton, QDialog, QApplication,
)
from PySide6.QtCore  import (Qt, Slot, QProcess, QUrl, Signal, QEvent, QBuffer,
                             QIODevice, QThread, QPoint, QTimer)
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtGui   import (QPixmap, QCursor, QImage, QPainter, QPen, QColor,
                             QFont)

from ..custon_label.interactive_imageLabel import Interactive_imageLabel
from ..custon_btn.btn_footer import BtnIco

from core.state_global.hwnd import hwndState
from core.capture_exaple import capture_window_by_hwnd, pil_image_to_png_bytes, window_exists, get_title
from core.window_controller import set_window_always_on_top
from core.dvr.hikconnect_channel_encoder import ChannelTypeDetector

# DVR
from workers.rtsp_worker import RTSPWorker

# Identidad estable de la camara (H-11). Primera rebanada del despiece de
# render_box: la logica vive en el nucleo y la comparten los cuatro clientes.
from elde_core.ui import identidad_camara as _identidad
# Captura compartida: start_dvr_stream y loop_show_result eran el mismo
# codigo en los cuatro clientes (0,97-1,00 de similitud entre tres de
# ellos; amazonas arrastraba una copia anterior).
from elde_core.ui.render_box_captura import CapturaDVRMixin
from elde_core.ui.render_box_puerta import PuertaDireccionalMixin
from elde_core.ui import overlay_munecas as _munecas
from elde_core.ui import etiqueta_seguimiento as _etiqueta_id
from elde_core.ui import telemetria_ciclo as _telemetria
from elde_core.ui import jpeg_a_medida as _jpeg_a_medida
from elde_core.ui.selector_local_camara import (
    SelectorLocalCamara, SIN_LOCAL_USA_EL_PIE)
_DVR_MIME = "application/x-dvr-channel"


class _EventWorker(QThread):
    """POSTea un frame al endpoint /vlm/events del servidor en un hilo aparte
    (la inferencia VLM tarda ~20-30s, no debe congelar la UI)."""
    done = Signal(dict)

    def __init__(self, url, jpeg_bytes, data=None, parent=None):
        super().__init__(parent)
        self._url = url
        self._jpeg = jpeg_bytes
        self._data = data or {}

    def run(self):
        try:
            import requests
            files = {"image": ("frame.jpg", self._jpeg, "image/jpeg")}
            r = requests.post(self._url, files=files, data=self._data,
                              timeout=180)
            self.done.emit(r.json())
        except Exception as e:
            self.done.emit({"status": "error", "message": str(e)})


class _ModelWorker(QThread):
    """POSTea la seleccion de modelo VLM (3b/7b) al servidor (form data)."""
    done = Signal(dict)

    def __init__(self, url, model, parent=None):
        super().__init__(parent)
        self._url = url
        self._model = model

    def run(self):
        try:
            import requests
            r = requests.post(self._url, data={"model": self._model}, timeout=60)
            self.done.emit(r.json())
        except Exception as e:
            self.done.emit({"status": "error", "message": str(e)})


class _PostWorker(QThread):
    """POST de form-data a un endpoint del servidor, en un hilo aparte."""
    done = Signal(dict)

    def __init__(self, url, data=None, parent=None):
        super().__init__(parent)
        self._url = url
        self._data = data or {}

    def run(self):
        try:
            import requests
            r = requests.post(self._url, data=self._data, timeout=60)
            self.done.emit(r.json())
        except Exception as e:
            self.done.emit({"status": "error", "message": str(e)})


class _GetWorker(QThread):
    """GET simple a un endpoint del servidor (para sincronizar estado)."""
    done = Signal(dict)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self._url = url

    def run(self):
        try:
            import requests
            r = requests.get(self._url, timeout=10)
            self.done.emit(r.json())
        except Exception as e:
            self.done.emit({"status": "error", "message": str(e)})


class Render_box(CapturaDVRMixin, PuertaDireccionalMixin, QFrame):

    # La puerta direccional la cablea `PuertaDireccionalMixin`. Su bandera
    # `PUERTA_TIENE_MODO_QUE_LA_LEE` no se redeclara aqui: ya vale True por
    # defecto y repetirlo solo anade un sitio mas donde quedarse desfasado.
    # Lo que de verdad lo sostiene es que el pie de este cliente ofrezca
    # `PermanenciaArea` —el unico modo que lee el umbral—, y de eso responde
    # `test_contrato_puerta.py::test_el_pie_de_cada_cliente_ofrece_el_modo_que_la_lee`.

    double_clicked_signal = Signal(int, bool)
    roi_change_signal     = Signal(list)
    alert_received        = Signal(dict)
    # Bloque `metadata['anaqueles']` del servidor (modo Anaqueles): el panel
    # lateral de areas lo consume por camara.
    anaqueles_received    = Signal(dict)

    def __init__(self,
                 frames_per_milliseconds=100,
                 index=0,
                 hwnd=None,
                 inferece_play=False,
                 roi=[[100,100],[900,100],[900,900],[100,900]],
                 roi_boolean=False,
                 order_zone=None,
                 order_zone_boolean=False,
                 delivery_zone=None,
                 delivery_zone_boolean=False,
                 # Areas de anaquel (modo Anaqueles): lista de poligonos en
                 # espacio 0..1000 del visor, una por nivel del estante.
                 anaqueles=None,
                 anaqueles_boolean=False,
                 # Numeros GLOBALES de cada area (paralelos a `anaqueles`).
                 # El numero es la IDENTIDAD del anaquel entre camaras:
                 # camara 1 = anaqueles 1..5, camara 2 = 6..10, etc. Los
                 # asigna el contador del cliente (callbacks de abajo) y
                 # quitar un area NO renumera las demas.
                 anaqueles_numeros=None,
                 asignar_numero_anaquel=None,
                 liberar_numero_anaquel=None,
                 # Productos dibujados DENTRO de las areas (25-sep-2026):
                 # `ProductosAnaquel.a_lista()` tal como se guardo.
                 anaqueles_productos=None,
                 # Áreas a ignorar (12-oct-2026, `areas_ignorar.py`).
                 zonas_ignorar=None,
                 zonas_ignorar_boolean=False,
                 vlm_enabled_boolean=False,
                 heatmap_boolean=False,
                 trace_boolean=True,
                 ellipse_style=False,
                 vlm_context="",
                 track_classes=None,
                 callback_save_data=None,
                 socket_services=None,
                 api_jarvis=None,
                 # Establecimiento de ESTA camara: a donde van sus
                 # alertas de Jarvis. Vacio = el del selector del pie.
                 # Se recibe pero NO se restaura de la config: cada
                 # sesion empieza sin local (ver el modulo del selector).
                 establecimiento="",
                 # Puerta direccional: umbral + flecha doble ENTRADA/SALIDA.
                 # La lee el modo PermanenciaArea, que el pie de este cliente
                 # ofrece desde el 7-sep-2026. `None` = posicion por defecto
                 # del componente.
                 puerta_direccional=None,
                 puerta_boolean=False,
                 # Pintar las munecas que manda el modo Anaqueles. Nace
                 # APAGADO: es una ayuda de calibracion, no algo que deba
                 # aparecer solo en la pantalla del operador.
                 munecas_boolean=False,
                 # Telemetria del ciclo a disco (24-sep-2026). Diagnostico:
                 # nace APAGADA.
                 telemetria_ciclo_boolean=False,
                 **_legacy_kwargs,
                 ):
        super().__init__()

        self.setAcceptDrops(True)
        self.hwnd        = hwnd
        self.smart_mode  = inferece_play
        self.index       = index
        self.api_jarvis  = api_jarvis
        # Local de esta camara. Con camaras de establecimientos
        # distintos en la misma ventana y la MISMA inferencia, cada una
        # alerta al suyo; el selector del pie es uno solo y no puede.
        self.establecimiento = str(establecimiento or "").strip()
        self.socket      = socket_services
        self.socket.connected_signal.connect(self.reconnect_socket)
        self.socket.disconnected_signal.connect(self.diconect_socket)

        self.roi                     = roi
        self.roi_boolean             = roi_boolean
        # ROI azul de Toma de orden
        self.order_zone              = order_zone if order_zone else [[200,400],[500,400],[500,700],[200,700]]
        self.order_zone_boolean      = bool(order_zone_boolean)
        # ROI rojo de Entrega de plato
        self.delivery_zone           = delivery_zone if delivery_zone else [[550,400],[850,400],[850,700],[550,700]]
        self.delivery_zone_boolean   = bool(delivery_zone_boolean)
        # Puerta direccional (umbral + flecha doble ENTRADA/SALIDA). Todo su
        # cableado vive en `PuertaDireccionalMixin`, compartido por los
        # cuatro clientes. Nace apagada.
        self.init_puerta(puerta_direccional, puerta_boolean)

        # Areas de anaquel (naranjas, numeradas). Sin areas por defecto: las
        # anade el operador desde el menu ROI, una por nivel del estante.
        self.anaqueles               = list(anaqueles or [])
        self.anaqueles_boolean       = bool(anaqueles_boolean)
        self.anaqueles_numeros       = [int(n) for n in
                                        (anaqueles_numeros or [])]
        self._asignar_numero_anaquel = asignar_numero_anaquel
        self._liberar_numero_anaquel = liberar_numero_anaquel
        # Migracion de areas dibujadas ANTES de la numeracion global (14-ago):
        # se les asigna numero ahora, una sola vez, y se persiste.
        if len(self.anaqueles_numeros) < len(self.anaqueles):
            while len(self.anaqueles_numeros) < len(self.anaqueles):
                self.anaqueles_numeros.append(self._numero_anaquel_nuevo())
            if callback_save_data:
                # El __init__ corre antes de conectar señales: se persiste
                # directo por el callback (self._save_all aun no sirve aqui).
                callback_save_data(index, "anaqueles_numeros",
                                   self.anaqueles_numeros[:])
        # Nombres derivados de los numeros globales ("Anaquel 7"). Este
        # atributo es el que leen LAS DOS rutas de payload.
        self.anaqueles_nombres       = [f"Anaquel {n}"
                                        for n in self.anaqueles_numeros]
        # Productos dentro de las areas (25-sep-2026). Los edita el lienzo;
        # aqui solo se guardan para restaurarlos al abrir.
        self.anaqueles_productos     = list(anaqueles_productos or [])
        self.zonas_ignorar           = list(zonas_ignorar or [])
        self.zonas_ignorar_boolean   = bool(zonas_ignorar_boolean)
        # ETAPA 2 — toggle VLM
        self.vlm_enabled_boolean     = bool(vlm_enabled_boolean)
        self.callback_save_data      = callback_save_data

        self.process              = None
        self.stop                 = False
        self.frames_per_milliseconds = frames_per_milliseconds
        self.frame_count          = 0
        self.last_fps_time        = time.time()
        self.current_fps          = 0
        # FPS de IA: frames procesados por el servidor y mostrados (el FPS
        # real percibido en modo IA, distinto de la tasa de captura local).
        self._ai_fps_count        = 0
        self._ai_fps_t0           = time.time()
        self.is_maximized         = False
        self.image_w              = 0
        self.image_h              = 0
        self.current_pixmap       = None
        self.component_key        = str(uuid.uuid4())
        self.can_send_next_frame  = True
        # MODO DIRECTO: el servidor devuelve solo detecciones (no la imagen) y
        # el cliente las dibuja sobre el frame que envio -> mucha menos
        # latencia/ancho de banda. _pending_frame_bytes guarda el JPEG en vuelo.
        self._direct_mode         = True
        self._pending_frame_bytes = None
        # Munecas del modo Anaqueles (23-sep-2026): el ultimo juego recibido
        # y CUANDO llego. La hora no es opcional: son un dibujo en vivo y un
        # punto viejo clavado en pantalla miente mas que no pintar nada.
        self._munecas_ultimas = []
        self._munecas_ts = None
        self._munecas_boolean = bool(munecas_boolean)
        # «#<tracker_id>» en cada caja, para comprobar que ByteTrack sigue
        # bien a cada persona (6-oct-2026, `etiqueta_seguimiento.py`). De
        # diagnóstico: no se guarda, arranca apagado.
        self._ids_boolean = False
        # Marcas de tiempo por fotograma, a un JSONL (ver telemetria_ciclo).
        # El servidor midio un ciclo de 156 ms con ~99 de «espera» que desde
        # alli no se puede desglosar; esto lo mide desde este lado.
        self._telemetria = _telemetria.TelemetriaCiclo(
            activa=telemetria_ciclo_boolean)
        # Overlay de Supervision (uno por camara: estelas/heatmap son per-camara).
        # Si no esta disponible queda None y se usa el dibujo con QPainter.
        self._sv_overlay = None
        if SupervisionOverlay is not None and cv2 is not None and np is not None:
            try:
                _ov = SupervisionOverlay()
                if getattr(_ov, "available", False):
                    _ov.show_trace = bool(trace_boolean)
                    _ov.style = "ellipse" if ellipse_style else "round"
                    self._sv_overlay = _ov
            except Exception:
                self._sv_overlay = None
        # Unpacker PERSISTENTE para el stream msgpack del capture worker.
        # Imprescindible: un frame JPEG (>64KB) llega partido en varios
        # chunks del pipe; recrearlo por chunk desincronizaba el stream y
        # tiraba casi todos los frames (causa raiz del ~1 fps).
        self._unpacker            = None
        # Ángulo de cámara para demografía (Personal de Amazonas):
        # "auto" => el servidor lo detecta solo (frontal/lateral/cenital) por
        # la forma de las personas. También admite "frontal"|"lateral"|"cenital"
        # si se quiere fijar manualmente.
        self.camera_angle         = "auto"
        # Mapa de calor: si True, el servidor pinta el overlay de zonas
        # calientes sobre el video procesado. Se activa desde el menu
        # de clases ("Mapa de calor"). Se restaura de la config guardada.
        self.heatmap_boolean      = bool(heatmap_boolean)
        # Preferencias de visualizacion Supervision (persistidas)
        self.trace_boolean        = bool(trace_boolean)
        self.ellipse_style        = bool(ellipse_style)
        # Contexto VLM de esta camara (se envia con "Detectar evento")
        self.vlm_context          = vlm_context or ""
        # Nombre legible de la camara (DVR: alias+canal) para que el
        # dashboard no muestre el UUID.
        self.camera_name_dvr      = ""
        # Reenvio de alertas a WhatsApp: lo gobierna el
        # interruptor GLOBAL del pie (main.py lo propaga a
        # todos los recuadros). El envio lo hace el servidor.
        self.whatsapp_boolean     = False
        # Identidad ESTABLE del canal DVR (numero de serie del equipo + canal).
        # No cambia entre reinicios, al contrario que component_key: es lo que
        # permite acumular historico por camara. Ver H-11 y _device_id().
        self._dvr_device_serial   = ""
        self._dvr_channel_id      = ""

        # DVR
        self._rtsp_worker: RTSPWorker | None = None
        self._dvr_mode:    bool = False

        # ── Clases para tracking (COCO) ──
        # Mapa: nombre_display → class_id (int).
        self._available_classes = {
            "Persona":    0,
            "Auto":       2,
            "Moto":       3,
            "Bus":        5,
            "Camion":     7,
            "Perro":     16,
        }
        # Por defecto: solo Persona (sistema personas + demografia).
        # Se restaura de la config guardada (lista de class_id COCO).
        self._selected_classes = (
            list(track_classes) if track_classes else [0])

        self.setup_ui()

        if self.hwnd is not None and window_exists(self.hwnd):
            self.get_hwnd_and_print(self.hwnd)
            self.title      = get_title(self.hwnd)
            self.id_windows = int(hwnd)
            if self.socket.is_connected() and self.smart_mode:
                self.init_loop()

        hwndState.change_hwnd.connect(self.get_hwnd_and_print)
        if self.socket is not None:
            self.socket.signal_inference.connect(self.on_text_message_received)


    def setup_ui(self):
        self.setObjectName("box-content")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setContentsMargins(0, 0, 0, 0)

        self.stack = QVBoxLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)

        # Barra info
        self.bar_info = QWidget(self)
        self.bar_info.setAttribute(Qt.WA_StyledBackground, True)
        self.bar_info.setMaximumHeight(30)
        self.bar_info.setObjectName("bar_options")
        bar_info_layout = QHBoxLayout(self.bar_info)

        self.text_fps  = QLabel(f"FPS: {self.current_fps}")
        self.text_fps.setObjectName("text-fps")
        self.text_size = QLabel("0x0")
        self.text_size.setObjectName("text-fps")
        self._dvr_label = QLabel("")
        self._dvr_label.setObjectName("text-fps")
        self._dvr_label.setVisible(False)

        bar_info_layout.addWidget(self.text_fps)
        bar_info_layout.addWidget(self.text_size)
        bar_info_layout.addStretch()
        bar_info_layout.addWidget(self._dvr_label)

        # Imagen
        self.imagen_label = Interactive_imageLabel(
            "viewing window",
            roi=self.roi,
            roi_active=self.roi_boolean,
            order_zone=self.order_zone,
            order_zone_active=self.order_zone_boolean,
            delivery_zone=self.delivery_zone,
            delivery_zone_active=self.delivery_zone_boolean,
            anaqueles=self.anaqueles,
            anaqueles_activate=self.anaqueles_boolean,
            anaqueles_etiquetas=self.anaqueles_nombres,
            anaqueles_productos=self.anaqueles_productos,
            zonas_ignorar=self.zonas_ignorar,
            zonas_ignorar_activate=self.zonas_ignorar_boolean,
            **self.kwargs_puerta(),
        )
        self.imagen_label.point_change.connect(self.save_point)
        self.imagen_label.anaqueles_change.connect(self.save_anaqueles)
        self.imagen_label.productos_change.connect(self.save_productos)
        self.imagen_label.ignorar_change.connect(self.save_ignorar)
        self.conectar_puerta(self.imagen_label)
        # La regla mide en píxeles de la imagen que se manda al servidor
        # (`image_w`/`image_h` los actualiza cada frame), no en los de la
        # pantalla: así se compara tal cual con los umbrales del servidor.
        self.imagen_label.regla.tamano_frame = (
            lambda: (self.image_w, self.image_h))
        self.imagen_label.setAlignment(Qt.AlignCenter)
        self.imagen_label.installEventFilter(self)
        self.stack.addWidget(self.imagen_label)

        # Barra botones
        self.bar_options = QWidget(self)
        self.bar_options.setAttribute(Qt.WA_StyledBackground, True)
        self.bar_options.setMaximumHeight(30)
        self.bar_options.setObjectName("bar_options")
        bar_opt_layout = QHBoxLayout(self.bar_options)
        bar_opt_layout.setContentsMargins(10, 0, 10, 0)
        bar_opt_layout.setSpacing(5)

        # ── Toggle principal: ENCENDER / APAGAR el analisis IA de la camara ──
        # (rojo = IA activa). Se habilita al conectar el socket/DVR. Es el unico
        # boton suelto de la barra; el resto de opciones viven en los menus.
        self.btn_smart = BtnIco(ico_path="resource/mode_ai.png",
                                title="Encender / apagar analisis IA", h=30, w=30)
        self.btn_smart.setObjectName("btn-bar")
        self.btn_smart.clicked.connect(self.activate_modesmart)
        self.btn_smart.setCheckable(True)
        self.btn_smart.setDisabled(True)

        # ── Estado de la barra ──
        # Sin botones dispersos: ROI / Vista / Asistente son menus que leen y
        # conmutan estos flags (self.*), reconstruyendose al abrirse.
        self._vlm_model_key = "7b"   # modelo VLM activo del servidor (3b/7b)
        self._event_busy    = False  # evita doble disparo de "Detectar evento"
        self._event_worker  = None
        self._model_worker  = None
        self._sync_worker   = None
        self._ask_worker    = None


        # Al cerrar la app hay que parar los QThread ANTES de que Qt destruya
        # este widget: un QThread destruido mientras corre hace que Qt aborte
        # el proceso (qFatal -> 0xc0000409). aboutToQuit se emite con los
        # widgets todavia vivos, que es justo lo que necesitamos.
        _app = QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._stop_all_workers)
        self._eval_avisados = set()      # personas evaluando producto

        # ── Grupo Clases (selector de qué trackear) ──
        self._btn_classes = BtnIco(ico_path="resource/camera_box.png",
                                   title="Seleccionar clases a detectar", h=30, w=30)
        self._btn_classes.setObjectName("btn-bar")
        self._btn_classes.clicked.connect(self._show_class_menu)

        # ── Grupo Controles de captura ──
        # Guarda el frame ACTUAL (lo que se ve, con cajas/zonas) a disco.
        # Antes este boton no hacia nada; ahora guarda la captura del cliente.
        self.btn_cap = BtnIco(ico_path="resource/camera_box.png",
                              title="Guardar captura de esta camara", h=30, w=30)
        self.btn_cap.setObjectName("btn-bar")
        self.btn_cap.clicked.connect(self._save_capture)

        btn_play  = BtnIco(ico_path="resource/play_box.png",  title="Iniciar", h=30, w=30)
        btn_pause = BtnIco(ico_path="resource/pause_box.png", title="Pausar",  h=30, w=30)
        btn_stop  = BtnIco(ico_path="resource/stop_box.png",  title="Parar",   h=30, w=30)
        for b in (btn_play, btn_pause, btn_stop):
            b.setObjectName("btn-bar")

        btn_play.clicked.connect(self.init_loop)
        btn_pause.clicked.connect(self.pause_loop)
        btn_stop.clicked.connect(self.detroy_loop)

        # Botón detener DVR
        self._btn_stop_dvr = BtnIco(ico_path="resource/stop_box.png",
                                    title="Detener DVR", h=30, w=30)
        self._btn_stop_dvr.setObjectName("btn-bar")
        self._btn_stop_dvr.setToolTip("Detener stream DVR")
        self._btn_stop_dvr.clicked.connect(self._stop_dvr_stream)
        self._btn_stop_dvr.setVisible(False)

        # ── Separador visual (línea vertical) ──
        def _sep():
            s = QFrame()
            s.setFrameShape(QFrame.VLine)
            s.setStyleSheet("color: #666;")
            s.setFixedWidth(2)
            s.setFixedHeight(20)
            return s

        # ── Barra de TIENDA: area de conteo, vista, asistente IA (VLM) y
        # selector de clases.
        # El menu "Tienda" (definir zonas, calibrar anaqueles, panel de
        # analitica) se retiro en el HITO 5: el servidor dejo de exponer
        # /retail/* y de emitir metadata['retail'] en la simplificacion del
        # 27-jul, asi que abria dialogos que no llevaban a ninguna parte. El
        # codigo estuvo en cuarentena en _legacy/ hasta el HITO 10 (31-jul);
        # hoy vive solo en el historial (tag pre-hito10-legacy). ──
        self._menu_roi = self._make_menu_button(
            "Area ▾",
            "Area de conteo (ROI) de la camara: activar, mostrar puntos, "
            "reiniciar",
            self._build_roi_menu)
        self._menu_vista = self._make_menu_button(
            "Vista ▾", "Visualizacion: estelas de movimiento y estilo de caja",
            self._build_vista_menu)
        self._menu_ia = self._make_menu_button(
            "IA / VLM ▾",
            "Asistente IA (VLM): preguntar a la imagen, detectar evento, "
            "modelo (3B/7B), contexto",
            self._build_ia_menu)

        # Layout: [IA | Tienda | Area | Vista | IA/VLM | Clases] --- [Captura ▶ ⏸ ⏹ DVR]
        # Establecimiento POR CAMARA: la lista carga async tras el login
        # de Jarvis y el selector se repuebla solo al llegar.
        self.selector_local = SelectorLocalCamara(
            api_jarvis=self.api_jarvis,
            establecimiento=self.establecimiento,
            texto_vacio=SIN_LOCAL_USA_EL_PIE,
            indice_camara=self.index)
        self.selector_local.local_cambiado.connect(
            self._set_establecimiento)

        bar_opt_layout.addWidget(self.btn_smart)
        bar_opt_layout.addWidget(_sep())
        bar_opt_layout.addWidget(self._menu_roi)
        bar_opt_layout.addWidget(self._menu_vista)
        bar_opt_layout.addWidget(self._menu_ia)
        # Puerta direccional: boton SUELTO, no una entrada de menu — se
        # dibuja mirando el video y escondida tras dos clics no se encuentra.
        bar_opt_layout.addWidget(self.crear_boton_puerta())
        bar_opt_layout.addWidget(_sep())
        bar_opt_layout.addWidget(self._btn_classes)
        bar_opt_layout.addWidget(self.selector_local)
        bar_opt_layout.addStretch(1)
        bar_opt_layout.addWidget(self.btn_cap)
        bar_opt_layout.addWidget(btn_play)
        bar_opt_layout.addWidget(btn_pause)
        bar_opt_layout.addWidget(btn_stop)
        bar_opt_layout.addWidget(self._btn_stop_dvr)

        self.bar_info.hide()
        self.bar_options.hide()


    # ── DVR: stream RTSP ─────────────────────────────────────


    def _on_dvr_frame(self, img: QImage):
        if not self._dvr_mode:
            return

        pix = QPixmap.fromImage(img)
        self.current_pixmap = pix
        w, h = img.width(), img.height()
        self.image_w = w
        self.image_h = h

        # FPS tracking
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now
            prefix = "AI" if self.smart_mode else "DVR"
            self.text_fps.setText(f"{prefix} FPS: {self.current_fps}")

        # Si IA activa, enviar frame al servidor (solo mostrar respuesta del servidor)
        if self.smart_mode and self.socket is not None and self.socket.is_connected():
            if (not self.can_send_next_frame
                    and (time.time() - getattr(self, '_frame_en_vuelo_ts',
                                               0.0)) > 10.0):
                # Red de seguridad: si la respuesta se perdio (reconexion,
                # frame descartado por el servidor), el recuadro no puede
                # quedarse mudo para siempre esperando una que no llegara.
                self.can_send_next_frame = True
            if self.can_send_next_frame:
                tel = self._telemetria if self._telemetria.activa else None
                if tel is not None:
                    tel.abrir(self._device_id(), 'dvr', t_captura=now)
                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                img.save(buf, "JPEG", 80)
                jpeg_bytes = bytes(buf.data())
                buf.close()
                if tel is not None:
                    tel.marcar('t_jpeg')
                    tel.dato('bytes', len(jpeg_bytes))

                roi_c       = self.imagen_label.get_coordinates(w, h)
                order_c     = self.imagen_label.get_order_zone_coordinates(w, h)
                delivery_c  = self.imagen_label.get_delivery_zone_coordinates(w, h)

                data = {
                    "image": jpeg_bytes,
                    "roi_coordinates": roi_c,
                    "roi_activate": self.roi_boolean,
                    "order_zone_coordinates":    order_c,
                    "order_zone_activate":       self.order_zone_boolean,
                    "delivery_zone_coordinates": delivery_c,
                    "delivery_zone_activate":    self.delivery_zone_boolean,
                    "enable_vlm":                self.vlm_enabled_boolean,
                    "camera_id": self._device_id(),
                    "camera_angle": self.camera_angle,
                    "enviar_whatsapp": self.whatsapp_boolean,
                    # Local de ESTA camara: el servidor segmenta por el
                    # los tableros, el conteo de alertas y los mapas de
                    # calor. Viaja por frame porque una sola conexion
                    # lleva camaras de locales distintos.
                    "establecimiento": self.establecimiento,
                    "heatmap_activate": self.heatmap_boolean,
                    "camera_name": self._camera_display_name(),
                    "track_classes": self._selected_classes,
                    "draw_server": (not self._direct_mode),
                    # Areas de anaquel (modo Anaqueles): poligonos en pixeles
                    # del frame + nombres paralelos. Deben viajar por ESTA
                    # ruta y por la de captura de ventana (hook del mixin).
                    "anaqueles_zonas": self.imagen_label
                                           .get_anaqueles_coordinates(w, h),
                    "anaqueles_activate": self.anaqueles_boolean,
                    "anaqueles_nombres": self._nombres_anaqueles(),
                    # Productos dentro de esas areas (25-sep-2026). Tambien
                    # por la ruta de ventana (`render_box_captura`).
                    "anaqueles_productos": self.imagen_label
                                               .get_productos_coordinates(w, h),
                    # Areas a ignorar (12-oct-2026). Tambien por la ruta de
                    # ventana (`render_box_captura`).
                    "zonas_ignorar": self.imagen_label
                                         .get_ignorar_coordinates(w, h),
                    # Umbral de cruce dibujado con la puerta direccional:
                    # [[x1,y1],[x2,y2]] en pixeles; DENTRO —el lado de la
                    # cabeza verde ENTRADA— a la derecha de P1→P2. Viaja
                    # aunque el modo de este cliente no lo lea: el contrato
                    # lo declara, asi que mandarlo vacio es inofensivo y el
                    # dia que se habilite PermanenciaArea aqui ya esta.
                    **self.payload_puerta(w, h),
                }
                # Modo directo: guardar el frame enviado para dibujar encima
                # las detecciones que devuelva el servidor.
                self._pending_frame_bytes = jpeg_bytes
                self.socket.send_binary_frame(self.component_key, data)
                self.can_send_next_frame = False
                self._frame_en_vuelo_ts = time.time()
                if tel is not None:
                    tel.marcar('t_envio', self._frame_en_vuelo_ts)
            elif self._telemetria.activa:
                self._telemetria.descartados += 1
            # No mostrar frame crudo — el frame procesado llega del servidor
            # (modo clasico) o se dibuja local (modo directo)
        else:
            # Sin IA: mostrar frame crudo directamente (con el lienzo de
            # zonas encima si esta activo, para poder revisarlas sin IA).
            self.imagen_label.setPixmap(
                pix.scaled(
                    self.imagen_label.size(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation)
            )

    def _stop_dvr_stream(self):
        if self._rtsp_worker and self._rtsp_worker.isRunning():
            self._rtsp_worker.stop()
        self._rtsp_worker  = None
        self._dvr_mode     = False
        self._dvr_label.setVisible(False)
        self._btn_stop_dvr.setVisible(False)
        # El selector HD/SD solo tiene sentido con un stream DVR vivo.
        if getattr(self, '_btn_calidad_dvr', None) is not None:
            self._btn_calidad_dvr.setVisible(False)
        self.smart_mode    = False
        self.btn_smart.setChecked(False)
        self.btn_smart.setStyleSheet("background-color:#BFBFBF;")
        self.olvidar_captura_guardada()
        self.can_send_next_frame = True
        self.text_fps.setText("FPS: 0")

    @Slot()
    def _stop_all_workers(self):
        """Para TODOS los QThread de este RenderBox al cerrar la aplicacion.

        Si un QThread se destruye mientras sigue corriendo, Qt llama a qFatal y
        aborta el proceso entero (0xc0000409), sin traceback de Python: es lo
        que tumbaba al cliente al cerrarlo. No toca la UI a proposito, porque
        aqui la ventana ya se esta yendo."""
        # RTSPWorker tiene bucle infinito propio: su stop() lo saca del bucle
        # y espera. Es el que mas seguro estaba corriendo al cerrar.
        if self._rtsp_worker is not None:
            try:
                if self._rtsp_worker.isRunning():
                    self._rtsp_worker.stop()
            except RuntimeError:
                pass                     # ya destruido por Qt
            self._rtsp_worker = None

        # Los demas estan bloqueados dentro de requests (hasta 180s en el VLM)
        # y no se pueden interrumpir de forma limpia: se les da un margen corto
        # y, si no acaban, se cortan. Cortarlos al salir es preferible a que Qt
        # aborte el proceso.
        for name in ("_event_worker", "_model_worker", "_sync_worker",
                     "_ask_worker"):
            worker = getattr(self, name, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.quit()
                    if not worker.wait(2000):
                        worker.terminate()
                        worker.wait(1000)
            except RuntimeError:
                pass                     # ya destruido por Qt
            setattr(self, name, None)


    # ── Drag & Drop (ventana Windows + canal DVR) ─────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-boxcap"):
            event.acceptProposedAction()
        elif event.mimeData().hasFormat(_DVR_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        try:
            if event.mimeData().hasFormat(_DVR_MIME):
                raw  = bytes(event.mimeData().data(_DVR_MIME)).decode("utf-8")
                data = json.loads(raw)
                self.start_dvr_stream(data)
                event.acceptProposedAction()
                return

            if event.mimeData().hasFormat("application/x-boxcap"):
                raw = bytes(event.mimeData().data("application/x-boxcap")).decode("utf-8")
                other_hwnd, other_title = raw.split("|", 1)
                # Ignorar el drop solo si esa ventana YA se esta
                # mostrando. Comparar contra un `id_windows` huerfano
                # (recuadro parado) rechazaba relanzar la misma ventana.
                if (self.hwnd is not None
                        and int(other_hwnd) == getattr(self, "id_windows",
                                                       None)):
                    event.ignore()
                    return
                self._stop_dvr_stream()
                self.get_hwnd_and_print(int(other_hwnd))
                self.id_windows = int(other_hwnd)
                self.title      = other_title
                self.hwnd       = self.id_windows
                event.acceptProposedAction()
                if callable(self.callback_save_data):
                    self.callback_save_data(self.index, "hwnd", self.id_windows)
                return

            event.ignore()
        except Exception as e:
            print(f"dropEvent error: {e}")


    # ── Resto del código original (sin cambios) ───────────────

    def _hideandclear_roy(self):
        """Activa/desactiva el ROI de personas individualmente."""
        self.imagen_label.toggle_points()
        self.roi_boolean = not self.roi_boolean
        self.imagen_label.update()
        self._save_all("roi_boolean", self.roi_boolean)

    def _toggle_points_visibility(self):
        """Oculta/muestra los puntos del ROI sin desactivar el ROI."""
        self.imagen_label.toggle_points_visibility()

    def _toggle_trace(self):
        """Prende/apaga las estelas de movimiento (Supervision). Persiste."""
        self.trace_boolean = not self.trace_boolean
        if self._sv_overlay is not None:
            self._sv_overlay.show_trace = self.trace_boolean
        self._save_all("trace_boolean", self.trace_boolean)

    def _toggle_box_style(self):
        """Alterna el estilo de deteccion: caja redondeada <-> elipse. Persiste."""
        self.ellipse_style = not self.ellipse_style
        if self._sv_overlay is not None:
            self._sv_overlay.style = "ellipse" if self.ellipse_style else "round"
        self._save_all("ellipse_style", self.ellipse_style)

    # ── Guardado de capturas del cliente ────────────────────────────
    def _save_capture(self):
        """Guarda a disco el frame ACTUAL de esta camara (lo que se ve, con
        cajas/zonas dibujadas). Carpeta: capturas/<camara>/<fecha>/.

        Es la captura MANUAL del cliente. Las fotos automaticas por evento
        (agarre, reposicion, caja...) las guarda el servidor en
        output/detecciones/. Defensivo: nunca lanza."""
        try:
            import time as _t
            # Preferir lo que se VE en pantalla (anotado); si no, el limpio.
            pix = self.imagen_label.pixmap() if hasattr(self, "imagen_label") \
                else None
            if pix is None or pix.isNull():
                pix = getattr(self, "current_pixmap", None)
            if pix is None or pix.isNull():
                print("[captura] no hay frame para guardar todavia")
                return
            cam = self._camera_display_name()
            safe = "".join(c for c in str(cam) if c.isalnum() or c in "_-")[:40] \
                or "cam"
            base = os.path.join("capturas", safe,
                                _t.strftime("%Y-%m-%d"))
            os.makedirs(base, exist_ok=True)
            ruta = os.path.join(base, _t.strftime("%H-%M-%S") + ".jpg")
            if pix.save(ruta, "JPG", 88):
                self._ultima_captura = os.path.abspath(ruta)
                print(f"[captura] guardada: {self._ultima_captura}")
                # Avisar en el sidebar (clickeable -> abre la carpeta).
                try:
                    self.alert_received.emit({
                        "event_type": "Captura guardada",
                        "class_name": "Captura",
                        "description": f"Captura manual de {cam}.",
                        "timestamp": _t.strftime("%Y-%m-%d %H:%M:%S"),
                        "image_base64": "", "crop_image": "",
                        "camera_id": self._device_id(),
                        "screenshot_path": self._ultima_captura,
                    })
                except Exception:
                    pass
            else:
                print(f"[captura] no se pudo escribir {ruta}")
        except Exception as e:
            print(f"[captura] error: {e}")

    # ── Deteccion de evento (VLM): compra / entrega de bandeja ───────────
    def _current_frame_bytes(self):
        """Bytes del frame actual: el ultimo enviado (IA activa) o captura
        fresca de la ventana."""
        if self._pending_frame_bytes:
            return bytes(self._pending_frame_bytes)
        try:
            if self.hwnd:
                buf = capture_window_by_hwnd(self.hwnd)
                if buf is not None:
                    return pil_image_to_png_bytes(buf)
        except Exception:
            pass
        return None

    def _http_base(self):
        # Derivacion unica en el nucleo (sin fallback silencioso a localhost,
        # ver base_http_del_websocket): el mismo criterio de H-02.
        from elde_core.ui.panel_capturas import base_http_del_websocket
        return base_http_del_websocket(getattr(self.socket, "url", "") or "")

    def _events_url(self):
        return self._http_base() + "/vlm/events"

    @Slot(dict)
    def _on_model_result(self, data):
        """Resultado de cambiar el modelo VLM (la seleccion se hace desde el
        menu 'Asistente' -> Modelo de IA)."""
        cur = (data.get("current") or {}).get("key")
        if data.get("status") == "ok" and cur:
            self._vlm_model_key = cur
            QMessageBox.information(
                self, "Modelo de IA",
                f"Modelo de IA del servidor: {cur.upper()} "
                f"({'calidad' if cur == '7b' else 'rapido'}).\n"
                "La proxima consulta puede tardar ~30s en cargar el modelo.")
        else:
            QMessageBox.warning(
                self, "Modelo de IA",
                f"No se pudo cambiar: {data.get('message', '?')}")

    def _sync_vlm_model(self):
        """GET /vlm/model -> guarda el modelo activo del servidor en
        self._vlm_model_key (el menu lo muestra). Se llama al (re)conectar."""
        try:
            self._sync_worker = _GetWorker(
                self._http_base() + "/vlm/model", self)
            self._sync_worker.done.connect(self._apply_model_state)
            self._sync_worker.start()
        except Exception:
            pass

    @Slot(dict)
    def _apply_model_state(self, data):
        cur = (data.get("current") or {}).get("key")
        if cur in ("3b", "7b"):
            self._vlm_model_key = cur

    def _detect_event(self):
        """Captura el frame actual y pregunta a la IA si hay compra / entrega."""
        if self._event_busy:
            return
        frame = self._current_frame_bytes()
        if not frame:
            QMessageBox.information(
                self, "Deteccion de evento",
                "No hay frame disponible. Activa el analisis IA o selecciona "
                "una ventana primero.")
            return
        self._event_busy = True
        self._event_worker = _EventWorker(
            self._events_url(), frame,
            {"context": self.vlm_context or ""}, self)
        self._event_worker.done.connect(self._on_event_result)
        self._event_worker.start()

    def _edit_vlm_context(self):
        """Edita el contexto VLM de esta camara (describe la escena). Se suma al
        contexto global y se envia con 'Detectar evento'. Persiste."""
        from PySide6.QtWidgets import QInputDialog
        txt, ok = QInputDialog.getMultiLineText(
            self, "Contexto VLM de esta camara",
            "Describe la escena para el VLM (donde esta la caja, la zona de "
            "entrega, como luce una compra, color de las bandejas...).\n"
            "Se suma al contexto global y se usa al 'Detectar evento':",
            self.vlm_context or "")
        if ok:
            self.vlm_context = (txt or "").strip()
            self._save_all("vlm_context", self.vlm_context)

    @Slot(dict)
    def _on_event_result(self, data):
        self._event_busy = False
        if data.get("status") == "error":
            QMessageBox.warning(
                self, "Deteccion de evento",
                f"Error: {data.get('message', '?')}")
            return
        compra = "SI" if data.get("compra") else "no"
        entrega = "SI" if data.get("entrega_bandeja") else "no"
        QMessageBox.information(
            self, "Deteccion de evento",
            f"Compra: {compra}\n"
            f"Entrega de bandeja: {entrega}\n\n"
            f"Confianza: {data.get('confianza', '?')}\n"
            f"{data.get('descripcion', '')}")

    def _reset_roi(self):
        """Reinicia el ROI de personas a un rectangulo centrado por defecto (lo
        deja visible/editable) y LIMPIA las estelas/heatmap acumulados."""
        default = [[150, 150], [850, 150], [850, 850], [150, 850]]  # 0..1000
        try:
            self.imagen_label.set_roi(default)
            self.imagen_label.set_edit_target('roi')
            self.imagen_label.points_hidden = False
            self.imagen_label.show_points_fn()
            self.roi_boolean = True
        except Exception as e:
            print(f"reset roi error: {e}")
        # limpiar el estado visual acumulado de Supervision (estelas + heatmap)
        if self._sv_overlay is not None:
            try:
                self._sv_overlay.reset_all()
            except Exception:
                pass
        self._save_all("roi", default)
        self._save_all("roi_boolean", self.roi_boolean)

    def _toggle_order_zone(self):
        """Activa/desactiva el ROI azul de Toma de orden."""
        self.order_zone_boolean = not self.order_zone_boolean
        self.imagen_label.toggle_order_zone(self.order_zone_boolean)
        if self.order_zone_boolean:
            self.imagen_label.set_edit_target('order')
            self.imagen_label.points_hidden = False  # no dejarlo globalmente oculto
        self.imagen_label.update()
        self._save_all("order_zone_boolean", self.order_zone_boolean)

    def _toggle_delivery_zone(self):
        """Activa/desactiva el ROI rojo de Entrega de plato."""
        self.delivery_zone_boolean = not self.delivery_zone_boolean
        self.imagen_label.toggle_delivery_zone(self.delivery_zone_boolean)
        if self.delivery_zone_boolean:
            self.imagen_label.set_edit_target('delivery')
            self.imagen_label.points_hidden = False
        self.imagen_label.update()
        self._save_all("delivery_zone_boolean", self.delivery_zone_boolean)

    # ── Areas de anaquel (modo Anaqueles) ────────────────────────────────
    def _toggle_anaqueles(self):
        self.anaqueles_boolean = not self.anaqueles_boolean
        self.imagen_label.toggle_anaqueles(self.anaqueles_boolean)
        if self.anaqueles_boolean:
            self.imagen_label.points_hidden = False
        self.imagen_label.update()
        self._save_all("anaqueles_boolean", self.anaqueles_boolean)

    def _numero_anaquel_nuevo(self):
        """El siguiente numero GLOBAL de anaquel (unico entre camaras).

        Sin callback (llamador viejo), respaldo local: max de esta camara +1
        — degradado pero funcional."""
        if callable(self._asignar_numero_anaquel):
            try:
                return int(self._asignar_numero_anaquel())
            except Exception:
                pass
        return (max(self.anaqueles_numeros) + 1
                if self.anaqueles_numeros else 1)

    def _sync_nombres_anaqueles(self):
        """Reconstruye los nombres desde los numeros, los persiste y los
        pinta en el visor. Es la UNICA fuente de los nombres enviados."""
        self.anaqueles_nombres = [f"Anaquel {n}"
                                  for n in self.anaqueles_numeros]
        self._save_all("anaqueles_numeros", self.anaqueles_numeros[:])
        try:
            self.imagen_label.set_anaqueles_etiquetas(self.anaqueles_nombres)
        except Exception:
            pass

    def _agregar_anaquel(self):
        """Un nivel mas del estante, con su numero global. La etiqueta lo
        crea apilado y emite `anaqueles_change` (persiste los poligonos)."""
        self.anaqueles_numeros.append(self._numero_anaquel_nuevo())
        self._sync_nombres_anaqueles()
        self.imagen_label.agregar_anaquel()
        if not self.anaqueles_boolean:
            self.anaqueles_boolean = True
            self._save_all("anaqueles_boolean", True)

    def _quitar_anaquel(self):
        if self.anaqueles_numeros:
            numero = self.anaqueles_numeros.pop()
            # Si era el ultimo asignado, el contador global retrocede: el
            # ciclo "me equivoque, quito y redibujo" no quema numeros.
            if callable(self._liberar_numero_anaquel):
                try:
                    self._liberar_numero_anaquel(numero)
                except Exception:
                    pass
            self._sync_nombres_anaqueles()
        self.imagen_label.quitar_anaquel()

    @Slot(list, bool)
    def save_anaqueles(self, poligonos, activo):
        self.anaqueles = poligonos
        self.anaqueles_boolean = bool(activo)
        self._save_all("anaqueles", poligonos)
        self._save_all("anaqueles_boolean", self.anaqueles_boolean)

    # ── Productos dentro de las areas (25-sep-2026) ──────────────────────
    def save_ignorar(self, areas, activo):
        self.zonas_ignorar = areas
        self.zonas_ignorar_boolean = bool(activo)
        self._save_all("zonas_ignorar", areas)
        self._save_all("zonas_ignorar_boolean", self.zonas_ignorar_boolean)

    def save_productos(self, productos):
        self.anaqueles_productos = productos
        self._save_all("anaqueles_productos", productos)

    def _pedir_nombre_producto(self, titulo, texto, actual=""):
        """El nombre lo escribe el operador («Nescafé Gold»): es lo que el
        informe de marketing enseña. `None` si cancela."""
        from PySide6.QtWidgets import QInputDialog, QLineEdit
        from elde_core.ui.productos_anaquel import MAX_NOMBRE
        # El texto inicial va POSICIONAL, como en `device_panel.py`: es la
        # forma de llamada que ya corre en este cliente.
        nombre, ok = QInputDialog.getText(self, titulo, texto,
                                          QLineEdit.Normal, actual)
        if not ok:
            return None
        return str(nombre or "").strip()[:MAX_NOMBRE]

    def _agregar_producto(self, anaquel):
        area = self._nombres_anaqueles_todas()[anaquel]
        nombre = self._pedir_nombre_producto(
            "Nuevo producto", f"Nombre del producto en {area}:")
        if nombre is None:
            return
        if self.imagen_label.agregar_producto(anaquel, nombre):
            # El lienzo enciende las areas para que el producto se vea; el
            # interruptor del recuadro tiene que decir lo mismo.
            if not self.anaqueles_boolean:
                self.anaqueles_boolean = True
                self._save_all("anaqueles_boolean", True)

    def _renombrar_producto(self, indice):
        actual = self.imagen_label.productos.nombre_visible(indice)
        nombre = self._pedir_nombre_producto(
            "Renombrar producto", "Nombre nuevo:", actual)
        if nombre is not None:
            self.imagen_label.renombrar_producto(indice, nombre)

    def _quitar_producto(self, anaquel):
        self.imagen_label.quitar_producto(anaquel)

    def _nombres_anaqueles_todas(self):
        """Nombre de CADA area dibujada, este o no encendido el interruptor
        (`_nombres_anaqueles` devuelve vacio con las areas apagadas, y el
        menu tiene que poder listarlas igual)."""
        nombres = list(self.anaqueles_nombres or [])
        return [(str(nombres[i]).strip()
                 if i < len(nombres) and str(nombres[i]).strip()
                 else f"Área {i + 1}")
                for i in range(len(self.imagen_label.anaqueles))]

    def _build_productos_menu(self, menu):
        """Submenu por area: añadir, renombrar y quitar el ULTIMO producto
        (solo el ultimo: el servidor identifica cada producto por su
        posicion dentro del area, ver `productos_anaquel.py`)."""
        productos = self.imagen_label.productos
        for i, area in enumerate(self._nombres_anaqueles_todas()):
            suyos = productos.de_anaquel(i)
            sub = menu.addMenu(f"{area} ({len(suyos)})")
            sub.setStyleSheet(self._MENU_QSS)
            sub.addAction("Añadir producto…",
                          lambda *_, i=i: self._agregar_producto(i))
            renombrar = sub.addMenu("Renombrar")
            renombrar.setStyleSheet(self._MENU_QSS)
            renombrar.setEnabled(bool(suyos))
            for j in suyos:
                renombrar.addAction(productos.nombre_visible(j),
                                    lambda *_, j=j: self._renombrar_producto(j))
            quitar = sub.addAction("Quitar ultimo producto",
                                   lambda *_, i=i: self._quitar_producto(i))
            quitar.setEnabled(bool(suyos))
        # «Posible agotado» (5-oct-2026, servidor `anaqueles_stock.py`): la
        # foto de referencia la pide el operador con la balda BIEN SURTIDA;
        # el servidor no la renueva sola.
        # 9-oct-2026: también por producto (p. ej. tras reponer solo uno).
        # El servidor la toma sola cada día a las 7:30 y la rehace tras cada
        # bajada confirmada; esta es la manual, con la balda bien surtida.
        menu.addSeparator()
        foto = menu.addMenu("📷 Foto de referencia de stock (balda surtida)")
        foto.setStyleSheet(self._MENU_QSS)
        foto.setEnabled(bool(productos.productos))
        foto.addAction("Todos los productos",
                       lambda *_: self._foto_referencia_stock())
        for i, area in enumerate(self._nombres_anaqueles_todas()):
            suyos = productos.de_anaquel(i)
            if not suyos:
                continue
            sub = foto.addMenu(area)
            sub.setStyleSheet(self._MENU_QSS)
            # La posición DENTRO del área es la identidad del producto en
            # el servidor (`productos_anaquel.py`).
            for posicion, j in enumerate(suyos):
                sub.addAction(productos.nombre_visible(j),
                              lambda *_, i=i, p=posicion:
                              self._foto_referencia_stock(i, p))

    def _foto_referencia_stock(self, area=None, producto=None):
        """Pide al servidor la foto «surtido» de todos los productos de esta
        camara, o solo del producto `producto` (posicion dentro del area
        `area`). Va por HTTP y no por el websocket: el payload lo comparten
        otros clientes (GUIA 4.7) y esto es una orden suelta, no un dato de
        cada frame."""
        from urllib.parse import quote
        try:
            url = (f"{self._http_base()}/api/anaqueles/"
                   f"{quote(str(self._device_id()), safe='')}"
                   "/stock/referencia")
            if area is not None and producto is not None:
                url += f"?area={int(area)}&producto={int(producto)}"
        except Exception as e:
            QMessageBox.warning(self, "Stock", f"Sin servidor: {e}")
            return
        self._stock_worker = _PostWorker(url, parent=self)
        self._stock_worker.done.connect(self._on_foto_referencia_stock)
        self._stock_worker.start()

    @Slot(dict)
    def _on_foto_referencia_stock(self, data):
        if data.get("status") == "ok":
            QMessageBox.information(
                self, "Stock",
                f"{data.get('mensaje', 'Foto pedida.')}\n\n"
                "Hazla con la balda bien surtida: pone a cero la cuenta de "
                "bajadas. Tras cada agarre, el servidor compara el producto "
                "con su foto: si falta algo lo cuenta y rehace la foto, y "
                "cada 8 bajadas avisa de que está cerca de vaciarse.")
        else:
            QMessageBox.warning(
                self, "Stock",
                "No se pudo pedir la foto: "
                f"{data.get('detail') or data.get('message') or '?'}\n\n"
                "¿La camara esta conectada en modo Anaqueles?")

    def _nombres_anaqueles(self):
        """Nombres paralelos a las zonas enviadas: la numeracion GLOBAL
        («Anaquel 7»), unica entre camaras. Respaldo por indice si algun
        area quedara sin numero (no deberia: la migracion del __init__)."""
        n = (len(self.imagen_label.anaqueles)
             if self.anaqueles_boolean else 0)
        nombres = list(self.anaqueles_nombres or [])
        return [
            (str(nombres[i]).strip()
             if i < len(nombres) and str(nombres[i]).strip()
             else f"Área {i + 1}")
            for i in range(n)]

    def _toggle_munecas(self):
        self._munecas_boolean = not self._munecas_boolean
        self._save_all("munecas_boolean", self._munecas_boolean)

    def _toggle_ids(self):
        self._ids_boolean = not self._ids_boolean

    def _toggle_regla(self):
        regla = self.imagen_label.regla
        regla.activa = not regla.activa
        if not regla.activa:
            regla.borrar()          # al apagarla, la línea se va con ella
        self.imagen_label.update()

    def _toggle_telemetria_ciclo(self):
        if self._telemetria.activa:
            self._telemetria.apagar()
        else:
            self._telemetria.activa = True
            print(f"[TELEMETRIA] escribiendo en {_telemetria.ruta_por_defecto()}",
                  flush=True)
        self._save_all("telemetria_ciclo_boolean", self._telemetria.activa)

    def _toggle_vlm(self):
        """Activa/desactiva el verificador de eventos (Etapa 2) en el servidor.
        El estado viaja en el payload websocket; el servidor hace lazy-load
        del modelo Qwen2.5-VL la primera vez que se enciende."""
        self.vlm_enabled_boolean = not self.vlm_enabled_boolean
        self._save_all("vlm_enabled_boolean", self.vlm_enabled_boolean)

    # ── Barra: menus desplegables (ROI / Vista / Asistente) ──────────────
    # Toda opcion vive en estos menus. El estado son flags en self.* y los
    # menus se reconstruyen al abrirse (aboutToShow) para reflejarlo.
    _MENU_QSS = (
        "QMenu{background:#2b2b2b;border:1px solid #555;padding:4px;}"
        "QMenu::item{color:#eee;padding:6px 22px 6px 24px;}"
        "QMenu::item:selected{background:#3d6fb0;}"
        "QMenu::separator{height:1px;background:#555;margin:4px 8px;}"
        "QMenu::indicator{width:14px;height:14px;left:6px;}"
        "QMenu::indicator:checked{background:#4CAF50;border:1px solid #666;border-radius:2px;}"
        "QMenu::indicator:non-exclusive:unchecked{background:#555;border:1px solid #666;border-radius:2px;}"
    )

    def _make_menu_button(self, text, tooltip, build_fn):
        """QToolButton con menu desplegable (popup instantaneo). El menu se
        RECONSTRUYE en cada apertura (build_fn) para reflejar el estado real."""
        btn = QToolButton()
        btn.setText(text)
        btn.setObjectName("btn-bar")
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setPopupMode(QToolButton.InstantPopup)
        btn.setFixedHeight(30)
        btn.setStyleSheet(
            "QToolButton{color:#fff;font-weight:bold;padding:2px 12px;"
            "border:1px solid #555;border-radius:4px;background:#333;}"
            "QToolButton:hover{background:#3d6fb0;}"
            "QToolButton::menu-indicator{image:none;}")
        menu = QMenu(btn)
        menu.setStyleSheet(self._MENU_QSS)
        menu.aboutToShow.connect(lambda: build_fn(menu))
        btn.setMenu(menu)
        return btn

    def _add_toggle(self, menu, text, checked, on_toggle):
        """Item checkable que refleja un flag (checked) y, al pulsarlo, corre
        on_toggle() (que conmuta ese flag)."""
        a = menu.addAction(text)
        a.setCheckable(True)
        a.setChecked(bool(checked))
        a.triggered.connect(lambda *_: on_toggle())
        return a

    def _build_roi_menu(self, menu):
        # ROI = area de conteo de la camara. Las zonas de comida (toma de
        # orden / entrega de plato) se quitaron: no aplican a tienda; las
        # zonas de tienda (pasillos/anaqueles/precio) se definen en el menu
        # "Tienda" -> "Definir zonas de la tienda".
        menu.clear()
        pts_visible = not getattr(self.imagen_label, "points_hidden", False)
        self._add_toggle(menu, "Activar area de conteo (ROI)",
                         self.roi_boolean, self._hideandclear_roy)
        self._add_toggle(menu, "Mostrar puntos del area",
                         pts_visible, self._toggle_points_visibility)
        menu.addSeparator()
        # Areas de anaquel (modo Anaqueles): un cuadrilatero por nivel del
        # estante. Solo el pipeline Anaqueles las lee; en modo Tiendas se
        # pueden dejar dibujadas sin que decidan nada.
        n = len(self.imagen_label.anaqueles)
        self._add_toggle(menu, f"Areas de anaquel ({n})",
                         self.anaqueles_boolean, self._toggle_anaqueles)
        menu.addAction("Añadir area de anaquel",
                       lambda *_: self._agregar_anaquel())
        quitar = menu.addAction("Quitar ultima area",
                                lambda *_: self._quitar_anaquel())
        quitar.setEnabled(n > 0)
        # Productos DENTRO de cada area (25-sep-2026): reparten sus agarres.
        # Sin areas no hay donde meterlos.
        productos = menu.addMenu(
            f"Productos del anaquel "
            f"({len(self.imagen_label.productos.productos)})")
        productos.setStyleSheet(self._MENU_QSS)
        productos.setEnabled(n > 0)
        self._build_productos_menu(productos)
        menu.addSeparator()
        # Areas a ignorar (12-oct-2026): quien queda ENTERO dentro no se
        # cuenta (gente del fondo que no esta en el pasillo).
        ign = self.imagen_label.ignorar
        self._add_toggle(menu, f"Areas a ignorar ({len(ign.areas)})",
                         ign.activa, lambda: self.imagen_label.toggle_ignorar())
        menu.addAction("Añadir area a ignorar (fondo)",
                       lambda *_: self.imagen_label.agregar_ignorar())
        quitar_ign = menu.addAction(
            "Quitar ultima area a ignorar",
            lambda *_: self.imagen_label.quitar_ignorar())
        quitar_ign.setEnabled(bool(ign.areas))
        menu.addSeparator()
        # La puerta tiene ADEMAS su boton en la barra (es lo que se busca al
        # calibrar); aqui se repite para quien venga al menu de ROI
        # esperando encontrar todas las zonas juntas.
        self._add_toggle(menu, "Puerta direccional (flecha doble ⇅ + umbral)",
                         self.puerta_boolean, self.toggle_puerta)
        invertir = menu.addAction("Invertir sentido (entrada ↔ salida)")
        invertir.setEnabled(self.puerta_boolean)
        invertir.triggered.connect(lambda *_: self.invertir_puerta())
        menu.addSeparator()
        menu.addAction("Reiniciar area y estelas", lambda *_: self._reset_roi())

    def _build_vista_menu(self, menu):
        menu.clear()
        self._add_toggle(menu, "Estelas de movimiento",
                         self.trace_boolean, self._toggle_trace)
        # Va en Vista y no en Area: no dibuja geometria, ensenia lo que el
        # servidor ESTA viendo. Solo pinta algo en modo Anaqueles, que es el
        # unico que manda munecas.
        self._add_toggle(menu, "Munecas del anaquel (diagnostico)",
                         self._munecas_boolean, self._toggle_munecas)
        # Comprobar que ByteTrack conserva el número de cada persona.
        self._add_toggle(menu, "IDs de seguimiento (diagnostico)",
                         self._ids_boolean, self._toggle_ids)
        # Diagnostico de rendimiento: el operador la enciende, usa el cliente
        # y la sesion de soporte lee el fichero despues. La ruta se ensenia
        # aqui mismo para que no haya que buscarla.
        self._add_toggle(menu, "Telemetria del ciclo (diagnostico)",
                         self._telemetria.activa,
                         self._toggle_telemetria_ciclo)
        ruta = menu.addAction(f"   → {_telemetria.ruta_por_defecto()}")
        ruta.setEnabled(False)
        # Regla para medir a la gente del fondo en píxeles de la imagen.
        # No se guarda: es una herramienta de un momento, y encendida
        # bloquea la edición de las zonas.
        self._add_toggle(menu, "Regla: medir en píxeles (clic y arrastrar)",
                         self.imagen_label.regla.activa, self._toggle_regla)
        self._add_toggle(menu, "Cajas estilo elipse",
                         self.ellipse_style, self._toggle_box_style)

    def _build_ia_menu(self, menu):
        menu.clear()
        menu.addAction("Preguntar a la IA…", lambda *_: self._ask_vlm())
        menu.addAction("Detectar evento (compra / entrega)",
                       lambda *_: self._detect_event())
        menu.addSeparator()
        sub = menu.addMenu("Modelo de IA")
        sub.setStyleSheet(self._MENU_QSS)
        is7 = (self._vlm_model_key == "7b")
        a3 = sub.addAction("3B (rapido)"); a3.setCheckable(True); a3.setChecked(not is7)
        a7 = sub.addAction("7B (calidad)"); a7.setCheckable(True); a7.setChecked(is7)
        a3.triggered.connect(lambda *_: self._select_vlm_model("3b"))
        a7.triggered.connect(lambda *_: self._select_vlm_model("7b"))
        menu.addAction("Contexto de la escena…",
                       lambda *_: self._edit_vlm_context())
        menu.addSeparator()
        self._add_toggle(menu, "Verificador de eventos (Etapa 2)",
                         self.vlm_enabled_boolean, self._toggle_vlm)








    # ── Lienzo de zonas sobre el video en vivo ───────────────────────

    _ZONA_COLORES = {
        "pasillos":   (0, 200, 255),    # cian
        "anaqueles":  (0, 220, 0),      # verde
        "mobiliario": (255, 0, 255),    # magenta
    }







    def _emit_alert(self, event_type: str, class_name: str, desc: str,
                    screenshot_path: str = ""):
        """Empuja una alerta al sidebar por el canal que ya existe.

        `screenshot_path`: ruta de la FOTO de evidencia guardada por el
        servidor (misma maquina) -> click en la alerta la abre en Explorer.
        """
        self.alert_received.emit({
            "event_type":   event_type,
            "class_name":   class_name,
            "description":  desc,
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_base64": "",
            "crop_image":   "",
            "camera_id":    self._device_id(),
            "screenshot_path": screenshot_path or "",
        })




    def _select_vlm_model(self, key):
        """Cambia el modelo VLM del servidor (3b/7b) -> POST /vlm/model."""
        if key not in ("3b", "7b") or key == self._vlm_model_key:
            return
        self._vlm_model_key = key  # feedback optimista; _on_model_result confirma
        self._model_worker = _ModelWorker(
            self._http_base() + "/vlm/model", key, self)
        self._model_worker.done.connect(self._on_model_result)
        self._model_worker.start()

    def _ask_vlm(self):
        """Pregunta libre a la IA (VQA) sobre el frame actual -> /vlm/ask."""
        frame = self._current_frame_bytes()
        if not frame:
            QMessageBox.information(
                self, "Preguntar a la IA",
                "No hay frame disponible. Activa el monitoreo o selecciona una "
                "ventana primero.")
            return
        from PySide6.QtWidgets import QInputDialog
        q, ok = QInputDialog.getText(
            self, "Preguntar a la IA", "¿Que quieres saber de la escena?")
        if not ok or not q.strip():
            return
        self._ask_worker = _EventWorker(
            self._http_base() + "/vlm/ask", frame,
            {"question": q.strip(), "context": self.vlm_context or ""}, self)
        self._ask_worker.done.connect(self._on_ask_result)
        self._ask_worker.start()

    @Slot(dict)
    def _on_ask_result(self, data):
        if data.get("status") == "error":
            QMessageBox.warning(self, "Preguntar a la IA",
                                f"Error: {data.get('message', '?')}")
            return
        QMessageBox.information(
            self, "Respuesta de la IA", data.get("answer") or "(sin respuesta)")

    def _show_class_menu(self):
        """Muestra un menú popup con checkboxes para seleccionar qué clases detectar."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2b2b2b; border: 1px solid #555; padding: 4px; }
            QCheckBox { color: white; padding: 4px 8px; font-size: 12px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QCheckBox::indicator:checked { background: #4CAF50; border: 1px solid #666; border-radius: 2px; }
            QCheckBox::indicator:unchecked { background: #555; border: 1px solid #666; border-radius: 2px; }
        """)

        for name, class_id in self._available_classes.items():
            cb = QCheckBox(name)
            cb.setChecked(class_id in self._selected_classes)
            cb.toggled.connect(
                lambda checked, cid=class_id: self._on_class_toggled(cid, checked)
            )
            action = QWidgetAction(menu)
            action.setDefaultWidget(cb)
            menu.addAction(action)

        # ── Mapa de calor (overlay del servidor) ──
        menu.addSeparator()
        cb_heat = QCheckBox("Mapa de calor")
        cb_heat.setChecked(self.heatmap_boolean)
        cb_heat.toggled.connect(self._on_heatmap_toggled)
        action_heat = QWidgetAction(menu)
        action_heat.setDefaultWidget(cb_heat)
        menu.addAction(action_heat)

        # Mostrar debajo del botón
        btn_pos = self._btn_classes.mapToGlobal(self._btn_classes.rect().bottomLeft())
        menu.exec(btn_pos)

    def _on_heatmap_toggled(self, checked: bool):
        """Activa/desactiva el overlay de mapa de calor del servidor."""
        self.heatmap_boolean = bool(checked)
        self._save_all("heatmap_boolean", self.heatmap_boolean)

    _slug = staticmethod(_identidad.slug)

    def _device_id(self) -> str:
        """Identificador ESTABLE de la camara, para el `camera_id` del payload.

        La logica vive en `elde_core.ui.identidad_camara`: los cuatro clientes
        tenian el mismo fallo (H-11) y ahora comparten el mismo arreglo. Aqui
        solo se leen los atributos del recuadro.
        """
        return _identidad.device_id(
            serie_dvr=self._dvr_device_serial,
            canal_dvr=self._dvr_channel_id,
            titulo_ventana=getattr(self, 'title', ''),
            indice=getattr(self, 'index', 0),
            # Separa esta INSTALACION de las demas de la misma PC
            # (9-sep-2026): el indice del recuadro solo separa
            # ventanas homonimas DENTRO de un cliente, y dos clientes
            # distintos capturando dos iVMS-4200 mandaban el mismo
            # `win-iVMS-4200` — un rastreador para dos escenas y el
            # local saltando entre los dos en cada frame.
            instalacion=_identidad.instalacion_de_app('tienda_view'))

    def _camera_display_name(self) -> str:
        """Nombre legible de esta camara para el dashboard."""
        return _identidad.nombre_visible(
            nombre_dvr=self.camera_name_dvr,
            titulo_ventana=getattr(self, 'title', ''),
            indice=getattr(self, 'index', 0))

    def _on_class_toggled(self, class_id: int, checked: bool):
        """Actualiza las clases seleccionadas para tracking."""
        if checked and class_id not in self._selected_classes:
            self._selected_classes.append(class_id)
        elif not checked and class_id in self._selected_classes:
            self._selected_classes.remove(class_id)
        # Guardar en configuración persistente
        self._save_all("track_classes", self._selected_classes[:])

    def init_loop(self):
        try:
            if self.hwnd is None:
                return
            if self.process is None:
                self.process = QProcess(self)
                # Canales SEPARADOS: por stdout van los frames en
                # msgpack y NADA mas. Con MergedChannels, cualquier
                # linea que el worker escribiera por stderr entraba en
                # el flujo y lo corrompia («Error deserializando
                # mensaje» en el servidor), asi que el worker no podia
                # avisar de nada — por eso una captura muerta estuvo
                # 19 h en silencio (8-ago-2026).
                self.process.setProcessChannelMode(
                    QProcess.SeparateChannels)
                try:
                    self.process.readyReadStandardError.connect(
                        self.leer_avisos_del_worker,
                        Qt.UniqueConnection)
                except (RuntimeError, TypeError):
                    pass
                self._unpacker = None  # stream nuevo: empezar limpio
                try:
                    self.process.readyReadStandardOutput.connect(
                        self.loop_show_result, Qt.UniqueConnection)
                except (RuntimeError, TypeError):
                    pass
                if getattr(sys, 'frozen', False):
                    # EMPAQUETADO: sys.executable es el propio .exe. Lanzarlo
                    # con un script como argumento abriria OTRA copia de la
                    # aplicacion (bucle de ventanas al dar Play), porque
                    # PyInstaller ignora el script que se le pase. Se re-invoca
                    # en modo worker: main.py lo detecta y captura sin GUI.
                    self.process.start(sys.executable,
                                       ["--capture-worker", str(self.hwnd)])
                else:
                    worker_script = "src/workers/capture_woker.py"
                    if not os.path.exists(worker_script):
                        return
                    self.process.start(sys.executable,
                                       [worker_script, str(self.hwnd)])
                if not self.process.waitForStarted(5000):
                    return
            else:
                try:
                    self.process.readyReadStandardOutput.connect(
                        self.loop_show_result, Qt.UniqueConnection)
                except (RuntimeError, TypeError):
                    pass
        except Exception as e:
            print(f"init_loop error: {e}")

    def pause_loop(self):
        if self.process:
            self.text_fps.setText("FPS: 0")

    def activate_modesmart(self):
        self.smart_mode = not self.smart_mode
        self.olvidar_captura_guardada()
        self._save_all("inference_play", self.smart_mode)
        if self.smart_mode:
            self.btn_smart.setStyleSheet("background-color:#FF0000;")
            self.stop = False
            self.can_send_next_frame = True
            self._ai_fps_count = 0
            self._ai_fps_t0 = time.time()
            self.text_fps.setText("IA: … FPS")
        else:
            self.btn_smart.setStyleSheet("background-color:#BFBFBF;")
            self.can_send_next_frame = True

    def detroy_loop(self):
        """Detiene el recuadro y lo deja LISTO para reutilizar.

        Antes dejaba `id_windows` con la ventana anterior y `stop` en
        True: volver a soltar la MISMA ventana caia en el guard del
        dropEvent («ya la estoy mostrando») y se ignoraba — el recuadro
        quedaba en negro con "viewing window" hasta reiniciar la
        aplicacion. Es el mismo arreglo que ya llevaba el detroy_loop
        de Amazonas (8-ago-2026).
        """
        self.btn_smart.setChecked(False)
        self.smart_mode = False
        # Fuente A: worker de captura de ventana.
        if self.process is not None:
            self.process.terminate()
            if not self.process.waitForFinished(1000):
                self.process.kill()
            self.process = None
        # Fuente B: stream DVR (sin esto, parar un canal DVR dejaba el
        # hilo RTSP vivo y la imagen no se iba).
        try:
            if (getattr(self, '_rtsp_worker', None) is not None
                    or getattr(self, '_dvr_mode', False)):
                self._stop_dvr_stream()
        except Exception:
            pass
        # Estado de la fuente: TODO a cero, incluido `id_windows` (el
        # guard del dropEvent compara contra el).
        self.title = None
        self.hwnd = None
        self.id_windows = None
        # `stop` significa "no sigas el bucle actual", no un cierre
        # permanente del recuadro: dejarla en True lo inutilizaba para
        # el siguiente lanzamiento.
        self.stop = False
        self.olvidar_captura_guardada()
        self.can_send_next_frame = True
        self.current_pixmap = None
        self.image_w = 0
        self.image_h = 0
        self.imagen_label.setPixmap(QPixmap())
        self.imagen_label.clear()
        self.imagen_label.setText("viewing window")
        self.text_fps.setText("FPS: 0")
        self.text_size.setText("0x0")
        if callable(self.callback_save_data):
            self.callback_save_data(self.index, "hwnd", None)

    @Slot(int)
    def get_hwnd_and_print(self, hwnd=None):
        try:
            if hwnd is not None:
                self.hwnd = hwnd
                self._preview_intentos = 0
            set_window_always_on_top(self.hwnd)
            buffer = capture_window_by_hwnd(self.hwnd)
            image = pil_image_to_png_bytes(buffer) if buffer is not None else None
            if image is None:
                # La vista previa es UN solo PrintWindow justo despues de
                # reordenar la ventana: en equipos lentos ese primer disparo
                # falla y el recuadro quedaba vacio hasta pulsar Play (el
                # worker si reintenta en bucle). Reintentar unas veces antes
                # de rendirse, solo mientras el worker no este corriendo.
                intentos = getattr(self, "_preview_intentos", 0)
                if (intentos < 5 and self.hwnd is not None
                        and getattr(self, "process", None) is None):
                    self._preview_intentos = intentos + 1
                    QTimer.singleShot(400, lambda: self.get_hwnd_and_print())
                return
            self._preview_intentos = 0
            self.update_streaming_frame(image, type_image="bytes", tets=False)
            self.bar_options.raise_()
        except Exception as e:
            print(f"get_hwnd error: {e}")


    def _active_zone_polygons(self, w, h):
        """Poligonos (en pixeles w x h del frame) de las zonas ACTIVAS, para
        que Supervision cuente personas dentro (PolygonZone)."""
        zones = []
        try:
            lbl = self.imagen_label
            if getattr(self, "roi_boolean", False):
                p = lbl.get_coordinates(w, h)
                if p and len(p) >= 3:
                    zones.append(p)
            if getattr(self, "order_zone_boolean", False):
                p = lbl.get_order_zone_coordinates(w, h)
                if p and len(p) >= 3:
                    zones.append(p)
            if getattr(self, "delivery_zone_boolean", False):
                p = lbl.get_delivery_zone_coordinates(w, h)
                if p and len(p) >= 3:
                    zones.append(p)
        except Exception:
            pass
        return zones

    # ── Salida a pantalla (unica ruta) ───────────────────────────────

    def _apply_label_pixmap(self, pixmap):
        """Punto UNICO por el que un frame llega al label.

        `current_pixmap` se guarda LIMPIO (sin el overlay de zonas) para que
        redibujar al redimensionar no acumule capas; las zonas se pintan
        sobre una copia justo antes de mostrar.
        """
        if pixmap is None or pixmap.isNull():
            return
        self.current_pixmap = pixmap
        self.imagen_label.setPixmap(
            pixmap.scaled(
                self.imagen_label.size(), Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation))

    def _show_bgr(self, bgr):
        """Muestra un frame numpy BGR (anotado por Supervision) en el label."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())  # copy: el buffer numpy es temporal
        self.image_w, self.image_h = w, h
        if hasattr(self, "text_size"):
            self.text_size.setText(f"{w}x{h}")
        self._apply_label_pixmap(pixmap)

    def _munecas_a_pintar(self):
        """Las munecas vigentes, o vacio si estan apagadas o caducadas.

        Un solo sitio decide: los dos caminos de pintado (Supervision y
        QPainter) tienen que enseniar LO MISMO, o el operador veria una cosa
        distinta segun lo que tenga instalado."""
        if not self._munecas_boolean:
            return []
        return _munecas.vigentes(self._munecas_ultimas, self._munecas_ts,
                                 time.time())

    def _try_supervision_overlay(self, detections, jpeg=None):
        """Anota con Supervision (cajas/estelas/zonas/heatmap). True si lo logro;
        False -> el caller cae al dibujo con QPainter."""
        if self._sv_overlay is None or cv2 is None or np is None:
            return False
        try:
            arr = np.frombuffer(jpeg or self._pending_frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            if frame is None:
                return False
            h, w = frame.shape[:2]
            self._sv_overlay.show_heatmap = bool(
                getattr(self, "heatmap_boolean", False))
            zones = self._active_zone_polygons(w, h)
            annotated = self._sv_overlay.annotate(frame, detections, zones)
            # DESPUES de Supervision: las munecas son la capa de diagnostico
            # y tienen que quedar por encima de las cajas y el heatmap.
            _munecas.pintar_cv2(annotated, self._munecas_a_pintar())
            self._show_bgr(annotated)
            return True
        except Exception as e:
            print(f"sv overlay error (fallback QPainter): {e}")
            return False

    def _draw_detections_and_show(self, detections, jpeg=None):
        """MODO DIRECTO: dibuja las detecciones del servidor sobre el frame que
        el cliente envio y lo muestra. Usa Supervision
        (cajas redondeadas + etiquetas + estelas + zonas + heatmap) si esta
        disponible; si no, cae al dibujo simple con QPainter. Las coords vienen
        en pixeles del frame enviado -> se dibujan a esa resolucion y luego se
        escala todo al label (alineado).

        `jpeg`: el frame de ESTA respuesta. Desde el 5-oct-2026 la siguiente
        imagen se envia antes de pintar, y `_pending_frame_bytes` ya es la
        nueva: sin esto las cajas irian sobre la imagen equivocada."""
        jpeg = jpeg or self._pending_frame_bytes
        if not jpeg:
            return
        if getattr(self, '_ids_boolean', False):
            detections = _etiqueta_id.con_id(detections)
        if self._try_supervision_overlay(detections, jpeg):
            return
        # 28-oct-2026: se abre ya al tamaño del recuadro (`jpeg_a_medida`,
        # ~9 ms menos por imagen) y el pintor se escala, asi que todo se
        # sigue dibujando en pixeles de la imagen ENVIADA, como antes.
        etiqueta = self.imagen_label.size()
        imagen, ancho_real, alto_real = _jpeg_a_medida.abrir(
            jpeg, etiqueta.width(), etiqueta.height())
        if imagen is None:
            return
        pixmap = QPixmap.fromImage(imagen)
        try:
            painter = QPainter(pixmap)
            if pixmap.width() != ancho_real or pixmap.height() != alto_real:
                # Sin suavizado, las rayas de 2 px reducidas saldrian
                # dentadas; antes las suavizaba el encogido final.
                painter.setRenderHint(QPainter.Antialiasing)
                painter.scale(pixmap.width() / ancho_real,
                              pixmap.height() / alto_real)
            font = QFont(); font.setPointSize(10); font.setBold(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            for d in (detections or []):
                box = d.get("box")
                if not box or len(box) < 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                col = d.get("color") or [0, 255, 0]
                qcol = QColor(int(col[0]), int(col[1]), int(col[2]))
                pen = QPen(qcol); pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                label = d.get("label", "")
                if label:
                    tw = fm.horizontalAdvance(label); th = fm.height()
                    ty = max(th, y1)
                    painter.fillRect(x1, ty - th, tw + 6, th, QColor(0, 0, 0))
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(x1 + 3, ty - 4, label)
            _munecas.pintar_qt(painter, self._munecas_a_pintar())
            painter.end()
        except Exception as e:
            print(f"draw detections error: {e}")
        # Los de la imagen enviada, no los del pixmap reducido: con ellos
        # pasan las zonas y la regla de 0-1000 a pixeles.
        self.image_w = ancho_real
        self.image_h = alto_real
        if hasattr(self, 'text_size'):
            self.text_size.setText(f"{self.image_w}x{self.image_h}")
        self._apply_label_pixmap(pixmap)

    def update_streaming_frame(self, frame, type_image="base64", tets=False):
        try:
            if tets: self.open = True
            pixmap = QPixmap()
            if type_image == "base64":
                frame_bytes = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", frame))
                pixmap.loadFromData(frame_bytes, "JPEG")
            elif type_image == "jpeg_bytes":
                pixmap.loadFromData(frame, "JPEG")
            else:
                pixmap.loadFromData(frame, "PNG")

            if not pixmap.isNull():
                self.image_w = pixmap.width()
                self.image_h = pixmap.height()
                self.text_size.setText(f"{self.image_w}x{self.image_h}")
                self._apply_label_pixmap(pixmap)
        except Exception as e:
            print(f"update_frame error: {e}")
        finally:
            if not self.stop and tets:
                self.loop_show_result()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        if hasattr(self, "bar_info"):
            self.bar_info.setGeometry(0, 0, w, 30)
        if hasattr(self, "bar_options"):
            hb = self.bar_options.height()
            self.bar_options.setGeometry(0, h - hb, w, hb)
        if self.current_pixmap:
            self._apply_label_pixmap(self.current_pixmap)

    def enterEvent(self, event):
        self.bar_info.show(); self.bar_options.show()
        self.bar_info.raise_(); self.bar_options.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):
        try:
            from PySide6.QtWidgets import QApplication
            if QApplication.activePopupWidget() is not None:
                return
        except Exception:
            pass
        self.bar_info.hide(); self.bar_options.hide()
        super().leaveEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            # Ceder ante la calibracion: un doble clic sobre un vertice
            # arrastrable (area de anaquel, ROI, zonas) o en medio de un
            # arrastre NO debe maximizar el recuadro — el operador esta
            # ajustando puntos, no pidiendo pantalla completa.
            try:
                etiqueta = self.imagen_label
                if watched is etiqueta and (
                        etiqueta.arrastrando_punto()
                        or etiqueta.hay_punto_en(event.pos())):
                    return super().eventFilter(watched, event)
            except Exception:
                pass
            self.is_maximized = not self.is_maximized
            self.double_clicked_signal.emit(self.index, self.is_maximized)
        return super().eventFilter(watched, event)

    @Slot(dict)
    def on_text_message_received(self, message):
        # Ping del keepalive del servidor: no es la respuesta de ningun
        # frame, asi que NO debe pasar por el finally que desbloquea el
        # envio (desbloquear sin respuesta multiplica el ritmo de envio).
        if message.get("status") == "ping":
            return
        # Enrutado: el sobre viaja con `component_key` (clave del recuadro)
        # y el payload con `camera_id` (identidad estable de la camara, ver
        # _device_id). Se aceptan las dos porque desde H-11 ya NO son el
        # mismo valor: comparar solo contra component_key descartaria las
        # respuestas que solo traen camera_id.
        msg_key = message.get("component_key") or message.get("camera_id") or ""
        if msg_key and msg_key not in (self.component_key, self._device_id()):
            # Respuesta de OTRO recuadro: fuera del try->finally A PROPOSITO.
            # Este return vivia dentro, y el finally desbloqueaba
            # `can_send_next_frame`: cualquier respuesta de cualquier camara
            # desbloqueaba TODOS los recuadros, el cliente enviaba N frames
            # por respuesta procesada y esa contrapresion acababa matando la
            # conexion con 1011 'keepalive ping timeout' cada ~3 minutos
            # (diagnostico del 7-ago-2026; ver server/src/app/ws_drenaje.py).
            return
        # Apartado: el siguiente fotograma puede abrirse antes de pintar este.
        tel = self._telemetria.apartar() if self._telemetria.en_vuelo() else None
        if tel is not None:
            tel.marcar('t_respuesta')
        estado_tel = 'error'
        # La imagen de ESTA respuesta: enviar la siguiente sobrescribe
        # `_pending_frame_bytes`, y las cajas irian sobre la imagen equivocada.
        frame_respuesta = self._pending_frame_bytes
        enviada = False
        try:
            # Tarea 2 (5-oct-2026): la captura guardada sale YA, antes de
            # pintar esta respuesta. El servidor trabaja en ella mientras el
            # cliente pinta (~17 ms) y no se espera a la captura siguiente
            # (~20 ms de media en modo G). Sigue habiendo UNA en vuelo.
            self.can_send_next_frame = True
            enviada = self.enviar_captura_guardada()
            metadata  = message.get("data", {}).get("metadata", {})
            if metadata:

                list_alert = metadata.get("alerts", []) or []
                for iteration in list_alert:
                    event_type = iteration.get("event_type", "Alerta")
                    img_b64 = (
                        iteration.get("image_base64", "")
                        or iteration.get("crop_image", "")
                        or ""
                    )
                    print(f'[ALERT-EMIT] event={event_type!r} img_len={len(img_b64)}', flush=True)
                    self.alert_received.emit({
                        "event_type":      event_type,
                        "class_name":      iteration.get("class_name", event_type),
                        "description":     iteration.get("description", ""),
                        "timestamp":       iteration.get("timestamp", ""),
                        "image_base64":    img_b64,
                        "crop_image":      img_b64,
                        "camera_id":       self._device_id(),
                        "screenshot_path": iteration.get("screenshot_path", ""),
                        # El reenviador de Jarvis manda la novedad a este
                        # local; vacio = al del selector del pie.
                        "establecimiento": self.establecimiento,
                    })

                # Bloque del modo Anaqueles (contadores/eventos/heatmap por
                # area). Se reemite con la camara para que el panel lateral
                # sepa de quien es; los clientes en otro modo no lo reciben
                # nunca (el servidor solo lo construye en Anaqueles).
                bloque_anaqueles = metadata.get("anaqueles")
                if bloque_anaqueles and bloque_anaqueles.get("munecas"):
                    self._munecas_ultimas = bloque_anaqueles["munecas"]
                    self._munecas_ts = time.time()
                if bloque_anaqueles:
                    self.anaqueles_received.emit({
                        **bloque_anaqueles,
                        "camera_id": self._device_id(),
                        "camera_name": self._camera_display_name(),
                    })
            if tel is not None:
                tel.marcar('t_paneles')
            data = message["data"]
            estado_tel = str(data.get("status") or "sin_status")
            # El servidor hace eco del `camera_id` que le enviamos, que desde
            # H-11 es el device_id estable y NO component_key. Comparar contra
            # component_key aqui fallaba siempre y la imagen procesada no se
            # mostraba nunca.
            if data["status"] == "success" and data["camera_id"] == self._device_id():
                proc_img = data.get("processed_image")
                detections = (data.get("metadata") or {}).get("detections")
                if proc_img:
                    # Modo clasico: el servidor envia la imagen ya dibujada.
                    self.update_streaming_frame(proc_img, type_image="base64", tets=False)
                elif detections is not None:
                    # Modo DIRECTO: dibujar las detecciones sobre el frame enviado.
                    self._draw_detections_and_show(detections, frame_respuesta)
                if tel is not None:
                    tel.marcar('t_pintado')
                    tel.dato('personas', len(detections or []))
                estado_tel = 'ok'
                # FPS REAL de la IA: frames procesados por el servidor/seg.
                self._ai_fps_count += 1
                _now = time.time()
                if _now - self._ai_fps_t0 >= 1.0:
                    self.current_fps = self._ai_fps_count
                    self.text_fps.setText(f"IA: {self._ai_fps_count} FPS")
                    self._ai_fps_count = 0
                    self._ai_fps_t0 = _now
            if data["status"] == "error":
                raise Exception(data.get("message", "Error del servidor"))
        except Exception as e:
            print(f"on_text_message_received error: {e}")
        finally:
            if tel is not None:
                tel.cerrar(estado_tel)
            # Si ya salio la siguiente, esta en vuelo: no liberar el envio
            # (serian dos en vuelo, y la cola del servidor no lo espera).
            if not enviada:
                self.can_send_next_frame = True

    def reconnect_socket(self, data):
        self.olvidar_captura_guardada()
        self.can_send_next_frame = True
        self.btn_smart.setEnabled(True)
        self.loop_show_result()
        self._sync_vlm_model()

    def diconect_socket(self, data):
        self.olvidar_captura_guardada()
        self.btn_smart.setEnabled(False)

    def _set_establecimiento(self, nombre):
        """Recibe el cambio del selector de la barra.

        SIN persistencia a proposito: el local no se recuerda entre sesiones,
        para que nadie se quede alertando al establecimiento de ayer.
        """
        self.establecimiento = str(nombre or "").strip()

    def _save_all(self, key, value):
        if self.callback_save_data:
            self.callback_save_data(self.index, key, value)

    def save_point(self, roi, roi_boolean,
                   order_zone=None, order_zone_boolean=False,
                   delivery_zone=None, delivery_zone_boolean=False):
        # Sincroniza estado interno con lo que el label acaba de emitir
        if order_zone is not None:
            self.order_zone = order_zone
            self.order_zone_boolean = bool(order_zone_boolean)
        if delivery_zone is not None:
            self.delivery_zone = delivery_zone
            self.delivery_zone_boolean = bool(delivery_zone_boolean)
        if self.callback_save_data:
            self.callback_save_data(self.index, "roi",         roi)
            self.callback_save_data(self.index, "roi_boolean", roi_boolean)
            if order_zone is not None:
                self.callback_save_data(self.index, "order_zone",         order_zone)
                self.callback_save_data(self.index, "order_zone_boolean", order_zone_boolean)
            if delivery_zone is not None:
                self.callback_save_data(self.index, "delivery_zone",         delivery_zone)
                self.callback_save_data(self.index, "delivery_zone_boolean", delivery_zone_boolean)
