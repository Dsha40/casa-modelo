from PySide6.QtWidgets import QStatusBar, QLabel, QWidget, QHBoxLayout, QComboBox, QCheckBox
from PySide6.QtCore import Slot, Qt, Signal

from .custon_btn.btn_footer import BtnIco

#: Nombre del unico modo de este cliente, tal como lo ve el operador Y como
#: viaja al servidor (`/ws/{type_inference}`). Son el MISMO string a
#: proposito: separar etiqueta y valor obliga a mantener una tabla de
#: traduccion, y basta que alguien anada una sin la otra para tener un modo
#: que solo sabe fallar (le paso a HummusVLM y a Restaurante).
MODO_TIENDA = 'Tiendas'

#: Modo Anaqueles (13-ago-2026): interaccion con los niveles del estante
#: (visitas/agarres por area + heatmap propio), para marketing. Mismo string
#: operador/servidor, por la misma razon que MODO_TIENDA. Para cambiar de
#: modo: "Cerrar conexion" habilita el selector.
MODO_ANAQUELES = 'Anaqueles'

#: Permanencia por area/umbral (7-sep-2026). Se ofrece aqui porque el
#: recuadro de este cliente ya trae la puerta direccional, y sin el modo que
#: la lee el boton seria un adorno: dibujar un umbral que no clasifica un
#: solo paso es peor que no ofrecerlo. Es un modo del SERVIDOR —se pide por
#: la URL del websocket— asi que basta con nombrarlo. Va el ULTIMO y NO
#: cambia el modo por defecto: quien no abra el desplegable no nota nada.
MODO_PERMANENCIA = 'PermanenciaArea'

#: Nombres antiguos que pueden venir guardados en `last_inference` de
#: instalaciones anteriores. El servidor los sigue aceptando; aqui se
#: traducen para que el selector no se quede en blanco.
MODOS_HEREDADOS = ('Personal de Amazonas',)


class CustomStatusBar(QStatusBar):
    
    
    inference_type_selected = Signal(str)
    
    
    def __init__(self,
                type_inference_default=None,
                # COMPAT: los llamadores historicos aun pasan estos dos. El
                # selector global de establecimiento se elimino (5-ago-2026),
                # asi que se aceptan y se ignoran para no romper a nadie.
                list_establishment=None,
                selected_establishment_default=None,
        ):
        
        super().__init__(parent=None)
        self.type_inference_default=type_inference_default
        self.setup_ui()
        
        
    def setup_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(35)
        self.setObjectName('FooterBar')
        # Sin stylesheet propio: #FooterBar se viste en styles/global.qss
        # (antes el gris #424242 inline pisaba cualquier tema).
        
        container = QWidget()
        self.container_layout = QHBoxLayout(container)
        self.container_layout.setSpacing(20)
        self.container_layout.setContentsMargins(0,0,0,0)
        "inserción______⤵️_______"
        self.addPermanentWidget(container)
        
        
        
        # El selector GLOBAL de establecimiento se ELIMINO el 5-ago-2026
        # (peticion del operador). El destino de las alertas es POR CAMARA:
        # cada recuadro tiene su propio select "Local" en la barra. Un global
        # aqui solo podia contradecirlo, y ademas reconstruirse mientras el
        # operador elegia — de ahi el "se recarga y no deja seleccionar".
        
        
        """____Indicador del server___"""
        self.msg_label = QLabel('Selecione el tipo de inferencia --->')
        self .indicator = QLabel('●')
        self.indicator.setStyleSheet('color: gray;')
        "inserción______⤵️_______"
        self.container_layout.addWidget(self.indicator)
        self.container_layout.addWidget(self.msg_label)
        self.container_layout.addStretch()
        
        # Cliente DEDICADO A TIENDA: un solo modo, el de analitica de
        # supermercado. Se llama 'Tiendas' desde el 6-ago-2026; antes
        # 'Personal de Amazonas', que describia el pipeline del servidor y no
        # lo que el operador de una tienda cree estar eligiendo.
        #
        # El servidor acepta LOS DOS nombres (ver _build_processor), asi que
        # las instalaciones que tengan guardado el viejo en `last_inference`
        # siguen funcionando y caen en este mismo selector.
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(['Seleccione...', MODO_TIENDA,
                                       MODO_ANAQUELES, MODO_PERMANENCIA])

        # Por defecto seleccionar el modo tienda si no viene otro fijado.
        if self.type_inference_default is None:
            self.type_inference_default = MODO_TIENDA

        if self.type_inference_default is not None:
            nombre = self.type_inference_default
            if nombre in MODOS_HEREDADOS:           # config antigua guardada
                nombre = MODO_TIENDA
            index_inference = self.layout_selector.findText(nombre)
            if index_inference < 0:                 # modo no soportado aqui
                index_inference = self.layout_selector.findText(MODO_TIENDA)
            self.layout_selector.setCurrentIndex(max(0, index_inference))
            self.layout_selector.setDisabled(True)
            
            
        self.layout_selector.currentTextChanged.connect(self._on_selector_changed)
        "inserción______⤵️_______"
        self.container_layout.addWidget(QLabel("Tipos de inferencias:")) # Etiqueta opcional
        self.container_layout.addWidget(self.layout_selector)

        """____Interruptor de ENVÍO a JARVIS (novedades)___
        Reenvia cada alerta al establecimiento seleccionado arriba como una
        novedad de Jarvis365. Su estado se persiste (ver main.py); el checkbox
        es solo la UI. Por defecto ACTIVADO: en este cliente la novedad es el
        producto (el visitante identificado), asi que arrancar sin enviar
        dejaria el establecimiento a ciegas."""
        self.chk_envio_jarvis = QCheckBox('Enviar a Jarvis')
        self.chk_envio_jarvis.setChecked(True)
        self.chk_envio_jarvis.setToolTip(
            'Activar/desactivar el envío de alertas a la API de Jarvis365.\n'
            'Desactivado: las alertas siguen viéndose en el panel lateral,\n'
            'pero NO se envían novedades al establecimiento.')
        self.chk_envio_jarvis.setStyleSheet(
            'QCheckBox { color: #3fb950; font-weight: bold; }')
        self.chk_envio_jarvis.toggled.connect(self._on_envio_jarvis_toggled)
        self.container_layout.addWidget(self.chk_envio_jarvis)

        """____Interruptor de ENVÍO por WHATSAPP (bot 'ava')___
        Reenvia cada alerta como imagen a un grupo de WhatsApp; el envio lo
        hace el servidor. Es GLOBAL para todas las camaras y su estado se
        persiste (ver main.py). Por defecto DESACTIVADO."""
        self.chk_envio_whatsapp = QCheckBox('Enviar por WhatsApp')
        self.chk_envio_whatsapp.setChecked(False)
        self.chk_envio_whatsapp.setToolTip(
            'Activar/desactivar el envío de alertas por WhatsApp.\n'
            'Desactivado: las alertas siguen viéndose en el panel lateral,\n'
            'pero NO se envían imágenes al grupo de WhatsApp.')
        self.chk_envio_whatsapp.setStyleSheet(
            'QCheckBox { color: #8b949e; }')
        self.chk_envio_whatsapp.toggled.connect(self._on_envio_whatsapp_toggled)
        self.container_layout.addWidget(self.chk_envio_whatsapp)

        self.btn_stopconection = BtnIco(ico_path='resource/finish_connection.png', title='Cerrar conexión con el servidor', h=25, w=25)
        "inserción______⤵️_______"
        self.container_layout.addWidget(self.btn_stopconection)

        """____Boton para selección de render_BOX___"""
        self.btn_layout = BtnIco(ico_path='resource/layout.png', title='Divisiones de ventanas: (3x3, 2x2, etc.)')
        "inserción______⤵️_______"
        self.container_layout.addWidget(self.btn_layout)

    def modo_seleccionado(self) -> str:
        """El modo de inferencia que REALMENTE muestra el selector, ya
        normalizado a los que este cliente soporta.

        Existe porque el valor persistido (`last_inference`) puede ser de otro
        negocio: en ese caso el selector cae a 'Tiendas' pero quien arranca la
        conexion seguia usando el valor crudo, asi que la pantalla decia una
        cosa y el websocket enviaba otra (H-14)."""
        texto = self.layout_selector.currentText()
        return '' if texto in ('', 'Seleccione...') else texto
    
    
    
    def _on_selector_changed(self, text):
        if text != 'Seleccione...':
            self.inference_type_selected.emit(text)
            self.layout_selector.setDisabled(True)
        
    
    
    

    def _on_envio_jarvis_toggled(self, activo):
        """Retroalimentacion visual del interruptor de envio a Jarvis."""
        if activo:
            self.chk_envio_jarvis.setStyleSheet(
                'QCheckBox { color: #3fb950; font-weight: bold; }')
            self.showMessage('Envío de alertas a Jarvis ACTIVADO', 4000)
        else:
            self.chk_envio_jarvis.setStyleSheet(
                'QCheckBox { color: #8b949e; }')
            self.showMessage('Envío de alertas a Jarvis DESACTIVADO '
                             '(el panel lateral sigue mostrando alertas)', 5000)

    def _on_envio_whatsapp_toggled(self, activo):
        """Retroalimentacion visual del interruptor de envio por WhatsApp."""
        if activo:
            self.chk_envio_whatsapp.setStyleSheet(
                'QCheckBox { color: #3fb950; font-weight: bold; }')
            self.showMessage('Envío de alertas por WhatsApp ACTIVADO', 4000)
        else:
            self.chk_envio_whatsapp.setStyleSheet(
                'QCheckBox { color: #8b949e; }')
            self.showMessage('Envío de alertas por WhatsApp DESACTIVADO '
                             '(el panel lateral sigue mostrando alertas)', 5000)

    @Slot(bool, str)
    def update_ui(self, is_connected, message):
        if is_connected:
            self.showMessage('Conexión establecida con el servidor', 3000)
            self.indicator.setStyleSheet('color: #3fb950; font-weight: bold;')
            self.msg_label.setStyleSheet('color: #3fb950; font-weight: bold;')
            self.layout_selector.setEnabled(False)
        else:
            self.showMessage('Conexión perdida con el servidor', 3000)
            self.indicator.setStyleSheet('color: #f85149; font-weight: bold;')
            self.msg_label.setStyleSheet('color: #e6edf3; font-weight: bold;')
            self.layout_selector.setEnabled(True)
        self.msg_label.setText(message)
       
    
    @Slot(str)
    def receive_message(self, mesagge):
        self.showMessage(mesagge, 3000)