from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QPixmap, QMouseEvent, QPainter, QBrush, QPen, QColor

from .flecha_direccion import INDICE_MEDIO, FlechaDireccion
from .linea_cruce import LineaCruce
from .puerta_direccional import PuertaDireccional


# Colores BGR-independientes para Qt (RGB)
_BLUE  = QColor(40, 120, 255)   # Toma de orden
_RED   = QColor(255, 60, 60)    # Entrega de plato
_NARANJA = QColor(255, 160, 40)  # Áreas de anaquel (modo Anaqueles)
# Gris frio para la zona del PERSONAL (modo comedor). Se elige gris a
# proposito: es la unica zona que NO cuenta gente, y en pantalla tiene que
# leerse como "aqui no pasa nada" al lado del amarillo del ROI, el azul de la
# toma de orden y el rojo de la entrega de plato.
_GRIS  = QColor(165, 170, 180)


class Interactive_imageLabel(QLabel):

    # Personas + Toma de orden + Entrega de plato
    point_change = Signal(list, bool, list, bool, list, bool)
    # Flecha de direccion (PermanenciaArea en modo puerta). Va en su PROPIA
    # senal a proposito: `point_change` tiene seis argumentos y la usan los
    # cuatro clientes; anadirle uno mas los rompe a todos de golpe.
    flecha_change = Signal(list, bool)
    # Linea de cruce (PermanenciaArea, lectura 'linea'). Tambien en su
    # propia senal, por la misma razon que la flecha.
    linea_change = Signal(list, bool)
    # Puerta direccional: la MISMA linea de cruce rotulada con la flecha
    # doble ENTRADA/SALIDA. Senal propia aunque comparta geometria con
    # `linea_change`: el recuadro necesita saber CUAL de las dos herramientas
    # movio el operador para guardar su interruptor y no encender la otra.
    puerta_change = Signal(list, bool)
    # Areas de anaquel (modo Anaqueles): lista de N poligonos en espacio
    # 0..1000 + activo. Senal PROPIA por la misma razon que la flecha:
    # `point_change` tiene seis argumentos fijos y la usan varios clientes.
    anaqueles_change = Signal(list, bool)
    # Zona del PERSONAL (modo comedor Hummus): poligono en espacio 0..1000 +
    # activo. Senal PROPIA, por la MISMA razon que la flecha, la linea y los
    # anaqueles: `point_change` tiene seis argumentos y la usan los cuatro
    # clientes; anadirle un septimo los rompe a todos de golpe.
    personal_change = Signal(list, bool)

    def __init__(self, parent=None,
                 roi=None,
                 roi_active=False,
                 order_zone=None,
                 order_zone_active=False,
                 delivery_zone=None,
                 delivery_zone_active=False,
                 personal_zone=None,
                 personal_zone_active=False,
                 **kwargs):
        super().__init__(parent)
        self.setMouseTracking(True)

        # Compatibilidad con dicts que vienen como '*_activate'
        roi_active_val            = kwargs.get('roi_activate', roi_active)
        order_zone_active_val     = kwargs.get('order_zone_activate', order_zone_active)
        delivery_zone_active_val  = kwargs.get('delivery_zone_activate', delivery_zone_active)
        personal_zone_active_val  = kwargs.get('personal_zone_activate', personal_zone_active)

        # ROI de personas (amarillo)
        self.show_points = bool(roi_active_val)
        self.points_hidden = False           # toggle visual global
        self.points_editable = True          # compat
        self.points = self.list_to_qpoints(roi or [])

        # ROI de Toma de orden (azul)
        self.order_zone = self.list_to_qpoints(order_zone or [])
        self.order_zone_active = bool(order_zone_active_val)

        # ROI de Entrega de plato (rojo)
        self.delivery_zone = self.list_to_qpoints(delivery_zone or [])
        self.delivery_zone_active = bool(delivery_zone_active_val)

        # Zona del PERSONAL (gris, modo comedor Hummus). Es la tercera zona
        # del comedor: quien esta dentro trabaja ahi y NO cuenta como cliente.
        # Nace apagada, asi que los otros clientes no la ven.
        self.personal_zone = self.list_to_qpoints(personal_zone or [])
        self.personal_zone_active = bool(personal_zone_active_val)

        # Flecha de direccion "hacia adentro" (PermanenciaArea, modo puerta).
        # Inactiva salvo que se encienda: los otros clientes no la ven.
        self.flecha = FlechaDireccion(
            kwargs.get('flecha'),
            kwargs.get('flecha_activate', kwargs.get('flecha_active', False)))
        # Agarre de la flecha en curso (0 cola, 1 punta, INDICE_MEDIO la
        # flecha entera, -1 ninguno). Aparte de `active_point_index` para no
        # enredarse con la edicion de poligonos, que tiene su propia
        # maquinaria.
        self.flecha_index = -1
        # Ultima posicion (0..1000) de un arrastre de flecha entera: el
        # cuerpo se mueve por DELTAS para que no de un salto al cogerlo
        # lejos del centro.
        self._flecha_prev = None
        # Extremos tambien por delta: se guarda el desfase agarre-extremo
        # para que coger el borde del circulo no teletransporte el punto
        # al cursor (hasta ~10 px de salto que ademas se guardaban).
        self._flecha_agarre = None
        # Foto de los puntos al empezar un agarre de la flecha: el press
        # de un doble clic tambien arrastra, y la inversion debe partir de
        # lo que habia ANTES de ese arrastre accidental, no descalibrar.
        self._flecha_puntos_press = None

        # Linea de cruce (PermanenciaArea, lectura 'linea'): se dibuja SOBRE
        # el umbral y el servidor mira de que lado a que lado se paso.
        # Comparte con la flecha toda la maquinaria de arrastre (los indices
        # `flecha_index` y compania sirven para "el utensilio activo").
        self.linea = LineaCruce(
            kwargs.get('linea_cruce'),
            kwargs.get('linea_cruce_activate',
                       kwargs.get('linea_cruce_active', False)))

        # Puerta direccional: la misma linea del umbral, pero rotulada con
        # una flecha doble ENTRADA/SALIDA en vez de con DENTRO/FUERA. Es el
        # utensilio que se le ofrece a quien calibra una puerta pensando en
        # por donde entra la gente y no en que lado es el interior.
        #
        # Geometria IDENTICA a la linea (dos puntos, DENTRO a la derecha de
        # P1->P2), asi que viaja al servidor por el mismo campo y la lee el
        # mismo detector. Guarda sus puntos APARTE de los de la linea: son
        # dos calibraciones distintas de la misma camara y cambiar de
        # herramienta no puede pisar la anterior.
        self.puerta = PuertaDireccional(
            kwargs.get('puerta_direccional'),
            kwargs.get('puerta_direccional_activate',
                       kwargs.get('puerta_direccional_active', False)))
        # Utensilio agarrado en el press, para que el move y el release
        # actuen sobre el mismo aunque algo conmute a media accion.
        self._gadget_press = None

        # Areas de anaquel (modo Anaqueles): N cuadrilateros arrastrables,
        # uno por nivel del estante. Cada area reusa la MISMA mecanica de
        # vertices que los poligonos fijos, pero con sus propios indices de
        # arrastre (_anq_*) para no tocar la maquinaria compartida que usan
        # los otros clientes.
        self.anaqueles = [self.list_to_qpoints(a)
                          for a in (kwargs.get('anaqueles') or [])]
        self.anaqueles_active = bool(
            kwargs.get('anaqueles_activate',
                       kwargs.get('anaqueles_active', False)))
        # Etiquetas por area (paralelas a self.anaqueles). Las fija el
        # cliente con la numeracion GLOBAL entre camaras ("Anaquel 7"); sin
        # etiqueta se cae al indice local ("Área N"), que era lo de antes.
        self.anaqueles_etiquetas = [str(e) for e in
                                    (kwargs.get('anaqueles_etiquetas') or [])]
        self._anq_area = -1          # area cuyo vertice se arrastra
        self._anq_vert = -1          # indice del vertice arrastrado

        # Qué conjunto de puntos edita el mouse:
        # 'roi' | 'order' | 'delivery' | 'personal'
        self.edit_target = 'roi'

        self.active_point_index = -1
        self.point_radius = 10
        self.current_pixmap = QPixmap()

        if not self.show_points:
            self.hide_points()

    # ------------------------------------------------------------
    # CONTROL DE VISIBILIDAD
    # ------------------------------------------------------------
    def hide_points(self):
        self.show_points = False
        self.update()

    def show_points_fn(self):
        self.show_points = True
        self.update()

    def toggle_points(self):
        self.show_points = not self.show_points
        self.update()

    def toggle_points_visibility(self):
        """Oculta/muestra TODO el overlay sin desactivar los ROI."""
        self.points_hidden = not self.points_hidden
        self.update()

    def toggle_order_zone(self, state: bool = None):
        if state is None:
            self.order_zone_active = not self.order_zone_active
        else:
            self.order_zone_active = bool(state)
        self.update()

    def toggle_delivery_zone(self, state: bool = None):
        if state is None:
            self.delivery_zone_active = not self.delivery_zone_active
        else:
            self.delivery_zone_active = bool(state)
        self.update()

    def toggle_personal_zone(self, state: bool = None):
        """Enciende/apaga la zona del personal (sin borrarla).

        Avisa por `personal_change` porque apagar la zona cambia lo que el
        servidor recibe: `get_personal_zone_coordinates` pasa a devolver lista
        vacia, que es la senal de «no dibujada».
        """
        if state is None:
            self.personal_zone_active = not self.personal_zone_active
        else:
            self.personal_zone_active = bool(state)
        self.update()
        self._emitir_personal()

    def toggle_flecha(self, state: bool = None):
        """Enciende/apaga la flecha de direccion."""
        if state is None:
            self.flecha.activa = not self.flecha.activa
        else:
            self.flecha.activa = bool(state)
        self.flecha_index = -1
        self.update()

    def toggle_linea(self, state: bool = None):
        """Enciende/apaga la linea de cruce."""
        if state is None:
            self.linea.activa = not self.linea.activa
        else:
            self.linea.activa = bool(state)
        self.flecha_index = -1
        self.update()

    def toggle_puerta(self, state: bool = None):
        """Enciende/apaga la puerta direccional (flecha doble + umbral)."""
        if state is None:
            self.puerta.activa = not self.puerta.activa
        else:
            self.puerta.activa = bool(state)
        self.flecha_index = -1
        self.update()

    # ── Areas de anaquel ─────────────────────────────────────────────
    def toggle_anaqueles(self, state: bool = None):
        """Enciende/apaga las areas de anaquel (sin borrarlas)."""
        if state is None:
            self.anaqueles_active = not self.anaqueles_active
        else:
            self.anaqueles_active = bool(state)
        self._anq_area = self._anq_vert = -1
        self.update()

    def agregar_anaquel(self):
        """Anade un area nueva (cuadrilatero arrastrable) y la activa.

        Las areas nacen APILADAS en bandas horizontales, como los niveles
        de un estante: la primera arriba, las siguientes debajo. El
        operador solo tiene que arrastrar las esquinas a su nivel.
        """
        n = len(self.anaqueles)
        y0 = 120 + (n * 170) % 720
        self.anaqueles.append([
            QPoint(150, y0), QPoint(850, y0),
            QPoint(850, y0 + 130), QPoint(150, y0 + 130)])
        self.anaqueles_active = True
        self.update()
        self._emitir_anaqueles()

    def quitar_anaquel(self):
        """Quita la ULTIMA area (los indices de las demas no cambian: la
        identidad de los contadores del servidor es el indice)."""
        if not self.anaqueles:
            return
        self.anaqueles.pop()
        self._anq_area = self._anq_vert = -1
        self.update()
        self._emitir_anaqueles()

    def set_anaqueles(self, data, activa: bool = None):
        self.anaqueles = [self.list_to_qpoints(a) for a in (data or [])]
        if activa is not None:
            self.anaqueles_active = bool(activa)
        self.update()

    def set_anaqueles_etiquetas(self, etiquetas):
        """Las etiquetas visibles de cada area (numeracion global)."""
        self.anaqueles_etiquetas = [str(e) for e in (etiquetas or [])]
        self.update()

    def anaqueles_a_lista(self):
        return [self.qpoints_to_list(a) for a in self.anaqueles]

    def get_anaqueles_coordinates(self, image_width: int, image_height: int):
        """Lista de poligonos en pixeles del frame. VACIA si la herramienta
        esta apagada: el servidor distingue «sin areas» de «areas en 0,0»."""
        if not self.anaqueles_active:
            return []
        return [self._coords_pixels(a, image_width, image_height)
                for a in self.anaqueles if len(a) >= 3]

    def _emitir_anaqueles(self):
        self.anaqueles_change.emit(self.anaqueles_a_lista(),
                                   self.anaqueles_active)

    # ── Zona del personal (modo comedor) ─────────────────────────────
    def _emitir_personal(self):
        self.personal_change.emit(self.qpoints_to_list(self.personal_zone),
                                  self.personal_zone_active)

    def _anq_vertice_en(self, pos):
        """`(area, indice)` del vertice de anaquel bajo `pos`, o `None`."""
        if not self.anaqueles_active:
            return None
        for a, pts in enumerate(self.anaqueles):
            for i, pt in enumerate(pts):
                p_pix = self.get_scaled_point(pt)
                dx = pos.x() - p_pix.x()
                dy = pos.y() - p_pix.y()
                if (dx * dx + dy * dy) ** 0.5 < self.point_radius:
                    return a, i
        return None

    #: Los utensilios de direccion, de mas a menos prioritario. El orden
    #: importa en dos sitios: quien manda si por un descuido hay dos
    #: encendidos, y quien se prueba primero bajo el cursor. La puerta va la
    #: primera por ser la mas explicita —dice ENTRADA y SALIDA con todas sus
    #: letras—, asi que si esta encendida es la que el operador eligio.
    #: El cliente los mantiene EXCLUYENTES; esto es solo el desempate.
    def _gadgets(self):
        return (self.puerta, self.linea, self.flecha)

    def _gadget_activo(self):
        """El utensilio de direccion que manda, o `None` si ninguno."""
        for g in self._gadgets():
            if g.activa:
                return g
        return None

    def _emitir_gadget(self, gadget):
        if gadget is self.puerta:
            senal = self.puerta_change
        elif gadget is self.linea:
            senal = self.linea_change
        else:
            senal = self.flecha_change
        senal.emit(gadget.a_lista(), gadget.activa)

    def gadget_direccion_en(self, pos) -> bool:
        """¿`pos` cae sobre alguno de los utensilios de direccion ACTIVOS?

        Lo consulta el recuadro para NO maximizar con el doble clic que va
        a invertir un utensilio de direccion.
        """
        for g in self._gadgets():
            if g.activa and g.indice_en(pos, self.get_scaled_point,
                                        self.point_radius) != -1:
                return True
        return False

    def invertir_flecha(self):
        """Intercambia ENTRADA y SALIDA sin mover la flecha del sitio.

        Existe como metodo publico para que el cliente lo ofrezca tambien
        en un menu: el doble clic es rapido pero nadie lo descubre solo.
        """
        if not self.flecha.activa:
            return
        self._invertir(self.flecha)

    def invertir_linea(self):
        """Voltea DENTRO y FUERA sin mover la linea del umbral."""
        if not self.linea.activa:
            return
        self._invertir(self.linea)

    def invertir_puerta(self):
        """Intercambia ENTRADA y SALIDA sin mover la puerta del umbral."""
        if not self.puerta.activa:
            return
        self._invertir(self.puerta)

    def _invertir(self, gadget):
        gadget.invertir()
        self.flecha_index = -1
        self._gadget_press = None
        self.update()
        self._emitir_gadget(gadget)

    def set_edit_target(self, target: str):
        if target in ('roi', 'order', 'delivery', 'personal'):
            self.edit_target = target
            self.active_point_index = -1
            self.update()

    # ------------------------------------------------------------
    # PIXMAP
    # ------------------------------------------------------------
    def setPixmap(self, pixmap):
        self.current_pixmap = pixmap
        super().setPixmap(pixmap)
        self.update()

    # ------------------------------------------------------------
    # CONVERSIÓN DE COORDENADAS (0-1000 ↔ pixeles)
    # ------------------------------------------------------------
    def get_scaled_point(self, point_percentage: QPoint) -> QPoint:
        width = self.width()
        height = self.height()
        x = int(point_percentage.x() * width / 1000)
        y = int(point_percentage.y() * height / 1000)
        return QPoint(x, y)

    def get_percentage_point(self, point_pixel: QPoint) -> QPoint:
        width = self.width()
        height = self.height()
        if width == 0 or height == 0:
            return QPoint(0, 0)
        x = int(point_pixel.x() * 1000 / width)
        y = int(point_pixel.y() * 1000 / height)
        x = max(0, min(x, 1000))
        y = max(0, min(y, 1000))
        return QPoint(x, y)

    # ------------------------------------------------------------
    # DIBUJO
    # ------------------------------------------------------------
    def _draw_polygon(self, painter: QPainter, pts, line_color: QColor,
                      fill_color: QColor, target_name: str,
                      etiqueta: str = ""):
        if not pts:
            return
        last = None
        for i, pt in enumerate(pts):
            pix = self.get_scaled_point(pt)
            if last is not None:
                painter.setPen(QPen(line_color, 2, Qt.SolidLine))
                painter.drawLine(last, pix)
            last = pix
            color = Qt.white if (self.edit_target == target_name
                                 and i == self.active_point_index) else fill_color
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pix, self.point_radius, self.point_radius)
        if last is not None and len(pts) > 1:
            first = self.get_scaled_point(pts[0])
            painter.setPen(QPen(line_color, 2, Qt.SolidLine))
            painter.drawLine(last, first)
        # Rotulo junto al primer vertice, con el mismo doble trazo que las
        # areas de anaquel (negro grueso debajo) para que se lea sobre
        # cualquier fondo. Opcional: las zonas que ya existian no lo llevan.
        if etiqueta:
            origen = self.get_scaled_point(pts[0])
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.black, 4))
            painter.drawText(origen.x() + 14, origen.y() - 9, etiqueta)
            painter.setPen(QPen(line_color, 2))
            painter.drawText(origen.x() + 13, origen.y() - 10, etiqueta)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.points_hidden:
            return
        any_visible = (self.show_points or self.order_zone_active
                       or self.delivery_zone_active or self.personal_zone_active
                       or self.flecha.activa or self.puerta.activa
                       or self.linea.activa or self.anaqueles_active)
        if not any_visible:
            return
        if self.width() < 1 or self.height() < 1 or self.current_pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # ROI de personas (amarillo, líneas blancas punteadas)
        if self.show_points and self.points:
            last_p = None
            for i, p_perc in enumerate(self.points):
                p_pix = self.get_scaled_point(p_perc)
                if last_p is not None:
                    painter.setPen(QPen(Qt.white, 1, Qt.DashLine))
                    painter.drawLine(last_p, p_pix)
                last_p = p_pix
                color = Qt.red if i == self.active_point_index and self.edit_target == 'roi' else Qt.yellow
                painter.setPen(QPen(Qt.black, 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(p_pix, self.point_radius, self.point_radius)
            if last_p is not None and len(self.points) > 1:
                first_p = self.get_scaled_point(self.points[0])
                painter.setPen(QPen(Qt.white, 1, Qt.DashLine))
                painter.drawLine(last_p, first_p)

        # Toma de orden (azul)
        if self.order_zone_active:
            self._draw_polygon(painter, self.order_zone, _BLUE, _BLUE, 'order')

        # Entrega de plato (rojo)
        if self.delivery_zone_active:
            self._draw_polygon(painter, self.delivery_zone, _RED, _RED, 'delivery')

        # Personal (gris): la unica zona que EXCLUYE gente del conteo, asi que
        # va rotulada; confundirla con la de entrega falsearia las ventas.
        if self.personal_zone_active:
            self._draw_polygon(painter, self.personal_zone, _GRIS, _GRIS,
                               'personal', "Personal")

        # Areas de anaquel (naranja, numeradas): un cuadrilatero por nivel.
        if self.anaqueles_active:
            for a, pts in enumerate(self.anaqueles):
                if not pts:
                    continue
                last = None
                for i, pt in enumerate(pts):
                    pix = self.get_scaled_point(pt)
                    if last is not None:
                        painter.setPen(QPen(_NARANJA, 2, Qt.SolidLine))
                        painter.drawLine(last, pix)
                    last = pix
                    activo = (a == self._anq_area and i == self._anq_vert)
                    painter.setPen(QPen(Qt.black, 2))
                    painter.setBrush(QBrush(Qt.white if activo else _NARANJA))
                    painter.drawEllipse(pix, self.point_radius,
                                        self.point_radius)
                if last is not None and len(pts) > 1:
                    first = self.get_scaled_point(pts[0])
                    painter.setPen(QPen(_NARANJA, 2, Qt.SolidLine))
                    painter.drawLine(last, first)
                # Etiqueta del nivel, junto al primer vertice: es lo que el
                # operador compara con los contadores del panel. La fija el
                # cliente (numeracion global entre camaras); sin ella, el
                # indice local de siempre.
                etiqueta = (self.anaqueles_etiquetas[a]
                            if a < len(self.anaqueles_etiquetas)
                            and str(self.anaqueles_etiquetas[a]).strip()
                            else f"Área {a + 1}")
                origen = self.get_scaled_point(pts[0])
                painter.setPen(QPen(Qt.black, 4))
                painter.drawText(origen.x() + 14, origen.y() - 9, etiqueta)
                painter.setPen(QPen(_NARANJA, 2))
                painter.drawText(origen.x() + 13, origen.y() - 10, etiqueta)

        # Los utensilios de direccion, los ULTIMOS: van encima de las zonas
        # para poder cogerlos aunque caigan sobre un vertice del ROI. El
        # indice de arrastre pertenece al utensilio agarrado (o al activo).
        gadget = self._gadget_press or self._gadget_activo()
        for g in self._gadgets():
            g.pintar(painter, self.get_scaled_point, self.point_radius,
                     self.flecha_index if g is gadget else -1)

        painter.end()

    # ------------------------------------------------------------
    # EVENTOS DEL MOUSE
    # ------------------------------------------------------------
    def _interactive(self) -> bool:
        if self.points_hidden:
            return False
        return (self.show_points or self.order_zone_active
                or self.delivery_zone_active or self.personal_zone_active
                or self.flecha.activa or self.puerta.activa
                or self.linea.activa or self.anaqueles_active)

    def _vertice_en(self, pos):
        """`(conjunto, indice)` del vertice de poligono bajo `pos`, o `None`.

        El mismo hit-test para el press y para el doble clic: los dos deben
        resolver igual el solape entre un vertice y el cuerpo de la flecha.
        """
        sets = [
            ('roi',      self.points,         self.show_points),
            ('order',    self.order_zone,     self.order_zone_active),
            ('delivery', self.delivery_zone,  self.delivery_zone_active),
            ('personal', self.personal_zone,  self.personal_zone_active),
        ]
        # Prioriza el conjunto seleccionado en el combo
        sets.sort(key=lambda s: 0 if s[0] == self.edit_target else 1)
        for name, pts, active in sets:
            if not active or not pts:
                continue
            for i, pt in enumerate(pts):
                p_pix = self.get_scaled_point(pt)
                dx = pos.x() - p_pix.x()
                dy = pos.y() - p_pix.y()
                if (dx * dx + dy * dy) ** 0.5 < self.point_radius:
                    return name, i
        return None

    def hay_punto_en(self, pos) -> bool:
        """Hay un punto arrastrable bajo `pos` (coordenadas del visor).

        Lo consultan las barras superpuestas al visor para cederle el
        arrastre cuando un punto queda justo debajo de un boton.
        """
        if not self._interactive():
            return False
        if self.gadget_direccion_en(pos):
            return True
        if self._anq_vertice_en(pos) is not None:
            return True
        for pts, activo in ((self.points, self.show_points),
                            (self.order_zone, self.order_zone_active),
                            (self.delivery_zone, self.delivery_zone_active),
                            (self.personal_zone, self.personal_zone_active)):
            if not activo or not pts:
                continue
            for pt in pts:
                p_pix = self.get_scaled_point(pt)
                dx = pos.x() - p_pix.x()
                dy = pos.y() - p_pix.y()
                if (dx * dx + dy * dy) ** 0.5 < self.point_radius:
                    return True
        return False

    def arrastrando_punto(self) -> bool:
        """Hay un arrastre de punto en curso."""
        return (self.active_point_index != -1 or self.flecha_index != -1
                or self._anq_vert != -1)

    def mousePressEvent(self, event: QMouseEvent):
        if not self._interactive():
            return super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            # El utensilio de direccion se pinta encima, asi que sus
            # EXTREMOS se cogen primero. Su cuerpo en cambio cede ante los
            # vertices de los poligonos: cruza medio visor y si ganara
            # siempre, un vertice de ROI que quedara debajo seria imposible
            # de editar.
            gadget = self._gadget_activo()
            idx = (-1 if gadget is None else
                   gadget.indice_en(event.pos(), self.get_scaled_point,
                                    self.point_radius))
            self._flecha_puntos_press = None
            if idx != -1 and idx != INDICE_MEDIO:
                self._gadget_press = gadget
                self.flecha_index = idx
                self._flecha_puntos_press = gadget.a_lista()
                punto = gadget.puntos[idx]
                perc = self.get_percentage_point(event.pos())
                self._flecha_agarre = QPoint(punto.x() - perc.x(),
                                             punto.y() - perc.y())
                self.update()
                return
            vertice = self._vertice_en(event.pos())
            if vertice is not None:
                self.edit_target, self.active_point_index = vertice
                self.update()
                return
            anq = self._anq_vertice_en(event.pos())
            if anq is not None:
                self._anq_area, self._anq_vert = anq
                self.update()
                return
            if idx == INDICE_MEDIO:
                self._gadget_press = gadget
                self.flecha_index = idx
                self._flecha_puntos_press = gadget.a_lista()
                self._flecha_prev = self.get_percentage_point(event.pos())
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._interactive():
            return super().mouseMoveEvent(event)
        if self.flecha_index != -1 and event.buttons() & Qt.LeftButton:
            gadget = self._gadget_press or self._gadget_activo()
            if gadget is None:
                return super().mouseMoveEvent(event)
            pos = event.pos()
            pos.setX(max(0, min(pos.x(), self.width())))
            pos.setY(max(0, min(pos.y(), self.height())))
            perc = self.get_percentage_point(pos)
            if self.flecha_index == INDICE_MEDIO:
                if self._flecha_prev is not None:
                    gadget.trasladar(perc.x() - self._flecha_prev.x(),
                                     perc.y() - self._flecha_prev.y())
                self._flecha_prev = perc
            else:
                # Con el desfase del agarre: el extremo sigue al raton desde
                # donde se cogio, sin saltar primero al cursor.
                off = self._flecha_agarre or QPoint(0, 0)
                destino = QPoint(
                    max(0, min(perc.x() + off.x(), 1000)),
                    max(0, min(perc.y() + off.y(), 1000)))
                gadget.mover(self.flecha_index, destino)
            self.update()
            return super().mouseMoveEvent(event)
        if self._anq_vert != -1 and event.buttons() & Qt.LeftButton:
            new_pos = event.pos()
            new_pos.setX(max(0, min(new_pos.x(), self.width())))
            new_pos.setY(max(0, min(new_pos.y(), self.height())))
            try:
                self.anaqueles[self._anq_area][self._anq_vert] = \
                    self.get_percentage_point(new_pos)
            except IndexError:
                self._anq_area = self._anq_vert = -1
            self.update()
            return super().mouseMoveEvent(event)
        if self.active_point_index != -1 and event.buttons() & Qt.LeftButton:
            new_pos = event.pos()
            new_pos.setX(max(0, min(new_pos.x(), self.width())))
            new_pos.setY(max(0, min(new_pos.y(), self.height())))
            new_perc = self.get_percentage_point(new_pos)
            target = self.edit_target
            if target == 'roi':
                self.points[self.active_point_index] = new_perc
            elif target == 'order':
                self.order_zone[self.active_point_index] = new_perc
            elif target == 'delivery':
                self.delivery_zone[self.active_point_index] = new_perc
            elif target == 'personal':
                self.personal_zone[self.active_point_index] = new_perc
            self.update()
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # Doble clic en cualquier parte del utensilio de direccion: la
        # flecha invierte entrada ↔ salida; la linea voltea DENTRO ↔ FUERA.
        gadget = self._gadget_activo()
        if (self._interactive() and event.button() == Qt.LeftButton
                and gadget is not None):
            idx = gadget.indice_en(event.pos(), self.get_scaled_point,
                                   self.point_radius)
            # El cuerpo cede ante un vertice de poligono, la MISMA prioridad
            # que el press: dos ajustes rapidos de un vertice llegan como
            # doble clic y no pueden acabar invirtiendo nada.
            if idx == INDICE_MEDIO and self._vertice_en(event.pos()) is not None:
                idx = -1
            if idx != -1:
                # El press de este mismo doble clic pudo arrastrar unos
                # pixeles el agarre; se restaura lo que habia antes para
                # que invertir no descalibre.
                if self._flecha_puntos_press is not None:
                    gadget.poner(self._flecha_puntos_press)
                self._invertir(gadget)
                return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._interactive():
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.flecha_index != -1:
            gadget = self._gadget_press or self._gadget_activo()
            self.flecha_index = -1
            self._flecha_prev = None
            self._flecha_agarre = None
            self._gadget_press = None
            # `_flecha_puntos_press` sobrevive al release A PROPOSITO: el
            # doble clic llega despues y lo necesita para no descalibrar.
            self.update()
            if gadget is not None:
                self._emitir_gadget(gadget)
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self._anq_vert != -1:
            self._anq_area = self._anq_vert = -1
            self.update()
            self._emitir_anaqueles()
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.active_point_index != -1:
            objetivo = self.edit_target
            self.active_point_index = -1
            self.update()
            if objetivo == 'personal':
                # La zona del personal NO viaja en `point_change`: esa senal
                # tiene seis argumentos fijos y la usan los cuatro clientes.
                # Va en la suya, como la flecha, la linea y los anaqueles.
                self._emitir_personal()
            else:
                self.point_change.emit(
                    self.qpoints_to_list(self.points),         self.show_points,
                    self.qpoints_to_list(self.order_zone),     self.order_zone_active,
                    self.qpoints_to_list(self.delivery_zone),  self.delivery_zone_active,
                )
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------
    # COORDENADAS EN PIXELES ORIGINALES
    # ------------------------------------------------------------
    def _coords_pixels(self, qpoints, image_width, image_height):
        if image_width <= 0 or image_height <= 0:
            return []
        return [
            [int(p.x() * image_width / 1000), int(p.y() * image_height / 1000)]
            for p in qpoints
        ]

    def get_coordinates(self, image_width: int, image_height: int):
        return self._coords_pixels(self.points, image_width, image_height)

    def get_order_zone_coordinates(self, image_width: int, image_height: int):
        return self._coords_pixels(self.order_zone, image_width, image_height)

    def get_delivery_zone_coordinates(self, image_width: int, image_height: int):
        return self._coords_pixels(self.delivery_zone, image_width, image_height)

    def get_personal_zone_coordinates(self, image_width: int, image_height: int):
        """Poligono del personal en pixeles del frame.

        Lista VACIA si la zona esta apagada, igual que
        `get_anaqueles_coordinates`: el servidor distingue «no dibujada» de
        «dibujada en 0,0», y sin esa distincion apagar la zona seguiria
        excluyendo del conteo a quien pasara por ese rincon.
        """
        if not self.personal_zone_active:
            return []
        return self._coords_pixels(self.personal_zone, image_width,
                                   image_height)

    def get_flecha_coordinates(self, image_width: int, image_height: int):
        """`[[x_cola, y_cola], [x_punta, y_punta]]` en pixeles del frame.

        Lista VACIA si la flecha esta apagada o es demasiado corta para
        significar algo: el servidor debe poder distinguir «no marcada» de
        «marcada apuntando a ninguna parte».
        """
        if not self.flecha.es_utilizable():
            return []
        return self.flecha.coordenadas_pixel(image_width, image_height)

    def get_linea_coordinates(self, image_width: int, image_height: int):
        """`[[x1, y1], [x2, y2]]` en pixeles del frame; DENTRO a la derecha
        de P1→P2. Lista VACIA si la linea esta apagada o es demasiado corta:
        el servidor debe poder distinguir «no marcada» de «inservible».
        """
        if not self.linea.es_utilizable():
            return []
        return self.linea.coordenadas_pixel(image_width, image_height)

    def get_puerta_coordinates(self, image_width: int, image_height: int):
        """`[[x1, y1], [x2, y2]]` del umbral en pixeles; DENTRO —el lado de
        la cabeza verde ENTRADA— a la derecha de P1→P2.

        MISMA forma y MISMA convencion que `get_linea_coordinates`, porque
        es la misma geometria: el cliente la manda por el campo `linea_cruce`
        y la lee el detector de linea de siempre. Lista VACIA si la puerta
        esta apagada o es demasiado corta.
        """
        if not self.puerta.es_utilizable():
            return []
        return self.puerta.coordenadas_pixel(image_width, image_height)

    def get_cruce_coordinates(self, image_width: int, image_height: int):
        """El umbral que rige, venga de la puerta o de la linea.

        Existe para que el recuadro no tenga que preguntar cual de las dos
        herramientas esta encendida cada vez que arma un frame: las dos
        producen el MISMO dato para el MISMO campo del contrato, y lo unico
        que cambia es como se dibuja. Devuelve lista VACIA si no hay ninguna
        utilizable.
        """
        return (self.get_puerta_coordinates(image_width, image_height)
                or self.get_linea_coordinates(image_width, image_height))

    # ------------------------------------------------------------
    # SERIALIZACIÓN
    # ------------------------------------------------------------
    def qpoints_to_list(self, qpoints):
        return [[p.x(), p.y()] for p in qpoints]

    def list_to_qpoints(self, data):
        return [QPoint(c[0], c[1]) for c in (data or [])]

    def set_roi(self, data):
        self.points = self.list_to_qpoints(data)
        self.update()

    def set_order_zone(self, data):
        self.order_zone = self.list_to_qpoints(data)
        self.update()

    def set_delivery_zone(self, data):
        self.delivery_zone = self.list_to_qpoints(data)
        self.update()

    def set_personal_zone(self, data, activa: bool = None):
        """Repone la zona del personal (y opcionalmente su interruptor).

        NO emite `personal_change` a proposito, igual que sus hermanas: se
        llama al restaurar lo guardado, y avisar ahi devolveria al cliente lo
        que el mismo acaba de dar.
        """
        self.personal_zone = self.list_to_qpoints(data)
        if activa is not None:
            self.personal_zone_active = bool(activa)
        self.update()

    def set_flecha(self, data, activa: bool = None):
        self.flecha.poner(data)
        if activa is not None:
            self.flecha.activa = bool(activa)
        self.update()

    def set_linea(self, data, activa: bool = None):
        self.linea.poner(data)
        if activa is not None:
            self.linea.activa = bool(activa)
        self.update()

    def set_puerta(self, data, activa: bool = None):
        self.puerta.poner(data)
        if activa is not None:
            self.puerta.activa = bool(activa)
        self.update()
