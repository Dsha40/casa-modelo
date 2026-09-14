"""
Ventana "¿A que servidor me conecto?" — la misma para los cuatro clientes.

Hasta ahora la respuesta estaba en el `.env` de cada carpeta: quien instalaba
el cliente en otra maquina tenia que abrir un fichero de texto y escribir una
IP a mano, y si el servidor cambiaba de direccion el cliente se quedaba
intentando conectar en silencio a una puerta cerrada. En un .exe distribuible
eso directamente no vale: no hay `.env` que editar ni carpeta del proyecto.

Aqui la ventana busca sola (baliza UDP + escaneo, ver
`elde_core.red.descubrimiento_servidores`), muestra lo que hay con su nombre,
sus clientes conectados y su GPU, y deja anadir a mano el que no se pueda
descubrir (el de la VPN). Con "Recordar" no vuelve a preguntar.

Uso:
    dlg = DialogoServidor(cliente='detecciones')
    if dlg.exec():
        url = dlg.ws_url          # 'ws://10.0.0.5:9000/ws'
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from ..red import descubrimiento_servidores as D
from ..red import servidores_guardados as G


class _BuscadorServidores(QThread):
    """El descubrimiento fuera del hilo GUI.

    Con `cancelar()` + `wait()` en el cierre: un QThread destruido mientras
    corre aborta el proceso entero con 0xc0000409, y este barre 254
    direcciones, asi que hay tiempo de sobra para cerrar la ventana antes."""

    sig_estado = Signal(str, int, int)      # texto, hechos, total
    sig_listo = Signal(list)                # list[ServidorELDE]

    def __init__(self, semillas: list[str], parent=None):
        super().__init__(parent)
        self._semillas = list(semillas)
        self._cancelado = False

    def cancelar(self) -> None:
        self._cancelado = True

    def run(self) -> None:
        try:
            encontrados = D.descubrir(
                semillas=self._semillas,
                progreso=lambda t, h=0, tot=0: self.sig_estado.emit(t, h, tot),
                cancelado=lambda: self._cancelado,
            )
        except Exception as e:  # noqa: BLE001  la GUI no se queda colgada
            self.sig_estado.emit(f"Error en la busqueda: {e}", 0, 0)
            encontrados = []
        self.sig_listo.emit(encontrados)


class DialogoServidor(QDialog):
    """Elegir el servidor de inferencia al que conectar este cliente."""

    _COLUMNAS = ["Servidor", "Direccion", "Puerto", "Clientes", "GPU",
                 "Detectado por"]

    def __init__(self, cliente: str = "", parent=None,
                 titulo: str = "Conectar a un servidor ELDE"):
        super().__init__(parent)
        self.cliente = cliente or ""
        self.setWindowTitle(titulo)
        self.resize(860, 480)
        self.seleccionado: D.ServidorELDE | None = None
        self._encontrados: list[D.ServidorELDE] = []
        self._buscador: _BuscadorServidores | None = None
        self._construir()
        self.buscar()

    # ------------------------------------------------------------ resultado
    @property
    def ws_url(self) -> str:
        return self.seleccionado.ws_url if self.seleccionado else ""

    @property
    def recordar(self) -> bool:
        return self.chk_recordar.isChecked()

    # ------------------------------------------------------------------- UI
    def _construir(self) -> None:
        self.setStyleSheet("""
            QDialog{background:#0f1115;color:#e6e9ef;
                    font-family:'Segoe UI',sans-serif}
            QLabel{color:#e6e9ef}
            QLabel#sub{color:#8b93a3;font-size:12px}
            QTableWidget{background:#1a1d24;color:#e6e9ef;
                         gridline-color:#2a2f3a;border:1px solid #2a2f3a;
                         selection-background-color:#00a8e8}
            QHeaderView::section{background:#232733;color:#e6e9ef;border:0;
                                 padding:6px;font-weight:bold}
            QLineEdit{background:#1a1d24;color:#e6e9ef;border:1px solid #2a2f3a;
                      border-radius:6px;padding:6px 10px}
            QPushButton{background:#00a8e8;color:#fff;font-weight:bold;
                        border:none;border-radius:6px;padding:8px 18px}
            QPushButton:hover{background:#33bff0}
            QPushButton:disabled{background:#3a3f4a;color:#888}
            QPushButton#sec{background:#2a2f3a;color:#e6e9ef}
            QPushButton#sec:hover{background:#3a4150}
            QProgressBar{background:#1a1d24;border:1px solid #2a2f3a;
                         border-radius:6px;height:6px;text-align:center}
            QProgressBar::chunk{background:#00a8e8;border-radius:6px}
            QCheckBox{color:#8b93a3}
        """)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(20, 16, 20, 14)
        raiz.setSpacing(8)

        titulo = QLabel("Servidores ELDE en la red")
        titulo.setStyleSheet("font-size:18px;font-weight:800")
        raiz.addWidget(titulo)
        sub = QLabel("Se busca solo al abrir. Si el servidor esta en otra red "
                     "(VPN, otra sede), anadelo abajo por su direccion.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        raiz.addWidget(sub)

        self.tabla = QTableWidget(0, len(self._COLUMNAS))
        self.tabla.setHorizontalHeaderLabels(self._COLUMNAS)
        self.tabla.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabla.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla.verticalHeader().setVisible(False)
        cab = self.tabla.horizontalHeader()
        cab.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(self._COLUMNAS)):
            cab.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.tabla.itemSelectionChanged.connect(self._al_seleccionar)
        self.tabla.itemDoubleClicked.connect(lambda *_: self._conectar())
        raiz.addWidget(self.tabla, 1)

        self.barra = QProgressBar()
        self.barra.setTextVisible(False)
        raiz.addWidget(self.barra)
        self.estado = QLabel("")
        self.estado.setObjectName("sub")
        raiz.addWidget(self.estado)

        # -- anadir a mano --
        manual = QHBoxLayout()
        manual.addWidget(QLabel("Anadir por direccion:"))
        self.txt_host = QLineEdit()
        self.txt_host.setPlaceholderText("192.168.1.50   o   servidor.local")
        self.txt_host.returnPressed.connect(self._anadir_manual)
        manual.addWidget(self.txt_host, 1)
        self.txt_puerto = QLineEdit(str(D.PUERTO_SERVIDOR))
        self.txt_puerto.setFixedWidth(70)
        self.txt_puerto.setAlignment(Qt.AlignCenter)
        self.txt_puerto.returnPressed.connect(self._anadir_manual)
        manual.addWidget(self.txt_puerto)
        self.btn_anadir = QPushButton("Anadir")
        self.btn_anadir.setObjectName("sec")
        self.btn_anadir.clicked.connect(self._anadir_manual)
        manual.addWidget(self.btn_anadir)
        raiz.addLayout(manual)

        # -- pie --
        pie = QHBoxLayout()
        self.chk_recordar = QCheckBox("Recordar y no volver a preguntar")
        self.chk_recordar.setChecked(True)
        pie.addWidget(self.chk_recordar)
        pie.addStretch(1)
        self.btn_buscar = QPushButton("Buscar de nuevo")
        self.btn_buscar.setObjectName("sec")
        self.btn_buscar.clicked.connect(self.buscar)
        pie.addWidget(self.btn_buscar)
        self.btn_cancelar = QPushButton("Salir")
        self.btn_cancelar.setObjectName("sec")
        self.btn_cancelar.clicked.connect(self.reject)
        pie.addWidget(self.btn_cancelar)
        self.btn_conectar = QPushButton("Conectar")
        self.btn_conectar.setEnabled(False)
        self.btn_conectar.setDefault(True)
        self.btn_conectar.clicked.connect(self._conectar)
        pie.addWidget(self.btn_conectar)
        raiz.addLayout(pie)

    # -------------------------------------------------------------- busqueda
    def buscar(self) -> None:
        if self._buscador is not None and self._buscador.isRunning():
            return
        self.btn_buscar.setEnabled(False)
        self.barra.setRange(0, 0)              # indeterminada mientras arranca
        self.estado.setText("Buscando…")
        self.tabla.setRowCount(0)
        self._encontrados = []
        self._buscador = _BuscadorServidores(G.manuales(), self)
        self._buscador.sig_estado.connect(self._al_progresar)
        self._buscador.sig_listo.connect(self._al_terminar)
        self._buscador.start()

    def _al_progresar(self, texto: str, hechos: int, total: int) -> None:
        self.estado.setText(texto)
        if total > 0:
            self.barra.setRange(0, total)
            self.barra.setValue(hechos)
        else:
            self.barra.setRange(0, 0)

    def _al_terminar(self, encontrados: list) -> None:
        self._encontrados = list(encontrados)
        self.barra.setRange(0, 1)
        self.barra.setValue(1)
        self.btn_buscar.setEnabled(True)
        self._pintar()
        if not self._encontrados:
            self.estado.setText(
                "No se encontro ningun servidor. Comprueba que el servidor "
                "este encendido, o anade su direccion abajo.")
        else:
            n = len(self._encontrados)
            self.estado.setText(
                f"{n} servidor(es). Elige uno y pulsa Conectar.")

    def _pintar(self) -> None:
        self.tabla.setRowCount(len(self._encontrados))
        for fila, s in enumerate(self._encontrados):
            celdas = [
                s.nombre or ("Este equipo" if s.local else s.host),
                s.host,
                str(s.puerto),
                "—" if s.clientes < 0 else str(s.clientes),
                s.gpu or "—",
                {"baliza": "baliza", "escaneo": "escaneo",
                 "semilla": "guardado"}.get(s.origen, s.origen),
            ]
            for col, texto in enumerate(celdas):
                item = QTableWidgetItem(texto)
                if not s.en_linea:
                    # Se muestra igual, apagado: que un servidor guardado
                    # desaparezca de la lista deja al operador sin saber si
                    # esta caido o si alguien lo borro.
                    item.setForeground(Qt.gray)
                self.tabla.setItem(fila, col, item)
        # Preseleccionar: el que ya estaba guardado para este cliente, y si no,
        # el primero que este en linea.
        guardado = G.ultimo(self.cliente or None)
        objetivo = 0
        for i, s in enumerate(self._encontrados):
            if guardado and (s.host, s.puerto) == guardado:
                objetivo = i
                break
            if s.en_linea and objetivo == 0:
                objetivo = i
        if self._encontrados:
            self.tabla.selectRow(objetivo)

    def _al_seleccionar(self) -> None:
        fila = self.tabla.currentRow()
        valido = 0 <= fila < len(self._encontrados)
        self.btn_conectar.setEnabled(valido)

    # --------------------------------------------------------------- manual
    def _anadir_manual(self) -> None:
        host = self.txt_host.text().strip()
        if not host:
            return
        try:
            puerto = int(self.txt_puerto.text().strip()
                         or D.PUERTO_SERVIDOR)
        except ValueError:
            puerto = D.PUERTO_SERVIDOR
        self.estado.setText(f"Probando {host}:{puerto}…")
        self.btn_anadir.setEnabled(False)
        try:
            srv = D.consultar(host, puerto, origen="manual")
        finally:
            self.btn_anadir.setEnabled(True)
        if srv is None:
            QMessageBox.warning(
                self, "Sin respuesta",
                f"{host}:{puerto} no responde como un servidor ELDE.\n\n"
                "Comprueba que el servidor este encendido y que el cortafuegos "
                f"deje pasar el puerto {puerto}.")
            self.estado.setText(f"{host}:{puerto} no respondio.")
            return
        # Se guarda aunque el operador no llegue a conectar: ya se ha
        # comprobado que existe, y volver a escribirlo cada vez es el tipo de
        # friccion que hace que la gente deje la IP en un post-it.
        G.anadir_manual(srv.host, srv.puerto)
        self._encontrados = [x for x in self._encontrados
                             if x.clave != srv.clave] + [srv]
        self._encontrados = D.ordenar(self._encontrados)
        self._pintar()
        for i, x in enumerate(self._encontrados):
            if x.clave == srv.clave:
                self.tabla.selectRow(i)
                break
        self.txt_host.clear()
        self.estado.setText(f"{srv.host}:{srv.puerto} anadido y guardado.")

    # ------------------------------------------------------------- conectar
    def _conectar(self) -> None:
        fila = self.tabla.currentRow()
        if not (0 <= fila < len(self._encontrados)):
            return
        srv = self._encontrados[fila]
        if not srv.en_linea and not D.alcanzable(srv.host, srv.puerto):
            respuesta = QMessageBox.question(
                self, "El servidor no responde",
                f"{srv.host}:{srv.puerto} no responde ahora mismo.\n\n"
                "¿Abrir el cliente igualmente? Se reconectara solo en cuanto "
                "el servidor este disponible.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if respuesta != QMessageBox.Yes:
                return
        self.seleccionado = srv
        if self.recordar:
            G.recordar(srv.host, srv.puerto, self.cliente or None)
        self.accept()

    # ---------------------------------------------------------------- cierre
    def _detener_buscador(self) -> None:
        if self._buscador is None:
            return
        self._buscador.cancelar()
        self._buscador.wait(4000)
        self._buscador = None

    def closeEvent(self, evento):  # noqa: N802  (API de Qt)
        self._detener_buscador()
        super().closeEvent(evento)

    def done(self, resultado):  # noqa: D102  (API de Qt)
        # `accept()`/`reject()` no pasan por closeEvent: sin esto, conectar
        # mientras el barrido sigue vivo destruiria el QThread en marcha.
        self._detener_buscador()
        super().done(resultado)
