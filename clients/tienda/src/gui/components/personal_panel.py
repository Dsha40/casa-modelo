"""Pestaña «Personal»: enrolar y gestionar al personal de la tienda.

Habla con la API `/api/personal` del servidor (src/app/personal_api.py):

    GET    /api/personal/            -> {total, activos, personal: [fichas]}
    POST   /api/personal/enrolar     -> {ok, ficha} | 400 {detail legible}
    PATCH  /api/personal/{id}        -> {'activo': bool}
    DELETE /api/personal/{id}
    GET    /api/personal/{id}/foto/0 -> jpeg

El personal enrolado queda EXCLUIDO del marketing (no cuenta como visitante)
y genera los avisos «Seguridad en pasillo» / «Empleado surte anaquel».

El POST de enrolar acepta `tipo`: 'rostro' (default, UN rostro claro que
identifica a la persona) o 'uniforme' (foto de cuerpo entero: identifica la
vestimenta por bordado/colores con OSNet y funciona de espaldas; el nombre
describe la prenda, p. ej. «Uniforme de seguridad»). GET devuelve `tipo` en
cada ficha.

## Reglas de hilos (patrón de elde_core.ui.panel_capturas)

Toda petición HTTP va en un QThread: el servidor puede estar en otra máquina
y un `requests` en el hilo GUI congela la ventana. El hilo de descarga de la
lista NO se encadena (si el anterior sigue corriendo, se espera al siguiente
tick) y `aboutToQuit` se conecta a `_detener()` que para el timer y hace
`wait(12000)` de TODOS los hilos vivos (con `terminate()` de último
recurso): un QThread destruido mientras corre
hace que Qt aborte el proceso entero (0xc0000409, ya tumbó a los clientes de
este repo — HALLAZGOS.md H-01).
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QByteArray, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from elde_core.ui.panel_capturas import base_http_del_websocket

# La misma paleta de anaqueles_panel: los laterales deben verse del mismo
# sistema.
FONDO = '#1e1e1e'
FONDO_CAB = '#252525'
BORDE = '#3a3a3a'
TEXTO = '#e8e8e8'
TEXTO_SUAVE = '#9a9a9a'

VERDE = '#4caf50'      # empleado activo
AMBAR = '#f0b429'      # avisos del servidor
ROJO = '#f85149'       # errores del formulario

_MINIATURA = 56

_ROL_ICONO = {'seguridad': '🛡️', 'pasillero': '📦'}
_ROL_LEGIBLE = {'seguridad': '🛡️ Seguridad', 'pasillero': '📦 Pasillero'}

_BTN = (
    'QPushButton { background:#2d2d2d; color:%s; border:1px solid %s;'
    ' border-radius:4px; padding:3px 10px; font-size:10px; }'
    'QPushButton:hover { background:#383838; }'
    'QPushButton:disabled { color:#666; }' % (TEXTO, BORDE))


class _Peticion(QThread):
    """Un POST/PATCH/DELETE fuera del hilo GUI.

    Emite SIEMPRE `done` con {'status': código http (0 = sin conexión),
    'cuerpo': json de la respuesta, 'error': texto si no hubo respuesta}."""

    done = Signal(dict)

    def __init__(self, metodo: str, url: str,
                 cuerpo: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._metodo = metodo
        self._url = url
        self._cuerpo = cuerpo

    def run(self):
        try:
            import requests
            # (connect, read) acotados: un servidor colgado no puede retener
            # el hilo más de lo que _detener() está dispuesto a esperar.
            r = requests.request(self._metodo, self._url,
                                 json=self._cuerpo, timeout=(3, 10))
            try:
                cuerpo = r.json()
            except Exception:
                cuerpo = {}
            self.done.emit({'status': r.status_code, 'cuerpo': cuerpo})
        except Exception as exc:                            # noqa: BLE001
            self.done.emit({'status': 0, 'cuerpo': {}, 'error': str(exc)})


class _DescargaPersonal(QThread):
    """Pide la lista del personal y las fotos que falten, fuera del GUI.

    `fotos_vistas` es {pid: n_fotos ya descargadas}: una foto solo se vuelve
    a pedir si el empleado tiene fotos nuevas."""

    listo = Signal(dict)

    def __init__(self, base: str, fotos_vistas: Dict[str, int], parent=None):
        super().__init__(parent)
        self._base = base
        self._vistas = dict(fotos_vistas)

    def run(self):
        try:
            import requests
            r = requests.get(f'{self._base}/api/personal/', timeout=(3, 5))
            if r.status_code == 404:
                # Servidor VIEJO: la ruta no existe todavía.
                self.listo.emit({'ok': False, 'motivo': 'sin_api'})
                return
            if r.status_code != 200:
                self.listo.emit({'ok': False, 'motivo': 'error',
                                 'error': f'HTTP {r.status_code}'})
                return
            datos = r.json()
        except Exception as exc:                            # noqa: BLE001
            self.listo.emit({'ok': False, 'motivo': 'sin_conexion',
                             'error': str(exc)})
            return

        fotos: Dict[str, Tuple[int, bytes]] = {}
        for ficha in (datos.get('personal') or []):
            pid = str(ficha.get('id') or '')
            url = ficha.get('foto_url')
            n = int(ficha.get('n_fotos') or 0)
            if not pid or not url or self._vistas.get(pid) == n:
                continue
            try:
                import requests
                ri = requests.get(f'{self._base}{url}', timeout=(3, 5))
                if ri.status_code == 200:
                    fotos[pid] = (n, ri.content)
            except Exception:
                pass    # sin foto la fila se pinta igual, con el icono
        self.listo.emit({'ok': True, 'datos': datos, 'fotos': fotos})


class PersonalPanel(QWidget):
    """Pestaña lateral para enrolar seguridad y pasilleros."""

    def __init__(self, url_ws: str = '', parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f'background:{FONDO}; color:{TEXTO};')

        self._base = base_http_del_websocket(
            url_ws or os.getenv('server_ws_url', ''))
        #: pid -> (n_fotos descargadas, jpeg)
        self._fotos: Dict[str, Tuple[int, bytes]] = {}
        self._hilo_lista: Optional[_DescargaPersonal] = None
        self._hilos_op: List[_Peticion] = []
        self._refresco_pendiente = False
        self._ruta_foto = ''

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(8, 8, 8, 8)
        raiz.setSpacing(8)

        # ── Cabecera ─────────────────────────────────────────────────────
        cab = QHBoxLayout()
        titulo = QLabel('Personal de la tienda')
        titulo.setStyleSheet('font-size:12px; font-weight:700;')
        cab.addWidget(titulo)
        cab.addStretch()
        btn_refresco = QPushButton('↻')
        btn_refresco.setToolTip('Refrescar la lista')
        btn_refresco.setFixedSize(24, 22)
        btn_refresco.setStyleSheet(_BTN)
        btn_refresco.clicked.connect(self.refrescar)
        cab.addWidget(btn_refresco)
        raiz.addLayout(cab)

        subtitulo = QLabel(
            'Enrolados = fuera del marketing; avisan seguridad/surtido.')
        subtitulo.setWordWrap(True)
        subtitulo.setStyleSheet(f'color:{TEXTO_SUAVE}; font-size:10px;')
        raiz.addWidget(subtitulo)

        # Aviso de servidor (sin API, sin conexión). Se pinta en cada tick
        # con el MISMO texto, sin diálogos ni logs: nada de spam.
        self.lbl_servidor = QLabel('')
        self.lbl_servidor.setWordWrap(True)
        self.lbl_servidor.setStyleSheet(
            f'color:{AMBAR}; font-size:10px; font-weight:600;')
        self.lbl_servidor.hide()
        raiz.addWidget(self.lbl_servidor)

        # ── Formulario de alta ───────────────────────────────────────────
        form = QFrame()
        form.setObjectName('FormPersonal')
        form.setStyleSheet(
            f'#FormPersonal {{ background:{FONDO_CAB};'
            f' border:1px solid {BORDE}; border-radius:6px; }}')
        caja_form = QVBoxLayout(form)
        caja_form.setContentsMargins(10, 8, 10, 8)
        caja_form.setSpacing(6)

        lbl_alta = QLabel('Enrolar empleado')
        lbl_alta.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; font-weight:700;'
            ' border:none;')
        caja_form.addWidget(lbl_alta)

        self.in_nombre = QLineEdit()
        self.in_nombre.setPlaceholderText('Nombre del empleado')
        self.in_nombre.setStyleSheet(
            f'background:{FONDO}; color:{TEXTO}; border:1px solid {BORDE};'
            ' border-radius:4px; padding:4px 6px; font-size:11px;')
        caja_form.addWidget(self.in_nombre)

        _COMBO_CSS = (
            f'QComboBox {{ background:{FONDO}; color:{TEXTO};'
            f' border:1px solid {BORDE}; border-radius:4px;'
            ' padding:4px 6px; font-size:11px; }}'
            f'QComboBox QAbstractItemView {{ background:{FONDO_CAB};'
            f' color:{TEXTO}; selection-background-color:{BORDE}; }}')

        self.combo_rol = QComboBox()
        self.combo_rol.addItem('🛡️ Seguridad', 'seguridad')
        self.combo_rol.addItem('📦 Pasillero', 'pasillero')
        self.combo_rol.setStyleSheet(_COMBO_CSS)
        caja_form.addWidget(self.combo_rol)

        # Tipo de ficha: «rostro» identifica a la persona por la cara;
        # «uniforme» identifica la VESTIMENTA (bordado/colores, OSNet en el
        # servidor) y funciona aunque el empleado esté de espaldas.
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItem('🙂 Rostro (la persona)', 'rostro')
        self.combo_tipo.addItem('👕 Uniforme (la vestimenta)', 'uniforme')
        self.combo_tipo.setToolTip(
            'Rostro: la foto debe tener UN rostro claro.\n'
            'Uniforme: foto del uniforme de cuerpo entero; identifica '
            'también de espaldas.')
        self.combo_tipo.setStyleSheet(_COMBO_CSS)
        self.combo_tipo.currentIndexChanged.connect(self._tipo_cambiado)
        caja_form.addWidget(self.combo_tipo)

        fila_foto = QHBoxLayout()
        fila_foto.setSpacing(6)
        btn_foto = QPushButton('Elegir foto…')
        btn_foto.setStyleSheet(_BTN)
        btn_foto.clicked.connect(self._elegir_foto)
        fila_foto.addWidget(btn_foto)
        self.lbl_foto = QLabel('sin foto elegida')
        self.lbl_foto.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; border:none;')
        fila_foto.addWidget(self.lbl_foto, stretch=1)
        caja_form.addLayout(fila_foto)

        self.btn_enrolar = QPushButton('Enrolar')
        self.btn_enrolar.setStyleSheet(_BTN)
        self.btn_enrolar.clicked.connect(self._enrolar)
        caja_form.addWidget(self.btn_enrolar)

        # El 400 del servidor trae un `detail` PARA EL OPERADOR («no se
        # detectó ningún rostro claro en la foto»): se muestra aquí en rojo,
        # pegado al formulario, no en un log.
        self.lbl_error = QLabel('')
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet(
            f'color:{ROJO}; font-size:10px; font-weight:600; border:none;')
        self.lbl_error.hide()
        caja_form.addWidget(self.lbl_error)

        raiz.addWidget(form)

        # ── Lista de enrolados ───────────────────────────────────────────
        fila_lista = QHBoxLayout()
        lbl_lista = QLabel('Enrolados')
        lbl_lista.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; font-weight:700;')
        fila_lista.addWidget(lbl_lista)
        fila_lista.addStretch()
        self.lbl_contador = QLabel('')
        self.lbl_contador.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px;')
        fila_lista.addWidget(self.lbl_contador)
        raiz.addLayout(fila_lista)

        self._lista = QVBoxLayout()
        self._lista.setSpacing(4)
        self._lista.setAlignment(Qt.AlignTop)
        cont = QWidget()
        cont.setStyleSheet('background:transparent;')
        cont.setLayout(self._lista)

        self._vacio = QLabel('Nadie enrolado todavía. Da de alta al primero '
                             'con el formulario de arriba.')
        self._vacio.setWordWrap(True)
        self._vacio.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; padding:10px 4px;')
        self._lista.addWidget(self._vacio)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            'QScrollArea { border:none; background:transparent; }')
        scroll.setWidget(cont)
        raiz.addWidget(scroll, stretch=1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── Refresco: al abrir + cada 30 s + tras cada operación ─────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refrescar)
        self._timer.start(30_000)
        QTimer.singleShot(400, self.refrescar)

        # closeEvent NO basta: al salir la aplicación Qt destruye los widgets
        # acoplados sin pasar por closeEvent, y un QThread vivo en ese
        # momento aborta el proceso (0xc0000409). aboutToQuit se emite con
        # los widgets todavía vivos: es el momento correcto para parar.
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._detener)

    # ── Refresco de la lista ─────────────────────────────────────────────
    def refrescar(self) -> None:
        """Pide la lista. Si ya hay una petición en curso NO se encadena
        otra: se anota y se relanza cuando la actual termine."""
        if self._hilo_lista is not None:
            try:
                if self._hilo_lista.isRunning():
                    self._refresco_pendiente = True
                    return
            except RuntimeError:
                pass    # el wrapper ya fue destruido por deleteLater
        self._hilo_lista = _DescargaPersonal(
            self._base, {p: n for p, (n, _) in self._fotos.items()}, self)
        self._hilo_lista.listo.connect(self._pintar)
        # El relanzado del refresco pendiente cuelga de `finished`, no de
        # `listo`: se emite SIEMPRE al acabar el hilo, así ningún refresco
        # anotado se pierde por la carrera con isRunning().
        self._hilo_lista.finished.connect(self._quiza_relanzar)
        self._hilo_lista.finished.connect(self._hilo_lista.deleteLater)
        self._hilo_lista.start()

    def _quiza_relanzar(self) -> None:
        """Al terminar el hilo de lista: si quedó un refresco anotado
        mientras corría, se relanza en el siguiente tick."""
        if self._refresco_pendiente:
            self._refresco_pendiente = False
            QTimer.singleShot(0, self.refrescar)

    def _pintar(self, res: Dict[str, Any]) -> None:
        if not res.get('ok'):
            if res.get('motivo') == 'sin_api':
                self._avisar_servidor(
                    'el servidor aún no tiene la API de personal '
                    '(reinícialo con la versión nueva)')
            else:
                self._avisar_servidor(
                    'sin respuesta del servidor de personal — se reintenta '
                    'solo en unos segundos')
            return
        self._avisar_servidor('')

        for pid, foto in (res.get('fotos') or {}).items():
            self._fotos[pid] = foto

        datos = res.get('datos') or {}
        personal = list(datos.get('personal') or [])
        self.lbl_contador.setText(
            f"{int(datos.get('activos') or 0)} activos · "
            f"{int(datos.get('total') or 0)} en total")

        _vaciar(self._lista)
        if not personal:
            self._vacio = QLabel('Nadie enrolado todavía. Da de alta al '
                                 'primero con el formulario de arriba.')
            self._vacio.setWordWrap(True)
            self._vacio.setStyleSheet(
                f'color:{TEXTO_SUAVE}; font-size:10px; padding:10px 4px;')
            self._lista.addWidget(self._vacio)
            return
        for ficha in personal:
            self._lista.addWidget(self._fila(ficha))

    def _fila(self, ficha: Dict[str, Any]) -> QFrame:
        pid = str(ficha.get('id') or '')
        nombre = str(ficha.get('nombre') or '?')
        rol = str(ficha.get('rol') or '')
        activo = bool(ficha.get('activo'))
        # Ficha de uniforme: identifica la VESTIMENTA, no un rostro. Se
        # marca con 👕 junto al nombre y al icono, y en el estado.
        uniforme = str(ficha.get('tipo') or '') == 'uniforme'

        fila = QFrame()
        fila.setObjectName('FilaPersonal')
        fila.setStyleSheet(
            f'#FilaPersonal {{ background:{FONDO_CAB};'
            f' border:1px solid {BORDE}; border-radius:6px; }}')
        lay = QHBoxLayout(fila)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        foto = QLabel()
        foto.setFixedSize(_MINIATURA, _MINIATURA)
        pintada = False
        guardada = self._fotos.get(pid)
        if guardada is not None:
            pix = QPixmap()
            if pix.loadFromData(QByteArray(guardada[1])):
                foto.setPixmap(pix)
                foto.setScaledContents(True)
                foto.setStyleSheet('border:none; border-radius:4px;')
                pintada = True
        if not pintada:
            icono = _ROL_ICONO.get(rol, '👤')
            if uniforme:
                icono = f'👕 {icono}'
            foto.setText(icono)
            foto.setAlignment(Qt.AlignCenter)
            foto.setStyleSheet(
                f'background:{FONDO}; border:1px solid {BORDE};'
                ' border-radius:4px;'
                f' font-size:{"14" if uniforme else "20"}px;')
        lay.addWidget(foto)

        texto = QVBoxLayout()
        texto.setSpacing(1)
        lbl_nombre = QLabel(f'👕 {nombre}' if uniforme else nombre)
        lbl_nombre.setStyleSheet(
            'font-size:11px; font-weight:700; border:none;')
        texto.addWidget(lbl_nombre)
        lbl_rol = QLabel(_ROL_LEGIBLE.get(rol, rol or '—'))
        lbl_rol.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; border:none;')
        texto.addWidget(lbl_rol)
        sufijo = ' (uniforme)' if uniforme else ''
        if activo:
            estado = QLabel(f'● Activo{sufijo}')
            estado.setStyleSheet(
                f'color:{VERDE}; font-size:10px; font-weight:600;'
                ' border:none;')
        else:
            estado = QLabel(f'● Inactivo{sufijo}')
            estado.setStyleSheet(
                f'color:{TEXTO_SUAVE}; font-size:10px; border:none;')
        texto.addWidget(estado)
        lay.addLayout(texto, stretch=1)

        botones = QVBoxLayout()
        botones.setSpacing(4)
        btn_estado = QPushButton('Desactivar' if activo else 'Activar')
        btn_estado.setStyleSheet(_BTN)
        btn_estado.clicked.connect(
            lambda _=False, p=pid, a=activo, b=btn_estado:
            self._cambiar_activo(p, a, b))
        botones.addWidget(btn_estado)
        btn_borrar = QPushButton('Borrar')
        btn_borrar.setStyleSheet(
            _BTN + f'QPushButton {{ color:{ROJO}; }}')
        btn_borrar.clicked.connect(
            lambda _=False, p=pid, n=nombre, b=btn_borrar:
            self._borrar(p, n, b))
        botones.addWidget(btn_borrar)
        lay.addLayout(botones)
        return fila

    # ── Alta ─────────────────────────────────────────────────────────────
    def _tipo_cambiado(self) -> None:
        """Con tipo uniforme el «nombre» describe la PRENDA, no a una
        persona: el placeholder lo dice antes de que el operador escriba
        un nombre propio."""
        if self.combo_tipo.currentData() == 'uniforme':
            self.in_nombre.setPlaceholderText(
                'nombre del uniforme (ej. Uniforme de seguridad)')
        else:
            self.in_nombre.setPlaceholderText('Nombre del empleado')

    def _elegir_foto(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(
            self, 'Elegir la foto del empleado', '',
            'Imágenes (*.jpg *.jpeg *.png);;Todos los archivos (*)')
        if ruta:
            self._ruta_foto = ruta
            self.lbl_foto.setText(os.path.basename(ruta))

    def _enrolar(self) -> None:
        nombre = self.in_nombre.text().strip()
        uniforme = self.combo_tipo.currentData() == 'uniforme'
        if not nombre:
            self._error_form(
                'escribe el nombre del uniforme (ej. Uniforme de seguridad)'
                if uniforme else 'escribe el nombre del empleado')
            return
        if not self._ruta_foto:
            self._error_form(
                'elige una foto del uniforme de cuerpo entero (que se '
                'vean bordado y colores)' if uniforme else
                'elige una foto del empleado (JPG o PNG, de frente y con '
                'buena luz)')
            return
        try:
            with open(self._ruta_foto, 'rb') as fh:
                foto_b64 = base64.b64encode(fh.read()).decode()
        except OSError as exc:
            self._error_form(f'no se pudo leer la foto: {exc}')
            return
        self._error_form('')
        self.btn_enrolar.setEnabled(False)
        self.btn_enrolar.setText('Enrolando…')
        self._lanzar('POST', f'{self._base}/api/personal/enrolar',
                     {'nombre': nombre,
                      'rol': self.combo_rol.currentData(),
                      'tipo': self.combo_tipo.currentData(),
                      'foto_b64': foto_b64},
                     self._enrolado)

    def _enrolado(self, res: Dict[str, Any]) -> None:
        self.btn_enrolar.setEnabled(True)
        self.btn_enrolar.setText('Enrolar')
        status = int(res.get('status') or 0)
        if status == 200:
            self.in_nombre.clear()
            self._ruta_foto = ''
            self.lbl_foto.setText('sin foto elegida')
            self._error_form('')
            self.refrescar()
            return
        detalle = str((res.get('cuerpo') or {}).get('detail') or '').strip()
        if status == 400:
            self._error_form(detalle or 'el servidor rechazó la foto')
        elif status == 0:
            self._error_form('no se pudo contactar al servidor: '
                             + str(res.get('error') or 'sin conexión'))
        else:
            self._error_form(f'error {status} del servidor'
                             + (f': {detalle}' if detalle else ''))

    # ── Activar/desactivar y borrar ──────────────────────────────────────
    def _cambiar_activo(self, pid: str, activo_actual: bool,
                        boton: Optional[QPushButton] = None) -> None:
        if boton is not None:
            boton.setEnabled(False)
        self._lanzar('PATCH', f'{self._base}/api/personal/{pid}',
                     {'activo': not activo_actual}, self._op_terminada)

    def _borrar(self, pid: str, nombre: str,
                boton: Optional[QPushButton] = None) -> None:
        respuesta = QMessageBox.question(
            self, 'Borrar empleado',
            f'¿Borrar a {nombre}?\n\nSe borran también TODAS sus fotos de '
            'la galería y volverá a contarse como un visitante más.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if respuesta != QMessageBox.Yes:
            return
        if boton is not None:
            boton.setEnabled(False)
        self._lanzar('DELETE', f'{self._base}/api/personal/{pid}',
                     None, self._op_terminada)

    def _op_terminada(self, res: Dict[str, Any]) -> None:
        status = int(res.get('status') or 0)
        if status == 200:
            self._error_form('')
        else:
            detalle = str((res.get('cuerpo') or {}).get('detail')
                          or res.get('error') or '').strip()
            self._error_form('la operación falló'
                             + (f': {detalle}' if detalle else ''))
        self.refrescar()

    # ── Interno ──────────────────────────────────────────────────────────
    def _lanzar(self, metodo: str, url: str, cuerpo: Optional[dict],
                callback) -> None:
        """Un hilo por operación, registrado para el wait del cierre."""
        hilo = _Peticion(metodo, url, cuerpo, self)
        hilo.done.connect(callback)
        hilo.finished.connect(self._podar_hilos)
        # Sin esto los QThread terminados se acumulaban para siempre (la poda
        # solo los saca de la lista, no los destruye).
        hilo.finished.connect(hilo.deleteLater)
        self._hilos_op.append(hilo)
        hilo.start()

    def _podar_hilos(self) -> None:
        vivos: List[_Peticion] = []
        for h in self._hilos_op:
            try:
                if not h.isFinished():
                    vivos.append(h)
            except RuntimeError:
                pass    # el wrapper ya fue destruido por deleteLater
        self._hilos_op = vivos

    def _error_form(self, texto: str) -> None:
        self.lbl_error.setText(texto)
        self.lbl_error.setVisible(bool(texto))

    def _avisar_servidor(self, texto: str) -> None:
        self.lbl_servidor.setText(texto)
        self.lbl_servidor.setVisible(bool(texto))

    def _detener(self) -> None:
        """Para el timer y espera a TODOS los hilos vivos. Idempotente: lo
        llaman closeEvent y aboutToQuit."""
        try:
            self._timer.stop()
            hilos: List[QThread] = list(self._hilos_op)
            if self._hilo_lista is not None:
                hilos.append(self._hilo_lista)
            for hilo in hilos:
                try:
                    if not hilo.isRunning():
                        continue
                    if not hilo.wait(12000):
                        # terminate() es feo (mata el hilo a media petición),
                        # pero es el último recurso: dejarlo vivo hace que el
                        # destructor de Qt aborte con qFatal (0xc0000409).
                        hilo.terminate()
                        hilo.wait()
                except RuntimeError:
                    pass    # el wrapper ya fue destruido por deleteLater
        except Exception:
            pass

    def closeEvent(self, event):
        self._detener()
        super().closeEvent(event)


def _vaciar(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
