"""
Panel lateral de alertas por COLUMNAS, generico y sin dominio.

## Por que existe

Cada cliente vigila algo distinto y su panel debe reflejarlo: perimetrales
separa **Vehiculos** y **Personas**, tienda no tiene vehiculos sino
**Visitantes** y **Capturas**, y los clientes multimodo necesitan una tercera
columna de eventos. Antes eso obligaba a copiar las ~480 lineas del panel en
cada cliente y retocarlas, con la divergencia que eso arrastra.

Aqui vive **solo el armazon**: la tarjeta de alerta, la columna con su titulo,
icono, color y contador, y el enrutado. Que columnas hay y como se decide en
cual cae cada alerta lo aporta cada cliente, que es justo la parte de dominio
que NO debe compartirse (criterio del HITO 2).

## Como se usa

    from elde_core.ui.panel_alertas import Columna, PanelAlertas

    COLUMNAS = [
        Columna('vehiculos', 'Vehículos', '🚗', '#e67e22'),
        Columna('personas',  'Personas',  '👤', '#3498db'),
    ]

    def clasificar(alerta: dict) -> str:
        ...devuelve la clave de la columna, o '' para descartar la alerta...

    panel = PanelAlertas(COLUMNAS, clasificar, titulo='Alertas perimetrales')

El estilo es propio y minimo a proposito: depender del modulo `tema` de un
cliente concreto volveria a acoplar el nucleo al cliente.
"""

from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QByteArray, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

MAX_ALERTAS = 200          # por columna, antes de descartar las mas viejas
_MINIATURA = 56            # lado de la miniatura, en pixeles


# ── Tiempos: llegada, salida y permanencia ───────────────────────────────
# Los emite la vigilancia perimetral y son su dato mas util («entro a las
# 21:14 y estuvo 3 min»). Viven en el nucleo, no en el cliente, para que
# cualquier pipeline que los publique los vea pintados sin duplicar el panel.

def _a_fecha(valor) -> Optional[datetime]:
    """epoch | ISO | None -> datetime local, o None. Tolerante a lo que venga."""
    if valor in (None, '', 0):
        return None
    try:
        if isinstance(valor, str):
            crudo = valor.strip()
            if not crudo:
                return None
            try:
                # Epoch como texto ('1786984080.55'): asi llega el
                # timestamp de las tarjetas del vigilante/estacionamiento.
                return datetime.fromtimestamp(float(crudo))
            except (ValueError, OverflowError, OSError):
                pass
            if crudo.endswith('Z'):
                crudo = crudo[:-1] + '+00:00'
            dt = datetime.fromisoformat(crudo)
            return (dt.astimezone().replace(tzinfo=None)
                    if dt.tzinfo is not None else dt)
        return datetime.fromtimestamp(float(valor))
    except (ValueError, TypeError, OSError):
        return None


def _hora(valor, vacio: str = '—') -> str:
    dt = _a_fecha(valor)
    return dt.strftime('%H:%M:%S') if dt else vacio


def _permanencia(segundos) -> str:
    """Segundos -> «3 min 12 s». Cadena vacia si no hay dato usable."""
    try:
        s = int(float(segundos))
    except (TypeError, ValueError):
        return ''
    if s < 0:
        return ''
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60} min {s % 60:02d} s'
    return f'{s // 3600} h {(s % 3600) // 60:02d} min'


@dataclass(frozen=True)
class Columna:
    """Definicion de una columna del panel.

    `clave` es lo que devuelve el clasificador del cliente; el resto es
    presentacion."""
    clave: str
    titulo: str
    icono: str
    color: str
    #: Tarjeta con la imagen ARRIBA, a todo el ancho y sin deformar, y el
    #: texto debajo (20-oct-2026): la foto «antes | ahora» del stock es
    #: apaisada y en el cuadrado de 56 px no se veía nada.
    imagen_arriba: bool = False
    #: Miniatura ARRIBA, sin deformar y centrada, con el texto debajo a todo
    #: el ancho (28-oct-2026). Para recortes de personas (altos y estrechos):
    #: en una columna estrecha, la miniatura cuadrada de 56 px a la izquierda
    #: los aplastaba y dejaba al texto ~40 px, que se cortaba y se solapaba.
    miniatura_arriba: bool = False


#: Alto de la miniatura con `miniatura_arriba`: basta para reconocer a la
#: persona sin que cada tarjeta ocupe media columna.
_MINIATURA_ALTO = 96


class TarjetaAlerta(QFrame):
    """Una alerta: miniatura, titular, descripcion y pie con hora y camara.

    Si la alerta trae `screenshot_path`, al pulsarla se abre la carpeta que la
    contiene — es la via rapida para llegar a la evidencia."""

    def __init__(self, alerta: dict, color: str, parent=None,
                 imagen_arriba: bool = False, miniatura_arriba: bool = False):
        super().__init__(parent)
        self._ruta = str(alerta.get('screenshot_path') or '')
        self.setFrameShape(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor if self._ruta else Qt.ArrowCursor)
        # Con el nombre del objeto (28-oct-2026): un «QFrame {...}» a secas
        # también se aplicaba a cada QLabel de dentro (una QLabel ES un
        # QFrame), y cada línea de texto salía con su propia barra de color.
        self.setObjectName('tarjetaAlerta')
        self.setStyleSheet(
            f'#tarjetaAlerta {{ background:#1e1e1e; border-left:3px solid {color};'
            f' border-radius:4px; }}'
            f'#tarjetaAlerta:hover {{ background:#282828; }}')
        chico = miniatura_arriba          # texto un punto más pequeño

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(6, 5, 6, 5)
        raiz.setSpacing(3)

        fila = QHBoxLayout()
        fila.setSpacing(6)
        if imagen_arriba:
            imagen = self._imagen_ancha(alerta)
            if imagen is not None:
                raiz.addWidget(imagen)
        elif miniatura_arriba:
            imagen = self._miniatura_alta(alerta)
            if imagen is not None:
                raiz.addWidget(imagen, alignment=Qt.AlignHCenter)
        else:
            miniatura = self._miniatura(alerta)
            if miniatura is not None:
                fila.addWidget(miniatura)

        texto = QVBoxLayout()
        texto.setSpacing(1)
        titular = _TextoAjustado(self._titular(alerta))
        titular.setStyleSheet(f'color:{color}; font-weight:bold; '
                              f'font-size:{10 if chico else 11}px;')
        titular.setWordWrap(True)
        texto.addWidget(titular)

        # Tiempos, si el pipeline los publica. Van ANTES de la descripcion
        # porque en vigilancia son el dato que se mira primero.
        tiempos = self._tiempos(alerta)
        if tiempos:
            lbl = QLabel(tiempos)
            lbl.setStyleSheet('color:#9ad; font-size:10px;')
            texto.addWidget(lbl)

        desc = str(alerta.get('description') or '').strip()
        if desc:
            lbl = _TextoAjustado(desc)
            lbl.setStyleSheet(f'color:#bbb; font-size:{9 if chico else 10}px;')
            lbl.setWordWrap(True)
            texto.addWidget(lbl)

        pie = self._pie(alerta)
        if pie:
            lbl = _TextoAjustado(pie)
            lbl.setStyleSheet(f'color:#777; font-size:{8 if chico else 9}px;')
            # Sin esto, «🕒 hora  📹 cámara» en una sola línea fijaba el
            # ancho mínimo de la columna y el panel se salía por la derecha.
            lbl.setWordWrap(True)
            texto.addWidget(lbl)

        fila.addLayout(texto, stretch=1)
        raiz.addLayout(fila)

    @staticmethod
    def _titular(alerta: dict) -> str:
        evento = str(alerta.get('event_type') or '').strip()
        clase = str(alerta.get('class_name') or '').strip()
        gid = str(alerta.get('global_id') or '').strip()
        base = (f'{evento} — {clase}'
                if evento and clase and evento.lower() != clase.lower()
                else (evento or clase or 'Detección'))
        return f'{base}  [{gid}]' if gid else base

    @staticmethod
    def _tiempos(alerta: dict) -> str:
        """Linea «llegada / salida / permanencia», solo con lo que haya.

        Un objeto que sigue en el sitio no tiene hora de salida: se marca con
        «sigue» en vez de dejar un hueco, que se confunde con un dato perdido.
        """
        partes = []
        llegada = alerta.get('hora_llegada')
        if llegada:
            partes.append(f'⬇ {_hora(llegada)}')
        if 'hora_salida' in alerta:
            salida = alerta.get('hora_salida')
            partes.append(f'⬆ {_hora(salida)}' if salida else '⬆ sigue')
        perm = _permanencia(alerta.get('permanencia_s'))
        if perm:
            partes.append(f'⏱ {perm}')
        return '   '.join(partes)

    @staticmethod
    def _pie(alerta: dict) -> str:
        partes = []
        # La hora legible venga como venga el timestamp: epoch (crudo o en
        # texto), ISO o ya formateada. El recorte de los ultimos 8 caracteres
        # queda de respaldo — sobre un epoch mostraba «.5534697».
        ts = alerta.get('timestamp')
        hora = _hora(ts, vacio='')
        if not hora:
            crudo = str(ts or '').strip()
            hora = crudo[-8:] if len(crudo) > 8 else crudo
        if hora:
            partes.append(f'🕒 {hora}')
        cam = str(alerta.get('camera_name') or '').strip()
        if cam:
            partes.append(f'📹 {cam}')
        # Local por camara (perimetrales): si la alerta viene etiquetada, se
        # ve a que establecimiento pertenece sin abrir Jarvis.
        local = str(alerta.get('establecimiento') or '').strip()
        if local:
            partes.append(f'🏢 {local}')
        return '   '.join(partes)

    def _miniatura(self, alerta: dict) -> Optional[QLabel]:
        b64 = alerta.get('crop_image') or alerta.get('image_base64') or ''
        if not b64:
            return None
        try:
            if ',' in b64[:40]:                 # data:image/...;base64,XXXX
                b64 = b64.split(',', 1)[1]
            pix = QPixmap()
            if not pix.loadFromData(QByteArray(base64.b64decode(b64))):
                return None
        except Exception:
            return None
        lbl = QLabel()
        lbl.setFixedSize(_MINIATURA, _MINIATURA)
        lbl.setScaledContents(True)
        lbl.setPixmap(pix)
        lbl.setStyleSheet('border-radius:3px;')
        return lbl

    def _miniatura_alta(self, alerta: dict) -> Optional[QLabel]:
        """Miniatura de `_MINIATURA_ALTO` px de alto, ancho según la
        proporción de la foto (sin deformar ni recortar)."""
        pix = self._pixmap(alerta)
        if pix is None:
            return None
        lbl = QLabel()
        lbl.setPixmap(pix.scaledToHeight(_MINIATURA_ALTO,
                                         Qt.SmoothTransformation))
        lbl.setStyleSheet('border-radius:3px;')
        return lbl

    def _imagen_ancha(self, alerta: dict) -> Optional[QLabel]:
        """La imagen a todo el ancho de la tarjeta, sin deformar."""
        pix = self._pixmap(alerta)
        if pix is None:
            return None
        lbl = _ImagenAjustada(pix)
        lbl.setStyleSheet('border-radius:3px;')
        return lbl

    @staticmethod
    def _pixmap(alerta: dict) -> Optional[QPixmap]:
        b64 = alerta.get('crop_image') or alerta.get('image_base64') or ''
        if not b64:
            return None
        try:
            if ',' in b64[:40]:
                b64 = b64.split(',', 1)[1]
            pix = QPixmap()
            return pix if pix.loadFromData(QByteArray(base64.b64decode(b64))) else None
        except Exception:
            return None

    def mouseReleaseEvent(self, event):
        """Abre la carpeta de la captura. Nunca lanza: si la ruta ya no existe
        o el explorador falla, no puede tumbar la interfaz."""
        if event.button() == Qt.LeftButton and self._ruta:
            try:
                carpeta = os.path.dirname(self._ruta) or self._ruta
                if os.path.isdir(carpeta):
                    subprocess.Popen(['explorer', os.path.normpath(carpeta)])
            except Exception:
                pass
        super().mouseReleaseEvent(event)


class _TextoAjustado(QLabel):
    """QLabel con salto de línea que pide el alto que necesita a SU ancho
    real (28-oct-2026). En las columnas estrechas (ancho «Ignored») Qt
    calculaba el alto para un ancho mayor y la última línea se cortaba:
    «…Primera» sin «visita registrada»."""

    def __init__(self, texto: str, parent=None):
        super().__init__(texto, parent)
        self.setWordWrap(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        alto = self.heightForWidth(event.size().width())
        if alto > 0 and alto != self.minimumHeight():
            self.setMinimumHeight(alto)


class _ImagenAjustada(QLabel):
    """Una imagen que ocupa el ancho disponible y escala su alto con la
    proporción original (una QLabel con `setScaledContents` deforma)."""

    def __init__(self, pix: QPixmap, parent=None):
        super().__init__(parent)
        self._pix = pix
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(40)
        self._ajustar(120)

    def _ajustar(self, ancho: int) -> None:
        ancho = max(40, ancho)
        alto = max(1, round(self._pix.height() * ancho / max(1, self._pix.width())))
        self.setFixedHeight(alto)
        self.setPixmap(self._pix.scaled(ancho, alto, Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._ajustar(event.size().width())
        super().resizeEvent(event)


class ColumnaAlertas(QWidget):
    """Una columna con cabecera (icono, titulo, contador) y lista con scroll."""

    def __init__(self, columna: Columna, max_alertas: int = MAX_ALERTAS,
                 parent=None):
        super().__init__(parent)
        self.columna = columna
        self.max_alertas = max_alertas
        self._n = 0

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(4)

        cab = QWidget()
        # Con selector (28-oct-2026): sin él, el borde superior y el fondo
        # se aplicaban también al icono, al título y al contador, y el
        # título salía metido en su propia caja y cortado.
        cab.setObjectName('cabColumna')
        cab.setAttribute(Qt.WA_StyledBackground, True)
        cab.setStyleSheet(f'#cabColumna {{ background:#252525; border-top:2px solid '
                          f'{columna.color}; border-radius:3px; }}')
        cl = QHBoxLayout(cab)
        cl.setContentsMargins(6, 4, 6, 4)
        cl.setSpacing(5)
        ico = QLabel(columna.icono)
        ico.setStyleSheet('font-size:13px;')
        cl.addWidget(ico)
        tit = QLabel(columna.titulo)
        tit.setStyleSheet(f'color:{columna.color}; font-weight:bold; '
                          f'font-size:11px;')
        cl.addWidget(tit)
        cl.addStretch()
        self.contador = QLabel('0')
        self.contador.setStyleSheet(
            f'color:#fff; background:{columna.color}; border-radius:7px;'
            f' padding:0 6px; font-size:10px; font-weight:bold;')
        cl.addWidget(self.contador)
        raiz.addWidget(cab)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet('QScrollArea { border:none; background:transparent; }')
        cont = QWidget()
        cont.setStyleSheet('background:transparent;')
        self.lista = QVBoxLayout(cont)
        self.lista.setContentsMargins(0, 0, 0, 0)
        self.lista.setSpacing(4)
        self.lista.setAlignment(Qt.AlignTop)

        self.vacio = QLabel('Sin detecciones todavía.')
        self.vacio.setAlignment(Qt.AlignCenter)
        self.vacio.setWordWrap(True)
        self.vacio.setStyleSheet('color:#666; font-size:10px; padding:12px 4px;')
        self.lista.addWidget(self.vacio)

        self.scroll.setWidget(cont)
        raiz.addWidget(self.scroll, stretch=1)
        # Horizontal «Ignored» (20-oct-2026): con «Expanding» la suma de los
        # anchos mínimos de las columnas (cabecera, tarjetas) pasaba del
        # ancho del panel y la última columna se cortaba por la derecha. Así
        # las columnas se reparten el ancho que hay y el texto salta de línea.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.setMinimumWidth(60)

    def anadir(self, alerta: dict) -> None:
        """Inserta la alerta arriba del todo (lo mas reciente primero)."""
        if self.vacio is not None:
            self.vacio.setVisible(False)
        self.lista.insertWidget(0, TarjetaAlerta(
            alerta, self.columna.color,
            imagen_arriba=self.columna.imagen_arriba,
            miniatura_arriba=self.columna.miniatura_arriba))
        self._n += 1
        self.contador.setText(str(self._n))
        # Descartar las mas viejas para que la memoria no crezca sin limite.
        while self.lista.count() > self.max_alertas + 1:
            item = self.lista.takeAt(self.lista.count() - 1)
            w = item.widget() if item else None
            if w is not None and w is not self.vacio:
                w.deleteLater()

    def limpiar(self) -> None:
        for i in reversed(range(self.lista.count())):
            w = self.lista.itemAt(i).widget()
            if w is not None and w is not self.vacio:
                w.setParent(None)
                w.deleteLater()
        self._n = 0
        self.contador.setText('0')
        if self.vacio is not None:
            self.vacio.setVisible(True)


class PanelAlertas(QWidget):
    """Panel de N columnas. El dominio lo pone el cliente.

    `clasificador(alerta) -> clave` decide la columna. Si devuelve una clave
    desconocida o vacia, la alerta se descarta en silencio: es como cada
    cliente filtra lo que no le concierne."""

    def __init__(self, columnas: List[Columna],
                 clasificador: Callable[[dict], str],
                 titulo: str = 'Alertas',
                 max_alertas: int = MAX_ALERTAS, parent=None):
        super().__init__(parent)
        if not columnas:
            raise ValueError('un panel necesita al menos una columna')
        self._clasificar = clasificador
        self.columnas: Dict[str, ColumnaAlertas] = {}

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(4, 4, 4, 4)
        raiz.setSpacing(5)

        cab = QWidget()
        cl = QHBoxLayout(cab)
        cl.setContentsMargins(2, 0, 2, 0)
        lbl = QLabel(titulo)
        lbl.setStyleSheet('color:#eee; font-weight:bold; font-size:12px;')
        cl.addWidget(lbl)
        cl.addStretch()
        btn = QPushButton('Limpiar')
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            'QPushButton { color:#bbb; background:#2b2b2b; border:1px solid '
            '#3a3a3a; border-radius:3px; padding:2px 8px; font-size:10px; }'
            'QPushButton:hover { color:#fff; background:#3a3a3a; }')
        btn.clicked.connect(self.limpiar)
        cl.addWidget(btn)
        raiz.addWidget(cab)

        cols = QWidget()
        cols.setStyleSheet('background:transparent;')
        hl = QHBoxLayout(cols)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(3)
        for i, c in enumerate(columnas):
            if i:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet('color:#3a3a3a;')
                hl.addWidget(sep)
            col = ColumnaAlertas(c, max_alertas=max_alertas)
            self.columnas[c.clave] = col
            hl.addWidget(col, stretch=1)
        raiz.addWidget(cols, stretch=1)

    @Slot(dict)
    def add_alert(self, alerta: dict) -> None:
        """Nombre en ingles a proposito: es el slot al que ya se conectan los
        `alert_received` de los clientes; cambiarlo obligaria a tocarlos."""
        try:
            if not isinstance(alerta, dict):
                return
            clave = self._clasificar(alerta) or ''
            col = self.columnas.get(clave)
            if col is not None:
                col.anadir(alerta)
        except Exception:
            pass            # una alerta mal formada no puede romper el panel

    def limpiar(self) -> None:
        for col in self.columnas.values():
            col.limpiar()

    # Alias por compatibilidad con los paneles antiguos de los clientes.
    clear_alerts = limpiar
