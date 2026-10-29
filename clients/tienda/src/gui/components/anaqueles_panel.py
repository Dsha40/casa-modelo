"""Panel en vivo de ANAQUELES: qué nivel del estante se visita y se agarra más.

Acompaña al modo `Anaqueles` del servidor. Responde, de arriba abajo:

  1. **¿Cuánto movimiento hubo hoy?** — visitas y agarres totales del día.
  2. **¿Qué área gana?** — ranking por agarres (la señal fuerte de marketing:
     «lo tocó»), con las visitas («se acercó») al lado y una barra
     proporcional para leerlo de un vistazo.
  3. **¿Qué acaba de pasar?** — feed de agarres/visitas con hora y
     demografía.

Los totales que llegan del servidor son ABSOLUTOS del día (persistidos en
disco allá): tras una reconexión el panel simplemente muestra el siguiente
bloque y ya está al día — no acumula nada por su cuenta.

## Por qué acumula por cámara

Mismo motivo que el panel de permanencia de detecciones: puede haber varias
cámaras en modo Anaqueles a la vez, cada una con sus áreas. El panel guarda
el último bloque de CADA cámara y pinta una sección por cámara; una cámara
callada más de `CADUCIDAD_CAMARA_S` se da por desconectada y deja de sumar,
para no enseñar cifras viejas como si fueran de ahora.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

# La misma paleta del panel de permanencia: los laterales deben verse del
# mismo sistema.
FONDO = '#1e1e1e'
FONDO_CAB = '#252525'
BORDE = '#3a3a3a'
TEXTO = '#e8e8e8'
TEXTO_SUAVE = '#9a9a9a'

NARANJA = '#ffa028'    # el color de las áreas en el visor
VERDE = '#4caf50'      # visitas
AMBAR = '#f0b429'      # avisos
AZUL = '#58a6ff'       # personal de seguridad en pasillo
VIOLETA = '#a371f7'    # empleado surtiendo el anaquel
ROJO = '#f85149'       # posible agotado (5-oct-2026)

#: Una cámara que lleva esto sin emitir se considera desconectada.
CADUCIDAD_CAMARA_S = 20.0

#: Sucesos que se conservan en el feed.
MAX_EVENTOS = 40

#: Ranking y feed se reconstruyen solo si cambia lo que enseñan, y como mucho
#: cada esto (28-oct-2026). Antes se rehacían con CADA respuesta del servidor
#: (~13 por segundo con el vídeo), casi siempre iguales: con 4 áreas y 40
#: sucesos, 22 ms por respuesta y otros 60-85 ms de lo que Qt hace después
#: (borrar los widgets viejos, dar estilo y maquetar los nuevos), medido en el
#: PC de la tienda. La telemetría del ciclo solo veía los 22. Las cifras, los
#: avisos y el stock se siguen escribiendo con cada respuesta: son baratos.
RECONSTRUIR_CADA_S = 0.5

_GENERO_LEGIBLE = {'male': 'Hombre', 'female': 'Mujer',
                   'Male': 'Hombre', 'Female': 'Mujer'}

_ROL_ICONO = {'seguridad': '🛡️', 'pasillero': '📦'}


class _Tarjeta(QFrame):
    """Un número grande con su etiqueta (mismo dibujo que permanencia)."""

    def __init__(self, titulo: str, color: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('TarjetaAnaqueles')
        self.setStyleSheet(
            f'#TarjetaAnaqueles {{ background:{FONDO_CAB};'
            f' border:1px solid {BORDE}; border-radius:6px; }}')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        caja = QVBoxLayout(self)
        caja.setContentsMargins(10, 8, 10, 8)
        caja.setSpacing(1)
        lbl = QLabel(titulo)
        lbl.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; font-weight:600;')
        self.valor = QLabel('0')
        fuente = QFont()
        fuente.setPointSize(24)
        fuente.setWeight(QFont.Bold)
        self.valor.setFont(fuente)
        self.valor.setStyleSheet(f'color:{color};')
        caja.addWidget(lbl)
        caja.addWidget(self.valor)


class AnaquelesPanel(QWidget):
    """Pestaña lateral del modo Anaqueles del cliente de tienda."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f'background:{FONDO}; color:{TEXTO};')

        #: camera_id -> {'ts': monotonic, 'areas': [...], 'salud': {...},
        #:               'nombre': str}
        self._camaras: Dict[str, Dict[str, Any]] = {}
        self._eventos: List[Dict[str, Any]] = []
        # Dict y no set A PROPÓSITO: conserva el orden de inserción, así la
        # poda se queda con las claves MÁS RECIENTES — superconjunto de las
        # 10 que el servidor reenvía íntegras en cada feed. Con un set, la
        # poda era arbitraria y los sucesos viejos reaparecían duplicados.
        self._vistos: Dict[tuple, None] = {}

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(8, 8, 8, 8)
        raiz.setSpacing(8)

        titulo = QLabel('Anaqueles — interacción por nivel')
        titulo.setStyleSheet('font-size:12px; font-weight:700;')
        raiz.addWidget(titulo)

        fila = QHBoxLayout()
        fila.setSpacing(8)
        self.card_visitas = _Tarjeta('VISITAS HOY', VERDE)
        self.card_agarres = _Tarjeta('AGARRES HOY', NARANJA)
        fila.addWidget(self.card_visitas)
        fila.addWidget(self.card_agarres)
        raiz.addLayout(fila)

        self.lbl_aviso = QLabel('')
        self.lbl_aviso.setWordWrap(True)
        self.lbl_aviso.setStyleSheet(
            f'color:{AMBAR}; font-size:10px; font-weight:600;')
        self.lbl_aviso.hide()
        raiz.addWidget(self.lbl_aviso)

        # «Posible agotado» (5-oct-2026, bloque['stock'] del servidor). Va
        # aparte de los avisos ámbar: esos son de la cámara, esto es de la
        # balda, y es lo que el encargado tiene que ver.
        self.lbl_stock = QLabel('')
        self.lbl_stock.setWordWrap(True)
        self.lbl_stock.setStyleSheet(
            f'color:{ROJO}; font-size:11px; font-weight:700;')
        self.lbl_stock.hide()
        raiz.addWidget(self.lbl_stock)

        # Personal enrolado a la vista AHORA (bloque['personal'] del
        # servidor): una línea discreta, no un aviso — que esté el pasillero
        # es lo normal.
        self.lbl_personal = QLabel('')
        self.lbl_personal.setWordWrap(True)
        self.lbl_personal.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px;')
        self.lbl_personal.hide()
        raiz.addWidget(self.lbl_personal)

        # Ranking (se reconstruye entero cuando cambia: son pocas filas).
        self._zona_ranking = QVBoxLayout()
        self._zona_ranking.setSpacing(4)
        cont_ranking = QWidget()
        cont_ranking.setLayout(self._zona_ranking)

        # Feed de sucesos.
        self._zona_feed = QVBoxLayout()
        self._zona_feed.setSpacing(2)
        cont_feed = QWidget()
        cont_feed.setLayout(self._zona_feed)

        contenedor = QWidget()
        caja = QVBoxLayout(contenedor)
        caja.setContentsMargins(0, 0, 0, 0)
        caja.setSpacing(10)
        caja.addWidget(cont_ranking)
        lbl_feed = QLabel('Últimos sucesos')
        lbl_feed.setStyleSheet(
            f'color:{TEXTO_SUAVE}; font-size:10px; font-weight:700;')
        caja.addWidget(lbl_feed)
        caja.addWidget(cont_feed)
        caja.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(contenedor)
        raiz.addWidget(scroll, stretch=1)

        # Lo último que se pintó en ranking y feed (ver RECONSTRUIR_CADA_S).
        # Si cambian antes del plazo, se pintan una vez al cumplirse.
        self._firma_ranking: Optional[tuple] = None
        self._version_feed = 0
        self._feed_pintado = -1
        self._listas_ts = -1e9
        self._timer_listas = QTimer(self)
        self._timer_listas.setSingleShot(True)
        self._timer_listas.timeout.connect(self._repintar)

        # Refresco periódico: expira cámaras calladas aunque no llegue nada.
        self._timer = QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self._repintar)
        self._timer.start()

        # «Lleva X» de cada posible agotado, de segundo en segundo (5-oct-
        # 2026, lo pidió el usuario). Solo reescribe ese letrero: el
        # repintado entero sigue a su ritmo de 5 s.
        self._timer_stock = QTimer(self)
        self._timer_stock.setInterval(1000)
        self._timer_stock.timeout.connect(self._refrescar_stock)
        self._timer_stock.start()

    # ── Entrada ─────────────────────────────────────────────────────────
    @Slot(dict)
    def actualizar(self, bloque: Dict[str, Any]) -> None:
        """Recibe `metadata['anaqueles']` reemitido por un render_box."""
        try:
            cam = str(bloque.get('camera_id') or bloque.get('camera') or '?')
            estado = self._camaras.setdefault(cam, {})
            estado['ts'] = time.monotonic()
            estado['nombre'] = str(bloque.get('camera_name') or cam)
            if 'areas' in bloque:
                estado['areas'] = list(bloque.get('areas') or [])
            if 'salud' in bloque:
                estado['salud'] = dict(bloque.get('salud') or {})
            if bloque.get('sin_areas'):
                estado['sin_areas'] = True
            elif 'areas' in bloque:
                estado['sin_areas'] = False
            if bloque.get('escena'):
                estado['escena_ts'] = time.monotonic()
            if 'stock' in bloque:
                estado['stock'] = dict(bloque.get('stock') or {})
                # Base del contador: el servidor manda los segundos que
                # llevaba al emitir; desde aquí se suma con el reloj PROPIO
                # (el del servidor puede ir en otra zona horaria).
                estado['stock_ts'] = time.monotonic()
            # Personal a la vista: se guarda POR CÁMARA con un TTL propio de
            # 10 s (aplicado en _repintar), además de la caducidad de la
            # cámara: un servidor viejo no reemite la lista vacía al salir el
            # último empleado y sin el TTL el letrero se quedaba pegado.
            if 'personal' in bloque:
                estado['personal'] = list(bloque.get('personal') or [])
                estado['personal_ts'] = time.monotonic()

            for ev in (bloque.get('eventos') or []):
                # 'nombre' además de 'area': un servidor en caliente puede no
                # mandar todavía 'area' en los eventos de personal, y sin el
                # nombre dos sucesos distintos del mismo segundo colisionan.
                clave = (cam, ev.get('ts'), ev.get('tipo'),
                         ev.get('area'), ev.get('nombre'),
                         ev.get('id_corto'))
                if clave in self._vistos:
                    continue
                self._vistos[clave] = None
                self._eventos.insert(0, dict(ev, camara=estado['nombre']))
                self._version_feed += 1
            del self._eventos[MAX_EVENTOS:]
            if len(self._vistos) > 400:
                for k in list(self._vistos)[:-200]:
                    del self._vistos[k]

            self._repintar()
        except Exception as exc:                            # noqa: BLE001
            print(f'[anaqueles-panel] error actualizando: {exc}')

    # ── Pintura ─────────────────────────────────────────────────────────
    def _vivas(self) -> Dict[str, Dict[str, Any]]:
        ahora = time.monotonic()
        return {c: e for c, e in self._camaras.items()
                if (ahora - float(e.get('ts') or 0)) < CADUCIDAD_CAMARA_S}

    def _repintar(self) -> None:
        vivas = self._vivas()
        areas_todas: List[Dict[str, Any]] = []
        avisos: List[str] = []
        for cam, estado in vivas.items():
            for a in (estado.get('areas') or []):
                areas_todas.append(dict(a, camara=estado.get('nombre', cam)))
            salud = estado.get('salud') or {}
            nombre = estado.get('nombre', cam)
            if estado.get('sin_areas'):
                avisos.append(f'{nombre}: sin áreas dibujadas — añade áreas '
                              'de anaquel en el menú ROI del recuadro.')
            if salud.get('suspendido'):
                avisos.append(f'{nombre}: conteo suspendido — la escena '
                              'acaba de cambiar (¿canal de iVMS?).')
            elif (time.monotonic()
                  - float(estado.get('escena_ts') or -1e9)) < 10.0:
                avisos.append(f'{nombre}: la escena cambió — revisa que las '
                              'áreas sigan sobre el estante correcto.')
            if salud.get('sin_personas_s'):
                mins = int(salud['sin_personas_s']) // 60
                avisos.append(f'{nombre}: {mins} min sin ver una persona — '
                              '¿la cámara muestra la escena correcta?')
            if salud.get('pose_disponible') is False:
                avisos.append(f'{nombre}: modelo de pose no disponible — '
                              'se cuentan visitas pero NO agarres.')

        self.card_visitas.valor.setText(
            str(sum(int(a.get('visitas') or 0) for a in areas_todas)))
        self.card_agarres.valor.setText(
            str(sum(int(a.get('agarres') or 0) for a in areas_todas)))

        if avisos:
            self.lbl_aviso.setText('\n'.join(f'⚠ {a}' for a in avisos[:4]))
            self.lbl_aviso.show()
        else:
            self.lbl_aviso.hide()

        self._refrescar_stock(vivas)

        en_camara: List[str] = []
        for estado in vivas.values():
            # TTL propio del bloque de personal: el servidor nuevo ya emite
            # la lista vacía al salir el último empleado; esto es el cinturón
            # para un servidor viejo que simplemente deja de mandar la clave.
            if (time.monotonic()
                    - float(estado.get('personal_ts') or 0)) > 10.0:
                continue
            for p in (estado.get('personal') or []):
                icono = _ROL_ICONO.get(str(p.get('rol') or ''), '👤')
                etiqueta = f"{icono} {p.get('nombre')}"
                if etiqueta not in en_camara:
                    en_camara.append(etiqueta)
        if en_camara:
            self.lbl_personal.setText(
                'En cámara ahora: ' + ' · '.join(en_camara))
            self.lbl_personal.show()
        else:
            self.lbl_personal.hide()

        self._reconstruir_listas(areas_todas, varias=len(vivas) > 1)

    def _reconstruir_listas(self, areas: List[Dict[str, Any]],
                            varias: bool) -> None:
        """Ranking y feed, solo si cambió lo que enseñan y como mucho cada
        `RECONSTRUIR_CADA_S`. Un cambio que llega antes del plazo se pinta
        al cumplirse, con lo último que haya: no se pierde ninguno."""
        firma = _firma_ranking(areas, varias)
        cambia_ranking = firma != self._firma_ranking
        cambia_feed = self._version_feed != self._feed_pintado
        if not (cambia_ranking or cambia_feed):
            return
        espera = RECONSTRUIR_CADA_S - (time.monotonic() - self._listas_ts)
        if espera > 0:
            if not self._timer_listas.isActive():
                self._timer_listas.start(int(espera * 1000) + 1)
            return
        self._listas_ts = time.monotonic()
        if cambia_ranking:
            self._firma_ranking = firma
            self._pintar_ranking(areas, varias)
        if cambia_feed:
            self._feed_pintado = self._version_feed
            self._pintar_feed()

    def _refrescar_stock(self, vivas=None) -> None:
        lineas = lineas_de_stock(self._vivas() if vivas is None else vivas)
        if lineas:
            self.lbl_stock.setText('\n'.join(lineas))
            self.lbl_stock.show()
        else:
            self.lbl_stock.hide()

    def _pintar_ranking(self, areas: List[Dict[str, Any]],
                        varias: bool) -> None:
        _vaciar(self._zona_ranking)
        if not areas:
            lbl = QLabel('Sin datos todavía: conecta una cámara en modo '
                         'Anaqueles y dibuja las áreas del estante.')
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f'color:{TEXTO_SUAVE}; font-size:10px;')
            self._zona_ranking.addWidget(lbl)
            return

        orden = sorted(areas, key=lambda a: (-int(a.get('agarres') or 0),
                                             -int(a.get('visitas') or 0)))
        tope = max(1, max(int(a.get('agarres') or 0) for a in orden))
        for pos, a in enumerate(orden, start=1):
            fila = QFrame()
            fila.setStyleSheet(
                f'background:{FONDO_CAB}; border:1px solid {BORDE};'
                'border-radius:4px;')
            caja = QVBoxLayout(fila)
            caja.setContentsMargins(8, 5, 8, 5)
            caja.setSpacing(2)

            nombre = str(a.get('nombre') or f"Área {int(a.get('idx', 0)) + 1}")
            if varias:
                nombre = f"{a.get('camara')} · {nombre}"
            ahora_n = int(a.get('personas_ahora') or 0)
            sufijo = f'  ·  {ahora_n} ahora' if ahora_n else ''
            cab = QLabel(
                f"<b>{pos}. {nombre}</b>"
                f"<span style='color:{TEXTO_SUAVE};'>{sufijo}</span>")
            cab.setStyleSheet('font-size:11px; border:none;')
            caja.addWidget(cab)

            cifras = QLabel(
                f"<span style='color:{NARANJA}; font-weight:700;'>"
                f"{int(a.get('agarres') or 0)} agarres</span>"
                f"<span style='color:{TEXTO_SUAVE};'>  ·  "
                f"{int(a.get('visitas') or 0)} visitas</span>")
            cifras.setStyleSheet('font-size:10px; border:none;')
            caja.addWidget(cifras)

            barra = QProgressBar()
            barra.setRange(0, tope)
            barra.setValue(min(tope, int(a.get('agarres') or 0)))
            barra.setTextVisible(False)
            barra.setFixedHeight(5)
            barra.setStyleSheet(
                f'QProgressBar {{ background:{BORDE}; border:none;'
                'border-radius:2px; }}'
                f'QProgressBar::chunk {{ background:{NARANJA};'
                'border-radius:2px; }}')
            caja.addWidget(barra)
            self._zona_ranking.addWidget(fila)

    def _pintar_feed(self) -> None:
        _vaciar(self._zona_feed)
        if not self._eventos:
            lbl = QLabel('—')
            lbl.setStyleSheet(f'color:{TEXTO_SUAVE}; font-size:10px;')
            self._zona_feed.addWidget(lbl)
            return
        for ev in self._eventos[:MAX_EVENTOS]:
            tipo = str(ev.get('tipo') or '')
            # En 'seguridad'/'surte' el id_corto es el NOMBRE del empleado
            # enrolado y 'nombre' el área (vacía en seguridad).
            if tipo == 'seguridad':
                color = AZUL
                cuerpo = f"🛡️ {ev.get('id_corto')} en el pasillo"
            elif tipo == 'surte':
                color = VIOLETA
                cuerpo = (f"📦 {ev.get('id_corto')} surte "
                          f"<b>{ev.get('nombre')}</b>")
            elif tipo == 'agotado':
                color = ROJO
                cuerpo = (f"🛒 Posible agotado: <b>{ev.get('producto')}</b>"
                          f" en {ev.get('nombre')}")
            elif tipo == 'repuesto':
                color = VERDE
                estuvo = (f" (estuvo {duracion(ev['duracion_s'])})"
                          if ev.get('duracion_s') is not None else '')
                cuerpo = (f"✅ Repuesto: <b>{ev.get('producto')}</b>"
                          f" en {ev.get('nombre')}{estuvo}")
            else:
                color = NARANJA if tipo == 'agarre' else VERDE
                verbo = 'agarró en' if tipo == 'agarre' else 'visitó'
                quien = _GENERO_LEGIBLE.get(str(ev.get('genero')),
                                            'Persona')
                rango = f" ({ev.get('rango_etario')})" if ev.get(
                    'rango_etario') else ''
                cuerpo = f"{quien}{rango} {verbo} <b>{ev.get('nombre')}</b>"
            lbl = QLabel(
                f"<span style='color:{TEXTO_SUAVE};'>{ev.get('ts')}</span>  "
                f"<span style='color:{color}; font-weight:700;'>●</span> "
                f"{cuerpo}")
            lbl.setStyleSheet('font-size:10px;')
            self._zona_feed.addWidget(lbl)


def _firma_ranking(areas: List[Dict[str, Any]], varias: bool) -> tuple:
    """Todo lo que enseña el ranking: si no cambia, no se reconstruye."""
    return (varias, tuple(
        (a.get('camara'), a.get('nombre'), a.get('idx'),
         int(a.get('agarres') or 0), int(a.get('visitas') or 0),
         int(a.get('personas_ahora') or 0)) for a in areas))


def duracion(segundos: Any) -> str:
    """«45 s», «2 min 05 s», «1 h 03 min» (el mismo formato que el log del
    servidor, `anaqueles_stock._duracion`)."""
    s = max(0, int(segundos or 0))
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60} min {s % 60:02d} s'
    return f'{s // 3600} h {(s % 3600) // 60:02d} min'


def lineas_de_stock(vivas: Dict[str, Dict[str, Any]],
                    ahora: Optional[float] = None) -> List[str]:
    """Texto del letrero de stock a partir del último `stock` de cada
    cámara viva. Suelta de la clase para probarla sin pintar nada.

    «lleva X» = los segundos que el servidor contaba al emitir
    (`vacio_s`) + lo que ha pasado desde que llegaron, por el reloj
    monotónico de este PC. Cuenta desde que se DETECTÓ, unos segundos
    después del agarre. Un servidor anterior al contador no manda
    `vacio_s`: entonces solo se dice la hora.
    """
    ahora = time.monotonic() if ahora is None else ahora
    lineas: List[str] = []
    varias = len(vivas) > 1
    for cam, estado in vivas.items():
        stock = estado.get('stock') or {}
        prefijo = f"{estado.get('nombre', cam)} · " if varias else ''
        pasado = max(0.0, ahora - float(estado.get('stock_ts') or ahora))
        for a in (stock.get('alertas') or []):
            lleva = (f" — lleva {duracion(a['vacio_s'] + pasado)}"
                     if a.get('vacio_s') is not None else '')
            lineas.append(f"🛒 {prefijo}Posible agotado: {a.get('producto')}"
                          f" ({a.get('area_nombre')}){lleva}"
                          f" (desde {a.get('desde')})")
        if stock.get('esperando_foto'):
            lineas.append(f"📷 {prefijo}Foto de referencia: esperando a que "
                          f"nadie tape {stock['esperando_foto']} producto(s)")
    return lineas


def _vaciar(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
