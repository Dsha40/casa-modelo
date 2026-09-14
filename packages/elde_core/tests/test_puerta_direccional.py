"""
Pruebas de la puerta direccional (`elde_core.ui.puerta_direccional`).

La puerta es la línea de cruce con otro rótulo: en vez de DENTRO/FUERA pinta
una flecha DOBLE con las cabezas ENTRADA (verde) y SALIDA (roja). Como la
geometría es la misma, viaja por el mismo campo del contrato y la lee el
mismo detector del servidor.

Lo que más se protege:

  - **la cabeza verde cae del lado DENTRO** — es el único puente entre lo
    que el operador ve y lo que el servidor mide. Si se despega, la puerta
    enseñaría ENTRADA donde el servidor cuenta salidas, en silencio y sin
    que nada falle;
  - **la puerta y la línea producen el MISMO dato** — porque comparten campo
    en el contrato; si divergieran, cambiar de herramienta cambiaría el
    resultado sin avisar;
  - **cada herramienta guarda sus propios puntos** — son dos calibraciones
    de la misma cámara y encender una no puede pisar la otra;
  - **los otros clientes no notan nada** — nace apagada y en su señal.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from elde_core.ui.flecha_direccion import (INDICE_MEDIO,  # noqa: E402
                                           LARGO_MINIMO)
from elde_core.ui.interactive_image_label import (  # noqa: E402
    Interactive_imageLabel)
from elde_core.ui.linea_cruce import LineaCruce  # noqa: E402
from elde_core.ui.puerta_direccional import PuertaDireccional  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _horizontal(activa=True):
    """P1 en (200,500), P2 en (800,500): DENTRO queda ABAJO (y > 500)."""
    return PuertaDireccional([[200, 500], [800, 500]], activa=activa)


def _raton(tipo, pos):
    return QMouseEvent(tipo, QPointF(pos.x(), pos.y()), Qt.LeftButton,
                       Qt.LeftButton, Qt.NoModifier)


# ── El puente entre lo que se ve y lo que se mide ───────────────────────
def test_la_cabeza_verde_cae_del_lado_dentro():
    """LA prueba: ENTRADA se pinta donde el servidor cuenta las entradas.

    El pintado sitúa `apex_dentro` en `medio + normal * brazo`. Aquí se
    reproduce esa cuenta y se comprueba con `lado()`, que es la MISMA
    fórmula que usa el detector del servidor.
    """
    puerta = _horizontal()
    nx, ny = puerta.normal_dentro()
    medio = puerta.punto_medio()
    apex_dentro = QPoint(int(medio.x() + nx * 100), int(medio.y() + ny * 100))
    apex_fuera = QPoint(int(medio.x() - nx * 100), int(medio.y() - ny * 100))
    assert puerta.lado(apex_dentro) == 1, 'la cabeza VERDE debe caer DENTRO'
    assert puerta.lado(apex_fuera) == -1, 'la cabeza ROJA debe caer FUERA'


def test_invertir_intercambia_entrada_y_salida_sin_mover_la_puerta():
    puerta = _horizontal()
    antes = puerta.normal_dentro()
    puerta.invertir()
    despues = puerta.normal_dentro()
    assert abs(despues[0] + antes[0]) < 1e-6
    assert abs(despues[1] + antes[1]) < 1e-6
    assert sorted(puerta.a_lista()) == [[200, 500], [800, 500]]


def test_la_misma_convencion_que_el_servidor():
    """El mismo punto cae del mismo lado aquí y en el detector de línea."""
    sys.path.insert(0, str(RAIZ.parents[0] / '..' / 'server'))
    try:
        from src.analityc.core.permanencia_linea import _geometria
    except ImportError:
        return                              # sin el server a mano, se omite
    puerta = _horizontal()
    geo = _geometria(puerta.coordenadas_pixel(1000, 1000))
    for punto in (QPoint(500, 700), QPoint(500, 300), QPoint(100, 800)):
        lado_ui = puerta.lado(punto)
        dist = geo.distancia_firmada((punto.x(), punto.y()))
        lado_servidor = 1 if dist > 0 else (-1 if dist < 0 else 0)
        assert lado_ui == lado_servidor, f'{punto} discrepa entre UI y server'


def test_produce_EL_MISMO_dato_que_la_linea():
    """Comparten campo en el contrato: no pueden dar resultados distintos."""
    puntos = [[200, 500], [800, 500]]
    puerta = PuertaDireccional(puntos, activa=True)
    linea = LineaCruce(puntos, activa=True)
    assert (puerta.coordenadas_pixel(1920, 1080)
            == linea.coordenadas_pixel(1920, 1080))
    assert puerta.normal_dentro() == linea.normal_dentro()
    for p in (QPoint(500, 700), QPoint(500, 300), QPoint(500, 500)):
        assert puerta.lado(p) == linea.lado(p)


# ── Geometría heredada ──────────────────────────────────────────────────
def test_una_puerta_corta_no_es_utilizable():
    corta = PuertaDireccional([[500, 500], [500 + int(LARGO_MINIMO) - 5, 500]],
                              activa=True)
    assert not corta.es_utilizable()


def test_una_puerta_degenerada_no_tiene_normal():
    assert PuertaDireccional([[500, 500], [500, 500]],
                             activa=True).normal_dentro() is None


def test_trasladarla_no_cambia_el_sentido():
    puerta = _horizontal()
    antes = puerta.normal_dentro()
    puerta.trasladar(100, -80)
    assert puerta.normal_dentro() == antes
    assert puerta.a_lista() == [[300, 420], [900, 420]]


def test_el_brazo_esta_acotado_por_los_dos_lados():
    """Ni una flecha fuera del encuadre ni una del tamaño de un punto."""
    puerta = _horizontal()
    assert puerta._brazo(4.0) >= 40.0, 'una puerta estrecha aún debe verse'
    assert puerta._brazo(5000.0) <= 95.0, 'no puede salirse del encuadre'
    medio = puerta._brazo(200.0)
    assert 40.0 <= medio <= 95.0


# ── Pintado ─────────────────────────────────────────────────────────────
def test_se_pinta_sin_reventar():
    lienzo = QPixmap(400, 300)
    painter = QPainter(lienzo)
    try:
        for puerta in (_horizontal(),
                       PuertaDireccional([[10, 10], [12, 12]], activa=True),
                       PuertaDireccional([[0, 0], [1000, 1000]], activa=True),
                       _horizontal(activa=False)):
            for indice in (-1, 0, 1, INDICE_MEDIO):
                puerta.pintar(painter,
                              lambda p: QPoint(p.x() // 4, p.y() // 4),
                              10, indice)
    finally:
        painter.end()


def test_apagada_no_pinta_nada():
    lienzo = QPixmap(50, 50)
    painter = QPainter(lienzo)
    try:
        pintado = []
        _horizontal(activa=False).pintar(
            painter, lambda p: pintado.append(p) or QPoint(0, 0), 10, -1)
        assert pintado == [], 'apagada no debe ni escalar sus puntos'
    finally:
        painter.end()


# ── El label compartido ─────────────────────────────────────────────────
def _label_con_puerta():
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_puerta(True)
    label.set_puerta([[200, 500], [800, 500]])
    return label


def test_el_label_nace_SIN_puerta():
    """Los clientes que no la enciendan no deben notar nada."""
    label = Interactive_imageLabel('x')
    assert label.puerta.activa is False
    assert label.get_puerta_coordinates(1920, 1080) == []
    assert label.get_cruce_coordinates(1920, 1080) == []
    assert label._interactive() is False


def test_el_label_no_manda_una_puerta_inservible():
    label = _label_con_puerta()
    assert len(label.get_puerta_coordinates(1920, 1080)) == 2
    label.set_puerta([[500, 500], [505, 500]])       # demasiado corta
    assert label.get_puerta_coordinates(1920, 1080) == []


def test_get_cruce_prefiere_la_puerta_y_cae_a_la_linea():
    """El recuadro pide «el umbral» sin preguntar cuál está encendida."""
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_linea(True)
    label.set_linea([[100, 300], [900, 300]])
    assert label.get_cruce_coordinates(1000, 1000) == [[100, 300], [900, 300]]

    label.toggle_puerta(True)
    label.set_puerta([[200, 700], [800, 700]])
    assert label.get_cruce_coordinates(1000, 1000) == [[200, 700], [800, 700]]

    label.toggle_puerta(False)
    assert label.get_cruce_coordinates(1000, 1000) == [[100, 300], [900, 300]]


def test_cada_herramienta_guarda_sus_PROPIOS_puntos():
    """Son dos calibraciones de la misma cámara: no pueden pisarse."""
    label = Interactive_imageLabel('x')
    label.set_linea([[100, 300], [900, 300]])
    label.set_puerta([[200, 700], [800, 700]])
    label.set_flecha([[400, 100], [400, 900]])
    assert label.linea.a_lista() == [[100, 300], [900, 300]]
    assert label.puerta.a_lista() == [[200, 700], [800, 700]]
    assert label.flecha.a_lista() == [[400, 100], [400, 900]]


def test_arrastrarla_emite_su_propia_senal():
    label = _label_con_puerta()
    recibido = {'puerta': [], 'linea': [], 'flecha': []}
    label.puerta_change.connect(lambda *a: recibido['puerta'].append(a))
    label.linea_change.connect(lambda *a: recibido['linea'].append(a))
    label.flecha_change.connect(lambda *a: recibido['flecha'].append(a))
    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(400, 500)))
    label.mouseMoveEvent(_raton(QEvent.MouseMove, QPoint(450, 450)))
    label.mouseReleaseEvent(_raton(QEvent.MouseButtonRelease,
                                   QPoint(450, 450)))
    assert label.puerta.a_lista() == [[250, 450], [850, 450]]
    assert len(recibido['puerta']) == 1
    assert recibido['linea'] == [], 'la señal de la línea no es la suya'
    assert recibido['flecha'] == [], 'la señal de la flecha no es la suya'


def test_doble_clic_sobre_la_puerta_intercambia_entrada_y_salida():
    label = _label_con_puerta()
    recibido = []
    label.puerta_change.connect(lambda *a: recibido.append(a))
    antes = label.puerta.normal_dentro()
    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(500, 500)))
    label.mouseDoubleClickEvent(
        _raton(QEvent.MouseButtonDblClick, QPoint(500, 500)))
    despues = label.puerta.normal_dentro()
    assert abs(despues[1] + antes[1]) < 1e-6, 'debe haberse volteado'
    assert sorted(label.puerta.a_lista()) == [[200, 500], [800, 500]]
    assert recibido, 'invertir debe avisar al cliente para que lo guarde'


def test_invertir_puerta_apagada_no_hace_nada():
    label = Interactive_imageLabel('x')
    label.set_puerta([[200, 500], [800, 500]])
    label.invertir_puerta()
    assert label.puerta.a_lista() == [[200, 500], [800, 500]]


def test_la_puerta_manda_sobre_la_linea_si_las_dos_estan_encendidas():
    """El cliente las mantiene excluyentes; esto es solo el desempate."""
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_linea(True)
    label.toggle_puerta(True)
    assert label._gadget_activo() is label.puerta


def test_la_puerta_se_agarra_aunque_haya_un_ROI_debajo():
    """Sus EXTREMOS ganan a los vértices del ROI, como los de la línea."""
    label = _label_con_puerta()
    label.toggle_roi(True) if hasattr(label, 'toggle_roi') else None
    label.show_points = True
    label.points = label.list_to_qpoints([[200, 500], [900, 900]])
    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(200, 500)))
    assert label.flecha_index == 0, 'debió agarrar el extremo de la puerta'
    assert label.active_point_index == -1
