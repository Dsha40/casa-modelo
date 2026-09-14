"""
Pruebas de la línea de cruce (`elde_core.ui.linea_cruce`).

Lo que más se protege:

  - **la convención del lado**: DENTRO a la derecha caminando de P1 a P2
    (con la `y` de imagen hacia abajo) — es TODO el contrato con
    `permanencia_linea.py` del servidor; si esto se rompe, las entradas se
    cuentan como salidas en silencio;
  - **invertir voltea el lado sin mover la línea** — el arreglo de un
    segundo cuando quedó al revés;
  - **el label compartido no cambia para los otros tres clientes** — la
    línea nace apagada y viaja en su propia señal.
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

from elde_core.ui.flecha_direccion import LARGO_MINIMO  # noqa: E402
from elde_core.ui.interactive_image_label import (  # noqa: E402
    Interactive_imageLabel)
from elde_core.ui.linea_cruce import LineaCruce  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _horizontal(activa=True):
    """P1 en (200,500), P2 en (800,500): DENTRO queda ABAJO (y > 500)."""
    return LineaCruce([[200, 500], [800, 500]], activa=activa)


def _raton(tipo, pos):
    return QMouseEvent(tipo, QPointF(pos.x(), pos.y()), Qt.LeftButton,
                       Qt.LeftButton, Qt.NoModifier)


# ── La convención del lado ──────────────────────────────────────────────
def test_dentro_queda_a_la_derecha_de_p1_p2():
    linea = _horizontal()
    assert linea.lado(QPoint(500, 700)) == 1, 'abajo debe ser DENTRO'
    assert linea.lado(QPoint(500, 300)) == -1, 'arriba debe ser FUERA'
    assert linea.lado(QPoint(500, 500)) == 0, 'sobre la línea no hay lado'


def test_la_normal_apunta_hacia_dentro():
    nx, ny = _horizontal().normal_dentro()
    assert abs(nx) < 1e-6 and abs(ny - 1.0) < 1e-6


def test_la_misma_convencion_que_el_servidor():
    """El mismo punto debe caer del mismo lado aquí y en el detector."""
    sys.path.insert(0, str(RAIZ.parents[0] / '..' / 'server'))
    try:
        from src.analityc.core.permanencia_linea import _geometria
    except ImportError:
        return                              # sin el server a mano, se omite
    linea = _horizontal()
    geo = _geometria(linea.coordenadas_pixel(1000, 1000))
    for punto in (QPoint(500, 700), QPoint(500, 300), QPoint(100, 800)):
        lado_ui = linea.lado(punto)
        dist = geo.distancia_firmada((punto.x(), punto.y()))
        lado_servidor = 1 if dist > 0 else (-1 if dist < 0 else 0)
        assert lado_ui == lado_servidor, f'{punto} discrepa entre UI y server'


def test_invertir_voltea_el_lado_sin_mover_la_linea():
    linea = _horizontal()
    linea.invertir()
    assert linea.lado(QPoint(500, 700)) == -1
    assert linea.lado(QPoint(500, 300)) == 1
    assert sorted(linea.a_lista()) == [[200, 500], [800, 500]]


def test_una_linea_degenerada_no_tiene_normal():
    assert LineaCruce([[500, 500], [500, 500]], activa=True).normal_dentro() \
        is None


def test_una_linea_corta_no_es_utilizable():
    corta = LineaCruce([[500, 500], [500 + int(LARGO_MINIMO) - 5, 500]],
                       activa=True)
    assert not corta.es_utilizable()


# ── Pintado ─────────────────────────────────────────────────────────────
def test_se_pinta_sin_reventar():
    from elde_core.ui.flecha_direccion import INDICE_MEDIO
    lienzo = QPixmap(400, 300)
    painter = QPainter(lienzo)
    try:
        for linea in (_horizontal(),
                      LineaCruce([[10, 10], [12, 12]], activa=True),
                      _horizontal(activa=False)):
            for indice in (-1, 0, INDICE_MEDIO):
                linea.pintar(painter,
                             lambda p: QPoint(p.x() // 4, p.y() // 4),
                             10, indice)
    finally:
        painter.end()


# ── El label compartido ─────────────────────────────────────────────────
def _label_con_linea():
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_linea(True)
    label.set_linea([[200, 500], [800, 500]])
    return label


def test_el_label_nace_SIN_linea():
    """Los otros tres clientes no deben notar nada."""
    label = Interactive_imageLabel('x')
    assert label.linea.activa is False
    assert label.get_linea_coordinates(1920, 1080) == []
    assert label._interactive() is False


def test_el_label_no_manda_una_linea_inservible():
    label = _label_con_linea()
    assert len(label.get_linea_coordinates(1920, 1080)) == 2
    label.set_linea([[500, 500], [505, 500]])        # demasiado corta
    assert label.get_linea_coordinates(1920, 1080) == []


def test_arrastrarla_emite_su_propia_senal():
    label = _label_con_linea()
    recibido = {'linea': [], 'flecha': []}
    label.linea_change.connect(lambda *a: recibido['linea'].append(a))
    label.flecha_change.connect(lambda *a: recibido['flecha'].append(a))
    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(400, 500)))
    label.mouseMoveEvent(_raton(QEvent.MouseMove, QPoint(450, 450)))
    label.mouseReleaseEvent(_raton(QEvent.MouseButtonRelease, QPoint(450, 450)))
    assert label.linea.a_lista() == [[250, 450], [850, 450]]
    assert len(recibido['linea']) == 1
    assert recibido['flecha'] == [], 'la señal de la flecha no es la suya'


def test_doble_clic_sobre_la_linea_voltea_los_lados():
    label = _label_con_linea()
    recibido = []
    label.linea_change.connect(lambda pts, act: recibido.append((pts, act)))
    label.mouseDoubleClickEvent(_raton(QEvent.MouseButtonDblClick,
                                       QPoint(500, 500)))
    assert label.linea.a_lista() == [[800, 500], [200, 500]]
    assert len(recibido) == 1 and recibido[0][1] is True


def test_invertir_linea_desde_el_menu_emite_para_guardar():
    label = _label_con_linea()
    recibido = []
    label.linea_change.connect(lambda pts, act: recibido.append((pts, act)))
    label.invertir_linea()
    assert label.linea.a_lista() == [[800, 500], [200, 500]]
    assert len(recibido) == 1


def test_con_linea_y_flecha_activas_manda_la_linea():
    """El cliente las hace excluyentes; si un estado raro deja las dos,
    el arrastre y el doble clic van a la línea (la herramienta elegida)."""
    label = _label_con_linea()
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])       # exactamente encima
    antes_flecha = label.flecha.a_lista()
    label.mouseDoubleClickEvent(_raton(QEvent.MouseButtonDblClick,
                                       QPoint(500, 500)))
    assert label.linea.a_lista() == [[800, 500], [200, 500]]
    assert label.flecha.a_lista() == antes_flecha


def test_la_linea_cede_el_arrastre_a_las_barras():
    label = _label_con_linea()
    assert label.hay_punto_en(QPoint(500, 500)) is True
    assert label.hay_punto_en(QPoint(500, 100)) is False
    assert label.gadget_direccion_en(QPoint(500, 500)) is True


# ── Ejecutable sin pytest (ver server/tests/test_stand_almacen.py) ───────
if __name__ == '__main__':
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        try:
            _fn()
            print(f"  OK    {_nombre}")
        except Exception as _exc:                          # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
