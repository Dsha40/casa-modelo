"""La zona del PERSONAL en la etiqueta compartida.

Es la tercera zona del modo comedor: quien la ocupa queda excluido del conteo
de clientes. Se anadio el 26-ago-2026, cuando Amazonas se reconcilio con el
nucleo.

Lo que mas se protege aqui es la SENAL. `point_change` lleva seis argumentos y
la usan los cuatro clientes: ampliarla los rompe a todos de golpe, asi que la
zona nueva viaja en `personal_change`, como ya hacen la flecha, la linea y las
areas de anaquel. La rama de `mouseReleaseEvent` que decide cual de las dos se
dispara es la linea mas delicada del cambio, y hasta esta prueba no la cubria
nada.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from elde_core.ui.interactive_image_label import Interactive_imageLabel  # noqa: E402

# Una sola instancia para todo el modulo, como en test_flecha_direccion.py.
_app = QApplication.instance() or QApplication([])

_ZONA = [[100, 100], [900, 100], [900, 900], [100, 900]]


def _soltar(label):
    """Suelta el boton donde el arrastre en curso decida su senal."""
    label.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, QPointF(0, 0), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))


# ── Nace apagada: los otros clientes no notan nada ───────────────────────

def test_la_zona_nace_apagada():
    label = Interactive_imageLabel('x')
    assert label.personal_zone_active is False
    assert label.get_personal_zone_coordinates(1920, 1080) == []


def test_apagada_devuelve_lista_vacia_aunque_este_dibujada():
    """La lista vacia es la senal de «no dibujada» que espera el servidor: sin
    ella, una zona apagada seguiria excluyendo del conteo a quien pasara por
    ese rincon."""
    label = Interactive_imageLabel('x', personal_zone=_ZONA)
    label.toggle_personal_zone(False)
    assert label.get_personal_zone_coordinates(1920, 1080) == []

    label.toggle_personal_zone(True)
    puntos = label.get_personal_zone_coordinates(1920, 1080)
    assert len(puntos) == 4, 'encendida SI debe viajar'
    # 100/1000 * 1920 = 192 ; 100/1000 * 1080 = 108
    assert puntos[0] == [192, 108], 'convierte 0..1000 a pixeles del frame'


# ── LA prueba: la señal propia ───────────────────────────────────────────

def test_la_zona_del_personal_viaja_en_su_propia_senal():
    """`point_change` tiene 6 argumentos y la usan los cuatro clientes.

    Si alguien invierte la rama de `mouseReleaseEvent` y la zona empieza a
    viajar en `point_change`, esta prueba cae.
    """
    label = Interactive_imageLabel('x')
    propias, del_roi = [], []
    label.personal_change.connect(lambda pts, act: propias.append((pts, act)))
    label.point_change.connect(lambda *a: del_roi.append(a))

    label.toggle_personal_zone(True)
    label.set_edit_target('personal')
    label.set_personal_zone(_ZONA)
    propias.clear()          # el toggle ya emitio: aqui interesa el arrastre
    label.active_point_index = 0     # el operador tiene un vertice agarrado
    _soltar(label)

    assert len(propias) == 1, 'debe emitir personal_change al soltar'
    assert propias[0][1] is True
    assert not del_roi, 'NO puede emitir point_change: rompe a los 4 clientes'


def test_el_roi_sigue_emitiendo_point_change_con_seis_argumentos():
    """La no-regresion del resto del producto."""
    label = Interactive_imageLabel('x', roi=_ZONA, roi_active=True)
    recibido = []
    label.point_change.connect(lambda *a: recibido.append(a))

    label.set_edit_target('roi')
    label.active_point_index = 0     # el operador tiene un vertice agarrado
    _soltar(label)

    assert len(recibido) == 1
    assert len(recibido[0]) == 6, 'los cuatro save_point esperan seis'


# ── Edición ──────────────────────────────────────────────────────────────

def test_set_edit_target_admite_personal():
    label = Interactive_imageLabel('x')
    label.set_edit_target('personal')
    assert label.edit_target == 'personal'


def test_un_destino_inventado_no_se_cuela():
    label = Interactive_imageLabel('x')
    label.set_edit_target('roi')
    label.set_edit_target('terraza')
    assert label.edit_target == 'roi'


def test_encenderla_la_hace_interactiva():
    """Si no, las barras flotantes se quedan el clic y no se puede dibujar."""
    label = Interactive_imageLabel('x')
    assert label._interactive() is False
    label.toggle_personal_zone(True)
    assert label._interactive() is True


def test_sus_vertices_se_pueden_agarrar():
    """`hay_punto_en` es lo que hace que el recuadro ceda el arrastre."""
    label = Interactive_imageLabel('x', personal_zone=_ZONA)
    label.toggle_personal_zone(True)
    label.resize(1000, 1000)
    assert label.hay_punto_en(QPoint(100, 100))


def test_set_personal_zone_no_emite():
    """Restaurar lo guardado no puede rebotar al cliente lo que acaba de
    cargar: seria una escritura por arranque, y con la zona a medio sembrar
    machacaria la que el operador tenia dibujada."""
    label = Interactive_imageLabel('x')
    emitidas = []
    label.personal_change.connect(lambda pts, act: emitidas.append(act))
    label.set_personal_zone(_ZONA, activa=True)
    assert emitidas == []
    assert label.personal_zone_active is True
