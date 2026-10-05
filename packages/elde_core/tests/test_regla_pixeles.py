"""
Regla manual para medir en píxeles (`elde_core.ui.regla_pixeles`, 2-oct-2026).

Lo que fijan estas pruebas es lo que haría mentir a la regla: medir en
píxeles de la PANTALLA en vez de los de la imagen que recibe el servidor, dar
una cifra sin saber todavía el tamaño de la imagen, o que con la regla
encendida un arrastre siga moviendo los puntos de las zonas.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.ui.regla_pixeles import ReglaPixeles  # noqa: E402


def _regla(ancho=1456, alto=868):
    r = ReglaPixeles()
    r.tamano_frame = lambda: (ancho, alto)
    return r


def test_mide_en_pixeles_de_la_imagen_no_de_la_pantalla():
    """De 0..1000 del visor a la imagen: x*ancho/1000, y*alto/1000."""
    r = _regla(1456, 868)
    r.empezar(500, 200)
    r.mover(500, 700)                 # vertical: media pantalla de alto
    m = r.medida()
    assert round(m['alto']) == 434    # 500/1000 * 868
    assert m['ancho'] == 0
    assert round(m['px']) == 434


def test_da_alto_y_ancho_por_separado():
    r = _regla(1000, 1000)
    r.empezar(100, 100)
    r.mover(400, 500)
    m = r.medida()
    assert (round(m['ancho']), round(m['alto']), round(m['px'])) == (300, 400, 500)


def test_sin_tamano_de_imagen_no_inventa_una_cifra():
    r = ReglaPixeles()                # tamano_frame por defecto: (0, 0)
    r.empezar(0, 0)
    r.mover(1000, 1000)
    assert r.medida() is None
    assert r.texto() == 'sin imagen todavía'


def test_borrar_quita_la_linea():
    r = _regla()
    r.empezar(1, 1)
    r.mover(2, 2)
    r.terminar()
    r.borrar()
    assert r.medida() is None and not r.arrastrando


def test_con_la_regla_encendida_arrastrar_mide_y_no_mueve_zonas():
    """En el lienzo REAL, con eventos de ratón de Qt. No se enseña ninguna
    ventana: los eventos se entregan directamente al lienzo. NO se fuerza
    `QT_QPA_PLATFORM=offscreen`: la QApplication es una por proceso, y la
    prueba del escáner (`test_escaner_propias_ventanas.py`) necesita ventanas
    REALES de Windows."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from elde_core.ui.interactive_image_label import Interactive_imageLabel

    app = QApplication.instance() or QApplication([])
    roi = [[100, 100], [900, 100], [900, 900], [100, 900]]   # como en la config
    lab = Interactive_imageLabel(roi=roi, roi_active=True)
    lab.resize(1000, 600)
    lab.regla.tamano_frame = lambda: (1456, 868)
    lab.regla.activa = True
    antes = [(p.x(), p.y()) for p in lab.points]

    def ev(tipo, x, y, botones):
        return QMouseEvent(tipo, QPointF(x, y), QPointF(x, y), Qt.LeftButton,
                           botones, Qt.NoModifier)

    # Empieza JUSTO sobre un vértice del ROI (100,100 -> 100,60 en el visor).
    lab.mousePressEvent(ev(QMouseEvent.MouseButtonPress, 100, 60, Qt.LeftButton))
    lab.mouseMoveEvent(ev(QMouseEvent.MouseMove, 100, 360, Qt.LeftButton))
    lab.mouseReleaseEvent(ev(QMouseEvent.MouseButtonRelease, 100, 360, Qt.NoButton))
    app.processEvents()

    assert [(p.x(), p.y()) for p in lab.points] == antes, 'la regla movio el ROI'
    m = lab.regla.medida()
    assert m is not None and round(m['alto']) == 434   # 300 px de 600 = 500/1000 de 868
    assert not lab.regla.arrastrando
