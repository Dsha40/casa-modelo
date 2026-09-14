"""
Pruebas de la flecha de direccion (`elde_core.ui.flecha_direccion`).

Lo que mas se protege:

  - **una flecha corta NO define una direccion** — dos puntos casi pegados dan
    un vector de ruido, y con el se contarian entradas al azar. Devuelve
    `None`, que es una respuesta que hay que tratar, no un cero disfrazado;
  - **la cola es la salida y la punta la entrada**, sin excepciones: es todo
    el contrato entre el operador y el servidor;
  - **el label compartido no cambia para los otros tres clientes** — la flecha
    nace apagada y viaja en su propia senal.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from elde_core.ui.flecha_direccion import (  # noqa: E402
    INDICE_MEDIO, LARGO_MINIMO, FlechaDireccion)
from elde_core.ui.interactive_image_label import (  # noqa: E402
    Interactive_imageLabel)

_app = QApplication.instance() or QApplication([])


def _horizontal(activa=True):
    """Cola en (200,500), punta en (800,500): apunta a la derecha."""
    return FlechaDireccion([[200, 500], [800, 500]], activa=activa)


# ── Contrato cola/punta ─────────────────────────────────────────────────
def test_la_punta_es_la_entrada_y_la_cola_la_salida():
    f = _horizontal()
    assert (f.cola.x(), f.cola.y()) == (200, 500)
    assert (f.punta.x(), f.punta.y()) == (800, 500)


def test_el_vector_apunta_hacia_adentro():
    dx, dy = _horizontal().vector_unitario()
    assert abs(dx - 1.0) < 1e-6 and abs(dy) < 1e-6


def test_el_vector_esta_normalizado():
    f = FlechaDireccion([[0, 0], [300, 400]], activa=True)   # 3-4-5
    dx, dy = f.vector_unitario()
    assert abs((dx ** 2 + dy ** 2) ** 0.5 - 1.0) < 1e-6
    assert abs(dx - 0.6) < 1e-6 and abs(dy - 0.8) < 1e-6


# ── Cuando NO hay direccion ─────────────────────────────────────────────
def test_una_flecha_apagada_no_da_vector():
    assert _horizontal(activa=False).vector_unitario() is None


def test_una_flecha_demasiado_corta_no_da_vector():
    """Ruido, no direccion: dos puntos casi pegados apuntan a cualquier lado."""
    corta = FlechaDireccion([[500, 500], [500 + int(LARGO_MINIMO) - 5, 500]],
                            activa=True)
    assert corta.vector_unitario() is None
    assert not corta.es_utilizable()


def test_una_flecha_de_longitud_cero_no_revienta():
    f = FlechaDireccion([[500, 500], [500, 500]], activa=True)
    assert f.vector_unitario() is None
    assert f.largo() == 0.0


def test_justo_en_el_limite_ya_vale():
    f = FlechaDireccion([[0, 0], [int(LARGO_MINIMO), 0]], activa=True)
    assert f.es_utilizable() and f.vector_unitario() is not None


# ── Serializacion ───────────────────────────────────────────────────────
def test_ida_y_vuelta_por_los_ajustes():
    f = _horizontal()
    otra = FlechaDireccion(f.a_lista(), activa=True)
    assert otra.a_lista() == f.a_lista()
    assert otra.vector_unitario() == f.vector_unitario()


def test_unos_ajustes_corruptos_caen_a_la_posicion_por_defecto():
    """Nace agarrable: una flecha de largo cero no se puede coger."""
    for basura in (None, [], [[1, 2]], 'x', [['a', 'b'], ['c', 'd']], [{}, {}]):
        f = FlechaDireccion(basura, activa=True)
        assert f.largo() >= LARGO_MINIMO, f'con {basura!r} quedo inservible'


def test_las_coordenadas_van_al_tamano_real_del_frame():
    f = _horizontal()
    assert f.coordenadas_pixel(1920, 1080) == [[384, 540], [1536, 540]]
    assert f.coordenadas_pixel(0, 0) == [], 'sin frame no hay coordenadas'


# ── Raton ───────────────────────────────────────────────────────────────
def _escala_identidad(p):
    return p


def test_se_coge_el_extremo_que_esta_bajo_el_cursor():
    f = _horizontal()
    assert f.indice_en(QPoint(202, 501), _escala_identidad, 10) == 0
    assert f.indice_en(QPoint(798, 499), _escala_identidad, 10) == 1
    assert f.indice_en(QPoint(500, 300), _escala_identidad, 10) == -1


def test_el_cuerpo_y_el_centro_cogen_la_flecha_entera():
    """Sobre el cuerpo (lejos de los extremos) el agarre es INDICE_MEDIO."""
    f = _horizontal()
    assert f.indice_en(QPoint(500, 500), _escala_identidad, 10) == INDICE_MEDIO
    assert f.indice_en(QPoint(400, 505), _escala_identidad, 10) == INDICE_MEDIO
    # Junto a un extremo gana el extremo: alli se quiere reorientar, no mover.
    assert f.indice_en(QPoint(205, 500), _escala_identidad, 10) == 0


def test_apagada_no_se_puede_coger():
    assert _horizontal(activa=False).indice_en(
        QPoint(200, 500), _escala_identidad, 10) == -1


def test_mover_un_extremo_cambia_el_vector():
    f = _horizontal()
    f.mover(1, QPoint(200, 900))          # la punta pasa a apuntar hacia abajo
    dx, dy = f.vector_unitario()
    assert abs(dx) < 1e-6 and dy > 0.99


def test_mover_un_indice_inexistente_no_revienta():
    f = _horizontal()
    antes = f.a_lista()
    f.mover(7, QPoint(0, 0))
    f.mover(-3, QPoint(0, 0))
    assert f.a_lista() == antes


# ── Invertir y trasladar ────────────────────────────────────────────────
def test_invertir_intercambia_entrada_y_salida():
    """El arreglo cuando las entradas se cuentan como salidas."""
    f = _horizontal()
    dx_antes, dy_antes = f.vector_unitario()
    f.invertir()
    assert (f.cola.x(), f.cola.y()) == (800, 500)
    assert (f.punta.x(), f.punta.y()) == (200, 500)
    dx, dy = f.vector_unitario()
    assert abs(dx + dx_antes) < 1e-6 and abs(dy + dy_antes) < 1e-6


def test_invertir_dos_veces_la_deja_como_estaba():
    f = _horizontal()
    antes = f.a_lista()
    f.invertir()
    f.invertir()
    assert f.a_lista() == antes


def test_trasladar_mueve_los_dos_extremos_sin_girar():
    f = _horizontal()
    vector = f.vector_unitario()
    largo = f.largo()
    f.trasladar(50, -100)
    assert f.a_lista() == [[250, 400], [850, 400]]
    assert f.vector_unitario() == vector
    assert f.largo() == largo


def test_trasladar_se_frena_en_el_borde_sin_deformarse():
    """Se recorta el paso, no cada punto: contra el borde no debe girar."""
    f = _horizontal()
    largo = f.largo()
    f.trasladar(9999, 9999)
    assert f.a_lista() == [[400, 1000], [1000, 1000]]
    assert f.largo() == largo
    f.trasladar(-9999, -9999)
    assert f.a_lista() == [[0, 0], [600, 0]]
    assert f.largo() == largo


# ── Pintado ─────────────────────────────────────────────────────────────
def test_se_pinta_sin_reventar():
    """Un widget que compila pero peta al pintarse no sirve de nada."""
    lienzo = QPixmap(400, 300)
    painter = QPainter(lienzo)
    try:
        for flecha in (_horizontal(),
                       FlechaDireccion([[10, 10], [12, 12]], activa=True),
                       _horizontal(activa=False)):
            for indice in (-1, 0, INDICE_MEDIO):
                flecha.pintar(painter,
                              lambda p: QPoint(p.x() // 4, p.y() // 4),
                              10, indice)
    finally:
        painter.end()


# ── El label compartido ─────────────────────────────────────────────────
def test_el_label_nace_SIN_flecha():
    """Los otros tres clientes no deben notar nada."""
    label = Interactive_imageLabel('x', roi=[[0, 0], [100, 0], [100, 100]],
                                   roi_active=True)
    assert label.flecha.activa is False
    assert label.get_flecha_coordinates(1920, 1080) == []


def test_la_flecha_viaja_en_su_propia_senal():
    """`point_change` tiene 6 argumentos y la usan los cuatro clientes."""
    label = Interactive_imageLabel('x')
    recibido = []
    label.flecha_change.connect(lambda pts, act: recibido.append((pts, act)))
    label.toggle_flecha(True)
    label.flecha_index = 1
    label.flecha.mover(1, QPoint(900, 500))
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    label.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, QPointF(0, 0), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    assert len(recibido) == 1 and recibido[0][1] is True


def test_encender_la_flecha_la_hace_interactiva():
    label = Interactive_imageLabel('x')
    assert label._interactive() is False
    label.toggle_flecha(True)
    assert label._interactive() is True


def test_el_label_no_manda_una_flecha_inservible():
    label = Interactive_imageLabel('x')
    label.toggle_flecha(True)
    label.set_flecha([[500, 500], [505, 500]])       # demasiado corta
    assert label.get_flecha_coordinates(1920, 1080) == []
    label.set_flecha([[200, 500], [800, 500]])
    assert len(label.get_flecha_coordinates(1920, 1080)) == 2


def test_la_flecha_cede_el_arrastre_a_las_barras():
    """`hay_punto_en` es lo que consultan las barras superpuestas al visor."""
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])
    assert label.hay_punto_en(QPoint(200, 500)) is True
    # El cuerpo tambien es agarrable ahora: mueve la flecha entera.
    assert label.hay_punto_en(QPoint(500, 500)) is True
    assert label.hay_punto_en(QPoint(500, 100)) is False


def _raton(tipo, pos):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent
    return QMouseEvent(tipo, QPointF(pos.x(), pos.y()), Qt.LeftButton,
                       Qt.LeftButton, Qt.NoModifier)


def _doble_clic(pos):
    from PySide6.QtCore import QEvent
    return _raton(QEvent.MouseButtonDblClick, pos)


def test_doble_clic_sobre_la_flecha_la_invierte_y_avisa():
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])
    recibido = []
    label.flecha_change.connect(lambda pts, act: recibido.append((pts, act)))
    label.mouseDoubleClickEvent(_doble_clic(QPoint(500, 500)))
    assert label.flecha.a_lista() == [[800, 500], [200, 500]]
    assert len(recibido) == 1 and recibido[0][0] == [[800, 500], [200, 500]]


def test_doble_clic_fuera_de_la_flecha_no_hace_nada():
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])
    antes = label.flecha.a_lista()
    label.mouseDoubleClickEvent(_doble_clic(QPoint(500, 100)))
    assert label.flecha.a_lista() == antes


def test_invertir_desde_el_menu_emite_para_guardar():
    """`invertir_flecha()` es lo que llama la entrada del menu del cliente."""
    label = Interactive_imageLabel('x')
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])
    recibido = []
    label.flecha_change.connect(lambda pts, act: recibido.append((pts, act)))
    label.invertir_flecha()
    assert label.flecha.a_lista() == [[800, 500], [200, 500]]
    assert len(recibido) == 1 and recibido[0][1] is True


def test_invertir_apagada_no_hace_nada():
    label = Interactive_imageLabel('x')
    label.set_flecha([[200, 500], [800, 500]])
    recibido = []
    label.flecha_change.connect(lambda pts, act: recibido.append((pts, act)))
    label.invertir_flecha()
    assert label.flecha.a_lista() == [[200, 500], [800, 500]]
    assert recibido == []


def test_arrastrar_el_cuerpo_mueve_la_flecha_entera():
    """Prensa en el cuerpo, arrastre y suelte: traslada sin girar."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])
    vector = label.flecha.vector_unitario()

    label.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, QPointF(400, 500), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    assert label.flecha_index == INDICE_MEDIO
    label.mouseMoveEvent(QMouseEvent(
        QEvent.MouseMove, QPointF(450, 450), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    label.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, QPointF(450, 450), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

    assert label.flecha.a_lista() == [[250, 450], [850, 450]]
    assert label.flecha.vector_unitario() == vector
    assert label.flecha_index == -1


def test_doble_clic_sobre_un_vertice_de_roi_NO_invierte():
    """El cuerpo cede ante los vertices tambien en el doble clic.

    Dos ajustes rapidos de un vertice llegan como MouseButtonDblClick; si
    la guardia solo mirara la flecha, ese gesto la invertiria en silencio.
    """
    from PySide6.QtCore import QEvent
    label = Interactive_imageLabel(
        'x', roi=[[500, 500], [700, 300], [300, 300]], roi_active=True)
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])   # el cuerpo pasa por (500,500)
    antes = label.flecha.a_lista()

    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(500, 500)))
    assert label.active_point_index == 0         # agarro el vertice, no el cuerpo
    assert label.flecha_index == -1
    label.mouseReleaseEvent(_raton(QEvent.MouseButtonRelease, QPoint(500, 500)))
    label.mouseDoubleClickEvent(_doble_clic(QPoint(500, 500)))
    assert label.flecha.a_lista() == antes, 'el doble clic invirtio la flecha'


def test_el_press_del_doble_clic_no_descalibra_al_invertir():
    """El jitter entre el primer clic y el doble no puede mover la flecha.

    Secuencia Qt real: press → move (jitter) → release → dblclick. La
    inversion debe partir de los puntos de ANTES del press, no de los
    arrastrados sin querer.
    """
    from PySide6.QtCore import QEvent
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])

    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(798, 499)))
    label.mouseMoveEvent(_raton(QEvent.MouseMove, QPoint(795, 502)))
    label.mouseReleaseEvent(_raton(QEvent.MouseButtonRelease, QPoint(795, 502)))
    label.mouseDoubleClickEvent(_doble_clic(QPoint(795, 502)))

    assert label.flecha.a_lista() == [[800, 500], [200, 500]]


def test_arrastrar_un_extremo_no_teletransporta_al_cursor():
    """Cogido por el borde del circulo, el extremo sigue al raton por DELTA.

    Antes `mover()` ponia el punto EN el cursor: coger a 7 px del centro ya
    lo desplazaba 7 px, y ese salto ademas se guardaba.
    """
    from PySide6.QtCore import QEvent
    label = Interactive_imageLabel('x')
    label.resize(1000, 1000)
    label.toggle_flecha(True)
    label.set_flecha([[200, 500], [800, 500]])

    label.mousePressEvent(_raton(QEvent.MouseButtonPress, QPoint(795, 505)))
    assert label.flecha_index == 1
    label.mouseMoveEvent(_raton(QEvent.MouseMove, QPoint(700, 400)))
    label.mouseReleaseEvent(_raton(QEvent.MouseButtonRelease, QPoint(700, 400)))

    assert label.flecha.a_lista()[1] == [705, 395]


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
