"""
El establecimiento se elige SOLO en la barra de cada cámara (5-ago-2026).

## Las dos cosas que se fijan aqui

**1. En el pie no hay selector de establecimiento.** Lo pidio el operador: con
un local por camara, un desplegable global en el pie solo puede contradecirlo.
`perimetrales` ya lo habia quitado el 1-ago; los otros tres lo arrastraban.

**2. Elegir no reconstruye la lista.** Este era el fallo de verdad: al elegir,
`_al_elegir` llamaba a `fijar()`, que llamaba a `repoblar()`, que hace
`clear()`. Y todo eso DENTRO del manejador de `activated`, que Qt emite
mientras el popup se esta cerrando: se destruian las filas que la vista aun
usaba. Sintoma del operador: «da un error y no deja seleccionar, como si se
recargara». Repoblar ahi ademas no hacia falta — el combo ya muestra lo
elegido.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402
from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from elde_core.ui.selector_local_camara import SelectorLocalCamara  # noqa: E402

# Son las CARPETAS, no los `client_type`: 'managers' se renombró a
# 'detecciones' el 6-ago-2026 y su identificador de contrato no cambió.
CLIENTES = ('perimetrales', 'tienda', 'amazonas', 'detecciones')
LOCALES = [{'name': 'La Comarca'}, {'name': 'Bodega Central'}]


@pytest.fixture(scope='module', autouse=True)
def _qt():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class _Jarvis(QObject):
    establishments_loaded = Signal(list)

    def __init__(self, locales=None):
        super().__init__()
        self.list_of_establishments = locales or []


# ── 1. El pie no tiene selector ──────────────────────────────────────

@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_pie_no_tiene_selector_de_establecimiento(cliente):
    pie = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui' / 'components'
           / 'custom_status_bar.py').read_text(encoding='utf-8')
    assert 'self.selector_establishment' not in pie, (
        f'{cliente}: el pie volvio a tener un selector global')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_la_ventana_no_cablea_ningun_selector_global(cliente):
    ventana = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
               / 'windows_main.py').read_text(encoding='utf-8')
    assert 'selector_establishment' not in ventana, (
        f'{cliente}: la ventana sigue cableando el selector del pie')
    assert 'clicked_selection_establishment' not in ventana, (
        f'{cliente}: quedo el manejador del selector global')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_local_solo_se_elige_en_la_camara(cliente):
    caja = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
            / 'components' / 'render_box' / 'render_box.py').read_text(
                encoding='utf-8')
    assert 'SelectorLocalCamara(' in caja, (
        f'{cliente}: la camara se quedo sin selector y ya no hay global')


# ── 2. Elegir no reconstruye la lista ────────────────────────────────

def test_elegir_no_reconstruye_la_lista():
    """El fallo real: `clear()` dentro del manejador de `activated`."""
    sel = SelectorLocalCamara(api_jarvis=_Jarvis(LOCALES))
    reconstrucciones = []
    original = sel.repoblar
    sel.repoblar = lambda: (reconstrucciones.append(1), original())[1]

    sel.setCurrentIndex(sel.findData('Bodega Central'))
    sel.activated.emit(sel.currentIndex())

    assert reconstrucciones == [], (
        'elegir reconstruyo la lista: el popup se destruye bajo el dedo '
        'del operador')
    assert sel.establecimiento == 'Bodega Central'


def test_elegir_deja_la_lista_intacta():
    sel = SelectorLocalCamara(api_jarvis=_Jarvis(LOCALES))
    antes = [sel.itemText(i) for i in range(sel.count())]
    sel.setCurrentIndex(sel.findData('La Comarca'))
    sel.activated.emit(sel.currentIndex())
    assert [sel.itemText(i) for i in range(sel.count())] == antes


def test_elegir_varias_veces_seguidas_no_se_atasca():
    """Sintoma reportado: a la segunda ya no dejaba elegir."""
    sel = SelectorLocalCamara(api_jarvis=_Jarvis(LOCALES))
    elegidos = []
    sel.local_cambiado.connect(elegidos.append)
    for nombre in ('La Comarca', 'Bodega Central', 'La Comarca', ''):
        sel.setCurrentIndex(sel.findData(nombre))
        sel.activated.emit(sel.currentIndex())
    assert elegidos == ['La Comarca', 'Bodega Central', 'La Comarca', '']
    assert sel.establecimiento == ''


def test_fijar_desde_fuera_si_sincroniza_la_vista():
    """`fijar` no viene del combo, asi que ahi si hay que poner la vista al
    dia — pero moviendo el indice, no reconstruyendo, si el nombre existe."""
    sel = SelectorLocalCamara(api_jarvis=_Jarvis(LOCALES))
    sel.fijar('Bodega Central')
    assert sel.currentData() == 'Bodega Central'
    assert sel.count() == 3, 'no debio anadir opciones: el nombre ya estaba'


def test_fijar_un_nombre_ausente_si_reconstruye():
    """Unico caso en que hay que reconstruir: crearle su opcion."""
    sel = SelectorLocalCamara(api_jarvis=_Jarvis(LOCALES))
    sel.fijar('Local Cerrado')
    assert sel.establecimiento == 'Local Cerrado'
    assert any('no está en la lista' in sel.itemText(i)
               for i in range(sel.count()))
