"""
Selector de establecimiento POR CÁMARA (5-ago-2026).

Una ventana puede tener 4 camaras de 4 locales distintos corriendo la MISMA
inferencia: el selector del pie es uno solo y no puede resolver eso. Cada
recuadro necesita el suyo, y ese local tiene que viajar en el payload de cada
frame (para que el servidor segmente) y en cada tarjeta de alerta (para que la
novedad de Jarvis llegue al local correcto).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402
from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from elde_core.ui.selector_local_camara import (  # noqa: E402
    SIN_LOCAL_NO_ENVIA, SIN_LOCAL_USA_EL_PIE, SelectorLocalCamara)

# Son las CARPETAS, no los `client_type`: 'managers' se renombró a
# 'detecciones' el 6-ago-2026 y su identificador de contrato no cambió.
CLIENTES = ('perimetrales', 'tienda', 'amazonas', 'detecciones')


@pytest.fixture(scope='module', autouse=True)
def _qt():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class _JarvisFalso(QObject):
    establishments_loaded = Signal(list)

    def __init__(self, locales=None):
        super().__init__()
        self.list_of_establishments = locales or []


LOCALES = [{'name': 'La Comarca'}, {'name': 'Bodega Central'},
           {'name': 'Sucursal Norte'}]


def test_arranca_sin_local_y_con_la_lista_vacia():
    sel = SelectorLocalCamara(api_jarvis=_JarvisFalso())
    assert sel.count() == 1 and sel.establecimiento == ''


def test_se_rellena_cuando_jarvis_carga_la_lista():
    api = _JarvisFalso()
    sel = SelectorLocalCamara(api_jarvis=api)
    assert sel.count() == 1, 'al construir, la lista aun no ha llegado'
    api.list_of_establishments = LOCALES
    api.establishments_loaded.emit(LOCALES)
    assert sel.count() == 4


def test_cada_camara_lleva_su_propio_local():
    """El caso que pidio el operador: 4 camaras, 4 locales, misma inferencia."""
    api = _JarvisFalso(LOCALES)
    camaras = [SelectorLocalCamara(api_jarvis=api, indice_camara=i)
               for i in range(3)]
    for sel, nombre in zip(camaras, ('La Comarca', 'Bodega Central',
                                     'Sucursal Norte')):
        sel.fijar(nombre)
    assert [s.establecimiento for s in camaras] == [
        'La Comarca', 'Bodega Central', 'Sucursal Norte']


def test_elegir_avisa_al_recuadro():
    api = _JarvisFalso(LOCALES)
    sel = SelectorLocalCamara(api_jarvis=api)
    avisos = []
    sel.local_cambiado.connect(avisos.append)
    sel.setCurrentIndex(sel.findData('Bodega Central'))
    sel.activated.emit(sel.currentIndex())      # como un clic del operador
    assert avisos == ['Bodega Central']
    assert sel.establecimiento == 'Bodega Central'


def test_repoblar_no_dispara_un_cambio_falso():
    """`activated` y no `currentIndexChanged`: repoblar mueve el indice y no
    debe parecer que el operador eligio algo."""
    api = _JarvisFalso()
    sel = SelectorLocalCamara(api_jarvis=api)
    sel.fijar('La Comarca')
    avisos = []
    sel.local_cambiado.connect(avisos.append)
    api.list_of_establishments = LOCALES
    api.establishments_loaded.emit(LOCALES)
    assert avisos == []
    assert sel.establecimiento == 'La Comarca'


def test_un_local_que_ya_no_esta_en_la_lista_se_conserva():
    """Quitarlo en silencio mandaria sus alertas a otro sitio sin avisar."""
    api = _JarvisFalso(LOCALES)
    sel = SelectorLocalCamara(api_jarvis=api, establecimiento='Local Cerrado')
    assert sel.establecimiento == 'Local Cerrado'
    textos = [sel.itemText(i) for i in range(sel.count())]
    assert any('no está en la lista' in t for t in textos), textos


def test_volver_a_la_primera_opcion_borra_el_local():
    api = _JarvisFalso(LOCALES)
    sel = SelectorLocalCamara(api_jarvis=api, establecimiento='La Comarca')
    sel.setCurrentIndex(0)
    sel.activated.emit(0)
    assert sel.establecimiento == ''


def test_el_texto_de_vacio_dice_la_verdad_de_cada_cliente():
    """perimetrales no tiene selector en el pie: decir «el del pie» seria
    mentir, porque sin local NO se envia nada."""
    api = _JarvisFalso()
    peri = SelectorLocalCamara(api_jarvis=api, texto_vacio=SIN_LOCAL_NO_ENVIA)
    otro = SelectorLocalCamara(api_jarvis=api, texto_vacio=SIN_LOCAL_USA_EL_PIE)
    assert 'NO se envían' in peri.toolTip()
    assert 'pie de la ventana' in otro.toolTip()


def test_sin_api_no_revienta():
    """El recuadro se construye aunque Jarvis no este disponible."""
    sel = SelectorLocalCamara(api_jarvis=None)
    assert sel.count() == 1


# ── Cableado en los cuatro clientes ──────────────────────────────────

@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_recuadro_tiene_su_selector(cliente):
    caja = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
            / 'components' / 'render_box' / 'render_box.py').read_text(
                encoding='utf-8')
    assert 'SelectorLocalCamara(' in caja, (
        f'{cliente}: el recuadro no tiene selector de local')
    assert 'bar_opt_layout.addWidget(self.selector_local)' in caja, (
        f'{cliente}: el selector no esta en la barra (no se ve)')
    assert 'local_cambiado.connect(' in caja, (
        f'{cliente}: elegir un local no avisa al recuadro')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_local_viaja_en_el_frame_y_en_la_alerta(cliente):
    caja = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
            / 'components' / 'render_box' / 'render_box.py').read_text(
                encoding='utf-8')
    # Dos sitios: el payload del frame y la tarjeta de alerta.
    assert caja.count('"establecimiento": self.establecimiento') >= 2, (
        f'{cliente}: el local debe viajar en el frame Y en la alerta '
        f'(encontrado {caja.count(chr(34) + "establecimiento" + chr(34) + ": self.establecimiento")} veces)')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_local_no_se_recuerda_entre_sesiones(cliente):
    """Peticion del operador: cada sesion se elige de cero, para que nadie
    se quede alertando al establecimiento de ayer."""
    ventana = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
               / 'windows_main.py').read_text(encoding='utf-8')
    assert 'establecimiento' in ventana, f'{cliente}: no se pasa el local'
    assert 'cfg.get("establecimiento"' not in ventana, (
        f'{cliente}: el local se esta restaurando de la config')


def test_el_servidor_acepta_el_local_por_frame():
    app = (RAIZ.parents[1] / 'server' / 'src' / 'app' / 'app.py').read_text(
        encoding='utf-8')
    assert 'data.get("establecimiento")' in app
