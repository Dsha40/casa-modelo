"""
Nada puede robarle el clic al desplegable de local (5-ago-2026).

## El fallo, que tuvo DOS causas distintas

El operador lo describio igual las dos veces —«da un error y no deja
seleccionar, como si se recargara»— pero por debajo eran cosas diferentes, y
arreglar solo la primera dejo el problema en pie:

1. **La barra flotante se comia el clic.** `_BarraFlotante.vigilar_hijos()`
   instala en TODOS sus hijos un filtro que, si bajo el cursor hay un punto
   del ROI, manda el evento al visor y **consume el clic**. Para un boton es
   correcto (su accion cabe en un clic); para un desplegable significa que el
   popup no llega a abrirse. Y como dependia de donde estuviera el ROI,
   parecia intermitente. Afectaba a `perimetrales` y `amazonas`.

2. **`amazonas` escondia la barra al abrirse el popup.** Su `leaveEvent` no
   tenia la guarda de `activePopupWidget` que si tenian los otros tres: el
   popup es una ventana aparte, asi que al abrirse el raton "sale" del
   recuadro, la barra se ocultaba con el desplegable dentro y el popup se
   cerraba solo.

Hubo una tercera, ya corregida y con prueba propia en
`test_sin_selector_global.py`: reconstruir la lista dentro del manejador de
`activated`.

## Por que estas pruebas leen el codigo fuente

El sintoma solo se ve con una pantalla de verdad. Lo que si se puede fijar sin
pantalla es que las dos guardas sigan ahi, que es lo que se rompio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
# Son las CARPETAS, no los `client_type`: 'managers' se renombró a
# 'detecciones' el 6-ago-2026 y su identificador de contrato no cambió.
CLIENTES = ('perimetrales', 'tienda', 'amazonas', 'detecciones')


def _caja(cliente: str) -> str:
    return (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
            / 'components' / 'render_box' / 'render_box.py').read_text(
                encoding='utf-8')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_al_abrirse_un_popup_no_se_esconde_la_barra(cliente):
    """Si la barra se esconde, se lleva el desplegable dentro."""
    src = _caja(cliente)
    inicio = src.index('def leaveEvent')
    cuerpo = src[inicio:inicio + 700]
    assert 'activePopupWidget' in cuerpo, (
        f'{cliente}: leaveEvent esconde la barra aunque haya un popup abierto; '
        'el desplegable se cierra solo al abrirlo')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_la_barra_no_le_roba_el_clic_a_un_desplegable(cliente):
    """Solo aplica a los clientes cuya barra roba clics para el ROI."""
    src = _caja(cliente)
    if 'def vigilar_hijos' not in src:
        pytest.skip(f'{cliente}: su barra no intercepta clics')
    assert '_maneja_su_clic' in src, (
        f'{cliente}: vigilar_hijos vuelve a instalar el filtro en los '
        'desplegables; el popup no podra abrirse')
    inicio = src.index('def eventFilter')
    cuerpo = src[inicio:inicio + 1400]
    assert 'activePopupWidget' in cuerpo, (
        f'{cliente}: el filtro roba clics con un popup abierto')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_selector_sigue_en_la_barra(cliente):
    """La guarda no sirve de nada si el selector desaparece."""
    src = _caja(cliente)
    assert 'bar_opt_layout.addWidget(self.selector_local)' in src, (
        f'{cliente}: el selector de local ya no esta en la barra')


def test_el_boton_de_video_no_es_un_play():
    """`amazonas` usaba el MISMO icono que el boton de analisis en vivo, a dos
    huecos de distancia: se confundian."""
    src = _caja('amazonas')
    inicio = src.index('self.btn_video')
    cuerpo = src[inicio:inicio + 300]
    assert 'play_box.png' not in cuerpo, (
        'el boton de analizar un video volvio a ser un triangulo de reproducir')
    assert 'video_file.png' in cuerpo
    icono = (RAIZ.parents[1] / 'clients' / 'amazonas' / 'resource'
             / 'video_file.png')
    assert icono.is_file(), 'falta el icono; el boton saldria vacio'
