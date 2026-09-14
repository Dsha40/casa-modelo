"""
Una configuracion de Jarvis incompleta debe VERSE, no fallar en silencio.

`amazonas` y `managers` arrancaban sin `jarvis_url/email/password` en su .env:
`Jarvis_api(None, None, None)` fallaba el login con un mensaje perdido entre
los print del arranque, y el interruptor «Enviar a Jarvis» seguia verde y
marcado. Parecia funcionar y no enviaba nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QCheckBox  # noqa: E402

from elde_core.alertas import jarvis_credenciales as cred  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def _qt():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def entorno(monkeypatch):
    """Entorno limpio: el .env real del equipo no debe decidir el resultado."""
    for v in cred.VARIABLES:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _completo(entorno):
    entorno.setenv('jarvis_url', 'https://ejemplo/api_jarvis/v1')
    entorno.setenv('jarvis_email', 'x@y.z')
    entorno.setenv('jarvis_password', 'secreto')


def test_con_las_tres_variables_se_puede_enviar(entorno):
    _completo(entorno)
    casilla = QCheckBox('Enviar a Jarvis')
    casilla.setChecked(True)
    assert cred.comprobar(casilla) is True
    assert casilla.isChecked() and casilla.isEnabled()


def test_sin_variables_la_casilla_se_apaga_y_se_bloquea(entorno):
    casilla = QCheckBox('Enviar a Jarvis')
    casilla.setChecked(True)
    assert cred.comprobar(casilla, cliente='amazonas') is False
    assert not casilla.isChecked(), 'no puede quedarse marcada fingiendo que envia'
    assert not casilla.isEnabled()


def test_el_tooltip_dice_QUE_falta_y_DONDE(entorno):
    entorno.setenv('jarvis_url', 'https://ejemplo/api_jarvis/v1')
    casilla = QCheckBox('Enviar a Jarvis')
    cred.comprobar(casilla, cliente='detecciones')
    ayuda = casilla.toolTip()
    assert 'jarvis_email' in ayuda and 'jarvis_password' in ayuda
    assert 'jarvis_url' not in ayuda, 'esa si estaba: no debe acusarla'
    assert 'clients/detecciones/.env' in ayuda


def test_una_variable_vacia_cuenta_como_ausente(entorno):
    _completo(entorno)
    entorno.setenv('jarvis_password', '   ')
    assert cred.faltantes() == ['jarvis_password']


def test_funciona_sin_interfaz(entorno):
    """Se usa tambien desde scripts y pruebas, sin QCheckBox."""
    assert cred.comprobar(None) is False


# Son las CARPETAS, no los `client_type`: 'managers' se renombró a
# 'detecciones' el 6-ago-2026 y su identificador de contrato no cambió.
@pytest.mark.parametrize('cliente',
                         ('perimetrales', 'tienda', 'amazonas', 'detecciones'))
def test_los_cuatro_clientes_comprueban_al_arrancar(cliente):
    principal = (RAIZ.parents[1] / 'clients' / cliente / 'src'
                 / 'main.py').read_text(encoding='utf-8')
    assert 'comprobar_jarvis(' in principal, (
        f'{cliente}: no comprueba las credenciales al arrancar')
    assert 'jarvis_forwarder.set_activo(False)' in principal, (
        f'{cliente}: sin credenciales el reenviador se queda encendido')
