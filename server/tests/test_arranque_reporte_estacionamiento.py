"""
Un modulo ausente NO es un error del servidor (22-sep-2026).

`_arrancar_reporte_estacionamiento` importa `estacionamiento_planificador`,
que vive en el monorepo y NO viaja en el corte de tiendas. Hasta hoy esa
ausencia caia en un `except Exception` con `logger.exception`, asi que CADA
arranque del servidor escupia un ERROR en rojo con traceback completo:

    ERROR | no se pudo programar el reporte del estacionamiento
    Traceback (most recent call last):
      ...
    ModuleNotFoundError: No module named '...estacionamiento_planificador'

Es exactamente la misma situacion que `vigilante_amazonas` y
`perimetral_nocturna`, que salen como un WARNING de una linea. El dano no es
el ruido: es que quien mira el log aprende a ignorar los ERROR del arranque,
y el dia que haya uno de verdad pasara desapercibido.

Lo que estas pruebas fijan es la DISTINCION, no el silencio:

- el modulo no esta en este corte      -> WARNING de una linea, sin traza
- el modulo esta y revienta al arrancar -> ERROR con traza (el reporte diario
  del estacionamiento se perdio de verdad y hay que enterarse)

El segundo caso no se puede reproducir con el repositorio tal cual —aqui el
modulo simplemente no existe—, asi que se inyecta uno falso en `sys.modules`,
que es lo mas cerca que se puede estar del monorepo sin tenerlo delante.
"""
from __future__ import annotations

import importlib.machinery
import logging
import sys
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
# `app.py` arrastra `baliza_descubrimiento`, que importa del nucleo
# compartido. No esta instalado en el entorno de pruebas: se anade a la ruta
# aqui, como ya hacen test_api_lectura.py y test_fallo_imagen_IA_pipeline.py,
# para que este fichero tambien corra SUELTO y no solo dentro de la suite
# entera (donde otro test lo habia puesto antes por orden alfabetico).
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

from src.app.app import _arrancar_reporte_estacionamiento  # noqa: E402

#: Nombre absoluto al que resuelve el import relativo de `app.py`.
MODULO = 'src.analityc.core.estacionamiento_planificador'

#: El logger de `app.py`, que es quien emite el aviso.
LOGGER_APP = 'src.app.app'


def _registros(caplog, nivel):
    """Las lineas del logger de app.py con EXACTAMENTE ese nivel."""
    return [r for r in caplog.records
            if r.name == LOGGER_APP and r.levelno == nivel]


@pytest.fixture
def sin_planificador():
    """Garantiza que el modulo no esta, pase lo que pase en otras pruebas."""
    previo = sys.modules.pop(MODULO, None)
    yield
    if previo is not None:
        sys.modules[MODULO] = previo


# ── El corte de tiendas: ausente, y eso es normal ───────────────────────

def test_el_modulo_ausente_sale_como_warning(caplog, sin_planificador):
    with caplog.at_level(logging.DEBUG, logger=LOGGER_APP):
        _arrancar_reporte_estacionamiento()

    avisos = _registros(caplog, logging.WARNING)
    assert len(avisos) == 1
    assert 'estacionamiento_planificador no disponible' in avisos[0].getMessage()


def test_el_modulo_ausente_no_deja_ningun_error(caplog, sin_planificador):
    """El control al reves: esto es lo que fallaba antes del arreglo."""
    with caplog.at_level(logging.DEBUG, logger=LOGGER_APP):
        _arrancar_reporte_estacionamiento()

    assert _registros(caplog, logging.ERROR) == []


def test_el_aviso_no_lleva_traceback(caplog, sin_planificador):
    """Una linea, como `vigilante_amazonas`. `logger.exception` adjunta la
    excepcion en `exc_info` y por eso imprimia el bloque entero."""
    with caplog.at_level(logging.DEBUG, logger=LOGGER_APP):
        _arrancar_reporte_estacionamiento()

    assert all(r.exc_info is None for r in caplog.records
               if r.name == LOGGER_APP)


# ── El monorepo: presente pero roto, eso SI es un error ─────────────────

def test_si_el_modulo_existe_y_revienta_sigue_siendo_error(caplog,
                                                           monkeypatch):
    """Lo que NO se podia perder al bajar el nivel del aviso.

    Se simula el monorepo: el planificador esta, pero `planificador()` falla
    (una dependencia suya rota, un fichero de configuracion ilegible...). Ahi
    el reporte diario se ha perdido de verdad y tiene que verse con su traza.
    """
    falso = types.ModuleType(MODULO)
    # Un modulo importado de verdad lleva `__spec__`, y `find_spec` lo
    # devuelve tal cual cuando el nombre ya esta en sys.modules. Sin ponerlo,
    # `find_spec` levanta ValueError y el postizo pasaria por ausente: se
    # estaria probando el caso contrario al que se quiere probar.
    falso.__spec__ = importlib.machinery.ModuleSpec(MODULO, loader=None)

    def _revienta():
        raise RuntimeError('la bitacora del estacionamiento no abre')

    falso.planificador = _revienta
    monkeypatch.setitem(sys.modules, MODULO, falso)

    import src.analityc.core as _core
    monkeypatch.setattr(_core, 'estacionamiento_planificador', falso,
                        raising=False)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_APP):
        _arrancar_reporte_estacionamiento()

    errores = _registros(caplog, logging.ERROR)
    assert len(errores) == 1
    assert errores[0].exc_info is not None          # la traza sigue ahi
    assert _registros(caplog, logging.WARNING) == []


# ── El filtro por vertical, que ya existia ──────────────────────────────

def test_si_esta_vertical_no_sirve_estacionamiento_no_dice_nada(
        caplog, monkeypatch, sin_planificador):
    """Con ELDE_PIPELINES acotado ni siquiera se mira el modulo: no hay
    reporte que programar, asi que tampoco hay nada que avisar."""
    monkeypatch.setenv('ELDE_PIPELINES', 'Tiendas,Anaqueles')

    with caplog.at_level(logging.DEBUG, logger=LOGGER_APP):
        _arrancar_reporte_estacionamiento()

    assert [r for r in caplog.records if r.name == LOGGER_APP] == []
