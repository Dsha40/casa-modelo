"""
Cerrar un cliente NO es un error crítico.

Starlette no lanza `WebSocketDisconnect` cuando el bucle intenta leer (o
escribir) una vez más después del mensaje de cierre: lanza un `RuntimeError`
con un texto concreto. Ese RuntimeError caía en el `except Exception` general
del endpoint y se registraba así **cada vez que alguien cerraba su cliente**:

    ERROR | Error critico client=...: Cannot call "receive" once a disconnect
            message has been received.

El problema no era el ruido, era el enmascaramiento: un fallo de verdad al
desconectar se confundía con esto, y quien mirase el log aprendía a ignorar
la línea. Ahora se distingue, y **solo** para esos dos mensajes: cualquier
otro `RuntimeError` sigue siendo un error crítico, que es lo que es.

Se comparan fragmentos y no el texto exacto porque el mensaje ha cambiado
entre versiones de Starlette y no hay una clase de excepción propia que
atrapar. Los fragmentos se verificaron contra
`starlette/websockets.py` (líneas 57 y 98).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app.app import _es_cierre_normal  # noqa: E402

#: Literales de Starlette, copiados de su fuente.
RECIBIR = 'Cannot call "receive" once a disconnect message has been received.'
ENVIAR = 'Cannot call "send" once a close message has been sent.'


# ── Lo que SÍ es una desconexión normal ─────────────────────────────────
def test_leer_despues_del_cierre_es_una_desconexion():
    assert _es_cierre_normal(RuntimeError(RECIBIR)) is True


def test_escribir_despues_del_cierre_es_una_desconexion():
    assert _es_cierre_normal(RuntimeError(ENVIAR)) is True


def test_los_mensajes_de_starlette_siguen_siendo_estos():
    """Si Starlette cambia el texto, esto avisa antes que el log."""
    fuente = (RAIZ / 'server' / 'venv' / 'Lib' / 'site-packages' / 'starlette'
              / 'websockets.py')
    if not fuente.is_file():
        return                       # sin venv instalado: nada que comprobar
    texto = fuente.read_text(encoding='utf-8')
    assert RECIBIR in texto, 'Starlette cambió el mensaje de recepción'
    assert ENVIAR in texto, 'Starlette cambió el mensaje de envío'


# ── Lo que NO puede colar como desconexión ──────────────────────────────
def test_un_fallo_de_verdad_sigue_siendo_critico():
    """Es el riesgo de este arreglo: tragarse errores reales."""
    reales = (
        RuntimeError('CUDA out of memory'),
        RuntimeError('Event loop is closed'),
        RuntimeError('dictionary changed size during iteration'),
        RuntimeError('WebSocket is not connected. Need to call "accept" first.'),
        RuntimeError(''),
        ValueError('Tipo de inferencia desconocido'),
    )
    for exc in reales:
        assert _es_cierre_normal(exc) is False, f'se tragó {exc!r}'


def test_no_revienta_con_una_excepcion_rara():
    class _Rara(Exception):
        def __str__(self):
            raise ValueError('ni el mensaje se puede leer')

    try:
        _es_cierre_normal(_Rara())
    except ValueError:
        pass                         # aceptable: lo que no puede es colar
    else:
        pass


# ── Contrato con el endpoint ────────────────────────────────────────────
def test_el_endpoint_distingue_los_dos_casos():
    import inspect
    from src.app.app import websocket_endpoint
    fuente = inspect.getsource(websocket_endpoint)
    assert 'except RuntimeError as exc:' in fuente
    assert '_es_cierre_normal(exc)' in fuente
    # El genérico sigue ahí: no se ha ampliado lo que se considera normal.
    assert 'except Exception as exc:' in fuente
    assert fuente.count('Error critico client=') == 2


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
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
