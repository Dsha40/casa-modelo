"""
La captura de ventana no puede quedarse muda ni sin respaldo (8-ago-2026).

## El incidente

El cliente de detecciones dejo de capturar la ventana del iVMS-4200 y estuvo
**19 horas** enviando la misma imagen, sin una sola linea en ningun registro.
Medido en la maquina, sobre las dos ventanas que el cliente capturaba:

    PrintWindow(0)   0 exitos de 8
    PrintWindow(2)   0 exitos de 8
    BitBlt           8 exitos de 8, con imagen VIVA (8 firmas distintas)

Habia respaldo de BitBlt en el codigo. No servia para nada, por tres fallos
apilados, y esta prueba fija los tres:

1. **El respaldo era codigo muerto.** `PyCDC.BitBlt` de win32ui devuelve `None`
   cuando tiene EXITO y avisa de los fallos LANZANDO. El codigo hacia
   `result = saveDC.BitBlt(...)` y luego `if result:` — siempre falso. Cada vez
   que PrintWindow fallaba se tiraba una captura perfectamente buena.

2. **El metodo se elegia una vez y nunca se reintentaba.** `pw_flag` se
   cacheaba en el primer frame. PrintWindow le manda un WM_PRINT a la
   aplicacion y ESPERA, asi que puede empezar a fallar sin que la ventana
   cambie; con el flag fijado para siempre, la captura no volvia sola jamas.

3. **El worker no podia avisar.** El QProcess se abria con `MergedChannels`:
   stderr caia dentro del flujo de frames en msgpack y lo corrompia (el
   servidor registraba «Error deserializando mensaje»). Para no romperlo, el
   worker se lo callaba todo — de ahi las 19 h en silencio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

CLIENTES = ('perimetrales', 'tienda', 'amazonas', 'detecciones')

win32 = pytest.importorskip('win32con', reason='solo aplica en Windows')

from elde_core.capture import capture_worker as cw  # noqa: E402


# ── 1. El respaldo de BitBlt ─────────────────────────────────────────

class _DCFalso:
    """Doble de PyCDC: BitBlt devuelve None al copiar bien, como el de verdad."""

    def __init__(self, falla=False):
        self.falla = falla
        self.llamadas = 0

    def BitBlt(self, *_a, **_k):
        self.llamadas += 1
        if self.falla:
            raise RuntimeError('no se pudo copiar')
        return None                      # <- exito, y aun asi None


def test_bitblt_que_copia_bien_cuenta_como_exito():
    """El defecto exacto: `None` significaba exito y se leia como fallo."""
    dc = _DCFalso()
    assert cw._copiar_con_bitblt(dc, _DCFalso(), 800, 600) is True
    assert dc.llamadas == 1


def test_bitblt_que_falla_cuenta_como_fallo():
    """win32ui avisa LANZANDO; eso si es un fallo de verdad."""
    assert cw._copiar_con_bitblt(_DCFalso(falla=True), _DCFalso(),
                                 800, 600) is False


def _asigna_desde_bitblt(ruta: Path) -> bool:
    """¿Hay algun `x = ...BitBlt(...)` en el codigo REAL del archivo?

    Se mira con AST y no buscando texto: los comentarios y docstrings de estas
    mismas correcciones CITAN la linea defectuosa como ejemplo, y una busqueda
    literal se dispararia con la propia explicacion.
    """
    import ast
    arbol = ast.parse(ruta.read_text(encoding='utf-8'))
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Assign):
            continue
        valor = nodo.value
        if (isinstance(valor, ast.Call)
                and isinstance(valor.func, ast.Attribute)
                and valor.func.attr == 'BitBlt'):
            return True
    return False


def test_el_respaldo_no_se_lee_como_booleano():
    """Blindaje: que nadie vuelva a decidir con el valor que BitBlt devuelve."""
    assert not _asigna_desde_bitblt(Path(cw.__file__)), (
        'el valor de BitBlt es None al tener exito: no sirve de condicion')
    assert '_copiar_con_bitblt(' in Path(cw.__file__).read_text(
        encoding='utf-8')

    otro = RAIZ / 'elde_core' / 'capture' / 'window_grab.py'
    assert not _asigna_desde_bitblt(otro), (
        'la otra copia de la captura sigue leyendo mal el respaldo')


# ── 2. Reintento del metodo de captura ───────────────────────────────

def test_hay_umbrales_para_avisar_y_reintentar():
    assert cw.FALLOS_PARA_AVISAR > 0
    assert cw.FALLOS_PARA_REPROBAR > cw.FALLOS_PARA_AVISAR, (
        'primero se avisa, y solo despues se vuelve a probar el metodo')
    assert cw.SEG_PARA_AVISAR_CONGELADA > 0
    assert cw.CADA_CUANTOS_COMPROBAR > 0


def test_los_umbrales_se_pueden_ajustar_por_entorno(monkeypatch):
    """Regla 6: nada de valores fijos en el codigo."""
    monkeypatch.setenv('CAPTURE_FALLOS_AVISO', '7')
    assert cw._entero_de_entorno('CAPTURE_FALLOS_AVISO', 180) == 7
    monkeypatch.setenv('CAPTURE_FALLOS_AVISO', 'no-es-un-numero')
    assert cw._entero_de_entorno('CAPTURE_FALLOS_AVISO', 180) == 180


def test_el_worker_reintenta_el_metodo_tras_fallar():
    fuente = Path(cw.__file__).read_text(encoding='utf-8')
    assert 'pw_flag, fallos = None, 0' in fuente, (
        'sin volver a poner pw_flag en None, un metodo que deja de funcionar '
        'no se reintenta nunca y la captura no vuelve sola')


def test_el_worker_vigila_que_la_imagen_CAMBIE():
    """Capturar bien y capturar algo vivo no es lo mismo."""
    fuente = Path(cw.__file__).read_text(encoding='utf-8')
    assert 'SEG_PARA_AVISAR_CONGELADA' in fuente
    assert 'MISMA' in fuente, 'debe avisar de una imagen que no cambia'


# ── 3. El worker tiene que poder hablar ──────────────────────────────

def test_los_avisos_van_por_stderr_nunca_por_stdout(capsys):
    """Por stdout van los frames: una linea de texto ahi rompe el msgpack."""
    cw._avisar('prueba')
    capturado = capsys.readouterr()
    assert 'prueba' in capturado.err
    assert capturado.out == '', 'stdout es SOLO para los frames'


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_cliente_separa_los_canales_y_lee_los_avisos(cliente):
    caja = (RAIZ.parents[1] / 'clients' / cliente / 'src' / 'gui'
            / 'components' / 'render_box' / 'render_box.py').read_text(
                encoding='utf-8')
    assert 'setProcessChannelMode(QProcess.MergedChannels)' not in caja, (
        f'{cliente}: stderr volveria a corromper el flujo de frames')
    assert 'QProcess.SeparateChannels' in caja
    assert 'readyReadStandardError' in caja, (
        f'{cliente}: nadie recoge lo que el worker avisa')


def test_el_nucleo_expone_el_lector_de_avisos():
    from elde_core.ui.render_box_captura import CapturaDVRMixin
    assert hasattr(CapturaDVRMixin, 'leer_avisos_del_worker')
