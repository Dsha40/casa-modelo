"""
Pruebas de la evidencia fotográfica (`permanencia_evidencia.py`).

Dos cosas que parecen menores y no lo son:

  - **el base64 se genera aunque el disco falle** — si escribir la copia de
    respaldo revienta, la alerta debe salir igual; perder el respaldo es un
    incidente, perder el aviso es el fallo;
  - **el banner se translitera** — `cv2.putText` solo dibuja Hershey, que no
    tiene tildes ni «ñ»: sin esto la foto diría «Entr? al ?rea».

Y tres cosas de la propia suite (4-sep-2026, al migrar el servidor a Linux):

  - **el disco que falla se simula con un FICHERO donde el gestor espera su
    carpeta**, dentro de un directorio temporal. Antes era `Z:/no/existe`:
    en Windows es una unidad inexistente, pero en Linux es una ruta RELATIVA
    perfectamente creable; `mkdir(parents=True)` la fabricaba debajo de
    `server/`, la foto se escribía, la prueba fallaba y quedaba un
    `server/Z:/` de residuo. Con el fichero-tapón `mkdir` revienta en
    cualquier SO (`NotADirectoryError` en Linux, `FileExistsError` en
    Windows) y no se toca nada fuera del temporal. Ojo: eso cubre SOLO la
    rama en que `mkdir` LANZA. La otra rama de «disco que falla» —carpeta
    existente pero sin permisos, o disco lleno— pasa por `cv2.imwrite`, que
    devuelve `False` SIN lanzar, y producción no mira ese retorno: hoy
    devuelve la ruta de un fichero que no existe. Esa prueba existe abajo
    marcada `xfail(strict=True)` para que grite el día que se corrija;
  - **el reloj se congela** (martes 1-sep-2026 12:00, hora local) en todo
    lo que nombra carpetas o purga por fecha. El gestor lee `datetime.now()`
    para nombrar `<fecha>/<cámara>/<HHMMSS>_…jpg` y `date.today()` para
    decidir qué carpetas están fuera de retención. Con el reloj real, las
    pruebas solo podían afirmar el nombre de la cámara (la fecha y la hora
    cambiaban en cada corrida) y los márgenes de la purga (90 y 1 día frente
    a 30) absorbían un cruce de medianoche por suerte, no por diseño. Ahora
    las fechas sembradas son RELATIVAS al instante congelado, la ruta se
    afirma completa y la frontera exacta de la retención tiene su prueba;
  - **la medianoche reparte las fotos de una visita en dos carpetas de
    fecha, y eso es DELIBERADO**: `_escribir` nombra la carpeta con el
    `now()` de CADA escritura («las fotos se agrupan por fecha», dice el
    módulo), así que una entrada a las 23:59:59 y su comparativa a las
    00:00:01 no caen juntas. En vez de dejar que la suite dependa de a qué
    hora corre, esa frontera es una prueba explícita con dos instantes
    congelados.
"""
from __future__ import annotations

import base64
import contextlib
import datetime
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import permanencia_evidencia as pe  # noqa: E402
from src.analityc.core.permanencia_evidencia import (  # noqa: E402
    COLORES, Evidencia, GestorEvidencia, a_b64, componer, de_b64, id_corto,
    nombre_seguro, recortar, sin_acentos)

FRAME = np.full((480, 640, 3), 90, dtype=np.uint8)
CAJA = (100.0, 60.0, 200.0, 400.0)


# ── Reloj congelado ─────────────────────────────────────────────────────
#: Martes 1-sep-2026 a las 12:00 en hora LOCAL (`time.mktime`, no UTC): así
#: la carpeta de fecha y el prefijo HHMMSS que escribe el gestor son los que
#: las pruebas afirman, sea cual sea la zona horaria de la máquina.
INSTANTE = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))
HOY = datetime.date.fromtimestamp(INSTANTE)              # 2026-09-01

#: Los dos lados de la medianoche del 1 al 2 de septiembre, también en hora
#: local: la prueba de la frontera captura la entrada en el primero y la
#: comparativa en el segundo.
VISPERA_235959 = time.mktime((2026, 9, 1, 23, 59, 59, 0, 0, -1))
SIGUIENTE_000001 = time.mktime((2026, 9, 2, 0, 0, 1, 0, 0, -1))


class _ModuloConReloj:
    """`datetime` o `time` con el reloj sustituido; todo lo demás, al real.

    Se cuelga del MÓDULO probado (`pe.datetime`, `pe.time`) y no del
    intérprete: congelar `datetime.date` a nivel global alcanzaría a
    cualquier otra prueba de la sesión y al propio `tempfile`.
    """

    def __init__(self, real, **sustitutos):
        self._real, self._sustitutos = real, sustitutos

    def __getattr__(self, nombre):
        if nombre in self._sustitutos:
            return self._sustitutos[nombre]
        return getattr(self._real, nombre)


@contextlib.contextmanager
def _reloj_congelado(instante: float = INSTANTE):
    """Fija `datetime.now()`, `date.today()` y `time.time()` tal como los ve
    `permanencia_evidencia`. Es un gestor de contexto y no un fixture de
    pytest porque este fichero también corre sin pytest (ver el final)."""
    class _DateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime.fromtimestamp(instante, tz)

    class _Date(datetime.date):
        @classmethod
        def today(cls):
            return datetime.date.fromtimestamp(instante)

    originales = pe.datetime, pe.time
    pe.datetime = _ModuloConReloj(datetime, datetime=_DateTime, date=_Date)
    pe.time = _ModuloConReloj(time, time=lambda: instante)
    try:
        yield
    finally:
        pe.datetime, pe.time = originales


# ── Texto ───────────────────────────────────────────────────────────────
def test_el_banner_se_translitera():
    assert sin_acentos('Entró al área · Niño') == 'Entro al area  Nino'


def test_nombre_de_camara_utilizable_como_carpeta():
    assert nombre_seguro('Puerta Almacén / 3') == 'PuertaAlmacen3'
    assert nombre_seguro('') == 'cam'
    assert nombre_seguro('////') == 'cam'


def test_id_corto_cabe_en_un_banner():
    assert id_corto('7d4f1a2b-9c3e-4f10-8a55-1122334455aa') == '7d4f1a2b'
    assert id_corto('prov:1234567890') == 'prov:1234567'
    assert id_corto(None) == 'sin-id'


# ── Recorte ─────────────────────────────────────────────────────────────
def test_el_recorte_deja_aire_alrededor():
    """Pegado al cuerpo se pierde la cabeza, que es lo que se mira."""
    r = recortar(FRAME, CAJA)
    assert r.shape[0] > (CAJA[3] - CAJA[1])
    assert r.shape[1] > (CAJA[2] - CAJA[0])


def test_el_recorte_no_se_sale_del_frame():
    r = recortar(FRAME, (600.0, 440.0, 700.0, 600.0))
    assert r.shape[0] <= 480 and r.shape[1] <= 640


def test_sin_caja_devuelve_el_frame_entero():
    assert recortar(FRAME, None).shape == FRAME.shape


def test_una_caja_corrupta_devuelve_el_frame_y_no_lanza():
    assert recortar(FRAME, ('x', None, 3, 4)).shape == FRAME.shape


def test_sin_frame_no_hay_recorte():
    assert recortar(None, CAJA) is None
    assert recortar(np.zeros((0, 0, 3), dtype=np.uint8), CAJA) is None


# ── Codificación ────────────────────────────────────────────────────────
def test_ida_y_vuelta_de_base64():
    b64 = a_b64(FRAME)
    assert b64 and base64.b64decode(b64)
    img = de_b64(b64)
    assert img is not None and img.shape == FRAME.shape


def test_base64_de_nada_es_cadena_vacia():
    """Cadena vacía, NO excepción: esto corre en el bucle de inferencia."""
    assert a_b64(None) == ''
    assert a_b64(np.zeros((0, 0, 3), dtype=np.uint8)) == ''
    assert de_b64('') is None
    assert de_b64('esto no es base64 ni una imagen') is None


# ── Comparativa ─────────────────────────────────────────────────────────
def test_la_comparativa_pone_las_dos_fotos_lado_a_lado():
    izq = np.zeros((100, 40, 3), dtype=np.uint8)
    der = np.zeros((160, 60, 3), dtype=np.uint8)
    par = componer(izq, der)
    assert par.shape[0] == 160, 'se igualan a la más alta'
    assert par.shape[1] == 40 + 4 + 60, 'ancho de las dos más el separador'


def test_si_falta_una_foto_la_comparativa_devuelve_la_otra():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert componer(None, img) is img
    assert componer(img, None) is img
    assert componer(None, None) is None


# ── Gestor ──────────────────────────────────────────────────────────────
def test_captura_escribe_el_archivo_y_devuelve_el_base64():
    """El respaldo va a `<fecha>/<cámara>/<HHMMSS>_<id>_<momento>.jpg`."""
    with tempfile.TemporaryDirectory() as tmp, _reloj_congelado():
        g = GestorEvidencia(Path(tmp))
        ev = g.capturar(FRAME, CAJA, camara='Puerta Almacén',
                        persistent_id='abc-123', momento='entrada',
                        titulo='Entrada por la puerta',
                        lineas=['Hora: 10:00:00'])
        assert ev and ev.b64 and ev.ruta
        ruta = Path(ev.ruta)
        assert ruta.is_file() and ruta.stat().st_size > 0
        # La ruta COMPLETA, no solo el nombre de la cámara: con el reloj
        # congelado, la carpeta de fecha y el prefijo de hora forman parte
        # del contrato (es lo que un operador busca a mano en el disco).
        assert ruta.relative_to(tmp) == (
            Path('2026-09-01') / 'PuertaAlmacen'
            / '120000_abc123_entrada.jpg')


def test_sin_guardar_en_disco_sigue_habiendo_alerta():
    """El respaldo es opcional; el aviso no."""
    with tempfile.TemporaryDirectory() as tmp:
        g = GestorEvidencia(Path(tmp), guardar_en_disco=False)
        ev = g.capturar(FRAME, CAJA, camara='cam', persistent_id='p1',
                        momento='entrada', titulo='Entrada')
        assert ev.b64 and ev.ruta == ''
        assert not any(Path(tmp).iterdir()), 'no debía tocar el disco'


def test_un_disco_que_falla_no_se_lleva_la_alerta():
    """Si `mkdir` LANZA, el respaldo se pierde y la alerta (el base64) sale.

    Cubre SOLO esa rama: el «disco que falla» es un FICHERO en el sitio
    donde el gestor espera su carpeta raíz, y `mkdir(parents=True)` no puede
    crear nada debajo de un fichero en ningún SO. (Hasta el 4-sep-2026 era
    `Z:/no/existe`, que en Linux es una ruta relativa creable: ver la
    cabecera del fichero.) La otra rama —carpeta creada pero sin permisos,
    donde quien falla es `cv2.imwrite` devolviendo `False` sin lanzar— NO
    pasa por aquí: ver `test_una_carpeta_sin_permisos_no_se_lleva_la_alerta`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tapon = Path(tmp) / 'disco'
        tapon.write_bytes(b'')
        g = GestorEvidencia(tapon / 'evidencia')
        ev = g.capturar(FRAME, CAJA, camara='cam', persistent_id='p1',
                        momento='salida', titulo='Salida')
        assert ev.b64, 'se perdió la foto de la alerta por un fallo de disco'
        assert ev.ruta == ''
        assert g._fallos_disco == 1, 'el fallo debe contarse para avisar'
        assert tapon.is_file(), 'el gestor no se abre paso por la fuerza'
        assert [p.name for p in Path(tmp).iterdir()] == ['disco'], \
            'no debía escribir nada fuera de su carpeta'


@pytest.mark.skipif(os.name == 'nt' or os.geteuid() == 0,
                    reason='chmod no frena a root, y en Windows los '
                           'permisos de carpeta no significan lo mismo')
@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason='permanencia_evidencia.py:406 no comprueba el retorno de '
           'cv2.imwrite, que devuelve False SIN lanzar cuando no puede '
           'abrir el fichero: hoy devuelve la ruta de un fichero que no '
           'existe y no cuenta el fallo (4-sep-2026)')
def test_una_carpeta_sin_permisos_no_se_lleva_la_alerta():
    """Carpeta `<fecha>/<cámara>` existente pero de solo lectura: el caso
    que nombra el propio comentario de producción («Disco lleno o carpeta
    sin permisos»). `mkdir(exist_ok=True)` pasa, `cv2.imwrite` devuelve
    `False` y NO lanza. Lo que debe ocurrir es lo mismo que cuando `mkdir`
    lanza: base64 sí, `ruta == ''`, un fallo contado (para que el aviso del
    `logger` salga) y ningún fichero fantasma.

    Hoy producción devuelve la ruta de un fichero inexistente, que además se
    persiste en el libro de estancias (`permanencia_area.py`, `entrada_foto`
    y `'ruta'`). Marcada `xfail(strict=True)`: el día que producción
    compruebe el retorno de `imwrite`, esta prueba pasará y pytest exigirá
    quitar la marca. `raises=AssertionError` para que un fallo de montaje
    del propio test (un chmod que no aplica, un tmp raro) no se disfrace
    de «esperado».
    """
    with tempfile.TemporaryDirectory() as tmp, _reloj_congelado():
        raiz = Path(tmp)
        destino = raiz / HOY.isoformat() / 'cam'
        destino.mkdir(parents=True)
        os.chmod(destino, 0o500)
        try:
            g = GestorEvidencia(raiz)
            ev = g.capturar(FRAME, CAJA, camara='cam', persistent_id='p1',
                            momento='salida', titulo='Salida')
            assert ev.b64, 'se perdió la foto de la alerta por falta de permisos'
            assert ev.ruta == '', 'ruta fantasma: el fichero no se escribió'
            assert g._fallos_disco == 1, 'el fallo debe contarse para avisar'
            assert not any(destino.iterdir()), 'nada se escribió sin permiso'
        finally:
            # Devolver los permisos ANTES de que `TemporaryDirectory` limpie:
            # que el test no deje una carpeta que ni él mismo pueda borrar.
            os.chmod(destino, 0o700)


def test_la_foto_se_encoge_para_no_reventar_el_websocket():
    with tempfile.TemporaryDirectory() as tmp:
        g = GestorEvidencia(Path(tmp), max_ancho=100, max_alto=100,
                            guardar_en_disco=False)
        ev = g.capturar(FRAME, None, camara='cam', persistent_id='p1',
                        momento='entrada', titulo='Entrada')
        img = de_b64(ev.b64)
        assert img is not None and img.shape[1] <= 260  # ancho mínimo legible


def test_la_comparativa_del_gestor_junta_entrada_y_salida():
    with tempfile.TemporaryDirectory() as tmp, _reloj_congelado():
        g = GestorEvidencia(Path(tmp))
        entrada = g.capturar(FRAME, CAJA, camara='cam', persistent_id='p1',
                             momento='entrada', titulo='Entrada')
        salida = de_b64(g.capturar(FRAME, CAJA, camara='cam',
                                   persistent_id='p1', momento='salida',
                                   titulo='Salida').b64)
        par = g.comparativa(entrada.b64, salida, camara='cam',
                            persistent_id='p1')
        assert par and Path(par.ruta).is_file()
        assert de_b64(par.b64).shape[1] > de_b64(entrada.b64).shape[1]
        # `comparativa` y `capturar` comparten el esquema `<fecha>/<cámara>`
        # y el sufijo de momento. Con el reloj congelado eso es LA MISMA
        # carpeta; con el reloj real no tiene por qué serlo (cada escritura
        # toma su propio `now()`): la medianoche tiene su prueba aparte.
        assert Path(par.ruta).relative_to(tmp) == (
            Path('2026-09-01') / 'cam' / '120000_p1_comparativa.jpg')
        assert Path(par.ruta).parent == Path(entrada.ruta).parent


def test_una_visita_que_cruza_la_medianoche_reparte_sus_fotos_por_fecha():
    """Entrada a las 23:59:59, comparativa a las 00:00:01: DOS carpetas.

    Es el comportamiento real y deliberado de `_escribir` (cada escritura
    nombra su carpeta con su propio `now()`; «las fotos se agrupan por
    fecha», dice el módulo): la comparativa de una visita que cruza la
    medianoche vive en la carpeta del día en que se cerró, no en la del día
    en que empezó. Quien busque a mano «la salida de anoche» la encuentra
    en la fecha de la salida. Documentado como prueba explícita para que la
    suite no dependa de a qué hora corre.
    """
    with tempfile.TemporaryDirectory() as tmp:
        g = GestorEvidencia(Path(tmp))
        with _reloj_congelado(VISPERA_235959):
            entrada = g.capturar(FRAME, CAJA, camara='cam',
                                 persistent_id='p1', momento='entrada',
                                 titulo='Entrada')
        with _reloj_congelado(SIGUIENTE_000001):
            par = g.comparativa(entrada.b64, FRAME, camara='cam',
                                persistent_id='p1')
        assert Path(entrada.ruta).is_file() and Path(par.ruta).is_file()
        assert Path(entrada.ruta).relative_to(tmp) == (
            Path('2026-09-01') / 'cam' / '235959_p1_entrada.jpg')
        assert Path(par.ruta).relative_to(tmp) == (
            Path('2026-09-02') / 'cam' / '000001_p1_comparativa.jpg')
        assert Path(par.ruta).parent != Path(entrada.ruta).parent


def test_la_purga_borra_las_carpetas_viejas_y_respeta_las_ajenas():
    with tempfile.TemporaryDirectory() as tmp, _reloj_congelado():
        raiz = Path(tmp)
        vieja = raiz / (HOY - datetime.timedelta(days=90)).isoformat()
        reciente = raiz / (HOY - datetime.timedelta(days=1)).isoformat()
        ajena = raiz / 'estancias'        # el libro NO es una carpeta de fecha
        for d in (vieja, reciente, ajena):
            d.mkdir(parents=True)
        g = GestorEvidencia(raiz, retencion_dias=30)
        assert g.purgar() == 1
        assert not vieja.exists()
        assert reciente.exists() and ajena.exists()


def test_la_purga_respeta_la_carpeta_del_dia_limite():
    """«30 días de retención» = la carpeta de hace 30 días AÚN se ve.

    El límite es `hoy - retención` y se borra lo ESTRICTAMENTE anterior:
    quien pide 30 días encuentra 31 carpetas (la de hoy incluida). Es la
    única frontera donde un día de desfase del reloj cambiaría el resultado;
    por eso vive aparte y con el reloj congelado.
    """
    with tempfile.TemporaryDirectory() as tmp, _reloj_congelado():
        raiz = Path(tmp)
        en_el_limite = raiz / (HOY - datetime.timedelta(days=30)).isoformat()
        justo_fuera = raiz / (HOY - datetime.timedelta(days=31)).isoformat()
        en_el_limite.mkdir()
        justo_fuera.mkdir()
        assert GestorEvidencia(raiz, retencion_dias=30).purgar() == 1
        assert en_el_limite.exists() and not justo_fuera.exists()


def test_la_purga_no_corre_en_cada_frame():
    with tempfile.TemporaryDirectory() as tmp:
        g = GestorEvidencia(Path(tmp), retencion_dias=30)
        g.purgar()
        assert g.purgar() == 0, 'una purga por hora, no una por frame'


def test_retencion_cero_desactiva_la_purga():
    with tempfile.TemporaryDirectory() as tmp, _reloj_congelado():
        raiz = Path(tmp)
        vieja = raiz / (HOY - datetime.timedelta(days=900)).isoformat()
        vieja.mkdir(parents=True)
        assert GestorEvidencia(raiz, retencion_dias=0).purgar() == 0
        assert vieja.exists()


def test_una_evidencia_vacia_es_falsa():
    assert not Evidencia()
    assert Evidencia(b64='x')


def test_hay_un_color_por_momento():
    for momento in ('entrada', 'salida', 'permanencia', 'sin_salida'):
        assert momento in COLORES


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
if __name__ == '__main__':
    # Sin pytest nadie interpreta las marcas: `skipif` se salta a mano y
    # `xfail` se espera fallando (y cuenta como fallo si PASA, igual que
    # `strict=True`, para que no se olvide quitar la marca).
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        _marcas = {m.name: m for m in getattr(_fn, 'pytestmark', [])}
        if 'skipif' in _marcas and _marcas['skipif'].args[0]:
            print(f"  SALTO {_nombre}: {_marcas['skipif'].kwargs.get('reason', '')}")
            continue
        try:
            _fn()
        except Exception as _exc:                          # noqa: BLE001
            if 'xfail' in _marcas:
                print(f"  XFALLO {_nombre} (esperado): "
                      f"{type(_exc).__name__}: {_exc}")
            else:
                fallos += 1
                print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
        else:
            if 'xfail' in _marcas:
                fallos += 1
                print(f"  XPASA {_nombre}: producción ya cumple; "
                      f"quita la marca xfail")
            else:
                print(f"  OK    {_nombre}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
