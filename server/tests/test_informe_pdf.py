"""
Prueba del informe PDF del día (14-ago-2026; reloj congelado el 4-sep-2026).

No valida el contenido visual (eso es mirar el PDF): valida el CONTRATO —
que el informe SIEMPRE se genera, con datos o sin ellos. Cada sección es
defensiva por diseño: una fuente caída degrada a «sin datos», jamás tumba
el documento entero.

Hasta el 4-sep-2026 esta prueba pedía el informe de `time.strftime(
'%Y-%m-%d')` (el día natural) contra el entorno REAL del servidor
(`output/`, la base de la tienda: «carpetas llenas o vacías»). Dentro,
`_sec_resumen` decide si el informe es «de hoy» con OTRA lectura del reloj
y con el corte de jornada del entorno (`fecha == jornada_de()`), así que lo
que se ejercitaba cambiaba con el día, con la medianoche y con
`ELDE_TIENDA_CORTE_JORNADA_H`: unas veces se probaba la rama «hoy, con
datos» y otras «un día sin medir», según a qué hora corriese la suite.
Ahora el reloj se congela en un instante FIJO, los datos se siembran
RELATIVOS a ese instante en un almacén de usar y tirar, y cada rama tiene
su prueba con nombre: el informe de hoy con datos, un día sin nada, y la
frontera del corte de jornada (a la 01:30 con corte a las 22:00 el informe
«de hoy» es el de AYER en el calendario, que es lo que quiere una tienda
que cierra de madrugada).

Las secciones que leen carpetas (`anaqueles`, `personal_tienda`, `heatmap`,
`captures`) se apuntan a `tmp_path`: en el servidor Linux `output/` está
montada dentro del contenedor de producción y la suite no debe ni leerla.
"""
from __future__ import annotations

import base64
import io
import re
import sys
import time
import zlib
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import anaqueles_registro  # noqa: E402
from src.analityc.core import personal_galeria  # noqa: E402
from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.app import api_lectura, api_tienda  # noqa: E402
from src.app.informe_pdf import construir_informe_tienda  # noqa: E402

#: Martes 1-sep-2026 a las 12:00 en hora LOCAL (`mktime`, no un número
#: mágico en UTC): lejos de la medianoche, y el «día anterior» del
#: comparativo es el lunes. La jornada es el día natural porque el fixture
#: fija el corte a 0; con `ELDE_TIENDA_CORTE_JORNADA_H=22` en el entorno,
#: las 12:00 pertenecerían aún al 31-ago y esta prueba no diría lo que dice.
MEDIODIA = time.mktime((2026, 9, 1, 12, 0, 0, -1, -1, -1))
JORNADA = '2026-09-01'

#: Lo que devuelve `_aforo_vivo` simulado. Un número que no puede salir de
#: ninguna otra cifra del informe (se siembran 1-2 personas): si aparece
#: como celda entera del PDF, la sección del resumen tomó la rama «este
#: informe es de hoy». Menor que mil a propósito: `_fmt` pone punto de
#: millar y 4321 saldría como «4.321».
AFORO_CENTINELA = 977
CELDA_AFORO = str(AFORO_CENTINELA).encode()


def _congelar(monkeypatch, instante: float) -> None:
    """`time.time()` y `time.strftime()` sin argumento devuelven `instante`.

    Basta con el módulo `time`: `tienda_almacen.jornada_de()` e
    `informe_pdf` hacen `import time` y leen `time.time` al llamar. El
    `strftime` va para la portada («generado el ...») y para el `_hoy()` de
    `anaqueles_registro`, que también leen el reloj sin pasar instante.
    """
    strftime_real = time.strftime
    monkeypatch.setattr(time, 'time', lambda: instante)
    monkeypatch.setattr(
        time, 'strftime',
        lambda formato, t=None: strftime_real(
            formato, time.localtime(instante) if t is None else t))


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Todo lo que lee el informe, en `tmp_path` y con el reloj a mediodía.

    Devuelve el almacén ya reiniciado (volcado inmediato) para que cada
    prueba siembre lo suyo relativo a `time.time()`.
    """
    _congelar(monkeypatch, MEDIODIA)
    # El corte de jornada del ENTORNO no manda aquí: estas pruebas presuponen
    # el día natural, y la del corte a las 22:00 lo fija ella misma.
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    monkeypatch.setattr(anaqueles_registro, 'raiz_salida',
                        lambda: tmp_path / 'anaqueles')
    monkeypatch.setattr(
        personal_galeria, '_galeria',
        personal_galeria.GaleriaPersonalTienda(raiz=tmp_path / 'personal'))
    monkeypatch.setattr(AnalyticsConfig, 'HEATMAP_DIR',
                        str(tmp_path / 'heatmap'))
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_DIR',
                        str(tmp_path / 'captures'))
    monkeypatch.setattr(api_lectura, '_salida', lambda: tmp_path / 'output')
    # «Personas ahora» pregunta a los procesadores vivos del servidor de
    # inferencia; aquí no hay ninguno, así que se simula con el centinela.
    monkeypatch.setattr(api_tienda, '_aforo_vivo', lambda cam: AFORO_CENTINELA)
    return ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')


def _sembrar_jornada(alm, ahora: float) -> None:
    """Una jornada con vida y la anterior con algo, ambas RELATIVAS a `ahora`.

    Dos personas hoy (una con demografía) y una ayer a la misma hora, para
    que el comparativo «vs día anterior» tenga base y no salga «— sin base».
    """
    alm.anotar_entrada('cam-1', 'p-1', ts=ahora - 3 * 3600)
    alm.anotar_demografia('cam-1', 'p-1', 'female', '25-34',
                          ts=ahora - 3 * 3600 + 5)
    alm.anotar_salida('cam-1', 'p-1', permanencia_s=90.0,
                      ts=ahora - 3 * 3600 + 90)
    alm.anotar_entrada('cam-1', 'p-2', ts=ahora - 2 * 3600)
    alm.anotar_entrada('cam-1', 'p-9', ts=ahora - 86400 - 3 * 3600)
    alm.anotar_salida('cam-1', 'p-9', permanencia_s=30.0,
                      ts=ahora - 86400 - 3 * 3600 + 30)
    alm.volcar(force=True)


def _cadenas(pdf: bytes) -> list:
    """Las cadenas literales `(...) Tj` de todas las páginas, decodificadas.

    reportlab escribe cada página como un stream ASCII85 + Flate; sin
    abrirlo, «el PDF contiene X» no se puede afirmar. Cada celda de tabla y
    cada tramo de párrafo es una cadena aparte; los acentos van como escapes
    octales de WinAnsi (`\\341`), así que se comparan prefijos ASCII.
    """
    salida = b''
    for m in re.finditer(rb'stream\n(.*?)endstream', pdf, re.S):
        bruto = m.group(1).strip()
        if bruto.endswith(b'~>'):
            bruto = bruto[:-2]
        try:
            salida += zlib.decompress(base64.a85decode(bruto, adobe=False))
        except Exception:                                   # noqa: BLE001
            continue                    # streams binarios (imágenes, fuentes)
    return re.findall(rb'\(([^)]*)\)\s*Tj', salida)


def _informe(fecha: str) -> bytes:
    buf = io.BytesIO()
    construir_informe_tienda(buf, fecha)
    return buf.getvalue()


def _dice_sin_medir(cadenas: list) -> bool:
    return any(c.startswith(b'No hay ning') for c in cadenas)


def test_el_informe_siempre_se_genera(entorno):
    """El informe de HOY, con datos: PDF entero y con la rama «de hoy».

    El reloj está congelado el martes a mediodía, así que «hoy» es una
    jornada conocida (`JORNADA`) y no lo que marque la máquina. Que el
    centinela del aforo esté en el PDF demuestra que `_sec_resumen`
    reconoció la fecha como la jornada en curso y consultó el aforo vivo.
    """
    assert ta.jornada_de() == JORNADA          # el reloj congelado manda
    _sembrar_jornada(entorno, time.time())

    datos = _informe(JORNADA)

    assert datos[:5] == b'%PDF-'
    assert len(datos) > 5000            # portada + secciones, no un PDF hueco
    cadenas = _cadenas(datos)
    assert CELDA_AFORO in cadenas
    assert not _dice_sin_medir(cadenas)     # hubo datos: sin aviso de «sin medir»


def test_una_fecha_sin_datos_no_lo_rompe(entorno):
    """Un día sin nada registrado sale igual, diciendo que no se midió.

    Y como no es hoy, no pregunta «Personas ahora»: el centinela no aparece.
    """
    _sembrar_jornada(entorno, time.time())

    datos = _informe('2020-01-01')

    assert datos[:5] == b'%PDF-'
    cadenas = _cadenas(datos)
    assert _dice_sin_medir(cadenas)
    assert CELDA_AFORO not in cadenas


def test_con_corte_de_jornada_el_informe_de_hoy_es_el_de_ayer_en_el_calendario(
        entorno, tmp_path, monkeypatch):
    """Frontera DELIBERADA: a la 01:30 con corte a las 22:00, «hoy» es ayer.

    Una tienda que cierra de madrugada pone `ELDE_TIENDA_CORTE_JORNADA_H=22`:
    la clienta de las 23:30 y el de la 01:00 pertenecen a la jornada del
    día en que abrió. A la 01:30 del 2-sep, `jornada_de()` es el 1-sep, así
    que el informe del 1-sep es el «de hoy» (aforo vivo, sin aviso de «sin
    medir», con las dos entradas de madrugada) y el del 2-sep, aunque sea la
    fecha del calendario, es un día que todavía NO existe para la tienda.
    Antes esto dependía de la hora a la que corriese la suite; ahora está
    escrito.
    """
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 22)
    madrugada = time.mktime((2026, 9, 2, 1, 30, 0, -1, -1, -1))
    _congelar(monkeypatch, madrugada)
    assert ta.jornada_de() == '2026-09-01'

    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda-corte.db')
    alm.anotar_entrada('cam-1', 'p-1', ts=madrugada - 2 * 3600)   # 23:30 del 1-sep
    alm.anotar_entrada('cam-1', 'p-2', ts=madrugada - 30 * 60)    # 01:00 del 2-sep
    alm.volcar(force=True)
    assert alm.jornadas_con_datos('2026-09-01', '2026-09-02') == ['2026-09-01']

    de_hoy = _cadenas(_informe('2026-09-01'))
    assert CELDA_AFORO in de_hoy
    assert not _dice_sin_medir(de_hoy)

    del_calendario = _cadenas(_informe('2026-09-02'))
    assert CELDA_AFORO not in del_calendario
    assert _dice_sin_medir(del_calendario)


def test_el_informe_por_defecto_es_el_de_la_jornada_en_curso(monkeypatch):
    """Con corte a las 22:00, a la 01:30 del día 2 la jornada sigue siendo la
    del día 1: el informe «de hoy» debe ser el del día 1, no el del día 2
    (que aún no tiene nada)."""
    import time as _time
    from src.app import informe_pdf

    class _Reloj:
        def __init__(self, instante):
            self.instante = instante

        def time(self):
            return self.instante

        def __getattr__(self, nombre):
            return getattr(_time, nombre)

    una_y_media = _time.mktime((2026, 9, 2, 1, 30, 0, 0, 0, -1))
    monkeypatch.setattr(ta, 'time', _Reloj(una_y_media))
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 22)
    pedidas = []
    monkeypatch.setattr(informe_pdf, 'construir_informe_tienda',
                        lambda destino, fecha: pedidas.append(fecha))
    # Desde el 13-oct-2026 se pide antes el del pasillo (mismo periodo); aquí
    # «no se puede imprimir» y sale el clásico, que es lo que se mira.
    del_pasillo = []
    monkeypatch.setattr(informe_pdf, '_pdf_pasillo',
                        lambda desde, hasta, camara: del_pasillo.append((desde, hasta)))
    informe_pdf.informe_tienda(None)
    assert pedidas == ['2026-09-01']
    assert del_pasillo == [('2026-09-01', '2026-09-01')]
    # Con el día natural (corte 0) sí es el día 2; y una fecha explícita manda.
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    informe_pdf.informe_tienda(None)
    informe_pdf.informe_tienda('2026-08-15')
    assert pedidas[1:] == ['2026-09-02', '2026-08-15']
