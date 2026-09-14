"""
Pruebas del almacén de tienda: la memoria CON FECHA de la analítica (HITO 1).

No se instancia el pipeline (cargaría YOLO y pediría GPU). El almacén se prueba
directo contra una base en `tmp_path`, y el cableado del pipeline llamando a sus
dos métodos con un doble de `self` — que es todo lo que necesitan.

Lo que más se protege, porque es justo lo que estaba roto:

  - **«hoy» es hoy** — la jornada se decide en UNA función, y el corte se puede
    mover (una tienda que cierra de madrugada) sin tocar código;
  - **una reconexión NO pone el marcador a cero** — el procesador renace cada
    vez que parpadea la red, y hasta ahora se llevaba el día por delante;
  - **dos sesiones del mismo día SUMAN**, y quien vuelve a entrar sigue siendo
    UN visitante único, no dos;
  - **dos cámaras no se cruzan** — se llavea por `camera_id`, nunca por el
    nombre visible: cuatro ventanas pueden llamarse «iVMS-4200» (H-39);
  - **un fallo de disco no tumba el frame** — perder analítica es malo; perder
    el vídeo en vivo, peor.

Las pruebas que anotan «ahora» viven en un instante CONGELADO (martes a
mediodía, ver `reloj`), no en la hora de la máquina: la jornada se sella con
el instante del hecho, y si la medianoche cae entre anotar y preguntar, «hoy»
sale vacío. Esa frontera no se evita sin más: se documenta en dos pruebas
propias — la del marcador que arranca en cero
(`test_a_medianoche_el_marcador_de_hoy_empieza_de_cero`) y la de la
demografía que converge pasada la medianoche, que hoy se PIERDE y queda
como `xfail` estricto (y solo por `AssertionError`: una excepción de otra
clase es una regresión y tiene que verse) hasta que producción lo decida.

Los mandos que el módulo lee del entorno al importar (`CORTE_JORNADA_H`,
`RETENCION_DIAS`) se fijan a sus valores de fábrica en `mandos_de_fabrica`:
la suite razona sobre el día natural y los 13 meses documentados, diga lo
que diga la máquina que la corre. Que 396 y 0 sean DE VERDAD lo de fábrica
lo comprueba aparte, en un intérprete limpio,
`test_los_mandos_de_fabrica_son_396_dias_y_corte_a_las_0`: con el módulo
parcheado en este proceso nadie más puede verlo.

El reloj se congela en la COSTURA de `tienda_almacen` (`ta.time`), nunca en
`time.time` global: tqdm y requests, que llegan con torch al importar
`tiendas.py`, hacen `from time import time` al cargarse y se quedarían con
la lambda congelada para el resto de la sesión (fuga demostrada el
4-sep-2026 corriendo este fichero solo).
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import tienda_almacen as ta  # noqa: E402


def _epoca_local(anio, mes, dia, hora, minuto=0) -> float:
    """Época (UTC) del instante LOCAL indicado. `-1` deja que el sistema
    resuelva el horario de verano, que es lo que hace el servidor real."""
    return time.mktime((anio, mes, dia, hora, minuto, 0, 0, 0, -1))


# ── Reloj congelado (4-sep-2026) ─────────────────────────────────────────
#: El instante en el que «viven» las pruebas que anotan sin `ts`: martes
#: 18-ago-2026 a las 12:00 en hora LOCAL. Es un literal, no `time.time()`,
#: para que ni el reloj de la máquina ni el entorno muevan lo que la suite
#: entiende por «hoy»; y es mediodía porque a esa hora la jornada es la misma
#: con cualquier corte razonable y ninguna frontera cae entre dos llamadas.
AHORA = _epoca_local(2026, 8, 18, 12, 0)
HOY = '2026-08-18'
AYER = '2026-08-17'
MANANA = '2026-08-19'


@pytest.fixture
def mandos_de_fabrica(monkeypatch):
    """Los mandos que `tienda_almacen` lee del ENTORNO valen lo de fábrica.

    `CORTE_JORNADA_H` y `RETENCION_DIAS` salen de `ELDE_TIENDA_*` al importar
    el módulo, y `jornada_de(ts)` sin `corte_h` y `purgar()` sin argumento
    caen en ellos. Las pruebas de este fichero razonan sobre el día natural
    (23:59 y 00:01 son jornadas distintas; lunes 12:00 es lunes) y sobre los
    13 meses documentados: con `ELDE_TIENDA_CORTE_JORNADA_H=22` en la
    máquina las 12:00 del 17 pertenecen a la jornada del 16 y caían dos
    pruebas; con `ELDE_TIENDA_RETENCION_DIAS=500` (o 400) la purga no borra
    lo de hace 400 días y caía otra (ambos demostrados el 4-sep-2026). El
    contrato del corte movible se prueba aparte, pasando `corte_h`
    explícito; el de «retención 0 = nunca», pasando `retencion_dias=0`.

    Fijarlos aquí tiene un precio (4-sep-2026): ninguna prueba que pase por
    esta fixture ve ya el literal de fábrica de tienda_almacen.py, así que
    un '396' cambiado a '30' por accidente pasaría en verde. Por eso el
    valor de fábrica se comprueba en un intérprete NUEVO, sin `ELDE_TIENDA_*`
    en el entorno (`test_los_mandos_de_fabrica_son_396_dias_y_corte_a_las_0`).
    """
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    monkeypatch.setattr(ta, 'RETENCION_DIAS', 396)


class _RelojFalso:
    """Doble del módulo `time` para la costura `ta.time`.

    `time()` devuelve el instante congelado; todo lo demás (`localtime`,
    `strftime`, `monotonic`...) va al módulo real. `jornada_de` llama a
    `localtime`/`strftime` siempre con argumento, así que no leen el reloj
    de la máquina por su cuenta.
    """

    def __init__(self, instante: float):
        self.instante = instante

    def time(self) -> float:
        return self.instante

    def __getattr__(self, nombre):
        return getattr(time, nombre)


@pytest.fixture
def reloj(monkeypatch, mandos_de_fabrica):
    """Congela el reloj que ve `tienda_almacen` en `AHORA`; devuelve una
    función para MOVERLO.

    Por qué (4-sep-2026): `anotar_*` sin `ts` sella el hecho con
    `time.time()` y `volcar` le asigna `jornada_de(ts)`; `contadores_de()` y
    `resumen()` sin jornada leen `jornada_de()` de OTRA llamada a
    `time.time()`, un instante después. Si la medianoche local (o la hora
    del corte) cae entre ambas, lo anotado queda en la jornada de ayer y el
    conteo de «hoy» sale 0. Reproducido arrancando cada prueba a las
    23:59:59.995: caían seis, y otras cinco tenían la misma ventana. Con un
    solo reloj congelado, lo sembrado y lo preguntado usan el mismo instante
    por construcción — y las pruebas de la frontera lo mueven a propósito
    para documentar qué pasa al cruzar la medianoche.

    Por qué la COSTURA (`ta.time`) y no `time.time` global (4-sep-2026): el
    primer arreglo parcheaba `time.time`, y las pruebas del cableado
    importaban `tiendas.py` (torch/ultralytics) con el parche puesto; tqdm
    (`std.py`, `_monitor.py`) hace `from time import time` y requests
    (`sessions.py`) `preferred_clock = time.time` al cargarse, y se quedaban
    con la lambda congelada DESPUÉS del `undo` de monkeypatch, para el resto
    de la sesión (`tqdm(...).format_dict['elapsed']` = 0.0). Sustituyendo
    solo el nombre `time` que ve `tienda_almacen`, `time.time` global no se
    toca nunca y no hay nada que un import pueda capturar, sea cual sea el
    orden de los tests.
    """
    falso = _RelojFalso(AHORA)
    monkeypatch.setattr(ta, 'time', falso)

    def mover(instante: float) -> None:
        falso.instante = instante

    assert ta.jornada_de() == HOY, 'el instante congelado no cae en HOY'
    return mover


@pytest.fixture
def almacen(tmp_path, reloj):
    """Almacén de usar y tirar, sin espera entre volcados, a mediodía."""
    return ta.AlmacenTienda(ruta=tmp_path / 'tienda.db', periodo_volcado_s=0.0)


# ── Mandos del entorno: lo de fábrica (4-sep-2026) ───────────────────────

_MANDOS_DEL_ENTORNO = ('ELDE_TIENDA_RETENCION_DIAS', 'ELDE_TIENDA_CORTE_JORNADA_H')


def _mandos_en_un_interprete_limpio(tmp_path, **variables: str) -> list:
    """`[RETENCION_DIAS, CORTE_JORNADA_H]` tal como los deja importar
    `tienda_almacen` en un intérprete NUEVO, con `ELDE_TIENDA_RETENCION_DIAS`
    y `ELDE_TIENDA_CORTE_JORNADA_H` fuera del entorno salvo lo que se pase
    en `variables`.

    Subproceso y no `importlib.reload` (4-sep-2026): el reload reiniciaría
    el singleton `_almacen` y redefiniría `AlmacenTienda` a mitad de sesión
    (las instancias vivas quedarían con la clase vieja), y en ESTE proceso
    `mandos_de_fabrica` ya ha parcheado el módulo, así que el valor de
    fábrica no se puede observar. El módulo importa solo biblioteca
    estándar y el intérprete tarda ~0,05 s. La base se apunta a `tmp_path`
    por si algún día la importación tocase disco: nada de `server/output`,
    que en el servidor Linux está montado en el contenedor de producción.
    """
    entorno = {k: v for k, v in os.environ.items()
               if k not in _MANDOS_DEL_ENTORNO}
    entorno['ELDE_TIENDA_DB'] = str(tmp_path / 'tienda.db')
    entorno.update(variables)
    hecho = subprocess.run(
        [sys.executable, '-c',
         'from src.analityc.core import tienda_almacen as ta; '
         'print(ta.RETENCION_DIAS, ta.CORTE_JORNADA_H)'],
        cwd=str(RAIZ / 'server'), env=entorno,
        capture_output=True, text=True, timeout=60, check=False)
    assert hecho.returncode == 0, hecho.stderr
    return hecho.stdout.split()


def test_los_mandos_de_fabrica_son_396_dias_y_corte_a_las_0(tmp_path):
    """Sin `ELDE_TIENDA_*` en el entorno, la retención es de 396 días (13
    meses) y la jornada empieza a las 0 h.

    Es la ÚNICA prueba de la suite que mira el literal de fábrica de
    tienda_almacen.py (`os.getenv('ELDE_TIENDA_RETENCION_DIAS', '396')` y
    `os.getenv('ELDE_TIENDA_CORTE_JORNADA_H', '0')`): el resto del fichero
    lo fija a mano en `mandos_de_fabrica` para no depender de la máquina, y
    por eso ya no lo verifica. Sin esta prueba, un '396' que pasara a '30'
    por accidente se llevaría un año de histórico del dashboard con la
    suite en verde (demostrado el 4-sep-2026: con
    `ELDE_TIENDA_RETENCION_DIAS=4000` y con `=30` el fichero pasaba igual).
    Los 13 meses los eligió el dueño del producto para comparar un mes
    contra el mismo mes del año anterior.
    """
    assert _mandos_en_un_interprete_limpio(tmp_path) == ['396', '0']


def test_los_mandos_se_leen_del_entorno_al_importar(tmp_path):
    """`ELDE_TIENDA_RETENCION_DIAS=30` y `ELDE_TIENDA_CORTE_JORNADA_H=22`
    llegan al módulo tal cual: es la tabla de variables de su cabecera, y lo
    que permite mover el corte de una tienda que cierra de madrugada, o
    acortar la retención en un disco pequeño, sin tocar código."""
    assert _mandos_en_un_interprete_limpio(
        tmp_path, ELDE_TIENDA_RETENCION_DIAS='30',
        ELDE_TIENDA_CORTE_JORNADA_H='22') == ['30', '22']


# ── La jornada ───────────────────────────────────────────────────────────

def test_la_jornada_cambia_a_medianoche(mandos_de_fabrica):
    """23:59 y 00:01 son días distintos. Es el contrato de «hoy».

    Se llama SIN `corte_h` a propósito: es el camino que usa producción
    (`contadores_de`, `resumen`...), y lo que se fija es que con el corte a
    las 0 h la jornada es el día natural. El 0 lo pone `mandos_de_fabrica`;
    que 0 sea el corte de FÁBRICA lo comprueba
    `test_los_mandos_de_fabrica_son_396_dias_y_corte_a_las_0`.
    """
    assert ta.jornada_de(_epoca_local(2026, 8, 18, 23, 59)) == '2026-08-18'
    assert ta.jornada_de(_epoca_local(2026, 8, 19, 0, 1)) == '2026-08-19'


def test_el_corte_de_jornada_se_puede_mover():
    """Con corte a las 22:00, la madrugada pertenece al día que empezó.

    Es el caso de la tienda que cierra tarde: el cliente de las 00:30 del
    martes cuenta en la jornada del lunes, no en una jornada nueva de 30 min.
    """
    noche = _epoca_local(2026, 8, 18, 23, 59)
    madrugada = _epoca_local(2026, 8, 19, 0, 30)
    antes_del_corte = _epoca_local(2026, 8, 18, 21, 59)

    assert ta.jornada_de(noche, corte_h=22) == '2026-08-18'
    assert ta.jornada_de(madrugada, corte_h=22) == '2026-08-18'
    # Lo anterior al corte todavía es del día previo.
    assert ta.jornada_de(antes_del_corte, corte_h=22) == '2026-08-17'


def test_a_medianoche_el_marcador_de_hoy_empieza_de_cero(tmp_path, reloj):
    """Lo anotado a las 23:59:59 es de AYER aunque se lea (o se escriba) a
    las 00:00:01.

    Es la frontera que el resto de pruebas evita a propósito (reloj a
    mediodía), documentada aquí explícita porque es comportamiento
    DELIBERADO: la jornada se sella con el instante del HECHO, no con el del
    volcado ni con el de la lectura. Al cruzar la medianoche «hoy»
    (`contadores_de()` sin jornada) arranca en cero, y lo de la víspera sigue
    ahí pidiéndolo por jornada. Es lo que hace que el dashboard no arrastre
    el día anterior a las 00:00 y que una reconexión reanude SU jornada.

    El segundo hecho se deja en memoria (periodo largo) y se escribe ya
    pasada la medianoche: si la jornada se calculase al volcar, caería en
    el día 19 y este test lo vería.
    """
    alm = ta.AlmacenTienda(ruta=tmp_path / 'tienda.db',
                           periodo_volcado_s=10_000.0)
    reloj(_epoca_local(2026, 8, 18, 23, 59) + 59.0)       # 23:59:59
    alm.anotar_entrada('cam-1', 'p1')      # el primero se escribe al momento
    alm.anotar_entrada('cam-1', 'p2')      # el segundo espera en memoria
    assert alm._pendientes, 'p2 sigue sin escribir cuando cambia el día'

    reloj(_epoca_local(2026, 8, 19, 0, 0) + 1.0)           # 00:00:01
    assert alm.contadores_de('cam-1')['entradas'] == 0, 'hoy empieza en cero'
    de_ayer = alm.contadores_de('cam-1', jornada=HOY)
    assert de_ayer['entradas'] == 2
    assert set(de_ayer['personas']) == {'p1', 'p2'}
    assert alm.resumen()['entradas'] == 0
    assert alm.resumen(HOY, HOY)['entradas'] == 2


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason='BUG de producción pendiente (4-sep-2026): tienda_almacen.py:353 '
           'llavea la demografía por jornada_de(ts) de SU propio instante y '
           ':583 solo cuenta filas con entradas > 0; la que converge pasada '
           'la medianoche cae en una fila nueva con entradas=0 y la persona '
           'no cuenta en ningún día')
def test_la_demografia_que_converge_pasada_la_medianoche_no_se_pierde(
        tmp_path, reloj):
    """Quien entra a las 23:59:59 y cuya cara converge a las 00:00:01 tiene
    que aparecer en la demografía de ESA visita.

    Es el caso real, no uno rebuscado: `tiendas_fotos.py:555` llama a
    `anotar_demografia` SIN `ts` cuando género y edad CONVERGEN, que es
    siempre después de la entrada. Hoy `volcar` sella la op 'demografia' con
    `jornada_de(ts)` de su propio instante (tienda_almacen.py:353),
    `_upsert_demografia` (:389) crea entonces una fila NUEVA en la jornada
    siguiente con `entradas = 0`, y `demografia()` (:583) exige
    `entradas > 0`: quedan dos filas — `(18, p1, entradas=1, sin género)` y
    `(19, p1, entradas=0, Mujer)` — y la persona no cuenta en el 18, ni en
    el 19, ni en el rango que cubre ambos. Su visita se queda sin
    género/edad para siempre.

    Se deja como `xfail` ESTRICTO, no como prueba verde a medias: cuando
    producción ate la demografía a la visita abierta de esa (cámara,
    persona), esta prueba pasará y el `strict` obligará a quitar la marca.
    Lo que se fija es el contrato que promete `demografia()` («quien tiene
    demografía consolidada y pisó dentro al menos una vez»), sin prejuzgar
    en qué fila se guarda.

    `raises=AssertionError` a propósito (4-sep-2026): el «fallo esperado» es
    SOLO la aserción del contrato. Sin él, un `demografia()` que reventase
    con `TypeError` también contaría como esperado y la marca taparía la
    regresión (demostrado con un mutante); con él, esa excepción sale como
    fallo normal.
    """
    alm = ta.AlmacenTienda(ruta=tmp_path / 'tienda.db', periodo_volcado_s=0.0)
    reloj(_epoca_local(2026, 8, 18, 23, 59) + 59.0)       # 23:59:59
    alm.anotar_entrada('cam-1', 'p1')

    reloj(_epoca_local(2026, 8, 19, 0, 0) + 1.0)           # 00:00:01
    # Sin `ts`, como hace producción al converger la cara.
    alm.anotar_demografia('cam-1', 'p1', 'Mujer', '26-35')
    alm.volcar(force=True)

    demo = alm.demografia(HOY, MANANA)
    assert demo['genero'] == {'Mujer': 1}, 'p1 pisó dentro y tiene demografía'
    assert demo['edad'] == {'26-35': 1}


# ── Conteo ───────────────────────────────────────────────────────────────

def test_una_entrada_se_cuenta_una_vez(almacen):
    almacen.anotar_entrada('cam-1', 'p1')
    c = almacen.contadores_de('cam-1')
    assert c['entradas'] == 1
    assert c['salidas'] == 0
    assert c['unicos'] == 1


def test_quien_vuelve_a_entrar_sigue_siendo_UN_unico(almacen):
    """Tres visitas de la misma persona son 3 entradas y 1 visitante."""
    for _ in range(3):
        almacen.anotar_entrada('cam-1', 'p1')
        almacen.anotar_salida('cam-1', 'p1', permanencia_s=12.0)
    c = almacen.contadores_de('cam-1')
    assert c['entradas'] == 3
    assert c['salidas'] == 3
    assert c['unicos'] == 1
    assert c['personas']['p1']['permanencia_s'] == pytest.approx(36.0)


def test_dos_camaras_no_se_cruzan(almacen):
    """El pasillo 1 no puede heredar los visitantes del pasillo 2.

    Es la lección de H-39: cuatro ventanas «iVMS-4200» son cuatro cámaras, y
    lo que las distingue es el `camera_id`, no el nombre que se ve.
    """
    almacen.anotar_entrada('cam-1', 'p1')
    almacen.anotar_entrada('cam-1', 'p2')
    almacen.anotar_entrada('cam-2', 'p3')

    uno, dos = almacen.contadores_de('cam-1'), almacen.contadores_de('cam-2')
    assert uno['entradas'] == 2 and uno['unicos'] == 2
    assert dos['entradas'] == 1 and dos['unicos'] == 1
    assert set(dos['personas']) == {'p3'}


# ── Sobrevivir a cortes y reinicios ──────────────────────────────────────

def test_el_reinicio_a_mitad_de_jornada_no_pierde_lo_contado(tmp_path, reloj):
    """Matar el proceso y volver: el marcador sigue donde estaba."""
    ruta = tmp_path / 'tienda.db'
    sesion1 = ta.AlmacenTienda(ruta=ruta, periodo_volcado_s=0.0)
    sesion1.anotar_entrada('cam-1', 'p1')
    sesion1.anotar_entrada('cam-1', 'p2')
    sesion1.volcar(force=True)

    sesion2 = ta.AlmacenTienda(ruta=ruta, periodo_volcado_s=0.0)
    c = sesion2.contadores_de('cam-1')
    assert c['entradas'] == 2
    assert c['unicos'] == 2


def test_dos_sesiones_del_mismo_dia_suman(tmp_path, reloj):
    """El caso real: la red parpadea, el procesador renace, y el día sigue.

    Y la persona que estaba antes del corte y vuelve después NO se cuenta dos
    veces como visitante única.
    """
    ruta = tmp_path / 'tienda.db'
    antes = ta.AlmacenTienda(ruta=ruta, periodo_volcado_s=0.0)
    antes.anotar_entrada('cam-1', 'p1')
    antes.anotar_salida('cam-1', 'p1', permanencia_s=30.0)
    antes.volcar(force=True)

    despues = ta.AlmacenTienda(ruta=ruta, periodo_volcado_s=0.0)
    despues.anotar_entrada('cam-1', 'p1')      # la misma persona, otra vez
    despues.anotar_entrada('cam-1', 'p9')      # y alguien nuevo

    c = despues.contadores_de('cam-1')
    assert c['entradas'] == 3
    assert c['unicos'] == 2, 'p1 vuelve, no se duplica'


# ── Ventanas de fechas ───────────────────────────────────────────────────

def test_los_unicos_de_un_rango_no_son_la_suma_de_los_diarios(almacen):
    """Semántica escrita: «únicos» = personas DISTINTAS dentro de la ventana.

    p1 viene lunes y martes: en el rango es UNA persona, aunque sumando los
    únicos de cada día darían dos. (Lunes 17 y martes 18 de agosto de 2026,
    a mediodía: con el corte a las 0 h que fija `almacen`, cada semilla es
    de su día natural.)
    """
    lunes = _epoca_local(2026, 8, 17, 12, 0)
    martes = _epoca_local(2026, 8, 18, 12, 0)
    almacen.anotar_entrada('cam-1', 'p1', ts=lunes)
    almacen.anotar_entrada('cam-1', 'p1', ts=martes)
    almacen.anotar_entrada('cam-1', 'p2', ts=martes)

    rango = almacen.resumen(desde=AYER, hasta=HOY)
    assert rango['entradas'] == 3
    assert rango['visitantes_unicos'] == 2
    assert rango['jornadas'] == 2

    solo_lunes = almacen.resumen(desde=AYER, hasta=AYER)
    assert solo_lunes['entradas'] == 1
    assert solo_lunes['visitantes_unicos'] == 1


def test_el_resumen_se_puede_acotar_a_una_camara(almacen):
    almacen.anotar_entrada('cam-1', 'p1')
    almacen.anotar_entrada('cam-2', 'p2')
    assert almacen.resumen(HOY, HOY, camera_id='cam-1')['entradas'] == 1
    assert almacen.resumen(HOY, HOY)['entradas'] == 2


def test_la_demografia_queda_atada_a_la_camara_y_al_dia(tmp_path, reloj):
    """Sin esto no hay desglose por pasillo, que es el punto #4 del mapa.

    Se mira la fila cruda: la entrada y la demografía tienen que caer en la
    MISMA fila (jornada, cámara, persona), no en dos.

    La demografía se deja en memoria (periodo largo) y se ESCRIBE ya pasada
    la medianoche (4-sep-2026): así la jornada de la fila solo puede salir
    del instante del hecho. Con el reloj quieto en ambas llamadas, esta
    prueba pasaba igual si la jornada de la demografía se calculase al
    volcar — y entonces no probaba «al día», solo «a la cámara».
    """
    alm = ta.AlmacenTienda(ruta=tmp_path / 'tienda.db',
                           periodo_volcado_s=10_000.0)
    alm.anotar_entrada('cam-1', 'p1')      # el primero se escribe al momento
    alm.anotar_demografia('cam-1', 'p1', 'Mujer', '26-35')   # este espera
    assert alm._pendientes, 'la demografía sigue en memoria al cambiar el día'

    reloj(_epoca_local(2026, 8, 19, 0, 0) + 1.0)           # 00:00:01
    alm.volcar(force=True)

    with sqlite3.connect(str(tmp_path / 'tienda.db')) as con:
        filas = con.execute(
            'SELECT jornada, camera_id, genero, edad FROM tienda_visitas '
            'WHERE persistent_id = ?', ('p1',)).fetchall()
    assert filas == [(HOY, 'cam-1', 'Mujer', '26-35')]


def test_una_demografia_vacia_no_pisa_la_que_ya_habia(almacen, tmp_path):
    almacen.anotar_demografia('cam-1', 'p1', 'Hombre', '36-50')
    almacen.anotar_demografia('cam-1', 'p1', '', None)
    almacen.volcar(force=True)

    with sqlite3.connect(str(tmp_path / 'tienda.db')) as con:
        fila = con.execute(
            'SELECT genero, edad FROM tienda_visitas '
            'WHERE persistent_id = ?', ('p1',)).fetchone()
    assert fila == ('Hombre', '36-50')


# ── Robustez ─────────────────────────────────────────────────────────────

def test_un_fallo_de_disco_no_tumba_el_frame(almacen, monkeypatch):
    """El pipeline sigue aunque la base no se pueda escribir.

    Se simula el disco caído: `sqlite3.connect` revienta en toda llamada. Ni
    anotar ni volcar ni leer pueden propagar la excepción al bucle de vídeo.
    """
    def _explota(*_a, **_k):
        raise sqlite3.OperationalError('disco lleno (simulado)')

    monkeypatch.setattr(ta.sqlite3, 'connect', _explota)

    almacen.anotar_entrada('cam-1', 'p1')      # no lanza
    almacen.volcar(force=True)                 # no lanza
    assert almacen.contadores_de('cam-1') == {
        'entradas': 0, 'salidas': 0, 'unicos': 0, 'personas': {}}
    assert almacen.fallos > 0, 'el fallo se cuenta, no se traga en silencio'


def test_la_cola_no_crece_sin_fin(tmp_path, monkeypatch):
    """Con el disco caído horas, la memoria del proceso está acotada."""
    alm = ta.AlmacenTienda(ruta=tmp_path / 'tienda.db',
                           periodo_volcado_s=10_000.0)   # nunca vuelca sola
    for i in range(20_050):
        alm.anotar_entrada('cam-1', f'p{i}')
    assert len(alm._pendientes) <= 20_000


def test_la_purga_respeta_la_retencion(almacen):
    """Con 396 días de retención, lo de hace 400 se va y lo de hoy se queda.

    `purgar()` se llama SIN argumento a propósito: es lo que hace el hilo de
    mantenimiento, y lo que se fija es que lee el mando del módulo
    (`RETENCION_DIAS`) y no un número escondido. Los 396 los pone
    `mandos_de_fabrica` (vía `almacen` -> `reloj`), NO el módulo: el mando
    sale de `ELDE_TIENDA_RETENCION_DIAS` al importar, y con la variable a
    500 en la máquina lo de hace 400 días es «reciente» y no se borraba nada
    (`assert 0 > 0`, demostrado el 4-sep-2026). Que 396 sea el valor de
    fábrica (los 13 meses documentados) no lo prueba esta función sino
    `test_los_mandos_de_fabrica_son_396_dias_y_corte_a_las_0`.
    """
    viejo = AHORA - 400 * 86400          # relativo al reloj congelado
    almacen.anotar_entrada('cam-1', 'antiguo', ts=viejo)
    almacen.anotar_entrada('cam-1', 'reciente')
    almacen.volcar(force=True)

    assert almacen.purgar() > 0
    hoy = almacen.contadores_de('cam-1')
    assert set(hoy['personas']) == {'reciente'}
    assert almacen.resumen(desde=ta.jornada_de(viejo),
                           hasta=ta.jornada_de(viejo))['entradas'] == 0


def test_purgar_con_cero_dias_no_borra_nada(almacen):
    """Retención 0 = «no purgues nunca», no «bórralo todo»."""
    almacen.anotar_entrada('cam-1', 'p1')
    almacen.volcar(force=True)
    assert almacen.purgar(retencion_dias=0) == 0
    assert almacen.contadores_de('cam-1')['entradas'] == 1


# ── Cableado del pipeline (sin cargar YOLO) ──────────────────────────────

@pytest.fixture(scope='module')
def tiendas():
    """`TiendasProcess`; si `tiendas.py` no importa (torch/ultralytics), se
    saltan las pruebas que lo piden.

    Fixture de MÓDULO y no función llamada dentro de cada prueba
    (4-sep-2026): la importación arrastra torch, tqdm y requests, que atan
    `time.time` al cargarse (`from time import time`,
    `preferred_clock = time.time`). Los fixtures de módulo se instancian
    ANTES que los de función, así que este import ocurre una sola vez y con
    nada parcheado, aunque la prueba pida también `reloj`. Es cinturón y
    tirantes: `reloj` ya no toca `time.time` global, pero quien vuelva a
    parchearlo algún día no debe poder contaminar tqdm por el orden de los
    tests.
    """
    try:
        from src.analityc.core.tiendas import TiendasProcess
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f'tiendas.py no importable en este entorno: {exc}')
    return TiendasProcess


class _Doble:
    """Lo mínimo que miran los dos métodos: la cámara en curso."""

    def __init__(self, camera_id='cam-1'):
        self._camera_id_actual = camera_id


def test_la_siembra_rehidrata_los_contadores_de_la_camara(tmp_path, reloj,
                                                          tiendas):
    """Al reconectar, la cámara reanuda su jornada en vez de empezar de cero."""
    TiendasProcess = tiendas
    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    alm.anotar_entrada('cam-1', 'p1')
    alm.anotar_salida('cam-1', 'p1', permanencia_s=20.0)
    alm.anotar_entrada('cam-1', 'p2')          # p2 se quedó dentro
    alm.volcar(force=True)

    estado = {}
    TiendasProcess._sembrar_desde_almacen(_Doble(), estado, 'cam-1')

    assert estado['_total_entries'] == 2
    assert estado['_total_exits'] == 1
    assert estado['_unique_entered'] == {'p1', 'p2'}
    # Quien tenía más entradas que salidas seguía dentro cuando se cortó.
    assert estado['_zone_state']['p2']['estado'] == 'DENTRO'
    assert estado['_zone_state']['p1']['estado'] == 'FUERA'
    # El tiempo ya acumulado se conserva; el tramo abierto no se inventa.
    assert estado['_zone_state']['p1']['tiempo_total_dentro'] == pytest.approx(20.0)


def test_la_siembra_de_una_camara_virgen_empieza_en_cero(tmp_path, tiendas):
    TiendasProcess = tiendas
    ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    estado = {}
    TiendasProcess._sembrar_desde_almacen(_Doble(), estado, 'cam-nueva')
    assert estado['_total_entries'] == 0
    assert estado['_unique_entered'] == set()
    assert estado['_zone_state'] == {}


def test_los_ids_provisionales_no_entran_en_la_memoria(tmp_path, reloj,
                                                      tiendas):
    """Un `prov:` es un track sin rostro: guardarlo inflaría los únicos.

    Una oclusión que parta el track da un id provisional nuevo, así que la
    misma persona aparecería como varias. Se cuenta de menos a propósito.
    """
    TiendasProcess = tiendas
    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')

    TiendasProcess._anotar_zona(_Doble(), ta.EVENTO_ENTRADA, 'prov:7', AHORA)
    TiendasProcess._anotar_zona(_Doble(), ta.EVENTO_ENTRADA, 'p1', AHORA)
    alm.volcar(force=True)

    c = alm.contadores_de('cam-1')
    assert set(c['personas']) == {'p1'}


def test_anotar_sin_camara_no_escribe_ni_revienta(tmp_path, reloj, tiendas):
    """Antes del primer frame no hay cámara en curso: no se inventa una."""
    TiendasProcess = tiendas
    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')

    TiendasProcess._anotar_zona(_Doble(camera_id=None), ta.EVENTO_ENTRADA,
                                'p1', AHORA)
    alm.volcar(force=True)
    assert alm.resumen()['entradas'] == 0


def test_al_desconectar_se_vuelca_lo_pendiente(tmp_path, reloj, tiendas):
    """Los últimos segundos de conteo no se van con el procesador.

    Es el momento en que MÁS falta hacen: la reconexión siguiente los lee para
    reanudar la jornada.
    """
    TiendasProcess = tiendas
    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db',
                                    periodo_volcado_s=10_000.0)
    # El PRIMER hecho se escribe al momento (ver `_ultimo_volcado = 0.0` en el
    # constructor): un corte a los dos segundos de conectar no se lo lleva.
    TiendasProcess._anotar_zona(_Doble(), ta.EVENTO_ENTRADA, 'p1', AHORA)
    assert not alm._pendientes

    # El segundo ya espera al periodo, que aquí no vence nunca.
    TiendasProcess._anotar_zona(_Doble(), ta.EVENTO_ENTRADA, 'p2', AHORA)
    assert alm._pendientes, 'en memoria: el periodo de volcado no ha vencido'

    TiendasProcess.cerrar_almacen_tienda(_Doble())

    assert not alm._pendientes
    assert alm.resumen()['entradas'] == 2


def test_el_cierre_de_conexion_llama_al_volcado():
    """`app.py` tiene que invocarlo: si no está en la lista, no se llama.

    La lista es un contrato flojo (busca métodos por nombre), así que esta
    prueba es la que impide que un renombrado lo deje mudo sin avisar.
    """
    from src.app.app import _METODOS_DE_CIERRE, _cerrar_procesadores

    assert 'cerrar_almacen_tienda' in _METODOS_DE_CIERRE

    llamadas = []

    class _ProcesadorFalso:
        def cerrar_almacen_tienda(self):
            llamadas.append('padre')

    class _PorCamara:
        def cerrar_almacen_tienda(self):
            llamadas.append('camara')

    padre = _ProcesadorFalso()
    padre._camera_processors = {'cam-1': _PorCamara()}
    _cerrar_procesadores(padre)

    assert llamadas == ['padre', 'camara']


# ── Fotos del legado: apagadas por defecto (inventario #17) ──────────────

def test_las_fotos_del_legado_vienen_apagadas(tmp_path, tiendas):
    """Cada entrada y salida de cada persona disparaba un JPEG anotado, un
    base64 y un POST HTTP dentro del bucle de inferencia, para una función
    (avisar del personal) que se eliminó el 27-jul-2026 y cuyo resultado no
    lee nadie."""
    TiendasProcess = tiendas
    from src.analityc.core.analytics.config import AnalyticsConfig
    assert AnalyticsConfig.LEGACY_STAFF_PHOTOS is False

    import numpy as np
    marca = {'llamadas': 0}

    class Doble:
        debug_mode = False
        image_quality = 70

        def create_annotated_image(self, *a, **k):
            marca['llamadas'] += 1
            return np.zeros((10, 10, 3), np.uint8)

    d = Doble()
    escrito = TiendasProcess.save_staff_photo(
        d, np.zeros((10, 10, 3), np.uint8), -1, 7, 'entrada')

    assert escrito is False
    assert marca['llamadas'] == 0, 'ni siquiera llega a anotar el frame'


def test_encendiendo_el_interruptor_el_legado_vuelve(monkeypatch, tiendas):
    """Apagado no es borrado: quien mirase esa carpeta a mano puede recuperarla.

    Se comprueba que la ejecución PASA del interruptor, sin dejarla llegar a
    escribir nada: el doble corta en el primer método que toca, que es lo que
    demuestra que el flag ya no la detiene.
    """
    TiendasProcess = tiendas
    from src.analityc.core.analytics.config import AnalyticsConfig
    monkeypatch.setattr(AnalyticsConfig, 'LEGACY_STAFF_PHOTOS', True)

    import numpy as np
    marca = {'paso': False}

    class Doble:
        debug_mode = False
        image_quality = 70

        def __init__(self):
            self._ultimo_evento_staff = {}

        def get_staff_display_name(self, _id):
            marca['paso'] = True
            raise RuntimeError('hasta aquí: la prueba no escribe en disco')

    TiendasProcess.save_staff_photo(
        Doble(), np.zeros((10, 10, 3), np.uint8), -1, 7, 'entrada')
    assert marca['paso'], 'con el interruptor puesto la ruta legado se ejecuta'
