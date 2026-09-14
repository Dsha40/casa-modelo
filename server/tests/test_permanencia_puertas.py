"""
Pruebas del reparto por PUERTAS (`permanencia_puertas.py`, 7-sep-2026).

Lo que se protege aquí no se ve mirando la pantalla: con el vídeo perfecto y
sin un solo error en el log, un reparto mal hecho cuenta gente dentro de un
sitio por el que no pasó. Cada prueba fija uno de esos fallos:

  - **Compatibilidad sagrada**: sin nada dibujado, `poner([])` no crea el
    fichero, no cierra nada y no toca a nadie. Es lo que manda el cliente en
    CADA frame de casi todas las cámaras.
  - **Quitar no corre los índices** (el fallo de usar una lista): borrar la
    puerta de en medio dejaría a la gente de la siguiente contada en la
    anterior.
  - **Renombrar y mover no reinician el aforo**: la identidad es el ÍNDICE.
  - **Una puerta borrada no resucita del disco** (el fallo de un fichero por
    puerta): la tercera puerta nueva heredaría a los de la tercera vieja.
  - **Un reinicio no vacía el almacén**, y el libro v1 (una sola puerta) se
    adopta entero como la puerta 0.
  - **La escritura es atómica y amortiguada**: el fichero lleva las fotos de
    entrada en base64; escribirlo en cada frame cuesta más que la inferencia,
    y escribirlo encima deja un JSON a medias si el proceso cae.
  - **A10**: el gestor NO puede entrar en la lista que `TiendasProcess`
    multiplexa por cámara (prueba de FUENTE; ver su docstring).

El reloj NO se congela con parches: todas las llamadas del módulo aceptan
`ahora`, así que las pruebas pasan un instante explícito y las duraciones y
los amortiguadores son exactos sin depender de lo que tarde la máquina
(lección de la limpieza del 4-sep-2026).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import permanencia_puertas as pp  # noqa: E402
from src.analityc.core.permanencia_estancias import (  # noqa: E402
    SIN_ATRIBUIR, Estancia)
from src.analityc.core.permanencia_puerta import ENTRADA, SALIDA  # noqa: E402
from src.analityc.core.permanencia_puertas import (  # noqa: E402
    ATRIBUTO_ACTIVATE, ATRIBUTO_GESTOR, ATRIBUTO_LINEAS, ATRIBUTO_NOMBRES,
    MAX_PUERTAS, VERSION_LIBRO, GestorPuertas, LibroPuertas, Puerta,
    clave_dedup, ruta_libro)

#: Instante base de todas las pruebas. Se toma UNA vez y se pasa explícito.
AHORA = time.time()

#: Dos umbrales horizontales, como los del anexo. Caminando de P1 a P2
#: (izquierda -> derecha) DENTRO queda a la DERECHA = abajo en la imagen, así
#: que bajar por el frame es ENTRAR.
LINEA_A = [[192.0, 324.0], [1728.0, 324.0]]
LINEA_B = [[192.0, 756.0], [1728.0, 756.0]]


# ── Utilidades ──────────────────────────────────────────────────────────
def _gestor(tmp_path: Path, camara: str = 'cam1', **cfg) -> GestorPuertas:
    base = {'estancia_ttl_horas': 24, 'persistir_estancias': True}
    base.update(cfg)
    return GestorPuertas(camara=camara,
                         ruta=ruta_libro(tmp_path, camara), cfg=base)


def _meter(puerta: Puerta, pid: str, hace: float = 0.0) -> Estancia:
    """Alguien que entró por esa puerta hace `hace` segundos."""
    est = Estancia(persistent_id=pid, camara='cam1', zona='linea',
                   entrada_ts=AHORA - hace, ultima_vista_ts=AHORA - hace)
    puerta.sellar(est)
    puerta.registro.abrir(est)
    return est


def _dentro(puerta: Puerta) -> set:
    return set(puerta.registro.abiertas)


def _caja(y_pie: float, x: float = 900.0, alto: float = 200.0):
    """Caja de una persona cuyos pies pisan `y_pie`."""
    return (x - 50.0, y_pie - alto, x + 50.0, y_pie)


def _visibles(pid: str, y_pie: float):
    return {pid: {'box': _caja(y_pie), 'confirmado': True}}


def _leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding='utf-8'))


# ── Compatibilidad: sin puertas dibujadas no pasa NADA ──────────────────
def test_sin_puertas_dibujadas_no_se_crea_ni_el_fichero(tmp_path):
    """El caso de casi todas las cámaras: el cliente manda [] en cada frame.

    Un `sin-camara.json` vacío o un cierre de estancias aquí rompería el
    comportamiento de siempre en recuadros que ni saben que esto existe.
    """
    gestor = _gestor(tmp_path)
    for i in range(50):                       # 50 frames sin nada dibujado
        assert gestor.poner([], [], ahora=AHORA + i) == []
    assert len(gestor) == 0
    assert gestor.activo is False
    assert gestor.guardar(forzar=True, ahora=AHORA) is False
    assert not ruta_libro(tmp_path, 'cam1').exists()
    assert list(tmp_path.rglob('*.json')) == []


def test_la_dedup_clave_sin_puerta_es_EXACTAMENTE_la_de_siempre():
    """Sin puertas no cambia ni un carácter de lo que agrupan WhatsApp
    y Jarvis."""
    assert clave_dedup('cam1', 'entrada', 'p7') == 'perm|cam1|entrada|p7'
    assert clave_dedup('cam1', 'salida', 'p7', -1) == 'perm|cam1|salida|p7'


def test_la_dedup_clave_con_puerta_lleva_la_puerta_tras_el_tipo():
    """Dos personas entrando a la vez por puertas distintas NO comparten clave.

    Si la compartieran, la segunda moriría en el enfriamiento: justo la
    alerta que demuestra que son dos sitios. Y el segmento va DESPUÉS del
    tipo para no romper a quien filtra por `perm|cam|tipo|`.
    """
    a = clave_dedup('cam1', 'entrada', 'p7', 0)
    b = clave_dedup('cam1', 'entrada', 'p8', 1)
    assert a == 'perm|cam1|entrada|p0|p7'
    assert b == 'perm|cam1|entrada|p1|p8'
    assert a != b
    assert a.startswith('perm|cam1|entrada|')


# ── Alta, nombres y geometría ───────────────────────────────────────────
def test_poner_crea_una_puerta_por_umbral_con_su_nombre(tmp_path):
    gestor = _gestor(tmp_path)
    assert gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'],
                        ahora=AHORA) == []
    assert sorted(gestor.puertas) == [0, 1]
    assert [p.nombre for p in gestor] == ['Almacen', 'Patio']
    assert gestor.get(0).linea == LINEA_A
    assert gestor.activo is True


def test_sin_nombre_la_puerta_se_llama_por_su_numero(tmp_path):
    """`puertas_nombres` NO se valida contra `puertas_lineas` (A3): un desfase
    es cosmético y convertirlo en frame descartado dejaría al operador sin
    vídeo."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen'], ahora=AHORA)
    assert [p.nombre for p in gestor] == ['Almacen', 'Puerta 2']
    gestor.poner([LINEA_A], ['Almacen', 'Patio', 'Sobra'], ahora=AHORA)
    assert [p.nombre for p in gestor] == ['Almacen']


def test_mas_de_ocho_umbrales_se_recortan_sin_perder_el_frame(tmp_path):
    gestor = _gestor(tmp_path)
    lineas = [[[10.0, 100.0 + i * 10], [1000.0, 100.0 + i * 10]]
              for i in range(12)]
    gestor.poner(lineas, [], ahora=AHORA)
    assert len(gestor) == MAX_PUERTAS == 8
    assert sorted(gestor.puertas) == list(range(8))


def test_un_umbral_ilegible_no_corre_los_indices_de_los_demas(tmp_path):
    """El segundo umbral es un punto repetido (largo 0): se salta ESE índice.

    Colapsar la lista para «no dejar huecos» correría las identidades, que es
    exactamente el fallo que el dict existe para evitar.
    """
    gestor = _gestor(tmp_path)
    degenerada = [[500.0, 500.0], [500.0, 500.0]]
    gestor.poner([LINEA_A, degenerada, LINEA_B], ['A', 'B', 'C'], ahora=AHORA)
    assert sorted(gestor.puertas) == [0, 2]
    assert gestor.get(2).nombre == 'C'
    assert gestor.get(2).linea == LINEA_B


# ── Un detector y un libro POR PUERTA ───────────────────────────────────
def test_cada_puerta_tiene_su_propio_detector_y_su_propio_libro(tmp_path):
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    p0, p1 = gestor.get(0), gestor.get(1)
    assert p0.detector is not p1.detector
    assert p0.registro is not p1.registro
    assert p0.entradas_en_espera is not p1.entradas_en_espera


def test_cruzar_los_dos_umbrales_da_una_entrada_en_cada_puerta(tmp_path):
    """El detector NO se puede compartir: `_Paso.lado_firme` está llaveado
    solo por la persona y dos puertas se pisarían el lado en el mismo frame.

    Aquí la MISMA persona baja por el frame y cruza los dos umbrales: cada
    puerta tiene que ver SU entrada, con su propio estado.
    """
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    vistos = []
    # Cuatro frames de vida arriba del todo (frames_minimos) y luego baja.
    for i, y in enumerate([200, 210, 220, 230, 500, 900]):
        ts = AHORA + i * 0.5
        for puerta, cruce in gestor.observar(ts, _visibles('p1', y)):
            vistos.append((puerta.indice, cruce.tipo))
    assert vistos == [(0, ENTRADA), (1, ENTRADA)]


def test_entrar_por_una_y_salir_por_otra_no_es_un_fallo(tmp_path):
    """A7: quien entra por «Almacén» y sale por «Patio» deja una visita
    ABIERTA en el primero y una salida sin pareja en el segundo. Son dos
    sitios: fundirlas sería inventar una visita a un sitio por el que esa
    persona no pasó."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    vistos = []
    for i, y in enumerate([200, 210, 220, 230, 500, 900, 500]):
        ts = AHORA + i * 0.5
        for puerta, cruce in gestor.observar(ts, _visibles('p1', y)):
            vistos.append((puerta.indice, cruce.tipo))
    assert vistos == [(0, ENTRADA), (1, ENTRADA), (1, SALIDA)]


# ── Quitar puertas ──────────────────────────────────────────────────────
def test_quitar_la_ultima_no_corre_los_indices(tmp_path):
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B, LINEA_A], ['A', 'B', 'C'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    _meter(gestor.get(1), 'beto')
    gestor.poner([LINEA_A, LINEA_B], ['A', 'B'], ahora=AHORA + 1)
    assert sorted(gestor.puertas) == [0, 1]
    assert gestor.get(0).nombre == 'A' and _dentro(gestor.get(0)) == {'ana'}
    assert gestor.get(1).nombre == 'B' and _dentro(gestor.get(1)) == {'beto'}


def test_quitar_la_de_EN_MEDIO_no_mueve_a_la_gente_de_la_siguiente(tmp_path):
    """Con una LISTA, borrar la 1 de 3 corre los índices y la gente de la 2
    pasa a contarse en la 1: dos aforos falsos y ni un error en el log. Con
    `Dict[int, Puerta]` el índice 2 sigue siendo el 2."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B, LINEA_A], ['A', 'B', 'C'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    _meter(gestor.get(2), 'caro')
    huerfanas = gestor.poner([LINEA_A, None, LINEA_A], ['A', 'B', 'C'],
                             ahora=AHORA + 1)
    assert huerfanas == []                    # la 1 estaba vacía
    assert sorted(gestor.puertas) == [0, 2]
    assert gestor.get(2).nombre == 'C'
    assert _dentro(gestor.get(2)) == {'caro'}
    assert _dentro(gestor.get(0)) == {'ana'}


def test_quitar_una_puerta_con_gente_dentro_devuelve_sus_estancias(tmp_path):
    """Se devuelven ya CERRADAS como `sin_salida` para que el llamador avise.

    Dejarlas caer en silencio borraría del aforo a gente contada dentro de un
    sitio, sin una sola línea en ninguna parte.
    """
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    _meter(gestor.get(1), 'beto', hace=120)
    _meter(gestor.get(1), 'caro', hace=60)

    huerfanas = gestor.poner([LINEA_A], ['Almacen'], ahora=AHORA + 5)

    assert sorted(e.persistent_id for e in huerfanas) == ['beto', 'caro']
    for est in huerfanas:
        assert est.salida_ts == AHORA + 5
        assert est.identidad_por == SIN_ATRIBUIR
        assert est.puerta == 1               # dice de qué puerta venía
        assert est.puerta_nombre == 'Patio'
    assert sorted(gestor.puertas) == [0]
    assert _dentro(gestor.get(0)) == {'ana'}  # la que queda, intacta


def test_quitarlas_todas_cierra_todo_y_no_deja_a_nadie_dentro(tmp_path):
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['A', 'B'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    _meter(gestor.get(1), 'beto')
    huerfanas = gestor.poner([], [], ahora=AHORA + 2)
    assert sorted(e.persistent_id for e in huerfanas) == ['ana', 'beto']
    assert len(gestor) == 0
    assert gestor.total_dentro == 0


# ── Renombrar y mover ───────────────────────────────────────────────────
def test_renombrar_no_reinicia_el_aforo(tmp_path):
    """La identidad es el ÍNDICE (misma regla que `RegistroAnaqueles.area`).

    Si fuese el nombre, renombrar a media tarde bifurcaría el aforo: las
    visitas nuevas irían a una sección nueva y la gente de dentro quedaría en
    una sección inalcanzable.
    """
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A], ['Almacen'], ahora=AHORA)
    puerta = gestor.get(0)
    _meter(puerta, 'ana', hace=300)
    puerta.anotar_visita(120.0)
    puerta.anotar_visita(180.0)
    puerta.anotar_sin_atribuir()
    libro_antes, detector_antes = puerta.registro, puerta.detector

    gestor.poner([LINEA_A], ['Deposito'], ahora=AHORA + 60)

    misma = gestor.get(0)
    assert misma is puerta
    assert misma.registro is libro_antes and misma.detector is detector_antes
    assert misma.nombre == 'Deposito'
    assert _dentro(misma) == {'ana'}
    assert misma.registro.get('ana').entrada_ts == AHORA - 300  # sin reiniciar
    assert misma.total_visitas == 2
    assert misma.suma_duraciones == 300.0
    assert misma.salidas_sin_atribuir == 1
    # El nombre es solo para mostrar, pero el panel tiene que enseñar el nuevo.
    assert misma.registro.get('ana').puerta_nombre == 'Deposito'
    assert misma.registro.get('ana').puerta == 0


def test_mover_la_linea_no_reinicia_el_aforo(tmp_path):
    """El operador arrastra el umbral con el vídeo corriendo: mismo sitio."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A], ['Almacen'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    otra = [[100.0, 400.0], [1800.0, 410.0]]
    gestor.poner([otra], ['Almacen'], ahora=AHORA + 1)
    assert gestor.get(0).linea == otra
    assert _dentro(gestor.get(0)) == {'ana'}


# ── Disco: reinicio, no-resurrección, v1 ────────────────────────────────
def test_un_reinicio_no_vacia_el_almacen(tmp_path):
    """Cada quien vuelve a SU sección, con su puerta sellada."""
    uno = _gestor(tmp_path)
    uno.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    _meter(uno.get(0), 'ana', hace=600)
    _meter(uno.get(1), 'beto', hace=30)
    assert uno.guardar(forzar=True, ahora=AHORA) is True

    otro = _gestor(tmp_path)                  # «reinicio del servidor»
    assert otro.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'],
                      ahora=AHORA + 1) == []
    assert _dentro(otro.get(0)) == {'ana'}
    assert _dentro(otro.get(1)) == {'beto'}
    assert otro.get(0).registro.get('ana').entrada_ts == AHORA - 600
    assert otro.get(1).registro.get('beto').puerta == 1
    assert otro.total_dentro == 2


def test_una_puerta_borrada_NO_resucita_del_disco(tmp_path):
    """El fallo de `<cam>__p3.json`: borrar la 3 dejaba un huérfano que
    revivía en cuanto el operador dibujaba otra tercera puerta, con otra
    geometría y en otro sitio."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B, LINEA_A], ['A', 'B', 'C'], ahora=AHORA)
    _meter(gestor.get(2), 'caro')
    gestor.guardar(forzar=True, ahora=AHORA)

    huerfanas = gestor.poner([LINEA_A, LINEA_B], ['A', 'B'], ahora=AHORA + 1)
    assert [e.persistent_id for e in huerfanas] == ['caro']

    # a) en disco ya no queda ni la sección
    en_disco = _leer(ruta_libro(tmp_path, 'cam1'))
    assert sorted(en_disco['puertas']) == ['0', '1']

    # b) otra tercera puerta, con OTRA geometría, nace vacía
    gestor.poner([LINEA_A, LINEA_B, [[0.0, 20.0], [50.0, 900.0]]],
                 ['A', 'B', 'Nueva'], ahora=AHORA + 2)
    assert _dentro(gestor.get(2)) == set()
    assert gestor.get(2).nombre == 'Nueva'

    # c) y tampoco resucita tras un reinicio
    otro = _gestor(tmp_path)
    otro.poner([LINEA_A, LINEA_B, LINEA_A], ['A', 'B', 'Nueva'],
               ahora=AHORA + 3)
    assert _dentro(otro.get(2)) == set()


def test_una_seccion_sin_puerta_se_cierra_como_sin_salida(tmp_path):
    """Arranca el servidor y el operador ya solo tiene dos puertas: la gente
    de la tercera no se deja caer en silencio."""
    uno = _gestor(tmp_path)
    uno.poner([LINEA_A, LINEA_B, LINEA_A], ['A', 'B', 'C'], ahora=AHORA)
    _meter(uno.get(2), 'caro', hace=90)
    uno.guardar(forzar=True, ahora=AHORA)

    otro = _gestor(tmp_path)
    huerfanas = otro.poner([LINEA_A, LINEA_B], ['A', 'B'], ahora=AHORA + 1)
    assert [e.persistent_id for e in huerfanas] == ['caro']
    assert huerfanas[0].salida_ts == AHORA + 1
    # Y con la identidad marcada, igual que la que sale por `_quitar`:
    # son la MISMA huérfana por dos caminos y el registro no puede
    # decir de ella dos cosas distintas según por dónde llegó.
    assert huerfanas[0].identidad_por == SIN_ATRIBUIR
    assert sorted(otro.puertas) == [0, 1]


def test_las_secciones_esperan_mientras_el_recuadro_venga_vacio(tmp_path):
    """Un frame sin puertas ANTES de saber cómo está el recuadro no puede
    vaciar el libro: el cliente todavía no las mandó. Solo se resuelve
    cuando llega una lista con puertas (o cuando ya las hubo)."""
    uno = _gestor(tmp_path)
    uno.poner([LINEA_A], ['Almacen'], ahora=AHORA)
    _meter(uno.get(0), 'ana')
    uno.guardar(forzar=True, ahora=AHORA)

    otro = _gestor(tmp_path)
    assert otro.poner([], [], ahora=AHORA + 1) == []      # no-op, no cierra
    assert otro.poner([LINEA_A], ['Almacen'], ahora=AHORA + 2) == []
    assert _dentro(otro.get(0)) == {'ana'}


def test_el_fichero_v1_se_adopta_entero_como_la_puerta_0(tmp_path):
    """Es lo que era: la única puerta de esa cámara."""
    ruta = ruta_libro(tmp_path, 'cam1')
    ruta.parent.mkdir(parents=True, exist_ok=True)
    est = Estancia(persistent_id='ana', camara='cam1', zona='linea',
                   entrada_ts=AHORA - 200, ultima_vista_ts=AHORA - 200)
    ruta.write_text(json.dumps({'version': 1, 'guardado_ts': AHORA - 100,
                                'abiertas': [est.a_dict()]}),
                    encoding='utf-8')

    gestor = _gestor(tmp_path)
    assert gestor.libro.version_leida == 1
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    assert _dentro(gestor.get(0)) == {'ana'}
    assert _dentro(gestor.get(1)) == set()
    # Se sella como puerta 0: si siguiera con -1 (el aforo único de v1), al
    # releer el libro su aforo saldría vacío con el vídeo perfecto.
    assert gestor.get(0).registro.get('ana').puerta == 0

    gestor.guardar(forzar=True, ahora=AHORA)
    assert _leer(ruta)['version'] == VERSION_LIBRO == 2


def test_una_seccion_fuera_de_rango_se_ignora_sin_reventar(tmp_path):
    """Ocho puertas es el tope del contrato; una sección 9 en el fichero solo
    puede venir de un libro tocado a mano o de otra versión. Se ignora y el
    resto del libro se lee entero: descartarlo todo dejaría a las puertas
    buenas sin su gente."""
    ruta = ruta_libro(tmp_path, 'cam1')
    ruta.parent.mkdir(parents=True, exist_ok=True)
    est = Estancia(persistent_id='ana', camara='cam1', zona='linea',
                   entrada_ts=AHORA - 10, ultima_vista_ts=AHORA - 10)
    ruta.write_text(json.dumps({
        'version': 2, 'guardado_ts': AHORA,
        'puertas': {'0': {'nombre': 'Almacen', 'abiertas': [est.a_dict()]},
                    '9': {'nombre': 'Fantasma', 'abiertas': [est.a_dict()]},
                    'x': {'nombre': 'Basura', 'abiertas': []}}}),
        encoding='utf-8')
    libro = LibroPuertas(ruta=ruta, ttl_horas=24)
    secciones = libro.cargar(ahora=AHORA)
    assert sorted(secciones) == [0]
    assert [e.persistent_id for e in secciones[0]['abiertas']] == ['ana']


def test_un_libro_ilegible_no_tumba_el_arranque(tmp_path):
    ruta = ruta_libro(tmp_path, 'cam1')
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text('{"version": 2, "puer', encoding='utf-8')  # JSON a medias
    gestor = _gestor(tmp_path)
    assert gestor.poner([LINEA_A], ['Almacen'], ahora=AHORA) == []
    assert _dentro(gestor.get(0)) == set()


def test_una_estancia_pasada_de_TTL_no_resucita_al_cargar(tmp_path):
    uno = _gestor(tmp_path, estancia_ttl_horas=1)
    uno.poner([LINEA_A], ['Almacen'], ahora=AHORA)
    _meter(uno.get(0), 'ana', hace=7200)      # 2 h con TTL de 1 h
    _meter(uno.get(0), 'beto', hace=60)
    uno.guardar(forzar=True, ahora=AHORA)

    otro = _gestor(tmp_path, estancia_ttl_horas=1)
    otro.poner([LINEA_A], ['Almacen'], ahora=AHORA + 1)
    assert _dentro(otro.get(0)) == {'beto'}


# ── Escritura: atómica y amortiguada ────────────────────────────────────
class _OsEspia:
    """`os` con `replace` vigilado. Se cuelga del MÓDULO probado y no del
    intérprete: parchear `os.replace` global alcanzaría a cualquier otra
    prueba de la sesión (y a `tempfile`)."""

    def __init__(self, real):
        self._real = real
        self.movimientos = []

    def replace(self, origen, destino):
        self.movimientos.append((str(origen), str(destino)))
        return self._real.replace(origen, destino)

    def __getattr__(self, nombre):
        return getattr(self._real, nombre)


def test_la_escritura_es_atomica(tmp_path, monkeypatch):
    """Temporal + `os.replace`. Escribir encima del fichero bueno deja un
    JSON a medias si el proceso cae, y ese fichero es el aforo de TODAS las
    puertas de la cámara."""
    espia = _OsEspia(os)
    monkeypatch.setattr(pp, 'os', espia)
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    assert gestor.guardar(forzar=True, ahora=AHORA) is True

    ruta = ruta_libro(tmp_path, 'cam1')
    assert espia.movimientos, "el volcado no pasó por un reemplazo atómico"
    origen, destino = espia.movimientos[-1]
    assert destino == str(ruta)
    assert origen != str(ruta) and origen.endswith('.tmp')
    assert not list(ruta.parent.glob('*.tmp'))   # sin restos
    datos = _leer(ruta)                          # y parsea entero
    assert [e['persistent_id'] for e in datos['puertas']['0']['abiertas']] \
        == ['ana']
    assert datos['puertas']['1']['abiertas'] == []
    assert datos['puertas']['1']['nombre'] == 'Patio'


def test_la_escritura_esta_amortiguada_a_una_cada_cinco_segundos(tmp_path):
    """El fichero lleva las fotos de entrada en base64: escribirlo en cada
    frame cuesta más que la inferencia."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A], ['Almacen'], ahora=AHORA)   # alta = escritura ya
    ruta = ruta_libro(tmp_path, 'cam1')
    assert ruta.is_file()

    _meter(gestor.get(0), 'ana')
    assert gestor.guardar(ahora=AHORA + 1.0) is False    # aún amortiguado
    assert _leer(ruta)['puertas']['0']['abiertas'] == []

    assert gestor.guardar(ahora=AHORA + 6.0) is True
    assert [e['persistent_id']
            for e in _leer(ruta)['puertas']['0']['abiertas']] == ['ana']

    # Sin cambios no se escribe aunque haya pasado el rato.
    assert gestor.guardar(ahora=AHORA + 100.0) is False


def test_quitar_una_puerta_se_escribe_al_instante(tmp_path):
    """El amortiguador está para no escribir en cada frame, no para dejar en
    disco durante 5 s una puerta que el operador acaba de borrar."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['A', 'B'], ahora=AHORA)
    gestor.poner([LINEA_A], ['A'], ahora=AHORA + 0.2)    # dentro del margen
    assert sorted(_leer(ruta_libro(tmp_path, 'cam1'))['puertas']) == ['0']


def test_sin_ruta_todo_funciona_en_memoria(tmp_path):
    """`persistir_estancias: False` (o cámara aún sin nombre) no puede
    cambiar ni una decisión del reparto."""
    gestor = GestorPuertas(camara='cam1', ruta=None, cfg={})
    gestor.poner([LINEA_A, LINEA_B], ['A', 'B'], ahora=AHORA)
    _meter(gestor.get(1), 'ana')
    assert gestor.guardar(forzar=True, ahora=AHORA) is False
    huerfanas = gestor.poner([LINEA_A], ['A'], ahora=AHORA + 1)
    assert [e.persistent_id for e in huerfanas] == ['ana']
    assert list(tmp_path.iterdir()) == []


# ── Lecturas para el panel y el TTL ─────────────────────────────────────
def test_el_resumen_lleva_una_fila_por_puerta_con_sus_numeros(tmp_path):
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    _meter(gestor.get(0), 'ana')
    _meter(gestor.get(0), 'beto')
    gestor.get(0).anotar_visita(100.0)
    gestor.get(0).anotar_visita(186.4)
    gestor.get(1).anotar_sin_atribuir()

    filas = gestor.resumen({0: [{'pid': 'ana'}, {'pid': 'beto'}]})
    assert [f['idx'] for f in filas] == [0, 1]
    assert filas[0] == {'idx': 0, 'nombre': 'Almacen', 'dentro_ahora': 2,
                        'personas': [{'pid': 'ana'}, {'pid': 'beto'}],
                        'visitas_completadas': 2,
                        'permanencia_media_s': 143.2,
                        'salidas_sin_atribuir': 0}
    assert filas[1]['dentro_ahora'] == 0
    assert filas[1]['permanencia_media_s'] == 0.0
    assert filas[1]['salidas_sin_atribuir'] == 1
    assert gestor.total_dentro == 2


def test_la_misma_persona_puede_estar_dentro_de_dos_puertas(tmp_path):
    """Entró por el almacén y por el patio sin que se le viera salir: son dos
    sitios y cada uno tiene su cronómetro. Por eso la llave de la fila del
    panel incluye la puerta — con solo la persona, las dos filas se pelearían
    el mismo widget y el cronómetro alternaría."""
    gestor = _gestor(tmp_path)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    _meter(gestor.get(0), 'ana', hace=300)
    _meter(gestor.get(1), 'ana', hace=30)
    encontradas = gestor.buscar_abierta('ana')
    assert [p.indice for p, _ in encontradas] == [0, 1]
    assert gestor.total_dentro == 2
    assert len(gestor.abiertas()) == 2


def test_las_caducadas_llegan_con_su_puerta(tmp_path):
    gestor = _gestor(tmp_path, estancia_ttl_horas=1)
    gestor.poner([LINEA_A, LINEA_B], ['Almacen', 'Patio'], ahora=AHORA)
    _meter(gestor.get(1), 'ana', hace=7200)
    _meter(gestor.get(1), 'beto', hace=60)
    vencidas = gestor.caducadas(AHORA)
    assert [(p.indice, e.persistent_id) for p, e in vencidas] == [(1, 'ana')]
    assert _dentro(gestor.get(1)) == {'beto'}


# ── A10: la trampa del estado multiplexado (prueba de FUENTE) ───────────
FUENTE_TIENDAS = RAIZ / 'server' / 'src' / 'analityc' / 'core' / 'tiendas.py'


def test_el_gestor_de_puertas_NO_va_en_la_lista_que_se_multiplexa():
    """`TiendasProcess._push_state` cambia por cámara todo lo que nombre esa
    lista: `setattr(self, n, cam_state.get(n))`. Un atributo del modo puerta
    metido ahí se pondría a `None` en cada frame, `_actualizar` reventaría
    dentro de un `try` que SOLO loguea y el modo quedaría MUDO con el vídeo
    perfecto (el bug del 6 y el 7-ago-2026).

    No hace falta multiplexar nada: ya hay una instancia del procesador por
    cámara. Esta prueba es de TEXTO a propósito, y solo pregunta «¿está esta
    clave en esta ruta?» (lección H-47): que el valor sea correcto lo fijan
    las otras pruebas de este fichero.
    """
    texto = FUENTE_TIENDAS.read_text(encoding='utf-8')
    m = re.search(r'def _push_state\(.*?names\s*=\s*\[(.*?)\]', texto, re.S)
    assert m, ("no se encontró la lista `names` de `_push_state` en "
               "tiendas.py: si se renombró, hay que re-apuntar esta prueba")
    lista = m.group(1)
    for atributo in (ATRIBUTO_GESTOR, ATRIBUTO_LINEAS, ATRIBUTO_NOMBRES,
                     ATRIBUTO_ACTIVATE):
        assert atributo not in lista, (
            f"`{atributo}` está en la lista que `_push_state` multiplexa por "
            "cámara: quedaría en None en cada frame y el modo puertas se "
            "volvería mudo sin un solo error (bug del 6/7-ago-2026). Es un "
            "atributo PLANO del procesador, ya hay uno por cámara.")
