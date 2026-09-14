"""
Pruebas de `/api/v1/tienda`: la API que SÍ entiende de fechas (HITO 2).

Se ejercitan los endpoints de verdad con `TestClient` (recorren FastAPI,
validación de parámetros incluida) contra una base temporal poblada a mano.

Lo que más se protege:

  - **las tres vistas cuadran**: para la misma ventana, las entradas del
    resumen, la suma de la serie horaria y la suma del ranking dan lo mismo.
    Si un día dejan de cuadrar, alguien cambió una semántica sin darse cuenta;
  - **«únicos» de un rango son personas distintas**, no la suma de los únicos
    diarios (quien viene lunes y martes es UNA persona);
  - **sin base de comparación no se inventa un porcentaje**: la variación es
    `null`, y la página dirá «sin base», no «+100 %»;
  - **«sin datos» no es «cero»**: la cobertura dice qué jornadas se midieron;
  - **compatibilidad aditiva**: los endpoints viejos siguen respondiendo lo
    mismo, porque los leen otras tres páginas y el PDF.

## El reloj está CONGELADO (4-sep-2026)

Los endpoints sin parámetros responden por «hoy», y «hoy» lo deciden
preguntando `jornada_de()` en CADA petición. Antes este módulo calculaba
`HOY = ta.jornada_de()` al importarse (en la recolección de pytest) y
sembraba ahí: bastaba con que la medianoche cayera entre la recolección y la
prueba para que la API mirase una jornada vacía (8 pruebas en rojo arrancando
la suite a las 23:59:57). Ahora todas las pruebas viven en un instante fijo
(`AHORA`, fixture `reloj`) y los datos se siembran relativos a él. La frontera
de la medianoche, que en producción es deliberada, tiene su prueba explícita
al final del bloque «Resumen», en vez de depender de a qué hora corre la suite.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.app.api_tienda import router  # noqa: E402

# ── Reloj congelado ──────────────────────────────────────────────────────
#: Las jornadas en las que viven estas pruebas: martes 1-sep-2026 («hoy»),
#: su víspera y su día siguiente. Son literales, no `ta.jornada_de()`, para
#: que ni el reloj de la máquina ni `ELDE_TIENDA_CORTE_JORNADA_H` en el
#: entorno puedan mover lo que la suite entiende por «hoy».
AYER = '2026-08-31'
HOY = '2026-09-01'
MANANA = '2026-09-02'


def _epoca_de(jornada: str, hora: int) -> float:
    """Época del día `jornada` a las HH:30 en hora LOCAL.

    `mktime` con `-1` en el último campo, como el servidor real: la época es
    la de la hora local de la máquina que corre la suite, sea cual sea.
    """
    a, m, d = (int(x) for x in jornada.split('-'))
    return time.mktime((a, m, d, hora, 30, 0, 0, 0, -1))


#: El instante fijo: martes 1-sep-2026 a las 12:00 en hora local. A mediodía
#: la jornada es la misma con cualquier corte razonable y quedan entradas
#: sembradas antes (10:30) y después (17:30) de «ahora».
AHORA = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))


@pytest.fixture
def reloj(monkeypatch):
    """Congela `time.time()` en `AHORA` y fija la jornada al día natural.

    Devuelve una función para MOVER el reloj dentro de una prueba (las de
    la frontera de la jornada la usan). Un solo `time.time` parcheado basta:
    `jornada_de()` sin argumento, los `anotar_*` sin `ts` y el propio módulo
    de pruebas leen el mismo reloj, así que lo sembrado y lo preguntado
    coinciden por construcción.

    `CORTE_JORNADA_H` se fija a 0 porque `_epoca_de` siembra a las 10:30 y
    17:30 del día natural y las pruebas cuentan esas horas como «hoy»: con
    `ELDE_TIENDA_CORTE_JORNADA_H=22` exportada en la máquina (demostrado el
    4-sep-2026) esas horas pertenecen a la jornada ANTERIOR y caían 14
    pruebas. El contrato del corte vive en test_tienda_almacen.py; aquí solo
    se comprueba que la API se lo pregunta a `jornada_de` (ver la prueba del
    corte a las 22, más abajo).
    """
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)

    def mover(instante: float) -> None:
        monkeypatch.setattr(time, 'time', lambda: instante)

    mover(AHORA)
    assert ta.jornada_de() == HOY, 'el instante congelado no cae en HOY'
    return mover


@pytest.fixture
def cliente(tmp_path, reloj):
    """API montada sola, con una base recién hecha. Sin modelos ni GPU."""
    ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def poblado(cliente):
    """Ayer: 2 entradas. Hoy: 3 entradas de 2 personas, en dos horas."""
    alm = ta.get_almacen()
    alm.anotar_entrada('cam-1', 'p1', ts=_epoca_de(AYER, 10))
    alm.anotar_entrada('cam-1', 'p9', ts=_epoca_de(AYER, 11))

    alm.anotar_entrada('cam-1', 'p1', ts=_epoca_de(HOY, 10))
    alm.anotar_salida('cam-1', 'p1', permanencia_s=60.0, ts=_epoca_de(HOY, 10))
    alm.anotar_demografia('cam-1', 'p1', 'Mujer', '26-35', ts=_epoca_de(HOY, 10))
    alm.anotar_entrada('cam-1', 'p2', ts=_epoca_de(HOY, 17))
    alm.anotar_demografia('cam-1', 'p2', 'Hombre', '36-50', ts=_epoca_de(HOY, 17))
    alm.anotar_entrada('cam-2', 'p3', ts=_epoca_de(HOY, 17))
    alm.volcar(force=True)
    return cliente


# ── Resumen ──────────────────────────────────────────────────────────────

def test_sin_parametros_responde_hoy(poblado):
    r = poblado.get('/api/v1/tienda/resumen')
    assert r.status_code == 200
    d = r.json()
    assert d['ventana']['desde'] == HOY and d['ventana']['hasta'] == HOY
    assert d['kpis']['entradas'] == 3
    assert d['kpis']['visitantes_unicos'] == 3


def test_ayer_es_ayer(poblado):
    d = poblado.get(f'/api/v1/tienda/resumen?desde={AYER}&hasta={AYER}').json()
    assert d['kpis']['entradas'] == 2
    assert d['kpis']['visitantes_unicos'] == 2


def test_los_unicos_del_rango_no_son_la_suma_de_los_dias(poblado):
    """p1 viene ayer y hoy: en el rango es UNA persona, no dos."""
    d = poblado.get(f'/api/v1/tienda/resumen?desde={AYER}&hasta={HOY}').json()
    assert d['kpis']['entradas'] == 5
    assert d['kpis']['visitantes_unicos'] == 4    # p1, p9, p2, p3


def test_la_comparativa_es_contra_el_periodo_anterior(poblado):
    d = poblado.get('/api/v1/tienda/resumen').json()
    comp = d['comparativa']
    assert comp['ventana'] == {'desde': AYER, 'hasta': AYER}
    assert comp['kpis']['entradas'] == 2
    assert comp['variacion_pct']['entradas'] == pytest.approx(50.0)


def test_sin_base_de_comparacion_la_variacion_es_nula(cliente):
    """Dividir entre cero no da «+100 %»: da NADA, y hay que poder decirlo."""
    alm = ta.get_almacen()
    alm.anotar_entrada('cam-1', 'p1')       # sin `ts`: se sella con AHORA
    alm.volcar(force=True)
    d = cliente.get('/api/v1/tienda/resumen').json()
    assert d['kpis']['entradas'] == 1
    assert d['comparativa']['variacion_pct']['entradas'] is None


def test_la_cobertura_distingue_sin_datos_de_cero(poblado):
    """Un día sin medir no puede pintarse como un día de cero visitas."""
    hace_un_mes = ta.jornada_de(AHORA - 30 * 86400)
    d = poblado.get(
        f'/api/v1/tienda/resumen?desde={hace_un_mes}&hasta={HOY}').json()
    cobertura = d['ventana']['jornadas_con_datos']
    assert HOY in cobertura and AYER in cobertura
    assert hace_un_mes not in cobertura, 'ese día nadie estaba midiendo'


def test_el_aforo_de_una_ventana_pasada_es_nulo_no_cero(poblado):
    """Nadie sabe cuánta gente había dentro el martes a esta hora."""
    d = poblado.get(f'/api/v1/tienda/resumen?desde={AYER}&hasta={AYER}').json()
    assert d['kpis']['aforo_actual'] is None


def test_se_puede_acotar_a_una_camara(poblado):
    d = poblado.get('/api/v1/tienda/resumen?camara=cam-2').json()
    assert d['camara'] == 'cam-2'
    assert d['kpis']['entradas'] == 1


def test_pasada_la_medianoche_hoy_es_la_jornada_nueva(poblado, reloj):
    """La frontera de la jornada, EXPLÍCITA (4-sep-2026).

    Sin parámetros, `/resumen` responde por la jornada del INSTANTE de la
    petición, no por la que había cuando arrancó el servidor. A las 00:30
    «hoy» es una jornada recién estrenada, SIN DATOS (que no es lo mismo que
    cero) y todo lo de anoche pasa a ser la base de comparación. Es lo que
    quiere la tienda; queda escrito aquí para que nadie lo lea como un fallo
    la noche que la suite cruce las 00:00.
    """
    reloj(_epoca_de(MANANA, 0))            # 00:30 del día siguiente
    d = poblado.get('/api/v1/tienda/resumen').json()
    assert d['ventana'] == {'desde': MANANA, 'hasta': MANANA, 'dias': 1,
                            'jornadas_con_datos': []}
    assert d['kpis']['entradas'] == 0
    comp = d['comparativa']
    assert comp['ventana'] == {'desde': HOY, 'hasta': HOY}
    assert comp['kpis']['entradas'] == 3
    assert comp['variacion_pct']['entradas'] == pytest.approx(-100.0)


def test_con_corte_a_las_22_la_noche_es_de_la_misma_jornada(cliente, reloj,
                                                            monkeypatch):
    """Con `ELDE_TIENDA_CORTE_JORNADA_H=22` la jornada NO es el día natural.

    El cliente de las 23:30 y el de la 1:30 son de la misma jornada, y a las
    1:31 «hoy» sigue siendo la de ayer por el calendario. La cuenta la hace
    `jornada_de` (su contrato vive en test_tienda_almacen.py); lo que se
    protege aquí es que los endpoints sin parámetros se lo preguntan a ella
    y no al calendario, y que lo sembrado y lo preguntado usan el MISMO
    corte (la trampa que tumbó 14 pruebas de este fichero el 4-sep-2026).
    """
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 22)
    alm = ta.get_almacen()
    alm.anotar_entrada('cam-1', 'n1', ts=_epoca_de(HOY, 23))      # 23:30
    alm.anotar_entrada('cam-1', 'n2', ts=_epoca_de(MANANA, 1))    # 01:30
    alm.volcar(force=True)                   # la jornada se sella al volcar

    reloj(_epoca_de(MANANA, 1) + 60)         # 01:31 del día siguiente
    d = cliente.get('/api/v1/tienda/resumen').json()
    assert d['ventana'] == {'desde': HOY, 'hasta': HOY, 'dias': 1,
                            'jornadas_con_datos': [HOY]}
    assert d['kpis']['entradas'] == 2


# ── Serie ────────────────────────────────────────────────────────────────

def test_la_curva_trae_las_24_horas(poblado):
    """Una curva con huecos se dibuja torcida y engaña sobre la hora punta."""
    d = poblado.get('/api/v1/tienda/serie').json()
    assert len(d['tramos']) == 24
    por_hora = {t['hora']: t['entradas'] for t in d['tramos']}
    assert por_hora['10'] == 1
    assert por_hora['17'] == 2
    assert por_hora['03'] == 0


def test_la_punta_sale_de_las_entradas_reales(poblado):
    d = poblado.get('/api/v1/tienda/serie').json()
    assert d['punta']['hora'] == '17'
    assert d['punta']['entradas'] == 2


def test_la_serie_por_dia_da_una_fila_por_jornada(poblado):
    d = poblado.get(
        f'/api/v1/tienda/serie?desde={AYER}&hasta={HOY}&grano=dia').json()
    jornadas = {t['jornada']: t['entradas'] for t in d['tramos']}
    assert jornadas == {AYER: 2, HOY: 3}


# ── Pasillos ─────────────────────────────────────────────────────────────

def test_el_ranking_ordena_por_trafico(poblado):
    d = poblado.get('/api/v1/tienda/pasillos').json()
    assert [f['camera_id'] for f in d['ranking']] == ['cam-1', 'cam-2']
    assert d['mas_concurrido']['camera_id'] == 'cam-1'
    assert d['menos_concurrido']['camera_id'] == 'cam-2'


def test_una_camara_sin_registrar_no_se_oculta(poblado):
    """Un dato que existe y no se ve es peor que uno feo (regla de la casa)."""
    d = poblado.get('/api/v1/tienda/pasillos').json()
    assert all(f['registrada'] is False for f in d['ranking'])
    assert d['total'] == 2


def test_el_ranking_suma_todas_las_sesiones_del_rango(poblado):
    """El viejo leía UN informe: el pasillo que se desconectó salía a medias."""
    d = poblado.get(f'/api/v1/tienda/pasillos?desde={AYER}&hasta={HOY}').json()
    cam1 = next(f for f in d['ranking'] if f['camera_id'] == 'cam-1')
    assert cam1['entradas'] == 4        # 2 de ayer + 2 de hoy


# ── Demografía ───────────────────────────────────────────────────────────

def test_la_demografia_se_desglosa_por_pasillo(poblado):
    d = poblado.get('/api/v1/tienda/demografia').json()
    assert d['genero'] == {'Mujer': 1, 'Hombre': 1}
    por_camara = {f['camera_id']: f for f in d['por_camara']}
    assert por_camara['cam-1']['genero'] == {'Mujer': 1, 'Hombre': 1}
    assert 'cam-2' not in por_camara     # p3 entró pero no tiene demografía


def test_las_camaras_con_datos_pueblan_el_filtro(poblado):
    d = poblado.get('/api/v1/tienda/camaras').json()
    assert [c['camera_id'] for c in d['camaras']] == ['cam-1', 'cam-2']


# ── Coherencia entre vistas ──────────────────────────────────────────────

def test_resumen_serie_y_ranking_cuadran(poblado):
    """El criterio de «hecho» del hito: las tres vistas cuentan lo mismo."""
    ventana = f'desde={AYER}&hasta={HOY}'
    resumen = poblado.get(f'/api/v1/tienda/resumen?{ventana}').json()
    serie = poblado.get(f'/api/v1/tienda/serie?{ventana}&grano=dia').json()
    pasillos = poblado.get(f'/api/v1/tienda/pasillos?{ventana}').json()

    entradas = resumen['kpis']['entradas']
    assert entradas == sum(t['entradas'] for t in serie['tramos'])
    assert entradas == sum(f['entradas'] for f in pasillos['ranking'])


# ── Contrato de entrada ──────────────────────────────────────────────────

def test_una_fecha_con_forma_rara_se_rechaza(cliente):
    assert cliente.get('/api/v1/tienda/resumen?desde=ayer').status_code == 422
    assert cliente.get(
        '/api/v1/tienda/resumen?desde=2026-8-1').status_code == 422


def test_una_ventana_invertida_se_endereza(poblado):
    d = poblado.get(
        f'/api/v1/tienda/resumen?desde={HOY}&hasta={AYER}').json()
    assert d['ventana'] == {'desde': AYER, 'hasta': HOY, 'dias': 2,
                            'jornadas_con_datos': [AYER, HOY]}


def test_una_ventana_absurda_se_rechaza_con_motivo(cliente):
    r = cliente.get('/api/v1/tienda/resumen?desde=2020-01-01&hasta=2026-01-01')
    assert r.status_code == 400
    assert 'máximo' in r.json()['detail']


def test_la_base_caida_no_devuelve_un_500(cliente, monkeypatch):
    """Sin base, la página enseña ceros y sigue viva; no se cae el servidor."""
    def _explota(*_a, **_k):
        raise OSError('disco caído (simulado)')

    monkeypatch.setattr(ta.sqlite3, 'connect', _explota)
    r = cliente.get('/api/v1/tienda/resumen')
    assert r.status_code == 200
    assert r.json()['kpis']['entradas'] == 0
