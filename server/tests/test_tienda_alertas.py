"""
Pruebas de los avisos de NEGOCIO de la tienda (inventario #15).

Lo que más se protege no es que avisen, sino **que se callen cuando toca**:
una alerta de negocio que se dispara sola convierte el grupo de WhatsApp en
ruido que nadie lee, y ese es el final de todas las alertas.

  - el aforo viene APAGADO: «demasiada gente» depende del local y no hay un
    número honesto por defecto;
  - un pasillo que aún no ha visto a nadie hoy NO está dormido, está
    empezando el día;
  - sin días con los que comparar no hay «caída de tráfico», hay
    desconocimiento — el mismo criterio que el «sin base de comparación» del
    dashboard;
  - la misma condición no vuelve a avisar dentro del cooldown;
  - y nada de esto puede lanzar: corre dentro del bucle de inferencia.

Todas corren con el reloj CONGELADO (4-sep-2026). Las reglas deciden «hoy»
(`jornada_de()`) y «esta hora» (`strftime('%H')`) con su PROPIO `time.time()`,
así que una prueba que sembrara datos relativos a la hora real cambiaba de
significado según a qué hora corriera la suite: entre las 00:00 y las 02:59
una entrada «de hace 3 h» es de AYER, `ultima_entrada` no la ve y el pasillo
no está dormido; en el último segundo de una hora la base sembrada cae en la
hora anterior y no hay con qué comparar; y con `ELDE_TIENDA_CORTE_JORNADA_H`
en el entorno la siembra «de hoy» aterrizaba en la jornada de ayer. El fixture
`reloj` fija un martes a las 12:30 en hora local y las pruebas siembran
RELATIVO a ese instante. La frontera de la jornada, que es un comportamiento
deliberado, tiene su prueba explícita en vez de colarse según la hora.

Y todas parten de las reglas en su estado de FÁBRICA. Las constantes de `ta`
se leen del entorno al importar (`ELDE_TIENDA_*`) y la suite corre dentro del
contenedor de la tienda, con su `.env`: el aforo de un local encendía «el
aforo viene apagado», una ventana de caída corta dejaba la base en dos
jornadas y un pasillo dormido de medio minuto convertía la entrada «reciente»
en dormida. El fixture `base` fija la tabla documentada de defectos y una
prueba comprueba que esa tabla es la que el módulo trae de verdad.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import tienda_alertas as ta  # noqa: E402
from src.analityc.core import tienda_almacen as alm  # noqa: E402


class Camara(ta.AlertasNegocioMixin):
    """Lo mínimo que miran las reglas."""

    def __init__(self, camera_id='cam-1', dentro=0):
        self._camera_id_actual = camera_id
        self._camera_display_name = 'Pasillo 3'
        self._zone_state = {
            f'p{i}': {'estado': 'DENTRO'} for i in range(dentro)}
        self._alertas_pendientes = []


# ── Reloj congelado ──────────────────────────────────────────────────────

#: Martes 1-sep-2026 a las 12:30, hora LOCAL. Un martes cualquiera a media
#: jornada: hay horas de sobra por detrás («hace 3 h» sigue siendo hoy) y el
#: minuto 30 ya cuenta como hora empezada (`CAIDA_MIN_MINUTOS` vale 15).
MARTES_MEDIODIA = (2026, 9, 1, 12, 30)


class _Reloj:
    """`time.time()` congelado en un instante que ELIGE la prueba.

    `ta` y `alm` hacen `import time` y llaman a `time.time()` en el momento
    de usarlo, así que basta con sustituir el atributo del módulo. Lo demás
    (`time.localtime(ts)`, `time.mktime`, el `localtime` de SQLite) no mira el
    reloj: convierte el instante que se le da, y sigue siendo el de verdad.
    """

    def __init__(self):
        self.ahora = 0.0

    def fijar(self, a, m, d, h, mi=0, s=0):
        """Congela en esa fecha y hora LOCALES; devuelve la época."""
        self.ahora = time.mktime((a, m, d, h, mi, s, 0, 0, -1))
        return self.ahora

    @property
    def hora(self):
        return int(time.strftime('%H', time.localtime(self.ahora)))

    def jornada(self, dias=0):
        """Jornada del instante congelado, o la de `dias` días antes/después."""
        return alm.jornada_de(self.ahora + dias * 86400)


@pytest.fixture
def reloj(monkeypatch):
    r = _Reloj()
    r.fijar(*MARTES_MEDIODIA)
    monkeypatch.setattr(time, 'time', lambda: r.ahora)
    # Estas pruebas razonan en día NATURAL. Con `ELDE_TIENDA_CORTE_JORNADA_H=22`
    # en el entorno, «hoy» a las 12:30 sería la jornada de ayer y la siembra
    # de hoy caía en la anterior (4-sep-2026). El corte tiene su propia prueba
    # en test_tienda_almacen.py; aquí se fija en 0 y quien lo necesite lo sube.
    monkeypatch.setattr(alm, 'CORTE_JORNADA_H', 0)
    return r


# ── Reglas en su estado de FÁBRICA ───────────────────────────────────────

#: La tabla de defectos documentada en `tienda_alertas`. El fixture `base` la
#: fija en el módulo y `test_los_defectos_de_fabrica_son_los_de_la_tabla`
#: comprueba que es la que el módulo trae de verdad cuando el entorno no dice
#: nada: si producción cambia un defecto, deja de coincidir y se ve AQUÍ, no
#: en una prueba que pasaba por casualidad con el entorno de una tienda.
DEFECTOS_DE_FABRICA = {
    'AFORO_MAXIMO': 0,
    'PASILLO_DORMIDO_MIN': 120,
    'CAIDA_PCT': 50,
    'CAIDA_MIN_DIAS': 3,
    'CAIDA_VENTANA_DIAS': 14,
    'CAIDA_MIN_MINUTOS': 15,
    'COOLDOWN_S': 1800,
    'EVAL_S': 60,
}


def _modulo_de_fabrica(monkeypatch):
    """Una copia FRESCA de `tienda_alertas` cargada sin ninguna `ELDE_TIENDA_*`.

    Las constantes se leen del entorno al importar (`_num(...)`), así que el
    `ta` ya importado trae lo que hubiera en el entorno de la suite y no sirve
    para afirmar un defecto de fábrica. Se vuelve a ejecutar el módulo con el
    entorno limpio, bajo otro nombre y sin tocar `sys.modules`: `ta` (y
    `Camara`, que hereda de su mixin) siguen siendo los de siempre. Cuesta
    milisegundos: solo define constantes y una clase.
    """
    for nombre in list(os.environ):
        if nombre.startswith('ELDE_TIENDA_'):
            monkeypatch.delenv(nombre)
    spec = importlib.util.spec_from_file_location(
        'src.analityc.core._tienda_alertas_de_fabrica', ta.__file__)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture
def base(tmp_path, monkeypatch, reloj):
    """Almacén limpio, reglas en su estado de FÁBRICA y reloj congelado.

    Las constantes de `ta` se leen del entorno al importar y la suite corre
    dentro del contenedor elde-tiendas con el `.env` de la tienda: con
    `ELDE_TIENDA_AFORO_MAXIMO=10` caía «el aforo viene apagado», con
    `ELDE_TIENDA_CAIDA_VENTANA_DIAS=2` la base de tres jornadas se quedaba en
    dos y con `ELDE_TIENDA_PASILLO_DORMIDO_MIN` por debajo de 1 una entrada de
    hace 60 s ya era un pasillo dormido (4-sep-2026). Cada prueba parte de la
    tabla y sube lo que necesite; `EVAL_S` se anula para que evalúe en cada
    llamada (el throttle tiene su propia prueba).
    """
    alm.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    for nombre, valor in DEFECTOS_DE_FABRICA.items():
        monkeypatch.setattr(ta, nombre, valor)
    monkeypatch.setattr(ta, 'EVAL_S', 0.0)       # que evalúe en cada llamada
    return alm.get_almacen()


def _avisos(c):
    return [a['description'] for a in c._alertas_pendientes]


# ── Aforo ────────────────────────────────────────────────────────────────

def test_los_defectos_de_fabrica_son_los_de_la_tabla(monkeypatch):
    """Lo que fija el fixture `base` es lo que el módulo trae sin entorno.

    Si producción cambiara un defecto, la suite seguiría probando la tabla
    vieja sin enterarse; aquí se entera (4-sep-2026).
    """
    fresco = _modulo_de_fabrica(monkeypatch)
    assert {k: getattr(fresco, k) for k in DEFECTOS_DE_FABRICA} == \
        DEFECTOS_DE_FABRICA


def test_el_aforo_viene_apagado(base, monkeypatch):
    """Sin un número del local, encender una alarma sería inventárselo.

    El defecto se lee en una copia del módulo cargada SIN `ELDE_TIENDA_*` en
    el entorno: `ta.AFORO_MAXIMO` ya lo fija el fixture y afirmarlo ahí sería
    afirmar el propio fixture (4-sep-2026). Con el aforo apagado, 50 personas
    dentro no dicen nada.
    """
    assert _modulo_de_fabrica(monkeypatch).AFORO_MAXIMO == 0
    c = Camara(dentro=50)
    c._evaluar_alertas_negocio()
    assert _avisos(c) == []


def test_el_aforo_avisa_cuando_se_configura(base, monkeypatch):
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 10)
    c = Camara(dentro=12)
    c._evaluar_alertas_negocio()
    assert len(_avisos(c)) == 1
    assert '12 personas dentro' in _avisos(c)[0]
    assert c._alertas_pendientes[0]['event_type'] == ta.EVENTO_AFORO


def test_por_debajo_del_aforo_no_dice_nada(base, monkeypatch):
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 10)
    c = Camara(dentro=9)
    c._evaluar_alertas_negocio()
    assert _avisos(c) == []


# ── Pasillo dormido ──────────────────────────────────────────────────────

def test_un_pasillo_que_aun_no_vio_a_nadie_no_esta_dormido(base):
    """No ha pasado nada raro: es que el día acaba de empezar."""
    c = Camara()
    c._evaluar_alertas_negocio()
    assert _avisos(c) == []


def test_un_pasillo_con_gente_reciente_no_esta_dormido(base, reloj):
    base.anotar_entrada('cam-1', 'p1', ts=reloj.ahora - 60)
    # La entrada está y es de HOY: si la regla calla es por su criterio, no
    # porque el almacén no la vea. (A las 00:00:30 reales una entrada de hace
    # 60 s era de ayer y la prueba pasaba sin probar nada; 4-sep-2026.)
    # Igualdad EXACTA, no `pytest.approx`: su tolerancia relativa (1e-6) sobre
    # una época de ~1,78e9 s admite ±1.788 s, media hora, y «es la entrada de
    # hace 60 s» pasaba a ser «alguna de la última media hora». `mktime`
    # devuelve un double entero y el REAL de SQLite lo devuelve sin pérdida,
    # así que no hay nada que tolerar.
    assert base.ultima_entrada('cam-1') == reloj.ahora - 60
    c = Camara()
    c._evaluar_alertas_negocio()
    assert _avisos(c) == []


def test_un_pasillo_con_gente_hoy_pero_parado_hace_horas_avisa(base, reloj,
                                                              monkeypatch):
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 120)
    base.anotar_entrada('cam-1', 'p1', ts=reloj.ahora - 3 * 3600)  # 09:30
    c = Camara()
    c._evaluar_alertas_negocio()
    assert len(_avisos(c)) == 1
    assert 'sin una sola entrada' in _avisos(c)[0]
    assert c._alertas_pendientes[0]['event_type'] == ta.EVENTO_PASILLO


def test_cada_camara_lleva_su_propia_cuenta(base, reloj, monkeypatch):
    """El pasillo 2 no hereda el silencio del 1 (ni al revés)."""
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 120)
    base.anotar_entrada('cam-1', 'p1', ts=reloj.ahora - 3 * 3600)
    base.anotar_entrada('cam-2', 'p2', ts=reloj.ahora - 60)
    dormida, viva = Camara('cam-1'), Camara('cam-2')
    dormida._evaluar_alertas_negocio()
    viva._evaluar_alertas_negocio()
    assert len(_avisos(dormida)) == 1
    assert _avisos(viva) == []


def test_la_gente_de_ayer_no_despierta_el_pasillo_de_hoy(base, reloj,
                                                          monkeypatch):
    """A la 01:30 con la última entrada a las 22:30 de la víspera: SIN aviso.

    Es deliberado, no un descuido: `ultima_entrada` filtra por la jornada de
    HOY (`jornada_de()`), y «hoy sí hubo gente» es justo lo que distingue un
    pasillo dormido de uno que empieza el día. Han pasado 3 h sin entradas,
    pero todas fueron ayer. Hasta el 4-sep-2026 esta frontera solo asomaba
    cuando la suite corría entre las 00:00 y las 02:59 (y tumbaba las dos
    pruebas de arriba); aquí queda a la vista y con nombre.
    """
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 120)
    reloj.fijar(2026, 9, 2, 1, 30)                      # miércoles, 01:30
    vispera = time.mktime((2026, 9, 1, 22, 30, 0, 0, 0, -1))
    base.anotar_entrada('cam-1', 'p1', ts=vispera)
    assert alm.jornada_de(vispera) != reloj.jornada()   # la frontera
    # La siembra tiene que estar DE VERDAD en la jornada de la víspera y no
    # asomar en la de hoy, que es lo que la regla consulta. Sin estas dos
    # líneas la prueba se cumplía igual con una siembra que se perdiera por el
    # camino: solo afirmaba aritmética de fechas y un silencio (4-sep-2026).
    assert base.ultima_entrada(
        'cam-1', jornada=alm.jornada_de(vispera)) == vispera
    assert base.ultima_entrada('cam-1') is None
    c = Camara()
    c._evaluar_alertas_negocio()
    assert _avisos(c) == []


def test_con_corte_de_jornada_la_madrugada_sigue_siendo_ayer(base, reloj,
                                                             monkeypatch):
    """La misma escena con `ELDE_TIENDA_CORTE_JORNADA_H=22` SÍ avisa.

    Para una tienda que cierra de madrugada la jornada arranca a las 22:00: la
    entrada de las 22:30 y la 01:30 son la MISMA jornada, así que el pasillo
    lleva 3 h dormido dentro de un día que sigue abierto. El corte se fija
    ANTES de sembrar: la jornada se graba al volcar, igual que en producción.
    """
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 120)
    monkeypatch.setattr(alm, 'CORTE_JORNADA_H', 22)
    reloj.fijar(2026, 9, 2, 1, 30)
    vispera = time.mktime((2026, 9, 1, 22, 30, 0, 0, 0, -1))
    base.anotar_entrada('cam-1', 'p1', ts=vispera)
    assert alm.jornada_de(vispera) == reloj.jornada() == '2026-09-01'
    c = Camara()
    c._evaluar_alertas_negocio()
    assert len(_avisos(c)) == 1
    assert 'desde hace 180 min' in _avisos(c)[0]


# ── Caída de tráfico ─────────────────────────────────────────────────────

def _sembrar_hora(base, camera, jornadas, hora, cuantas):
    """`cuantas` entradas a esa hora (minuto 20, LOCAL) en cada jornada.

    Por fecha y hora locales, no restando 86.400 s al instante congelado: es
    la misma cuenta que hace la regla («la misma hora en otros días», con
    `strftime('%H', ..., 'localtime')`), y no se mueve una hora en las zonas
    con horario de verano. Las jornadas llegan del `reloj`, así que son las
    del día natural del instante congelado.
    """
    for jornada in jornadas:
        a, m, d = (int(x) for x in jornada.split('-'))
        for i in range(cuantas):
            ts = time.mktime((a, m, d, hora, 20, 0, 0, 0, -1))
            base.anotar_entrada(camera, f'{jornada}-{i}', ts=ts)


def test_sin_base_suficiente_no_se_inventa_una_caida(base, reloj, monkeypatch):
    """Dos días no son «lo normal a esta hora»."""
    monkeypatch.setattr(ta, 'CAIDA_MIN_DIAS', 3)
    monkeypatch.setattr(ta, 'CAIDA_PCT', 50)
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 0)   # aislar la regla
    monkeypatch.setattr(ta, 'CAIDA_MIN_MINUTOS', 0)     # idem: tiene su prueba
    previas = [reloj.jornada(-1), reloj.jornada(-2)]
    _sembrar_hora(base, 'cam-1', previas, reloj.hora, 10)

    c = Camara()
    c._evaluar_alertas_negocio()
    assert not [a for a in _avisos(c) if 'entradas' in a]

    # Contraste: con el mínimo en 2 esos mismos dos días YA son base, y el 0
    # de hoy es una caída. Prueba que lo que calló arriba fue el criterio y
    # no una base que la regla nunca llegó a ver (antes, en los minutos :00
    # a :14 y en el cruce de hora, la regla ni se evaluaba; 4-sep-2026).
    monkeypatch.setattr(ta, 'CAIDA_MIN_DIAS', 2)
    c._evaluar_alertas_negocio()
    assert [a for a in _avisos(c) if 'lo normal a esta hora' in a]


def test_con_base_y_caida_real_avisa(base, reloj, monkeypatch):
    monkeypatch.setattr(ta, 'CAIDA_MIN_DIAS', 3)
    monkeypatch.setattr(ta, 'CAIDA_PCT', 50)
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 0)   # aislar la regla
    monkeypatch.setattr(ta, 'CAIDA_MIN_MINUTOS', 0)     # idem: tiene su prueba
    previas = [reloj.jornada(-n) for n in (1, 2, 3)]
    _sembrar_hora(base, 'cam-1', previas, reloj.hora, 10)   # normal: 10/hora
    base.anotar_entrada('cam-1', 'hoy-1', ts=reloj.ahora)   # hoy: 1

    c = Camara()
    c._evaluar_alertas_negocio()
    avisos = [a for a in _avisos(c) if 'lo normal a esta hora' in a]
    assert len(avisos) == 1
    assert '90 % menos' in avisos[0]


def test_un_dia_normal_no_avisa_de_caida(base, reloj, monkeypatch):
    """Base de 10 y hoy 9 (un 10 % menos): lejos del 50 %, la regla calla."""
    monkeypatch.setattr(ta, 'CAIDA_MIN_DIAS', 3)
    monkeypatch.setattr(ta, 'CAIDA_PCT', 50)
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 0)
    monkeypatch.setattr(ta, 'CAIDA_MIN_MINUTOS', 0)
    previas = [reloj.jornada(-n) for n in (1, 2, 3)]
    _sembrar_hora(base, 'cam-1', previas, reloj.hora, 10)
    _sembrar_hora(base, 'cam-1', [reloj.jornada()], reloj.hora, 9)
    # La premisa, comprobada donde la regla la lee (`serie` por hora): tres
    # jornadas de base con 10 y la de hoy con 9, todas en la MISMA hora del
    # reloj congelado. Una prueba que solo afirma silencio se cumplía igual
    # con la siembra perdida por el camino (4-sep-2026).
    hora = f'{reloj.hora:02d}'
    filas = base.serie(reloj.jornada(-3), reloj.jornada(), camera_id='cam-1',
                       grano='hora')
    assert [(f['jornada'], f['hora'], f['entradas']) for f in filas] == [
        (reloj.jornada(-3), hora, 10), (reloj.jornada(-2), hora, 10),
        (reloj.jornada(-1), hora, 10), (reloj.jornada(), hora, 9)]

    c = Camara()
    c._evaluar_alertas_negocio()
    assert not [a for a in _avisos(c) if 'lo normal a esta hora' in a]

    # Contraste: con el umbral en el 5 % ese mismo 10 % de bajada SÍ es una
    # caída. Prueba que arriba calló el criterio (50 %) y no una base que la
    # regla nunca llegó a ver.
    monkeypatch.setattr(ta, 'CAIDA_PCT', 5)
    c._evaluar_alertas_negocio()
    avisos = [a for a in _avisos(c) if 'lo normal a esta hora' in a]
    assert len(avisos) == 1
    assert '10 % menos' in avisos[0]


# ── Anti-ruido ───────────────────────────────────────────────────────────

def test_la_misma_condicion_no_avisa_dos_veces(base, monkeypatch):
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 5)
    c = Camara(dentro=9)
    for _ in range(10):
        c._evaluar_alertas_negocio()
    assert len(_avisos(c)) == 1, 'el cooldown evita convertir el grupo en ruido'


def test_pasado_el_cooldown_vuelve_a_avisar(base, monkeypatch):
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 5)
    monkeypatch.setattr(ta, 'COOLDOWN_S', 0.0)
    c = Camara(dentro=9)
    c._evaluar_alertas_negocio()
    c._evaluar_alertas_negocio()
    assert len(_avisos(c)) == 2


def test_no_se_evalua_en_cada_frame(base, monkeypatch):
    """Con 25 fps, evaluar reglas por frame sería tirar el FPS a la basura."""
    monkeypatch.setattr(ta, 'EVAL_S', 3600.0)
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 5)
    c = Camara(dentro=9)
    c._evaluar_alertas_negocio()          # esta sí evalúa
    c._zone_state = {f'p{i}': {'estado': 'DENTRO'} for i in range(99)}
    for _ in range(100):
        c._evaluar_alertas_negocio()      # estas salen por el primer if
    assert len(_avisos(c)) == 1


def test_la_cola_de_avisos_esta_acotada(base, monkeypatch):
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 5)
    monkeypatch.setattr(ta, 'COOLDOWN_S', 0.0)
    c = Camara(dentro=9)
    for _ in range(320):
        c._evaluar_alertas_negocio()
    assert len(c._alertas_pendientes) <= 200


def test_una_base_caida_no_tumba_el_frame(base, monkeypatch):
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 1)

    def _explota(*_a, **_k):
        raise OSError('disco caído (simulado)')

    monkeypatch.setattr(alm.sqlite3, 'connect', _explota)
    c = Camara()
    c._evaluar_alertas_negocio()          # no lanza
    assert _avisos(c) == []


def test_sin_camara_en_curso_no_hace_nada(base, monkeypatch):
    """Antes del primer frame no hay cámara: no se inventa una."""
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 1)
    c = Camara(dentro=9)
    c._camera_id_actual = None
    c._evaluar_alertas_negocio()
    assert _avisos(c) == []


def test_el_aviso_llega_con_la_forma_que_espera_el_reenviador(base,
                                                              monkeypatch):
    """Contrato con whatsapp_alertas: sin estos campos el mensaje sale mudo."""
    monkeypatch.setattr(ta, 'AFORO_MAXIMO', 5)
    c = Camara(dentro=9)
    c._evaluar_alertas_negocio()
    a = c._alertas_pendientes[0]
    for clave in ('event_type', 'class_name', 'description', 'timestamp',
                  'camera_name'):
        assert a.get(clave), f'falta {clave}'

    # Y tiene que caer del lado INFORMATIVO (👤), no del de intrusión (🚨):
    # una métrica de negocio con alarma roja confunde a quien lee el grupo.
    from src.app.whatsapp_alertas import _EVENTOS_INFORMATIVOS
    evento = a['event_type'].lower()
    assert any(e in evento for e in _EVENTOS_INFORMATIVOS)


def test_una_hora_recien_empezada_no_se_compara(base, reloj, monkeypatch):
    """En el minuto 5 de las 12:00 SIEMPRE se va por debajo de la hora entera
    de otros días. Avisar de eso sería avisar de que el reloj avanza."""
    monkeypatch.setattr(ta, 'CAIDA_MIN_DIAS', 3)
    monkeypatch.setattr(ta, 'CAIDA_PCT', 50)
    monkeypatch.setattr(ta, 'PASILLO_DORMIDO_MIN', 0)
    monkeypatch.setattr(ta, 'CAIDA_MIN_MINUTOS', 15)
    reloj.fijar(2026, 9, 1, 12, 5)
    previas = [reloj.jornada(-n) for n in (1, 2, 3)]
    _sembrar_hora(base, 'cam-1', previas, reloj.hora, 10)

    c = Camara()
    c._evaluar_alertas_negocio()
    assert not [a for a in _avisos(c) if 'lo normal a esta hora' in a]

    # Contraste: en el minuto 15 la hora ya «ha empezado» y la misma base (10
    # de media, 0 hoy) sí es una caída. Antes se probaba con el umbral en 99,
    # que callaba a cualquier minuto: no había forma de que fallara
    # (4-sep-2026).
    reloj.fijar(2026, 9, 1, 12, 15)
    c._evaluar_alertas_negocio()
    assert [a for a in _avisos(c) if 'lo normal a esta hora' in a]
