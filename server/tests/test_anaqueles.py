"""
Pruebas del modo `Anaqueles` (visitas y agarres por nivel del estante).

No se instancia el pipeline completo (cargaría YOLO y pediría GPU): se prueba
`MedidorAnaqueles` directamente con dobles inyectados — una pose falsa que
devuelve muñecas donde la prueba manda, y una guardia de escena controlable.
El registro escribe en un directorio temporal, nunca en `server/output/`.

Lo que más se protege:

  - **una estancia = una visita** — quedarse 10 minutos frente al anaquel no
    puede inflar el contador, y el cooldown evita recontar al que vuelve a
    los dos minutos;
  - **el agarre exige racha de muñeca** — una muñeca que roza el polígono un
    solo frame no es un agarre; dos seguidos sí, y el cooldown por
    persona+área evita que un solo gesto cuente tres veces;
  - **la pose solo corre para quien está en compuerta** — es el contrato de
    coste del modo (N personas = N inferencias);
  - **el cambio de escena SUSPENDE el conteo** — la causa #1 del stand roto:
    zonas dibujadas sobre otra escena fabricando cifras creíbles;
  - **los totales sobreviven al renacer del procesador** — el registro
    reanuda la jornada del día desde disco;
  - **el reloj de pared va congelado** (4-sep-2026) — la JORNADA del
    registro la fija `time.strftime` real, no el `ahora` simulado, y cada
    `procesar` hace rollover si cambió el día: a las 23:59:59.99 la suite
    veía las áreas reabrirse a cero a mitad de test. Ver `reloj`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.anaqueles import (  # noqa: E402
    MODO, MedidorAnaqueles, _DEFECTOS)
# Estas pruebas fijan la MÁQUINA del gesto y la visita del cuerpo: desde
# el 12-oct-2026 los defectos cuentan la visita con la muñeca y esperan
# la bajada para contar el agarre (test_anaqueles_confirmacion.py).
_REGLAS_DEL_GESTO = {'visita_por_muneca': False, 'agarre_requiere_bajada': False}
from src.analityc.core.anaqueles_registro import (  # noqa: E402
    RegistroAnaqueles)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

#: Reloj de pared CONGELADO de toda la suite: martes 1-sep-2026 a las 12:00
#: en hora LOCAL (`time.mktime` respeta la zona de la máquina, así la
#: jornada es la que el test presupone en cualquier servidor). Los `ahora`
#: de cada prueba se siembran RELATIVOS a este instante, como en producción,
#: donde `ahora` ES `time.time()`. Antes cada prueba usaba `t0 = 1000.0`
#: (1970) con el reloj de pared libre: ver `reloj`.
MEDIODIA = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))

#: Un nivel del estante: rectángulo x 100..300, y 100..200 (píxeles).
AREA = [[100, 100], [300, 100], [300, 200], [100, 200]]

#: Persona pegada al anaquel: su franja superior toca el área.
BOX_EN_AREA = [120, 120, 220, 460]
#: Persona lejos, en otra parte del pasillo.
BOX_LEJOS = [500, 300, 560, 470]


class PoseFalsa:
    """Devuelve las muñecas que la prueba decida y cuenta las llamadas."""

    def __init__(self, munecas=None):
        self.respuesta = list(munecas or [])
        self.llamadas = 0
        self.disponible = True

    def munecas(self, frame, box):
        self.llamadas += 1
        return list(self.respuesta)


class EscenaControlada:
    """`evaluar` devuelve lo que la prueba programe (None = estable)."""

    def __init__(self):
        self.veredicto = None

    def evaluar(self, camara, frame, ahora=None):
        v, self.veredicto = self.veredicto, None
        return v


def _medidor(tmp_path, pose=None, escena=None, **cfg_extra):
    cfg = dict(_DEFECTOS, **_REGLAS_DEL_GESTO)
    cfg.update(cfg_extra)
    registro = RegistroAnaqueles('cam-test', raiz=tmp_path)
    m = MedidorAnaqueles('cam-test', cfg,
                         pose=pose or PoseFalsa(),
                         registro=registro,
                         escena=escena or EscenaControlada())
    m.poner_zonas([AREA], ['Área 1'])
    return m


def _tracks(box, tid=1, pid=None):
    return {tid: {'class': -1, 'box': list(box), 'persistent_id': pid}}


def _estar(m, box, t0, n, paso=0.5, tid=1, pid=None):
    """La persona SE QUEDA quieta frente al área `n` frames. Devuelve el
    ts del último frame. (v2: la quietud necesita trayectoria, no bastan
    dos frames sueltos.)"""
    t = t0
    for i in range(n):
        t = t0 + i * paso
        m.procesar(FRAME, _tracks(box, tid=tid, pid=pid), ahora=t)
    return t


@pytest.fixture(autouse=True)
def reloj(monkeypatch):
    """Congela `time.time()` y `time.strftime(fmt)` sin tupla en MEDIODIA.

    El medidor recibe el tiempo por `ahora`, pero la JORNADA del registro
    la decide `_hoy()` = `time.strftime('%Y-%m-%d')`, que lee el reloj de
    pared REAL, y cada `procesar` llama a `rollover_si_cambio_dia()`. Con
    `ahora` simulado y el reloj de pared libre, una prueba que arrancara
    en los últimos ~10 ms de un día natural veía el rollover a mitad de
    cuerpo: cooldowns limpios, áreas reabiertas a cero y `visitas == 1`
    saliendo 0 (reproducido el 4-sep-2026 a las 23:59:59.995: 14 pruebas
    de este archivo caían). Congelado, `time.time()` y `_hoy()` cuentan la
    misma jornada que los `ahora` sembrados desde MEDIODIA.

    `time.strftime(fmt)` SIN tupla lee el reloj en C, no pasa por
    `time.time`, por eso se envuelve; con tupla explícita se respeta tal
    cual (así `_hora(ts)` del medidor sigue formateando el `ahora` que se
    le da). Devuelve un mando con `.fijar(ts)` para la prueba que mueve el
    reloj de pared a propósito (el cruce de medianoche).
    """
    estado = {'ahora': MEDIODIA}
    strftime_real = time.strftime
    localtime_real = time.localtime

    def strftime_congelado(formato, tupla=None):
        if tupla is None:
            tupla = localtime_real(estado['ahora'])
        return strftime_real(formato, tupla)

    monkeypatch.setattr(time, 'time', lambda: estado['ahora'])
    monkeypatch.setattr(time, 'strftime', strftime_congelado)

    class _Mando:
        @staticmethod
        def fijar(ts: float) -> None:
            estado['ahora'] = float(ts)

    return _Mando()


# ── Visitas ──────────────────────────────────────────────────────────────
def test_visita_exige_detenerse_y_cuenta_una_sola_vez(tmp_path):
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0)          # llega
    m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + 0.5)
    assert m.registro.area(0)['visitas'] == 0                  # aún sin dwell
    _estar(m, BOX_EN_AREA, t0 + 1.0, 4)                        # sigue quieto
    assert m.registro.area(0)['visitas'] == 1
    # Sigue dentro 5 minutos: la MISMA estancia no re-cuenta.
    for i in range(10):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + 3.0 + i * 30)
    assert m.registro.area(0)['visitas'] == 1


def test_visita_cooldown_evita_recontar_al_que_vuelve(tmp_path):
    m = _medidor(tmp_path, visita_cooldown_s=120.0)
    t0 = MEDIODIA
    _estar(m, BOX_EN_AREA, t0, 6)
    assert m.registro.area(0)['visitas'] == 1
    # Se va (varios frames: la gracia de estancia exige VER la ausencia
    # sostenida) y vuelve al minuto: dentro del cooldown, no cuenta.
    _estar(m, BOX_LEJOS, t0 + 10.0, 4)
    _estar(m, BOX_EN_AREA, t0 + 60.0, 6)
    assert m.registro.area(0)['visitas'] == 1
    # Vuelve pasado el cooldown: visita nueva.
    _estar(m, BOX_LEJOS, t0 + 70.0, 4)
    _estar(m, BOX_EN_AREA, t0 + 200.0, 6)
    assert m.registro.area(0)['visitas'] == 2


def test_persona_lejos_no_cuenta_ni_gasta_pose(tmp_path):
    pose = PoseFalsa([(150, 150, 0.9)])
    m = _medidor(tmp_path, pose=pose)
    t0 = MEDIODIA
    for i in range(5):
        m.procesar(FRAME, _tracks(BOX_LEJOS), ahora=t0 + i)
    assert m.registro.area(0)['visitas'] == 0
    assert m.registro.area(0)['agarres'] == 0
    # El contrato de coste: sin nadie en compuerta, la pose NO corre.
    assert pose.llamadas == 0


# ── Agarres (semántica de GESTO, corregida el 14-ago) ────────────────────
class PoseSecuencia:
    """Guion de muñecas: una lista por llamada; la última se repite."""

    def __init__(self, guion):
        self.guion = list(guion)
        self.llamadas = 0
        self.disponible = True

    def munecas(self, frame, box):
        i = min(self.llamadas, len(self.guion) - 1)
        self.llamadas += 1
        return list(self.guion[i])


#: Muñeca PROFUNDA en el área 100..300 x 100..200 (≥12 px de cada borde).
DENTRO = [(150, 150, 0.9)]
#: Muñeca fuera del polígono (re-arma el gesto).
FUERA = [(400, 300, 0.9)]


def test_LA_CORRECCION_parado_con_la_mano_dentro_no_cuenta(tmp_path):
    """El bug reportado: la persona se acerca, NO agarra nada, y su muñeca
    colgando queda «dentro» del polígono por la proyección 2D. Antes: un
    agarre falso cada cooldown. Ahora: CERO, para siempre."""
    pose = PoseFalsa(DENTRO)                      # dentro desde el frame 1
    m = _medidor(tmp_path, pose=pose, agarre_cooldown_s=10.0)
    t0 = MEDIODIA
    for i in range(120):                          # UN MINUTO parado ahí
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0     # jamás se armó el gesto
    assert m.registro.area(0)['visitas'] == 1     # la visita sí, UNA


def test_gesto_fuera_dentro_cuenta_uno_y_solo_uno(tmp_path):
    # 4 frames con la mano fuera (asienta la compuerta y arma el gesto),
    # luego mete la mano y la deja dentro: UN agarre, no uno por cooldown.
    pose = PoseSecuencia([FUERA] * 4 + [DENTRO] * 40)
    m = _medidor(tmp_path, pose=pose, agarre_cooldown_s=10.0)
    t0 = MEDIODIA
    for i in range(44):                           # 22 s en total
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 1


def test_rearme_permite_el_segundo_gesto(tmp_path):
    # Gesto 1, saca la mano un rato (re-arma y pasa el cooldown), gesto 2.
    pose = PoseSecuencia([FUERA] * 4 + [DENTRO] * 4
                         + [FUERA] * 22 + [DENTRO] * 4)
    m = _medidor(tmp_path, pose=pose, agarre_cooldown_s=10.0)
    t0 = MEDIODIA
    for i in range(34):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 2


def test_asentamiento_descarta_el_transitorio_de_llegada(tmp_path):
    """Meter la mano ANTES de asentarse (1.2 s en compuerta) no cuenta: es
    el cuerpo entero «entrando» al polígono al caminar hacia el estante."""
    pose = PoseSecuencia([FUERA] * 2 + [DENTRO] * 2 + [FUERA] * 10)
    m = _medidor(tmp_path, pose=pose, agarre_asentamiento_s=5.0)
    t0 = MEDIODIA
    for i in range(14):                           # los DENTRO caen a t+1.0s
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0


def test_muneca_superficial_no_dispara_ni_calienta(tmp_path):
    # (106,150): a 6 px del borde — en la ZONA GRIS de la profundidad
    # efectiva (bbox de 100 px → exige 8, histéresis 4): ni agarre ni
    # calor, aunque esté «dentro» del polígono.
    pose = PoseSecuencia([FUERA] * 4 + [[(106, 150, 0.9)]] * 10)
    m = _medidor(tmp_path, pose=pose)
    t0 = MEDIODIA
    for i in range(14):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0
    assert float(m.registro.area(0)['_grid'].sum()) == 0.0


def test_muneca_fuera_del_area_no_agarra(tmp_path):
    pose = PoseFalsa(FUERA)                       # muñeca FUERA del polígono
    m = _medidor(tmp_path, pose=pose)
    t0 = MEDIODIA
    for i in range(6):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0
    assert pose.llamadas > 0                      # la pose SÍ corrió (compuerta)


def test_agarre_alimenta_el_heatmap_del_area(tmp_path):
    # Gesto real: fuera → dentro profundo en (140,130).
    pose = PoseSecuencia([FUERA] * 4 + [[(140, 130, 0.9)]] * 4)
    m = _medidor(tmp_path, pose=pose)
    t0 = MEDIODIA
    for i in range(8):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 1
    grid = m.registro.area(0)['_grid']
    assert float(grid.sum()) > 0.0
    # La muñeca (140,130) cae en el cuarto superior-izquierdo del área
    # (100..300 x 100..200): la celda caliente debe estar en esa esquina.
    filas, cols = grid.shape
    assert grid[: filas // 2, : cols // 2].sum() > 0.0
    # Y solo ahí: el resto de la rejilla sigue fría.
    assert grid[filas // 2:, :].sum() == 0.0


# ── Guardia de escena ────────────────────────────────────────────────────
def test_cambio_de_escena_suspende_el_conteo(tmp_path):
    escena = EscenaControlada()
    m = _medidor(tmp_path, escena=escena, escena_estabilizacion_s=3.0)
    t0 = MEDIODIA
    escena.veredicto = 'cambio'                   # el operador cambió el canal
    _estar(m, BOX_EN_AREA, t0, 6)                 # t0..t0+2.5: suspendido
    assert m.registro.area(0)['visitas'] == 0
    # Pasada la estabilización, se vuelve a contar desde cero (dwell nuevo).
    _estar(m, BOX_EN_AREA, t0 + 3.0, 8)
    assert m.registro.area(0)['visitas'] == 1


def test_el_aviso_de_escena_viaja_en_el_bloque(tmp_path):
    escena = EscenaControlada()
    m = _medidor(tmp_path, escena=escena)
    escena.veredicto = 'cambio'
    bloque = m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=MEDIODIA)
    assert bloque and bloque.get('escena', {}).get('tipo') == 'cambio'


# ── Persistencia ─────────────────────────────────────────────────────────
def test_los_totales_sobreviven_al_renacer(tmp_path):
    # Gesto real (fuera → dentro) para que haya un agarre que persistir.
    m = _medidor(tmp_path,
                 pose=PoseSecuencia([FUERA] * 4 + [DENTRO] * 4))
    t0 = MEDIODIA
    for i in range(8):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 1
    assert m.registro.area(0)['agarres'] == 1
    calor = float(m.registro.area(0)['_grid'].sum())
    m.cerrar()                                    # volcado forzado a disco

    # El «procesador renace» (reconexión): registro nuevo, mismo disco.
    r2 = RegistroAnaqueles('cam-test', raiz=tmp_path)
    assert r2.area(0)['visitas'] == 1
    assert r2.area(0)['agarres'] == 1
    assert float(r2.area(0)['_grid'].sum()) == calor
    assert r2.area(0)['nombre'] == 'Área 1'


def test_renombrar_el_area_conserva_los_contadores(tmp_path):
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    _estar(m, BOX_EN_AREA, t0, 6)
    assert m.registro.area(0)['visitas'] == 1
    # El operador renombra a mitad de jornada: identidad = ÍNDICE.
    m.poner_zonas([AREA], ['Nivel salsas'])
    m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + 3.0)
    assert m.registro.area(0)['visitas'] == 1
    assert m.registro.area(0)['nombre'] == 'Nivel salsas'


# ── Bloque en vivo ───────────────────────────────────────────────────────
def test_el_bloque_lleva_totales_absolutos_y_personas_ahora(tmp_path):
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    _estar(m, BOX_EN_AREA, t0, 6)
    bloque = m.procesar(FRAME, _tracks(BOX_EN_AREA),
                        ahora=t0 + 5.0)           # pasa el throttle de 2 s
    assert bloque is not None
    areas = {a['idx']: a for a in bloque['areas']}
    assert areas[0]['visitas'] == 1               # absoluto, no delta
    assert areas[0]['personas_ahora'] == 1
    assert bloque['camera'] == 'cam-test'


def test_sin_areas_dibujadas_lo_dice_y_no_revienta(tmp_path):
    m = _medidor(tmp_path)
    m.poner_zonas([], [])
    bloque = m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=MEDIODIA)
    assert bloque is not None and bloque.get('sin_areas') is True


def test_modo_es_el_string_del_combo():
    # El literal exacto: combo del cliente + /ws/Anaqueles + _build_processor.
    assert MODO == 'Anaqueles'


# ── v2 (14-ago tarde): «estar» vs «pasar» — los falsos que quedaban ──────
def test_V2_caminar_junto_al_estante_no_cuenta_nada(tmp_path):
    """EL falso positivo de visitas: cualquiera que camina por el pasillo
    se proyecta sobre el área durante todo el trayecto (>1.5 s). Una visita
    es una DETENCIÓN: el dwell solo corre con el cuerpo quieto. Y su braceo
    tampoco puede agarrar: la pose ni siquiera corre para quien camina."""
    pose = PoseFalsa(DENTRO)
    area_larga = [[100, 100], [600, 100], [600, 200], [100, 200]]
    m = _medidor(tmp_path, pose=pose)
    m.poner_zonas([area_larga], ['Anaquel 1'])
    t0 = MEDIODIA
    for i in range(12):                           # camina 6 s a lo largo
        x = 100 + i * 30                          # 60 px/s ≈ 0.6 anchos/s
        m.procesar(FRAME, _tracks([x, 120, x + 100, 460]), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 0
    assert m.registro.area(0)['agarres'] == 0
    assert pose.llamadas == 0                     # ni GPU gastada en él
    assert m._descartes.get('visita_movimiento', 0) > 0


def test_V2_el_roce_del_pasillo_no_abre_compuerta(tmp_path):
    """Quien va por el MEDIO del pasillo apenas roza el área en proyección:
    ese sliver no puede abrir compuerta (solape mínimo del 15%)."""
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    _estar(m, [290, 120, 390, 460], t0, 8)        # solo 10 px pisan el área
    assert m.registro.area(0)['visitas'] == 0
    assert m._descartes.get('visita_solape', 0) > 0


def test_V2_track_partido_no_recuenta_la_visita(tmp_path):
    """A 2-3 FPS las oclusiones parten los tracks: la MISMA persona parada
    recibe un `prov:` nuevo y el cooldown por persona no la reconoce. El
    dedup ESPACIAL sí: mismo sitio + hace poco = la misma persona."""
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    _estar(m, BOX_EN_AREA, t0, 6, tid=1)          # visita contada
    assert m.registro.area(0)['visitas'] == 1
    # El track se parte: id nuevo, MISMO sitio.
    _estar(m, BOX_EN_AREA, t0 + 4.0, 6, tid=7)
    assert m.registro.area(0)['visitas'] == 1     # no re-cuenta
    assert m._descartes.get('visita_dedup_espacial', 0) >= 1


def test_V2_el_temblor_de_la_muneca_no_agarra(tmp_path):
    """La muñeca colgando oscila 10-30 px alrededor del borde del área
    baja: re-armaba y disparaba. La AMPLITUD del gesto lo mata — la mano
    debe VIAJAR desde fuera hasta dentro (≥30% del ancho del cuerpo)."""
    pose = PoseSecuencia([[(102, 150, 0.9)]] * 3     # fuera (prof 2 < 4)
                         + [[(112, 150, 0.9)]] * 6)  # dentro (prof 12 ≥ 8)
    m = _medidor(tmp_path, pose=pose)
    t0 = MEDIODIA
    for i in range(9):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0
    assert m._descartes.get('agarre_amplitud', 0) >= 1


def test_V2_track_partido_no_recuenta_el_agarre(tmp_path):
    """El gesto contado con tid=1 no puede re-contarse cuando el track se
    parte a tid=9 en el mismo sitio dentro del cooldown."""
    m = _medidor(tmp_path,
                 pose=PoseSecuencia([FUERA] * 4 + [DENTRO] * 20),
                 agarre_cooldown_s=10.0)
    t0 = MEDIODIA
    for i in range(8):
        m.procesar(FRAME, _tracks(BOX_EN_AREA, tid=1), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 1
    # Track partido: id nuevo, la MISMA mano en el mismo sitio. La pose del
    # guion re-arma (FUERA no aplica ya) — forzamos el peor caso: el estado
    # del tid nuevo nace desarmado, se arma con el guion y reintenta.
    m2_pose = PoseSecuencia([FUERA] * 3 + [DENTRO] * 6)
    m.pose = m2_pose
    for i in range(9):
        m.procesar(FRAME, _tracks(BOX_EN_AREA, tid=9),
                   ahora=t0 + 4.0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 1     # dedup espacial
    assert m._descartes.get('agarre_dedup_espacial', 0) >= 1


# ── v3 (revisión adversarial): los 16 agujeros confirmados ───────────────
IZQ_FUERA = (400, 300, 0.9, 'izq')
DER_DENTRO = (150, 150, 0.9, 'der')


def test_V3_alternancia_de_munecas_no_agarra(tmp_path):
    """EL agujero grave: la pose pierde la muñeca interior unos frames, la
    OTRA mano (colgando fuera) «armaba» el gesto y regalaba la amplitud —
    un agarre falso cada cooldown para cualquiera parado. v3: la máquina
    es POR MUÑECA — la evidencia de una mano jamás arma a la otra."""
    guion = ([[IZQ_FUERA, DER_DENTRO]] * 3 + [[IZQ_FUERA]] * 2) * 4
    m = _medidor(tmp_path, pose=PoseSecuencia(guion))
    t0 = MEDIODIA
    for i in range(20):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0


def test_V3_caminante_radial_no_es_quieto(tmp_path):
    """Quien camina HACIA la cámara casi no desplaza su centro en píxeles
    (el bbox CRECE): pasaba por quieto y renacían las visitas en el eje
    principal del pasillo. v3 suma la velocidad RADIAL (cambio de ancho)."""
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    for i in range(8):
        w = 50 + i * 16                           # el bbox crece al acercarse
        cx = 170 + i * 16                         # y deriva apenas
        m.procesar(FRAME, _tracks([cx - w / 2, 120, cx + w / 2, 460]),
                   ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 0
    assert m._descartes.get('visita_movimiento', 0) > 0


def test_V3_el_acompanante_cuenta_su_propia_visita(tmp_path):
    """El dedup espacial se tragaba al acompañante hombro con hombro. v3:
    dos tracks que COEXISTEN en el frame jamás son fragmentos."""
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    for i in range(6):
        tracks = {**_tracks([120, 120, 220, 460], tid=1),
                  **_tracks([150, 120, 250, 460], tid=2)}
        m.procesar(FRAME, tracks, ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 2


def test_V3_transito_apilado_cuenta_solo_donde_se_asienta(tmp_path):
    """Un reach que ATRAVIESA el nivel de abajo camino al de arriba
    contaba en los dos. v3: el conteo queda pendiente un frame y se
    arbitra — solo cuenta donde la mano se asienta."""
    # La balda ALTA más arriba en la imagen (y menor), como en la tienda.
    # Hasta el 6-oct estaban al revés; desde la regla de la «bajada» (una
    # mano que viene de DENTRO de un área de encima no agarra abajo,
    # `test_anaqueles_bajada.py`) ese dibujo era otro gesto: bajar.
    area_baja = [[100, 210], [300, 210], [300, 310], [100, 310]]
    area_alta = [[100, 100], [300, 100], [300, 200], [100, 200]]
    guion = ([[(400, 260, 0.9, 'der')]] * 2       # fuera: arma
             + [[(150, 260, 0.9, 'der')]] * 2     # atraviesa el nivel bajo
             + [[(150, 150, 0.9, 'der')]] * 5)    # se asienta en el alto
    m = _medidor(tmp_path, pose=PoseSecuencia(guion))
    m.poner_zonas([area_baja, area_alta], ['Anaquel 1', 'Anaquel 2'])
    t0 = MEDIODIA
    for i in range(9):
        m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 0     # el de paso: NO
    assert m.registro.area(1)['agarres'] == 1     # donde se asentó: SÍ
    assert m._descartes.get('agarre_transito', 0) >= 1


def test_V3_estancia_larga_con_track_partido_no_recuenta(tmp_path):
    """Parado >2 min (más que el cooldown) y el track se parte: el sitio
    del dedup se REFRESCA mientras la estancia vive, así el fragmento
    tardío tampoco re-cuenta."""
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    _estar(m, BOX_EN_AREA, t0, 27, paso=5.0, tid=1)   # 130 s parado
    assert m.registro.area(0)['visitas'] == 1
    _estar(m, BOX_EN_AREA, t0 + 131.0, 6, tid=8)      # track partido
    assert m.registro.area(0)['visitas'] == 1
    assert m._descartes.get('visita_dedup_espacial', 0) >= 1


def test_V3_bbox_gigante_frente_a_nivel_angosto_si_cuenta(tmp_path):
    """El 15% se exigía sobre la FRANJA: un bbox gigante frente a un nivel
    angosto no podía cubrirlo jamás. v3 acota por el MENOR de (franja,
    polígono)."""
    nivel_angosto = [[150, 140], [500, 140], [500, 190], [150, 190]]
    m = _medidor(tmp_path)
    m.poner_zonas([nivel_angosto], ['Anaquel 1'])
    t0 = MEDIODIA
    _estar(m, [100, 100, 580, 700], t0, 6)        # bbox enorme, quieto
    assert m.registro.area(0)['visitas'] == 1


def test_V3_tras_cambio_de_canal_no_reanuda_hasta_volver(tmp_path):
    """A los 3 s reanudaba sobre el canal EQUIVOCADO, indefinidamente. v3:
    suspendido hasta que la cámara VUELVE a la escena de las zonas."""
    fr_zonas = np.zeros((480, 640, 3), dtype=np.uint8)
    fr_zonas[100:300, 80:560] = 170               # la escena «casa»
    fr_otro = np.zeros((480, 640, 3), dtype=np.uint8)
    fr_otro[10:80, 10:200] = 220
    fr_otro[300:460, 400:630] = 90
    escena = EscenaControlada()
    m = _medidor(tmp_path, escena=escena)
    t0 = MEDIODIA
    for i in range(4):                            # captura la huella «casa»
        m.procesar(fr_zonas, _tracks(BOX_LEJOS), ahora=t0 + i * 0.5)
    escena.veredicto = 'cambio'                   # el operador cambió canal
    for i in range(24):                           # 12 s en el canal AJENO
        m.procesar(fr_otro, _tracks(BOX_EN_AREA), ahora=t0 + 2.0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 0     # jamás reanudó por timer
    assert m._suspendido_hasta == float('inf')
    # Vuelve al canal correcto: reanuda (y cuenta) solo.
    for i in range(16):
        m.procesar(fr_zonas, _tracks(BOX_EN_AREA), ahora=t0 + 14.0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 1


def test_V31_el_refresco_no_adopta_la_entrada_del_vecino(tmp_path):
    """v3.1 (lo cazó la verificación de punta a punta): el acompañante, al
    refrescar «su» sitio, adoptaba por radio la entrada del comprador de
    al lado — y el fragmento del comprador quedaba exento del dedup y
    re-contaba."""
    m = _medidor(tmp_path)
    t0 = MEDIODIA
    for i in range(8):                            # comprador + acompañante
        tracks = {**_tracks([120, 120, 220, 460], tid=2),
                  **_tracks([200, 120, 300, 460], tid=3)}
        m.procesar(FRAME, tracks, ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 2     # los dos cuentan
    # El comprador (tid 2) se va; su track RENACE como tid 9 en su sitio,
    # con el acompañante aún al lado refrescando el suyo.
    for i in range(8):
        tracks = {**_tracks([120, 120, 220, 460], tid=9),
                  **_tracks([200, 120, 300, 460], tid=3)}
        m.procesar(FRAME, tracks, ahora=t0 + 5.0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 2     # el fragmento NO re-cuenta
    assert m._descartes.get('visita_dedup_espacial', 0) >= 1


def test_V31_confirmaciones_simultaneas_mismo_punto_cuentan_una(tmp_path):
    """v3.1: dos gestos que completaban en el MISMO frame y el mismo punto
    pasaban ambos (el dedup solo veía agarres ya confirmados). Dos muñecas
    en el mismo punto en <10 s es una oclusión de la pose, no dos agarres."""
    class PoseDoble:
        disponible = True

        def __init__(self):
            self.n = 0

        def munecas(self, frame, box):
            # AMBAS personas «hacen» el mismo gesto en el mismo punto (la
            # pose confundida por la oclusión): fuera 3 frames, dentro 6.
            self.n += 1
            fase_dentro = self.n > 6              # 2 personas x 3 frames
            return ([(150, 150, 0.9, 'der')] if fase_dentro
                    else [(400, 300, 0.9, 'der')])

    m = _medidor(tmp_path, pose=PoseDoble())
    t0 = MEDIODIA
    for i in range(9):
        tracks = {**_tracks([120, 120, 220, 460], tid=2),
                  **_tracks([125, 120, 225, 460], tid=3)}
        m.procesar(FRAME, tracks, ahora=t0 + i * 0.5)
    assert m.registro.area(0)['agarres'] == 1
    assert m._descartes.get('agarre_dedup_espacial', 0) >= 1


def test_V31_escenario_compuesto_de_tienda(tmp_path):
    """La verificación de punta a punta, como test permanente: caminante
    que pasa + comprador que se detiene y agarra UNA vez + acompañante +
    track partido. El libro final debe decir exactamente 2 visitas y 1
    agarre."""
    class PoseComprador:
        disponible = True
        fase = 'inicio'

        def munecas(self, frame, box):
            if int(box[0]) != 120:                # solo el comprador
                return []
            if self.fase == 'fuera':
                return [(400, 300, 0.9, 'der')]
            if self.fase == 'gesto':
                return [(150, 150, 0.9, 'der')]
            return []

    pose = PoseComprador()
    m = _medidor(tmp_path, pose=pose)
    t0 = MEDIODIA
    for i in range(28):
        tracks = {}
        if i < 12:                                # caminante cruza 6 s
            x = 90 + i * 25
            tracks[1] = {'class': -1, 'box': [x, 120, x + 100, 460],
                         'persistent_id': None}
        if i >= 4:                                # comprador (track partido en 20)
            tid_c = 2 if i < 20 else 9
            tracks[tid_c] = {'class': -1, 'box': [120, 120, 220, 460],
                             'persistent_id': None}
            pose.fase = ('fuera' if 6 <= i < 12
                         else 'gesto' if 12 <= i < 15
                         else 'fuera' if i >= 15 else 'inicio')
        if i >= 8:                                # acompañante
            tracks[3] = {'class': -1, 'box': [200, 120, 300, 460],
                         'persistent_id': None}
        m.procesar(FRAME, tracks, ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 2
    assert m.registro.area(0)['agarres'] == 1
    assert m._descartes.get('visita_movimiento', 0) > 0    # el caminante
    assert m._descartes.get('visita_dedup_espacial', 0) >= 1  # el fragmento


def test_V2_la_salud_declara_la_logica_que_corre(tmp_path):
    """Si «siguen los falsos», lo PRIMERO es verificar qué lógica corre:
    un servidor sin reiniciar sigue con la vieja. El bloque lo declara."""
    m = _medidor(tmp_path)
    bloque = m.procesar(FRAME, _tracks(BOX_EN_AREA), ahora=MEDIODIA)
    assert bloque['salud']['logica'] == 'gesto-v3-munecas'
    assert 'descartes' in bloque['salud']


# ── Regresiones de la revisión adversarial (13-ago-2026) ─────────────────
def test_arrastrar_un_vertice_interior_actualiza_el_poligono(tmp_path):
    """La firma vieja solo miraba p[0] y p[-1]: mover el vértice 1 o 2
    dejaba al servidor midiendo la geometría anterior en silencio."""
    m = _medidor(tmp_path)
    # El operador estira el vértice 1 (interior en la firma vieja) hasta
    # x=600: el área ahora también cubre BOX_LEJOS (x 500..560, y sup 300+).
    area_estirada = [[100, 100], [600, 100], [600, 420], [100, 200]]
    m.poner_zonas([area_estirada], ['Área 1'])
    t0 = MEDIODIA
    _estar(m, BOX_LEJOS, t0, 6)
    assert m.registro.area(0)['visitas'] == 1


def test_nivel_entre_filas_de_sondas_si_cuenta(tmp_path):
    """El muestreo de 9 puntos dejaba ciega a una banda fina de anaquel que
    cayera entre las filas de sondas de un bbox grande; la intersección
    real rectángulo-polígono no tiene ese agujero."""
    m = _medidor(tmp_path)
    # Banda de 130 px de alto (y 150..280) frente a un bbox 400x800 cuya
    # franja superior pone las filas de sondas en y=100, 340 y 580.
    banda = [[50, 150], [550, 150], [550, 280], [50, 280]]
    m.poner_zonas([banda], ['Área 1'])
    t0 = MEDIODIA
    _estar(m, [100, 100, 500, 900], t0, 6)
    assert m.registro.area(0)['visitas'] == 1


def test_cruce_de_medianoche_abre_jornada_nueva(tmp_path, reloj):
    """Conexión viva a las 00:00: los eventos de hoy no pueden inflar el
    JSON de ayer, y el archivo de hoy debe existir para la API.

    La frontera es DELIBERADA y la manda el reloj de PARED (`_hoy()`), no
    el `ahora` del frame (4-sep-2026): se cruza moviendo el reloj congelado
    de las 23:59:50 del martes a las 00:00:10 del miércoles, en hora local,
    con `ahora` avanzando a la par como en producción. Antes se stubeaba
    `_hoy` con dos literales; así se ejercita el `_hoy()` real."""
    vispera = time.mktime((2026, 9, 1, 23, 59, 50, 0, 0, -1))
    reloj.fijar(vispera)
    m = _medidor(tmp_path)
    assert m.registro._fecha == '2026-09-01'
    _estar(m, BOX_EN_AREA, vispera, 6)            # 23:59:50 .. 23:59:52.5
    assert m.registro.area(0)['visitas'] == 1

    reloj.fijar(vispera + 20.0)                   # 00:00:10 del miércoles
    m.procesar(FRAME, _tracks(BOX_LEJOS), ahora=vispera + 20.0)
    assert m.registro._fecha == '2026-09-02'
    assert m.registro.area(0)['visitas'] == 0     # jornada nueva, de cero
    # La visita de ayer quedó en el archivo de AYER, intacta.
    ayer = json.loads((tmp_path / 'cam-test' / '2026-09-01.json')
                      .read_text(encoding='utf-8'))
    assert ayer['areas']['0']['visitas'] == 1
    # Y hoy se puede volver a contar (cooldowns y dedup espacial limpios).
    _estar(m, BOX_EN_AREA, vispera + 30.0, 6)
    assert m.registro.area(0)['visitas'] == 1


def test_cooldowns_vencidos_se_podan(tmp_path):
    """Los pid prov: son infinitos: sin poda, los dicts de cooldown crecen
    toda la jornada."""
    m = _medidor(tmp_path, visita_cooldown_s=30.0)
    t0 = MEDIODIA
    _estar(m, BOX_EN_AREA, t0, 6)
    assert m._visita_ts                           # cooldown vivo
    # Mucho después (cooldown vencido de sobra), otro frame cualquiera.
    m.procesar(FRAME, _tracks(BOX_LEJOS), ahora=t0 + 200.0)
    assert not m._visita_ts                       # podado


def test_carpeta_segura_sin_traversal_ni_colisiones(tmp_path):
    from src.analityc.core.anaqueles_registro import _carpeta_nombre
    # `..` no puede escapar de la raíz.
    assert '..' not in _carpeta_nombre('..')
    assert _carpeta_nombre('..').startswith('cam_')
    # Dos ids que el saneo simple aplastaría al mismo nombre NO colisionan.
    assert _carpeta_nombre('cam 1') != _carpeta_nombre('cam:1')
    # Un id ya limpio se usa tal cual: el round-trip carpeta→API funciona.
    assert _carpeta_nombre('win-iVMS-4200') == 'win-iVMS-4200'
    assert _carpeta_nombre(_carpeta_nombre('cam:1')) == _carpeta_nombre('cam:1')


# ── Resumen de TODAS las cámaras de una vez (HITO 3) ─────────────────────
# El dashboard pedía la lista de cámaras y luego una petición por cada una,
# cada 30 s: con ocho pasillos, nueve viajes para pintar una sección.

def test_el_resumen_global_trae_todas_las_camaras(monkeypatch):
    from src.app import anaqueles_api

    datos = {
        'cam-1': {'fecha': '2026-08-18', 'grid_dim': [12, 24], 'areas': {
            '0': {'nombre': 'Anaquel 1', 'visitas': 5, 'agarres': 2,
                  'grid': [[0, 1], [2, 0]]}}},
        'cam-2': {'fecha': '2026-08-18', 'grid_dim': [12, 24], 'areas': {
            '0': {'nombre': 'Anaquel 7', 'visitas': 1, 'agarres': 0,
                  'grid': []}}},
    }
    monkeypatch.setattr(anaqueles_api, 'camaras_registradas',
                        lambda: ['cam-1', 'cam-2', 'cam-apagada'])
    monkeypatch.setattr(anaqueles_api, 'ultimo_registro',
                        lambda cam, fecha=None: datos.get(cam))

    r = anaqueles_api.resumen_de_todas(fecha='2026-08-18')

    assert r['total'] == 2
    assert [c['camera_id'] for c in r['camaras']] == ['cam-1', 'cam-2']
    # La rejilla viaja dentro: no hace falta una petición por área.
    assert r['camaras'][0]['areas'][0]['grid'] == [[0, 1], [2, 0]]
    assert r['camaras'][0]['total_agarres'] == 2


def test_una_camara_sin_datos_no_desaparece_en_silencio(monkeypatch):
    """«No midió hoy» es información; que se caiga de la lista, no."""
    from src.app import anaqueles_api

    monkeypatch.setattr(anaqueles_api, 'camaras_registradas',
                        lambda: ['cam-1', 'cam-apagada'])
    monkeypatch.setattr(anaqueles_api, 'ultimo_registro',
                        lambda cam, fecha=None: (
                            {'fecha': '2026-08-18', 'areas': {}}
                            if cam == 'cam-1' else None))

    r = anaqueles_api.resumen_de_todas(fecha='2026-08-18')
    assert r['sin_datos'] == ['cam-apagada']
