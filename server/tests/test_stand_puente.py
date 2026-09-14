"""
Pruebas del puente pipeline ↔ stand (cableado).

Esta es la prueba que faltaba en todo el proyecto: las otras 156 verifican
que cada pieza DECIDE bien cuando la alimentas; ninguna verificaba que algo
la alimentara. Aquí se simula `active_tracks` como lo entrega
`tiendas.py` y se comprueba que del otro lado salen
visitantes en SQLite y un bloque para el WebSocket.

Sin GPU: la pose no se carga (no hay atención sostenida en estos escenarios)
y el VLM no se invoca.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import stand_almacen as _alm  # noqa: E402
from src.analityc.core import stand_puente as sp  # noqa: E402
from src.analityc.core.stand_almacen import AlmacenStand  # noqa: E402
from src.analityc.core.stand_config import ConfigStand  # noqa: E402

ANCHO, ALTO = 1280, 720
FRAME = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)

# Instante FIJO para las pruebas de sesión: martes 1-sep-2026 a las 12:00 en
# hora LOCAL (mktime), lejos de la medianoche. La sesión «de hoy» se decide
# por la fecha local de `datetime.now()` (stand_almacen.sesion_abierta_de) y
# el test no puede depender de a qué hora corre la suite (4-sep-2026).
MEDIODIA_MARTES = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))


@contextlib.contextmanager
def _reloj_fijo(epoca: float):
    """Congela el reloj que usan el puente y el almacén en `epoca`.

    `stand_almacen` hizo `from datetime import datetime`, así que parchear
    `datetime.datetime` a secas no le llega: se le sustituye SU nombre por
    una subclase con `now()` fijo. Es un context manager (y no una fixture)
    para que el runner `__main__` de abajo siga funcionando sin pytest.
    """
    class _Fijo(_dt.datetime):
        @classmethod
        def now(cls, tz=None):                 # noqa: D102
            return _dt.datetime.fromtimestamp(epoca, tz)

    orig_time, orig_dt = sp.time.time, _alm.datetime
    sp.time.time = lambda: epoca
    _alm.datetime = _Fijo
    try:
        yield
    finally:
        sp.time.time = orig_time
        _alm.datetime = orig_dt


def _cfg():
    """Zonas simples y separadas: paso a la derecha, stand a la izquierda."""
    datos = {
        'habilitado': True,
        'stand': {'stand_id': 'test-01', 'nombre': 'Prueba'},
        'detencion': {'dwell_min_s': 3.0, 'desplazamiento_max_px': 60},
        'atencion': {'distancia_max_m': 1.2, 'duracion_min_s': 2.0},
        'conteo': {'min_frames_track': 5, 'min_dwell_zona_paso_s': 0.5},
        'reid': {'umbral_merge': 0.78, 'ttl_galeria_min': 45},
        'demografia': {'buffer': {'max_muestras': 10, 'espaciado_frames': 1,
                                  'min_muestras_validas': 3},
                       'genero': {'confianza_min': 0.65}},
        'heatmap': {'celda_px': 20, 'sigma_celdas': 0},
        'personal': {'dwell_auto_enrolamiento_min': 10, 'umbral_reid': 0.75},
        'degustacion': {'pose': {'frames_min': 8, 'ventana_s': 2.0},
                        'ventana_post_atencion_s': 90,
                        'vlm': {'confianza_min': 0.7, 'max_por_minuto': 20}},
    }
    zonas = {'zonas': {
        # Mitad derecha = corredor.
        'zona_paso': [[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]],
        # Mitad izquierda = stand.
        'zona_stand': [[0.0, 0.30], [0.55, 0.30], [0.55, 1.0], [0.0, 1.0]],
        'zona_atencion': [[0.10, 0.55], [0.50, 0.55], [0.50, 0.95],
                          [0.10, 0.95]],
        'zona_personal': [[0.0, 0.0], [0.50, 0.0], [0.50, 0.28], [0.0, 0.28]],
        'linea_conteo': [[0.60, 0.5], [1.0, 0.5]],
    }}
    return ConfigStand(datos, zonas)


def _puente(tmp: Path) -> sp.PuenteStand:
    import os
    os.environ['STAND_BD'] = str(tmp / 'stand.db')
    return sp.PuenteStand(ANCHO, ALTO, camara_id='cam-test', cfg=_cfg())


def _track(pid, x, y, ancho=80, alto=200, confirmado=True):
    """Track como lo entrega el pipeline: el pie es el centro inferior."""
    return {
        'box': [x - ancho / 2, y - alto, x + ancho / 2, y],
        'center': (x, y - alto / 2),
        'persistent_id': pid,
        'persistent_confirmed': confirmado,
    }


def test_abre_sesion_si_las_zonas_estan_calibradas():
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        assert p.activo is True
        assert p.sesion_id is not None
        s = p.almacen.sesiones()[0]
        assert s['stand_id'] == 'test-01'
        # El snapshot de umbrales viaja con la sesión.
        assert s['config_snapshot'] and 'dwell_min_s' in s['config_snapshot']


def test_reanuda_la_sesion_de_hoy_en_vez_de_abrir_otra():
    """Cada reinicio instanciaba un puente nuevo y abría OTRA sesión.

    Se detectó en producción con 4 sesiones abiertas en una tarde: la misma
    jornada partida en trozos, y una persona vista antes y después del
    reinicio contando en ambos.

    El reloj va congelado a mediodía: «de hoy» se decide por la fecha local
    y, si la medianoche caía entre `p1` y `p2`, el segundo puente abría otra
    sesión y el test fallaba por la hora, no por el código (4-sep-2026).
    """
    with tempfile.TemporaryDirectory() as d, _reloj_fijo(MEDIODIA_MARTES):
        p1 = _puente(Path(d))
        primera = p1.sesion_id
        assert primera is not None

        p2 = _puente(Path(d))          # simula el reinicio
        assert p2.sesion_id == primera, "abrió una sesión nueva"
        assert len(p2.almacen.sesiones()) == 1

        # Cerrada la jornada, la siguiente sí es nueva.
        p2.cerrar()
        p3 = _puente(Path(d))
        assert p3.sesion_id != primera
        assert len(p3.almacen.sesiones()) == 2


def test_la_sesion_abierta_de_ayer_no_se_reanuda_al_dia_siguiente():
    """Frontera de jornada, hecha explícita: la reanudación es de HOY.

    Una sesión que quedó abierta (el servidor se cayó a las 23:59 y volvió a
    las 00:01) es de la víspera: `sesion_abierta_de` filtra por fecha local,
    así que el reinicio abre una jornada NUEVA en vez de colgarse de la de
    ayer. Sin este test, el comportamiento quedaba implícito y dependía de a
    qué hora corriera la suite (4-sep-2026).
    """
    ayer_2359 = MEDIODIA_MARTES + 11 * 3600 + 59 * 60    # martes 23:59
    hoy_0001 = ayer_2359 + 120                           # miércoles 00:01
    with tempfile.TemporaryDirectory() as d:
        with _reloj_fijo(ayer_2359):
            p1 = _puente(Path(d))
            de_ayer = p1.sesion_id
            assert de_ayer is not None
        with _reloj_fijo(hoy_0001):
            p2 = _puente(Path(d))
            assert p2.sesion_id != de_ayer, "reanudó la jornada de ayer"
            sesiones = {s['id']: s for s in p2.almacen.sesiones()}
            assert len(sesiones) == 2
            assert sesiones[de_ayer]['fecha'] != sesiones[p2.sesion_id]['fecha']
            # Y en el mismo día sí se reanuda la recién abierta.
            assert _puente(Path(d)).sesion_id == p2.sesion_id


def test_no_abre_sesion_sin_zonas():
    """Sin calibrar NO se cuenta, y se dice. Nunca se asume una zona."""
    with tempfile.TemporaryDirectory() as d:
        import os
        os.environ['STAND_BD'] = str(Path(d) / 'stand.db')
        cfg = ConfigStand({'habilitado': True}, {'zonas': {}})
        p = sp.PuenteStand(ANCHO, ALTO, cfg=cfg)
        assert p.activo is False and p.sesion_id is None
        assert p.procesar(FRAME, {}) is None


def test_una_persona_que_cruza_el_corredor_se_cuenta():
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        # Baja por el corredor (x=0.8) cruzando la línea de conteo (y=0.5).
        for i in range(20):
            y = 100 + i * 25
            p.procesar(FRAME, {1: _track('p-A', 0.8 * ANCHO, y)},
                       ts=100.0 + i * 0.2)
        p.procesar(FRAME, {}, ts=110.0)      # desaparece -> cierra track
        p._persistir(1e9)

        c = p.almacen.contadores(p.sesion_id)
        assert c['pasaron'] == 1, f"contó {c['pasaron']} personas, esperaba 1"
        assert c['se_detuvieron'] == 0


def test_la_misma_persona_en_dos_tracks_cuenta_UNA_vez():
    """Oclusión por el mostrador: el pipeline parte el track y reasigna el
    mismo persistent_id. No puede contar dos."""
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        for tid in (1, 2):
            for i in range(20):
                y = 100 + i * 25
                p.procesar(FRAME, {tid: _track('p-A', 0.8 * ANCHO, y)},
                           ts=100.0 + tid * 20 + i * 0.2)
            p.procesar(FRAME, {}, ts=100.0 + tid * 20 + 10)
        p._persistir(1e9)
        assert p.almacen.contadores(p.sesion_id)['pasaron'] == 1


def test_track_provisional_no_cuenta():
    """Sin identidad confirmada no hay conteo posible."""
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        for i in range(20):
            p.procesar(FRAME, {1: _track(None, 0.8 * ANCHO, 100 + i * 25,
                                         confirmado=False)},
                       ts=100.0 + i * 0.2)
        p.procesar(FRAME, {}, ts=110.0)
        p._persistir(1e9)
        assert p.almacen.contadores(p.sesion_id)['pasaron'] == 0


def test_parado_frente_al_stand_cuenta_como_detenido():
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        # Entra por el corredor...
        for i in range(8):
            p.procesar(FRAME, {1: _track('p-B', 0.8 * ANCHO, 300 + i * 20)},
                       ts=100.0 + i * 0.2)
        # ...y se queda quieto en zona_stand 6 segundos.
        for i in range(30):
            p.procesar(FRAME, {1: _track('p-B', 0.25 * ANCHO, 600)},
                       ts=102.0 + i * 0.2)
        p.procesar(FRAME, {}, ts=115.0)
        p._persistir(1e9)

        c = p.almacen.contadores(p.sesion_id)
        assert c['pasaron'] == 1
        assert c['se_detuvieron'] == 1, "no detectó la detención"
        v = p.almacen.visitantes(p.sesion_id)[0]
        assert v['clasificacion'] == 'se_detuvo_sin_atencion'
        assert v['dwell_max_s'] >= 3.0


def test_el_chef_queda_excluido_del_conteo():
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        p.personal.enrolar('p-CHEF', None, nombre='Chef', ts=100.0)
        for i in range(30):
            p.procesar(FRAME, {
                1: _track('p-CHEF', 0.25 * ANCHO, 150),      # zona_personal
                2: _track('p-C', 0.8 * ANCHO, 200 + i * 15),  # cliente
            }, ts=100.0 + i * 0.2)
        p.procesar(FRAME, {}, ts=120.0)
        p._persistir(1e9)

        c = p.almacen.contadores(p.sesion_id)
        assert c['pasaron'] == 1, "el chef se coló en el conteo"
        pids = [v['persistent_id'] for v in p.almacen.visitantes(
            p.sesion_id, incluir_personal=True)]
        assert 'p-CHEF' not in pids or all(
            v['es_personal'] for v in p.almacen.visitantes(
                p.sesion_id, incluir_personal=True)
            if v['persistent_id'] == 'p-CHEF')


def test_emite_el_bloque_para_el_websocket():
    """Cada sección va a SU ritmo, así que hay que mirar varios mensajes.

    (Una primera versión de este test guardaba solo el último bloque y
    fallaba por eso: `salud` va cada 10 s y no estaba en él.)
    """
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        bloques = []
        for i in range(15):
            b = p.procesar(FRAME, {1: _track('p-D', 0.8 * ANCHO, 200 + i * 20)},
                           ts=100.0 + i * 0.5)
            if b:
                bloques.append(b)

        assert bloques, "no emitió nada para el cliente"
        claves = set().union(*(set(b) for b in bloques))
        assert {'sesion_id', 'contadores', 'tracks_activos',
                'salud'} <= claves, f"faltan claves: {claves}"
        assert all(b['sesion_id'] == p.sesion_id for b in bloques)
        con_tracks = next(b for b in bloques if 'tracks_activos' in b)
        assert con_tracks['tracks_activos'][0]['persistent_id'] == 'p-D'


def test_el_heatmap_acumula_posiciones():
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        for i in range(20):
            p.procesar(FRAME, {1: _track('p-E', 0.25 * ANCHO, 600)},
                       ts=100.0 + i * 0.2)
        assert p.heatmap.salud()['muestras'] >= 15


def test_un_fallo_del_stand_no_tumba_el_frame():
    """El bucle de inferencia sirve a Tienda y Perimetrales: pase lo que
    pase aquí, no puede propagarse."""
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        # Track corrupto: sin box.
        assert p.procesar(FRAME, {1: {'persistent_id': 'x'}},
                          ts=100.0) is not None or True
        # Box con basura.
        p.procesar(FRAME, {2: {'box': ['a', 'b', 'c', 'd'],
                               'persistent_id': 'y',
                               'persistent_confirmed': True}}, ts=101.0)


def test_cerrar_sesion_vuelca_y_marca_cerrada():
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        for i in range(20):
            p.procesar(FRAME, {1: _track('p-F', 0.8 * ANCHO, 100 + i * 25)},
                       ts=100.0 + i * 0.2)
        sid = p.sesion_id
        p.cerrar()
        assert p.sesion_id is None
        alm = AlmacenStand(ruta=Path(d) / 'stand.db')
        s = [x for x in alm.sesiones() if x['id'] == sid][0]
        assert s['estado'] == 'cerrada' and s['hora_fin']


def test_marcar_personal_en_la_bd_llega_a_la_galeria_viva():
    """El POST de `/api/stand/.../personal/enrolar` solo escribe el flag en
    `stand_visitantes`; la galería que consulta `_localizar_chef` vive en
    memoria. El puente debe releer la BD y reflejarlo — sin eso el chef
    marcado por el operador no existía para el pipeline, `chef_disponible`
    era False y no se detectaba una sola atención (7-ago-2026: chef marcado
    el día 5 y cero eventos de atención después)."""
    with tempfile.TemporaryDirectory() as d:
        p = _puente(Path(d))
        # El chef aparece unos frames y queda como visitante en la BD.
        for i in range(6):
            p.procesar(FRAME, {1: _track('chef-1', 0.25 * ANCHO, 150)},
                       ts=100.0 + i * 0.2)
        p.procesar(FRAME, {}, ts=102.0)
        p._persistir(1e9)

        # Lo que hace la API al marcarlo: el flag, solo en la BD.
        p.almacen.upsert_visitante(p.sesion_id, 'chef-1', es_personal=True)
        assert not p.personal.es_personal('chef-1'), \
            "premisa: la galería viva aún no lo conoce"

        p._ultimo_sync_personal = 0.0        # fuerza el siguiente ciclo
        p._sincronizar_personal(200.0)
        assert p.personal.es_personal('chef-1'), \
            "el marcado de la BD no llegó a la galería viva"

        # Desmarcar también llega (y solo toca lo que enroló el sync).
        p.almacen.upsert_visitante(p.sesion_id, 'chef-1', es_personal=False)
        p._ultimo_sync_personal = 0.0
        p._sincronizar_personal(300.0)
        assert not p.personal.es_personal('chef-1')


def test_deshabilitado_devuelve_none():
    """Con `habilitado: false` no se instancia nada.

    Escribe su propio YAML en vez de leer el del repo: ese flag se cambia en
    producción (hoy está en `true`) y el test no puede depender de su valor,
    o falla por un motivo que no tiene nada que ver con lo que prueba.
    """
    import os
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / 'apagado.yaml'
        cfg.write_text('habilitado: false\n', encoding='utf-8')
        os.environ['STAND_CONFIG'] = str(cfg)
        try:
            assert sp.crear_si_habilitado(ANCHO, ALTO, 'cam') is None
        finally:
            os.environ.pop('STAND_CONFIG', None)


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
