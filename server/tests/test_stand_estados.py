"""
Pruebas de la máquina de estados del visitante (HITO 7).

Cubre los cuatro estados terminales de la máquina con trayectorias
sintéticas, más el quinto desenlace posible de un track —`descartado`, que
decide `stand_identidad` antes de llegar aquí— y la invariante que impide un
embudo invertido:

    degustaron <= atendidos <= se_detuvieron <= pasaron

Las coordenadas van en METROS (`usar_metros=True`) para que los umbrales del
Anexo A se lean tal cual y los tests no dependan de la resolución.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_config import ConfigStand  # noqa: E402
from src.analityc.core.stand_estados import (  # noqa: E402
    CLASIFICACION_DE, Estado, MaquinaVisitante)


def _cfg(**over):
    datos = {
        'detencion': {'dwell_min_s': 3.0, 'desplazamiento_max_m': 0.8},
        'paso_largo': {'velocidad_min_ms': 0.6},
        'atencion': {'distancia_max_m': 1.2, 'duracion_min_s': 2.0},
    }
    for ruta, valor in over.items():
        nodo = datos
        partes = ruta.split('.')
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = valor
    return ConfigStand(datos, {})


def _maquina(pid='p1', eventos=None):
    cb = None
    if eventos is not None:
        def cb(tipo, zona, payload):
            eventos.append((tipo, zona, payload))
    return MaquinaVisitante(pid, _cfg(), al_evento=cb, usar_metros=True)


# ── Los cuatro estados terminales ───────────────────────────────────────
def test_terminal_paso_de_largo():
    m = _maquina()
    # Cruza el corredor a 1 m/s sin entrar nunca a zona_stand.
    for i in range(10):
        m.actualizar(ts=i * 0.5, punto_pie=(i * 0.5, 0.0), en_zona_paso=True)
    r = m.cerrar(ts=5.0)
    assert r.estado == Estado.PASO_DE_LARGO
    assert r.clasificacion == 'paso_de_largo'
    assert r.se_detuvo is False and r.fue_atendido is False
    assert r.velocidad_media is not None and r.velocidad_media > 0.6


def test_terminal_se_retiro_sin_atencion():
    m = _maquina()
    # 5 s parado en zona_stand, sin chef cerca.
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True)
    r = m.cerrar(ts=5.5)
    assert r.estado == Estado.SE_RETIRO_SIN_ATENCION
    assert r.clasificacion == 'se_detuvo_sin_atencion'
    assert r.se_detuvo is True and r.fue_atendido is False
    assert r.dwell_max_s >= 3.0


def test_terminal_atendido_sin_degustar():
    m = _maquina()
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True)
    # 3 s con el chef a 0.8 m, en zona de atención.
    for i in range(7):
        m.actualizar(ts=5.5 + i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_stand=True, en_zona_atencion=True,
                     distancia_chef=0.8, chef_disponible=True)
    r = m.cerrar(ts=9.0)
    assert r.estado == Estado.ATENDIDO_SIN_DEGUSTAR
    assert r.clasificacion == 'atendido_sin_degustar'
    assert r.fue_atendido is True and r.degusto is False
    assert r.duracion_atencion_s >= 2.0


def test_terminal_degusto():
    m = _maquina()
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True)
    for i in range(7):
        m.actualizar(ts=5.5 + i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_stand=True, en_zona_atencion=True,
                     distancia_chef=0.8, chef_disponible=True)
    m.marcar_degustacion(ts=10.0, confianza=0.88, metodo='pose+vlm')
    r = m.cerrar(ts=12.0)
    assert r.estado == Estado.DEGUSTO
    assert r.clasificacion == 'degusto'
    assert r.confianza_degustacion == 0.88
    assert r.metodo_deteccion_degustacion == 'pose+vlm'


# ── Las reglas que evitan falsos positivos ──────────────────────────────
def test_caminar_despacio_por_la_zona_no_es_detenerse():
    """Sin la comprobación de desplazamiento, cruzar despacio contaría."""
    m = _maquina()
    # 5 s dentro de zona_stand pero recorriendo 5 metros.
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(i * 0.5, 0.0),
                     en_zona_paso=True, en_zona_stand=True)
    r = m.cerrar(ts=5.5)
    assert r.se_detuvo is False, "se contó como detenido caminando"
    assert r.clasificacion == 'paso_de_largo'


def test_chef_lejos_no_es_atencion():
    m = _maquina()
    for i in range(21):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True,
                     en_zona_atencion=True,
                     distancia_chef=3.0,          # > 1.2 m
                     chef_disponible=True)
    r = m.cerrar(ts=10.5)
    assert r.fue_atendido is False


def test_chef_no_disponible_no_es_atencion():
    """Cerca del mostrador pero el chef está atendiendo a otro."""
    m = _maquina()
    for i in range(21):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True,
                     en_zona_atencion=True, distancia_chef=0.5,
                     chef_disponible=False)
    r = m.cerrar(ts=10.5)
    assert r.fue_atendido is False
    assert r.clasificacion == 'se_detuvo_sin_atencion'


def test_atencion_acumula_no_exige_continuidad():
    """El chef se gira a cortar y vuelve: sigue siendo una atención."""
    m = _maquina()
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True)
    t = 5.5
    for _ciclo in range(3):                  # 3 tramos de 1 s, con pausas
        for i in range(3):
            m.actualizar(ts=t + i * 0.5, punto_pie=(2.0, 1.0),
                         en_zona_stand=True, en_zona_atencion=True,
                         distancia_chef=0.7, chef_disponible=True)
        t += 3.0
        m.actualizar(ts=t, punto_pie=(2.0, 1.0), en_zona_stand=True)
    r = m.cerrar(ts=t + 1)
    assert r.fue_atendido is True


# ── Monotonía ───────────────────────────────────────────────────────────
def test_degustacion_enciende_los_flags_anteriores():
    """Aunque no hayamos visto la atención, el embudo debe cuadrar."""
    m = _maquina()
    m.actualizar(ts=0.0, punto_pie=(1.0, 1.0), en_zona_paso=True)
    m.marcar_degustacion(ts=1.0, confianza=0.9, metodo='vlm')
    r = m.cerrar(ts=2.0)
    assert r.degusto and r.fue_atendido and r.se_detuvo


def test_los_flags_nunca_se_apagan():
    """El embudo mide lo que ocurrió, no el estado final."""
    m = _maquina()
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True)
    assert m.se_detuvo is True
    # Se va: sale de todas las zonas.
    for i in range(10):
        m.actualizar(ts=6.0 + i * 0.5, punto_pie=(10.0 + i, 5.0))
    assert m.se_detuvo is True, "el flag se apagó al salir de la zona"
    assert m.cerrar(ts=12.0).clasificacion == 'se_detuvo_sin_atencion'


def test_marcar_degustacion_es_idempotente():
    m = _maquina()
    m.actualizar(ts=0.0, punto_pie=(1.0, 1.0), en_zona_paso=True)
    m.marcar_degustacion(ts=1.0, confianza=0.9, metodo='vlm')
    m.marcar_degustacion(ts=5.0, confianza=0.5, metodo='pose')
    r = m.cerrar(ts=6.0)
    assert r.confianza_degustacion == 0.9      # gana la primera
    assert r.ts_degustacion == 1.0


def test_invariante_del_embudo_sobre_una_poblacion():
    """degustaron <= atendidos <= se_detuvieron <= pasaron, siempre."""
    resumenes = []

    # 5 pasan de largo.
    for k in range(5):
        m = _maquina(f'largo{k}')
        for i in range(10):
            m.actualizar(ts=i * 0.5, punto_pie=(i * 0.5, 0.0),
                         en_zona_paso=True)
        resumenes.append(m.cerrar(ts=5.0))

    # 3 se detienen sin atención.
    for k in range(3):
        m = _maquina(f'det{k}')
        for i in range(11):
            m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                         en_zona_paso=True, en_zona_stand=True)
        resumenes.append(m.cerrar(ts=5.5))

    # 2 atendidos, 1 de ellos degusta.
    for k in range(2):
        m = _maquina(f'at{k}')
        for i in range(11):
            m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                         en_zona_paso=True, en_zona_stand=True)
        for i in range(7):
            m.actualizar(ts=5.5 + i * 0.5, punto_pie=(2.0, 1.0),
                         en_zona_stand=True, en_zona_atencion=True,
                         distancia_chef=0.8, chef_disponible=True)
        if k == 0:
            m.marcar_degustacion(ts=10.0, confianza=0.9, metodo='vlm')
        resumenes.append(m.cerrar(ts=11.0))

    pasaron = len(resumenes)
    detuvieron = sum(1 for r in resumenes if r.se_detuvo)
    atendidos = sum(1 for r in resumenes if r.fue_atendido)
    degustaron = sum(1 for r in resumenes if r.degusto)

    assert (pasaron, detuvieron, atendidos, degustaron) == (10, 5, 2, 1)
    assert degustaron <= atendidos <= detuvieron <= pasaron


# ── Trazabilidad y contratos ────────────────────────────────────────────
def test_cada_transicion_emite_su_evento():
    eventos = []
    m = _maquina(eventos=eventos)
    for i in range(11):
        m.actualizar(ts=i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_paso=True, en_zona_stand=True)
    for i in range(7):
        m.actualizar(ts=5.5 + i * 0.5, punto_pie=(2.0, 1.0),
                     en_zona_stand=True, en_zona_atencion=True,
                     distancia_chef=0.8, chef_disponible=True)
    m.marcar_degustacion(ts=10.0, confianza=0.9, metodo='vlm')
    m.cerrar(ts=11.0)

    tipos = [e[0] for e in eventos]
    for esperado in ('entrada_escena', 'entrada_zona_stand', 'detencion',
                     'inicio_atencion', 'degustacion', 'salida_escena'):
        assert esperado in tipos, f"falta el evento {esperado}"


def test_campos_visitante_encajan_con_el_almacen():
    from src.analityc.core.stand_almacen import AlmacenStand
    import inspect
    m = _maquina()
    m.actualizar(ts=0.0, punto_pie=(1.0, 1.0), en_zona_paso=True)
    r = m.cerrar(ts=1.0)
    campos = m.campos_visitante(r)
    # Todos deben ser aceptados por upsert_visitante (que valida nombres).
    fuente = inspect.getsource(AlmacenStand.upsert_visitante)
    for clave in campos:
        assert f"'{clave}'" in fuente, f"{clave} no existe en stand_visitantes"


def test_todas_las_clasificaciones_estan_mapeadas():
    for estado in (Estado.PASO_DE_LARGO, Estado.SE_RETIRO_SIN_ATENCION,
                   Estado.ATENDIDO_SIN_DEGUSTAR, Estado.DEGUSTO):
        assert estado in CLASIFICACION_DE


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
