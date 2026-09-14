"""
Pruebas del índice de capturas (HITO 3): leer deja de costar lo que pesa
el disco.

El índice es una CACHÉ reconciliada, no una segunda fuente de verdad: el disco
sigue mandando. Por eso lo que más se protege aquí es que el espejo no mienta.

  - lo nuevo entra, lo borrado sale, y lo que cambió se vuelve a leer
    (la demografía se escribe MÁS TARDE, cuando converge, y el reanálisis con
    VLM también reescribe el sidecar: «está o no está» no basta);
  - una foto sin sidecar, o un sidecar sin foto, no se cuela;
  - el coste de contar y de listar NO crece con las capturas acumuladas;
  - si la base falla, la galería sigue saliendo por el camino lento en vez de
    quedarse en blanco.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.analityc.core import tienda_capturas as tc  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402


@pytest.fixture
def capturas(tmp_path, monkeypatch):
    """Carpeta de capturas y base, ambas de usar y tirar."""
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_DIR',
                        str(tmp_path / 'captures'))
    (tmp_path / 'captures' / 'persons').mkdir(parents=True)
    (tmp_path / 'captures' / 'faces').mkdir(parents=True)
    ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    return tmp_path / 'captures'


def _escribir(base: Path, stem: str, *, meta=None, con_foto=True,
              con_rostro=False, con_sidecar=True):
    if con_foto:
        (base / 'persons' / f'{stem}.jpg').write_bytes(b'\xff\xd8jpg')
    if con_sidecar:
        datos = {'timestamp': stem[:15], 'camera': 'iVMS-4200'}
        datos.update(meta or {})
        (base / 'persons' / f'{stem}.json').write_text(
            json.dumps(datos), encoding='utf-8')
    if con_rostro:
        (base / 'faces' / f'{stem}.jpg').write_bytes(b'\xff\xd8cara')


#: 1 ms en nanosegundos: la tolerancia con la que `sincronizar` da dos mtimes
#: por iguales (`abs(anterior - mtime) < 0.001`, tienda_capturas.py).
UN_MS_NS = 1_000_000


def _mtime_sidecar(base: Path, stem: str) -> int:
    return (base / 'persons' / f'{stem}.json').stat().st_mtime_ns


def _fechar_sidecar(base: Path, stem: str, mtime_ns: int) -> int:
    """Fija el mtime del sidecar a un instante EXACTO y devuelve el que quedó.

    El índice decide «cambió / no cambió» por el mtime (`sincronizar`, con
    tolerancia de 1 ms). La prueba de reescritura confiaba en un
    `time.sleep(0.01)` para que el reloj del disco avanzara, y eso depende del
    sistema de ficheros: FAT/exFAT redondean a 2 s, SMB/DrvFs y HFS+ a 1 s,
    así que una reescritura 10 ms después conservaba el mtime y la prueba
    fallaba sin que nada estuviera roto (4-sep-2026). Con `os.utime` el
    instante es el que la prueba pide, RELATIVO al de la foto, y da igual lo
    que tarde la prueba entre una escritura y la otra.

    Se devuelve el mtime que el disco dejó de verdad: en un sistema de
    ficheros grueso puede no ser el pedido, y la prueba tiene que saberlo
    (ver `_saltar_si_el_disco_redondeo`).
    """
    ruta = base / 'persons' / f'{stem}.json'
    os.utime(ruta, ns=(ruta.stat().st_atime_ns, int(mtime_ns)))
    return ruta.stat().st_mtime_ns


def _saltar_si_el_disco_redondeo(foto_ns: int, sidecar_ns: int) -> None:
    """Si el disco no pudo representar el salto pedido, la prueba no prueba
    nada: mejor un `skip` que diga por qué que un fallo que no lo es (o un
    pase por casualidad, si redondeó hacia arriba)."""
    if sidecar_ns - foto_ns < UN_MS_NS:
        pytest.skip('el sistema de ficheros de tmp_path redondea el mtime '
                    f'(quedó a {sidecar_ns - foto_ns} ns de la foto): no '
                    'puede representar una reescritura milisegundos después')


# ── Reconciliación ───────────────────────────────────────────────────────

def test_lo_nuevo_entra_en_el_indice(capturas):
    _escribir(capturas, '20260818_101500_cam1_t1',
              meta={'device_id': 'win-cam1', 'gender': 'Mujer'})
    info = tc.sincronizar(force=True)
    assert info['nuevas'] == 1
    d = tc.listar()
    assert d['total'] == 1
    assert d['capturas'][0]['gender'] == 'Mujer'
    assert d['capturas'][0]['device_id'] == 'win-cam1'


def test_un_sidecar_reescrito_se_vuelve_a_leer(capturas):
    """La demografía llega DESPUÉS: si el índice no la recoge, la galería se
    queda para siempre con «sin demografía».

    En producción la reescritura va en la MISMA vuelta del bucle de frames que
    la foto: `_capture_established_tracks` escribe el sidecar y, acto seguido,
    `_update_captures_demographics` lo reescribe en cuanto la demografía
    converge (tiendas.py / tiendas_fotos.py). O sea, decenas o centenas de
    milisegundos después de la foto, no segundos. Por eso la reescritura se
    fecha 10 ms tras la foto — lo que el antiguo `time.sleep(0.01)` intentaba
    — y no más tarde: si alguien «relajara» la tolerancia de 1 ms de
    `sincronizar` a medio segundo, la galería se quedaría con «sin demografía»
    y esta prueba tiene que cazarlo (4-sep-2026).
    """
    stem = '20260818_101500_cam1_t1'
    _escribir(capturas, stem)
    foto_ns = _mtime_sidecar(capturas, stem)
    tc.sincronizar(force=True)
    assert tc.listar()['capturas'][0]['gender'] is None

    _escribir(capturas, stem, meta={'gender': 'Hombre', 'age_range': '36-50'})
    sidecar_ns = _fechar_sidecar(capturas, stem, foto_ns + 10 * UN_MS_NS)
    _saltar_si_el_disco_redondeo(foto_ns, sidecar_ns)
    info = tc.sincronizar(force=True)

    assert info['actualizadas'] == 1
    fila = tc.listar()['capturas'][0]
    assert fila['gender'] == 'Hombre' and fila['age_range'] == '36-50'


def test_dos_milisegundos_despues_ya_es_otro_sidecar(capturas):
    """Frontera de la tolerancia: `sincronizar` da por iguales dos mtimes a
    menos de 1 ms, y nada más. Reescrito 2 ms después de la foto, el sidecar
    YA cuenta como cambio. Aquí se pinza el 1 ms con precisión; la prueba de
    arriba lo hace a 10 ms, la distancia real de producción (4-sep-2026)."""
    stem = '20260818_101500_cam1_t1'
    _escribir(capturas, stem)
    foto_ns = _mtime_sidecar(capturas, stem)
    tc.sincronizar(force=True)

    _escribir(capturas, stem, meta={'gender': 'Mujer'})
    sidecar_ns = _fechar_sidecar(capturas, stem, foto_ns + 2 * UN_MS_NS)
    _saltar_si_el_disco_redondeo(foto_ns, sidecar_ns)
    info = tc.sincronizar(force=True)

    assert info['actualizadas'] == 1
    assert tc.listar()['capturas'][0]['gender'] == 'Mujer'


def test_un_sidecar_con_el_mismo_mtime_no_se_reabre(capturas):
    """El mtime es LA señal de cambio: es lo que hace barato reconciliar
    20.000 sidecars cada 3 s. Si alguien «arreglara» el índice reparseando
    todo, esta prueba lo caza; y deja escrito que las dos anteriores no pasan
    por suerte del reloj del disco, sino porque el mtime cambió
    (4-sep-2026)."""
    stem = '20260818_101500_cam1_t1'
    _escribir(capturas, stem)
    original_ns = _mtime_sidecar(capturas, stem)
    tc.sincronizar(force=True)

    _escribir(capturas, stem, meta={'gender': 'Hombre'})
    _fechar_sidecar(capturas, stem, original_ns)            # mismo instante
    info = tc.sincronizar(force=True)

    assert info['actualizadas'] == 0 and info['nuevas'] == 0
    assert tc.listar()['capturas'][0]['gender'] is None, \
        'con el mismo mtime el índice no vuelve a abrir el sidecar'


def test_lo_que_desaparece_del_disco_sale_del_indice(capturas):
    stem = '20260818_101500_cam1_t1'
    _escribir(capturas, stem)
    tc.sincronizar(force=True)
    assert tc.total() == 1

    (capturas / 'persons' / f'{stem}.jpg').unlink()
    (capturas / 'persons' / f'{stem}.json').unlink()
    info = tc.sincronizar(force=True)

    assert info['borradas'] == 1
    assert tc.total() == 0


def test_un_sidecar_sin_foto_no_se_cuela(capturas):
    """La foto manda: un JSON huérfano no es una captura."""
    _escribir(capturas, '20260818_101500_cam1_t1', con_foto=False)
    tc.sincronizar(force=True)
    assert tc.total() == 0


def test_un_sidecar_ilegible_no_tumba_la_sincronizacion(capturas):
    _escribir(capturas, '20260818_101500_cam1_t1')
    (capturas / 'persons' / '20260818_101600_cam1_t2.jpg').write_bytes(b'x')
    (capturas / 'persons' / '20260818_101600_cam1_t2.json').write_text(
        '{roto', encoding='utf-8')
    tc.sincronizar(force=True)
    assert tc.total() == 2               # entra, pero sin metadatos


def test_la_sincronizacion_va_amortiguada(capturas):
    _escribir(capturas, '20260818_101500_cam1_t1')
    tc.sincronizar(force=True)
    _escribir(capturas, '20260818_101600_cam1_t2')
    assert tc.sincronizar()['omitido'] == 1, 'no re-enumera en cada petición'


# ── Consultas ────────────────────────────────────────────────────────────

def test_las_mas_recientes_primero_y_paginadas(capturas):
    for i in range(5):
        _escribir(capturas, f'20260818_1015{i:02d}_cam1_t{i}')
    tc.sincronizar(force=True)

    primera = tc.listar(limite=2)
    assert primera['total'] == 5
    assert [c['stem'][-2:] for c in primera['capturas']] == ['t4', 't3']

    segunda = tc.listar(limite=2, offset=2)
    assert [c['stem'][-2:] for c in segunda['capturas']] == ['t2', 't1']


def test_se_puede_acotar_por_ventana(capturas, monkeypatch):
    """`desde`/`hasta` son JORNADAS (`jornada_de`), no días del calendario.

    Aquí la jornada es el día natural (corte a las 00:00) para que los
    literales de abajo signifiquen lo que parecen. Se fija a mano y no se
    hereda: `ELDE_TIENDA_CORTE_JORNADA_H` la lee `tienda_almacen` al
    importarse, y en una máquina con corte a las 22 las 12:00 del 18 caen en
    la jornada del 17 y la primera ventana daba 0 (4-sep-2026). El caso del
    corte movido tiene su propia prueba justo debajo.
    """
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    _escribir(capturas, '20260817_120000_cam1_t1')
    _escribir(capturas, '20260818_120000_cam1_t2')
    tc.sincronizar(force=True)

    assert tc.listar(desde='2026-08-18', hasta='2026-08-18')['total'] == 1
    assert tc.listar(desde='2026-08-17', hasta='2026-08-18')['total'] == 2


def test_la_ventana_sigue_la_jornada_de_la_tienda_no_el_calendario(capturas,
                                                                  monkeypatch):
    """Con corte a las 22:00, la captura de las 00:30 del martes es de la
    jornada del lunes — igual que las entradas y el aforo, que salen del mismo
    `jornada_de`. Si la galería usara el día natural, la ventana «lunes» del
    dashboard enseñaría 3 entradas y 2 fotos. Es deliberado, y esta prueba lo
    deja dicho (4-sep-2026)."""
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 22)
    _escribir(capturas, '20260818_003000_cam1_t1')   # 00:30 del 18: jornada 17
    _escribir(capturas, '20260818_223000_cam1_t2')   # 22:30 del 18: jornada 18
    tc.sincronizar(force=True)

    lunes = tc.listar(desde='2026-08-17', hasta='2026-08-17')
    martes = tc.listar(desde='2026-08-18', hasta='2026-08-18')
    assert [c['stem'][-2:] for c in lunes['capturas']] == ['t1']
    assert [c['stem'][-2:] for c in martes['capturas']] == ['t2']
    assert lunes['total'] == martes['total'] == 1


def test_se_puede_acotar_por_camara(capturas):
    _escribir(capturas, '20260818_120000_a_t1', meta={'device_id': 'win-cam1'})
    _escribir(capturas, '20260818_120100_b_t2', meta={'device_id': 'win-cam2'})
    tc.sincronizar(force=True)
    assert tc.listar(camara='win-cam1')['total'] == 1


def test_una_captura_vieja_sin_device_id_se_filtra_por_nombre(capturas):
    """Las de antes del 18-ago-2026 no traen device_id. No pueden quedar
    fuera del filtro en silencio."""
    _escribir(capturas, '20260818_120000_a_t1', meta={'camera': 'Pasillo 3'})
    tc.sincronizar(force=True)
    assert tc.listar(camara='Pasillo 3')['total'] == 1


# ── Respaldo ─────────────────────────────────────────────────────────────

def test_sin_base_la_galeria_sale_igual_por_el_camino_lento(capturas,
                                                            monkeypatch):
    """Un índice roto no puede dejar la galería en blanco."""
    _escribir(capturas, '20260818_101500_cam1_t1', meta={'gender': 'Mujer'},
              con_rostro=True)

    def _explota(*_a, **_k):
        raise OSError('base caída (simulada)')

    monkeypatch.setattr(ta.sqlite3, 'connect', _explota)

    from src.app.dashboard import _captures_index
    filas = _captures_index()
    assert len(filas) == 1
    assert filas[0]['gender'] == 'Mujer'
    assert filas[0]['has_face'] is True


def test_el_recorrido_de_disco_y_el_indice_dicen_lo_mismo(capturas):
    """Si divergen, la caché estaría mintiendo: es la prueba que lo caza."""
    for i in range(4):
        _escribir(capturas, f'20260818_1015{i:02d}_cam1_t{i}',
                  meta={'gender': 'Mujer' if i % 2 else 'Hombre',
                        'person_uuid': f'u{i}'},
                  con_rostro=(i % 2 == 0))
    tc.sincronizar(force=True)

    disco = {c['stem']: c for c in tc.indice_desde_disco()}
    indice = {c['stem']: c for c in tc.listar(limite=100)['capturas']}
    assert set(disco) == set(indice)
    for stem, fila in disco.items():
        for clave in ('gender', 'age_range', 'camera', 'device_id',
                      'person_uuid', 'has_face', 'timestamp'):
            assert fila[clave] == indice[stem][clave], f'{stem}.{clave}'


def test_si_cambia_el_corte_de_jornada_se_reindexa_todo(capturas, monkeypatch):
    """Las filas guardan la jornada calculada con el corte de ENTONCES. Si el
    operador cambia ELDE_TIENDA_CORTE_JORNADA_H y reinicia, el primer
    sincronizar del proceso lo nota (la captura más reciente ya no cuadra)
    y reindexa; si no cambió, no vuelve a parsear nada."""
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    _escribir(capturas, '20260901_100000_000')
    _escribir(capturas, '20260901_113000_000')
    tc.reiniciar_para_pruebas()
    assert tc.sincronizar(force=True)['nuevas'] == 2

    def jornadas():
        with ta.get_almacen()._conectar() as con:
            return sorted(str(f['jornada']) for f in con.execute(
                'SELECT jornada FROM tienda_capturas'))
    assert jornadas() == ['2026-09-01', '2026-09-01']
    # Reinicio sin cambio de corte: nada que reindexar.
    tc.reiniciar_para_pruebas()
    r = tc.sincronizar(force=True)
    assert r['nuevas'] == 0 and r['actualizadas'] == 0
    # Reinicio CON corte a las 22:00: las 10:00 del día 1 pertenecen a la
    # jornada del 31 de agosto; el índice se rehace solo.
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 22)
    tc.reiniciar_para_pruebas()
    r = tc.sincronizar(force=True)
    assert r['nuevas'] + r['actualizadas'] == 2
    assert jornadas() == ['2026-08-31', '2026-08-31']
