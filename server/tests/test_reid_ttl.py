"""
Pruebas de la retencion de la galeria facial (21-sep-2026).

## Por que existen

El pipeline de tiendas nacio para un stand: alli interesaba recordar a la
gente mucho tiempo, asi que `REID_RESET_POLICY` era `"never"` y la galeria
solo crecia. En un pasillo de supermercado eso no se sostiene: la busqueda de
`_best_in_gallery` recorre TODA la galeria en un bucle de Python, y medido en
este repo (512-d, 5 plantillas por persona) pasa de 3.6 ms con 100 personas a
67.6 ms con 2 000 y 170.6 ms con 5 000 — con un presupuesto de 66 ms por frame
a 15 fps, y pagandose una vez por cada cara del frame. Con ~200 personas/hora
se llega a ~2 400 en un solo dia.

La politica `"ttl"` convierte la galeria en memoria de trabajo: cada persona
caduca pasado `REID_TTL_S` desde su ULTIMA aparicion y nada se escribe a
disco. Aqui se fija ese contrato.

Todo con embeddings sinteticos y tiempos inyectados: sin GPU, sin camara y en
milisegundos, igual que `test_stand_identidad.py`.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics.face_reidentifier import (  # noqa: E402
    FaceReidentifier)

TTL = 1800.0          # 30 min, el defecto de REID_TTL_S


class _SesionArcFaceFalsa:
    """Devuelve un embedding determinista por 'rostro'.

    El rostro sintetico es un parche de color constante, asi que la misma
    persona da siempre el mismo vector y dos personas distintas dan vectores
    casi ortogonales (coseno ~0 en 512-d, muy por debajo del umbral 0.35).
    Asi un match solo puede venir de la galeria, nunca del azar.
    """

    def get_inputs(self):
        return [types.SimpleNamespace(name="in")]

    def get_outputs(self):
        return [types.SimpleNamespace(name="out")]

    def run(self, _salidas, entradas):
        blob = next(iter(entradas.values()))
        semilla = int(round(float(np.mean(blob)) * 1e6)) % (2 ** 31)
        v = np.random.default_rng(semilla).standard_normal(512)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        return [v[None, :]]


def _rostro(persona: int) -> np.ndarray:
    """Parche BGR constante que representa a una persona concreta."""
    return np.full((120, 120, 3), (persona * 7) % 250, dtype=np.uint8)


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "person_db" / "persons.pkl")


def _reid(db_path, ttl=TTL, persistir=False):
    return FaceReidentifier(
        embedding_session=_SesionArcFaceFalsa(),
        db_path=db_path,
        similarity_threshold=0.35,
        reset_policy="ttl" if ttl > 0 else "never",
        ttl_seconds=ttl,
        persistir=persistir,
    )


def _envejecer(reid: FaceReidentifier, uid: str, segundos: float) -> None:
    """Retrasa el `last_seen` de una persona para simular el paso del tiempo."""
    reid._db[uid]['last_seen'] -= segundos


# ── El TTL en si ────────────────────────────────────────────────────────

def test_persona_caduca_pasado_el_ttl(db_path):
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(1))[0]
    assert reid.total_unique_persons == 1

    _envejecer(reid, uid, TTL + 1)
    assert reid.total_unique_persons == 0
    assert reid.session_expired_count == 1


def test_no_caduca_dentro_de_la_ventana(db_path):
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(2))[0]

    _envejecer(reid, uid, TTL - 60)     # 29 min: aun dentro
    assert reid.total_unique_persons == 1


def test_el_ttl_cuenta_desde_la_ultima_vez_no_desde_la_primera(db_path):
    """Quien sigue en el pasillo no caduca aunque lleve horas dentro.

    Es la diferencia entre purgar por `last_seen` y purgar por `first_seen`:
    con el segundo criterio, un cliente que se demora mas que el TTL se
    partiria en dos visitantes sin haberse movido de sitio.
    """
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(3))[0]

    # Lleva 2 h en escena, pero se la ha visto hace un instante.
    reid._db[uid]['first_seen'] -= 7200.0
    assert reid.total_unique_persons == 1

    # Se la sigue viendo: cada re-identificacion refresca `last_seen`.
    for _ in range(3):
        _envejecer(reid, uid, TTL - 30)
        reid.remove_track(100001)
        again = reid.identify_or_register(100002, _rostro(3))
        assert again is not None and again[0] == uid
    assert reid.total_unique_persons == 1


def test_ttl_cero_conserva_el_comportamiento_historico(db_path):
    """Con ttl=0 nada caduca: las instalaciones con `never` no cambian."""
    reid = _reid(db_path, ttl=0.0)
    uid = reid.identify_or_register(100001, _rostro(4))[0]

    reid._db[uid]['last_seen'] -= 30 * 24 * 3600.0     # un mes
    assert reid.total_unique_persons == 1
    assert reid.session_expired_count == 0


def test_tras_caducar_la_misma_cara_es_un_recorrido_nuevo(db_path):
    """Contrato explicito de la politica: se cuentan RECORRIDOS, no personas.

    Quien vuelve pasado el TTL recibe un persistent_id nuevo y suma otra fila
    en `tienda_visitas`. Es el precio aceptado a cambio de que la galeria no
    crezca; esta prueba existe para que el cambio sea deliberado y no un
    descubrimiento a los seis meses.
    """
    reid = _reid(db_path)
    primero = reid.identify_or_register(100001, _rostro(5))[0]

    _envejecer(reid, primero, TTL + 1)
    reid.remove_track(100001)

    segundo = reid.identify_or_register(100002, _rostro(5))[0]
    assert segundo != primero
    assert reid.total_unique_persons == 1


def test_dentro_de_la_ventana_la_misma_cara_mantiene_el_uuid(db_path):
    reid = _reid(db_path)
    primero = reid.identify_or_register(100001, _rostro(6))[0]

    _envejecer(reid, primero, TTL - 120)
    reid.remove_track(100001)

    segundo = reid.identify_or_register(100002, _rostro(6))[0]
    assert segundo == primero


def test_caducar_limpia_el_mapeo_de_tracks(db_path):
    """El atajo `_track_to_person` no puede sobrevivir a su persona.

    Si el mapeo quedara, `identify_or_register` devolveria por la via rapida
    un uuid que ya no esta en la galeria.
    """
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(7))[0]
    assert reid.get_persistent_id(100001) == uid

    _envejecer(reid, uid, TTL + 1)
    reid.total_unique_persons          # dispara la purga

    assert reid.get_persistent_id(100001) is None


def test_caducar_borra_la_foto_del_rostro(db_path):
    """Si caducara el registro pero no el jpg, `faces/` creceria igual."""
    reid = _reid(db_path)
    uid = reid.identify_or_register(100001, _rostro(8))[0]
    foto = reid.get_face_image_path(uid)
    assert foto is not None and os.path.isfile(foto)

    _envejecer(reid, uid, TTL + 1)
    reid.total_unique_persons

    assert not os.path.isfile(foto)


# ── Galeria en RAM ──────────────────────────────────────────────────────

def test_sin_persistencia_no_se_escribe_el_pickle(db_path):
    reid = _reid(db_path, persistir=False)
    reid.identify_or_register(100001, _rostro(9))
    reid.force_save()

    assert not os.path.isfile(db_path)


def test_dos_galerias_en_ram_no_se_pisan(db_path):
    """El fallo que motivo el cambio: dos conexiones websocket, un archivo.

    Cada conexion construye su propio FaceReidentifier
    (`_build_processor` corre una vez por websocket). Con galeria en disco,
    `_save_db_now` escribia su memoria ENTERA encima y la segunda borraba a
    la gente de la primera; al apagar, `atexit` las ejecutaba en orden
    inverso y el archivo acababa VACIO. Sin archivo no hay carrera.
    """
    a = _reid(db_path, persistir=False)
    b = _reid(db_path, persistir=False)

    ua = a.identify_or_register(100001, _rostro(10))[0]
    ub = b.identify_or_register(100001, _rostro(11))[0]
    a.force_save()
    b.force_save()

    assert a.total_unique_persons == 1
    assert b.total_unique_persons == 1
    assert ua != ub
    assert not os.path.isfile(db_path)


def test_con_persistencia_la_galeria_sobrevive_al_reinicio(db_path):
    """La opcion `never`/`daily` sigue funcionando como siempre."""
    uno = _reid(db_path, ttl=0.0, persistir=True)
    uid = uno.identify_or_register(100001, _rostro(12))[0]
    uno.force_save()

    dos = _reid(db_path, ttl=0.0, persistir=True)
    assert dos.total_unique_persons == 1
    assert dos.identify_or_register(100002, _rostro(12))[0] == uid


def test_un_pickle_viejo_en_disco_no_vacia_la_galeria_en_ram(db_path):
    """Un `persons.pkl` de una instalacion anterior no manda sobre la RAM.

    `_sync_external_deletions` existe para propagar los borrados del
    dashboard, y lo hace calculando `memoria - disco`. Si se dejara correr
    con la galeria en RAM, ese conjunto seria TODA la ventana viva: un
    archivo heredado (o recien tocado por otro proceso) se llevaria por
    delante a la gente que esta ahora mismo en el pasillo.
    """
    viejo = _reid(db_path, ttl=0.0, persistir=True)
    viejo.identify_or_register(100001, _rostro(13))
    viejo.force_save()
    assert os.path.isfile(db_path)

    en_ram = _reid(db_path, persistir=False)
    uid = en_ram.identify_or_register(100001, _rostro(14))[0]
    # El pickle es mas reciente que nada de lo que hay en RAM.
    os.utime(db_path, None)
    en_ram.identify_or_register(100002, _rostro(15))

    assert en_ram.get_persistent_id(100001) == uid
    assert en_ram.total_unique_persons == 2


# ── Cableado: la politica por defecto y como la consume tiendas.py ──────

@pytest.mark.skipif(
    'ELDE_REID_RESET_POLICY' in os.environ or 'ELDE_REID_TTL_S' in os.environ,
    reason='el entorno sobreescribe la politica; aqui se fija el DEFECTO')
def test_el_defecto_es_ttl_de_treinta_minutos():
    from src.analityc.core.analytics.config import AnalyticsConfig
    assert AnalyticsConfig.REID_RESET_POLICY == 'ttl'
    assert AnalyticsConfig.REID_TTL_S == 1800.0


def test_la_firma_admite_los_parametros_que_pasa_tiendas():
    """Un typo en un kwarg solo se veria al arrancar el servidor de verdad."""
    import inspect
    params = inspect.signature(FaceReidentifier.__init__).parameters
    assert 'ttl_seconds' in params
    assert 'persistir' in params


def test_tiendas_cablea_el_ttl_y_deja_atexit_solo_con_disco():
    """Se prueba sobre el codigo: instanciar TiendasProcess cargaria los
    modelos y pediria GPU (mismo motivo que
    `test_el_procesador_POR_CAMARA_conserva_el_modo`).

    `atexit.register` guarda una referencia FUERTE al metodo, asi que con
    galeria en disco cada conexion websocket deja vivo su FaceReidentifier
    hasta que muere el proceso, y al apagar se ejecutan todos en orden
    inverso sobrescribiendo el mismo fichero. Sin disco no debe registrarse.
    """
    import inspect
    from src.analityc.core.tiendas import TiendasProcess
    fuente = inspect.getsource(TiendasProcess.__init__)

    assert 'ttl_seconds=' in fuente
    assert 'persistir=' in fuente
    # El atexit vive DENTRO de la rama sin TTL.
    rama_sin_ttl = fuente.split('if _con_ttl:', 1)[1]
    assert 'atexit.register' in rama_sin_ttl.split('else:', 1)[1]


# ── Multicamara: el namespace de track_id (sin cobertura hasta hoy) ─────

def test_namespace_por_camara_evita_colisiones(db_path):
    """Cuatro camaras, cuatro personas, todas con track_id local 1.

    ByteTrack numera desde 1 en CADA camara y el reidentificador es unico,
    asi que sin offset las cuatro compartirian entrada en `_track_to_person`
    y la camara 2 heredaria la identidad de la 1 sin mirar un solo rostro.
    """
    reid = _reid(db_path)
    camaras = ["pasillo-1", "pasillo-2", "caja", "entrada"]
    offsets = [reid.camera_track_namespace(c) for c in camaras]

    assert len(set(offsets)) == 4
    assert reid.camera_track_namespace("caja") == offsets[2]

    uuids = [reid.identify_or_register(1 + off, _rostro(20 + i))[0]
             for i, off in enumerate(offsets)]
    assert len(set(uuids)) == 4
    assert reid.total_unique_persons == 4


def test_una_persona_vista_por_cuatro_camaras_es_un_solo_uuid(db_path):
    reid = _reid(db_path)
    camaras = ["pasillo-1", "pasillo-2", "caja", "entrada"]
    offsets = [reid.camera_track_namespace(c) for c in camaras]

    uuids = [reid.identify_or_register(1 + off, _rostro(30))[0]
             for off in offsets]

    assert len(set(uuids)) == 1
    assert reid.total_unique_persons == 1
    assert reid.get_all_persons()[0]['visit_count'] == 4


def test_sin_offset_la_segunda_camara_hereda_la_identidad(db_path):
    """Control negativo: demuestra QUE evita el namespace.

    Mismo track_id local sin offset -> el atajo de `_track_to_person`
    devuelve el uuid de la primera persona sin llegar a mirar la cara.
    """
    reid = _reid(db_path)
    a = reid.identify_or_register(1, _rostro(40))[0]
    b = reid.identify_or_register(1, _rostro(41))[0]

    assert a == b
    assert reid.total_unique_persons == 1
