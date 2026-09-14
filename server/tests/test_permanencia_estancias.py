"""
Pruebas del libro de estancias abiertas (`permanencia_estancias.py`).

Lo que más se protege aquí es lo que no se ve mirando la pantalla: que un
**reinicio del servidor no vacíe el almacén**, y que emparejar a quien sale
sea una decisión con umbral y no «la más parecida», que siempre da respuesta
y acierta la mitad de las veces.

El reloj se **congela** (martes 1-sep-2026 12:00, hora local) en todas las
pruebas (4-sep-2026): las estancias se siembran «hace N segundos» respecto a
`time.time()` y las duraciones se miden contra `time.time()` otra vez.
Con el reloj real eso obligaba a ventanas de tolerancia (88..92, 590..610)
que dependían de cuánto tardase la máquina entre sembrar y medir; con el
reloj fijo la duración es EXACTA y la prueba afirma el número que promete.
"""
from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import permanencia_estancias as pe  # noqa: E402
from src.analityc.core.permanencia_estancias import (  # noqa: E402
    CIERRE_SIN_SALIDA, SIN_ATRIBUIR, Estancia, RegistroEstancias, coseno)


# ── Reloj congelado ─────────────────────────────────────────────────────
#: Martes 1-sep-2026 a las 12:00 en hora LOCAL (`time.mktime`, no UTC), el
#: mismo instante que usa test_permanencia_evidencia.py. Al libro de
#: estancias no le importa la hora del día (solo resta épocas), pero fijar
#: un instante con sentido evita que un TTL de 24 h o 48 h cruce nada raro
#: y deja las duraciones exactas.
INSTANTE = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))


class _ModuloConReloj:
    """`time` con `time()` sustituido; todo lo demás (strftime...), al real.

    Se cuelga del MÓDULO probado (`pe.time`) y no del intérprete: congelar
    `time.time` a nivel global alcanzaría a cualquier otra prueba de la
    sesión y al propio `tempfile`.
    """

    def __init__(self, real, **sustitutos):
        self._real, self._sustitutos = real, sustitutos

    def __getattr__(self, nombre):
        if nombre in self._sustitutos:
            return self._sustitutos[nombre]
        return getattr(self._real, nombre)


@contextlib.contextmanager
def _reloj_congelado(instante: float = INSTANTE):
    """Fija `time.time()` tal como lo ven `permanencia_estancias` y este
    fichero (`_ahora`). Es un gestor de contexto y no solo un fixture porque
    este fichero también corre sin pytest (ver el final)."""
    global _AHORA
    original, original_ahora = pe.time, _AHORA
    pe.time = _ModuloConReloj(time, time=lambda: instante)
    _AHORA = lambda: instante
    try:
        yield instante
    finally:
        pe.time, _AHORA = original, original_ahora


#: Reloj que usan las pruebas para sembrar y medir. Por defecto el real; el
#: fixture / `_reloj_congelado` lo sustituye por el instante fijo.
_AHORA = time.time


@pytest.fixture(autouse=True)
def _reloj_fijo():
    """Todas las pruebas del fichero con el reloj congelado (4-sep-2026)."""
    with _reloj_congelado():
        yield


def _emb(semilla: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _estancia(pid: str, hace: float = 0.0, emb=None) -> Estancia:
    ahora = _AHORA()
    return Estancia(persistent_id=pid, camara='cam1', zona='puerta',
                    entrada_ts=ahora - hace, ultima_vista_ts=ahora - hace,
                    emb_cuerpo=emb)


# ── Coseno ──────────────────────────────────────────────────────────────
def test_coseno_de_un_vector_consigo_mismo_es_uno():
    v = _emb(1)
    assert abs(coseno(v, v) - 1.0) < 1e-5


def test_coseno_con_nada_no_revienta():
    """`-1.0`, no una excepción: se llama dentro del bucle de inferencia."""
    assert coseno(None, _emb(1)) == -1.0
    assert coseno(_emb(1), None) == -1.0
    assert coseno(np.zeros(512, dtype=np.float32), _emb(1)) == -1.0
    assert coseno(_emb(1), _emb(2, dim=128)) == -1.0     # dimensiones distintas


# ── Abrir y cerrar ──────────────────────────────────────────────────────
def test_abrir_dos_veces_NO_reinicia_el_cronometro():
    """El bug clásico: parpadear en el borde pondría el reloj a cero."""
    reg = RegistroEstancias(persistir=False)
    reg.abrir(_estancia('p1', hace=300))
    primera = reg.get('p1').entrada_ts
    reg.abrir(_estancia('p1', hace=0))
    assert reg.get('p1').entrada_ts == primera
    assert len(reg) == 1


def test_reabrir_completa_lo_que_faltaba():
    """Si entró sin cuerpo medible y luego sí, se aprovecha."""
    reg = RegistroEstancias(persistir=False)
    reg.abrir(_estancia('p1', hace=10, emb=None))
    reg.abrir(_estancia('p1', hace=0, emb=_emb(7)))
    assert reg.get('p1').emb_cuerpo is not None


def test_cerrar_pone_hora_de_salida_y_saca_del_libro():
    """`cerrar()` sella la salida con el reloj del MÓDULO (`pe.time.time`),
    así que con el reloj congelado la duración es exactamente lo sembrado."""
    reg = RegistroEstancias(persistir=False)
    reg.abrir(_estancia('p1', hace=90))
    est = reg.cerrar('p1')
    assert est is not None and 'p1' not in reg
    assert est.salida_ts == INSTANTE
    assert est.duracion(_AHORA()) == pytest.approx(90.0)


def test_cerrar_algo_que_no_estaba_devuelve_None():
    assert RegistroEstancias(persistir=False).cerrar('fantasma') is None


# ── Emparejar por cuerpo ────────────────────────────────────────────────
def test_empareja_a_quien_sale_con_su_propia_entrada():
    reg = RegistroEstancias(persistir=False, umbral_cuerpo=0.86)
    reg.abrir(_estancia('p1', emb=_emb(1)))
    reg.abrir(_estancia('p2', emb=_emb(2)))
    par = reg.emparejar_por_cuerpo(_emb(2))
    assert par is not None
    assert par[0].persistent_id == 'p2' and par[1] > 0.99


def test_sin_llegar_al_umbral_NO_empareja():
    """Devolver «el más parecido» sin umbral acierta la mitad de las veces."""
    reg = RegistroEstancias(persistir=False, umbral_cuerpo=0.86)
    reg.abrir(_estancia('p1', emb=_emb(1)))
    assert reg.emparejar_por_cuerpo(_emb(99)) is None


def test_una_estancia_sin_cuerpo_no_rompe_la_busqueda():
    reg = RegistroEstancias(persistir=False)
    reg.abrir(_estancia('sin_cuerpo', emb=None))
    reg.abrir(_estancia('con_cuerpo', emb=_emb(3)))
    par = reg.emparejar_por_cuerpo(_emb(3))
    assert par is not None and par[0].persistent_id == 'con_cuerpo'


def test_unica_abierta_solo_cuando_de_verdad_hay_una():
    reg = RegistroEstancias(persistir=False)
    assert reg.unica_abierta() is None
    reg.abrir(_estancia('p1'))
    assert reg.unica_abierta().persistent_id == 'p1'
    reg.abrir(_estancia('p2'))
    assert reg.unica_abierta() is None, 'con dos dentro no se puede saber'


# ── Caducidad ───────────────────────────────────────────────────────────
def test_una_estancia_vencida_se_cierra_como_sin_salida():
    reg = RegistroEstancias(persistir=False, ttl_horas=1)
    reg.abrir(_estancia('p1', hace=2 * 3600))
    reg.abrir(_estancia('p2', hace=60))
    vencidas = reg.caducadas()
    assert [e.persistent_id for e in vencidas] == ['p1']
    assert vencidas[0].identidad_por == SIN_ATRIBUIR
    assert 'p2' in reg, 'la reciente no se toca'


def test_la_caducidad_es_estrictamente_mayor_que_el_ttl():
    """La frontera exacta: con el reloj congelado se puede sembrar una
    estancia de EXACTAMENTE `ttl` segundos y afirmar que aún no vence (el
    código compara con `>`); un segundo más y sí. Con el reloj real este
    borde era imposible de probar."""
    reg = RegistroEstancias(persistir=False, ttl_horas=1)
    reg.abrir(_estancia('justo', hace=3600))
    reg.abrir(_estancia('pasado', hace=3601))
    assert [e.persistent_id for e in reg.caducadas()] == ['pasado']
    assert 'justo' in reg


def test_el_libro_no_crece_sin_limite():
    reg = RegistroEstancias(persistir=False, max_abiertas=3)
    for i in range(6):
        reg.abrir(_estancia(f'p{i}', hace=100 - i))
    assert len(reg) == 3


# ── Persistencia ────────────────────────────────────────────────────────
def test_un_reinicio_NO_vacia_el_almacen():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'cam1.json'
        reg = RegistroEstancias(ruta=ruta, ttl_horas=24)
        reg.abrir(_estancia('p1', hace=600, emb=_emb(5)))
        reg.abrir(_estancia('p2', hace=30))
        assert reg.guardar(forzar=True)

        # El servidor se reinicia: proceso nuevo, mismo archivo.
        otro = RegistroEstancias(ruta=ruta, ttl_horas=24)
        assert otro.cargar() == 2
        assert set(otro.abiertas) == {'p1', 'p2'}
        assert otro.get('p1').emb_cuerpo is not None, 'perdió el Re-ID'
        assert otro.get('p1').duracion(_AHORA()) == pytest.approx(600.0)


def test_al_recargar_NO_resucita_lo_ya_vencido():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'cam1.json'
        reg = RegistroEstancias(ruta=ruta, ttl_horas=24)
        reg.abrir(_estancia('viejo', hace=48 * 3600))
        reg.guardar(forzar=True)
        otro = RegistroEstancias(ruta=ruta, ttl_horas=24)
        assert otro.cargar() == 0


def test_un_json_corrupto_no_tumba_el_arranque():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'cam1.json'
        ruta.write_text('{esto no es json', encoding='utf-8')
        reg = RegistroEstancias(ruta=ruta)
        assert reg.cargar() == 0 and len(reg) == 0


def test_una_estancia_corrupta_no_se_lleva_a_las_demas():
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'cam1.json'
        buena = _estancia('buena', hace=10).a_dict()
        ruta.write_text(json.dumps({'abiertas': [{'sin': 'id'}, buena]}),
                        encoding='utf-8')
        reg = RegistroEstancias(ruta=ruta)
        assert reg.cargar() == 1 and 'buena' in reg


def test_guardar_no_escribe_en_cada_frame():
    """A 25 FPS, escribir siempre costaría más que la inferencia."""
    with tempfile.TemporaryDirectory() as tmp:
        reg = RegistroEstancias(ruta=Path(tmp) / 'cam1.json')
        reg.abrir(_estancia('p1'))
        assert reg.guardar() is True            # la 1a sí
        reg.abrir(_estancia('p2'))
        assert reg.guardar() is False           # demasiado pronto
        assert reg.guardar(forzar=True) is True


def test_sin_ruta_no_toca_el_disco():
    reg = RegistroEstancias(persistir=False)
    reg.abrir(_estancia('p1'))
    assert reg.guardar(forzar=True) is False
    assert reg.cargar() == 0


# ── La puerta de cada estancia (7-sep-2026) ─────────────────────────────
# Con varias puertas en la misma cámara, la estancia tiene que recordar POR
# CUÁL entró: es lo que impide que una salida por la trasera se empareje con
# una entrada por la principal. Y como el libro se relee tras un reinicio, ese
# dato tiene que sobrevivir al disco SIN romper los ficheros que ya existen.
def test_sin_puertas_una_estancia_es_el_aforo_de_la_camara():
    """El camino de siempre: sin nada dibujado, `-1` y nombre vacío."""
    est = _estancia('p1')
    assert est.puerta == -1 and est.puerta_nombre == ''


def test_la_puerta_sobrevive_a_la_ida_y_vuelta():
    est = _estancia('p1', hace=120, emb=_emb(4))
    est.puerta, est.puerta_nombre = 3, 'Patio'
    copia = Estancia.desde_dict(est.a_dict())
    assert copia is not None
    assert copia.puerta == 3 and copia.puerta_nombre == 'Patio'
    assert copia.persistent_id == 'p1', 'no se tocó nada más'


def test_la_ida_y_vuelta_pasa_por_JSON_de_verdad():
    """Como en disco: `json.dumps` + `json.loads`, no solo el dict."""
    est = _estancia('p1', hace=30)
    est.puerta, est.puerta_nombre = 7, 'Depósito'
    copia = Estancia.desde_dict(json.loads(json.dumps(est.a_dict())))
    assert copia is not None
    assert copia.puerta == 7 and copia.puerta_nombre == 'Depósito'


def test_la_puerta_CERO_no_se_lee_como_aforo_unico():
    """La trampa del `or`: la puerta 0 es la PRIMERA, no un contador a cero.

    Con `int(datos.get('puerta') or -1)` toda la gente de la primera puerta
    —la más transitada, la principal— volvería del disco como aforo de la
    cámara en cada reinicio, y su sección se quedaría vacía sin un solo error.
    """
    est = _estancia('p1')
    est.puerta, est.puerta_nombre = 0, 'Almacén'
    copia = Estancia.desde_dict(est.a_dict())
    assert copia is not None
    assert copia.puerta == 0 and copia.puerta_nombre == 'Almacén'


def test_un_libro_SIN_las_claves_nuevas_se_sigue_leyendo():
    """Compatibilidad hacia atrás: los ficheros que ya están en disco fueron
    escritos antes de que existieran las puertas, y eran el aforo único."""
    viejo = _estancia('p1', hace=60).a_dict()
    viejo.pop('puerta')
    viejo.pop('puerta_nombre')
    copia = Estancia.desde_dict(viejo)
    assert copia is not None
    assert copia.puerta == -1 and copia.puerta_nombre == ''
    assert copia.persistent_id == 'p1' and copia.zona == 'puerta'


def test_una_puerta_nula_en_el_json_cae_al_aforo_unico():
    """`null` es lo que escribe cualquiera que serialice un `Optional`."""
    crudo = _estancia('p1').a_dict()
    crudo['puerta'], crudo['puerta_nombre'] = None, None
    copia = Estancia.desde_dict(crudo)
    assert copia is not None
    assert copia.puerta == -1 and copia.puerta_nombre == ''


def test_la_puerta_tambien_va_en_el_volcado_sin_foto():
    """`con_foto=False` quita EVIDENCIA (los base64), no la identidad."""
    est = _estancia('p1')
    est.puerta, est.puerta_nombre = 2, 'Patio'
    ligero = est.a_dict(con_foto=False)
    assert 'entrada_b64' not in ligero, 'esto sí se queda fuera'
    assert ligero['puerta'] == 2 and ligero['puerta_nombre'] == 'Patio'


def test_un_reinicio_devuelve_a_cada_uno_a_SU_puerta():
    """Sin esto, tras reiniciar todos caerían en el aforo único (-1) y los
    aforos por puerta quedarían a cero con el vídeo perfecto."""
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'cam1.json'
        reg = RegistroEstancias(ruta=ruta, ttl_horas=24)
        primera = _estancia('p1', hace=600)
        primera.puerta, primera.puerta_nombre = 0, 'Almacén'
        segunda = _estancia('p2', hace=30)
        segunda.puerta, segunda.puerta_nombre = 1, 'Patio'
        reg.abrir(primera)
        reg.abrir(segunda)
        assert reg.guardar(forzar=True)

        otro = RegistroEstancias(ruta=ruta, ttl_horas=24)
        assert otro.cargar() == 2
        assert (otro.get('p1').puerta, otro.get('p1').puerta_nombre) == \
            (0, 'Almacén')
        assert (otro.get('p2').puerta, otro.get('p2').puerta_nombre) == \
            (1, 'Patio')


def test_un_fichero_EN_DISCO_de_antes_no_se_pierde():
    """El fichero real que hay hoy en las cámaras: sin `puerta` ni
    `puerta_nombre`. Tiene que cargar entero, como el aforo de la cámara."""
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'cam1.json'
        antiguo = _estancia('p1', hace=600).a_dict()
        antiguo.pop('puerta')
        antiguo.pop('puerta_nombre')
        ruta.write_text(json.dumps({'version': 1, 'abiertas': [antiguo]}),
                        encoding='utf-8')
        reg = RegistroEstancias(ruta=ruta, ttl_horas=24)
        assert reg.cargar() == 1
        est = reg.get('p1')
        assert est.puerta == -1 and est.puerta_nombre == ''
        assert est.duracion(_AHORA()) == pytest.approx(600.0)


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
if __name__ == '__main__':
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        try:
            with _reloj_congelado():        # sin pytest no corre el fixture
                _fn()
            print(f"  OK    {_nombre}")
        except Exception as _exc:                          # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
