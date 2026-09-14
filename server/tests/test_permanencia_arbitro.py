"""
Pruebas del árbitro de cruces ambiguos (`permanencia_arbitro.py`).

Sin GPU y sin modelo: se sustituye el router multimodal por un doble. Lo que
se prueba aquí es la **política**, que es donde están los errores caros:

  - **el pipeline nunca espera** — la cola está acotada y lo que no cabe se
    descarta y se cuenta, en vez de acumularse por detrás del vídeo;
  - **una respuesta rara no contamina ni mata el hilo** — los VLM devuelven
    prosa, markdown y a veces basura;
  - **«ninguno» no se convierte en un cruce** — que el modelo no sepa es una
    respuesta válida, y la correcta es no contar nada;
  - **el veredicto conserva la hora ORIGINAL del cruce**, no la de la
    respuesta: el cronómetro cuenta desde que la persona pasó.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.permanencia_arbitro import (  # noqa: E402
    ANCHO_RECUERDO, POR_VLM, RECUERDOS, ArbitroPuerta, MemoriaVisual,
    Veredicto, _parsear)

FRAME = np.full((480, 640, 3), 120, dtype=np.uint8)
CAJA = (100.0, 60.0, 200.0, 400.0)


class _RouterFalso:
    """Doble del router multimodal: responde lo que se le diga."""

    def __init__(self, respuesta='{"accion": "entro", "seguridad": "alta"}'):
        self.respuesta = respuesta
        self.llamadas = []

    def vqa(self, imagen, pregunta, max_new_tokens=None):
        self.llamadas.append((imagen, pregunta))
        return self.respuesta


def _con_router(respuesta):
    """Sustituye el router real. Devuelve el doble."""
    from src.analityc.core import multimodal_router as mm
    falso = _RouterFalso(respuesta)
    mm.get_multimodal_router = lambda: falso
    return falso


# ── Lectura de la respuesta ─────────────────────────────────────────────
def test_lee_un_json_limpio():
    v = _parsear('{"accion": "entro", "seguridad": "alta"}', 'p1', 10.0, None)
    assert v.tipo == 'entrada' and v.seguridad == 'alta' and v.resuelto


def test_lee_el_json_envuelto_en_prosa():
    """Es lo que hacen los VLM de verdad."""
    crudo = ('Claro, analizando las imagenes:\n```json\n'
             '{"accion": "salio", "seguridad": "media"}\n```\nEspero que ayude.')
    v = _parsear(crudo, 'p1', 10.0, None)
    assert v.tipo == 'salida' and v.seguridad == 'media'


def test_ninguno_NO_es_un_cruce():
    v = _parsear('{"accion": "ninguno", "seguridad": "baja"}', 'p1', 10.0, None)
    assert v.tipo is None and not v.resuelto


def test_una_respuesta_ilegible_se_descarta():
    for crudo in ('', 'no tengo ni idea', '{roto', None,
                  '{"otra_cosa": 1}', '[1, 2, 3]'):
        v = _parsear(crudo, 'p1', 10.0, None)
        assert v is None or not v.resuelto, f'{crudo!r} coló como cruce'


def test_una_seguridad_inventada_cae_a_baja():
    v = _parsear('{"accion": "entro", "seguridad": "altisima"}', 'p1', 1.0, None)
    assert v.seguridad == 'baja'


def test_el_veredicto_conserva_la_hora_del_CRUCE():
    v = _parsear('{"accion": "entro"}', 'p1', 1234.5, None)
    assert v.ts == 1234.5


# ── Memoria visual ──────────────────────────────────────────────────────
def test_guarda_los_momentos_reducidos():
    m = MemoriaVisual(cada_n_frames=1)
    m.recordar('p1', FRAME, CAJA)
    m.recordar('p1', FRAME, CAJA)
    tira = m.tira('p1')
    assert tira is not None
    assert tira.shape[1] <= 2 * ANCHO_RECUERDO + 10, 'no redujo los frames'


def test_con_un_solo_momento_no_hay_movimiento_que_ver():
    m = MemoriaVisual(cada_n_frames=1)
    m.recordar('p1', FRAME, CAJA)
    assert m.tira('p1') is None


def test_no_guarda_mas_de_tres_momentos():
    m = MemoriaVisual(cada_n_frames=1)
    for _ in range(20):
        m.recordar('p1', FRAME, CAJA)
    assert len(m._por_clave['p1']) == RECUERDOS


def test_conserva_el_PRIMER_momento_aunque_entren_muchos():
    """Primero y último son los que marcan el recorrido."""
    m = MemoriaVisual(cada_n_frames=1)
    primero = np.full((480, 640, 3), 10, dtype=np.uint8)
    m.recordar('p1', primero, None)
    for tono in range(50, 200, 20):
        m.recordar('p1', np.full((480, 640, 3), tono, dtype=np.uint8), None)
    assert int(m._por_clave['p1'][0].mean()) < 30, 'perdió el primer momento'


def test_no_guarda_en_cada_frame():
    m = MemoriaVisual(cada_n_frames=5)
    for _ in range(4):
        m.recordar('p1', FRAME, CAJA)
    assert len(m._por_clave.get('p1', [])) == 1


def test_olvidar_libera_al_track():
    m = MemoriaVisual(cada_n_frames=1)
    m.recordar('p1', FRAME, CAJA)
    m.olvidar('p1')
    assert 'p1' not in m._por_clave


def test_la_memoria_no_crece_sin_limite():
    m = MemoriaVisual(max_tracks=5, cada_n_frames=1)
    for i in range(40):
        m.recordar(f'p{i}', FRAME, CAJA)
    assert len(m._por_clave) <= 5


def test_un_frame_vacio_no_revienta():
    m = MemoriaVisual(cada_n_frames=1)
    m.recordar('p1', None, CAJA)
    m.recordar('p1', np.zeros((0, 0, 3), dtype=np.uint8), CAJA)
    m.recordar('p1', FRAME, ('x', None, 1, 2))     # caja corrupta
    assert m.tira('p1') is None or True             # lo que no puede es lanzar


# ── El árbitro apagado ──────────────────────────────────────────────────
def test_apagado_no_hace_absolutamente_nada():
    """Sin activarlo, el modelo ni se carga."""
    a = ArbitroPuerta(activo=False)
    a.recordar('p1', FRAME, CAJA)
    assert a.encolar('p1', 10.0) is False
    assert a.recoger() == []
    assert a.stats['encolados'] == 0


# ── El árbitro encendido ────────────────────────────────────────────────
def test_sin_recuerdos_no_se_consulta():
    """Preguntar con una sola imagen sería pedirle que adivine."""
    a = ArbitroPuerta(activo=True)
    try:
        assert a.encolar('desconocido', 10.0) is False
        assert a.stats['descartados'] == 1
    finally:
        a.detener()


def test_la_cola_llena_descarta_en_vez_de_acumular():
    """El vídeo no puede quedarse esperando a la cola del VLM."""
    a = ArbitroPuerta(activo=True, tam_cola=2, cada_n_frames=1)
    a.detener()                     # sin consumidor: la cola se llena
    try:
        for i in range(6):
            a.memoria.recordar(f'p{i}', FRAME, CAJA)
            a.memoria.recordar(f'p{i}', FRAME, CAJA)
            a.encolar(f'p{i}', 10.0)
        assert a.stats['encolados'] == 2
        assert a.stats['descartados'] == 4
    finally:
        a.detener()


def test_un_veredicto_util_llega_a_la_bandeja():
    a = ArbitroPuerta(activo=True)
    a.detener()                     # se procesa a mano, sin depender del hilo
    try:
        _con_router('{"accion": "entro", "seguridad": "alta"}')
        a._procesar('p1', FRAME, 555.0, CAJA)
        listos = a.recoger()
        assert len(listos) == 1
        assert listos[0].tipo == 'entrada' and listos[0].ts == 555.0
        assert a.recoger() == [], 'la bandeja debe vaciarse al recogerla'
    finally:
        a.detener()


def test_un_ninguno_NO_llega_a_la_bandeja():
    a = ArbitroPuerta(activo=True)
    a.detener()
    try:
        _con_router('{"accion": "ninguno", "seguridad": "baja"}')
        a._procesar('p1', FRAME, 555.0, CAJA)
        assert a.recoger() == []
        assert a.stats['procesados'] == 1 and a.stats['resueltos'] == 0
    finally:
        a.detener()


def test_una_respuesta_ilegible_se_cuenta_y_no_contamina():
    a = ArbitroPuerta(activo=True)
    a.detener()
    try:
        _con_router('perdona, no puedo ayudarte con eso')
        a._procesar('p1', FRAME, 1.0, None)
        assert a.recoger() == [] and a.stats['no_parseables'] == 1
    finally:
        a.detener()


def test_el_hilo_consume_la_cola_de_verdad():
    a = ArbitroPuerta(activo=True, cada_n_frames=1)
    try:
        _con_router('{"accion": "salio", "seguridad": "media"}')
        a.memoria.recordar('p1', FRAME, CAJA)
        a.memoria.recordar('p1', FRAME, CAJA)
        assert a.encolar('p1', 99.0) is True
        for _ in range(60):                    # hasta 3 s
            listos = a.recoger()
            if listos:
                assert listos[0].tipo == 'salida' and listos[0].ts == 99.0
                return
            time.sleep(0.05)
        raise AssertionError('el hilo no procesó nada en 3 s')
    finally:
        a.detener()


def test_un_fallo_del_modelo_no_mata_el_hilo():
    a = ArbitroPuerta(activo=True, cada_n_frames=1)
    try:
        from src.analityc.core import multimodal_router as mm

        def _explota():
            raise RuntimeError('el modelo no está cargado')

        mm.get_multimodal_router = _explota
        for clave in ('p1', 'p2'):
            a.memoria.recordar(clave, FRAME, CAJA)
            a.memoria.recordar(clave, FRAME, CAJA)
            a.encolar(clave, 1.0)
        for _ in range(60):
            if a.stats['errores'] >= 2:
                break
            time.sleep(0.05)
        assert a.stats['errores'] >= 2
        assert a._hilo is not None and a._hilo.is_alive(), 'el hilo murió'
    finally:
        a.detener()


def test_encolar_libera_los_recuerdos():
    """Si no, la memoria de un track resuelto se queda ocupando sitio."""
    a = ArbitroPuerta(activo=True, cada_n_frames=1)
    a.detener()
    try:
        a.memoria.recordar('p1', FRAME, CAJA)
        a.memoria.recordar('p1', FRAME, CAJA)
        a.encolar('p1', 1.0)
        assert 'p1' not in a.memoria._por_clave
    finally:
        a.detener()


# ── Contrato con el pipeline ────────────────────────────────────────────
def test_el_veredicto_se_marca_como_opinion_no_como_medida():
    assert POR_VLM == 'vlm'
    v = Veredicto(clave='p1', tipo='entrada', ts=1.0)
    assert v.resuelto and v.seguridad == 'baja'


def test_el_pipeline_enciende_los_ambiguos_junto_con_el_arbitro():
    """Un árbitro sin cruces que arbitrar no sirve de nada."""
    import inspect
    from src.analityc.core.permanencia_area import PermanenciaArea
    fuente = inspect.getsource(PermanenciaArea._iniciar_permanencia)
    assert "emitir_ambiguos'] = arbitro_on" in fuente


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
