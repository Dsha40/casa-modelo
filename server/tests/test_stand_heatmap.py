"""
Pruebas del mapa de calor de tres capas (HITO 9).

Además de lo mecánico, se verifica la propiedad que da sentido al hito:
**tráfico y permanencia son mapas distintos**. Alguien que cruza rápido deja
mucho tráfico y poca permanencia; alguien parado, lo contrario. Si las dos
capas salieran iguales, el mapa no diría nada útil sobre el stand.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_almacen import AlmacenStand  # noqa: E402
from src.analityc.core.stand_config import ConfigStand  # noqa: E402
from src.analityc.core.stand_heatmap import (  # noqa: E402
    TIPO_DETENCIONES, TIPO_PERMANENCIA, TIPO_TRAFICO, MapaCalorStand)


def _cfg(celda=20, sigma=0):
    # sigma=0 en las pruebas: sin suavizar, cada muestra queda en su celda
    # y se puede afirmar sobre índices concretos.
    return ConfigStand({'heatmap': {'celda_px': celda,
                                    'sigma_celdas': sigma}}, {})


def _mapa(celda=20, sigma=0, ancho=1280, alto=720):
    return MapaCalorStand(_cfg(celda, sigma), ancho, alto)


def test_geometria_de_la_rejilla():
    m = _mapa()
    assert m.cols == 64 and m.filas == 36        # 1280/20, 720/20
    assert m.grid() == {'ancho': 64, 'alto': 36, 'celda_px': 20}


def test_una_posicion_cae_en_su_celda():
    m = _mapa()
    assert m.registrar_posicion((0.5, 0.5), dt=1.0) is True
    bruto = m.matriz(TIPO_TRAFICO, suavizar=False, normalizar=False)
    assert bruto[18, 32] == 1.0                  # centro exacto
    assert bruto.sum() == 1.0


def test_puntos_fuera_del_frame_se_ignoran():
    m = _mapa()
    assert m.registrar_posicion((1.5, 0.5)) is False
    assert m.registrar_posicion((-0.1, 0.5)) is False
    assert m.matriz(TIPO_TRAFICO, normalizar=False).sum() == 0.0


def test_trafico_y_permanencia_son_mapas_DISTINTOS():
    """La propiedad que justifica tener dos capas.

    A: cruza deprisa (10 muestras de 0.04 s cada una, celdas distintas).
    B: parado en una celda (2 muestras de 5 s).
    En tráfico gana A; en permanencia, B.
    """
    m = _mapa()
    for i in range(10):                          # A, cruzando
        m.registrar_posicion((0.1 + i * 0.05, 0.9), dt=0.04)
    for _ in range(2):                           # B, parado
        m.registrar_posicion((0.5, 0.5), dt=5.0)

    trafico = m.matriz(TIPO_TRAFICO, suavizar=False, normalizar=False)
    permanencia = m.matriz(TIPO_PERMANENCIA, suavizar=False, normalizar=False)

    # La celda de B tiene menos tráfico que el recorrido de A...
    assert trafico[18, 32] == 2.0
    assert trafico.sum() == 12.0
    # ...pero acapara la permanencia.
    assert permanencia[18, 32] == 10.0
    assert permanencia[18, 32] / permanencia.sum() > 0.9


def test_capa_de_detenciones_es_independiente():
    m = _mapa()
    m.registrar_posicion((0.2, 0.2), dt=1.0)
    m.registrar_detencion((0.8, 0.8))
    det = m.matriz(TIPO_DETENCIONES, suavizar=False, normalizar=False)
    assert det.sum() == 1.0
    assert det[28, 51] == 1.0
    # Una detención no ensucia el tráfico.
    assert m.matriz(TIPO_TRAFICO, suavizar=False, normalizar=False).sum() == 1.0


# ── Filtros del dashboard ───────────────────────────────────────────────
def test_filtro_por_genero():
    m = _mapa()
    m.registrar_posicion((0.25, 0.25), dt=1.0, genero='mujer')
    m.registrar_posicion((0.75, 0.75), dt=1.0, genero='hombre')

    solo_m = m.matriz(TIPO_TRAFICO, genero='mujer', suavizar=False,
                      normalizar=False)
    assert solo_m.sum() == 1.0 and solo_m[9, 16] == 1.0
    # Sin filtro salen los dos.
    assert m.matriz(TIPO_TRAFICO, suavizar=False,
                    normalizar=False).sum() == 2.0


def test_filtro_por_clasificacion_revela_el_diagnostico():
    """Por dónde caminan los que pasan de largo vs los que se detienen.

    Esta comparación es la que revela problemas de ubicación del stand.
    """
    m = _mapa()
    for i in range(5):
        m.registrar_posicion((0.1 + i * 0.02, 0.95), dt=0.1,
                             clasificacion='paso_de_largo')
    m.registrar_posicion((0.5, 0.5), dt=3.0, clasificacion='degusto')

    largo = m.matriz(TIPO_TRAFICO, clasificacion='paso_de_largo',
                     suavizar=False, normalizar=False)
    degusto = m.matriz(TIPO_TRAFICO, clasificacion='degusto',
                       suavizar=False, normalizar=False)
    assert largo.sum() == 5.0 and degusto.sum() == 1.0
    # Están en zonas distintas del frame.
    assert largo[34, :].sum() == 5.0
    assert degusto[18, 32] == 1.0


def test_filtro_por_horas():
    m = _mapa()
    m.registrar_posicion((0.5, 0.5), dt=1.0, hora=10)
    m.registrar_posicion((0.5, 0.5), dt=1.0, hora=11)
    m.registrar_posicion((0.5, 0.5), dt=1.0, hora=18)
    assert m.matriz(TIPO_TRAFICO, horas=[10, 11], suavizar=False,
                    normalizar=False).sum() == 2.0
    assert m.matriz(TIPO_TRAFICO, horas=[18], suavizar=False,
                    normalizar=False).sum() == 1.0


def test_filtros_se_combinan():
    m = _mapa()
    m.registrar_posicion((0.5, 0.5), dt=1.0, genero='mujer',
                         rango_etario='26-35', hora=10)
    m.registrar_posicion((0.5, 0.5), dt=1.0, genero='mujer',
                         rango_etario='36-50', hora=10)
    r = m.matriz(TIPO_TRAFICO, genero='mujer', rango_etario='26-35',
                 horas=[10], suavizar=False, normalizar=False)
    assert r.sum() == 1.0


# ── Salida ──────────────────────────────────────────────────────────────
def test_normalizacion_deja_el_pico_en_uno():
    m = _mapa()
    for _ in range(7):
        m.registrar_posicion((0.5, 0.5), dt=1.0)
    m.registrar_posicion((0.1, 0.1), dt=1.0)
    norm = m.matriz(TIPO_TRAFICO, suavizar=False)
    assert norm.max() == 1.0
    assert 0.0 < norm[3, 6] < 1.0


def test_mapa_vacio_no_revienta():
    m = _mapa()
    r = m.matriz(TIPO_TRAFICO)
    assert r.shape == (36, 64) and r.max() == 0.0
    assert m.zonas_calientes(TIPO_TRAFICO) == []


def test_suavizado_reparte_a_las_vecinas():
    m = _mapa(sigma=2)
    m.registrar_posicion((0.5, 0.5), dt=1.0)
    suave = m.matriz(TIPO_TRAFICO, suavizar=True, normalizar=False)
    assert suave[18, 33] > 0, "el suavizado no tocó a la celda vecina"
    assert suave[18, 32] > suave[18, 33], "el pico debe seguir en el centro"


def test_zonas_calientes():
    m = _mapa()
    for _ in range(5):
        m.registrar_posicion((0.5, 0.5), dt=1.0)
    for _ in range(2):
        m.registrar_posicion((0.2, 0.8), dt=1.0)
    top = m.zonas_calientes(TIPO_TRAFICO, k=2)
    assert len(top) == 2
    assert abs(top[0]['x'] - 0.5078) < 0.02      # centro de la celda 32
    assert top[0]['intensidad'] >= top[1]['intensidad']


def test_payload_para_websocket():
    m = _mapa()
    m.registrar_posicion((0.5, 0.5), dt=1.0)
    p = m.como_payload(TIPO_PERMANENCIA)
    assert p['tipo'] == TIPO_PERMANENCIA
    assert p['grid']['celda_px'] == 20
    assert len(p['matriz']) == 36 and len(p['matriz'][0]) == 64
    assert isinstance(p['matriz'][18][32], float)


def test_snapshot_persiste_solo_las_capas_con_datos():
    with tempfile.TemporaryDirectory() as d:
        alm = AlmacenStand(ruta=Path(d) / 'stand.db')
        sid = alm.abrir_sesion(stand_id='s1')
        m = _mapa()
        m.registrar_posicion((0.5, 0.5), dt=1.0)   # tráfico + permanencia
        # Sin detenciones: esa capa no debe guardarse vacía.
        assert m.snapshot(alm, sid) == 2
        assert alm.ultimo_heatmap(sid, TIPO_TRAFICO) is not None
        assert alm.ultimo_heatmap(sid, TIPO_PERMANENCIA) is not None
        assert alm.ultimo_heatmap(sid, TIPO_DETENCIONES) is None

        m.registrar_detencion((0.3, 0.3))
        assert m.snapshot(alm, sid) == 3


def test_overlay_genera_imagen():
    import numpy as np
    m = _mapa()
    for _ in range(5):
        m.registrar_posicion((0.5, 0.5), dt=1.0)
    fondo = np.zeros((720, 1280, 3), dtype=np.uint8)
    img = m.overlay(TIPO_TRAFICO, fondo=fondo)
    assert img is not None and img.shape == (720, 1280, 3)
    # Sin fondo también funciona (mapa a pelo).
    assert m.overlay(TIPO_TRAFICO).shape == (720, 1280, 3)


def test_tipo_desconocido_falla():
    m = _mapa()
    try:
        m.matriz('inventado')
    except ValueError:
        pass
    else:
        raise AssertionError("aceptó un tipo de mapa inexistente")


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
