"""
Telemetría de fps por cámara (28-sep-2026): `src/app/telemetria_fps.py`.

Responde a «¿cuántos frames recibe el servidor?» sin deducirlo de otros
logs. Lo que se fija:

  - **recibidos y procesados se cuentan por cámara**;
  - **los descartados son los que la cola reemplazó**, no la resta
    recibidos − procesados (30-sep-2026): la resta contaba como perdido el
    frame EN VUELO que cerraba cada periodo;
  - **el tiempo de proceso** sale como mediana y p90 del periodo;
  - **cada periodo empieza de cero** tras escribir su línea: las cifras son
    del periodo, no acumuladas desde el arranque;
  - **nunca lanza**: medir no puede costar un frame.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app import telemetria_fps as tf  # noqa: E402


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    """Estado propio por prueba y reloj controlado (el periodo no se cumple
    solo a mitad de una prueba)."""
    reloj = {'t': 1000.0}
    monkeypatch.setattr(tf.time, 'monotonic', lambda: reloj['t'])
    monkeypatch.setattr(tf, '_por_camara', {})
    monkeypatch.setattr(tf, '_desde', 1000.0)
    monkeypatch.setattr(tf, '_historial', {})
    monkeypatch.setattr(tf, '_ultima_imagen', {})
    monkeypatch.setattr(tf, '_ultimo_tamano', {})
    return reloj


def test_cuenta_por_camara(limpio):
    for _ in range(10):
        tf.anotar_recibido('cam-a')
    for ms in (30, 40, 50, 60, 70, 80):
        tf.anotar_procesado('cam-a', ms)
    tf.anotar_recibido('cam-b')
    limpio['t'] = 1010.0                          # 10 s de periodo
    por_cam = {r['camera_id']: r for r in tf.resumen()}
    a = por_cam['cam-a']
    assert a['recibidos_s'] == 1.0 and a['procesados_s'] == 0.6
    assert a['proceso_ms_mediana'] == 60.0 and a['proceso_ms_p90'] == 80.0
    assert por_cam['cam-b']['recibidos_s'] == 0.1   # no se mezclan


def test_los_descartados_son_los_que_reemplazo_la_cola(limpio):
    """10 recibidos y 6 procesados NO son 4 descartados: los que faltan
    pueden estar en vuelo. Solo cuenta lo que la cola dice que reemplazó."""
    for _ in range(10):
        tf.anotar_recibido('cam-a')
    for _ in range(6):
        tf.anotar_procesado('cam-a', 50)
    tf.anotar_descartado('cam-a')
    tf.anotar_descartado('cam-a')
    tf.anotar_recibido('cam-b')
    limpio['t'] = 1010.0
    por_cam = {r['camera_id']: r for r in tf.resumen()}
    assert por_cam['cam-a']['descartados'] == 2      # la resta daría 4
    assert por_cam['cam-b']['descartados'] == 0      # la resta daría 1


def test_cada_periodo_escribe_su_linea_y_empieza_de_cero(limpio, caplog):
    for _ in range(3):
        tf.anotar_recibido('cam-a')
        tf.anotar_procesado('cam-a', 50)
    limpio['t'] = 1000.0 + tf.PERIODO_S          # se cumple el periodo
    with caplog.at_level(logging.INFO):
        tf.anotar_recibido('cam-a')               # el que dispara el volcado
    lineas = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith('FPS [cam-a]')]
    # El frame que cierra el periodo acaba de llegar y está EN VUELO. Con la
    # resta de antes salía «descartados=1» en 70 de las 246 líneas de la
    # tarde del 29-sep sin que la cola hubiera tirado ninguno (30-sep-2026).
    assert len(lineas) == 1 and 'descartados=0' in lineas[0]
    assert tf.resumen() == []                     # periodo nuevo, vacío


def test_la_linea_lleva_los_descartes_de_la_cola(limpio, caplog):
    for _ in range(3):
        tf.anotar_recibido('cam-a')
        tf.anotar_procesado('cam-a', 50)
    for _ in range(2):                            # dos reemplazos reales
        tf.anotar_recibido('cam-a')
        tf.anotar_descartado('cam-a')
    limpio['t'] = 1000.0 + tf.PERIODO_S
    with caplog.at_level(logging.INFO):
        tf.anotar_recibido('cam-a')
    lineas = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith('FPS [cam-a]')]
    assert len(lineas) == 1 and 'descartados=2' in lineas[0]   # la resta: 3


def test_guarda_en_memoria_cada_periodo_escrito(limpio, monkeypatch):
    """La página «En vivo» (22-oct-2026) pinta la última hora sin leer el
    log: cada línea escrita queda también en `historial`, sin pasar de
    `HISTORIAL_PERIODOS` por cámara, `ultima_imagen` dice cuándo llegó la
    última y `ultimo_tamano`, cuánto pesaba."""
    monkeypatch.setattr(tf, 'HISTORIAL_PERIODOS', 2)
    monkeypatch.setattr(tf.time, 'time', lambda: 5000.0)
    for periodo in range(1, 4):
        tf.anotar_recibido('cam-a', 1000 * periodo)
        tf.anotar_procesado('cam-a', 40 + periodo)
        limpio['t'] = 1000.0 + periodo * tf.PERIODO_S
        tf.anotar_recibido('cam-a')               # cierra el periodo
    historial = tf.historial('cam-a')
    assert [h['proceso_ms_mediana'] for h in historial] == [42.0, 43.0]
    assert historial[-1]['t'] == 5000.0
    assert tf.ultima_imagen('cam-a') == 5000.0
    # La que cerró el último periodo no dijo su tamaño: vale la anterior.
    assert tf.ultimo_tamano('cam-a') == 3000
    assert tf.ultima_imagen('cam-b') is None and tf.historial('cam-b') == []
    assert tf.ultimo_tamano('cam-b') is None
    assert tf.camaras_vistas() == ['cam-a']


def test_mide_lo_que_pesan_las_imagenes(limpio):
    """El peso por el que se sabe si el techo es la conexión (29-sep)."""
    imagen = b'\xff\xd8' + b'x' * (300 * 1024 - 2)       # 300 KB en binario
    for _ in range(5):
        tf.anotar_recibido('cam-a', tf.tamano_de_payload(('proc', imagen)))
    limpio['t'] = 1010.0
    r = tf.resumen()[0]
    assert r['kb_por_frame'] == 300
    assert r['entrada_kb_s'] == 150.0                    # 1500 KB en 10 s
    # El formato antiguo manda la imagen en base64: 4 caracteres = 3 bytes.
    assert tf.tamano_imagen('data:image/jpeg;base64,' + 'A' * 400) == 300


def test_nunca_lanza(limpio):
    tf.anotar_procesado('cam-a', 'no es un numero')
    tf.anotar_recibido(None)
    tf.anotar_descartado(None)
    for raro in (None, (), 'x', 3, ('solo-uno',)):
        assert tf.tamano_de_payload(raro) == 0
    tf.resumen()
