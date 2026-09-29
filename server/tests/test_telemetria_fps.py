"""
Telemetría de fps por cámara (28-sep-2026): `src/app/telemetria_fps.py`.

Responde a «¿cuántos frames recibe el servidor?» sin deducirlo de otros
logs. Lo que se fija:

  - **recibidos y procesados se cuentan por cámara**, y la diferencia son
    los descartados de la cola (se queda con el último frame por cámara);
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
    return reloj


def test_cuenta_por_camara_y_saca_los_descartados(limpio):
    for _ in range(10):
        tf.anotar_recibido('cam-a')
    for ms in (30, 40, 50, 60, 70, 80):
        tf.anotar_procesado('cam-a', ms)
    tf.anotar_recibido('cam-b')
    limpio['t'] = 1010.0                          # 10 s de periodo
    por_cam = {r['camera_id']: r for r in tf.resumen()}
    a = por_cam['cam-a']
    assert a['recibidos_s'] == 1.0 and a['procesados_s'] == 0.6
    assert a['descartados'] == 4
    assert a['proceso_ms_mediana'] == 60.0 and a['proceso_ms_p90'] == 80.0
    assert por_cam['cam-b']['recibidos_s'] == 0.1   # no se mezclan


def test_cada_periodo_escribe_su_linea_y_empieza_de_cero(limpio, caplog):
    for _ in range(3):
        tf.anotar_recibido('cam-a')
        tf.anotar_procesado('cam-a', 50)
    limpio['t'] = 1000.0 + tf.PERIODO_S          # se cumple el periodo
    with caplog.at_level(logging.INFO):
        tf.anotar_recibido('cam-a')               # el que dispara el volcado
    lineas = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith('FPS [cam-a]')]
    assert len(lineas) == 1 and 'descartados=1' in lineas[0]
    assert tf.resumen() == []                     # periodo nuevo, vacío


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
    for raro in (None, (), 'x', 3, ('solo-uno',)):
        assert tf.tamano_de_payload(raro) == 0
    tf.resumen()
