"""El bloque `anaqueles` lleva las muñecas del fotograma (23-sep-2026).

Sirven para que el operador vea en el recuadro DÓNDE cree el servidor que está
la mano mientras calibra un área: sin eso, ajustar un polígono es adivinar —
medido sobre vídeo real, un área ceñida a los productos no abre compuerta y no
cuenta nada, y desde el cliente no había forma de saberlo.

Las dos cosas que se fijan aquí:

1. **Van SIN throttle.** El resto del bloque se emite cada 2, 10 o 30 s; un
   punto con 2 s de retraso se pinta donde la mano YA no está, que es peor que
   no pintarlo.
2. **Se reponen en cada fotograma.** Si acumularan, el recuadro acabaría
   sembrado de puntos viejos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / 'src'))

from analityc.core.anaqueles import MedidorAnaqueles, cargar_config  # noqa: E402

# Un area generosa y una persona pegada a ella: la compuerta tiene que abrir
# para que se pida pose (es la condicion de que haya munecas).
_AREA = [[20, 25], [980, 25], [980, 620], [20, 620]]
_BOX = (840.0, 351.0, 1439.0, 797.0)
_FRAME = np.full((816, 1438, 3), 40, dtype=np.uint8)


def _medidor(tmp_path, munecas):
    """Un medidor con la pose sustituida por `munecas` fijas.

    Se sustituye la pose a proposito: lo que se prueba es el TRANSPORTE de las
    munecas hasta el bloque, no que ViTPose las encuentre (eso depende de los
    pesos y no cabe en una prueba unitaria).
    """
    from analityc.core import anaqueles_registro
    anaqueles_registro.raiz_salida = lambda: tmp_path      # nada toca output/
    m = MedidorAnaqueles(f'cam-{tmp_path.name}', cargar_config(''), device=None)
    m.poner_zonas([_AREA], ['Anaquel 1'])
    m.pose.munecas = lambda frame, box: list(munecas)
    # Desde el 6-oct el medidor pide todas las personas a la vez
    # (`munecas_lote`, pose_gpu.py): se sustituye también.
    m.pose.munecas_lote = lambda frame, boxes: [list(munecas) for _ in boxes]
    return m


def _procesar(m, n=6, t0=1000.0):
    tracks = {1: {'class': -1, 'box': _BOX, 'persistent_id': 'p1'}}
    bloque = None
    for i in range(n):
        bloque = m.procesar(_FRAME, tracks, demografia=None,
                            ahora=t0 + i * 0.4, reidentifier=None)
    return bloque


def test_el_bloque_lleva_las_munecas(tmp_path):
    m = _medidor(tmp_path, [(934.0, 453.0, 0.78, 'der')])
    bloque = _procesar(m)
    assert bloque is not None
    assert 'munecas' in bloque, 'el recuadro no puede pintar lo que no llega'
    mun = bloque['munecas'][0]
    assert mun['x'] == 934.0 and mun['y'] == 453.0
    assert mun['lado'] == 'der'
    assert mun['area'] == 0
    assert mun['prof'] is not None
    assert isinstance(mun['dentro'], bool)


def test_la_profundidad_dice_si_el_servidor_la_cuenta_dentro(tmp_path):
    """`dentro` es el COLOR del punto en el recuadro: es toda la ayuda que
    el operador recibe para saber por que un agarre no cuenta."""
    hondo = _medidor(tmp_path / 'a', [(500.0, 300.0, 0.8, 'der')])
    (tmp_path / 'a').mkdir(exist_ok=True)
    b = _procesar(hondo)
    assert b['munecas'][0]['dentro'] is True

    fuera = _medidor(tmp_path / 'b', [(1200.0, 700.0, 0.8, 'der')])
    (tmp_path / 'b').mkdir(exist_ok=True)
    b2 = _procesar(fuera)
    if b2 and b2.get('munecas'):
        assert b2['munecas'][0]['dentro'] is False


def test_no_acumulan_entre_fotogramas(tmp_path):
    """Si no se repusieran, el recuadro acabaria sembrado de puntos viejos."""
    m = _medidor(tmp_path, [(934.0, 453.0, 0.78, 'der'),
                            (900.0, 450.0, 0.70, 'izq')])
    _procesar(m, n=10)
    assert len(m._munecas_frame) == 2, 'dos munecas por fotograma, no veinte'


def test_sin_munecas_la_clave_NO_viaja(tmp_path):
    """Un bloque con `munecas: []` haria que el cliente borrara y repintara
    cada 2 s sin motivo. Si no hay, no se manda la clave."""
    m = _medidor(tmp_path, [])
    bloque = _procesar(m)
    assert bloque is None or 'munecas' not in bloque


def test_las_munecas_no_esperan_al_throttle_de_contadores(tmp_path):
    """Los contadores van cada 2 s; las munecas, en CADA fotograma. Con el
    throttle, el punto se pintaria donde la mano ya no esta."""
    m = _medidor(tmp_path, [(934.0, 453.0, 0.78, 'der')])
    tracks = {1: {'class': -1, 'box': _BOX, 'persistent_id': 'p1'}}
    t = 1000.0
    m.procesar(_FRAME, tracks, demografia=None, ahora=t, reidentifier=None)
    # 0,4 s despues: demasiado pronto para los contadores (periodo 2 s).
    b = m.procesar(_FRAME, tracks, demografia=None, ahora=t + 0.4,
                   reidentifier=None)
    assert b is not None and 'munecas' in b
    assert 'areas' not in b, 'los contadores SI deben respetar su ritmo'
