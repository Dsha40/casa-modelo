"""Áreas a ignorar (12-oct-2026, `zonas_ignorar.py`).

Lo que fijan: se tira a quien tiene la caja ENTERA dentro de un área y a
nadie más (con un trozo fuera cuenta), sin áreas no cambia nada, un fallo
nunca se come detecciones, y el filtro va ANTES del seguimiento y llega
desde el mensaje del cliente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analityc.core import zonas_ignorar  # noqa: E402

#: El fondo: x 0..400, y 0..300.
FONDO = [[0, 0], [400, 0], [400, 300], [0, 300]]


def _det(cajas):
    return sv.Detections(xyxy=np.array(cajas, dtype=float),
                         confidence=np.full(len(cajas), 0.9),
                         class_id=np.zeros(len(cajas), dtype=int))


def test_tira_solo_a_quien_esta_entero_dentro():
    d = zonas_ignorar.filtrar(_det([
        [50, 50, 120, 250],      # entera dentro: fuera
        [350, 100, 450, 400],    # un trozo fuera: cuenta
        [500, 100, 600, 500],    # fuera del todo: cuenta
    ]), [FONDO], 'cam')
    assert d.xyxy.tolist() == [[350, 100, 450, 400], [500, 100, 600, 500]]


def test_el_borde_cuenta_como_dentro():
    d = zonas_ignorar.filtrar(_det([[0, 0, 400, 300]]), [FONDO], 'cam')
    assert len(d) == 0


def test_varias_areas():
    otra = [[500, 0], [700, 0], [700, 200], [500, 200]]
    d = zonas_ignorar.filtrar(_det([[10, 10, 50, 50], [550, 10, 600, 60],
                                    [800, 10, 900, 60]]), [FONDO, otra], 'cam')
    assert d.xyxy.tolist() == [[800, 10, 900, 60]]


def test_sin_areas_o_sin_detecciones_no_cambia_nada():
    entrada = _det([[50, 50, 120, 250]])
    assert zonas_ignorar.filtrar(entrada, None, 'cam') is entrada
    assert zonas_ignorar.filtrar(entrada, [], 'cam') is entrada
    assert zonas_ignorar.filtrar(entrada, [[[0, 0], [1, 1]]], 'cam') is entrada
    vacio = sv.Detections.empty()
    assert zonas_ignorar.filtrar(vacio, [FONDO], 'cam') is vacio


def test_un_fallo_no_se_come_detecciones():
    raro = object()
    assert zonas_ignorar.filtrar(raro, [FONDO], 'cam') is raro


def test_va_antes_del_seguimiento_y_detras_del_filtro_del_fondo():
    src = (Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    i_fondo = src.index('filtro_fondo.filtrar(')
    i_ign = src.index('zonas_ignorar.filtrar(', i_fondo)
    i_trk = src.index('self._byte_tracker.update_with_detections(', i_fondo)
    assert i_fondo < i_ign < i_trk


def test_llega_del_mensaje_al_procesador():
    """Las cuatro piezas de `app.py`: se lee del mensaje, viaja en la tupla
    posicional y se cuelga en las DOS ramas del procesador."""
    src = (Path(__file__).resolve().parents[1] / 'src' / 'app'
           / 'app.py').read_text(encoding='utf-8')
    assert 'data.get("zonas_ignorar")' in src
    assert src.count('zonas_ignorar=zonas_ignorar') == 1
    assert '._zonas_ignorar = zonas_ignorar' in src
    assert src.count('_zonas_ignorar = zonas_ignorar') == 2


def test_el_contrato_declara_el_campo():
    from elde_core.contracts import eventos
    campos = set()
    for nombre in dir(eventos):
        modelo = getattr(eventos, nombre)
        campos |= set(getattr(modelo, 'model_fields', {}) or {})
    assert 'zonas_ignorar' in campos
