"""Presupuesto de tareas pesadas por imagen (26-oct-2026, `presupuesto_imagen.py`).

Se fija: apagado no cambia nada (devuelve el turno de siempre); encendido,
como mucho una tarea por imagen, las demás se hacen en las siguientes sin
perderse, las pendientes van antes que las nuevas y se olvidan las de quien
ya no está.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import presupuesto_imagen as pi  # noqa: E402


class Proc:
    pass


def test_apagado_devuelve_el_turno_de_siempre(monkeypatch):
    monkeypatch.setenv('ELDE_PRESUPUESTO_IMAGEN', '0')
    p = Proc()
    pi.nueva_imagen(p)
    assert [pi.toca(p, t, 'genero', True) for t in (1, 2, 3)] == [True, True, True]
    assert pi.toca(p, 4, 'huella', False) is False
    assert not hasattr(p, pi._ESTADO)                     # ni siquiera guarda estado


def test_una_por_imagen_y_las_demas_despues(monkeypatch):
    monkeypatch.setenv('ELDE_PRESUPUESTO_IMAGEN', '1')
    p = Proc()
    hechas = []
    # Imagen 1: tres tareas a la vez (género de 1 y 2, huella de 3).
    pi.nueva_imagen(p)
    for tid, tarea in ((1, 'genero'), (2, 'genero'), (3, 'huella')):
        if pi.toca(p, tid, tarea, True):
            hechas.append((1, tid))
    # Imágenes siguientes: el turno ya no toca (False), pero las pendientes salen.
    for n in (2, 3, 4):
        pi.nueva_imagen(p)
        for tid, tarea in ((1, 'genero'), (2, 'genero'), (3, 'huella')):
            if pi.toca(p, tid, tarea, False):
                hechas.append((n, tid))
    assert hechas == [(1, 1), (2, 2), (3, 3)]             # ninguna perdida, una por imagen


def test_las_pendientes_van_antes_que_las_nuevas(monkeypatch):
    monkeypatch.setenv('ELDE_PRESUPUESTO_IMAGEN', '1')
    p = Proc()
    pi.nueva_imagen(p)
    assert pi.toca(p, 1, 'genero', True)
    assert not pi.toca(p, 2, 'genero', True)              # queda pendiente
    pi.nueva_imagen(p)
    assert not pi.toca(p, 5, 'huella', True)              # nueva: espera su turno
    assert pi.toca(p, 2, 'genero', False)                 # la pendiente sale antes


def test_se_olvida_la_tarea_de_quien_ya_no_esta(monkeypatch):
    monkeypatch.setenv('ELDE_PRESUPUESTO_IMAGEN', '1')
    p = Proc()
    pi.nueva_imagen(p)
    pi.toca(p, 1, 'genero', True)
    pi.toca(p, 2, 'genero', True)                         # pendiente; 2 se va
    for _ in range(pi.OLVIDAR_TRAS + 2):
        pi.nueva_imagen(p)
    assert pi.toca(p, 9, 'genero', True)                  # ya no bloquea a nadie
