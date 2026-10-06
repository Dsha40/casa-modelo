"""
Cronómetro por etapas (`tiempos_etapas.py`, 6-oct-2026).

Lo que fijan estas pruebas: cada etapa cuenta solo su tiempo PROPIO (la de
dentro se le resta a la de fuera), la suma de etapas da el total, sale una
línea por imagen y, sin la variable, no se toca nada.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import tiempos_etapas as te  # noqa: E402


def _modulo_falso(monkeypatch):
    m = types.ModuleType('falso_etapas')

    def dentro():
        time.sleep(0.03)

    def fuera():
        time.sleep(0.02)
        m.dentro()

    def imagen():
        m.fuera()
        time.sleep(0.01)

    m.dentro, m.fuera, m.imagen = dentro, fuera, imagen
    monkeypatch.setitem(sys.modules, 'falso_etapas', m)
    monkeypatch.setattr(te, '_OBJETIVOS', [
        ('falso_etapas', None, 'imagen', 'TOTAL'),
        ('falso_etapas', None, 'fuera', 'fuera'),
        ('falso_etapas', None, 'dentro', 'dentro')])
    monkeypatch.setattr(te, '_fichero', None)
    return m


def test_cada_etapa_cuenta_su_tiempo_propio(tmp_path, monkeypatch):
    m = _modulo_falso(monkeypatch)
    ruta = tmp_path / 'tiempos.jsonl'
    monkeypatch.setenv(te.VARIABLE, str(ruta))
    assert te.instalar()
    m.imagen()
    m.imagen()
    te._fichero.close()
    lineas = [json.loads(x) for x in ruta.read_text('utf-8').splitlines()]
    assert len(lineas) == 2                            # una por imagen
    e = lineas[0]['etapas']
    # Propios: fuera ~20 ms (sin los 30 de dentro), dentro ~30, resto ~10.
    assert 15 <= e['fuera'] < 28
    assert 25 <= e['dentro'] < 40
    assert 5 <= e['resto'] < 20
    assert abs(sum(e.values()) - lineas[0]['total_ms']) < 1.0


def test_sin_la_variable_no_se_toca_nada(monkeypatch):
    m = _modulo_falso(monkeypatch)
    monkeypatch.delenv(te.VARIABLE, raising=False)
    original = m.imagen
    assert not te.instalar()
    assert m.imagen is original
