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


def test_separa_procesador_de_espera(tmp_path, monkeypatch):
    """29-oct-2026: `cpu` cuenta el procesador del hilo, no la espera. 200 ms
    por etapa: en Windows ese reloj va a saltos de ~15,6 ms, y con el PC
    ocupado el hilo no siempre tiene el procesador (medido: 78 de 120 ms y
    125-140 de 200): por eso el listón de «calcula» es bajo."""
    m = types.ModuleType('falso_cpu')

    def calcula():                      # procesador todo el rato
        fin = time.perf_counter() + 0.2
        while time.perf_counter() < fin:
            pass

    def espera():                       # como esperar a la GPU
        time.sleep(0.2)

    def imagen():
        m.calcula()
        m.espera()

    m.calcula, m.espera, m.imagen = calcula, espera, imagen
    monkeypatch.setitem(sys.modules, 'falso_cpu', m)
    monkeypatch.setattr(te, '_OBJETIVOS', [
        ('falso_cpu', None, 'imagen', 'TOTAL'),
        ('falso_cpu', None, 'calcula', 'calcula'),
        ('falso_cpu', None, 'espera', 'espera')])
    monkeypatch.setattr(te, '_fichero', None)
    ruta = tmp_path / 'cpu.jsonl'
    monkeypatch.setenv(te.VARIABLE, str(ruta))
    assert te.instalar()
    m.imagen()
    te._fichero.close()
    linea = json.loads(ruta.read_text('utf-8').splitlines()[0])
    assert linea['cpu']['calcula'] >= 50
    assert linea['cpu']['espera'] < 40
    assert linea['etapas']['espera'] >= 180


def test_sin_la_variable_no_se_toca_nada(monkeypatch):
    m = _modulo_falso(monkeypatch)
    monkeypatch.delenv(te.VARIABLE, raising=False)
    original = m.imagen
    assert not te.instalar()
    assert m.imagen is original
