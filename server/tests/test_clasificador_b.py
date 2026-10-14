"""Clasificador oficial «B sin herencia» (9-oct-2026, clasificador_b.py).

Se fija lo que se validó en vivo el 8-oct contra etiquetas a mano:
- la regla de decisión por evidencia (2 lecturas muy claras bastan; flojas
  no; un desacuerdo fuerte bloquea);
- el interruptor ELDE_CLASIFICADOR (b por defecto, `clasico` vuelve atrás);
- NO hereda el género del reconocimiento de caras;
- NO vuelve a mirar a quien ya decidió (sin histéresis).
"""
import numpy as np
import pytest

from src.analityc.core.analytics import clasificador_b as cb
from src.analityc.core.analytics.config import AnalyticsConfig


def _acc(lecturas, peso=1.0):
    a = cb._AcumuladorEvidencia(AnalyticsConfig)
    for ph in lecturas:
        a.add(np.array([ph, 1.0 - ph]), np.full(8, 1 / 8), 30.0, weight=peso, source='body')
    return a


def test_dos_lecturas_muy_claras_deciden():
    r = _acc([0.98, 0.98]).try_commit(5, 0.85, 0.35)
    assert r is not None and r['gender'] == 'Hombre'


def test_una_sola_lectura_no_decide():
    assert _acc([0.999]).try_commit(5, 0.85, 0.35) is None


def test_lecturas_flojas_no_deciden():
    assert _acc([0.6, 0.62, 0.58]).try_commit(5, 0.85, 0.35) is None


def test_desacuerdo_fuerte_bloquea():
    # Evidencia neta hacia Mujer, pero 2 de 5 votos en contra (40 % > 25 %).
    assert _acc([0.02, 0.02, 0.02, 0.9, 0.9]).try_commit(5, 0.85, 0.35) is None


def test_interruptor(monkeypatch):
    monkeypatch.delenv('ELDE_CLASIFICADOR', raising=False)
    assert cb.usar_b() is True
    monkeypatch.setenv('ELDE_CLASIFICADOR', 'clasico')
    assert cb.usar_b() is False


def _b_sin_modelos():
    """Un ClasificadorB sin cargar modelos: solo para probar su lógica."""
    c = object.__new__(cb.ClasificadorB)
    c._accum = {}
    return c


def test_no_hereda_el_genero_del_reconocimiento():
    c = _b_sin_modelos()
    acc = _acc([])
    assert c._commit_from_reid(acc, {'gender': 'Hombre', 'demo_confidence': 0.99}, 7) is None
    assert acc.committed is False
    reid = cb._ReidSinDemografia(type('R', (), {'get_demographics_for_person':
                                                 lambda self, t: {'gender': 'Hombre'},
                                                 'is_available': True})())
    assert reid.get_demographics_for_person(7) is None
    assert reid.is_available is True                    # el resto, intacto


def test_quien_ya_decidio_no_se_vuelve_a_mirar():
    c = _b_sin_modelos()
    acc = _acc([0.98, 0.98])
    acc.try_commit(5, 0.85, 0.35)
    c._accum[7] = acc
    # Si llamara al clasificador de siempre (histéresis), fallaría: no tiene
    # modelos. Debe devolver el resultado guardado sin mirar la imagen.
    assert c.classify(None, (0, 0, 10, 10), 7)['gender'] == 'Hombre'


def _acc_fuente(lecturas, fuente):
    a = cb._AcumuladorEvidencia(AnalyticsConfig)
    for ph in lecturas:
        a.add(np.array([ph, 1.0 - ph]), np.full(8, 1 / 8), 30.0, weight=1.0, source=fuente)
    return a


def test_con_cara_basta_el_90_y_solo_cuerpo_exige_el_95():
    """12-oct-2026: con cara acertaba el 97 % en vivo y solo con cuerpo el
    93 %. Dos lecturas de 0,92 (evidencia 2,44) pasan el 90 % (2,20) pero no
    el 95 % (2,94)."""
    assert _acc_fuente([0.92, 0.92], 'face').try_commit(5, 0.85, 0.35) is not None
    assert _acc_fuente([0.92, 0.92], 'body').try_commit(5, 0.85, 0.35) is None
