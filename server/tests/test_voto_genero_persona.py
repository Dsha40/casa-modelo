"""
El voto de genero por persona (22-sep-2026).

## Que se rompio

`_maybe_update_demographics` agrega el genero de una persona por VOTO
PONDERADO: cada lectura suma su confianza al cubo de su genero y se reporta el
mayoritario. Es lo que evita que una estimacion mala decida la etiqueta.

El filtro que decidia si un voto contaba pedia `('Hombres','Mujeres','Ninos')`
—el vocabulario de `result['category']`, que sirve para PINTAR el recuadro—
mientras que `demographics.py` manda `_GENDER_LIST`, o sea 'Hombre'/'Mujer' en
singular. La condicion no se cumplia nunca, `votos_genero` quedaba vacio y la
consolidacion por mayoria no se ejecutaba jamas.

## Por que importaba

El efecto no era solo perder la mayoria. Sin votos, la unica rama que escribe
el genero es la de respaldo, que solo actua si el valor guardado es `None` o
`'Desconocido'`; como al registrar ya se guarda la primera lectura, el genero
quedaba CONGELADO en ella. Y esa primera lectura es la peor: la de cuando la
persona asoma al fondo del pasillo, de lado y con la cara a ~20 px.

Peor aun, ese valor congelado se PROPAGA: `demographics.py:1371-1386` impone a
proposito el genero guardado cuando reconoce a alguien, en vez de volver a
mirarle la cara. Es correcto si lo guardado es el consenso; con el voto muerto,
era la primera impresion pisando a todas las lecturas buenas posteriores.

La EDAD no tenia este filtro (solo exige `age_value > 0`), asi que si se
corregia sola por la mediana. El mecanismo estaba medio vivo.

Todo con lecturas inyectadas: sin GPU, sin modelos y en milisegundos, igual
que `test_reid_ttl.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics.face_reidentifier import (  # noqa: E402
    FaceReidentifier, _GENEROS_VOTABLES)
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402

# Reutiliza el doble de ArcFace de la suite del TTL: un embedding
# determinista por 'rostro', sin red ni modelo.
from test_reid_ttl import _SesionArcFaceFalsa, _rostro  # noqa: E402

TRACK = 100001

#: Por encima de DEMO_VOTE_MIN_CONFIDENCE (0.50): estas lecturas SI votan.
ALTA = 0.95
MEDIA = 0.80
#: Por debajo del umbral: no deben contar.
BAJA = 0.20


@pytest.fixture()
def reid(tmp_path):
    return FaceReidentifier(
        embedding_session=_SesionArcFaceFalsa(),
        db_path=str(tmp_path / 'person_db' / 'persons.pkl'),
        similarity_threshold=0.35,
        reset_policy='ttl', ttl_seconds=1800.0, persistir=False)


def _ver(reid, genero, conf, *, edad='26-35', edad_v=30.0, cara=1):
    """Una lectura mas de la MISMA persona, como un frame mas del video."""
    reid.identify_or_register(track_id=TRACK, face_bgr=_rostro(cara),
                              gender=genero, age_range=edad,
                              age_value=edad_v, demo_confidence=conf)


def _ficha(reid):
    return reid._db[reid._track_to_person[TRACK]]


# ── El contrato que se rompio ───────────────────────────────────────────

def test_la_mayoria_gana_a_la_primera_lectura(reid):
    """EL caso del pasillo: entra de lado (mal) y luego se gira (bien)."""
    _ver(reid, 'Mujer', MEDIA)           # primera, de lejos
    for _ in range(5):
        _ver(reid, 'Hombre', ALTA)       # ya cerca, mirando el anaquel

    assert _ficha(reid)['gender'] == 'Hombre'


def test_sin_el_arreglo_mandaria_la_primera(reid):
    """Control explicito: la primera lectura NO puede imponerse."""
    _ver(reid, 'Mujer', MEDIA)
    primera = _ficha(reid)['gender']
    for _ in range(5):
        _ver(reid, 'Hombre', ALTA)

    assert primera == 'Mujer'                      # asi empezo
    assert _ficha(reid)['gender'] != primera       # y NO se quedo asi


def test_los_votos_se_acumulan_de_verdad(reid):
    _ver(reid, 'Hombre', ALTA)
    _ver(reid, 'Hombre', ALTA)

    votos = _ficha(reid)['votos_genero']
    assert set(votos) == {'Hombre'}
    assert votos['Hombre'] == pytest.approx(ALTA * 2)


def test_el_voto_pesa_por_confianza_no_por_cantidad(reid):
    """Tres lecturas flojas no tumban a dos muy seguras."""
    _ver(reid, 'Mujer', 0.55)
    _ver(reid, 'Mujer', 0.55)
    _ver(reid, 'Mujer', 0.55)            # 1.65 en total
    _ver(reid, 'Hombre', ALTA)
    _ver(reid, 'Hombre', ALTA)           # 1.90 en total

    assert _ficha(reid)['gender'] == 'Hombre'


# ── Lo que NO debe votar ────────────────────────────────────────────────

def test_una_lectura_dudosa_no_vota(reid):
    _ver(reid, 'Hombre', ALTA)
    _ver(reid, 'Mujer', BAJA)            # por debajo del umbral

    assert set(_ficha(reid)['votos_genero']) == {'Hombre'}


def test_desconocido_no_entra_en_el_recuento(reid):
    """'Desconocido' es el estado transitorio del clasificador, no un genero."""
    _ver(reid, 'Hombre', ALTA)
    _ver(reid, 'Desconocido', ALTA)

    assert set(_ficha(reid)['votos_genero']) == {'Hombre'}
    assert _ficha(reid)['gender'] == 'Hombre'


def test_una_correccion_manual_es_intocable(reid):
    """Si alguien corrigio a mano, ningun voto puede pisarlo."""
    _ver(reid, 'Mujer', ALTA)
    _ficha(reid)['manual_override'] = True
    for _ in range(5):
        _ver(reid, 'Hombre', ALTA)

    assert _ficha(reid)['gender'] == 'Mujer'


# ── El guardian: que los dos vocabularios no se separen otra vez ────────

def test_el_vocabulario_del_voto_es_el_que_manda_demographics():
    """ESTA es la prueba que habria cazado el fallo.

    El voto solo funciona si acepta exactamente los valores que produce
    `demographics.py`. Estaban escritos por separado, se desincronizaron y
    nadie se entero porque el sintoma era silencioso. Aqui se atan.
    """
    from src.analityc.core.analytics.demographics import _GENDER_LIST

    assert tuple(_GENDER_LIST) == _GENEROS_VOTABLES


def test_el_umbral_del_voto_sigue_siendo_el_de_la_config():
    """Las confianzas de estas pruebas se eligieron contra este numero."""
    umbral = float(AnalyticsConfig.DEMO_VOTE_MIN_CONFIDENCE)
    assert BAJA < umbral <= 0.55 < MEDIA < ALTA


# ── Que no se haya roto la edad, que ya funcionaba ──────────────────────

def test_la_edad_se_sigue_corrigiendo_por_mediana(reid):
    _ver(reid, 'Hombre', ALTA, edad='51-65', edad_v=60.0)   # disparate inicial
    for _ in range(5):
        _ver(reid, 'Hombre', ALTA, edad='26-35', edad_v=30.0)

    assert _ficha(reid)['age_value'] == pytest.approx(30.0)


def test_una_edad_ausente_no_rompe_el_voto_de_genero(reid):
    _ver(reid, 'Hombre', ALTA, edad=None, edad_v=0.0)

    assert _ficha(reid)['votos_genero'] == {'Hombre': pytest.approx(ALTA)}
