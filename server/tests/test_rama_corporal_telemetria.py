"""
La telemetria tiene que decir la verdad sobre la rama corporal (22-sep-2026).

## Que se rompio

`get_quality_report()["rama_corporal_activa"]` se calculaba mirando SOLO las
dos vias legadas: `self._par` (PAR ONNX) y `self._mivolo` (el MiVOLO ONNX
antiguo). Desde el Hito 4 la ruta PRINCIPAL es otra —`_estimador_cuerpo`,
MiVOLO v2, que `_infer_body_only` prueba ANTES que las dos anteriores— y la
bandera no la miraba.

## Por que importaba

Medido sobre un video real del pasillo (22-sep-2026): la telemetria reportaba
`rama_corporal_activa: False` mientras la rama habia ACEPTADO 235 muestras
corporales. Leida junto al contador `body_sin_modelo` —cuyo nombre sugiere
"falta el modelo" cuando en realidad cuenta "MiVOLO corrio y no dio un
resultado valido"— lleva derecho a dos conclusiones falsas:

  1. que la rama corporal esta apagada, y
  2. que generar `par_gender.onnx` la encenderia.

Lo segundo es ademas inutil: `_infer_body_only` ni mira `_par` mientras
`_estimador_cuerpo` exista. Se perderia el trabajo entero.

Sin cargar modelos: el estimador se sustituye por un doble.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics import (  # noqa: E402
    estimador_edad_genero as _est)
from src.analityc.core.analytics.demographics import (  # noqa: E402
    DemographicsClassifier)


class _EstimadorFalso:
    """Doble de MiVOLO v2: dice que esta listo y no carga nada."""
    disponible = True
    backend = 'doble'


def _clasificador(monkeypatch, con_estimador: bool):
    """DemographicsClassifier sin sesiones ONNX reales."""
    if con_estimador:
        monkeypatch.setattr(_est, 'obtener_estimador', lambda: _EstimadorFalso())
    else:
        monkeypatch.setattr(_est, 'obtener_estimador',
                            lambda: (_ for _ in ()).throw(
                                RuntimeError('no disponible en esta prueba')))
    return DemographicsClassifier()


def test_con_mivolo_v2_la_bandera_dice_activa(monkeypatch):
    """ESTA es la que habria cazado el fallo."""
    clas = _clasificador(monkeypatch, con_estimador=True)

    assert clas._estimador_cuerpo is not None        # la via principal existe
    assert clas._par is None and clas._mivolo is None  # las legadas no
    assert clas.get_quality_report()['rama_corporal_activa'] is True


def test_sin_ninguna_via_la_bandera_dice_inactiva(monkeypatch):
    """Control negativo: sin MiVOLO v2 ni las legadas, sigue siendo False."""
    clas = _clasificador(monkeypatch, con_estimador=False)

    assert clas._estimador_cuerpo is None
    assert clas.get_quality_report()['rama_corporal_activa'] is False


def test_una_via_legada_sola_sigue_contando(monkeypatch):
    """No se rompe lo que ya funcionaba: PAR a secas sigue marcando activa."""
    clas = _clasificador(monkeypatch, con_estimador=False)
    clas._par = object()

    assert clas.get_quality_report()['rama_corporal_activa'] is True


def test_mivolo_v2_se_prueba_antes_que_las_vias_legadas():
    """El orden es el que hace inutil generar PAR teniendo MiVOLO v2.

    Se comprueba sobre el codigo: dentro de `_infer_body_only`, la rama de
    `_estimador_cuerpo` aparece ANTES que la de `_par`. Si alguien invirtiera
    el orden, generar un PAR pasaria a tener efecto y este comentario —y la
    recomendacion que se dio al usuario— dejarian de ser ciertos.
    """
    import inspect
    fuente = inspect.getsource(DemographicsClassifier._infer_body_only)

    assert fuente.index('self._estimador_cuerpo') < fuente.index('self._par')
