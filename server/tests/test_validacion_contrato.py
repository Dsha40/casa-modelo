"""Un campo que el contrato no conoce NO puede dejar ciega a una cámara.

## El día que esto se escribió (7-sep-2026)

El operador añadió una herramienta a su recuadro —una zona de personal— y su
cliente empezó a mandar `personal_zone_coordinates` y `personal_zone_activate`.
El contrato (`FrameInference`) es `extra='forbid'` y la validación va en modo
`estricto`, así que el servidor DESCARTÓ el 100 % de sus frames y respondió un
error por cada uno. El síntoma que llegó fue: «le doy play, pero no detecta».
Ninguna cámara, ningún log de detección, ninguna pista.

Había pasado unas horas antes con `linea_cruce_estilo`, el rótulo de la puerta
direccional. Dos veces el mismo día, con la misma forma: el cliente gana una
herramienta y el servidor se queda mudo.

`extra='forbid'` está para que una clave nueva o mal escrita se VEA, y eso se
conserva: queda en el log y en los contadores de `resumen()`. Lo que cambia es
el castigo. Un campo de más es un aviso; no es motivo para tirar el frame.
Cualquier OTRO error —tipo equivocado, obligatorio ausente, imagen ilegible—
sigue descartando el mensaje, que es de lo que el modo estricto protege.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

import cv2  # noqa: E402

from src.app import validacion_contrato as vc  # noqa: E402

JPEG = cv2.imencode('.jpg', np.zeros((64, 64, 3), np.uint8))[1].tobytes()


def _mensaje(**extra):
    """Un frame del formato que mandan los clientes de verdad."""
    return {'event': 'frame', 'id_connection': 'c-1',
            'type_inference': 'VigilanteAmazonas', 'component_key': 'k',
            'data': {'image': JPEG, 'roi_coordinates': [],
                     'roi_activate': False, **extra}}


@pytest.fixture(autouse=True)
def _estricto(monkeypatch):
    """El modo de producción desde el 31-jul-2026."""
    monkeypatch.setattr(vc, 'MODO', 'estricto')


def test_un_campo_que_el_contrato_no_declara_no_tumba_el_frame():
    """El caso real: el cliente estrena una herramienta y sigue viendo."""
    ok, motivo = vc.revisar(_mensaje(herramienta_nueva_del_recuadro=[1, 2]))
    assert ok is True and motivo is None


def test_pero_queda_anotado_para_que_se_vea():
    """Si no se registra, `extra='forbid'` deja de servir para lo suyo."""
    antes = dict(vc.resumen()['contadores'])
    vc.revisar(_mensaje(campo_mal_escrito_por_un_dedazo=True))
    despues = vc.resumen()['contadores']
    nuevas = [k for k in despues if k.startswith('campos_desconocidos')]
    assert nuevas, 'un campo desconocido tiene que dejar rastro'
    clave = nuevas[0]
    assert despues[clave] > antes.get(clave, 0)


def test_los_errores_de_verdad_siguen_descartando_el_mensaje():
    """Lo que el modo estricto protege no se toca."""
    ok, motivo = vc.revisar(_mensaje(roi_activate='ni booleano ni nada'))
    assert ok is False and 'payload invalido' in motivo
    # Y un obligatorio ausente tampoco pasa.
    sin_imagen = _mensaje()
    del sin_imagen['data']['image']
    ok, _ = vc.revisar(sin_imagen)
    assert ok is False


def test_un_campo_de_mas_junto_a_un_error_de_verdad_sigue_siendo_rechazo():
    """La tolerancia es SOLO cuando el único problema son campos de más."""
    ok, _ = vc.revisar(_mensaje(campo_desconocido=1,
                                roi_activate='ni booleano ni nada'))
    assert ok is False


def test_las_dos_herramientas_que_lo_provocaron_pasan():
    """La zona de personal y el rótulo de la puerta, que ya están
    declarados: pasan por el camino normal, no por la tolerancia."""
    for extra in ({'personal_zone_coordinates': [], 'personal_zone_activate': False},
                  {'linea_cruce': [[300, 0], [300, 400]],
                   'linea_cruce_activate': True,
                   'linea_cruce_estilo': 'puerta'}):
        assert vc.revisar(_mensaje(**extra)) == (True, None), extra
    from elde_core.contracts.eventos import FrameInference
    for campo in ('personal_zone_coordinates', 'personal_zone_activate',
                  'linea_cruce_estilo'):
        assert campo in FrameInference.model_fields, campo


def test_el_ayudante_distingue_bien_los_dos_casos():
    """La pieza que decide, aislada: mira el detalle de Pydantic y no el
    texto del error, que cambia entre versiones."""
    from elde_core.contracts.eventos import FrameInference
    import pydantic

    def error_de(payload):
        try:
            FrameInference(**payload)
        except pydantic.ValidationError as e:
            return e
        pytest.fail('se esperaba un error de validación')

    solo_extras = error_de({'image': JPEG, 'uno_de_mas': 1, 'otro_de_mas': 2})
    assert vc._solo_campos_desconocidos(solo_extras) == ['otro_de_mas', 'uno_de_mas']
    # Ojo: 'no' o 'false' SÍ son booleanos para Pydantic (coerción laxa);
    # hace falta algo que de verdad no lo sea.
    mezclado = error_de({'image': JPEG, 'uno_de_mas': 1,
                         'roi_activate': 'ni booleano ni nada'})
    assert vc._solo_campos_desconocidos(mezclado) is None
    assert vc._solo_campos_desconocidos(RuntimeError('nada que ver')) is None
