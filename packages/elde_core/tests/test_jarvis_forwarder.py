"""
Pruebas del reenviador de alertas a Jarvis compartido (5-ago-2026).

Cubren las DOS politicas que conviven, porque la diferencia entre clientes ya
no es codigo duplicado sino parametros:

  - PERIMETRAL (`clases_permitidas` por defecto, `local_global=False`):
    filtro de clases, agrupacion por grupo, y el destino sale SOLO del select
    "Local" de la camara.
  - TIENDA/MANAGERS/AMAZONAS (`clases_permitidas=None`, `local_global=True`):
    sin filtro de clase (el class_name es demografia), un visitante = una
    novedad, y el destino cae al selector global del pie.

Sin red y sin GUI: se usa un doble de `Jarvis_api`.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402

from elde_core.alertas.jarvis_forwarder import JarvisAlertForwarder  # noqa: E402


def test_el_titulo_propio_de_la_alerta_manda():
    """`titulo_novedad` va tal cual: sin el titulo de la ventana de iVMS.

    Lo manda PermanenciaArea (10-ago-2026) para que la novedad diga
    «Permanencia en el pasillo de cooler» y no «… camara iVMS-4200».
    """
    titulo = JarvisAlertForwarder._titulo(
        {'titulo_novedad': 'Permanencia en el pasillo de cooler',
         'camera_name': 'iVMS-4200'}, 'salida', 'PERSONA')
    assert titulo == 'Permanencia en el pasillo de cooler'
    assert 'iVMS' not in titulo


def test_el_titulo_lleva_el_local_no_el_titulo_de_ivms():
    """El cierre del titulo es el LOCAL de destino (17-ago-2026); el titulo
    de la ventana de iVMS queda solo de respaldo cuando no hay local."""
    con_local = JarvisAlertForwarder._titulo(
        {'camera_name': 'iVMS-4200'}, 'llegada', 'CARRO', 'AVILA')
    assert con_local == 'CARRO detectado en el área — AVILA'
    sin_local = JarvisAlertForwarder._titulo(
        {'camera_name': 'iVMS-4200'}, 'llegada', 'CARRO', '')
    assert sin_local == 'CARRO detectado en el área — iVMS-4200'
    # El centinela '?' de _nombre_local (destino sin nombre legible) no debe
    # colarse en el titulo: cae a la camara igual que el local vacio.
    centinela = JarvisAlertForwarder._titulo(
        {'camera_name': 'iVMS-4200'}, 'llegada', 'CARRO', '?')
    assert centinela == 'CARRO detectado en el área — iVMS-4200'


@pytest.fixture(scope='module', autouse=True)
def _qt():
    """QObject necesita un QCoreApplication vivo."""
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    yield app


class _JarvisFalso:
    def __init__(self, global_=None):
        self.session_user = {'name': 'X'}
        self.selected_establishment = global_
        self.llamadas: list[dict] = []

    def enviar_novedad_async(self, base64_image='', title='', message='',
                             establecimiento=None):
        self.llamadas.append({'title': title, 'img': base64_image,
                              'destino': establecimiento})


def _alerta(evento, clase, gid='', cam='patio', local='', **extra):
    base = {'event_type': evento, 'class_name': clase, 'global_id': gid,
            'camera_name': cam, 'description': f'{clase} {evento}',
            'image_base64': 'AAAA', 'establecimiento': local}
    base.update(extra)
    return base


def _envejecer(fwd):
    """Empuja las claves fuera de la ventana de agrupacion."""
    for k in list(fwd._ultimo):
        fwd._ultimo[k] -= (fwd.VENTANA_GRUPO_SEG + 1.0)


# ── Politica PERIMETRAL ──────────────────────────────────────────────

def test_perimetral_agrupa_el_grupo_de_una_clase():
    api = _JarvisFalso()
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T1', local='La Comarca'))
    assert len(api.llamadas) == 1
    # Cinco mas de la misma clase y camara dentro de la ventana: 1 sola novedad.
    for i in range(2, 7):
        fwd.on_alert(_alerta('llegada', 'PERSONA', f'T{i}', local='La Comarca'))
    assert len(api.llamadas) == 1 and fwd.agrupadas == 5
    # Pasada la ventana, vuelve a alertar.
    _envejecer(fwd)
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T7', local='La Comarca'))
    assert len(api.llamadas) == 2


def test_perimetral_clases_distintas_no_se_agrupan_entre_si():
    api = _JarvisFalso()
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T1', local='L'))
    fwd.on_alert(_alerta('llegada', 'CARRO', 'C1', local='L'))
    # 'objeto' esta fuera de la politica: no se reenvia.
    fwd.on_alert(_alerta('llegada', 'OBJETO', 'O1', local='L'))
    assert len(api.llamadas) == 2
    # La salida es un evento distinto: alerta aunque acabe de entrar.
    fwd.on_alert(_alerta('salida', 'PERSONA', 'T1', local='L'))
    assert len(api.llamadas) == 3


def test_perimetral_reid_no_se_agrupa_por_clase():
    api = _JarvisFalso()
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('alerta', 'Juan Perez', 'VIG-RID-1', local='L'))
    fwd.on_alert(_alerta('alerta', 'Ana Gomez', 'VIG-RID-2', local='L'))
    assert len(api.llamadas) == 2, 'cada persona identificada es una novedad'


def test_perimetral_sin_local_no_envia_y_no_consume_la_ventana():
    """Una camara sin local no envia; y al elegirlo, la siguiente SI sale.

    Regresion: si la clave de agrupacion se marcara antes de comprobar el
    destino, elegir el local dejaria al operador 30 s sin novedades sin
    ninguna razon visible.
    """
    api = _JarvisFalso()
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T1', local=''))
    assert api.llamadas == [] and fwd.sin_local == 1
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T2', local='La Comarca'))
    assert len(api.llamadas) == 1


def test_perimetral_ignora_el_global_aunque_exista():
    """El respaldo global se elimino a proposito el 1-ago-2026."""
    api = _JarvisFalso(global_={'name': 'Otro Local'})
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T1', local=''))
    assert api.llamadas == [], 'no debe caer al global: desviaria la alerta'


# ── Politica TIENDA / MANAGERS / AMAZONAS ────────────────────────────

def _fwd_generico(global_=None):
    return JarvisAlertForwarder(_JarvisFalso(global_),
                                clases_permitidas=None, local_global=True)


# ── Clave de agrupacion propia del pipeline ──────────────────────────

def test_un_pipeline_puede_fijar_su_propia_unidad_de_agrupacion():
    """`dedup_clave` gana a la politica por clase.

    PermanenciaArea la manda con la PERSONA dentro. Sin este escape, dos
    personas entrando por la misma puerta con pocos segundos de diferencia
    compartirian clave (camara + tipo de evento) y la segunda se agruparia:
    el operador se quedaria sin su aviso y sin su tiempo de permanencia.
    """
    fwd = _fwd_generico({'name': 'L'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('Entrada al área', 'Entrada al área',
                         dedup_clave='perm|cam1|entrada|p1'))
    fwd.on_alert(_alerta('Entrada al área', 'Entrada al área',
                         dedup_clave='perm|cam1|entrada|p2'))
    assert len(api.llamadas) == 2, 'la segunda persona se perdio'


def test_la_misma_persona_repetida_si_se_agrupa():
    """El escape afina la clave; no desactiva el anti-flood."""
    fwd = _fwd_generico({'name': 'L'})
    api = fwd._jarvis
    for _ in range(4):
        fwd.on_alert(_alerta('Permanencia prolongada', 'Permanencia prolongada',
                             dedup_clave='perm|cam1|permanencia|p1'))
    assert len(api.llamadas) == 1 and fwd.agrupadas == 3


def test_entrada_y_salida_de_la_misma_persona_son_avisos_distintos():
    fwd = _fwd_generico({'name': 'L'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('Entrada al área', 'Entrada al área',
                         dedup_clave='perm|cam1|entrada|p1'))
    fwd.on_alert(_alerta('Salida del área', 'Salida del área',
                         dedup_clave='perm|cam1|salida|p1'))
    assert len(api.llamadas) == 2


def test_sin_dedup_clave_la_politica_de_siempre_no_cambia():
    api = _JarvisFalso()
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T1', local='L'))
    fwd.on_alert(_alerta('llegada', 'PERSONA', 'T2', local='L'))
    assert len(api.llamadas) == 1 and fwd.agrupadas == 1


def test_generico_envia_visitantes_con_demografia_como_clase():
    fwd = _fwd_generico({'name': 'Tienda 1', '_id': 'id-1'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('Visitante', 'Femenino 25-32',
                         cam='pasillo', person_uuid='p-1'))
    assert len(api.llamadas) == 1, (
        'el filtro de clases perimetrales tiraria esta alerta')
    assert api.llamadas[0]['destino'] == {'name': 'Tienda 1', '_id': 'id-1'}
    assert 'Femenino 25-32' in api.llamadas[0]['title']


def test_generico_una_novedad_por_persona_no_por_demografia():
    """Dos mujeres del mismo rango de edad son DOS visitantes."""
    fwd = _fwd_generico({'name': 'Tienda 1'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('Visitante', 'Femenino 25-32', person_uuid='p-1'))
    fwd.on_alert(_alerta('Visitante', 'Femenino 25-32', person_uuid='p-2'))
    assert len(api.llamadas) == 2
    # La misma persona repetida dentro de la ventana SI se agrupa.
    fwd.on_alert(_alerta('Visitante', 'Femenino 25-32', person_uuid='p-1'))
    assert len(api.llamadas) == 2 and fwd.agrupadas == 1


def test_generico_sin_local_seleccionado_no_envia():
    fwd = _fwd_generico(None)
    api = fwd._jarvis
    fwd.on_alert(_alerta('Visitante', 'Masculino 33-45', person_uuid='p-9'))
    assert api.llamadas == [] and fwd.sin_local == 1


def test_generico_el_local_de_la_camara_gana_al_global():
    fwd = _fwd_generico({'name': 'Global'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('Visitante', 'Masculino 33-45',
                         local='Sucursal Norte', person_uuid='p-3'))
    assert api.llamadas[0]['destino'] == 'Sucursal Norte'


def test_captura_manual_no_llega_a_jarvis():
    """Pulsar 'capturar' es un evento de UI, no una novedad del local."""
    fwd = _fwd_generico({'name': 'Tienda 1'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('Captura guardada', 'Captura', person_uuid='c-1'))
    assert api.llamadas == [] and fwd.ignoradas == 1


# ── Interruptor ──────────────────────────────────────────────────────

def test_el_interruptor_corta_el_envio_sin_tocar_el_panel():
    api = _JarvisFalso({'name': 'Tienda 1'})
    fwd = JarvisAlertForwarder(api, clases_permitidas=None, local_global=True)
    fwd.set_activo(False)
    fwd.on_alert(_alerta('Visitante', 'Femenino 25-32', person_uuid='p-1'))
    assert api.llamadas == []
    fwd.set_activo(True)
    fwd.on_alert(_alerta('Visitante', 'Femenino 25-32', person_uuid='p-2'))
    assert len(api.llamadas) == 1


def test_titulo_de_un_evento_desconocido_no_pierde_informacion():
    """Un modo nuevo (estacionamiento…) debe leerse, no caer en 'Alerta'."""
    fwd = _fwd_generico({'name': 'L'})
    api = fwd._jarvis
    fwd.on_alert(_alerta('pernocta', 'CARRO', 'T1'))
    assert 'Pernocta' in api.llamadas[0]['title']
    assert 'CARRO' in api.llamadas[0]['title']


def test_el_envio_completo_titula_con_el_local_de_destino():
    """De punta a punta: la novedad enviada dice el local, no la ventana."""
    api = _JarvisFalso()
    fwd = JarvisAlertForwarder(api)
    fwd.on_alert(_alerta('merodeo', 'CARRO', 'VIG-MER-1',
                         cam='iVMS-4200', local='AVILA'))
    assert len(api.llamadas) == 1
    titulo = api.llamadas[0]['title']
    assert 'AVILA' in titulo and 'iVMS' not in titulo
