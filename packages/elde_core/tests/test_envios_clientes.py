"""
Los CUATRO clientes deben poder enviar por WhatsApp y por Jarvis (5-ago-2026).

## Por que una prueba que lee el codigo fuente

El hueco que esto blinda no era un fallo de logica: era **cableado que no
existia**. `tienda`, `managers` y `amazonas` construian `Jarvis_api`, abrian
sesion y poblaban el selector de establecimientos... y jamas llamaban a
`enviar_novedad_async`. Todo compilaba, la UI parecia completa y no se enviaba
ni una novedad. Ninguna prueba unitaria del reenviador habria detectado eso,
porque el reenviador estaba bien: lo que faltaba era el `connect`.

Asi que aqui se comprueba la CONEXION, leyendo el `main.py` de cada cliente:
que se cree el reenviador, que las alertas de los recuadros lleguen a el, y que
el interruptor del pie lo gobierne. Es una prueba de integracion barata (sin
Qt, sin venvs, sin red) que falla el dia que alguien borre una linea de
cableado.

`amazonas` no tiene panel de alertas (su lateral son las capturas de disco), asi
que ahi la senal va solo a Jarvis; por eso el panel no se exige.
"""
from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[3]
# Son las CARPETAS, no los `client_type`: 'managers' se renombró a
# 'detecciones' el 6-ago-2026 y su identificador de contrato no cambió.
CLIENTES = ('perimetrales', 'tienda', 'amazonas', 'detecciones')


def _fuente(cliente: str, *partes: str) -> str:
    ruta = RAIZ / 'clients' / cliente / 'src' / Path(*partes)
    assert ruta.is_file(), f'no existe {ruta}'
    return ruta.read_text(encoding='utf-8')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_pie_tiene_los_dos_interruptores(cliente):
    """Sin las dos casillas en el pie, el operador no puede encender nada."""
    src = _fuente(cliente, 'gui', 'components', 'custom_status_bar.py')
    for casilla in ('chk_envio_jarvis', 'chk_envio_whatsapp'):
        assert f'self.{casilla} = QCheckBox' in src, (
            f'{cliente}: falta la casilla {casilla} en el pie')
        assert f'self.{casilla}.toggled.connect' in src, (
            f'{cliente}: la casilla {casilla} no notifica a nadie')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_las_alertas_del_recuadro_llegan_al_reenviador(cliente):
    """La senal existe, se emite y alguien la escucha."""
    caja = _fuente(cliente, 'gui', 'components', 'render_box', 'render_box.py')
    assert 'alert_received' in caja and 'Signal(dict)' in caja, (
        f'{cliente}: el recuadro no declara la senal de alertas')
    assert 'self.alert_received.emit(' in caja, (
        f'{cliente}: la senal de alertas no se emite nunca')

    principal = _fuente(cliente, 'main.py')
    assert 'alert_received.connect(jarvis_forwarder.on_alert)' in principal, (
        f'{cliente}: las alertas no llegan al reenviador de Jarvis')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_reenviador_de_jarvis_esta_construido_y_gobernado(cliente):
    principal = _fuente(cliente, 'main.py')
    assert 'JarvisAlertForwarder(' in principal, (
        f'{cliente}: no se crea el reenviador de Jarvis')
    assert 'chk_envio_jarvis' in principal, (
        f'{cliente}: el interruptor del pie no se cablea al reenviador')
    assert 'jarvis_envio_activo' in principal, (
        f'{cliente}: el estado del interruptor no se persiste')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_flag_de_whatsapp_viaja_en_cada_frame(cliente):
    """El envio lo hace el SERVIDOR: si el flag no va en el payload, no hay
    forma de que llegue nada al grupo por muy marcada que este la casilla."""
    caja = _fuente(cliente, 'gui', 'components', 'render_box', 'render_box.py')
    assert '"enviar_whatsapp": self.whatsapp_boolean' in caja, (
        f'{cliente}: el flag de WhatsApp no viaja en el payload del frame')

    principal = _fuente(cliente, 'main.py')
    assert 'box.whatsapp_boolean' in principal, (
        f'{cliente}: la casilla del pie no se propaga a los recuadros')
    assert 'whatsapp_envio_activo' in principal, (
        f'{cliente}: el estado del interruptor no se persiste')


@pytest.mark.parametrize('cliente', CLIENTES)
def test_el_reenviador_es_el_del_nucleo(cliente):
    """Una sola implementacion. Cuatro copias divergentes fue como se llego a
    que tres clientes se quedaran sin envio a Jarvis."""
    principal = _fuente(cliente, 'main.py')
    importa_nucleo = 'elde_core.alertas.jarvis_forwarder' in principal
    # perimetrales importa por su shim historico, que redirige al nucleo.
    importa_shim = 'jarvis_alert_forwarder' in principal
    assert importa_nucleo or importa_shim, (
        f'{cliente}: no usa el reenviador compartido')
    if importa_shim:
        shim = _fuente(cliente, 'core', 'network', 'jarvis_alert_forwarder.py')
        assert 'elde_core.alertas' in shim, (
            f'{cliente}: el shim ya no redirige al nucleo')


def test_el_servidor_reenvia_a_whatsapp_los_pipelines_que_no_son_vigilante():
    """VigilanteWS tiene su ruta propia; el resto pasa por whatsapp_alertas.

    Sin esta rama, la casilla de tienda/amazonas/managers seria decorativa.

    Desde el 7-ago-2026 el reenvio ya no vive en el bucle del endpoint: lo
    hace el drenador (`ws_drenaje.DrenajeCliente`), que recibe
    `_whatsapp.reenviar` cableado desde app.py. Se comprueban las dos partes:
    que app.py lo cablea y que el drenador excluye a los pipelines con ruta
    propia (VigilanteAmazonas y Estacionamiento, 17-ago-2026: reenviarlos
    duplicaba cada aviso).
    """
    app = (RAIZ / 'server' / 'src' / 'app' / 'app.py').read_text(
        encoding='utf-8')
    assert 'data.get("enviar_whatsapp")' in app
    assert 'reenviar_whatsapp=_whatsapp.reenviar' in app

    drenaje = (RAIZ / 'server' / 'src' / 'app' / 'ws_drenaje.py').read_text(
        encoding='utf-8')
    assert 'self._reenviar_whatsapp(' in drenaje
    assert 'type_inference not in RUTA_PROPIA_WHATSAPP' in drenaje
    assert '"VigilanteAmazonas", "Estacionamiento"' in drenaje
