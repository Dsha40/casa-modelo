"""
El reenvio a WhatsApp de los pipelines que NO son VigilanteWS (5-ago-2026).

`VigilanteAmazonas` tiene su ruta propia dentro de su adaptador. Todo lo demas
—tienda, amazonas, managers y cualquier modo futuro— pasa por
`src/app/whatsapp_alertas.reenviar()`. Si ese modulo se rompe, la casilla
«Enviar por WhatsApp» de esos tres clientes queda decorativa: marcada, sin
enviar nada, y sin ninguna senal de que no funciona.

Se prueba con un emisor falso: aqui interesa QUE se encola y CON QUE texto, no
que el bot conteste (el bot es red, y ademas Proton VPN lo bloquea en este
equipo).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

from src.app import whatsapp_alertas  # noqa: E402


class _EmisorFalso:
    def __init__(self):
        self.envios: list[tuple[str, str, str]] = []

    def enviar_b64(self, imagen_b64: str, texto: str, clave: str = "") -> bool:
        self.envios.append((imagen_b64, texto, clave))
        return True


@pytest.fixture
def emisor(monkeypatch):
    falso = _EmisorFalso()
    monkeypatch.setattr(whatsapp_alertas, '_obtener_emisor', lambda: falso)
    return falso


def _visitante(**extra):
    """La alerta tal como la publica el pipeline de tienda."""
    base = {'event_type': 'Visitante', 'class_name': 'Femenino 25-32',
            'description': 'Visitante Femenino, 25-32 años. Primera visita '
                           'registrada.',
            'timestamp': '2026-08-05 14:42:00', 'crop_image': 'AAAA'}
    base.update(extra)
    return base


def test_una_alerta_de_tienda_se_encola(emisor):
    n = whatsapp_alertas.reenviar({'alerts': [_visitante()]},
                                  camera_name='pasillo', camera_id='cam-1')
    assert n == 1 and len(emisor.envios) == 1


def test_el_visitante_no_lleva_icono_de_alarma(emisor):
    """Un cliente de tienda con 🚨 delante confunde a quien lee el grupo:
    debe distinguirse de un vistazo de una intrusion."""
    whatsapp_alertas.reenviar({'alerts': [_visitante()]},
                              camera_name='pasillo', camera_id='cam-1')
    texto = emisor.envios[0][1]
    assert texto.startswith('👤'), texto
    assert '🚨' not in texto


def test_una_intrusion_si_lleva_icono_de_alarma(emisor):
    whatsapp_alertas.reenviar(
        {'alerts': [{'event_type': 'intrusion', 'class_name': 'PERSONA',
                     'description': 'Persona en zona restringida',
                     'timestamp': '14:42:00', 'crop_image': 'AAAA'}]},
        camera_name='patio', camera_id='cam-2')
    assert emisor.envios[0][1].startswith('🚨')


def test_el_texto_lleva_camara_descripcion_y_hora(emisor):
    whatsapp_alertas.reenviar({'alerts': [_visitante()]},
                              camera_name='pasillo', camera_id='cam-1')
    texto = emisor.envios[0][1]
    assert 'pasillo' in texto
    assert 'Primera visita' in texto
    assert '14:42:00' in texto


def test_el_local_sustituye_a_la_camara_y_la_hora_va_legible(emisor):
    """Formato del 31-ago-2026: encabezado con SOLO el local + detalle +
    hora legible. La clase no se repite en el encabezado (ya va en la
    descripcion); antes salia «CARRO · AVILA» y luego «MERODEO: CARRO…»."""
    epoch = 1786984080.5534697
    whatsapp_alertas.reenviar(
        {'alerts': [{'event_type': 'merodeo', 'class_name': 'CARRO',
                     'description': 'MERODEO: CARRO entró 3 veces al área '
                                    'en 1 min (similitud 0.98)',
                     'timestamp': str(epoch), 'crop_image': 'AAAA'}]},
        camera_name='iVMS-4200', camera_id='cam-9', establecimiento='AVILA')
    lineas = emisor.envios[0][1].split('\n')
    assert lineas[0] == '🚨 AVILA'
    assert lineas[1].startswith('MERODEO: CARRO entró 3 veces')
    hora = time.strftime('%H:%M:%S', time.localtime(epoch))
    assert lineas[2] == f'🕒 {hora}'
    assert 'iVMS-4200' not in '\n'.join(lineas)
    assert str(epoch) not in '\n'.join(lineas)


def test_sin_descripcion_el_evento_no_desaparece(emisor):
    """Con el encabezado sin clase (31-ago-2026), una alerta SIN descripcion
    debe decir igual QUE paso: la clase (o el tipo de evento) ocupa esa
    linea. Sin esto el mensaje seria solo local + hora."""
    whatsapp_alertas.reenviar(
        {'alerts': [{'event_type': 'intrusion', 'class_name': 'PERSONA',
                     'description': '', 'timestamp': '14:42:00',
                     'crop_image': 'AAAA'}]},
        camera_name='patio', camera_id='cam-2', establecimiento='AVILA')
    lineas = emisor.envios[0][1].split('\n')
    assert lineas[0] == '🚨 AVILA'
    assert lineas[1] == 'PERSONA'


def test_la_clave_separa_camaras_para_el_antiflood(emisor):
    """Dos camaras distintas no deben taparse la una a la otra."""
    for cam in ('cam-1', 'cam-2'):
        whatsapp_alertas.reenviar({'alerts': [_visitante()]},
                                  camera_name=cam, camera_id=cam)
    claves = {c for _, _, c in emisor.envios}
    assert len(claves) == 2, claves


def test_una_alerta_sin_imagen_no_se_envia(emisor):
    """El bot publica imagenes: sin recorte no hay nada que mandar."""
    n = whatsapp_alertas.reenviar(
        {'alerts': [_visitante(crop_image='', image_base64='')]},
        camera_name='pasillo', camera_id='cam-1')
    assert n == 0 and emisor.envios == []


def test_prefiere_el_recorte_al_frame_completo(emisor):
    whatsapp_alertas.reenviar(
        {'alerts': [_visitante(crop_image='RECORTE',
                               image_base64='FRAME-ENTERO')]},
        camera_name='pasillo', camera_id='cam-1')
    assert emisor.envios[0][0] == 'RECORTE'


def test_metadata_sin_alertas_no_hace_nada(emisor):
    assert whatsapp_alertas.reenviar({'detections': []}) == 0
    assert whatsapp_alertas.reenviar(None) == 0
    assert emisor.envios == []


def test_un_fallo_del_bot_no_tumba_la_inferencia(monkeypatch):
    """Esto corre DENTRO del bucle de inferencia: no puede propagar."""
    class _Explota:
        def enviar_b64(self, *a, **k):
            raise RuntimeError('bot caído')

    monkeypatch.setattr(whatsapp_alertas, '_obtener_emisor', lambda: _Explota())
    assert whatsapp_alertas.reenviar({'alerts': [_visitante()]}) == 0


def test_el_pipeline_de_tienda_publica_alertas():
    """Contrato con `tiendas`: si dejara de publicar `alerts`, la casilla de
    WhatsApp de tienda/amazonas no enviaria nada.

    La emision del evento se comprueba por CONDUCTA (¿existe el metodo en el
    procesador?) y no buscando su `def` en un archivo concreto: el 19-ago-2026
    se movio a `tiendas_fotos.py` (HITO 5) y esta prueba se puso roja sin que
    nada del contrato hubiera cambiado. Lo que importa es que el pipeline
    siga sabiendo emitirlo, no en que archivo vive.
    """
    fuente = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
              / 'tiendas.py').read_text(encoding='utf-8')
    assert "'alerts': self._drenar_alertas()" in fuente

    import sys
    sys.path.insert(0, str(RAIZ / 'server'))
    from src.analityc.core.tiendas import TiendasProcess
    assert callable(getattr(TiendasProcess, '_emitir_visitante', None))
    assert callable(getattr(TiendasProcess, '_drenar_alertas', None))
