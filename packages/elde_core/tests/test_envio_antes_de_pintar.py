"""
Tarea 2: la captura que llega con una imagen en vuelo se GUARDA y se envía al
volver la respuesta, antes de pintarla (5-oct-2026).

Con el servidor en modo G (57 ms por imagen), el ciclo del cliente pasaba
~20 ms esperando la captura SIGUIENTE (las que llegaban mientras tanto se
tiraban) y ~17 ms pintando antes de liberar el envío. Lo que se fija aquí,
en la parte compartida (`render_box_captura` y `telemetria_ciclo`):

- con una en vuelo, la última captura se guarda (no se tira);
- `enviar_captura_guardada` la manda y deja UNA en vuelo; sin guardada, sin
  IA o con una ya en vuelo no manda nada;
- `olvidar_captura_guardada` la descarta (parar, apagar IA, desconexión);
- la telemetría «aparta» el fotograma respondido para marcar su pintado
  aunque el siguiente ya esté en vuelo, sin escribirlo como sin_respuesta.

La parte del recuadro del cliente (orden enviar -> pintar, sobre la imagen
correcta) se prueba en clients/tienda/test/test_envio_antes_de_pintar.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elde_core.ui import telemetria_ciclo as tc  # noqa: E402
from test_render_box_captura import (  # noqa: E402
    EtiquetaConZonas, ProcesoFalso, RecuadroFalso)


def _con_captura(imagen: bytes):
    r = RecuadroFalso(EtiquetaConZonas())
    r.process = ProcesoFalso(msgpack.packb(
        {'header': {'x': 1}, 'image_bytes': imagen}, use_bin_type=True))
    return r


def test_con_una_en_vuelo_la_ultima_se_guarda_y_no_se_envia():
    r = _con_captura(b'\xff\xd8B')
    r.can_send_next_frame = False
    r.loop_show_result()
    assert not r.socket.enviado
    assert r._captura_guardada is not None
    assert r._captura_guardada[1] == b'\xff\xd8B'


def test_enviar_la_guardada_deja_una_en_vuelo():
    r = _con_captura(b'\xff\xd8B')
    r.can_send_next_frame = False
    r.loop_show_result()
    r.can_send_next_frame = True              # llegó la respuesta
    assert r.enviar_captura_guardada() is True
    assert len(r.socket.enviado) == 1
    assert r._pending_frame_bytes == b'\xff\xd8B'
    assert r.can_send_next_frame is False
    assert r._captura_guardada is None
    assert r.enviar_captura_guardada() is False     # no hay otra


def test_no_envia_la_guardada_con_una_en_vuelo_ni_sin_ia():
    r = _con_captura(b'\xff\xd8B')
    r.can_send_next_frame = False
    r.loop_show_result()
    assert r.enviar_captura_guardada() is False     # sigue una en vuelo
    r.can_send_next_frame = True
    r.smart_mode = False
    assert r.enviar_captura_guardada() is False     # IA apagada
    assert not r.socket.enviado


def test_olvidar_la_guardada():
    r = _con_captura(b'\xff\xd8B')
    r.can_send_next_frame = False
    r.loop_show_result()
    r.olvidar_captura_guardada()
    r.can_send_next_frame = True
    assert r.enviar_captura_guardada() is False


def test_con_el_envio_libre_va_la_mas_reciente_y_se_olvida_la_guardada():
    r = _con_captura(b'\xff\xd8B')
    r.can_send_next_frame = False
    r.loop_show_result()                      # B guardada
    r.process = ProcesoFalso(msgpack.packb(
        {'header': {'x': 2}, 'image_bytes': b'\xff\xd8C'}, use_bin_type=True))
    r.can_send_next_frame = True
    r.loop_show_result()                      # C llega con el envío libre
    assert r.socket.enviado and r._pending_frame_bytes == b'\xff\xd8C'
    assert r._captura_guardada is None


def test_la_telemetria_aparta_el_respondido_sin_perderlo(tmp_path):
    e = tc.EscritorJsonl(tmp_path / 'tel.jsonl', vaciar_cada_s=0)
    t = tc.TelemetriaCiclo(activa=True, salida=e)
    t.abrir('cam', 'ventana')                 # A en vuelo
    a = t.apartar()                           # llega su respuesta
    assert a is not None and not t.en_vuelo()
    t.abrir('cam', 'ventana')                 # B sale ANTES de pintar A
    a.marcar('t_pintado')
    a.cerrar('ok')
    t.cerrar('ok')
    filas = [json.loads(l) for l in e.ruta.read_text('utf-8').splitlines()]
    assert [f['estado'] for f in filas] == ['ok', 'ok']   # nada sin_respuesta
    assert [f['seq'] for f in filas] == [1, 2]
    assert 't_pintado' in filas[0]
    assert t.apartar() is None                # sin nada en vuelo
