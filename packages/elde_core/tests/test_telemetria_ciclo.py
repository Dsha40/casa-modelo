"""
Pruebas de la telemetria del ciclo (`elde_core.ui.telemetria_ciclo`).

Se escribio para medir, desde el cliente, en que se van los ~99 ms de
«espera» que el servidor vio el 24-sep-2026 en cada ciclo de 156 ms. Lo que
fijan estas pruebas es lo que haria mentir al fichero: que se pierdan lineas
al cerrar, que un fotograma sin respuesta desaparezca sin dejar rastro, o que
el recuadro con la telemetria APAGADA pague algo o escriba algo.

Corre sin pantalla y sin servidor.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elde_core.ui import telemetria_ciclo as tc  # noqa: E402
from test_render_box_captura import (  # noqa: E402
    EtiquetaConZonas, ProcesoFalso, RecuadroFalso)


def _lineas(ruta: Path):
    return [json.loads(l) for l in ruta.read_text(encoding='utf-8').splitlines()]


def _escritor(tmp_path, **kw):
    return tc.EscritorJsonl(tmp_path / 'tel.jsonl', **kw)


# ── El escritor ──────────────────────────────────────────────────────────

def test_amortigua_y_vacia_al_pedirlo(tmp_path):
    e = _escritor(tmp_path, vaciar_cada_s=3600)
    e.escribir({'a': 1})
    assert not e.ruta.exists(), 'con intervalo largo no debe tocar disco aun'
    e.vaciar()
    assert _lineas(e.ruta) == [{'a': 1}]


def test_vacia_solo_pasado_el_intervalo(tmp_path):
    """Lo que salva el fichero si el operador cierra el cliente de golpe."""
    e = _escritor(tmp_path, vaciar_cada_s=0)
    e.escribir({'a': 1})
    e.escribir({'a': 2})
    assert [r['a'] for r in _lineas(e.ruta)] == [1, 2]


def test_rota_a_una_copia_al_pasar_el_tope(tmp_path):
    e = _escritor(tmp_path, vaciar_cada_s=0, max_bytes=50)
    for i in range(10):
        e.escribir({'i': i, 'relleno': 'x' * 20})
    copia = e.ruta.with_name(e.ruta.name + '.1')
    assert copia.exists()
    # Nunca se pierde lo ultimo: esta en el fichero vivo.
    assert _lineas(e.ruta)[-1]['i'] == 9


def test_la_ruta_por_defecto_va_a_localappdata(monkeypatch, tmp_path):
    monkeypatch.delenv('ELDE_TELEMETRIA_CICLO', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    assert tc.ruta_por_defecto() == tmp_path / 'tienda_view' / 'telemetria_ciclo.jsonl'
    monkeypatch.setenv('ELDE_TELEMETRIA_CICLO', str(tmp_path / 'otra.jsonl'))
    assert tc.ruta_por_defecto() == tmp_path / 'otra.jsonl'


def test_la_cabecera_del_worker_se_lee_en_el_mismo_reloj():
    ahora = datetime.now()
    assert abs(tc.epoch_de_cabecera({'timestamp': ahora.isoformat()})
               - ahora.timestamp()) < 1e-3
    assert tc.epoch_de_cabecera({'x': 1}) is None
    assert tc.epoch_de_cabecera(None) is None


# ── Un recuadro ──────────────────────────────────────────────────────────

def test_apagada_no_escribe_nada(tmp_path):
    e = _escritor(tmp_path, vaciar_cada_s=0)
    t = tc.TelemetriaCiclo(activa=False, salida=e)
    t.abrir('cam', 'ventana', t_recibido=1.0)
    t.marcar('t_envio')
    t.cerrar('ok')
    e.vaciar()
    assert not e.ruta.exists()


def test_un_fotograma_sin_respuesta_queda_escrito(tmp_path):
    """Si la red de seguridad libera el envio, el fotograma perdido no puede
    desaparecer: un hueco sin explicar falsearia el desglose."""
    e = _escritor(tmp_path, vaciar_cada_s=0)
    t = tc.TelemetriaCiclo(activa=True, salida=e)
    t.abrir('cam', 'ventana', t_recibido=1.0)
    t.abrir('cam', 'ventana', t_recibido=2.0)
    t.cerrar('ok')
    filas = _lineas(e.ruta)
    assert [f['estado'] for f in filas] == ['sin_respuesta', 'ok']
    assert [f['seq'] for f in filas] == [1, 2]


# ── Enganchada a la captura por ventana (la de iVMS) ─────────────────────

def _recuadro_con_telemetria(tmp_path, cabecera):
    r = RecuadroFalso(EtiquetaConZonas())
    r.process = ProcesoFalso(msgpack.packb(
        {'header': cabecera, 'image_bytes': b'\xff\xd8jpeg'}, use_bin_type=True))
    e = _escritor(tmp_path, vaciar_cada_s=0)
    r._telemetria = tc.TelemetriaCiclo(activa=True, salida=e)
    return r, e


def test_la_ruta_de_ventana_marca_captura_recibido_y_envio(tmp_path):
    ahora = datetime.now()
    r, e = _recuadro_con_telemetria(tmp_path, {'timestamp': ahora.isoformat()})
    r.loop_show_result()
    assert r.socket.enviado, 'la telemetria no puede impedir el envio'
    t = r._telemetria
    assert t.en_vuelo()
    t.cerrar('ok')
    (fila,) = _lineas(e.ruta)
    assert fila['ruta'] == 'ventana'
    assert fila['camera_id'] == 'dvr-J12345678-2'
    assert fila['bytes'] == len(b'\xff\xd8jpeg')
    assert fila['t_captura'] <= fila['t_recibido'] <= fila['t_envio']


def test_cuenta_las_imagenes_que_llegan_con_un_fotograma_en_vuelo(tmp_path):
    """Son la prueba de que la captura ya trabaja en paralelo al servidor.

    Desde el 5-oct-2026 (tarea 2) la ULTIMA que llega con una en vuelo se
    guarda para enviarla al volver la respuesta: de dos, una se guarda y una
    se tira. Al enviar la siguiente, la guardada (reemplazada) cuenta."""
    r, e = _recuadro_con_telemetria(tmp_path, {'x': 1})
    r.can_send_next_frame = False
    r.loop_show_result()
    r.loop_show_result()
    assert r._telemetria.descartados == 1
    assert r._captura_guardada is not None
    r.can_send_next_frame = True
    r.loop_show_result()
    r._telemetria.cerrar('ok')
    (fila,) = _lineas(e.ruta)
    assert fila['descartados'] == 2
    assert 't_captura' not in fila, 'sin timestamp en la cabecera no se inventa'


def test_sin_atributo_de_telemetria_el_mixin_sigue_igual():
    """Los otros clientes del nucleo no la tienen: no puede romperles nada."""
    r = RecuadroFalso(EtiquetaConZonas())
    assert not hasattr(r, '_telemetria')
    r.loop_show_result()
    assert r.socket.enviado
