"""El lector del websocket nunca espera a la inferencia.

Fija el arreglo del 7-ago-2026 al ciclo de muerte 1011: el bucle del endpoint
esperaba cada resultado antes de volver a leer, la contrapresión llenaba el
buffer del websocket, uvicorn pausaba el TCP y el PONG del keepalive del
cliente quedaba atascado — cierre 1011 «keepalive ping timeout» cada ~3
minutos, reconexión, procesador nuevo y contadores a cero.

Se prueba:
  - que `ColaUltimoFrame` conserva SOLO el frame más reciente de cada cámara
    (un frame viejo sin procesar es un dato caducado, no pendiente);
  - que `DrenajeCliente` procesa, responde y reenvía a WhatsApp por su cuenta;
  - que un fallo de envío detiene el drenador sin ruido (el lector reporta la
    desconexión con su propio criterio);
  - el contrato con el endpoint: ya no existe el `await future` en el bucle.
"""
from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app.ws_drenaje import (ColaUltimoFrame, DrenajeCliente,  # noqa: E402
                                ItemFrame)


def _item(camera_id, marca, **kwargs):
    return ItemFrame(request={'marca': marca}, binario=False,
                     payload=(marca,), camera_id=camera_id, **kwargs)


# ── ColaUltimoFrame ──────────────────────────────────────────────────────
def test_conserva_solo_el_frame_mas_reciente_de_cada_camara():
    async def caso():
        cola = ColaUltimoFrame()
        cola.poner(_item('cam1', 'viejo'))
        cola.poner(_item('cam2', 'unico'))
        cola.poner(_item('cam1', 'nuevo'))       # reemplaza a 'viejo'
        assert len(cola) == 2
        assert cola.descartados == 1
        primero = await cola.sacar()
        segundo = await cola.sacar()
        # cam1 conserva su TURNO original pero con el dato fresco.
        assert (primero.camera_id, primero.request['marca']) == ('cam1', 'nuevo')
        assert (segundo.camera_id, segundo.request['marca']) == ('cam2', 'unico')
    asyncio.run(caso())


def test_sacar_espera_a_que_llegue_algo():
    async def caso():
        cola = ColaUltimoFrame()

        async def productor():
            await asyncio.sleep(0.02)
            cola.poner(_item('cam1', 'tardio'))

        tarea = asyncio.ensure_future(productor())
        item = await asyncio.wait_for(cola.sacar(), timeout=1.0)
        await tarea
        assert item.request['marca'] == 'tardio'
    asyncio.run(caso())


# ── Dobles para el drenador ──────────────────────────────────────────────
class _WebsocketFalso:
    def __init__(self, fallar: bool = False):
        self.enviados = []
        self.fallar = fallar

    async def send_json(self, data):
        if self.fallar:
            raise RuntimeError('socket roto')
        self.enviados.append(('json', data))

    async def send_bytes(self, data):
        if self.fallar:
            raise RuntimeError('socket roto')
        self.enviados.append(('bytes', data))


def _drenador(ws, executor, procesar=None, **kwargs):
    return DrenajeCliente(
        'client_test', ws,
        procesar=procesar or (lambda marca: {'status': 'success',
                                             'marca': marca}),
        executor_getter=lambda: executor,
        candado_envio=asyncio.Lock(),
        **kwargs)


async def _espera(condicion, timeout=2.0):
    limite = asyncio.get_event_loop().time() + timeout
    while not condicion():
        if asyncio.get_event_loop().time() > limite:
            raise AssertionError('condición no alcanzada a tiempo')
        await asyncio.sleep(0.01)


# ── DrenajeCliente ───────────────────────────────────────────────────────
def test_procesa_y_responde_sin_que_nadie_espere():
    async def caso():
        with ThreadPoolExecutor(max_workers=1) as executor:
            ws = _WebsocketFalso()
            d = _drenador(ws, executor)
            d.start()
            d.encolar(_item('cam1', 'f1'))
            await _espera(lambda: d.enviados == 1)
            await d.stop()
        formato, respuesta = ws.enviados[0]
        assert formato == 'json'
        # La respuesta es el request ORIGINAL con `data` = resultado: el
        # contrato de siempre, que el cliente enruta por component_key.
        assert respuesta['marca'] == 'f1'
        assert respuesta['data'] == {'status': 'success', 'marca': 'f1'}
    asyncio.run(caso())


def test_un_fallo_de_envio_detiene_el_drenador():
    async def caso():
        with ThreadPoolExecutor(max_workers=1) as executor:
            ws = _WebsocketFalso(fallar=True)
            d = _drenador(ws, executor)
            d.start()
            d.encolar(_item('cam1', 'f1'))
            await _espera(lambda: d._task.done())
            assert d.enviados == 0
            await d.stop()
    asyncio.run(caso())


def test_un_fallo_de_inferencia_responde_error_y_sigue_vivo():
    async def caso():
        def procesar(marca):
            if marca == 'malo':
                raise ValueError('modelo roto')
            return {'status': 'success', 'marca': marca}

        with ThreadPoolExecutor(max_workers=1) as executor:
            ws = _WebsocketFalso()
            d = _drenador(ws, executor, procesar=procesar)
            d.start()
            d.encolar(_item('cam1', 'malo'))
            d.encolar(_item('cam2', 'bueno'))
            await _espera(lambda: d.enviados == 2)
            await d.stop()
        assert ws.enviados[0][1]['data']['status'] == 'error'
        assert ws.enviados[1][1]['data']['status'] == 'success'
    asyncio.run(caso())


def test_reenvia_whatsapp_solo_cuando_el_frame_lo_pide():
    async def caso():
        llamadas = []

        def reenviar(metadata, camera_name='', camera_id='',
                     establecimiento=''):
            llamadas.append((metadata, camera_name, camera_id,
                             establecimiento))
            return 1

        def procesar(marca):
            # Como el pipeline real: las alertas van dentro de `metadata`.
            return {'status': 'success', 'metadata': {'marca': marca}}

        with ThreadPoolExecutor(max_workers=1) as executor:
            ws = _WebsocketFalso()
            d = _drenador(ws, executor, procesar=procesar,
                          reenviar_whatsapp=reenviar)
            d.start()
            d.encolar(_item('cam1', 'sin_wa'))
            d.encolar(_item('cam2', 'con_wa', enviar_whatsapp=True,
                            camera_name='Puerta', camera_id_raw='7',
                            establecimiento='AVILA'))
            # VigilanteAmazonas y Estacionamiento (VigilanteWS) tienen su
            # ruta propia: reenviarlos aqui duplicaria cada aviso.
            d.encolar(_item('cam3', 'vigilante', enviar_whatsapp=True,
                            type_inference='VigilanteAmazonas'))
            d.encolar(_item('cam4', 'parking', enviar_whatsapp=True,
                            type_inference='Estacionamiento'))
            await _espera(lambda: d.enviados == 4)
            await d.stop()
        assert len(llamadas) == 1
        metadata, nombre, cam, local = llamadas[0]
        assert metadata == {'marca': 'con_wa'}
        assert (nombre, cam) == ('Puerta', '7')
        # El LOCAL de la camara viaja hasta el mensaje (encabeza el texto).
        assert local == 'AVILA'
    asyncio.run(caso())


def test_stop_con_cola_pendiente_no_se_cuelga():
    async def caso():
        with ThreadPoolExecutor(max_workers=1) as executor:
            ws = _WebsocketFalso()
            d = _drenador(ws, executor)
            d.start()
            for i in range(5):
                d.encolar(_item(f'cam{i}', f'f{i}'))
            await asyncio.wait_for(d.stop(), timeout=2.0)
            d.encolar(_item('cam9', 'tarde'))       # tras stop: se ignora
            assert len(d.cola) == 0
    asyncio.run(caso())


# ── Contrato con el endpoint ─────────────────────────────────────────────
def test_el_lector_ya_no_espera_el_resultado():
    """La espera inline era la causa raíz del 1011: no puede volver."""
    import inspect
    from src.app.app import websocket_endpoint
    fuente = inspect.getsource(websocket_endpoint)
    assert 'worker.encolar(ItemFrame(' in fuente
    assert 'await future' not in fuente
    assert 'DrenajeCliente(' in fuente


def test_el_procesador_se_construye_fuera_del_event_loop():
    """Cargar modelos congelaba a TODOS los clientes en cada reconexión."""
    import inspect
    from src.app.app import websocket_endpoint
    fuente = inspect.getsource(websocket_endpoint)
    assert 'run_in_executor(\n                    None, _build_processor' \
        in fuente.replace('  ', '  ')
    assert '_candado_construccion' in fuente


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
if __name__ == '__main__':
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        try:
            _fn()
            print(f"  OK    {_nombre}")
        except Exception as _exc:                          # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
