"""Estrategia Device Gateway: sondeo de canales, URLs y tolerancia.

Fija el contrato del 8-ago-2026: el gateway re-sirve por RTSP los equipos que
se le registran por ISUP, y la estrategia debe:

  - listar equipos aunque el envoltorio JSON cambie de nombre (la guía cambió
    entre versiones y el cargador no puede romperse por eso);
  - sondear canales y CORTAR tras 2 fallos seguidos (no 16 timeouts en un
    DVR de 4 canales);
  - no sondear equipos que el gateway ya declara offline;
  - devolver URLs rtsp:// absolutas y con credenciales embebidas (el RTSP
    del gateway exige el mismo digest que el HTTP);
  - ser ADITIVA: Hik-Connect y EZVIZ siguen registradas tal cual.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.dvr.device_gateway import (DeviceGatewayStrategy,  # noqa: E402
                                          _extraer_equipos)


def _estrategia() -> DeviceGatewayStrategy:
    return DeviceGatewayStrategy('192.168.1.50', 81, 'admin', 'clave123')


def _respuesta(status: int = 200, json_data=None):
    r = mock.Mock()
    r.status_code = status
    if json_data is None:
        r.json.side_effect = ValueError('sin json')
    else:
        r.json.return_value = json_data
    return r


# ── Extraer equipos de cualquier envoltorio ──────────────────────────────
def test_extrae_equipos_del_envoltorio_de_la_guia():
    datos = {'SearchResult': {'numOfMatches': 1, 'MatchList': [
        {'Device': {'devIndex': 'abc-123', 'devName': 'DVR Bodega',
                    'devStatus': 'online'}}]}}
    equipos = _extraer_equipos(datos)
    assert len(equipos) == 1
    assert equipos[0]['devIndex'] == 'abc-123'


def test_extrae_equipos_sin_envoltorio_device():
    datos = {'DeviceList': [{'devIndex': 'x1'}, {'devIndex': 'x2'}]}
    assert [e['devIndex'] for e in _extraer_equipos(datos)] == ['x1', 'x2']


def test_sin_equipos_devuelve_lista_vacia():
    assert _extraer_equipos({'ResponseStatus': {'statusCode': 1}}) == []


# ── URLs ─────────────────────────────────────────────────────────────────
def test_url_relativa_se_completa_con_gateway_y_credenciales():
    e = _estrategia()
    url = e._completar_rtsp('/media/live/ch1')
    assert url == 'rtsp://admin:clave123@192.168.1.50:554/media/live/ch1'


def test_url_absoluta_solo_recibe_credenciales():
    e = _estrategia()
    url = e._completar_rtsp('rtsp://192.168.1.50:8554/live/1')
    assert url == 'rtsp://admin:clave123@192.168.1.50:8554/live/1'


# ── get_device_info: sondeo y estados ────────────────────────────────────
def _simular(e: DeviceGatewayStrategy, equipos, canales_con_stream):
    """Parcha _http: deviceInfo OK, deviceList con `equipos`, y streamMedia
    responde URL solo para los canales en `canales_con_stream`."""
    def _http(method, path, json_body=None):
        if 'System/deviceInfo' in path and 'devIndex' not in path:
            return {'DeviceInfo': {'deviceName': 'GW', 'model': 'DG',
                                   'firmwareVersion': 'V1.8.0'}}
        if 'deviceList' in path:
            return {'SearchResult': {'MatchList': [
                {'Device': dict(eq)} for eq in equipos]}}
        if 'streamMedia' in path:
            canal = int(json_body['StreamInfo']['id'])
            dev = path.split('devIndex=')[1]
            if (dev, canal) in canales_con_stream:
                return {'MediaAccessInfo': {'URL': f'/live/{dev}/{canal}'}}
            return None
        return None
    return mock.patch.object(e, '_http', side_effect=_http)


def test_sondea_canales_y_corta_tras_dos_fallos_seguidos():
    e = _estrategia()
    con_stream = {('d1', 1), ('d1', 2), ('d1', 3), ('d1', 4)}

    with _simular(e, [{'devIndex': 'd1', 'devName': 'DVR Local',
                       'devSerial': 'S1', 'devStatus': 'online'}],
                  con_stream):
        info = e.get_device_info()

    assert info.num_video_channels == 4
    assert [c.id for c in info.channels] == ['1', '2', '3', '4']
    # El canal trae lo que el recuadro necesita para el drop y la identidad.
    extra = info.channels[0].extra
    assert extra['device_serial'] == 'S1'
    assert extra['device_alias'] == 'DVR Local'
    assert extra['channel_id'] == 'S1-1'
    assert info.channels[0].rtsp_main.startswith('rtsp://admin:clave123@')


def test_no_sondea_equipos_offline():
    e = _estrategia()
    with _simular(e, [{'devIndex': 'd2', 'devName': 'Apagado',
                       'devStatus': 'offline'}], set()) as parche:
        info = e.get_device_info()
    assert info.num_video_channels == 0
    rutas = [c.args[1] for c in parche.call_args_list]
    assert not any('streamMedia' in r for r in rutas), (
        'un equipo offline no debe generar sondeos')
    # Pero el equipo SÍ queda listado para que el operador lo vea.
    assert any('Apagado' in d for d in info.extra['devices'])


def test_gateway_caido_lanza_connection_error():
    e = _estrategia()
    with mock.patch.object(e, '_http', return_value=None):
        try:
            e.get_device_info()
        except ConnectionError as exc:
            assert 'no responde' in str(exc)
        else:
            raise AssertionError('debió lanzar ConnectionError')


def test_credenciales_malas_lanzan_permission_error():
    e = _estrategia()
    with mock.patch('elde_core.dvr.device_gateway.requests.request',
                    return_value=_respuesta(401)):
        try:
            e._http('GET', '/ISAPI/System/deviceInfo?format=json')
        except PermissionError:
            pass
        else:
            raise AssertionError('debió lanzar PermissionError')


# ── Aditiva: no desplaza a nadie ─────────────────────────────────────────
def test_el_contexto_registra_la_marca_sin_quitar_las_demas():
    from elde_core.dvr.context import DVRContext
    marcas = DVRContext.brands()
    assert 'Device Gateway' in marcas
    assert 'Hik-Connect' in marcas          # la nube de los DVR remotos sigue
    assert 'EZVIZ' in marcas                # la via de consumo sigue
    assert DVRContext.default_port('Device Gateway') == 80


# ── Ejecutable sin pytest ────────────────────────────────────────────────
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
