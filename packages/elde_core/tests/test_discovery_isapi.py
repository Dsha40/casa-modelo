"""Descubrimiento de equipos por ISAPI y filtro anti-eco (8-ago-2026).

Nace de una red real (72.68.60.x) donde un middlebox contesta el SYN de TODOS
los puertos en TODAS las IPs: el barrido TCP devolvía 150 "equipos" idénticos,
ninguno real, y los 3 equipos de verdad (2 DVR + 1 cámara) quedaban enterrados
en puertos no estándar (8080/8081/8082).

Se fija:
  - `_huella_http` reconoce un Hikvision por el reto Digest de ISAPI, con realm
    genérico (DVRNVRDVS), de cámara ("IP Camera(..)") o hasta un hash — el bug
    era exigir que el realm terminara en dígito;
  - el realm distingue cámara de DVR;
  - un `Server` "DNVRS-Webs" es Hikvision, NO Dahua (regresión del heurístico
    viejo "Webs" -> Dahua);
  - `es_equipo_video` da por bueno lo confirmado por ISAPI y RECHAZA el host
    con 8000 y 37777 abiertos a la vez (firma del eco).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.dvr import discovery  # noqa: E402
from elde_core.dvr.discovery import DispositivoRed, _tipo_por_realm  # noqa: E402


class _HTTPError(Exception):
    """Doble de urllib.error.HTTPError con headers y code."""
    def __init__(self, code, headers):
        self.code = code
        self.headers = headers


def _huella_con_realm(realm: str, server: str = "") -> dict:
    """Ejecuta _huella_http simulando el 401 Digest de ISAPI con ese realm."""
    import urllib.error
    hdrs = {"WWW-Authenticate": f'Digest realm="{realm}", qop="auth"',
            "Server": server}
    err = urllib.error.HTTPError("u", 401, "Unauthorized", hdrs, None)
    with mock.patch("urllib.request.urlopen", side_effect=err):
        return discovery._huella_http("10.0.0.9", 8080, 1.0)


# ── La huella ISAPI reconoce los 3 realms reales ─────────────────────────
def test_realm_dvr_generico_se_reconoce():
    # El realm de tu DVR .117; NO termina en dígito (el bug viejo lo perdía).
    h = _huella_con_realm("DVRNVRDVS", server="DNVRS-Webs")
    assert h["marca"] == "Hikvision"
    assert h["isapi"] is True
    assert h["tipo"] == "dvr"


def test_realm_de_camara_se_reconoce_como_camara():
    h = _huella_con_realm("IP Camera(FF877)", server="webserver")
    assert h["marca"] == "Hikvision"
    assert h["tipo"] == "camara"


def test_realm_hash_sigue_siendo_hikvision_confirmado():
    # .118 devuelve un realm hash: no da modelo, pero ISAPI lo confirma.
    h = _huella_con_realm("8b39ba3faff5289b080cc4d7")
    assert h["marca"] == "Hikvision"
    assert h["isapi"] is True
    assert h.get("tipo", "") == ""


def test_realm_con_modelo_ds_extrae_el_modelo():
    h = _huella_con_realm("DS-7208HUHI-K2")
    assert h["modelo"] == "DS-7208HUHI-K2"


def test_tipo_por_realm():
    assert _tipo_por_realm("IP Camera(FF877)") == "camara"
    assert _tipo_por_realm("DVRNVRDVS") == "dvr"
    assert _tipo_por_realm("Network Camera") == "camara"
    assert _tipo_por_realm("hash123") == ""


# ── es_equipo_video: ISAPI vale, doble-SDK es eco ────────────────────────
def test_isapi_confirmado_es_equipo():
    d = DispositivoRed(ip="1.1.1.1", marca="Hikvision", origen={"TCP", "ISAPI"})
    assert d.es_equipo_video is True


def test_host_con_ambos_sdk_es_eco_no_equipo():
    # 8000 (Hik) y 37777 (Dahua) a la vez = firewall contestando, no un equipo.
    d = DispositivoRed(ip="1.1.1.1",
                       puertos_abiertos=[80, 8000, 554, 37777, 8080, 443],
                       origen={"TCP"})
    assert d.es_equipo_video is False


def test_solo_web_no_basta_para_equipo():
    d = DispositivoRed(ip="1.1.1.1", puertos_abiertos=[80, 443], origen={"TCP"})
    assert d.es_equipo_video is False


# ── descripción y marca_ui ───────────────────────────────────────────────
def test_descripcion_incluye_el_tipo():
    d = DispositivoRed(ip="1.1.1.1", marca="Hikvision", tipo="camara")
    assert "Cámara IP" in d.descripcion
    dvr = DispositivoRed(ip="1.1.1.2", marca="Hikvision", tipo="dvr")
    assert "DVR/NVR" in dvr.descripcion


def test_marca_ui_de_hikvision():
    d = DispositivoRed(ip="1.1.1.1", marca="Hikvision", origen={"ISAPI"})
    assert d.marca_ui == "Hikvision HTTP"


# ── Los puertos amplios cubren el 8082 real ──────────────────────────────
def test_los_puertos_no_estandar_de_los_equipos_reales_se_barren():
    # Los 3 equipos estaban en 8080/8081/8082; deben estar en la lista TCP.
    for p in (8080, 8081, 8082):
        assert p in discovery.PUERTOS_SONDA, f"falta el puerto {p} en el barrido"
    # Y en la lista de confirmación ISAPI.
    for p in (8080, 8081, 8082, 82):
        assert p in discovery._PUERTOS_WEB_ISAPI


# ── Parseo ISAPI agnóstico al namespace (regresión 8-ago-2026) ───────────
def test_el_xml_se_parsea_con_cualquier_namespace():
    """El DVR-216Q-F1 real usa el namespace std-cgi.com, no hikvision.com.

    La estrategia fijaba hikvision.com/ver20, así que devolvía 0 canales y un
    modelo vacío CONTRA UN DVR QUE RESPONDÍA PERFECTAMENTE. Se compara por
    nombre local de la etiqueta.
    """
    import xml.etree.ElementTree as ET
    from elde_core.dvr.hikvision_http import _find, _findall_local

    for ns in ("http://www.std-cgi.com/ver20/XMLSchema",     # el DVR real
               "http://www.hikvision.com/ver20/XMLSchema",   # firmware nuevo
               "http://www.hikvision.com/ver10/XMLSchema"):
        info = ET.fromstring(
            f'<DeviceInfo xmlns="{ns}"><model>DVR-216Q-F1</model>'
            f'<serialNumber>ABC123</serialNumber></DeviceInfo>')
        assert _find(info, "model") == "DVR-216Q-F1", f"falla con {ns}"
        assert _find(info, "serialNumber") == "ABC123"

        canales = ET.fromstring(
            f'<VideoInputChannelList xmlns="{ns}">'
            f'<VideoInputChannel><id>1</id><name>Camera 01</name></VideoInputChannel>'
            f'<VideoInputChannel><id>2</id><name>Camera 02</name></VideoInputChannel>'
            f'</VideoInputChannelList>')
        chs = _findall_local(canales, "VideoInputChannel")
        assert len(chs) == 2, f"falla con {ns}"
        assert _find(chs[0], "name") == "Camera 01"


def test_la_url_rtsp_sigue_el_formato_de_hikvision():
    from elde_core.dvr.hikvision_http import HikvisionHTTPStrategy
    e = HikvisionHTTPStrategy("10.0.0.5", 8080, "admin", "clave")
    # Canal 1 principal = 101, canal 1 secundario = 102, canal 12 = 1201.
    assert e.build_rtsp_url(1, False).endswith(":554/Streaming/Channels/101")
    assert e.build_rtsp_url(1, True).endswith(":554/Streaming/Channels/102")
    assert e.build_rtsp_url(12, False).endswith(":554/Streaming/Channels/1201")


if __name__ == '__main__':
    fallos = 0
    for _n, _f in sorted(globals().items()):
        if not _n.startswith('test_') or not callable(_f):
            continue
        try:
            _f()
            print(f"  OK    {_n}")
        except Exception as _e:                            # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_n}: {type(_e).__name__}: {_e}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
