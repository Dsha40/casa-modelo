"""Estrategia DVR para Hik Device Gateway (8-ago-2026).

## Qué es y por qué entra al ecosistema

Hik Device Gateway es un servicio (Windows/Linux) al que los equipos Hikvision
se REGISTRAN por ISUP/EHome: en el DVR se configura Platform Access → Ehome
con la IP y el puerto del gateway, y a partir de ahí el gateway re-sirve los
canales LOCALMENTE — API HTTP con digest y streams RTSP. Como es el equipo el
que inicia la conexión, sirve igual para un DVR en la red local que para uno
remoto detrás de NAT (solo el gateway necesita un puerto alcanzable).

Documentación leída para esto: `extras/hik-connect/Hik DeviceGateway(V1.8...)`
— API Developer Guide V1.8 (flujo 4.1/4.2.1, mensajes JSON_StreamInfo y
JSON_MediaAccessInfo), FAQ y datasheet.

## Qué NO toca

Es ADITIVA. Hik-Connect (nube, hikconnect.py) sigue siendo el camino de los
DVR remotos de Richard, EZVIZ el de las cuentas de consumo, y la captura de
ventana el de la cámara compartida como invitado. Esta estrategia cubre los
equipos que se registren al gateway (los DVR de la red local primero).

## Cómo descubre los canales

La guía no ofrece un listado de canales fiable para equipos ISUP (FAQ: el de
ISUP2.0 "is not accurate"), así que se SONDEA: se pide la URL de stream canal
por canal (`streamMedia`, método `preview`) y el canal existe si el gateway la
devuelve. El sondeo se corta tras `_MAX_FALLOS_SEGUIDOS` fallos consecutivos:
los canales reales son contiguos y esto evita 16 timeouts en un equipo de 4.

Cada acierto rellena `rtsp_main`/`rtsp_sub` con la URL lista para el
RTSPWorker del recuadro (credenciales embebidas: el RTSP del gateway exige el
mismo digest que el HTTP).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPDigestAuth

from .base import ChannelInfo, DeviceInfo, DVRStrategy

logger = logging.getLogger(__name__)

#: Canales 1..N que se sondean por equipo registrado.
_MAX_CANALES = 16
#: Fallos consecutivos que dan por terminado el sondeo de un equipo.
_MAX_FALLOS_SEGUIDOS = 2


class DeviceGatewayStrategy(DVRStrategy):
    """Hik Device Gateway: equipos registrados por ISUP, re-servidos en RTSP.

    `host`/`port` son los del GATEWAY (no los de ningún DVR), y `username`/
    `password` los de su usuario admin — el mismo con el que se entra a su
    web. El gateway autentica todo (HTTP y RTSP) con digest.
    """

    # ── HTTP ────────────────────────────────────────────────────────────
    def _http(self, method: str, path: str,
              json_body: Optional[dict] = None) -> Optional[dict]:
        """Petición con digest que devuelve el JSON o None (nunca lanza,
        salvo credenciales malas: eso SÍ debe llegar al diálogo)."""
        url = f"{self.base_http_url}{path}"
        try:
            r = requests.request(
                method, url, json=json_body,
                auth=HTTPDigestAuth(self.username, self.password),
                timeout=self.timeout)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("DeviceGateway %s %s: %s", method, path, exc)
            return None
        if r.status_code == 401:
            raise PermissionError(
                "Credenciales del Device Gateway incorrectas (HTTP 401).")
        if r.status_code >= 400:
            logger.debug("DeviceGateway %s %s -> HTTP %s", method, path,
                         r.status_code)
            return None
        try:
            return r.json()
        except ValueError:
            return None

    # ── Piezas ──────────────────────────────────────────────────────────
    def _listar_equipos(self) -> List[Dict[str, Any]]:
        """Equipos registrados en el gateway, tolerante con el envoltorio.

        La guía (5.1.4) define POST deviceList con un SearchDescription
        paginado; el nombre exacto del nodo de la lista ha cambiado entre
        versiones, así que se busca la PRIMERA lista de dicts con pinta de
        equipo (devIndex presente) en cualquier nivel razonable.
        """
        cuerpo = {"SearchDescription": {"position": 0, "maxResult": 100,
                                        "Filter": {"key": ""}}}
        datos = (self._http("POST",
                            "/ISAPI/ContentMgmt/DeviceMgmt/deviceList?format=json",
                            cuerpo)
                 or self._http("POST",
                               "/ISAPI/ContentMgmt/DeviceMgmt/deviceList?format=json",
                               {}))
        if not isinstance(datos, dict):
            return []
        return _extraer_equipos(datos)

    def _url_stream(self, dev_index: str, canal: int,
                    sub_stream: bool) -> str:
        """Pide la URL de vivo de UN canal. Cadena vacía si no existe."""
        cuerpo = {"StreamInfo": {
            "id": str(canal),
            "streamType": "sub" if sub_stream else "main",
            "method": "preview",
        }}
        datos = self._http(
            "POST",
            f"/ISAPI/System/streamMedia?format=json&devIndex={dev_index}",
            cuerpo)
        if not isinstance(datos, dict):
            return ""
        url = ((datos.get("MediaAccessInfo") or {}).get("URL")
               or (datos.get("data") or {}).get("url") or "")
        return self._completar_rtsp(str(url)) if url else ""

    def _completar_rtsp(self, url: str) -> str:
        """URL absoluta con credenciales: lista para ffmpeg/RTSPWorker.

        El gateway puede devolver la URL completa (`rtsp://ip:puerto/...`) o
        solo la ruta; y su RTSP exige el mismo digest que el HTTP, así que
        las credenciales van embebidas (ffmpeg las negocia solo).
        """
        if not url.startswith("rtsp://"):
            url = f"rtsp://{self.host}:554{url if url.startswith('/') else '/' + url}"
        if "@" not in url.split("//", 1)[1].split("/", 1)[0]:
            cred = f"{self.username}:{self.password}@"
            url = url.replace("rtsp://", f"rtsp://{cred}", 1)
        return url

    # ── Contrato DVRStrategy ────────────────────────────────────────────
    def get_device_info(self) -> DeviceInfo:
        # Valida credenciales y da identidad al gateway. Si esto devuelve
        # None sin haber lanzado PermissionError, el gateway no responde.
        propio = self._http("GET", "/ISAPI/System/deviceInfo?format=json")
        if propio is None:
            raise ConnectionError(
                f"Hik Device Gateway no responde en {self.base_http_url}. "
                f"¿Servicio DeviceGatewayService corriendo y puerto correcto?")

        equipos = self._listar_equipos()
        info = DeviceInfo(brand="Device Gateway", ip_address=self.host)
        nodo = (propio.get("DeviceInfo") or propio.get("deviceInfo") or {})
        info.device_name = str(nodo.get("deviceName") or "Hik Device Gateway")
        info.firmware_version = str(nodo.get("firmwareVersion") or "")
        info.model = str(nodo.get("model") or "Device Gateway")
        info.extra["total_devices"] = len(equipos)
        info.extra["devices"] = []

        canales: List[ChannelInfo] = []
        for eq in equipos:
            dev_index = str(eq.get("devIndex") or "")
            if not dev_index:
                continue
            alias = str(eq.get("devName") or eq.get("deviceName")
                        or eq.get("deviceID") or dev_index[:8])
            serial = str(eq.get("devSerial") or eq.get("deviceSerial")
                         or eq.get("deviceID") or dev_index)
            estado = str(eq.get("devStatus") or eq.get("status") or "")
            en_linea = estado.lower() in ("online", "1", "true", "")
            info.extra["devices"].append(
                f"{alias} ({serial}) estado={estado or '?'}")
            if not en_linea:
                # Un equipo apagado no se sondea: serían N timeouts para
                # confirmar lo que el gateway ya dijo.
                continue

            fallos_seguidos = 0
            for ch in range(1, _MAX_CANALES + 1):
                principal = self._url_stream(dev_index, ch, sub_stream=False)
                if not principal:
                    fallos_seguidos += 1
                    if fallos_seguidos >= _MAX_FALLOS_SEGUIDOS:
                        break
                    continue
                fallos_seguidos = 0
                canales.append(ChannelInfo(
                    id=str(ch),
                    name=f"{alias} · Canal {ch}",
                    rtsp_main=principal,
                    rtsp_sub=self._url_stream(dev_index, ch, sub_stream=True),
                    extra={
                        # Lo que start_dvr_stream y la identidad H-11 leen.
                        "device_serial": serial,
                        "device_alias": alias,
                        "channel_id": f"{serial}-{ch}",
                        "channel_no": str(ch),
                        "dev_index": dev_index,
                        "gateway": self.base_http_url,
                    }))

        info.channels = canales
        info.num_video_channels = len(canales)
        # Para build_rtsp_url sin repetir el listado.
        self._primer_dev_index = next(
            (str(e.get("devIndex")) for e in equipos if e.get("devIndex")), "")
        return info

    def build_rtsp_url(self, channel: int, sub_stream: bool = False) -> str:
        dev_index = getattr(self, "_primer_dev_index", "")
        if not dev_index:
            equipos = self._listar_equipos()
            dev_index = next(
                (str(e.get("devIndex")) for e in equipos if e.get("devIndex")),
                "")
            self._primer_dev_index = dev_index
        if not dev_index:
            return ""
        return self._url_stream(dev_index, channel, sub_stream)


def _extraer_equipos(datos: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Encuentra la lista de equipos venga como venga envuelta.

    Se aceptan los envoltorios vistos en la guía y variantes defensivas:
    `SearchResult.MatchList[].Device`, `MatchList[]`, `DeviceList[]`,
    `data.device[]`. Un dict "parece equipo" si trae devIndex.
    """
    def _parece_equipo(d: Any) -> bool:
        return isinstance(d, dict) and bool(d.get("devIndex"))

    def _desenvolver(item: Any) -> Any:
        if isinstance(item, dict) and "Device" in item:
            return item["Device"]
        return item

    pila: List[Any] = [datos]
    while pila:
        nodo = pila.pop()
        if isinstance(nodo, list):
            candidatos = [_desenvolver(x) for x in nodo]
            equipos = [c for c in candidatos if _parece_equipo(c)]
            if equipos:
                return equipos
            pila.extend(candidatos)
        elif isinstance(nodo, dict):
            pila.extend(nodo.values())
    return []
