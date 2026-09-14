"""
src/app/whatsapp_alertas.py — Reenvio de alertas a WhatsApp para CUALQUIER
pipeline, no solo VigilanteAmazonas.

## El problema que resuelve

El interruptor «Enviar por WhatsApp» existia unicamente en `perimetrales-view`,
y el servidor solo lo consumia en `VigilanteWS`: `app.py` fijaba
`processor._enviar_whatsapp` y su propio comentario reconocia que era
«inofensivo para el resto de procesadores» — es decir, en los demas pipelines
el flag no hacia nada.

Al llevar el interruptor a los otros clientes, un boton que no envia nada seria
peor que no tenerlo. Este modulo cierra ese hueco: reenvia las alertas de
cualquier procesador que las publique en `metadata["alerts"]`.

## Que pipelines se benefician

| Pipeline | Publica `alerts` | Resultado |
|---|---|---|
| `VigilanteAmazonas` | si (via su adaptador) | ya funcionaba; se respeta su ruta propia |
| `Hummus`, `Misters` | si | funciona por aqui |
| `Perimetrales*` (base_perimeter) | si | funciona por aqui |
| `Personal de Amazonas` (tienda) | si (`_emitir_visitante`) | funciona por aqui |
| `Estacionamiento` | si (hereda de VigilanteWS) | ruta propia de VigilanteWS |

El pipeline de tienda no publicaba ninguna alerta cuando se escribio este
modulo: su metadata solo traia `detections`, `demographics`, `heatmap` y
`analytics_report`. El evento que faltaba definir —cual merece un WhatsApp— se
resolvio con `_emitir_visitante`: se avisa cuando el genero y la edad de una
persona CONVERGEN, que ocurre una sola vez por visitante. Ver
`test_whatsapp_otros_pipelines.py`, que fija ese contrato.

## Diseno

- **No decide el "si"**: solo actua cuando el cliente activo su interruptor.
- **No bloquea**: `EmisorWhatsApp` publica en un hilo demonio y trae anti-flood
  por clave, asi que no hay que anadir throttling aqui.
- **Nunca lanza**: un fallo de notificacion jamas puede tumbar la inferencia.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_emisor: Optional[Any] = None
_intentado = False


def _obtener_emisor():
    """Carga perezosa del emisor. Si VIGILANTE no esta disponible, se anota una
    vez y el reenvio queda inactivo sin molestar mas."""
    global _emisor, _intentado
    if _intentado:
        return _emisor
    _intentado = True
    try:
        from vigilante_amazonas.servicios.emisor_whatsapp import EmisorWhatsApp
        _emisor = EmisorWhatsApp()
    except Exception as exc:
        logger.info("reenvio a WhatsApp no disponible (%s)", exc)
        _emisor = None
    return _emisor


# Eventos que NO son una alarma. Un visitante de tienda con un 🚨 delante
# confunde: quien lee el grupo debe distinguir de un vistazo una intrusion de
# una metrica de negocio.
#
# 'retiro_de_comida' y 'compra' cubren los eventos del comedor Hummus
# (Retiro_De_Comida / Compra / Compra_Probable / Compra_Sin_Retiro). OJO: NO
# poner 'retiro' a secas — casaria con 'retiro_sin_pago', que es LA alarma de
# negocio del comedor y debe conservar su sirena.
_EVENTOS_INFORMATIVOS = ('visitante', 'captura', 'pedido', 'entrega',
                         'lavado', 'aforo', 'retiro_de_comida', 'compra')


def _hora_legible(ts: Any) -> str:
    """'1786984080.55' -> '17:40:25' (hora local). Un texto ya formateado
    pasa tal cual ('2026-08-05 14:42:00' se queda con la hora); vacio -> ''.

    Copia local del `hora_legible` de vigilante a proposito: este modulo no
    importa `vigilante_amazonas` en el nivel del modulo (si el paquete falta,
    el reenvio queda inactivo sin tumbar `app.py`)."""
    crudo = str(ts or '').strip()
    if not crudo:
        return ''
    # Sello de captura '20260817_144210' ANTES del float(): PEP 515 acepta el
    # guion bajo entre digitos y lo convertiria en un epoch absurdo.
    if re.fullmatch(r'\d{8}_\d{6}', crudo):
        return f'{crudo[9:11]}:{crudo[11:13]}:{crudo[13:15]}'
    try:
        return time.strftime('%H:%M:%S', time.localtime(float(crudo)))
    except (ValueError, OverflowError, OSError):
        pass
    if re.fullmatch(r'.+\s\d{2}:\d{2}:\d{2}', crudo):
        return crudo[-8:]
    return crudo


def _texto(alerta: Dict[str, Any], camara: str, local: str = '') -> str:
    """Titular del mensaje. Formato pedido por el operador (31-ago-2026),
    el mismo del vigilante para que el grupo siga homogeneo — el encabezado
    lleva SOLO el local (la clase ya la dice la descripcion; repetirla era
    redundante):

        🚨 AVILA                   <- LOCAL de la camara
        MERODEO: CARRO entró 3 veces al área en 1 min (similitud 0.98)
        🕒 17:40:25                <- hora legible, no el epoch
    """
    clase = str(alerta.get('class_name') or alerta.get('event_type')
                or 'deteccion')
    desc = str(alerta.get('description') or '').strip()
    if alerta.get('texto_plano') and desc:
        # La alerta trae su mensaje YA redactado (permanencia, 10-ago-2026):
        # va tal cual, sin icono, sin tipo en mayusculas y sin «cámara
        # {titulo de la ventana}» — ese titulo es el de iVMS y no dice nada.
        return desc
    evento = str(alerta.get('event_type') or '').strip().lower()
    icono = '👤' if any(e in evento for e in _EVENTOS_INFORMATIVOS) else '🚨'
    # El LOCAL de la camara encabeza; el nombre de la camara (titulo de la
    # ventana de iVMS) es solo el respaldo cuando no hay local elegido.
    encabezado = (str(alerta.get('establecimiento') or '').strip()
                  or (local or '').strip())
    # Solo el LOCAL encabeza (31-ago-2026): la clase ya viaja en la
    # descripcion o en el tipo de evento. Sin local, la camara; y si
    # tampoco hay camara, la clase — el encabezado nunca queda vacio.
    if encabezado:
        titulo = f"{icono} {encabezado}"
    elif camara:
        titulo = f"{icono} cámara {camara}"
    else:
        titulo = f"{icono} {clase.upper()}"
    if desc:
        titulo += f"\n{desc}"
    else:
        # Sin descripcion, el QUE paso es la clase o el tipo de evento: sin
        # esta linea el mensaje seria solo local + hora, sin decir que ocurrio.
        titulo += f"\n{clase.upper()}"
    hora = _hora_legible(alerta.get('timestamp'))
    if hora:
        titulo += f"\n🕒 {hora}"
    return titulo


def reenviar(metadata: Any, camera_name: str = '', camera_id: str = '',
             establecimiento: str = '') -> int:
    """Reenvia a WhatsApp las alertas que traiga `metadata`.

    `establecimiento` es el LOCAL de la camara (el select del recuadro, que
    viaja en el payload del frame): encabeza el mensaje en lugar del titulo
    de la ventana de iVMS.

    Devuelve cuantas se encolaron. Nunca lanza: si algo falla, se registra y se
    sigue, porque esto corre dentro del bucle de inferencia.
    """
    try:
        if not isinstance(metadata, dict):
            return 0
        alertas = metadata.get('alerts') or []
        if not alertas:
            return 0
        emisor = _obtener_emisor()
        if emisor is None:
            return 0

        enviadas = 0
        for alerta in alertas:
            if not isinstance(alerta, dict):
                continue
            # El recorte de la deteccion es mas util que el frame completo;
            # si no viene, se usa la imagen de la alerta.
            imagen = (alerta.get('crop_image')
                      or alerta.get('image_base64') or '')
            if not imagen:
                continue
            # La clave alimenta el anti-flood del emisor: misma camara + mismo
            # tipo de evento no se repite dentro del periodo de enfriamiento.
            #
            # `dedup_clave` deja que el pipeline decida cual es su unidad de
            # repeticion. PermanenciaArea la manda con la PERSONA dentro: dos
            # personas entrando seguidas son dos avisos, no uno, que es lo que
            # se pidio. Sin ese escape, la segunda caia en el enfriamiento de
            # la primera por compartir camara y tipo de evento.
            clave = (str(alerta.get('dedup_clave') or '').strip()
                     or f"{camera_id or camera_name}|"
                        f"{alerta.get('event_type') or alerta.get('class_name') or ''}")
            try:
                if emisor.enviar_b64(imagen,
                                     _texto(alerta, camera_name,
                                            establecimiento),
                                     clave=clave):
                    enviadas += 1
            except Exception as exc:
                logger.debug("no se pudo reenviar una alerta: %s", exc)
        return enviadas
    except Exception as exc:
        logger.debug("reenvio a WhatsApp fallo: %s", exc)
        return 0
