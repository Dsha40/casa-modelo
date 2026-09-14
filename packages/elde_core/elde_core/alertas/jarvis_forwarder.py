"""
Reenviador de alertas hacia la API de Jarvis365 — COMPARTIDO por los cuatro
clientes (5-ago-2026).

Recibe las MISMAS alertas que se muestran en el panel lateral (las tarjetas que
emite cada `render_box` por `alert_received`) y las reenvia a Jarvis como
"novedades" (POST /novelties, con la imagen subida a /multimedia).

## Por que vive aqui

Hasta hoy este reenviador existia SOLO en `clients/perimetrales`. Los otros tres
clientes construian `Jarvis_api` (login, lista de establecimientos, selector del
pie) pero **jamas enviaban una novedad**: la sesion se abria, el selector se
poblaba, y ahi moria. Tener media integracion cableada es peor que no tenerla,
porque desde la UI parece que funciona.

Al subirlo al nucleo, los cuatro clientes comparten una sola implementacion y
las diferencias reales quedan como PARAMETROS, no como copias divergentes:

| Cliente | `clases_permitidas` | `local_global` |
|---|---|---|
| perimetrales | las 6 perimetrales | `False` — cada camara tiene su select "Local" |
| tienda / amazonas / managers | `None` (todas) | `True` — un solo local, el del selector del pie |

`clases_permitidas=None` no es dejadez: el pipeline de tienda emite visitantes
cuyo `class_name` es la demografia ("Femenino 25-32"), que no encaja en ninguna
lista de clases de vehiculo. Filtrar por clase ahi tiraria el 100% de las
alertas.

## Politica de envio

  - CLASES: en modo perimetral se reenvian persona, personal_seguridad, moto,
    carro, camioneta y camion (la clase `objeto` NO). En los demas clientes no
    se filtra por clase.
  - EVENTOS DE UI: los que solo ocurren en el cliente (una captura manual) NO
    viajan a Jarvis: son ruido para quien lee las novedades.
  - AGRUPACION POR GRUPO: si entra/sale un GRUPO de la misma clase junto
    (varias personas, varios carros...), se envia UNA SOLA alerta, no una por
    objeto. Se logra con una ventana de agrupacion por (evento, camara, clase):
    la primera dispara la novedad y las siguientes del mismo evento y clase
    dentro de VENTANA_GRUPO_SEG se agrupan. Entrada y salida son eventos
    distintos, asi que no se agrupan entre si.
  - POR OBJETO: personas de interes (Re-ID), intrusiones, merodeo y visitantes
    de tienda NO se agrupan por clase — cada individuo identificado genera su
    propia novedad (dedup por objeto).

Se conecta en main.py junto al sidebar:

    reenviador = JarvisAlertForwarder(jarvis_api)
    for box in window_containter.list_box:
        box.alert_received.connect(reenviador.on_alert)
"""

from __future__ import annotations

import time
import unicodedata

from PySide6.QtCore import QObject, Slot

from ..logging import obtener

# Al .log del cliente, no a la consola: los `print` de antes solo se veian
# con la consola abierta, asi que "¿Jarvis envio o no?" no se podia responder
# despues de los hechos (8-ago-2026).
log = obtener(__name__)

# Centinela: "no me pasaron nada, usa la politica de la clase". No sirve None
# porque None es un valor CON significado propio (no filtrar por clase).
_HEREDA = object()


def _normaliza(texto: str) -> str:
    """MAYUSCULAS sin acentos: 'Camión' -> 'CAMION'."""
    sin_acento = unicodedata.normalize("NFKD", texto or "")
    sin_acento = "".join(c for c in sin_acento if not unicodedata.combining(c))
    return sin_acento.upper().strip()


def _nombre_local(destino) -> str:
    """Nombre legible del establecimiento para el log (viene como dict o str)."""
    if isinstance(destino, dict):
        return str(destino.get('name') or destino.get('nombre')
                   or destino.get('id') or '?')
    return str(destino or '?')


class JarvisAlertForwarder(QObject):
    """Puente panel de alertas -> API de Jarvis con agrupacion por grupo/clase."""

    # Prefijos de clase que SI se reenvian en modo perimetral. 'OBJETO' queda
    # fuera a proposito. Se compara contra el class_name normalizado
    # (mayusculas sin acentos); "PERSONA SEGURIDAD" cae bajo "PERSONA".
    CLASES_PERMITIDAS: tuple[str, ...] = (
        "PERSONA", "MOTO", "CARRO", "CAMIONETA", "CAMION",
    )
    # Ventana de AGRUPACION por (camara, clase): dentro de este lapso, un grupo
    # de la misma clase = 1 sola alerta. Subela para agrupar grupos que entran
    # mas espaciados; bajala para alertar con mas frecuencia.
    VENTANA_GRUPO_SEG: float = 30.0
    # Eventos que NO se agrupan por clase (cada objeto/persona su alerta).
    #   merodeo:   cada visitante recurrente merece su propia novedad (su
    #              global_id VIG-MER-<cam>-<id> es estable entre visitas).
    #   visitante: pipeline de tienda. Agruparlos por clase seria agrupar por
    #              "Femenino 25-32", es decir, perder personas distintas.
    EVENTOS_POR_OBJETO: tuple[str, ...] = ("alerta", "intrusion", "intrusión",
                                           "merodeo", "visitante")
    # Eventos que nacen y mueren en el CLIENTE (el operador pulsa un boton).
    # No son una novedad del establecimiento y no deben llegar a Jarvis.
    EVENTOS_IGNORADOS: tuple[str, ...] = ("captura guardada", "captura")
    # Purga de la tabla anti-fuga.
    MAX_CLAVES: int = 2000

    def __init__(self, jarvis_api, parent=None, *,
                 clases_permitidas=_HEREDA,
                 local_global: bool = False) -> None:
        """
        `clases_permitidas`: tupla de prefijos, o None para no filtrar por
        clase. Omitido = la politica perimetral de la clase.

        `local_global`: de donde sale el establecimiento cuando la alerta no
        trae uno propio. False (perimetrales) = de ninguna parte, la alerta se
        omite; el destino es SIEMPRE el select "Local" de la camara y un
        respaldo escondido mandaria la novedad al establecimiento equivocado.
        True (los otros tres) = del selector unico del pie, que es como esos
        clientes eligen el local.
        """
        super().__init__(parent)
        self._jarvis = jarvis_api
        self.local_global = bool(local_global)
        if clases_permitidas is _HEREDA:
            self.clases_permitidas: tuple[str, ...] | None = tuple(
                self.CLASES_PERMITIDAS)
        else:
            self.clases_permitidas = (tuple(clases_permitidas)
                                      if clases_permitidas else None)
        # Interruptor maestro (checkbox "Enviar a Jarvis" del pie). Con False
        # NO se envia nada a la API; el sidebar sigue mostrando las alertas.
        self.activo: bool = True
        # clave -> timestamp del ultimo envio.
        self._ultimo: dict[tuple, float] = {}
        self.enviadas: int = 0
        self.agrupadas: int = 0     # entradas suprimidas por pertenecer a un grupo
        self.sin_local: int = 0     # omitidas: no habia establecimiento
        self.ignoradas: int = 0     # eventos de UI y clases fuera de politica

    @Slot(bool)
    def set_activo(self, activo: bool) -> None:
        """Activa/desactiva el envio a la API (conectado al checkbox del pie)."""
        self.activo = bool(activo)
        log.info("envío a la API de Jarvis %s",
                 'ACTIVADO' if self.activo else 'DESACTIVADO')

    @Slot(dict)
    def on_alert(self, alerta: dict) -> None:
        """Recibe una tarjeta de alerta y decide si reenviarla a Jarvis."""
        try:
            if self._jarvis is None or not self.activo:
                return
            evento = str(alerta.get("event_type", "")).strip().lower()
            if evento in self.EVENTOS_IGNORADOS:
                self.ignoradas += 1
                return
            clase_norm = _normaliza(str(alerta.get("class_name") or ""))
            por_objeto = evento in self.EVENTOS_POR_OBJETO

            # Filtro de clases: las 'llegadas' solo de las permitidas; los
            # eventos por objeto (Re-ID/intrusion/visitante) pasan aunque el
            # class_name sea un nombre propio o una franja demografica.
            if (self.clases_permitidas is not None and not por_objeto
                    and not clase_norm.startswith(self.clases_permitidas)):
                self.ignoradas += 1
                return

            clave = self._clave(alerta, evento, clase_norm, por_objeto)
            ahora = time.time()
            if ahora - self._ultimo.get(clave, 0.0) < self.VENTANA_GRUPO_SEG:
                # Mismo grupo (misma camara+clase) dentro de la ventana, o el
                # mismo objeto repetido: se agrupa (no se reenvia).
                self.agrupadas += 1
                return

            # Destino ANTES de marcar la clave: si no hay local, la alerta no
            # se envio y no debe consumir la ventana de agrupacion — al elegir
            # el local, la siguiente igual se enviaria de inmediato.
            destino = self._destino(alerta)
            if not destino:
                self.sin_local += 1
                log.warning(
                    "alerta NO enviada a Jarvis: no hay establecimiento para "
                    "la cámara '%s' %s",
                    alerta.get('camera_name') or '?',
                    ("(select «Local» del recuadro)" if not self.local_global
                     else "(selector del pie de la ventana)"))
                return

            self._ultimo[clave] = ahora
            self._purgar(ahora)

            titulo = self._titulo(alerta, evento, alerta.get("class_name") or "",
                                  _nombre_local(destino))
            mensaje = str(alerta.get("description", "") or "")
            imagen = alerta.get("image_base64") or alerta.get("crop_image") or ""

            self._jarvis.enviar_novedad_async(
                base64_image=imagen, title=titulo, message=mensaje,
                establecimiento=destino)
            self.enviadas += 1
            # La evidencia que faltaba: antes un envio exitoso no dejaba NI
            # UNA linea y "¿Jarvis envio?" solo se respondia mirando el grupo.
            log.info("novedad enviada a Jarvis: «%s» (cámara '%s', local %s) "
                     "— %d enviadas en la sesión",
                     titulo, alerta.get('camera_name') or '?',
                     _nombre_local(destino), self.enviadas)
        except Exception as e:
            log.error("error reenviando alerta a Jarvis: %s", e)

    def _destino(self, alerta: dict):
        """Establecimiento (nombre o dict) al que va esta novedad, o ''.

        El de la CAMARA manda siempre. El global del pie solo entra en juego
        en los clientes que se configuraron con `local_global=True`, que son
        los que tienen un unico local por sesion.
        """
        propio = str(alerta.get("establecimiento") or "").strip()
        if propio:
            return propio
        if not self.local_global:
            return ""
        global_ = getattr(self._jarvis, "selected_establishment", None)
        if isinstance(global_, dict):
            return global_          # `Jarvis_api` acepta el dict tal cual
        return str(global_ or "").strip()

    def _clave(self, alerta: dict, evento: str, clase_norm: str,
               por_objeto: bool) -> tuple:
        """Clave de deduplicacion:
        - `dedup_clave` en la alerta -> la que diga el pipeline. Es el escape
          para los modos cuya unidad de agrupacion no es la clase ni encaja en
          `EVENTOS_POR_OBJETO`. PermanenciaArea la usa: una entrada y una
          salida POR PERSONA, que es justo lo que se le pide; sin esto, dos
          personas cruzando la misma puerta en menos de VENTANA_GRUPO_SEG se
          agruparian en una sola novedad y el tiempo de la segunda se perderia.
        - llegada/salida de area -> ('grp', evento, camara, clase): agrupa el
          GRUPO. El evento va en la clave para que entrada y salida NO se
          agrupen entre si (un grupo que sale si alerta aunque acabe de entrar).
        - Re-ID/intrusion/visitante -> ('obj', id): una por individuo.
        """
        propia = str(alerta.get("dedup_clave") or "").strip()
        if propia:
            return ("propia", propia)
        if por_objeto:
            gid = str(alerta.get("global_id")
                      or alerta.get("person_uuid")
                      or f"{alerta.get('camera_name','?')}:{alerta.get('tracker_id','?')}")
            return ("obj", gid)
        cam = str(alerta.get("camera_name") or alerta.get("camera_id") or "?")
        return ("grp", evento, cam, clase_norm)

    def _purgar(self, ahora: float) -> None:
        if len(self._ultimo) <= self.MAX_CLAVES:
            return
        self._ultimo = {k: t for k, t in self._ultimo.items()
                        if ahora - t < self.VENTANA_GRUPO_SEG}

    @staticmethod
    def _titulo(alerta: dict, evento: str, clase: str, local: str = "") -> str:
        """Titulo legible para la novedad de Jarvis."""
        propio = str(alerta.get("titulo_novedad") or "").strip()
        if propio:
            # La alerta trae su titulo YA redactado (permanencia): va tal
            # cual, sin el nombre de la camara — que aqui es el titulo de
            # la ventana de iVMS y no le dice nada a quien lee la novedad.
            return propio
        # El LOCAL de destino encabeza el cierre del titulo (17-ago-2026,
        # mismo criterio que WhatsApp); el titulo de la ventana de iVMS es
        # solo el respaldo cuando no hay establecimiento.
        lugar = (local or "").strip()
        if lugar == "?":            # _nombre_local sin nombre legible
            lugar = ""
        camara = (lugar
                  or str(alerta.get("camera_name") or alerta.get("camera_id") or ""))
        if evento == "llegada":
            base = f"{clase} detectado en el área"
        elif evento == "permanencia":
            base = f"{clase} permanece en el área"
        elif evento == "salida":
            base = f"{clase} salió del área"
        elif evento in ("intrusion", "intrusión"):
            base = f"Intrusión: {clase}"
        elif evento == "merodeo":
            base = f"⚠ MERODEO: {clase} entrando y saliendo del área"
        elif evento == "alerta":
            base = f"Persona de interés: {clase}"
        elif evento == "visitante":
            # Tienda: el "que" util es la demografia, que ya viene en `clase`.
            base = f"Visitante: {clase}" if clase else "Visitante identificado"
        elif evento:
            # Modo nuevo o pipeline que aun no tiene frase propia: mejor el
            # nombre del evento que un generico que no dice nada.
            base = f"{evento.capitalize()}: {clase}" if clase else evento.capitalize()
        else:
            base = f"Alerta: {clase}" if clase else "Alerta"
        return f"{base} — {camara}" if camara else base
