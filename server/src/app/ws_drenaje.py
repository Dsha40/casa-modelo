"""Drenaje del websocket: el lector NUNCA espera a la inferencia.

## El ciclo de muerte que corta este módulo (7-ago-2026)

El bucle del endpoint hacía `receive() -> procesar -> send()` en serie: mientras
la inferencia corría, el socket NO se leía. Si el cliente enviaba más de lo que
el servidor drenaba (tenía un bug que multiplicaba su ritmo por el número de
recuadros), el buffer del websocket llegaba a su tope y uvicorn PAUSABA la
lectura del TCP. A partir de ahí:

  - el cliente veía `Unable to write` (su buffer de envío lleno);
  - su PONG del keepalive quedaba sin leer en esa misma tubería;
  - uvicorn cerraba la conexión VIVA con 1011 `keepalive ping timeout`;
  - el cliente reconectaba a los 5 s, el procesador renacía y TODOS los
    contadores (permanencia, stand, visitantes) volvían a cero.

Cada conexión moría a los ~3 minutos, medido en `server.log` y `amazonas.log`.
Subir el timeout del ping (H-26) solo retrasó la muerte.

## El arreglo

Dos tareas por cliente en vez de una:

  - el **lector** (el bucle del endpoint) solo lee, valida y encola: el socket
    se drena SIEMPRE, así que el pong nunca queda atascado;
  - el **drenador** (`DrenajeCliente`) saca de la cola, procesa en el executor
    y envía la respuesta.

La cola es `ColaUltimoFrame`: guarda SOLO el frame más reciente de cada
cámara. Un frame viejo sin procesar no es un dato pendiente, es un dato
caducado: procesarlo retrasa al que lo reemplazó. Con el cliente sano (un
frame en vuelo por recuadro) la cola nunca descarta nada; los descartes se
cuentan y protegen contra clientes viejos o con bugs.

El envío comparte un candado (`candado_envio`) con el lector, que también
escribe (pings de aplicación, errores de contrato): dos tareas escribiendo a
la vez en el mismo websocket intercalarían mensajes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import msgpack

from . import telemetria_fps

logger = logging.getLogger(__name__)

#: Pipelines cuyo procesador reenvía WhatsApp por su cuenta (VigilanteWS y
#: sus subclases): reenviarlos también aquí duplicaría cada aviso — y el
#: duplicado salía con el formato viejo (cámara de iVMS + epoch crudo).
RUTA_PROPIA_WHATSAPP = ("VigilanteAmazonas", "Estacionamiento")


@dataclass
class ItemFrame:
    """Todo lo que hace falta para procesar un frame y responderlo."""

    #: El mensaje ORIGINAL del cliente: la respuesta es este mismo dict con
    #: `data` reemplazada por el resultado (el contrato de siempre).
    request: Dict[str, Any]
    #: Con qué formato llegó (y por tanto con cuál se responde).
    binario: bool
    #: Argumentos posicionales de `_full_frame_sync`, ya en orden.
    payload: Tuple[Any, ...]
    camera_id: Any
    camera_id_raw: Any = None
    camera_name: str = ''
    #: LOCAL de la cámara en ESTE frame (select del recuadro): encabeza el
    #: mensaje de WhatsApp en lugar del título de la ventana de iVMS.
    establecimiento: str = ''
    #: El interruptor del cliente en ESTE frame.
    enviar_whatsapp: bool = False
    type_inference: str = ''


class ColaUltimoFrame:
    """Cola por cámara que conserva SOLO el frame más reciente de cada una.

    El orden de atención entre cámaras es el de llegada (round-robin natural:
    una cámara nueva se pone al final). Reemplazar el frame de una cámara que
    ya esperaba NO le cambia el turno: solo refresca el dato.
    """

    def __init__(self) -> None:
        self._orden: deque = deque()
        self._items: Dict[Any, ItemFrame] = {}
        self._evento = asyncio.Event()
        #: Frames reemplazados sin procesar. Con el cliente sano es 0; si
        #: crece, hay un cliente enviando sin esperar sus respuestas.
        self.descartados = 0

    def __len__(self) -> int:
        return len(self._items)

    def poner(self, item: ItemFrame) -> bool:
        """Encola `item`. True si REEMPLAZÓ un frame sin procesar de su
        cámara: el drenador lo anota en la telemetría, que hasta el 30-sep
        adivinaba los descartes restando recibidos menos procesados."""
        if item.camera_id in self._items:
            self.descartados += 1
            if self.descartados in (1, 10, 100) or self.descartados % 1000 == 0:
                logger.warning(
                    "ColaUltimoFrame: %d frame(s) reemplazados sin procesar "
                    "(camara %s). El cliente envia mas rapido de lo que se "
                    "procesa.", self.descartados, item.camera_id)
            reemplazo = True
        else:
            self._orden.append(item.camera_id)
            reemplazo = False
        self._items[item.camera_id] = item
        self._evento.set()
        return reemplazo

    async def sacar(self) -> ItemFrame:
        """Espera hasta que haya algo y devuelve el frame más antiguo en turno."""
        while not self._orden:
            self._evento.clear()
            await self._evento.wait()
        camera_id = self._orden.popleft()
        return self._items.pop(camera_id)

    def vaciar(self) -> None:
        self._orden.clear()
        self._items.clear()
        self._evento.set()


class DrenajeCliente:
    """La tarea que procesa y responde los frames de UN cliente.

    Sustituye al `ClientWorker` de futures: aquí el lector no espera nada.
    Las dependencias entran por el constructor para que este módulo no importe
    `app.py` (ciclo) y para que las pruebas puedan pasar dobles.
    """

    def __init__(self, client_id: str, websocket: Any,
                 procesar: Callable[..., Dict[str, Any]],
                 executor_getter: Callable[[], Any],
                 candado_envio: asyncio.Lock,
                 reenviar_whatsapp: Optional[Callable[..., int]] = None,
                 registrar_saliente: Optional[Callable[..., None]] = None
                 ) -> None:
        self.client_id = client_id
        self.websocket = websocket
        self._procesar = procesar
        self._executor_getter = executor_getter
        self._candado_envio = candado_envio
        self._reenviar_whatsapp = reenviar_whatsapp
        self._registrar_saliente = registrar_saliente
        self.cola = ColaUltimoFrame()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        #: Cuántas respuestas se enviaron. Telemetría para /health si hace falta.
        self.enviados = 0

    # -- Ciclo de vida ----------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(
            self._run(), name=f"drenaje-{self.client_id}")

    async def stop(self) -> None:
        self._stopped = True
        self.cola.vaciar()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def encolar(self, item: ItemFrame) -> None:
        """La llama el lector. Nunca bloquea, nunca lanza."""
        if not self._stopped:
            telemetria_fps.anotar_recibido(
                item.camera_id, telemetria_fps.tamano_de_payload(item.payload))
            if self.cola.poner(item):
                telemetria_fps.anotar_descartado(item.camera_id)

    # -- La tarea ---------------------------------------------------------
    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._stopped:
            try:
                item = await self.cola.sacar()
            except asyncio.CancelledError:
                break

            try:
                # perf_counter y no monotonic (5-oct-2026): en Windows,
                # monotonic va a saltos de ~15,6 ms y el «proceso» de la
                # telemetría salía en escalones (94, 109, 125…). El 2-oct no
                # distinguió una diferencia de 10 ms entre dos pasadas A/B
                # que el cliente sí vio. En Linux los dos son finos.
                t_proc = time.perf_counter()
                result = await loop.run_in_executor(
                    self._executor_getter(), self._procesar, *item.payload)
                telemetria_fps.anotar_procesado(
                    item.camera_id, (time.perf_counter() - t_proc) * 1000.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:                        # noqa: BLE001
                logger.error("Error procesando frame client=%s: %s",
                             self.client_id, exc)
                result = {"status": "error", "message": str(exc)}

            if self._stopped:
                break
            if not await self._responder(item, result):
                break

    async def _responder(self, item: ItemFrame,
                         result: Dict[str, Any]) -> bool:
        """Reenvía a WhatsApp si toca y responde al cliente.

        `False` = el socket ya no sirve y el drenador debe terminar (el
        lector verá la desconexión por su lado y hará la limpieza).
        """
        try:
            if (item.enviar_whatsapp and self._reenviar_whatsapp is not None
                    and item.type_inference not in RUTA_PROPIA_WHATSAPP
                    and isinstance(result, dict)):
                self._reenviar_whatsapp(
                    result.get("metadata"),
                    camera_name=str(item.camera_name or ""),
                    camera_id=str(item.camera_id_raw or ""),
                    establecimiento=str(item.establecimiento or ""))
        except Exception as exc:                            # noqa: BLE001
            logger.debug("reenvio WhatsApp fallo client=%s: %s",
                         self.client_id, exc)

        request = item.request
        request["data"] = result
        try:
            if self._registrar_saliente is not None:
                self._registrar_saliente("saliente", item.type_inference,
                                         request)
        except Exception:                                   # noqa: BLE001
            pass

        try:
            async with self._candado_envio:
                if item.binario:
                    await self.websocket.send_bytes(
                        msgpack.packb(request, use_bin_type=True))
                else:
                    await self.websocket.send_json(request)
            self.enviados += 1
            return True
        except asyncio.CancelledError:
            return False
        except Exception as exc:                            # noqa: BLE001
            # Casi siempre es que el cliente se fue entre el proceso y la
            # respuesta. No es un error del drenador: se termina en silencio
            # y el lector reporta la desconexión con su propio criterio.
            logger.info("Drenaje %s: no se pudo responder (%s); se detiene",
                        self.client_id, exc)
            return False
