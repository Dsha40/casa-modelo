"""Telemetria del ciclo cliente <-> servidor, escrita a disco por fotograma.

## Para que

El 24-sep-2026 se midio desde el servidor un ciclo de 156 ms por fotograma
(6,4 fps con una camara) con la GPU al 37 %: ~57 ms de calculo y ~99 ms de
«espera». Desde el servidor no se ve en que se va esa espera. Esto pone
marcas de tiempo en el CLIENTE, una linea JSON por fotograma, para sacar el
desglose.

## Por que a disco y no a la pantalla

El operador no puede tener abierto el cliente y la sesion de diagnostico a la
vez: usa el cliente un rato, lo cierra, y las medidas se leen despues. Por eso
se escribe amortiguado pero se vacia cada `VACIAR_CADA_S` segundos y al salir:
un cierre de golpe pierde como mucho esos segundos, no el fichero.

## Las marcas (todas en epoch, `time.time()`)

Ruta de VENTANA (iVMS; la captura corre en OTRO proceso, `capture_worker`):

    t_captura    el worker saco la imagen (cabecera `timestamp` del worker)
    t_recibido   el mensaje llego por el pipe a `loop_show_result`
    t_envio      volvio `send_binary_frame` (msgpack + encolar en el socket)
    t_respuesta  llego la respuesta de ESTE recuadro
    t_paneles    se emitieron alertas y bloque de Anaqueles a los paneles
    t_pintado    se entrego el pixmap a la etiqueta (Qt lo repinta despues)

Ruta de DVR: igual, pero `t_captura` es la llegada del QImage del hilo RTSP y
hay ademas `t_jpeg` (la compresion ocurre aqui, en el hilo de la interfaz).

Entre fotogramas queda un hueco que NO es de nadie: desde el `t_pintado` de
uno hasta el `t_recibido` del siguiente, el recuadro esta libre esperando a
que el worker entregue una imagen nueva. `descartados` cuenta cuantas
imagenes se tiraron DESDE EL ENVIO ANTERIOR (llegaron con uno en vuelo; solo
vale la ultima): va en la linea del fotograma siguiente, no en la del que
estaba en vuelo.

## Estados

    ok             respuesta con status success
    error          respuesta con status error (o excepcion al tratarla)
    sin_respuesta  se abrio otro fotograma antes de que llegara la respuesta
                   (red de seguridad de 10 s, reconexion, stop...)
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

#: Cada cuanto se vacia el buffer a disco. Es lo maximo que se pierde si el
#: cliente se cierra de golpe (un kill no pasa por atexit).
VACIAR_CADA_S = 2.0

#: A 6 fps son ~21.600 lineas/hora de ~300 bytes: ~6,5 MB/hora. Con 20 MB y
#: una copia caben ~6 horas, de sobra para una sesion de medida.
MAX_BYTES = 20 * 1024 * 1024


def ruta_por_defecto() -> Path:
    """`ELDE_TELEMETRIA_CICLO` manda; si no, %LOCALAPPDATA%\\tienda_view\\."""
    explicita = (os.getenv('ELDE_TELEMETRIA_CICLO') or '').strip()
    if explicita:
        return Path(explicita)
    base = (os.getenv('LOCALAPPDATA') or '').strip()
    if base:
        return Path(base) / 'tienda_view' / 'telemetria_ciclo.jsonl'
    return Path.home() / '.elde' / 'telemetria_ciclo.jsonl'


class EscritorJsonl:
    """Un fichero JSONL amortiguado, con vaciado periodico y una rotacion."""

    def __init__(self, ruta: Path, max_bytes: int = MAX_BYTES,
                 vaciar_cada_s: float = VACIAR_CADA_S):
        self.ruta = Path(ruta)
        self.max_bytes = int(max_bytes)
        self.vaciar_cada_s = float(vaciar_cada_s)
        self._buffer = []
        self._ultimo_vaciado = time.monotonic()
        self._lock = threading.Lock()

    def escribir(self, registro: Dict[str, Any]) -> None:
        linea = json.dumps(registro, ensure_ascii=False, separators=(',', ':'))
        with self._lock:
            self._buffer.append(linea)
            if time.monotonic() - self._ultimo_vaciado >= self.vaciar_cada_s:
                self._vaciar_locked()

    def vaciar(self) -> None:
        with self._lock:
            self._vaciar_locked()

    def _vaciar_locked(self) -> None:
        self._ultimo_vaciado = time.monotonic()
        if not self._buffer:
            return
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            if (self.ruta.exists()
                    and self.ruta.stat().st_size >= self.max_bytes):
                # Una sola copia: `.1` es la tanda anterior entera.
                viejo = self.ruta.with_name(self.ruta.name + '.1')
                os.replace(self.ruta, viejo)
            with open(self.ruta, 'a', encoding='utf-8') as f:
                f.write('\n'.join(self._buffer) + '\n')
            self._buffer.clear()
        except OSError:
            # Medir nunca puede tumbar el cliente. Se descarta la tanda para
            # que el buffer no crezca sin fin si el disco no acepta.
            self._buffer.clear()


_escritor: Optional[EscritorJsonl] = None
_escritor_lock = threading.Lock()


def escritor() -> EscritorJsonl:
    """El escritor del proceso: UN fichero para todos los recuadros (cada
    linea lleva su `camera_id`)."""
    global _escritor
    with _escritor_lock:
        if _escritor is None:
            _escritor = EscritorJsonl(ruta_por_defecto())
            atexit.register(_escritor.vaciar)
        return _escritor


def epoch_de_cabecera(header: Any) -> Optional[float]:
    """El `timestamp` isoformat que pone el capture_worker, en epoch.

    El worker usa `datetime.now()` (hora local, sin zona) en la MISMA maquina,
    asi que `.timestamp()` lo lleva al mismo reloj que `time.time()`."""
    try:
        return datetime.fromisoformat(str(header['timestamp'])).timestamp()
    except Exception:                                       # noqa: BLE001
        return None


class Apartado:
    """Un fotograma ya respondido que se sigue marcando aparte (ver
    `TelemetriaCiclo.apartar`). Mismas `marcar`, `dato` y `cerrar`."""

    def __init__(self, registro: Dict[str, Any], salida: EscritorJsonl):
        self._registro: Optional[Dict[str, Any]] = registro
        self._salida = salida

    def marcar(self, nombre: str, t: Optional[float] = None) -> None:
        if self._registro is not None:
            self._registro[nombre] = round(time.time() if t is None else t, 4)

    def dato(self, nombre: str, valor: Any) -> None:
        if self._registro is not None:
            self._registro[nombre] = valor

    def cerrar(self, estado: str) -> None:
        if self._registro is None:
            return
        registro, self._registro = self._registro, None
        registro['estado'] = estado
        self._salida.escribir(registro)


class TelemetriaCiclo:
    """Las marcas de UN recuadro. Un fotograma en vuelo a la vez, como el
    propio recuadro: `abrir` empieza uno y `cerrar` lo escribe.

    Con `activa=False` todo es un no-op barato: los enganches del recuadro se
    pueden dejar puestos siempre."""

    def __init__(self, activa: bool = False, salida: Optional[EscritorJsonl] = None):
        self.activa = bool(activa)
        self._salida = salida
        self._actual: Optional[Dict[str, Any]] = None
        self._seq = 0
        #: Imagenes que entrego el worker mientras habia uno en vuelo.
        self.descartados = 0

    def _escritor(self) -> EscritorJsonl:
        return self._salida if self._salida is not None else escritor()

    def abrir(self, camera_id: str, ruta: str, **marcas: Optional[float]) -> None:
        if not self.activa:
            return
        if self._actual is not None:
            # El anterior no tuvo respuesta: la red de seguridad, una
            # reconexion o un stop liberaron el envio. Se escribe igual: un
            # hueco sin explicar en el fichero enganaria al analisis.
            self.cerrar('sin_respuesta')
        self._seq += 1
        self._actual = {'seq': self._seq, 'camera_id': camera_id,
                        'ruta': ruta, 'descartados': self.descartados}
        self.descartados = 0
        for nombre, valor in marcas.items():
            if valor is not None:
                self._actual[nombre] = round(float(valor), 4)

    def marcar(self, nombre: str, t: Optional[float] = None) -> None:
        if self._actual is not None:
            self._actual[nombre] = round(time.time() if t is None else t, 4)

    def dato(self, nombre: str, valor: Any) -> None:
        if self._actual is not None:
            self._actual[nombre] = valor

    def en_vuelo(self) -> bool:
        return self._actual is not None

    def apartar(self) -> Optional['Apartado']:
        """Saca el fotograma en vuelo SIN escribirlo, para seguir marcandolo
        (paneles, pintado) mientras YA vuela el siguiente (5-oct-2026: el
        recuadro envia la siguiente imagen ANTES de pintar la respuesta).
        Sin esto, `abrir` del siguiente lo cerraria como 'sin_respuesta'."""
        if self._actual is None:
            return None
        registro, self._actual = self._actual, None
        return Apartado(registro, self._escritor())

    def cerrar(self, estado: str) -> None:
        if self._actual is None:
            return
        registro, self._actual = self._actual, None
        registro['estado'] = estado
        self._escritor().escribir(registro)

    def apagar(self) -> None:
        """Al desactivarla: se tira el fotograma a medias y se vacia."""
        self.activa = False
        self._actual = None
        self.descartados = 0
        self._escritor().vaciar()
