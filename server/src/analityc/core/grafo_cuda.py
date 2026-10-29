"""src/analityc/core/grafo_cuda.py — repetir un modelo con un grafo de CUDA.

## Por qué (6-oct-2026)

La tanda de clasificación (caras + edad/género) era lo que hacía saltar los
fps: con ella una imagen pasaba de ~52 a ~137 ms (cronómetro por etapas,
pasada del 6-oct 12:02). Dentro de edad/género (MiVOLO v2, PyTorch FP16 en
la Quadro RTX 4000) el modelo tardaba lo MISMO con 1 persona que con 3
(26,5 frente a 28,0 ms): no lo frenaba la GPU sino LANZAR sus cientos de
operaciones una a una desde la CPU (un i5-6500 compartido).

Un grafo de CUDA graba esa secuencia una vez y luego la repite entera con
una sola orden. Medido en un banco con el modelo real: 30,1 → 10,5 ms por
llamada, y la salida es idéntica (diferencia máxima 0,000000): son los
mismos cálculos, solo cambia cómo se lanzan.

## Cómo

`GrafosPorLote(modelo)` guarda un grafo por tamaño de lote (se graba la
primera vez que llega ese tamaño, hasta `LOTE_MAX`). La entrada se copia en
un tensor fijo y se repite el grafo; la salida se COPIA antes de devolverla
(la siguiente repetición la sobrescribe). Quien llama debe tenerlo bajo su
candado: un grafo no se puede repetir desde dos hilos a la vez.

`ELDE_GRAFOS_CUDA=0` lo apaga. Sin CUDA, con otro tamaño de lote o si algo
falla al grabar, devuelve None y quien llama usa el camino de siempre; tras
un fallo no se vuelve a intentar (un error en cada imagen llenaría el log).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VARIABLE = 'ELDE_GRAFOS_CUDA'
#: Tamaño de lote más grande que se graba (cada uno ocupa su memoria).
LOTE_MAX = 4


def activado() -> bool:
    """False con `ELDE_GRAFOS_CUDA` = 0 / false / no / off."""
    valor = os.getenv(VARIABLE, '1').strip().lower()
    return valor not in ('0', 'false', 'no', 'off')


class GrafosPorLote:
    """Grafos de CUDA de un modelo, uno por tamaño de lote."""

    def __init__(self, modelo: Any, nombre: str = 'modelo') -> None:
        self._modelo = modelo
        self._nombre = nombre
        self._grafos: Dict[tuple, Any] = {}
        self._roto = False

    def ejecutar(self, entrada: Any) -> Optional[Any]:
        """Salida del modelo para `entrada` (tensor CUDA) repitiendo su
        grafo, o None si no se puede (el que llama usa el camino normal)."""
        if self._roto or not activado():
            return None
        try:
            import torch
            if not entrada.is_cuda or entrada.shape[0] > LOTE_MAX:
                return None
            clave = (tuple(entrada.shape), entrada.dtype)
            grafo = self._grafos.get(clave)
            if grafo is None:
                grafo = self._grabar(entrada)
                self._grafos[clave] = grafo
            g, estatica, salida = grafo
            estatica.copy_(entrada)
            g.replay()
            # Un modelo puede devolver varias salidas (el detector RF-DETR
            # da cajas y puntuaciones, 29-oct-2026): se copia cada una.
            if isinstance(salida, (tuple, list)):
                return type(salida)(t.clone() for t in salida)
            return salida.clone()
        except Exception as exc:                            # noqa: BLE001
            self._roto = True
            logger.warning("grafo de CUDA de %s desactivado (%s): se sigue "
                           "con el camino de siempre", self._nombre, exc)
            return None

    def _grabar(self, entrada: Any):
        import torch
        estatica = torch.zeros_like(entrada)
        estatica.copy_(entrada)
        # Calentar en un stream aparte, como pide PyTorch antes de grabar.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._modelo(estatica)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        # 'thread_local': otros hilos (otras cámaras, el detector) pueden
        # seguir usando la GPU mientras se graba este.
        with torch.cuda.graph(g, capture_error_mode='thread_local'), \
                torch.no_grad():
            salida = self._modelo(estatica)
        logger.info("grafo de CUDA de %s grabado para la forma %s",
                    self._nombre, tuple(entrada.shape))
        return g, estatica, salida
