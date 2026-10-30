"""src/analityc/core/detector_grafo.py — el detector RF-DETR con un grafo de CUDA.

## NO ENCENDER TODAVÍA: desactiva los grafos de MiVOLO y ViTPose (29-oct-2026)

En la pasada con grafo de la A/B de abajo, los grafos de MiVOLO (16:34:04)
y de ViTPose (16:34:24) se desactivaron solos con «Inplace update to
inference tensor outside InferenceMode is not allowed». En los 18 arranques
sin este grafo (26 a 29-oct) no pasó nunca. Todo siguió funcionando, pero
MiVOLO y ViTPose por el camino lento.

Por qué: rfdetr llama al modelo dentro de `@torch.inference_mode()`
(`predict`, rfdetr/detr.py), así que este grafo es el PRIMERO del proceso y
se graba en modo inferencia. Al empezar a grabar un grafo, torch actualiza un
estado interno que crea con el primero; nacido en modo inferencia, ya no se
puede actualizar fuera de él, que es donde graban MiVOLO y ViTPose: falla
`capture_begin` y `GrafosPorLote` apaga su grafo. Reproducido con dos modelos
de juguete y el mismo `GrafosPorLote`: solo falla si el primer grafo se grabó
en modo inferencia. Que ese estado sea el del generador de números aleatorios
de la GPU es deducción: lo sugiere el segundo error que queda tras el fallo
(«Offset increment outside graph capture»); el código de torch es compilado.

Arreglo probado solo con los modelos de juguete: grabar siempre con
`torch.inference_mode(False)` en `GrafosPorLote._grabar`. Antes de encender
esto hay que hacerlo y repetir la A/B: la de abajo no vale para decidir.

## Por qué (29-oct-2026)

El detector es la etapa más cara del servidor (~35 ms por imagen en el PC
de la tienda, cronómetro por etapas). Con el perfilador de torch: manda 482
operaciones sueltas a la GPU en cada imagen y el procesador tarda en
lanzarlas casi lo mismo que la GPU en hacerlas (~28 frente a ~27 ms). Es el
caso de ViTPose y MiVOLO (`grafo_cuda.py`): grabar la secuencia una vez y
repetirla de golpe.

Banco con el detector real (`banco_grafo_detector.py` y
`banco_grafo_forward.py` del scratchpad, PC de la tienda con el servidor en
vivo al lado):

  * mismas detecciones: 93 de 93 en 40 imágenes del vídeo de la cámara 3,
    diferencia máxima 0,000000;
  * solo el modelo, 300 llamadas alternando: 32,1 → 24,1 ms.

Preparar la imagen y filtrar los resultados (`predict` de rfdetr) siguen
como antes: solo se graba el modelo de inferencia que rfdetr ya traza.

A/B en el servidor (29-oct, vídeo de la cámara 2 en Media Player, 9,5 min
cada pasada, cronómetro por etapas): el detector pasa de 34,8 a 30,0 ms de
media por imagen, y baja en cada grupo (0 personas 35,0 → 30,0; 1 persona
33,8 → 29,8; 2 personas 36,5 → 31,3). El procesador del detector solo baja
1,6 ms (28,5 → 26,9): se ahorra espera, no CPU. Las imágenes por segundo no
cambiaron de forma visible (13,4 frente a 13,1): en la pasada con grafo,
MiVOLO y ViTPose fueron SIN sus grafos (ver arriba), y además hubo más
gente. Sin aumento visible de memoria de la GPU; grabar el grafo al
arrancar, ~3 s. Sin probar: iVMS en vivo y varias cámaras.

## Cómo

`instalar(rfdetr)` sustituye `rfdetr.model.inference_model` por un
envoltorio que repite el grafo (`GrafosPorLote`, un grafo por forma de
entrada: siempre 1×3×576×576 hoy). Si no hay CUDA, si rfdetr no trazó el
modelo o si grabar falla, se usa el modelo de siempre. Quien llama ya lo
tiene bajo su candado (`_DetectorCompartido._candado`): un grafo no se
puede repetir desde dos hilos a la vez.

## Interruptor

APAGADO por defecto, y no encender hasta el arreglo de arriba:
`ELDE_GRAFO_DETECTOR=1` lo enciende. `ELDE_GRAFOS_CUDA=0` apaga todos los
grafos, este incluido.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

VARIABLE = 'ELDE_GRAFO_DETECTOR'


def activado() -> bool:
    """True solo con `ELDE_GRAFO_DETECTOR` = 1 / true / si / on."""
    return os.getenv(VARIABLE, '').strip().lower() in ('1', 'true', 'si', 'sí', 'on')


class _ConGrafo:
    """Se llama como el modelo; repite su grafo o, si no se puede, lo llama."""

    def __init__(self, modelo: Any) -> None:
        from .grafo_cuda import GrafosPorLote
        self.original = modelo
        self._grafos = GrafosPorLote(modelo, 'RF-DETR')

    def __call__(self, entrada: Any) -> Any:
        salida = self._grafos.ejecutar(entrada)
        return self.original(entrada) if salida is None else salida


def instalar(rfdetr: Any) -> bool:
    """Pone el grafo en el modelo de inferencia de `rfdetr` (un RFDETR ya
    optimizado con `inference()`). Nunca lanza; True si quedó puesto."""
    if not activado():
        return False
    try:
        nucleo = rfdetr.model
        modelo = getattr(nucleo, 'inference_model', None)
        if modelo is None or isinstance(modelo, _ConGrafo):
            return False
        if getattr(getattr(nucleo, 'device', None), 'type', None) != 'cuda':
            return False
        nucleo.inference_model = _ConGrafo(modelo)
        logger.warning("detector: grafo de CUDA ENCENDIDO (%s=1)", VARIABLE)
        return True
    except Exception as exc:                                # noqa: BLE001
        logger.warning("detector: sin grafo de CUDA (%s)", exc)
        return False
