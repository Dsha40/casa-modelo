"""
src/analityc/core/estacionamiento_escena.py — guardia de cambio de escena.

## El problema que resuelve (verificado el 10-ago-2026)

La fuente del modo Estacionamiento es la CAPTURA de la ventana del iVMS-4200:
cuando el operador cambia el canal, la escena cambia de golpe pero la cámara
lógica sigue siendo la misma, y nada reseteaba tracker ni área
(`GestorRastreadores.reiniciar` y `RastreadorArea.olvidar_camara` eran código
muerto). Cada flip fabricaba:

  * a los 10 s, tarjetas de SALIDA con permanencia inventada para vehículos
    que siguen estacionados en el canal anterior;
  * al volver, una oleada de LLEGADAS de vehículos aparcados hace horas
    (y otro «estacionado» a los 300 s);
  * en la ventana intermedia, herencia de identidad ENTRE escenas por la
    re-vinculación IoU (un carro del canal 2 continuaba el cronómetro del
    carro del canal 1).

## Cómo decide

Huella de 24x18 en gris NORMALIZADA (media 0, desviación 1) por cámara; la
normalización hace la comparación inmune a cambios GLOBALES de brillo y
contraste (el conmutador día/IR no debe disparar el reset, el cambio de canal
sí). Diferencia media |a-b| entre huellas: misma escena con movimiento normal
~0.1-0.3; escena distinta ~1.0 (descorrelacionadas). Umbral 0.65
(ESTACIONAMIENTO_ESCENA_UMBRAL).

Dos protecciones contra falsos disparos:

  * CAMBIO exige DOS frames: el frame que difiere de la escena estable solo
    propone una candidata; si el siguiente vuelve a la escena de antes era un
    glitch de compresión y no pasa nada; si sostiene la candidata, es un
    cambio real y se avisa.
  * SIN_SENAL (frame casi uniforme: negro, azul «NO VIDEO») solo se avisa si
    dura ESTACIONAMIENTO_ESCENA_SIN_SENAL_SEG (8 s por defecto, justo antes
    de que la purga del área a los 10 s empiece a emitir salidas falsas).
    Un parpadeo negro corto no toca el estado.

Quien decide QUÉ resetear es el modo (`estacionamiento.py`); este módulo solo
observa frames y avisa, con cooldown para no repetir el aviso en ráfaga.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

#: Resultados de evaluar(): None = escena estable.
CAMBIO = "cambio"
SIN_SENAL = "sin_senal"

#: Tamaño de la huella (ancho, alto): grande para captar la estructura,
#: chico para que la comparación cueste microsegundos.
_HUELLA = (24, 18)

#: Desviación estándar (gris 0-255) por debajo de la cual el frame es SIN
#: SEÑAL. Una escena real, incluso nocturna en IR, la supera de sobra.
_STD_MINIMA = 4.0


def _flotante(nombre: str, defecto: float) -> float:
    try:
        return float((os.getenv(nombre) or "").strip() or defecto)
    except ValueError:
        return defecto


def activa() -> bool:
    return (os.getenv("ESTACIONAMIENTO_ESCENA_ACTIVA", "1").strip()
            not in ("0", "false", "no"))


def umbral() -> float:
    return _flotante("ESTACIONAMIENTO_ESCENA_UMBRAL", 0.65)


def sin_senal_seg() -> float:
    return _flotante("ESTACIONAMIENTO_ESCENA_SIN_SENAL_SEG", 8.0)


def cooldown_seg() -> float:
    return _flotante("ESTACIONAMIENTO_ESCENA_COOLDOWN_SEG", 5.0)


@dataclass(eq=False)
class _EstadoCamara:
    estable: Optional[np.ndarray] = None       # huella de la escena vigente
    candidata: Optional[np.ndarray] = None     # posible escena nueva (1 frame)
    sin_senal_desde: float = 0.0               # 0 = hay señal
    ultimo_aviso: float = float("-inf")        # nunca ha avisado


class DetectorEscena:
    """Detecta el cambio brusco de escena (o la pérdida de señal) por cámara."""

    def __init__(self) -> None:
        self._camaras: Dict[str, _EstadoCamara] = {}

    def evaluar(self, camara: str, frame: np.ndarray,
                ahora: Optional[float] = None) -> Optional[str]:
        """None si la escena sigue siendo la misma; CAMBIO o SIN_SENAL si no.

        Nunca lanza: ante cualquier fallo se comporta como escena estable
        (el guardia jamás debe tumbar el frame loop).
        """
        if not activa():
            return None
        try:
            return self._evaluar(camara, frame,
                                 time.time() if ahora is None else ahora)
        except Exception:                              # noqa: BLE001
            return None

    # ------------------------------------------------------------ interno
    @staticmethod
    def _diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def _evaluar(self, camara: str, frame: np.ndarray,
                 ahora: float) -> Optional[str]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        est = self._camaras.setdefault(camara, _EstadoCamara())

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 \
            else frame
        chica = cv2.resize(gris, _HUELLA,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        desviacion = float(chica.std())

        # ── Sin señal: solo cuenta si es SOSTENIDA ───────────────────
        if desviacion < _STD_MINIMA:
            if est.sin_senal_desde == 0.0:
                est.sin_senal_desde = ahora
                return None
            if ahora - est.sin_senal_desde < sin_senal_seg():
                return None
            # No se pisa la huella estable: si el video vuelve al MISMO
            # canal no habrá un falso CAMBIO encima del SIN_SENAL.
            return self._avisar(est, ahora, SIN_SENAL)
        est.sin_senal_desde = 0.0

        huella = (chica - chica.mean()) / max(1e-6, desviacion)
        if est.estable is None:
            est.estable = huella
            return None

        if est.candidata is None:
            if self._diff(huella, est.estable) < umbral():
                est.estable = huella       # rodar: la deriva lenta no dispara
                return None
            est.candidata = huella         # posible escena nueva: confirmar
            return None

        # Había candidata: ¿el frame sostiene la escena nueva, volvió la
        # vieja (glitch de un frame) o sigue la animación de transición?
        if self._diff(huella, est.estable) < umbral():
            est.candidata = None           # era un glitch: nada que hacer
            est.estable = huella
            return None
        if self._diff(huella, est.candidata) < umbral():
            est.estable = huella           # cambio REAL confirmado
            est.candidata = None
            return self._avisar(est, ahora, CAMBIO)
        est.candidata = huella             # transición en curso: esperar
        return None

    @staticmethod
    def _avisar(est: _EstadoCamara, ahora: float,
                veredicto: str) -> Optional[str]:
        """Aplica el cooldown: el mismo aviso no se repite en ráfaga."""
        if ahora - est.ultimo_aviso < cooldown_seg():
            return None
        est.ultimo_aviso = ahora
        return veredicto
