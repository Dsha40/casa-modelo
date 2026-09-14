"""src/analityc/core/tienda_alertas.py — avisos de NEGOCIO de la tienda.

## Por qué existe

El pipeline avisaba de una sola cosa: «hay un visitante nuevo»
(`_emitir_visitante`), más las alertas de personal. Todo lo demás había que
descubrirlo mirando el dashboard: que el aforo se disparó, que un pasillo lleva
tres horas sin que entre nadie, que el tráfico de esta hora es la mitad de lo
normal. Un dato que solo existe si alguien mira la pantalla no sirve para
reaccionar; sirve para explicar después lo que pasó.

Estas tres reglas miran lo mismo que miraría el encargado, y avisan por el
camino que ya existe: la cola de `metadata['alerts']`, que el cliente pinta en
su panel lateral y —solo si el operador encendió su interruptor— reenvía a
WhatsApp y a Jarvis.

## Las reglas

| Regla | Qué mira | Por defecto |
|---|---|---|
| **Aforo alto** | personas dentro del área ahora | **apagada** |
| **Pasillo dormido** | una cámara que hoy tuvo gente y lleva N minutos sin una sola entrada | 120 min |
| **Caída de tráfico** | las entradas de esta hora contra la media de esa misma hora en los días anteriores | −50 % |

**El aforo viene apagado a propósito.** «Demasiada gente» depende del local: no
hay un número honesto que poner por defecto, y encender una alarma con una
cifra inventada es peor que no tenerla. Se activa poniendo
`ELDE_TIENDA_AFORO_MAXIMO`.

Las otras dos SÍ vienen encendidas porque no dependen de ninguna constante
inventada: se comparan con **la propia historia de esa tienda**. Aun así solo
llegan al WhatsApp si el operador tiene su interruptor puesto; sin él se
quedan en el panel del cliente.

## Reglas de la casa que cumple

- **Nunca inventa una comparación.** La caída de tráfico calla mientras no
  haya al menos `CAIDA_MIN_DIAS` jornadas con datos a esa hora: sin base, no
  hay caída, hay desconocimiento.
- **No repite.** Una alerta por regla y cámara cada `COOLDOWN_S`.
- **No corre en cada frame.** Se evalúa como mucho cada `EVAL_S`, y las
  consultas al almacén viven detrás de ese throttle.
- **Nunca lanza.** Esto se llama dentro del bucle de inferencia.
- **Cero valores incrustados**: todo sale del entorno.

| Variable | Defecto | Qué es |
|---|---|---|
| `ELDE_TIENDA_AFORO_MAXIMO` | `0` (apagada) | personas dentro que disparan el aviso |
| `ELDE_TIENDA_PASILLO_DORMIDO_MIN` | `120` | minutos sin una entrada |
| `ELDE_TIENDA_CAIDA_PCT` | `50` | caída porcentual que se avisa |
| `ELDE_TIENDA_CAIDA_MIN_DIAS` | `3` | jornadas de base mínimas para comparar |
| `ELDE_TIENDA_ALERTAS_COOLDOWN_S` | `1800` | mínimo entre avisos iguales |
| `ELDE_TIENDA_ALERTAS_EVAL_S` | `60` | cada cuánto se evalúan las reglas |
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from . import tienda_almacen as _almacen

logger = logging.getLogger(__name__)


def _num(nombre: str, defecto: float) -> float:
    try:
        return float(os.getenv(nombre, str(defecto)))
    except (TypeError, ValueError):
        return defecto


#: 0 = apagada. No hay un aforo «normal» que valga para toda tienda.
AFORO_MAXIMO: int = int(_num('ELDE_TIENDA_AFORO_MAXIMO', 0))

#: Minutos sin una sola entrada para considerar dormido un pasillo que hoy YA
#: tuvo gente. 0 = apagada.
PASILLO_DORMIDO_MIN: float = _num('ELDE_TIENDA_PASILLO_DORMIDO_MIN', 120)

#: Caída porcentual de la hora en curso contra su media histórica. 0 = apagada.
CAIDA_PCT: float = _num('ELDE_TIENDA_CAIDA_PCT', 50)

#: Jornadas con datos mínimas para que la media signifique algo.
CAIDA_MIN_DIAS: int = int(_num('ELDE_TIENDA_CAIDA_MIN_DIAS', 3))

#: Días hacia atrás en los que buscar esa base.
CAIDA_VENTANA_DIAS: int = int(_num('ELDE_TIENDA_CAIDA_VENTANA_DIAS', 14))

#: Minutos que tiene que llevar la hora en curso antes de compararla con su
#: media. En el minuto 1 de las 10:00 SIEMPRE «va por debajo» de la hora
#: entera de otros días, y avisar de eso sería avisar de que el reloj avanza.
CAIDA_MIN_MINUTOS: int = int(_num('ELDE_TIENDA_CAIDA_MIN_MINUTOS', 15))

COOLDOWN_S: float = _num('ELDE_TIENDA_ALERTAS_COOLDOWN_S', 1800)
EVAL_S: float = _num('ELDE_TIENDA_ALERTAS_EVAL_S', 60)

#: `whatsapp_alertas` pinta 🚨 salvo que el tipo esté en su lista de
#: informativos, donde ya figuran «aforo» y «visitante». Estos avisos son de
#: negocio, no intrusiones: se nombran para caer del lado informativo y que el
#: grupo distinga de un vistazo una alarma de una métrica.
EVENTO_AFORO = 'Aforo'
EVENTO_PASILLO = 'Aforo pasillo dormido'
EVENTO_CAIDA = 'Aforo caida de trafico'


class AlertasNegocioMixin:
    """Las tres reglas de negocio, evaluadas fuera del camino caliente."""

    def _evaluar_alertas_negocio(self) -> None:
        """Punto de entrada desde `process_frame`. Barata y muda por defecto.

        Sale por la puerta en el primer `if` la inmensa mayoría de los frames:
        con 25 fps y `EVAL_S = 60`, 1.499 de cada 1.500 llamadas son una resta
        de flotantes.
        """
        try:
            ahora = time.time()
            if (ahora - getattr(self, '_alertas_negocio_ts', 0.0)) < EVAL_S:
                return
            self._alertas_negocio_ts = ahora

            camara = getattr(self, '_camera_id_actual', None)
            if camara is None:
                return
            self._regla_aforo_alto(ahora)
            self._regla_pasillo_dormido(camara, ahora)
            self._regla_caida_trafico(camara, ahora)
        except Exception:                                   # noqa: BLE001
            pass                    # jamás tumbar el vídeo por un aviso

    # ── Reglas ──────────────────────────────────────────────────────────
    def _regla_aforo_alto(self, ahora: float) -> None:
        if AFORO_MAXIMO <= 0:
            return
        dentro = sum(1 for st in (getattr(self, '_zone_state', None) or {}).values()
                     if st.get('estado') == 'DENTRO')
        if dentro < AFORO_MAXIMO:
            return
        self._encolar_aviso_negocio(
            'aforo', EVENTO_AFORO,
            f'{dentro} personas dentro del área (el aviso salta a partir de '
            f'{AFORO_MAXIMO}). Conviene abrir otra caja o reforzar el pasillo.',
            ahora)

    def _regla_pasillo_dormido(self, camara: Any, ahora: float) -> None:
        """Un pasillo que hoy tuvo gente y de pronto no tiene ninguna.

        Solo avisa si HUBO tráfico hoy: una cámara que aún no ha visto a nadie
        no está dormida, está empezando el día.
        """
        if PASILLO_DORMIDO_MIN <= 0:
            return
        ultima = _almacen.get_almacen().ultima_entrada(camara)
        if ultima is None:
            return
        minutos = (ahora - ultima) / 60.0
        if minutos < PASILLO_DORMIDO_MIN:
            return
        self._encolar_aviso_negocio(
            'pasillo_dormido', EVENTO_PASILLO,
            f'sin una sola entrada desde hace {int(minutos)} min, y hoy sí '
            'hubo gente. Revisar si algo tapa la cámara, si se movió un '
            'expositor o si se cerró el paso.',
            ahora)

    def _regla_caida_trafico(self, camara: Any, ahora: float) -> None:
        """La hora en curso contra la media de esa misma hora otros días.

        Calla mientras no haya base suficiente: sin días con los que comparar
        no hay caída, hay desconocimiento — el mismo criterio que el «sin base
        de comparación» del dashboard.
        """
        if CAIDA_PCT <= 0:
            return
        alm = _almacen.get_almacen()
        hoy = _almacen.jornada_de(ahora)
        hora = time.strftime('%H', time.localtime(ahora))

        # Solo tiene sentido comparar una hora ya empezada de verdad: en el
        # primer minuto de las 10:00 «va por debajo» siempre.
        if int(time.strftime('%M', time.localtime(ahora))) < CAIDA_MIN_MINUTOS:
            return

        desde = _almacen.jornada_de(ahora - CAIDA_VENTANA_DIAS * 86400)
        filas = alm.serie(desde, hoy, camera_id=camara, grano='hora')
        actual = 0
        base: List[int] = []
        for f in filas:
            if str(f.get('hora') or '') != hora:
                continue
            if f['jornada'] == hoy:
                actual = f['entradas']
            else:
                base.append(f['entradas'])
        if len(base) < CAIDA_MIN_DIAS:
            return                      # sin base: no se inventa una caída
        media = sum(base) / len(base)
        if media <= 0:
            return
        caida = 100.0 * (media - actual) / media
        if caida < CAIDA_PCT:
            return
        self._encolar_aviso_negocio(
            f'caida_{hora}', EVENTO_CAIDA,
            f'a las {hora}:00 van {actual} entradas y lo normal a esta hora '
            f'son {media:.0f} ({caida:.0f} % menos, sobre {len(base)} días).',
            ahora)

    # ── Emisión ─────────────────────────────────────────────────────────
    def _encolar_aviso_negocio(self, clave: str, tipo: str, detalle: str,
                               ahora: float) -> None:
        """Mete el aviso en la MISMA cola que el resto del pipeline.

        Con anti-repetición por (regla, cámara): un pasillo dormido lo sigue
        estando el minuto siguiente, y avisar cada minuto convierte el grupo
        de WhatsApp en ruido que nadie lee.
        """
        camara = getattr(self, '_camera_id_actual', None)
        llave = f'{clave}:{camara}'
        vistas = getattr(self, '_alertas_negocio_vistas', None)
        if vistas is None:
            vistas = {}
            self._alertas_negocio_vistas = vistas
        if (ahora - vistas.get(llave, 0.0)) < COOLDOWN_S:
            return
        vistas[llave] = ahora

        nombre = str(getattr(self, '_camera_display_name', '') or camara or '')
        alerta: Dict[str, Any] = {
            'event_type': tipo,
            'class_name': tipo,
            'description': detalle,
            'timestamp': ahora,
            'camera_name': nombre,
            'camera_id': str(camara or ''),
            'negocio': True,        # marca: no es una deteccion, es una metrica
        }
        pendientes = getattr(self, '_alertas_pendientes', None)
        if pendientes is None:
            pendientes = []
            self._alertas_pendientes = pendientes
        pendientes.append(alerta)
        if len(pendientes) > 200:   # misma cota que el resto de la cola
            self._alertas_pendientes = pendientes[-100:]
        logger.info('tienda: aviso de negocio (%s) en %s — %s',
                    tipo, nombre, detalle)
