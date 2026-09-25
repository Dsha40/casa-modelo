"""Lo que falta para cerrar la ficha de una persona: esperarlo, con caducidad.

## El fallo que corrige (medido el 24-sep-2026)

`_update_captures_demographics` cerraba la ficha de cada persona **en cuanto
se conocía el género O la identidad**, lo primero que llegara, y ya no la
volvía a mirar (`_captured_demo_done`). Lo que llegaba después se perdía, en
las dos direcciones:

  - **género primero, identidad después**: el género solo se escribe en la
    base si la persona tiene identidad, así que quedaba fuera. Ese día, 3 de 3
    clasificados no llegaron a la base.
  - **identidad primero, género después**: la ficha se sellaba con el motivo
    «muestras_insuficientes» mientras el clasificador SEGUÍA reuniendo
    muestras. Comprobado: la ficha del track 100004 se cerró a las 21:02:53 y
    a los 34 s el servidor la clasificó como Mujer, 36-50. Ni la ficha ni la
    base se enteraron.

## Qué hace este módulo

Nada con disco ni con modelos: lleva la lista de fichas que esperan algo, y
en cada fotograma pregunta si ya llegó. Quien lo usa decide qué escribir. Así
la lógica de la espera se prueba sola, sin levantar el pipeline.

Cada espera recuerda su CÁMARA y solo se resuelve mientras se procesa esa
cámara: la demografía y el seguimiento son por cámara, y resolver un track
de la cámara A con el clasificador de la B daría un resultado ajeno.

Caduca a los `TTL_S` segundos: si en cinco minutos no llegó lo que faltaba,
la persona ya se fue y la ficha se queda como estaba.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Cinco minutos. Más que de sobra para que converja el género (se observó a
#: los 34 s) o llegue la identidad, y poco para que la lista crezca: en un
#: pasillo pasan ~200 personas por hora.
TTL_S = 300.0

#: Cota de memoria por si el TTL no bastara (reloj detenido, ráfaga enorme).
MAX_ESPERAS = 5000

_GENEROS = ('Hombre', 'Mujer')


class EsperaDemografia:
    """Fichas cerradas a medias, esperando identidad o género."""

    def __init__(self, ttl_s: float = TTL_S) -> None:
        self.ttl_s = float(ttl_s)
        #: tid -> ficha que YA tiene género y espera la identidad.
        self._identidad: Dict[Any, Dict[str, Any]] = {}
        #: tid -> ficha que YA tiene identidad y espera el género.
        self._genero: Dict[Any, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._identidad) + len(self._genero)

    # ── Altas ────────────────────────────────────────────────────────────
    def esperar_identidad(self, tid: Any, stem: str, camara: str,
                          genero: str, edad: Optional[str],
                          ahora: Optional[float] = None) -> None:
        """Clasificada sin identidad: esperar a que la reconozcan."""
        if len(self) >= MAX_ESPERAS:
            return
        self._identidad[tid] = {
            'stem': stem, 'camara': str(camara), 'genero': genero,
            'edad': edad, 't': float(ahora if ahora is not None
                                     else time.time())}

    def esperar_genero(self, tid: Any, stem: str, camara: str,
                       person_uuid: str,
                       ahora: Optional[float] = None) -> None:
        """Reconocida sin género: esperar a que el género converja."""
        if len(self) >= MAX_ESPERAS:
            return
        self._genero[tid] = {
            'stem': stem, 'camara': str(camara), 'person_uuid': person_uuid,
            't': float(ahora if ahora is not None else time.time())}

    # ── Revisión, una vez por fotograma ─────────────────────────────────
    def revisar(self, camara: str,
                identidad_de: Callable[[Any], Optional[str]],
                demografia_de: Callable[[Any], Optional[Dict[str, Any]]],
                ahora: Optional[float] = None
                ) -> Tuple[List[Tuple], List[Tuple]]:
        """Devuelve lo que se resolvió en ESTA cámara y lo olvida.

        - `identidades`: `(tid, stem, person_uuid, genero, edad)` de fichas
          que tenían género y acaban de recibir identidad;
        - `generos`: `(tid, stem, person_uuid, demo)` de fichas que tenían
          identidad y acaban de recibir género. `demo` es el dict del
          clasificador, con su confianza y su origen.

        Nunca lanza: una consulta que falla cuenta como «todavía no».
        """
        t = float(ahora if ahora is not None else time.time())
        camara = str(camara)
        for pendientes in (self._identidad, self._genero):
            for tid in [k for k, v in pendientes.items()
                        if t - v['t'] > self.ttl_s]:
                del pendientes[tid]

        identidades: List[Tuple] = []
        for tid, v in list(self._identidad.items()):
            if v['camara'] != camara:
                continue
            try:
                uuid = identidad_de(tid)
            except Exception:                               # noqa: BLE001
                uuid = None
            if uuid:
                identidades.append((tid, v['stem'], uuid, v['genero'],
                                    v['edad']))
                del self._identidad[tid]

        generos: List[Tuple] = []
        for tid, v in list(self._genero.items()):
            if v['camara'] != camara:
                continue
            try:
                demo = demografia_de(tid)
            except Exception:                               # noqa: BLE001
                demo = None
            if demo and demo.get('gender') in _GENEROS:
                generos.append((tid, v['stem'], v['person_uuid'], demo))
                del self._genero[tid]
        return identidades, generos
