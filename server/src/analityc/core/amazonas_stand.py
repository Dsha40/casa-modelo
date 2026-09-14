"""Pipeline `AmazonasStand` — stand de degustación (modo propio).

## Por qué es un pipeline aparte y no un flag dentro de otro

La primera versión enganchaba el módulo del stand dentro de
`tiendas.py`, detrás de `habilitado` en la configuración. Eso
tenía un problema que se vio en cuanto se activó: **el cliente de Tienda usa
ese mismo pipeline**, así que al encender el flag empezó a abrir sesiones y a
contar visitantes de un stand que no está midiendo. Un interruptor global
para algo que es de una sola cámara.

Como modo propio, la elección es explícita: el operador selecciona
`AmazonasStand` en el cliente y solo entonces se mide. Tienda y Perimetrales
no se enteran de que esto existe.

## Qué añade sobre `TiendasProcess`

Hereda TODO el pipeline de personas —detección, BoT-SORT, Re-ID, demografía,
capturas, heatmap propio— y le suma, después de construir el `metadata`:

  - identidad única con reglas anti-doble-conteo (`stand_identidad`);
  - exclusión del chef (`stand_personal`);
  - máquina de estados del visitante (`stand_estados`);
  - degustación por pose + VLM (`stand_degustacion`);
  - mapa de calor de tres capas (`stand_heatmap`);
  - persistencia en SQLite y bloque en vivo para el cliente.

Hereda por subclase a propósito: `app.py` despacha muchas cosas por
`isinstance(processor, TiendasProcess)`, así que todo eso sigue funcionando
sin tocar una línea. Es el mismo patrón que usa `Estacionamiento` sobre
`VigilanteWS`.

## Zonas

Sigue exigiendo `config/stand_zonas.yaml` calibrado. Sin `zona_paso` y
`zona_stand` no cuenta nada y lo dice en el log: un polígono inventado da
cifras creíbles y falsas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from .tiendas import TiendasProcess

logger = logging.getLogger(__name__)

#: Nombre del modo tal como lo pide el cliente por WebSocket.
MODO = "AmazonasStand"


class AmazonasStand(TiendasProcess):
    """`TiendasProcess` + medición del stand de degustación."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._puente = None
        self._puente_intentado = False
        logger.info("AmazonasStand: pipeline del stand de degustación activo")

    # ── Bucle ───────────────────────────────────────────────────────────
    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False,
                      camera_id: Any = 1, heatmap_activate: bool = False
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """El frame de `TiendasProcess` + el bloque `stand` en el metadata."""
        imagen, metadata = super().process_frame(
            image, roi=roi, activate_roi=activate_roi, camera_id=camera_id,
            heatmap_activate=heatmap_activate)

        try:
            bloque = self._medir_stand(image, camera_id)
            if bloque:
                metadata['stand'] = bloque
        except Exception as exc:                            # noqa: BLE001
            # Nunca propaga: si la medición del stand falla, el cliente debe
            # seguir viendo vídeo y detecciones.
            logger.error("AmazonasStand: fallo midiendo el stand: %s", exc)

        return imagen, metadata

    def _medir_stand(self, image: np.ndarray, camera_id: Any):
        if not self._puente_intentado:
            self._puente_intentado = True
            try:
                from .stand_puente import PuenteStand
                from .stand_config import cargar
                alto, ancho = image.shape[:2]
                cfg = cargar()
                # En este modo NO se consulta `habilitado`: elegir el pipeline
                # ES la forma de habilitarlo. El flag del YAML existía para el
                # enganche global, que ya no está.
                self._puente = PuenteStand(ancho, alto, camera_id, cfg)
                if not self._puente.activo:
                    logger.warning(
                        "AmazonasStand: zonas sin calibrar; se procesa vídeo "
                        "pero NO se cuenta. Ejecuta tools/calibrar_stand.py")
            except Exception as exc:                        # noqa: BLE001
                logger.error("AmazonasStand: no se pudo iniciar el módulo: "
                             "%s", exc)
                self._puente = None

        if self._puente is None:
            return None
        return self._puente.procesar(
            image, self._tracks_de(camera_id),
            getattr(self, '_demographics', None))

    def _tracks_de(self, camera_id: Any) -> Dict[Any, Any]:
        """Los tracks de ESTA cámara, no los que quedan tras `_pop_state`.

        `TiendasProcess` multiplexa cámaras sobre un mismo objeto: al entrar
        en `process_frame` mueve `active_tracks` (entre otros) a `self`, y al
        salir **restaura los de antes**. Este método corre después, así que
        `self.active_tracks` era un diccionario vacío que no era de nadie: el
        stand recibía cero personas por muchas que hubiera delante, sin un
        solo error. Encontrado el 7-ago-2026 depurando PermanenciaArea, que
        tenía el mismo fallo; explica que `stand.db` siguiera vacía.

        Se lee `camera_states` a pelo: `_get_cam_state` crea y MIGRA el estado
        (repone `_zone_state` y construye un clasificador demográfico), que no
        es lo que quiere quien solo viene a consultar.
        """
        estado = (getattr(self, 'camera_states', None) or {}).get(camera_id)
        if isinstance(estado, dict):
            tracks = estado.get('active_tracks')
            if isinstance(tracks, dict):
                return tracks
        return getattr(self, 'active_tracks', {}) or {}

    # ── Cierre ──────────────────────────────────────────────────────────
    def cerrar_sesion_stand(self) -> None:
        """Cierra la jornada del stand. **Hoy no la llama nadie**, a propósito.

        La versión anterior de este docstring decía que la llamaba el servidor
        al desconectarse el cliente. No era cierto, y además no debe serlo:
        una reconexión por un corte de red cerraría la jornada y la siguiente
        abriría otra, partiendo el día en trozos que no se pueden sumar (una
        persona vista antes y después cuenta en los dos). Es exactamente el
        fallo de las cuatro sesiones colgadas que se arregló con
        `stand_almacen.sesion_abierta_de`, que reanuda la jornada del día.

        Queda como el gancho para cerrar la jornada **a propósito** —fin de la
        degustación— desde una herramienta o un endpoint. Mientras no exista
        ese disparador explícito, la sesión se reanuda sola cada día.
        """
        if self._puente is not None:
            self._puente.cerrar()


def crear(**kwargs: Any) -> AmazonasStand:
    """Fábrica, para que `app.py` no importe la clase directamente."""
    return AmazonasStand(**kwargs)
