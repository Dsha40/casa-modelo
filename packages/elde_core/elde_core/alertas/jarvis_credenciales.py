"""
Comprobacion de las credenciales de Jarvis365 al arrancar un cliente.

## Por que existe

`amazonas` y `managers` no tenian `jarvis_url`, `jarvis_email` ni
`jarvis_password` en su `.env`. Construian `Jarvis_api(None, None, None)`, el
login fallaba con un «Protocol "" is unknown» perdido entre los `print` del
arranque, y a partir de ahi todo seguia como si nada: la ventana abria, el
interruptor «Enviar a Jarvis» se veia marcado y verde, y no salia ni una
novedad. Un fallo de configuracion disfrazado de funcionamiento normal.

Esto lo hace visible: si falta algo, el interruptor del pie se apaga, se
deshabilita y su tooltip dice exactamente que variable falta y donde ponerla.
Nadie tiene que leer la consola para descubrirlo.

    from elde_core.alertas.jarvis_credenciales import comprobar
    comprobar(footer_bar.chk_envio_jarvis, cliente='amazonas')
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Las tres que `Jarvis_api` necesita para abrir sesion. Sin cualquiera de
# ellas no hay sesion, y sin sesion `enviar_novedad_async` se rinde.
VARIABLES = ('jarvis_url', 'jarvis_email', 'jarvis_password')


def faltantes() -> list[str]:
    """Las variables de Jarvis que no estan puestas (o estan vacias)."""
    return [v for v in VARIABLES if not (os.getenv(v) or '').strip()]


def comprobar(casilla=None, cliente: str = '') -> bool:
    """True si se puede enviar a Jarvis. Si no, apaga y explica la casilla.

    `casilla`: el QCheckBox «Enviar a Jarvis» del pie. Es opcional para que
    esto se pueda usar tambien desde una prueba o un script sin UI.
    """
    ausentes = faltantes()
    if not ausentes:
        return True

    lista = ', '.join(ausentes)
    donde = f'clients/{cliente}/.env' if cliente else 'el .env del cliente'
    logger.warning(
        'Jarvis DESACTIVADO: faltan %s en %s. No se enviara ninguna novedad.',
        lista, donde)
    print(f'[jarvis] envío DESACTIVADO: faltan {lista} en {donde}')

    if casilla is not None:
        try:
            casilla.setChecked(False)
            casilla.setEnabled(False)
            casilla.setToolTip(
                'Envío a Jarvis no disponible.\n'
                f'Faltan estas variables en {donde}:\n  {lista}\n'
                'Añádelas y vuelve a abrir el cliente.')
        except Exception as exc:      # una UI rota no puede tumbar el arranque
            logger.debug('no se pudo marcar la casilla de Jarvis: %s', exc)
    return False
