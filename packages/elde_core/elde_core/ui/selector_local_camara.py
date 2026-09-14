"""
Selector de establecimiento POR CÁMARA, en la barra de cada recuadro.

## Para que sirve

Una misma ventana puede tener 4 camaras de 4 locales distintos corriendo la
MISMA inferencia. El selector del pie es uno solo para toda la ventana, asi que
no puede resolver eso: hace falta que cada recuadro diga a que establecimiento
van SUS alertas.

Nacio en `perimetrales` (1-ago-2026) y aqui se comparte con los cuatro
clientes, que es donde el operador lo necesita igual.

## Dos decisiones que parecen detalles y no lo son

**El local NO se recuerda entre sesiones** (3-ago-2026, peticion del operador):
cada vez que el cliente abre, el selector arranca en blanco y hay que elegirlo.
Es lo contrario de lo que se hace con casi todo lo demas, y a proposito: dejar
camaras apuntando al local de ayer es como se mandan alertas al sitio
equivocado sin que nadie lo note. Quien lo guarde debe hacerlo fuera de aqui.

**Un local guardado que ya no esta en la lista se conserva como opcion**, con
la etiqueta «(no está en la lista)». Quitarlo en silencio dejaria la camara
apuntando a otro sitio sin avisar.

## Que hace la primera opcion

Depende del cliente, y por eso es un parametro:

  - `perimetrales`: no hay selector global (se elimino), asi que una camara sin
    local **no envia**. Su texto debe decir exactamente eso.
  - `tienda`, `amazonas`, `managers`: si hay selector en el pie, que actua de
    respaldo. Ahi el texto es «el del pie».

Decirlo mal es peor que no decir nada: el operador creeria que sus alertas van
a alguna parte.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox

# Se ve igual en los cuatro clientes: la barra del recuadro es oscura en todos.
_ESTILO = (
    "QComboBox{color:#fff;background:#333;border:1px solid #555;"
    "border-radius:4px;padding:1px 8px;font-size:11px;}"
    "QComboBox:hover{border-color:#3d6fb0;}"
    "QComboBox::drop-down{border:none;width:16px;}"
    "QComboBox QAbstractItemView{background:#2b2b2b;color:#eee;"
    "border:1px solid #555;selection-background-color:#3d6fb0;}")

# Textos de la primera opcion (sin local elegido).
SIN_LOCAL_NO_ENVIA = "Local: sin elegir"
SIN_LOCAL_USA_EL_PIE = "Local: el del pie"


class SelectorLocalCamara(QComboBox):
    """Desplegable de establecimiento de UNA camara."""

    # Nombre elegido (o '' al volver a la primera opcion).
    local_cambiado = Signal(str)

    def __init__(self, api_jarvis=None, establecimiento: str = '',
                 texto_vacio: str = SIN_LOCAL_USA_EL_PIE,
                 indice_camara=None, parent=None) -> None:
        super().__init__(parent)
        self._api = api_jarvis
        self._texto_vacio = texto_vacio
        self._indice = indice_camara
        self.establecimiento = str(establecimiento or '').strip()

        self.setObjectName('selector-local')
        self.setFixedHeight(24)
        self.setMinimumWidth(130)
        self.setMaximumWidth(190)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(_ESTILO)

        self.repoblar()
        # `activated` y no `currentIndexChanged`: solo interesa lo que ELIGE
        # el operador, no los cambios que provoca repoblar.
        self.activated.connect(self._al_elegir)

        # La lista de Jarvis llega DESPUES del login (async): al construirse
        # esto casi siempre esta vacio y hay que repoblar cuando responda.
        try:
            self._api.establishments_loaded.connect(lambda *_: self.repoblar())
        except Exception:
            pass

    # ── Lista ────────────────────────────────────────────────────────

    def _nombres(self) -> list[str]:
        try:
            return [str(e.get('name', '')).strip()
                    for e in (getattr(self._api, 'list_of_establishments', None)
                              or [])
                    if isinstance(e, dict) and e.get('name')]
        except Exception:
            return []

    def repoblar(self) -> None:
        """Rellena las opciones dejando seleccionada la de esta camara.

        Idempotente: se llama al construir y cada vez que Jarvis carga la lista.
        """
        self.blockSignals(True)
        self.clear()
        self.addItem(self._texto_vacio, '')
        nombres = self._nombres()
        for nombre in nombres:
            self.addItem(nombre, nombre)
        if self.establecimiento and self.establecimiento not in nombres:
            self.addItem(f'{self.establecimiento} (no está en la lista)',
                         self.establecimiento)
        indice = self.findData(self.establecimiento)
        self.setCurrentIndex(indice if indice != -1 else 0)
        self.blockSignals(False)
        self._refrescar_tooltip()

    # ── Seleccion ────────────────────────────────────────────────────

    def _al_elegir(self, indice: int) -> None:
        """El operador eligio una opcion.

        NO se repuebla aqui. `activated` se emite mientras el popup se esta
        cerrando, asi que llamar a `repoblar()` —que hace `clear()`— destruye
        las filas que la vista todavia esta usando: el desplegable se
        reconstruye bajo el dedo del operador, parece que «se recarga» y la
        eleccion no llega a cuajar. Y no hace ninguna falta: el combo YA esta
        mostrando lo que se acaba de elegir.
        """
        self._aplicar(self.itemData(indice) or '')

    def fijar(self, nombre: str) -> None:
        """Cambia el local desde FUERA del combo (restaurar, pruebas, codigo).

        A diferencia de `_al_elegir`, aqui la vista no esta sincronizada y hay
        que ponerla al dia — reconstruyendo solo si el nombre no esta en la
        lista, que es el unico caso en que no basta con mover el indice.
        """
        nombre = str(nombre or '').strip()
        self._aplicar(nombre)
        indice = self.findData(nombre)
        if indice == -1:
            self.repoblar()          # hay que crearle su opcion
        elif self.currentIndex() != indice:
            self.blockSignals(True)
            self.setCurrentIndex(indice)
            self.blockSignals(False)

    def _aplicar(self, nombre: str) -> None:
        """Guarda el local, refresca la ayuda y avisa al recuadro."""
        self.establecimiento = str(nombre or '').strip()
        self._refrescar_tooltip()
        etiqueta = (f'[box {self._indice}]' if self._indice is not None
                    else '[camara]')
        print(f'{etiqueta} establecimiento de la camara: '
              f'{self.establecimiento or "(sin local)"}')
        self.local_cambiado.emit(self.establecimiento)

    def _refrescar_tooltip(self) -> None:
        if self.establecimiento:
            self.setToolTip(
                f'Establecimiento de ESTA cámara: {self.establecimiento}\n'
                'Sus alertas de Jarvis van ahí.')
            return
        respaldo = ('Ahora usa el seleccionado en el pie de la ventana.'
                    if self._texto_vacio == SIN_LOCAL_USA_EL_PIE else
                    'Sin local elegido NO se envían alertas a Jarvis.')
        self.setToolTip(
            'Establecimiento de ESTA cámara (a dónde van sus alertas de '
            f'Jarvis).\n{respaldo}')
