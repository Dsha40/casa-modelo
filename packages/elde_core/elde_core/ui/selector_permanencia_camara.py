"""
Segundos que un objeto debe estar en el área antes de alertar, POR CÁMARA.

## Para que sirve

El umbral vivia SOLO en el servidor (`vigilante_amazonas/config.py`), asi que
afinarlo obligaba a editar un archivo y REINICIAR el servidor — con todas las
camaras de todos los locales cayendose de paso. Y el valor bueno no es uno:
un porton concurrido y un patio trasero necesitan tiempos distintos, igual que
necesitan locales distintos.

Este control lo pone donde el operador puede usarlo: la barra de SU recuadro,
al lado del select «Local», y el valor viaja al servidor en cada frame (misma
via que `establecimiento`). Cambiarlo surte efecto en el siguiente frame, sin
reiniciar nada.

## Que clase usa cada numero

El de personas rige `persona`. El otro rige TODO lo demas: `moto`, `carro`,
`camioneta`, `camion` y tambien `objeto` (mochila/bolso/maleta abandonados).
`personal_seguridad` no usa ninguno de los dos — ver abajo.

## Por que DOS numeros y no uno

Es la distincion que el operador ya venia afinando a mano en el servidor
(45 s personas / 60 s vehiculos, 18-ago-2026): una persona parada medio
minuto en el perimetro es dato antes que un vehiculo, que necesita mas tiempo
para distinguir «se estaciono» de «esta maniobrando».

## Lo que este control NO toca

El **personal de seguridad** sigue alertando INMEDIATO pase lo que pase. No es
un olvido: su presencia es dato operativo que no admite espera, y dejar que un
umbral por camara lo silenciara seria un modo de fallo silencioso. Quien
quiera cambiar eso, que lo haga explicito en el servidor.

## Se RECUERDA entre sesiones

Al reves que el select «Local» (que arranca en blanco a proposito para que
nadie alerte al local de ayer), esto es un ajuste de la CAMARA — como la
lectura de placas. El recuadro lo persiste con su `callback_save_data`.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QWidget

# La barra del recuadro es oscura en los cuatro clientes.
_ESTILO_SPIN = (
    "QSpinBox{color:#fff;background:#333;border:1px solid #555;"
    "border-radius:4px;padding:1px 2px;font-size:11px;}"
    "QSpinBox:hover{border-color:#3d6fb0;}"
    "QSpinBox::up-button,QSpinBox::down-button{width:12px;}")
_ESTILO_ETIQUETA = "color:#bbb;font-size:11px;"

#: Tope de 1 hora: mas alla no es un umbral, es apagar la alerta — y para eso
#: esta el 0 del otro extremo (alertar de inmediato), que si es explicito.
MAX_SEGUNDOS = 3600

#: Lo que trae el servidor si la camara no manda nada (config.py del vigilante).
POR_DEFECTO_PERSONAS = 45
POR_DEFECTO_VEHICULOS = 60


class SelectorPermanenciaCamara(QWidget):
    """Dos contadores de segundos (personas / vehiculos) de UNA camara.

    Emite `permanencia_cambiada(personas, vehiculos)` cada vez que el
    operador mueve cualquiera de los dos.
    """

    permanencia_cambiada = Signal(int, int)

    def __init__(self, personas: int | None = None,
                 vehiculos: int | None = None, parent=None) -> None:
        super().__init__(parent)
        raiz = QHBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(3)

        etiqueta = QLabel("⏱")
        etiqueta.setStyleSheet(_ESTILO_ETIQUETA)
        etiqueta.setToolTip(
            "Segundos que el objeto debe permanecer DENTRO del área antes de\n"
            "que se envíe la alerta. Un cruce más corto no genera ni llegada\n"
            "ni salida.\n\n"
            "El personal de seguridad alerta siempre de inmediato.")
        raiz.addWidget(etiqueta)

        self._personas = self._crear_spin(
            personas, POR_DEFECTO_PERSONAS,
            "Personas: segundos dentro del área antes de alertar")
        self._vehiculos = self._crear_spin(
            vehiculos, POR_DEFECTO_VEHICULOS,
            "Vehículos (carro, camioneta, moto, camión) y objetos:\n"
            "segundos dentro del área antes de alertar.\n\n"
            "Este contador rige TODO lo que no sea una persona.")
        raiz.addWidget(QLabel())          # separador fino entre los dos pares
        for icono, spin, ayuda in (("👤", self._personas, "personas"),
                                   ("🚗", self._vehiculos, "vehículos")):
            lbl = QLabel(icono)
            lbl.setStyleSheet(_ESTILO_ETIQUETA)
            lbl.setToolTip(f"Segundos para {ayuda}")
            raiz.addWidget(lbl)
            raiz.addWidget(spin)

    def _crear_spin(self, valor: int | None, defecto: int,
                    ayuda: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, MAX_SEGUNDOS)
        spin.setSuffix(" s")
        spin.setFixedWidth(62)
        spin.setStyleSheet(_ESTILO_SPIN)
        spin.setToolTip(ayuda)
        # `None` = el recuadro no traia valor guardado: se parte del mismo
        # numero que usa el servidor, para que el control no MIENTA sobre lo
        # que esta pasando antes de que el operador lo toque.
        spin.setValue(int(valor) if valor is not None else defecto)
        spin.valueChanged.connect(self._emitir)
        return spin

    def _emitir(self) -> None:
        self.permanencia_cambiada.emit(self._personas.value(),
                                       self._vehiculos.value())

    # ── Lectura desde el recuadro ────────────────────────────────────────
    def personas(self) -> int:
        return int(self._personas.value())

    def vehiculos(self) -> int:
        return int(self._vehiculos.value())
