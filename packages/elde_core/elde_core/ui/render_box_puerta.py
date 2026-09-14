"""Cableado de la puerta direccional en el recuadro, para los cuatro clientes.

## Por qué un mixin y no cuatro copias

Los cuatro `render_box.py` son copias que llevan tiempo divergiendo: cambian
los nombres de los métodos, el menú y hasta qué botones existen. Cablear la
puerta a mano en cada uno significaría cuatro sitios donde arreglar el mismo
fallo, y la historia del repo dice cómo acaba eso: el 8-ago-2026 hubo que
portar el mismo `destroy_loop` a los cuatro `render_box` porque el `stop` del
recuadro impedía relanzar la ventana, y el bug estaba cuatro veces.

Aquí está una sola vez. Cada cliente aporta lo poco que es suyo —dónde va el
botón en su barra y qué modo lee la puerta— y hereda lo demás.

## Qué NO hace

No toca la flecha ni la línea de cruce. Solo las **apaga** cuando existen y
se enciende la puerta, porque las tres responden a la misma pregunta y el
servidor admite una. Se comprueban con `getattr`: tres de los cuatro clientes
no tienen flecha ni línea, y llamarlas a ciegas les rompería el arranque.

## El dato que viaja

La puerta produce el MISMO umbral que la línea de cruce —dos puntos, DENTRO a
la derecha de P1→P2— así que viaja por el mismo campo (`linea_cruce`) y la lee
el mismo detector del servidor. `payload_puerta()` arma las tres claves; el
tercero (`linea_cruce_estilo`) no cambia la medida, solo deja escrito en el
registro del servidor qué tenía delante el operador.

## Requisitos del recuadro que lo herede

  - `self.imagen_label`: un `Interactive_imageLabel` del núcleo.
  - `self._save_all(clave, valor)`: la persistencia por cámara.
  - `self.modo_inferencia` (opcional): si existe, el tooltip dice si el modo
    activo lee la puerta o la ignora.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton

#: El único modo del servidor que lee el umbral. Un cliente cuyo selector no
#: lo ofrezca puede dibujar y guardar la puerta, pero el servidor no la mirará
#: mientras corra otro modo; el tooltip lo dice en vez de fingir.
MODO_QUE_LEE_LA_PUERTA = "PermanenciaArea"

#: Estilo del botón, calcado del de la flecha y la línea del cliente
#: detecciones para que los tres se lean como una familia. Marcado en verde
#: azulado: ni el verde de la flecha ni el azul de la línea.
_ESTILO_BOTON = (
    "QToolButton{color:#fff;font-weight:bold;padding:2px 12px;"
    "border:1px solid #555;border-radius:4px;background:#333;}"
    "QToolButton:hover{background:#3d6fb0;}"
    "QToolButton:checked{background:#00897b;border-color:#26a69a;}")

_TOOLTIP = (
    "Puerta direccional: dibuja la línea SOBRE el umbral, de lado a lado\n"
    "del vano. Encima aparece una flecha DOBLE que la atraviesa: la\n"
    "cabeza verde marca por dónde se ENTRA y la roja por dónde se SALE.\n"
    "¿Al revés? Doble clic sobre ella (o menú ROI → Invertir sentido)\n"
    "intercambia entrada y salida sin moverla del sitio. Arrastra los\n"
    "extremos para ajustarla al vano; el cuerpo o el rombo central la\n"
    "mueven entera.")


class PuertaDireccionalMixin:
    """Estado, botón, menú, persistencia y payload de la puerta."""

    #: ¿El selector de modos de ESTE cliente ofrece el modo que lee el
    #: umbral? Hoy los cuatro lo ofrecen —a tienda y perimetrales se les
    #: añadió el 7-sep-2026, junto con este botón y por su causa: un
    #: utensilio que se dibuja pero que ningún modo mira es un adorno, y el
    #: operador no tiene cómo enterarse de que no sirve—, así que la rama
    #: `False` no la usa nadie ahora mismo.
    #:
    #: Se conserva porque la alternativa es peor: un quinto cliente que
    #: herede el mixin sin ese modo enseñaría un botón mudo. Poniéndolo a
    #: `False`, el tooltip lo dice y la puerta se sigue pudiendo calibrar.
    PUERTA_TIENE_MODO_QUE_LA_LEE = True

    # ── Estado ───────────────────────────────────────────────────────────
    def init_puerta(self, puntos: Any = None, activa: bool = False) -> None:
        """Fija el estado. Se llama en el `__init__`, ANTES del label.

        Sanea de paso el caso de unos ajustes que traigan la puerta y otro
        utensilio encendidos a la vez (editados a mano, o guardados por una
        versión anterior): manda la puerta.
        """
        self.puerta_direccional = puntos
        self.puerta_boolean = bool(activa)
        if self.puerta_boolean:
            for otro in ('linea_cruce_boolean', 'flecha_boolean'):
                if getattr(self, otro, False):
                    setattr(self, otro, False)

    def kwargs_puerta(self) -> Dict[str, Any]:
        """Lo que hay que pasarle al `Interactive_imageLabel` al crearlo."""
        return {
            'puerta_direccional': self.puerta_direccional,
            'puerta_direccional_activate': self.puerta_boolean,
        }

    def conectar_puerta(self, label) -> None:
        """Engancha la señal del label a la persistencia. Tras crearlo."""
        label.puerta_change.connect(self.save_puerta)

    # ── Botón ────────────────────────────────────────────────────────────
    def crear_boton_puerta(self) -> QToolButton:
        """El botón de la barra. El cliente decide dónde lo coloca.

        Botón SUELTO y no una entrada de menú por la misma razón que la
        flecha: la puerta se dibuja mirando el vídeo, y escondida tras dos
        clics no se encuentra.
        """
        boton = QToolButton()
        # '⇅' (U+21C5): la doble flecha partida existe en las fuentes de
        # Windows. La de cuerpo único ('⇕') sale como un cuadro vacío.
        boton.setText("⇅ Puerta")
        boton.setObjectName("btn-bar")
        boton.setCursor(Qt.PointingHandCursor)
        boton.setCheckable(True)
        boton.setChecked(self.puerta_boolean)
        boton.setFixedHeight(30)
        boton.setStyleSheet(_ESTILO_BOTON)
        boton.clicked.connect(lambda marcado: self.toggle_puerta(marcado))
        self._btn_puerta = boton
        self.actualizar_tooltip_puerta()
        return boton

    def actualizar_tooltip_puerta(self) -> None:
        """Dice para qué sirve y si el modo activo la lee de verdad."""
        boton = getattr(self, '_btn_puerta', None)
        if boton is None:
            return
        modo = str(getattr(self, 'modo_inferencia', '') or '').strip()
        if not self.PUERTA_TIENE_MODO_QUE_LA_LEE:
            estado = ("\n\nOJO: este cliente no ofrece "
                      f"{MODO_QUE_LEE_LA_PUERTA} en el pie,\nque es el único "
                      "modo que lee el umbral. Puedes dejarla\ncalibrada y "
                      "queda guardada, pero desde aquí no generará\nalertas "
                      "de entrada ni de salida.")
        elif modo == MODO_QUE_LEE_LA_PUERTA:
            estado = ("\n\nModo activo: PermanenciaArea — SÍ la usa.\n"
                      "Dibujarla activa las alertas por cruce: «Persona\n"
                      "entrando al área» / «Persona saliendo del área», con\n"
                      "hora y foto. Borrarla vuelve al área clásica.")
        elif modo:
            estado = (f"\n\nModo activo: {modo} — este modo la ignora.\n"
                      f"Solo la lee {MODO_QUE_LEE_LA_PUERTA}.")
        else:
            # NO se dice «sin conexión»: seria mentira en tres de los cuatro
            # clientes. Solo Detecciones tiene `set_modo_inferencia`, asi que
            # en los otros `modo` esta vacio SIEMPRE, tambien con el servidor
            # conectado y corriendo PermanenciaArea. Un cartel que afirma un
            # estado que no ha comprobado manda a revisar la red a quien lo
            # unico que tiene que hacer es dibujar la puerta.
            estado = (f"\n\nPara que el servidor la lea, la cámara tiene que "
                      f"correr\n{MODO_QUE_LEE_LA_PUERTA} (se elige en el "
                      "pie de la ventana).")
        boton.setToolTip(_TOOLTIP + estado)

    # ── Interruptor ──────────────────────────────────────────────────────
    def toggle_puerta(self, estado: Optional[bool] = None) -> None:
        """Enciende/apaga la puerta y apaga los otros utensilios.

        Los `getattr` no son adorno: tres de los cuatro clientes no tienen
        flecha ni línea, y llamar a sus toggles a ciegas les lanzaría un
        `AttributeError` al pulsar el botón.
        """
        self.puerta_boolean = (not self.puerta_boolean if estado is None
                               else bool(estado))
        if self.puerta_boolean:
            for bandera, apagar in (('flecha_boolean', '_toggle_flecha'),
                                    ('linea_cruce_boolean', '_toggle_linea')):
                if getattr(self, bandera, False):
                    metodo = getattr(self, apagar, None)
                    if callable(metodo):
                        metodo(False)
        self.imagen_label.toggle_puerta(self.puerta_boolean)
        boton = getattr(self, '_btn_puerta', None)
        if boton is not None and boton.isChecked() != self.puerta_boolean:
            boton.setChecked(self.puerta_boolean)
        self._save_all("puerta_boolean", self.puerta_boolean)

    def invertir_puerta(self) -> None:
        """Intercambia ENTRADA y SALIDA sin moverla del umbral."""
        self.imagen_label.invertir_puerta()

    # ── Persistencia ─────────────────────────────────────────────────────
    def save_puerta(self, puntos, activa) -> None:
        """Clave PROPIA en los ajustes, no la de la línea.

        Son dos calibraciones distintas de la misma cámara: volver a la otra
        herramienta debe encontrar lo que se dejó, no lo último que se movió.
        """
        self.puerta_direccional = puntos
        self.puerta_boolean = bool(activa)
        self._save_all("puerta_direccional", puntos)
        self._save_all("puerta_boolean", self.puerta_boolean)

    # ── Lo que viaja al servidor ─────────────────────────────────────────
    def estilo_cruce(self) -> str:
        """Qué utensilio dibujó el umbral: `'puerta'`, `'linea'` o vacío."""
        if self.puerta_boolean:
            return "puerta"
        if getattr(self, 'linea_cruce_boolean', False):
            return "linea"
        return ""

    def payload_puerta(self, ancho: int, alto: int) -> Dict[str, Any]:
        """Las tres claves del umbral, listas para el `data` del frame.

        `get_cruce_coordinates` devuelve el umbral que rija —de la puerta o
        de la línea, que producen el mismo dato para el mismo campo— así que
        el recuadro no tiene que preguntar cuál está encendida.
        """
        umbral: List[Any] = self.imagen_label.get_cruce_coordinates(ancho,
                                                                    alto)
        return {
            "linea_cruce": umbral,
            "linea_cruce_activate": bool(umbral),
            "linea_cruce_estilo": self.estilo_cruce(),
        }
