"""Pruebas de la PUERTA DIRECCIONAL en el recuadro de este cliente.

El utensilio —el umbral en cian con la flecha doble ENTRADA/SALIDA
atravesandolo— vive en el nucleo (`elde_core.ui.puerta_direccional`) y su
cableado en `elde_core.ui.render_box_puerta`, asi que la geometria y el
pintado ya se prueban alli. Lo que se prueba AQUI es lo unico que el nucleo
no puede saber: que ESTE cliente lo monto bien.

Se instancia Qt de verdad en modo offscreen. Un widget que compila pero
revienta al pintarse no sirve de nada, y eso solo se ve construyendolo.

Lo que mas se protege:

  - **nace apagada** — quien no la encienda no puede notar ningun cambio;
  - **la cabeza verde cae del lado que el servidor llama DENTRO** — si el
    dibujo y la medida divergen, el operador calibra mirando una cosa y el
    servidor cuenta otra, y nadie se entera hasta que las entradas salen
    contadas como salidas;
  - **viaja por el campo de la LINEA** — es la misma geometria con otro
    rotulo, asi que la lee el detector de linea de siempre; mandarla por un
    campo propio habria obligado a estrenar un detector para medir lo mismo.

Este fichero es DELIBERADAMENTE el mismo en los cuatro clientes: lo que
comprueba es que los cuatro montaron identico un componente compartido, y
que uno se desvie es justo el fallo que busca.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))

from PySide6.QtCore import QObject, QPoint, Signal  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _SocketFalso(QObject):
    connected_signal = Signal(dict)
    disconnected_signal = Signal(dict)
    signal_inference = Signal(dict)

    def is_connected(self):
        return False


def _recuadro(**kw):
    from gui.components.render_box.render_box import Render_box
    guardado = {}
    caja = Render_box(index=0, socket_services=_SocketFalso(),
                      callback_save_data=lambda i, k, v:
                      guardado.__setitem__(k, v),
                      **kw)
    return caja, guardado


def _pintar(caja):
    """Fuerza un pintado real del visor con la puerta encima."""
    caja.imagen_label.resize(640, 480)
    caja.imagen_label.current_pixmap = QPixmap(640, 480)
    caja.imagen_label.render(QPixmap(640, 480))


# ── Nace apagada ─────────────────────────────────────────────────────────
def test_nace_apagada_y_no_manda_nada():
    """Quien no la encienda no puede notar que existe."""
    caja, _ = _recuadro()
    assert caja.puerta_boolean is False
    assert caja.imagen_label.puerta.activa is False
    datos = caja.payload_puerta(1920, 1080)
    assert datos['linea_cruce'] == []
    assert datos['linea_cruce_activate'] is False
    assert datos['linea_cruce_estilo'] == ''


# ── El boton ─────────────────────────────────────────────────────────────
def test_el_boton_esta_en_la_barra_y_es_pulsable():
    """Suelto y siempre visible: la puerta se dibuja mirando el video.

    Escondida tras dos clics de menu no se encuentra, que es lo que ya paso
    con la flecha antes de sacarla a la barra.
    """
    caja, _ = _recuadro()
    boton = caja._btn_puerta
    assert caja.bar_options.layout().indexOf(boton) >= 0
    assert boton.isCheckable() and boton.isEnabled()
    assert 'Puerta' in boton.text()


def test_el_boton_no_se_deshabilita_nunca():
    """El modo llega del pie DESPUES de conectar.

    Si el boton dependiera de eso, quien abriera el cliente sin conectar no
    podria ni calibrar la puerta.
    """
    caja, _ = _recuadro()
    assert caja._btn_puerta.isEnabled()


def test_el_tooltip_explica_las_dos_cabezas():
    caja, _ = _recuadro()
    ayuda = caja._btn_puerta.toolTip()
    assert 'ENTRA' in ayuda and 'SALE' in ayuda
    assert 'umbral' in ayuda


# ── Encender, guardar, restaurar ─────────────────────────────────────────
def test_pulsar_el_boton_la_enciende_y_la_guarda():
    caja, guardado = _recuadro()
    caja.toggle_puerta(True)
    assert caja.puerta_boolean is True
    assert caja.imagen_label.puerta.activa is True
    assert guardado.get('puerta_boolean') is True


def test_se_restaura_de_los_ajustes():
    """Guardar sin restaurar es el fallo clasico de este cableado."""
    caja, _ = _recuadro(puerta_direccional=[[100, 200], [900, 200]],
                        puerta_boolean=True)
    assert caja._btn_puerta.isChecked() is True
    assert caja.imagen_label.puerta.a_lista() == [[100, 200], [900, 200]]


def test_la_restauracion_esta_cableada_en_windows_main():
    ventana = (RAIZ / 'src' / 'gui' / 'windows_main.py').read_text(
        encoding='utf-8')
    assert 'cfg.get("puerta_direccional"' in ventana
    assert 'cfg.get("puerta_boolean"' in ventana


# ── Lo que viaja al servidor ─────────────────────────────────────────────
def test_viaja_en_pixeles_por_el_campo_de_la_linea():
    """Misma geometria que la linea = mismo campo = mismo detector."""
    caja, _ = _recuadro(puerta_direccional=[[100, 500], [900, 500]],
                        puerta_boolean=True)
    datos = caja.payload_puerta(1920, 1080)
    assert datos['linea_cruce'] == [[192, 540], [1728, 540]]
    assert datos['linea_cruce_activate'] is True
    assert datos['linea_cruce_estilo'] == 'puerta'


def test_demasiado_corta_no_manda_nada():
    """Dos puntos casi pegados dan una perpendicular aleatoria.

    Vale mas el silencio: el servidor cae entonces al area clasica en vez de
    clasificar pasos con una direccion inventada.
    """
    caja, _ = _recuadro(puerta_direccional=[[500, 500], [510, 500]],
                        puerta_boolean=True)
    assert caja.payload_puerta(1920, 1080)['linea_cruce'] == []


def test_apagarla_vacia_el_payload():
    caja, _ = _recuadro(puerta_direccional=[[100, 500], [900, 500]],
                        puerta_boolean=True)
    caja.toggle_puerta(False)
    assert caja.payload_puerta(1920, 1080)['linea_cruce'] == []


# ── Que lo pintado y lo medido no diverjan ───────────────────────────────
def test_la_cabeza_verde_cae_del_lado_que_el_servidor_llama_DENTRO():
    """LA prueba que importa.

    El cliente pinta ENTRADA hacia `normal_dentro()`; el servidor cuenta como
    entrada a quien pasa a la derecha de P1->P2. Es la MISMA cuenta. Si un
    dia divergen, el operador calibraria mirando una cosa y el servidor
    contaria otra, y el sintoma —entradas contadas como salidas— tarda dias
    en aparecer y no apunta a ningun sitio.
    """
    caja, _ = _recuadro(puerta_direccional=[[100, 500], [900, 500]],
                        puerta_boolean=True)
    puerta = caja.imagen_label.puerta
    normal = puerta.normal_dentro()
    assert normal is not None
    # P1->P2 va hacia +x, luego DENTRO (su derecha, con la `y` de imagen
    # hacia abajo) queda hacia +y: la mitad de ABAJO del encuadre.
    assert abs(normal[0]) < 1e-6 and abs(normal[1] - 1.0) < 1e-6
    assert puerta.lado(QPoint(500, 700)) == 1, 'abajo es DENTRO'
    assert puerta.lado(QPoint(500, 300)) == -1, 'arriba es FUERA'


def test_invertir_intercambia_entrada_y_salida_sin_moverla():
    """El fallo tipico no es una puerta torcida: es una al reves."""
    caja, _ = _recuadro(puerta_direccional=[[100, 500], [900, 500]],
                        puerta_boolean=True)
    antes = caja.imagen_label.puerta.a_lista()
    lado_antes = caja.imagen_label.puerta.lado(QPoint(500, 700))
    caja.invertir_puerta()
    assert caja.imagen_label.puerta.a_lista() == antes[::-1]
    assert caja.imagen_label.puerta.lado(QPoint(500, 700)) == -lado_antes


# ── Que no tumbe el visor ────────────────────────────────────────────────
def test_pintarla_no_revienta():
    """Un fallo pintando no puede tumbar el visor de video."""
    caja, _ = _recuadro(puerta_direccional=[[100, 500], [900, 500]],
                        puerta_boolean=True)
    _pintar(caja)


def test_pintarla_pegada_al_borde_no_revienta():
    """La puerta al borde del encuadre es el caso MAS comun, no el raro."""
    caja, _ = _recuadro(puerta_direccional=[[0, 0], [0, 990]],
                        puerta_boolean=True)
    _pintar(caja)


def test_pintarla_demasiado_corta_no_revienta():
    """El estado intermedio de CADA calibracion: se dibuja arrastrando."""
    caja, _ = _recuadro(puerta_direccional=[[500, 500], [505, 500]],
                        puerta_boolean=True)
    _pintar(caja)
