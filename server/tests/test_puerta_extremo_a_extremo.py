"""La PUERTA que dibuja el operador clasifica bien en el servidor.

## Qué prueba esto que no prueben las otras

`test_permanencia_linea.py` prueba el detector con una línea escrita a mano en
el propio fichero. Las pruebas de los clientes prueban que el recuadro monta
el utensilio y arma el payload. En medio queda el hueco que de verdad importa:

    lo que el operador VE al dibujar  ==  lo que el servidor CUENTA

El cliente pinta la cabeza verde ENTRADA hacia `normal_dentro()`. El servidor
cuenta como entrada a quien pasa a la derecha de P1→P2. Son la misma cuenta —
mientras nadie toque una de las dos. El día que diverjan, el operador
calibrará mirando una cosa y el servidor contará otra, y el síntoma (entradas
contadas como salidas) tarda días en aparecer y no apunta a ningún sitio.

Aquí se recorre la cadena entera con las clases REALES de los dos lados: se
dibuja la puerta con el widget del cliente, se saca su geometría en píxeles
como la saca el payload, y se le da EL MISMO dato al detector del servidor.

## Por qué no hace falta un detector nuevo

Porque el eje de la flecha doble ES la normal de la línea. La puerta y la
línea de cruce producen, punto por punto, la MISMA geometría; lo único que
cambia es cómo se rotula en pantalla. Eso es lo que fija
`test_los_dos_utensilios_producen_el_mismo_dato`: si un día alguien cambia el
pintado de la puerta y con él su geometría, esa prueba se pone roja antes de
que llegue a una cámara.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from src.analityc.core.permanencia_linea import (  # noqa: E402
    POR_LINEA, DetectorLinea)
from src.analityc.core.permanencia_puerta import ENTRADA, SALIDA  # noqa: E402

#: Tamaño del frame real que enviaría el cliente.
ANCHO, ALTO_FRAME = 1920, 1080
#: Alto de la persona en píxeles: fija la histéresis (0.15 alturas = 30 px).
ALTO = 200


def _puerta(puntos):
    """La puerta del CLIENTE, dibujada en el espacio 0..1000 del visor."""
    PySide6 = pytest.importorskip('PySide6')          # noqa: F841
    from elde_core.ui.puerta_direccional import PuertaDireccional
    return PuertaDireccional(puntos, activa=True)


def _obs(x, y, confirmado=True):
    """Una persona cuyo PIE está en `(x, y)`, en píxeles del frame."""
    return {'confirmado': confirmado,
            'box': [x - 30, y - ALTO, x + 30, y]}


def _caminar(detector, clave, recorrido, umbral, t0=0.0, dt=0.2):
    cruces = []
    for i, (x, y) in enumerate(recorrido):
        cruces.extend(detector.observar(t0 + i * dt,
                                        {clave: _obs(x, y)}, umbral))
    return cruces


# ── La puerta horizontal: se entra hacia ABAJO ───────────────────────────
#: Dibujada de izquierda a derecha, a media altura del visor. P1→P2 va hacia
#: +x, luego DENTRO —donde el cliente pinta la cabeza verde— queda hacia +y:
#: la mitad de ABAJO del encuadre.
HORIZONTAL = [[100, 500], [900, 500]]

#: En píxeles del frame eso es la recta y = 540. Bajar la cruza hacia DENTRO.
BAJANDO = [(960, 400), (960, 460), (960, 505), (960, 560), (960, 620),
           (960, 680)]
SUBIENDO = [(960, 680), (960, 620), (960, 560), (960, 505), (960, 460),
            (960, 400)]


def test_el_umbral_llega_al_servidor_en_pixeles_del_frame():
    """Lo que el payload manda es lo que el detector recibe."""
    puerta = _puerta(HORIZONTAL)
    umbral = puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
    assert umbral == [[192, 540], [1728, 540]]


def test_bajar_hacia_la_cabeza_verde_cuenta_como_ENTRADA():
    """El recorrido va hacia donde el operador ve pintado ENTRADA."""
    puerta = _puerta(HORIZONTAL)
    umbral = puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
    # El cliente pinta ENTRADA hacia la normal: comprobado aparte para que,
    # si esto cambia, se sepa que fue el DIBUJO y no el detector.
    assert puerta.normal_dentro()[1] > 0, 'la cabeza verde mira hacia abajo'

    cruces = _caminar(DetectorLinea(), 'p-A', BAJANDO, umbral)
    assert [c.tipo for c in cruces] == [ENTRADA]
    assert cruces[0].via == POR_LINEA


def test_subir_hacia_la_cabeza_roja_cuenta_como_SALIDA():
    puerta = _puerta(HORIZONTAL)
    umbral = puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
    cruces = _caminar(DetectorLinea(), 'p-B', SUBIENDO, umbral)
    assert [c.tipo for c in cruces] == [SALIDA]


def test_invertir_la_puerta_invierte_lo_que_cuenta_el_servidor():
    """EL arreglo de un segundo cuando las entradas salen como salidas.

    El fallo típico no es una puerta torcida: es una puesta al revés. Que el
    doble clic la arregle de verdad —y no solo en el dibujo— es lo que se
    fija aquí.
    """
    puerta = _puerta(HORIZONTAL)
    puerta.invertir()
    umbral = puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
    cruces = _caminar(DetectorLinea(), 'p-C', BAJANDO, umbral)
    assert [c.tipo for c in cruces] == [SALIDA], 'invertida, bajar es salir'


# ── La puerta vertical: se entra hacia la IZQUIERDA ──────────────────────
#: Dibujada de arriba abajo. P1→P2 va hacia +y, luego DENTRO (su derecha con
#: la `y` de imagen hacia abajo) queda hacia -x: la mitad IZQUIERDA.
VERTICAL = [[500, 100], [500, 900]]


def test_una_puerta_vertical_tambien_concuerda():
    """La convención no puede depender de cómo caiga la puerta."""
    puerta = _puerta(VERTICAL)
    umbral = puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
    assert puerta.normal_dentro()[0] < 0, 'la cabeza verde mira a la izquierda'

    hacia_la_izquierda = [(1200, 540), (1100, 540), (1020, 540), (940, 540),
                          (860, 540), (780, 540)]
    cruces = _caminar(DetectorLinea(), 'p-D', hacia_la_izquierda, umbral)
    assert [c.tipo for c in cruces] == [ENTRADA]


# ── Que el dibujo y la medida no puedan divergir ─────────────────────────
def test_los_dos_utensilios_producen_el_mismo_dato():
    """La puerta es la línea con otro rótulo. Punto por punto.

    Es lo que permite que no haya detector nuevo: la geometría ya estaba
    probada, con su histéresis, sus desempates de vano y su cooldown. Si esto
    deja de ser cierto, el servidor necesitaría un detector propio y hay que
    enterarse aquí, no en una cámara.
    """
    pytest.importorskip('PySide6')
    from elde_core.ui.linea_cruce import LineaCruce
    from elde_core.ui.puerta_direccional import PuertaDireccional

    for puntos in (HORIZONTAL, VERTICAL, [[120, 880], [760, 240]]):
        puerta = PuertaDireccional(puntos, activa=True)
        linea = LineaCruce(puntos, activa=True)
        assert (puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
                == linea.coordenadas_pixel(ANCHO, ALTO_FRAME))
        assert puerta.normal_dentro() == linea.normal_dentro()


def test_una_puerta_demasiado_corta_no_clasifica_nada():
    """Vale más el silencio que un sentido inventado.

    Dos puntos casi pegados dan una perpendicular aleatoria: el paso se
    clasificaría a cara o cruz. El cliente manda lista vacía y el detector
    calla; el modo cae al área clásica, que sigue contando permanencia.
    """
    puerta = _puerta([[500, 500], [508, 500]])
    assert puerta.es_utilizable() is False
    assert puerta.coordenadas_pixel(ANCHO, ALTO_FRAME) != []   # geometría hay
    # ...pero el recuadro NO la manda, y sin umbral el detector no emite.
    cruces = _caminar(DetectorLinea(), 'p-E', BAJANDO, [])
    assert cruces == []


def test_el_jitter_sobre_el_umbral_no_fabrica_cruces():
    """Quien se queda parado EN la puerta no entra ni sale.

    Es el caso de todos los días: alguien que se para en el vano a hablar.
    Sin histéresis, el temblor del tracker lo contaría entrando y saliendo
    varias veces por segundo.
    """
    puerta = _puerta(HORIZONTAL)
    umbral = puerta.coordenadas_pixel(ANCHO, ALTO_FRAME)
    temblor = [(960, 538), (960, 542), (960, 539), (960, 543), (960, 537),
               (960, 541), (960, 540), (960, 542)]
    cruces = _caminar(DetectorLinea(), 'p-F', temblor, umbral)
    assert cruces == []
