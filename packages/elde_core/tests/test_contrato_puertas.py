"""
El payload del frame acepta N PUERTAS con nombre (7-sep-2026).

Que se fija aqui y por que:

  - **lo que manda hoy TODO recuadro sin puertas sigue pasando** — la lista
    externa vacia y el cliente que no manda nada. Es la parte que no puede
    fallar: el contrato es `extra='forbid'` y el modo por defecto es
    `estricto`, asi que cualquier severidad de mas no da un aviso, deja a
    esa camara SIN VIDEO. Ya paso con el ROI vacio (31-jul), con los
    anaqueles (13-ago) y con `linea_cruce_estilo` (7-sep);
  - **el tope de 8 y la forma de cada umbral** — un segmento son 2 puntos
    de 2 coordenadas, ni uno mas;
  - **los nombres desparejados NO revientan** — un rotulo de mas o de
    menos es cosmetico; tirar el frame por eso seria mucho peor que el
    problema;
  - **POR QUE ES UN CAMPO NUEVO y no `linea_cruce` con otro tipo** — con
    el test que lo demuestra en vez del razonamiento: con EXACTAMENTE DOS
    puertas metidas en `linea_cruce` el validador viejo no se entera, y el
    servidor apaga el modo en silencio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ_PAQUETE = Path(__file__).resolve().parents[1]      # packages/elde_core
RAIZ = Path(__file__).resolve().parents[3]              # ELDE/
sys.path.insert(0, str(RAIZ_PAQUETE))

from pydantic import ValidationError                            # noqa: E402

from elde_core.contracts.eventos import (MAX_PUERTAS,           # noqa: E402
                                         FrameInference)

JPEG = b'\xff\xd8\xff\xe0' + b'\x00' * 64       # cabecera JPEG valida

#: Un umbral horizontal cualquiera, en pixeles del frame (1920x1080).
def _umbral(y: int):
    return [[192, y], [1728, y]]


# ── Lo que manda el cliente de verdad ───────────────────────────────────
def test_el_payload_del_anexo_valida_entero():
    """Los cinco campos del recuadro con dos puertas dibujadas."""
    frame = FrameInference(
        image=JPEG,
        linea_cruce=[[192, 540], [1728, 540]],
        linea_cruce_activate=True,
        linea_cruce_estilo='puerta',
        puertas_lineas=[_umbral(324), _umbral(756)],
        puertas_activate=True,
        puertas_nombres=['Almacen', 'Patio'],
    )
    assert frame.puertas_activate is True
    assert len(frame.puertas_lineas) == 2
    assert frame.puertas_nombres == ['Almacen', 'Patio']
    # El umbral suelto sigue llegando: quien manda sobre quien lo decide el
    # servidor (puertas ganan), no el contrato. Aqui solo se comprueba que
    # los dos caben en el MISMO payload; si el contrato rechazara uno de
    # los dos, el cliente no podria mandar lo que ya manda.
    assert frame.linea_cruce == [[192.0, 540.0], [1728.0, 540.0]]


def test_el_cliente_que_no_manda_nada_sigue_pasando():
    """Los tres campos son opcionales: un cliente sin actualizar —o
    cualquier pipeline que no sea permanencia— no se entera de que existen.

    Si esto se pusiera rojo, la version nueva del contrato apagaria a los
    cuatro clientes de golpe."""
    frame = FrameInference(image=JPEG)
    assert frame.puertas_activate is False
    assert frame.puertas_lineas is None
    assert frame.puertas_nombres is None


def test_la_lista_vacia_se_acepta():
    """Lo que manda en CADA frame todo recuadro SIN puertas dibujadas.

    No es un caso raro: es el caso normal de casi todas las camaras. Una
    lista vacia significa «no hay puertas», no «hay una puerta rota», y
    viaja acompañada de su `puertas_activate: False`."""
    frame = FrameInference(image=JPEG, puertas_lineas=[],
                           puertas_activate=False, puertas_nombres=[])
    assert frame.puertas_lineas == []
    assert frame.puertas_nombres == []


# ── El tope ─────────────────────────────────────────────────────────────
def test_el_tope_acordado_son_ocho():
    """El numero es del anexo del operador, no del codigo: se fija aqui
    para que bajarlo o subirlo sea una decision explicita y no un descuido
    en un refactor."""
    assert MAX_PUERTAS == 8


def test_ocho_puertas_pasan():
    lineas = [_umbral(100 + 50 * i) for i in range(MAX_PUERTAS)]
    frame = FrameInference(image=JPEG, puertas_lineas=lineas,
                           puertas_activate=True)
    assert len(frame.puertas_lineas) == MAX_PUERTAS


def test_nueve_puertas_revientan():
    """El tope se aplica en el contrato para que el error se VEA: el
    cliente recibe el motivo con el numero que mando. Dejarlo pasar solo
    daria un video a tirones sin causa aparente (N detectores y N libros
    de estancias se recorren en cada frame)."""
    lineas = [_umbral(100 + 50 * i) for i in range(MAX_PUERTAS + 1)]
    with pytest.raises(ValidationError) as exc:
        FrameInference(image=JPEG, puertas_lineas=lineas)
    assert 'como mucho 8' in str(exc.value)
    assert 'llegaron 9' in str(exc.value)


# ── La forma de cada umbral ─────────────────────────────────────────────
def test_un_umbral_de_tres_puntos_revienta():
    """Tres puntos no son un umbral: son un poligono a medio dibujar. El
    servidor solo leeria los dos primeros y mediria una linea que el
    operador no ve pintada."""
    with pytest.raises(ValidationError) as exc:
        FrameInference(image=JPEG,
                       puertas_lineas=[[[0, 0], [10, 10], [20, 20]]])
    assert 'puertas_lineas[0]' in str(exc.value)
    assert '2 puntos' in str(exc.value)


def test_un_umbral_de_un_solo_punto_revienta():
    with pytest.raises(ValidationError):
        FrameInference(image=JPEG, puertas_lineas=[[[0, 0]]])


def test_un_punto_de_tres_coordenadas_revienta():
    """`Poligono` es `List[List[float]]`: pydantic acepta un punto de tres
    numeros sin rechistar y el detector se quedaria con los dos primeros.
    El error tiene que salir aqui, con el punto en el mensaje."""
    with pytest.raises(ValidationError) as exc:
        FrameInference(image=JPEG,
                       puertas_lineas=[[[0, 0, 0], [10, 10]]])
    assert 'cada punto es' in str(exc.value)


def test_el_indice_de_la_puerta_mala_va_en_el_mensaje():
    """Con ocho umbrales, «uno esta mal» no sirve de nada al operador."""
    lineas = [_umbral(100), _umbral(200), [[0, 0], [1, 1], [2, 2]]]
    with pytest.raises(ValidationError) as exc:
        FrameInference(image=JPEG, puertas_lineas=lineas)
    assert 'puertas_lineas[2]' in str(exc.value)


# ── Los nombres ─────────────────────────────────────────────────────────
def test_los_nombres_desparejados_no_revientan():
    """Un rotulo de mas, de menos o ninguno es COSMETICO: el servidor
    rotula «Puerta N» y sigue contando igual.

    Validarlo contra `puertas_lineas` convertiria un desfase de un frame
    —renombrar mientras el video corre— en un frame DESCARTADO, y en modo
    estricto eso es la camara parada. El remedio seria peor."""
    dos = [_umbral(324), _umbral(756)]
    assert FrameInference(image=JPEG, puertas_lineas=dos,
                          puertas_nombres=['Almacen']).puertas_nombres == \
        ['Almacen']
    assert len(FrameInference(image=JPEG, puertas_lineas=dos,
                              puertas_nombres=['A', 'B', 'C', 'D']
                              ).puertas_nombres) == 4
    assert FrameInference(image=JPEG, puertas_lineas=dos,
                          puertas_nombres=[]).puertas_nombres == []
    assert FrameInference(image=JPEG,
                          puertas_lineas=dos).puertas_nombres is None
    # Y al reves: nombres sin lineas tampoco tumban el frame.
    assert FrameInference(image=JPEG,
                          puertas_nombres=['Almacen']).puertas_lineas is None


# ── Por que un campo nuevo y no cambiarle el tipo a `linea_cruce` ───────
class _Info:
    """El `info` que pydantic le pasa a un validador; aqui solo se usa
    `field_name` para el texto del error."""
    field_name = 'linea_cruce'


def _validador_de_segmento():
    """El validador REAL de `linea_cruce`, sacado del modelo de verdad.

    Se llama a mano —y no a traves de `FrameInference(...)`— porque lo que
    hay que medir es al VALIDADOR solo: en el modelo de hoy la anotacion
    `Poligono` rechaza la lista de segmentos antes de que el validador la
    vea, y esa red es justo la que desaparece si a alguien le da por
    ensanchar el tipo (que es la alternativa que este contrato descarta).
    Es el codigo del repo, no una copia: si `_segmento_ok` cambia, esto
    cambia con el."""
    return FrameInference.__pydantic_decorators__.field_validators[
        '_segmento_ok'].func


def test_el_validador_de_segmento_no_distingue_dos_puertas_de_un_umbral():
    """LA razon de que exista `puertas_lineas`, demostrada.

    `_segmento_ok` solo pregunta `len(v) == 2` y `len(p) == 2`, y las dos
    son ciertas para un segmento de dos puntos Y para una lista de DOS
    segmentos. Este test NO describe algo deseable: describe la trampa por
    la que las puertas NO viajan en `linea_cruce`."""
    validar = _validador_de_segmento()
    dos_puertas = [_umbral(324), _umbral(756)]

    assert validar(dos_puertas, _Info()) is dos_puertas, (
        "dos puertas disfrazadas de umbral pasan el validador de segmento "
        "INTACTAS: por eso el tipo de `linea_cruce` no se puede versionar")

    # Con UNA o con TRES, el MISMO validador revienta -> en modo estricto
    # se pierde el 100 % de los frames de ese cliente. Silencio con dos,
    # apagon con una o con tres: ninguna de las dos es una migracion.
    for n in (1, 3):
        puertas = [_umbral(100 + 50 * i) for i in range(n)]
        with pytest.raises(ValueError) as exc:
            validar(puertas, _Info())
        assert '2 puntos' in str(exc.value)


def test_hoy_linea_cruce_rechaza_a_gritos_la_lista_de_segmentos():
    """La otra cara: mientras el tipo de `linea_cruce` NO se toque, la
    propia anotacion `Poligono` (`List[List[float]]`) tumba esa lista
    antes del validador, y el cliente se entera.

    Este test es el centinela del campo viejo: si algun dia se pone rojo
    es que alguien ensancho el tipo, y con el se habra ido la unica cosa
    que separa «un umbral» de «dos puertas contadas en silencio»."""
    with pytest.raises(ValidationError) as exc:
        FrameInference(image=JPEG, linea_cruce=[_umbral(324), _umbral(756)])
    assert 'linea_cruce' in str(exc.value)


def test_las_mismas_tres_formas_por_el_campo_nuevo_se_comportan_igual():
    """El campo nuevo no tiene el punto ciego: 1, 2 y 3 puertas valen."""
    for n in (1, 2, 3):
        lineas = [_umbral(100 + 50 * i) for i in range(n)]
        frame = FrameInference(image=JPEG, puertas_lineas=lineas,
                               puertas_activate=True)
        assert len(frame.puertas_lineas) == n


def test_el_servidor_apaga_el_modo_en_silencio_con_dos_puertas():
    """La otra mitad de la demostracion, del lado del servidor.

    `_geometria` hace `float(datos[0][0])` sobre lo que ya es un PUNTO,
    salta `TypeError` y su `except` lo convierte en `None`: el detector
    devuelve `[]` y el video sigue perfecto. Nadie mira el log de una
    camara que se ve bien — ese es el modo de fallo que el campo nuevo
    evita."""
    sys.path.insert(0, str(RAIZ / 'server'))
    try:
        from src.analityc.core.permanencia_linea import _geometria
    except ImportError:                     # sin el server a mano, se omite
        pytest.skip('el servidor no esta en este arbol')

    dos_puertas = [_umbral(324), _umbral(756)]
    assert _geometria(dos_puertas) is None, (
        "si esto deja de ser None, el sintoma cambia y hay que revisar el "
        "comentario de `puertas_lineas` en el contrato")
    # Un umbral de verdad si se parsea: la diferencia no esta en el
    # servidor, esta en que nadie puede distinguir los dos payloads.
    assert _geometria(_umbral(540)) is not None


# ── El modo estricto sigue en pie ───────────────────────────────────────
def test_estricto_sigue_rechazando_lo_desconocido():
    """Anadir campos no afloja `extra='forbid'`. Un nombre casi igual
    (singular en vez de plural) tiene que sonar, no colarse en silencio."""
    with pytest.raises(ValidationError):
        FrameInference(image=JPEG, puerta_lineas=[_umbral(324)])
    with pytest.raises(ValidationError):
        FrameInference(image=JPEG, puertas_linea=[_umbral(324)])
