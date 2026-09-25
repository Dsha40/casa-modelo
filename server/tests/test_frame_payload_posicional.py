"""`frame_payload` es una tupla POSICIONAL: sus huecos tienen que casar.

## Por qué esto necesita una prueba

El bucle del websocket arma una tupla y la encola; el drenador la desempaqueta
contra `_full_frame_sync(*payload)`, que a su vez llama a `process_image_sync`.
No hay nombres de por medio en el primer salto: solo posiciones.

Insertar un argumento **en medio** de cualquiera de las tres listas corre
todos los siguientes una posición. Y el desastre es silencioso, porque los
tipos vecinos son compatibles: un `bool` cae donde iba otro `bool`, un
`Optional[str]` donde iba otro. El servidor no lanza; simplemente empieza a
leer el nombre de la cámara como establecimiento, o la flecha como línea. Se
descubre mirando alertas raras días después.

El código ya avisa —«Van AL FINAL porque `frame_payload` es una tupla
POSICIONAL»— pero un comentario no detiene a nadie. Esto sí.

La prueba que ya existía (`test_payload_a_frame_sync.py`) solo CUENTA los
huecos: una permutación de dos argumentos del mismo tipo la pasa entera.
Esta los compara uno a uno.

## Por qué con `ast` y no leyendo el fuente como texto

Una prueba que busca una línea literal se pone roja cuando alguien reordena o
parte una línea, sin que nada esté roto (pasó el 7-sep-2026 con otra prueba de
este mismo cambio). El árbol sintáctico aguanta el formato; lo que compara son
nombres y posiciones, que es lo que de verdad importa aquí.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

APP = RAIZ / 'server' / 'src' / 'app' / 'app.py'

#: Los argumentos que se añadieron AL FINAL y que no se pueden mover: los dos
#: umbrales de permanencia (19-ago-2026), el rótulo del utensilio con el que
#: se dibujó el umbral, las tres piezas de las PUERTAS —N umbrales con
#: nombre, uno por sitio— (7-sep-2026) y los PRODUCTOS dibujados dentro de
#: las áreas del anaquel (25-sep-2026).
ULTIMOS = ('min_permanencia_personas', 'min_permanencia_vehiculos',
           'linea_cruce_estilo',
           'puertas_lineas', 'puertas_nombres', 'puertas_activate',
           'anaqueles_productos')

#: Dos puertas como las manda el recuadro: píxeles del frame, 2 puntos por
#: umbral, la POSICIÓN en la lista es la identidad de cada puerta.
PUERTAS = [[[192, 324], [1728, 324]], [[192, 756], [1728, 756]]]

#: El umbral suelto de toda la vida (`linea_cruce`).
LINEA = [[300, 0], [300, 400]]

#: `TiendasProcess` multiplexa estado por cámara; lo que entre ahí queda en
#: None en cada frame (ver la última prueba del fichero).
TIENDAS = RAIZ / 'server' / 'src' / 'analityc' / 'core' / 'tiendas.py'


def _arbol():
    return ast.parse(APP.read_text(encoding='utf-8'))


def _firma(arbol, nombre):
    for n in ast.walk(arbol):
        if isinstance(n, ast.FunctionDef) and n.name == nombre:
            return [a.arg for a in n.args.args]
    pytest.fail(f'no se encontró la función {nombre} en app.py')


def _tupla(arbol):
    for n in ast.walk(arbol):
        if (isinstance(n, ast.Assign) and n.targets
                and getattr(n.targets[0], 'id', '') == 'frame_payload'
                and isinstance(n.value, ast.Tuple)):
            return [getattr(e, 'id', None) or '<expresión>'
                    for e in n.value.elts]
    pytest.fail('no se encontró la tupla frame_payload en app.py')


def test_la_tupla_casa_hueco_a_hueco_con_full_frame_sync():
    """El salto sin nombres: `_full_frame_sync(*frame_payload)`."""
    arbol = _arbol()
    tupla, firma = _tupla(arbol), _firma(arbol, '_full_frame_sync')
    desalineados = [(i, t, firma[i] if i < len(firma) else '<SOBRA>')
                    for i, t in enumerate(tupla)
                    if i >= len(firma) or t != firma[i]]
    assert not desalineados, (
        'la tupla y la firma se desalinearon; cada hueco pasa a leer el '
        f'argumento de al lado: {desalineados}')


def test_las_dos_firmas_llevan_los_mismos_argumentos_en_el_mismo_orden():
    """`_full_frame_sync` reenvía a `process_image_sync`."""
    arbol = _arbol()
    a, b = _firma(arbol, '_full_frame_sync'), _firma(arbol, 'process_image_sync')
    # `process_image_sync` recibe el procesador y la imagen ya decodificada,
    # así que el primer par difiere por diseño; de ahí en adelante, no.
    assert a[2:] == b[2:], (
        'las dos firmas divergieron a partir del tercer argumento')


def test_los_argumentos_nuevos_estan_AL_FINAL_y_en_el_mismo_orden():
    """Lo único que hace segura la ampliación de una tupla posicional."""
    arbol = _arbol()
    tupla = _tupla(arbol)
    assert tuple(tupla[-len(ULTIMOS):]) == ULTIMOS, (
        f'los últimos argumentos son {tupla[-len(ULTIMOS):]}; añadir uno en '
        'medio corre todos los siguientes en silencio')
    for nombre in ('_full_frame_sync', 'process_image_sync'):
        assert tuple(_firma(arbol, nombre)[-len(ULTIMOS):]) == ULTIMOS, (
            f'{nombre} no termina en {ULTIMOS}')


def test_las_dos_ramas_cuelgan_el_rotulo_en_el_procesador():
    """Hay DOS ramas que preparan el procesador (multicámara y simple). Si
    solo una cuelga el rótulo, la mitad de los modos lo pierde y la alerta
    sale muda sin que nada falle."""
    arbol = _arbol()
    puestas = [n for n in ast.walk(arbol)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute)
                       and t.attr == '_linea_cruce_estilo' for t in n.targets)]
    assert len(puestas) >= 2, (
        f'solo {len(puestas)} rama(s) cuelgan _linea_cruce_estilo; la otra '
        'perdería el rótulo')
    # Y donde se cuelga la línea, se cuelga el rótulo: van juntos o no van.
    lineas = [n for n in ast.walk(arbol)
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Attribute)
                      and t.attr == '_linea_cruce' for t in n.targets)]
    assert len(puestas) == len(lineas), (
        f'{len(lineas)} ramas cuelgan la línea pero {len(puestas)} el rótulo')


def test_el_rotulo_y_el_interruptor_se_leen_del_payload_del_cliente():
    """Comportamiento, no texto: lo que el recuadro manda es lo que llega."""
    from src.app.app import umbral_del_payload
    LINEA = [[300, 0], [300, 400]]
    assert umbral_del_payload({'linea_cruce': LINEA,
                               'linea_cruce_activate': True,
                               'linea_cruce_estilo': 'puerta'}) == (LINEA, 'puerta')
    # El interruptor apagado quita el umbral aunque siga dibujado.
    assert umbral_del_payload({'linea_cruce': LINEA,
                               'linea_cruce_activate': False,
                               'linea_cruce_estilo': 'puerta'})[0] is None
    # Un cliente anterior al 7-sep-2026 no manda ninguno de los dos.
    assert umbral_del_payload({'linea_cruce': LINEA}) == (LINEA, None)


def test_el_contrato_acepta_el_rotulo_del_umbral():
    """El servidor valida los frames con `extra='forbid'` en modo estricto:
    sin este campo en el contrato, un cliente que mande la puerta ve
    RECHAZADO el 100 % de sus frames (verificado el 7-sep-2026)."""
    pydantic = pytest.importorskip('pydantic')
    sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))
    from elde_core.contracts.eventos import FrameInference

    def errores(payload):
        try:
            FrameInference(**payload)
            return []
        except pydantic.ValidationError as e:
            return [er for er in e.errors() if er['type'] == 'extra_forbidden']

    base = {'image': 'x', 'linea_cruce': [[300, 0], [300, 400]],
            'linea_cruce_activate': True}
    assert errores({**base, 'linea_cruce_estilo': 'puerta'}) == []
    assert errores({**base, 'linea_cruce_estilo': 'linea'}) == []
    # Y sigue acotado: no se acepta cualquier cosa del cliente.
    with pytest.raises(pydantic.ValidationError):
        FrameInference(**{**base, 'linea_cruce_estilo': 'x' * 64})


def _cuelgues(arbol, atributo):
    """Las asignaciones `algo.<atributo> = ...` que hay en el fuente."""
    return [n for n in ast.walk(arbol)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == atributo
                    for t in n.targets)]


def test_las_dos_ramas_cuelgan_LAS_PUERTAS_en_el_procesador():
    """A4, puntos 5 y 6. `process_image_sync` prepara el procesador en DOS
    ramas distintas: la multicámara (`cam_proc`) y la simple, dentro del
    `except`. Colgar las puertas en una sola deja a la otra mitad de las
    cámaras sin ver ninguna: caen al aforo único de la cámara con el vídeo
    perfecto y sin un solo error que lo diga."""
    arbol = _arbol()
    for atributo in ('_puertas_lineas', '_puertas_nombres',
                     '_puertas_activate'):
        cuantas = len(_cuelgues(arbol, atributo))
        assert cuantas >= 2, (
            f'solo {cuantas} rama(s) cuelgan {atributo}; la otra se queda '
            'sin puertas y nadie se entera')
    # Y las tres viajan JUNTAS. Una rama con las líneas pero sin los nombres
    # rotularía «Puerta 1/2/3» sobre puertas que el operador ya bautizó, y
    # sin el interruptor no habría forma de apagarlas desde el recuadro.
    assert (len(_cuelgues(arbol, '_puertas_lineas'))
            == len(_cuelgues(arbol, '_puertas_nombres'))
            == len(_cuelgues(arbol, '_puertas_activate'))), (
        'las tres piezas de las puertas se cuelgan en distinto número de '
        'ramas: alguna rama recibe media configuración')


def test_las_dos_ramas_cuelgan_LOS_PRODUCTOS_del_anaquel():
    """Los productos (25-sep-2026) viajan con sus áreas: en cada rama donde
    se cuelgan las áreas tienen que colgarse ellos. Si faltaran en una, esa
    mitad de las cámaras contaría por área sin repartir entre productos, y
    el síntoma sería un producto que nunca suma, no un error."""
    arbol = _arbol()
    zonas = len(_cuelgues(arbol, '_anaqueles_zonas'))
    productos = len(_cuelgues(arbol, '_anaqueles_productos'))
    assert zonas >= 2, f'solo {zonas} rama(s) cuelgan las áreas'
    assert productos == zonas, (
        f'las áreas se cuelgan en {zonas} ramas y los productos en '
        f'{productos}: alguna rama recibe las áreas sin sus productos')


def test_las_puertas_se_leen_del_payload_tal_como_las_manda_el_recuadro():
    """Comportamiento, no texto: lo que dibuja el operador es lo que llega."""
    from src.app.app import puertas_del_payload
    assert puertas_del_payload({'puertas_lineas': PUERTAS,
                                'puertas_activate': True,
                                'puertas_nombres': ['Almacen', 'Patio']}) == (
        PUERTAS, ['Almacen', 'Patio'], True)
    # El interruptor AUSENTE cuenta como True, igual que en el umbral suelto:
    # un cliente que manda las líneas sin el campo las quiere.
    lineas, _, activado = puertas_del_payload({'puertas_lineas': PUERTAS})
    assert (lineas, activado) == (PUERTAS, True)
    # Los nombres NO se cruzan con las líneas: un desfase de rótulos es
    # cosmético y descartar el frame por eso sería mucho peor.
    assert puertas_del_payload({'puertas_lineas': PUERTAS,
                                'puertas_nombres': ['Almacen']}) == (
        PUERTAS, ['Almacen'], True)


def test_sin_puertas_dibujadas_todo_queda_EXACTAMENTE_como_hoy():
    """Compatibilidad: el 99 % de los recuadros no dibuja puertas."""
    from src.app.app import puertas_del_payload
    # Lo que manda HOY, en CADA frame, todo recuadro sin puertas.
    assert puertas_del_payload({'puertas_lineas': [],
                                'puertas_activate': False,
                                'puertas_nombres': []}) == (None, None, False)
    # Un cliente anterior al 7-sep-2026 no manda ninguno de los tres campos.
    # El interruptor NO puede salir en True por el «ausente = True»: quien
    # mirase solo el interruptor cambiaría de modo sin una puerta dibujada.
    assert puertas_del_payload({}) == (None, None, False)
    assert puertas_del_payload({'linea_cruce': LINEA}) == (None, None, False)
    # Interruptor apagado con las puertas dibujadas: vuelve el aforo de la
    # cámara, igual que apagar el umbral suelto devuelve el área clásica.
    assert puertas_del_payload({'puertas_lineas': PUERTAS,
                                'puertas_activate': False}) == (
        None, None, False)
    # Y lo que no es una lista se ignora: una cadena dejaría al gestor
    # iterando letras y montando una puerta por carácter.
    assert puertas_del_payload({'puertas_lineas': 'dos'}) == (None, None, False)


def test_las_puertas_MANDAN_sobre_el_umbral_suelto():
    """Precedencia de A3, que decide el transporte y no el contrato: los dos
    campos son válidos y pueden llegar en el MISMO payload. Medir el mismo
    umbral dos veces —una en el aforo de la cámara y otra en la puerta—
    emitiría DOS entradas por cada persona que pasa UNA vez."""
    from src.app.app import umbrales_del_payload
    linea, estilo, puertas, nombres, activado = umbrales_del_payload({
        'linea_cruce': LINEA, 'linea_cruce_activate': True,
        'linea_cruce_estilo': 'puerta',
        'puertas_lineas': PUERTAS, 'puertas_activate': True,
        'puertas_nombres': ['Almacen', 'Patio']})
    assert linea is None, 'el umbral suelto tiene que quedar IGNORADO'
    assert (puertas, nombres, activado) == (PUERTAS, ['Almacen', 'Patio'], True)
    assert estilo == 'puerta'   # cosmético: no pesa en ninguna cuenta
    # Sin puertas, la línea suelta rige EXACTAMENTE igual que hoy.
    assert umbrales_del_payload({'linea_cruce': LINEA,
                                 'puertas_lineas': [],
                                 'puertas_activate': False}) == (
        LINEA, None, None, None, False)
    # Y con las puertas dibujadas pero apagadas, vuelve a mandar la línea.
    assert umbrales_del_payload({'linea_cruce': LINEA,
                                 'puertas_lineas': PUERTAS,
                                 'puertas_activate': False})[0] == LINEA


def test_el_contrato_acepta_las_claves_de_las_puertas():
    """Con `extra='forbid'` en modo estricto, un campo que el contrato no
    conoce RECHAZA el 100 % de los frames del cliente que lo manda."""
    pydantic = pytest.importorskip('pydantic')
    sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))
    from elde_core.contracts.eventos import FrameInference

    def errores(payload):
        try:
            FrameInference(**payload)
            return []
        except pydantic.ValidationError as e:
            return [er for er in e.errors() if er['type'] == 'extra_forbidden']

    base = {'image': 'x'}
    assert errores({**base, 'puertas_lineas': PUERTAS,
                    'puertas_activate': True,
                    'puertas_nombres': ['Almacen', 'Patio']}) == []
    # Lo que manda en cada frame un recuadro SIN puertas: si esto se
    # rechazara, todos los recuadros de hoy se quedarían sin vídeo.
    assert errores({**base, 'puertas_lineas': [], 'puertas_activate': False,
                    'puertas_nombres': []}) == []
    # Y sigue acotado, pinzado por los DOS lados. Con `image='x'` esto no
    # medía nada: la imagen ya revienta por su propio validador y el
    # `raises` se cumplía tuviera tope o no (mutar MAX_PUERTAS a 999 lo
    # dejaba pasar). Hace falta un JPEG que valide para que el único
    # motivo posible de error sea el tope.
    jpeg = {'image': b'\xff\xd8\xff' + b'\x00' * 64}
    FrameInference(**{**jpeg, 'puertas_lineas': [PUERTAS[0]] * 8})   # 8: pasa
    with pytest.raises(pydantic.ValidationError) as ocho_mas_una:
        FrameInference(**{**jpeg, 'puertas_lineas': [PUERTAS[0]] * 9})
    assert '9' in str(ocho_mas_una.value)


def test_las_puertas_NO_entran_en_el_estado_que_tiendas_multiplexa():
    """La trampa del 6 y el 7-ago-2026, a propósito (A10).

    `TiendasProcess._push_state` intercambia por cámara los atributos de una
    lista de nombres. Lo que entre en esa lista y no esté en el `cam_state`
    queda en None EN CADA FRAME: las puertas desaparecerían, `_actualizar`
    reventaría dentro de un `try` que solo loguea, y el modo quedaría MUDO
    con el vídeo perfecto. No hay nada que multiplexar —ya existe una
    instancia del procesador por cámara—, así que los atributos de las
    puertas son PLANOS y tienen que quedarse fuera.

    Prueba de fuente y solo para eso (lección H-47): «esta clave no está en
    esta ruta», nunca «este valor es correcto».
    """
    arbol = ast.parse(TIENDAS.read_text(encoding='utf-8'))
    push = next((n for n in ast.walk(arbol)
                 if isinstance(n, ast.FunctionDef) and n.name == '_push_state'),
                None)
    assert push is not None, (
        'TiendasProcess._push_state cambió de nombre: revisa a mano que las '
        'puertas sigan fuera del estado multiplexado')
    multiplexados = {n.value for n in ast.walk(push)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for atributo in ('_puertas_lineas', '_puertas_nombres',
                     '_puertas_activate', '_gestor_puertas'):
        assert atributo not in multiplexados, (
            f'{atributo} entró en el estado que _push_state multiplexa: cada '
            'frame lo dejará en None y el modo puertas quedará mudo sin un '
            'solo error')


def test_las_dos_ramas_dejan_LAS_PUERTAS_puestas_de_verdad(monkeypatch):
    """Lo mismo que la prueba anterior, pero EJECUTANDO (lección H-47: el
    fuente contesta «esta clave está en esta ruta», nunca «este valor es el
    correcto»). La segunda rama es la del `except`, así que se llega a ella
    haciendo reventar `get_camera_processor` — que es justo lo que pasa con
    un procesador de una sola cámara."""
    from src.app import app as modulo

    class Falso:
        """Pasa el `isinstance(processor, TiendasProcess)` y no carga nada."""

        def __init__(self, sin_multicamara):
            self.sin_multicamara = sin_multicamara

        def get_camera_processor(self, camera_id):
            if self.sin_multicamara:
                raise RuntimeError('procesador simple: no hay multicámara')
            return self

        def set_camera_angle(self, angulo):
            pass

        def process_frame(self, *a, **k):
            return 'img', {}

    monkeypatch.setattr(modulo, 'TiendasProcess', Falso)
    for sin_multicamara in (False, True):
        proc = Falso(sin_multicamara)
        modulo.process_image_sync(proc, None, None, False,
                                  puertas_lineas=PUERTAS,
                                  puertas_nombres=['Almacen', 'Patio'],
                                  puertas_activate=True)
        rama = 'simple' if sin_multicamara else 'multicámara'
        assert proc._puertas_lineas == PUERTAS, f'rama {rama} sin las líneas'
        assert proc._puertas_nombres == ['Almacen', 'Patio'], (
            f'rama {rama} sin los rótulos: las puertas saldrían como '
            '«Puerta 1/2» aunque el operador las bautizara')
        assert proc._puertas_activate is True, f'rama {rama} sin interruptor'

    # Y un cliente que no manda puertas deja las dos ramas en el camino de
    # siempre: ninguna puerta y el interruptor apagado (aforo de la cámara).
    for sin_multicamara in (False, True):
        proc = Falso(sin_multicamara)
        modulo.process_image_sync(proc, None, None, False)
        assert (proc._puertas_lineas, proc._puertas_nombres,
                proc._puertas_activate) == (None, None, False)
