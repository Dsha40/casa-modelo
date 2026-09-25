"""
Payloads por evento. Un modelo por cada forma de mensaje que existe hoy.

Todos los campos salen del codigo real (`render_box.py` los emite,
`app.py` los lee con `data.get()`), no de un diseno ideal: el objetivo del
HITO 3 es formalizar lo que YA circula, no cambiarlo. Lo que se anade es la
validacion, que hoy no hay ninguna.

Los campos opcionales lo son porque distintos clientes mandan distintos
subconjuntos: el de tienda no manda `pay_roi`, el de autolavado no manda
`heatmap_activate`. Marcarlos obligatorios romperia clientes vivos.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Un poligono es una lista de puntos [x, y] normalizados o en pixeles.
Punto = List[float]
Poligono = List[Punto]

#: Tope de umbrales por camara (7-sep-2026). No es un numero estetico: cada
#: puerta se lleva en el servidor su PROPIO detector de linea y su propio
#: libro de estancias, y los N se recorren en CADA frame. Sin tope, un
#: cliente con la lista duplicada por un fallo suyo multiplica el trabajo
#: por frame de esa camara hasta comerse el turno de las demas, y el sintoma
#: seria «el video va a tirones», no «hay 400 puertas». Ocho es el tope que
#: fijo el operador, y con el error en el contrato lo ve al instante.
MAX_PUERTAS = 8


class CameraAngle(str, Enum):
    """Perfil de angulo de la camara. Ajusta el filtro de aspecto del
    detector: en cenital una persona se ve mucho mas ancha que alta.

    `auto` deja que el servidor lo deduzca solo, y es el valor por defecto
    real (`tiendas.py:860`). Una version anterior de este
    contrato no lo contemplaba y habria **rechazado payloads reales y
    funcionales**: lo detecto la captura de una sesion de verdad, no la
    lectura del codigo emisor."""
    AUTO = "auto"
    FRONTAL = "frontal"
    LATERAL = "lateral"
    CENITAL = "cenital"


def _valida_poligono(v: Optional[Poligono], nombre: str) -> Optional[Poligono]:
    if v is None:
        return v
    if not v:
        # Lista VACIA = "no hay poligono", que no es lo mismo que un poligono
        # mal formado. Es lo que devuelve `get_coordinates()` cuando la zona no
        # tiene puntos, y viaja acompañada de su `*_activate: False`.
        #
        # Rechazarla habria tumbado a los cuatro clientes al pasar la
        # validacion a `estricto`: basta con no tener ROI dibujado. Lo encontro
        # el modo `observar` del HITO 8 con trafico real, que es justo para lo
        # que existe.
        return v
    if len(v) < 3:
        raise ValueError(f"{nombre}: un poligono necesita 3 puntos o mas, "
                         f"llegaron {len(v)}")
    for p in v:
        if len(p) != 2:
            raise ValueError(f"{nombre}: cada punto es [x, y], llego {p!r}")
    return v


class ProductoAnaquel(BaseModel):
    """Un producto dibujado DENTRO de un área de anaquel (25-sep-2026).

    Sirve para una sola cosa: repartir los AGARRES del área entre lo que hay
    en la balda («Nescafé Gold» frente a «Colcafé»). El servidor ya sabía en
    qué punto exacto entraba la muñeca; con esto sabe además qué producto
    había ahí.

    La validación es SOLO de forma, y a propósito. Un campo declarado con un
    valor mal formado tira el frame ENTERO en modo estricto (lo que sí
    perdona `validacion_contrato` es el campo DESCONOCIDO, no el malo), así
    que aquí no se comprueba que `anaquel` exista entre las áreas del frame:
    un producto huérfano lo ignora el servidor, igual que un rótulo de
    puerta sobrante (ver `puertas_nombres`). Convertirlo en error dejaría la
    cámara sin vídeo por un desfase que no le quita nada a nadie.
    """

    model_config = ConfigDict(extra='forbid')

    #: A qué área pertenece: su ÍNDICE en `anaqueles_zonas` del mismo frame.
    anaquel: int
    #: Lo escribe el operador. Mismo tope que el resto de nombres del
    #: contrato; el cliente recorta antes para no llegar nunca aquí.
    nombre: str = Field(default='', max_length=120)
    #: En píxeles del frame enviado, como las áreas.
    poligono: Poligono = Field(default_factory=list)

    @field_validator('poligono')
    @classmethod
    def _poligono_ok(cls, v):
        return _valida_poligono(v, 'poligono')


class FrameInference(BaseModel):
    """Cliente -> servidor. Un frame a analizar y la configuracion activa.

    Corresponde al `data` de hoy (`render_box.py:645-669`)."""

    model_config = ConfigDict(extra='forbid')  # endurecido el 31-jul; ver nota al pie

    image: bytes = Field(..., description="JPEG del frame.")

    # ── Region de interes principal ──
    roi_activate: bool = False
    roi_coordinates: Optional[Poligono] = None

    # ── Camara ──
    # Duplica el `device_id` del envelope DENTRO del payload. No es un
    # descuido: el formato en transicion lo lleva ahi (`data.camera_id`) y el
    # servidor lo lee de ahi; la capa de compatibilidad lo promociona al
    # envelope SIN sacarlo del payload (sacarlo mutaria el request que el
    # servidor procesa despues). Sin este campo, `extra='forbid'` rechazaria
    # TODOS los frames.
    camera_id: Optional[str] = Field(default=None, max_length=96)
    camera_angle: CameraAngle = CameraAngle.AUTO
    camera_name: Optional[str] = Field(default=None, max_length=48)
    # Cabecera del frame que anade el cliente de perimetrales: timestamp de
    # captura, tamano y formato. Visto en la captura real, no en el codigo de
    # tienda. Se deja abierto porque cada cliente mete lo suyo.
    header: Optional[Dict[str, Any]] = None

    # ── Analitica ──
    heatmap_activate: bool = False
    track_classes: Optional[List[int]] = None
    draw_server: bool = True
    enable_vlm: bool = False
    # Notificacion por WhatsApp de las alertas (clientes perimetrales).
    enviar_whatsapp: bool = False
    # Establecimiento (local) de ESTA camara (3-ago-2026): lo elige el
    # operador en el select "Local" del recuadro. El servidor lo usa como
    # encabezado del mensaje de WhatsApp en lugar del "VIGILANTE" generico.
    establecimiento: Optional[str] = Field(default=None, max_length=120)
    # Lectura de placas (ALPR) de ESTA camara (5-ago-2026): boton propio en
    # la barra del recuadro. Va por camara porque el OCR cuesta y solo
    # interesa donde se ven las placas (entrada del estacionamiento).
    leer_placas: bool = False
    # Segundos DENTRO del area antes de alertar, de ESTA camara (contadores
    # de la barra del recuadro, 19-ago-2026). Se declaran aqui o el modo
    # estricto los rechaza: el contrato es extra=forbid y un campo sin
    # declarar tira el frame ENTERO — asi se perdio el 100% de los frames de
    # permanencia el 13-ago. None = el cliente no los maneja y rige config.
    min_permanencia_personas: Optional[float] = Field(default=None, ge=0, le=3600)
    min_permanencia_vehiculos: Optional[float] = Field(default=None, ge=0, le=3600)

    # ── Zonas de Hummus (pedido / entrega de bandeja) ──
    order_zone_activate: bool = False
    order_zone_coordinates: Optional[Poligono] = None
    delivery_zone_activate: bool = False
    delivery_zone_coordinates: Optional[Poligono] = None

    # ── Zonas de autolavado (pago / retiro) ──
    pay_roi_activate: bool = False
    pay_roi: Optional[Poligono] = None
    withdraw_roi_activate: bool = False
    withdraw_roi: Optional[Poligono] = None

    # ── Puerta (perimetrales) ──
    door_roi_activate: bool = False
    door_roi_coordinates: Optional[Poligono] = None
    door_direction_activate: bool = False
    door_direction: Optional[str] = None

    # ── PermanenciaArea: flecha, línea y nombre del área ──
    # REPARACIÓN (13-ago-2026): el cliente detecciones emite estos campos
    # desde el 8/10-ago y `app.py` los lee, pero nunca se declararon aquí.
    # Con `extra='forbid'` y el modo `estricto` por defecto, el contrato
    # rechazaba el 100% de esos frames — verificado ejecutando la validación
    # contra un payload real. Son 2 puntos, no un polígono, así que tienen
    # su propio validador (`_valida_segmento`).
    flecha_direccion: Optional[Poligono] = None
    flecha_direccion_activate: bool = False
    linea_cruce: Optional[Poligono] = None
    #: AUSENTE no es False: el servidor lo trata como True porque los
    #: clientes anteriores a la puerta direccional mandan la linea sin
    #: este campo y exigirlo les apagaria la lectura que ya usan (ver
    #: app.umbral_del_payload). Por eso `None` y no `False`: declarar
    #: False aqui hacia que este fichero —que se presenta como fuente
    #: unica de verdad— dijera lo contrario que el servidor justo en el
    #: caso que se queria proteger (7-sep-2026).
    linea_cruce_activate: Optional[bool] = None
    #: Con QUÉ utensilio se dibujó el umbral que viaja en `linea_cruce`:
    #: `'linea'` (rótulos DENTRO/FUERA) o `'puerta'` (flecha doble
    #: ENTRADA/SALIDA). La geometría de los dos es la misma y la lee el
    #: mismo detector; esto viaja SOLO para que el registro del servidor
    #: diga qué vio el operador cuando haya que revisar una calibración
    #: al revés. Opcional a propósito: un cliente viejo que no lo mande
    #: sigue valiendo. SIN ESTE CAMPO el contrato es `extra='forbid'` y
    #: el modo por defecto es `estricto`: un cliente que lo mandara
    #: vería rechazado el 100 % de sus frames (7-sep-2026).
    linea_cruce_estilo: Optional[str] = Field(default=None, max_length=16)
    area_nombre: Optional[str] = Field(default=None, max_length=120)
    #: Zona de PERSONAL del recuadro (7-sep-2026). El servidor todavia
    #: no hace nada con ella; se declara porque el contrato es
    #: `extra='forbid'` en modo estricto y, sin declararla, el cliente
    #: que la dibuja ve rechazados TODOS sus frames — que es como se
    #: descubrio: «le doy play y no detecta».
    personal_zone_coordinates: Optional[Poligono] = None
    personal_zone_activate: bool = False

    # ── Anaqueles: N áreas del estante dibujadas por el operador ──
    # Un polígono por nivel del anaquel, en píxeles del frame enviado, con
    # su lista PARALELA de nombres. Solo los lee el pipeline `Anaqueles`.
    anaqueles_activate: bool = False
    anaqueles_zonas: Optional[List[Poligono]] = None
    anaqueles_nombres: Optional[List[str]] = None
    #: Productos dibujados dentro de esas áreas (25-sep-2026). Lista vacía o
    #: ausente = nadie dibujó productos: los agarres se cuentan por área,
    #: exactamente como antes. Un servidor anterior a este campo NO pierde
    #: frames por él: `validacion_contrato` deja pasar los campos que no
    #: conoce (desde el 7-sep) y solo los apunta en el log.
    anaqueles_productos: Optional[List[ProductoAnaquel]] = None

    # ── Puertas: N umbrales con nombre, uno por acceso (7-sep-2026) ──
    # Cada puerta es un SEGMENTO `[[x1,y1],[x2,y2]]` en pixeles del frame,
    # con la MISMA convencion que `linea_cruce`: caminando de P1 a P2,
    # DENTRO queda a la DERECHA. No hay geometria nueva ni detector nuevo:
    # el eje de la flecha doble que pinta el recuadro ES la normal de esta
    # linea, y el servidor instancia el detector de siempre una vez por
    # puerta. `puertas_nombres` es la lista PARALELA de rotulos.
    #
    # POR QUE UN CAMPO NUEVO Y NO CAMBIARLE EL TIPO A `linea_cruce`
    # ------------------------------------------------------------
    # La alternativa era reutilizar `linea_cruce` ENSANCHANDO su tipo a
    # una lista de segmentos: es la unica forma de que N umbrales quepan
    # en ese campo. Se descarto porque el guardia que quedaria no
    # distingue los dos significados, y esto esta COMPROBADO, no razonado
    # (`test_contrato_puertas.py` llama al validador de verdad):
    #
    #   `_segmento_ok` solo pregunta `len(v) == 2` y `len(p) == 2`, y las
    #   dos cosas son ciertas para UN segmento de dos puntos Y para una
    #   LISTA DE DOS SEGMENTOS. Con el tipo ensanchado, DOS puertas pasan
    #   INTACTAS: llegan al servidor, `_geometria` hace
    #   `float(datos[0][0])` sobre lo que ya es un PUNTO, salta el
    #   `TypeError` y el `except (TypeError, ValueError, IndexError)` de
    #   `permanencia_linea.py:431` lo devuelve como `None`. El modo se
    #   apaga EN SILENCIO con el video perfecto, y nadie mira el log de
    #   una camara que se ve bien.
    #
    #   Con UNA o con TRES puertas ese mismo validador SI revienta y, en
    #   modo estricto, se pierde el 100 % de los frames de ese cliente.
    #   O sea: el campo significaria una cosa u otra segun su LONGITUD,
    #   y el fallo seria silencio o apagon segun cuantas puertas dibujara
    #   el operador esa tarde.
    #
    # Un tipo que el validador no puede distinguir del viejo no se puede
    # versionar. Con un campo propio, `linea_cruce` sigue siendo
    # EXACTAMENTE lo que era —hoy su anotacion `Poligono` rechaza esa
    # lista de segmentos a gritos, y ese ruido es justo la proteccion que
    # se pierde al ensanchar el tipo—, y el modo nuevo viaja por un campo
    # que un servidor viejo rechaza A LA VISTA (`extra='forbid'` lo
    # registra) en lugar de contar mal en la sombra.
    puertas_activate: bool = False
    puertas_lineas: Optional[List[Poligono]] = None
    #: NO se valida contra `puertas_lineas`, y es deliberado: un desfase de
    #: nombres es COSMETICO (el servidor rotula «Puerta 3» y sigue
    #: contando), mientras que convertirlo en error tira el frame ENTERO en
    #: modo estricto. Cambiar un rotulo a media tarde dejaria a esa camara
    #: sin video hasta que el operador adivinara por que.
    puertas_nombres: Optional[List[str]] = None

    @field_validator('image')
    @classmethod
    def _parece_jpeg(cls, v: bytes) -> bytes:
        """Un frame vacio o que no es JPEG hace fallar al detector mas
        adelante, con un error mucho menos claro que este."""
        if not v:
            raise ValueError("image: llego vacio")
        if not v.startswith(b'\xff\xd8'):
            raise ValueError("image: no empieza por la cabecera JPEG (FF D8)")
        return v

    @field_validator('roi_coordinates')
    @classmethod
    def _roi_ok(cls, v):
        return _valida_poligono(v, 'roi_coordinates')

    @field_validator('flecha_direccion', 'linea_cruce')
    @classmethod
    def _segmento_ok(cls, v, info):
        """Flecha y línea son SEGMENTOS: exactamente 2 puntos [x, y].

        Lista vacía = herramienta apagada (lo que emite el cliente cuando el
        utensilio no está dibujado), igual que en los polígonos."""
        if v is None or not v:
            return v
        if len(v) != 2:
            raise ValueError(f"{info.field_name}: un segmento son 2 puntos "
                             f"[[x1,y1],[x2,y2]], llegaron {len(v)}")
        for p in v:
            if len(p) != 2:
                raise ValueError(f"{info.field_name}: cada punto es [x, y], "
                                 f"llego {p!r}")
        return v

    @field_validator('anaqueles_zonas')
    @classmethod
    def _anaqueles_ok(cls, v):
        if v is None:
            return v
        for i, poligono in enumerate(v):
            _valida_poligono(poligono, f'anaqueles_zonas[{i}]')
            if not poligono:
                raise ValueError(f"anaqueles_zonas[{i}]: un área vacía no "
                                 f"significa nada; se omite de la lista")
        return v

    @field_validator('puertas_lineas')
    @classmethod
    def _puertas_ok(cls, v):
        """Cada puerta es un SEGMENTO exacto: 2 puntos de 2 coordenadas.

        La lista externa VACIA se acepta a proposito: es lo que manda en
        CADA frame todo recuadro que no tiene puertas dibujadas (junto a
        `puertas_activate: False`), que hoy son casi todos.
        Rechazarla los dejaria SIN VIDEO — el mismo fallo que ya
        costo el ROI vacio al endurecer a `estricto` el 31-jul-2026, y que
        por eso `_valida_poligono` tambien deja pasar.

        El tope se comprueba aqui y no en el servidor porque aqui el error
        se ve: en modo estricto el cliente recibe el mensaje con el numero
        que mando. Aguas abajo solo habria un video a tirones sin causa.
        """
        if v is None:
            return v
        if len(v) > MAX_PUERTAS:
            raise ValueError(f"puertas_lineas: como mucho {MAX_PUERTAS} "
                             f"puertas por camara, llegaron {len(v)}")
        for i, segmento in enumerate(v):
            # Exactamente 2: es la comprobacion que `linea_cruce` NO puede
            # hacer sobre una lista de puertas (ver el comentario del
            # campo), y la unica que separa «dos umbrales» de «un umbral».
            if len(segmento) != 2:
                raise ValueError(f"puertas_lineas[{i}]: un umbral son 2 "
                                 f"puntos [[x1,y1],[x2,y2]], llegaron "
                                 f"{len(segmento)}")
            for p in segmento:
                if len(p) != 2:
                    raise ValueError(f"puertas_lineas[{i}]: cada punto es "
                                     f"[x, y], llego {p!r}")
        return v

    @field_validator('order_zone_coordinates')
    @classmethod
    def _order_ok(cls, v):
        return _valida_poligono(v, 'order_zone_coordinates')

    @field_validator('delivery_zone_coordinates')
    @classmethod
    def _deliv_ok(cls, v):
        return _valida_poligono(v, 'delivery_zone_coordinates')

    @field_validator('personal_zone_coordinates')
    @classmethod
    def _personal_ok(cls, v):
        """La zona gris del comedor (personal). Sin esto, un poligono de dos
        puntos entraba y el motor marcaba como PERSONAL a quien pasara por una
        franja sin area — y un empleado queda fuera del conteo de clientes y
        de los grupos, asi que el fallo se ve como ventas que no cuadran, no
        como un error."""
        return _valida_poligono(v, 'personal_zone_coordinates')


class Deteccion(BaseModel):
    """Una caja detectada, con lo que el pipeline haya podido inferir."""
    model_config = ConfigDict(extra='forbid')

    track_id: Optional[int] = None
    bbox: Optional[List[float]] = None
    confianza: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clase: Optional[str] = None
    gender: Optional[str] = None
    age_range: Optional[str] = None


class FrameResult(BaseModel):
    """Servidor -> cliente. Resultado del analisis de un frame.

    Hoy el servidor reutiliza el mismo sobre y sustituye `data` por el
    resultado (`app.py:785`), asi que cliente y servidor comparten forma
    exterior pero no interior."""

    model_config = ConfigDict(extra='forbid')

    status: Optional[str] = None           # "ok" o "success" segun pipeline
    message: Optional[str] = None
    processing_time: Optional[float] = Field(default=None, ge=0.0)

    # Eco del device_id, para que el cliente sepa a que camara corresponde
    # la respuesta cuando tiene varias en vuelo.
    camera_id: Optional[str] = None

    # El frame anotado viaja con DOS nombres segun el pipeline: `image` en
    # los de tienda y `processed_image` en los perimetrales. Ambos se
    # modelan porque ambos circulan; unificarlos es trabajo del HITO 8, no
    # de este hito, cuyo objetivo es formalizar lo que hay.
    image: Optional[bytes] = None
    processed_image: Optional[bytes] = None

    # Igual con las detecciones: `tracks` en tienda, `metadata.detections`
    # en perimetrales.
    tracks: Optional[List[Deteccion]] = None
    metadata: Optional[Dict[str, Any]] = None

    detections_in_roi: Optional[int] = None
    valid_tracks_in_roi: Optional[int] = None


class ConnectionInit(BaseModel):
    """Servidor -> cliente, al aceptar la conexion (`app.py:647`)."""
    model_config = ConfigDict(extra='forbid')

    id_connection: int
    roi: bool = False


class Heartbeat(BaseModel):
    """Latido en cualquier direccion. Hoy el servidor manda
    `{"status": "ping"}` cada `WEBSOCKET_TIMEOUT` sin trafico."""
    model_config = ConfigDict(extra='forbid')

    status: str = "ping"


class ErrorEvento(BaseModel):
    """Error estructurado (`app.py:_send_error`)."""
    model_config = ConfigDict(extra='forbid')

    status: str = "error"
    message: str
    codigo: Optional[str] = None


# Que modelo valida cada `event_type`.
POR_EVENTO: Dict[str, Any] = {
    "frame.inference": FrameInference,
    "frame.result": FrameResult,
    "connection.init": ConnectionInit,
    "heartbeat": Heartbeat,
    "error": ErrorEvento,
}


# NOTA sobre `extra='forbid'`
# ---------------------------
# Hasta el 31-jul-2026 los payloads permitian claves desconocidas
# (`extra='allow'`), a proposito y solo durante la migracion: endurecer antes
# habria roto a los clientes sin actualizar. Con los cuatro clientes migrados
# y el envio compartido en el nucleo (loop_show_result), la forma del payload
# es una sola y conocida, asi que una clave desconocida ya no es "un cliente
# viejo": es un error o una deriva del contrato, y debe sonar. Si un campo
# nuevo hace falta, se anade AQUI primero y sube `event_version`.
#
# Cuando los 4 clientes esten migrados (fin del HITO 7), estos modelos deben
# pasar a `extra='forbid'` y las claves sobrantes que aparezcan en el log de
# compatibilidad hay que resolverlas una a una.
