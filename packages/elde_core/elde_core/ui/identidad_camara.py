"""
Identidad estable de una camara. Primera rebanada del despiece de `render_box`.

El problema (H-11)
------------------
Los cuatro clientes mandaban como `camera_id` su `component_key`, que es un
`uuid.uuid4()` generado al **construir el panel**. Cada arranque de la
aplicacion inventaba camaras nuevas a ojos del servidor, asi que los heatmaps,
el conteo y la demografia se fragmentaban en un UUID por sesion: no habia
historico por zona. El planograma tambien se guardaba por ese id, asi que las
zonas dibujadas se perdian en cada reinicio.

Se corrigio en tienda el 30-jul-2026, con la idea de replicarlo cuando
`render_box` pasara al nucleo. Como el despiece se retraso, los otros tres
clientes se quedaron con el fallo. Esto es esa parte del despiece: la identidad
vive **una vez** y los cuatro la usan.

`component_key` no desaparece: sigue siendo la clave de enrutado del widget (el
servidor no la usa para nada).

La segunda leccion (H-39, 8-ago-2026)
-------------------------------------
La primera version daba a las ventanas capturadas la identidad del TITULO a
secas, con la idea de que dos recuadros mostrando la misma camara compartieran
`device_id`. El caso que no se previo: dos recuadros capturando dos ventanas
DISTINTAS del mismo programa — iVMS-4200 permite abrir varias vistas y todas se
titulan igual. Ambos recuadros mandaban `win-iVMS-4200`, el servidor los trataba
como UNA camara y el rastreador recibia las dos escenas entrelazadas: cocina,
generador, cocina... Medido con el pipeline real: 37/40 frames con tracks al ver
una sola escena, **0/40** con las dos alternadas. Ceguera total, sin ningun
error.

Por eso el indice del recuadro forma parte de la identidad de ventana desde el
recuadro 2 en adelante: `win-iVMS-4200` (recuadro 1), `win-iVMS-4200-c2`
(recuadro 2)... El primero conserva su id historico; los demas se separan. El
coste asumido: dos recuadros que capturen la MISMA ventana ya no comparten
historico — eso duplica datos, mientras que compartirlo con escenas distintas
los corrompe todos.

La tercera leccion (9-sep-2026)
-------------------------------
El sufijo del recuadro separa ventanas DENTRO de un cliente. No separa dos
CLIENTES distintos, y ese caso llego: una misma PC con dos iVMS-4200 abiertos
—la 3.1 y la 2.8, que ponen el mismo titulo— capturados por dos clientes, cada
uno con su propio «recuadro 1». Los dos mandaban `win-iVMS-4200`.

En el servidor se vio asi, varias veces por segundo:

    local de 'win-iVMS-4200': 'Camaras CY Aruba' -> 'Carpinteria'
    local de 'win-iVMS-4200': 'Carpinteria' -> 'Camaras CY Aruba'

Dos locales distintos peleandose la misma llave: un solo rastreador para dos
escenas, un solo contador de area y alertas atribuidas al local equivocado. Lo
delato la puerta de cruce, que se ponia y se quitaba en cada frame (357 veces
en dos minutos) y por eso no confirmaba un solo paso; el entrelazado llevaba
ahi desde antes, callado.

Se arregla con `instalacion`: un identificador corto y ESTABLE de cada
instalacion del cliente (ver `id_instalacion`), que entra en la identidad de
las camaras por VENTANA. No toca las de DVR, que ya son unicas por serie y
canal, y es opcional: sin el, `device_id` se comporta exactamente igual que
antes. El coste, otra vez el mismo y por la misma razon: al estrenarlo, esas
camaras cambian de id UNA vez y parten su historico. Duplicar datos es malo;
mezclar dos locales en el mismo marcador es peor.
"""
from __future__ import annotations

from typing import Any

#: Longitud maxima por defecto. El valor acaba siendo un nombre de archivo.
LIMITE = 64


def slug(texto: str, limite: int = LIMITE) -> str:
    """Deja el texto apto para identificador Y para nombre de archivo.

    Importa porque el `device_id` acaba siendo el nombre de los heatmaps
    (`output/heatmap/<device_id>.png`) y de las capturas.

    Se admiten letras, digitos, `_`, `-` y puntos sueltos. Deliberadamente MAS
    restrictivo que el contrato del HITO 3, que tambien permitiria `:`: los dos
    puntos son validos en el envelope pero **ilegales en un nombre de archivo de
    Windows**, y este valor termina siendo uno. Ademas se colapsan los puntos
    seguidos, para que ningun `..` sobreviva y nadie pueda salirse de la
    carpeta.
    """
    limpio = ''.join(
        c if (c.isalnum() or c in '_-.') else '_' for c in (texto or ''))
    while '..' in limpio:
        limpio = limpio.replace('..', '.')
    return limpio.strip('._-')[:limite].strip('._-') or 'sin_nombre'


#: Nombre del archivo donde vive el identificador de instalacion.
ARCHIVO_INSTALACION = 'instalacion.id'


def id_instalacion(carpeta: Any, largo: int = 6) -> str:
    """Identificador corto y estable de ESTA instalacion del cliente.

    Se guarda en un archivo dentro de la carpeta de configuracion, asi que
    sobrevive a los reinicios: es justo lo que hacia falta y lo que el
    `component_key` (un uuid4 por sesion) no daba (H-11). Se crea la primera
    vez y despues solo se lee.

    Deliberadamente CORTO: este valor acaba dentro de un nombre de archivo
    (`output/heatmap/<device_id>.png`) junto al titulo de la ventana, y un
    uuid entero se comeria el presupuesto de caracteres.

    Nunca lanza. Si la carpeta no se puede leer ni escribir devuelve `''`, y
    entonces `device_id` se comporta como siempre: es preferible seguir con
    el id de antes que quedarse sin identidad.
    """
    try:
        import os
        import uuid
        ruta = os.path.join(str(carpeta), ARCHIVO_INSTALACION)
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                guardado = slug(f.read().strip(), largo)
            if guardado:
                return guardado
        except FileNotFoundError:
            pass
        nuevo = slug(uuid.uuid4().hex[:largo], largo)
        os.makedirs(str(carpeta), exist_ok=True)
        # Escritura atomica: dos clientes de la misma PC pueden arrancar a la
        # vez, y un archivo a medias daria una identidad distinta en cada
        # lectura — que es el fallo que esto viene a cerrar.
        tmp = ruta + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(nuevo)
        os.replace(tmp, ruta)
        return nuevo
    except Exception:                                       # noqa: BLE001
        return ''


def instalacion_de_app(app_name: str) -> str:
    """El identificador de instalacion de un cliente, por su nombre de app.

    Cada cliente guarda su configuracion en su propia carpeta
    (`SettingsModel(app_name=...)`), asi que dos clientes DISTINTOS de la
    misma PC tienen carpetas distintas y por tanto identificadores distintos:
    es lo que separa sus ventanas homonimas.

    LIMITE CONOCIDO: dos copias del MISMO cliente lanzadas en la misma PC
    comparten carpeta, y por tanto identificador. Ese caso sigue chocando y
    hay que separarlo por el indice del recuadro.

    Se cachea: esto se llama por frame y por recuadro, y no puede ir al disco
    cada vez.
    """
    global _CACHE_INSTALACION
    if app_name in _CACHE_INSTALACION:
        return _CACHE_INSTALACION[app_name]
    try:
        from appdirs import user_config_dir
        valor = id_instalacion(user_config_dir(app_name))
    except Exception as exc:                                # noqa: BLE001
        valor = ''
    if not valor:
        # NUNCA en silencio. Sin identificador las camaras por ventana vuelven
        # a su id de antes, que es lo correcto —mejor el historico de siempre
        # que ninguna identidad—, pero dos clientes de la misma PC volverian a
        # chocar y el sintoma seria otra vez mudo: dos escenas en un
        # rastreador y el local saltando en cada frame. Que se vea.
        import logging
        logging.getLogger(__name__).warning(
            "sin identificador de instalacion para '%s': las camaras "
            "capturadas por VENTANA conservan su id de siempre y podrian "
            "chocar con las de otro cliente de esta misma PC", app_name)
    _CACHE_INSTALACION[app_name] = valor
    return valor


#: Memoria del identificador por app: `device_id` se llama por frame.
_CACHE_INSTALACION: dict = {}


def device_id(serie_dvr: str = '',
              canal_dvr: str = '',
              titulo_ventana: str = '',
              indice: int = 0,
              instalacion: str = '') -> str:
    """Identificador ESTABLE de la camara, para el `camera_id` del payload.

    Prioridad, de mas estable a menos:

    1. **Canal DVR** — numero de serie del equipo + canal. Identifica una
       camara fisica y no cambia nunca.
    2. **Ventana capturada** — su titulo, MAS el recuadro a partir del
       segundo (H-39): programas como iVMS-4200 abren varias ventanas con el
       mismo titulo mostrando escenas distintas, y si comparten id el
       servidor entrelaza las escenas en un solo rastreador (que se queda
       ciego: 0/40 frames con tracks, medido). El recuadro 1 conserva el id
       sin sufijo para no partir el historico del caso comun.
    3. **Posicion del recuadro** — ultimo recurso; estable dentro de una misma
       disposicion de la rejilla.

    `instalacion` (9-sep-2026) separa dos CLIENTES distintos que capturan
    ventanas homonimas: el sufijo del recuadro solo separa DENTRO de un
    cliente, y una misma PC con dos iVMS-4200 abiertos mandaba `win-iVMS-4200`
    por duplicado (ver la cabecera del modulo). Solo entra en las camaras por
    VENTANA: las de DVR ya son unicas por serie y canal, y las de posicion no
    identifican nada que merezca historico. Vacio = comportamiento de antes,
    identico, para que estrenarlo sea una decision y no un efecto colateral.
    """
    if serie_dvr or canal_dvr:
        serie = slug(serie_dvr, 40)
        canal = slug(canal_dvr, 12)
        return f'dvr-{serie}-{canal}' if canal else f'dvr-{serie}'

    titulo = (titulo_ventana or '').strip()
    if titulo:
        base = f'win-{slug(titulo, 72)}'
        if int(indice) > 0:
            base = f'{base}-c{int(indice) + 1}'
        # El sufijo de instalacion va el ULTIMO: asi el id sigue empezando por
        # el titulo, que es lo que un operador reconoce de un vistazo en el
        # tablero y en los nombres de archivo.
        # OJO con `slug('')`: devuelve 'sin_nombre', no cadena vacia (es su
        # ultimo recurso para que un id nunca quede vacio). Sin este guardia,
        # el sufijo se colaba SIEMPRE y cambiaba el id de todas las camaras
        # por ventana, que es justo lo que este parametro promete no hacer.
        marca = slug(instalacion, 12) if str(instalacion or '').strip() else ''
        return f'{base}-i{marca}' if marca else base

    return f'box-{int(indice) + 1}'


def nombre_visible(nombre_dvr: str = '',
                   titulo_ventana: str = '',
                   indice: int = 0) -> str:
    """Nombre legible de la camara para el dashboard.

    Es el compañero de `device_id`: uno identifica, el otro se enseña. Van
    juntos porque se deducen de las mismas tres fuentes y en el mismo orden.
    """
    if nombre_dvr:
        return nombre_dvr
    titulo = (titulo_ventana or '').strip()
    if titulo:
        # El mismo sufijo que el device_id (H-39): sin el, el tablero
        # enseñaria dos camaras distintas con el mismo nombre.
        return (titulo[:48] if int(indice) <= 0
                else f'{titulo[:44]} ({int(indice) + 1})')
    return f'Camara {int(indice) + 1}'
