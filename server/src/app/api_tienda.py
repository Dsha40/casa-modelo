"""src/app/api_tienda.py — la API de tienda CON VENTANA TEMPORAL (HITO 2).

## Por qué un router propio y no más campos en `/api/v1/resumen`

`/api/v1/resumen` no sabe de fechas: devuelve lo que lleve viva la sesión de
los procesadores conectados. Eso lo leen el dashboard de tienda, el de
amazonas, el de managers y el informe PDF, así que cambiarle el significado a
sus campos rompería tres páginas de golpe para arreglar una.

Aquí viven los endpoints que SÍ entienden de días. Los de siempre se quedan
exactamente como estaban (compatibilidad aditiva): lo nuevo se añade, no se
sustituye. El día que las cuatro páginas hayan migrado, aquellos se podrán
jubilar con una deprecación ordenada.

## De dónde salen los números

De `tienda_almacen` (SQLite con jornada), NO de los procesadores vivos ni de
`analytics_report_*.json`. Por eso estos endpoints responden igual con el
cliente conectado o desconectado, y suman TODAS las sesiones del rango — que
es justo lo que el ranking viejo no hacía (leía un solo informe, el más
reciente por fecha de archivo).

La única excepción es `aforo_actual` («personas ahora»), que por definición es
del instante y sale de los procesadores vivos. Solo se informa cuando la
ventana incluye hoy; en una ventana pasada va a `null`, no a cero: nadie sabe
cuánta gente había dentro el martes a esta hora.

## Semántica escrita (para que ninguna cifra cambie de significado en silencio)

- **visitantes_unicos** de una ventana = `persistent_id` DISTINTOS dentro de
  la ventana. **No** es la suma de los únicos diarios: quien viene lunes y
  martes es UNA persona, y sumando daría dos.
- **entradas** = cruces confirmados del área. Siempre ≥ visitantes únicos.
- **permanencia_media_s** = media del tiempo dentro de quienes entraron, sobre
  las filas de visita del rango.
- **personas** de un tramo horario = gente distinta que entró EN ESE TRAMO;
  sumar tramos no da los únicos del día (quien entra a las 10 y a las 12 está
  en los dos).
- **cobertura**: qué jornadas de la ventana tienen algún dato. Sirve para que
  la página distinga «0 visitas» de «el servidor estaba apagado». Pintadas
  igual, las dos se leen como una caída de tráfico que nunca ocurrió.

Solo GET, como manda la regla de `/api/v1`.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..analityc.core import tienda_almacen as _almacen
from . import registro_dispositivos as _registro
from .api_lectura import _saneado

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v1/tienda', tags=['tienda'])

#: Formato de fecha aceptado en la URL. Cualquier otra cosa es un 422 de
#: FastAPI antes de tocar la base.
_FECHA = r'^\d{4}-\d{2}-\d{2}$'

#: Tope de la ventana. Por encima de la retención (13 meses) no hay nada que
#: leer, y una ventana absurda solo sirve para castigar al servidor.
_MAX_DIAS = 400


def _ventana(desde: Optional[str], hasta: Optional[str]) -> Dict[str, Any]:
    """Normaliza la ventana pedida. Sin parámetros, HOY."""
    hoy = _almacen.jornada_de()
    desde = desde or hoy
    hasta = hasta or desde
    if hasta < desde:
        desde, hasta = hasta, desde     # invertida: se endereza, no se rechaza
    dias = _almacen.dias_de(desde, hasta)
    if dias > _MAX_DIAS:
        raise HTTPException(
            status_code=400,
            detail=f'la ventana pedida son {dias} días y el máximo es '
                   f'{_MAX_DIAS}; la retención de la base son 13 meses')
    return {'desde': desde, 'hasta': hasta, 'dias': dias, 'hoy': hoy}


def _camara(camara: Optional[str]) -> Optional[str]:
    """Mismo criterio de saneado que el resto de la API de lectura."""
    if camara is None:
        return None
    limpio = _saneado(camara)
    return limpio or None


def _aforo_vivo(camara: Optional[str]) -> Optional[int]:
    """Personas DENTRO ahora mismo, de los procesadores conectados.

    `None` (no cero) cuando no hay ninguna cámara transmitiendo: no es que no
    haya nadie en la tienda, es que nadie está mirando.

    Se mira en `sys.modules` en vez de importar: `app.py` arrastra torch y
    ultralytics, y un `import` perezoso dentro de la petición se convertía en
    13 segundos de espera la primera vez que alguien abría el dashboard en un
    proceso que aún no lo tuviera cargado. Si el servidor de inferencia no
    está en este proceso, no hay procesadores vivos que consultar y punto.
    """
    servidor = sys.modules.get(f'{__package__}.app')
    if servidor is None:
        return None
    try:
        procs = list(servidor._iter_person_processors())
    except Exception:                                       # noqa: BLE001
        return None
    if not procs:
        return None
    total = 0
    for _cid, cam_id, proc in procs:
        if camara is not None and str(cam_id) != camara:
            continue
        estado = getattr(proc, '_zone_state', None) or {}
        total += sum(1 for st in estado.values()
                     if st.get('estado') == 'DENTRO')
    return total


def _variacion(actual: float, anterior: float) -> Optional[float]:
    """Variación porcentual, o `None` si no hay base con la que comparar.

    Dividir entre cero no da «+100 %» ni «0 %»: da NADA. La página tiene que
    poder decir «sin base de comparación» en vez de inventarse una mejora.
    """
    if not anterior:
        return None
    return round(100.0 * (actual - anterior) / anterior, 1)


@router.get('/resumen')
def resumen(desde: Optional[str] = Query(None, pattern=_FECHA),
            hasta: Optional[str] = Query(None, pattern=_FECHA),
            camara: Optional[str] = Query(None),
            comparar: bool = Query(True)) -> Dict[str, Any]:
    """KPIs de la ventana, con su comparativa contra el periodo anterior."""
    v = _ventana(desde, hasta)
    cam = _camara(camara)
    alm = _almacen.get_almacen()

    kpis = alm.resumen(v['desde'], v['hasta'], camera_id=cam)
    # «Ahora» solo tiene sentido si la ventana llega hasta hoy.
    kpis['aforo_actual'] = (_aforo_vivo(cam) if v['hasta'] >= v['hoy']
                            else None)

    salida: Dict[str, Any] = {
        'ventana': {
            'desde': v['desde'], 'hasta': v['hasta'], 'dias': v['dias'],
            # Cobertura: sin esto, un día con el servidor apagado se pinta
            # como un día de cero visitas.
            'jornadas_con_datos': alm.jornadas_con_datos(v['desde'],
                                                         v['hasta']),
        },
        'camara': cam,
        'kpis': kpis,
    }
    if comparar:
        ant_desde, ant_hasta = _almacen.ventana_anterior(v['desde'],
                                                         v['hasta'])
        previo = alm.resumen(ant_desde, ant_hasta, camera_id=cam)
        salida['comparativa'] = {
            'ventana': {'desde': ant_desde, 'hasta': ant_hasta},
            'kpis': previo,
            'variacion_pct': {
                clave: _variacion(kpis.get(clave, 0) or 0,
                                  previo.get(clave, 0) or 0)
                for clave in ('entradas', 'salidas', 'visitantes_unicos',
                              'permanencia_media_s')
            },
        }
    return salida


@router.get('/serie')
def serie(desde: Optional[str] = Query(None, pattern=_FECHA),
          hasta: Optional[str] = Query(None, pattern=_FECHA),
          camara: Optional[str] = Query(None),
          grano: str = Query('hora', pattern=r'^(hora|dia)$')) -> Dict[str, Any]:
    """La curva: entradas y salidas por hora (o por día) de la ventana.

    Con `grano=hora` se devuelven las 24 horas SIEMPRE, rellenando con ceros
    las que no tuvieron movimiento: una curva con huecos se dibuja torcida y
    engaña sobre la hora punta.
    """
    v = _ventana(desde, hasta)
    cam = _camara(camara)
    alm = _almacen.get_almacen()
    filas = alm.serie(v['desde'], v['hasta'], camera_id=cam, grano=grano)

    if grano == 'hora':
        # Agregado de toda la ventana por hora del día, que es lo que pinta la
        # curva. El detalle por jornada va aparte, sin rellenar.
        por_hora = {f'{h:02d}': {'hora': f'{h:02d}', 'entradas': 0,
                                 'salidas': 0, 'personas': 0}
                    for h in range(24)}
        for f in filas:
            rec = por_hora.get(str(f['hora'] or '').zfill(2))
            if rec is None:
                continue
            rec['entradas'] += f['entradas']
            rec['salidas'] += f['salidas']
            rec['personas'] += f['personas']
        tramos = list(por_hora.values())
    else:
        tramos = [{'jornada': f['jornada'], 'entradas': f['entradas'],
                   'salidas': f['salidas'], 'personas': f['personas']}
                  for f in filas]

    activos = [t for t in tramos if t['entradas'] > 0]
    punta = max(activos, key=lambda t: t['entradas']) if activos else None
    valle = min(activos, key=lambda t: t['entradas']) if activos else None
    return {
        'ventana': {'desde': v['desde'], 'hasta': v['hasta'],
                    'dias': v['dias'],
                    'jornadas_con_datos': alm.jornadas_con_datos(v['desde'],
                                                                 v['hasta'])},
        'camara': cam,
        'grano': grano,
        'tramos': tramos,
        'detalle_por_jornada': filas if grano == 'hora' else [],
        'punta': punta,
        'valle': valle,
        'total_entradas': sum(t['entradas'] for t in tramos),
    }


@router.get('/pasillos')
def pasillos(desde: Optional[str] = Query(None, pattern=_FECHA),
             hasta: Optional[str] = Query(None, pattern=_FECHA),
             comparar: bool = Query(True)) -> Dict[str, Any]:
    """Ranking de cámaras/pasillos de la ventana, con quién sube y quién baja.

    Suma TODAS las sesiones del rango (el ranking viejo leía solo el informe
    más reciente, así que el pasillo que se desconectó a media mañana aparecía
    con la mitad de su tráfico).

    Las cámaras que no están dadas de alta en el registro **no se ocultan**:
    salen con `registrada: false`, como hacen los mapas de calor con las
    huérfanas. Un dato que existe y no se ve es peor que uno feo.
    """
    v = _ventana(desde, hasta)
    alm = _almacen.get_almacen()
    filas = alm.por_camara(v['desde'], v['hasta'])

    nombres: Dict[str, Dict[str, Any]] = {
        str(d['device_id']): d for d in _registro.dispositivos()}

    previo: Dict[str, Dict[str, Any]] = {}
    ant_desde = ant_hasta = None
    if comparar:
        ant_desde, ant_hasta = _almacen.ventana_anterior(v['desde'],
                                                         v['hasta'])
        previo = {f['camera_id']: f
                  for f in alm.por_camara(ant_desde, ant_hasta)}

    ranking: List[Dict[str, Any]] = []
    for f in filas:
        cam = f['camera_id']
        ficha = nombres.get(cam) or {}
        antes = previo.get(cam) or {}
        ranking.append({
            **f,
            'camera_name': ficha.get('camera_name') or cam,
            'registrada': bool(ficha),
            'client_type': ficha.get('client_type'),
            'entradas_previas': antes.get('entradas'),
            'variacion_pct': (_variacion(f['entradas'],
                                         antes.get('entradas', 0) or 0)
                              if comparar else None),
        })
    return {
        'ventana': {'desde': v['desde'], 'hasta': v['hasta'],
                    'dias': v['dias']},
        'comparativa': ({'desde': ant_desde, 'hasta': ant_hasta}
                        if comparar else None),
        'total': len(ranking),
        'ranking': ranking,
        'mas_concurrido': ranking[0] if ranking else None,
        # Con una sola cámara medida no hay «menos concurrido»: sería la misma.
        'menos_concurrido': ranking[-1] if len(ranking) > 1 else None,
    }


@router.get('/demografia')
def demografia(desde: Optional[str] = Query(None, pattern=_FECHA),
               hasta: Optional[str] = Query(None, pattern=_FECHA),
               camara: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Género y edad de la ventana, global y POR PASILLO.

    El desglose por pasillo era imposible hasta el HITO 1: la captura solo
    guardaba el nombre visible de la ventana, y cuatro cámaras pueden
    llamarse «iVMS-4200». Ahora la demografía se guarda con el `device_id`.
    """
    v = _ventana(desde, hasta)
    cam = _camara(camara)
    datos = _almacen.get_almacen().demografia(v['desde'], v['hasta'],
                                              camera_id=cam)
    nombres = {str(d['device_id']): d.get('camera_name')
               for d in _registro.dispositivos()}
    por_camara = [
        {'camera_id': cid, 'camera_name': nombres.get(cid) or cid,
         'genero': rec['genero'], 'edad': rec['edad']}
        for cid, rec in sorted(datos['por_camara'].items())
    ]
    return {
        'ventana': {'desde': v['desde'], 'hasta': v['hasta'],
                    'dias': v['dias']},
        'camara': cam,
        'genero': datos['genero'],
        'edad': datos['edad'],
        'por_camara': por_camara,
    }


@router.get('/clasificaciones')
def clasificaciones(desde: Optional[str] = Query(None, pattern=_FECHA),
                    hasta: Optional[str] = Query(None, pattern=_FECHA),
                    camara: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Clasificados de la ventana, separados por si se les reconoció o no.

    `/demografia` solo ve a quien tiene identidad, porque sale de las
    visitas. Aquí están TODOS los clasificados (24-sep-2026): ese día, 3 de 3
    no tenían identidad y `/demografia` no los veía. `con_identidad` cuenta
    personas; `sin_identidad` cuenta capturas, porque sin identidad no hay
    forma de saber que dos capturas son la misma persona.
    """
    v = _ventana(desde, hasta)
    cam = _camara(camara)
    datos = _almacen.get_almacen().clasificaciones(v['desde'], v['hasta'],
                                                   camera_id=cam)
    return {
        'ventana': {'desde': v['desde'], 'hasta': v['hasta'],
                    'dias': v['dias']},
        'camara': cam,
        **datos,
    }


@router.get('/capturas')
def capturas(desde: Optional[str] = Query(None, pattern=_FECHA),
             hasta: Optional[str] = Query(None, pattern=_FECHA),
             camara: Optional[str] = Query(None),
             limite: int = Query(60, ge=1, le=500),
             offset: int = Query(0, ge=0)) -> Dict[str, Any]:
    """La galería, acotada por ventana y por cámara, y PAGINADA.

    El endpoint viejo (`/api/v1/capturas`) no sabe de fechas y su tope de 500
    era el techo de la galería. Este lee del índice, así que su coste no crece
    con el número de fotos acumuladas, y `offset` permite pasar página de
    verdad en vez de pedir «las 500 más recientes» una y otra vez.
    """
    from ..analityc.core import tienda_capturas as _indice
    from .api_lectura import _saneado as _limpio

    v = _ventana(desde, hasta) if (desde or hasta) else None
    datos = _indice.listar(
        limite=limite, offset=offset,
        desde=(v['desde'] if v else None), hasta=(v['hasta'] if v else None),
        camara=_camara(camara))
    for c in datos['capturas']:
        stem = _limpio(c.get('stem') or '')
        c['url'] = f'/dashboard/img/capture/{stem}.jpg'
        c['url_miniatura'] = (f'/dashboard/img/capface/{stem}.jpg'
                              if c.get('has_face') else c['url'])
    return {
        'ventana': ({'desde': v['desde'], 'hasta': v['hasta'],
                     'dias': v['dias']} if v else None),
        'camara': _camara(camara),
        'total': datos['total'],
        'limite': limite,
        'offset': offset,
        'capturas': datos['capturas'],
    }


@router.get('/camaras')
def camaras() -> Dict[str, Any]:
    """Las cámaras con datos guardados, para poblar el filtro de la página."""
    nombres = {str(d['device_id']): d for d in _registro.dispositivos()}
    filas = []
    for cid in _almacen.get_almacen().camaras():
        ficha = nombres.get(cid) or {}
        filas.append({
            'camera_id': cid,
            'camera_name': ficha.get('camera_name') or cid,
            'registrada': bool(ficha),
            'client_type': ficha.get('client_type'),
        })
    return {'total': len(filas), 'camaras': filas}
