"""KPIs, embudo y agregados del stand (HITO 10).

## La regla que gobierna todo este módulo

**Denominador cero devuelve `null`, nunca 0.** Una tasa de conversión del 0 % y
«todavía no ha pasado nadie» son cosas distintas: la primera dice que el stand
funciona mal, la segunda que no hay dato. Mostrarlas igual lleva a decidir mal,
que es justo lo que este sistema existe para evitar.

## Por qué se precalcula

El dashboard no debe recalcular KPIs en cada request: con una jornada entera
en `stand_visitantes` eso son escaneos completos por cada refresco. El job de
agregación deja los buckets de 15 min ya resueltos en `stand_metricas`, y los
endpoints leen de ahí.

Los conteos SIEMPRE salen de `stand_visitantes` (una fila por persona única),
nunca de `stand_eventos` — una persona genera muchos eventos y contarlos
inflaría todo. Ver AUDITORIA_STAND.md §4.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

ORDEN_EMBUDO = ('pasaron', 'se_detuvieron', 'atendidos', 'degustaron')

ETIQUETA_ESCALON = {
    'pasaron': 'Pasaron por delante',
    'se_detuvieron': 'Se detuvieron',
    'atendidos': 'Atendidos por el chef',
    'degustaron': 'Degustaron el producto',
}


def tasa(numerador: float, denominador: float) -> Optional[float]:
    """Cociente, o None si no hay denominador. Nunca 0 por defecto."""
    if not denominador:
        return None
    return round(float(numerador) / float(denominador), 4)


def _media(valores: Sequence[float]) -> Optional[float]:
    limpios = [float(v) for v in valores if v is not None]
    if not limpios:
        return None
    return round(sum(limpios) / len(limpios), 2)


def contadores(visitantes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Las cuatro cifras del embudo a partir de la lista de visitantes."""
    return {
        'pasaron': len(visitantes),
        'se_detuvieron': sum(1 for v in visitantes if v.get('se_detuvo')),
        'atendidos': sum(1 for v in visitantes if v.get('fue_atendido')),
        'degustaron': sum(1 for v in visitantes if v.get('degusto')),
    }


def kpis(visitantes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bloque completo de KPIs de una sesión (o de un subconjunto)."""
    c = contadores(visitantes)
    detenidos = [v for v in visitantes if v.get('se_detuvo')]
    atendidos = [v for v in visitantes if v.get('fue_atendido')]

    return {
        **c,
        'tasa_captacion': tasa(c['se_detuvieron'], c['pasaron']),
        'tasa_atencion': tasa(c['atendidos'], c['se_detuvieron']),
        'tasa_degustacion': tasa(c['degustaron'], c['atendidos']),
        'tasa_global': tasa(c['degustaron'], c['pasaron']),
        'tiempo_medio_permanencia_s': _media(
            [v.get('dwell_max_s') for v in detenidos]),
        'tiempo_medio_atencion_s': _media(
            [v.get('duracion_atencion_s') for v in atendidos]),
        'tiempo_medio_estancia_s': _media(
            [v.get('duracion_total_s') for v in visitantes]),
    }


def embudo(visitantes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Escalones con su caída absoluta y porcentual.

    Marca el escalón de MAYOR pérdida: es la respuesta directa a «dónde
    estamos fallando», y el dashboard lo resalta sin que nadie tenga que
    comparar cuatro números a ojo.
    """
    c = contadores(visitantes)
    escalones: List[Dict[str, Any]] = []
    peor_idx, peor_caida = None, -1

    for i, clave in enumerate(ORDEN_EMBUDO):
        valor = c[clave]
        if i == 0:
            escalones.append({
                'clave': clave, 'etiqueta': ETIQUETA_ESCALON[clave],
                'valor': valor, 'caida_absoluta': None,
                'caida_pct': None, 'conversion_desde_anterior': None,
            })
            continue
        previo = c[ORDEN_EMBUDO[i - 1]]
        caida = previo - valor
        caida_pct = tasa(caida, previo)
        escalones.append({
            'clave': clave, 'etiqueta': ETIQUETA_ESCALON[clave],
            'valor': valor, 'caida_absoluta': caida,
            'caida_pct': caida_pct,
            'conversion_desde_anterior': tasa(valor, previo),
        })
        if caida > peor_caida:
            peor_caida, peor_idx = caida, i

    return {
        'escalones': escalones,
        'mayor_perdida': (ORDEN_EMBUDO[peor_idx]
                          if peor_idx is not None and peor_caida > 0 else None),
        'mayor_perdida_absoluta': peor_caida if peor_caida > 0 else None,
    }


def demografia(visitantes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribuciones y el cruce género × rango etario.

    El cruce es la tabla que dice a qué segmento le está conectando el
    producto — el dato accionable de todo el dashboard.
    """
    dist_genero: Dict[str, int] = {}
    dist_rango: Dict[str, int] = {}
    cruce: Dict[str, Dict[str, Dict[str, int]]] = {}
    sin_demografia = 0

    for v in visitantes:
        if not v.get('demografia_valida'):
            sin_demografia += 1
            continue
        g = v.get('genero') or '?'
        r = v.get('rango_etario') or '?'
        dist_genero[g] = dist_genero.get(g, 0) + 1
        dist_rango[r] = dist_rango.get(r, 0) + 1
        celda = cruce.setdefault(g, {}).setdefault(
            r, {'personas': 0, 'atendidos': 0, 'degustaron': 0})
        celda['personas'] += 1
        if v.get('fue_atendido'):
            celda['atendidos'] += 1
        if v.get('degusto'):
            celda['degustaron'] += 1

    # Tasa de degustación por celda del cruce.
    for g, rangos in cruce.items():
        for r, celda in rangos.items():
            celda['tasa_degustacion'] = tasa(celda['degustaron'],
                                             celda['atendidos'])
            celda['tasa_global'] = tasa(celda['degustaron'], celda['personas'])

    total = len(visitantes)
    return {
        'distribucion_genero': dist_genero,
        'distribucion_rango_etario': dist_rango,
        'cruce_genero_edad': cruce,
        'sin_demografia': sin_demografia,
        'null_rate': tasa(sin_demografia, total),
        'total': total,
    }


def _bucket_de(marca: Optional[str], minutos: int = 15) -> Optional[str]:
    """Marca de tiempo -> etiqueta de bucket. Acepta ISO o 'Y-m-d H:M:S'."""
    if not marca:
        return None
    for formato in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            dt = datetime.strptime(str(marca)[:19], formato)
            break
        except ValueError:
            continue
    else:
        return None
    minuto = (dt.minute // minutos) * minutos
    return dt.replace(minute=minuto, second=0,
                      microsecond=0).strftime('%Y-%m-%dT%H:%M')


def temporal(visitantes: List[Dict[str, Any]],
             minutos: int = 15) -> List[Dict[str, Any]]:
    """Serie por bucket: tráfico, atenciones y degustaciones.

    Identifica horas pico y horas muertas. Se agrupa por la PRIMERA detección
    de cada persona: es cuando entró, no cuando se fue.
    """
    buckets: Dict[str, Dict[str, int]] = {}
    for v in visitantes:
        b = _bucket_de(v.get('primera_deteccion'), minutos)
        if b is None:
            continue
        d = buckets.setdefault(b, {'pasaron': 0, 'se_detuvieron': 0,
                                   'atendidos': 0, 'degustaron': 0})
        d['pasaron'] += 1
        if v.get('se_detuvo'):
            d['se_detuvieron'] += 1
        if v.get('fue_atendido'):
            d['atendidos'] += 1
        if v.get('degusto'):
            d['degustaron'] += 1

    return [{'bucket': b, **d,
             'tasa_global': tasa(d['degustaron'], d['pasaron'])}
            for b, d in sorted(buckets.items())]


def trafico_perdido(visitantes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gente que se detuvo y se fue sin que la atendieran.

    Es la cifra de oportunidad perdida: si es alta, o el chef no da abasto o
    no está viendo a quien se acerca.
    """
    perdidos = [v for v in visitantes
                if v.get('se_detuvo') and not v.get('fue_atendido')]
    detenidos = sum(1 for v in visitantes if v.get('se_detuvo'))
    return {
        'personas': len(perdidos),
        'tasa_sobre_detenidos': tasa(len(perdidos), detenidos),
        'permanencia_media_s': _media(
            [v.get('dwell_max_s') for v in perdidos]),
    }


def calidad_dato(visitantes: List[Dict[str, Any]],
                 descartados: Optional[List[Dict[str, Any]]] = None,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Panel de auditoría. Ninguna cifra es definitiva sin esto al lado."""
    descartados = descartados or []
    por_motivo: Dict[str, int] = {}
    for d in descartados:
        m = d.get('motivo_descarte') or d.get('motivo') or '?'
        por_motivo[m] = por_motivo.get(m, 0) + 1

    demo = demografia(visitantes)
    return {
        'visitantes_contados': len(visitantes),
        'null_rate_demografia': demo['null_rate'],
        'sin_demografia': demo['sin_demografia'],
        'tracks_descartados': len(descartados),
        'descartes_por_motivo': por_motivo,
        **(extra or {}),
    }


def resumen_sesion(visitantes: List[Dict[str, Any]],
                   descartados: Optional[List[Dict[str, Any]]] = None,
                   extra_calidad: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """Todo lo que consume el dashboard, en una sola pasada."""
    return {
        'kpis': kpis(visitantes),
        'embudo': embudo(visitantes),
        'demografia': demografia(visitantes),
        'temporal': temporal(visitantes),
        'trafico_perdido': trafico_perdido(visitantes),
        'calidad_dato': calidad_dato(visitantes, descartados, extra_calidad),
    }
