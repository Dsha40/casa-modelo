"""
Pruebas de los agregados, el emisor y la API REST del stand (HITO 10).

La regla que más se prueba aquí: **denominador cero devuelve None, nunca 0**.
Una tasa del 0 % («el stand funciona mal») y «todavía no ha pasado nadie» son
cosas distintas, y mostrarlas igual lleva a decidir mal.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import stand_metricas as m  # noqa: E402
from src.analityc.core.stand_almacen import (  # noqa: E402
    AlmacenStand, Clasificacion)
from src.analityc.core.stand_emisor import EmisorStand  # noqa: E402


def _v(pid, **campos):
    base = {'persistent_id': pid, 'se_detuvo': 0, 'fue_atendido': 0,
            'degusto': 0, 'demografia_valida': 0, 'dwell_max_s': 0.0,
            'duracion_atencion_s': 0.0, 'duracion_total_s': 0.0,
            'primera_deteccion': '2026-08-05 15:07:00'}
    base.update(campos)
    return base


def _poblacion():
    """10 visitantes: 5 de largo, 3 detenidos, 2 atendidos (1 degusta)."""
    vis = [_v(f'l{i}', duracion_total_s=4.0) for i in range(5)]
    vis += [_v(f'd{i}', se_detuvo=1, dwell_max_s=6.0) for i in range(3)]
    vis.append(_v('a0', se_detuvo=1, fue_atendido=1, dwell_max_s=20.0,
                  duracion_atencion_s=15.0, demografia_valida=1,
                  genero='hombre', rango_etario='36-50'))
    vis.append(_v('a1', se_detuvo=1, fue_atendido=1, degusto=1,
                  dwell_max_s=30.0, duracion_atencion_s=25.0,
                  demografia_valida=1, genero='mujer', rango_etario='26-35'))
    return vis


# ── KPIs y tasas ────────────────────────────────────────────────────────
def test_tasa_con_denominador_cero_es_none():
    assert m.tasa(0, 0) is None
    assert m.tasa(5, 0) is None
    assert m.tasa(0, 10) == 0.0        # esto SÍ es un cero real


def test_kpis_sobre_poblacion_conocida():
    k = m.kpis(_poblacion())
    assert (k['pasaron'], k['se_detuvieron'], k['atendidos'],
            k['degustaron']) == (10, 5, 2, 1)
    assert k['tasa_captacion'] == 0.5
    assert k['tasa_atencion'] == 0.4
    assert k['tasa_degustacion'] == 0.5
    assert k['tasa_global'] == 0.1
    assert k['tiempo_medio_atencion_s'] == 20.0


def test_kpis_de_sesion_vacia_no_inventan_ceros():
    k = m.kpis([])
    assert k['pasaron'] == 0
    for clave in ('tasa_captacion', 'tasa_atencion', 'tasa_degustacion',
                  'tasa_global'):
        assert k[clave] is None, f"{clave} debería ser None sin datos"


# ── Embudo ──────────────────────────────────────────────────────────────
def test_embudo_marca_el_escalon_de_mayor_perdida():
    e = m.embudo(_poblacion())
    assert [x['valor'] for x in e['escalones']] == [10, 5, 2, 1]
    # De 10 a 5 se pierden 5; de 5 a 2, 3; de 2 a 1, 1.
    assert e['mayor_perdida'] == 'se_detuvieron'
    assert e['mayor_perdida_absoluta'] == 5
    assert e['escalones'][1]['caida_pct'] == 0.5
    assert e['escalones'][0]['caida_absoluta'] is None   # el primero no cae


def test_embudo_vacio_no_marca_perdida():
    e = m.embudo([])
    assert e['mayor_perdida'] is None


# ── Demografía ──────────────────────────────────────────────────────────
def test_demografia_y_cruce():
    d = m.demografia(_poblacion())
    assert d['distribucion_genero'] == {'hombre': 1, 'mujer': 1}
    assert d['sin_demografia'] == 8
    assert d['null_rate'] == 0.8
    # El cruce: la mujer de 26-35 fue atendida y degustó.
    celda = d['cruce_genero_edad']['mujer']['26-35']
    assert celda['personas'] == 1 and celda['degustaron'] == 1
    assert celda['tasa_degustacion'] == 1.0
    # El hombre de 36-50 fue atendido y NO degustó.
    assert d['cruce_genero_edad']['hombre']['36-50']['tasa_degustacion'] == 0.0


# ── Serie temporal ──────────────────────────────────────────────────────
def test_temporal_agrupa_en_buckets():
    vis = [_v('a', primera_deteccion='2026-08-05 15:07:00'),
           _v('b', primera_deteccion='2026-08-05 15:14:59'),
           _v('c', primera_deteccion='2026-08-05 15:16:00', se_detuvo=1)]
    serie = m.temporal(vis, minutos=15)
    assert len(serie) == 2
    assert serie[0]['bucket'] == '2026-08-05T15:00' and serie[0]['pasaron'] == 2
    assert serie[1]['bucket'] == '2026-08-05T15:15' and serie[1]['pasaron'] == 1
    assert serie[1]['se_detuvieron'] == 1


def test_temporal_ignora_marcas_ilegibles():
    assert m.temporal([_v('x', primera_deteccion='no es fecha')]) == []
    assert m.temporal([_v('x', primera_deteccion=None)]) == []


# ── Tráfico perdido y calidad ───────────────────────────────────────────
def test_trafico_perdido():
    t = m.trafico_perdido(_poblacion())
    assert t['personas'] == 3            # detenidos sin atender
    assert t['tasa_sobre_detenidos'] == 0.6
    assert t['permanencia_media_s'] == 6.0


def test_calidad_dato_incluye_descartes():
    q = m.calidad_dato(
        _poblacion(),
        descartados=[{'motivo_descarte': 'track_corto'},
                     {'motivo_descarte': 'track_corto'},
                     {'motivo_descarte': 'nunca_en_zona'}])
    assert q['visitantes_contados'] == 10
    assert q['null_rate_demografia'] == 0.8
    assert q['tracks_descartados'] == 3
    assert q['descartes_por_motivo']['track_corto'] == 2


# ── Emisor ──────────────────────────────────────────────────────────────
def test_emisor_respeta_el_throttling():
    e = EmisorStand(periodo_contadores=2.0, periodo_salud=10.0)
    c = {'pasaron': 5, 'se_detuvieron': 3, 'atendidos': 2, 'degustaron': 1}

    b = e.construir(contadores=c, salud={'fps': 20}, ahora=0.0)
    assert 'contadores' in b and 'salud' in b

    # A los 0.5 s no toca ninguno de los dos.
    assert e.construir(contadores=c, salud={'fps': 20}, ahora=0.5) is None
    # A los 2 s tocan los contadores, pero la salud aún no.
    b = e.construir(contadores=c, salud={'fps': 20}, ahora=2.0)
    assert 'contadores' in b and 'salud' not in b
    # A los 10 s toca también la salud.
    b = e.construir(contadores=c, salud={'fps': 20}, ahora=10.0)
    assert 'salud' in b


def test_emisor_manda_totales_absolutos_no_deltas():
    """Así una reconexión se autocorrige sola con el siguiente mensaje."""
    e = EmisorStand(periodo_contadores=0.0)
    b1 = e.construir(contadores={'pasaron': 5, 'se_detuvieron': 0,
                                 'atendidos': 0, 'degustaron': 0}, ahora=0.0)
    b2 = e.construir(contadores={'pasaron': 9, 'se_detuvieron': 0,
                                 'atendidos': 0, 'degustaron': 0}, ahora=1.0)
    assert b1['contadores']['pasaron'] == 5
    assert b2['contadores']['pasaron'] == 9      # total, no "+4"


def test_emisor_calcula_tasas_con_none():
    e = EmisorStand(periodo_contadores=0.0)
    b = e.construir(contadores={'pasaron': 0, 'se_detuvieron': 0,
                                'atendidos': 0, 'degustaron': 0}, ahora=0.0)
    assert b['contadores']['tasas']['captacion'] is None


def test_feed_de_eventos_se_acota_a_diez():
    e = EmisorStand(periodo_contadores=0.0)
    for i in range(15):
        e.anotar_evento('degustacion', genero='mujer', rango_etario='26-35')
    b = e.construir(ahora=0.0)
    assert len(b['ultimos_eventos']) == 10
    # Sin fotos: privacidad (regla 7).
    assert 'crop_image' not in b['ultimos_eventos'][0]
    assert set(b['ultimos_eventos'][0]) >= {'tipo', 'ts', 'genero',
                                            'rango_etario'}


def test_feed_solo_se_manda_cuando_cambia():
    e = EmisorStand(periodo_contadores=0.0)
    e.anotar_evento('atencion')
    assert 'ultimos_eventos' in e.construir(ahora=0.0)
    # Sin eventos nuevos, no se reenvía la lista entera.
    assert 'ultimos_eventos' not in (e.construir(ahora=1.0) or {})


def test_sesion_id_viaja_en_todo_mensaje():
    """El cliente lo necesita para construir la URL del POST de personal."""
    e = EmisorStand(periodo_contadores=0.0, sesion_id=7)
    b = e.construir(contadores={'pasaron': 1, 'se_detuvieron': 0,
                                'atendidos': 0, 'degustaron': 0}, ahora=0.0)
    assert b['sesion_id'] == 7
    # Sin sesión asignada, la clave no aparece (no se inventa un 0).
    e2 = EmisorStand(periodo_contadores=0.0)
    b2 = e2.construir(contadores={'pasaron': 1, 'se_detuvieron': 0,
                                  'atendidos': 0, 'degustaron': 0}, ahora=0.0)
    assert 'sesion_id' not in b2


def test_tracks_activos_solo_llevan_lo_minimo():
    """Ni embeddings ni recortes: solo lo justo para señalar a alguien."""
    e = EmisorStand(periodo_tracks=0.0, sesion_id=1)
    b = e.construir(tracks=[{
        'persistent_id': 'p-1', 'genero': 'hombre', 'rango_etario': '36-50',
        'zona': 'zona_personal', 'es_personal': False, 'dwell_s': 612.44,
        'embedding': [0.1] * 512, 'crop': 'base64...',
    }], ahora=0.0)
    t = b['tracks_activos'][0]
    assert set(t) == {'persistent_id', 'genero', 'rango_etario', 'zona',
                      'es_personal', 'dwell_s'}
    assert 'embedding' not in t and 'crop' not in t
    assert t['dwell_s'] == 612.4


def test_tracks_provisionales_no_se_envian():
    """Un `prov:` cambia de id al confirmarse: enrolarlo apuntaría a nadie."""
    e = EmisorStand(periodo_tracks=0.0, sesion_id=1)
    b = e.construir(tracks=[{'persistent_id': 'prov:17'},
                            {'persistent_id': 'p-ok'},
                            {'persistent_id': ''}], ahora=0.0)
    assert [t['persistent_id'] for t in b['tracks_activos']] == ['p-ok']


def test_tracks_se_acotan_para_que_el_selector_sea_usable():
    e = EmisorStand(periodo_tracks=0.0, sesion_id=1)
    b = e.construir(tracks=[{'persistent_id': f'p-{i}'} for i in range(50)],
                    ahora=0.0)
    assert len(b['tracks_activos']) == 20


def test_tracks_respetan_su_propio_ritmo():
    e = EmisorStand(periodo_tracks=2.0, sesion_id=1)
    assert 'tracks_activos' in e.construir(tracks=[{'persistent_id': 'p'}],
                                           ahora=0.0)
    assert e.construir(tracks=[{'persistent_id': 'p'}], ahora=0.5) is None
    assert 'tracks_activos' in e.construir(tracks=[{'persistent_id': 'p'}],
                                           ahora=2.0)


def test_adjuntar_no_pisa_el_metadata_existente():
    """Regla 2: aditivo. Tienda y Perimetrales no deben enterarse."""
    e = EmisorStand(periodo_contadores=0.0)
    meta = {'detections': [1, 2, 3], 'alerts': ['x'], 'heatmap': 'algo'}
    salida = e.adjuntar(meta, contadores={'pasaron': 1, 'se_detuvieron': 0,
                                          'atendidos': 0, 'degustaron': 0},
                        ahora=0.0)
    assert salida['detections'] == [1, 2, 3]
    assert salida['alerts'] == ['x']
    assert salida['heatmap'] == 'algo'
    assert 'stand' in salida


def test_adjuntar_no_ensucia_si_no_toca_emitir():
    e = EmisorStand(periodo_contadores=100.0)
    c = {'pasaron': 1, 'se_detuvieron': 0, 'atendidos': 0, 'degustaron': 0}
    meta = {'detections': []}
    e.adjuntar(meta, contadores=c, ahora=0.0)
    meta2 = {'detections': []}
    e.adjuntar(meta2, contadores=c, ahora=1.0)
    assert 'stand' not in meta2, "emitió fuera de su ritmo"


def test_heatmap_solo_se_serializa_cuando_toca():
    llamadas = []

    def construir_mapa():
        llamadas.append(1)
        return {'tipo': 'permanencia', 'matriz': [[0]]}

    e = EmisorStand(periodo_heatmap=30.0)
    e.construir(heatmap=construir_mapa, ahora=0.0)
    e.construir(heatmap=construir_mapa, ahora=5.0)
    e.construir(heatmap=construir_mapa, ahora=10.0)
    assert len(llamadas) == 1, "serializó la matriz fuera de su ritmo"


# ── Integración con el almacén ──────────────────────────────────────────
def test_contadores_del_almacen_coinciden_con_las_metricas():
    """El contador SQL y el cálculo en Python no pueden discrepar."""
    with tempfile.TemporaryDirectory() as d:
        alm = AlmacenStand(ruta=Path(d) / 'stand.db')
        sid = alm.abrir_sesion(stand_id='s1')
        for v in _poblacion():
            alm.upsert_visitante(
                sid, v['persistent_id'], se_detuvo=v['se_detuvo'],
                fue_atendido=v['fue_atendido'], degusto=v['degusto'])
        # El chef no debe aparecer en ninguno de los dos.
        alm.upsert_visitante(sid, 'chef', es_personal=True, se_detuvo=True,
                             fue_atendido=True, degusto=True)

        sql = alm.contadores(sid)
        py = m.contadores(alm.visitantes(sid))
        assert sql == py
        assert sql['pasaron'] == 10


def test_clasificaciones_del_almacen_son_las_del_embudo():
    assert set(Clasificacion.TODAS) == {
        'paso_de_largo', 'se_detuvo_sin_atencion', 'atendido_sin_degustar',
        'degusto'}


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
if __name__ == '__main__':
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        try:
            _fn()
            print(f"  OK    {_nombre}")
        except Exception as _exc:                          # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)


# ── El frame en vivo (añadido el 6-ago-2026) ────────────────────────────
#
# `/api/stand/frame` devolvía 503 SIEMPRE, aunque el cliente estuviera
# enviando vídeo: leía `cam_proc.last_processed_frame`, y ese atributo vuelve
# a None al salir de process_frame porque `_pop_state` restaura el valor
# anterior. El frame se queda en `camera_states[camera_id]`.

def test_el_frame_se_busca_tambien_en_el_estado_por_camara():
    import numpy as np
    from src.app.stand_api import _ultimo_frame

    class _Proc:
        last_processed_frame = None
        camera_states = {'cam1': {'last_processed_frame': np.zeros((4, 4, 3),
                                                                   np.uint8)}}
    assert _ultimo_frame(_Proc()) is not None


def test_sin_frame_en_ninguno_de_los_dos_sitios_devuelve_None():
    from src.app.stand_api import _ultimo_frame

    class _Vacio:
        last_processed_frame = None
        camera_states = {'cam1': {'last_processed_frame': None}}
    assert _ultimo_frame(_Vacio()) is None
    assert _ultimo_frame(object()) is None


def test_cuenta_solo_las_personas():
    """Distinguir «no detecta a nadie» de «detecta pero el ROI está mal»."""
    from src.app.stand_api import _cuantas_personas

    class _Proc:
        active_tracks = {1: {'class': -1}, 2: {'class': -1}, 3: {'class': 7}}
    assert _cuantas_personas(_Proc()) == 2

    class _PorCamara:
        active_tracks = {}
        camera_states = {'c': {'active_tracks': {9: {'class': -1}}}}
    assert _cuantas_personas(_PorCamara()) == 1
    assert _cuantas_personas(object()) == 0
