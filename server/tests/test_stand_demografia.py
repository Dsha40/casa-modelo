"""
Pruebas de la agregación demográfica del stand (HITO 5).

La idea que se protege: **un null siempre lleva motivo, y no tener demografía
nunca impide contar a la persona**. Ese desacople es el arreglo del hallazgo
§3.2 de la auditoría, donde un visitante sin género desaparecía entero.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_config import ConfigStand  # noqa: E402
from src.analityc.core.stand_demografia import (  # noqa: E402
    FUENTE_CORPORAL, FUENTE_FACIAL, AgregadorDemografia, Descarte,
    MotivoSinDemografia, rango_de_edad)


def _cfg(**over):
    datos = {'demografia': {
        'buffer': {'max_muestras': 10, 'espaciado_frames': 5,
                   'min_muestras_validas': 3},
        'genero': {'confianza_min': 0.65},
        'peso_muestra_corporal': 0.5,
    }}
    for ruta, valor in over.items():
        nodo = datos
        partes = ruta.split('.')
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = valor
    return ConfigStand(datos, {})


def _alimentar(ag, pid, muestras, *, fuente=FUENTE_CORPORAL, paso=5):
    """Mete muestras respetando el espaciado mínimo."""
    for i, (genero, conf, edad) in enumerate(muestras):
        ag.muestra(pid, genero=genero, genero_conf=conf, edad=edad,
                   fuente=fuente, frame_idx=i * paso)


def test_muestras_consistentes_dan_veredicto_valido():
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'p1', [('mujer', 0.8, 30), ('mujer', 0.9, 32),
                          ('mujer', 0.85, 28)])
    v = ag.veredicto('p1')
    assert v.valida and v.genero == 'mujer'
    assert v.genero_muestras == 3
    assert v.rango_etario == '26-35'
    assert v.motivo is None


def test_sin_muestras_da_null_con_motivo():
    ag = AgregadorDemografia(_cfg())
    v = ag.veredicto('fantasma')
    assert v.valida is False
    assert v.genero is None
    assert v.motivo == MotivoSinDemografia.SIN_MUESTRAS


def test_pocas_muestras_es_track_corto():
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'p1', [('hombre', 0.9, 40), ('hombre', 0.9, 41)])
    v = ag.veredicto('p1')
    assert v.valida is False
    assert v.motivo == MotivoSinDemografia.TRACK_CORTO


def test_muestras_contradictorias_dan_baja_confianza():
    """Mejor un null honesto que una moneda al aire presentada como dato."""
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'p1', [('mujer', 0.9, 30), ('hombre', 0.9, 30),
                          ('mujer', 0.9, 30), ('hombre', 0.9, 30)])
    v = ag.veredicto('p1')
    assert v.valida is False
    assert v.motivo == MotivoSinDemografia.BAJA_CONFIANZA
    # La confianza medida se conserva: es dato para el panel de calidad.
    assert 0.4 < v.genero_confianza < 0.65


def test_la_edad_usa_mediana_no_media():
    """Con media, un solo outlier arrastra el track entero."""
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'p1', [('hombre', 0.9, 30), ('hombre', 0.9, 32),
                          ('hombre', 0.9, 31), ('hombre', 0.9, 95)])
    v = ag.veredicto('p1')
    assert v.valida
    assert v.edad_estimada == 31.5          # mediana; la media daría 47
    assert v.rango_etario == '26-35'


def test_la_rama_facial_pesa_mas_que_la_corporal():
    ag = AgregadorDemografia(_cfg())
    for i in range(3):
        ag.muestra('p1', genero='mujer', genero_conf=0.9, edad=30,
                   fuente=FUENTE_FACIAL, frame_idx=i * 5)
    for i in range(2):
        ag.muestra('p1', genero='hombre', genero_conf=0.9, edad=30,
                   fuente=FUENTE_CORPORAL, frame_idx=(i + 3) * 5)
    v = ag.veredicto('p1')
    # 3*0.9*1.0 = 2.7 vs 2*0.9*0.5 = 0.9 -> ratio 0.75 >= 0.65
    assert v.valida and v.genero == 'mujer'
    assert v.fuente == 'mixto'


def test_solo_corporal_tambien_es_valido():
    """La rama corporal es el pipeline PRIMARIO: funciona sin rostro."""
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'p1', [('hombre', 0.9, 45)] * 4, fuente=FUENTE_CORPORAL)
    v = ag.veredicto('p1')
    assert v.valida and v.fuente == FUENTE_CORPORAL
    assert v.rango_etario == '36-50'


def test_espaciado_rechaza_muestras_pegadas():
    ag = AgregadorDemografia(_cfg())
    assert ag.muestra('p1', genero='mujer', genero_conf=0.9, frame_idx=0)
    # frame 1..4 están dentro del espaciado mínimo (5) -> se descartan
    for f in (1, 2, 3, 4):
        assert ag.muestra('p1', genero='mujer', genero_conf=0.9,
                          frame_idx=f) is False
    assert ag.muestra('p1', genero='mujer', genero_conf=0.9, frame_idx=5)
    assert ag.salud()['descartes_por_condicion'][Descarte.ESPACIADO] == 4


def test_buffer_circular_acota_memoria():
    ag = AgregadorDemografia(_cfg(**{'demografia.buffer.max_muestras': 4}))
    _alimentar(ag, 'p1', [('mujer', 0.9, 30)] * 10)
    assert len(ag._buffers['p1']) == 4
    assert ag.veredicto('p1').valida


def test_null_rate_y_desglose():
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'ok1', [('mujer', 0.9, 30)] * 3, fuente=FUENTE_FACIAL)
    _alimentar(ag, 'ok2', [('hombre', 0.9, 40)] * 3, fuente=FUENTE_CORPORAL)
    _alimentar(ag, 'corto', [('hombre', 0.9, 40)])
    # 'ninguno' no tiene ni una muestra.

    r = ag.null_rate(['ok1', 'ok2', 'corto', 'ninguno'])
    assert r['total'] == 4 and r['con_demografia'] == 2
    assert r['null_rate'] == 0.5
    assert r['por_motivo'][MotivoSinDemografia.TRACK_CORTO] == 1
    assert r['por_motivo'][MotivoSinDemografia.SIN_MUESTRAS] == 1
    # Desglose por rama, que es lo que pide el criterio de aceptación.
    assert r['por_rama'][FUENTE_FACIAL] == 1
    assert r['por_rama'][FUENTE_CORPORAL] == 1

    assert ag.null_rate([]) == {'total': 0, 'con_demografia': 0,
                                'null_rate': None, 'por_motivo': {},
                                'por_rama': {}}


def test_instrumentacion_registra_donde_se_pierde():
    """Sin estos contadores no se puede saber DÓNDE se pierde la demografía
    — que es justo lo que la auditoría no pudo responder con números."""
    ag = AgregadorDemografia(_cfg())
    ag.descartar_crop(Descarte.BBOX_PEQUENO)
    ag.descartar_crop(Descarte.BBOX_PEQUENO)
    ag.descartar_crop(Descarte.BORROSO)
    ag.muestra('p1', genero='mujer', genero_conf=0.9, frame_idx=0)

    s = ag.salud()
    assert s['crops_recibidos'] == 4
    assert s['crops_aceptados'] == 1
    assert s['crops_descartados'] == 3
    assert s['descartes_por_condicion'][Descarte.BBOX_PEQUENO] == 2
    assert s['descartes_por_condicion'][Descarte.BORROSO] == 1


def test_rangos_etarios_son_los_del_sistema():
    """No se inventa vocabulario: son los de AnalyticsConfig.AGE_RANGES."""
    assert rango_de_edad(5) == '0-12'
    assert rango_de_edad(15) == '13-17'
    assert rango_de_edad(22) == '18-25'
    assert rango_de_edad(30) == '26-35'
    assert rango_de_edad(45) == '36-50'
    assert rango_de_edad(60) == '51-65'
    assert rango_de_edad(80) == '65+'
    assert rango_de_edad(None) is None


def test_traduccion_a_campos_del_visitante():
    ag = AgregadorDemografia(_cfg())
    _alimentar(ag, 'p1', [('mujer', 0.9, 30)] * 3)
    campos = ag.veredicto('p1').como_campos_visitante()
    # Las claves deben coincidir EXACTAMENTE con stand_visitantes.
    assert set(campos) == {
        'genero', 'genero_confianza', 'genero_muestras', 'edad_estimada',
        'rango_etario', 'edad_muestras', 'demografia_valida',
        'motivo_sin_demografia'}
    assert campos['demografia_valida'] is True
    assert campos['motivo_sin_demografia'] is None


def test_un_null_nunca_impide_contar():
    """El arreglo del §3.2: sin género, la persona sigue existiendo."""
    ag = AgregadorDemografia(_cfg())
    campos = ag.veredicto('nadie').como_campos_visitante()
    assert campos['genero'] is None
    assert campos['demografia_valida'] is False
    assert campos['motivo_sin_demografia'] == MotivoSinDemografia.SIN_MUESTRAS
    # Nada aquí bloquea el alta del visitante: el conteo vive en
    # stand_identidad/stand_almacen y no consulta la demografía.


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
