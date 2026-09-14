"""
Pruebas de identificación del chef / personal (HITO 6).

El criterio de aceptación del hito es asimétrico y conviene tenerlo presente:
el chef debe quedar identificado y excluido, y **los falsos positivos deben
ser cero**. Marcar por error a un cliente como personal lo borra de las
métricas para siempre y hunde el denominador sin dejar rastro visible; no
detectar al chef, en cambio, se ve enseguida en las cifras.

Por eso `test_cliente_que_se_queda_mucho_no_se_marca_solo` es el test central.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_config import ConfigStand  # noqa: E402
from src.analityc.core.stand_personal import (  # noqa: E402
    GaleriaPersonal, OrigenPersonal)

DIM = 128


def _cfg(**over):
    datos = {'personal': {'dwell_auto_enrolamiento_min': 10,
                          'umbral_reid': 0.75}}
    for ruta, valor in over.items():
        nodo = datos
        partes = ruta.split('.')
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = valor
    return ConfigStand(datos, {})


def _emb(semilla: int) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


def test_enrolamiento_manual_excluye_al_chef():
    gp = GaleriaPersonal(_cfg())
    gp.enrolar('chef-1', _emb(1), nombre='Chef', ts=0.0)
    assert gp.es_personal('chef-1') is True
    assert gp.personal()[0]['origen'] == OrigenPersonal.MANUAL


def test_cliente_normal_no_es_personal():
    gp = GaleriaPersonal(_cfg())
    gp.enrolar('chef-1', _emb(1), ts=0.0)
    assert gp.es_personal('cliente-1', _emb(999)) is False


def test_cliente_que_se_queda_mucho_no_se_marca_solo():
    """CERO FALSOS POSITIVOS: el automático propone, nunca confirma.

    Un cliente que se queda 20 minutos charlando frente al stand supera de
    largo el umbral de permanencia. Debe quedar PROPUESTO y seguir contando
    como visitante hasta que un humano diga lo contrario.
    """
    gp = GaleriaPersonal(_cfg())
    for s in range(0, 1300, 10):        # ~21 min dentro de zona_personal
        gp.observar('cliente-pegajoso', en_zona_personal=True, ts=float(s))

    assert gp.es_personal('cliente-pegajoso') is False, (
        "se auto-marcó como personal sin confirmación humana")
    assert len(gp.propuestas()) == 1
    assert gp.propuestas()[0]['persistent_id'] == 'cliente-pegajoso'


def test_propuesta_confirmada_pasa_a_personal():
    gp = GaleriaPersonal(_cfg())
    for s in range(0, 1300, 10):
        gp.observar('chef-2', en_zona_personal=True, ts=float(s))
    assert gp.confirmar_propuesta('chef-2', _emb(3), nombre='Chef 2') is True
    assert gp.es_personal('chef-2') is True
    assert gp.propuestas() == []
    assert gp.calidad_dato()['por_origen'][OrigenPersonal.CONFIRMADO] == 1


def test_propuesta_rechazada_no_vuelve_a_proponerse():
    gp = GaleriaPersonal(_cfg())
    for s in range(0, 1300, 10):
        gp.observar('cliente-1', en_zona_personal=True, ts=float(s))
    assert gp.rechazar_propuesta('cliente-1') is True
    assert gp.propuestas() == []

    # Sigue dentro de la zona: no debe reaparecer la propuesta.
    for s in range(1300, 2600, 10):
        gp.observar('cliente-1', en_zona_personal=True, ts=float(s))
    assert gp.propuestas() == []
    assert gp.es_personal('cliente-1') is False


def test_dwell_solo_cuenta_dentro_de_la_zona():
    gp = GaleriaPersonal(_cfg())
    for s in range(0, 1300, 10):
        gp.observar('x', en_zona_personal=False, ts=float(s))
    assert gp.propuestas() == [], "acumuló permanencia fuera de zona_personal"


def test_reid_reconoce_al_chef_con_id_nuevo():
    """El chef cambia de persistent_id cada vez que el tracker lo pierde.

    Sin verificación por Re-ID habría que re-enrolarlo a cada rato, y entre
    medias contaría como visitante.
    """
    gp = GaleriaPersonal(_cfg())
    emb_chef = _emb(7)
    gp.enrolar('chef-1', emb_chef, ts=0.0)

    # Mismo embedding, identidad nueva -> se reconoce y se ancla.
    assert gp.es_personal('chef-1-reaparecido', emb_chef) is True
    assert gp.es_personal('chef-1-reaparecido') is True   # ya sin embedding
    assert gp.calidad_dato()['por_origen'][OrigenPersonal.REID] == 1


def test_reid_no_confunde_a_un_desconocido():
    gp = GaleriaPersonal(_cfg())
    gp.enrolar('chef-1', _emb(7), ts=0.0)
    assert gp.es_personal('otro', _emb(4242)) is False
    assert gp.calidad_dato()['por_origen'][OrigenPersonal.REID] == 0


def test_dar_de_baja_devuelve_a_la_persona_al_conteo():
    """Un enrolamiento erróneo tiene que ser reversible."""
    gp = GaleriaPersonal(_cfg())
    gp.enrolar('confundido', _emb(2), ts=0.0)
    assert gp.es_personal('confundido') is True
    gp.dar_de_baja('confundido')
    assert gp.es_personal('confundido') is False


def test_metricas_de_rendimiento_del_chef():
    gp = GaleriaPersonal(_cfg())
    gp.enrolar('chef-1', _emb(1), ts=0.0)
    gp.observar('chef-1', en_zona_personal=True, ts=0.0)
    for i in range(6):                       # 6 atenciones de 60 s
        gp.registrar_atencion('chef-1', 60.0, ts=float(i * 300))

    m = gp.metricas('chef-1', ahora=3600.0)   # 1 hora de jornada
    assert m['atenciones'] == 6
    assert m['atenciones_por_hora'] == 6.0
    assert m['tiempo_medio_atencion_s'] == 60.0
    assert m['tiempo_activo_s'] == 360.0
    # Lo que revela si el stand está infrautilizado.
    assert m['tiempo_muerto_s'] == 3240.0


def test_metricas_sin_actividad_no_inventan_numeros():
    gp = GaleriaPersonal(_cfg())
    m = gp.metricas('nadie')
    assert m['atenciones'] == 0
    assert m['atenciones_por_hora'] is None   # None, no 0: no hay jornada
    assert m['tiempo_medio_atencion_s'] is None


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
