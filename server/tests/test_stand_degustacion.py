"""
Pruebas de la detección de degustación (HITO 8).

Keypoints sintéticos en formato COCO (17 puntos, cada uno [x, y, conf]) y un
cliente VLM falso. Así se prueban las dos etapas —geometría y decisión— sin
GPU y sin cargar Qwen.

Lo que NO prueba: precisión y recall reales. Eso exige un clip con
degustaciones anotadas a mano (HITO 13); aquí solo se verifica que las reglas
hacen lo que dicen hacer.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_config import ConfigStand  # noqa: E402
from src.analityc.core.stand_degustacion import (  # noqa: E402
    METODO_POSE_VLM, DetectorDegustacion, DisparadorPose, PlanificadorVLM,
    ancho_cabeza, angulo_codo, parsear_respuesta_vlm)


def _cfg(**over):
    datos = {'degustacion': {
        'pose': {'ratio_muneca_boca': 0.35, 'frames_min': 8,
                 'ventana_s': 2.0, 'angulo_codo_max': 120.0},
        'ventana_post_atencion_s': 90,
        'vlm': {'confianza_min': 0.7, 'max_por_minuto': 20},
    }}
    for ruta, valor in over.items():
        nodo = datos
        partes = ruta.split('.')
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = valor
    return ConfigStand(datos, {})


def _kps(*, muneca_der=(200.0, 200.0), nariz=(200.0, 100.0),
         codo_der=(240.0, 160.0)):
    """17 keypoints COCO. Cabeza de 40 px (orejas a 180 y 220)."""
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    k[0] = [nariz[0], nariz[1], 0.9]            # nariz
    k[3] = [180.0, 100.0, 0.9]                  # oreja izq
    k[4] = [220.0, 100.0, 0.9]                  # oreja der
    k[5] = [170.0, 160.0, 0.9]                  # hombro izq
    k[6] = [230.0, 160.0, 0.9]                  # hombro der
    k[8] = [codo_der[0], codo_der[1], 0.9]      # codo der
    k[10] = [muneca_der[0], muneca_der[1], 0.9]  # muñeca der
    return k


# ── Geometría ───────────────────────────────────────────────────────────
def test_ancho_cabeza_prefiere_las_orejas():
    assert ancho_cabeza(_kps()) == 40.0


def test_ancho_cabeza_cae_a_hombros_sin_cara():
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    k[5] = [100.0, 160.0, 0.9]
    k[6] = [200.0, 160.0, 0.9]
    assert ancho_cabeza(k) == 45.0        # 100 px de hombros * 0.45


def test_ancho_cabeza_none_sin_referencias():
    assert ancho_cabeza([[0.0, 0.0, 0.0] for _ in range(17)]) is None


def test_angulo_codo():
    # Brazo estirado: hombro, codo y muñeca alineados.
    assert round(angulo_codo((0, 0), (10, 0), (20, 0))) == 180
    # Doblado en ángulo recto.
    assert round(angulo_codo((0, 0), (10, 0), (10, 10))) == 90


# ── Etapa 1: disparador de pose ─────────────────────────────────────────
def test_mano_en_la_boca_es_candidata():
    d = DisparadorPose(_cfg())
    # Muñeca a 10 px de la nariz; cabeza 40 px -> ratio 0.25 < 0.35.
    g = d.evaluar_frame(_kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0)))
    assert g.candidato is True
    assert g.ratio is not None and g.ratio < 0.35


def test_mano_lejos_no_es_candidata():
    d = DisparadorPose(_cfg())
    g = d.evaluar_frame(_kps(muneca_der=(400.0, 400.0)))
    assert g.candidato is False
    assert g.motivo == 'mano_lejos'


def test_codo_estirado_descarta_aunque_la_mano_este_cerca():
    """Un brazo estirado no puede estar llevando nada a la boca."""
    d = DisparadorPose(_cfg())
    # Hombro(230,160), codo(215,135), muñeca(200,110): casi alineados.
    g = d.evaluar_frame(_kps(muneca_der=(200.0, 110.0),
                             codo_der=(215.0, 135.0)))
    assert g.candidato is False
    assert g.motivo == 'codo_estirado'


def test_sin_cara_no_dispara():
    d = DisparadorPose(_cfg())
    k = _kps()
    k[0] = [0.0, 0.0, 0.0]                # nariz sin confianza
    assert d.evaluar_frame(k).motivo == 'sin_cara'


def test_hacen_falta_frames_min_dentro_de_la_ventana():
    d = DisparadorPose(_cfg())
    buena = _kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0))
    for i in range(7):
        dispara, _ = d.observar('p1', buena, ts=i * 0.1)
        assert dispara is False, "disparó antes de los 8 frames"
    dispara, _ = d.observar('p1', buena, ts=0.7)
    assert dispara is True


def test_los_frames_no_tienen_que_ser_consecutivos():
    """El gesto tiene oclusiones: la mano tapa la cara, alguien cruza."""
    d = DisparadorPose(_cfg())
    buena = _kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0))
    mala = _kps(muneca_der=(400.0, 400.0))
    dispara = False
    ts = 0.0
    for _ in range(8):
        dispara, _ = d.observar('p1', buena, ts=ts)
        ts += 0.05
        d.observar('p1', mala, ts=ts)     # frame perdido entre medias
        ts += 0.05
    assert dispara is True


def test_hits_fuera_de_la_ventana_caducan():
    d = DisparadorPose(_cfg())
    buena = _kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0))
    for i in range(7):
        d.observar('p1', buena, ts=i * 0.1)
    # 10 s después: los hits viejos ya no valen.
    dispara, _ = d.observar('p1', buena, ts=10.0)
    assert dispara is False


# ── Planificador / control de coste ─────────────────────────────────────
def test_la_cola_respeta_el_tope_por_minuto():
    p = PlanificadorVLM(_cfg(**{'degustacion.vlm.max_por_minuto': 3}))
    for i in range(10):
        p.encolar(f'p{i}', ts=0.0)
    consumidas = [p.siguiente(0.0) for _ in range(10)]
    assert sum(1 for c in consumidas if c is not None) == 3
    # Lo demás NO se descarta: sigue en cola.
    assert p.en_cola == 7


def test_pasado_el_minuto_se_recupera_el_cupo():
    p = PlanificadorVLM(_cfg(**{'degustacion.vlm.max_por_minuto': 2}))
    for i in range(5):
        p.encolar(f'p{i}', ts=0.0)
    assert p.siguiente(0.0) and p.siguiente(0.0)
    assert p.siguiente(0.0) is None
    assert p.siguiente(61.0) is not None


def test_no_se_encola_dos_veces_a_la_misma_persona():
    p = PlanificadorVLM(_cfg())
    assert p.encolar('p1', ts=0.0) is True
    assert p.encolar('p1', ts=0.5) is False
    assert p.en_cola == 1


# ── Parseo de la respuesta del VLM ──────────────────────────────────────
def test_parsea_json_envuelto_en_prosa():
    """Los VLM añaden texto pese a pedirles JSON puro."""
    r = parsear_respuesta_vlm(
        'Claro, aquí tienes:\n```json\n'
        '{"degustando": true, "confianza": 0.9, "evidencia": "come jamón"}\n'
        '```\nEspero que ayude.')
    assert r is not None and r['degustando'] is True
    assert r['confianza'] == 0.9


def test_parseo_devuelve_none_si_no_hay_json():
    assert parsear_respuesta_vlm("no puedo responder") is None
    assert parsear_respuesta_vlm("") is None
    assert parsear_respuesta_vlm("{roto") is None


# ── Orquestación ────────────────────────────────────────────────────────
def _detector(respuesta, **over):
    llamadas = []

    def vlm(frames, prompt):
        llamadas.append((len(frames), prompt))
        return respuesta
    return DetectorDegustacion(_cfg(**over), cliente_vlm=vlm), llamadas


def _disparar(det, pid='p1', ts0=0.0):
    buena = _kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0))
    encolado = False
    for i in range(9):
        encolado = det.observar(pid, buena, ts=ts0 + i * 0.1,
                                fue_atendido=True) or encolado
    return encolado


def test_flujo_completo_confirma_degustacion():
    det, llamadas = _detector(
        '{"degustando": true, "confianza": 0.88, "evidencia": "se lleva '
        'un trozo a la boca"}')
    assert _disparar(det) is True
    confs = det.procesar_cola(lambda pid: ['f1', 'f2', 'f3'], ahora=1.0)
    assert len(confs) == 1
    assert confs[0].degustando and confs[0].confianza == 0.88
    assert confs[0].metodo == METODO_POSE_VLM
    assert llamadas[0][0] == 3                 # ráfaga de 3 frames
    assert det.salud()['confirmadas'] == 1


def test_vlm_dice_que_no_y_no_se_confirma():
    det, _ = _detector(
        '{"degustando": false, "confianza": 0.95, "evidencia": "se rasca"}')
    _disparar(det)
    assert det.procesar_cola(lambda pid: ['f1'], ahora=1.0) == []
    assert det.salud()['rechazadas_vlm'] == 1


def test_confianza_baja_no_se_confirma():
    det, _ = _detector(
        '{"degustando": true, "confianza": 0.4, "evidencia": "quizas"}')
    _disparar(det)
    assert det.procesar_cola(lambda pid: ['f1'], ahora=1.0) == []
    assert det.salud()['rechazadas_vlm'] == 1


def test_sin_vlm_no_se_confirma_por_pose_sola():
    """La pose confunde comer con rascarse: no basta por sí sola."""
    det = DetectorDegustacion(_cfg(), cliente_vlm=None)
    _disparar(det)
    assert det.procesar_cola(lambda pid: ['f1'], ahora=1.0) == []
    assert det.salud()['confirmadas'] == 0


def test_el_vlm_que_revienta_no_tumba_el_pipeline():
    def vlm_roto(frames, prompt):
        raise RuntimeError("CUDA out of memory")
    det = DetectorDegustacion(_cfg(), cliente_vlm=vlm_roto)
    _disparar(det)
    assert det.procesar_cola(lambda pid: ['f1'], ahora=1.0) == []
    assert det.salud()['errores_vlm'] == 1


def test_no_elegible_si_no_fue_atendido():
    """Nadie degusta sin que le hayan dado la muestra."""
    det, _ = _detector('{"degustando": true, "confianza": 0.9}')
    buena = _kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0))
    for i in range(20):
        det.observar('p1', buena, ts=i * 0.1, fue_atendido=False)
    assert det.salud()['frames_pose'] == 0


def test_fuera_de_la_ventana_post_atencion_no_se_evalua():
    det, _ = _detector('{"degustando": true, "confianza": 0.9}')
    buena = _kps(muneca_der=(200.0, 110.0), codo_der=(240.0, 150.0))
    for i in range(20):
        det.observar('p1', buena, ts=1000.0 + i * 0.1, fue_atendido=True,
                     ts_fin_atencion=0.0)          # hace 1000 s
    assert det.salud()['frames_pose'] == 0


def test_no_se_reconfirma_a_quien_ya_degusto():
    det, llamadas = _detector(
        '{"degustando": true, "confianza": 0.9, "evidencia": "come"}')
    _disparar(det)
    det.procesar_cola(lambda pid: ['f1'], ahora=1.0)
    # Segundo intento sobre la misma persona.
    _disparar(det, ts0=10.0)
    det.procesar_cola(lambda pid: ['f1'], ahora=11.0)
    assert len(llamadas) == 1, "se gastó VLM en alguien ya confirmado"


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
