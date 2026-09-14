"""
Pruebas de la lectura CRUCE de `PermanenciaArea` (paso visible + flecha).

Pedida el 8-ago-2026 para el cliente de detecciones: al atravesar el ROI, la
flecha decide si el paso es «Persona entrando al área» o «Persona saliendo
del área», con hora y foto, y la alerta sale por la cola de siempre
(WhatsApp/Jarvis según los interruptores del cliente).

Lo que más se protege:

  - **sin flecha NO se emite nada** — aquí la dirección se mide, no se
    adivina; un modo que «funciona» sin flecha estaría inventando;
  - **el paseo previo no envenena la medida** — la base del recorrido sigue a
    la persona hasta que pisa el ROI: caminar en contra por el pasillo y
    luego girar y entrar debe dar ENTRADA, no salida;
  - **la salida sin pareja avisa sin inventar tiempo** — aquí es un caso
    normal (la persona pudo estar dentro desde antes de arrancar).

El doble hereda de `PermanenciaArea` sin `__init__` (no carga YOLO ni GPU) y
llama al mismo `_iniciar_permanencia` de producción, igual que
`test_permanencia_area.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.permanencia_area import (  # noqa: E402
    MODO_AREA, MODO_CRUCE, MODO_PUERTA, PermanenciaArea)
from src.analityc.core.permanencia_cruce import DetectorCruce  # noqa: E402
from src.analityc.core.permanencia_estancias import SIN_ATRIBUIR  # noqa: E402
from src.analityc.core.permanencia_puerta import (  # noqa: E402
    ENTRADA, POR_FLECHA, SALIDA)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
#: Flecha «hacia adentro» apuntando a +x. Con cajas de 200 px de alto, medio
#: cuerpo (el umbral por defecto) son 100 px de avance.
ADENTRO = [[0, 0], [100, 0]]
ALTO = 200


def _obs(x, en_zona, confirmado=True):
    return {'en_zona': bool(en_zona), 'confirmado': confirmado,
            'box': [x, 50, x + 60, 50 + ALTO]}


def _caminar(det, clave, pasos, t0=0.0, dt=0.2, flecha=ADENTRO):
    """`pasos` = [(x, en_zona), ...]. Devuelve todos los cruces emitidos."""
    cruces = []
    for i, (x, en_zona) in enumerate(pasos):
        cruces.extend(det.observar(t0 + i * dt, {clave: _obs(x, en_zona)},
                                   flecha))
    return cruces


# ── DetectorCruce ───────────────────────────────────────────────────────
def test_paso_a_favor_de_la_flecha_es_entrada():
    det = DetectorCruce()
    pasos = [(0, False), (40, False), (80, True), (120, True),
             (160, True), (200, True), (240, False)]
    cruces = _caminar(det, 'p-A', pasos)
    assert [c.tipo for c in cruces] == [ENTRADA]
    assert cruces[0].via == POR_FLECHA
    assert cruces[0].proyeccion >= 0.5


def test_paso_en_contra_es_salida():
    det = DetectorCruce()
    pasos = [(240, True), (200, True), (160, True), (120, True), (80, False)]
    cruces = _caminar(det, 'p-B', pasos)
    assert [c.tipo for c in cruces] == [SALIDA]
    assert cruces[0].proyeccion <= -0.5


def test_sin_flecha_no_emite_nada():
    det = DetectorCruce()
    pasos = [(0, False), (40, False), (80, True), (120, True),
             (160, True), (200, True), (240, True)]
    assert _caminar(det, 'p-C', pasos, flecha=None) == []


def test_sin_pisar_el_roi_no_hay_cruce():
    """Caminar con decisión por el encuadre no es cruzar el umbral."""
    det = DetectorCruce()
    pasos = [(x, False) for x in range(0, 400, 40)]
    assert _caminar(det, 'p-D', pasos) == []


def test_merodear_bajo_el_umbral_calla():
    """Pasitos dentro del ROI que no llegan a medio cuerpo: silencio."""
    det = DetectorCruce()
    pasos = [(100, True), (130, True), (110, True), (140, True),
             (120, True), (150, True)]
    assert _caminar(det, 'p-E', pasos) == []


def test_el_paseo_previo_no_envenena_la_medida():
    """Camina EN CONTRA por el pasillo, gira y entra: debe dar ENTRADA.

    La base del recorrido sigue a la persona hasta que pisa el ROI; sin esa
    ventana deslizante, el tramo previo (aquí -1.6 alturas) taparía el avance
    real hacia adentro y el veredicto saldría al revés.
    """
    det = DetectorCruce()
    pasos = [(400, False), (320, False), (240, False),   # -160 px (en contra)
             (320, False), (400, True), (440, True),     # gira y entra
             (480, True), (520, True)]
    cruces = _caminar(det, 'p-F', pasos)
    assert [c.tipo for c in cruces] == [ENTRADA]


def test_entrar_y_luego_salir_emite_los_dos():
    """Tras cada emisión la base se rearma: la vuelta se mide desde ahí."""
    det = DetectorCruce()
    entrada = [(0, False), (40, False), (80, True), (120, True),
               (160, True), (200, True)]
    vuelta = [(240, True), (200, True), (160, True), (120, True),
              (80, True), (40, False)]
    cruces = _caminar(det, 'p-G', entrada + vuelta)
    assert [c.tipo for c in cruces] == [ENTRADA, SALIDA]


def test_el_mismo_sentido_no_se_repite_dentro_del_cooldown():
    """Quien sigue avanzando tras la entrada no dispara otra entrada."""
    det = DetectorCruce()
    pasos = [(0, False), (40, False), (80, True), (120, True), (160, True),
             (200, True), (300, True), (400, True), (500, True), (600, True)]
    cruces = _caminar(det, 'p-H', pasos)
    assert [c.tipo for c in cruces] == [ENTRADA]


# ── Integración con PermanenciaArea ─────────────────────────────────────
class _Doble(PermanenciaArea):
    """`PermanenciaArea` sin cargar modelos, sin disco y sin cámara."""

    def __init__(self, **over):
        from src.analityc.core.permanencia_area import _DEFECTOS
        cfg = {**_DEFECTOS,
               'modo': 'cruce',
               'guardar_fotos': False,        # nada de JPEG en output/
               'persistir_estancias': False,  # nada de JSON en output/
               'estancia_ttl_horas': 100}
        cfg.update(over)
        self._zone_state = {}
        self.active_tracks = {}
        self._alertas_pendientes = []
        self._camera_display_name = 'cam-test'
        self._iniciar_permanencia(cfg, 'cam-test')
        self._flecha_direccion = ADENTRO

    def ve_en(self, pid, en_zona, t, x):
        self.active_tracks = {pid: {'persistent_id': pid,
                                    'box': [x, 50, x + 60, 50 + ALTO],
                                    'in_zone': en_zona,
                                    'persistent_confirmed': True}}
        self._modo_vigente = self._modo_efectivo()
        self._ciclo_cruce(FRAME, t)

    def alertas(self):
        a = list(self._alertas_pendientes)
        self._alertas_pendientes = []
        return a


def _entra(p, pid, t0=0.0):
    for i, (x, z) in enumerate([(0, False), (40, False), (80, True),
                                (120, True), (160, True), (200, True)]):
        p.ve_en(pid, z, t0 + i * 0.2, x)


def _sale(p, pid, t0):
    for i, (x, z) in enumerate([(240, True), (200, True), (160, True),
                                (120, True), (80, True), (40, False)]):
        p.ve_en(pid, z, t0 + i * 0.2, x)


def test_en_modo_cruce_entrar_avisa_persona_entrando():
    p = _Doble()
    _entra(p, 'p-1')
    alertas = p.alertas()
    assert len(alertas) == 1
    a = alertas[0]
    assert a['event_type'] == 'Persona entrando al área'
    assert a['description'].startswith('Persona entrando al área a las ')
    assert a['crop_image'], 'la alerta de entrada debe llevar foto'
    assert a['clase_gruesa'] == 'persona'
    assert a['dedup_clave'].endswith('|p-1')
    assert 'p-1' in p._registro, 'la entrada abre la estancia (cronómetro)'


def test_salida_con_pareja_lleva_el_tiempo_dentro():
    p = _Doble()
    _entra(p, 'p-2', t0=0.0)
    p.alertas()
    _sale(p, 'p-2', t0=120.0)
    alertas = p.alertas()
    assert len(alertas) == 1
    a = alertas[0]
    assert a['event_type'] == 'Persona saliendo del área'
    # Formato pedido el 10-ago-2026: el texto redactado, no una frase suelta.
    assert a['description'].startswith('Permanencia en ')
    assert 'Tiempo total: 00:0' in a['description']
    assert a['duracion_s'] and a['duracion_s'] > 100
    assert a['entrada_base64'] is not None


def test_salida_sin_pareja_avisa_sin_inventar_tiempo():
    """La persona estaba dentro desde antes de arrancar: se avisa igual."""
    p = _Doble()
    _sale(p, 'p-3', t0=0.0)
    alertas = p.alertas()
    assert len(alertas) == 1
    a = alertas[0]
    assert a['event_type'] == 'Persona saliendo del área'
    assert a['description'].startswith('Persona saliendo del área a las ')
    assert a['duracion_s'] is None, 'sin entrada vista no se inventa tiempo'
    assert a['identidad_por'] == SIN_ATRIBUIR


def test_sin_flecha_el_modo_cruce_calla():
    p = _Doble()
    p._flecha_direccion = None
    _entra(p, 'p-4')
    assert p.alertas() == []


# ── El interruptor: la flecha activa la lectura cruce ───────────────────
def test_la_flecha_activa_el_cruce_en_modo_area():
    p = _Doble(modo='area')
    assert p._modo_efectivo() == MODO_CRUCE


def test_sin_flecha_el_area_sigue_siendo_area():
    p = _Doble(modo='area')
    p._flecha_direccion = None
    assert p._modo_efectivo() == MODO_AREA


def test_el_modo_puerta_explicito_no_se_toca():
    p = _Doble(modo='puerta')
    assert p._modo_efectivo() == MODO_PUERTA


def test_el_resumen_dice_el_modo_vigente():
    p = _Doble(modo='area')
    p._modo_vigente = p._modo_efectivo()
    assert p._resumen(0.0)['modo'] == MODO_CRUCE


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
