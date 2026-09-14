"""
Pruebas del detector de cruces de puerta (`permanencia_puerta.py`).

La cámara mira al marco y **no ve el interior**. La pregunta que este módulo
responde —¿entró o se marchó?— no se puede contestar con geometría, así que
se contesta con el ciclo de vida del seguimiento. Estas pruebas son las que
fijan esa lectura:

  - nace fuera, muere dentro del vano  -> **entró**;
  - nace dentro del vano, se aleja     -> **salió**;
  - nace y muere dentro sin moverse    -> **nada**, porque es ambiguo.

El último es el importante: es la tentación de reportar algo cuando no se
sabe, y aquí se prueba explícitamente que no se reporta.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.permanencia_puerta import (  # noqa: E402
    AMBIGUO, ENTRADA, POR_CICLO, POR_FLECHA, SALIDA, DEFECTOS_PUERTA,
    DetectorPuerta, _unitario, centro_pie, diagonal)

CAJA = (100.0, 50.0, 160.0, 250.0)


def _ver(det, t, en_zona, clave='p1', confirmado=True):
    """Un frame con la persona visible."""
    return det.observar(t, {clave: {'en_zona': en_zona, 'box': CAJA,
                                    'confirmado': confirmado}})


def _nadie(det, t):
    return det.observar(t, {})


def _tipos(cruces):
    return [c.tipo for c in cruces]


# ── Entrada ─────────────────────────────────────────────────────────────
def test_nace_fuera_muere_dentro_del_vano_es_ENTRADA():
    det = DetectorPuerta()
    t = 0.0
    for _ in range(4):                       # se acerca, aún fuera del vano
        t += 0.1
        assert _ver(det, t, False) == []
    for _ in range(4):                       # pisa el vano
        t += 0.1
        assert _ver(det, t, True) == []
    # Desaparece: se metió. Hace falta esperar `segundos_ausencia`.
    assert _nadie(det, t + 0.5) == []
    cruces = _nadie(det, t + 3.0)
    assert _tipos(cruces) == [ENTRADA]
    assert cruces[0].clave == 'p1' and cruces[0].box == CAJA


def test_la_entrada_se_emite_UNA_vez():
    det = DetectorPuerta()
    t = 0.0
    for en_zona in (False, False, False, False, True, True, True, True):
        t += 0.1
        _ver(det, t, en_zona)
    assert _tipos(_nadie(det, t + 3.0)) == [ENTRADA]
    assert _nadie(det, t + 4.0) == []
    assert _nadie(det, t + 60.0) == []


def test_quien_se_va_por_donde_vino_NO_entra():
    """Última posición fuera del vano = se marchó, no se metió."""
    det = DetectorPuerta()
    t = 0.0
    for en_zona in (False, True, True, True, False, False, False):
        t += 0.1
        _ver(det, t, en_zona)
    assert _nadie(det, t + 3.0) == []


def test_una_oclusion_corta_no_fabrica_una_entrada():
    det = DetectorPuerta()
    t = 0.0
    for en_zona in (False, False, True, True, True):
        t += 0.1
        _ver(det, t, en_zona)
    # Se pierde medio segundo detrás de una columna y vuelve: por debajo de
    # `segundos_ausencia`, así que no hay cruce.
    assert _nadie(det, t + 0.5) == []
    t += 1.0
    assert _ver(det, t, True) == []


def test_un_parpadeo_del_detector_no_es_una_entrada():
    """Dos frames de vida no bastan: `frames_minimos`."""
    det = DetectorPuerta()
    _ver(det, 0.1, True)
    _ver(det, 0.2, True)
    assert _nadie(det, 5.0) == []


# ── Salida ──────────────────────────────────────────────────────────────
def test_nace_en_el_vano_y_se_aleja_es_SALIDA():
    det = DetectorPuerta()
    t = 0.0
    cruces = []
    for _ in range(3):                       # aparece dentro del vano
        t += 0.1
        cruces += _ver(det, t, True)
    assert cruces == [], 'todavía podría estar solo parada ahí'
    for _ in range(3):                       # se aleja
        t += 0.1
        cruces += _ver(det, t, False)
    assert _tipos(cruces) == [SALIDA]


def test_la_salida_no_se_convierte_luego_en_entrada():
    """Quien salió y se va del encuadre no puede además haber entrado."""
    det = DetectorPuerta()
    t = 0.0
    for en_zona in (True, True, True, False, False, False):
        t += 0.1
        _ver(det, t, en_zona)
    assert _nadie(det, t + 5.0) == []


def test_quien_ya_estaba_en_el_encuadre_no_sale():
    det = DetectorPuerta()
    t = 0.0
    for en_zona in (False, False, False, True, True, False, False, False):
        t += 0.1
        cruces = _ver(det, t, en_zona)
        assert cruces == [], f'cruce inventado en t={t}'


# ── El caso ambiguo ─────────────────────────────────────────────────────
def test_alguien_parado_en_el_vano_no_genera_NADA():
    """Nació dentro (¿salió?) y murió dentro (¿entró?). No se decide."""
    det = DetectorPuerta()
    t = 0.0
    for _ in range(10):
        t += 0.1
        assert _ver(det, t, True) == []
    assert _nadie(det, t + 5.0) == [], 'reportó un cruce que no puede saber'


def test_sin_exigir_fuera_de_zona_ese_caso_SI_cuenta_como_entrada():
    """Escape para el ROI que cubre todo el pasillo (ver el YAML)."""
    det = DetectorPuerta({'exigir_fuera_de_zona': False})
    t = 0.0
    for _ in range(10):
        t += 0.1
        _ver(det, t, True)
    assert _tipos(_nadie(det, t + 5.0)) == [ENTRADA]


# ── Varias personas ─────────────────────────────────────────────────────
def test_dos_personas_a_la_vez_no_se_mezclan():
    det = DetectorPuerta()
    t = 0.0
    for _ in range(5):                       # 'entra' se acerca; 'sale' surge
        t += 0.1                             # del vano
        det.observar(t, {
            'entra': {'en_zona': False, 'box': CAJA, 'confirmado': True},
            'sale': {'en_zona': True, 'box': CAJA, 'confirmado': True}})
    cruces = []
    for _ in range(4):                       # 'entra' pisa el vano;
        t += 0.1                             # 'sale' se aleja
        cruces += det.observar(t, {
            'entra': {'en_zona': True, 'box': CAJA, 'confirmado': True},
            'sale': {'en_zona': False, 'box': CAJA, 'confirmado': True}})
    assert [(c.clave, c.tipo) for c in cruces] == [('sale', SALIDA)]

    cruces = det.observar(t + 3.0, {})
    assert [(c.clave, c.tipo) for c in cruces] == [('entra', ENTRADA)]


# ── Robustez ────────────────────────────────────────────────────────────
def test_una_caja_corrupta_no_tumba_el_bucle():
    det = DetectorPuerta()
    t = 0.0
    for en_zona in (False, False, True, True, True):
        t += 0.1
        det.observar(t, {'p1': {'en_zona': en_zona, 'box': ['x', None],
                                'confirmado': True}})
    cruces = det.observar(t + 3.0, {})
    assert _tipos(cruces) == [ENTRADA] and cruces[0].box is None


def test_las_huellas_no_crecen_sin_limite():
    det = DetectorPuerta()
    t = 0.0
    for i in range(800):
        t += 1.0
        det.observar(t, {f'p{i}': {'en_zona': False, 'box': CAJA}})
    assert det.rastreados <= 600, 'fuga de memoria en sesiones largas'


def test_la_config_desconocida_se_ignora_sin_romper():
    det = DetectorPuerta({'inventado': 1, 'frames_minimos': 9})
    assert det.cfg['frames_minimos'] == 9
    assert 'inventado' not in det.cfg
    assert set(det.cfg) == set(DEFECTOS_PUERTA)


# ── La flecha de dirección ──────────────────────────────────────────────
#
# La dibuja el operador en el recuadro: apunta HACIA ADENTRO. Aquí llega ya
# como par de puntos. Confirma y desempata; no sustituye al ciclo de vida.

#: Flecha que apunta a la DERECHA (la puerta está a la derecha del encuadre).
DERECHA = [[0, 0], [100, 0]]
IZQUIERDA = [[100, 0], [0, 0]]


def _caja_en(x, alto=200):
    """Caja de 60x`alto` con su borde izquierdo en `x`."""
    return (float(x), 50.0, float(x + 60), float(50 + alto))


def _ver_en(det, t, en_zona, x, flecha=None, clave='p1', alto=200):
    return det.observar(t, {clave: {'en_zona': en_zona,
                                    'box': _caja_en(x, alto),
                                    'confirmado': True}}, flecha)


def test_la_flecha_confirma_la_entrada_y_lo_dice():
    det = DetectorPuerta()
    t, x = 0.0, 100
    for en_zona in (False, False, False, False, True, True, True, True):
        t += 0.1
        x += 40                              # avanza hacia la puerta
        _ver_en(det, t, en_zona, x, DERECHA)
    cruces = det.observar(t + 3.0, {}, DERECHA)
    assert _tipos(cruces) == [ENTRADA]
    assert cruces[0].via == POR_FLECHA
    assert cruces[0].proyeccion > 0.5, 'avanzó más de media altura hacia dentro'


def test_la_flecha_DESCARTA_una_entrada_que_iba_al_reves():
    """Muere en el vano, pero venía alejándose: no se metió."""
    det = DetectorPuerta()
    t, x = 0.0, 600
    for en_zona in (False, False, False, False, True, True, True, True):
        t += 0.1
        x -= 40                              # se aleja de la puerta
        _ver_en(det, t, en_zona, x, DERECHA)
    assert det.observar(t + 3.0, {}, DERECHA) == []


def test_la_flecha_DESCARTA_una_salida_que_iba_al_reves():
    """Nace en el vano pero se mete: no es una salida, es una entrada."""
    det = DetectorPuerta()
    t, x = 0.0, 100
    cruces = []
    for en_zona in (True, True, True, False, False, False):
        t += 0.1
        x += 60                              # se adentra
        cruces += _ver_en(det, t, en_zona, x, DERECHA)
    assert cruces == [], 'contó una salida yendo hacia adentro'


def test_la_flecha_confirma_la_salida():
    det = DetectorPuerta()
    t, x = 0.0, 600
    cruces = []
    for en_zona in (True, True, True, False, False, False):
        t += 0.1
        x -= 60
        cruces += _ver_en(det, t, en_zona, x, DERECHA)
    assert _tipos(cruces) == [SALIDA]
    assert cruces[0].via == POR_FLECHA and cruces[0].proyeccion < -0.5


def test_la_flecha_RESCATA_el_caso_que_sin_ella_se_callaba():
    """Nace y muere en el vano: sin flecha es ambiguo; con ella, se decide."""
    det = DetectorPuerta()
    t, x = 0.0, 100
    for _ in range(8):
        t += 0.1
        x += 40
        _ver_en(det, t, True, x, DERECHA)    # SIEMPRE dentro del vano
    cruces = det.observar(t + 3.0, {}, DERECHA)
    assert _tipos(cruces) == [ENTRADA]
    assert cruces[0].via == POR_FLECHA


def test_sin_moverse_lo_suficiente_sigue_sin_decidirse():
    det = DetectorPuerta()
    t, x = 0.0, 100
    for _ in range(8):
        t += 0.1
        x += 3                               # 24 px sobre 200 de alto: 0.12
        _ver_en(det, t, True, x, DERECHA)
    assert det.observar(t + 3.0, {}, DERECHA) == []


def test_el_umbral_va_en_ALTURAS_no_en_pixeles():
    """La misma escena al doble de resolución da el mismo veredicto.

    En píxeles, el mismo ajuste serviría para una cámara y no para la otra.
    """
    veredictos = []
    for escala in (1, 2):
        det = DetectorPuerta()
        t, x = 0.0, 100 * escala
        for en_zona in (False, False, False, False, True, True, True, True):
            t += 0.1
            x += 40 * escala
            _ver_en(det, t, en_zona, x, DERECHA, alto=200 * escala)
        cruces = det.observar(t + 3.0, {}, DERECHA)
        veredictos.append((_tipos(cruces), round(cruces[0].proyeccion, 2)))
    assert veredictos[0] == veredictos[1]


def test_la_flecha_al_reves_invierte_el_veredicto():
    det = DetectorPuerta()
    t, x = 0.0, 100
    for _ in range(8):
        t += 0.1
        x += 40
        _ver_en(det, t, True, x, IZQUIERDA)   # la puerta está a la IZQUIERDA
    cruces = det.observar(t + 3.0, {}, IZQUIERDA)
    assert cruces == [], 'se aleja de la puerta: no entró'


def test_sin_flecha_todo_sigue_como_estaba():
    """La heurística de siempre es el respaldo, no un modo degradado."""
    det = DetectorPuerta()
    t, x = 0.0, 100
    for en_zona in (False, False, False, False, True, True, True, True):
        t += 0.1
        x += 40
        _ver_en(det, t, en_zona, x, None)
    cruces = det.observar(t + 3.0, {}, None)
    assert _tipos(cruces) == [ENTRADA]
    assert cruces[0].via == POR_CICLO
    assert cruces[0].proyeccion is None, 'no había flecha que proyectar'


# ── Cruces ambiguos para el árbitro ─────────────────────────────────────
def test_por_defecto_los_ambiguos_se_callan():
    det = DetectorPuerta()
    t = 0.0
    for _ in range(8):
        t += 0.1
        _ver(det, t, True)
    assert det.observar(t + 3.0, {}) == []


def test_con_el_arbitro_activo_los_ambiguos_se_emiten():
    det = DetectorPuerta({'emitir_ambiguos': True})
    t = 0.0
    for _ in range(8):
        t += 0.1
        _ver(det, t, True)
    cruces = det.observar(t + 3.0, {})
    assert _tipos(cruces) == [AMBIGUO]
    assert cruces[0].clave == 'p1'


def test_lo_que_la_flecha_resuelve_NO_va_al_arbitro():
    det = DetectorPuerta({'emitir_ambiguos': True})
    t, x = 0.0, 100
    for _ in range(8):
        t += 0.1
        x += 40
        _ver_en(det, t, True, x, DERECHA)
    assert _tipos(det.observar(t + 3.0, {}, DERECHA)) == [ENTRADA]


# ── Normalización de la flecha que llega del cliente ────────────────────
def test_se_acepta_el_par_de_puntos_del_recuadro():
    assert _unitario([[0, 0], [10, 0]]) == (1.0, 0.0)


def test_se_acepta_un_vector_ya_calculado():
    assert _unitario((0.0, 3.0)) == (0.0, 1.0)


def test_una_flecha_inservible_es_None_y_no_una_excepcion():
    for basura in (None, [], [[1, 1]], 'xy', [[0, 0], [0, 0]],
                   [['a', 'b'], ['c', 'd']], {}):
        assert _unitario(basura) is None, f'{basura!r} debería ser None'


# ── Geometría de apoyo ──────────────────────────────────────────────────
def test_el_ancla_es_el_pie_no_el_centro():
    """Con el centroide, una persona alta entra en la puerta un metro antes."""
    assert centro_pie(CAJA) == (130.0, 250.0)
    assert centro_pie(None) is None


def test_diagonal_de_una_caja():
    assert abs(diagonal((0, 0, 3, 4)) - 5.0) < 1e-6
    assert diagonal(None) == 0.0


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
