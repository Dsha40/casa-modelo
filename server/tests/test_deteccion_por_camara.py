"""Detección ajustable POR CÁMARA y el aviso de escena sin personas.

Fija lo del 8-ago-2026, nacido de una tarde en la que «no avanza» resultó ser
que la ventana de iVMS mostraba la cámara del GENERADOR (patio vacío) y nadie
lo notó: cero detecciones no dejaba NI UNA señal.

Dos piezas:
  - los perfiles config/camera_profiles/<camera_id>.json, que ya ajustaban la
    demografía, ahora ajustan también la DETECCIÓN (PERSON_DETECT_CONF,
    BYTETRACK_TRACK_ACTIVATION_THRESH) para esa cámara y solo esa;
  - PermanenciaArea delata (log + panel) cuando lleva demasiado sin ver ni
    una persona, en vez de callar como si no hubiera tránsito.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.analytics.demographics import (  # noqa: E402
    _CameraConfigView)
from src.analityc.core.permanencia_area import (  # noqa: E402
    AVISO_SIN_PERSONAS_S, PermanenciaArea)

PERFIL_IVMS = (RAIZ / 'server' / 'config' / 'camera_profiles'
               / 'win-iVMS-4200.json')


# ── El perfil de la cámara ───────────────────────────────────────────────
def test_todas_las_claves_del_perfil_existen_en_analytics_config():
    """El cargador IGNORA en silencio las claves desconocidas: un typo aquí
    no daría error, daría una cámara sin el ajuste que se cree que tiene."""
    datos = json.loads(PERFIL_IVMS.read_text(encoding='utf-8'))
    desconocidas = [k for k in datos
                    if not k.startswith('_')
                    and not hasattr(AnalyticsConfig, k)]
    assert not desconocidas, (
        f'claves que el cargador descartaría: {desconocidas}')


def test_el_perfil_ivms_baja_el_piso_de_deteccion():
    """La razón de ser del cambio: gente pequeña/parcial de las cámaras
    exteriores en picado que a 0.25 el detector perdía."""
    datos = json.loads(PERFIL_IVMS.read_text(encoding='utf-8'))
    assert float(datos['PERSON_DETECT_CONF']) < float(
        AnalyticsConfig.PERSON_DETECT_CONF)
    assert float(datos['BYTETRACK_TRACK_ACTIVATION_THRESH']) < float(
        AnalyticsConfig.BYTETRACK_TRACK_ACTIVATION_THRESH)


def test_la_vista_por_camara_resuelve_override_y_fallback():
    vista = _CameraConfigView({'PERSON_DETECT_CONF': 0.18})
    assert float(vista.PERSON_DETECT_CONF) == 0.18
    # Lo no sobreescrito cae al valor global, intacto.
    assert (float(vista.BYTETRACK_TRACK_ACTIVATION_THRESH)
            == float(AnalyticsConfig.BYTETRACK_TRACK_ACTIVATION_THRESH))


def test_la_deteccion_usa_el_perfil_de_la_camara():
    """Contrato con tiendas.py: el predict y el tracker leen el perfil,
    no la config global a pelo."""
    import inspect
    from src.analityc.core.tiendas import TiendasProcess
    fuente = inspect.getsource(TiendasProcess.process_frame)
    assert '_cfg_cam = self._perfil_camara(camera_id)' in fuente
    assert '_cfg_cam.PERSON_DETECT_CONF' in fuente
    assert 'self._make_byte_tracker(camera_id)' in fuente
    fuente_state = inspect.getsource(TiendasProcess._get_cam_state)
    assert 'self._make_byte_tracker(camera_id)' in fuente_state


# ── El aviso de escena sin personas ──────────────────────────────────────
def _permanencia_pelada() -> PermanenciaArea:
    """Instancia SIN __init__ (sin modelos): solo lo que usa _vigilar_escena."""
    p = object.__new__(PermanenciaArea)
    p._camara_permanencia = 'test-cam'
    p._ultimo_aviso_escena = 0.0
    p._sin_personas_s = None
    return p


def _instante(anio: int, mes: int, dia: int, hora: int, minuto: int) -> float:
    """Época de un instante en hora LOCAL (la máquina corre en America/Caracas).

    4-sep-2026: los tests del aviso dejan de arrancar de time.time(). El
    comportamiento no depende de la hora (_vigilar_escena solo resta deltas),
    pero un test que lee el reloj real no lo DEMUESTRA: al fijar el instante
    queda escrito que da igual cuándo corra la suite.
    """
    return time.mktime((anio, mes, dia, hora, minuto, 0, -1, -1, -1))


MARTES_MEDIODIA = _instante(2026, 9, 8, 12, 0)      # martes 8-sep-2026 12:00


def test_con_personas_no_hay_aviso():
    p = _permanencia_pelada()
    p._ultima_persona_ts = MARTES_MEDIODIA - 10_000  # llevaba HORAS sin gente
    p._vigilar_escena(MARTES_MEDIODIA, hay_personas=True)
    assert p._sin_personas_s is None                 # verla la absuelve


def test_sin_personas_poco_tiempo_no_alarma():
    p = _permanencia_pelada()
    ahora = MARTES_MEDIODIA
    p._ultima_persona_ts = ahora - (AVISO_SIN_PERSONAS_S / 2)
    p._vigilar_escena(ahora, hay_personas=False)
    assert p._sin_personas_s is None


def test_sin_personas_demasiado_tiempo_delata_la_escena():
    p = _permanencia_pelada()
    ahora = MARTES_MEDIODIA
    p._ultima_persona_ts = ahora - (AVISO_SIN_PERSONAS_S + 60)
    p._vigilar_escena(ahora, hay_personas=False)
    assert p._sin_personas_s is not None
    assert p._sin_personas_s >= AVISO_SIN_PERSONAS_S


def test_el_aviso_cruza_la_medianoche_sin_reiniciarse():
    """El silencio se mide por DELTA de época, no por jornada ni fecha.

    4-sep-2026: frontera explícita. Si alguien reescribiera _vigilar_escena
    para "contar desde hoy" (como hace tienda_alertas con jornada_de), una
    cámara que enmudece a las 23:56 quedaría absuelta a las 00:01 solo por
    el cambio de día. La última persona vista a las 23:56 del 31-dic y el
    ciclo a las 00:01 del 1-ene son 300 s de silencio, y deben delatarse.
    """
    p = _permanencia_pelada()
    ahora = _instante(2027, 1, 1, 0, 1)
    p._ultima_persona_ts = _instante(2026, 12, 31, 23, 56)
    assert ahora - p._ultima_persona_ts == AVISO_SIN_PERSONAS_S
    p._vigilar_escena(ahora, hay_personas=False)
    assert p._sin_personas_s == AVISO_SIN_PERSONAS_S
    # Y el primer aviso de log queda fechado en ese ciclo, no en 0.0.
    assert p._ultimo_aviso_escena == ahora


def test_el_resumen_publica_el_aviso_para_el_panel():
    import inspect
    fuente = inspect.getsource(PermanenciaArea._resumen)
    assert "'sin_personas_s'" in fuente
    # Y el panel del cliente detecciones lo convierte en texto visible.
    panel = (RAIZ / 'clients' / 'detecciones' / 'src' / 'gui' / 'components'
             / 'permanencia_panel.py').read_text(encoding='utf-8')
    assert 'sin_personas_s' in panel


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
