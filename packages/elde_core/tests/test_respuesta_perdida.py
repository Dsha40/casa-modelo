"""
Red de seguridad de la ruta de VENTANA (8-oct-2026, hallazgo 30): si la
imagen en vuelo lleva más de `ESPERA_RESPUESTA_MAX_S` sin respuesta, se da
por perdida y el recuadro vuelve a enviar.

El 8-oct a las 05:45:20 se perdió una respuesta y el recuadro estuvo 3 h 27
min sin enviar: esperaba una respuesta que no iba a llegar. Lo que se fija:

- antes del plazo no se libera nada;
- pasado el plazo, la captura que llega se ENVÍA (no se guarda);
- sin IA no hay nada que liberar;
- se apunta en el registro (una vez por minuto, con la cuenta).
"""
from __future__ import annotations

import sys
from pathlib import Path

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from elde_core.ui import render_box_captura as rbc  # noqa: E402
from test_render_box_captura import (  # noqa: E402
    EtiquetaConZonas, ProcesoFalso, RecuadroFalso)


def _con_una_en_vuelo(hace_s: float):
    r = RecuadroFalso(EtiquetaConZonas())
    r.process = ProcesoFalso(msgpack.packb(
        {'header': {'x': 1}, 'image_bytes': b'\xff\xd8N'}, use_bin_type=True))
    r.can_send_next_frame = False
    r._frame_en_vuelo_ts = rbc.time.time() - hace_s
    return r


def test_antes_del_plazo_se_sigue_esperando():
    r = _con_una_en_vuelo(rbc.ESPERA_RESPUESTA_MAX_S - 1.0)
    r.loop_show_result()
    assert not r.socket.enviado
    assert r._captura_guardada is not None


def test_pasado_el_plazo_se_da_por_perdida_y_se_envia():
    r = _con_una_en_vuelo(rbc.ESPERA_RESPUESTA_MAX_S + 1.0)
    r.loop_show_result()
    assert len(r.socket.enviado) == 1
    assert r.can_send_next_frame is False          # la nueva queda en vuelo
    assert r._respuestas_perdidas == 1


def test_sin_ia_no_hay_nada_que_liberar():
    r = _con_una_en_vuelo(60.0)
    r.smart_mode = False
    assert r._liberar_si_respuesta_perdida() is False
    assert r.can_send_next_frame is False


def test_se_apunta_en_el_registro(caplog=None):
    avisos = []
    viejo = rbc.log.warning
    rbc.log.warning = lambda *a, **k: avisos.append(a)
    try:
        r = _con_una_en_vuelo(rbc.ESPERA_RESPUESTA_MAX_S + 1.0)
        assert r._liberar_si_respuesta_perdida() is True
        r.can_send_next_frame = False
        r._frame_en_vuelo_ts = rbc.time.time() - 10.0
        assert r._liberar_si_respuesta_perdida() is True
    finally:
        rbc.log.warning = viejo
    assert len(avisos) == 1                         # una vez por minuto
    assert r._respuestas_perdidas == 2
