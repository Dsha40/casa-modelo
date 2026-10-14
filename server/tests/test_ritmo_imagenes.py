"""
Ajustes contados en imágenes pasados a SEGUNDOS (`ritmo_imagenes.py`,
5-oct-2026).

Lo que fijan estas pruebas:

- la regla «nunca menos exigente que hoy»: a los fps ya validados (~5-8) los
  ajustes quedan IGUAL que antes; solo se endurecen al subir los fps;
- la tabla del módulo, con los valores reales de la configuración;
- el ByteTrack de verdad recibe la memoria y la confirmación en segundos;
- el agarre: a 15 fps, 2 imágenes con la mano dentro ya NO bastan (eran
  0,13 s) y sí 4; a 2 fps, 2 siguen bastando, como siempre;
- tiendas.py ajusta el ByteTrack ANTES de cada actualización.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import ritmo_imagenes as ri  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.anaqueles import _DEFECTOS, MedidorAnaqueles  # noqa: E402
# Estas pruebas fijan la MÁQUINA del gesto: desde el 12-oct-2026 los
# defectos esperan la bajada para contar el agarre
# (test_anaqueles_confirmacion.py).
_REGLAS_DEL_GESTO = {'visita_por_muneca': False, 'agarre_requiere_bajada': False}
from src.analityc.core.anaqueles_registro import RegistroAnaqueles  # noqa: E402


# ── Ritmo e imagenes_para ────────────────────────────────────────────────
def test_el_ritmo_promedia_y_no_cuenta_las_pausas():
    r = ri.Ritmo()
    assert r.fps is None                          # sin medir todavía
    for i in range(30):
        r.anotar(100.0 + i * 0.1)                 # 10 fps
    assert r.fps == pytest.approx(10.0, rel=1e-6)
    r.anotar(100.0 + 29 * 0.1 + 60.0)             # vídeo parado un minuto
    assert r.fps == pytest.approx(10.0, rel=1e-6)


def test_imagenes_para_nunca_baja_del_suelo():
    assert ri.imagenes_para(0.25, 2, None) == 2   # sin fps: el suelo
    assert ri.imagenes_para(0.0, 2, 30.0) == 2    # 0 s: solo imágenes
    assert ri.imagenes_para(0.25, 2, 3.0) == 2    # pocos fps: el suelo
    assert ri.imagenes_para(0.25, 2, 8.0) == 2    # 0,25 x 8 = 2 exacto
    assert ri.imagenes_para(0.25, 2, 15.0) == 4


@pytest.mark.parametrize('fps, agarre, perdido, confirmar', [
    (5.2, 2, 90, 3),       # pasada C del 5-oct: como siempre
    (7.7, 2, 90, 3),       # pasada G del 5-oct: como siempre
    (15.0, 4, 135, 6),     # lo que se endurece al subir los fps
])
def test_la_tabla_con_los_valores_reales(fps, agarre, perdido, confirmar):
    assert ri.imagenes_para(_DEFECTOS['agarre_min_s'],
                            _DEFECTOS['agarre_frames_min'], fps) == agarre
    assert ri.imagenes_para(_DEFECTOS['agarre_rearme_s'],
                            _DEFECTOS['agarre_rearme_frames'], fps) == agarre
    base = int(AnalyticsConfig.BYTETRACK_FRAME_RATE / 30.0
               * AnalyticsConfig.BYTETRACK_LOST_BUFFER)
    assert ri.imagenes_para(AnalyticsConfig.BYTETRACK_LOST_S, base,
                            fps) == perdido
    assert ri.imagenes_para(AnalyticsConfig.BYTETRACK_MIN_CONSECUTIVE_S,
                            AnalyticsConfig.BYTETRACK_MIN_CONSECUTIVE_FRAMES,
                            fps) == confirmar


# ── ByteTrack ────────────────────────────────────────────────────────────
def _bytetrack():
    return sv.ByteTrack(
        lost_track_buffer=AnalyticsConfig.BYTETRACK_LOST_BUFFER,
        frame_rate=AnalyticsConfig.BYTETRACK_FRAME_RATE,
        minimum_consecutive_frames=AnalyticsConfig.BYTETRACK_MIN_CONSECUTIVE_FRAMES)


@pytest.mark.parametrize('fps, perdido, confirmar', [
    (5.0, 90, 3), (15.0, 135, 6)])
def test_el_bytetrack_recibe_los_segundos(fps, perdido, confirmar):
    bt = _bytetrack()
    for i in range(20):
        ri.ajustar_bytetrack(bt, 1000.0 + i / fps, AnalyticsConfig)
        bt.update_with_detections(sv.Detections.empty())
    assert bt.max_time_lost == perdido
    assert bt.minimum_consecutive_frames == confirmar


def test_ajustar_el_bytetrack_nunca_lanza():
    ri.ajustar_bytetrack(object(), 1.0, AnalyticsConfig)


def test_tiendas_ajusta_antes_de_actualizar():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    i_aj = src.index('ritmo_imagenes.ajustar_bytetrack(')
    i_up = src.index('self._byte_tracker.update_with_detections(', i_aj)
    assert i_aj < i_up


# ── El resto de ajustes en imágenes (ROI, fichas, género, track único) ───
#: Los valores de hoy de cada ajuste convertido con `imagenes_ref`.
_DE_HOY = [1, 2, 3, 4, 5, 10, 12, 30, 200]


@pytest.mark.parametrize('fps', [5.0, 7.7])
def test_a_los_fps_validados_todo_queda_igual(fps):
    for n in _DE_HOY:
        assert ri.imagenes_ref(n, fps) == n
    assert ri.imagenes_ref(3, None) == 3          # sin fps medidos


def test_a_10_fps_duran_el_mismo_tiempo_que_a_7_7():
    # 3 imágenes = 0,39 s -> 4 a 10 fps; 1 -> 2; 30 (anti-duplicado) -> 39.
    assert [ri.imagenes_ref(n, 10.0) for n in (1, 3, 5, 10, 30)] == [2, 4, 7, 13, 39]


def test_tiendas_toma_los_fps_de_la_camara():
    from types import SimpleNamespace
    from src.analityc.core.tiendas import TiendasProcess
    bt = SimpleNamespace(_elde_ritmo=ri.Ritmo())
    for i in range(20):
        bt._elde_ritmo.anotar(100.0 + i * 0.1)                  # 10 fps
    assert TiendasProcess._imagenes(SimpleNamespace(_byte_tracker=bt), 3) == 4
    assert TiendasProcess._imagenes(SimpleNamespace(), 3) == 3   # sin tracker


def test_ningun_ajuste_del_roi_ni_del_genero_queda_en_imagenes_sueltas():
    """Las comparaciones van por `_imagenes`. Si alguien vuelve a comparar
    con el número a pelo, a más fps vuelve a durar menos tiempo."""
    core = RAIZ / 'server' / 'src' / 'analityc' / 'core'
    src = (core / 'tiendas.py').read_text(encoding='utf-8')
    viejos = [
        ">= self.max_frames_without_detection:",
        "% self._classify_every_n ==",
        "hyst = max(1, int(AnalyticsConfig.ZONE_HYSTERESIS_FRAMES))",
        "_every = max(1, int(AnalyticsConfig.BODY_REID_EVERY_N))",
        "_min_seen = int(AnalyticsConfig.BODY_REID_MIN_SEEN_FRAMES)",
        "(self.frame_counter - counted_frame) < 30",
        ">= self.exit_frames_threshold:",
        "> self.max_frames_out:",
        ">= self.min_track_frames and",
        ">= self.min_time_in_roi and",
    ]
    for viejo in viejos:
        assert viejo not in src, f'vuelve a contar en imágenes: {viejo}'
    fotos = (core / 'tiendas_fotos.py').read_text(encoding='utf-8')
    assert 'kmin = _imagenes(kmin)' in fotos


# ── Agarre (montaje de test_anaqueles_cercania) ──────────────────────────
FRAME = np.zeros((700, 640, 3), dtype=np.uint8)
MEDIODIA = time.mktime((2026, 10, 5, 12, 0, 0, 0, 0, -1))
AREA = [[100, 100], [300, 100], [300, 200], [100, 200]]       # balda alta
BOX_DEBAJO = [150, 260, 250, 600]                             # la clienta
ABAJO = [(200, 300, 0.9)]                                     # mano fuera
ARRIBA = [(200, 150, 0.9)]                                    # mano dentro


class _Pose:
    def __init__(self, guion):
        self.guion, self.llamadas, self.disponible = list(guion), 0, True

    def munecas(self, frame, box):
        i = min(self.llamadas, len(self.guion) - 1)
        self.llamadas += 1
        return list(self.guion[i])


class _Escena:
    def evaluar(self, camara, frame, ahora=None):
        return None


class _SinPersonal:
    def ficha_de(self, pid, reidentifier):
        return None


@pytest.fixture(autouse=True)
def _reloj(monkeypatch):
    strftime_real, localtime_real = time.strftime, time.localtime
    monkeypatch.setattr(time, 'time', lambda: MEDIODIA)
    monkeypatch.setattr(time, 'strftime', lambda f, t=None: strftime_real(
        f, localtime_real(MEDIODIA) if t is None else t))


def _agarres(tmp_path, guion, fps):
    cfg = dict(_DEFECTOS, **_REGLAS_DEL_GESTO)
    cfg['pose_cercania'] = True
    m = MedidorAnaqueles('cam', cfg, pose=_Pose(guion),
                         registro=RegistroAnaqueles('cam', raiz=tmp_path),
                         escena=_Escena(), identificador=_SinPersonal())
    m.poner_zonas([AREA], ['Anaquel 1'])
    for i in range(len(guion)):
        m.procesar(FRAME, {1: {'class': -1, 'box': list(BOX_DEBAJO),
                               'persistent_id': None}},
                   ahora=MEDIODIA + i / fps)
    return m.registro.area(0)['agarres']


def test_a_2_fps_dos_imagenes_dentro_siguen_bastando(tmp_path):
    """Como siempre: a 2 fps, 2 imágenes son 1 s de mano dentro."""
    assert _agarres(tmp_path, [ABAJO] * 4 + [ARRIBA] * 2 + [ABAJO] * 2,
                    fps=2.0) == 1


def test_a_15_fps_dos_imagenes_dentro_ya_no_bastan(tmp_path):
    """0,13 s de mano dentro: antes contaba, ahora no (exige 0,25 s)."""
    assert _agarres(tmp_path, [ABAJO] * 25 + [ARRIBA] * 2 + [ABAJO] * 5,
                    fps=15.0) == 0


def test_a_15_fps_el_agarre_real_cuenta(tmp_path):
    assert _agarres(tmp_path, [ABAJO] * 25 + [ARRIBA] * 8, fps=15.0) == 1
