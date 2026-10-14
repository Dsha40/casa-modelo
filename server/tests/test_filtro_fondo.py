"""
Descarte de la gente del fondo por su alto en píxeles (`filtro_fondo.py`,
2-oct-2026).

Lo que fijan estas pruebas: que se tira a quien está por debajo del umbral y
a nadie más, que con el umbral en 0 (el valor global) no cambia nada, que un
fallo nunca se come detecciones, y que el filtro va ANTES del seguimiento: si
se moviera detrás, la gente del fondo volvería a dejar seguimientos y calor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analityc.core import filtro_fondo  # noqa: E402


def _det(altos):
    xyxy = np.array([[10.0, 100.0, 60.0, 100.0 + h] for h in altos])
    return sv.Detections(xyxy=xyxy, confidence=np.full(len(altos), 0.9),
                         class_id=np.zeros(len(altos), dtype=int))


def test_tira_a_los_bajos_y_deja_al_resto():
    d = filtro_fondo.filtrar(_det([120, 234, 235, 400]), 235, 'cam')
    altos = (d.xyxy[:, 3] - d.xyxy[:, 1]).tolist()
    assert altos == [235.0, 400.0]
    assert d.confidence.tolist() == [0.9, 0.9]


def test_con_umbral_cero_no_cambia_nada():
    entrada = _det([10, 20])
    assert filtro_fondo.filtrar(entrada, 0, 'cam') is entrada
    assert filtro_fondo.filtrar(entrada, None, 'cam') is entrada


def test_sin_detecciones_devuelve_lo_mismo():
    vacio = sv.Detections.empty()
    assert filtro_fondo.filtrar(vacio, 235, 'cam') is vacio
    assert filtro_fondo.filtrar(None, 235, 'cam') is None


def test_un_fallo_no_se_come_detecciones():
    """Mejor contar de más que perder clientes por un error."""
    raro = object()
    assert filtro_fondo.filtrar(raro, 235, 'cam') is raro


def test_el_global_esta_apagado():
    from src.analityc.core.analytics.config import AnalyticsConfig
    assert AnalyticsConfig.PERSON_MIN_HEIGHT_PX == 0


def test_el_perfil_del_video_lo_activa_y_las_demas_camaras_no():
    """El perfil se carga por el nombre SIN sufijo de instalación, y la
    clave nueva no se descarta como desconocida."""
    from src.analityc.core.analytics.demographics import _load_camera_profile
    video = _load_camera_profile('win-Media_Player-c3-i58bf7b')
    assert video.PERSON_MIN_HEIGHT_PX == 50        # 235 hasta el 12-oct, 180 hasta el 14-oct-2026
    # La cámara 3 en vivo tiene su propio perfil desde el 6-oct (200 px;
    # 220 desde el 9-oct, pedido por el usuario); una cámara sin perfil
    # sigue sin filtro.
    assert _load_camera_profile('win-iVMS-4200-c3-i58bf7b').PERSON_MIN_HEIGHT_PX == 150  # 220 hasta el 12-oct-2026
    assert _load_camera_profile('win-iVMS-4200-c4-i58bf7b').PERSON_MIN_HEIGHT_PX == 0


def test_el_filtro_va_antes_del_seguimiento():
    """Sobre el código: detectar -> filtro_fondo (MARCA) -> ByteTrack, en ese
    orden. Desde el 13-oct-2026 marca en vez de borrar; quien nunca valió lo
    quita `filtro_memoria` detrás de ByteTrack (test_filtro_memoria.py)."""
    src = (Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    i_det = src.index('deteccion_compartida.detectar_sv(')
    i_fil = src.index('filtro_fondo.mascara(', i_det)
    i_trk = src.index('self._byte_tracker.update_with_detections(', i_det)
    assert i_det < i_fil < i_trk


def test_la_mascara_marca_lo_mismo_que_quitaba_el_filtro():
    det = _det([120, 234, 235, 400])
    assert filtro_fondo.mascara(det, 235, 'cam').tolist() == [False, False, True, True]
    assert filtro_fondo.mascara(det, 0, 'cam').tolist() == [True] * 4
    assert filtro_fondo.mascara(object(), 235, 'cam').tolist() == []
