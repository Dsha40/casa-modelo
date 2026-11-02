"""
Pruebas del ROI que ya no tira ni borra a nadie (30-oct-2026,
`ROI_FILTRA_DETECCIONES`) y de la imagen actual (`api_imagen_actual.py`).

En la casa modelo (cámara 4) el ROI cubre la entrada. Quien pasaba al resto
de la casa: (1) se tiraba antes de seguirlo («Filtro ROI») y (2) se borraba
en cuanto pisaba fuera (`exit_frames_threshold = 1`), y al volver al ROI
desde dentro era persona y visita nuevas. Lo vio el usuario en vivo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402

SRC = (RAIZ / 'server' / 'src' / 'analityc' / 'core' / 'tiendas.py'
       ).read_text(encoding='utf-8')


def test_por_defecto_el_roi_no_filtra():
    assert AnalyticsConfig.ROI_FILTRA_DETECCIONES is False


def test_el_filtro_de_detecciones_depende_del_interruptor():
    i = SRC.index('# Filtro ROI.')
    trozo = SRC[i:i + 900]
    assert "'ROI_FILTRA_DETECCIONES', False)):" in trozo
    assert 'cv2.pointPolygonTest(roi_poly' in trozo


def test_salir_del_roi_no_borra_el_seguimiento():
    i = SRC.index('def process_entry_exit_logic')
    cuerpo = SRC[i:SRC.index('def _count_staff_safe')]
    assert "_quitar_fuera = bool(getattr(AnalyticsConfig, 'ROI_FILTRA_DETECCIONES'" \
        in cuerpo
    # El `continue` va ANTES de apuntar a nadie para borrar.
    assert cuerpo.index('if not _quitar_fuera:\n                    continue') \
        < cuerpo.index('tracks_to_remove.append(track_id)')
    assert 'if track_id in self.active_tracks and _quitar_fuera:' in cuerpo


def test_la_imagen_actual_pinta_roi_e_identidades():
    from src.app import api_imagen_actual as ai

    class _Origen:
        @staticmethod
        def nacio_dentro(pid):
            return {'aaaaaaaa-1': True, 'bbbbbbbb-2': False}.get(pid)

    img = np.zeros((200, 300, 3), np.uint8)
    tracks = {
        1: {'class': -1, 'box': [10, 10, 60, 120],
            'persistent_id': 'aaaaaaaa-1', 'identity_source': 'body'},
        2: {'class': -1, 'box': [100, 10, 150, 120],
            'persistent_id': 'bbbbbbbb-2', 'identity_source': 'face'},
        3: {'class': -1, 'box': [200, 10, 250, 120],
            'persistent_id': 'prov:3'},
        4: {'class': 5, 'box': [0, 0, 5, 5]},          # no es persona
    }
    out = ai.dibujar(img, [[0, 0], [299, 0], [299, 199], [0, 199]], tracks,
                     None, _Origen())
    assert out.shape == img.shape and out.any()
    assert not img.any()                     # no toca la original


def test_app_registra_la_imagen_actual():
    src = (RAIZ / 'server' / 'src' / 'app' / 'app.py').read_text(
        encoding='utf-8')
    assert 'app.include_router(api_imagen_actual_router)' in src
