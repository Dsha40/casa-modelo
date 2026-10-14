"""Pausar durante los cortes de vídeo de iVMS (13-oct-2026,
`cortes_video.py`). OPCIONAL: apagado por defecto.

Lo que fijan: está apagado salvo que se pida; la imagen repetida solo se
salta pasados 0,4 s (el vídeo normal repite como mucho una); al volver el
vídeo se alarga la espera de los géneros pendientes y, tras una pausa
larga, se reinicia el seguimiento; la respuesta en pausa no lleva cajas ni
repite alertas; y `tiendas.py`/`anaqueles.py` lo usan donde deben.

Y `sin_video` (SIEMPRE activo): reconoce la imagen gris del corte —no una
balda vacía— y con ella ni se guarda el fondo del mapa de calor ni la foto
del anaquel (lo del stock está en `test_anaqueles_stock_bajadas.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analityc.core import cortes_video as cv_  # noqa: E402
from src.analityc.core import demografia_aplazada as da  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core'


def _img(valor):
    im = np.zeros((90, 160, 3), np.uint8)
    im[10:40, 10:60] = valor
    return im


def test_apagado_por_defecto_y_se_enciende_con_la_variable(monkeypatch):
    monkeypatch.delenv('ELDE_CONGELAR_CORTES', raising=False)
    assert cv_.activo() is False
    monkeypatch.setenv('ELDE_CONGELAR_CORTES', '1')
    assert cv_.activo() is True


def test_solo_salta_la_imagen_repetida_pasados_04_s():
    est = {}
    assert cv_.congelada(est, _img(100), 0.0) is False       # nueva
    assert cv_.congelada(est, _img(100), 0.1) is False       # una repetida (normal)
    assert cv_.congelada(est, _img(120), 0.2) is False       # el vídeo sigue
    assert cv_.congelada(est, _img(120), 0.5) is False       # 0,3 s igual: aún no
    assert cv_.congelada(est, _img(120), 0.7) is True        # 0,5 s igual: corte
    assert cv_.congelada(est, _img(120), 30.0) is True
    assert cv_.congelada(est, _img(140), 30.1) is False      # vuelve el vídeo


class _Demo:
    pass


class _Tracker:
    def __init__(self):
        self.reinicios = 0

    def reset(self):
        self.reinicios += 1


def test_al_volver_alarga_los_generos_pendientes_y_no_reinicia_si_fue_corto():
    est, demo, bt = {}, _Demo(), _Tracker()
    da.aplazar(demo, 7, 100.0)
    cv_.congelada(est, _img(1), 100.0, demo, bt)
    cv_.congelada(est, _img(1), 100.5, demo, bt)            # pausa desde 100,5
    cv_.congelada(est, _img(1), 129.5, demo, bt)
    cv_.congelada(est, _img(2), 130.5, demo, bt)            # vuelve: 30 s de pausa
    assert bt.reinicios == 0
    assert da.purgar(demo, 130.5, 9.0, lambda t: None) == 0  # sin el parche, vencía
    assert da.purgar(demo, 140.0, 9.0, lambda t: None) == 1


def test_tras_una_pausa_larga_se_reinicia_el_seguimiento():
    est, bt = {}, _Tracker()
    cv_.congelada(est, _img(1), 0.0, None, bt)
    cv_.congelada(est, _img(1), 1.0, None, bt)
    cv_.congelada(est, _img(2), 1000.0, None, bt)
    assert bt.reinicios == 1


def test_la_respuesta_en_pausa_no_lleva_cajas_ni_repite_alertas():
    est = {}
    cv_.recordar(est, {'alerts': [{'x': 1}], 'detections': [{'box': [0, 0, 1, 1]}], 'zone_count': 2})
    m = cv_.metadata_congelada(est)
    assert m['alerts'] == [] and m['detections'] == [] and m['congelado'] is True
    assert m['zone_count'] == 2
    assert cv_.metadata_congelada({})['congelado'] is True


def test_un_fallo_procesa_la_imagen():
    assert cv_.congelada({}, None, 0.0) is False


def test_tiendas_y_anaqueles_lo_usan_donde_deben():
    src = (RAIZ / 'tiendas.py').read_text(encoding='utf-8')
    i = src.index('if cortes_video.activo():')
    assert i < src.index('self.frame_counter += 1', i)
    assert src.index('try:', i - 400) < i                    # dentro del try (se restaura el estado)
    assert 'cortes_video.recordar(cam_state, metadata)' in src
    anq = (RAIZ / 'anaqueles.py').read_text(encoding='utf-8')
    assert "metadata.get('congelado')" in anq


# ── Imagen sin vídeo (siempre activo) ────────────────────────────────────
def _con_video(alto=480, ancho=640):
    """Una escena: balda clara con cajas oscuras y suelo a rayas."""
    f = np.full((alto, ancho, 3), 90, np.uint8)
    f[alto * 5 // 8:] = (((np.arange(ancho) // 16) % 2) * 120 + 40).astype(np.uint8)[None, :, None]
    f[60:140, 80:560] = 200
    f[70:130, 100:180] = 30
    return f


def _gris_de_ivms(alto=852, ancho=1440):
    """La del corte: gris casi liso (ruido de JPEG) y alrededor la ventana
    de iVMS (barras, borde, botones), fuera del centro."""
    rng = np.random.default_rng(7)
    f = np.clip(rng.normal(78, 3, (alto, ancho, 1)), 0, 255).astype(np.uint8).repeat(3, 2)
    f[:40] = 30
    f[-40:] = 30
    f[:, :20] = 200
    f[5:35, 50:400:40] = 230
    return f


def test_sin_video_reconoce_la_gris_de_ivms_y_no_una_balda_vacia():
    gris = _gris_de_ivms()
    assert gris.std() > 10              # entera NO es lisa: por eso se mira el centro
    assert cv_.sin_video(gris) is True
    assert cv_.sin_video(np.zeros((480, 640, 3), np.uint8)) is True
    assert cv_.sin_video(_con_video()) is False
    vacia = _con_video()
    vacia[60:140, 80:560] = 120         # balda vacía y lisa, pero hay vídeo
    assert cv_.sin_video(vacia) is False


def test_sin_video_nunca_lanza():
    assert cv_.sin_video(None) is False
    assert cv_.sin_video(np.zeros((0, 0, 3), np.uint8)) is False
    assert cv_.sin_video('no es una imagen') is False


def test_el_mapa_de_calor_no_guarda_la_gris_de_fondo(tmp_path):
    """Antes cada 60 s: en un corte no hay nadie y el mapa no revisaba la
    escena. Con la gris se queda el último fondo bueno."""
    from src.analityc.core.analytics.heatmap import HeatmapAccumulator
    acc = HeatmapAccumulator()
    acc.add_person((100, 100, 140, 300), 640, 480)

    def foto(fondo):
        acc._last_snapshot_ts = acc._last_bg_ts = 0.0
        acc.maybe_save_snapshot('cam-gris', background=fondo,
                                out_dir=str(tmp_path), every_s=0, personas=0)
        rutas = [r for r in (tmp_path / 'bg').glob('*.jpg') if '.tmp' not in r.name]
        return cv2.imread(str(rutas[0])) if rutas else None

    assert foto(_gris_de_ivms(480, 640)) is None
    bueno = foto(_con_video())
    assert bueno is not None
    assert np.array_equal(foto(_gris_de_ivms(480, 640)), bueno)


def test_la_gris_no_sirve_de_foto_del_anaquel(tmp_path):
    from src.analityc.core.anaqueles import MedidorAnaqueles, _DEFECTOS
    from src.analityc.core.anaqueles_registro import RegistroAnaqueles

    class PoseNula:
        disponible = False

        def munecas(self, frame, box):
            return []

    class EscenaMuda:
        def evaluar(self, *a, **k):
            return None

    m = MedidorAnaqueles(
        'cam-gris', dict(_DEFECTOS), pose=PoseNula(),
        registro=RegistroAnaqueles('cam-gris', raiz=tmp_path),
        escena=EscenaMuda(), identificador=None)
    m.poner_zonas([[[100, 100], [300, 100], [300, 200], [100, 200]]], ['Anaquel 1'])
    ref = tmp_path / 'cam-gris' / 'fotos_ref' / '0.jpg'
    m.procesar(_gris_de_ivms(480, 640), {}, ahora=1000.0)
    assert not ref.exists()
    # La gris no gastó el turno de 10 min: con vídeo, la foto sale ya.
    m.procesar(_con_video(), {}, ahora=1001.0)
    assert ref.exists()
