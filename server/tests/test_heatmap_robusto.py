"""
Pruebas del mapa de calor robusto (14-ago-2026).

Lo que más se protege:

  - **la normalización por percentil** — una sola zona muy caliente (la
    fila de la caja) NO puede aplastar el resto del mapa a invisible, que
    era el defecto del render lineal;
  - **el ancla de escena** — si el operador cambia el canal de iVMS, los
    pies de OTRA escena no contaminan el acumulado; al volver al canal,
    se reanuda; si el cambio se sostiene, se re-ancla con mapa nuevo;
  - **el ancla sobrevive al reinicio** (viaja en el estado .npz);
  - **la foto de referencia del anaquel** solo se toma con el área VACÍA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics.heatmap import (  # noqa: E402
    HeatmapAccumulator, _mapa_uint8, render_calor_sobre)

FRAME_A = np.zeros((180, 240, 3), dtype=np.uint8)
FRAME_A[40:90, 30:200] = 180                    # escena A: franja clara
FRAME_B = np.zeros((180, 240, 3), dtype=np.uint8)
FRAME_B[100:170, 10:120] = 90                   # escena B: otra estructura
FRAME_B[10:40, 150:230] = 220


# ── Normalización robusta ────────────────────────────────────────────────
def test_percentil_no_deja_que_un_pico_aplaste_el_mapa():
    grid = np.zeros((54, 96), np.float32)
    grid[10, 10] = 1000.0                       # la fila de la caja
    grid[30, 20:70] = 10.0                      # el resto de la tienda
    norm = _mapa_uint8(grid, clip_pct=98.0, gamma=0.7)
    # Render lineal viejo: 10/1000*255 ≈ 2 (invisible). Robusto: visible.
    assert int(norm[30, 40]) >= 150
    assert int(norm[10, 10]) == 255             # el pico satura, no manda


def test_render_no_revienta_con_grilla_vacia_o_imagen_chica():
    vacia = np.zeros((54, 96), np.float32)
    out = render_calor_sobre(FRAME_A.copy(), vacia)
    assert out.shape == FRAME_A.shape
    grid = np.zeros((54, 96), np.float32)
    grid[5, 5] = 3.0
    chica = np.zeros((40, 60, 3), np.uint8)     # muy chica para leyenda
    out2 = render_calor_sobre(chica, grid, con_leyenda=True, titulo='x')
    assert out2.shape == chica.shape


def test_el_calor_pinta_y_la_leyenda_se_graba():
    grid = np.zeros((54, 96), np.float32)
    grid[27, 48] = 8.0
    out = render_calor_sobre(FRAME_A.copy(), grid, con_leyenda=True,
                             titulo='cam-test', subtitulo='99 registros')
    assert out.shape == FRAME_A.shape
    assert int(np.abs(out.astype(int) - FRAME_A.astype(int)).sum()) > 0


# ── Ancla de escena ──────────────────────────────────────────────────────
def _persona(acc, t=None):
    acc.add_person((100, 40, 140, 160), 240, 180)


def test_escena_ajena_no_contamina_y_al_volver_reanuda():
    acc = HeatmapAccumulator()
    t0 = 1000.0
    acc.observar_escena(FRAME_A, ahora=t0)      # ancla en la escena A
    _persona(acc)
    base = acc._samples
    assert base == 1
    # El operador cambia el canal: escena B — NADA se acumula.
    acc.observar_escena(FRAME_B, ahora=t0 + 2)
    for _ in range(5):
        _persona(acc)
    assert acc._samples == base
    # Vuelve al canal correcto: se reanuda solo.
    acc.observar_escena(FRAME_A, ahora=t0 + 4)
    _persona(acc)
    assert acc._samples == base + 1


def test_sin_senal_no_acumula():
    acc = HeatmapAccumulator()
    acc.observar_escena(FRAME_A, ahora=1000.0)
    negro = np.zeros((180, 240, 3), np.uint8)   # «NO VIDEO»
    acc.observar_escena(negro, ahora=1002.0)
    _persona(acc)
    assert acc._samples == 0


def test_cambio_sostenido_re_ancla_con_mapa_nuevo():
    acc = HeatmapAccumulator()
    t0 = 1000.0
    acc.observar_escena(FRAME_A, ahora=t0)
    _persona(acc)
    assert acc._samples == 1
    # Escena B sostenida más allá del umbral de re-ancla (1800 s).
    acc.observar_escena(FRAME_B, ahora=t0 + 10)
    acc.observar_escena(FRAME_B, ahora=t0 + 2000)
    # Re-anclado: el mapa viejo se descartó (el histórico horario queda en
    # disco) y la escena B es la nueva casa.
    assert acc._samples == 0
    _persona(acc)
    assert acc._samples == 1


def test_el_ancla_sobrevive_al_reinicio(tmp_path):
    acc = HeatmapAccumulator()
    acc.observar_escena(FRAME_A, ahora=1000.0)
    _persona(acc)
    assert acc.guardar_estado('cam-x', out_dir=str(tmp_path))

    acc2 = HeatmapAccumulator()
    assert acc2.cargar_estado('cam-x', out_dir=str(tmp_path))
    assert acc2._ancla is not None
    # El proceso renació y lo primero que ve es OTRA escena: el ancla
    # restaurada la rechaza (antes, la primera escena vista re-anclaba).
    acc2.observar_escena(FRAME_B, ahora=2000.0)
    muestras = acc2._samples
    _persona(acc2)
    assert acc2._samples == muestras


# ── Foto de referencia del anaquel ───────────────────────────────────────
def test_ref_del_area_solo_con_el_area_vacia(tmp_path):
    from src.analityc.core.anaqueles import MedidorAnaqueles, _DEFECTOS
    from src.analityc.core.anaqueles_registro import RegistroAnaqueles

    class PoseNula:
        disponible = False

        def munecas(self, frame, box):
            return []

    class EscenaMuda:
        def evaluar(self, *a, **k):
            return None

    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    area = [[100, 100], [300, 100], [300, 200], [100, 200]]
    m = MedidorAnaqueles(
        'cam-ref', dict(_DEFECTOS), pose=PoseNula(),
        registro=RegistroAnaqueles('cam-ref', raiz=tmp_path),
        escena=EscenaMuda(), identificador=None)
    m.poner_zonas([area], ['Anaquel 1'])
    ref = tmp_path / 'cam-ref' / 'fotos_ref' / '0.jpg'

    # Con una persona EN el área: no se toma la foto.
    ocupado = {1: {'class': -1, 'box': [120, 120, 220, 460],
                   'persistent_id': None}}
    m.procesar(frame, ocupado, ahora=1000.0)
    assert not ref.exists()

    # Área vacía: la foto de referencia aparece.
    m.procesar(frame, {}, ahora=1010.0)
    assert ref.exists()
