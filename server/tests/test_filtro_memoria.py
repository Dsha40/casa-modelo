"""Filtro del fondo y áreas a ignorar CON MEMORIA (13-oct-2026,
`filtro_memoria.py`).

Con el ByteTrack real (supervision): quien rondaba el umbral ya no se parte
en dos números; la gente del fondo sigue sin contar; un fallo nunca deja
pasar al fondo; y `tiendas.py` marca ANTES de ByteTrack y filtra DESPUÉS.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analityc.core import filtro_fondo, filtro_memoria, zonas_ignorar  # noqa: E402

warnings.filterwarnings('ignore')
UMBRAL = 150


def _tracker():
    return sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=90,
                        minimum_matching_threshold=0.8, frame_rate=15,
                        minimum_consecutive_frames=1)


def _det(cajas):
    return sv.Detections(xyxy=np.array(cajas, dtype=float),
                         confidence=np.full(len(cajas), 0.9),
                         class_id=np.zeros(len(cajas), dtype=int))


def _paso(bt, cajas, umbral=UMBRAL, zonas=None, memoria=True):
    det = _det(cajas)
    valida = filtro_fondo.mascara(det, umbral, 'cam') & zonas_ignorar.mascara(det, zonas, 'cam')
    if not memoria:                          # lo de antes: borrar ANTES de ByteTrack
        det = det[valida]
        return bt.update_with_detections(det)
    det = filtro_memoria.marcar(det, valida)
    return filtro_memoria.filtrar(bt, bt.update_with_detections(det), 'cam')


def _camina(bt, memoria):
    """Una persona que camina; en las imágenes 10-14 se agacha y su caja
    baja a 140 px (bajo el umbral de 150). Devuelve los números que tuvo."""
    numeros = []
    for i in range(30):
        alto = 140 if 10 <= i < 15 else 220
        x = 100 + 4 * i
        r = _paso(bt, [[x, 500 - alto, x + 70, 500]], memoria=memoria)
        numeros += [int(t) for t in r.tracker_id] if len(r) else []
    return set(numeros), len(numeros)


def test_quien_encoge_un_momento_conserva_su_numero():
    numeros, vistas = _camina(_tracker(), memoria=True)
    assert len(numeros) == 1 and vistas == 30     # sin huecos: se le sigue contando


def test_lo_de_antes_lo_perdia_esas_imagenes():
    """Al revés: borrando antes de ByteTrack, la persona desaparece las 5
    imágenes en que su caja baja del umbral (así se partía en dos números
    cuando el hueco duraba más de lo que ByteTrack aguanta)."""
    _numeros, vistas = _camina(_tracker(), memoria=False)
    assert vistas == 25


def test_el_fondo_nunca_cuenta():
    bt = _tracker()
    for i in range(20):
        r = _paso(bt, [[600 + i, 300, 640 + i, 400]])     # 100 px: siempre fondo
        assert len(r) == 0


def test_el_area_a_ignorar_tambien_tiene_memoria():
    """Quien estuvo FUERA del área y entra entera en ella sigue contando;
    quien siempre estuvo dentro, no."""
    zona = [[[0, 0], [400, 0], [400, 600], [0, 600]]]
    bt = _tracker()
    for i in range(20):
        x = 450 - 10 * i                                    # entra en el área
        r = _paso(bt, [[x, 200, x + 60, 500], [50, 250, 110, 520]], zonas=zona)
        assert len(r) == 1                                  # solo el que vino de fuera


def test_sin_marca_todo_pasa_y_un_fallo_no_deja_pasar_al_fondo():
    det = _det([[0, 0, 10, 10]])
    assert filtro_memoria.filtrar(_tracker(), det) is det   # sin marca
    roto = object()                                         # ByteTrack que no acepta memoria
    bt = _tracker()
    d = filtro_memoria.marcar(_det([[0, 0, 50, 300], [0, 0, 10, 10]]), np.array([True, False]))
    r = bt.update_with_detections(d)
    assert len(filtro_memoria.filtrar(roto, r, 'cam')) == 1  # cae a «solo las válidas»


def test_tiendas_marca_antes_y_filtra_despues_del_seguimiento():
    src = (Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    i_det = src.index('deteccion_compartida.detectar_sv(')
    i_marca = src.index('filtro_memoria.marcar(', i_det)
    i_trk = src.index('self._byte_tracker.update_with_detections(', i_det)
    i_filtra = src.index('filtro_memoria.filtrar(', i_det)
    assert i_det < i_marca < i_trk < i_filtra
    assert src.index('filtro_fondo.mascara(', i_det) < i_trk
    assert src.index('zonas_ignorar.mascara(', i_det) < i_trk
