"""El modulo de deteccion RF-DETR compartido (2-sep-2026): traduccion,
filtro de clases y el sustituto de .track(persist=True). Sin GPU: todo con
dobles — la carga real la cubre el smoke de despliegue."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

import supervision as sv  # noqa: E402

from src.analityc.core import deteccion_compartida as dc  # noqa: E402


def test_la_tabla_cubre_lo_que_piden_los_pipelines():
    """base_perimeter pide [0,1,2,3,5,7,16]; autolavado [2,3,5,7]; todos
    deben tener nombre en la tabla (si no, el filtro los descartaria)."""
    pedidos = {0, 1, 2, 3, 5, 7, 16}
    assert pedidos <= set(dc.NOMBRE_A_PSEUDO.values())


def test_la_tabla_coincide_con_el_espacio_91_real():
    rfdetr = pytest.importorskip('rfdetr')
    from rfdetr.detr import COCO_CLASSES
    ids = {n: i for i, n in COCO_CLASSES.items()}
    # Por NOMBRE: person=1, car=3, motorcycle=4, bus=6, truck=8 en el 91.
    assert ids['person'] == 1 and ids['car'] == 3 and ids['truck'] == 8
    # Y train (7 en el 91) NO esta en nuestra tabla: por numero seria truck.
    assert 'train' not in dc.NOMBRE_A_PSEUDO


class _DetFalso:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy, self.confidence, self.class_id = xyxy, confidence, class_id
    def __len__(self):
        return len(self.class_id)


def _detector_doble():
    d = dc._DetectorCompartido.__new__(dc._DetectorCompartido)
    d._id91_a_pseudo = {1: 0, 3: 2, 4: 3, 6: 5, 8: 7, 18: 16}
    d._candado = threading.Lock()
    return d


def test_detectar_traduce_filtra_y_devuelve_sv(monkeypatch):
    d = _detector_doble()
    falso = _DetFalso(
        xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 11, 11]]),
        confidence=np.array([0.9, 0.8, 0.7]),
        class_id=np.array([1, 3, 7]),   # person, car, train(91)->descartado
    )
    d._modelo = type('M', (), {'predict': lambda self, *a, **k: falso})()
    det = d.detectar(np.zeros((10, 10, 3), np.uint8))
    assert isinstance(det, sv.Detections) and len(det) == 2
    assert det.class_id.tolist() == [0, 2]          # pseudo-80
    # Filtro de clases (el classes= de ultralytics):
    solo_persona = d.detectar(np.zeros((10, 10, 3), np.uint8), clases=[0])
    assert solo_persona.class_id.tolist() == [0]


def test_detectar_vacio_es_detections_vacio():
    d = _detector_doble()
    d._modelo = type('M', (), {'predict': lambda self, *a, **k: None})()
    det = d.detectar(np.zeros((4, 4, 3), np.uint8))
    assert isinstance(det, sv.Detections) and len(det) == 0


def test_rastreador_persistente_mantiene_ids(monkeypatch):
    """Dos llamadas con la misma caja => el MISMO track_id (persist=True)."""
    caja = sv.Detections(xyxy=np.array([[100., 100., 200., 300.]]),
                         confidence=np.array([0.9]),
                         class_id=np.array([0]))
    monkeypatch.setattr(dc, 'detectar_sv', lambda *a, **k: caja)
    r = dc.RastreadorPersistente(frame_rate=10)
    frame = np.zeros((480, 640, 3), np.uint8)
    ids = set()
    for _ in range(6):
        for t in r.seguir(frame):
            ids.add(t['track_id'])
            assert set(t) == {'track_id', 'bbox', 'conf', 'cls_coco'}
    assert len(ids) == 1, ids
