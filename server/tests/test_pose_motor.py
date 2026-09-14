"""El motor de pose (ViTPose, único desde el 3-sep-2026): la fábrica sin
elección, el error ÚNICO cuando faltan los pesos, el aviso ante un
POSE_MOTOR heredado y el contrato de `keypoints_17` — 17 keypoints COCO
[x, y, conf] en coordenadas de la imagen dada. Sin GPU y sin cargar modelos:
todo con dobles, salvo el test de carga real que se salta solo mientras los
pesos no estén descargados (y aun entonces corre en CPU: los tests no tocan
la GPU)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import pose_motor as pm  # noqa: E402


# ── Fábrica ─────────────────────────────────────────────────────────────
def _con_pesos_falsos(tmp_path, monkeypatch):
    """Simula que los pesos de ViTPose están en disco (sin pesar 700 MB)."""
    (tmp_path / 'model.safetensors').write_bytes(b'x')
    (tmp_path / 'config.json').write_text('{}')   # la compuerta exige ambos
    monkeypatch.setattr(pm, 'RUTA_VITPOSE', tmp_path)


def _sin_avisos_previos(monkeypatch):
    """Cada test arranca como un proceso nuevo: sin avisos ya emitidos."""
    monkeypatch.setattr(pm, '_avisos_emitidos', set())


def test_defecto_es_vitpose_y_no_carga_nada(tmp_path, monkeypatch):
    monkeypatch.delenv('POSE_MOTOR', raising=False)
    _con_pesos_falsos(tmp_path, monkeypatch)
    motor = pm.crear_motor_pose()
    assert isinstance(motor, pm._MotorViTPose)
    # La fábrica se llama dentro del bucle de inferencia: crear el motor no
    # puede cargar el modelo (eso pasa perezosamente en el primer uso).
    assert motor._modelo is None and not motor._intentado


def test_sin_pesos_devuelve_none_y_un_solo_error(tmp_path, monkeypatch,
                                                 caplog):
    """Sin rollback: la fábrica entrega igual un ViTPose, keypoints_17 da
    None (los consumidores ya lo toleran) y el error sale UNA vez aunque el
    stand y anaqueles creen cada uno su motor y lo usen varias veces."""
    monkeypatch.delenv('POSE_MOTOR', raising=False)
    monkeypatch.setattr(pm, 'RUTA_VITPOSE', tmp_path)   # carpeta vacía
    _sin_avisos_previos(monkeypatch)
    # Si el motor intentara importar transformers sin pesos, el ImportError
    # acabaría en un SEGUNDO error de log y el test lo cazaría.
    monkeypatch.setitem(sys.modules, 'transformers', None)
    imagen = np.zeros((80, 40, 3), np.uint8)
    with caplog.at_level(logging.DEBUG, logger=pm.logger.name):
        m1 = pm.crear_motor_pose()
        m2 = pm.crear_motor_pose(device='cuda:0')   # anaqueles pasa device
        assert isinstance(m1, pm._MotorViTPose)
        assert isinstance(m2, pm._MotorViTPose)
        assert m1.keypoints_17(imagen) is None
        assert m1.keypoints_17(imagen) is None
        assert m2.keypoints_17(imagen) is None
    assert not m1.disponible and not m2.disponible
    errores = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errores) == 1, 'el error de pesos ausentes debe salir UNA vez'
    assert 'ViTPose' in errores[0].message
    assert str(tmp_path) in errores[0].message


def test_pose_motor_yolo_solo_avisa(tmp_path, monkeypatch, caplog):
    """Un compose viejo con ELDE_POSE_MOTOR=yolo no elige nada: sale un
    aviso (una vez) diciendo que el rollback se retiró y se sigue con
    ViTPose."""
    monkeypatch.setenv('POSE_MOTOR', 'yolo')
    _con_pesos_falsos(tmp_path, monkeypatch)
    _sin_avisos_previos(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=pm.logger.name):
        m1 = pm.crear_motor_pose(device='cuda:0')
        m2 = pm.crear_motor_pose()
    assert isinstance(m1, pm._MotorViTPose)
    assert isinstance(m2, pm._MotorViTPose)
    assert not hasattr(pm, '_MotorYoloPose')
    avisos = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(avisos) == 1
    assert 'el rollback yolo11m-pose se retiro el 3-sep-2026' in avisos[0].message
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_pose_motor_desconocido_tambien_avisa(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('POSE_MOTOR', 'banana')
    _con_pesos_falsos(tmp_path, monkeypatch)
    _sin_avisos_previos(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=pm.logger.name):
        motor = pm.crear_motor_pose()
    assert isinstance(motor, pm._MotorViTPose)
    assert any('POSE_MOTOR' in r.message for r in caplog.records)


def test_pose_motor_vitpose_explicito_no_avisa(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('POSE_MOTOR', 'vitpose')
    _con_pesos_falsos(tmp_path, monkeypatch)
    _sin_avisos_previos(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=pm.logger.name):
        pm.crear_motor_pose()
    assert not caplog.records


# ── Contrato del adaptador ViTPose (con dobles, sin transformers) ───────
class _ProcesadorFalso:
    """Registra lo que recibe y devuelve tensores canned de torch."""

    def __init__(self, kps, scores):
        import torch
        self.imagenes = []
        self.cajas = []
        self._kps = torch.tensor(kps, dtype=torch.float32)
        self._scores = torch.tensor(scores, dtype=torch.float32)

    def __call__(self, images, boxes, return_tensors):
        import torch
        assert return_tensors == 'pt'
        self.imagenes.append(np.asarray(images))
        self.cajas.append(boxes)
        return {'pixel_values': torch.zeros((1, 3, 256, 192))}

    def post_process_pose_estimation(self, outputs, boxes):
        self.cajas.append(boxes)
        return [[{'keypoints': self._kps, 'scores': self._scores}]]


def _motor_vitpose_doble(kps, scores):
    m = pm._MotorViTPose(device='cpu')
    m._intentado = True                  # que no intente cargar nada real
    m._dispositivo = 'cpu'
    m._fp16 = False
    m._procesador = _ProcesadorFalso(kps, scores)
    m._modelo = lambda pixel_values: object()   # la salida solo se re-pasa
    return m


def test_vitpose_convierte_bgr_caja_entera_y_17x3():
    pytest.importorskip('torch')
    kps = [[float(i), float(i) + 0.5] for i in range(17)]
    scores = [0.9] * 17
    m = _motor_vitpose_doble(kps, scores)
    # Pixel BGR = [255, 10, 0] (azul puro): si llega igual al procesador,
    # nadie convirtió a RGB y las poses saldrían con los colores cruzados.
    imagen = np.zeros((80, 40, 3), dtype=np.uint8)
    imagen[:, :] = (255, 10, 0)
    salida = m.keypoints_17(imagen)
    assert salida is not None and len(salida) == 17
    # muñeca izq: [x, y, conf] (approx: el 0.9 viaja como float32)
    assert salida[9] == pytest.approx([9.0, 9.5, 0.9])
    assert all(len(k) == 3 for k in salida)
    proc = m._procesador
    assert proc.imagenes[0][0, 0].tolist() == [0, 10, 255]   # RGB, no BGR
    # La caja del top-down es la imagen ENTERA en formato COCO (x, y, w, h):
    # quien llama ya recortó a la persona.
    assert proc.cajas[0] == [[[0.0, 0.0, 40.0, 80.0]]]
    assert proc.cajas[1] == proc.cajas[0]        # las mismas al post-proceso


def test_vitpose_sin_17_completos_es_none():
    pytest.importorskip('torch')
    m = _motor_vitpose_doble([[1.0, 2.0]] * 5, [0.9] * 5)   # solo 5 kps
    assert m.keypoints_17(np.zeros((80, 40, 3), np.uint8)) is None


def test_vitpose_sin_modelo_es_none_y_nunca_lanza():
    m = pm._MotorViTPose()
    m._intentado = True                  # la carga «falló»: modelo None
    assert m.keypoints_17(np.zeros((80, 40, 3), np.uint8)) is None
    assert not m.disponible


# ── Carga real (se salta sola mientras los pesos se descargan) ──────────
_PESOS = pm.RUTA_VITPOSE / 'model.safetensors'


@pytest.mark.skipif(not _PESOS.is_file(),
                    reason='pesos ViTPose aún no descargados')
def test_carga_real_vitpose_en_cpu():
    """Humo de verdad: pesos locales + forward en CPU (la GPU es de los
    servidores; ver la regla de los tests)."""
    motor = pm._MotorViTPose(device='cpu')
    rng = np.random.default_rng(7)
    imagen = rng.integers(0, 255, (160, 96, 3)).astype(np.uint8)
    kps = motor.keypoints_17(imagen)
    assert motor.disponible, 'el modelo local debió cargar sin red'
    assert kps is not None and len(kps) == 17
    assert all(len(k) == 3 and all(np.isfinite(v) for v in k) for k in kps)
