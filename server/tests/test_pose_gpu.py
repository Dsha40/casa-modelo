"""
Recortes de ViTPose preparados en la GPU y todas las personas en una sola
llamada (`pose_gpu.py`, 6-oct-2026).

Lo que fijan estas pruebas:

- `pose_gpu.preparar` da la MISMA entrada al modelo que el preparador de
  Hugging Face (scipy en la CPU), también en los bordes del recorte: ahí
  `grid_sample` mezclaba el borde con negro y scipy pone negro (el primer
  intento salía con 35-250 niveles de más en una fila). Corre en la CPU de
  torch, así que no necesita GPU;
- `munecas_lote` sin `keypoints_lote` en el motor (o si devuelve None) da
  lo mismo que `munecas` caja a caja;
- el interruptor `ELDE_POSE_PREPARAR_EN_GPU=0` apaga el lote;
- el medidor de Anaqueles pide la pose de todas las personas de una vez.

Medido aparte (no es prueba: usa las fotos de las fichas): sobre 400 fotos,
las 592 muñecas a ≤ 0,11 px del camino de antes; 3 personas, 74,5 → 26,8 ms
en la Quadro RTX 4000.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import pose_gpu  # noqa: E402
from src.analityc.core.anaqueles_pose import PoseMunecas  # noqa: E402

torch = pytest.importorskip('torch')


def _procesador():
    from src.analityc.core.pose_motor import RUTA_VITPOSE
    if not (Path(RUTA_VITPOSE) / 'preprocessor_config.json').exists():
        pytest.skip('sin la configuración de ViTPose en disco')
    from transformers import VitPoseImageProcessor
    return VitPoseImageProcessor.from_pretrained(str(RUTA_VITPOSE),
                                                 local_files_only=True)


def _imagen(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([xx / w * 250, yy / h * 250,
                     128 + 100 * np.sin(xx / 7) * np.cos(yy / 9)],
                    -1).astype(np.uint8)


# Apaisado, cuadrado y alto: cada uno rellena por un lado distinto.
@pytest.mark.parametrize('h, w', [(300, 120), (200, 200), (400, 150),
                                  (90, 260)])
def test_la_entrada_es_la_de_hugging_face(h, w):
    proc = _procesador()
    img = _imagen(h, w)                                   # BGR
    cpu = proc(images=np.ascontiguousarray(img[:, :, ::-1]),
               boxes=[[[0.0, 0.0, float(w), float(h)]]],
               return_tensors='pt')['pixel_values'][0]
    gpu = pose_gpu.preparar(img, [(0, 0, w, h)], proc.image_mean,
                            proc.image_std, 'cpu', fp16=False)[0]
    s = torch.tensor(proc.image_std).view(3, 1, 1)
    # En niveles de gris 0-255: como mucho 1 (redondeo de float32 vs 64).
    assert float(((cpu - gpu).abs() * s * 255).max()) <= 1.01


def test_el_recorte_sale_del_frame_entero():
    """Un recorte dentro de un frame grande = ese recorte suelto."""
    proc = _procesador()
    img = _imagen(400, 600)
    caja = (130, 40, 290, 380)
    suelto = np.ascontiguousarray(img[40:380, 130:290])
    a = pose_gpu.preparar(img, [caja], proc.image_mean, proc.image_std,
                          'cpu', False)
    b = pose_gpu.preparar(suelto, [(0, 0, 160, 340)], proc.image_mean,
                          proc.image_std, 'cpu', False)
    assert torch.equal(a, b)


# ── munecas_lote ─────────────────────────────────────────────────────────
class _MotorSinLote:
    """17 keypoints con las muñecas en el centro del recorte."""

    def __init__(self):
        self.llamadas = 0

    def keypoints_17(self, recorte):
        self.llamadas += 1
        h, w = recorte.shape[:2]
        kps = [[0.0, 0.0, 0.0]] * 17
        kps[9] = [w / 2, h / 2, 0.9]
        kps[10] = [w / 4, h / 4, 0.9]
        return kps


class _MotorConLote(_MotorSinLote):
    def __init__(self, devuelve=True):
        super().__init__()
        self.devuelve, self.lotes = devuelve, []

    def keypoints_lote(self, frame, cajas):
        self.lotes.append(list(cajas))
        if not self.devuelve:
            return None
        return [self.keypoints_17(frame[y1:y2, x1:x2])
                for x1, y1, x2, y2 in cajas]


FRAME = np.zeros((500, 800, 3), np.uint8)
CAJAS = [[10, 10, 110, 310], [-20, 50, 90, 400], [300, 100, 305, 120],
         [400, 60, 560, 480]]


def _pose(motor):
    pm = PoseMunecas()
    pm._intentado, pm._modelo = True, motor
    return pm


@pytest.mark.parametrize('motor', [_MotorSinLote(), _MotorConLote(),
                                   _MotorConLote(devuelve=False)])
def test_el_lote_da_lo_mismo_que_caja_a_caja(motor):
    pm = _pose(motor)
    assert pm.munecas_lote(FRAME, CAJAS) == [pm.munecas(FRAME, c)
                                            for c in CAJAS]


def test_el_lote_es_una_sola_llamada_y_salta_las_cajas_chicas():
    motor = _MotorConLote()
    salida = _pose(motor).munecas_lote(FRAME, CAJAS)
    assert len(motor.lotes) == 1
    # La caja de 5x20 px no llega al modelo, y la que se sale del frame
    # va recortada, igual que en `munecas`.
    assert motor.lotes[0] == [(10, 10, 110, 310), (0, 50, 90, 400),
                              (400, 60, 560, 480)]
    assert salida[2] == []
    # En coordenadas del FRAME: la muñeca izquierda del doble va al centro
    # del recorte (400..560, 60..480) -> (480, 270).
    izq = [m for m in salida[3] if m[3] == 'izq'][0]
    assert (izq[0], izq[1]) == (480.0, 270.0)


def test_el_interruptor_apaga_el_lote(monkeypatch):
    from src.analityc.core.pose_motor import _MotorViTPose
    m = _MotorViTPose('cuda')
    m._intentado, m._modelo, m._procesador = True, object(), object()
    m._dispositivo = 'cuda'
    monkeypatch.setenv('ELDE_POSE_PREPARAR_EN_GPU', '0')
    assert m.keypoints_lote(FRAME, [(0, 0, 100, 300)]) is None


def test_sin_cuda_no_hay_lote():
    from src.analityc.core.pose_motor import _MotorViTPose
    m = _MotorViTPose('cpu')
    m._intentado, m._modelo, m._procesador = True, object(), object()
    assert m.keypoints_lote(FRAME, [(0, 0, 100, 300)]) is None


def test_el_medidor_pide_todas_las_poses_de_una_vez():
    src = (RAIZ / 'server' / 'src' / 'analityc' / 'core'
           / 'anaqueles.py').read_text(encoding='utf-8')
    i = src.index("lote = getattr(self.pose, 'munecas_lote', None)")
    assert i < src.index('for (pid, (tid, box, areas_de_pid)), munecas in '
                         'zip(orden, todas):')
