"""El motor open-vocab del router multimodal: OWLv2 (Apache-2.0), sin rollback.

Desde el 3-sep-2026 el rollback a YOLO-World se retiro junto con ultralytics
(AGPL), asi que ELDE_WORLD_MOTOR admite solo 'owlv2' (default) | 'off'.
Nada aqui carga modelos: se prueba el DESPACHO por ELDE_WORLD_MOTOR (owlv2
default, off apagado, cualquier otro valor -incluido el viejo 'yoloworld'-
avisa y cae en owlv2 sin tocar ultralytics) y el contrato de salida de
MotorMundoOWLv2.detect con dobles del processor/modelo — el mismo formato
[{box, label, confidence}] que consumen busqueda_vlm._correr y los endpoints
/vlm/detect y /vlm/route. La carga real (CPU) solo corre si los pesos
locales ya estan descargados; si no, se salta.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import multimodal_router as mr  # noqa: E402

PESOS = Path(mr.MotorMundoOWLv2.RUTA_PESOS)


@pytest.fixture
def sin_ultralytics(monkeypatch):
    """Un None en sys.modules hace que `import ultralytics` reviente con
    ImportError: asi el test demuestra que NINGUN camino del router lo
    importa ya (la imagen deja de llevar el paquete AGPL el 3-sep-2026)."""
    monkeypatch.setitem(sys.modules, 'ultralytics', None)


# ── despacho por ELDE_WORLD_MOTOR ────────────────────────────────────────

def test_owlv2_es_el_motor_por_defecto(monkeypatch, sin_ultralytics, caplog):
    """Sin env el motor es OWLv2 (la envoltura, sin cargar pesos), se
    memoriza (dos llamadas devuelven el MISMO objeto) y no avisa de nada."""
    monkeypatch.delenv('ELDE_WORLD_MOTOR', raising=False)
    r = mr.MultimodalRouter(device='cpu')
    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        m = r._ensure_world()
        assert r._ensure_world() is m
    assert isinstance(m, mr.MotorMundoOWLv2)
    assert m._modelo is None                 # perezoso: aun sin pesos
    assert not [x for x in caplog.records if 'ELDE_WORLD_MOTOR' in x.message]


def test_owlv2_explicito_tampoco_avisa(monkeypatch, sin_ultralytics, caplog):
    """'owlv2' (con mayusculas/espacios de un .env escrito a mano) es el
    valor canonico: mismo motor y sin aviso de 'no reconocido'."""
    monkeypatch.setenv('ELDE_WORLD_MOTOR', '  OWLv2 ')
    r = mr.MultimodalRouter(device='cpu')
    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        assert isinstance(r._ensure_world(), mr.MotorMundoOWLv2)
    assert not [x for x in caplog.records if 'ELDE_WORLD_MOTOR' in x.message]


def test_off_apaga_la_deteccion_sin_reventar(monkeypatch, sin_ultralytics):
    """'off': detect devuelve [] (busqueda_vlm degrada con cero resultados)
    y no se instancia ningun motor."""
    monkeypatch.setenv('ELDE_WORLD_MOTOR', 'off')
    r = mr.MultimodalRouter(device='cpu')
    assert r._ensure_world() is None
    img = np.zeros((32, 32, 3), np.uint8)
    assert r.detect(img, 'person') == []
    assert r.detect(img, ['red car', 'dog']) == []
    assert r._world is None


@pytest.mark.parametrize('valor', ['yoloworld', 'YoloWorld', 'yolo-world'])
def test_yoloworld_retirado_avisa_y_usa_owlv2(monkeypatch, sin_ultralytics,
                                              caplog, valor):
    """Un .env viejo con ELDE_WORLD_MOTOR=yoloworld (el rollback AGPL,
    retirado el 3-sep-2026) NO deja la busqueda sin motor ni importa
    ultralytics: avisa UNA vez (diciendo que se retiro) y usa OWLv2."""
    monkeypatch.setenv('ELDE_WORLD_MOTOR', valor)
    r = mr.MultimodalRouter(device='cpu')
    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        m = r._ensure_world()
        assert r._ensure_world() is m        # memorizado, y sin 2o aviso
    assert isinstance(m, mr.MotorMundoOWLv2)
    avisos = [x for x in caplog.records if 'ELDE_WORLD_MOTOR' in x.message]
    assert len(avisos) == 1
    assert 'yoloworld' in avisos[0].message.lower()
    assert 'retiro' in avisos[0].message
    assert 'owlv2' in avisos[0].message
    assert 'off' in avisos[0].message        # dice cuales SI valen


def test_valor_desconocido_avisa_y_usa_owlv2(monkeypatch, sin_ultralytics,
                                             caplog):
    """Cualquier otro typo se trata igual: aviso una vez y owlv2."""
    monkeypatch.setenv('ELDE_WORLD_MOTOR', 'grounding-dino')
    r = mr.MultimodalRouter(device='cpu')
    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        assert isinstance(r._ensure_world(), mr.MotorMundoOWLv2)
        r._ensure_world()
    avisos = [x for x in caplog.records if 'ELDE_WORLD_MOTOR' in x.message]
    assert len(avisos) == 1 and "'grounding-dino'" in avisos[0].message


def test_el_router_ya_no_acepta_pesos_yolo():
    """La firma perdio world_weights (nadie lo pasaba; los pesos .pt de
    YOLO-World ya no se montan): pasarlo debe fallar, no ignorarse en
    silencio con una ruta que ya no existe."""
    with pytest.raises(TypeError):
        mr.MultimodalRouter(device='cpu', world_weights='pesos-yolo-world.pt')
    assert mr.MultimodalRouter.MOTORES_WORLD == ('owlv2', 'off')


# ── contrato de salida de MotorMundoOWLv2.detect (con dobles) ────────────

class _ProcFalso:
    """Doble del Owlv2Processor: registra lo que recibe y devuelve dos
    cajas, una normal y una asomada al relleno cuadrado del preproceso."""

    def __init__(self, capturado):
        self._cap = capturado

    def __call__(self, text=None, images=None, return_tensors=None):
        import torch
        self._cap['text'] = text
        return {'input_ids': torch.zeros((1, 2), dtype=torch.long),
                'pixel_values': torch.zeros((1, 3, 8, 8))}

    def post_process_grounded_object_detection(self, outputs, threshold=0.1,
                                               target_sizes=None):
        import torch
        self._cap['threshold'] = threshold
        self._cap['target_sizes'] = target_sizes
        return [{'scores': torch.tensor([0.31, 0.12]),
                 'labels': torch.tensor([1, 0]),
                 'boxes': torch.tensor([[10., 20., 110., 220.],
                                        [-5., -3., 700., 500.]])}]


class _ModeloFalso:
    """Doble del Owlv2ForObjectDetection: solo device/dtype y el forward."""
    def __init__(self):
        import torch
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def __call__(self, **kwargs):
        return object()


def test_detect_owlv2_respeta_el_contrato():
    """[{box:[x1,y1,x2,y2] float px, label: terms[idx] tal cual,
    confidence: float}] — lo que consumen busqueda_vlm y /vlm/detect
    (el contrato heredado del motor anterior, que no cambio al retirarlo)."""
    capturado = {}
    motor = mr.MotorMundoOWLv2(device='cpu')
    motor._proc = _ProcFalso(capturado)   # _modelo publicado: _load no corre
    motor._modelo = _ModeloFalso()
    img = np.zeros((480, 640, 3), np.uint8)

    out = motor.detect(img, ['red car', 'person'], conf=0.07)

    # Lo que viajo al modelo: los terminos como UNA lista de consultas, el
    # conf tal cual (sin recalibrar aqui) y el tamano REAL del frame.
    assert capturado['text'] == [['red car', 'person']]
    assert capturado['threshold'] == pytest.approx(0.07)
    assert capturado['target_sizes'] == [(480, 640)]

    assert [sorted(d) for d in out] == [['box', 'confidence', 'label']] * 2
    assert out[0]['box'] == [10.0, 20.0, 110.0, 220.0]
    assert out[0]['label'] == 'person'            # labels[0]=1 -> terms[1]
    assert out[0]['confidence'] == pytest.approx(0.31)
    # La caja asomada al relleno cuadrado queda recortada al marco real.
    assert out[1]['box'] == [0.0, 0.0, 640.0, 480.0]
    assert out[1]['label'] == 'red car'
    assert all(isinstance(v, float) for d in out
               for v in d['box'] + [d['confidence']])


def test_router_detect_delega_en_owlv2_con_terminos_limpios(monkeypatch,
                                                            sin_ultralytics):
    """MultimodalRouter.detect: str -> lista, recorta espacios, descarta
    vacios y delega en MotorMundoOWLv2.detect con el conf que llega."""
    monkeypatch.delenv('ELDE_WORLD_MOTOR', raising=False)
    r = mr.MultimodalRouter(device='cpu')
    llamadas = []

    def _detect(image_bgr, terms, conf=0.05):
        llamadas.append((terms, conf))
        return [{'box': [0., 0., 1., 1.], 'label': terms[0],
                 'confidence': 0.5}]
    monkeypatch.setattr(mr.MotorMundoOWLv2, 'detect', staticmethod(_detect))
    img = np.zeros((32, 32, 3), np.uint8)
    assert r.detect(img, ' person ', conf=0.2)[0]['label'] == 'person'
    assert r.detect(img, ['', '  ']) == []       # sin terminos: ni llama
    assert llamadas == [(['person'], 0.2)]


def test_pesos_ausentes_dan_error_claro(monkeypatch):
    """Sin pesos locales (o descarga a medias) el error debe decir DONDE
    ponerlos y COMO apagar la busqueda (ya no hay motor anterior al que
    volver) — no un stacktrace de HF."""
    monkeypatch.setattr(mr.MotorMundoOWLv2, 'RUTA_PESOS',
                        '/no/existe/owlv2-base-patch16-ensemble')
    motor = mr.MotorMundoOWLv2(device='cpu')
    with pytest.raises(RuntimeError) as exc:
        motor.detect(np.zeros((32, 32, 3), np.uint8), ['person'])
    assert '/no/existe/owlv2-base-patch16-ensemble' in str(exc.value)
    assert 'ELDE_WORLD_MOTOR=off' in str(exc.value)
    assert 'yoloworld' not in str(exc.value)


# ── carga real (solo si los pesos ya estan descargados; CPU) ─────────────

_PESOS_COMPLETOS = ((PESOS / 'config.json').is_file()
                    and any((PESOS / f).is_file()
                            for f in ('model.safetensors',
                                      'pytorch_model.bin')))


@pytest.mark.skipif(not _PESOS_COMPLETOS,
                    reason='pesos OWLv2 locales ausentes (descarga en curso)')
def test_carga_real_owlv2_en_cpu_cumple_el_contrato():
    motor = mr.MotorMundoOWLv2(device='cpu')
    out = motor.detect(np.zeros((96, 128, 3), np.uint8), ['person'],
                       conf=0.05)
    assert isinstance(out, list)
    for d in out:
        assert sorted(d) == ['box', 'confidence', 'label']
        assert d['label'] == 'person'
        x1, y1, x2, y2 = d['box']
        assert 0.0 <= x1 <= x2 <= 128.0 and 0.0 <= y1 <= y2 <= 96.0
