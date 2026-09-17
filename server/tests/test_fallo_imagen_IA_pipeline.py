"""Un fallo atrapado del pipeline viaja al cliente como `error` (17-sep-2026).

`TiendasProcess.process_frame` atrapa sus excepciones y devuelve
`{'error': mensaje}`; `_full_frame_sync` lo respondia como `success`. Sin
`detections` el cliente en modo directo no repintaba y el contador de FPS de
IA seguia subiendo: imagen congelada y consola del cliente limpia. Asi estuvo
el detector caido por «No module named 'vigilante_amazonas'».

Se prueba:
  - que la forma exacta del fallo produce `status: "error"` con su mensaje;
  - que una respuesta buena, o una que solo ANOTA un `error` junto a sus
    resultados, NO se convierte en fallo;
  - que la respuesta de error cumple el contrato `FrameResult`;
  - el cableado: `_full_frame_sync` la consulta antes de responder.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app import fallo_imagen_IA_pipeline as fp  # noqa: E402

_MENSAJE = "detector compartido no disponible: No module named 'vigilante_amazonas'"


def test_el_fallo_atrapado_viaja_como_error():
    r = fp.respuesta_si_fallo({'error': _MENSAJE}, 'win-iVMS-4200', 0.004)
    assert r == {
        'camera_id': 'win-iVMS-4200',
        'status': 'error',
        'message': _MENSAJE,
        'metadata': {'error': _MENSAJE},
        'processed_image': None,
        'processing_time': 0.004,
    }


def test_un_fallo_sin_texto_sigue_siendo_error_con_mensaje():
    r = fp.respuesta_si_fallo({'error': ''}, 'cam', 0.0)
    assert r is not None and r['status'] == 'error' and r['message']


def test_una_respuesta_buena_no_es_fallo():
    assert fp.respuesta_si_fallo(
        {'detections': [], 'people_counter': 0}, 'cam', 0.1) is None
    assert fp.respuesta_si_fallo({}, 'cam', 0.1) is None
    assert fp.respuesta_si_fallo(None, 'cam', 0.1) is None


def test_un_error_anotado_junto_a_resultados_no_es_fallo():
    """La misma imagen sirve otras verticales: un `error` informativo al lado
    de resultados validos no puede tumbar el frame (congelaria al cliente)."""
    meta = {'error': 'VLM no disponible', 'detections': [{'box': [0, 0, 1, 1]}]}
    assert fp.respuesta_si_fallo(meta, 'cam', 0.1) is None


def test_la_respuesta_de_error_cumple_el_contrato():
    sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))
    from elde_core.contracts.eventos import FrameResult
    FrameResult.model_validate(
        fp.respuesta_si_fallo({'error': _MENSAJE}, 'cam', 0.01))


def test_full_frame_sync_consulta_el_fallo_antes_de_responder():
    """Si la consulta queda despues de las ramas de respuesta, vuelve el
    `success` silencioso."""
    import inspect
    from src.app.app import _full_frame_sync
    fuente = inspect.getsource(_full_frame_sync)
    consulta = fuente.index('_fallo.respuesta_si_fallo(')
    assert consulta < fuente.index('if not draw_server:')
    assert consulta < fuente.index('_encode_jpeg(')


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
if __name__ == '__main__':
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        try:
            _fn()
            print(f"  OK    {_nombre}")
        except Exception as _exc:                          # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
