"""
Pruebas del pipeline `AmazonasStand` (modo propio del stand).

No instancia la clase completa: eso cargaría YOLO y necesitaría GPU. Lo que
se verifica es el CONTRATO del que depende todo lo demás:

  - que herede de `TiendasProcess`, porque `app.py` despacha media docena de
    cosas con `isinstance(processor, TiendasProcess)`;
  - que el stand ya NO esté enganchado dentro de `TiendasProcess`, que es el
    fallo que hizo a Tienda abrir sesiones de un stand que no mide;
  - que un fallo del stand no rompa el frame.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.amazonas_stand import (  # noqa: E402
    MODO, AmazonasStand)
from src.analityc.core.tiendas import (  # noqa: E402
    TiendasProcess)


def test_es_subclase_de_tiendas_process():
    """`app.py` despacha por isinstance: sin esto se rompe medio servidor."""
    assert issubclass(AmazonasStand, TiendasProcess)


def test_sobrescribe_process_frame():
    assert AmazonasStand.process_frame is not TiendasProcess.process_frame


def test_el_stand_ya_no_vive_dentro_de_tiendas_process():
    """El enganche global fue un error: Tienda usa ese mismo pipeline y al
    encender el flag empezó a contar visitantes de un stand ajeno."""
    src = inspect.getsource(TiendasProcess.process_frame)
    assert 'stand_puente' not in src
    assert 'crear_si_habilitado' not in src


def test_el_modo_esta_registrado_en_el_servidor():
    from src.app.app import _build_processor
    src = inspect.getsource(_build_processor)
    assert f'"{MODO}"' in src or f"'{MODO}'" in src


def test_el_cliente_amazonas_ofrece_el_modo():
    ruta = (RAIZ / 'clients' / 'amazonas' / 'src' / 'gui' / 'components' /
            'custom_status_bar.py')
    assert MODO in ruta.read_text(encoding='utf-8'), (
        "el selector del cliente no ofrece el modo del stand")


def test_un_fallo_del_stand_no_rompe_el_frame():
    """Si la medición revienta, el cliente debe seguir viendo vídeo."""
    class _Falso(AmazonasStand):
        def __init__(self):                    # sin cargar modelos
            self._puente = None
            self._puente_intentado = True
            self.active_tracks = {}

        def _medir_stand(self, image, camera_id):
            raise RuntimeError("boom")

    import numpy as np
    falso = _Falso()
    # Se le inyecta el super().process_frame para no cargar YOLO.
    _Falso.__mro__  # noqa: B018
    esperado = (np.zeros((4, 4, 3), np.uint8), {'detections': []})
    original = TiendasProcess.process_frame
    try:
        TiendasProcess.process_frame = lambda self, image, **kw: esperado
        img, meta = AmazonasStand.process_frame(
            falso, np.zeros((4, 4, 3), np.uint8))
    finally:
        TiendasProcess.process_frame = original

    assert meta == {'detections': []}, "el metadata original se perdió"
    assert 'stand' not in meta


def test_sin_puente_no_añade_bloque():
    class _Falso(AmazonasStand):
        def __init__(self):
            self._puente = None
            self._puente_intentado = True
            self.active_tracks = {}

    import numpy as np
    assert _Falso()._medir_stand(np.zeros((4, 4, 3), np.uint8), 1) is None


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
