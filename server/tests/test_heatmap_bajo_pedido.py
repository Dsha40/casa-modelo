"""
Mapa de calor vigente pintado BAJO PEDIDO y en JPEG (`heatmap_bajo_pedido.py`,
5-oct-2026).

Lo que fijan estas pruebas:

- que el procesado de imágenes YA NO pinta el mapa (era el objetivo: pintar
  y guardar el PNG costaba ~430 ms cada 5 s dentro del hilo de imágenes);
- que pedirlo devuelve un JPEG válido con el mapa encima del fondo;
- que se reutiliza durante CACHE_S y se vuelve a pintar después;
- que un nombre desconocido no crea acumuladores fantasma en el registro;
- que dashboard, API y lista encuentran la cámara por su JSON, sin PNG.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

from src.analityc.core.analytics import heatmap_bajo_pedido as bp  # noqa: E402
from src.analityc.core.analytics import heatmap_registro as reg  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.analytics.heatmap import HeatmapAccumulator  # noqa: E402

_FRAME = np.full((360, 640, 3), 80, np.uint8)


@pytest.fixture
def carpeta(tmp_path, monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'HEATMAP_DIR', str(tmp_path))
    monkeypatch.setattr(bp, '_cache', {})
    return tmp_path


def _camara_con_mapa(nombre):
    """Una cámara con gente acumulada y su JSON en disco, como la deja el
    procesado normal."""
    for _ in range(5):
        reg.acumular_desde_metadata(nombre, 'Pasillo', _FRAME,
                                    {'detections': [{'box': [100, 100, 160, 300]}]})
    assert reg.obtener(nombre).flush(nombre)


def test_el_procesado_ya_no_pinta_el_mapa(carpeta, monkeypatch):
    acc = HeatmapAccumulator()
    for _ in range(3):
        acc.add_person((100, 100, 160, 300), 640, 360)

    def no_debe_pintar(*a, **k):
        raise AssertionError('el procesado pinto el mapa')
    monkeypatch.setattr(acc, '_render_grid', no_debe_pintar)
    assert acc.maybe_save_snapshot('cam-sin-pintar', background=_FRAME,
                                   every_s=0.0)
    assert (carpeta / 'cam-sin-pintar.json').is_file()
    assert not (carpeta / 'cam-sin-pintar.png').exists()


def test_pedirlo_da_un_jpeg_con_el_mapa(carpeta):
    _camara_con_mapa('cam-bp-1')
    datos = bp.imagen('cam-bp-1')
    assert datos is not None and datos[:2] == b'\xff\xd8'       # JPEG
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    # Hay calor pintado: algo distinto del gris uniforme del fondo.
    assert int(np.abs(img.astype(int) - 80).max()) > 40


def test_se_reutiliza_y_luego_se_vuelve_a_pintar(carpeta, monkeypatch):
    _camara_con_mapa('cam-bp-2')
    primera = bp.imagen('cam-bp-2')
    assert bp.imagen('cam-bp-2') is primera                     # de la caché
    reloj = bp.time.time() + bp.CACHE_S + 1
    monkeypatch.setattr(bp.time, 'time', lambda: reloj)
    assert bp.imagen('cam-bp-2') is not primera                 # pintada de nuevo


def test_un_nombre_desconocido_no_crea_acumuladores(carpeta):
    assert bp.imagen('nadie-la-conoce') is None
    assert 'nadie-la-conoce' not in reg._POR_CAMARA


def test_la_lista_sale_del_json_y_admite_png_viejos(carpeta):
    (carpeta / 'a.json').write_text('{}', encoding='utf-8')
    (carpeta / 'b.png').write_bytes(b'\x89PNGviejo')
    (carpeta / 'c.tmp.png').write_bytes(b'x')
    (carpeta / 'd.json.tmp').write_text('{}', encoding='utf-8')
    (carpeta / 'state').mkdir()
    assert [h['camera'] for h in bp.listado()] == ['a', 'b']
    assert bp.existe('a') and bp.existe('b') and not bp.existe('c')


def test_el_dashboard_sirve_el_jpeg_y_lista_por_json(carpeta, monkeypatch):
    from src.app import dashboard as db
    monkeypatch.setattr(db, '_heatmap_dir', lambda: str(carpeta))
    _camara_con_mapa('cam-bp-3')
    respuesta = db.img_heatmap('cam-bp-3')
    assert respuesta.media_type == 'image/jpeg'
    assert respuesta.body[:2] == b'\xff\xd8'
    lista = db.dashboard_heatmaps()
    assert [h['camera'] for h in lista['heatmaps']] == ['cam-bp-3']


def test_la_api_v1_encuentra_la_camara_sin_png(carpeta, monkeypatch):
    from src.app import api_lectura as api
    (carpeta / 'heatmap').mkdir()
    monkeypatch.setattr(AnalyticsConfig, 'HEATMAP_DIR', str(carpeta / 'heatmap'))
    monkeypatch.setattr(api, '_salida', lambda: carpeta)
    _camara_con_mapa('cam-bp-4')
    assert api._hay_heatmap('cam-bp-4')
    fuera = api.listar_heatmaps()
    assert [h['device_id'] for h in fuera['heatmaps']] == ['cam-bp-4']
    assert fuera['heatmaps'][0]['url'] == '/dashboard/img/heatmap/cam-bp-4.png'
    meta = json.loads((carpeta / 'heatmap' / 'cam-bp-4.json').read_text('utf-8'))
    assert fuera['heatmaps'][0]['modificado'] >= int(meta['actualizado']) - 1
