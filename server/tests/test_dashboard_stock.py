"""Apartado «Stock» del dashboard (9-oct-2026, dashboard_stock.py).

- el historial sale de las fotos de evidencia, más reciente primero, con los
  nombres de área y producto del registro del día;
- una alerta viva del pipeline marca su foto como ACTIVA y trae sus segundos;
- la ruta de imagen no deja salir de la carpeta de la cámara;
- la página del dashboard lleva el apartado.
"""
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.app import dashboard_stock as ds


def _camara(tmp_path: Path) -> Path:
    cam = tmp_path / 'win-iVMS-4200-c3-i58bf7b'
    (cam / 'stock_alertas').mkdir(parents=True)
    (cam / '2026-10-09.json').write_text(json.dumps({'areas': {
        '0': {'nombre': 'Anaquel 1', 'productos': {'1': {'nombre': 'coffemate'}}},
        '1': {'nombre': 'Anaquel 2', 'productos': {'1': {'nombre': 'colcafe'}}}}}),
        encoding='utf-8')
    for nombre in ('2026-10-09_101328_1_1.jpg', '2026-10-09_163803_0_1.jpg', 'otra_cosa.jpg'):
        (cam / 'stock_alertas' / nombre).write_bytes(b'\xff\xd8')
    return cam


def test_historial_con_nombres_y_mas_reciente_primero(tmp_path, monkeypatch):
    _camara(tmp_path)
    monkeypatch.setattr(ds, '_activas', lambda: {})
    r = ds.resumen_stock(tmp_path)
    assert [h['hora'] for h in r['historial']] == ['16:38:03', '10:13:28']
    assert r['historial'][1]['producto'] == 'colcafe'
    assert r['historial'][1]['area'] == 'Anaquel 2'
    assert r['activas'] == [] and r['total'] == 2       # 'otra_cosa.jpg' no cuenta


def test_la_alerta_viva_marca_su_foto_como_activa(tmp_path, monkeypatch):
    _camara(tmp_path)
    monkeypatch.setattr(ds, '_activas', lambda: {'win-iVMS-4200-c3-i58bf7b': [
        {'area': 0, 'producto_idx': 1, 'desde': '16:38:03', 'vacio_s': 125,
         'producto': 'coffemate', 'area_nombre': 'Anaquel 1', 'cambio': 0.71,
         'tipo': 'agotado'}]})
    r = ds.resumen_stock(tmp_path)
    assert len(r['activas']) == 1
    a = r['activas'][0]
    assert a['producto'] == 'coffemate' and a['lleva_s'] == 125 and a['cambio'] == 0.71


@pytest.mark.parametrize('camara,archivo', [
    ('..', '2026-10-09_101328_1_1.jpg'),
    ('win-iVMS-4200-c3-i58bf7b', '..\\..\\tienda.db'),
    ('win-iVMS-4200-c3-i58bf7b', 'cualquiera.jpg'),
])
def test_la_imagen_no_sale_de_su_carpeta(camara, archivo):
    with pytest.raises(HTTPException):
        ds.img_stock(camara, archivo)


def test_la_pagina_lleva_el_apartado():
    from src.app.dashboard import _HTML
    assert 'Stock — posible agotado' in _HTML and '/dashboard/api/stock' in _HTML


def test_las_bajadas_salen_con_su_cambio_y_su_cuenta(tmp_path, monkeypatch):
    """12-oct-2026: `..._bajada.jpg` y su `.json` (cambio y cuenta)."""
    cam = _camara(tmp_path)
    (cam / 'stock_alertas' / '2026-10-12_090000_1_1_bajada.jpg').write_bytes(b'\xff\xd8')
    (cam / 'stock_alertas' / '2026-10-12_090000_1_1_bajada.json').write_text(
        json.dumps({'tipo': 'bajada', 'cambio': 0.52, 'bajadas': 3}), encoding='utf-8')
    monkeypatch.setattr(ds, '_activas', lambda: {})
    h = ds.resumen_stock(tmp_path)['historial'][0]
    assert h['tipo'] == 'bajada' and h['cambio'] == 0.52 and h['bajadas'] == 3
    assert h['producto'] == 'colcafe' and h['activa'] is False


def test_el_dashboard_de_tienda_lleva_el_apartado():
    """El botón «Dashboard» del cliente abre /dashboards/tienda/, no
    /dashboard: ahí tiene que estar el apartado (12-oct-2026)."""
    raiz = Path(__file__).resolve().parents[2] / 'dashboards' / 'tienda'
    assert 'id="stock-sec"' in (raiz / 'index.html').read_text(encoding='utf-8')
    assert '/dashboard/api/stock' in (raiz / 'app.js').read_text(encoding='utf-8')
