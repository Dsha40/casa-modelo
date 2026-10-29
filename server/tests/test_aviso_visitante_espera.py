"""El aviso de «visitante» espera unos segundos (20-oct-2026).

El género llegaba a veces antes que la marca de personal del uniforme, y el
cliente recibía avisos de visitante de empleados y vigilantes (77 fichas de
personal del 19-20 oct). Se fija: el aviso no sale antes de la espera; sale
si la ficha sigue en la galería; no sale si la ficha se apartó (personal).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.tiendas_fotos import FotosMixin  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402


class P(FotosMixin):
    def _ficha_personal_tienda(self, pid):
        return None

    def _captura_en_b64(self, stem):
        return ''


def test_espera_y_no_avisa_si_la_ficha_se_aparto(tmp_path, monkeypatch):
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_DIR', str(tmp_path), raising=False)
    monkeypatch.setattr(AnalyticsConfig, 'VISITANTE_ESPERA_S', 8.0, raising=False)
    from src.analityc.core import tienda_almacen as ta
    ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    (tmp_path / 'persons').mkdir()
    (tmp_path / 'persons' / 'cliente.json').write_text('{}')   # 'empleado' ya no está
    p = P()
    meta = {'gender': 'Hombre', 'age_range': '26-35', 'timestamp': '20261020_100000'}
    p._emitir_visitante('cliente', dict(meta))
    p._emitir_visitante('empleado', dict(meta))
    t_listo = p._visitantes_en_espera[0][0]
    p._soltar_visitantes(t_listo - 1)
    assert not getattr(p, '_alertas_pendientes', [])              # aún espera
    p._soltar_visitantes(t_listo + 1)
    assert len(p._alertas_pendientes) == 1                        # solo el cliente
    assert p._visitantes_en_espera == []
