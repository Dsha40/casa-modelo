"""El panel de Anaqueles enseña el «posible agotado» (5-oct-2026).

El servidor (`anaqueles_stock.py`) manda `bloque['stock']` con las alertas
activas y, en el feed, sucesos `agotado` / `repuesto`. Lo que se protege:

  - la alerta sale en el letrero rojo, con producto, área y hora;
  - el letrero se apaga cuando el servidor deja de mandarla (repuesto);
  - los sucesos del feed no caen en la rama genérica («Persona visitó…»);
  - mientras la foto de referencia espera a que nadie tape, se dice.

No se fuerza `QT_QPA_PLATFORM=offscreen` (CLAUDE.md 12.5): la QApplication es
una por proceso y forzarlo rompe las pruebas que necesitan ventanas reales.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.components.anaqueles_panel import (  # noqa: E402
    AnaquelesPanel, lineas_de_stock)

ALERTA = {'area': 1, 'producto_idx': 0, 'area_nombre': 'Anaquel 2',
          'producto': 'Colcafé', 'desde': '11:21:30', 'cambio': 0.71}


def _bloque(stock, eventos=()):
    return {'camera_id': 'cam-1', 'camera_name': 'Cámara 3',
            'stock': stock, 'eventos': list(eventos)}


def test_la_alerta_sale_en_el_letrero_rojo():
    panel = AnaquelesPanel()
    panel.actualizar(_bloque({'referencias': 2, 'alertas': [ALERTA]}))
    assert not panel.lbl_stock.isHidden()
    texto = panel.lbl_stock.text()
    assert 'Posible agotado: Colcafé' in texto
    assert 'Anaquel 2' in texto and '11:21:30' in texto


def test_al_reponer_el_letrero_se_apaga():
    panel = AnaquelesPanel()
    panel.actualizar(_bloque({'alertas': [ALERTA]}))
    panel.actualizar(_bloque({'alertas': []}))
    assert panel.lbl_stock.isHidden()


def test_los_sucesos_de_stock_no_parecen_visitas():
    panel = AnaquelesPanel()
    panel.actualizar(_bloque({'alertas': [ALERTA]}, eventos=[
        {'tipo': 'agotado', 'area': 1, 'nombre': 'Anaquel 2',
         'producto': 'Colcafé', 'ts': '11:21:30',
         'id_corto': 'stock-1-0'}]))
    textos = [panel._zona_feed.itemAt(i).widget().text()
              for i in range(panel._zona_feed.count())]
    assert any('Posible agotado' in t and 'Colcafé' in t for t in textos)
    assert not any('visitó' in t or 'agarró' in t for t in textos)


def test_se_dice_que_la_foto_espera():
    lineas = lineas_de_stock({'cam-1': {'nombre': 'Cámara 3', 'stock': {
        'alertas': [], 'esperando_foto': 2}}})
    assert lineas == ['📷 Foto de referencia: esperando a que nadie tape '
                      '2 producto(s)']


def test_con_varias_camaras_se_dice_cual():
    vivas = {'a': {'nombre': 'Cámara 3', 'stock': {'alertas': [ALERTA]}},
             'b': {'nombre': 'Cámara 4', 'stock': {'alertas': []}}}
    assert lineas_de_stock(vivas)[0].startswith('🛒 Cámara 3 · ')


def test_el_agotado_va_a_su_columna_y_no_a_visitantes():
    from gui.components.sidebar.alerts_sidebar import clasificar
    assert clasificar({'event_type': 'Posible agotado',
                       'class_name': 'coffemate'}) == 'stock'
    # Lo de antes, igual.
    assert clasificar({'event_type': 'Visitante',
                       'class_name': 'Femenino 25-32'}) == 'visitantes'
    assert clasificar({'event_type': 'Empleado surte anaquel'}) == \
        'visitantes'


# ── Cuánto lleva (5-oct, lo pidió el usuario) ───────────────────────────
def test_el_letrero_dice_cuanto_lleva_y_sigue_contando():
    """`vacio_s` es lo que contaba el servidor al emitir; el panel suma lo
    que pasa desde que lo recibió (`stock_ts`), con su propio reloj."""
    vivas = {'c': {'nombre': 'Cámara 3', 'stock_ts': 1000.0,
                   'stock': {'alertas': [dict(ALERTA, vacio_s=125)]}}}
    assert 'lleva 2 min 05 s' in lineas_de_stock(vivas, ahora=1000.0)[0]
    assert 'lleva 2 min 15 s' in lineas_de_stock(vivas, ahora=1010.0)[0]


def test_el_panel_avanza_el_contador_cada_segundo(monkeypatch):
    import gui.components.anaqueles_panel as P
    reloj = [5000.0]
    monkeypatch.setattr(P.time, 'monotonic', lambda: reloj[0])
    panel = AnaquelesPanel()
    assert panel._timer_stock.interval() == 1000
    assert panel._timer_stock.isActive()
    panel.actualizar(_bloque({'alertas': [dict(ALERTA, vacio_s=59)]}))
    assert 'lleva 59 s' in panel.lbl_stock.text()
    reloj[0] += 3
    panel._refrescar_stock()             # lo que hace el temporizador
    assert 'lleva 1 min 02 s' in panel.lbl_stock.text()


def test_el_repuesto_dice_cuanto_estuvo():
    panel = AnaquelesPanel()
    panel.actualizar(_bloque({'alertas': []}, eventos=[
        {'tipo': 'repuesto', 'area': 1, 'nombre': 'Anaquel 2',
         'producto': 'Colcafé', 'ts': '11:26:42', 'id_corto': 'stock-1-0',
         'duracion_s': 312}]))
    textos = [panel._zona_feed.itemAt(i).widget().text()
              for i in range(panel._zona_feed.count())]
    assert any('Repuesto' in t and 'estuvo 5 min 12 s' in t for t in textos)


def test_la_duracion_se_lee_bien():
    from gui.components.anaqueles_panel import duracion
    assert duracion(45) == '45 s'
    assert duracion(125) == '2 min 05 s'
    assert duracion(3780) == '1 h 03 min'
