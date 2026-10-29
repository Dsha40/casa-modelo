"""El panel de Anaqueles no se reconstruye con cada respuesta (28-oct-2026).

Antes ranking y feed se rehacían enteros con cada respuesta del servidor
(~13 por segundo), casi siempre con lo mismo. Lo que se protege:

  - el mismo bloque no reconstruye nada;
  - un cambio que llega antes de `RECONSTRUIR_CADA_S` se pinta al cumplirse
    el plazo (no se pierde) y las cifras de arriba van al momento;
  - un suceso nuevo rehace el feed y no el ranking.

No se fuerza `QT_QPA_PLATFORM=offscreen` (CLAUDE.md 12.5).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

import gui.components.anaqueles_panel as P  # noqa: E402


def _bloque(visitas=5, eventos=()):
    areas = [{'idx': 0, 'nombre': 'Anaquel 1', 'visitas': visitas,
              'agarres': 1, 'personas_ahora': 0}]
    return {'camera_id': 'cam-1', 'camera_name': 'Cámara 3', 'areas': areas,
            'salud': {}, 'eventos': list(eventos)}


def _panel_contado(monkeypatch, reloj):
    monkeypatch.setattr(P.time, 'monotonic', lambda: reloj[0])
    panel = P.AnaquelesPanel()
    cuenta = {'ranking': 0, 'feed': 0}
    ranking, feed = panel._pintar_ranking, panel._pintar_feed

    def contar_ranking(*a, **k):
        cuenta['ranking'] += 1
        return ranking(*a, **k)

    def contar_feed(*a, **k):
        cuenta['feed'] += 1
        return feed(*a, **k)

    panel._pintar_ranking = contar_ranking
    panel._pintar_feed = contar_feed
    return panel, cuenta


def _textos(layout):
    from PySide6.QtWidgets import QLabel
    textos = []
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        textos += [w.text()] if isinstance(w, QLabel) else [
            lbl.text() for lbl in w.findChildren(QLabel)]
    return ' | '.join(textos)


def test_el_mismo_bloque_no_reconstruye_nada(monkeypatch):
    reloj = [1000.0]
    panel, cuenta = _panel_contado(monkeypatch, reloj)
    panel.actualizar(_bloque())
    assert cuenta == {'ranking': 1, 'feed': 1}
    for _ in range(5):
        reloj[0] += 1.0                    # pasado el plazo, y aun así
        panel.actualizar(_bloque())
    assert cuenta == {'ranking': 1, 'feed': 1}


def test_un_cambio_seguido_se_pinta_al_cumplirse_el_plazo(monkeypatch):
    reloj = [1000.0]
    panel, cuenta = _panel_contado(monkeypatch, reloj)
    panel.actualizar(_bloque(visitas=5))
    reloj[0] += 0.1
    panel.actualizar(_bloque(visitas=6))
    # Las cifras de arriba, al momento; el ranking, todavía no.
    assert panel.card_visitas.valor.text() == '6'
    assert cuenta['ranking'] == 1
    assert panel._timer_listas.isActive()
    assert '5 visitas' in _textos(panel._zona_ranking)
    reloj[0] += P.RECONSTRUIR_CADA_S
    panel._repintar()                      # lo que hace el temporizador
    assert cuenta['ranking'] == 2
    assert '6 visitas' in _textos(panel._zona_ranking)


def test_un_suceso_nuevo_rehace_el_feed_y_no_el_ranking(monkeypatch):
    reloj = [1000.0]
    panel, cuenta = _panel_contado(monkeypatch, reloj)
    panel.actualizar(_bloque())
    reloj[0] += 1.0
    panel.actualizar(_bloque(eventos=[
        {'tipo': 'agarre', 'area': 0, 'nombre': 'Anaquel 1',
         'ts': '15:31:03', 'id_corto': '0021', 'genero': 'Mujer'}]))
    assert cuenta == {'ranking': 1, 'feed': 2}
    assert 'agarró en' in _textos(panel._zona_feed)
