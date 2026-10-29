"""Unir seguimientos cortados, en sombra (20-oct-2026, `costura_sombra.py`).

Se fija la regla medida con las 47 parejas del usuario: une si el nuevo
aparece en <= 3 s, a <= 1 alto de persona y con el cuerpo parecido (>= 0,80);
no une si falta cualquiera de las tres; no toca los seguimientos; apunta una
línea con su foto; solo mira a quien se ve en la imagen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import costura_sombra as cs  # noqa: E402

FRAME = np.full((600, 800, 3), 120, np.uint8)
E1 = np.array([1.0, 0.0]); E2 = np.array([0.9, 0.4359])   # coseno 0,90
E3 = np.array([0.0, 1.0])                                   # coseno 0


def _correr(tmp_path, b_box, b_emb, hueco_s):
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': 'prov:1', 'last_seen': 0}
    fc, t = 0, 100.0
    for i in range(5):                                      # A visible 2 s
        fc += 1; t += 0.4; a['last_seen'] = fc
        c.observar(FRAME, {1: a}, t, fc)
    t += hueco_s; fc += 1                                   # hueco
    b = {'box': b_box, '_emb_cuerpo': b_emb, 'persistent_id': 'prov:2', 'last_seen': fc}
    tracks = {1: dict(a), 2: b}                             # A sigue en el dict, sin verse
    c.observar(FRAME, tracks, t, fc)
    c.observar(FRAME, tracks, t + 0.1, fc)
    lineas = [json.loads(l) for f in tmp_path.glob('*.jsonl') for l in f.read_text(encoding='utf-8').splitlines()]
    return c, lineas, tracks


def test_une_corte_corto_cercano_y_parecido(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '0')               # solo apunta
    c, lineas, tracks = _correr(tmp_path, [110, 105, 210, 405], E2, 1.0)
    assert len(lineas) == 1 and lineas[0]['a'] == '1' and lineas[0]['b'] == '2'
    assert lineas[0]['foto'] and (tmp_path / lineas[0]['foto']).exists()
    assert tracks[2]['persistent_id'] == 'prov:2'           # en sombra no toca nada


def test_no_une_si_falta_alguna_condicion(tmp_path):
    for nombre, box, emb, hueco in (('lejos', [500, 100, 600, 400], E2, 1.0),
                                     ('distinto', [110, 105, 210, 405], E3, 1.0),
                                     ('tarde', [110, 105, 210, 405], E2, 5.0)):
        d = tmp_path / nombre
        _c, lineas, _t = _correr(d, box, emb, hueco)
        assert lineas == [], nombre


def test_en_pausa_no_hace_nada(monkeypatch):
    """20-oct-2026: el usuario la pausó; con la variable a 0 no apunta nada."""
    monkeypatch.setenv('ELDE_COSTURA_SOMBRA', '0')
    monkeypatch.setattr(cs, '_POR_CAMARA', {})
    cs.observar('cam', FRAME, {1: {'box': [0, 0, 10, 10], 'last_seen': 1}}, 1)
    assert cs._POR_CAMARA == {}
    monkeypatch.setenv('ELDE_COSTURA_SOMBRA', '1')
    cs.observar('cam', FRAME, {1: {'box': [0, 0, 10, 10], 'last_seen': 1}}, 1)
    assert 'cam' in cs._POR_CAMARA


def test_no_une_a_traves_de_un_salto_de_video(tmp_path, monkeypatch):
    """21-oct-2026: al reabrir o mover el vídeo todos desaparecen en la misma
    imagen; el que aparece después en el mismo sitio es OTRA persona."""
    monkeypatch.setattr(cs, '_detectar_saltos', lambda: True)
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a1 = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1}
    a2 = {'box': [400, 100, 500, 400], '_emb_cuerpo': E3}
    a3 = {'box': [600, 100, 700, 400], '_emb_cuerpo': E3}
    fc, t = 0, 100.0
    for _ in range(5):
        fc += 1; t += 0.4
        a1['last_seen'] = a2['last_seen'] = a3['last_seen'] = fc
        c.observar(FRAME, {1: a1, 2: a2, 3: a3}, t, fc)
    t += 1.0; fc += 1                                      # los tres se van a la vez
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E2, 'last_seen': fc}
    c.observar(FRAME, {4: b}, t, fc)
    c.observar(FRAME, {4: b}, t + 0.1, fc)
    assert c.saltos == 1 and c.uniones == 0
    assert not list(tmp_path.glob('*.jsonl'))


def test_dos_que_se_van_a_la_vez_no_son_un_salto(tmp_path):
    """21-oct-2026: con 2 saltaba de más en un vídeo con mucha gente."""
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a1 = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1}
    a2 = {'box': [400, 100, 500, 400], '_emb_cuerpo': E3}
    fc, t = 0, 100.0
    for _ in range(5):
        fc += 1; t += 0.4
        a1['last_seen'] = a2['last_seen'] = fc
        c.observar(FRAME, {1: a1, 2: a2}, t, fc)
    t += 1.0; fc += 1
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E2, 'last_seen': fc}
    c.observar(FRAME, {3: b}, t, fc)
    c.observar(FRAME, {3: b}, t + 0.1, fc)
    assert c.saltos == 0 and c.uniones == 1


def test_con_la_deteccion_apagada_no_marca_saltos(monkeypatch):
    """21-oct-2026: el usuario la apagó; el código sigue, sin efecto."""
    monkeypatch.setattr(cs, '_detectar_saltos', lambda: False)
    c = cs.CosturaSombra('cam', carpeta=Path('.'))
    caja = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1}
    pistas = {i: dict(caja) for i in range(1, 5)}
    for fc in range(1, 4):
        for v in pistas.values():
            v['last_seen'] = fc
        c.observar(FRAME, pistas, 100.0 + fc * 0.4, fc)
    c.observar(FRAME, {}, 103.0, 10)
    assert c.saltos == 0


def test_modo_activo_hereda_la_identidad(tmp_path, monkeypatch):
    """22-oct-2026: tras 58 de 61 uniones bien, el nuevo hereda la identidad
    (y la marca de personal) del anterior, y se avisa al pipeline."""
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '1')
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': 'uuid-A',
         'persistent_confirmed': True, 'es_personal': True, 'personal_rol': 'pasillero'}
    fc, t = 0, 100.0
    for _ in range(5):
        fc += 1; t += 0.4; a['last_seen'] = fc
        c.observar(FRAME, {1: a}, t, fc)
    t += 1.0; fc += 1
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E2, 'persistent_id': 'prov:2', 'last_seen': fc}
    unidos = c.observar(FRAME, {2: b}, t, fc) or c.observar(FRAME, {2: b}, t + 0.1, fc)
    assert unidos == [(1, 2)]
    assert b['persistent_id'] == 'uuid-A' and b['persistent_confirmed'] is True
    # 23-oct-2026: la marca de personal espera a los votos propios.
    assert b['_heredado'] == ('uuid-A', True) and not b.get('es_personal')
    assert b['_personal_heredado']['rol'] == 'pasillero'
    assert json.loads(next(tmp_path.glob('*.jsonl')).read_text(encoding='utf-8'))['unido'] is True


def _unir_empleado_con(c, votos_b, monkeypatch):
    """Une un empleado (1) con un seguimiento nuevo (2) cuyos votos propios
    del uniforme son `votos_b`; devuelve el seguimiento nuevo."""
    from src.analityc.core import clasificador_uniforme as cu
    monkeypatch.setattr(cu, 'votos', lambda cam, tid: list(votos_b) if tid == 2 else [])
    a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': 'prov:1',
         'es_personal': True, 'personal_rol': 'pasillero'}
    fc, t = 0, 100.0
    for _ in range(5):
        fc += 1; t += 0.4; a['last_seen'] = fc
        c.observar(FRAME, {1: a}, t, fc)
    t += 1.0; fc += 1
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E2, 'persistent_id': 'prov:2', 'last_seen': fc}
    c.deshechos_vistos, c.confirmados_vistos = [], []
    for dt in (0.0, 0.1):
        c.observar(FRAME, {2: b}, t + dt, fc)
        c.deshechos_vistos += c.deshechos
        c.confirmados_vistos += c.personal_confirmado
    return b


def test_la_marca_heredada_se_confirma_con_votos_de_uniforme(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '1')
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    b = _unir_empleado_con(c, [0.95, 0.99], monkeypatch)
    assert b['es_personal'] is True and b['personal_via'] == 'costura'
    assert b['persistent_id'] == 'prov:1'
    assert c.confirmados_vistos == [2] and c.deshechos_vistos == []


def test_votos_sin_uniforme_deshacen_la_union(tmp_path, monkeypatch):
    """23-oct-2026, 08:56:13 en vivo: un cliente heredó la marca de un
    empleado con 2 votos propios a 0,0 y su ficha se apartó."""
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '1')
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    b = _unir_empleado_con(c, [0.0, 0.01], monkeypatch)
    assert not b.get('es_personal') and '_heredado' not in b
    assert b['persistent_id'] == 'prov:2' and b['persistent_confirmed'] is False
    assert c.deshechos_vistos == [2] and c.confirmados_vistos == []


def test_con_un_solo_voto_espera(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '1')
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    b = _unir_empleado_con(c, [0.0], monkeypatch)
    assert not b.get('es_personal') and b.get('_personal_heredado') and b['_heredado']


def test_modo_activo_la_cara_manda(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '1')
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': 'prov:1'}
    fc, t = 0, 100.0
    for _ in range(5):
        fc += 1; t += 0.4; a['last_seen'] = fc
        c.observar(FRAME, {1: a}, t, fc)
    t += 1.0; fc += 1
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E2, 'persistent_id': 'uuid-B',
         'persistent_confirmed': True, 'identity_source': 'face', 'last_seen': fc}
    c.observar(FRAME, {2: b}, t, fc); c.observar(FRAME, {2: b}, t + 0.1, fc)
    assert b['persistent_id'] == 'uuid-B' and '_heredado' not in b


def _cara_contra(tmp_path, monkeypatch, pid_a):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '1')
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': pid_a}
    fc, t = 0, 100.0
    for _ in range(5):
        fc += 1; t += 0.4; a['last_seen'] = fc
        c.observar(FRAME, {1: a}, t, fc)
    t += 1.0; fc += 1
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E2, 'persistent_id': 'uuid-B',
         'persistent_confirmed': True, 'identity_source': 'face', 'last_seen': fc}
    c.observar(FRAME, {2: b}, t, fc); c.observar(FRAME, {2: b}, t + 0.1, fc)
    return a, json.loads(next(tmp_path.glob('*.jsonl')).read_text(encoding='utf-8'))


def test_hacia_atras_se_apunta_si_el_de_antes_era_provisional(tmp_path, monkeypatch):
    """27-oct-2026: vídeo del 26-oct, 7 de 8 de estas parejas eran la misma
    persona. A la sombra: se apunta, no se toca a nadie."""
    a, linea = _cara_contra(tmp_path, monkeypatch, 'prov:1')
    assert linea['unido'] is False and linea['hacia_atras'] is True
    assert a['persistent_id'] == 'prov:1'


def test_hacia_atras_no_si_el_de_antes_tenia_su_cara(tmp_path, monkeypatch):
    a, linea = _cara_contra(tmp_path, monkeypatch, 'uuid-A')
    assert linea['unido'] is False and linea['hacia_atras'] is False


# ── Huella inmediata (26-oct-2026) ──────────────────────────────────────
def _huella_por_color(r):
    """Huella de mentira: el color del recorte (blanco o gris)."""
    return np.array([1.0, 0.0]) if r.mean() > 200 else np.array([0.0, 1.0])


def _corte_sin_huella_del_pipeline(tmp_path, color_b):
    """A y B sin `_emb_cuerpo` (trozos de ~1 s: el pipeline no llegó a
    calcularla); A blanco, B del color dado; B reaparece 0,6 s después."""
    frame_a = FRAME.copy(); frame_a[100:400, 100:200] = 255
    frame_b = FRAME.copy(); frame_b[105:405, 110:210] = color_b
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a = {'box': [100, 100, 200, 400], 'persistent_id': 'prov:1'}
    fc, t = 0, 100.0
    for _ in range(3):
        fc += 1; t += 0.2; a['last_seen'] = fc
        c.observar(frame_a, {1: a}, t, fc)
    t += 0.6; fc += 1
    b = {'box': [110, 105, 210, 405], 'persistent_id': 'prov:2', 'last_seen': fc}
    c.observar(frame_b, {2: b}, t, fc)
    c.observar(frame_b, {2: b}, t + 0.1, fc)
    return [json.loads(l) for f in tmp_path.glob('*.jsonl') for l in f.read_text(encoding='utf-8').splitlines()]


def test_huella_inmediata_une_trozos_cortos(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '0')
    monkeypatch.setenv('ELDE_COSTURA_HUELLA_INMEDIATA', '1')
    monkeypatch.setattr(cs, '_huella_fn', _huella_por_color)
    lineas = _corte_sin_huella_del_pipeline(tmp_path / 'igual', 255)
    assert [(l['a'], l['b']) for l in lineas] == [('1', '2')]
    assert _corte_sin_huella_del_pipeline(tmp_path / 'distinto', 60) == []    # otra ropa


def test_sin_huella_inmediata_esos_trozos_no_se_unen(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '0')
    monkeypatch.setattr(cs, '_huella_fn', _huella_por_color)
    monkeypatch.setenv('ELDE_COSTURA_HUELLA_INMEDIATA', '0')
    assert _corte_sin_huella_del_pipeline(tmp_path, 255) == []


def test_forma_valida_salvo_en_el_borde_de_abajo(monkeypatch):
    from src.analityc.core import filtro_fondo as ff
    from src.analityc.core.analytics.config import AnalyticsConfig
    ancha_medio = [100, 300, 300, 400]                      # alto 100, ancho 200
    ancha_borde = [100, 500, 300, 595]                      # toca el borde (600)
    alta = [100, 100, 200, 400]
    assert ff.forma_valida(alta, 0.8, 600)
    assert not ff.forma_valida(ancha_medio, 0.8, 600)
    assert ff.forma_valida(ancha_borde, 0.8, 600)
    assert ff.forma_valida(ancha_medio, 0.0, 600)           # filtro apagado
    monkeypatch.setattr(AnalyticsConfig, 'ASPECTO_EXENTO_BORDE_PX', 0)
    assert not ff.forma_valida(ancha_borde, 0.8, 600)       # como antes


# ── Sin rebote (26-oct-2026) ────────────────────────────────────────────
def _ver(c, tracks, t, fc):
    for v in tracks.values():
        v['last_seen'] = fc
    c.observar(FRAME, tracks, t, fc)


def _rebote(tmp_path):
    """A se pierde y nace B (se unen); luego B se pierde y A vuelve con su
    número: la unión contraria B->A sería un rebote."""
    c = cs.CosturaSombra('cam', carpeta=tmp_path)
    a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': 'prov:1'}
    b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E1, 'persistent_id': 'prov:2'}
    t, fc = 100.0, 0
    for _ in range(3):
        fc += 1; t += 0.2; _ver(c, {1: a}, t, fc)
    t += 0.6
    for _ in range(3):
        fc += 1; t += 0.2; _ver(c, {2: b}, t, fc)
    t += 0.6
    a2 = dict(a)
    for _ in range(3):
        fc += 1; t += 0.2; _ver(c, {1: a2}, t, fc)
    return [(l['a'], l['b']) for f in tmp_path.glob('*.jsonl') for l in
            (json.loads(x) for x in f.read_text(encoding='utf-8').splitlines())]


def test_sin_rebote_no_deshace_la_union_al_reves(tmp_path, monkeypatch):
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '0')
    monkeypatch.setenv('ELDE_COSTURA_SIN_REBOTE', '1')
    assert _rebote(tmp_path / 'con') == [('1', '2')]
    monkeypatch.setenv('ELDE_COSTURA_SIN_REBOTE', '0')
    assert _rebote(tmp_path / 'sin') == [('1', '2'), ('2', '1')]     # como antes


def test_sin_rebote_si_une_numeros_vistos_a_la_vez(tmp_path, monkeypatch):
    """Hueco 0,0 s (el nuevo se vio en la MISMA imagen que el anterior) SÍ se
    une: no unirlo se probó y se quitó (en «empleados» eran recuadros
    duplicados de la misma persona)."""
    monkeypatch.setenv('ELDE_COSTURA_UNIR', '0')
    for valor, esperado in (('1', [('1', '2')]), ('0', [('1', '2')])):
        monkeypatch.setenv('ELDE_COSTURA_SIN_REBOTE', valor)
        d = tmp_path / valor
        c = cs.CosturaSombra('cam', carpeta=d)
        a = {'box': [100, 100, 200, 400], '_emb_cuerpo': E1, 'persistent_id': 'prov:1'}
        b = {'box': [110, 105, 210, 405], '_emb_cuerpo': E1, 'persistent_id': 'prov:2'}
        t, fc = 100.0, 0
        for _ in range(3):
            fc += 1; t += 0.2; _ver(c, {1: a}, t, fc)
        fc += 1; t += 0.2; _ver(c, {1: a, 2: b}, t, fc)              # los dos a la vez
        for _ in range(4):
            fc += 1; t += 0.2; _ver(c, {2: b}, t, fc)
        lineas = [(l['a'], l['b']) for f in d.glob('*.jsonl') for l in
                  (json.loads(x) for x in f.read_text(encoding='utf-8').splitlines())]
        assert lineas == esperado, valor
