"""Informe del pasillo, el PDF del botón «Exportar PDF» (13-oct-2026,
`informe_pasillo.py` y `informe_pdf.py`).

Fija el contrato, no el dibujo (eso se mira en el PDF):

  - el botón pide el del pasillo con el periodo y la cámara; si no se puede
    imprimir (sin Edge/Chrome, p. ej. el servidor de Linux) o falla, sale
    el clásico del último día, como antes;
  - las cifras salen de la base (capturas, identificadas, clasificadas, con
    cara o solo cuerpo, entradas, permanencia) y de los anaqueles (cada
    agarre con su producto y el género de la ficha de quien agarró);
  - la página sale con datos y sin ellos;
  - la foto del stock se recorta a «antes | ahora», sin la del cliente;
  - el mapa general no usa una hora pintada sobre la imagen gris de un corte;
  - con navegador, el PDF sale de verdad (sin él, esa prueba se salta).

Todo en `tmp_path`: la suite no lee `output/` ni la base real.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import anaqueles_eventos, anaqueles_registro  # noqa: E402
from src.analityc.core import anaqueles_stock as st  # noqa: E402
from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.app import informe_pasillo as ip  # noqa: E402
from src.app import informe_pdf  # noqa: E402

CAM = 'win-prueba-c3-i1'


# ── El botón ─────────────────────────────────────────────────────────────
def test_el_boton_pide_el_del_pasillo_con_periodo_y_camara(monkeypatch):
    pedidos = []
    monkeypatch.setattr(informe_pdf, '_pdf_pasillo',
                        lambda d, h, c: pedidos.append((d, h, c)) or b'%PDF-pasillo')
    r = informe_pdf.informe_tienda(desde='2026-10-13', hasta='2026-10-12', camara=CAM)
    assert pedidos == [('2026-10-12', '2026-10-13', CAM)]   # invertido: se endereza
    assert r.body == b'%PDF-pasillo'
    assert 'Informe_Pasillo_2026-10-12_a_2026-10-13.pdf' in r.headers['content-disposition']
    r = informe_pdf.informe_tienda('2026-10-09')             # `fecha`, como antes
    assert pedidos[-1] == ('2026-10-09', '2026-10-09', None)
    assert 'Informe_Pasillo_2026-10-09.pdf' in r.headers['content-disposition']


def test_sin_poder_imprimirlo_sale_el_clasico_del_ultimo_dia(monkeypatch):
    clasico = []
    monkeypatch.setattr(informe_pdf, 'construir_informe_tienda',
                        lambda destino, fecha: clasico.append(fecha))
    monkeypatch.setattr(ip, 'navegador', lambda: None)
    monkeypatch.setattr(ip, 'construir', lambda *a: '<html></html>')
    informe_pdf.informe_tienda(desde='2026-10-10', hasta='2026-10-12')
    assert clasico == ['2026-10-12']

    def roto(*a):
        raise RuntimeError('base caída')
    monkeypatch.setattr(ip, 'construir', roto)       # si el nuevo falla, igual
    informe_pdf.informe_tienda('2026-10-11')
    assert clasico[-1] == '2026-10-11'

    monkeypatch.setattr(informe_pdf, '_pdf_pasillo',
                        lambda *a: pytest.fail('con formato=clasico no se pide'))
    informe_pdf.informe_tienda('2026-10-09', formato='clasico')
    assert clasico[-1] == '2026-10-09'


# ── Los datos ────────────────────────────────────────────────────────────
@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Base, anaqueles, mapas de calor y logs, todo en `tmp_path`."""
    ruta = tmp_path / 'tienda.db'
    ta.reiniciar_para_pruebas(ruta=ruta)
    monkeypatch.setattr(ip, '_bd', lambda: sqlite3.connect(str(ruta)))
    monkeypatch.setattr(ip, 'SERVIDOR', tmp_path)
    for modulo in (anaqueles_registro, anaqueles_eventos):
        monkeypatch.setattr(modulo, 'raiz_salida', lambda: tmp_path / 'anaqueles')
    monkeypatch.setattr(AnalyticsConfig, 'HEATMAP_DIR', str(tmp_path / 'heatmap'))
    # El informe pone al día el índice de fichas: que mire una carpeta vacía.
    monkeypatch.setattr(AnalyticsConfig, 'CAPTURE_DIR', str(tmp_path / 'captures'))
    return ruta


def _sembrar(ruta: Path) -> str:
    """Un día: 3 personas (y un «no es persona»), 2 entradas y 1 agarre."""
    reg = anaqueles_registro.RegistroAnaqueles(CAM, raiz=ruta.parent / 'anaqueles')
    hoy = reg._fecha                  # el registro escribe el día de hoy
    base = time.mktime(time.strptime(f'{hoy} 10:00:00', '%Y-%m-%d %H:%M:%S'))
    con = sqlite3.connect(str(ruta))
    for stem, ts, genero, edad, uuid, falso in (
            ('f1_t11', base, 'Hombre', '26-35', 'uuid-aaaa1111', 0),
            ('f2_t12', base + 60, 'Mujer', '18-25', None, 0),
            ('f3_t13', base + 120, None, None, None, 0),
            ('f4_t14', base + 180, 'Hombre', '26-35', None, 1)):
        con.execute(
            "insert into tienda_capturas (stem, ts, jornada, camera, device_id, "
            "genero, edad, person_uuid, no_es_persona) values (?,?,?,?,?,?,?,?,?)",
            (stem, ts, hoy, 'Prueba', CAM, genero, edad, uuid, falso))
    for fila in (('f1_t11', base, hoy, CAM, 'Hombre', '26-35', 'uuid-aaaa1111', 0),
                 ('f2_t12', base + 60, hoy, CAM, 'Mujer', '18-25', None, 1)):
        con.execute("insert into tienda_clasificaciones values (?,?,?,?,?,?,?,?)", fila)
    for ts, pid, tipo, perm in ((base, 'uuid-aaaa1111', 'entrada', None),
                                (base + 40, 'uuid-aaaa1111', 'salida', 40.0),
                                (base + 60, 'prov:12', 'entrada', None),
                                (base + 68, 'prov:12', 'salida', 8.0)):
        con.execute("insert into tienda_eventos (ts, jornada, camera_id, persistent_id, "
                    "tipo, permanencia_s) values (?,?,?,?,?,?)", (ts, hoy, CAM, pid, tipo, perm))
    con.commit()
    con.close()
    # Anaqueles: los totales del día y cada visita/agarre con su hora.
    reg.area(0, 'Anaquel 1')
    reg.producto(0, 0, 'colcafe')
    reg.sumar(0, 'visitas')
    reg.sumar(0, 'agarres')
    reg.sumar_producto(0, 0)
    reg.volcar(force=True)
    anaqueles_eventos.apuntar(reg.carpeta, {'tipo': 'visita', 'area': 0,
                                            'area_nombre': 'Anaquel 1',
                                            'pid': 'uuid-aaaa1111'}, base + 10)
    anaqueles_eventos.apuntar(reg.carpeta, {'tipo': 'agarre', 'area': 0,
                                            'area_nombre': 'Anaquel 1', 'producto': 'colcafe',
                                            'pid': 'uuid-aaaa1111', 'genero': None}, base + 20)
    # Seguimientos: 4 números distintos en el registro por persona.
    salida = ruta.parent / 'output'
    salida.mkdir(exist_ok=True)
    with open(salida / f'analytics_log_{CAM}.jsonl', 'w', encoding='utf-8') as fh:
        for i, tid in enumerate((11, 11, 12, 13, 15)):
            fh.write(json.dumps({'event': 'person_detected',
                                 'timestamp': dt.datetime.fromtimestamp(base + i).isoformat(),
                                 'data': {'track_id': tid}}) + '\n')
    return hoy


def test_las_cifras_salen_de_la_base_y_de_los_anaqueles(entorno):
    hoy = _sembrar(entorno)
    d = ip.datos(hoy, hoy)
    assert d['camara'] == CAM and d['camara_corta'] == 'cámara 3'
    r = d['resumen']
    assert (r['capturadas'], r['identificadas'], r['clasificadas']) == (3, 1, 2)
    assert (d['demo']['hombres'], d['demo']['mujeres']) == (1, 1)
    assert (d['demo']['con_cara'], d['demo']['solo_cuerpo']) == (1, 1)
    assert d['trafico']['entradas'] == 2 and d['perm']['mediana'] == 24.0
    assert d['seguimientos'] == 4
    anq = d['anaqueles']
    assert anq['por_area'][0] == [1, 1]                  # 1 visita y 1 agarre
    assert anq['agarres'][0]['producto'] == 'colcafe'
    assert anq['agarres'][0]['genero'] == 'Hombre'        # el de su ficha
    pagina = ip.html(d)
    for texto in ('Informe del pasillo', 'cámara 3', 'Resumen del periodo',
                  'Demografía', 'Anaqueles', 'colcafe'):
        assert texto in pagina
    assert 'Sin datos para esta sección' not in pagina    # ninguna sección rota


def test_un_periodo_sin_datos_sale_igual(entorno):
    d = ip.datos('2020-01-01', '2020-01-03')
    assert d['vacio']
    pagina = ip.html(d)
    assert 'No hay datos registrados' in pagina and '1-ene al 3-ene' in pagina


# ── Imágenes ─────────────────────────────────────────────────────────────
def test_la_foto_del_stock_queda_en_antes_y_ahora(tmp_path):
    """La alerta guarda ANTES | AHORA | QUIÉN AGARRÓ; el informe no lleva la
    foto del cliente."""
    rng = np.random.default_rng(3)
    antes = rng.integers(0, 255, (60, 80, 1), dtype=np.uint8).repeat(3, 2)  # gris: nada rojo
    ahora = np.full((60, 80, 3), 120, np.uint8)
    persona = np.zeros((120, 50, 3), np.uint8)
    persona[..., 2] = 255                                  # rojo puro
    ruta = tmp_path / 'alerta.jpg'
    cv2.imwrite(str(ruta), st.componer_evidencia(antes, ahora, persona, 'colcafe'))
    completa = cv2.imread(str(ruta))
    recorte = cv2.imdecode(np.frombuffer(ip._antes_ahora(ruta), np.uint8), cv2.IMREAD_COLOR)

    def rojos(img):
        return int(((img[..., 2] > 200) & (img[..., 1] < 60) & (img[..., 0] < 60)).sum())
    assert rojos(completa) > 1000
    assert rojos(recorte) == 0
    assert recorte.shape[1] < completa.shape[1]


def test_los_huecos_oscuros_entre_botes_no_cortan_el_antes(tmp_path):
    """15-oct-2026: en los botes de Nescafé los huecos oscuros entre botes
    parecían las franjas que separan los paneles y el PDF cortaba dentro del
    ANTES. ANTES y AHORA miden lo mismo: el corte bueno es el que tiene su
    pareja al doble."""
    balda = np.full((60, 80, 3), 150, np.uint8)
    for x in (18, 38, 58):
        balda[:, x:x + 3] = 30                             # hueco del color del marco
    persona = np.zeros((120, 50, 3), np.uint8)
    persona[..., 2] = 255
    ruta = tmp_path / 'botes.jpg'
    cv2.imwrite(str(ruta), st.componer_evidencia(balda, balda.copy(), persona, 'Nescafe Gold'))
    completa = cv2.imread(str(ruta))
    assert len(ip._separadores(completa)) > 2              # los huecos sí engañan
    recorte = cv2.imdecode(np.frombuffer(ip._antes_ahora(ruta), np.uint8), cv2.IMREAD_COLOR)
    panel = (80 * 240 // 60) + 8                           # ancho de ANTES (y de AHORA)
    assert abs(recorte.shape[1] - 2 * panel) <= 4


# ── Medidas de stock tomadas durante un corte de iVMS ───────────────────
def _gris_ivms():
    """Un recorte de la imagen gris de un corte (gris 78 con ruido de JPEG)."""
    rng = np.random.default_rng(9)
    return np.clip(rng.normal(78, 2, (60, 80, 1)), 0, 255).astype(np.uint8).repeat(3, 2)


def _balda(semilla):
    rng = np.random.default_rng(semilla)
    return rng.integers(0, 255, (60, 80, 1), dtype=np.uint8).repeat(3, 2)


def _foto_stock(ruta, ahora, antes=None):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    antes = _balda(5) if antes is None else antes
    cv2.imwrite(str(ruta), st.componer_evidencia(antes, ahora, None, 'colcafe', titulo='Bajada 1'))


def test_la_foto_de_stock_de_un_corte_se_reconoce(tmp_path):
    """13-oct-2026: dos bajadas de las 08:11 se midieron sobre la imagen gris
    de un corte de iVMS. Una balda vacía y lisa de otro tono NO es un corte."""
    casos = {'ahora_gris': (_gris_ivms(), None, True),
             'antes_gris': (_balda(4), _gris_ivms(), True),   # foto de referencia gris
             'balda': (_balda(4), None, False),
             'vacia_clara': (np.full((60, 80, 3), 170, np.uint8), None, False)}
    for nombre, (ahora, antes, esperado) in casos.items():
        ruta = tmp_path / f'{nombre}.jpg'
        _foto_stock(ruta, ahora, antes)
        assert ip._medida_en_corte(ruta) is esperado, nombre


def test_el_agarre_que_solo_confirmo_un_corte_no_cuenta(entorno):
    """El agarre se cuenta en el mismo segundo que la bajada que lo confirma:
    si esa bajada se midió sobre la imagen gris, ni la foto sale ni el
    agarre cuenta, y el informe lo dice. Con una bajada de verdad en ese
    momento, sí cuenta."""
    hoy = _sembrar(entorno)                      # 1 agarre a las 10:00:20 (Anaquel 1)
    t = time.mktime(time.strptime(f'{hoy} 10:00:21', '%Y-%m-%d %H:%M:%S'))
    carpeta = entorno.parent / 'anaqueles' / CAM / 'stock_alertas'
    nombre = f"{time.strftime('%Y-%m-%d_%H%M%S', time.localtime(t))}_0_0_bajada.jpg"
    _foto_stock(carpeta / nombre, _gris_ivms())
    d = ip.datos(hoy, hoy)
    assert d['anaqueles']['agarres'] == [] and d['anaqueles']['en_corte'] == 1
    assert d['anaqueles']['por_area'][0] == [1, 0]
    assert d['stock'] == [] and d['stock_en_corte'] == 1
    pagina = ip.html(d)
    assert '1 agarre no se cuenta' in pagina and 'Se quitó 1 foto de stock' in pagina

    _foto_stock(carpeta / nombre.replace('_0_0_', '_0_1_'), _balda(6))
    d = ip.datos(hoy, hoy)
    assert len(d['anaqueles']['agarres']) == 1 and d['anaqueles']['en_corte'] == 0
    assert len(d['stock']) == 1 and d['stock_en_corte'] == 1


def test_el_mapa_general_se_salta_una_hora_gris(entorno, tmp_path):
    carpeta = tmp_path / 'heatmap' / 'history' / CAM
    carpeta.mkdir(parents=True)
    hoy = dt.date.today()
    gris = np.full((360, 640, 3), 78, np.uint8)            # pintada sobre un corte
    bueno = np.full((360, 640, 3), 90, np.uint8)
    bueno[200:] = (((np.arange(640) // 16) % 2) * 120 + 40).astype(np.uint8)[None, :, None]
    cv2.imwrite(str(carpeta / f'{hoy.isoformat()}_10.png'), gris)
    cv2.imwrite(str(carpeta / f'{hoy.isoformat()}_11.png'), bueno)
    m = ip._mapa_general(CAM, Counter({(hoy, 10): 9, (hoy, 11): 2}))
    assert m is not None and m['hora'] == 11


def test_con_navegador_el_pdf_sale_de_verdad():
    if ip.navegador() is None:
        pytest.skip('sin Edge/Chrome (p. ej. el servidor de Linux): sale el clásico')
    pdf = ip.a_pdf('<!doctype html><html><body><h1>Informe del pasillo</h1></body></html>')
    assert pdf is not None and pdf[:5] == b'%PDF-'


def test_seccion_del_personal_con_su_permanencia(tmp_path, monkeypatch):
    """19-oct-2026: el informe trae la permanencia propia del personal de
    camiseta (`personal_mapa.py`), solo de las horas del periodo."""
    import datetime as _dt
    import json as _json
    import time as _time
    from src.app import informe_pasillo as ip
    from src.analityc.core.anaqueles_registro import _carpeta_nombre
    monkeypatch.setattr(ip, 'SERVIDOR', tmp_path)
    carpeta = tmp_path / 'output' / 'personal_permanencia' / _carpeta_nombre('cam-p')
    carpeta.mkdir(parents=True)
    (carpeta / '2026-10-19.json').write_text(_json.dumps({'horas': {
        '09': {'con_personal_s': 600.0, 'persona_s': 900.0},
        '10': {'con_personal_s': 300.0, 'persona_s': 300.0},
        '15': {'con_personal_s': 999.0, 'persona_s': 999.0}}}), encoding='utf-8')
    f = _dt.date(2026, 10, 19)
    a = _time.mktime(_dt.datetime(2026, 10, 19, 8).timetuple())
    b = _time.mktime(_dt.datetime(2026, 10, 19, 12).timetuple())
    p = ip._datos_personal('cam-p', [f], a, b)
    assert p['con_personal_s'] == 900.0 and p['persona_s'] == 1200.0   # sin las 15h
    html = ip._sec_personal({'personal_camiseta': p})
    assert 'Personal de tienda' in html and html.count('<tr><td>') == 2
    assert ip._sec_personal({'personal_camiseta': None}) == ''


def test_el_pdf_ensena_las_reposiciones_del_personal():
    """21-oct-2026: el surtido (`..._surtido.jpg`) no salía en el PDF."""
    from src.app import informe_pasillo as ip
    assert ip._FOTO.match('2026-10-21_143116_0_2_surtido.jpg').group(5) == '_surtido'
    html = ip._sec_stock({'stock': [{'t': 1792540000.0, 'tipo': 'surtido', 'area': 'Anaquel 1',
                                     'producto': 'illy', 'imagen': b'x'}]})
    assert 'repuesto por el personal' in html and 'antes y después de la reposición' in html
