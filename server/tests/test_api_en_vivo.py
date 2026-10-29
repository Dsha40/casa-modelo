"""
Página «En vivo» del dashboard de tienda (22-oct-2026): `src/app/api_en_vivo.py`.

Lo que se fija:

  - **vídeo, imagen gris de un corte, captura vacía de una ventana
    minimizada y cámara parada** se distinguen, y el «desde» solo se da por
    bueno cuando se vio el cambio: la primera vez que se mira una cámara no
    se sabe cuándo empezó su estado;
  - **confirmados, sin confirmar y esperando bajada** salen de donde viven:
    los dos primeros del archivo del día, los que esperan de la memoria del
    MEDIDOR de la cámara (no se guardan en disco). La primera versión los
    buscaba en el procesador y en vivo nunca encontraba ninguno: las
    pruebas usaban un doble con otra forma. Ahora el medidor es el de
    verdad (`MedidorAnaqueles`);
  - **en modo Tiendas no hay agarres** (`null`, no ceros);
  - **las fichas clasificadas suman**: una fila por ficha, de esa cámara y
    esa jornada;
  - **la GPU** se lee del CSV del vigilante, con «por calor» según los bits
    de nvidia-smi;
  - **un bloque que falla no rompe los demás** ni la respuesta.

Nada lee los datos reales: el día de anaqueles, la base y el CSV viven en
`tmp_path`. Los procesadores son de mentira (crear uno de verdad carga los
modelos), pero su medidor es el real.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.analityc.core import anaqueles  # noqa: E402
from src.analityc.core import anaqueles_eventos, anaqueles_registro  # noqa: E402
from src.analityc.core import tienda_almacen as ta  # noqa: E402
from src.analityc.core.anaqueles_registro import RegistroAnaqueles  # noqa: E402
from src.app import api_en_vivo as ev  # noqa: E402
from src.app import telemetria_fps as tf  # noqa: E402

CAM = 'win-iVMS-4200-c3-prueba'
FECHA = '2026-09-01'
AHORA = 1_788_000_000.0


def _medidor(tmp_path):
    """El medidor DE VERDAD de una cámara, con dos gestos esperando bajada.
    Se añaden a su propia lista: si esa lista cambiara de nombre, esto
    fallaría en vez de seguir probando un doble que ya no se parece."""
    m = anaqueles.MedidorAnaqueles(
        CAM, dict(anaqueles._DEFECTOS, stock_ventana_s=180.0),
        pose=object(), escena=object(), identificador=object(),
        registro=RegistroAnaqueles(CAM, raiz=tmp_path / 'medidor'))
    m._agarres_por_confirmar.extend([
        {'idx': 2, 't': AHORA - 45.0, 'pid': 'prov:7'},
        {'idx': 3, 't': AHORA - 5.0, 'pid': 'prov:8'},
    ])
    return m


class Anaqueles:
    """Lo que la página lee de un procesador de Anaqueles vivo. Su medidor
    es None hasta la primera imagen medida, como en el de verdad."""

    def __init__(self, frame, medidor=None):
        self.last_processed_frame = frame
        self.camera_states = {}
        self.active_tracks = {1: {'class': -1}, 2: {'class': -1}}
        self._zone_state = {1: {'estado': 'DENTRO'}, 2: {'estado': 'FUERA'}}
        self._medidor = medidor


def test_el_procesador_de_verdad_guarda_su_medidor_donde_se_lee():
    """El doble de arriba copia este atributo del `Anaqueles` real."""
    assert 'self._medidor' in inspect.getsource(anaqueles.Anaqueles.__init__)


class TiendasProcess:
    def __init__(self, frame):
        self.last_processed_frame = frame
        self.camera_states = {}
        self.active_tracks = {}
        self._zone_state = {}


def _gris():
    """La pantalla lisa de un corte de iVMS."""
    return np.full((868, 1456, 3), 128, dtype=np.uint8)


def _video():
    """Bloques claros y oscuros de 100 px. Ruido aleatorio no sirve: al
    reducirlo a 160x90 se promedia y queda casi liso, como la gris."""
    filas, cols = np.indices((868, 1456))
    tablero = ((filas // 100 + cols // 100) % 2 * 200).astype(np.uint8)
    return np.dstack([tablero] * 3)


@pytest.fixture(autouse=True)
def limpio(tmp_path, monkeypatch):
    """Estado propio por prueba; anaqueles y GPU en `tmp_path`."""
    monkeypatch.setattr(ev, '_video', {})
    monkeypatch.setattr(ev, '_cache_hoy', {})
    monkeypatch.setattr(tf, '_historial', {})
    monkeypatch.setattr(tf, '_ultima_imagen', {CAM: AHORA - 0.3})
    monkeypatch.setattr(tf, '_ultimo_tamano', {CAM: 39 * 1024})
    monkeypatch.setattr(anaqueles_registro, 'raiz_salida',
                        lambda: tmp_path / 'anaqueles')
    monkeypatch.setattr(anaqueles_eventos, 'raiz_salida',
                        lambda: tmp_path / 'anaqueles')
    monkeypatch.setattr(anaqueles_registro, '_hoy', lambda: FECHA)
    monkeypatch.setattr(ev, 'RUTA_GPU', tmp_path / 'gpu_temp.csv')
    # La gente de hoy tiene su propia prueba: aquí no toca ninguna base.
    monkeypatch.setattr(ev, '_gente_hoy', lambda cam, ahora: None)
    return tmp_path


def _conectar(monkeypatch, *procs):
    """Hace de `app.py`: `_iter_person_processors` con estos procesadores."""
    falso = types.ModuleType('src.app.app')
    falso._iter_person_processors = lambda: [
        (f'cliente-{i}', cam, proc) for i, (cam, proc) in enumerate(procs)]
    monkeypatch.setitem(sys.modules, 'src.app.app', falso)


def _dia(tmp_path, areas, eventos=()):
    carpeta = tmp_path / 'anaqueles' / CAM
    (carpeta / 'eventos').mkdir(parents=True)
    (carpeta / f'{FECHA}.json').write_text(json.dumps(
        {'fecha': FECHA, 'camera_id': CAM, 'areas': areas}), encoding='utf-8')
    (carpeta / 'eventos' / f'{FECHA}.jsonl').write_text(
        ''.join(json.dumps(e) + '\n' for e in eventos), encoding='utf-8')


# ── Vídeo, imagen gris o parada ──────────────────────────────────────────

def test_imagen_gris_y_desde_cuando():
    proc = Anaqueles(_gris())
    primera = ev._camara(CAM, proc, AHORA)['video']
    assert primera['estado'] == 'gris'
    # La primera vez no se sabe cuándo empezó: «al menos desde».
    assert primera['cambio_visto'] is False

    proc.last_processed_frame = _video()
    segunda = ev._camara(CAM, proc, AHORA + 3)['video']
    assert segunda['estado'] == 'video'
    assert segunda['cambio_visto'] is True and segunda['desde'] == AHORA + 3

    # Sigue igual: el «desde» no se mueve.
    tercera = ev._camara(CAM, proc, AHORA + 6)['video']
    assert tercera['desde'] == AHORA + 3


def test_captura_vacia_de_una_ventana_minimizada_no_es_un_corte():
    """22-oct-2026, 17:40: iVMS minimizado, imágenes de 1 KB. Lisa como la
    gris, pero se arregla restaurando la ventana, no en iVMS."""
    tf._ultimo_tamano[CAM] = 1024
    video = ev._camara(CAM, Anaqueles(_gris()), AHORA)['video']
    assert video['estado'] == 'vacia' and video['ultima_imagen_kb'] == 1.0

    tf._ultimo_tamano[CAM] = 39 * 1024           # la pantalla de un corte
    video = ev._camara(CAM, Anaqueles(_gris()), AHORA + 3)['video']
    assert video['estado'] == 'gris' and video['cambio_visto'] is True


def test_el_arranque_del_servidor_no_es_un_cambio_de_estado():
    """22-oct-2026: tras un reinicio decía «vacía desde las 17:47:22», la
    hora del arranque, con iVMS minimizado desde antes. Sin ninguna imagen
    aún no se sabe nada: el primer estado de verdad es «al menos desde»."""
    del tf._ultima_imagen[CAM]
    antes = ev._camara(CAM, Anaqueles(_gris()), AHORA)['video']
    assert antes['estado'] == 'parada' and antes['desde'] is None

    tf._ultima_imagen[CAM] = AHORA + 2.9
    primera = ev._camara(CAM, Anaqueles(_gris()), AHORA + 3)['video']
    assert primera['estado'] == 'gris' and primera['cambio_visto'] is False


def test_sin_imagenes_es_parada_con_la_hora_de_la_ultima():
    tf._ultima_imagen[CAM] = AHORA - 60.0
    video = ev._camara(CAM, Anaqueles(_video()), AHORA)['video']
    assert video['estado'] == 'parada'
    assert video['desde'] == AHORA - 60.0 and video['cambio_visto'] is True


# ── Agarres ──────────────────────────────────────────────────────────────

def test_confirmados_sin_confirmar_y_esperando(limpio):
    _dia(limpio, {
        '1': {'nombre': 'Anaquel 2', 'visitas': 4, 'agarres': 1,
              'agarres_sin_confirmar': 0},
        '2': {'nombre': 'Anaquel 3', 'visitas': 9, 'agarres': 2,
              'agarres_sin_confirmar': 3},
    }, eventos=[
        {'tipo': 'visita', 'hora': '10:00:00', 'area_nombre': 'Anaquel 3'},
        {'tipo': 'agarre', 'hora': '10:01:00', 'area_nombre': 'Anaquel 3',
         'producto': 'Blue'},
        {'tipo': 'agarre', 'hora': '11:30:00', 'area_nombre': 'Anaquel 2',
         'producto': None},
    ])
    agarres = ev._camara(CAM, Anaqueles(_video(), _medidor(limpio)),
                         AHORA)['agarres']
    por_area = {a['idx']: a for a in agarres['areas']}

    assert por_area[2]['confirmados'] == 2
    assert por_area[2]['sin_confirmar'] == 3
    assert por_area[2]['esperando'] == 1
    assert por_area[2]['esperando_max_s'] == 45.0
    assert por_area[1]['esperando'] == 0 and por_area[1]['esperando_max_s'] is None
    # Un gesto en un área que aún no llegó al disco también se ve.
    assert por_area[3]['esperando'] == 1 and por_area[3]['confirmados'] == 0
    assert agarres['ventana_s'] == 180.0
    # Los últimos confirmados, el más nuevo primero; las visitas no.
    assert [u['hora'] for u in agarres['ultimos_confirmados']] == [
        '11:30:00', '10:01:00']


def test_antes_de_la_primera_imagen_medida_salen_los_del_disco(limpio):
    """El medidor nace con la primera imagen medida: hasta entonces no hay
    nadie esperando, pero los totales del día ya están en el disco."""
    _dia(limpio, {'2': {'nombre': 'Anaquel 3', 'visitas': 9, 'agarres': 2,
                        'agarres_sin_confirmar': 3}})
    agarres = ev._camara(CAM, Anaqueles(_video()), AHORA)['agarres']
    assert agarres['areas'] == [{
        'idx': 2, 'nombre': 'Anaquel 3', 'visitas': 9, 'confirmados': 2,
        'sin_confirmar': 3, 'esperando': 0, 'esperando_max_s': None}]


def test_modo_tiendas_no_tiene_agarres():
    cam = ev._camara(CAM, TiendasProcess(_video()), AHORA)
    assert cam['modo'] == 'Tiendas'
    assert cam['agarres'] is None


def test_un_bloque_roto_no_rompe_los_demas():
    class Roto(Anaqueles):
        @property
        def _medidor(self):
            raise RuntimeError('a medio cambiar')

        @_medidor.setter
        def _medidor(self, valor):
            pass

    cam = ev._camara(CAM, Roto(_gris()), AHORA)
    assert cam['agarres'] is None
    assert cam['video']['estado'] == 'gris'
    assert cam['gente_ahora']['en_imagen'] == 2


# ── Gente ────────────────────────────────────────────────────────────────

def test_gente_ahora_en_imagen_y_dentro_del_area(monkeypatch):
    proc = Anaqueles(_video())
    _conectar(monkeypatch, (CAM, proc))
    gente = ev._camara(CAM, proc, AHORA)['gente_ahora']
    assert gente == {'en_imagen': 2, 'dentro_area': 1}


def test_fichas_clasificadas_suman(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, 'CORTE_JORNADA_H', 0)
    alm = ta.reiniciar_para_pruebas(ruta=tmp_path / 'tienda.db')
    ts = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))
    jornada = ta.jornada_de(ts)
    for stem, genero, uuid in (('a', 'Hombre', 'u1'), ('b', 'Hombre', 'u1'),
                               ('c', 'Mujer', None)):
        alm.anotar_clasificacion(CAM, stem, genero, 'Adulto',
                                 person_uuid=uuid, ts=ts)
    alm.anotar_clasificacion('otra-camara', 'd', 'Mujer', 'Adulto', ts=ts)
    alm.anotar_clasificacion(CAM, 'e', 'Mujer', 'Adulto', ts=ts - 86400)
    alm.volcar(force=True)

    clasificadas = ev._clasificadas(alm, jornada, CAM)
    # Una fila por ficha: la misma identidad en dos fichas cuenta dos.
    assert clasificadas == {'total': 3, 'genero': {'Hombre': 2, 'Mujer': 1}}


# ── GPU ──────────────────────────────────────────────────────────────────

def test_gpu_del_csv_del_vigilante(limpio):
    (limpio / 'gpu_temp.csv').write_text(
        '﻿hora,temp_C,uso_pct,memoria_MB,consumo_W,reloj_MHz,'
        'ventilador_pct,limitacion\n'
        '2026-10-22 17:19:14,84,44,2594,76.00,1650,73,0x0000000000000001\n'
        '2026-10-22 17:20:1\n'                          # fila cortada
        '2026-10-22 17:20:14,85,50,2598,67.98,1620,[N/A],0x0000000000000020\n',
        encoding='utf-8')
    gpu = ev._gpu()
    ultima = gpu['ultima']
    assert ultima['temp_c'] == 85 and ultima['uso_pct'] == 50
    assert ultima['ventilador_pct'] is None
    assert ultima['por_calor'] is True and ultima['motivos'] == ['calor']
    assert [f['temp_c'] for f in gpu['historial']] == [84, 85]


def test_gpu_en_reposo_no_es_calor(limpio):
    (limpio / 'gpu_temp.csv').write_text(
        'hora,temp_C\n2026-10-22 17:19:14,60,0,100,20,300,30,0x1\n',
        encoding='utf-8')
    ultima = ev._gpu()['ultima']
    assert ultima['por_calor'] is False and ultima['motivos'] == ['en reposo']


def test_sin_csv_de_la_gpu_es_null():
    assert ev._gpu() is None


# ── El endpoint ──────────────────────────────────────────────────────────

def test_endpoint_junta_camaras_conectadas_y_vistas(monkeypatch):
    _conectar(monkeypatch, (CAM, Anaqueles(_gris())))
    tf._ultima_imagen['cam-que-se-fue'] = AHORA - 600
    monkeypatch.setattr(ev.time, 'time', lambda: AHORA)
    from src.app import recursos_pc
    pc = {'ultima': {'cpu_pct': 89.2, 'ram_libre_mb': 1957}, 'historial': []}
    monkeypatch.setattr(recursos_pc, 'resumen', lambda: pc)
    app = FastAPI()
    app.include_router(ev.router)

    r = TestClient(app).get('/api/v1/en-vivo')
    assert r.status_code == 200
    datos = r.json()
    assert datos['ahora'] == AHORA and datos['gpu'] is None
    assert datos['pc'] == pc                     # la CPU y la RAM del PC
    camaras = [c['camera_id'] for c in datos['camaras']]
    # Las conectadas primero.
    assert camaras == [CAM, 'cam-que-se-fue']
    assert datos['camaras'][0]['video']['estado'] == 'gris'
    se_fue = datos['camaras'][1]
    assert se_fue['conectada'] is False
    assert se_fue['video']['estado'] == 'parada'
