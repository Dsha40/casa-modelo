"""
Pruebas del identificador estable de camara (H-11).

Verifican la propiedad que da sentido a todo el historico por zona: **el mismo
dispositivo fisico produce el mismo `device_id` en ejecuciones distintas**.
Antes de H-11 el valor era `uuid.uuid4()` por panel, asi que esta propiedad no
se cumplia nunca y los heatmaps se fragmentaban en un UUID por sesion.

Hasta el 30-jul-2026 estas pruebas extraian las funciones del `render_box.py`
de tienda leyendo su codigo fuente, porque solo ese cliente tenia el arreglo.
Ahora la logica esta en el nucleo y la comparten los cuatro, asi que se prueba
donde vive. Las dos ultimas pruebas son las que vigilan que siga siendo asi.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
# Son las CARPETAS, no los `client_type`: 'managers' se renombró a
# 'detecciones' el 6-ago-2026 y su identificador de contrato no cambió.
CLIENTES = ('tienda', 'perimetrales', 'detecciones', 'amazonas')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elde_core.contracts import ClientType, Envelope, EventType  # noqa: E402
from elde_core.ui.identidad_camara import (device_id, id_instalacion,  # noqa: E402
                                           nombre_visible,
                                           slug)


def _render_box(cliente: str) -> Path:
    return (RAIZ / 'clients' / cliente / 'src' / 'gui' / 'components' /
            'render_box' / 'render_box.py')


def test_el_mismo_canal_dvr_da_el_mismo_id():
    """LA propiedad de H-11: estable entre 'reinicios' de la aplicacion."""
    uno = device_id(serie_dvr='J12345678', canal_dvr='2', indice=0)
    dos = device_id(serie_dvr='J12345678', canal_dvr='2', indice=3)
    assert uno == dos
    assert uno == 'dvr-J12345678-2'


def test_canales_distintos_dan_ids_distintos():
    ids = {device_id(serie_dvr='J12345678', canal_dvr='1'),
           device_id(serie_dvr='J12345678', canal_dvr='2'),
           device_id(serie_dvr='OTRO9999', canal_dvr='1')}
    assert len(ids) == 3


def test_cae_al_titulo_de_la_ventana():
    assert device_id(titulo_ventana='iVMS-4200') == 'win-iVMS-4200'


def test_ventanas_con_el_mismo_titulo_en_recuadros_distintos_se_separan():
    """H-39: iVMS-4200 abre varias ventanas y TODAS se titulan igual.

    Cuando dos recuadros capturaban dos de esas ventanas, ambos mandaban el
    mismo camera_id y el servidor entrelazaba las dos escenas en un solo
    rastreador — que se quedaba ciego (0/40 frames con tracks, medido con el
    pipeline real; 37/40 con una sola escena). Cada recuadro debe tener
    identidad propia a partir del segundo.
    """
    assert device_id(titulo_ventana='iVMS-4200', indice=0) == 'win-iVMS-4200'
    assert device_id(titulo_ventana='iVMS-4200', indice=1) == 'win-iVMS-4200-c2'
    assert device_id(titulo_ventana='iVMS-4200', indice=3) == 'win-iVMS-4200-c4'


def test_el_recuadro_1_conserva_su_id_historico():
    """El sufijo empieza en el recuadro 2: el caso comun (una sola ventana,
    recuadro 0) no debe partir su historico de heatmaps y tableros."""
    assert device_id(titulo_ventana='iVMS-4200', indice=0) == 'win-iVMS-4200'


def test_ultimo_recurso_por_posicion():
    assert device_id(indice=0) == 'box-1'
    assert device_id(indice=4) == 'box-5'


def test_el_dvr_manda_sobre_el_titulo():
    """Si hay canal DVR, es la identidad mas fiable y gana."""
    assert device_id(serie_dvr='J1', canal_dvr='3',
                     titulo_ventana='iVMS-4200', indice=7) == 'dvr-J1-3'


def test_el_id_es_valido_como_nombre_de_archivo():
    """Los device_id acaban siendo `output/heatmap/<id>.png`."""
    ident = device_id(titulo_ventana='C:\\ruta\\../etc passwd?*<>')
    for prohibido in '\\/:*?"<>| ':
        if prohibido == ':':
            continue          # los dos puntos si son validos en el contrato
        assert prohibido not in ident, f'{prohibido!r} en {ident!r}'
    assert '..' not in ident


def test_el_id_pasa_la_validacion_del_envelope():
    """Cierra el circulo: lo que genera el cliente lo acepta el contrato."""
    for ident in (device_id(serie_dvr='J12345678', canal_dvr='2'),
                  device_id(titulo_ventana='Camara del pasillo 3'),
                  device_id(indice=2),
                  device_id(titulo_ventana='C:\\ruta\\rara ../x')):
        env = Envelope(client_type=ClientType.TIENDA, site_id='lacomarca',
                       device_id=ident, event_type=EventType.FRAME_INFERENCE)
        assert env.device_id == ident


def test_slug_no_devuelve_cadena_vacia():
    assert slug('') == 'sin_nombre'
    assert slug('///') == 'sin_nombre'


def test_nombre_visible_sigue_el_mismo_orden_de_fuentes():
    assert nombre_visible(nombre_dvr='Pasillo 3', titulo_ventana='x') == 'Pasillo 3'
    assert nombre_visible(titulo_ventana='iVMS-4200') == 'iVMS-4200'
    assert nombre_visible(indice=2) == 'Camara 3'


def test_nombre_visible_distingue_recuadros_del_mismo_titulo():
    """Companero del sufijo del device_id: sin esto el tablero enseñaria dos
    camaras DISTINTAS con el mismo nombre y nadie sabria cual es cual."""
    assert nombre_visible(titulo_ventana='iVMS-4200', indice=1) == 'iVMS-4200 (2)'
    assert nombre_visible(titulo_ventana='iVMS-4200', indice=0) == 'iVMS-4200'


# ---------------------------------------------------------------------------
# Los dos guardias: que los CUATRO clientes usen esto y no una copia propia.
# ---------------------------------------------------------------------------

def test_ningun_cliente_manda_el_uuid_de_sesion_como_camera_id():
    """La regresion concreta de H-11.

    `component_key` es un `uuid.uuid4()` por panel. Si vuelve a viajar como
    `camera_id`, el historico por camara deja de acumularse otra vez.
    """
    culpables = []
    for cliente in CLIENTES:
        archivo = _render_box(cliente)
        if not archivo.is_file():
            culpables.append(f'{cliente}: no existe {archivo}')
            continue
        texto = archivo.read_text(encoding='utf-8', errors='replace')
        for n, linea in enumerate(texto.splitlines(), 1):
            if re.search(r'["\']camera_id["\']\s*:\s*self\.component_key',
                         linea):
                culpables.append(f'{cliente}:{n}')
    assert not culpables, ('vuelve a viajar el uuid de sesion como camera_id: '
                           + ', '.join(culpables))


def test_los_cuatro_clientes_delegan_en_el_nucleo():
    """Que nadie se quede una copia propia de la logica de identidad.

    Es como empezo H-11: el arreglo existia en un cliente y los otros tres
    seguian con el fallo.
    """
    faltan = []
    for cliente in CLIENTES:
        archivo = _render_box(cliente)
        if not archivo.is_file():
            faltan.append(f'{cliente}: no existe el archivo')
            continue
        texto = archivo.read_text(encoding='utf-8', errors='replace')
        if 'identidad_camara' not in texto:
            faltan.append(f'{cliente}: no importa el nucleo')
        elif '_identidad.device_id(' not in texto:
            faltan.append(f'{cliente}: no llama a _identidad.device_id()')
    assert not faltan, 'clientes con identidad propia: ' + ', '.join(faltan)


# ── La tercera lección: dos CLIENTES en la misma PC (9-sep-2026) ─────────

def test_dos_clientes_de_la_misma_pc_ya_no_chocan(tmp_path):
    """El caso real que se vio en producción.

    Una misma PC con dos iVMS-4200 abiertos —la 3.1 y la 2.8, que ponen el
    MISMO título— capturados por dos clientes, cada uno con su «recuadro 1».
    Los dos mandaban `win-iVMS-4200`, y el servidor los trataba como una sola
    cámara: un rastreador para dos escenas y el local saltando entre
    «Cámaras CY Aruba» y «Carpinteria» varias veces por segundo.
    """
    uno = id_instalacion(tmp_path / 'cliente_a')
    dos = id_instalacion(tmp_path / 'cliente_b')
    assert uno and dos and uno != dos

    a = device_id(titulo_ventana='iVMS-4200', indice=0, instalacion=uno)
    b = device_id(titulo_ventana='iVMS-4200', indice=0, instalacion=dos)
    assert a != b, 'los dos «recuadro 1» siguen chocando'


def test_sin_instalacion_el_id_es_EXACTAMENTE_el_de_antes():
    """El parámetro es aditivo: estrenarlo tiene que ser una decisión.

    Si el defecto cambiara el id, actualizar el núcleo partiría el histórico
    de TODAS las cámaras por ventana de golpe, sin que nadie lo pidiera.
    """
    assert device_id(titulo_ventana='iVMS-4200') == 'win-iVMS-4200'
    assert device_id(titulo_ventana='iVMS-4200', indice=1) == 'win-iVMS-4200-c2'
    assert device_id(titulo_ventana='iVMS-4200', instalacion='') == 'win-iVMS-4200'


def test_la_instalacion_no_toca_las_camaras_de_dvr(tmp_path):
    """Un canal de DVR ya es único por serie y canal: meterle la instalación
    partiría su histórico sin arreglar nada."""
    marca = id_instalacion(tmp_path)
    con = device_id(serie_dvr='J12345678', canal_dvr='2', instalacion=marca)
    sin = device_id(serie_dvr='J12345678', canal_dvr='2')
    assert con == sin == 'dvr-J12345678-2'


def test_el_id_de_instalacion_sobrevive_al_reinicio(tmp_path):
    """Es lo que el `component_key` no daba (H-11): un uuid4 por sesión
    inventaba cámaras nuevas en cada arranque."""
    primero = id_instalacion(tmp_path)
    assert primero
    for _ in range(3):
        assert id_instalacion(tmp_path) == primero


def test_el_indice_sigue_separando_dentro_del_mismo_cliente(tmp_path):
    """La lección anterior (H-39) no se pierde: dos ventanas homónimas del
    MISMO cliente siguen separándose por el recuadro."""
    marca = id_instalacion(tmp_path)
    ids = {device_id(titulo_ventana='iVMS-4200', indice=i, instalacion=marca)
           for i in range(4)}
    assert len(ids) == 4


def test_una_carpeta_imposible_no_deja_al_cliente_sin_identidad():
    """Sin poder escribir, se devuelve '' y `device_id` se comporta como
    siempre. Quedarse sin id sería peor que no separar las cámaras."""
    assert id_instalacion('/proc/no_puedo_escribir_aqui/jamas') == ''


def test_el_id_sigue_sirviendo_de_nombre_de_archivo(tmp_path):
    """El `device_id` acaba siendo `output/heatmap/<id>.png`."""
    marca = id_instalacion(tmp_path)
    ident = device_id(titulo_ventana='C:\\ruta rara/iVMS 4200 (2.8)',
                      indice=2, instalacion=marca)
    assert all(c.isalnum() or c in '_-.' for c in ident), ident
    assert '..' not in ident and '/' not in ident and '\\' not in ident


def test_los_cuatro_clientes_mandan_su_instalacion():
    """El cableado, vigilado por AST y no por confianza.

    De nada sirve el parámetro si un cliente se queda sin pasarlo: ese
    cliente seguiría chocando con los demás y el síntoma volvería a ser el de
    siempre —dos escenas en un rastreador, el local saltando— sin un error.

    Se lee el AST y no el texto: un `grep` pasaría con la palabra dentro de un
    comentario.
    """
    import ast

    raiz = Path(__file__).resolve().parents[3]
    faltan = []
    for cliente in ('tienda', 'perimetrales', 'detecciones', 'amazonas'):
        ruta = (raiz / 'clients' / cliente / 'src' / 'gui' / 'components' /
                'render_box' / 'render_box.py')
        assert ruta.is_file(), f'no existe {ruta}'
        arbol = ast.parse(ruta.read_text(encoding='utf-8'))
        ok = False
        for n in ast.walk(arbol):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == 'device_id':
                if any(k.arg == 'instalacion' for k in n.keywords):
                    ok = True
        if not ok:
            faltan.append(cliente)
    assert not faltan, (
        f'estos clientes no mandan su instalación: {faltan}; sus ventanas '
        f'seguirán chocando con las de otro cliente de la misma PC')


def test_si_no_hay_identificador_se_avisa_en_vez_de_callarse(monkeypatch, caplog):
    """El arreglo no puede apagarse en silencio.

    Pasó al escribirlo: `appdirs` faltaba en el entorno de pruebas, la función
    devolvía '' por un `except` mudo y todo «pasaba» mientras la separación de
    cámaras no ocurría. Es el mismo fallo que este cambio viene a cerrar, y
    por eso ahora deja rastro.
    """
    import logging
    import elde_core.ui.identidad_camara as m

    monkeypatch.setattr(m, '_CACHE_INSTALACION', {})
    monkeypatch.setattr(m, 'id_instalacion', lambda *a, **k: '')
    with caplog.at_level(logging.WARNING,
                         logger='elde_core.ui.identidad_camara'):
        assert m.instalacion_de_app('cliente_de_prueba') == ''
    assert any('cliente_de_prueba' in r.getMessage() for r in caplog.records), (
        'quedarse sin identificador tiene que dejar rastro en el log')


def test_el_identificador_se_lee_del_disco_una_sola_vez(monkeypatch, tmp_path):
    """`device_id` se llama por frame y por recuadro: ir al disco cada vez
    sería un `open()` por frame."""
    import elde_core.ui.identidad_camara as m
    monkeypatch.setattr(m, '_CACHE_INSTALACION', {})
    veces = []
    monkeypatch.setattr(m, 'id_instalacion',
                        lambda *a, **k: (veces.append(1), 'abc123')[1])
    monkeypatch.setitem(__import__('sys').modules, 'appdirs',
                        type(__import__('sys'))('appdirs'))
    import sys as _s
    _s.modules['appdirs'].user_config_dir = lambda app: str(tmp_path / app)
    for _ in range(5):
        assert m.instalacion_de_app('un_cliente') == 'abc123'
    assert len(veces) == 1, f'fue al disco {len(veces)} veces'
