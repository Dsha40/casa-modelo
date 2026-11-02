"""
Perfiles de camara y el sufijo de instalacion (21-sep-2026).

## Por que existen

Los perfiles `config/camera_profiles/win-iVMS-4200.json` y
`win-iVMS-4200-c2.json` se calibraron el 7 y 8 de agosto de 2026 con
mediciones reales (bbox 40x90 en vez de 60x140, blur 16 en vez de 25, YuNet a
640...). El 9 de septiembre, `device_id()` estreno el parametro `instalacion`
para separar dos clientes que capturan ventanas homonimas, y paso a pegar un
`-i<marca>` al final del id. Desde entonces esos dos ficheros no los carga
nadie: el servidor arranca con `por_id={}` y la calibracion no se aplica.

`_resolver_perfil` arregla eso reintentando sin el sufijo. Aqui se fija que
lo hace, y —mas importante— que NO se lleva por delante lo que el sufijo vino
a proteger.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.analytics import demographics as demo  # noqa: E402

PERFILES = RAIZ / 'server' / 'config' / 'camera_profiles'


@pytest.fixture()
def perfiles(tmp_path, monkeypatch):
    """Directorio de perfiles aislado (no toca los del repositorio)."""
    destino = tmp_path / 'config' / 'camera_profiles'
    destino.mkdir(parents=True)
    monkeypatch.setattr(demo, '_PROJECT_ROOT', str(tmp_path))
    return destino


def _escribir(dirp: Path, nombre: str, datos: dict) -> Path:
    ruta = dirp / f'{nombre}.json'
    ruta.write_text(json.dumps(datos), encoding='utf-8')
    return ruta


# ── El caso que motiva el cambio ────────────────────────────────────────

def test_el_perfil_sin_sufijo_sirve_a_la_camara_con_sufijo(perfiles):
    _escribir(perfiles, 'win-iVMS-4200', {'DEMO_MIN_BLUR_VAR': 16.0})

    vista = demo._load_camera_profile('win-iVMS-4200-i58bf7b')

    assert float(vista.DEMO_MIN_BLUR_VAR) == 16.0


def test_tambien_con_el_sufijo_de_recuadro(perfiles):
    """El '-c2' va ANTES del de instalacion y debe conservarse."""
    _escribir(perfiles, 'win-iVMS-4200-c2', {'DEMO_MIN_PERSON_BBOX_H': 90})

    vista = demo._load_camera_profile('win-iVMS-4200-c2-i58bf7b')

    assert int(vista.DEMO_MIN_PERSON_BBOX_H) == 90


def test_el_recuadro_2_no_hereda_el_perfil_del_recuadro_1(perfiles):
    """Quitar el sufijo no puede colapsar `-c2` sobre el recuadro 1.

    Son escenas DISTINTAS (H-39): iVMS-4200 abre varias ventanas homonimas
    con camaras diferentes.
    """
    _escribir(perfiles, 'win-iVMS-4200', {'DEMO_MIN_BLUR_VAR': 16.0})

    vista = demo._load_camera_profile('win-iVMS-4200-c2-i58bf7b')

    # Sin `win-iVMS-4200-c2.json` no hay perfil: se queda con el global.
    assert float(vista.DEMO_MIN_BLUR_VAR) == float(
        demo.AnalyticsConfig.DEMO_MIN_BLUR_VAR)


# ── Lo que el sufijo vino a proteger ────────────────────────────────────

def test_el_id_completo_gana_al_nombre_base(perfiles):
    """Quien necesite separar dos instalaciones crea el fichero completo."""
    _escribir(perfiles, 'win-iVMS-4200', {'DEMO_MIN_BLUR_VAR': 16.0})
    _escribir(perfiles, 'win-iVMS-4200-i58bf7b', {'DEMO_MIN_BLUR_VAR': 9.0})

    vista = demo._load_camera_profile('win-iVMS-4200-i58bf7b')

    assert float(vista.DEMO_MIN_BLUR_VAR) == 9.0, 'el especifico debe mandar'


def test_sin_ningun_perfil_se_usa_la_config_global(perfiles):
    vista = demo._load_camera_profile('win-loquesea-i58bf7b')

    assert float(vista.DEMO_MIN_BLUR_VAR) == float(
        demo.AnalyticsConfig.DEMO_MIN_BLUR_VAR)
    assert int(vista.DEMO_MIN_PERSON_BBOX_W) == int(
        demo.AnalyticsConfig.DEMO_MIN_PERSON_BBOX_W)


def test_una_camara_de_dvr_no_se_toca(perfiles):
    """`device_id()` no le pone sufijo de instalacion a las camaras de DVR
    (ya son unicas por serie y canal), asi que aqui no hay nada que quitar."""
    _escribir(perfiles, 'dvr-ABC123-04', {'DEMO_MIN_BLUR_VAR': 12.0})

    assert float(demo._load_camera_profile('dvr-ABC123-04')
                 .DEMO_MIN_BLUR_VAR) == 12.0


# ── Bordes del recorte ──────────────────────────────────────────────────

def test_no_se_recorta_un_titulo_que_solo_parece_un_sufijo(perfiles):
    """'-i' + palabra corta puede darse en un titulo real.

    Mientras exista el fichero con el nombre completo, gana ese; el recorte
    es solo el segundo intento.
    """
    _escribir(perfiles, 'win-Panel-informe', {'DEMO_MIN_BLUR_VAR': 11.0})

    assert float(demo._load_camera_profile('win-Panel-informe')
                 .DEMO_MIN_BLUR_VAR) == 11.0


def test_el_nombre_se_sanitiza_y_se_trunca_a_64(perfiles):
    """Los puntos se caen y el nombre se corta: el fichero debe llamarse
    como el id SANITIZADO, no como el id crudo."""
    largo = ('win-HD_CCTV_Camera_video_3MP_4MP_iProx_CCTV_'
             'HDCCTVCameras.net_retail_store-i58bf7b')
    esperado = 'win-HD_CCTV_Camera_video_3MP_4MP_iProx_CCTV_HDCCTVCamerasnet_ret'
    assert len(esperado) == 64
    _escribir(perfiles, esperado, {'DEMO_MIN_BLUR_VAR': 7.0})

    assert float(demo._load_camera_profile(largo).DEMO_MIN_BLUR_VAR) == 7.0


def test_camera_id_vacio_o_none_no_rompe(perfiles):
    for valor in (None, '', '///'):
        vista = demo._load_camera_profile(valor)
        assert float(vista.DEMO_MIN_BLUR_VAR) == float(
            demo.AnalyticsConfig.DEMO_MIN_BLUR_VAR)


# ── Los perfiles que YA estan en el repositorio ─────────────────────────

def test_los_perfiles_del_repo_los_carga_su_camara_real():
    """Sin monkeypatch: se leen los ficheros de verdad de `config/`.

    Es la prueba de que la calibracion de agosto vuelve a aplicarse a los
    `device_id` que manda el cliente HOY, con sufijo de instalacion.
    """
    # El de c2 (stand de degustación) se quitó el 2-nov-2026: en la casa
    # modelo c2 es de solo permanencia y no lleva calibración (abajo).
    c2 = PERFILES / 'win-iVMS-4200-c2.json'
    if c2.is_file():
        assert demo._load_camera_profile(
            'win-iVMS-4200-c2-i58bf7b').SOLO_PERMANENCIA is True
    for archivo, camera_id in (
            ('win-iVMS-4200.json', 'win-iVMS-4200-i58bf7b'),):
        if not (PERFILES / archivo).is_file():
            pytest.skip(f'{archivo} no esta en este arbol')
        vista = demo._load_camera_profile(camera_id)
        esperado = json.loads((PERFILES / archivo).read_text(encoding='utf-8'))
        assert float(vista.DEMO_MIN_BLUR_VAR) == float(
            esperado['DEMO_MIN_BLUR_VAR'])
        assert int(vista.DEMO_MIN_PERSON_BBOX_H) == int(
            esperado['DEMO_MIN_PERSON_BBOX_H'])
