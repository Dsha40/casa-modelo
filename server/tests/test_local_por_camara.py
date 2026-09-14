"""A qué local pertenece una foto vieja, cuando el sidecar no lo dice.

## El fallo (10-sep-2026)

El sidecar de una alerta guarda el NOMBRE de la cámara, no su identificador. Y
el iVMS-4200 titula igual todas sus ventanas: en el registro del perimetral
hay seis dispositivos llamados «iVMS-4200» repartidos en CUATRO locales.

    win-iVMS-4200     iVMS-4200   Cámaras CY Aruba
    win-iVMS-4200-c2  iVMS-4200   Prolicor
    win-iVMS-4200-c3  iVMS-4200   Galpon 21
    win-iVMS-4200-c4  iVMS-4200   Ávila

El mapa se construía a secas, así que ganaba el último que se leyera y **1.467
fotos de local desconocido salían estampadas como «Cámaras CY Aruba»**,
mientras la pestaña de Prolicor enseñaba una sola foto. Peor: al registrar una
cámara nueva, esas 1.467 cambiaban de dueño solas, sin que nadie tocara nada.

Lo que el operador veía era «se me mezclan las fotos de los clientes». No era
la galería ni el Re-ID: era esto.

La regla ahora es que **un nombre repetido no imputa local**. Esas fotos salen
en «(sin local)», que es la verdad, en vez de una mentira que parece un dato.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app import api_lectura as al  # noqa: E402

#: El registro real del perimetral, tal cual estaba el 10-sep-2026.
REGISTRO_REAL = [
    {'device_id': 'win-iVMS-4200', 'camera_name': 'iVMS-4200',
     'establecimiento': 'Cámaras CY Aruba'},
    {'device_id': 'win-iVMS-4200-c2', 'camera_name': 'iVMS-4200',
     'establecimiento': 'Prolicor'},
    {'device_id': 'win-iVMS-4200-c3', 'camera_name': 'iVMS-4200',
     'establecimiento': 'Galpon 21'},
    {'device_id': 'win-iVMS-4200-c4', 'camera_name': 'iVMS-4200',
     'establecimiento': 'Ávila'},
    {'device_id': 'CAM-PACIENTE', 'camera_name': 'CAM-PACIENTE',
     'establecimiento': 'Ávila'},
]


def _con_registro(monkeypatch, dispositivos):
    monkeypatch.setattr(al, '_registro',
                        type('R', (), {'dispositivos': staticmethod(
                            lambda: list(dispositivos))})())
    # El aviso es una vez por arranque; se rearma para cada prueba.
    monkeypatch.setattr(al._locales_por_camara, '_avisado', False,
                        raising=False)


# ── La regla ─────────────────────────────────────────────────────────────

def test_un_nombre_repetido_no_imputa_local(monkeypatch):
    """El caso real: cuatro locales bajo el mismo nombre."""
    _con_registro(monkeypatch, REGISTRO_REAL)
    mapa = al._locales_por_camara()
    assert 'iVMS-4200' not in mapa, (
        'un nombre en cuatro locales no puede imputar ninguno: la version '
        'anterior devolvia el ultimo leido y estampaba 1.467 fotos mal')


def test_un_nombre_unico_si_imputa(monkeypatch):
    """La regla no puede volverse tan estricta que deje de servir."""
    _con_registro(monkeypatch, REGISTRO_REAL)
    assert al._locales_por_camara().get('CAM-PACIENTE') == 'Ávila'


def test_el_mismo_local_repetido_no_es_ambiguo(monkeypatch):
    """Tres ventanas del mismo local (el caso de tienda) sí imputan: no hay
    nada que decidir, todas son del mismo sitio."""
    _con_registro(monkeypatch, [
        {'camera_name': 'iVMS-4200', 'establecimiento': 'Automercados-IA'},
        {'camera_name': 'iVMS-4200', 'establecimiento': 'Automercados-IA'},
    ])
    assert al._locales_por_camara().get('iVMS-4200') == 'Automercados-IA'


def test_una_camara_sin_local_no_estorba(monkeypatch):
    """Una cámara sin local asignado no puede volver ambigua a las demás."""
    _con_registro(monkeypatch, [
        {'camera_name': 'iVMS-4200', 'establecimiento': 'Prolicor'},
        {'camera_name': 'iVMS-4200', 'establecimiento': ''},
    ])
    assert al._locales_por_camara().get('iVMS-4200') == 'Prolicor'


# ── Que no sea mudo ──────────────────────────────────────────────────────

def test_la_ambiguedad_se_avisa_con_los_locales_implicados(monkeypatch, caplog):
    """La solución de fondo la tiene el operador —renombrar las ventanas— y
    no puede tomarla si nadie le dice que hay un choque."""
    _con_registro(monkeypatch, REGISTRO_REAL)
    with caplog.at_level(logging.WARNING, logger='src.app.api_lectura'):
        al._locales_por_camara()
    texto = ' '.join(r.getMessage() for r in caplog.records)
    assert 'iVMS-4200' in texto
    for local in ('Prolicor', 'Galpon 21', 'Ávila'):
        assert local in texto, f'el aviso no nombra «{local}»'


def test_el_aviso_no_se_repite_en_cada_peticion(monkeypatch, caplog):
    """`/api/v1/alertas` llama a esto en cada consulta: un aviso por llamada
    llenaría el log y dejaría de leerse."""
    _con_registro(monkeypatch, REGISTRO_REAL)
    with caplog.at_level(logging.WARNING, logger='src.app.api_lectura'):
        for _ in range(5):
            al._locales_por_camara()
    assert len([r for r in caplog.records if 'iVMS-4200' in r.getMessage()]) == 1


# ── Lo que ve el dashboard ───────────────────────────────────────────────

def test_el_sidecar_manda_sobre_el_registro(monkeypatch):
    """Si la foto guardó su local, ese es el bueno: es lo que se vio en su
    momento, no lo que la cámara sea hoy."""
    _con_registro(monkeypatch, REGISTRO_REAL)
    mapa = al._locales_por_camara()
    fila = {'camara': 'iVMS-4200', 'establecimiento': 'Prolicor'}
    assert al._local_de(fila, mapa) == 'Prolicor'


def test_sin_sidecar_y_con_nombre_ambiguo_queda_sin_local(monkeypatch):
    """El caso de las 1.467: antes decían «Cámaras CY Aruba»."""
    _con_registro(monkeypatch, REGISTRO_REAL)
    mapa = al._locales_por_camara()
    assert al._local_de({'camara': 'iVMS-4200'}, mapa) == ''


def test_un_registro_vacio_no_revienta(monkeypatch):
    _con_registro(monkeypatch, [])
    assert al._locales_por_camara() == {}
