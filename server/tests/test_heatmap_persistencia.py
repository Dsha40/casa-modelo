"""
Pruebas del heatmap persistente e historico (1-ago-2026).

Lo pedido: los mapas de calor se generan para TODAS las camaras, PERMANECEN
aunque la camara se cierre o el servidor se reinicie, y conservan un
historico por horas. Aqui se prueba la mecanica sin GPU ni pipelines:
el acumulador con estado, el registro global y la API del historico.

Reloj (4-sep-2026): el acumulador sella la hora con el reloj de la maquina
(`HeatmapAccumulator._current_stamp` -> `time.strftime('%Y-%m-%d_%H')`).
Las pruebas que dependen de ese sello congelan el reloj en un instante
FIJO (fixture `reloj_congelado`) en vez de leer la hora real: la version
anterior comparaba dos lecturas del reloj separadas por la escritura del
historico y fallaba si la corrida caia en los ultimos milisegundos de una
hora (reproducido a las 10:59:59.99).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))
sys.path.insert(0, str(RAIZ / 'packages' / 'elde_core'))

from fastapi import HTTPException  # noqa: E402

from src.analityc.core.analytics.config import AnalyticsConfig  # noqa: E402
from src.analityc.core.analytics.heatmap import HeatmapAccumulator  # noqa: E402
from src.analityc.core.analytics import heatmap_registro as reg  # noqa: E402
from src.app import api_lectura as api  # noqa: E402

_FRAME = np.zeros((360, 640, 3), np.uint8)

# Un martes a mediodia, hora local: lejos de cualquier frontera de hora,
# de dia o de anio. Las pruebas de frontera fijan sus propios instantes.
_MARTES_MEDIODIA = '2026-09-08 12:00'


@pytest.fixture
def reloj_congelado(monkeypatch):
    """Congela el reloj del modulo `time` en un instante local fijo.

    Devuelve `congelar(fecha_hora)`: recibe 'YYYY-MM-DD HH:MM' en hora LOCAL
    (via `time.mktime`, para que el sello de hora sea el que la prueba
    presupone en cualquier zona horaria) y devuelve la epoca fijada.

    Se parchean TRES puntos y no solo `time.time` (4-sep-2026): el sello de
    hora del acumulador es `time.strftime('%Y-%m-%d_%H')` SIN tupla, y en
    CPython esa llamada (igual que `time.localtime()` sin argumento) lee el
    reloj del sistema directamente en C, sin pasar por `time.time()`.
    Congelar solo `time.time` deja el sello leyendo la hora real. Con
    tupla/argumento explicito las funciones pasan tal cual, para no alterar
    a logging, zipfile (np.savez) ni a la propia prueba cuando formatea
    instantes que ya calculo.

    Se congela el reloj y NO se stubbea `_current_stamp`: la prueba debe
    ejercitar la lectura real del reloj que hace el codigo de produccion.
    """
    localtime_real = time.localtime
    strftime_real = time.strftime

    def congelar(fecha_hora: str) -> float:
        instante = time.mktime(time.strptime(fecha_hora, '%Y-%m-%d %H:%M'))

        def localtime_fijo(segundos=None):
            return localtime_real(instante if segundos is None else segundos)

        def strftime_fijo(formato, tupla=None):
            return strftime_real(
                formato, localtime_fijo() if tupla is None else tupla)

        monkeypatch.setattr(time, 'time', lambda: instante)
        monkeypatch.setattr(time, 'localtime', localtime_fijo)
        monkeypatch.setattr(time, 'strftime', strftime_fijo)
        return instante

    return congelar


def _estampar(acc: HeatmapAccumulator, veces: int = 10) -> None:
    for _ in range(veces):
        acc.add_person((100, 100, 140, 200), 640, 360)


def test_el_acumulado_sobrevive_a_un_reinicio(reloj_congelado):
    # Reloj fijo: los dos acumuladores nacen en la MISMA hora, asi que la
    # comparacion de grillas no depende de que la corrida cruce una hora.
    reloj_congelado(_MARTES_MEDIODIA)
    with tempfile.TemporaryDirectory() as tmp:
        acc = HeatmapAccumulator()
        _estampar(acc, 10)
        assert acc.guardar_estado('cam-x', tmp)

        renacido = HeatmapAccumulator()
        assert renacido.cargar_estado('cam-x', tmp)
        assert renacido._samples == 10
        assert np.allclose(renacido._total, acc._total)
        # Y sigue acumulando ENCIMA, no desde cero.
        _estampar(renacido, 1)
        assert renacido._samples == 11


def test_flush_escribe_json_y_estado_sin_frame_a_mano(reloj_congelado):
    """Desde el 5-oct-2026 el mapa vigente NO deja PNG: se pinta bajo pedido
    (`heatmap_bajo_pedido.py`). El volcado deja el JSON (con el que se lista
    la camara) y el estado; el PNG solo con HEATMAP_PNG_PERIODICO."""
    reloj_congelado(_MARTES_MEDIODIA)
    with tempfile.TemporaryDirectory() as tmp:
        acc = HeatmapAccumulator()
        _estampar(acc)
        assert acc.flush('cam-y', out_dir=tmp)
        assert not (Path(tmp) / 'cam-y.png').exists()
        assert (Path(tmp) / 'cam-y.json').is_file()
        assert (Path(tmp) / 'state' / 'cam-y.npz').is_file()
        # Dentro de la misma hora el flush NO cierra ninguna hora: el
        # historico solo se escribe cuando cambia el sello (con el reloj
        # real, una corrida que cruzara la hora dejaba aqui un history/
        # que nadie afirmaba).
        assert not (Path(tmp) / 'history').exists()
        # Sin muestras no se escribe nada (no ensuciar el disco con vacios).
        assert not HeatmapAccumulator().flush('cam-vacia', out_dir=tmp)
        assert not (Path(tmp) / 'cam-vacia.png').exists()


@pytest.mark.parametrize(
    'ahora, hora_cerrada, hora_vigente',
    [
        # Caso corriente: la hora anterior del mismo dia.
        ('2026-09-08 12:00', '2026-09-08_11', '2026-09-08_12'),
        # Madrugada: la hora interrumpida es la ultima de la VISPERA y el
        # historico debe llevar la fecha de la vispera, no la de hoy.
        ('2026-09-08 00:30', '2026-09-07_23', '2026-09-08_00'),
        # Cambio de anio: la hora cerrada pertenece al anio anterior.
        ('2027-01-01 00:30', '2026-12-31_23', '2027-01-01_00'),
    ],
    ids=['martes-mediodia', 'madrugada-vispera', 'anio-nuevo'])
def test_la_hora_interrumpida_se_cierra_al_volver(
        reloj_congelado, ahora, hora_cerrada, hora_vigente):
    """Camara cerrada a mitad de hora: al volver, esa hora parcial se guarda
    en el historico en vez de perderse. OJO: el sello debe caer DENTRO de la
    retencion (90 dias para .npz/.json) — con un sello del 2020 la retencion
    los borra nada mas escribirlos, que es lo correcto.

    El reloj va congelado en `ahora` y `hora_cerrada` es la hora anterior a
    ese instante, escrita a mano para que las fronteras de dia y de anio
    queden documentadas en vez de depender de a que hora corre la suite."""
    reloj_congelado(ahora)
    with tempfile.TemporaryDirectory() as tmp:
        acc = HeatmapAccumulator()
        assert acc._hour_stamp == hora_vigente     # el reloj esta congelado
        _estampar(acc, 5)
        acc._hour_stamp = hora_cerrada          # hora anterior, quedo a medias
        acc.guardar_estado('cam-z', tmp)

        renacido = HeatmapAccumulator()
        assert renacido.cargar_estado('cam-z', tmp)
        assert renacido._hour_stamp == hora_cerrada
        renacido.maybe_save_snapshot('cam-z', background=_FRAME,
                                     out_dir=tmp, every_s=0.0)
        hist = Path(tmp) / 'history' / 'cam-z'
        assert (hist / f'{hora_cerrada}.png').is_file()
        assert (hist / f'{hora_cerrada}.npz').is_file()
        with open(hist / f'{hora_cerrada}.json', encoding='utf-8') as f:
            meta = json.load(f)
        assert meta['muestras'] == 5
        assert meta['stamp'] == hora_cerrada
        # El buffer horario quedo vaciado y en la hora vigente.
        assert renacido._hour_samples == 0
        assert renacido._hour_stamp == hora_vigente


def test_el_registro_comparte_acumulador_y_no_cuenta_doble():
    original = AnalyticsConfig.HEATMAP_DIR
    with tempfile.TemporaryDirectory() as tmp:
        AnalyticsConfig.HEATMAP_DIR = tmp
        try:
            a1 = reg.obtener('cam-reg-1', 'Pasillo 1')
            assert reg.obtener('cam-reg-1') is a1

            n = reg.acumular_desde_metadata(
                'cam-reg-1', 'Pasillo 1', _FRAME,
                {'detections': [{'box': [10, 10, 50, 90]}],
                 'tracks': [{'bbox': [60, 10, 100, 90]}]})
            assert n == 2 and a1._samples == 2

            # Metadata con clave `heatmap` = el pipeline YA acumulo por
            # dentro (TiendasProcess): estampar aqui seria contar doble.
            n2 = reg.acumular_desde_metadata(
                'cam-reg-1', None, _FRAME,
                {'heatmap': {'muestras': 9},
                 'detections': [{'box': [10, 10, 50, 90]}]})
            assert n2 == 0 and a1._samples == 2

            # Metadata sin cajas o roto: no pasa nada.
            assert reg.acumular_desde_metadata('cam-reg-1', None, _FRAME, {}) == 0
            assert reg.acumular_desde_metadata('cam-reg-1', None, _FRAME, None) == 0
            assert reg.acumular_desde_metadata('cam-reg-1', None, None, {}) == 0
        finally:
            AnalyticsConfig.HEATMAP_DIR = original


def test_volcar_todos_escribe_cada_camara():
    original = AnalyticsConfig.HEATMAP_DIR
    with tempfile.TemporaryDirectory() as tmp:
        AnalyticsConfig.HEATMAP_DIR = tmp
        try:
            reg.acumular_desde_metadata(
                'cam-volcado', 'Volcada', _FRAME,
                {'detections': [{'box': [10, 10, 50, 90]}]})
            assert reg.volcar_todos() >= 1
            # Sin PNG desde el 5-oct-2026 (bajo pedido): JSON + estado.
            assert (Path(tmp) / 'cam-volcado.json').is_file()
            assert (Path(tmp) / 'state' / 'cam-volcado.npz').is_file()
        finally:
            AnalyticsConfig.HEATMAP_DIR = original


def test_con_el_interruptor_vuelve_el_png_periodico(reloj_congelado,
                                                     monkeypatch):
    """HEATMAP_PNG_PERIODICO=True = como antes del 5-oct-2026, por si algo
    de fuera del repositorio lee el PNG del disco."""
    reloj_congelado(_MARTES_MEDIODIA)
    monkeypatch.setattr(AnalyticsConfig, 'HEATMAP_PNG_PERIODICO', True)
    with tempfile.TemporaryDirectory() as tmp:
        acc = HeatmapAccumulator()
        _estampar(acc)
        assert acc.flush('cam-png', out_dir=tmp)
        assert (Path(tmp) / 'cam-png.png').is_file()


def test_api_historico_lista_y_sirve_saneado():
    salida_real = api._salida
    with tempfile.TemporaryDirectory() as tmp:
        hist = Path(tmp) / 'heatmap' / 'history' / 'dev-1'
        hist.mkdir(parents=True)
        (hist / '2026-08-01_10.png').write_bytes(b'\x89PNGfake')
        (hist / '2026-08-01_10.json').write_text(
            json.dumps({'muestras': 42, 'zonas_calientes': []}),
            encoding='utf-8')
        (hist / '2026-08-01_09.png').write_bytes(b'\x89PNGfake')
        api._salida = lambda: Path(tmp)
        try:
            fuera = api.historico_de_heatmap('dev-1')
            assert fuera['total'] == 2
            assert fuera['horas'][0]['stamp'] == '2026-08-01_10'  # reciente 1o
            assert fuera['horas'][0]['muestras'] == 42
            assert fuera['horas'][1]['muestras'] is None          # sin json

            servida = api.foto_de_historico('dev-1', '2026-08-01_10')
            assert Path(servida.path).resolve().parent == hist.resolve()
            for malo in (('dev-1', '../../secreto'), ('../dev-1', 'x'),
                         ('dev-1', 'no-existe')):
                try:
                    api.foto_de_historico(*malo)
                    assert False, f'{malo} debio dar 404'
                except HTTPException as e:
                    assert e.status_code == 404
            assert api.historico_de_heatmap('sin-historia')['total'] == 0
        finally:
            api._salida = salida_real
