"""
CPU y RAM del PC, por programa (23-oct-2026): `src/app/recursos_pc.py`.

Lo que se fija:

  - **cada proceso va a su programa**: el servidor por su pid, el programa
    de la tienda y su capturador por su línea de órdenes (el archivo del
    cliente se llama `capture_woker.py`, así), iVMS y el reproductor por su
    nombre; cualquier otro `python` no se sigue;
  - **la CPU de cada programa va en % del PC entero**: psutil da 100 por
    hilo, y con 4 hilos un programa que ocupa uno entero es el 25 %. Así
    suman con el total y el «resto» sale bien;
  - **el resto nunca es negativo**: la memoria compartida cuenta en cada
    programa y la suma puede pasar de la usada;
  - **un punto por minuto de reloj**, con la CPU media del minuto, y la
    misma línea en el CSV, con su cabecera la primera vez;
  - **un proceso que se cierra se deja de seguir**, sin romper la muestra;
  - **sin psutil no se mide nada**, y no lanza.

Los procesos y psutil son de mentira: ninguna prueba mide este PC.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.app import recursos_pc as rp  # noqa: E402

MI_PID = 500
GB = 1024 ** 3


def _muestra(t, cpu_pc, servidor_cpu=0.0, ram_libre=4 * GB):
    return rp.construir(t, cpu_pc, 12 * GB, ram_libre,
                        {'servidor': {'cpu': servidor_cpu, 'ram': 3 * GB,
                                      'procesos': 1}}, 4)


# ── De qué programa es cada proceso ──────────────────────────────────────

@pytest.mark.parametrize('pid, nombre, linea, esperado', [
    (MI_PID, 'python.exe', 'iniciar_servidor_headless.py 9000', 'servidor'),
    (7, 'python.exe', r'C:\x\clients\tienda\venv\Scripts\python.exe src\main.py', 'tienda'),
    (8, 'python.exe', r'python.exe src/workers/capture_woker.py 132206', 'capturador'),
    (9, 'iVMS-4200.Video.C.exe', '', 'ivms'),
    (10, 'vlc.exe', '', 'reproductor'),
    (11, 'Microsoft.Media.Player.exe', '', 'reproductor'),
    # El lanzador del venv del servidor no es el servidor: es otro pid.
    (12, 'python.exe', 'iniciar_servidor_headless.py 9000', None),
    (13, 'python.exe', 'python -I resumen_dia.py', None),
    (14, 'chrome.exe', '', None),
])
def test_cada_proceso_va_a_su_programa(pid, nombre, linea, esperado):
    assert rp.papel(pid, nombre, linea, MI_PID) == esperado


# ── Las cifras ───────────────────────────────────────────────────────────

def test_la_cpu_va_en_porcentaje_del_pc_entero():
    """Con 4 hilos, el servidor ocupando hilo y medio (150 según psutil) es
    el 37,5 % del PC; con el PC al 80 %, el resto es 42,5."""
    m = rp.construir(0.0, 80.0, 12 * GB, 4 * GB,
                     {'servidor': {'cpu': 150.0, 'ram': 3 * GB, 'procesos': 1}}, 4)
    programas = {p['clave']: p for p in m['programas']}
    assert programas['servidor']['cpu_pct'] == 37.5
    assert programas['resto']['cpu_pct'] == 42.5
    assert programas['servidor']['ram_mb'] == 3 * 1024
    assert programas['tienda'] == {'clave': 'tienda', 'nombre': 'Programa de la tienda',
                                   'cpu_pct': 0.0, 'ram_mb': 0, 'procesos': 0}
    assert m['ram_usada_mb'] == 8 * 1024 and m['ram_libre_mb'] == 4 * 1024


def test_el_resto_nunca_es_negativo():
    """La memoria compartida cuenta en cada programa: la suma puede pasar de
    la usada, y la CPU de un periodo, del total del PC."""
    m = rp.construir(0.0, 10.0, 12 * GB, 10 * GB,
                     {'servidor': {'cpu': 100.0, 'ram': 3 * GB, 'procesos': 1}}, 4)
    resto = m['programas'][-1]
    assert resto['clave'] == 'resto'
    assert resto['cpu_pct'] == 0.0 and resto['ram_mb'] == 0


def test_un_punto_por_minuto_y_su_linea_en_el_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'psutil', types.SimpleNamespace(
        cpu_count=lambda: 4, cpu_percent=lambda intervalo=None: 0.0))
    csv = tmp_path / 'logs' / 'recursos_pc.csv'
    medidor = rp.Medidor(ruta_csv=csv)
    minuto = 29_800_000 * 60.0
    for i, cpu in enumerate((40.0, 60.0, 80.0)):         # el mismo minuto
        medidor.anotar(_muestra(minuto + 5 * i, cpu, servidor_cpu=100.0))
    assert medidor.resumen()['historial'] == []           # aún no se cerró

    medidor.anotar(_muestra(minuto + 61, 20.0, ram_libre=2 * GB))
    historial = medidor.resumen()['historial']
    assert len(historial) == 1
    punto = historial[0]
    assert punto['t'] == minuto and punto['cpu_pct'] == 60.0
    assert punto['cpu_servidor'] == 25.0 and punto['ram_libre_mb'] == 4 * 1024
    assert medidor.resumen()['ultima']['ram_libre_mb'] == 2 * 1024

    lineas = csv.read_text(encoding='utf-8').splitlines()
    assert lineas[0].split(',') == rp.CABECERA_CSV
    hora = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(minuto))
    assert lineas[1].startswith(f'{hora},60.0,{8 * 1024},{4 * 1024},25.0,3072,')
    assert len(lineas) == 2


# ── El muestreo, con procesos de mentira ─────────────────────────────────

class _Cerrado(Exception):
    pass


class _Proceso:
    vivos = {}

    def __init__(self, pid):
        datos = _Proceso.vivos.get(pid)
        if datos is None:
            raise _Cerrado(pid)
        self.pid = pid
        self._linea, self._cpu, self._ram = datos

    def cmdline(self):
        return self._linea.split()

    def cpu_percent(self, intervalo=None):
        if self.pid not in _Proceso.vivos:
            raise _Cerrado(self.pid)
        return self._cpu

    def memory_info(self):
        return types.SimpleNamespace(rss=self._ram)

    def oneshot(self):
        import contextlib
        return contextlib.nullcontext()


@pytest.fixture
def pc_de_mentira(monkeypatch):
    _Proceso.vivos = {
        MI_PID: ('python -u iniciar_servidor_headless.py 9000', 140.0, 3 * GB),
        20: (r'python.exe src\main.py', 40.0, 350 * 1024 ** 2),
        21: ('python.exe src/workers/capture_woker.py 1', 40.0, 160 * 1024 ** 2),
        22: ('', 8.0, 80 * 1024 ** 2),
        23: ('python -I otro_script.py', 99.0, GB),
    }
    nombres = {MI_PID: 'python.exe', 20: 'python.exe', 21: 'python.exe',
               22: 'iVMS-4200.Video.C.exe', 23: 'python.exe', 24: 'explorer.exe'}
    monkeypatch.setattr(rp, '_listar', lambda: [
        (pid, nombres[pid]) for pid in nombres if pid in _Proceso.vivos or pid == 24])
    monkeypatch.setattr(rp.os, 'getpid', lambda: MI_PID)
    monkeypatch.setattr(rp, 'psutil', types.SimpleNamespace(
        Process=_Proceso, NoSuchProcess=_Cerrado, cpu_count=lambda: 4,
        cpu_percent=lambda intervalo=None: 90.0,
        virtual_memory=lambda: types.SimpleNamespace(total=12 * GB,
                                                     available=2 * GB)))


def test_muestrea_cada_programa(pc_de_mentira, tmp_path):
    medidor = rp.Medidor(ruta_csv=tmp_path / 'r.csv')
    m = medidor.muestrear(t=1000.0)
    programas = {p['clave']: p for p in m['programas']}
    assert programas['servidor']['cpu_pct'] == 35.0
    assert programas['tienda']['cpu_pct'] == 10.0
    assert programas['capturador']['ram_mb'] == 160
    assert programas['ivms']['procesos'] == 1 and programas['ivms']['cpu_pct'] == 2.0
    # El otro python no es de ningún programa: va al resto.
    assert programas['resto']['cpu_pct'] == round(90.0 - 35.0 - 10.0 - 10.0 - 2.0, 1)


def test_un_proceso_que_se_cierra_se_deja_de_seguir(pc_de_mentira, tmp_path):
    medidor = rp.Medidor(ruta_csv=tmp_path / 'r.csv')
    medidor.muestrear(t=1000.0)
    del _Proceso.vivos[21]                       # se cerró el capturador
    m = medidor.muestrear(t=1005.0)
    programas = {p['clave']: p for p in m['programas']}
    assert programas['capturador']['procesos'] == 0
    assert programas['tienda']['procesos'] == 1
    assert 21 not in medidor._procesos


def test_sin_psutil_no_mide_y_no_lanza(monkeypatch):
    monkeypatch.setattr(rp, 'psutil', None)
    monkeypatch.setattr(rp, '_medidor', None)
    monkeypatch.setattr(rp, '_hilo', None)
    rp.iniciar()
    assert rp._hilo is None and rp.resumen() is None
