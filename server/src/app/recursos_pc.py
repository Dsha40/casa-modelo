"""src/app/recursos_pc.py — CPU y RAM del PC, por programa (23-oct-2026).

## Por qué

En el PC de la tienda (i5-6500 de 4 hilos, 12 GB) el cuello es el
procesador, que se reparten el servidor, el programa de la tienda, su
capturador, iVMS y, en las pruebas, el reproductor de vídeo. Y la memoria
se ha acabado más de una vez: 420 MB libres el 1-oct y 0,7 GB el 8-oct, con
el programa de la tienda creciendo. La página «En vivo» del dashboard
enseñaba la GPU, pero no esto; el usuario lo pidió el 23-oct.

## Cómo, y lo que cuesta (medido el 23-oct en ese PC)

Un hilo mide cada `MUESTREO_S`:

  * el PC entero (`psutil.cpu_percent` y `virtual_memory`): 0,4 ms;
  * los procesos de cada programa (`cpu_percent` y `memory_info` de los 17
    que había): 11 ms.

Qué proceso es de qué programa se mira cada `BUSCAR_S`, con la lista de
procesos de Windows (`CreateToolhelp32Snapshot`, ~100 ms), y la línea de
órdenes solo de los `python.exe` (1,4 ms). `psutil.process_iter` tardó 27 s
en recorrer los 278 procesos del PC: pide el nombre de uno en uno y los
protegidos lo mandan por el camino lento. Fuera de Windows se usa
`process_iter`, que allí es rápido.

Los programas que corren como Administrador (el de la tienda e iVMS) se
leen igual: comprobado el 23-oct desde el servidor, que no lo es.

## Las cifras

  * La CPU de cada programa va en % del PC ENTERO (100 % = todos los
    hilos), para que sume con el total. «Resto del PC» es la diferencia.
  * La RAM de cada programa es su conjunto de trabajo (`memory_info().rss`,
    el «conjunto de trabajo» del Administrador de tareas). La memoria
    compartida cuenta en cada uno, así que la suma puede pasar de la usada:
    el «resto» se queda en 0, nunca negativo.
  * La última hora, un punto por minuto de reloj (la CPU media del minuto,
    la RAM del final), y la misma línea en `logs/recursos_pc.csv` para el
    resumen del día. No rota: unos 150 KB por día.

Sin psutil (el servidor de Linux puede no tenerlo) no mide nada y la página
dice «sin datos». Nunca lanza: medir no puede costarle nada al servidor.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:                                         # pragma: no cover
    psutil = None

logger = logging.getLogger(__name__)

MUESTREO_S = 5.0
BUSCAR_S = 60.0
HISTORIAL_PUNTOS = 60                   # un punto por minuto: la última hora

#: `server/logs/recursos_pc.csv`, junto a `gpu_temp.csv` del vigilante.
RUTA_CSV = Path(__file__).resolve().parents[2] / 'logs' / 'recursos_pc.csv'

#: Los programas que se siguen, en el orden en que salen en la página.
PROGRAMAS = (
    ('servidor', 'Servidor de IA'),
    ('tienda', 'Programa de la tienda'),
    ('capturador', 'Capturador de ventanas'),
    ('ivms', 'iVMS-4200'),
    ('reproductor', 'Reproductor de vídeo'),
)
REPRODUCTORES = ('vlc.exe', 'microsoft.media.player.exe', 'wmplayer.exe',
                 'video.ui.exe')

CABECERA_CSV = (['hora', 'cpu_pct', 'ram_usada_mb', 'ram_libre_mb']
                + [f'{medida}_{clave}' for clave, _ in PROGRAMAS
                   for medida in ('cpu', 'ram')])

MB = 1024 * 1024


def papel(pid: int, nombre: str, linea: str, mi_pid: int) -> Optional[str]:
    """De qué programa es un proceso, o None si no es de ninguno que se siga.
    `linea` es su línea de órdenes; solo hace falta para los `python`."""
    nombre = (nombre or '').lower()
    if pid == mi_pid:
        return 'servidor'
    if nombre.startswith('ivms-4200'):
        return 'ivms'
    if nombre in REPRODUCTORES:
        return 'reproductor'
    if nombre.startswith('python'):
        linea = (linea or '').replace('\\', '/').lower()
        # En el cliente el archivo se llama `capture_woker.py`, así, sin la r.
        if 'capture_wo' in linea:
            return 'capturador'
        if 'src/main.py' in linea:
            return 'tienda'
    return None


def _listar_windows() -> List[Tuple[int, str]]:
    """(pid, nombre) de todos los procesos, de la lista de Windows."""
    import ctypes
    from ctypes import wintypes as w

    class Entrada(ctypes.Structure):
        _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD),
                    ('th32ProcessID', w.DWORD),
                    ('th32DefaultHeapID', ctypes.c_size_t),
                    ('th32ModuleID', w.DWORD), ('cntThreads', w.DWORD),
                    ('th32ParentProcessID', w.DWORD),
                    ('pcPriClassBase', ctypes.c_long), ('dwFlags', w.DWORD),
                    ('szExeFile', ctypes.c_wchar * 260)]

    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = w.HANDLE
    foto = k.CreateToolhelp32Snapshot(0x2, 0)       # TH32CS_SNAPPROCESS
    if not foto or foto == w.HANDLE(-1).value:
        return []
    try:
        e = Entrada()
        e.dwSize = ctypes.sizeof(Entrada)
        salida = []
        hay = k.Process32FirstW(foto, ctypes.byref(e))
        while hay:
            salida.append((int(e.th32ProcessID), e.szExeFile))
            hay = k.Process32NextW(foto, ctypes.byref(e))
        return salida
    finally:
        k.CloseHandle(foto)


def _listar() -> List[Tuple[int, str]]:
    if os.name == 'nt':
        return _listar_windows()
    return [(p.pid, p.info.get('name') or '')
            for p in psutil.process_iter(['name'])]


def construir(t: float, cpu_pc: float, ram_total: int, ram_libre: int,
              por_papel: Dict[str, Dict[str, Any]],
              nucleos: int) -> Dict[str, Any]:
    """Una muestra: el PC entero y cada programa. `por_papel` trae la CPU
    sumada de sus procesos como la da psutil (100 = un hilo) y su RAM en
    bytes."""
    nucleos = max(1, int(nucleos or 1))
    programas = []
    for clave, nombre in PROGRAMAS:
        d = por_papel.get(clave) or {}
        programas.append({
            'clave': clave, 'nombre': nombre,
            'cpu_pct': round(min(100.0, float(d.get('cpu', 0.0)) / nucleos), 1),
            'ram_mb': round(int(d.get('ram', 0)) / MB),
            'procesos': int(d.get('procesos', 0))})
    usada_mb = round((ram_total - ram_libre) / MB)
    programas.append({
        'clave': 'resto', 'nombre': 'Resto del PC',
        'cpu_pct': round(max(0.0, cpu_pc - sum(p['cpu_pct'] for p in programas)), 1),
        'ram_mb': max(0, usada_mb - sum(p['ram_mb'] for p in programas)),
        'procesos': None})
    return {'t': round(t, 1), 'cpu_pct': round(float(cpu_pc), 1),
            'nucleos': nucleos, 'ram_total_mb': round(ram_total / MB),
            'ram_usada_mb': usada_mb, 'ram_libre_mb': round(ram_libre / MB),
            'programas': programas}


def resumir(muestras: List[Dict[str, Any]], t: float) -> Dict[str, Any]:
    """El punto de un minuto: la CPU media y la RAM de la última muestra."""
    ultima = muestras[-1]
    punto: Dict[str, Any] = {
        't': t,
        'cpu_pct': round(sum(m['cpu_pct'] for m in muestras) / len(muestras), 1),
        'ram_usada_mb': ultima['ram_usada_mb'],
        'ram_libre_mb': ultima['ram_libre_mb']}
    for i, (clave, _) in enumerate(PROGRAMAS):
        punto[f'cpu_{clave}'] = round(
            sum(m['programas'][i]['cpu_pct'] for m in muestras) / len(muestras), 1)
        punto[f'ram_{clave}'] = ultima['programas'][i]['ram_mb']
    return punto


class Medidor:
    """El estado del muestreo. Lo usa un solo hilo (`_bucle`); la página lee
    la última muestra y el historial con `resumen`."""

    def __init__(self, ruta_csv: Path = RUTA_CSV) -> None:
        self.ruta_csv = Path(ruta_csv)
        self._procesos: Dict[int, Tuple[Any, str]] = {}
        self._buscado: Optional[float] = None
        self._ultima: Optional[Dict[str, Any]] = None
        self._historial: Deque[Dict[str, Any]] = deque(maxlen=HISTORIAL_PUNTOS)
        self._minuto: Optional[int] = None
        self._del_minuto: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._nucleos = psutil.cpu_count() or 1
        psutil.cpu_percent(None)        # la primera llamada solo arranca la cuenta

    def buscar(self) -> None:
        """Qué procesos son de cada programa. Los que ya se seguían conservan
        su objeto: `cpu_percent` cuenta desde su llamada anterior."""
        mi_pid = os.getpid()
        vistos = set()
        for pid, nombre in _listar():
            vistos.add(pid)
            if pid in self._procesos:
                continue
            bajo = (nombre or '').lower()
            if not (pid == mi_pid or bajo.startswith(('python', 'ivms-4200'))
                    or bajo in REPRODUCTORES):
                continue
            try:
                proc = psutil.Process(pid)
                linea = ' '.join(proc.cmdline()) if bajo.startswith('python') else ''
                clave = papel(pid, nombre, linea, mi_pid)
                if clave is None:
                    continue
                proc.cpu_percent(None)
                self._procesos[pid] = (proc, clave)
            except Exception:                               # noqa: BLE001
                continue                    # se cerró, o no deja leerse
        for pid in list(self._procesos):
            if pid not in vistos:
                del self._procesos[pid]

    def muestrear(self, t: Optional[float] = None) -> Dict[str, Any]:
        ahora = time.monotonic()
        if self._buscado is None or (ahora - self._buscado) >= BUSCAR_S:
            self.buscar()
            self._buscado = ahora
        cpu_pc = psutil.cpu_percent(None)
        memoria = psutil.virtual_memory()
        por_papel: Dict[str, Dict[str, Any]] = {}
        for pid, (proc, clave) in list(self._procesos.items()):
            try:
                with proc.oneshot():
                    cpu = proc.cpu_percent(None)
                    ram = proc.memory_info().rss
            except psutil.NoSuchProcess:
                self._procesos.pop(pid, None)
                continue
            except Exception:                               # noqa: BLE001
                continue
            d = por_papel.setdefault(clave, {'cpu': 0.0, 'ram': 0, 'procesos': 0})
            d['cpu'] += cpu
            d['ram'] += ram
            d['procesos'] += 1
        muestra = construir(time.time() if t is None else t, cpu_pc,
                            memoria.total, memoria.available, por_papel,
                            self._nucleos)
        self.anotar(muestra)
        return muestra

    def anotar(self, muestra: Dict[str, Any]) -> None:
        """Guarda la muestra y, al cambiar el minuto de reloj, cierra el
        punto del minuto anterior (historial y una línea del CSV)."""
        minuto = int(muestra['t'] // 60)
        punto = None
        with self._lock:
            self._ultima = muestra
            if self._minuto is not None and minuto != self._minuto and self._del_minuto:
                punto = resumir(self._del_minuto, self._minuto * 60.0)
                self._historial.append(punto)
                self._del_minuto = []
            self._minuto = minuto
            self._del_minuto.append(muestra)
        if punto is not None:
            self._al_csv(punto)

    def _al_csv(self, punto: Dict[str, Any]) -> None:
        try:
            nuevo = not self.ruta_csv.exists()
            self.ruta_csv.parent.mkdir(parents=True, exist_ok=True)
            hora = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(punto['t']))
            fila = [hora] + [str(punto[c]) for c in CABECERA_CSV[1:]]
            with open(self.ruta_csv, 'a', encoding='utf-8') as fh:
                if nuevo:
                    fh.write(','.join(CABECERA_CSV) + '\n')
                fh.write(','.join(fila) + '\n')
        except Exception as exc:                            # noqa: BLE001
            logger.debug("recursos_pc: no se escribió el CSV: %s", exc)

    def resumen(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._ultima is None:
                return None
            return {'ultima': self._ultima, 'historial': list(self._historial)}


_medidor: Optional[Medidor] = None
_hilo: Optional[threading.Thread] = None
_arranque = threading.Lock()


def _bucle(medidor: Medidor) -> None:
    while True:
        inicio = time.monotonic()
        try:
            medidor.muestrear()
        except Exception as exc:                            # noqa: BLE001
            logger.debug("recursos_pc: muestra fallida: %s", exc)
        time.sleep(max(0.5, MUESTREO_S - (time.monotonic() - inicio)))


def iniciar() -> None:
    """Arranca el hilo una sola vez (lo llama el arranque del servidor)."""
    global _medidor, _hilo
    if psutil is None:
        logger.info("recursos_pc: sin psutil, no se mide la CPU ni la RAM")
        return
    with _arranque:
        if _hilo is not None and _hilo.is_alive():
            return
        _medidor = Medidor()
        _hilo = threading.Thread(target=_bucle, args=(_medidor,),
                                 name='recursos_pc', daemon=True)
        _hilo.start()
    logger.info("recursos_pc: midiendo CPU y RAM cada %.0f s", MUESTREO_S)


def resumen() -> Optional[Dict[str, Any]]:
    """La última muestra y la última hora, o None si aún no se midió."""
    medidor = _medidor
    return medidor.resumen() if medidor is not None else None
