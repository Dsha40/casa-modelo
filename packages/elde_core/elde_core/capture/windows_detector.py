from PySide6.QtCore import Signal, QThread

import win32gui
import win32con
import win32process
import pythoncom
import ctypes
from ctypes import wintypes
import os
import time
import psutil


def es_del_propio_proceso(hwnd) -> bool:
    """¿La ventana es de ESTE cliente (sus menús, diálogos, la principal)?

    La lista `ignore_hwnds` se arma al importar `window_monitor`, cuando el
    cliente todavía no tiene ninguna ventana: en la práctica no ignoraba
    nada. Cada menú desplegable de Qt es una ventana de nivel superior con
    título («python»), así que el escáner la tomaba por una fuente de
    captura y su miniatura le hacía SetWindowPos + BringWindowToTop +
    parpadeo con 0,2 s de sleep en el hilo de la interfaz. Visto el
    28-sep-2026: «Productos del anaquel > Área > Añadir producto…» cerraba
    los menús sin abrir nunca el diálogo del nombre. Filtrar por PID vale
    para cualquier ventana que el cliente abra en el futuro, sin mantener
    una lista."""
    try:
        return win32process.GetWindowThreadProcessId(hwnd)[1] == os.getpid()
    except Exception:                                       # noqa: BLE001
        return False


# Definir la estructura de la función de callback para SetWinEventHook
WINEVENTPROC = ctypes.WINFUNCTYPE(
    None, 
    wintypes.HANDLE, 
    wintypes.DWORD, 
    wintypes.HWND, 
    wintypes.LONG, 
    wintypes.LONG, 
    wintypes.DWORD, 
    wintypes.DWORD
)





class WindowScannerThread(QThread):

    """Hilo que escanea periódicamente las ventanas abiertas."""
    window_opened = Signal(int, str)  # hwnd, title
    window_closed = Signal(int)       # hwnd



    def __init__(self, ignore_hwnds=None, parent=None):
        super().__init__(parent)
        self.running = True
        self.current_windows = {}  # hwnd -> title
        self.ignore_hwnds = ignore_hwnds or []  # Lista de HWNDs a ignorar



    def run(self):
        while self.running:
            self._check_window_changes()
            time.sleep(0.3)  # Escanear cada 300ms



    def _check_window_changes(self):
        new_windows = {}
        
        def enum_windows(hwnd, _):
            # Si este HWND está en la lista de ignorados, saltarlo
            if hwnd in self.ignore_hwnds or es_del_propio_proceso(hwnd):
                return True

            if (win32gui.IsWindowVisible(hwnd) and 
                win32gui.GetParent(hwnd) == 0 and
                win32gui.GetWindowText(hwnd) != ""):
                
                title = win32gui.GetWindowText(hwnd)
                new_windows[hwnd] = title
            return True
        
        win32gui.EnumWindows(enum_windows, None)
        
        # Detectar nuevas ventanas
        for hwnd, title in new_windows.items():
            if hwnd not in self.current_windows:
                self.window_opened.emit(hwnd, title)
        
        # Detectar ventanas cerradas
        for hwnd in list(self.current_windows.keys()):
            if hwnd not in new_windows:
                self.window_closed.emit(hwnd)
        
        self.current_windows = new_windows






    def stop(self, *_ignorado, msec: int = 2000):
        """Saca al hilo del bucle y ESPERA a que salga de verdad.

        Sin el wait() el objeto se destruia mientras run() seguia dentro del
        sleep(0.3): Qt ve un QThread vivo destruyendose, llama a qFatal y
        aborta el proceso entero (0xc0000409, sin traceback de Python). Era la
        causa de que el cliente 'crashease' al cerrarlo.
        quit() no sirve aqui porque run() esta sobrescrito y no hay event loop:
        lo unico que saca al hilo del bucle es running = False.

        `*_ignorado` no es un adorno: este metodo se conecta a la senal
        `QObject.destroyed`, que entrega el objeto destruido como argumento.
        Con la firma `stop(self, msec=2000)`, ese QObject aterrizaba en `msec`
        y `self.wait(QObject)` lanzaba TypeError. `msec` queda como argumento
        de solo palabra clave para que no vuelva a pasar."""
        if not self.isRunning():
            return
        self.running = False
        if not self.wait(msec):
            self.terminate()
            self.wait(500)
