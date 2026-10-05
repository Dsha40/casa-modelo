"""
El escáner de ventanas no puede ofrecer como fuente de captura las ventanas
del PROPIO cliente (28-sep-2026).

Cada menú desplegable de Qt es una ventana de nivel superior con título. El
escáner las tomaba por ventanas nuevas y su miniatura les hacía
SetWindowPos + BringWindowToTop + parpadeo: «Productos del anaquel > Área >
Añadir producto…» cerraba los menús sin abrir nunca el diálogo del nombre.

Usa ventanas REALES de Windows (no hay otra forma de probar EnumWindows), así
que solo corre en Windows con escritorio. La ventana de prueba se abre fuera
de la pantalla.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.platform != 'win32':
    pytest.skip('EnumWindows solo existe en Windows', allow_module_level=True)

import win32process  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

# Solo el detector: importar `window_monitor` arrancaría su hilo singleton.
from elde_core.capture.windows_detector import (  # noqa: E402
    WindowScannerThread, es_del_propio_proceso)


def _ventana_propia():
    app = QApplication.instance() or QApplication([])
    # La QApplication es una por proceso. Si otra prueba de la misma pasada
    # la creó antes con QT_QPA_PLATFORM=offscreen (test_flecha_direccion y
    # otras cuatro), Qt no crea ventanas de Windows: winId() vale 1 y la
    # prueba fallaba sin que el escáner tuviera nada que ver (2-oct-2026,
    # suite completa de elde_core). Sola sí corre: ahí es donde vale.
    if app.platformName() == 'offscreen':
        pytest.skip('Qt va sin pantalla en esta pasada: no hay ventanas reales')
    w = QWidget()
    w.setWindowTitle('python')          # el título que tienen los menús de Qt
    w.setGeometry(-3000, -3000, 120, 80)
    w.show()
    app.processEvents()
    return app, w


def test_el_escaner_ignora_las_ventanas_del_propio_cliente():
    app, w = _ventana_propia()
    try:
        hwnd = int(w.winId())
        assert es_del_propio_proceso(hwnd)
        escaner = WindowScannerThread(ignore_hwnds=[])
        escaner._check_window_changes()
        assert hwnd not in escaner.current_windows, (
            'el escaner ofrece como fuente una ventana del propio cliente')
    finally:
        w.close()
        app.processEvents()


def test_sigue_viendo_las_ventanas_de_otros_programas():
    """El filtro no puede comerse iVMS: solo descarta NUESTRO proceso."""
    escaner = WindowScannerThread(ignore_hwnds=[])
    escaner._check_window_changes()
    ajenas = [h for h in escaner.current_windows
              if win32process.GetWindowThreadProcessId(h)[1] != os.getpid()]
    if not ajenas:
        pytest.skip('no hay ninguna ventana de otro programa abierta')
    assert all(not es_del_propio_proceso(h) for h in ajenas)
