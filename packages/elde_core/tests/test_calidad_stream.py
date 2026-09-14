"""Selector de calidad del stream DVR: principal (main) o secundario (sub).

Todo DVR publica dos flujos del mismo canal: el PRINCIPAL a resolución nativa
(más píxeles por persona = mejor detección) y el SECUNDARIO reducido (mucho
menos ancho de banda y CPU). `rtsp_sub` ya lo calculaban las estrategias y
viajaba hasta el recuadro, pero `start_dvr_stream` cogía SIEMPRE el principal:
la opción no existía (8-ago-2026).

Se elige POR RECUADRO: en la misma ventana puede convivir la cámara del área
que importa en principal con las de contexto en secundario.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elde_core.ui.render_box_captura import CapturaDVRMixin  # noqa: E402

CANAL = {"rtsp_main": "rtsp://u:p@10.0.0.5:554/Streaming/Channels/101",
         "rtsp_sub":  "rtsp://u:p@10.0.0.5:554/Streaming/Channels/102"}


class _Recuadro(CapturaDVRMixin):
    """Lo mínimo del widget que el mixin toca para esta funcionalidad."""
    def __init__(self):
        self.index = 0
        self.guardado = {}

    def _save_all(self, clave, valor):
        self.guardado[clave] = valor

    def _pintar_boton_calidad(self):
        pass                            # sin Qt en las pruebas

    def start_dvr_stream(self, channel_data):
        self.reabierto_con = channel_data


# ── Elección de URL ──────────────────────────────────────────────────────
def test_por_defecto_usa_el_stream_principal():
    r = _Recuadro()
    assert r.calidad_stream() == "main"
    assert r.url_del_canal(CANAL) == CANAL["rtsp_main"]


def test_en_secundario_usa_la_url_sub():
    r = _Recuadro()
    r._usar_substream = True
    assert r.calidad_stream() == "sub"
    assert r.url_del_canal(CANAL) == CANAL["rtsp_sub"]


def test_sin_sub_publicado_cae_al_principal():
    """Mejor verla en alta que no verla: el respaldo es explícito."""
    r = _Recuadro()
    r._usar_substream = True
    assert r.url_del_canal({"rtsp_main": CANAL["rtsp_main"], "rtsp_sub": ""}) \
        == CANAL["rtsp_main"]


def test_canal_sin_urls_no_revienta():
    r = _Recuadro()
    assert r.url_del_canal({}) == ""


# ── Alternar ─────────────────────────────────────────────────────────────
def test_alternar_cambia_persiste_y_reabre_el_stream():
    r = _Recuadro()
    r._dvr_channel_data = dict(CANAL)

    r.alternar_calidad_stream()
    assert r.calidad_stream() == "sub"
    assert r.guardado["usar_substream"] is True
    # Reabrir es obligatorio: son dos URL distintas del equipo.
    assert r.reabierto_con == CANAL

    r.alternar_calidad_stream()
    assert r.calidad_stream() == "main"
    assert r.guardado["usar_substream"] is False


def test_alternar_sin_canal_activo_no_reabre_nada():
    r = _Recuadro()
    r.alternar_calidad_stream()
    assert r.calidad_stream() == "sub"
    assert not hasattr(r, "reabierto_con")


def test_un_cliente_sin_persistencia_no_rompe():
    """`_save_all` puede no existir o fallar; la opción debe cambiar igual."""
    class _SinGuardado(_Recuadro):
        def _save_all(self, clave, valor):
            raise RuntimeError("este cliente no persiste")

    r = _SinGuardado()
    r.alternar_calidad_stream()
    assert r.calidad_stream() == "sub"


# ── Contrato con start_dvr_stream ────────────────────────────────────────
def test_start_dvr_stream_usa_el_selector_y_recuerda_el_canal():
    import inspect
    fuente = inspect.getsource(CapturaDVRMixin.start_dvr_stream)
    assert "self.url_del_canal(channel_data)" in fuente, (
        "debe elegir la URL por calidad, no coger rtsp_main a pelo")
    assert 'rtsp_url = channel_data.get("rtsp_main"' not in fuente
    assert "self._dvr_channel_data = dict(channel_data)" in fuente, (
        "hay que recordar el canal para poder reabrirlo al cambiar calidad")


if __name__ == '__main__':
    fallos = 0
    for _n, _f in sorted(globals().items()):
        if not _n.startswith('test_') or not callable(_f):
            continue
        try:
            _f()
            print(f"  OK    {_n}")
        except Exception as _e:                            # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_n}: {type(_e).__name__}: {_e}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
