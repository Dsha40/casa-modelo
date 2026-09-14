"""El umbral de cruce pasa el contrato ESTRICTO, y lo mandan las DOS rutas.

## Por que esta prueba existe

El contrato valida con `extra='forbid'` y el servidor arranca en `estricto`
desde el 31-jul-2026. Eso convierte cualquier clave nueva en un interruptor
de apagado: un campo que el cliente manda y el modelo no declara **rechaza el
100% de los frames** de ese recuadro, no el campo. Ya paso una vez, con
`flecha_direccion` / `linea_cruce` / `area_nombre` sin declarar, y el sintoma
—recuadros a un frame cada diez segundos— no apunta al contrato.

## Y las dos rutas

Hay DOS constructores de payload: el `render_box.py` de cada cliente (stream
DVR) y el mixin de captura por VENTANA que comparten los cuatro (las camaras
de iVMS-4200). Que una mande una clave y la otra no es el fallo del
17-ago-2026, cuando la ruta de ventana no llevaba `establecimiento` y las
alertas decian «camara iVMS-4200» con el local bien elegido.

Hasta el 7-sep-2026 la ruta de ventana no mandaba NI flecha NI linea: una
puerta calibrada sobre una ventana de iVMS no clasificaba un solo paso y
nada lo decia. Aqui se cierra esa puerta.
"""
from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[3]

#: JPEG minimo valido: el validador exige la cabecera FF D8.
_JPEG = bytes.fromhex('ffd8ffe000104a46494600010100000100010000ffd9')

#: (descripcion, archivo) de cada sitio que arma el dict del payload.
RUTAS = [
    ("captura por VENTANA (mixin compartido)",
     RAIZ / "packages/elde_core/elde_core/ui/render_box_captura.py"),
    ("stream DVR del cliente amazonas",
     RAIZ / "clients/amazonas/src/gui/components/render_box/render_box.py"),
    ("stream DVR del cliente detecciones",
     RAIZ / "clients/detecciones/src/gui/components/render_box/render_box.py"),
    ("stream DVR del cliente perimetrales",
     RAIZ / "clients/perimetrales/src/gui/components/render_box/render_box.py"),
    ("stream DVR del cliente tienda",
     RAIZ / "clients/tienda/src/gui/components/render_box/render_box.py"),
]

CLIENTES = ("amazonas", "detecciones", "perimetrales", "tienda")


def _modelo():
    try:
        from elde_core.contracts.eventos import FrameInference
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"contrato no importable: {exc}")
    return FrameInference


# ── El contrato acepta lo que el cliente manda ───────────────────────────
def test_el_umbral_y_su_estilo_pasan_el_contrato():
    """Lo que emite `payload_puerta()`, tal cual, con la puerta encendida."""
    modelo = _modelo()
    frame = modelo(image=_JPEG, camera_id="1",
                   linea_cruce=[[192, 540], [1728, 540]],
                   linea_cruce_activate=True,
                   linea_cruce_estilo="puerta")
    assert frame.linea_cruce_estilo == "puerta"


def test_el_estilo_linea_tambien_pasa():
    """La otra herramienta que dibuja el MISMO umbral."""
    modelo = _modelo()
    frame = modelo(image=_JPEG, camera_id="1",
                   linea_cruce=[[0, 0], [100, 100]],
                   linea_cruce_activate=True,
                   linea_cruce_estilo="linea")
    assert frame.linea_cruce_estilo == "linea"


def test_apagada_manda_vacio_y_tambien_pasa():
    """Es lo que emiten los cuatro clientes en CADA frame sin puerta.

    O sea: el caso comun. Que este rechazara dejaria sin video a todo
    recuadro que no haya dibujado nada.
    """
    modelo = _modelo()
    frame = modelo(image=_JPEG, camera_id="1", linea_cruce=[],
                   linea_cruce_activate=False, linea_cruce_estilo="")
    assert not frame.linea_cruce


def test_un_cliente_viejo_que_no_lo_mande_sigue_valiendo():
    """El campo es opcional: servidor nuevo + cliente viejo es seguro.

    El sentido contrario NO lo es, y por eso el orden de despliegue es
    contrato y servidor primero, clientes despues.
    """
    modelo = _modelo()
    frame = modelo(image=_JPEG, camera_id="1",
                   linea_cruce=[[0, 0], [100, 100]],
                   linea_cruce_activate=True)
    assert frame.linea_cruce_estilo is None


def test_el_umbral_sigue_siendo_un_segmento_de_dos_puntos():
    """La puerta no relaja la regla de la linea: son los mismos dos puntos."""
    modelo = _modelo()
    with pytest.raises(Exception):
        modelo(image=_JPEG, camera_id="1",
               linea_cruce=[[0, 0], [10, 10], [20, 20]],
               linea_cruce_activate=True, linea_cruce_estilo="puerta")


def test_una_clave_no_declarada_seguiria_tumbando_el_frame():
    """Fija POR QUE hay que declarar cada campo nuevo.

    Si un dia alguien relaja `extra='forbid'`, esta prueba se pone roja y
    obliga a decidirlo a proposito en vez de descubrirlo en produccion.
    """
    modelo = _modelo()
    with pytest.raises(Exception):
        modelo(image=_JPEG, camera_id="1", umbral_inventado=[[0, 0], [1, 1]])


# ── Las dos rutas mandan lo mismo ────────────────────────────────────────
@pytest.mark.parametrize("descripcion,archivo", RUTAS,
                         ids=[r[0] for r in RUTAS])
def test_todas_las_rutas_mandan_el_umbral(descripcion, archivo):
    assert archivo.is_file(), f"no existe {archivo}"
    fuente = archivo.read_text(encoding="utf-8")
    manda = ('"linea_cruce"' in fuente          # dict literal (ruta ventana)
             or "payload_puerta(" in fuente)     # el mixin lo arma (clientes)
    assert manda, (
        f"la ruta «{descripcion}» no manda el umbral: una puerta calibrada "
        f"por ahi no clasificaria un solo paso, y nada lo diria")


@pytest.mark.parametrize("cliente", CLIENTES)
def test_cada_cliente_persiste_y_restaura_la_puerta(cliente):
    """Guardar sin restaurar es el fallo clasico: se calibra cada manana."""
    ventana = (RAIZ / "clients" / cliente / "src/gui/windows_main.py"
               ).read_text(encoding="utf-8")
    assert 'cfg.get("puerta_direccional"' in ventana, (
        f"{cliente}: guarda la puerta pero no la restaura")
    assert 'cfg.get("puerta_boolean"' in ventana, (
        f"{cliente}: no restaura el interruptor")


@pytest.mark.parametrize("cliente", CLIENTES)
def test_cada_cliente_hereda_el_mixin_del_nucleo(cliente):
    """Cuatro copias del mismo cableado es como se propaga un fallo.

    El 8-ago-2026 hubo que portar el mismo arreglo del `stop` del recuadro a
    los cuatro `render_box` porque el bug estaba cuatro veces.
    """
    fuente = (RAIZ / "clients" / cliente
              / "src/gui/components/render_box/render_box.py"
              ).read_text(encoding="utf-8")
    assert "PuertaDireccionalMixin" in fuente, (
        f"{cliente}: cableo la puerta a mano en vez de heredar el mixin")


@pytest.mark.parametrize("cliente", CLIENTES)
def test_el_pie_de_cada_cliente_ofrece_el_modo_que_la_lee(cliente):
    """Un utensilio que ningun modo mira es un adorno.

    El umbral solo lo lee `PermanenciaArea`. Si un cliente ofrece el boton
    pero no el modo, el operador dibuja una puerta que no clasifica nada y
    no tiene forma de enterarse.
    """
    pie = (RAIZ / "clients" / cliente
           / "src/gui/components/custom_status_bar.py"
           ).read_text(encoding="utf-8")
    assert "PermanenciaArea" in pie, (
        f"{cliente}: tiene el boton de la puerta pero su pie no ofrece "
        f"PermanenciaArea, el unico modo que lee el umbral")
