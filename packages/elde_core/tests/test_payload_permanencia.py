"""
Las DOS rutas de envio de frames llevan los segundos de permanencia.

Por que una prueba que lee el codigo fuente y no una unitaria: el widget
`Render_box` necesita Qt, un socket y una ventana viva, asi que instanciarlo
en CI no es realista. Lo que se puede fijar barato es lo unico que fallo de
verdad el 17-ago-2026: que una de las dos rutas de payload se quede SIN una
clave que la otra si manda.

Ese dia el mixin de captura por VENTANA no incluia `establecimiento` (solo lo
hacia la ruta por stream DVR), y el sintoma fue que las alertas decian
«cámara iVMS-4200» aunque el operador tuviera su local elegido. Nadie lo vio
hasta que se reviso el mensaje real. Aqui se cierra esa puerta para los
contadores nuevos.
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]

#: (descripcion, archivo) de cada sitio que arma el dict del payload.
RUTAS = [
    ("captura por VENTANA (mixin compartido)",
     RAIZ / "packages/elde_core/elde_core/ui/render_box_captura.py"),
    ("stream DVR del cliente perimetral",
     RAIZ / "clients/perimetrales/src/gui/components/render_box/render_box.py"),
]

CLAVES = ("min_permanencia_personas", "min_permanencia_vehiculos")

#: La ruta por VENTANA y la del cliente que dibuja el umbral tienen que
#: mandar LO MISMO. El 7-sep-2026 no era asi: el mixin de ventana no mandaba
#: `linea_cruce` ni la flecha, asi que una puerta calibrada sobre una ventana
#: de iVMS no clasificaba un solo paso y NADA lo decia (el servidor recibia el
#: umbral vacio y caia al area clasica en silencio). Tercera reincidencia de
#: la misma clase, despues de `establecimiento` (17-ago) y los anaqueles
#: (13-ago) — por eso esto se fija aqui y no en un comentario.
RUTAS_UMBRAL = [
    ("captura por VENTANA (mixin compartido)",
     RAIZ / "packages/elde_core/elde_core/ui/render_box_captura.py"),
    ("stream DVR del cliente de detecciones",
     RAIZ / "clients/detecciones/src/gui/components/render_box/render_box.py"),
]

CLAVES_UMBRAL = ("flecha_direccion", "flecha_direccion_activate",
                 "linea_cruce", "linea_cruce_activate", "area_nombre")

#: Mixin compartido al que las rutas pueden DELEGAR las claves del umbral.
#: Desde el 7-sep-2026 el recuadro no arma las tres claves a mano: llama a
#: `payload_puerta(w, h)`, que vive aqui y las devuelve juntas. Buscarlas solo
#: en el fichero de la ruta daba un falso positivo —«la ruta DVR no manda
#: linea_cruce_activate»— cuando en realidad las manda todas, delegando
#: (8-sep-2026). Se mira el mixin SOLO si la ruta lo llama de verdad.
MIXIN_PUERTA = RAIZ / "packages/elde_core/elde_core/ui/render_box_puerta.py"


def _fuente_efectiva(archivo):
    """El texto de la ruta mas el del mixin, si la ruta delega en el."""
    fuente = archivo.read_text(encoding="utf-8")
    if "payload_puerta(" in fuente and MIXIN_PUERTA.is_file():
        fuente += "\n" + MIXIN_PUERTA.read_text(encoding="utf-8")
    return fuente


def test_las_dos_rutas_mandan_los_segundos():
    for descripcion, archivo in RUTAS:
        assert archivo.is_file(), f"no existe {archivo}"
        fuente = archivo.read_text(encoding="utf-8")
        for clave in CLAVES:
            assert f'"{clave}"' in fuente, (
                f"la ruta «{descripcion}» no manda «{clave}» en el payload: "
                f"esa camara alertaria con el umbral de otra")


def test_las_dos_rutas_mandan_el_umbral_de_permanencia():
    """Una puerta o una linea dibujadas tienen que llegar por las DOS."""
    for descripcion, archivo in RUTAS_UMBRAL:
        assert archivo.is_file(), f"no existe {archivo}"
        fuente = _fuente_efectiva(archivo)
        faltan = [c for c in CLAVES_UMBRAL if f'"{c}"' not in fuente]
        assert not faltan, (
            f"la ruta «{descripcion}» no manda {faltan}: un umbral dibujado "
            f"en esa camara no clasificaria ni un paso, y en silencio")


def test_todas_las_claves_del_payload_estan_en_el_contrato():
    """El contrato es `extra='forbid'` en modo estricto: una clave que el
    payload manda y el contrato no declara hace que el servidor DESCARTE el
    100 % de los frames de ese cliente. Paso el 7-sep-2026 con
    `linea_cruce_estilo`."""
    import ast
    import sys
    sys.path.insert(0, str(RAIZ / "packages" / "elde_core"))
    from elde_core.contracts.eventos import FrameInference

    #: Claves del transporte que el contrato traduce o consume aparte.
    FUERA = {"header", "image", "event", "id_connection", "type_inference",
             "component_key", "data"}
    conocidas = set(FrameInference.model_fields) | FUERA
    # Alias del formato antiguo que la capa de compatibilidad renombra.
    for alias in ("roi_coordinates", "order_zone_coordinates",
                  "delivery_zone_coordinates", "door_roi_coordinates",
                  "door_direction", "door_direction_activate",
                  "pay_roi_coordinates", "withdraw_roi_coordinates",
                  "pay_roi_activate", "withdraw_roi_activate"):
        conocidas.add(alias)

    for descripcion, archivo in RUTAS_UMBRAL:
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        claves = {n.value for d in ast.walk(arbol) if isinstance(d, ast.Dict)
                  for n in d.keys
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        # Solo las que parecen del payload (las que el contrato podria
        # conocer): las demas son dicts internos del widget.
        sospechosas = {c for c in claves
                       if c.startswith(("linea_cruce", "flecha_direccion",
                                        "anaqueles_", "min_permanencia_"))
                       or c in ("area_nombre", "establecimiento",
                                "leer_placas", "camera_name", "camera_id")}
        desconocidas = sorted(sospechosas - conocidas)
        assert not desconocidas, (
            f"«{descripcion}» manda {desconocidas}, que el contrato no "
            f"declara: con extra='forbid' en modo estricto el servidor "
            f"rechazaria el 100 % de sus frames")


def test_el_recuadro_persiste_lo_que_el_operador_elige():
    """A diferencia del local (que arranca en blanco a proposito), esto es
    un ajuste de la camara y se recuerda entre sesiones."""
    recuadro = (RAIZ / "clients/perimetrales/src/gui/components/render_box"
                / "render_box.py").read_text(encoding="utf-8")
    assert '_save_all("min_perm_personas"' in recuadro
    assert '_save_all("min_perm_vehiculos"' in recuadro
    ventana = (RAIZ / "clients/perimetrales/src/gui/windows_main.py"
               ).read_text(encoding="utf-8")
    assert 'cfg.get("min_perm_personas")' in ventana, (
        "se guarda pero no se restaura: el operador lo pondria cada mañana")
    assert 'cfg.get("min_perm_vehiculos")' in ventana
