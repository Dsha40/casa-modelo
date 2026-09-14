"""
El JavaScript de los dashboards, dentro de la suite (19-ago-2026).

Hasta ahora el JS solo se comprobaba a mano. La regla de la casa es «verifica
ejecutando, no compilando», pero es que ni siquiera se compilaba en CI: un
paréntesis de más en `app.js` dejaba la página en blanco y las 672 pruebas
seguían en verde tan tranquilas.

Aquí se hacen las dos cosas:

  - **parsea** — `node --check` sobre cada archivo de `dashboards/`;
  - **funciona** — `js/prueba_util.js` ejecuta la lógica de `shared/util.js`
    (la plantilla que escapa sola y el respaldo de imágenes rotas) contra un
    DOM mínimo. Ahí se cazó que anidar `html` dentro de `html` doble-escapaba,
    que es el patrón más usado en las cuatro páginas.

Si `node` no está instalado, se salta: no todo el que corre las pruebas del
servidor tiene que tener Node, y un fallo aquí sería ruido, no información.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
DASHBOARDS = RAIZ / 'dashboards'

_NODE = shutil.which('node')
_SIN_NODE = pytest.mark.skipif(_NODE is None,
                               reason='node no está instalado en esta máquina')


def _js_de_dashboards():
    return sorted(DASHBOARDS.rglob('*.js'))


@_SIN_NODE
def test_todo_el_javascript_parsea():
    """Un paréntesis de más deja la página en blanco sin avisar a nadie."""
    archivos = _js_de_dashboards()
    assert archivos, 'no se encontró ni un .js en dashboards/'
    for js in archivos:
        r = subprocess.run([_NODE, '--check', str(js)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f'{js.name} no parsea:\n{r.stderr}'


@_SIN_NODE
def test_la_logica_de_util_js_hace_lo_que_dice():
    """Escapado de plantillas y respaldo de imágenes, ejecutados de verdad."""
    prueba = Path(__file__).parent / 'js' / 'prueba_util.js'
    r = subprocess.run([_NODE, str(prueba)], capture_output=True, text=True,
                       cwd=str(RAIZ))
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'Todo OK' in r.stdout


def test_ningun_dashboard_pone_javascript_en_un_atributo():
    """`onerror=`/`onclick=` en línea: JS dentro de una cadena, dentro de un
    atributo, dentro de otra cadena.

    Era la construcción más frágil de estas páginas y además repetía el
    data-URI del placeholder (636 caracteres) en CADA imagen: ~311 KB de
    atributo duplicado con 500 capturas. Ahora hay un solo oyente delegado y
    las imágenes solo declaran `data-respaldo` / `data-rota-texto`.

    Esta prueba no necesita node: es sobre el texto.
    """
    culpables = []
    for js in _js_de_dashboards():
        texto = js.read_text(encoding='utf-8')
        for n, linea in enumerate(texto.splitlines(), 1):
            despojada = linea.strip()
            if despojada.startswith(('*', '//', '/*')):
                continue                      # los comentarios sí lo nombran
            if 'onerror="' in linea or 'onclick="' in linea:
                culpables.append(f'{js.name}:{n}')
    assert not culpables, ('JavaScript en un atributo HTML: '
                           + ', '.join(culpables))
