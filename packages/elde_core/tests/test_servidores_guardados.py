"""
Pruebas de la lista de servidores que se recuerda entre arranques
(12-ago-2026).

Es el fichero que hace que no haya que volver a escribir la IP de la VPN cada
vez, asi que lo que importa es que no se pierda ni se corrompa: un JSON a
medias dejaria al operador sin lista, y un archivo ilegible no puede impedir
que el cliente arranque.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402

from elde_core.red import servidores_guardados as G  # noqa: E402


@pytest.fixture(autouse=True)
def carpeta_propia(tmp_path, monkeypatch):
    monkeypatch.setenv("ELDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("ELDE_SERVIDORES", raising=False)
    yield


def test_arranca_en_vacio_sin_fichero():
    assert G.cargar() == {"version": 1, "manuales": [], "ultimo": None,
                          "por_cliente": {}}
    assert G.manuales() == []
    assert G.ultimo() is None


def test_anadir_y_quitar_manuales():
    assert G.anadir_manual("10.8.0.1")
    assert G.anadir_manual("10.8.0.1")          # idempotente, no duplica
    assert G.manuales() == ["10.8.0.1:9000"]
    assert G.anadir_manual("otra.sede", 9100)
    assert G.manuales() == ["10.8.0.1:9000", "otra.sede:9100"]
    assert G.quitar_manual("10.8.0.1")
    assert G.manuales() == ["otra.sede:9100"]
    assert not G.quitar_manual("10.8.0.1")      # ya no estaba


def test_no_se_guarda_un_host_vacio():
    assert not G.anadir_manual("   ")
    assert G.manuales() == []


def test_la_variable_de_entorno_siembra_la_lista(monkeypatch):
    """Existe para el despliegue: el instalador puede dejar puesta la
    direccion del servidor de esa sucursal sin tocar codigo ni obligar al
    operador a escribirla."""
    G.anadir_manual("10.8.0.1")
    monkeypatch.setenv("ELDE_SERVIDORES", "1.2.3.4, 5.6.7.8:9100 ;9.9.9.9")
    assert G.manuales() == ["10.8.0.1:9000", "1.2.3.4:9000", "5.6.7.8:9100",
                            "9.9.9.9:9000"]


def test_lo_del_entorno_no_duplica_lo_ya_guardado(monkeypatch):
    G.anadir_manual("1.2.3.4")
    monkeypatch.setenv("ELDE_SERVIDORES", "1.2.3.4")
    assert G.manuales() == ["1.2.3.4:9000"]


def test_recordar_y_olvidar():
    assert G.ultimo() is None
    G.recordar("10.0.0.1")
    assert G.ultimo() == ("10.0.0.1", 9000)
    G.recordar("10.0.0.2", 9100, "tienda")
    assert G.ultimo("tienda") == ("10.0.0.2", 9100)
    # Un cliente sin preferencia propia cae en la comun.
    assert G.ultimo("perimetrales") == ("10.0.0.1", 9000)
    G.olvidar("tienda")
    assert G.ultimo("tienda") == ("10.0.0.1", 9000)
    G.olvidar()
    assert G.ultimo() is None


def test_un_json_corrupto_no_tumba_el_arranque(tmp_path):
    """Preferible empezar de cero que dejar al operador sin cliente."""
    (tmp_path / G.NOMBRE_ARCHIVO).write_text("{esto no es json",
                                             encoding="utf-8")
    assert G.cargar()["manuales"] == []
    assert G.ultimo() is None
    # Y se puede volver a escribir encima.
    assert G.anadir_manual("10.0.0.1")
    assert G.manuales() == ["10.0.0.1:9000"]


def test_un_json_con_la_forma_equivocada_se_ignora(tmp_path):
    (tmp_path / G.NOMBRE_ARCHIVO).write_text('["una lista"]', encoding="utf-8")
    assert G.cargar()["por_cliente"] == {}


def test_campos_de_tipo_raro_no_rompen(tmp_path):
    (tmp_path / G.NOMBRE_ARCHIVO).write_text(
        json.dumps({"manuales": "no soy una lista", "por_cliente": 7,
                    "ultimo": "tampoco"}), encoding="utf-8")
    datos = G.cargar()
    assert datos["manuales"] == []
    assert datos["por_cliente"] == {}
    assert G.ultimo() is None


def test_la_escritura_no_deja_temporales(tmp_path):
    """Se escribe en un temporal y se renombra: un corte a mitad no puede
    dejar el JSON truncado."""
    G.recordar("10.0.0.1")
    assert (tmp_path / G.NOMBRE_ARCHIVO).exists()
    assert list(tmp_path.glob("*.tmp")) == []
