"""
Pruebas de a QUE servidor se conecta un cliente al arrancar (12-ago-2026).

La decision tiene cinco fuentes y un orden que importa. Lo que se prueba aqui
es justo ese orden, porque equivocarlo tiene consecuencias visibles:

  - Si el `.env` ganara a la eleccion guardada, elegir un servidor nuevo en la
    ventana de conexion no serviria de nada mientras el `.env` viejo siguiera
    ahi (que es lo que pasaba en la maquina de desarrollo).
  - Si no se distinguiera el `server_ws_url` que INYECTA el selector del que
    trae el `.env`, el cliente volveria a preguntar aunque acabaras de elegir
    servidor en el hub.
  - Si un servidor guardado que no responde no hiciera preguntar, un cambio de
    IP dejaria al cliente reintentando en silencio para siempre.

Sin GUI y sin red: se sustituye `alcanzable` y nunca se abre el dialogo.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pytest  # noqa: E402

from elde_core.red import arranque_servidor as A  # noqa: E402
from elde_core.red import servidores_guardados as G  # noqa: E402


@pytest.fixture(autouse=True)
def entorno_limpio(tmp_path, monkeypatch):
    """Cada prueba con su propia carpeta de configuracion y sin heredar el
    entorno de la anterior (ni el de la maquina que corre las pruebas)."""
    monkeypatch.setenv("ELDE_CONFIG_DIR", str(tmp_path))
    for variable in ("server_ws_url", A.MARCA_FIJADO, "ELDE_NO_PREGUNTAR",
                     "ELDE_SERVIDORES"):
        monkeypatch.delenv(variable, raising=False)
    # Ninguna prueba debe acabar abriendo una ventana.
    monkeypatch.setattr(A, "preguntar",
                        lambda *a, **k: pytest.fail(
                            "no deberia haberse abierto la ventana"))
    yield


def _responde(monkeypatch, valor: bool):
    monkeypatch.setattr(A.D, "alcanzable", lambda *a, **k: valor)


# ---------------------------------------------------------------------------
#  1) Linea de ordenes
# ---------------------------------------------------------------------------
def test_la_linea_de_ordenes_manda_y_se_recuerda(monkeypatch):
    _responde(monkeypatch, True)
    url = A.resolver("managers", argv=["x", "--servidor", "10.0.0.5:9100"])
    assert url == "ws://10.0.0.5:9100/ws"
    # Queda fijado en el entorno para `ajustes.cargar()`, que es quien lo lee.
    import os
    assert os.environ["server_ws_url"] == url
    # La marca lleva el NOMBRE del cliente, no un "1": asi no arrastra a un
    # proceso hijo de otro cliente (ver MARCA_FIJADO).
    assert os.environ[A.MARCA_FIJADO] == "managers"
    # Y recordado, para no tener que repetir el argumento cada vez.
    assert G.ultimo("managers") == ("10.0.0.5", 9100)


def test_admite_las_tres_formas_de_escribirlo(monkeypatch):
    _responde(monkeypatch, True)
    for entrada in ("10.0.0.5", "10.0.0.5:9000", "ws://10.0.0.5:9000/ws"):
        assert A.resolver("x", argv=["p", f"--servidor={entrada}"]) \
            == "ws://10.0.0.5:9000/ws"


# ---------------------------------------------------------------------------
#  2) El selector ya eligio
# ---------------------------------------------------------------------------
def test_lo_que_inyecta_el_selector_no_vuelve_a_preguntar(monkeypatch):
    monkeypatch.setenv("server_ws_url", "ws://5.5.5.5:9000/ws")
    monkeypatch.setenv(A.MARCA_FIJADO, "1")
    # Aunque el servidor NO responda: el operador ya lo eligio en el hub y el
    # cliente se reconecta solo. Volver a preguntarle seria preguntar dos
    # veces lo mismo.
    _responde(monkeypatch, False)
    assert A.resolver("managers", argv=["x"]) == "ws://5.5.5.5:9000/ws"


def test_sin_la_marca_el_env_es_solo_un_candidato(monkeypatch):
    """El mismo valor, pero sin la marca, es un `.env`: se usa si responde."""
    monkeypatch.setenv("server_ws_url", "ws://5.5.5.5:9000/ws")
    _responde(monkeypatch, True)
    assert A.resolver("managers", argv=["x"]) == "ws://5.5.5.5:9000/ws"


# ---------------------------------------------------------------------------
#  3) y 4) Lo guardado gana al .env
# ---------------------------------------------------------------------------
def test_la_eleccion_guardada_gana_al_env(monkeypatch):
    monkeypatch.setenv("server_ws_url", "ws://1.1.1.1:9000/ws")
    G.recordar("2.2.2.2", 9000, "managers")
    _responde(monkeypatch, True)
    assert A.resolver("managers", argv=["x"]) == "ws://2.2.2.2:9000/ws"


def test_lo_guardado_por_cliente_gana_a_lo_comun(monkeypatch):
    G.recordar("1.1.1.1", 9000)                     # comun
    G.recordar("2.2.2.2", 9000, "tienda")           # solo tienda
    _responde(monkeypatch, True)
    assert A.resolver("tienda", argv=["x"]) == "ws://2.2.2.2:9000/ws"
    assert A.resolver("managers", argv=["x"]) == "ws://1.1.1.1:9000/ws"


def test_la_marca_de_un_cliente_no_arrastra_a_otro(monkeypatch):
    """Las variables de entorno se HEREDAN. Si la marca fuera un "1" a secas,
    un proceso hijo de otro cliente la veria, se saltaria su propia eleccion y
    acabaria conectado al servidor del padre."""
    monkeypatch.setenv("server_ws_url", "ws://1.1.1.1:9000/ws")
    monkeypatch.setenv(A.MARCA_FIJADO, "tienda")
    G.recordar("2.2.2.2", 9000, "managers")
    _responde(monkeypatch, True)
    # 'tienda' se fia de la marca; 'managers' no, y usa lo suyo.
    assert A.resolver("tienda", argv=["x"]) == "ws://1.1.1.1:9000/ws"
    monkeypatch.setenv(A.MARCA_FIJADO, "tienda")
    assert A.resolver("managers", argv=["x"]) == "ws://2.2.2.2:9000/ws"


def test_la_marca_del_selector_vale_para_cualquier_cliente(monkeypatch):
    """El "1" lo pone el SELECTOR, que ya pregunto por todos."""
    monkeypatch.setenv("server_ws_url", "ws://1.1.1.1:9000/ws")
    monkeypatch.setenv(A.MARCA_FIJADO, "1")
    G.recordar("2.2.2.2", 9000, "managers")
    _responde(monkeypatch, True)
    assert A.resolver("managers", argv=["x"]) == "ws://1.1.1.1:9000/ws"


# ---------------------------------------------------------------------------
#  La excepcion: el elegido no responde
# ---------------------------------------------------------------------------
def test_si_el_guardado_no_responde_se_pregunta(monkeypatch):
    G.recordar("2.2.2.2", 9000, "managers")
    _responde(monkeypatch, False)
    monkeypatch.setattr(A, "preguntar", lambda *a, **k: "ws://9.9.9.9:9000/ws")
    assert A.resolver("managers", argv=["x"]) == "ws://9.9.9.9:9000/ws"


def test_no_preguntar_sirve_para_instalaciones_desatendidas(monkeypatch):
    """Un puesto sin nadie delante no puede quedarse esperando una respuesta."""
    G.recordar("2.2.2.2", 9000, "managers")
    monkeypatch.setenv("ELDE_NO_PREGUNTAR", "1")
    _responde(monkeypatch, False)
    assert A.resolver("managers", argv=["x"]) == "ws://2.2.2.2:9000/ws"


def test_cancelar_la_ventana_cae_al_candidato(monkeypatch):
    """Cerrar la ventana no debe cerrar el cliente si habia un servidor
    razonable: se abre con el de antes y se reconectara solo."""
    G.recordar("2.2.2.2", 9000, "managers")
    _responde(monkeypatch, False)
    monkeypatch.setattr(A, "preguntar", lambda *a, **k: None)
    assert A.resolver("managers", argv=["x"]) == "ws://2.2.2.2:9000/ws"


def test_sin_nada_y_cancelando_no_hay_servidor(monkeypatch):
    """Aqui si: sin servidor no hay cliente que valga, y quien llama debe
    cerrar en vez de abrir una ventana que no va a recibir nada."""
    _responde(monkeypatch, False)
    monkeypatch.setattr(A, "preguntar", lambda *a, **k: None)
    assert A.resolver("managers", argv=["x"]) is None


# ---------------------------------------------------------------------------
#  Forzar la ventana
# ---------------------------------------------------------------------------
def test_elegir_servidor_fuerza_la_ventana(monkeypatch):
    """`--elegir-servidor` pregunta aunque todo lo demas este decidido: es la
    unica forma de cambiar de servidor sin borrar la configuracion."""
    monkeypatch.setenv("server_ws_url", "ws://1.1.1.1:9000/ws")
    monkeypatch.setenv(A.MARCA_FIJADO, "1")
    G.recordar("2.2.2.2", 9000, "managers")
    _responde(monkeypatch, True)
    monkeypatch.setattr(A, "preguntar", lambda *a, **k: "ws://3.3.3.3:9000/ws")
    assert A.resolver("managers", argv=["x", "--elegir-servidor"]) \
        == "ws://3.3.3.3:9000/ws"


def test_sin_dialogo_permitido_no_se_bloquea(monkeypatch):
    _responde(monkeypatch, False)
    assert A.resolver("x", argv=["p"], permitir_dialogo=False) is None
    G.recordar("2.2.2.2", 9000, "x")
    assert A.resolver("x", argv=["p"], permitir_dialogo=False) \
        == "ws://2.2.2.2:9000/ws"
