"""Mapa de calor diferido por seguimiento (22-oct-2026, `calor_diferido.py`).

Se fija: el calor no se pinta mientras dura el seguimiento; al terminar va al
mapa de clientes si nunca fue personal, al del personal si fue de camiseta
(TAMBIÉN lo de antes de la marca) y a ninguno si fue vigilante; uno largo se
reparte cada ESPERA_MAX_S.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import calor_diferido as cd  # noqa: E402


class Mapa:
    def __init__(self):
        self.n = 0

    def add_person(self, box, w, h):
        self.n += 1


def _camiseta(tr):
    return tr.get('personal_rol') == 'pasillero'


def test_reparte_al_terminar_segun_lo_que_resulto_ser():
    c = cd.CalorDiferido()
    cli, per = Mapa(), Mapa()
    for i in range(5):                                       # 3 personas, 5 imágenes
        for tid in (1, 2, 3):
            c.anotar(tid, [0, 0, 1, 1], 100, 100, t=100.0 + i)
    tracks = {1: {}, 2: {'es_personal': True, 'personal_rol': 'pasillero'},
              3: {'es_personal': True, 'personal_rol': 'vigilante'}}
    assert c.repartir(tracks, cli, lambda: per, _camiseta, t=105.0) == 0   # siguen vivos
    assert cli.n == 0 and per.n == 0
    # Ya no están en `tracks`: se usa lo que se recordó mientras se los veía.
    c.repartir({}, cli, lambda: per, _camiseta, t=110.0)
    assert (cli.n, per.n) == (5, 5)                          # el vigilante, a ninguno


def test_uno_largo_se_reparte_cada_rato():
    c = cd.CalorDiferido()
    cli = Mapa()
    for i in range(70):
        c.anotar(7, [0, 0, 1, 1], 100, 100, t=100.0 + i)
        c.repartir({7: {}}, cli, lambda: None, _camiseta, t=100.0 + i)
    assert cli.n >= 60                                       # sin esperar a que acabe
