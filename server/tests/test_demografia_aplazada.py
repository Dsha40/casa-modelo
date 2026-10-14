"""Conservar el género mientras ByteTrack recuerde el número (12-oct-2026,
`demografia_aplazada.py`).

Lo que se fija: quien vuelve antes de `BYTETRACK_LOST_S` conserva su jurado
(no se borra nada), quien no vuelve se borra una sola vez al vencer, y
`tiendas.py` aplaza en vez de borrar, purga en cada imagen y cancela al
volver el mismo número.
"""
from pathlib import Path

from src.analityc.core import demografia_aplazada as da


class _Demo:
    pass


def test_quien_vuelve_a_tiempo_no_se_borra():
    demo, borrados = _Demo(), []
    da.aplazar(demo, 7, 100.0)
    assert da.purgar(demo, 105.0, 9.0, borrados.append) == 0
    assert da.cancelar(demo, 7) is True
    assert da.purgar(demo, 200.0, 9.0, borrados.append) == 0
    assert borrados == []


def test_quien_no_vuelve_se_borra_una_vez_al_vencer():
    demo, borrados = _Demo(), []
    da.aplazar(demo, 7, 100.0)
    da.aplazar(demo, 8, 104.0)
    assert da.purgar(demo, 109.0, 9.0, borrados.append) == 1
    assert borrados == [7]
    assert da.purgar(demo, 120.0, 9.0, borrados.append) == 1
    assert borrados == [7, 8]
    assert da.purgar(demo, 130.0, 9.0, borrados.append) == 0


def test_un_fallo_al_borrar_no_para_a_los_demas():
    demo, borrados = _Demo(), []

    def borrar(t):
        if t == 1:
            raise RuntimeError('roto')
        borrados.append(t)
    da.aplazar(demo, 1, 0.0)
    da.aplazar(demo, 2, 0.0)
    assert da.purgar(demo, 10.0, 9.0, borrar) == 2
    assert borrados == [2]


def test_sin_clasificador_no_hace_nada():
    assert da.purgar(None, 10.0, 9.0, lambda t: None) == 0
    assert da.cancelar(None, 3) is False


def test_tiendas_aplaza_purga_y_cancela():
    """Sobre el código: `_remove_track` ya no borra la demografía en el acto;
    `_update_tracks_botsort` purga al empezar y cancela al volver un número."""
    src = (Path(__file__).resolve().parents[1] / 'src' / 'analityc' / 'core'
           / 'tiendas.py').read_text(encoding='utf-8')
    quitar = src[src.index('    def _remove_track('):src.index('    def _borrar_demografia(')]
    assert 'demografia_aplazada.aplazar(' in quitar
    assert 'remove_track(track_id)' not in quitar
    assert '_sellar_captura_sin_demografia' not in quitar
    botsort = src[src.index('    def _update_tracks_botsort('):src.index('    def _resolve_track_identities(')]
    assert botsort.index('self._purgar_demografia()') < botsort.index('for det in detections')
    nuevo = botsort[botsort.index('# Nuevo track de BoTSORT'):]
    assert nuevo.index('demografia_aplazada.cancelar(') < nuevo.index('self._classify_person(')
