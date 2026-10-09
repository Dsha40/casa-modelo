"""El número de seguimiento en la etiqueta de la caja (6-oct-2026,
`elde_core/ui/etiqueta_seguimiento.py`): para comprobar a ojo que ByteTrack
conserva el número de cada persona."""
from elde_core.ui.etiqueta_seguimiento import con_id


def test_pone_el_numero_delante_de_la_etiqueta():
    salida = con_id([{'tracker_id': 100013, 'label': 'Personas 0.91'}])
    assert salida[0]['label'] == '#100013 Personas 0.91'


def test_no_toca_la_lista_original():
    original = [{'tracker_id': 7, 'label': 'Personas'}]
    con_id(original)
    assert original[0]['label'] == 'Personas'


def test_sin_numero_o_sin_etiqueta():
    salida = con_id([{'label': 'Personas'}, {'tracker_id': 3}])
    assert salida[0]['label'] == 'Personas'
    assert salida[1]['label'] == '#3'
    assert con_id(None) == []
