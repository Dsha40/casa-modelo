"""
Pruebas de identidad única y anti-doble-conteo (HITO 4).

Todo con embeddings sintéticos: la lógica de decisión —que es donde nacen los
errores de conteo— se prueba sin GPU, sin cámara y en milisegundos.

El caso que da nombre a este módulo es `test_persona_fragmentada_en_muchos_
tracks`: reproduce el escenario que produjo el sobreconteo del ~314% (una
persona partida en N tracks por oclusión) y verifica que hoy sigue contando 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_config import ConfigStand  # noqa: E402
from src.analityc.core.stand_identidad import (  # noqa: E402
    GestorIdentidad, MotivoDescarte, coseno)

DIM = 128


def _cfg(**over):
    """ConfigStand mínimo con una línea de conteo horizontal en y=0.5."""
    datos = {
        'reid': {'umbral_merge': 0.78, 'umbral_merge_ventana': 0.65,
                 'ventana_reentrada_s': 120, 'ttl_galeria_min': 45},
        'conteo': {'min_frames_track': 15, 'min_dwell_zona_paso_s': 1.0},
        'demografia': {'buffer': {'max_muestras': 10}},
    }
    for ruta, valor in over.items():
        nodo = datos
        partes = ruta.split('.')
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = valor
    zonas = {'zonas': {'linea_conteo': [[0.0, 0.5], [1.0, 0.5]]}}
    return ConfigStand(datos, zonas)


def _emb(semilla: int, ruido: float = 0.0) -> np.ndarray:
    """Embedding reproducible. Misma semilla = misma persona."""
    rng = np.random.default_rng(semilla)
    base = rng.normal(size=DIM)
    if ruido:
        base = base + np.random.default_rng(semilla + 9999).normal(
            size=DIM) * ruido
    return base / np.linalg.norm(base)


def _emb_a_coseno(base: np.ndarray, objetivo: float) -> np.ndarray:
    """Vector unitario cuya similitud coseno con `base` es EXACTAMENTE
    `objetivo`.

    Necesario para probar la zona gris: con ruido gaussiano la similitud sale
    donde sale, y un test que no controla ese número no prueba lo que dice
    probar (la primera versión de este test pasaba por accidente porque la
    similitud era 1.0 y disparaba siempre el umbral alto).
    """
    rng = np.random.default_rng(4321)
    aleatorio = rng.normal(size=base.shape)
    # Componente ortogonal a base, unitaria.
    orto = aleatorio - np.dot(aleatorio, base) * base
    orto /= np.linalg.norm(orto)
    v = objetivo * base + np.sqrt(1.0 - objetivo ** 2) * orto
    return v / np.linalg.norm(v)


def _recorrer_emb(g: GestorIdentidad, tracker_id: int, emb: np.ndarray, *,
                  frames: int = 20, t0: float = 0.0) -> None:
    """Como `_recorrer` pero con un embedding fijo dado."""
    for i in range(frames):
        y = 0.2 + (0.6 * i / max(1, frames - 1))
        g.observar(tracker_id, embedding=emb, calidad=100.0 - i,
                   punto_pie=(0.5, y), en_zona_paso=True, ts=t0 + i * 0.1)


def _recorrer(g: GestorIdentidad, tracker_id: int, semilla: int, *,
              frames: int = 20, t0: float = 0.0, ruido: float = 0.02,
              cruzar: bool = True, dwell: bool = True) -> None:
    """Simula un track completo que cruza la línea de arriba a abajo."""
    for i in range(frames):
        y = 0.2 + (0.6 * i / max(1, frames - 1)) if cruzar else 0.2
        g.observar(tracker_id, embedding=_emb(semilla, ruido),
                   calidad=100.0 - i, punto_pie=(0.5, y),
                   en_zona_paso=dwell, ts=t0 + i * 0.1)


def test_una_persona_un_track_cuenta_uno():
    g = GestorIdentidad(_cfg())
    _recorrer(g, 1, semilla=11)
    r = g.cerrar(1, ts=5.0)
    assert r.contable and not r.descartado
    assert g.total_contable() == 1


def test_dos_personas_distintas_cuentan_dos():
    g = GestorIdentidad(_cfg())
    _recorrer(g, 1, semilla=11)
    g.cerrar(1, ts=5.0)
    _recorrer(g, 2, semilla=77, t0=10.0)
    g.cerrar(2, ts=15.0)
    assert g.total_contable() == 2


def test_persona_fragmentada_en_muchos_tracks():
    """El escenario del sobreconteo del ~314%.

    Una sola persona a la que el tracker pierde y recupera 5 veces (el
    mostrador la ocluye). Contando `tracker_id` saldrían 5 personas.
    """
    g = GestorIdentidad(_cfg())
    for i, tid in enumerate([10, 11, 12, 13, 14]):
        _recorrer(g, tid, semilla=42, t0=i * 12.0)
        g.cerrar(tid, ts=i * 12.0 + 5.0)

    assert g.total_contable() == 1, (
        f"la misma persona contó {g.total_contable()} veces")
    assert len(g.fusiones()) == 4       # 4 reapariciones fusionadas
    assert g.calidad_dato()['fusiones_realizadas'] == 4


def test_track_corto_se_descarta_con_motivo():
    g = GestorIdentidad(_cfg())
    _recorrer(g, 1, semilla=11, frames=5)      # < min_frames_track (15)
    r = g.cerrar(1, ts=1.0)
    assert r.descartado and r.motivo_descarte == MotivoDescarte.TRACK_CORTO
    assert g.total_contable() == 0
    # No se borra: alimenta el panel de calidad del dato.
    assert g.descartes()[0]['motivo'] == MotivoDescarte.TRACK_CORTO


def test_track_que_nunca_pisa_la_zona_se_descarta():
    g = GestorIdentidad(_cfg())
    _recorrer(g, 1, semilla=11, cruzar=False, dwell=False)
    r = g.cerrar(1, ts=5.0)
    assert r.descartado
    assert r.motivo_descarte == MotivoDescarte.NUNCA_EN_ZONA
    assert g.total_contable() == 0


def test_cruzar_la_linea_basta_aunque_no_haya_dwell():
    g = GestorIdentidad(_cfg())
    _recorrer(g, 1, semilla=11, cruzar=True, dwell=False)
    r = g.cerrar(1, ts=5.0)
    assert r.contable, "cruzar la línea de conteo debería bastar"


def test_permanecer_en_zona_stand_tambien_hace_contable():
    """Quien llega al stand por el frente, sin pisar el pasillo, CUENTA.

    Reproduce el descarte del 7-ago-2026: un visitante con detención
    registrada frente al mostrador salía 'nunca_en_zona' porque solo la
    línea o zona_paso ganaban el conteo — y es justo la persona que el
    módulo existe para medir.
    """
    g = GestorIdentidad(_cfg())
    emb = _emb(11)
    for i in range(20):
        g.observar(1, embedding=emb, calidad=100.0 - i,
                   punto_pie=(0.5, 0.7), en_zona_paso=False,
                   en_zona_stand=True, ts=i * 0.1)
    r = g.cerrar(1, ts=5.0)
    assert r.contable and not r.descartado
    assert g.total_contable() == 1


def test_un_roce_con_zona_stand_no_basta():
    """Menos dwell en zona_stand que el mínimo sigue descartándose: la regla
    nueva no puede convertir en contable a un falso positivo que parpadea
    un par de frames dentro de la zona."""
    g = GestorIdentidad(_cfg())
    emb = _emb(11)
    for i in range(20):
        g.observar(1, embedding=emb, calidad=100.0 - i,
                   punto_pie=(0.5, 0.7), en_zona_paso=False,
                   en_zona_stand=(i < 5),      # ~0.4 s dentro, luego fuera
                   ts=i * 0.1)
    r = g.cerrar(1, ts=5.0)
    assert r.descartado
    assert r.motivo_descarte == MotivoDescarte.NUNCA_EN_ZONA


def test_track_sin_embedding_nunca_compromete_identidad():
    """Sin crop utilizable no hay identidad, y sin identidad no se cuenta."""
    g = GestorIdentidad(_cfg())
    for i in range(20):
        pid = g.observar(1, punto_pie=(0.5, 0.2 + 0.03 * i),
                         en_zona_paso=True, ts=i * 0.1)
        assert pid.startswith('prov:'), "no debe salir del estado provisional"
    r = g.cerrar(1, ts=5.0)
    assert r.descartado
    assert r.motivo_descarte == MotivoDescarte.SIN_EMBEDDING


def test_zona_gris_solo_fusiona_dentro_de_la_ventana():
    """Similitud entre 0.65 y 0.78: la reaparición debe ser plausible.

    Sin la ventana temporal, dos personas parecidas a lo largo de la jornada
    se colapsarían en una sola (subconteo).
    """
    cfg = _cfg(**{'reid.umbral_merge': 0.90,
                  'reid.umbral_merge_ventana': 0.60,
                  'reid.ventana_reentrada_s': 30})
    base = _emb(5)
    gris = _emb_a_coseno(base, 0.75)     # 0.60 < 0.75 < 0.90 => zona gris
    assert 0.60 < coseno(base, gris) < 0.90

    # Dentro de la ventana (10 s < 30 s): fusiona.
    g1 = GestorIdentidad(cfg)
    _recorrer_emb(g1, 1, base)
    g1.cerrar(1, ts=2.0)
    _recorrer_emb(g1, 2, gris, t0=10.0)
    g1.cerrar(2, ts=12.0)
    assert g1.total_contable() == 1, "debió fusionar dentro de la ventana"

    # Fuera de la ventana (500 s > 30 s): identidad nueva.
    g2 = GestorIdentidad(cfg)
    _recorrer_emb(g2, 1, base)
    g2.cerrar(1, ts=2.0)
    _recorrer_emb(g2, 2, gris, t0=500.0)
    g2.cerrar(2, ts=502.0)
    assert g2.total_contable() == 2, "no debió fusionar fuera de la ventana"


def test_ttl_vacia_la_galeria():
    """La galería es efímera (regla 7 de privacidad), no un registro."""
    cfg = _cfg(**{'reid.ttl_galeria_min': 1})     # 60 s
    g = GestorIdentidad(cfg)
    _recorrer(g, 1, semilla=11)
    g.cerrar(1, ts=2.0)
    assert g.calidad_dato()['identidades_en_galeria'] == 1
    # Una observación muy posterior fuerza la caducidad.
    _recorrer(g, 2, semilla=77, t0=1000.0)
    assert g.calidad_dato()['identidades_en_galeria'] == 1


def test_provisional_hasta_tener_muestras_suficientes():
    g = GestorIdentidad(_cfg())
    p1 = g.observar(1, embedding=_emb(11), calidad=9, punto_pie=(0.5, 0.2),
                    ts=0.0)
    p2 = g.observar(1, embedding=_emb(11), calidad=9, punto_pie=(0.5, 0.3),
                    ts=0.1)
    assert p1.startswith('prov:') and p2.startswith('prov:')
    p3 = g.observar(1, embedding=_emb(11), calidad=9, punto_pie=(0.5, 0.4),
                    ts=0.2)
    assert not p3.startswith('prov:'), "con 3 muestras ya debe comprometerse"


def test_solo_se_promedian_las_mejores_muestras():
    cfg = _cfg(**{'demografia.buffer.max_muestras': 3})
    g = GestorIdentidad(cfg)
    # 3 crops buenos de la persona real + 10 basura de baja calidad.
    for i in range(3):
        g.observar(1, embedding=_emb(11), calidad=1000.0,
                   punto_pie=(0.5, 0.2), ts=i * 0.1)
    for i in range(10):
        g.observar(1, embedding=_emb(999 + i), calidad=0.1,
                   punto_pie=(0.5, 0.3), ts=1.0 + i * 0.1)
    # La identidad debe seguir pareciéndose a la persona real.
    pid = g.observar(1, embedding=_emb(11), calidad=1000.0,
                     punto_pie=(0.5, 0.6), ts=3.0)
    emb_ident = g._galeria[pid].embedding
    assert coseno(emb_ident, _emb(11)) > 0.9, (
        "los crops basura contaminaron el embedding")


def test_calidad_dato_reporta_todo():
    g = GestorIdentidad(_cfg())
    _recorrer(g, 1, semilla=11)
    g.cerrar(1, ts=5.0)
    _recorrer(g, 2, semilla=11, t0=10.0)        # misma persona -> fusión
    g.cerrar(2, ts=15.0)
    _recorrer(g, 3, semilla=77, frames=4, t0=20.0)   # basura
    g.cerrar(3, ts=21.0)

    q = g.calidad_dato()
    assert q['personas_contables'] == 1
    assert q['fusiones_realizadas'] == 1
    assert q['tracks_descartados'] == 1
    assert q['descartes_por_motivo'][MotivoDescarte.TRACK_CORTO] == 1


# ── Ejecutable sin pytest (ver test_stand_almacen.py) ────────────────────
if __name__ == '__main__':
    fallos = 0
    for _nombre, _fn in sorted(list(globals().items())):
        if not _nombre.startswith('test_') or not callable(_fn):
            continue
        try:
            _fn()
            print(f"  OK    {_nombre}")
        except Exception as _exc:                          # noqa: BLE001
            fallos += 1
            print(f"  FALLO {_nombre}: {type(_exc).__name__}: {_exc}")
    print(f"\n{'TODO VERDE' if not fallos else f'{fallos} FALLO(S)'}")
    sys.exit(1 if fallos else 0)
