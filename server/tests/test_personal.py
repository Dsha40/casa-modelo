"""
Pruebas del personal de tienda enrolado con fotos (14-ago-2026).

Sin modelos reales: el «foto → embedding» se inyecta (probar YuNet/ArcFace es
probar los modelos, no nuestra lógica), el reidentificador es un doble con la
misma forma (`_db`/`_lock`) y el identificador del medidor es un stub.

Lo que más se protege:

  - **el personal NO cuenta en el marketing** — un pasillero frente al
    anaquel no suma visitas ni agarres ni calienta el mapa: es LA razón de
    ser de la funcionalidad;
  - **las alertas operativas salen con su cooldown** — «seguridad en el
    pasillo» al verlo y «surte el área» tras un dwell sostenido, sin
    inundar;
  - **el enrolamiento persiste y sobrevive al reinicio** — fichas y
    embeddings se releen del disco;
  - **el error barato es el correcto** — sin galería, con galería vacía o
    ante cualquier fallo, TODOS cuentan como clientes (nunca al revés).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.anaqueles import (  # noqa: E402
    MedidorAnaqueles, _DEFECTOS)
from src.analityc.core.anaqueles_registro import (  # noqa: E402
    RegistroAnaqueles)
from src.analityc.core.personal_galeria import (  # noqa: E402
    GaleriaPersonalTienda, IdentificadorPersonal)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FOTO = np.full((200, 160, 3), 128, dtype=np.uint8)
AREA = [[100, 100], [300, 100], [300, 200], [100, 200]]
BOX_EN_AREA = [120, 120, 220, 460]


def _vec(idx: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[idx] = 1.0
    return v


def _galeria(tmp_path, monkeypatch, emb=None) -> GaleriaPersonalTienda:
    gal = GaleriaPersonalTienda(raiz=tmp_path / 'personal')
    monkeypatch.setattr(gal, '_embedding_de_foto',
                        lambda foto: (emb if emb is not None else _vec(0)))
    return gal


class ReidFalso:
    """La forma mínima que consulta IdentificadorPersonal."""

    def __init__(self, embeddings_por_pid):
        import threading
        self._lock = threading.RLock()
        self._db = {pid: {'embeddings': list(embs)}
                    for pid, embs in embeddings_por_pid.items()}


# ── Galería ──────────────────────────────────────────────────────────────
def test_enrolar_listar_y_persistir(tmp_path, monkeypatch):
    gal = _galeria(tmp_path, monkeypatch)
    ficha = gal.enrolar('Ana Pérez', 'seguridad', FOTO)
    assert ficha['nombre'] == 'Ana Pérez'
    assert ficha['rol'] == 'seguridad'
    assert ficha['activo'] and ficha['n_fotos'] == 1
    assert gal.foto_ruta(ficha['id'], 0) is not None

    # «Reinicio del servidor»: otra instancia sobre el mismo disco.
    gal2 = GaleriaPersonalTienda(raiz=tmp_path / 'personal')
    assert [f['nombre'] for f in gal2.listar()] == ['Ana Pérez']
    assert gal2.identificar(_vec(0)) is not None


def test_identificar_respeta_umbral_y_estado(tmp_path, monkeypatch):
    gal = _galeria(tmp_path, monkeypatch, emb=_vec(0))
    ficha = gal.enrolar('Luis', 'pasillero', FOTO)
    # Mismo rostro: identifica; otro rostro (ortogonal): no.
    assert gal.identificar(_vec(0))[1]['nombre'] == 'Luis'
    assert gal.identificar(_vec(1)) is None
    # Desactivado deja de identificarse (vacaciones), sin perder la ficha.
    gal.poner(ficha['id'], activo=False)
    assert gal.identificar(_vec(0)) is None
    gal.poner(ficha['id'], activo=True)
    assert gal.identificar(_vec(0)) is not None


def test_enrolar_valida_datos(tmp_path, monkeypatch):
    gal = _galeria(tmp_path, monkeypatch)
    for nombre, rol in (('', 'seguridad'), ('Ana', 'gerente')):
        try:
            gal.enrolar(nombre, rol, FOTO)
            raise AssertionError('debió rechazar')
        except ValueError:
            pass


def test_borrar_elimina_fotos_y_deja_de_identificar(tmp_path, monkeypatch):
    gal = _galeria(tmp_path, monkeypatch)
    ficha = gal.enrolar('Eva', 'pasillero', FOTO)
    ruta = gal.foto_ruta(ficha['id'], 0)
    gal.borrar(ficha['id'])
    assert not ruta.exists()
    assert gal.identificar(_vec(0)) is None
    assert gal.listar() == []


# ── Identificador (pid → ficha) ──────────────────────────────────────────
def test_identificador_resuelve_por_embeddings_del_reid(tmp_path, monkeypatch):
    gal = _galeria(tmp_path, monkeypatch, emb=_vec(3))
    gal.enrolar('Ana', 'seguridad', FOTO)
    ident = IdentificadorPersonal(galeria=gal)
    reid = ReidFalso({'uuid-ana': [_vec(3)], 'uuid-cliente': [_vec(7)]})

    ficha = ident.ficha_de('uuid-ana', reid)
    assert ficha and ficha['nombre'] == 'Ana' and ficha['rol'] == 'seguridad'
    assert ident.ficha_de('uuid-cliente', reid) is None
    assert ident.ficha_de('prov:12345', reid) is None     # sin rostro = cliente
    # El caché sirve el veredicto aunque luego falte el reid (correcto);
    # un identificador FRESCO sin reid no puede resolver: cliente.
    assert IdentificadorPersonal(galeria=gal).ficha_de('uuid-ana',
                                                       None) is None


def test_identificador_se_invalida_al_cambiar_la_galeria(tmp_path, monkeypatch):
    gal = _galeria(tmp_path, monkeypatch, emb=_vec(3))
    ficha = gal.enrolar('Ana', 'seguridad', FOTO)
    ident = IdentificadorPersonal(galeria=gal)
    reid = ReidFalso({'uuid-ana': [_vec(3)]})
    assert ident.ficha_de('uuid-ana', reid) is not None
    gal.borrar(ficha['id'])                    # sube la versión de la galería
    assert ident.ficha_de('uuid-ana', reid) is None


# ── Exclusión en el modo Anaqueles ───────────────────────────────────────
class IdentStub:
    def __init__(self, fichas_por_pid):
        self.fichas = dict(fichas_por_pid)

    def ficha_de(self, pid, reid=None):
        return self.fichas.get(str(pid))


class PoseKlienta:
    disponible = True

    def munecas(self, frame, box):
        return [(150, 150, 0.9)]               # muñeca dentro del área


def _medidor_con_personal(tmp_path, fichas, **cfg_extra):
    cfg = dict(_DEFECTOS)
    cfg.update(cfg_extra)
    m = MedidorAnaqueles(
        'cam-test', cfg, pose=PoseKlienta(),
        registro=RegistroAnaqueles('cam-test', raiz=tmp_path),
        escena=type('E', (), {'evaluar': lambda *a, **k: None})(),
        identificador=IdentStub(fichas))
    m.poner_zonas([AREA], ['Área 1'])
    return m


def _track(pid, tid=1):
    return {tid: {'class': -1, 'box': list(BOX_EN_AREA),
                  'persistent_id': pid}}


def test_pasillero_no_suma_visitas_ni_agarres_pero_avisa_surte(tmp_path):
    m = _medidor_con_personal(
        tmp_path, {'uuid-luis': {'nombre': 'Luis', 'rol': 'pasillero'}},
        surte_dwell_s=3.0)
    t0 = 1000.0
    for i in range(10):                        # 5 s frente al anaquel
        m.procesar(FRAME, _track('uuid-luis'), ahora=t0 + i * 0.5)
    # NADA de marketing: es la razón de ser de la funcionalidad.
    assert m.registro.area(0)['visitas'] == 0
    assert m.registro.area(0)['agarres'] == 0
    assert float(m.registro.area(0)['_grid'].sum()) == 0.0
    # Y UNA alerta de surtido (dwell 3 s, cooldown evita repetir).
    alertas = m.drenar_alertas_personal()
    assert len(alertas) == 1
    assert alertas[0]['tipo'] == 'surte'
    assert alertas[0]['nombre'] == 'Luis'
    assert alertas[0]['area'] == 'Área 1'


def test_seguridad_avisa_al_verla_con_cooldown(tmp_path):
    m = _medidor_con_personal(
        tmp_path, {'uuid-ana': {'nombre': 'Ana', 'rol': 'seguridad'}},
        alerta_seguridad_cooldown_s=300.0)
    t0 = 1000.0
    m.procesar(FRAME, _track('uuid-ana'), ahora=t0)
    m.procesar(FRAME, _track('uuid-ana'), ahora=t0 + 5.0)
    alertas = m.drenar_alertas_personal()
    assert len(alertas) == 1 and alertas[0]['tipo'] == 'seguridad'
    # Pasado el cooldown vuelve a avisar (patrulla siguiente).
    m.procesar(FRAME, _track('uuid-ana'), ahora=t0 + 400.0)
    assert len(m.drenar_alertas_personal()) == 1


def test_cliente_sigue_contando_junto_al_personal(tmp_path):
    m = _medidor_con_personal(
        tmp_path, {'uuid-luis': {'nombre': 'Luis', 'rol': 'pasillero'}})
    t0 = 1000.0
    tracks = {**_track('uuid-luis', tid=1),
              **_track('uuid-cliente', tid=2)}
    for i in range(6):                            # quieto frente al área
        m.procesar(FRAME, tracks, ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 1     # SOLO el cliente
    assert 1 in m.personal_tids                   # el empleado, etiquetado
    assert 2 not in m.personal_tids


# ── Uniformes (identifican de espaldas) ──────────────────────────────────
def _galeria_uniforme(tmp_path, monkeypatch, emb):
    gal = GaleriaPersonalTienda(raiz=tmp_path / 'personal')
    monkeypatch.setattr(gal, '_embedding_de_uniforme', lambda foto: emb)
    return gal


def test_uniforme_se_enrola_e_identifica_por_cuerpo(tmp_path, monkeypatch):
    gal = _galeria_uniforme(tmp_path, monkeypatch, _vec(5))
    ficha = gal.enrolar('Uniforme de seguridad', 'seguridad', FOTO,
                        tipo='uniforme')
    assert ficha['tipo'] == 'uniforme'
    # El MISMO uniforme (embedding cercano) identifica; otro no.
    assert gal.identificar_cuerpo(_vec(5))[1]['rol'] == 'seguridad'
    assert gal.identificar_cuerpo(_vec(6)) is None
    # Y NO contamina la vía de rostro (tipos separados).
    assert gal.identificar(_vec(5)) is None
    assert gal.tiene_uniformes()


def test_umbral_uniforme_es_estricto(tmp_path, monkeypatch):
    """Un cliente con ropa PARECIDA (coseno 0.7) no puede quedar excluido."""
    gal = _galeria_uniforme(tmp_path, monkeypatch, _vec(5))
    gal.enrolar('Uniforme pasillero', 'pasillero', FOTO, tipo='uniforme')
    parecido = 0.7 * _vec(5) + (1 - 0.7 ** 2) ** 0.5 * _vec(9)  # cos = 0.7
    assert gal.identificar_cuerpo(parecido) is None
    casi_igual = 0.9 * _vec(5) + (1 - 0.9 ** 2) ** 0.5 * _vec(9)  # cos = 0.9
    assert gal.identificar_cuerpo(casi_igual) is not None


def test_ficha_por_cuerpo_del_identificador(tmp_path, monkeypatch):
    gal = _galeria_uniforme(tmp_path, monkeypatch, _vec(5))
    gal.enrolar('Uniforme de seguridad', 'seguridad', FOTO, tipo='uniforme')
    ident = IdentificadorPersonal(galeria=gal)
    ficha = ident.ficha_por_cuerpo(_vec(5))
    assert ficha and ficha['via'] == 'uniforme'
    assert ficha['rol'] == 'seguridad'
    assert ident.ficha_por_cuerpo(_vec(6)) is None
    assert ident.ficha_por_cuerpo(None) is None


def test_track_marcado_por_uniforme_queda_excluido_en_anaqueles(tmp_path):
    """La vía del flag: el pipeline base marcó es_personal por uniforme
    (funciona con prov:, de espaldas) y el modo lo excluye y alerta."""
    m = _medidor_con_personal(tmp_path, {})       # galería de rostros VACÍA
    t0 = 1000.0
    tracks = {1: {'class': -1, 'box': list(BOX_EN_AREA),
                  'persistent_id': 'prov:1',      # sin rostro: solo uniforme
                  'es_personal': True,
                  'personal_rol': 'pasillero',
                  'personal_nombre': 'Uniforme pasillero'}}
    for i in range(10):
        m.procesar(FRAME, tracks, ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 0     # excluido del marketing
    assert m.registro.area(0)['agarres'] == 0
    alertas = m.drenar_alertas_personal()
    assert len(alertas) == 1 and alertas[0]['tipo'] == 'surte'
    assert alertas[0]['nombre'] == 'Uniforme pasillero'


def test_sin_galeria_todos_cuentan_como_clientes(tmp_path):
    """El fallo barato: sin identificador utilizable, nadie se excluye."""
    class IdentRoto:
        def ficha_de(self, pid, reid=None):
            raise RuntimeError('galería corrupta')

    cfg = dict(_DEFECTOS)
    m = MedidorAnaqueles(
        'cam-test', cfg, pose=PoseKlienta(),
        registro=RegistroAnaqueles('cam-test', raiz=tmp_path),
        escena=type('E', (), {'evaluar': lambda *a, **k: None})(),
        identificador=IdentRoto())
    m.poner_zonas([AREA], ['Área 1'])
    t0 = 1000.0
    for i in range(6):                            # quieto frente al área
        m.procesar(FRAME, _track('uuid-x'), ahora=t0 + i * 0.5)
    assert m.registro.area(0)['visitas'] == 1


# ── Surtido: la MUÑECA del personal dentro del área (19-oct-2026) ────────
class PoseFuera:
    disponible = True

    def munecas(self, frame, box):
        return [(150, 400, 0.9)]               # mano abajo, fuera del área


def test_el_surtido_se_vigila_solo_si_la_muneca_del_personal_entra(tmp_path):
    for pose, esperado in ((PoseKlienta(), [0]), (PoseFuera(), [])):
        m = _medidor_con_personal(
            tmp_path, {'uuid-luis': {'nombre': 'Luis', 'rol': 'pasillero'}})
        m.pose = pose
        llamadas = []
        m._stock.tras_personal = lambda area, t: llamadas.append(area)
        m.procesar(FRAME, _track('uuid-luis'), ahora=1000.0)
        m.procesar(FRAME, _track('uuid-luis'), ahora=1000.1)   # < 0,5 s: sin pose
        assert llamadas == esperado, type(pose).__name__


def test_surte_pide_que_la_muneca_entre_no_basta_el_cuerpo(tmp_path):
    """19-oct-2026: de pie delante del área con la mano abajo no «surte»."""
    m = _medidor_con_personal(
        tmp_path, {'uuid-luis': {'nombre': 'Luis', 'rol': 'pasillero'}},
        surte_dwell_s=3.0)
    m.pose = PoseFuera()
    for i in range(10):                        # 5 s frente al anaquel
        m.procesar(FRAME, _track('uuid-luis'), ahora=1000.0 + i * 0.5)
    assert m.drenar_alertas_personal() == []


def test_vigilante_y_mantenimiento_del_modelo_no_surten(tmp_path):
    """19-oct-2026 (lo pidió el usuario): solo dejan de contar."""
    for rol in ('vigilante', 'mantenimiento'):
        m = _medidor_con_personal(tmp_path, {'uuid-x': {'nombre': rol, 'rol': rol}},
                                  surte_dwell_s=3.0)
        llamadas = []
        m._stock.tras_personal = lambda area, t: llamadas.append(area)
        for i in range(10):
            m.procesar(FRAME, _track('uuid-x'), ahora=1000.0 + i * 0.5)
        assert m.drenar_alertas_personal() == [] and llamadas == [], rol
        assert m.registro.area(0)['visitas'] == 0 and m.registro.area(0)['agarres'] == 0


def test_gesto_pendiente_de_quien_resulta_personal_se_descarta(tmp_path):
    """19-oct-2026, 16:36:47: el gesto llegó un segundo antes de la marca."""
    m = _medidor_con_personal(tmp_path, {})
    llamadas = []
    m._stock.tras_personal = lambda area, t: llamadas.append(area)
    m._agarres_por_confirmar = [{'idx': 0, 'pid': 'prov:7', 'tid': 7, 't': 99.0,
                                 'genero': None, 'rango': None},
                                {'idx': 0, 'pid': 'prov:8', 'tid': 8, 't': 99.0,
                                 'genero': None, 'rango': None}]
    m.personal_tids = {7: {'rol': 'pasillero'}}
    m._gestos_de_personal(100.0)
    assert [a['tid'] for a in m._agarres_por_confirmar] == [8]   # el cliente sigue
    assert llamadas == [0]
