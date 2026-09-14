"""
Pruebas del modo `PermanenciaArea` (cronómetro de personas, área y puerta).

No instancia la clase completa (cargaría YOLO y pediría GPU): se construye un
doble que hereda de `PermanenciaArea` sin `__init__` y llama al mismo
`_iniciar_permanencia` que usa producción, para que el estado del doble no se
desvíe del real. Se le inyecta el `_zone_state` y los `active_tracks` tal como
los deja `TiendasProcess`.

Lo que más se protege:

  - **el cronómetro va por `persistent_id`, no por track** — cronometrando
    tracks, alguien que pasa por detrás de una columna sale con tres estancias
    cortas en vez de una larga, y el umbral no salta nunca;
  - **en modo puerta la estancia sobrevive a la desaparición** — es la razón
    de ser del modo: quien entra no se ve, y el reloj no puede pararse;
  - **lo que no se sabe se dice** — una salida sin emparejar se reporta como
    tal, sin tiempo de permanencia inventado.

El doble desactiva fotos en disco y persistencia: las pruebas no deben dejar
JPEG ni JSON en `server/output/`.
"""
from __future__ import annotations

import contextlib
import sys
import time as _t
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.permanencia_area import (  # noqa: E402
    MODO, MODO_AREA, MODO_PUERTA, PermanenciaArea, formato_duracion)
from src.analityc.core.permanencia_estancias import (  # noqa: E402
    POR_CUERPO, POR_ZONA, SIN_ATRIBUIR)
from src.analityc.core.tiendas import (  # noqa: E402
    TiendasProcess)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
CAJA = [10, 10, 60, 200]

# Un martes a las 12:00 en hora LOCAL (America/Caracas en el servidor), lejos
# de cualquier medianoche. Las pruebas que siembran estancias «hace N horas»
# se congelan aquí (4-sep-2026) para que lo que cuenta el TTL y la hora del
# cruce no dependan de a qué hora corre la suite.
MARTES_MEDIODIA = _t.mktime((2026, 9, 8, 12, 0, 0, 0, 0, -1))


@contextlib.contextmanager
def _reloj_congelado(instante: float = MARTES_MEDIODIA):
    """Fija `time.time()` en `instante` para producción y para el test.

    `permanencia_estancias` y `permanencia_area` hacen `import time` y llaman
    `time.time()` al vencer estancias y al fechar alertas, así que basta con
    parchear el atributo del módulo. Es un context manager y no un fixture
    para que el bloque `__main__` del final siga pudiendo ejecutar las
    pruebas sin pytest.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_t, 'time', lambda: instante)
        yield instante


class _Doble(PermanenciaArea):
    """`PermanenciaArea` sin cargar modelos, sin disco y sin cámara."""

    def __init__(self, **over):
        from src.analityc.core.permanencia_area import _DEFECTOS
        cfg = {**_DEFECTOS,
               'guardar_fotos': False,        # nada de JPEG en output/
               'persistir_estancias': False,  # nada de JSON en output/
               'estancia_ttl_horas': 100}     # el TTL se prueba aparte
        cfg.update(over)
        self._zone_state = {}
        self.active_tracks = {}
        self._alertas_pendientes = []
        self._camera_display_name = 'cam-test'
        self._iniciar_permanencia(cfg, 'cam-test')

    # -- Modo área: se manipula la máquina de zonas del padre -------------
    def dentro(self, pid, desde):
        self._zone_state[pid] = {'estado': 'DENTRO', 'entry_time': desde}
        self.active_tracks[pid] = {'persistent_id': pid, 'box': CAJA,
                                   'persistent_confirmed': True}

    def fuera(self, pid):
        self._zone_state[pid] = {'estado': 'FUERA', 'entry_time': None}
        self.active_tracks.pop(pid, None)

    # -- Modo puerta: se manipulan los tracks vivos -----------------------
    def ve(self, pid, en_zona, t):
        self.active_tracks = {pid: {'persistent_id': pid, 'box': CAJA,
                                    'in_zone': en_zona,
                                    'persistent_confirmed': True}}
        self._ciclo_puerta(FRAME, t)

    def ve_en(self, pid, en_zona, t, x):
        """Como `ve`, pero con la persona en una posición concreta."""
        self.active_tracks = {pid: {'persistent_id': pid,
                                    'box': [x, 50, x + 60, 250],
                                    'in_zone': en_zona,
                                    'persistent_confirmed': True}}
        self._ciclo_puerta(FRAME, t)

    def no_ve_a_nadie(self, t):
        self.active_tracks = {}
        self._ciclo_puerta(FRAME, t)

    def alertas(self):
        a = list(self._alertas_pendientes)
        self._alertas_pendientes = []
        return a


def _tipos(alertas):
    return [a['tipo_permanencia'] for a in alertas]


# ── Formato ─────────────────────────────────────────────────────────────
def test_duracion_legible():
    assert formato_duracion(0) == '0 s'
    assert formato_duracion(45) == '45 s'
    assert formato_duracion(60) == '1 min'
    assert formato_duracion(75) == '1 min 15 s'
    assert formato_duracion(3600) == '1 h'
    assert formato_duracion(5400) == '1 h 30 min'


# ── Contrato con el servidor y los clientes ─────────────────────────────
def test_es_subclase_de_tiendas_process():
    """`app.py` despacha por isinstance."""
    assert issubclass(PermanenciaArea, TiendasProcess)


def test_el_modo_esta_registrado_en_el_servidor():
    import inspect
    from src.app.app import _build_processor
    assert MODO in inspect.getsource(_build_processor)


def test_los_clientes_ofrecen_el_modo():
    for cliente in ('amazonas', 'detecciones'):
        ruta = (RAIZ / 'clients' / cliente / 'src' / 'gui' / 'components' /
                'custom_status_bar.py')
        assert MODO in ruta.read_text(encoding='utf-8'), (
            f"el selector de {cliente} no ofrece {MODO}")


def test_al_desconectar_el_libro_se_vuelca_a_disco():
    """El guardado normal va limitado a uno cada 5 s.

    Sin volcado al cerrar, apagar el cliente justo después de que alguien
    entre por la puerta perdería esa estancia: al reconectar, esa persona ya
    no estaría dentro y su salida saldría como «no atribuida».
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = _Doble(persistir_estancias=True,
                   carpeta_evidencia=tmp, guardar_fotos=False)
        p.dentro('p1', desde=_t.time())
        p._actualizar(FRAME)
        p.dentro('p2', desde=_t.time())      # dentro de los 5 s: no se guarda
        p._actualizar(FRAME)

        p.cerrar_permanencia()

        from src.analityc.core.permanencia_estancias import RegistroEstancias
        otro = RegistroEstancias(ruta=p._registro.ruta, ttl_horas=100)
        assert otro.cargar() == 2, 'se perdió una estancia al cerrar'


def test_el_servidor_cierra_los_procesadores_POR_CAMARA():
    """El cierre tiene el mismo agujero que tenía el procesado.

    `app.py` cerraba solo el objeto que construyó, que es justo el que nunca
    ve un frame; el que tiene el estado vivo es el de `get_camera_processor`.
    """
    import inspect
    from src.app import app as _app
    fuente = inspect.getsource(_app._cerrar_procesadores)
    assert '_camera_processors' in fuente
    assert 'cerrar_permanencia' in _app._METODOS_DE_CIERRE
    # Cerrar la jornada del stand en cada desconexión partiría el día.
    assert 'cerrar_sesion_stand' not in _app._METODOS_DE_CIERRE


def test_el_procesador_POR_CAMARA_conserva_el_modo():
    """El fallo que dejaba el modo mudo sin que se notara.

    `app.py` no procesa el frame con el objeto que construyó, sino con el que
    devuelve `get_camera_processor(camera_id)`. Ese método instanciaba
    `TiendasProcess` con la clase incrustada, así que en un modo derivado
    salía el PADRE: se veía el vídeo y las detecciones, y no se emitía un
    solo evento del modo. Se prueba sobre el código porque instanciarlo de
    verdad cargaría YOLO y pediría GPU.
    """
    import inspect
    fuente = inspect.getsource(TiendasProcess.get_camera_processor)
    assert 'cam_proc = type(self)(' in fuente
    assert 'cam_proc = TiendasProcess(' not in fuente, (
        'vuelve a fabricar el padre: los modos derivados quedan mudos')


def test_la_alerta_tiene_el_formato_que_esperan_whatsapp_y_jarvis():
    p = _Doble()
    p.dentro('p1', desde=_t.time())
    p._actualizar(FRAME)
    a = p.alertas()[0]
    # Claves que leen render_box.py (Jarvis) y whatsapp_alertas.py.
    for clave in ('event_type', 'class_name', 'description', 'timestamp',
                  'camera_name', 'person_uuid', 'crop_image', 'image_base64'):
        assert clave in a, f"falta {clave}, que el reenviador espera"


def test_la_alerta_deduplica_POR_PERSONA_no_por_camara():
    """WhatsApp y Jarvis agrupan por (cámara, evento) si nadie dice otra cosa.

    Sin esto, dos personas entrando con pocos segundos de diferencia darían
    un solo aviso, y el tiempo de la segunda no llegaría a nadie.
    """
    ahora = _t.time()
    p = _Doble()
    p.dentro('p1', desde=ahora)
    p.dentro('p2', desde=ahora)
    p._actualizar(FRAME)
    claves = [a['dedup_clave'] for a in p.alertas()]
    assert len(claves) == 2 and len(set(claves)) == 2
    assert all(c.startswith('perm|cam-test|entrada|') for c in claves)


def test_el_video_marca_a_quien_esta_dentro_con_su_cronometro():
    """El cliente pinta `label` tal cual: el tiempo tiene que venir ahí."""
    p = _Doble(alertar_entrada=False)
    p.dentro('p1', desde=_t.time() - 90)
    # El tracker numera con enteros; el atajo del doble usa el pid de clave.
    p.active_tracks = {7: {'persistent_id': 'p1', 'box': CAJA, 'class': -1,
                           'category': 'Personas', 'confidence': 0.9,
                           'persistent_confirmed': True}}
    p._actualizar(FRAME)
    detecciones = p._build_detections()
    assert detecciones, 'no hay detecciones que marcar'
    assert 'dentro' in detecciones[0]['label']
    assert detecciones[0]['permanencia_s'] >= 89


def test_las_alertas_van_a_la_columna_de_personas():
    """Sin `clase_gruesa` caerían en «Eventos», con los pedidos de Hummus."""
    p = _Doble()
    p.dentro('p1', desde=_t.time())
    p._actualizar(FRAME)
    assert p.alertas()[0]['clase_gruesa'] == 'persona'


# ── Modo ÁREA ───────────────────────────────────────────────────────────
def test_entrada_genera_alerta():
    p = _Doble()
    # Marca RECIENTE: con un timestamp antiguo la persona llevaría décadas
    # dentro y saltaría también la alerta de permanencia (lo cazó la primera
    # versión de este test, que usaba 1000.0 = época Unix).
    p.dentro('p1', desde=_t.time())
    p._actualizar(FRAME)
    a = p.alertas()
    assert _tipos(a) == ['entrada']
    assert a[0]['person_uuid'] == 'p1'
    assert a[0]['crop_image'], "la alerta salió sin foto (WhatsApp la tira)"


# ── Filtro de roces: confirmar la entrada (11-ago-2026) ─────────────────
def test_un_roce_no_avisa_ni_entrada_ni_salida():
    """Quien toca el área y se va al instante no genera NADA."""
    p = _Doble(confirmar_entrada_s=4)
    p.dentro('p-roce', desde=_t.time())
    p._actualizar(FRAME)
    assert p.alertas() == [], 'la entrada debe quedar retenida'
    p.fuera('p-roce')
    p._actualizar(FRAME)
    assert p.alertas() == [], 'el roce no puede avisar nada'


def test_la_entrada_confirmada_llega_con_su_hora_real():
    """Confirmar filtra, no atrasa: la alerta sale con la hora de entrada."""
    from src.analityc.core.permanencia_area import _hora
    p = _Doble(confirmar_entrada_s=4)
    inicio = _t.time() - 10                 # lleva 10 s dentro: confirma ya
    p.dentro('p-real', desde=inicio)
    p._actualizar(FRAME)
    a = p.alertas()
    assert _tipos(a) == ['entrada']
    assert _hora(inicio) in a[0]['description']


def test_la_visita_corta_pero_valida_avisa_al_salir():
    """Sale ANTES de confirmar pero pasó `min_estancia`: entrada + salida."""
    p = _Doble(confirmar_entrada_s=30, min_estancia_para_avisar_s=3)
    p.dentro('p-visita', desde=_t.time() - 8)
    p._actualizar(FRAME)
    assert p.alertas() == []
    p.fuera('p-visita')
    p._actualizar(FRAME)
    assert _tipos(p.alertas()) == ['entrada', 'salida']


def test_sin_confirmacion_la_entrada_avisa_al_instante():
    """`confirmar_entrada_s: 0` = comportamiento clásico intacto."""
    p = _Doble()
    p.dentro('p-cero', desde=_t.time())
    p._actualizar(FRAME)
    assert _tipos(p.alertas()) == ['entrada']


def test_salida_lleva_el_tiempo_total():
    p = _Doble()
    inicio = _t.time() - 90          # entró hace 90 s
    p.dentro('p1', desde=inicio)
    p._actualizar(FRAME)
    p.alertas()
    p.fuera('p1')
    p._actualizar(FRAME)
    a = p.alertas()
    assert _tipos(a) == ['salida']
    # Formato pedido el 10-ago-2026: Inicio / Fin / Tiempo total en HH:MM:SS.
    assert 'Inicio: ' in a[0]['description']
    assert 'Fin: ' in a[0]['description']
    assert 'Tiempo total: 00:01:' in a[0]['description']
    assert 88 <= a[0]['duracion_s'] <= 92
    assert a[0]['duracion_legible'] == formato_duracion(a[0]['duracion_s'])


def test_la_salida_lleva_tambien_la_foto_de_la_entrada():
    """Es lo que deja comprobar que es la misma persona que entró."""
    p = _Doble()
    p.dentro('p1', desde=_t.time() - 90)
    p._actualizar(FRAME)
    p.alertas()
    p.fuera('p1')
    p._actualizar(FRAME)
    a = p.alertas()[0]
    assert a['entrada_base64'], 'se perdió la foto de entrada'
    assert a['image_base64'], 'la alerta de salida salió sin imagen'


def test_estancia_corta_no_avisa_al_salir():
    """Quien solo cruza el área no debe mandar un WhatsApp."""
    p = _Doble(alertar_entrada=False, min_estancia_para_avisar_s=3)
    p.dentro('p1', desde=_t.time() - 1)      # 1 segundo dentro
    p._actualizar(FRAME)
    p.fuera('p1')
    p._actualizar(FRAME)
    assert p.alertas() == []


def test_permanencia_prolongada_avisa_una_vez():
    p = _Doble(alertar_entrada=False, umbral_permanencia_s=60,
               realertar_cada_s=0)
    p.dentro('p1', desde=_t.time() - 120)    # lleva 2 min
    p._actualizar(FRAME)
    assert _tipos(p.alertas()) == ['permanencia']
    # Sigue dentro, pero `realertar_cada_s: 0` = una sola vez.
    p._actualizar(FRAME)
    assert p.alertas() == []


def test_permanencia_no_avisa_antes_del_umbral():
    p = _Doble(alertar_entrada=False, umbral_permanencia_s=300)
    p.dentro('p1', desde=_t.time() - 10)
    p._actualizar(FRAME)
    assert p.alertas() == []


def test_umbral_cero_desactiva_el_aviso_de_permanencia():
    p = _Doble(alertar_entrada=False, umbral_permanencia_s=0)
    p.dentro('p1', desde=_t.time() - 99999)
    p._actualizar(FRAME)
    assert p.alertas() == []


def test_el_cronometro_sigue_a_la_PERSONA_no_al_track():
    """Con oclusión, el pipeline reasigna el mismo persistent_id.

    Si cronometráramos tracks, esto serían dos estancias cortas y el umbral
    no saltaría nunca.
    """
    inicio = _t.time() - 200
    p = _Doble(alertar_entrada=False, umbral_permanencia_s=120)
    p.dentro('p1', desde=inicio)
    p._actualizar(FRAME)
    p.alertas()

    # El tracker pierde el track pero la persona sigue DENTRO para el padre
    # (la histéresis no la saca), y vuelve con otro tracker_id.
    p.active_tracks.pop('p1', None)
    p._actualizar(FRAME)
    p.active_tracks['p1'] = {'persistent_id': 'p1', 'box': [20, 20, 70, 210]}
    r = p._actualizar(FRAME)

    assert r['dentro_ahora'] == 1
    assert r['personas'][0]['segundos'] >= 200, "el cronómetro se reinició"


def test_varias_personas_a_la_vez():
    ahora = _t.time()
    p = _Doble(alertar_entrada=False)
    p.dentro('p1', desde=ahora - 300)
    p.dentro('p2', desde=ahora - 30)
    r = p._actualizar(FRAME)
    assert r['dentro_ahora'] == 2
    # Ordenadas por permanencia descendente: primero quien lleva más.
    assert r['personas'][0]['persistent_id'] == 'p1'
    assert r['personas'][1]['persistent_id'] == 'p2'


def test_resumen_acumula_visitas_y_media():
    p = _Doble(alertar_entrada=False, alertar_salida=False)
    for pid, hace in (('p1', 100), ('p2', 200)):
        p.dentro(pid, desde=_t.time() - hace)
        p._actualizar(FRAME)
        p.fuera(pid)
        r = p._actualizar(FRAME)
    assert r['visitas_completadas'] == 2
    assert 140 <= r['permanencia_media_s'] <= 160    # media de 100 y 200


def test_un_zone_state_vacio_no_revienta():
    p = _Doble()
    r = p._actualizar(FRAME)
    assert r['dentro_ahora'] == 0 and r['personas'] == []
    assert r['permanencia_media_s'] is None      # None, no 0: no hay dato


# ── Modo PUERTA ─────────────────────────────────────────────────────────
def _entra_por_la_puerta(p, pid='p1', t0=1000.0):
    """Se acerca, pisa el vano y desaparece dentro."""
    t = t0
    for en_zona in (False, False, False, False, True, True, True, True):
        t += 0.1
        p.ve(pid, en_zona, t)
    p.no_ve_a_nadie(t + 3.0)
    return t + 3.0


def _sale_por_la_puerta(p, pid='p1', t0=2000.0):
    """Aparece en el vano y se aleja."""
    t = t0
    for en_zona in (True, True, True, False, False, False):
        t += 0.1
        p.ve(pid, en_zona, t)
    return t


def test_en_modo_puerta_entrar_abre_la_estancia_y_avisa():
    p = _Doble(modo='puerta')
    assert p.modo_permanencia == MODO_PUERTA
    _entra_por_la_puerta(p)
    a = p.alertas()
    assert _tipos(a) == ['entrada']
    assert a[0]['modo_permanencia'] == MODO_PUERTA
    assert a[0]['crop_image'], 'la entrada salió sin foto'
    assert 'p1' in p._registro


def test_la_estancia_SOBREVIVE_a_que_la_persona_desaparezca():
    """Es la razón de ser del modo: dentro no se ve, y el reloj no para."""
    p = _Doble(modo='puerta', alertar_entrada=False)
    _entra_por_la_puerta(p)
    for _ in range(50):                      # 50 frames sin nadie a la vista
        p.no_ve_a_nadie(_t.time())
    assert len(p._registro) == 1, 'se perdió a quien está dentro'


def test_salir_cierra_la_estancia_con_el_tiempo_total():
    p = _Doble(modo='puerta', alertar_entrada=False)
    t_entrada = 1000.0
    _entra_por_la_puerta(p, t0=t_entrada)
    p.alertas()
    _sale_por_la_puerta(p, t0=t_entrada + 7200)      # dos horas dentro
    a = p.alertas()
    assert _tipos(a) == ['salida']
    assert a[0]['duracion_s'] > 7000
    assert 'h' in a[0]['duracion_legible']
    assert 'p1' not in p._registro


def test_una_salida_sin_pareja_se_reporta_SIN_inventar_tiempo():
    p = _Doble(modo='puerta', atribuir_si_unica=True)
    _sale_por_la_puerta(p, pid='desconocido')        # nadie había entrado
    a = p.alertas()
    assert _tipos(a) == ['no_atribuida']
    assert a[0]['duracion_s'] is None, 'se inventó una permanencia'
    assert a[0]['identidad_por'] == SIN_ATRIBUIR
    assert p._resumen(_t.time())['salidas_sin_atribuir'] == 1


def test_con_una_sola_persona_dentro_la_salida_se_le_atribuye():
    """Quien sale solo puede ser esa. Se dice que fue por descarte de zona."""
    p = _Doble(modo='puerta', alertar_entrada=False, atribuir_si_unica=True)
    _entra_por_la_puerta(p, pid='p1', t0=1000.0)
    _sale_por_la_puerta(p, pid='otro-id-distinto', t0=1600.0)
    a = p.alertas()
    assert _tipos(a) == ['salida']
    assert a[0]['identidad_por'] == POR_ZONA
    assert a[0]['duracion_s'] > 500


def test_sin_atribuir_si_unica_no_se_adivina():
    p = _Doble(modo='puerta', alertar_entrada=False, atribuir_si_unica=False)
    _entra_por_la_puerta(p, pid='p1', t0=1000.0)
    _sale_por_la_puerta(p, pid='otro-id-distinto', t0=1600.0)
    assert _tipos(p.alertas()) == ['no_atribuida']
    assert 'p1' in p._registro, 'cerró la estancia de otra persona'


def test_se_puede_callar_la_alerta_no_atribuida():
    p = _Doble(modo='puerta', alertar_no_atribuida=False)
    _sale_por_la_puerta(p, pid='desconocido')
    assert p.alertas() == []


def test_quien_pasa_de_largo_por_delante_de_la_puerta_no_entra():
    p = _Doble(modo='puerta')
    t = 1000.0
    for en_zona in (False, True, True, False, False, False):
        t += 0.1
        p.ve('p1', en_zona, t)
    p.no_ve_a_nadie(t + 3.0)
    assert p.alertas() == []
    assert len(p._registro) == 0


# ── El estado que el padre restaura al salir ────────────────────────────
#
# Encontrado en producción el 7-ago-2026, y es LA razón de que no detectara
# nada. `TiendasProcess` multiplexa cámaras sobre un mismo objeto: `_push_state`
# mueve el estado de la cámara a `self` al entrar en `process_frame`, y
# `_pop_state` **restaura el de antes** al salir. `active_tracks` y
# `_zone_state` están en esa lista. Este módulo trabaja DESPUÉS, así que leía
# dos diccionarios vacíos que no eran de nadie.
#
# Lo delató el aforo: `_zone_count` NO está en la lista de intercambio, así que
# sobrevivía al pop y marcaba 1 mientras `_zone_state` estaba vacío.

def test_el_cronometro_lee_los_tracks_de_SU_camara():
    p = _Doble()
    # Así queda el objeto tras `_pop_state`: los atributos, vacíos; el estado
    # real de la cámara, en `camera_states`.
    reales = {7: {'persistent_id': 'p1', 'box': CAJA, 'in_zone': True,
                  'persistent_confirmed': True}}
    zonas = {'p1': {'estado': 'DENTRO', 'entry_time': _t.time() - 30}}
    p.active_tracks = {}
    p._zone_state = {}
    p.camera_states = {'cam1': {'active_tracks': reales, '_zone_state': zonas}}

    p._recuperar_estado_de_camara('cam1')

    assert p.active_tracks is reales, 'no son copias: es el MISMO diccionario'
    assert p._zone_state is zonas


def test_sin_ese_rescate_el_cronometro_no_ve_a_nadie():
    """La prueba de que el fallo era ese y no otro."""
    p = _Doble()
    p.active_tracks = {}
    p._zone_state = {}
    p.camera_states = {'cam1': {
        'active_tracks': {7: {'persistent_id': 'p1', 'box': CAJA,
                              'in_zone': True}},
        '_zone_state': {'p1': {'estado': 'DENTRO',
                               'entry_time': _t.time() - 30}}}}
    assert p._actualizar(FRAME)['dentro_ahora'] == 0      # sin rescatar
    p._recuperar_estado_de_camara('cam1')
    assert p._actualizar(FRAME)['dentro_ahora'] == 1      # rescatado


def test_un_estado_de_camara_ausente_no_revienta():
    p = _Doble()
    p.camera_states = {}
    p._recuperar_estado_de_camara('la-que-sea')
    p.camera_states = {'cam1': {'active_tracks': None, '_zone_state': 'basura'}}
    p._recuperar_estado_de_camara('cam1')
    assert isinstance(p.active_tracks, dict)


def test_el_rescate_ocurre_ANTES_de_actualizar():
    """Si se llama después, se lee el estado vacío y no sirve de nada."""
    import inspect
    fuente = inspect.getsource(PermanenciaArea.process_frame)
    assert (fuente.index('_recuperar_estado_de_camara')
            < fuente.index('self._actualizar('))


def test_el_stand_tiene_el_mismo_rescate():
    """Compartían el fallo: `stand.db` seguía vacía por esto."""
    import inspect
    from src.analityc.core.amazonas_stand import AmazonasStand
    fuente = inspect.getsource(AmazonasStand)
    assert '_tracks_de' in fuente and '_get_cam_state' in fuente
    assert 'self._puente.procesar(\n            image, self._tracks_de(' in fuente


# ── Cronometrar a quien no muestra la cara ──────────────────────────────
#
# Se vio en producción el 6-ago-2026: cámara funcionando, personas detectadas
# y capturadas, y el modo sin emitir NADA. Causa: la máquina DENTRO/FUERA del
# padre solo admite identidades confirmadas por rostro, y esa cámara no sacaba
# ninguna (`galeria_total: 0`). Sin un error en el log: solo silencio.

def test_por_defecto_se_cronometra_a_quien_no_tiene_rostro():
    """Medir una permanencia no necesita identidad única global."""
    p = _Doble()
    assert p.cfg_permanencia['contar_sin_rostro'] is True
    assert p._zone_requiere_confirmado is False


def test_el_padre_respeta_la_politica_de_ESTA_instancia():
    """El flag es por instancia: Tienda sigue exigiendo rostro."""
    from src.analityc.core.analytics.config import AnalyticsConfig
    import inspect
    fuente = inspect.getsource(TiendasProcess._process_zone_transitions)
    assert "getattr(self, '_zone_requiere_confirmado'" in fuente
    # Y el defecto global no se ha tocado.
    assert AnalyticsConfig.ZONE_ENTRY_REQUIRES_CONFIRMED is True


def test_una_persona_provisional_arranca_el_cronometro():
    p = _Doble()
    ahora = _t.time()
    # Así la deja el padre cuando no hay rostro: pid 'prov:<track>'.
    p._zone_state['prov:41'] = {'estado': 'DENTRO', 'entry_time': ahora - 45}
    p.active_tracks = {41: {'persistent_id': 'prov:41', 'box': CAJA,
                            'in_zone': True, 'persistent_confirmed': False}}
    r = p._actualizar(FRAME)
    a = p.alertas()
    assert _tipos(a) == ['entrada']
    assert a[0]['crop_image'], 'sin caja no habría foto'
    assert r['dentro_ahora'] == 1


def test_con_contar_sin_rostro_apagado_se_vuelve_al_comportamiento_estricto():
    p = _Doble(contar_sin_rostro=False)
    assert p._zone_requiere_confirmado is True
    p.active_tracks = {41: {'persistent_id': 'prov:41', 'box': CAJA,
                            'in_zone': True, 'persistent_confirmed': False}}
    assert p._cajas_por_persona() == {}, 'un provisional no debería contar'


def test_en_puerta_los_provisionales_cuentan_pase_lo_que_pase():
    """Nadie muestra la cara mientras cruza un umbral."""
    p = _Doble(modo='puerta', contar_sin_rostro=False)
    p.active_tracks = {41: {'persistent_id': 'prov:41', 'box': CAJA,
                            'in_zone': True, 'persistent_confirmed': False}}
    assert 'prov:41' in p._cajas_por_persona()


# ── La flecha de dirección ──────────────────────────────────────────────
#
# Llega como atributo (`app.py` la fija junto al nombre de la cámara) para no
# cambiar la firma de `process_frame`, que comparte toda la familia.

DERECHA = [[0, 0], [100, 0]]        # la puerta está a la derecha


def test_la_flecha_del_recuadro_llega_al_detector():
    p = _Doble(modo='puerta')
    p._flecha_direccion = DERECHA
    t, x = 1000.0, 100
    for _ in range(8):
        t += 0.1
        x += 40
        p.ve_en('p1', True, t, x)    # SIEMPRE en el vano: sin flecha, ambiguo
    p.no_ve_a_nadie(t + 3.0)
    a = p.alertas()
    assert _tipos(a) == ['entrada'], 'la flecha no llegó: se quedó callado'
    assert a[0]['direccion_por'] == 'flecha'
    assert a[0]['direccion_avance'] > 0.5


def test_sin_flecha_la_alerta_dice_que_fue_por_el_ciclo_de_vida():
    """Que sea una medida o una inferencia cambia cuánto fiarse."""
    p = _Doble(modo='puerta')
    _entra_por_la_puerta(p)
    a = p.alertas()
    assert a[0]['direccion_por'] == 'ciclo_de_vida'
    assert a[0]['direccion_avance'] is None


def test_una_flecha_al_reves_evita_contar_una_entrada_que_no_fue():
    p = _Doble(modo='puerta')
    p._flecha_direccion = [[100, 0], [0, 0]]     # la puerta, a la izquierda
    t, x = 1000.0, 100
    for en_zona in (False, False, False, False, True, True, True, True):
        t += 0.1
        x += 40                                   # se aleja de la puerta
        p.ve_en('p1', en_zona, t, x)
    p.no_ve_a_nadie(t + 3.0)
    assert p.alertas() == []
    assert len(p._registro) == 0


# ── El árbitro (VLM fuera del bucle) ────────────────────────────────────
def test_el_arbitro_viene_APAGADO():
    """Sin activarlo, el modelo ni se carga."""
    p = _Doble(modo='puerta')
    assert p._arbitro.activo is False
    assert p._detector_puerta.cfg['emitir_ambiguos'] is False


def test_encender_el_arbitro_enciende_los_cruces_ambiguos():
    """Un árbitro sin cruces que arbitrar no sirve de nada."""
    p = _Doble(modo='puerta', arbitro_vlm=True)
    try:
        assert p._arbitro.activo is True
        assert p._detector_puerta.cfg['emitir_ambiguos'] is True
    finally:
        p._arbitro.detener()


def test_un_veredicto_del_arbitro_abre_la_estancia_con_la_hora_del_CRUCE():
    """No con la de la respuesta: el reloj cuenta desde que la persona pasó.

    Reloj congelado (4-sep-2026): el veredicto llega a las 12:00 y el cruce
    fue a las 11:50. Si la estancia se abriera con la hora de la respuesta,
    `entrada_ts` valdría `ahora` y no `cruce_ts`; con el reloj libre ambos
    valores se parecían tanto que la diferencia era difícil de leer.
    """
    from src.analityc.core.permanencia_arbitro import Veredicto
    with _reloj_congelado() as ahora:
        p = _Doble(modo='puerta', arbitro_vlm=True)
        try:
            cruce_ts = ahora - 600           # cruzó hace 10 minutos
            p._arbitro._veredictos.append(
                Veredicto(clave='p1', tipo='entrada', ts=cruce_ts,
                          seguridad='alta'))
            p._aplicar_veredictos(FRAME)
            a = p.alertas()
            assert _tipos(a) == ['entrada']
            assert a[0]['direccion_por'] == 'vlm'
            assert a[0]['arbitro_seguridad'] == 'alta'
            assert p._registro.get('p1').entrada_ts == cruce_ts
            assert p._registro.get('p1').entrada_ts != ahora, (
                'se abrió con la hora de la respuesta, no la del cruce')
        finally:
            p._arbitro.detener()


def test_con_el_arbitro_apagado_no_se_recogen_veredictos():
    p = _Doble(modo='puerta')
    _entra_por_la_puerta(p)              # no debe intentar arbitrar nada
    assert p._arbitro.recoger() == []


# ── TTL: la estancia que nunca se cierra ────────────────────────────────
def test_una_estancia_vencida_se_cierra_diciendo_que_no_se_vio_salir():
    """Con TTL de 1 h, quien entró a las 10:00 vence a las 12:00.

    El TTL compara deltas de `time.time()` (sin jornadas ni fechas), así que
    el reloj se congela a mediodía (4-sep-2026) solo para que «hace 2 h» sea
    un instante concreto y reproducible y no el que toque al correr la suite.
    """
    with _reloj_congelado() as ahora:
        p = _Doble(alertar_entrada=False, umbral_permanencia_s=0,
                   estancia_ttl_horas=1)
        p.dentro('p1', desde=ahora - 2 * 3600)   # entró a las 10:00
        p._actualizar(FRAME)     # la abre y, en el mismo frame, la vence
        a = p.alertas()
    assert _tipos(a) == ['sin_salida']
    assert a[0]['duracion_es_minima'] is True, 'la duración es una cota, no un dato'
    assert a[0]['identidad_por'] == SIN_ATRIBUIR
    assert 'nunca se registró su salida' in a[0]['description']
    assert a[0]['duracion_s'] >= 2 * 3600 - 1, 'la cota mínima son las 2 h'


# ── Configuración ───────────────────────────────────────────────────────
def test_el_modo_por_defecto_es_area():
    assert _Doble().modo_permanencia == MODO_AREA


def test_sin_nombre_de_camara_no_se_escribe_ningun_libro():
    """Se vio en producción: cada conexión dejaba un `sin-camara.json` vacío.

    El procesador se construye al conectar el cliente, antes del primer frame
    y por tanto antes de saber de qué cámara se trata. Ese libro no es de
    nadie: no debe existir.
    """
    import tempfile
    from src.analityc.core.permanencia_area import _DEFECTOS
    with tempfile.TemporaryDirectory() as tmp:
        p = _Doble.__new__(_Doble)
        p._zone_state, p.active_tracks, p._alertas_pendientes = {}, {}, []
        p._iniciar_permanencia(
            {**_DEFECTOS, 'guardar_fotos': False, 'persistir_estancias': True,
             'carpeta_evidencia': tmp}, camara='')
        p._registro.guardar(forzar=True)
        assert p._registro.ruta is None
        assert not (Path(tmp) / 'estancias').exists()


def test_conocer_el_nombre_de_la_camara_no_borra_lo_contado():
    """`app.py` fija el nombre justo antes del primer frame.

    El modo se remonta entonces para poder aplicar los ajustes de ESA cámara.
    Es un renombrado del mismo `camera_id`, no otra cámara: reiniciar los
    contadores ahí perdería las visitas ya cerradas.
    """
    p = _Doble(alertar_entrada=False, alertar_salida=False)
    p.dentro('p1', desde=_t.time() - 100)
    p._actualizar(FRAME)
    p.fuera('p1')
    p._actualizar(FRAME)
    assert p._resumen(_t.time())['visitas_completadas'] == 1

    p._camera_display_name = 'Puerta Almacen'
    p._actualizar(FRAME)                     # dispara `_asegurar_camara`
    assert p._camara_permanencia == 'Puerta Almacen'
    assert p._resumen(_t.time())['visitas_completadas'] == 1


def test_una_camara_puede_tener_su_propio_modo(tmp_path=None):
    """`camaras:` del YAML manda sobre lo global."""
    import tempfile
    from src.analityc.core import permanencia_area as pa
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'permanencia_config.yaml'
        ruta.write_text(
            "modo: area\n"
            "umbral_permanencia_s: 300\n"
            "camaras:\n"
            "  \"Puerta Almacen\":\n"
            "    modo: puerta\n"
            "    umbral_permanencia_s: 3600\n",
            encoding='utf-8')
        import os
        os.environ['PERMANENCIA_CONFIG'] = str(ruta)
        try:
            global_ = pa._cargar_config('Otra Camara')
            propia = pa._cargar_config('Puerta Almacen')
        finally:
            os.environ.pop('PERMANENCIA_CONFIG', None)
    assert global_['modo'] == 'area' and global_['umbral_permanencia_s'] == 300
    assert propia['modo'] == 'puerta' and propia['umbral_permanencia_s'] == 3600


def test_una_clave_inventada_en_el_yaml_no_entra_en_la_config():
    import os
    import tempfile
    from src.analityc.core import permanencia_area as pa
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'permanencia_config.yaml'
        ruta.write_text("inventada_por_alguien: 42\n", encoding='utf-8')
        os.environ['PERMANENCIA_CONFIG'] = str(ruta)
        try:
            cfg = pa._cargar_config()
        finally:
            os.environ.pop('PERMANENCIA_CONFIG', None)
    assert 'inventada_por_alguien' not in cfg
    assert set(cfg) == set(pa._DEFECTOS)


def test_un_yaml_roto_degrada_a_los_valores_por_defecto():
    import os
    import tempfile
    from src.analityc.core import permanencia_area as pa
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / 'permanencia_config.yaml'
        ruta.write_text("modo: [sin cerrar\n", encoding='utf-8')
        os.environ['PERMANENCIA_CONFIG'] = str(ruta)
        try:
            cfg = pa._cargar_config()
        finally:
            os.environ.pop('PERMANENCIA_CONFIG', None)
    assert cfg['modo'] == pa._DEFECTOS['modo']


def test_el_yaml_de_produccion_es_legible_y_completo():
    from src.analityc.core import permanencia_area as pa
    assert pa.ruta_config().is_file(), 'falta config/permanencia_config.yaml'
    cfg = pa._cargar_config()
    assert set(cfg) == set(pa._DEFECTOS)
    assert cfg['modo'] in (MODO_AREA, MODO_PUERTA)


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
