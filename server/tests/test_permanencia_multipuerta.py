"""
Pruebas del PIPELINE con varias puertas en una misma cámara (7-sep-2026).

Aquí no se prueba el reparto por dentro —eso es `test_permanencia_puertas.py`—
sino `PermanenciaArea` ENTERO con dos umbrales dibujados: se llama a
`_actualizar` frame a frame, como en producción, y se lee lo que sale por
donde sale de verdad (las alertas que drena el padre y el resumen que va en
`metadata['permanencia']`).

Lo que fija cada prueba, y el fallo concreto que evita:

  - **Compatibilidad sagrada**: sin nada dibujado, la `dedup_clave` no cambia
    ni un carácter, la estancia sigue en el libro de la cámara y el gestor no
    llega a existir. Es el 99% de los recuadros.
  - **Aforos separados**: la puerta 0 y la puerta 1 son dos SITIOS. Con un
    libro común, la segunda entrada de la misma persona moría por duplicada.
  - **Dos cronómetros para la misma persona**: estar dentro del almacén desde
    hace diez minutos y dentro del patio desde hace dos es cierto a la vez.
  - **Salir por otra puerta NO cierra la visita de la primera**: con el libro
    común, esa salida habría casado con la entrada de la principal (por
    «única abierta») y habría fundido las dos en una visita a un sitio por el
    que esa persona no pasó.
  - **Las alertas de dos puertas no se deduplican entre sí**: WhatsApp y
    Jarvis agrupan por `dedup_clave`; sin el segmento de la puerta, la
    SEGUNDA muere en el enfriamiento — justo la que demuestra que son dos
    sitios.
  - **El umbral suelto se ignora con puertas**: medirlo también emitiría DOS
    entradas por cada paso.
  - **A10**: el gestor es un atributo PLANO y NO puede entrar en la lista que
    `TiendasProcess` multiplexa por cámara (prueba de FUENTE).

Geometría (la del anexo): dos umbrales HORIZONTALES, `LINEA_A` en y=324 y
`LINEA_B` en y=756, los dos de izquierda a derecha. Caminando de P1 a P2,
DENTRO queda a la DERECHA = abajo en la imagen, así que BAJAR por el frame es
ENTRAR. `LINEA_B_INVERTIDA` es la misma con los puntos intercambiados: por ahí
bajar es SALIR, que es lo que hace falta para «entró por el almacén y salió
por el patio».

El reloj se congela y se avanza a mano: las duraciones son exactas y no
dependen de lo que tarde la máquina (lección de la limpieza del 4-sep-2026).
"""
from __future__ import annotations

import contextlib
import inspect
import re
import sys
import time as _t
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import permanencia_area as pa  # noqa: E402
from src.analityc.core.permanencia_area import (  # noqa: E402
    MODO_LINEA, PermanenciaArea)
from src.analityc.core.permanencia_estancias import SIN_ATRIBUIR  # noqa: E402
from src.analityc.core.permanencia_puertas import (  # noqa: E402
    ATRIBUTO_ACTIVATE, ATRIBUTO_GESTOR, ATRIBUTO_LINEAS, ATRIBUTO_NOMBRES)

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)

#: Los dos umbrales del anexo. Bajar por el frame = entrar.
LINEA_A = [[192, 324], [1728, 324]]
LINEA_B = [[192, 756], [1728, 756]]
#: El mismo umbral B con P1 y P2 intercambiados (doble clic en el recuadro):
#: aquí bajar es SALIR.
LINEA_B_INVERTIDA = [[1728, 756], [192, 756]]

ALTO = 200                      # el margen firme por defecto son 30 px
#: Cuatro frames arriba del todo: `frames_minimos` del detector, sin cruzar.
NACER_ARRIBA = [200, 210, 220, 230]
#: Cuatro frames entre los dos umbrales, firme para las dos puertas.
NACER_EN_MEDIO = [500, 510, 520, 530]
DENTRO_DE_A = 500               # pasado el umbral A, sin llegar al B
DENTRO_DE_B = 900               # pasado también el umbral B


class _Reloj:
    """`time.time()` a mano. Se avanza en la propia prueba."""

    def __init__(self, t0: float) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def avanzar(self, segundos: float) -> float:
        self.t += float(segundos)
        return self.t


@contextlib.contextmanager
def _reloj_de_prueba(t0: float = 1757260800.0):
    """Congela el reloj de producción (`time.time`) en uno manejable.

    `permanencia_area` y `permanencia_estancias` hacen `import time` y llaman
    `time.time()` dentro del bucle, así que basta con parchear el atributo
    del módulo `time`, igual que hace `test_permanencia_area.py`.

    El instante es un martes de septiembre de 2026 a mediodía en UTC: fijo,
    lejos de medianoche y sin depender de la hora de la máquina.
    """
    reloj = _Reloj(t0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_t, 'time', reloj)
        yield reloj


class _Doble(PermanenciaArea):
    """`PermanenciaArea` de verdad, sin modelos, sin disco y sin cámara.

    Hereda sin `__init__` (no carga RF-DETR ni GPU) y llama al mismo
    `_iniciar_permanencia` de producción, igual que
    `test_permanencia_linea.py`. Lo que se ejercita después es `_actualizar`
    entero: reconciliación, ciclo, confirmación, prolongada, vencidas y
    resumen.
    """

    def __init__(self, reloj: _Reloj, puertas=(LINEA_A, LINEA_B),
                 nombres=('Almacen', 'Patio'), linea_suelta=None, **over):
        cfg = {**pa._DEFECTOS,
               'modo': 'area',                # las puertas activan la LÍNEA
               'guardar_fotos': False,         # nada de JPEG en output/
               'persistir_estancias': False,   # nada de JSON en output/
               'estancia_ttl_horas': 100}
        cfg.update(over)
        self._reloj = reloj
        self._zone_state = {}
        self.active_tracks = {}
        self._alertas_pendientes = []
        self._camera_display_name = 'cam-test'
        self._iniciar_permanencia(cfg, 'cam-test')
        # Lo que `app.py` cuelga del procesador con cada frame.
        self._linea_cruce = linea_suelta
        self._linea_cruce_estilo = 'linea'
        self._flecha_direccion = None
        self._puertas_lineas = [list(p) for p in (puertas or [])]
        self._puertas_nombres = list(nombres or [])
        self._puertas_activate = bool(puertas)

    # -- El frame -------------------------------------------------------
    def frame(self, posiciones, dt: float = 0.5):
        """Un frame del pipeline REAL. `posiciones` = {pid: y del pie}."""
        self._reloj.avanzar(dt)
        self.active_tracks = {
            pid: {'persistent_id': pid,
                  'box': [850, y - ALTO, 950, y],
                  'in_zone': False,
                  'persistent_confirmed': True}
            for pid, y in posiciones.items()}
        return self._actualizar(FRAME)

    def caminar(self, pid: str, alturas, dt: float = 0.5):
        for y in alturas:
            self.frame({pid: y}, dt=dt)

    def quieto(self, posiciones, segundos: float, dt: float = 2.0):
        """Deja pasar el tiempo con la gente donde está (sin cruzar nada)."""
        for _ in range(int(segundos / dt)):
            self.frame(posiciones, dt=dt)

    def alertas(self):
        a = list(self._alertas_pendientes)
        self._alertas_pendientes = []
        return a

    def puerta(self, indice: int):
        return self._gestor_puertas.get(indice)


def _entrar_por_a(p: _Doble, pid: str = 'p1') -> None:
    """Nace arriba del todo y cruza el umbral A: entrada en la puerta 0."""
    p.caminar(pid, NACER_ARRIBA + [DENTRO_DE_A])


def _seguir_a_b(p: _Doble, pid: str = 'p1') -> None:
    """Sigue bajando y cruza el umbral B."""
    p.caminar(pid, [DENTRO_DE_B])


def _tipos(alertas):
    return [a['tipo_permanencia'] for a in alertas]


def _claves(alertas):
    return [a['dedup_clave'] for a in alertas]


# ── Compatibilidad sagrada: sin puertas, todo como hoy ──────────────────
def test_sin_puertas_dibujadas_todo_sale_EXACTAMENTE_como_hoy():
    """El recuadro manda `puertas_lineas: []` en CADA frame de casi todas las
    cámaras. Nada de lo que ya funciona puede notar que este reparto existe:
    ni la clave que agrupa WhatsApp y Jarvis, ni el libro donde vive la
    estancia, ni el aforo del panel.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=None, linea_suelta=LINEA_A)
        _entrar_por_a(p)
        alertas = p.alertas()

        assert _tipos(alertas) == ['entrada']
        a = alertas[0]
        assert a['dedup_clave'] == 'perm|cam-test|entrada|p1', (
            'sin puertas la clave no puede cambiar ni un carácter: WhatsApp y '
            'Jarvis agrupan por ella')
        assert a['puerta_idx'] == -1 and a['puerta_nombre'] == ''
        assert a['description'].startswith('Persona entrando al área')

        # La estancia vive en el libro de la CÁMARA, como siempre.
        assert 'p1' in p._registro
        assert len(p._gestor_puertas) == 0

        r = p._resumen(reloj.t)
        assert r['modo'] == MODO_LINEA
        assert r['dentro_ahora'] == 1
        assert r['personas'][0]['persistent_id'] == 'p1'
        assert r['personas'][0]['puerta'] == -1
        assert r['puertas'] == [], 'sin puertas la lista va vacía, no ausente'


def test_sin_los_atributos_del_payload_el_modo_de_siempre_sigue_igual():
    """Un cliente anterior a las puertas no manda NADA de esto.

    `app.py` no cuelga los atributos y todo se lee con `getattr`: si faltar
    apagara la lectura de línea, esos recuadros se quedarían sin sus alertas
    sin un solo error en el log.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=None, linea_suelta=LINEA_A)
        for atributo in (ATRIBUTO_LINEAS, ATRIBUTO_NOMBRES, ATRIBUTO_ACTIVATE):
            delattr(p, atributo)
        _entrar_por_a(p)
        assert _claves(p.alertas()) == ['perm|cam-test|entrada|p1']


def test_una_camara_sin_puertas_no_crea_el_fichero_de_puertas(tmp_path):
    """Con `persistir_estancias` encendido y sin nada dibujado, el libro de
    puertas NO se escribe: un JSON vacío que no es de nadie confunde al mirar
    el disco (la lección del `sin-camara.json`)."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=None, linea_suelta=LINEA_A,
                   persistir_estancias=True,
                   carpeta_evidencia=str(tmp_path))
        _entrar_por_a(p)
        p.cerrar_permanencia()
        assert not (tmp_path / 'estancias' / 'puertas').exists()


# ── Dos puertas, dos sitios ─────────────────────────────────────────────
def test_dos_puertas_llevan_aforos_separados():
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        r = p._resumen(reloj.t)
        assert [f['dentro_ahora'] for f in r['puertas']] == [1, 0]
        assert [f['nombre'] for f in r['puertas']] == ['Almacen', 'Patio']
        assert 'p1' in p.puerta(0) and 'p1' not in p.puerta(1)
        # El libro de la CÁMARA sigue vacío: con puertas, quien cuenta son
        # ellas (si no, el mismo paso se contaría dos veces).
        assert len(p._registro) == 0


def test_la_misma_persona_dentro_de_las_dos_tiene_dos_cronometros():
    """Entra por el almacén, se queda un minuto y entra también por el patio.

    Con un libro común, esta segunda entrada moría por duplicada («ya estaba
    dentro») y el patio no contaba a nadie. Y los dos cronómetros son
    distintos a propósito: llevar 60 s en un sitio y 0 en el otro es lo que
    de verdad pasó.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        p.quieto({'p1': DENTRO_DE_A}, segundos=60)
        p.alertas()
        _seguir_a_b(p, 'p1')

        assert _tipos(p.alertas()) == ['entrada'], (
            'la entrada por la segunda puerta tiene que avisar: es otro sitio')
        r = p._resumen(reloj.t)
        assert [f['dentro_ahora'] for f in r['puertas']] == [1, 1]
        # Aforo UNIÓN: la misma persona cuenta en los dos sitios.
        assert r['dentro_ahora'] == 2
        assert {f['persistent_id'] for f in r['personas']} == {'p1'}

        por_puerta = {f['puerta']: f['segundos'] for f in r['personas']}
        assert por_puerta[0] >= 60, 'el cronómetro del almacén sigue corriendo'
        assert por_puerta[1] < 5, 'el del patio acaba de arrancar'


def test_salir_por_otra_puerta_no_cierra_la_visita_de_la_primera():
    """Entra por «Almacen» y sale por «Patio» (umbral B invertido).

    A7: eso deja una visita ABIERTA en el primero y una salida SIN PAREJA en
    el segundo, y no es un fallo. Con el libro común, `_emparejar_salida`
    habría casado esta salida con la entrada del almacén —por «única
    abierta»— y habría fabricado una visita al patio que nadie hizo, además
    de vaciar el almacén.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=(LINEA_A, LINEA_B_INVERTIDA))
        _entrar_por_a(p, 'p1')
        p.quieto({'p1': DENTRO_DE_A}, segundos=20)
        p.alertas()
        _seguir_a_b(p, 'p1')            # cruza B hacia abajo = SALIR de B

        alertas = p.alertas()
        assert _tipos(alertas) == ['salida']
        a = alertas[0]
        assert a['identidad_por'] == SIN_ATRIBUIR
        assert a['puerta_idx'] == 1 and a['puerta_nombre'] == 'Patio'
        assert a['duracion_s'] is None, 'no se inventa un tiempo de estancia'

        assert 'p1' in p.puerta(0), 'la visita del almacén sigue abierta'
        assert len(p.puerta(1).registro) == 0
        r = p._resumen(reloj.t)
        assert [f['dentro_ahora'] for f in r['puertas']] == [1, 0]
        assert [f['salidas_sin_atribuir'] for f in r['puertas']] == [0, 1]


def test_entrar_y_salir_por_la_misma_puerta_cierra_SU_visita():
    """El caso normal: el cronómetro de esa puerta se cierra con su tiempo y
    el de la otra ni se entera."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        _seguir_a_b(p, 'p1')            # entra también por el patio
        p.quieto({'p1': DENTRO_DE_B}, segundos=120)
        p.alertas()
        p.caminar('p1', [DENTRO_DE_A])  # vuelve a subir: sale del patio

        alertas = p.alertas()
        assert _tipos(alertas) == ['salida']
        a = alertas[0]
        assert a['puerta_idx'] == 1 and a['puerta_nombre'] == 'Patio'
        assert a['duracion_s'] >= 120
        assert 'Permanencia en Patio' in a['description']
        assert a['titulo_novedad'] == 'Permanencia en Patio'

        assert 'p1' in p.puerta(0), 'la del almacén no se toca'
        assert 'p1' not in p.puerta(1)
        r = p._resumen(reloj.t)
        assert [f['visitas_completadas'] for f in r['puertas']] == [0, 1]
        assert r['puertas'][1]['permanencia_media_s'] >= 120
        assert r['visitas_completadas'] == 1, 'el total es la unión'


# ── Las alertas no se pisan entre puertas ───────────────────────────────
def test_las_alertas_de_dos_puertas_NO_se_deduplican_entre_si():
    """Dos personas entrando a la vez por puertas distintas.

    WhatsApp y Jarvis agrupan por `dedup_clave`: si la compartieran, la
    SEGUNDA moriría en el enfriamiento y se perdería justo la alerta que
    demuestra que son dos sitios.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        for i in range(4):              # los dos nacen y se afirman
            p.frame({'ana': NACER_ARRIBA[i], 'beto': NACER_EN_MEDIO[i]})
        p.frame({'ana': DENTRO_DE_A, 'beto': DENTRO_DE_B})

        alertas = p.alertas()
        assert _tipos(alertas) == ['entrada', 'entrada']
        claves = _claves(alertas)
        assert sorted(claves) == ['perm|cam-test|entrada|p0|ana',
                                  'perm|cam-test|entrada|p1|beto']
        assert len(set(claves)) == 2
        # El segmento va DESPUÉS del tipo: quien filtre por el prefijo de
        # siempre sigue viendo las dos.
        assert all(c.startswith('perm|cam-test|entrada|') for c in claves)


def test_la_misma_persona_por_dos_puertas_tampoco_comparte_clave():
    """El caso que más duele: una sola persona, dos sitios. Sin el segmento
    de la puerta, la segunda entrada muere deduplicada contra la primera."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        _seguir_a_b(p, 'p1')
        claves = _claves(p.alertas())
        assert claves == ['perm|cam-test|entrada|p0|p1',
                          'perm|cam-test|entrada|p1|p1']


def test_la_alerta_lleva_el_formato_que_esperan_whatsapp_y_jarvis():
    """Los campos nuevos se AÑADEN: los reenviadores no cambian."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        a = p.alertas()[0]
        for clave in ('event_type', 'class_name', 'description', 'timestamp',
                      'camera_name', 'person_uuid', 'crop_image',
                      'image_base64', 'dedup_clave', 'clase_gruesa'):
            assert clave in a, f"falta {clave}, que el reenviador espera"
        assert a['clase_gruesa'] == 'persona'
        assert a['puerta_idx'] == 0 and a['puerta_nombre'] == 'Almacen'
        assert 'Almacen' in a['description'], (
            'en WhatsApp hay que poder leer POR DÓNDE entró')
        assert a['umbral_dibujado_con'] == 'puerta', (
            'las puertas se dibujan con la flecha doble; el estilo del umbral '
            'suelto, que está ignorado, no puede colarse aquí')


# ── Precedencia: las puertas mandan sobre el umbral suelto ──────────────
def test_con_puertas_el_umbral_suelto_se_IGNORA():
    """`linea_cruce` dibujado ENCIMA del mismo vano que la puerta 0.

    Si se midieran los dos, cada paso emitiría DOS entradas —dos cronómetros
    y dos alertas para una sola persona— y el aforo de la cámara diría el
    doble de gente de la que hay.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=(LINEA_A, LINEA_B), linea_suelta=LINEA_A)
        _entrar_por_a(p, 'p1')
        alertas = p.alertas()
        assert _tipos(alertas) == ['entrada'], 'una sola entrada por paso'
        assert alertas[0]['puerta_idx'] == 0
        assert len(p._registro) == 0, (
            'el detector del umbral suelto no puede abrir nada mientras haya '
            'puertas')


def test_las_puertas_ganan_tambien_al_modo_explicito_del_yaml():
    """Una cámara con `modo: cruce` en el YAML y tres umbrales dibujados.

    Si el modo explícito ganara, esos umbrales no medirían NI UNO: el
    operador dibuja tres puertas, no pasa nada y no hay un solo error en el
    log. Ese silencio no lo depura nadie, así que dibujar manda.
    """
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, modo='cruce')
        assert p.modo_permanencia == 'cruce'
        _entrar_por_a(p, 'p1')
        alertas = p.alertas()
        assert _tipos(alertas) == ['entrada']
        assert alertas[0]['puerta_idx'] == 0
        assert p._modo_vigente == MODO_LINEA


def test_borrar_las_puertas_devuelve_el_umbral_suelto():
    """El operador borra los umbrales de puerta y deja la línea de siempre:
    la lectura de antes tiene que volver, con su clave de siempre."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=(LINEA_A,), nombres=('Almacen',),
                   linea_suelta=LINEA_B)
        _entrar_por_a(p, 'p1')
        p.alertas()
        p._puertas_lineas = []
        p._puertas_activate = False
        p.frame({'p1': DENTRO_DE_A})    # primer frame ya sin puertas
        assert 'sin_salida' in _tipos(p.alertas())

        p.caminar('p2', NACER_EN_MEDIO + [DENTRO_DE_B])
        alertas = p.alertas()
        assert _tipos(alertas) == ['entrada']
        assert _claves(alertas) == ['perm|cam-test|entrada|p2']
        assert 'p2' in p._registro


# ── Reconciliación en caliente ──────────────────────────────────────────
def test_borrar_una_puerta_con_gente_dentro_avisa_sin_salida():
    """Nadie se cae en silencio: quien estaba contado dentro de un sitio que
    el operador acaba de borrar se cierra DICIÉNDOLO, y con la clave de SU
    puerta (si no, ese aviso podría morir deduplicado contra el de otro)."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        _seguir_a_b(p, 'p1')
        p.quieto({'p1': DENTRO_DE_B}, segundos=30)
        p.alertas()

        p._puertas_lineas = [LINEA_A]           # se borra la última puerta
        p._puertas_nombres = ['Almacen']
        p.frame({'p1': DENTRO_DE_B})

        alertas = p.alertas()
        assert _tipos(alertas) == ['sin_salida']
        a = alertas[0]
        assert a['puerta_idx'] == 1 and a['puerta_nombre'] == 'Patio'
        assert a['dedup_clave'] == 'perm|cam-test|sin_salida|p1|p1'
        assert a['identidad_por'] == SIN_ATRIBUIR
        assert 'ya no está dibujada' in a['description']
        assert 'p1' in p.puerta(0), 'la otra puerta no se toca'


def test_renombrar_una_puerta_no_bifurca_el_aforo():
    """La identidad es el ÍNDICE. Renombrar a media tarde no puede reiniciar
    nada: la gente de dentro sigue dentro, y solo cambia el rótulo."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        p.alertas()
        p._puertas_nombres = ['Deposito', 'Patio']
        r = p.frame({'p1': DENTRO_DE_A})

        assert r['puertas'][0]['nombre'] == 'Deposito'
        assert r['puertas'][0]['dentro_ahora'] == 1
        assert [f['puerta_nombre'] for f in r['personas']] == ['Deposito']


def test_mover_el_umbral_no_reinicia_el_cronometro():
    """El operador arrastra la línea unos píxeles: el cronómetro no puede
    volver a cero (ni la persona salir y volver a entrar)."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        p.quieto({'p1': DENTRO_DE_A}, segundos=40)
        p.alertas()
        p._puertas_lineas = [[[192, 330], [1728, 330]], LINEA_B]
        r = p.frame({'p1': DENTRO_DE_A})

        assert p.alertas() == []
        assert r['puertas'][0]['dentro_ahora'] == 1
        assert r['personas'][0]['segundos'] >= 40


# ── El panel ────────────────────────────────────────────────────────────
def test_el_panel_amplia_los_totales_sin_sustituirlos():
    """`metadata['permanencia']` conserva sus claves de siempre con la UNIÓN y
    añade una fila por puerta. Y cada fila de personas dice su puerta: con
    solo la persona, quien está dentro de dos sitios genera dos filas que se
    pelean el mismo widget y el cronómetro alterna."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj)
        _entrar_por_a(p, 'p1')
        _seguir_a_b(p, 'p1')
        r = p.frame({'p1': DENTRO_DE_B})

        for clave in ('modo', 'camara', 'dentro_ahora', 'personas',
                      'visitas_completadas', 'permanencia_media_s',
                      'umbral_s', 'salidas_sin_atribuir', 'sin_personas_s'):
            assert clave in r, f'el panel de siempre lee {clave}'
        assert r['dentro_ahora'] == 2 == len(r['personas'])
        assert {(f['puerta'], f['puerta_nombre']) for f in r['personas']} == {
            (0, 'Almacen'), (1, 'Patio')}

        fila = r['puertas'][0]
        assert set(fila) == {'idx', 'nombre', 'dentro_ahora', 'personas',
                             'visitas_completadas', 'permanencia_media_s',
                             'salidas_sin_atribuir'}
        assert fila['idx'] == 0 and fila['nombre'] == 'Almacen'
        assert [f['persistent_id'] for f in fila['personas']] == ['p1']


def test_el_video_marca_el_cronometro_tambien_con_puertas():
    """El cliente pinta `label` tal cual. Con puertas, el libro de la cámara
    está vacío: sin mirar los de las puertas el vídeo se quedaría SIN
    cronómetros y nadie sabría por qué."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, alertar_entrada=False)
        _entrar_por_a(p, 'p1')
        p.quieto({'p1': DENTRO_DE_A}, segundos=90)
        p.active_tracks = {7: {'persistent_id': 'p1', 'box': [850, 300, 950,
                                                              DENTRO_DE_A],
                               'class': -1, 'category': 'Personas',
                               'confidence': 0.9,
                               'persistent_confirmed': True}}
        detecciones = p._build_detections()
        assert detecciones, 'no hay detecciones que marcar'
        marcada = [d for d in detecciones if d.get('permanencia_s')]
        assert marcada and marcada[0]['permanencia_s'] >= 90
        assert 'dentro' in marcada[0]['label']


# ── Permanencia prolongada, puerta a puerta ─────────────────────────────
def test_el_aviso_de_permanencia_es_de_la_puerta_y_no_se_deduplica():
    """Quien lleva mucho dentro de dos sitios genera DOS avisos, uno por
    sitio y con su tiempo. Con la clave de siempre, el segundo moriría en el
    enfriamiento y el panel diría que solo lleva mucho en uno."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, umbral_permanencia_s=60, realertar_cada_s=0)
        _entrar_por_a(p, 'p1')
        _seguir_a_b(p, 'p1')
        p.alertas()
        p.quieto({'p1': DENTRO_DE_B}, segundos=70)

        avisos = [a for a in p.alertas()
                  if a['tipo_permanencia'] == 'permanencia']
        assert len(avisos) == 2
        assert sorted(_claves(avisos)) == ['perm|cam-test|permanencia|p0|p1',
                                           'perm|cam-test|permanencia|p1|p1']
        assert {a['puerta_nombre'] for a in avisos} == {'Almacen', 'Patio'}
        assert all('dentro de «' in a['description'] for a in avisos)


# ── Entradas retenidas (`confirmar_entrada_s`) ──────────────────────────
def test_un_roce_en_una_puerta_no_tira_la_entrada_retenida_de_la_otra():
    """Las colas de espera son por puerta y la clave es LA PERSONA: con una
    cola común, el roce de alguien en el patio se llevaría por delante la
    alerta buena que esperaba confirmación en el almacén."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, confirmar_entrada_s=30,
                   min_estancia_para_avisar_s=3)
        _entrar_por_a(p, 'p1')
        assert p.alertas() == [], 'la entrada se retiene hasta confirmar'
        assert 'p1' in p.puerta(0).entradas_en_espera
        assert p.puerta(1).entradas_en_espera == {}

        _seguir_a_b(p, 'p1')                    # entra también por el patio
        p.caminar('p1', [DENTRO_DE_A])          # y sale enseguida: un ROCE
        assert 'p1' not in p.puerta(1).registro
        assert p.alertas() == [], 'el roce del patio muere en silencio'
        assert 'p1' in p.puerta(0).entradas_en_espera, (
            'la espera del almacén no es del patio')

        p.quieto({'p1': DENTRO_DE_A}, segundos=40)
        alertas = p.alertas()
        assert _tipos(alertas) == ['entrada']
        assert alertas[0]['dedup_clave'] == 'perm|cam-test|entrada|p0|p1'


# ── A10: el gestor es un atributo PLANO ─────────────────────────────────
FUENTE_TIENDAS = RAIZ / 'server' / 'src' / 'analityc' / 'core' / 'tiendas.py'


def test_el_gestor_de_puertas_es_un_atributo_plano_del_procesador():
    """Prueba de FUENTE: `_iniciar_permanencia` cuelga el gestor de `self`.

    Solo pregunta «¿está esta clave en esta ruta?» (lección H-47): que el
    gestor haga lo correcto lo fijan las otras pruebas de este fichero. Aquí
    lo que se protege es el SITIO, que es lo que no se ve fallar: no hay que
    multiplexar nada porque ya existe una instancia del procesador por
    cámara.
    """
    fuente = inspect.getsource(PermanenciaArea._iniciar_permanencia)
    assert f'self.{ATRIBUTO_GESTOR} = GestorPuertas(' in fuente, (
        'el gestor tiene que montarse aquí, con el resto del estado de la '
        'cámara: si no, la primera lectura revienta dentro del try que solo '
        'loguea y el modo se queda mudo')


def test_el_gestor_de_puertas_NO_entra_en_la_lista_que_se_multiplexa():
    """`TiendasProcess._push_state` hace `setattr(self, n, cam_state.get(n))`
    con todo lo que nombre esa lista. El gestor ahí dentro quedaría en `None`
    en cada frame, `_actualizar` reventaría dentro de un `try` que SOLO
    loguea y el modo se quedaría MUDO con el vídeo perfecto: es exactamente
    el bug del 6 y el 7-ago-2026.

    Hermana de la de `test_permanencia_puertas.py`: aquella protege el
    módulo del reparto, esta el pipeline que lo usa. La duplicación es
    deliberada — quien «arregle» el atributo metiéndolo en la lista tiene que
    tropezar con las dos.
    """
    texto = FUENTE_TIENDAS.read_text(encoding='utf-8')
    m = re.search(r'def _push_state\(.*?names\s*=\s*\[(.*?)\]', texto, re.S)
    assert m, ('no se encontró la lista `names` de `_push_state` en '
               'tiendas.py: si se renombró, hay que re-apuntar esta prueba')
    for atributo in (ATRIBUTO_GESTOR, ATRIBUTO_LINEAS, ATRIBUTO_NOMBRES,
                     ATRIBUTO_ACTIVATE):
        assert atributo not in m.group(1), (
            f'`{atributo}` está en la lista que `_push_state` multiplexa por '
            'cámara: quedaría en None en cada frame y las puertas no '
            'contarían a nadie, sin un solo error en el log.')


def test_el_modo_sobrevive_a_que_el_gestor_desaparezca():
    """Red de seguridad del mismo fallo: si alguien deja el atributo en
    `None`, el modo se repone y sigue contando en vez de quedarse mudo."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, puertas=None, linea_suelta=LINEA_A)
        setattr(p, ATRIBUTO_GESTOR, None)
        _entrar_por_a(p, 'p1')
        assert _claves(p.alertas()) == ['perm|cam-test|entrada|p1']
        assert getattr(p, ATRIBUTO_GESTOR) is not None


# ── Disco: el libro de puertas ──────────────────────────────────────────
def test_un_reinicio_no_vacia_las_puertas(tmp_path):
    """El libro se escribe al cerrar el cliente y se vuelve a leer al montar
    el modo: quien estaba dentro sigue dentro y su salida encuentra pareja."""
    with _reloj_de_prueba() as reloj:
        p = _Doble(reloj, persistir_estancias=True,
                   carpeta_evidencia=str(tmp_path))
        _entrar_por_a(p, 'p1')
        p.quieto({'p1': DENTRO_DE_A}, segundos=30)
        p.cerrar_permanencia()
        libro = tmp_path / 'estancias' / 'puertas' / 'cam-test.json'
        assert libro.is_file(), 'el libro de puertas tiene que estar en disco'

        otro = _Doble(reloj, persistir_estancias=True,
                      carpeta_evidencia=str(tmp_path))
        otro.frame({})                      # primer frame: se adoptan
        assert 'p1' in otro.puerta(0)
        r = otro._resumen(reloj.t)
        assert r['puertas'][0]['dentro_ahora'] == 1
        assert r['personas'][0]['segundos'] >= 30, (
            'el cronómetro no se reinicia al reiniciar el servidor')


if __name__ == '__main__':                                  # pragma: no cover
    raise SystemExit(pytest.main([__file__, '-v']))
