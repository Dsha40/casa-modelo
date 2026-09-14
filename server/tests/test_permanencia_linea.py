"""
Pruebas de la lectura LÍNEA de `PermanenciaArea` (cruce por cambio de lado).

Pedida el 10-ago-2026 para la puerta de Kitchen Productions: la cámara gran
angular curva las trayectorias y la proyección sobre la flecha confundía
entradas con salidas. La línea se dibuja SOBRE el umbral y el sentido sale de
qué lado a qué lado se pasó.

Lo que más se protege:

  - **el jitter sobre el umbral no fabrica cruces** — solo cuenta el lado
    FIRME (histéresis en alturas de la persona);
  - **cruzar la prolongación de la línea no cuenta** — el cruce tiene que
    cortar el SEGMENTO dibujado, no su recta infinita;
  - **los desempates de vano no inventan regresos** — quien acaba de cruzar
    y el muro lo oculta NO está volviendo a salir;
  - **sin línea NO se emite nada** — la dirección se mide, no se adivina.

Geometría de las pruebas: línea VERTICAL de (300,0) a (300,400), dibujada
hacia abajo. DENTRO queda a la derecha de P1→P2 = x < 300. Personas de 200 px
de alto: el margen firme por defecto (0.15 alturas) son 30 px.

El doble hereda de `PermanenciaArea` sin `__init__` (no carga YOLO ni GPU) y
llama al mismo `_iniciar_permanencia` de producción, igual que
`test_permanencia_cruce.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.permanencia_area import (  # noqa: E402
    MODO_AREA, MODO_CRUCE, MODO_LINEA, PermanenciaArea)
from src.analityc.core.permanencia_linea import (  # noqa: E402
    POR_LINEA, DetectorLinea)
from src.analityc.core.permanencia_puerta import ENTRADA, SALIDA  # noqa: E402

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
#: Línea vertical sobre el umbral. DENTRO = x < 300.
LINEA = [[300, 0], [300, 400]]
ALTO = 200


def _obs(x, y=250, confirmado=True, alto=ALTO):
    """Una persona cuyo PIE está en `(x, y)`."""
    return {'confirmado': confirmado,
            'box': [x - 30, y - alto, x + 30, y]}


def _caminar(det, clave, pies, t0=0.0, dt=0.2, linea=LINEA, **kw):
    """`pies` = [x, ...] o [(x, y), ...]. Devuelve los cruces emitidos."""
    cruces = []
    for i, pie in enumerate(pies):
        x, y = pie if isinstance(pie, tuple) else (pie, 250)
        cruces.extend(det.observar(t0 + i * dt,
                                   {clave: _obs(x, y, **kw)}, linea))
    return cruces


ENTRA = [450, 410, 370, 330, 290, 250, 210]     # de FUERA (x>300) a DENTRO
SALE = [170, 230, 290, 330, 370]                # de DENTRO a FUERA


# ── DetectorLinea ───────────────────────────────────────────────────────
def test_cruzar_hacia_dentro_es_entrada():
    cruces = _caminar(DetectorLinea(), 'p-A', ENTRA)
    assert [c.tipo for c in cruces] == [ENTRADA]
    assert cruces[0].via == POR_LINEA


def test_cruzar_hacia_fuera_es_salida():
    cruces = _caminar(DetectorLinea(), 'p-B', SALE)
    assert [c.tipo for c in cruces] == [SALIDA]


def test_el_jitter_sobre_el_umbral_no_fabrica_cruces():
    """Dentro del margen de histéresis no hay lado, y sin lado no hay cruce."""
    pies = [200, 250, 285, 265, 290, 255, 200]   # oscila a <30 px de la línea
    assert _caminar(DetectorLinea(), 'p-C', pies) == []


def test_cruzar_la_prolongacion_no_cuenta():
    """El mismo paso, pero 300 px por debajo del final del segmento."""
    pies = [(x, 700) for x in SALE]
    assert _caminar(DetectorLinea(), 'p-D', pies) == []


def test_rodear_la_linea_no_emite_y_el_reingreso_respeta_el_cooldown():
    """Salir por FUERA del segmento no avisa; volver a entrar en caliente
    tampoco (mismo sentido dentro de `no_reemitir_s`)."""
    det = DetectorLinea()
    pasos = ([(x, 250) for x in ENTRA]            # ENTRADA de verdad
             + [(250, 700), (330, 700)]           # rodea la línea por abajo
             + [(330, 250), (290, 250), (250, 250)])   # y reingresa ya
    cruces = _caminar(det, 'p-E', pasos)
    assert [c.tipo for c in cruces] == [ENTRADA], \
        'ni el rodeo ni el reingreso en caliente deben avisar'


def test_reingresar_pasado_el_cooldown_si_avisa():
    det = DetectorLinea()
    _caminar(det, 'p-F', ENTRA, t0=0.0)
    pasos = [(330, 250), (290, 250), (250, 250)]
    cruces = _caminar(det, 'p-F', pasos, t0=20.0)
    # A los 20 s: la vuelta a FUERA es una salida real y el reingreso avisa.
    assert [c.tipo for c in cruces] == [SALIDA, ENTRADA]


def test_morir_junto_al_umbral_es_cruce_al_otro_lado():
    """El tracker la pierde EN el marco viniendo de dentro: salió."""
    det = DetectorLinea()
    _caminar(det, 'p-G', [180, 240, 270, 285, 292])
    cruces = det.observar(4.0, {}, LINEA)
    assert [c.tipo for c in cruces] == [SALIDA]
    assert cruces[0].via == POR_LINEA


def test_morir_lejos_de_la_linea_no_infiere_nada():
    det = DetectorLinea()
    _caminar(det, 'p-H', [100, 110, 120, 130, 140])
    assert det.observar(4.0, {}, LINEA) == []


def test_morir_tras_cruzar_no_inventa_el_regreso():
    """Cruza, el muro lo oculta enseguida: NO es una salida fantasma."""
    det = DetectorLinea()
    cruces = _caminar(det, 'p-I', ENTRA)
    assert [c.tipo for c in cruces] == [ENTRADA]
    assert det.observar(5.0, {}, LINEA) == []


def test_el_desempate_de_muerte_se_puede_apagar():
    det = DetectorLinea({'desempate_muerte': False})
    _caminar(det, 'p-J', [180, 240, 270, 285, 292])
    assert det.observar(4.0, {}, LINEA) == []


def test_nacer_en_el_umbral_y_afirmarse_dentro_es_entrada():
    """Aparece EN el marco (venía del otro lado de la puerta) y entra."""
    cruces = _caminar(DetectorLinea(), 'p-K', [310, 300, 280, 250, 230])
    assert [c.tipo for c in cruces] == [ENTRADA]


def test_el_desempate_de_nacimiento_se_puede_apagar():
    det = DetectorLinea({'desempate_nacimiento': False})
    assert _caminar(det, 'p-L', [310, 300, 280, 250, 230]) == []


def test_nacer_lejos_no_es_ningun_cruce():
    """Reaparecer ya dentro (re-track de alguien que estaba) no avisa."""
    assert _caminar(DetectorLinea(), 'p-M', [180, 185, 190, 195, 200]) == []


def test_reaparicion_tardia_es_paso_nuevo():
    det = DetectorLinea()
    cruces = _caminar(det, 'p-N', ENTRA, t0=0.0)
    cruces += _caminar(det, 'p-N', [150, 230, 290, 330, 370], t0=50.0)
    assert [c.tipo for c in cruces] == [ENTRADA, SALIDA]


def test_sin_linea_no_emite_nada():
    assert _caminar(DetectorLinea(), 'p-O', ENTRA, linea=None) == []


def test_una_linea_de_largo_cero_no_emite():
    assert _caminar(DetectorLinea(), 'p-P', ENTRA,
                    linea=[[100, 100], [100, 100]]) == []


def test_el_provisional_cuenta_pero_va_marcado():
    cruces = _caminar(DetectorLinea(), 'prov:7', ENTRA, confirmado=False)
    assert [c.tipo for c in cruces] == [ENTRADA]
    assert cruces[0].confirmado is False


def test_una_persona_diminuta_no_decide():
    """Sin escala (caja de 6 px) no hay margen creíble: silencio."""
    assert _caminar(DetectorLinea(), 'p-Q', ENTRA, alto=6) == []


# ── Los agujeros que cazó la revisión adversarial (10-ago-2026) ─────────
def test_cruzar_recien_nacido_emite_en_cuanto_hay_vida_minima():
    """Cruzar en el frame 3 de vida no puede perder el paso para siempre:
    queda pendiente y se emite al juntar `frames_minimos`."""
    cruces = _caminar(DetectorLinea(), 'p-R', [370, 290, 250, 210])
    assert [c.tipo for c in cruces] == [ENTRADA]


def test_cruzar_recien_nacido_y_morir_no_invierte_el_sentido():
    """Antes: el cruce joven se descartaba, `lado_firme` quedaba apuntando
    al DESTINO y el desempate de muerte emitía el sentido CONTRARIO."""
    det = DetectorLinea()
    cruces = _caminar(det, 'p-S', [370, 290, 250, 250])
    assert [c.tipo for c in cruces] == [ENTRADA]
    assert det.observar(3.5, {}, LINEA) == [], \
        'la muerte tras el cruce no puede fabricar el regreso'


def test_nacer_en_la_banda_y_afirmarse_del_mismo_lado_no_emite():
    """Jitter de quien nace pegado al umbral SIN cambiar de lado: silencio."""
    assert _caminar(DetectorLinea(), 'p-T', [310, 312, 308, 311, 335]) == []


def test_parado_junto_a_la_puerta_que_se_pierde_no_es_cruce():
    """Sin evidencia de aproximación no se infiere nada, ni con churn."""
    det = DetectorLinea()
    for i, clave in enumerate(('prov:1', 'prov:2', 'prov:3')):
        t0 = i * 10.0
        _caminar(det, clave, [360, 361, 359, 360, 361], t0=t0)
        assert det.observar(t0 + 4.0, {}, LINEA) == [], \
            f'el re-track {clave} de la persona parada fabricó un cruce'


def test_ida_y_vuelta_de_un_parpadeo_se_cancela():
    """Cruza y regresa antes de la vida mínima: ninguno de los dos vale."""
    det = DetectorLinea()
    assert _caminar(det, 'p-U', [370, 250, 340, 345, 350]) == []
    assert det.observar(4.0, {}, LINEA) == []


# ── Integración con PermanenciaArea ─────────────────────────────────────
class _Doble(PermanenciaArea):
    """`PermanenciaArea` sin cargar modelos, sin disco y sin cámara."""

    def __init__(self, **over):
        from src.analityc.core.permanencia_area import _DEFECTOS
        cfg = {**_DEFECTOS,
               'modo': 'linea',
               'guardar_fotos': False,        # nada de JPEG en output/
               'persistir_estancias': False,  # nada de JSON en output/
               'estancia_ttl_horas': 100}
        cfg.update(over)
        self._zone_state = {}
        self.active_tracks = {}
        self._alertas_pendientes = []
        self._camera_display_name = 'cam-test'
        self._iniciar_permanencia(cfg, 'cam-test')
        self._linea_cruce = LINEA
        self._flecha_direccion = None

    def ve_en(self, pid, t, x, y=250):
        self.active_tracks = {pid: {'persistent_id': pid,
                                    'box': [x - 30, y - ALTO, x + 30, y],
                                    'in_zone': False,
                                    'persistent_confirmed': True}}
        self._modo_vigente = self._modo_efectivo()
        self._ciclo_linea(FRAME, t)

    def alertas(self):
        a = list(self._alertas_pendientes)
        self._alertas_pendientes = []
        return a


def _entra(p, pid, t0=0.0):
    for i, x in enumerate(ENTRA):
        p.ve_en(pid, t0 + i * 0.2, x)


def _sale(p, pid, t0):
    for i, x in enumerate(SALE):
        p.ve_en(pid, t0 + i * 0.2, x)


def test_en_modo_linea_entrar_avisa_persona_entrando():
    p = _Doble()
    _entra(p, 'p-1')
    alertas = p.alertas()
    assert len(alertas) == 1
    a = alertas[0]
    assert a['event_type'] == 'Persona entrando al área'
    assert a['crop_image'], 'la alerta de entrada debe llevar foto'
    assert a['clase_gruesa'] == 'persona'
    assert 'p-1' in p._registro, 'la entrada abre la estancia (cronómetro)'


def test_salida_con_pareja_lleva_el_tiempo_dentro():
    p = _Doble()
    _entra(p, 'p-2', t0=0.0)
    p.alertas()
    _sale(p, 'p-2', t0=120.0)
    alertas = p.alertas()
    assert len(alertas) == 1
    a = alertas[0]
    assert a['event_type'] == 'Persona saliendo del área'
    assert a['duracion_s'] and a['duracion_s'] > 100


def test_sin_linea_el_modo_linea_calla():
    p = _Doble()
    p._linea_cruce = None
    _entra(p, 'p-3')
    assert p.alertas() == []


# ── El formato pedido de la alerta de permanencia (10-ago-2026) ─────────
def test_la_alerta_de_salida_lleva_el_formato_pedido():
    """Cámaras {local} / Permanencia en {area} / Inicio / Fin / Total."""
    p = _Doble()
    p._establecimiento = 'CY Aruba'
    p._area_nombre = 'el pasillo de cooler'
    _entra(p, 'p-F1', t0=0.0)
    p.alertas()
    _sale(p, 'p-F1', t0=120.0)
    alertas = p.alertas()
    assert len(alertas) == 1
    a = alertas[0]
    lineas = a['description'].split('\n')
    assert lineas[0] == 'Cámaras CY Aruba'
    assert lineas[1] == 'Permanencia en el pasillo de cooler'
    assert lineas[2].startswith('Inicio: ')
    assert lineas[3].startswith('Fin: ')
    assert lineas[4].startswith('Tiempo total: 00:01:5')
    assert 'iVMS' not in a['description']
    assert a['texto_plano'] is True
    assert a['titulo_novedad'] == 'Permanencia en el pasillo de cooler'
    assert a['crop_image'], 'la foto apilada Inicio/Fin debe viajar'


def test_el_local_puede_llegar_como_dict():
    """El selector a veces manda el local entero, no solo el nombre."""
    p = _Doble()
    p._establecimiento = {'id': 7, 'nombre': 'CY Aruba'}
    p._area_nombre = 'el pasillo de cooler'
    _entra(p, 'p-F2', t0=0.0)
    p.alertas()
    _sale(p, 'p-F2', t0=60.0)
    assert p.alertas()[0]['description'].startswith('Cámaras CY Aruba\n')


def test_sin_local_ni_area_no_se_inventa_nada():
    """Sin selector la línea «Cámaras …» se omite; sin nombre, «el área».
    JAMÁS se cae al título de la ventana de iVMS."""
    p = _Doble()
    _entra(p, 'p-F3', t0=0.0)
    p.alertas()
    _sale(p, 'p-F3', t0=60.0)
    lineas = p.alertas()[0]['description'].split('\n')
    assert lineas[0] == 'Permanencia en el área'
    assert 'cam-test' not in '\n'.join(lineas)


def test_whatsapp_manda_el_texto_plano_tal_cual():
    """Con `texto_plano` el mensaje va SIN prefijo de evento ni cámara."""
    from src.app.whatsapp_alertas import _texto
    detalle = ('Cámaras CY Aruba\nPermanencia en el pasillo de cooler\n'
               'Inicio: 15:14:04\nFin: 15:14:21\nTiempo total: 00:00:17')
    alerta = {'event_type': 'salida', 'description': detalle,
              'texto_plano': True, 'timestamp': '2026-08-10 15:14:21'}
    assert _texto(alerta, 'iVMS-4200') == detalle
    sin_marca = _texto({'event_type': 'salida', 'description': 'x',
                        'timestamp': ''}, 'cam')
    # Formato clásico del 31-ago-2026: encabezado con la cámara (sin clase)
    # y la descripción en su línea — distinto del texto plano, que va solo.
    assert sin_marca.startswith('🚨'), 'sin la marca, el formato clásico sigue'
    assert 'cámara cam' in sin_marca and '\nx' in sin_marca


def test_la_escena_de_entrada_viaja_en_la_estancia():
    p = _Doble()
    _entra(p, 'p-F4', t0=0.0)
    estancia = p._registro.abiertas.get('p-F4')
    assert estancia is not None and estancia.entrada_escena_b64, \
        'sin escena de entrada no hay cuadro «Inicio» que apilar'


# ── El interruptor: la línea activa la lectura en modo area ─────────────
def test_la_linea_activa_la_lectura_en_modo_area():
    p = _Doble(modo='area')
    assert p._modo_efectivo() == MODO_LINEA


def test_la_linea_gana_a_la_flecha():
    """Si por descuido llegan las dos, manda la herramienta nueva."""
    p = _Doble(modo='area')
    p._flecha_direccion = [[0, 0], [100, 0]]
    assert p._modo_efectivo() == MODO_LINEA


def test_sin_linea_la_flecha_sigue_activando_el_cruce():
    p = _Doble(modo='area')
    p._linea_cruce = None
    p._flecha_direccion = [[0, 0], [100, 0]]
    assert p._modo_efectivo() == MODO_CRUCE


def test_sin_dibujos_el_area_sigue_siendo_area():
    p = _Doble(modo='area')
    p._linea_cruce = None
    assert p._modo_efectivo() == MODO_AREA


def test_el_resumen_dice_el_modo_vigente():
    p = _Doble(modo='area')
    p._modo_vigente = p._modo_efectivo()
    assert p._resumen(0.0)['modo'] == MODO_LINEA


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


# ── La PUERTA del recuadro (5-sep-2026) ─────────────────────────────────
#
# El cliente gana un botón «⇅ Puerta»: se dibuja el umbral igual que la
# línea y encima aparece una flecha doble que lo atraviesa —cabeza verde
# ENTRADA, roja SALIDA—, y el doble clic la invierte. La geometría que
# manda al servidor es EXACTAMENTE la de la línea (invertir = intercambiar
# P1 y P2, que es justo lo que ya significa aquí), así que no hay detector
# nuevo: solo llegan dos campos, el interruptor y el estilo.

def test_el_interruptor_apagado_devuelve_el_area_clasica():
    """`linea_cruce_activate: false` = dibujado pero desactivado."""
    from src.app.app import umbral_del_payload
    linea, _ = umbral_del_payload({'linea_cruce': LINEA,
                                   'linea_cruce_activate': False})
    assert linea is None
    p = _Doble(modo='area')
    p._linea_cruce = linea
    assert p._modo_efectivo() == MODO_AREA


def test_sin_el_campo_activate_la_linea_sigue_valiendo():
    """Los clientes anteriores a la puerta solo mandan la línea: exigirles
    el interruptor apagaría la lectura que ya usan."""
    from src.app.app import umbral_del_payload
    linea, estilo = umbral_del_payload({'linea_cruce': LINEA})
    assert linea == LINEA and estilo is None
    p = _Doble(modo='area')
    p._linea_cruce = linea
    assert p._modo_efectivo() == MODO_LINEA


def test_la_puerta_del_recuadro_activa_la_misma_lectura_de_linea():
    from src.app.app import umbral_del_payload
    linea, estilo = umbral_del_payload({'linea_cruce': LINEA,
                                        'linea_cruce_activate': True,
                                        'linea_cruce_estilo': 'puerta'})
    assert linea == LINEA and estilo == 'puerta'
    p = _Doble(modo='area')
    p._linea_cruce, p._linea_cruce_estilo = linea, estilo
    assert p._modo_efectivo() == MODO_LINEA


def test_borrar_el_umbral_apaga_la_lectura_aunque_quede_el_estilo():
    from src.app.app import umbral_del_payload
    linea, estilo = umbral_del_payload({'linea_cruce': [],
                                        'linea_cruce_activate': True,
                                        'linea_cruce_estilo': 'puerta'})
    assert linea is None and estilo == 'puerta'
    p = _Doble(modo='area')
    p._linea_cruce, p._linea_cruce_estilo = linea, estilo
    assert p._modo_efectivo() == MODO_AREA


def test_la_puerta_entra_y_sale_igual_que_la_linea():
    """La flecha doble no cambia la geometría: mismos cruces, y darle la
    vuelta (intercambiar P1 y P2) cambia el sentido sin mover el umbral."""
    det = DetectorLinea()
    assert [c.tipo for c in _caminar(det, 'p-puerta', ENTRA)] == [ENTRADA]
    al_reves = DetectorLinea()
    invertida = [LINEA[1], LINEA[0]]
    cruces = _caminar(al_reves, 'p-inv', ENTRA, linea=invertida)
    assert [c.tipo for c in cruces] == [SALIDA]


def test_el_estilo_viaja_en_la_alerta_pero_no_decide_nada():
    """`umbral_dibujado_con` contesta qué veía el operador cuando trazó el
    umbral. La alerta es la misma con puerta que con línea: el rótulo no
    cambia el sentido ni el texto, solo acompaña."""
    con_puerta = _Doble()
    con_puerta._linea_cruce_estilo = 'puerta'
    _entra(con_puerta, 'p-igual')
    a = con_puerta.alertas()
    assert len(a) == 1 and a[0]['umbral_dibujado_con'] == 'puerta'

    con_linea = _Doble()
    con_linea._linea_cruce_estilo = 'linea'
    _entra(con_linea, 'p-igual')
    b = con_linea.alertas()
    assert b[0]['umbral_dibujado_con'] == 'linea'
    # Mismo recorrido, mismo ID: lo ÚNICO que cambia es el rótulo (las
    # fotos se comparan aparte porque son JPEG del mismo cuadro).
    fuera = ('umbral_dibujado_con', 'crop_image', 'image_base64')
    assert {k: v for k, v in a[0].items() if k not in fuera} == \
           {k: v for k, v in b[0].items() if k not in fuera}
    assert a[0]['description'] == b[0]['description']


def test_el_rotulo_va_tambien_en_la_salida_y_en_la_permanencia():
    """La pregunta «¿estaba al revés?» se hace igual ante una salida."""
    p = _Doble()
    p._linea_cruce_estilo = 'puerta'
    _entra(p, 'p-2')
    p.alertas()
    _sale(p, 'p-2', t0=5.0)
    salida = p.alertas()
    assert salida and all(a['umbral_dibujado_con'] == 'puerta' for a in salida)


def test_el_rotulo_NO_sale_cuando_no_rige_ningun_umbral():
    """Un campo puesto para contestar «¿estaba al revés?» no puede ser lo
    primero que engañe: en el área clásica —porque el operador apagó el
    interruptor, o borró la línea y el cliente sigue mandando el último
    estilo— la alerta no puede decir que hubo umbral."""
    p = _Doble(modo='area')
    p._linea_cruce = None                      # borrado / interruptor apagado
    p._linea_cruce_estilo = 'puerta'           # el cliente aún lo manda
    assert p._modo_efectivo() == MODO_AREA
    p._modo_vigente = MODO_AREA
    assert p._estilo_umbral() is None


def test_un_estilo_que_no_es_del_recuadro_no_entra_en_la_alerta():
    """Lo que llega del cliente no se copia a ciegas en el registro."""
    p = _Doble()                               # modo linea: hay umbral
    p._modo_vigente = MODO_LINEA
    for basura in ('', None, 'PUERTA falsa', 'circulo', 123, ['puerta']):
        p._linea_cruce_estilo = basura
        assert p._estilo_umbral() is None, basura
    for bueno, esperado in (('Puerta', 'puerta'), ('  LINEA ', 'linea')):
        p._linea_cruce_estilo = bueno
        assert p._estilo_umbral() == esperado
    sucio = _Doble()
    sucio._linea_cruce_estilo = 'circulo'
    _entra(sucio, 'p-3')
    assert sucio.alertas()[0]['umbral_dibujado_con'] == ''


def test_sin_estilo_la_alerta_sale_igual_con_el_campo_vacio():
    """Un cliente viejo no pierde ni gana nada: el campo va siempre (el
    panel lee con .get()) y queda en blanco."""
    p = _Doble()
    assert p._estilo_umbral() is None
    _entra(p, 'p-4')
    assert p.alertas()[0]['umbral_dibujado_con'] == ''


def test_la_cadena_completa_acepta_el_estilo_del_umbral():
    """Regresión que ya costó un servidor caído (leer_placas, 5-ago-2026):
    la tupla POSICIONAL de app.py y las firmas encadenadas van juntas."""
    import inspect
    from src.app.app import _full_frame_sync, process_image_sync
    for fn in (_full_frame_sync, process_image_sync):
        assert 'linea_cruce_estilo' in inspect.signature(fn).parameters, fn
