"""Pipeline `PermanenciaArea` — cronómetro de personas, en un área o en una puerta.

Un solo modo con **dos lecturas del ROI** que dibujas en el recuadro del
cliente, según lo que haya delante de la cámara:

| `modo`   | El ROI es… | La persona, mientras dura la permanencia… |
|----------|------------|-------------------------------------------|
| `area`   | la zona vigilada | está **a la vista** |
| `puerta` | el vano de la puerta | **no se ve**: se metió |

En los dos casos el ciclo es el mismo y es lo que se pidió: entra → foto y
cronómetro en marcha → sale → foto y tiempo total. Cambia solo *cómo se sabe*
que entró y salió.

## Por qué la puerta necesita otra cosa

En `area` basta con leer la máquina DENTRO/FUERA que `TiendasProcess` ya
mantiene por `persistent_id` y con histéresis. En `puerta` esa máquina diría
«salió» en el instante en que la persona cruza y desaparece del encuadre, y el
cronómetro marcaría cero siempre. El sentido del cruce se deduce del ciclo de
vida del track (`permanencia_puerta.py`), y la estancia queda **abierta y
persistida en disco** mientras la persona está dentro, sin tope de duración
—que era el requisito: no se sabe cuánto puede tardar en salir—.

## Cómo se reconoce a quien sale

Por orden, y siempre diciendo cuál se usó (`identidad_por` en la alerta):

1. **rostro** — el `persistent_id` de la galería facial coincide con el de una
   estancia abierta. Es la vía normal y la más fiable: esa galería es
   persistente, así que sobrevive a las dos horas que la persona pasó dentro.
2. **cuerpo** — embedding OSNet guardado al entrar contra el de quien sale.
   Cubre a quien entra o sale de espaldas. Umbral alto y configurable.
3. **zona** — hay **exactamente una** estancia abierta: quien sale solo puede
   ser esa persona.
4. **no atribuida** — ninguna de las anteriores. Se avisa igual, diciendo que
   la salida no se pudo emparejar. No se inventa una duración.

## Qué NO reimplementa

Detección, BoT-SORT, Re-ID facial y corporal, y la máquina de zonas: todo eso
es de `TiendasProcess` y está calibrado en producción. Este módulo hereda en
vez de copiar porque dos implementaciones del mismo conteo acaban discrepando,
y entonces «cuánto estuvo dentro» tendría dos respuestas.

## Varias PUERTAS en la misma cámara (7-sep-2026)

El recuadro deja dibujar hasta ocho umbrales (`puertas_lineas`), y cada uno
es un SITIO distinto con su aforo y su cronómetro: el almacén, el patio, la
trastienda. No hay detector nuevo —el eje de la flecha doble ES la normal de
la línea— sino uno por puerta, cada uno con su libro (`permanencia_puertas`).

Tres reglas que sostienen todo lo demás:

  - **Sin puertas dibujadas no cambia NADA.** `puerta=None` es el camino de
    siempre en `_reg`/`_espera` y en todo lo que cuelga de ellos: área,
    puerta, cruce y línea suelta se comportan exactamente igual que antes de
    que este reparto existiera, con la misma `dedup_clave` y el mismo libro.
  - **Con puertas, el umbral suelto `linea_cruce` se IGNORA.** Medir el
    mismo vano dos veces emitiría DOS entradas por cada paso.
  - **Cada puerta se pregunta lo suyo.** «¿Ya estaba dentro?» y «¿de quién
    es esta salida?» se le preguntan al libro de ESA puerta; con un libro
    común, una salida por la trasera casaría con una entrada por la
    principal y las fundiría en una visita a un sitio por el que esa persona
    no pasó.

## Reparto en archivos

Cada pieza en el suyo, para que ninguna crezca sin control:

  - `permanencia_estancias.py` — el libro de quién está dentro (+ disco).
  - `permanencia_puerta.py` — cuándo un track cruzó y hacia dónde.
  - `permanencia_puertas.py` — el reparto cuando hay VARIAS puertas.
  - `permanencia_evidencia.py` — las fotos (recorte, banner, disco, base64).

## Alertas

Se encolan en `self._alertas_pendientes`, la cola que el padre drena a
`metadata['alerts']` en cada frame. Desde ahí:

  - **WhatsApp**: lo reenvía el servidor (`app.py` -> `whatsapp_alertas`)
    cuando el cliente manda `enviar_whatsapp`;
  - **Jarvis**: lo reenvía el cliente (`JarvisAlertForwarder`) con el local de
    cada cámara.

No hay que tocar ninguno de los dos: emitir en el formato de siempre basta.
Las alertas llevan `clase_gruesa: 'persona'` para que el panel del cliente las
ponga en la columna de personas y no en el cajón de sastre de eventos.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .analytics.config import AnalyticsConfig
from .permanencia_arbitro import POR_VLM, ArbitroPuerta
from .permanencia_cruce import DetectorCruce
from .permanencia_linea import DetectorLinea, _geometria as _linea_geometria

#: Herramientas del recuadro con las que se puede dibujar el umbral. La
#: «puerta» (flecha doble entrada/salida, 5-sep-2026) produce EXACTAMENTE
#: la misma geometría que la línea: solo cambia lo que ve el operador, y
#: por eso comparten detector. Ver `_estilo_umbral`.
ESTILOS_UMBRAL: tuple[str, ...] = ('linea', 'puerta')
from .permanencia_estancias import (POR_CUERPO, POR_ROSTRO, POR_ZONA,
                                    SIN_ATRIBUIR, Estancia, RegistroEstancias)
from .permanencia_evidencia import (GestorEvidencia, a_b64, escena_b64,
                                    id_corto, recortar)
from .permanencia_puerta import (AMBIGUO, ENTRADA, POR_FLECHA, SALIDA,
                                 DetectorPuerta, _unitario)
from .permanencia_puertas import (ATRIBUTO_ACTIVATE, ATRIBUTO_GESTOR,
                                  ATRIBUTO_LINEAS, ATRIBUTO_NOMBRES,
                                  GestorPuertas, Puerta, clave_dedup,
                                  ruta_libro)
from .tiendas import TiendasProcess

logger = logging.getLogger(__name__)

#: Nombre del modo tal como lo pide el cliente por WebSocket.
MODO = "PermanenciaArea"

MODO_AREA = 'area'
MODO_PUERTA = 'puerta'
#: Tercera lectura (8-ago-2026): el ROI es un umbral VISIBLE que la gente
#: atraviesa sin desaparecer, y la flecha decide si el paso es una entrada o
#: una salida. Se elige explícitamente (`modo: cruce`) o sola: dibujar la
#: flecha con el modo `area` activa esta lectura para esa cámara.
MODO_CRUCE = 'cruce'
#: Cuarta lectura (10-ago-2026): una LÍNEA sobre el umbral de la puerta y el
#: sentido sale de QUÉ LADO a QUÉ LADO se pasó. Para las cámaras gran angular
#: donde la trayectoria se curva y la proyección sobre la flecha se confunde.
#: Se elige explícitamente (`modo: linea`) o sola: dibujar la línea con el
#: modo `area` activa esta lectura para esa cámara (gana a la flecha).
MODO_LINEA = 'linea'

_RAIZ_SERVIDOR = Path(__file__).resolve().parents[3]

#: Sin UNA persona detectada durante este tiempo, el modo lo dice (log + panel)
#: en vez de callar. El silencio ya costó una tarde dos veces: el 6-ago (modos
#: mudos) y el 7-ago (la ventana de iVMS mostraba OTRA cámara y nadie lo notó).
AVISO_SIN_PERSONAS_S = 300.0

#: Alias histórico: las pruebas y el código anterior nombraban así la estancia.
_Estancia = Estancia

# Valores por defecto. Se sobrescriben con config/permanencia_config.yaml.
_DEFECTOS: Dict[str, Any] = {
    # 'area' (la persona se ve mientras dura) | 'puerta' (desaparece dentro).
    'modo': MODO_AREA,
    'alertar_entrada': True,
    'alertar_salida': True,
    # Aviso mientras la persona SIGUE dentro. 0 = desactivado.
    'umbral_permanencia_s': 300,
    # Cada cuanto repetir ese aviso si no se va. 0 = una sola vez.
    'realertar_cada_s': 300,
    # Estancias mas cortas que esto no generan alerta de salida: son gente
    # que cruza el area, no que se queda.
    'min_estancia_para_avisar_s': 3,
    # Filtro de ROCES (11-ago-2026): la alerta de ENTRADA se retiene hasta
    # que la persona lleve estos segundos dentro. Quien roza el area, se
    # asoma un paso o pasa por delante y se va antes de confirmar no genera
    # NI entrada NI salida. El cronometro no pierde nada: la alerta
    # confirmada sale con la hora REAL de entrada. 0 = avisar al instante.
    'confirmar_entrada_s': 0,
    'incluir_foto': True,
    # Cronometrar tambien a quien NO tiene el rostro confirmado.
    #
    # El pipeline padre exige identidad confirmada para su conteo de
    # VISITANTES UNICOS, que tiene que ser exacto. Cronometrar una estancia
    # es otra pregunta: basta con saber que hay alguien ahi. Con `False` en
    # una camara que no saca rostros la maquina de zonas se queda vacia y el
    # modo no mide nada, sin un solo error en el log.
    'contar_sin_rostro': True,
    # -- Evidencia fotografica -------------------------------------------
    'guardar_fotos': True,
    'carpeta_evidencia': 'output/permanencia',
    'retencion_dias': 30,
    'foto_max_ancho': 420,
    'foto_calidad': 80,
    # Foto de entrada + foto de salida en una sola imagen (la de la alerta
    # de salida). Es la que deja ver de un vistazo si es la misma persona.
    'foto_comparativa': True,
    # La ESCENA completa de la entrada apilada sobre la de la salida, con
    # las cintas «Inicio»/«Fin» (formato pedido el 10-ago-2026). Cuando hay
    # escena de entrada guardada, esta foto es la que viaja en la alerta;
    # la comparativa de recortes queda de respaldo.
    'foto_escena': True,
    # -- Re-identificacion para cerrar la estancia -----------------------
    'reid_cuerpo_umbral': 0.86,
    'atribuir_si_unica': True,
    'alertar_no_atribuida': True,
    # -- Libro de estancias ----------------------------------------------
    # Cota de seguridad, NO un tiempo maximo esperado: quien pase de aqui sin
    # salir se cierra como 'sin salida registrada', diciendolo.
    'estancia_ttl_horas': 24,
    'persistir_estancias': True,
    'max_estancias_abiertas': 200,
    # -- Deteccion de cruce (solo modo puerta) ---------------------------
    'puerta': {},
    # -- Paso visible con direccion (solo modo cruce) --------------------
    'cruce': {},
    # -- Linea de cruce (solo modo linea) --------------------------------
    # OJO: si la clave no esta aqui, `_cargar_config` descarta el bloque
    # del YAML aunque exista (solo copia claves conocidas).
    'linea': {},
    # Arbitraje con VLM de los cruces que la geometria NO resuelve. Fuera del
    # bucle de inferencia y APAGADO por defecto: sin activarlo el modelo ni se
    # carga. No decide la direccion (eso lo hace la flecha), solo el residuo.
    'arbitro_vlm': False,
    'arbitro_max_cola': 4,
    # -- Ajustes POR CAMARA ----------------------------------------------
    # {'Puerta Almacen': {'modo': 'puerta', 'umbral_permanencia_s': 3600}}
    'camaras': {},
}

_TITULOS = {
    'entrada': 'Entrada al área',
    'salida': 'Salida del área',
    'permanencia': 'Permanencia prolongada',
    'sin_salida': 'Estancia sin salida',
    'no_atribuida': 'Salida sin identificar',
}
_TITULOS_PUERTA = {
    'entrada': 'Entrada por la puerta',
    'salida': 'Salida por la puerta',
    'permanencia': 'Permanencia prolongada',
    'sin_salida': 'Estancia sin salida',
    'no_atribuida': 'Salida sin identificar',
}
# En cruce el título ES el mensaje pedido: es lo que se lee en WhatsApp y en
# la columna de personas del panel, tal cual.
_TITULOS_CRUCE = {
    'entrada': 'Persona entrando al área',
    'salida': 'Persona saliendo del área',
    'permanencia': 'Permanencia prolongada',
    'sin_salida': 'Estancia sin salida',
    'no_atribuida': 'Persona saliendo del área',
}


def formato_duracion(segundos: float) -> str:
    """`75` -> `'1 min 15 s'`. Legible en un WhatsApp, que es donde se lee."""
    s = int(max(0, segundos))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s} s" if s else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min" if m else f"{h} h"


class PermanenciaArea(TiendasProcess):
    """`TiendasProcess` + cronómetro, evidencia fotográfica y alertas."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._iniciar_permanencia(_cargar_config())

    # ── Arranque ────────────────────────────────────────────────────────
    def _iniciar_permanencia(self, cfg: Dict[str, Any],
                             camara: str = '') -> None:
        """Monta (o remonta) todo el estado del modo para una cámara.

        Se llama en el constructor y otra vez en cuanto se conoce el nombre
        real de la cámara, que `app.py` fija justo antes del primer frame: es
        lo que permite tener `modo: puerta` en una cámara y `area` en otra.
        """
        self.cfg_permanencia = cfg
        self._camara_permanencia = str(camara or '')
        # Politica de la máquina DENTRO/FUERA del padre, POR INSTANCIA. Es lo
        # que decide si una persona sin rostro confirmado llega siquiera a
        # existir para el cronómetro.
        self._zone_requiere_confirmado = not bool(cfg['contar_sin_rostro'])

        carpeta = Path(cfg['carpeta_evidencia'])
        if not carpeta.is_absolute():
            carpeta = _RAIZ_SERVIDOR / carpeta
        self._evidencia = GestorEvidencia(
            carpeta=carpeta,
            retencion_dias=int(cfg['retencion_dias']),
            max_ancho=int(cfg['foto_max_ancho']),
            calidad=int(cfg['foto_calidad']),
            guardar_en_disco=bool(cfg['guardar_fotos']))

        # Sin nombre de cámara NO se persiste. Esta primera construcción pasa
        # al conectarse el cliente, antes del primer frame; `_asegurar_camara`
        # vuelve a montar todo en cuanto se sabe de qué cámara se trata. Sin
        # esta guarda, cada conexión dejaba un `sin-camara.json` vacío en
        # disco: un archivo que no es de nadie y que confunde al mirar.
        libro = None
        nombre = _archivo_seguro(camara)
        if cfg['persistir_estancias'] and nombre:
            libro = carpeta / 'estancias' / f'{nombre}.json'
        self._registro = RegistroEstancias(
            ruta=libro,
            ttl_horas=float(cfg['estancia_ttl_horas']),
            umbral_cuerpo=float(cfg['reid_cuerpo_umbral']),
            max_abiertas=int(cfg['max_estancias_abiertas']),
            persistir=bool(cfg['persistir_estancias']))
        self._registro.cargar()
        #: Alias histórico. Es el MISMO dict que `self._registro.abiertas`.
        self._estancias: Dict[str, Estancia] = self._registro.abiertas

        # ── Varias puertas en esta cámara (7-sep-2026) ──────────────────
        # Atributo PLANO del procesador, y FUERA de la lista que
        # `TiendasProcess._push_state` multiplexa por cámara: ahí dentro
        # `setattr(self, n, cam_state.get(n))` lo dejaría en `None` en cada
        # frame, `_actualizar` reventaría dentro del `try` que SOLO loguea y
        # el modo se quedaría MUDO con el vídeo perfecto (el bug del 6 y el
        # 7-ago-2026). No hay nada que multiplexar: ya existe una instancia
        # del procesador por cámara. Lo fija una prueba de FUENTE en
        # `test_permanencia_multipuerta.py`.
        #
        # El libro de puertas va en su propio fichero
        # (`estancias/puertas/<cam>.json`) y NO en el de la cámara: los dos
        # conviven —el de la cámara sigue siendo el aforo del modo área— y
        # dos escritores sobre el mismo fichero se pisarían el libro entero.
        self._gestor_puertas = GestorPuertas(
            camara=self._camara_permanencia,
            ruta=(ruta_libro(carpeta, nombre)
                  if cfg['persistir_estancias'] else None),
            cfg=cfg)

        # El árbitro solo tiene algo que hacer si el detector le pasa los
        # cruces que no supo resolver, así que se encienden juntos.
        arbitro_on = bool(cfg['arbitro_vlm'])
        cfg_puerta = dict(cfg.get('puerta') or {})
        cfg_puerta['emitir_ambiguos'] = arbitro_on
        self._detector_puerta = DetectorPuerta(cfg_puerta)
        self._detector_cruce = DetectorCruce(dict(cfg.get('cruce') or {}))
        self._detector_linea = DetectorLinea(dict(cfg.get('linea') or {}))
        # Para loguear UNA vez el encendido automático de cada lectura.
        self._aviso_cruce_auto = False
        self._aviso_linea_auto = False
        self._aviso_puertas_auto = False
        # Alertas de entrada RETENIDAS por `confirmar_entrada_s`, por clave.
        # Se emiten cuando la estancia confirma; un roce las tira sin ruido.
        self._entradas_en_espera: Dict[str, Dict[str, Any]] = {}
        if getattr(self, '_arbitro', None) is not None:
            self._arbitro.detener()          # remonte por cambio de cámara
        self._arbitro = ArbitroPuerta(
            activo=arbitro_on, tam_cola=int(cfg['arbitro_max_cola']))
        self._dentro_previo: set = set()
        # Se conservan si ya existían. Este método se vuelve a llamar cuando
        # llega el nombre real de la cámara, y ese procesador es SIEMPRE el
        # mismo `camera_id`: es un renombrado, no otra cámara. Reiniciarlos
        # borraría las visitas contadas hasta el primer frame.
        self._total_visitas = getattr(self, '_total_visitas', 0)
        self._suma_duraciones = getattr(self, '_suma_duraciones', 0.0)
        self._salidas_sin_atribuir = getattr(self, '_salidas_sin_atribuir', 0)
        if not hasattr(self, '_alertas_pendientes'):
            self._alertas_pendientes = []
        # Vigilancia de la escena: desde cuándo no se ve NI UNA persona.
        self._ultima_persona_ts = getattr(self, '_ultima_persona_ts',
                                          time.time())
        self._ultimo_aviso_escena = getattr(self, '_ultimo_aviso_escena', 0.0)
        self._sin_personas_s: Optional[float] = None

        logger.info(
            "PermanenciaArea [%s]: modo=%s, umbral=%s s, entrada=%s, "
            "salida=%s, fotos=%s, %d estancia(s) abiertas",
            camara or 'sin cámara', cfg['modo'], cfg['umbral_permanencia_s'],
            cfg['alertar_entrada'], cfg['alertar_salida'],
            cfg['guardar_fotos'], len(self._registro))

    def _asegurar_camara(self) -> None:
        """Reaplica la configuración cuando se conoce el nombre de la cámara."""
        actual = str(getattr(self, '_camera_display_name', '') or '')
        if actual == getattr(self, '_camara_permanencia', ''):
            return
        if self._registro.abiertas:
            # El libro de la cámara anterior no se pierde: se cierra bien.
            self._registro.guardar(forzar=True)
        # Y el de sus puertas, por lo mismo: `_iniciar_permanencia` monta un
        # gestor nuevo y el anterior se va con lo que tuviera dentro.
        anterior = getattr(self, ATRIBUTO_GESTOR, None)
        if anterior is not None and len(anterior):
            anterior.guardar(forzar=True)
        self._iniciar_permanencia(_cargar_config(actual), actual)

    # ── Cámara o puerta: el MISMO código para las dos (A7) ──────────────
    def _gestor_de_puertas(self) -> GestorPuertas:
        """El gestor de puertas de esta cámara. Siempre en pie.

        Se lee con `getattr` y se repone si falta a propósito: todo lo que
        cuelga de aquí corre dentro del `try` de `process_frame`, que SOLO
        loguea. Un `AttributeError` ahí no se ve por ninguna parte —el vídeo
        sigue perfecto— y el modo entero se queda mudo, que es exactamente el
        fallo del 6 y el 7-ago-2026 que A10 manda evitar. Si hay que reponer
        el gestor se dice en el log: significa que alguien metió el atributo
        en la lista que `TiendasProcess` multiplexa por cámara, o que una
        subclase no pasó por `_iniciar_permanencia`.
        """
        gestor = getattr(self, ATRIBUTO_GESTOR, None)
        if gestor is None:
            logger.warning(
                "PermanenciaArea [%s]: no había gestor de puertas y se repone "
                "en caliente. Es un atributo PLANO del procesador: si acabó "
                "en la lista de `TiendasProcess._push_state` se pondrá a None "
                "en cada frame y las puertas no contarán a nadie.",
                getattr(self, '_camara_permanencia', '') or 'cam')
            gestor = GestorPuertas(
                camara=str(getattr(self, '_camara_permanencia', '') or ''),
                cfg=getattr(self, 'cfg_permanencia', None))
            setattr(self, ATRIBUTO_GESTOR, gestor)
        return gestor

    def _reg(self, puerta: Optional[Puerta] = None) -> RegistroEstancias:
        """El libro de estancias que rige: el de la cámara o el de la puerta.

        `puerta=None` es EL CAMINO DE SIEMPRE —área, puerta, cruce y línea
        suelta mientras no haya nada dibujado— y devuelve exactamente el
        `self._registro` de toda la vida: el comportamiento existente no
        cambia ni una línea por que este reparto exista.

        Con puerta, el libro es el SUYO, y de ahí salen las dos consecuencias
        que A7 exige (7-sep-2026):

          - `_emparejar_salida` busca SOLO aquí. En un libro común, una
            salida por la trasera casaría con una entrada por la principal y
            fundiría las dos en una visita a un sitio por el que esa persona
            no pasó.
          - «¿Ya estaba dentro?» se pregunta SOLO aquí. Con el libro común,
            la segunda entrada de la misma persona (por otra puerta) moría
            por duplicada y ese segundo sitio no contaba a nadie.
        """
        return self._registro if puerta is None else puerta.registro

    def _espera(self, puerta: Optional[Puerta] = None
                ) -> Dict[str, Dict[str, Any]]:
        """Las alertas de entrada RETENIDAS (`confirmar_entrada_s`) que rigen.

        También por puerta, y por la misma razón: la clave de una espera es
        LA PERSONA. Con un diccionario común, el roce de alguien en el patio
        tiraría la alerta buena que estaba esperando confirmación en el
        almacén — la misma clave, dos sitios.
        """
        return (self._entradas_en_espera if puerta is None
                else puerta.entradas_en_espera)

    @property
    def modo_permanencia(self) -> str:
        modo = str(self.cfg_permanencia.get('modo') or MODO_AREA).lower()
        if modo.startswith('puert'):
            return MODO_PUERTA
        if modo.startswith('cruc'):
            return MODO_CRUCE
        if modo.startswith('line'):
            return MODO_LINEA
        return MODO_AREA

    def _estilo_umbral(self) -> Optional[str]:
        """Con qué herramienta se dibujó el umbral: 'linea' o 'puerta'.

        Lo manda el recuadro (`linea_cruce_estilo`) y solo se acepta uno de
        esos dos: un valor cualquiera del cliente no entra en la alerta. No
        pesa en ninguna decisión — solo acompaña al registro (ver
        `_como_se_decidio`).
        """
        # Solo si HAY umbral rigiendo (7-sep-2026). Antes se estampaba
        # tambien en el area clasica: bastaba con que el cliente siguiera
        # mandando el ultimo estilo, o que el operador apagara el
        # interruptor sin borrar el dibujo, para que una alerta de area
        # dijera «umbral dibujado con puerta» sin haber umbral alguno. Un
        # campo puesto para contestar «¿estaba al reves?» no puede ser lo
        # primero que engane.
        if getattr(self, '_modo_vigente', None) != MODO_LINEA:
            return None
        if self._gestor_de_puertas().activo:
            # Con puertas rigiendo, `linea_cruce` está IGNORADO y su estilo
            # no describe nada de lo que se midió. Las puertas se dibujan
            # con la flecha doble entrada/salida, así que eso es lo que
            # tenía delante quien las trazó (7-sep-2026).
            return 'puerta'
        estilo = str(getattr(self, '_linea_cruce_estilo', '') or '').strip().lower()
        return estilo if estilo in ESTILOS_UMBRAL else None

    def _modo_efectivo(self) -> str:
        """El modo que rige ESTE frame.

        `area` + línea dibujada = `linea`; `area` + flecha dibujada =
        `cruce`. Sin tocar el YAML: dibujar es una declaración de intención
        («me importa el sentido del paso») y es el único dato que esas
        lecturas necesitan. Borrar el dibujo devuelve el área clásica. Los
        modos explícitos (`puerta`, `cruce`, `linea`) no se tocan. Si por un
        descuido llegan la línea Y la flecha, gana la LÍNEA: es la
        herramienta nueva y la que el operador dibujó a propósito.

        Y por encima de TODO eso, incluidos los modos explícitos, las
        PUERTAS (7-sep-2026): dibujar umbrales con la flecha doble es la
        misma declaración de intención, y son varias. Rigen la lectura LÍNEA
        (un `DetectorLinea` por puerta), el umbral suelto `linea_cruce` se
        ignora mientras haya alguna —medir el mismo vano dos veces emitiría
        DOS entradas por cada paso— y también se ignora el `modo` del YAML.

        Que ganen a un modo explícito es a propósito: la alternativa es que
        una cámara con `modo: cruce` y tres puertas dibujadas no mida NI UNA
        —el operador dibujó tres umbrales y no pasa nada, sin un error en el
        log—, y ese silencio no lo depura nadie. Ningún recuadro anterior
        puede caer aquí: hasta hoy no existía el campo `puertas_lineas`, así
        que solo llega cuando alguien acaba de dibujarlas.
        """
        modo = self.modo_permanencia
        if self._gestor_de_puertas().activo:
            if not self._aviso_puertas_auto:
                self._aviso_puertas_auto = True
                logger.info(
                    "PermanenciaArea [%s]: %d puerta(s) dibujadas — se activa "
                    "la lectura LÍNEA con un detector por puerta. El modo "
                    "'%s' del YAML y el umbral suelto `linea_cruce` quedan "
                    "ignorados mientras haya puertas.",
                    self._camara_permanencia,
                    len(self._gestor_de_puertas()), modo)
            return MODO_LINEA
        if modo != MODO_AREA:
            return modo
        if _linea_geometria(getattr(self, '_linea_cruce', None)) is not None:
            if not self._aviso_linea_auto:
                self._aviso_linea_auto = True
                logger.info(
                    "PermanenciaArea [%s]: línea dibujada con modo 'area' — "
                    "se activa la lectura LÍNEA (entrada/salida por lado del "
                    "cruce). Borrar la línea vuelve al área clásica.",
                    self._camara_permanencia)
            return MODO_LINEA
        if _unitario(getattr(self, '_flecha_direccion', None)) is None:
            return MODO_AREA
        if not self._aviso_cruce_auto:
            self._aviso_cruce_auto = True
            logger.info(
                "PermanenciaArea [%s]: flecha dibujada con modo 'area' — se "
                "activa la lectura CRUCE (entrada/salida por dirección). "
                "Borrar la flecha vuelve al área clásica.",
                self._camara_permanencia)
        return MODO_CRUCE

    # ── Bucle ───────────────────────────────────────────────────────────
    def process_frame(self, image: np.ndarray, roi=None, activate_roi=False,
                      camera_id: Any = 1, heatmap_activate: bool = False
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
        imagen, metadata = super().process_frame(
            image, roi=roi, activate_roi=activate_roi, camera_id=camera_id,
            heatmap_activate=heatmap_activate)

        try:
            self._recuperar_estado_de_camara(camera_id)
            metadata['permanencia'] = self._actualizar(image)
        except Exception as exc:                            # noqa: BLE001
            # Nunca propaga: si el cronómetro falla, el cliente debe seguir
            # viendo vídeo y detecciones.
            logger.error("PermanenciaArea: fallo actualizando: %s", exc,
                         exc_info=True)

        return imagen, metadata

    def _recuperar_estado_de_camara(self, camera_id: Any) -> None:
        """Reapunta `active_tracks` y `_zone_state` al estado de ESTA cámara.

        `TiendasProcess` multiplexa varias cámaras sobre un mismo objeto con
        `_push_state` / `_pop_state`: al entrar en `process_frame` mueve el
        estado de la cámara a los atributos de `self`, y **al salir restaura
        los de antes**. `active_tracks` y `_zone_state` están en esa lista.

        Como este módulo trabaja DESPUÉS de que el padre termine, leía los
        atributos ya restaurados: dos diccionarios vacíos que no eran de nadie.
        El cronómetro veía cero personas aunque el pipeline acabara de seguir a
        una, y no había forma de notarlo desde fuera — no hay error, solo
        silencio. Lo delató el aforo: `_zone_count` NO está en esa lista, así
        que sobrevivía al pop y marcaba 1 mientras `_zone_state` estaba vacío.

        Se reapunta a los MISMOS diccionarios que usa el pipeline (no copias),
        así que lo que se lea aquí es exactamente lo que acaba de calcular.

        Se lee `camera_states` a pelo y NO se llama a `_get_cam_state`: ese
        método no es un lector inocente —crea el estado si falta y, si le
        falta alguna clave, lo migra reponiendo `_zone_state` a vacío y
        construyendo un clasificador demográfico—. Justo lo que no debe pasar
        al consultar.
        """
        estado = (getattr(self, 'camera_states', None) or {}).get(camera_id)
        if not isinstance(estado, dict):
            return                      # instancia sin multiplexado: ya valen
        tracks = estado.get('active_tracks')
        if isinstance(tracks, dict):
            self.active_tracks = tracks
        zonas = estado.get('_zone_state')
        if isinstance(zonas, dict):
            self._zone_state = zonas

    def _actualizar(self, frame: np.ndarray) -> Dict[str, Any]:
        """Un frame: detecta transiciones, emite alertas y devuelve el resumen."""
        ahora = time.time()
        self._asegurar_camara()
        # Las puertas de ESTE frame, antes de decidir el modo: el cliente
        # manda la lista completa en cada frame y el operador añade, mueve,
        # renombra y quita en caliente.
        self._reconciliar_puertas(ahora)
        # El modo VIGENTE de este frame: los títulos y textos de las alertas
        # se deciden con él (no con el del YAML, que la flecha puede anular).
        self._modo_vigente = self._modo_efectivo()
        cajas = self._cajas_por_persona()
        self._vigilar_escena(ahora, bool(cajas))

        if self._modo_vigente == MODO_PUERTA:
            self._ciclo_puerta(frame, ahora)
        elif self._modo_vigente == MODO_CRUCE:
            self._ciclo_cruce(frame, ahora)
        elif self._modo_vigente == MODO_LINEA:
            self._ciclo_linea(frame, ahora)
        else:
            self._ciclo_area(frame, ahora, cajas)

        self._confirmar_entradas(ahora)
        self._permanencia_prolongada(frame, ahora, cajas)
        self._cerrar_vencidas(ahora)
        self._registro.guardar()
        # Amortiguado (una escritura cada 5 s) y mudo si no hay puertas: se
        # puede llamar en cada frame.
        self._gestor_de_puertas().guardar(ahora=ahora)
        return self._resumen(ahora)

    def _reconciliar_puertas(self, ahora: float) -> None:
        """Deja las puertas EXACTAMENTE como las manda este frame.

        Se hace SIEMPRE, también con el modo área y con el recuadro vacío, y
        no dentro de `_ciclo_linea`: si solo se reconciliara ahí, borrar la
        última puerta dejaría a quien estuviera dentro contado para siempre
        en un sitio que ya no existe — `_ciclo_linea` no volvería a correr y
        nadie lo sacaría nunca.

        Las estancias que se quedan sin puerta vuelven YA CERRADAS y aquí se
        avisan como 'sin salida': no se dejan caer en silencio, porque
        alguien contado dentro y luego desaparecido del aforo sin una sola
        línea es el fallo que no se puede depurar después.

        Con un gestor que nunca tuvo puertas esto es un no-op TOTAL (ni
        fichero, ni cierres, ni log): es lo que manda el cliente en cada
        frame de casi todas las cámaras.
        """
        gestor = self._gestor_de_puertas()
        lineas = getattr(self, ATRIBUTO_LINEAS, None)
        # Interruptor del recuadro. AUSENTE cuenta como True, igual que
        # `linea_cruce_activate`: los clientes que no lo mandan no pueden
        # quedarse sin la lectura. Presente y a False = dibujadas pero
        # desactivadas, que es lo mismo que borrarlas (vuelve el umbral
        # suelto o el área, y quien estuviera dentro se cierra avisando).
        if not bool(getattr(self, ATRIBUTO_ACTIVATE, True)):
            lineas = []
        for estancia in gestor.poner(lineas,
                                     getattr(self, ATRIBUTO_NOMBRES, None),
                                     ahora):
            self._avisar_sin_salida(
                estancia, ahora,
                razon=("el cronómetro se cierra aquí porque esa puerta ya "
                       "no está dibujada."),
                # Las entradas retenidas de esa puerta murieron con ella
                # (avisar de una entrada a un sitio que el operador acaba de
                # borrar sería avisar de una visita a ninguna parte), y la
                # espera de la CÁMARA no es suya: mismo `persistent_id`,
                # otro sitio.
                espera={})

    def _vigilar_escena(self, ahora: float, hay_personas: bool) -> None:
        """Delata a la cámara que lleva demasiado sin ver NI UNA persona.

        No es un error del pipeline: puede que de verdad no pase nadie. Pero
        cuando el operador cambia el canal de la ventana de iVMS, o el ROI
        quedó dibujado sobre otra escena, el síntoma es EXACTAMENTE este
        silencio — y sin este aviso se confunde con "no hay tránsito" hasta
        que alguien pregunta por qué no avanza (7-ago-2026, cámara GENERATOR).
        """
        if hay_personas:
            self._ultima_persona_ts = ahora
            self._sin_personas_s = None
            return
        sin = ahora - self._ultima_persona_ts
        if sin < AVISO_SIN_PERSONAS_S:
            self._sin_personas_s = None
            return
        self._sin_personas_s = sin
        if (ahora - self._ultimo_aviso_escena) >= AVISO_SIN_PERSONAS_S:
            self._ultimo_aviso_escena = ahora
            logger.warning(
                "PermanenciaArea [%s]: %s sin detectar UNA sola persona. Si "
                "ahí debería haber tránsito, revisa que la ventana capturada "
                "muestre la escena correcta (canal de iVMS), y que el ROI y "
                "la flecha estén dibujados sobre ESA escena.",
                self._camara_permanencia or 'cam', formato_duracion(sin))

    # ── Modo ÁREA ───────────────────────────────────────────────────────
    def _ciclo_area(self, frame: np.ndarray, ahora: float,
                    cajas: Dict[str, Tuple[float, float, float, float]]
                    ) -> None:
        """Lee la máquina DENTRO/FUERA del padre (la persona está a la vista)."""
        estados = getattr(self, '_zone_state', {}) or {}
        dentro_ahora = {pid for pid, st in estados.items()
                        if st.get('estado') == 'DENTRO'}

        for pid in dentro_ahora - self._dentro_previo:
            entrada = ahora
            st = estados.get(pid) or {}
            if st.get('entry_time'):
                entrada = float(st['entry_time'])   # el instante real
            self._abrir_estancia(frame, entrada, pid, cajas.get(pid),
                                 zona=MODO_AREA, confirmado=True)

        for pid in self._dentro_previo - dentro_ahora:
            self._cerrar_estancia(frame, ahora, pid, cajas.get(pid),
                                  identidad_por=POR_ROSTRO)

        self._dentro_previo = dentro_ahora

    # ── Modo PUERTA ─────────────────────────────────────────────────────
    def _ciclo_puerta(self, frame: np.ndarray, ahora: float) -> None:
        """El interior no se ve: el sentido del cruce sale del track.

        La flecha «hacia adentro» que dibuja el operador llega como atributo
        (`app.py` la fija junto al nombre de la cámara). Si no la dibujó vale
        `None` y el detector cae a su heurística: no se inventa una dirección.
        """
        visibles: Dict[str, Dict[str, Any]] = {}
        for tid, tr in (getattr(self, 'active_tracks', {}) or {}).items():
            pid = tr.get('persistent_id') or f'prov:{tid}'
            visibles[str(pid)] = {
                'en_zona': bool(tr.get('in_zone')),
                'box': tr.get('box'),
                'confirmado': bool(tr.get('persistent_confirmed')),
            }

        if self._arbitro.activo:
            for clave, obs in visibles.items():
                self._arbitro.recordar(clave, frame, obs.get('box'))

        flecha = getattr(self, '_flecha_direccion', None)
        for cruce in self._detector_puerta.observar(ahora, visibles, flecha):
            if cruce.tipo == ENTRADA:
                self._abrir_estancia(frame, cruce.ts, cruce.clave, cruce.box,
                                     zona=MODO_PUERTA,
                                     confirmado=cruce.confirmado,
                                     cruce=cruce)
            elif cruce.tipo == SALIDA:
                self._salida_por_puerta(frame, ahora, cruce.clave, cruce.box,
                                        cruce.confirmado, cruce=cruce)
            elif cruce.tipo == AMBIGUO:
                self._arbitrar(cruce)

        if self._arbitro.activo:
            self._aplicar_veredictos(frame)

    # ── Modo CRUCE ──────────────────────────────────────────────────────
    def _ciclo_cruce(self, frame: np.ndarray, ahora: float) -> None:
        """El ROI es un umbral VISIBLE: la flecha decide el sentido del paso.

        Mismos tracks y misma flecha que el modo puerta; cambia el detector.
        `DetectorCruce` no infiere nada del ciclo de vida del track — mide el
        desplazamiento sobre la flecha — así que sirve para quien atraviesa
        el ROI y sigue a la vista, que es el caso que `puerta` calla.
        """
        visibles: Dict[str, Dict[str, Any]] = {}
        for tid, tr in (getattr(self, 'active_tracks', {}) or {}).items():
            pid = tr.get('persistent_id') or f'prov:{tid}'
            visibles[str(pid)] = {
                'en_zona': bool(tr.get('in_zone')),
                'box': tr.get('box'),
                'confirmado': bool(tr.get('persistent_confirmed')),
            }

        flecha = getattr(self, '_flecha_direccion', None)
        for cruce in self._detector_cruce.observar(ahora, visibles, flecha):
            if cruce.tipo == ENTRADA:
                self._abrir_estancia(frame, cruce.ts, cruce.clave, cruce.box,
                                     zona=MODO_CRUCE,
                                     confirmado=cruce.confirmado,
                                     cruce=cruce)
            elif cruce.tipo == SALIDA:
                self._salida_por_cruce(frame, ahora, cruce.clave, cruce.box,
                                       cruce.confirmado, cruce=cruce)

    # ── Modo LÍNEA ──────────────────────────────────────────────────────
    def _ciclo_linea(self, frame: np.ndarray, ahora: float) -> None:
        """Una LÍNEA sobre el umbral: el sentido es de qué lado a qué lado.

        Mismos tracks que cruce; cambia la geometría. `DetectorLinea` no
        proyecta recorridos —mira el cambio de lado firme— así que aguanta
        las trayectorias curvas del gran angular que confunden a la flecha.
        No necesita ROI: el propio segmento localiza la puerta.

        Con PUERTAS dibujadas (7-sep-2026) es lo mismo, N veces: un detector
        y un libro por puerta, cada uno con SU segmento. Entrar por «Almacén»
        y salir por «Patio» deja una visita abierta en el primero y una
        salida sin pareja en el segundo, y eso NO es un fallo: son dos sitios
        y esa persona sigue dentro del almacén.
        """
        visibles: Dict[str, Dict[str, Any]] = {}
        for tid, tr in (getattr(self, 'active_tracks', {}) or {}).items():
            pid = tr.get('persistent_id') or f'prov:{tid}'
            visibles[str(pid)] = {
                'box': tr.get('box'),
                'confirmado': bool(tr.get('persistent_confirmed')),
            }

        gestor = self._gestor_de_puertas()
        if gestor.activo:
            # PRECEDENCIA (A3): `puertas_lineas` MANDA sobre `linea_cruce`.
            # El umbral suelto no se mide aquí ni aunque venga dibujado y
            # activado: sería el mismo vano medido dos veces, y cada paso
            # emitiría DOS entradas, con dos cronómetros y dos alertas para
            # una sola persona.
            for puerta, cruce in gestor.observar(ahora, visibles):
                if cruce.tipo == ENTRADA:
                    self._abrir_estancia(frame, cruce.ts, cruce.clave,
                                         cruce.box, zona=MODO_LINEA,
                                         confirmado=cruce.confirmado,
                                         cruce=cruce, puerta=puerta)
                elif cruce.tipo == SALIDA:
                    self._salida_por_cruce(frame, ahora, cruce.clave,
                                           cruce.box, cruce.confirmado,
                                           cruce=cruce, puerta=puerta)
            return

        linea = getattr(self, '_linea_cruce', None)
        for cruce in self._detector_linea.observar(ahora, visibles, linea):
            if cruce.tipo == ENTRADA:
                self._abrir_estancia(frame, cruce.ts, cruce.clave, cruce.box,
                                     zona=MODO_LINEA,
                                     confirmado=cruce.confirmado,
                                     cruce=cruce)
            elif cruce.tipo == SALIDA:
                self._salida_por_cruce(frame, ahora, cruce.clave, cruce.box,
                                       cruce.confirmado, cruce=cruce)

    def _salida_por_cruce(self, frame: np.ndarray, ahora: float, clave: str,
                          box: Optional[Tuple[float, float, float, float]],
                          confirmado: bool, cruce: Any = None,
                          puerta: Optional[Puerta] = None) -> None:
        """Salida en modo cruce: sin pareja también se avisa, sin drama.

        Aquí «no hay estancia abierta» es un caso NORMAL: la persona pudo
        estar dentro desde antes de arrancar el servidor, o entrar por otra
        puerta. Se avisa «Persona saliendo del área» con su hora y su foto, y
        sin inventar un tiempo de permanencia — no el mensaje de «salida sin
        identificar» de puerta, que aquí solo alarmaría en vano.

        Con puertas eso se vuelve todavía más normal: quien entró por el
        almacén y sale por el patio no tiene pareja EN EL PATIO, y así se
        cuenta (`salidas_sin_atribuir` de esa puerta). El aforo del almacén
        no se toca: esa persona sigue dentro.
        """
        estancia, via, score = self._emparejar_salida(frame, clave, box,
                                                      confirmado, puerta)
        if estancia is not None:
            self._cerrar_estancia(frame, ahora, estancia.persistent_id, box,
                                  identidad_por=via, score=score, cruce=cruce,
                                  puerta=puerta)
            return

        if puerta is not None:
            puerta.anotar_sin_atribuir()
        titulo = self._titulo('salida')
        foto = self._foto(frame, box, clave, 'salida', titulo,
                          [f"Hora: {_hora(ahora)}",
                           f"ID: {id_corto(clave)}"])
        donde = (f"Persona saliendo por «{puerta.nombre}»"
                 if puerta is not None else "Persona saliendo del área")
        self._alerta(clave, 'salida',
                     f"{donde} a las {_hora(ahora)}.",
                     foto.b64, ahora=ahora, estancia=None,
                     extra={'foto_salida': foto.ruta,
                            'identidad_por': SIN_ATRIBUIR,
                            **_como_se_decidio(cruce)},
                     puerta=puerta)

    def _salida_por_puerta(self, frame: np.ndarray, ahora: float, clave: str,
                           box: Optional[Tuple[float, float, float, float]],
                           confirmado: bool, cruce: Any = None,
                           puerta: Optional[Puerta] = None) -> None:
        """Empareja a quien sale con una estancia abierta, o lo dice."""
        estancia, via, score = self._emparejar_salida(frame, clave, box,
                                                      confirmado, puerta)
        if estancia is None:
            if puerta is None:
                self._salidas_sin_atribuir += 1
            else:
                puerta.anotar_sin_atribuir()
            if self.cfg_permanencia['alertar_no_atribuida']:
                self._alerta_no_atribuida(frame, ahora, clave, box, puerta)
            return

        self._cerrar_estancia(frame, ahora, estancia.persistent_id, box,
                              identidad_por=via, score=score, cruce=cruce,
                              puerta=puerta)

    def _emparejar_salida(self, frame: Optional[np.ndarray], clave: str,
                          box: Optional[Tuple[float, float, float, float]],
                          confirmado: bool, puerta: Optional[Puerta] = None
                          ) -> Tuple[Optional[Estancia], str, Optional[float]]:
        """A quién corresponde esta salida: rostro -> cuerpo -> única abierta.

        Compartido por puerta y cruce; lo que cambia entre modos es qué se
        hace cuando NO hay pareja, no cómo se busca.

        Se busca SOLO en el libro de SU puerta (A7). En un libro común, una
        salida por la trasera casaría —por cuerpo, o peor, por «única
        abierta»— con una entrada por la principal, y las fundiría en una
        visita a un sitio por el que esa persona no pasó: dos aforos mal y
        una duración inventada, sin un solo error en el log.
        """
        registro = self._reg(puerta)
        estancia = registro.get(clave)
        via = POR_ROSTRO if confirmado else POR_ZONA
        score: Optional[float] = None

        if estancia is None:
            emparejado = registro.emparejar_por_cuerpo(
                self._embedding_cuerpo(frame, box))
            if emparejado is not None:
                estancia, score = emparejado
                via = POR_CUERPO

        if estancia is None and self.cfg_permanencia['atribuir_si_unica']:
            unica = registro.unica_abierta()
            if unica is not None:
                estancia, via = unica, POR_ZONA

        return estancia, via, score

    def _arbitrar(self, cruce: Any) -> None:
        """Manda a la cola del árbitro un cruce que la geometría no resolvió."""
        if not self._arbitro.encolar(cruce.clave, cruce.ts, cruce.box):
            logger.debug("PermanenciaArea [%s]: cruce ambiguo de %s sin "
                         "arbitrar (cola llena o sin recuerdos)",
                         self._camara_permanencia, id_corto(cruce.clave))

    def _aplicar_veredictos(self, frame: np.ndarray) -> None:
        """Recoge lo que el VLM haya resuelto y lo aplica.

        La estancia se abre o se cierra con el instante ORIGINAL del cruce,
        no con el de la respuesta: el cronómetro tiene que contar desde que la
        persona pasó, no desde que el modelo terminó de pensar.
        """
        for veredicto in self._arbitro.recoger():
            extra = {'direccion_por': POR_VLM,
                     'arbitro_seguridad': veredicto.seguridad}
            if veredicto.tipo == 'entrada':
                self._abrir_estancia(frame, veredicto.ts, veredicto.clave,
                                     veredicto.box, zona=MODO_PUERTA,
                                     confirmado=False)
                logger.info("PermanenciaArea [%s]: el árbitro resolvió que %s "
                            "ENTRÓ (seguridad %s)", self._camara_permanencia,
                            id_corto(veredicto.clave), veredicto.seguridad)
            elif veredicto.tipo == 'salida':
                self._salida_por_puerta(frame, veredicto.ts, veredicto.clave,
                                        veredicto.box, confirmado=False)
                logger.info("PermanenciaArea [%s]: el árbitro resolvió que %s "
                            "SALIÓ (seguridad %s)", self._camara_permanencia,
                            id_corto(veredicto.clave), veredicto.seguridad)
            # Se anota en la última alerta encolada de quién vino el veredicto.
            if self._alertas_pendientes:
                self._alertas_pendientes[-1].update(extra)

    # ── Abrir / cerrar una estancia ─────────────────────────────────────
    def _abrir_estancia(self, frame: Optional[np.ndarray], entrada_ts: float,
                        clave: str,
                        box: Optional[Tuple[float, float, float, float]],
                        zona: str, confirmado: bool,
                        cruce: Any = None,
                        puerta: Optional[Puerta] = None) -> None:
        """Abre el cronómetro de una persona en la cámara o en una puerta.

        La pregunta «¿ya estaba dentro?» se le hace al libro de ESA puerta
        (A7): con un libro común, la segunda entrada de la misma persona
        —por otra puerta, que es un sitio distinto— moría aquí por duplicada
        y ese segundo sitio no contaba a nadie.
        """
        if clave in self._reg(puerta):
            return                              # ya estaba dentro: no reinicia

        rasgos = self._rasgos_de(clave)
        titulo = self._titulo('entrada')
        foto = self._foto(frame, box, clave, 'entrada', titulo,
                          [f"Hora: {_hora(entrada_ts)}",
                           f"ID: {id_corto(clave)}"] + _lineas_rasgos(rasgos))

        estancia = Estancia(
            persistent_id=str(clave),
            camara=self._camara_permanencia,
            zona=zona,
            entrada_ts=float(entrada_ts),
            entrada_foto=foto.ruta,
            entrada_b64=foto.b64,
            # La escena ENTERA de este instante: el cuadro «Inicio» de la
            # foto apilada que viajará en la alerta de salida.
            entrada_escena_b64=(escena_b64(
                frame, self.cfg_permanencia['foto_calidad'])
                if self.cfg_permanencia['foto_escena'] else ''),
            box=box,
            emb_cuerpo=self._embedding_cuerpo(frame, box),
            rasgos=rasgos,
            ultima_vista_ts=float(entrada_ts),
            identidad_por=POR_ROSTRO if confirmado else POR_ZONA)
        if puerta is not None:
            # POR QUÉ PUERTA fue, grabado en la estancia y persistido: sin
            # esto, al releer el libro tras un reinicio toda la gente caería
            # en el aforo único (-1) y los aforos por puerta saldrían a cero
            # con el vídeo perfecto.
            puerta.sellar(estancia)
        self._reg(puerta).abrir(estancia)

        if self.cfg_permanencia['alertar_entrada']:
            pendiente = {'clave': str(clave), 'ts': float(entrada_ts),
                         'zona': zona, 'b64': foto.b64, 'ruta': foto.ruta,
                         'extra': _como_se_decidio(cruce),
                         'estancia': estancia, 'puerta': puerta}
            if float(self.cfg_permanencia['confirmar_entrada_s'] or 0) > 0:
                # Se retiene: si resulta ser un roce, morira en silencio.
                self._espera(puerta)[str(clave)] = pendiente
            else:
                self._emitir_entrada(pendiente)

    def _emitir_entrada(self, datos: Dict[str, Any]) -> None:
        puerta = datos.get('puerta')
        donde = {MODO_PUERTA: 'Entró por la puerta',
                 MODO_CRUCE: 'Persona entrando al área',
                 MODO_LINEA: 'Persona entrando al área'}.get(
                     datos['zona'], 'Entró al área')
        if puerta is not None:
            # Con varias puertas, «entrando al área» no dice a cuál de los
            # sitios: el nombre que puso el operador es justo lo que hay que
            # leer en WhatsApp para saber dónde mirar.
            donde = f"Entró por «{puerta.nombre}»"
        self._alerta(datos['clave'], 'entrada',
                     f"{donde} a las {_hora(datos['ts'])}.",
                     datos['b64'], ahora=datos['ts'],
                     estancia=datos['estancia'],
                     extra={'foto_entrada': datos['ruta'],
                            **datos['extra']},
                     puerta=puerta)

    def _confirmar_entradas(self, ahora: float) -> None:
        """Emite las entradas retenidas que ya cumplieron la espera.

        La espera se mide desde la hora REAL de entrada, y la alerta sale
        con esa hora: confirmar no atrasa el cronometro, solo filtra a
        quien no llego a quedarse.

        Cada libro con SU cola: la de la cámara y la de cada puerta. Con una
        cola común, el roce de alguien en el patio tiraría la alerta buena
        que esperaba en el almacén (la clave es la persona, no el sitio).
        """
        self._confirmar_espera(None, ahora)
        for puerta in self._gestor_de_puertas():
            self._confirmar_espera(puerta, ahora)

    def _confirmar_espera(self, puerta: Optional[Puerta],
                          ahora: float) -> None:
        cola = self._espera(puerta)
        if not cola:
            return
        registro = self._reg(puerta)
        espera = float(self.cfg_permanencia['confirmar_entrada_s'] or 0)
        for clave in list(cola):
            datos = cola[clave]
            if clave not in registro.abiertas:
                # La estancia se cerro por otra via sin pasar por
                # `_cerrar_estancia` (no deberia, pero un pendiente
                # huerfano no puede quedarse esperando para siempre).
                cola.pop(clave, None)
                continue
            if (ahora - datos['ts']) >= espera:
                cola.pop(clave, None)
                self._emitir_entrada(datos)

    def _cerrar_estancia(self, frame: Optional[np.ndarray], ahora: float,
                         clave: str,
                         box: Optional[Tuple[float, float, float, float]],
                         identidad_por: str,
                         score: Optional[float] = None,
                         cruce: Any = None,
                         puerta: Optional[Puerta] = None) -> None:
        estancia = self._reg(puerta).cerrar(clave, ahora, score=score)
        if estancia is None:
            return
        estancia.identidad_por = identidad_por
        dur = estancia.duracion(ahora)
        if puerta is None:
            self._total_visitas += 1
            self._suma_duraciones += dur
        else:
            # Los contadores son de la puerta: la media de permanencia del
            # almacén y la del patio son dos números distintos, y sumarlos
            # en uno solo es la respuesta que nadie pidió.
            puerta.anotar_visita(dur)

        cfg = self.cfg_permanencia
        pendiente = self._espera(puerta).pop(str(clave), None)
        if dur < cfg['min_estancia_para_avisar_s']:
            # Un roce: si la entrada seguia retenida, muere aqui con el —
            # NI entrada NI salida. Ese silencio es el filtro.
            return
        if pendiente is not None:
            # La visita confirma al salir: la entrada retenida sale AHORA,
            # antes que la salida y con su hora original.
            self._emitir_entrada(pendiente)
        if not cfg['alertar_salida']:
            return

        titulo = self._titulo('salida')
        lineas = [f"Salió: {_hora(ahora)}",
                  f"Entró: {_hora(estancia.entrada_ts)}",
                  f"Total: {formato_duracion(dur)}",
                  f"ID: {id_corto(clave)} ({identidad_por})"]
        foto = self._foto(frame, box or estancia.box, clave, 'salida',
                          titulo, lineas)

        imagen = foto.b64
        ruta_comparativa = ''
        if (cfg['foto_comparativa'] and estancia.entrada_b64 and foto.b64):
            par = self._evidencia.comparativa(
                estancia.entrada_b64, _de_b64(foto.b64),
                camara=self._camara_permanencia, persistent_id=clave)
            if par:
                imagen, ruta_comparativa = par.b64, par.ruta
        # El formato pedido (10-ago-2026): la ESCENA entera de la entrada
        # apilada sobre la de la salida, con sus cintas «Inicio»/«Fin». Si
        # se puede componer, ES la foto de la alerta; los recortes quedan
        # de respaldo en los campos foto_*.
        if cfg['foto_escena'] and estancia.entrada_escena_b64:
            escenas = self._evidencia.escenas_inicio_fin(
                estancia.entrada_escena_b64, frame,
                camara=self._camara_permanencia, persistent_id=clave)
            if escenas:
                imagen, ruta_comparativa = escenas.b64, escenas.ruta

        # El texto, tal cual se lee en WhatsApp/Jarvis (por eso viaja con
        # `texto_plano`: sin prefijos de evento ni nombre de ventana):
        #     Cámaras {local del selector}
        #     Permanencia en {nombre del área de la cámara}
        #     Inicio / Fin / Tiempo total
        area = (str(getattr(self, '_area_nombre', '') or '').strip()
                or 'el área')
        if puerta is not None:
            # Con varias puertas el nombre del área es el mismo para todas y
            # no dice de cuál se salió; el de la puerta sí.
            area = puerta.nombre
        lineas_detalle = []
        local = str(_nombre_local(getattr(self, '_establecimiento', ''))
                    or '').strip()
        if local:
            lineas_detalle.append(f"Cámaras {local}")
        lineas_detalle += [f"Permanencia en {area}",
                           f"Inicio: {_hora(estancia.entrada_ts)}",
                           f"Fin: {_hora(ahora)}",
                           f"Tiempo total: {_hms(dur)}"]
        detalle = "\n".join(lineas_detalle)

        self._alerta(clave, 'salida', detalle, imagen, ahora=ahora,
                     estancia=estancia, duracion_s=dur,
                     extra={'foto_entrada': estancia.entrada_foto,
                            'foto_salida': foto.ruta,
                            'foto_comparativa': ruta_comparativa,
                            'entrada_base64': estancia.entrada_b64,
                            'identidad_por': identidad_por,
                            'identidad_score': (round(score, 3)
                                                if score is not None else None),
                            # WhatsApp manda el texto TAL CUAL y Jarvis usa
                            # este titulo, sin «cámara {ventana de iVMS}».
                            'texto_plano': True,
                            'titulo_novedad': f"Permanencia en {area}",
                            **_como_se_decidio(cruce)},
                     puerta=puerta)

    def _cerrar_vencidas(self, ahora: float) -> None:
        """Estancias que pasaron el TTL: se cierran DICIENDO que no se vio salir."""
        for estancia in self._registro.caducadas(ahora):
            self._avisar_sin_salida(estancia, ahora)
        # Cada puerta tiene su propio TTL corriendo sobre su propio libro:
        # `caducadas` las devuelve YA cerradas, con su puerta al lado.
        for puerta, estancia in self._gestor_de_puertas().caducadas(ahora):
            self._avisar_sin_salida(estancia, ahora, puerta)

    def _avisar_sin_salida(self, estancia: Estancia, ahora: float,
                           puerta: Optional[Puerta] = None,
                           razon: str = '',
                           espera: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> None:
        """Aviso de una estancia que se cierra sin haber visto salir a nadie.

        La comparten el TTL (`_cerrar_vencidas`) y las HUÉRFANAS de la
        reconciliación —el operador borró un umbral con gente dentro—: el
        aviso es el mismo y lo único que cambia es POR QUÉ se para el
        cronómetro, que es justo lo que hay que leer en WhatsApp. `razon`
        vacía = el límite del TTL, el texto de siempre, carácter por carácter.

        `espera` deja decir de qué cola se rescata la entrada retenida.
        Por defecto la que le toca al libro; las huérfanas pasan `{}` porque
        la suya murió con la puerta y la de la CÁMARA no es suya (mismo
        `persistent_id`, otro sitio).
        """
        cola = self._espera(puerta) if espera is None else espera
        dur = estancia.duracion(ahora)
        # De qué sitio se está hablando: con ocho puertas, «estancia de a3f2
        # vencida» no dice dónde había que ir a mirar. Vacío sin puertas, que
        # es el mensaje de siempre.
        _, sitio = _sello_puerta(puerta, estancia)
        logger.warning("PermanenciaArea [%s]: estancia de %s%s cerrada sin "
                       "salida (%s)", self._camara_permanencia,
                       id_corto(estancia.persistent_id),
                       f' en «{sitio}»' if sitio else '',
                       formato_duracion(dur))
        pendiente = cola.pop(str(estancia.persistent_id), None)
        if pendiente is not None:
            # Entro de verdad (tanto que nunca se le vio salir): la
            # entrada retenida se emite antes del aviso de sin-salida.
            self._emitir_entrada(pendiente)
        if not self.cfg_permanencia['alertar_salida']:
            return
        razon = razon or (
            f"el cronómetro se cierra aquí por el límite de "
            f"{self.cfg_permanencia['estancia_ttl_horas']} h.")
        self._alerta(
            estancia.persistent_id, 'sin_salida',
            f"Entró a las {_hora(estancia.entrada_ts)} y nunca se "
            f"registró su salida. Lleva al menos "
            f"{formato_duracion(dur)}; {razon}",
            estancia.entrada_b64, ahora=ahora, estancia=estancia,
            duracion_s=dur,
            extra={'identidad_por': SIN_ATRIBUIR,
                   'duracion_es_minima': True},
            puerta=puerta)

    # ── Permanencia prolongada ──────────────────────────────────────────
    def _permanencia_prolongada(self, frame: np.ndarray, ahora: float,
                                cajas: Dict[str, Tuple[float, float,
                                                       float, float]]) -> None:
        cfg = self.cfg_permanencia
        umbral = float(cfg['umbral_permanencia_s'] or 0)
        repetir = float(cfg['realertar_cada_s'] or 0)
        if umbral <= 0:
            return

        # El libro de la cámara y, si las hay, el de cada puerta. Los
        # cronómetros son independientes y este aviso también: la misma
        # persona puede llevar diez minutos en el almacén y dos en el patio,
        # y las dos cosas son ciertas a la vez.
        self._prolongada_en(None, frame, ahora, cajas, umbral, repetir)
        for puerta in self._gestor_de_puertas():
            self._prolongada_en(puerta, frame, ahora, cajas, umbral, repetir)

    def _prolongada_en(self, puerta: Optional[Puerta], frame: np.ndarray,
                       ahora: float,
                       cajas: Dict[str, Tuple[float, float, float, float]],
                       umbral: float, repetir: float) -> None:
        for pid, est in list(self._reg(puerta).abiertas.items()):
            if cajas.get(pid) is not None:
                est.box = cajas[pid]            # foto lo más fresca posible
                est.ultima_vista_ts = ahora
            dur = est.duracion(ahora)
            if dur < umbral:
                continue
            if est.avisos_permanencia == 0:
                pass                            # primer aviso: pasa
            elif repetir <= 0:
                continue                        # solo una vez
            elif (ahora - est.ultimo_aviso_ts) < repetir:
                continue                        # todavía no toca repetir
            est.avisos_permanencia += 1
            est.ultimo_aviso_ts = ahora

            visible = cajas.get(pid) is not None
            titulo = self._titulo('permanencia')
            foto = (self._foto(frame, est.box, pid, 'permanencia', titulo,
                               [f"Dentro desde: {_hora(est.entrada_ts)}",
                                f"Lleva: {formato_duracion(dur)}"])
                    if visible else None)
            imagen = foto.b64 if foto else est.entrada_b64
            if puerta is not None:
                donde = f'dentro de «{puerta.nombre}»'
            else:
                donde = ('dentro (entró por la puerta)'
                         if est.zona == MODO_PUERTA else 'dentro del área')
            self._alerta(pid, 'permanencia',
                         f"Lleva {formato_duracion(dur)} {donde} "
                         f"(desde las {_hora(est.entrada_ts)}).",
                         imagen, ahora=ahora, estancia=est, duracion_s=dur,
                         extra={'foto_en_vivo': visible}, puerta=puerta)

    # ── Cierre ──────────────────────────────────────────────────────────
    def cerrar_permanencia(self) -> None:
        """Vuelca el libro de estancias a disco. La llama `app.py` al
        desconectarse el cliente.

        El guardado normal está limitado a uno cada 5 s (escribir en cada
        frame costaría más que la inferencia). Sin este volcado, cerrar el
        cliente justo después de que alguien entre por la puerta perdería esa
        estancia: al reconectar, esa persona ya no estaría dentro y su salida
        saldría como «no atribuida».

        NO cierra las estancias abiertas: quien está dentro sigue dentro
        aunque el operador cierre la ventana.
        """
        try:
            self._registro.guardar(forzar=True)
            # El de las puertas es OTRO fichero: sin este volcado, cerrar la
            # ventana justo después de que alguien entre por el almacén
            # perdería esa estancia y su salida saldría sin pareja.
            gestor = self._gestor_de_puertas()
            gestor.guardar(forzar=True)
            self._arbitro.detener()
            logger.info("PermanenciaArea [%s]: %d estancia(s) abiertas "
                        "guardadas al cerrar (+%d en %d puerta(s))",
                        self._camara_permanencia, len(self._registro),
                        gestor.total_dentro, len(gestor))
        except Exception as exc:                            # noqa: BLE001
            logger.error("PermanenciaArea: no se pudo guardar el libro al "
                         "cerrar: %s", exc)

    # ── Marcado sobre el vídeo ──────────────────────────────────────────
    def _build_detections(self) -> list:
        """Las detecciones de siempre, con el cronómetro pegado a la etiqueta.

        Este cliente corre en **modo directo**: el servidor no dibuja, manda
        las cajas y el recuadro las pinta. Añadiendo aquí el tiempo, la
        persona queda marcada en el vídeo con su cronómetro sin tocar una
        línea del cliente.

        En modo clásico (el servidor dibuja) esto no se usa; el tiempo se ve
        igual en el panel lateral y en las alertas.
        """
        detecciones = super()._build_detections()
        try:
            abiertas = self._registro.abiertas
            gestor = self._gestor_de_puertas()
            if gestor.activo:
                # Con puertas, el libro de la cámara está vacío y sin esto el
                # vídeo se quedaría SIN cronómetros. En la etiqueta cabe uno
                # solo, así que se muestra el más antiguo: quien está dentro
                # de dos sitios a la vez tiene sus dos tiempos en el panel, y
                # elegir uno al azar haría parpadear la etiqueta cada frame.
                abiertas = dict(abiertas)
                for puerta, est in gestor.abiertas():
                    previa = abiertas.get(est.persistent_id)
                    if previa is None or est.entrada_ts < previa.entrada_ts:
                        abiertas[est.persistent_id] = est
            if not abiertas:
                return detecciones
            ahora = time.time()
            por_track = {}
            for tid, tr in (getattr(self, 'active_tracks', {}) or {}).items():
                pid = str(tr.get('persistent_id') or f'prov:{tid}')
                estancia = abiertas.get(pid)
                if estancia is not None:
                    por_track[int(tid)] = estancia
            for det in detecciones:
                estancia = por_track.get(det.get('tracker_id'))
                if estancia is None:
                    continue
                segundos = estancia.duracion(ahora)
                det['label'] = (f"{det.get('label', '')} · "
                                f"{formato_duracion(segundos)} dentro")
                det['permanencia_s'] = round(segundos, 1)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("PermanenciaArea: no se pudo marcar el vídeo (%s)",
                         exc)
        return detecciones

    # ── Piezas ──────────────────────────────────────────────────────────
    def _cajas_por_persona(self) -> Dict[str, Tuple[float, float, float, float]]:
        """`persistent_id` -> bbox, de los tracks vivos de este frame."""
        salida: Dict[str, Tuple[float, float, float, float]] = {}
        for tid, tr in (getattr(self, 'active_tracks', {}) or {}).items():
            pid = tr.get('persistent_id')
            box = tr.get('box')
            if box is None:
                continue
            # Los provisionales SÍ cuentan cuando el modo los cronometra: en
            # puerta y cruce siempre (nadie muestra la cara al cruzar) y en
            # área si `contar_sin_rostro`. Sin sus cajas, esas personas
            # saldrían en la lista del panel pero sus alertas irían sin foto.
            modo = getattr(self, '_modo_vigente', None) or self.modo_permanencia
            acepta_provisionales = (modo in (MODO_PUERTA, MODO_CRUCE,
                                             MODO_LINEA)
                                    or not self._zone_requiere_confirmado)
            if not pid:
                pid = f'prov:{tid}'
            elif str(pid).startswith('prov:') and not acepta_provisionales:
                continue
            try:
                salida[str(pid)] = tuple(float(v) for v in box[:4])
            except (TypeError, ValueError):
                continue
        return salida

    def _embedding_cuerpo(self, frame: Optional[np.ndarray],
                          box: Optional[Tuple[float, float, float, float]]
                          ) -> Optional[np.ndarray]:
        """Firma de apariencia para reconocer a quien sale de espaldas.

        `None` cuando no hay modelo corporal o el recorte es demasiado pequeño
        para dar un embedding útil: mejor sin respaldo que con uno basura, que
        emparejaría a cualquiera.
        """
        body = getattr(self, '_body_reidentifier', None)
        if body is None or not getattr(body, 'is_available', False):
            return None
        if frame is None or box is None:
            return None
        try:
            crop = self._body_crop(frame, box,
                                   int(AnalyticsConfig.BODY_REID_MIN_CROP_H))
            if crop is None:
                return None
            return body.compute_embedding(crop)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("PermanenciaArea: embedding corporal fallido (%s)",
                         exc)
            return None

    def _rasgos_de(self, persistent_id: str) -> Dict[str, Any]:
        """Género y rango de edad consolidados, si la galería los tiene."""
        reid = getattr(self, '_reidentifier', None)
        if reid is None or str(persistent_id).startswith('prov:'):
            return {}
        try:
            for tid, tr in (getattr(self, 'active_tracks', {}) or {}).items():
                if str(tr.get('persistent_id')) != str(persistent_id):
                    continue
                demo = reid.get_demographics_for_person(tid)
                if not demo:
                    continue
                return {'genero': demo.get('gender'),
                        'rango_edad': demo.get('age_range'),
                        'visitas': demo.get('visit_count')}
        except Exception as exc:                            # noqa: BLE001
            logger.debug("PermanenciaArea: rasgos no disponibles (%s)", exc)
        return {}

    def _titulo(self, tipo: str) -> str:
        # El modo VIGENTE, no el del YAML: la flecha puede haber activado la
        # lectura cruce y sus alertas deben titularse como tal.
        modo = getattr(self, '_modo_vigente', None) or self.modo_permanencia
        # La línea comparte títulos con cruce: son el MISMO mensaje pedido
        # («Persona entrando/saliendo del área»), cambia cómo se midió.
        tabla = {MODO_PUERTA: _TITULOS_PUERTA,
                 MODO_CRUCE: _TITULOS_CRUCE,
                 MODO_LINEA: _TITULOS_CRUCE}.get(modo, _TITULOS)
        return tabla.get(tipo, 'Permanencia')

    def _foto(self, frame: Optional[np.ndarray],
              box: Optional[Tuple[float, float, float, float]],
              clave: str, momento: str, titulo: str, lineas: List[str]):
        from .permanencia_evidencia import Evidencia
        if not self.cfg_permanencia['incluir_foto'] or frame is None:
            return Evidencia()
        return self._evidencia.capturar(
            frame, box, camara=self._camara_permanencia,
            persistent_id=clave, momento=momento, titulo=titulo,
            lineas=lineas)

    def _alerta_no_atribuida(self, frame: Optional[np.ndarray], ahora: float,
                             clave: str,
                             box: Optional[Tuple[float, float, float, float]],
                             puerta: Optional[Puerta] = None) -> None:
        """Alguien salió y no se pudo decir quién. Se avisa SIN inventar tiempo.

        El número de estancias abiertas es el del libro de SU puerta: decir
        «3 abiertas» contando las de los otros sitios haría creer que había
        con quién emparejar.
        """
        titulo = self._titulo('no_atribuida')
        foto = self._foto(frame, box, clave, 'salida', titulo,
                          [f"Hora: {_hora(ahora)}",
                           "No se pudo emparejar con ninguna entrada"])
        abiertas = len(self._reg(puerta))
        donde = (f"por «{puerta.nombre}»" if puerta is not None
                 else "por la puerta")
        self._alerta(
            clave, 'no_atribuida',
            f"Salió alguien {donde} a las {_hora(ahora)} y no se pudo "
            f"emparejar con ninguna entrada registrada "
            f"({abiertas} estancia(s) abierta(s)). Sin tiempo de permanencia.",
            foto.b64, ahora=ahora, estancia=None,
            extra={'identidad_por': SIN_ATRIBUIR,
                   'estancias_abiertas': abiertas},
            puerta=puerta)

    def _alerta(self, pid: str, tipo: str, descripcion: str,
                imagen_b64: str, ahora: float,
                estancia: Optional[Estancia] = None,
                duracion_s: Optional[float] = None,
                extra: Optional[Dict[str, Any]] = None,
                puerta: Optional[Puerta] = None) -> None:
        """Encola una alerta en el formato que ya entienden WhatsApp y Jarvis.

        Nunca lanza: esto corre dentro del bucle de inferencia.
        """
        try:
            titulo = self._titulo(tipo)
            idx_puerta, nombre_puerta = _sello_puerta(puerta, estancia)
            alerta: Dict[str, Any] = {
                'event_type': titulo,
                'class_name': titulo,
                'description': descripcion,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S',
                                           time.localtime(ahora)),
                'camera_name': self._camara_permanencia,
                'person_uuid': str(pid),
                'crop_image': imagen_b64 or '',
                'image_base64': imagen_b64 or '',
                # El panel del cliente reparte por aquí: sin esto las alertas
                # caerían en la columna «Eventos» junto a los pedidos de Hummus.
                'clase_gruesa': 'persona',
                # Unidad de repetición para el anti-flood de WhatsApp y la
                # agrupación de Jarvis: LA PERSONA, y desde el 7-sep-2026
                # LA PERSONA EN ESA PUERTA. Los dos deduplican por (cámara,
                # tipo de evento) por defecto, así que dos personas entrando
                # con pocos segundos de diferencia habrían generado un solo
                # aviso y el tiempo de la segunda se habría perdido; y sin el
                # segmento de la puerta, la misma persona entrando por dos
                # sitios (o dos personas a la vez por puertas distintas)
                # comparte clave y la SEGUNDA muere en el enfriamiento —
                # justo la alerta que demuestra que son dos sitios.
                #
                # El segmento va DESPUÉS del tipo (`perm|cam|tipo|p1|pid`)
                # para no romper a quien filtra por el prefijo
                # `perm|cam|tipo|`, y SIN puertas la clave es exactamente la
                # de siempre, carácter por carácter.
                'dedup_clave': clave_dedup(self._camara_permanencia, tipo,
                                           pid, idx_puerta),
                # Por qué puerta pasó esto. -1 = el aforo de la CÁMARA, que
                # es el único caso mientras no haya nada dibujado.
                'puerta_idx': idx_puerta,
                'puerta_nombre': nombre_puerta,
                # Extras que el panel ignora sin romperse (lee con .get()).
                'tipo_permanencia': tipo,
                'modo_permanencia': self.modo_permanencia,
                # Con QUÉ utensilio dibujó el operador el umbral: 'linea'
                # (rótulos DENTRO/FUERA) o 'puerta' (flecha doble
                # ENTRADA/SALIDA). No interviene en la decisión —los dos dan
                # la misma geometría y la lee el mismo detector—, pero es lo
                # primero que hay que saber cuando alguien reporta entradas
                # contadas como salidas: ese fallo casi siempre es un umbral
                # dibujado al revés, y saber qué tenía delante quien lo trazó
                # ahorra media investigación. Vacío en clientes anteriores.
                'umbral_dibujado_con': self._estilo_umbral() or '',
                'duracion_s': (round(float(duracion_s), 1)
                               if duracion_s is not None else None),
                'duracion_legible': (formato_duracion(duracion_s)
                                     if duracion_s is not None else ''),
                'id_corto': id_corto(pid),
            }
            if estancia is not None:
                alerta.update({
                    'entrada_hora': _hora(estancia.entrada_ts),
                    'entrada_ts': estancia.entrada_ts,
                    'zona': estancia.zona,
                    'identidad_por': estancia.identidad_por,
                    'genero': estancia.rasgos.get('genero'),
                    'rango_edad': estancia.rasgos.get('rango_edad'),
                })
            if extra:
                alerta.update(extra)

            if not hasattr(self, '_alertas_pendientes'):
                self._alertas_pendientes = []
            self._alertas_pendientes.append(alerta)
            # Cota de memoria: si el cliente no recoge, no se acumula sin fin.
            if len(self._alertas_pendientes) > 200:
                self._alertas_pendientes = self._alertas_pendientes[-100:]
            logger.info("PermanenciaArea [%s/%s] %s: %s",
                        self._camara_permanencia or 'cam', tipo,
                        id_corto(pid), descripcion)
        except Exception as exc:                            # noqa: BLE001
            logger.debug("PermanenciaArea: no se pudo encolar alerta: %s", exc)

    def _fila_persona(self, pid: Any, est: Estancia, ahora: float,
                      puerta: Optional[Puerta] = None) -> Dict[str, Any]:
        """Una fila de `personas` para el panel.

        Lleva la puerta a propósito (A8): la llave de cada fila del panel
        incluye el sitio. Con solo la persona, alguien que está dentro de dos
        puertas genera dos filas que se pelean el mismo widget y el
        cronómetro alterna entre los dos tiempos. Sin puertas es `-1` y `''`,
        que es lo que esas filas siempre fueron: el aforo de la cámara.
        """
        idx, nombre = _sello_puerta(puerta, est)
        return {'persistent_id': pid,
                'id_corto': id_corto(pid),
                'segundos': round(est.duracion(ahora), 1),
                'legible': formato_duracion(est.duracion(ahora)),
                'desde': _hora(est.entrada_ts),
                'desde_ts': est.entrada_ts,
                'zona': est.zona,
                'visible': (ahora - est.ultima_vista_ts) < 3.0,
                'genero': est.rasgos.get('genero'),
                'rango_edad': est.rasgos.get('rango_edad'),
                'avisado': est.avisos_permanencia > 0,
                'puerta': idx,
                'puerta_nombre': nombre}

    def _resumen(self, ahora: float) -> Dict[str, Any]:
        """Lo que ve el cliente en `metadata['permanencia']`.

        Con puertas dibujadas esto se AMPLÍA, no se sustituye (A8): los
        totales de arriba siguen siendo la UNIÓN —el panel de siempre sigue
        leyendo `dentro_ahora` y `personas` sin enterarse de nada— y debajo
        se añade una fila por puerta con sus propios números. La misma
        persona dentro de dos puertas cuenta dos veces en la unión: son dos
        sitios y cada uno tiene su cronómetro corriendo.
        """
        gestor = self._gestor_de_puertas()
        filas = [self._fila_persona(pid, est, ahora)
                 for pid, est in self._registro.abiertas.items()]
        por_puerta: Dict[int, List[Dict[str, Any]]] = {}
        for puerta, est in gestor.abiertas():
            fila = self._fila_persona(est.persistent_id, est, ahora, puerta)
            por_puerta.setdefault(puerta.indice, []).append(fila)
            filas.append(fila)
        dentro = sorted(filas, key=lambda x: x['segundos'], reverse=True)

        visitas = self._total_visitas + sum(p.total_visitas for p in gestor)
        suma = self._suma_duraciones + sum(p.suma_duraciones for p in gestor)
        media = (suma / visitas) if visitas else None
        sin_atribuir = (self._salidas_sin_atribuir
                        + sum(p.salidas_sin_atribuir for p in gestor))
        return {
            # El modo VIGENTE: si la flecha activó la lectura cruce, es lo que
            # el panel debe decir que está pasando.
            'modo': getattr(self, '_modo_vigente', None) or self.modo_permanencia,
            'camara': self._camara_permanencia,
            'dentro_ahora': len(dentro),
            'personas': dentro,
            'visitas_completadas': visitas,
            # `None` (no 0) mientras no haya ninguna visita cerrada: sin datos
            # no hay media, y un 0 se leería como «nadie se queda».
            'permanencia_media_s': (round(media, 1)
                                    if media is not None else None),
            'umbral_s': self.cfg_permanencia['umbral_permanencia_s'],
            'salidas_sin_atribuir': sin_atribuir,
            # Una fila por puerta dibujada, con SUS números (A8). Lista
            # VACÍA mientras no haya ninguna: el panel lee siempre la misma
            # clave y no tiene que distinguir «no hay» de «no existe».
            'puertas': gestor.resumen(por_puerta),
            # Segundos sin ver NI UNA persona (None mientras todo es normal).
            # El panel del cliente lo convierte en el aviso "¿la cámara
            # muestra la escena correcta?".
            'sin_personas_s': (round(self._sin_personas_s)
                               if self._sin_personas_s else None),
        }


# ── Utilidades ──────────────────────────────────────────────────────────
def _hora(ts: float) -> str:
    return time.strftime('%H:%M:%S', time.localtime(ts))


def _hms(segundos: float) -> str:
    """`17` -> `'00:00:17'`. El formato del «Tiempo total» pedido."""
    s = int(max(0, segundos))
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _nombre_local(establecimiento: Any) -> str:
    """El selector puede mandar el nombre a secas o el dict del local."""
    if isinstance(establecimiento, dict):
        return str(establecimiento.get('nombre')
                   or establecimiento.get('name') or '')
    return str(establecimiento or '')


def _como_se_decidio(cruce: Any) -> Dict[str, Any]:
    """Qué llevó a decidir el sentido del cruce, para que viaje en la alerta.

    Que la dirección la haya dado la flecha (una medida) o el ciclo de vida
    del track (una inferencia sobre la ausencia) cambia cuánto se puede fiar
    uno del dato. Se dice; no se esconde detrás de un mismo mensaje.

    CON QUÉ se dibujó el umbral (línea o puerta) NO va aquí: viaja en
    `umbral_dibujado_con`, que `_alerta` pone en TODAS las alertas del modo
    —no solo en las de cruce— porque la pregunta «¿estaba al revés?» se hace
    igual ante una permanencia que ante una entrada.
    """
    if cruce is None:
        return {}
    proyeccion = getattr(cruce, 'proyeccion', None)
    return {
        'direccion_por': getattr(cruce, 'via', None),
        'direccion_avance': (round(float(proyeccion), 2)
                             if proyeccion is not None else None),
    }


def _sello_puerta(puerta: Optional[Puerta],
                  estancia: Optional[Estancia] = None) -> Tuple[int, str]:
    """`(índice, nombre)` de la puerta a la que pertenece esto. `(-1, '')` si
    no hay ninguna, que es el aforo de la CÁMARA y el caso de siempre.

    Cuando la puerta ya no existe —una huérfana de la reconciliación: el
    operador borró el umbral con gente dentro— el sello se lee de la
    estancia, que lo lleva grabado desde que se abrió. Sin eso, ese aviso
    saldría con la clave del aforo único y podría morir deduplicado contra
    el de otra persona.

    Se compara con `None` a propósito en vez de `int(x or -1)`: la puerta 0
    es la PRIMERA (es un índice, no un contador) y ese `or` la mandaría al
    aforo de la cámara, borrando de las alertas justo la puerta principal.
    """
    if isinstance(puerta, Puerta):
        return int(puerta.indice), str(puerta.nombre)
    idx = getattr(estancia, 'puerta', None) if estancia is not None else None
    if idx is None:
        return -1, ''
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return -1, ''
    if idx < 0:
        return -1, ''
    return idx, str(getattr(estancia, 'puerta_nombre', '') or '')


def _lineas_rasgos(rasgos: Dict[str, Any]) -> List[str]:
    genero = rasgos.get('genero')
    edad = rasgos.get('rango_edad')
    if not genero and not edad:
        return []
    return [f"{genero or '?'} · {edad or '?'}"]


def _de_b64(dato: str):
    from .permanencia_evidencia import de_b64
    return de_b64(dato)


def _archivo_seguro(texto: Any) -> str:
    limpio = ''.join(c for c in str(texto or '')
                     if c.isalnum() or c in '_-')
    return limpio[:40]


def _recorte_b64(frame: np.ndarray,
                 box: Optional[Tuple[float, float, float, float]]) -> str:
    """Recorte de la persona en base64. Cadena vacía si no se puede.

    Se conserva porque otros módulos y las pruebas la nombran; la lógica vive
    ahora en `permanencia_evidencia`.
    """
    return a_b64(recortar(frame, box))


def ruta_config() -> Path:
    propia = (os.getenv('PERMANENCIA_CONFIG') or '').strip()
    return (Path(propia) if propia
            else _RAIZ_SERVIDOR / 'config' / 'permanencia_config.yaml')


def _cargar_config(camara: str = '') -> Dict[str, Any]:
    """Umbrales del YAML sobre los defectos, con override por cámara.

    Nunca lanza: un YAML roto degrada a los valores por defecto y lo dice en
    el log, en vez de tumbar el pipeline.
    """
    cfg = dict(_DEFECTOS)
    ruta = ruta_config()
    if not ruta.is_file():
        logger.info("PermanenciaArea: sin %s; se usan los valores por "
                    "defecto", ruta.name)
        return cfg
    try:
        import yaml
        with open(ruta, encoding='utf-8') as f:
            datos = yaml.safe_load(f) or {}
        for clave in cfg:
            if clave in datos:
                cfg[clave] = datos[clave]
        # Override por cámara: lo que no diga hereda de lo global.
        propio = (cfg.get('camaras') or {}).get(str(camara or ''))
        if isinstance(propio, dict):
            for clave, valor in propio.items():
                if clave in cfg:
                    cfg[clave] = valor
            logger.info("PermanenciaArea: %s usa ajustes propios (%s)",
                        camara, ', '.join(sorted(propio)))
    except Exception as exc:                                # noqa: BLE001
        logger.error("PermanenciaArea: %s ilegible (%s); valores por defecto",
                     ruta, exc)
    return cfg
