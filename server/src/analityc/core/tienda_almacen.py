"""src/analityc/core/tienda_almacen.py — memoria con FECHA de la analítica de
tienda (HITO 1 de la obra «tienda al 100 %», 18-ago-2026).

## Por qué existe

El pipeline mide bien y **recuerda mal**. Entradas, salidas, visitantes únicos
y permanencia viven en atributos del procesador (`_total_entries`,
`_unique_entered`, `_zone_state`), y el procesador **renace en cada
reconexión** del cliente (~5 s tras cualquier corte, ver `app.py`). Encima, el
único rastro en disco era `analytics_report_client_<id>_<device>.json`, donde
`<id>` sale de `client_id = f"client_{id(websocket)}"`: la dirección de memoria
del websocket. Cambia en cada sesión y Python la **reutiliza** tras el GC, así
que ni identifica la sesión ni se puede sumar.

Consecuencia para el dashboard: la sección «Hoy en la tienda» enseñaba «lo que
lleve viva esta sesión». Si el servidor llevaba tres días encendido, «hoy» eran
tres días; si se había reiniciado hacía diez minutos, «hoy» eran diez minutos.

Aquí los hechos quedan con **marca de tiempo y jornada**, así que «hoy» es hoy,
«ayer» es ayer, y dos sesiones del mismo día SUMAN.

## Qué guarda (y qué no)

Dos tablas, con la misma distinción que hizo bien `stand_almacen.py`:

- `tienda_visitas` — **la unidad de verdad**: una fila por
  (jornada, cámara, persona). De aquí salen todos los conteos. Los «únicos» de
  un rango son `COUNT(DISTINCT persistent_id)`, no la suma de los únicos
  diarios (esa suma contaría dos veces a quien vuelve el martes).
- `tienda_eventos` — bitácora inmutable, una fila por transición. Sirve para
  la serie por hora (la curva del día) y **no se cuenta**: una misma persona
  genera muchos eventos.

**El personal enrolado no entra**, y no por un filtro de aquí: quien llama ya
lo excluye. `_process_zone_transitions` salta los tracks con `es_personal` y
`_emitir_visitante` corta en seco al personal. Las cifras son solo de clientes,
igual que en el resto del producto.

## Cómo escribe sin frenar la inferencia

Esto se llama DENTRO del bucle de frames, así que:

- Las anotaciones solo **encolan en memoria** (una tupla, sin tocar disco).
- El volcado va **amortiguado**: como mucho uno cada `PERIODO_VOLCADO_S`, en un
  solo lote y una sola transacción. Es el patrón que ya demostró
  `anaqueles_registro.py`.
- **Un fallo de disco NUNCA tumba el frame**: todo método público es
  defensivo y se limita a contar el fallo en `fallos`. Perder analítica es
  malo; perder el vídeo en vivo es peor.
- La conexión se **cierra de verdad** en cada volcado. Gotcha ya pagado en
  `estacionamiento_placas.py` y documentado en `stand_almacen.py`:
  `with sqlite3.connect(...) as con` gestiona la TRANSACCIÓN, no la conexión;
  sin cerrarla se fugan descriptores y en Windows el `.db` queda bloqueado.

## Configuración (regla 6: cero valores incrustados)

Todo se puede mover por entorno, sin tocar código:

| Variable | Defecto | Qué es |
|---|---|---|
| `ELDE_TIENDA_DB` | `output/tienda.db` | dónde vive la base |
| `ELDE_TIENDA_VOLCADO_S` | `5.0` | mínimo entre escrituras a disco |
| `ELDE_TIENDA_RETENCION_DIAS` | `396` | 13 meses: permite comparar contra el mismo mes del año pasado |
| `ELDE_TIENDA_CORTE_JORNADA_H` | `0` | hora local a la que empieza la jornada |

## La jornada, en UNA sola función

`jornada_de()` es el único sitio del árbol que decide a qué día pertenece un
instante. `ts` siempre es época (UTC, `time.time()`); la jornada se deriva en
hora **local**, que es la que vive el operador de la tienda. Con
`ELDE_TIENDA_CORTE_JORNADA_H = 0` la jornada va de medianoche a medianoche; con
`22` iría de las 22:00 a las 22:00, para una tienda que cierra de madrugada.
"""

from __future__ import annotations

import atexit
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import date as _date
from datetime import timedelta as _timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: `src/analityc/core/tienda_almacen.py` -> parents[3] es `server/`.
_RAIZ_SERVIDOR = Path(__file__).resolve().parents[3]

#: Mínimo entre escrituras a disco. 5 s pierde como mucho 5 s de conteo en un
#: corte duro, y no convierte el disco en el cuello del bucle de inferencia.
PERIODO_VOLCADO_S: float = float(os.getenv('ELDE_TIENDA_VOLCADO_S', '5.0'))

#: 13 meses (~396 días). El plazo lo eligió el dueño del producto: es lo que
#: hace falta para comparar un mes contra el mismo mes del año pasado. Con
#: tráfico alto son ~1-2 millones de filas, que para SQLite no es nada.
RETENCION_DIAS: int = int(os.getenv('ELDE_TIENDA_RETENCION_DIAS', '396'))

#: Hora LOCAL a la que arranca la jornada. 0 = medianoche.
CORTE_JORNADA_H: int = int(os.getenv('ELDE_TIENDA_CORTE_JORNADA_H', '0'))

#: Tipos de la bitácora. Constantes para que un typo no cree una categoría
#: fantasma que luego no cuadre en los totales.
EVENTO_ENTRADA = 'entrada'
EVENTO_SALIDA = 'salida'


def ruta_bd() -> Path:
    """`server/output/tienda.db`, junto al resto de salidas del servidor."""
    valor = os.getenv('ELDE_TIENDA_DB', 'output/tienda.db')
    ruta = Path(valor)
    return ruta if ruta.is_absolute() else _RAIZ_SERVIDOR / ruta


def jornada_de(ts: Optional[float] = None,
               corte_h: Optional[int] = None) -> str:
    """`AAAA-MM-DD` de la jornada a la que pertenece un instante.

    EL ÚNICO sitio que hace esta cuenta. `ts` es época (UTC); la jornada se
    calcula en hora LOCAL, que es la que vive quien atiende la tienda.

    Con `corte_h = 0` (defecto) la jornada es el día natural. Con `corte_h = 22`
    la jornada empieza a las 22:00: un cliente de las 23:30 del lunes pertenece
    a la jornada del lunes, y uno de la 1:00 del martes, también.
    """
    if ts is None:
        ts = time.time()
    corte = CORTE_JORNADA_H if corte_h is None else int(corte_h)
    # Restar el corte en segundos y leer el día local resultante: así el tramo
    # que va de medianoche al corte cae en la jornada del día ANTERIOR.
    return time.strftime('%Y-%m-%d', time.localtime(float(ts) - corte * 3600))


def dias_de(desde: str, hasta: str) -> int:
    """Cuántas jornadas abarca la ventana, ambos extremos incluidos."""
    d = _date.fromisoformat(desde)
    h = _date.fromisoformat(hasta)
    return max(1, (h - d).days + 1)


def ventana_anterior(desde: str, hasta: str) -> Tuple[str, str]:
    """La ventana de IGUAL longitud justo anterior, para comparar.

    Hoy contra ayer; esta semana contra la anterior. Se compara con el tramo
    equivalente y no con «el mes pasado» a ojo, que es la trampa clásica de
    los dashboards: 7 días contra 30 siempre «cae».
    """
    largo = dias_de(desde, hasta)
    fin = _date.fromisoformat(desde) - _timedelta(days=1)
    ini = fin - _timedelta(days=largo - 1)
    return ini.isoformat(), fin.isoformat()


class AlmacenTienda:
    """Las dos tablas de la analítica de tienda, con su volcado amortiguado."""

    def __init__(self, ruta: Optional[Path] = None,
                 periodo_volcado_s: Optional[float] = None) -> None:
        self.ruta = Path(ruta or ruta_bd())
        self._periodo = (PERIODO_VOLCADO_S if periodo_volcado_s is None
                         else float(periodo_volcado_s))
        self._lock = threading.Lock()
        #: Cola en memoria. Un solo escritor: el que llame a `volcar`.
        self._pendientes: List[Tuple] = []
        #: En cero a proposito: el PRIMER hecho de la sesion se escribe al
        #: momento, sin esperar un periodo. Un corte a los dos segundos de
        #: conectar no puede llevarselo, y a partir de ahi ya amortigua.
        self._ultimo_volcado = 0.0
        #: Escrituras que fallaron. Se cuentan en vez de reventar el frame; el
        #: dashboard puede enseñarlo el día que haga falta (lección de H-25:
        #: un envío mudo no distingue «apagado» de «roto»).
        self.fallos = 0
        self._preparado = False
        self._preparar()

    # ── Conexión y esquema ──────────────────────────────────────────────
    @contextmanager
    def _conectar(self):
        """Conexión que se CIERRA de verdad (ver el gotcha de la cabecera)."""
        con = sqlite3.connect(str(self.ruta), timeout=10)
        con.row_factory = sqlite3.Row
        try:
            # WAL: lectores (los endpoints del dashboard) no bloquean al
            # escritor (el bucle de inferencia). busy_timeout evita el
            # «database is locked» cuando coinciden.
            con.execute('PRAGMA journal_mode=WAL')
            con.execute('PRAGMA busy_timeout=10000')
            with con:
                yield con
        finally:
            con.close()

    def _preparar(self) -> None:
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            with self._conectar() as con:
                con.execute('''
                    CREATE TABLE IF NOT EXISTS tienda_visitas (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        jornada       TEXT    NOT NULL,
                        camera_id     TEXT    NOT NULL,
                        persistent_id TEXT    NOT NULL,
                        ts_primera    REAL    NOT NULL,
                        ts_ultima     REAL    NOT NULL,
                        entradas      INTEGER NOT NULL DEFAULT 0,
                        salidas       INTEGER NOT NULL DEFAULT 0,
                        permanencia_s REAL    NOT NULL DEFAULT 0.0,
                        genero        TEXT,
                        edad          TEXT,
                        UNIQUE (jornada, camera_id, persistent_id)
                    )''')
                con.execute('''
                    CREATE TABLE IF NOT EXISTS tienda_eventos (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts            REAL NOT NULL,
                        jornada       TEXT NOT NULL,
                        camera_id     TEXT NOT NULL,
                        persistent_id TEXT NOT NULL,
                        tipo          TEXT NOT NULL,
                        permanencia_s REAL
                    )''')
                # Índice de CAPTURAS (HITO 3). No es otra fuente de verdad:
                # el disco lo sigue siendo. Esto es una caché para que el
                # coste de una petición deje de crecer con el número de fotos
                # acumuladas — antes cada refresco del dashboard abría y
                # parseaba TODOS los sidecars, cinco veces, cada 15 s.
                # `mtime` es lo que permite re-parsear solo lo que cambió.
                con.execute('''
                    CREATE TABLE IF NOT EXISTS tienda_capturas (
                        stem              TEXT PRIMARY KEY,
                        ts                REAL NOT NULL,
                        jornada           TEXT NOT NULL,
                        camera            TEXT,
                        device_id         TEXT,
                        genero            TEXT,
                        edad              TEXT,
                        person_uuid       TEXT,
                        visitas           INTEGER,
                        tiene_rostro      INTEGER NOT NULL DEFAULT 0,
                        no_es_persona     INTEGER NOT NULL DEFAULT 0,
                        revisado_vlm      INTEGER NOT NULL DEFAULT 0,
                        origen_demografia TEXT,
                        motivo            TEXT,
                        mtime             REAL NOT NULL DEFAULT 0
                    )''')
                # Una fila por PERSONA CLASIFICADA, tenga identidad o no
                # (24-sep-2026). `tienda_visitas` solo puede guardar a quien
                # tiene `persistent_id`, y el género se decidía muchas veces
                # sobre alguien sin identidad todavía: medido ese día, 3 de 3
                # clasificados no llegaron a la base. Aquí se guardan todos, y
                # `person_uuid` se rellena si la identidad llega DESPUÉS. La
                # clave es la captura (`stem`), única por persona-track: sirve
                # para completar la fila más tarde sin duplicarla.
                con.execute('''
                    CREATE TABLE IF NOT EXISTS tienda_clasificaciones (
                        stem         TEXT PRIMARY KEY,
                        ts           REAL NOT NULL,
                        jornada      TEXT NOT NULL,
                        camera_id    TEXT NOT NULL,
                        genero       TEXT NOT NULL,
                        edad         TEXT,
                        person_uuid  TEXT,
                        solo_cuerpo  INTEGER NOT NULL DEFAULT 0
                    )''')
                # Los índices son los del dashboard: siempre se pregunta por
                # ventana de fechas, y casi siempre acotando a una cámara.
                for sql in (
                    'CREATE INDEX IF NOT EXISTS idx_vis_jornada '
                    'ON tienda_visitas(jornada)',
                    'CREATE INDEX IF NOT EXISTS idx_vis_jornada_cam '
                    'ON tienda_visitas(jornada, camera_id)',
                    'CREATE INDEX IF NOT EXISTS idx_vis_pid '
                    'ON tienda_visitas(persistent_id)',
                    'CREATE INDEX IF NOT EXISTS idx_ev_ts '
                    'ON tienda_eventos(ts)',
                    'CREATE INDEX IF NOT EXISTS idx_ev_jornada_cam '
                    'ON tienda_eventos(jornada, camera_id)',
                    'CREATE INDEX IF NOT EXISTS idx_cap_ts '
                    'ON tienda_capturas(ts DESC)',
                    'CREATE INDEX IF NOT EXISTS idx_cap_jornada '
                    'ON tienda_capturas(jornada)',
                    'CREATE INDEX IF NOT EXISTS idx_cla_jornada_cam '
                    'ON tienda_clasificaciones(jornada, camera_id)',
                ):
                    con.execute(sql)
            self._preparado = True
        except Exception as exc:                            # noqa: BLE001
            # Sin base de datos el pipeline sigue midiendo y dibujando: lo que
            # se pierde es la memoria, no el vídeo.
            self.fallos += 1
            logger.warning('tienda_almacen: no se pudo preparar %s: %s',
                           self.ruta, exc)

    # ── Anotaciones (baratas: solo encolan) ─────────────────────────────
    def anotar_entrada(self, camera_id: Any, persistent_id: Any,
                       ts: Optional[float] = None) -> None:
        """Una persona ENTRÓ al área de esa cámara (transición confirmada)."""
        self._encolar((EVENTO_ENTRADA, float(ts if ts is not None else time.time()),
                       str(camera_id), str(persistent_id), 0.0, None, None))

    def anotar_salida(self, camera_id: Any, persistent_id: Any,
                      permanencia_s: float = 0.0,
                      ts: Optional[float] = None) -> None:
        """Una persona SALIÓ, con lo que estuvo dentro en ESA estancia."""
        self._encolar((EVENTO_SALIDA, float(ts if ts is not None else time.time()),
                       str(camera_id), str(persistent_id),
                       max(0.0, float(permanencia_s or 0.0)), None, None))

    def anotar_demografia(self, camera_id: Any, persistent_id: Any,
                          genero: Optional[str], edad: Optional[str],
                          ts: Optional[float] = None) -> None:
        """Género y edad ya CONSOLIDADOS de una persona (convergieron).

        Va a la fila de la visita, no a una tabla aparte: así la demografía
        queda atada a la jornada y a la CÁMARA, que es justo lo que hace
        posible el desglose por pasillo.
        """
        g = (genero or '').strip() or None
        e = (edad or '').strip() or None
        if not g and not e:
            return                      # nada que consolidar
        self._encolar(('demografia', float(ts if ts is not None else time.time()),
                       str(camera_id), str(persistent_id), 0.0, g, e))

    def anotar_clasificacion(self, camera_id: Any, stem: str,
                             genero: Optional[str], edad: Optional[str],
                             person_uuid: Optional[str] = None,
                             solo_cuerpo: bool = False,
                             ts: Optional[float] = None) -> None:
        """Una persona quedó CLASIFICADA, con o sin identidad.

        A diferencia de `anotar_demografia`, no exige `persistent_id`: es
        justo lo que se perdía. Si `person_uuid` llega después, completarla
        con `identificar_clasificacion`.
        """
        g = (genero or '').strip()
        if not g or g.lower() == 'desconocido' or not stem:
            return                      # sin género no es una clasificación
        self._encolar(('clasificacion',
                       float(ts if ts is not None else time.time()),
                       str(camera_id), str(stem), 0.0, g,
                       (edad or '').strip() or None,
                       (person_uuid or '').strip() or None,
                       1 if solo_cuerpo else 0))

    def identificar_clasificacion(self, stem: str, person_uuid: str) -> None:
        """La identidad de una clasificación llegó TARDE: se completa.

        Solo rellena el hueco (`person_uuid IS NULL`): una identidad ya
        escrita no se pisa, igual que la demografía nunca pisa lo sabido.
        """
        if not stem or not (person_uuid or '').strip():
            return
        self._encolar(('clasif_identidad', time.time(), '', str(stem), 0.0,
                       None, None, str(person_uuid).strip()))

    def quitar_persona(self, camera_id: Any, persistent_id: Any,
                       ts: Optional[float] = None) -> None:
        """Era PERSONAL (19-oct-2026, `personal_descuento.py`): borra su
        visita y sus eventos de esa jornada y cámara. Va por la MISMA cola
        que las entradas, así que también se lleva las aún sin volcar."""
        if not persistent_id:
            return
        self._encolar(('quitar_persona',
                       float(ts if ts is not None else time.time()),
                       str(camera_id), str(persistent_id), 0.0, None, None))

    def quitar_clasificacion(self, stem: str) -> None:
        """La ficha `stem` era de personal: fuera de las clasificaciones."""
        if stem:
            self._encolar(('quitar_clasif', time.time(), '', str(stem), 0.0,
                           None, None))

    def _encolar(self, op: Tuple) -> None:
        try:
            with self._lock:
                self._pendientes.append(op)
                # Cota de memoria: si el disco no responde durante horas, esto
                # no puede crecer sin fin dentro del proceso de inferencia.
                if len(self._pendientes) > 20000:
                    del self._pendientes[:10000]
                    self.fallos += 1
                    logger.warning('tienda_almacen: cola llena, se descartó '
                                   'la mitad más antigua')
            self._volcar_si_toca()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.debug('tienda_almacen: anotación descartada: %s', exc)

    # ── Volcado ─────────────────────────────────────────────────────────
    def _volcar_si_toca(self) -> None:
        if (time.time() - self._ultimo_volcado) >= self._periodo:
            self.volcar()

    def volcar(self, force: bool = False) -> None:
        """Escribe el lote pendiente. Nunca lanza: cuenta el fallo y sigue.

        El lote se saca bajo el lock, pero el disco se toca FUERA de el: el
        bucle de inferencia no se queda esperando a la E/S de un lector del
        dashboard. Si dos hilos vuelcan a la vez cada uno lleva su lote y su
        conexion; el resultado no depende del orden porque los upserts SUMAN
        (`entradas = entradas + excluded.entradas`) y la demografia solo
        rellena huecos (COALESCE), nunca pisa.
        """
        with self._lock:
            if not self._pendientes:
                # Sin tocar `_ultimo_volcado`: asi el primer hecho tras un rato
                # parado se escribe enseguida en vez de esperar otro periodo.
                return
            if not force and (time.time() - self._ultimo_volcado) < self._periodo:
                return
            lote, self._pendientes = self._pendientes, []
            self._ultimo_volcado = time.time()
        try:
            if not self._preparado:
                self._preparar()        # reintento: quizá el disco ya volvió
            with self._conectar() as con:
                for op in lote:
                    # Las 7 primeras posiciones son las de siempre; las
                    # clasificaciones (24-sep) añaden sus campos DETRÁS, así
                    # que las operaciones viejas no cambian de forma.
                    tipo, ts, cam, pid, permanencia, genero, edad = op[:7]
                    extra = op[7:]
                    jornada = jornada_de(ts)
                    if tipo == 'clasificacion':
                        self._upsert_clasificacion(con, jornada, cam, pid, ts,
                                                   genero, edad, extra[0],
                                                   extra[1])
                        continue
                    if tipo == 'clasif_identidad':
                        con.execute(
                            'UPDATE tienda_clasificaciones SET person_uuid = ? '
                            'WHERE stem = ? AND person_uuid IS NULL',
                            (extra[0], pid))
                        continue
                    if tipo == 'quitar_persona':
                        for tabla in ('tienda_visitas', 'tienda_eventos'):
                            con.execute(
                                f'DELETE FROM {tabla} WHERE jornada = ? '
                                'AND camera_id = ? AND persistent_id = ?',
                                (jornada, cam, pid))
                        continue
                    if tipo == 'quitar_clasif':
                        con.execute('DELETE FROM tienda_clasificaciones '
                                    'WHERE stem = ?', (pid,))
                        continue
                    if tipo == 'demografia':
                        self._upsert_demografia(con, jornada, cam, pid, ts,
                                                genero, edad)
                        continue
                    con.execute(
                        'INSERT INTO tienda_eventos '
                        '(ts, jornada, camera_id, persistent_id, tipo, '
                        ' permanencia_s) VALUES (?,?,?,?,?,?)',
                        (ts, jornada, cam, pid, tipo,
                         permanencia if tipo == EVENTO_SALIDA else None))
                    self._upsert_visita(con, jornada, cam, pid, ts, tipo,
                                        permanencia)
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: volcado fallido (%d hechos '
                           'perdidos): %s', len(lote), exc)

    @staticmethod
    def _upsert_visita(con, jornada: str, cam: str, pid: str, ts: float,
                       tipo: str, permanencia: float) -> None:
        entrada = 1 if tipo == EVENTO_ENTRADA else 0
        salida = 1 if tipo == EVENTO_SALIDA else 0
        dentro = permanencia if tipo == EVENTO_SALIDA else 0.0
        con.execute(
            'INSERT INTO tienda_visitas '
            '(jornada, camera_id, persistent_id, ts_primera, ts_ultima, '
            ' entradas, salidas, permanencia_s) VALUES (?,?,?,?,?,?,?,?) '
            'ON CONFLICT(jornada, camera_id, persistent_id) DO UPDATE SET '
            '  ts_ultima     = MAX(ts_ultima, excluded.ts_ultima), '
            '  entradas      = entradas + excluded.entradas, '
            '  salidas       = salidas + excluded.salidas, '
            '  permanencia_s = permanencia_s + excluded.permanencia_s',
            (jornada, cam, pid, ts, ts, entrada, salida, dentro))

    @staticmethod
    def _upsert_demografia(con, jornada: str, cam: str, pid: str, ts: float,
                           genero: Optional[str], edad: Optional[str]) -> None:
        # COALESCE con el valor nuevo PRIMERO: la demografía consolidada es
        # mejor que la que hubiera, pero un null nunca pisa lo ya sabido.
        con.execute(
            'INSERT INTO tienda_visitas '
            '(jornada, camera_id, persistent_id, ts_primera, ts_ultima, '
            ' genero, edad) VALUES (?,?,?,?,?,?,?) '
            'ON CONFLICT(jornada, camera_id, persistent_id) DO UPDATE SET '
            '  ts_ultima = MAX(ts_ultima, excluded.ts_ultima), '
            '  genero    = COALESCE(excluded.genero, genero), '
            '  edad      = COALESCE(excluded.edad, edad)',
            (jornada, cam, pid, ts, ts, genero, edad))

    @staticmethod
    def _upsert_clasificacion(con, jornada: str, cam: str, stem: str,
                              ts: float, genero: str, edad: Optional[str],
                              person_uuid: Optional[str],
                              solo_cuerpo: int) -> None:
        # Una segunda clasificación de la MISMA captura (el género se revisó)
        # actualiza el género y la edad; la identidad solo rellena huecos.
        con.execute(
            'INSERT INTO tienda_clasificaciones '
            '(stem, ts, jornada, camera_id, genero, edad, person_uuid, '
            ' solo_cuerpo) VALUES (?,?,?,?,?,?,?,?) '
            'ON CONFLICT(stem) DO UPDATE SET '
            '  genero      = excluded.genero, '
            '  edad        = COALESCE(excluded.edad, edad), '
            '  person_uuid = COALESCE(person_uuid, excluded.person_uuid), '
            '  solo_cuerpo = excluded.solo_cuerpo',
            (stem, ts, jornada, cam, genero, edad, person_uuid, solo_cuerpo))

    # ── Lectura ─────────────────────────────────────────────────────────
    def contadores_de(self, camera_id: Any,
                      jornada: Optional[str] = None) -> Dict[str, Any]:
        """Lo contado HOY (o en `jornada`) por esa cámara, para reanudar.

        Es lo que permite que una reconexión no ponga el marcador a cero y que
        dos sesiones del mismo día SUMEN. Devuelve también el detalle por
        persona, con el que se rehidrata la máquina DENTRO/FUERA sin inventarse
        nada: quien vuelve a entrar sigue contando una sola visita única.
        """
        vacio = {'entradas': 0, 'salidas': 0, 'unicos': 0, 'personas': {}}
        try:
            self.volcar(force=True)     # que lo recién anotado también cuente
            jor = jornada or jornada_de()
            with self._conectar() as con:
                filas = con.execute(
                    'SELECT persistent_id, entradas, salidas, permanencia_s '
                    'FROM tienda_visitas WHERE jornada = ? AND camera_id = ?',
                    (jor, str(camera_id))).fetchall()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: no se pudo leer contadores: %s',
                           exc)
            return vacio
        personas = {}
        entradas = salidas = unicos = 0
        for f in filas:
            personas[f['persistent_id']] = {
                'entradas': int(f['entradas'] or 0),
                'salidas': int(f['salidas'] or 0),
                'permanencia_s': float(f['permanencia_s'] or 0.0),
            }
            entradas += int(f['entradas'] or 0)
            salidas += int(f['salidas'] or 0)
            if int(f['entradas'] or 0) > 0:
                unicos += 1             # único = pisó dentro al menos una vez
        return {'entradas': entradas, 'salidas': salidas, 'unicos': unicos,
                'personas': personas}

    def resumen(self, desde: Optional[str] = None,
                hasta: Optional[str] = None,
                camera_id: Optional[Any] = None) -> Dict[str, Any]:
        """Totales de una ventana de jornadas (ambos extremos incluidos).

        «Únicos» son `persistent_id` DISTINTOS dentro de la ventana — no la
        suma de los únicos diarios, que contaría dos veces a quien vuelve.
        """
        vacio = {'entradas': 0, 'salidas': 0, 'visitantes_unicos': 0,
                 'permanencia_media_s': 0.0, 'jornadas': 0}
        desde = desde or jornada_de()
        hasta = hasta or desde
        cond = 'jornada BETWEEN ? AND ?'
        args: List[Any] = [desde, hasta]
        if camera_id is not None:
            cond += ' AND camera_id = ?'
            args.append(str(camera_id))
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                f = con.execute(
                    'SELECT COALESCE(SUM(entradas),0) AS e, '
                    '       COALESCE(SUM(salidas),0)  AS s, '
                    '       COUNT(DISTINCT CASE WHEN entradas > 0 '
                    '             THEN persistent_id END) AS u, '
                    '       COUNT(DISTINCT jornada) AS j, '
                    '       COALESCE(SUM(permanencia_s),0) AS p, '
                    '       COUNT(CASE WHEN entradas > 0 THEN 1 END) AS n '
                    f'FROM tienda_visitas WHERE {cond}', args).fetchone()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: resumen fallido: %s', exc)
            return vacio
        n = int(f['n'] or 0)
        return {
            'entradas': int(f['e'] or 0),
            'salidas': int(f['s'] or 0),
            'visitantes_unicos': int(f['u'] or 0),
            'permanencia_media_s': round(float(f['p'] or 0.0) / n, 1) if n else 0.0,
            'jornadas': int(f['j'] or 0),
        }

    def serie(self, desde: Optional[str] = None, hasta: Optional[str] = None,
              camera_id: Optional[Any] = None,
              grano: str = 'hora') -> List[Dict[str, Any]]:
        """Entradas y salidas por hora (o por día) de la ventana.

        Sale de la BITÁCORA, no de las visitas: es la única forma de saber a
        qué HORA pasó cada cosa. La suma de la serie tiene que cuadrar con las
        entradas de `resumen()` — lo fija una prueba.

        `personas` es cuánta gente DISTINTA entró en ese tramo; sumarla entre
        tramos no da los únicos del día (quien entra a las 10 y a las 12 está
        en los dos), para eso está `resumen()`.
        """
        desde = desde or jornada_de()
        hasta = hasta or desde
        cond = 'jornada BETWEEN ? AND ?'
        args: List[Any] = [desde, hasta]
        if camera_id is not None:
            cond += ' AND camera_id = ?'
            args.append(str(camera_id))
        # `localtime` para que la hora sea la que vivió la tienda, no UTC.
        bucket = ("strftime('%H', ts, 'unixepoch', 'localtime')"
                  if grano == 'hora' else "''")
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                filas = con.execute(
                    f'SELECT jornada, {bucket} AS hora, '
                    "  SUM(tipo = 'entrada') AS entradas, "
                    "  SUM(tipo = 'salida')  AS salidas, "
                    '  COUNT(DISTINCT CASE WHEN tipo = \'entrada\' '
                    '        THEN persistent_id END) AS personas '
                    f'FROM tienda_eventos WHERE {cond} '
                    'GROUP BY jornada, hora ORDER BY jornada, hora',
                    args).fetchall()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: serie fallida: %s', exc)
            return []
        return [{'jornada': f['jornada'], 'hora': f['hora'] or None,
                 'entradas': int(f['entradas'] or 0),
                 'salidas': int(f['salidas'] or 0),
                 'personas': int(f['personas'] or 0)} for f in filas]

    def por_camara(self, desde: Optional[str] = None,
                   hasta: Optional[str] = None) -> List[Dict[str, Any]]:
        """Totales por cámara en la ventana, de más a menos tráfico.

        Esto es lo que sustituye al ranking que leía UN solo informe: aquí
        **suman todas las sesiones** de todas las jornadas del rango.
        """
        desde = desde or jornada_de()
        hasta = hasta or desde
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                filas = con.execute(
                    'SELECT camera_id, '
                    '  COALESCE(SUM(entradas),0) AS e, '
                    '  COALESCE(SUM(salidas),0)  AS s, '
                    '  COUNT(DISTINCT CASE WHEN entradas > 0 '
                    '        THEN persistent_id END) AS u, '
                    '  COALESCE(SUM(permanencia_s),0) AS p, '
                    '  COUNT(CASE WHEN entradas > 0 THEN 1 END) AS n '
                    'FROM tienda_visitas WHERE jornada BETWEEN ? AND ? '
                    'GROUP BY camera_id ORDER BY e DESC',
                    (desde, hasta)).fetchall()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: por_camara fallido: %s', exc)
            return []
        salida = []
        for f in filas:
            n = int(f['n'] or 0)
            salida.append({
                'camera_id': f['camera_id'],
                'entradas': int(f['e'] or 0),
                'salidas': int(f['s'] or 0),
                'visitantes_unicos': int(f['u'] or 0),
                'permanencia_media_s': (round(float(f['p'] or 0.0) / n, 1)
                                        if n else 0.0),
            })
        return salida

    def demografia(self, desde: Optional[str] = None,
                   hasta: Optional[str] = None,
                   camera_id: Optional[Any] = None) -> Dict[str, Any]:
        """Género y edad de la ventana, global y POR CÁMARA.

        Cuenta personas DISTINTAS, no filas: quien viene lunes y martes es una
        persona. Solo entra quien tiene demografía consolidada y pisó dentro
        al menos una vez; los demás no se rellenan con adivinanzas.

        Aviso honesto: si a la misma persona se le estimó un género distinto
        en dos días, aparecerá en los dos cubos. Es raro y preferimos que se
        vea a maquillarlo.
        """
        desde = desde or jornada_de()
        hasta = hasta or desde
        cond = 'jornada BETWEEN ? AND ? AND entradas > 0'
        args: List[Any] = [desde, hasta]
        if camera_id is not None:
            cond += ' AND camera_id = ?'
            args.append(str(camera_id))
        vacio = {'genero': {}, 'edad': {}, 'por_camara': {}}
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                filas = con.execute(
                    'SELECT camera_id, genero, edad, '
                    '  COUNT(DISTINCT persistent_id) AS n '
                    f'FROM tienda_visitas WHERE {cond} '
                    'GROUP BY camera_id, genero, edad', args).fetchall()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: demografia fallida: %s', exc)
            return vacio
        genero: Dict[str, int] = {}
        edad: Dict[str, int] = {}
        por_camara: Dict[str, Dict[str, Dict[str, int]]] = {}
        for f in filas:
            n = int(f['n'] or 0)
            cam = str(f['camera_id'])
            if not f['genero'] and not f['edad']:
                # Cámara con tráfico pero sin ninguna cara clara: no se lista
                # con los cubos vacíos. Que tuvo gente ya lo dice /pasillos;
                # aquí solo sería una fila hueca en la tabla y en el PDF.
                continue
            rec = por_camara.setdefault(cam, {'genero': {}, 'edad': {}})
            if f['genero']:
                genero[f['genero']] = genero.get(f['genero'], 0) + n
                rec['genero'][f['genero']] = rec['genero'].get(f['genero'], 0) + n
            if f['edad']:
                edad[f['edad']] = edad.get(f['edad'], 0) + n
                rec['edad'][f['edad']] = rec['edad'].get(f['edad'], 0) + n
        return {'genero': genero, 'edad': edad, 'por_camara': por_camara}

    def clasificaciones(self, desde: Optional[str] = None,
                        hasta: Optional[str] = None,
                        camera_id: Optional[Any] = None) -> Dict[str, Any]:
        """Clasificados de la ventana, separados POR IDENTIDAD.

        `con_identidad` son personas reconocidas (cuentan en las visitas);
        `sin_identidad`, clasificadas sin que se las pudiera reconocer. Cada
        fila es una persona-track: quien sale de cámara y vuelve puede contar
        dos veces en `sin_identidad`, porque sin identidad no hay forma de
        saber que es la misma. Por eso no se mezclan con las visitas.
        """
        desde = desde or jornada_de()
        hasta = hasta or desde
        cond = 'jornada BETWEEN ? AND ?'
        args: List[Any] = [desde, hasta]
        if camera_id is not None:
            cond += ' AND camera_id = ?'
            args.append(str(camera_id))
        def cubo() -> Dict[str, Any]:
            return {'total': 0, 'genero': {}, 'edad': {}}
        salida = {'con_identidad': cubo(), 'sin_identidad': cubo(), 'total': 0}
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                # Con identidad se cuentan PERSONAS (la misma, reconocida en
                # dos capturas, es una); sin identidad, capturas: no hay forma
                # de saber que dos son la misma persona.
                filas = con.execute(
                    'SELECT genero, edad, '
                    '  COUNT(DISTINCT person_uuid) AS n_ident, '
                    '  SUM(person_uuid IS NULL) AS n_sin '
                    f'FROM tienda_clasificaciones WHERE {cond} '
                    'GROUP BY genero, edad', args).fetchall()
                total_ident = con.execute(
                    'SELECT COUNT(DISTINCT person_uuid) '
                    f'FROM tienda_clasificaciones WHERE {cond}',
                    args).fetchone()[0] or 0
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: clasificaciones fallidas: %s', exc)
            return salida
        for f in filas:
            for clave, n in (('con_identidad', int(f['n_ident'] or 0)),
                             ('sin_identidad', int(f['n_sin'] or 0))):
                if not n:
                    continue
                c = salida[clave]
                c['genero'][f['genero']] = c['genero'].get(f['genero'], 0) + n
                if f['edad']:
                    c['edad'][f['edad']] = c['edad'].get(f['edad'], 0) + n
                if clave == 'sin_identidad':
                    c['total'] += n
        # El total con identidad sale de su propia consulta, no de sumar los
        # cubos: alguien con dos géneros distintos en dos capturas aparece en
        # los dos cubos, pero es UNA persona.
        salida['con_identidad']['total'] = int(total_ident)
        salida['total'] = (salida['con_identidad']['total']
                           + salida['sin_identidad']['total'])
        return salida

    def ultima_entrada(self, camera_id: Any,
                       jornada: Optional[str] = None) -> Optional[float]:
        """Cuándo entró la última persona por esa cámara. `None` si nadie.

        Lo usa la regla del «pasillo dormido»: un pasillo que hoy tuvo gente y
        lleva horas sin una sola entrada es una señal de negocio (algo tapa la
        cámara, se movió un expositor, se cerró el paso). Sin este dato habría
        que recorrer toda la bitácora del día en el bucle de inferencia.
        """
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                f = con.execute(
                    'SELECT MAX(ts) AS t FROM tienda_eventos '
                    "WHERE tipo = 'entrada' AND camera_id = ? AND jornada = ?",
                    (str(camera_id), jornada or jornada_de())).fetchone()
            return float(f['t']) if f and f['t'] is not None else None
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: ultima_entrada fallida: %s', exc)
            return None

    def camaras(self) -> List[str]:
        """Las cámaras con algún dato guardado (para poblar el filtro)."""
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                filas = con.execute(
                    'SELECT DISTINCT camera_id FROM tienda_visitas '
                    'ORDER BY camera_id').fetchall()
            return [str(f['camera_id']) for f in filas]
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: camaras fallido: %s', exc)
            return []

    def jornadas_con_datos(self, desde: str, hasta: str) -> List[str]:
        """Qué jornadas de la ventana tienen algo.

        Sirve para que el dashboard distinga «0 visitas» de «sin datos» (el
        servidor estaba apagado): dos cosas muy distintas que pintadas igual
        se leen como una caída de tráfico que nunca ocurrió.
        """
        try:
            self.volcar(force=True)
            with self._conectar() as con:
                filas = con.execute(
                    'SELECT DISTINCT jornada FROM tienda_visitas '
                    'WHERE jornada BETWEEN ? AND ? ORDER BY jornada',
                    (desde, hasta)).fetchall()
            return [str(f['jornada']) for f in filas]
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: jornadas_con_datos fallido: %s',
                           exc)
            return []

    # ── Índice de capturas (HITO 3) ─────────────────────────────────────
    # Ojo: aquí NO hay cola ni volcado amortiguado. Estas escrituras no salen
    # del bucle de inferencia, sino de la reconciliación con el disco, que es
    # esporádica y ya viene por lotes.

    def capturas_indexadas(self) -> Dict[str, float]:
        """`{stem: mtime}` de lo ya indexado, para saber qué falta o cambió."""
        try:
            with self._conectar() as con:
                filas = con.execute(
                    'SELECT stem, mtime FROM tienda_capturas').fetchall()
            return {str(f['stem']): float(f['mtime'] or 0.0) for f in filas}
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: no se pudo leer el índice de '
                           'capturas: %s', exc)
            return {}

    def captura_mas_reciente(self) -> Optional[Dict[str, Any]]:
        """`{'ts', 'jornada'}` de la última captura indexada, o None. Sirve
        para detectar que el corte de jornada cambió desde que se indexó
        (ver tienda_capturas.sincronizar)."""
        try:
            with self._conectar() as con:
                f = con.execute('SELECT ts, jornada FROM tienda_capturas '
                                'ORDER BY ts DESC LIMIT 1').fetchone()
            return ({'ts': float(f['ts'] or 0.0), 'jornada': str(f['jornada'] or '')}
                    if f else None)
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: captura_mas_reciente fallo: %s', exc)
            return None

    def guardar_capturas(self, filas: List[Dict[str, Any]]) -> int:
        """Alta o actualización de capturas en el índice. Devuelve cuántas."""
        if not filas:
            return 0
        try:
            with self._conectar() as con:
                con.executemany(
                    'INSERT INTO tienda_capturas '
                    '(stem, ts, jornada, camera, device_id, genero, edad, '
                    ' person_uuid, visitas, tiene_rostro, no_es_persona, '
                    ' revisado_vlm, origen_demografia, motivo, mtime) '
                    'VALUES (:stem, :ts, :jornada, :camera, :device_id, '
                    ' :genero, :edad, :person_uuid, :visitas, :tiene_rostro, '
                    ' :no_es_persona, :revisado_vlm, :origen_demografia, '
                    ' :motivo, :mtime) '
                    'ON CONFLICT(stem) DO UPDATE SET '
                    '  ts=excluded.ts, jornada=excluded.jornada, '
                    '  camera=excluded.camera, device_id=excluded.device_id, '
                    '  genero=excluded.genero, edad=excluded.edad, '
                    '  person_uuid=excluded.person_uuid, '
                    '  visitas=excluded.visitas, '
                    '  tiene_rostro=excluded.tiene_rostro, '
                    '  no_es_persona=excluded.no_es_persona, '
                    '  revisado_vlm=excluded.revisado_vlm, '
                    '  origen_demografia=excluded.origen_demografia, '
                    '  motivo=excluded.motivo, mtime=excluded.mtime',
                    filas)
            return len(filas)
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: no se pudieron indexar %d '
                           'capturas: %s', len(filas), exc)
            return 0

    def borrar_capturas(self, stems: List[str]) -> int:
        """Saca del índice lo que ya no está en disco."""
        if not stems:
            return 0
        try:
            with self._conectar() as con:
                con.executemany('DELETE FROM tienda_capturas WHERE stem = ?',
                                [(s,) for s in stems])
            return len(stems)
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: no se pudieron borrar capturas '
                           'del índice: %s', exc)
            return 0

    def capturas(self, desde: Optional[str] = None,
                 hasta: Optional[str] = None,
                 camera_id: Optional[Any] = None,
                 limite: int = 60, offset: int = 0) -> Dict[str, Any]:
        """Capturas de la ventana, más recientes primero, con su total.

        Sin `desde`/`hasta` devuelve las más recientes de todas — que es lo que
        hacía el índice de disco, para no cambiarle el significado a la galería.
        """
        cond: List[str] = []
        args: List[Any] = []
        if desde:
            cond.append('jornada >= ?')
            args.append(desde)
        if hasta:
            cond.append('jornada <= ?')
            args.append(hasta)
        if camera_id is not None:
            # Las capturas anteriores al 18-ago-2026 no tienen device_id: se
            # acepta también el nombre visible para no dejarlas fuera del
            # filtro sin avisar.
            cond.append('(device_id = ? OR camera = ?)')
            args.extend([str(camera_id), str(camera_id)])
        donde = (' WHERE ' + ' AND '.join(cond)) if cond else ''
        try:
            with self._conectar() as con:
                total = con.execute(
                    f'SELECT COUNT(*) AS n FROM tienda_capturas{donde}',
                    args).fetchone()['n']
                filas = con.execute(
                    f'SELECT * FROM tienda_capturas{donde} '
                    'ORDER BY ts DESC, stem DESC LIMIT ? OFFSET ?',
                    args + [max(1, int(limite)), max(0, int(offset))]
                ).fetchall()
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: consulta de capturas fallida: %s',
                           exc)
            return {'total': 0, 'capturas': []}
        return {'total': int(total or 0),
                'capturas': [dict(f) for f in filas]}

    def total_capturas(self) -> int:
        """`COUNT(*)`, que es lo que antes costaba abrir todos los sidecars."""
        try:
            with self._conectar() as con:
                return int(con.execute(
                    'SELECT COUNT(*) AS n FROM tienda_capturas'
                ).fetchone()['n'] or 0)
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: total de capturas fallido: %s',
                           exc)
            return 0

    # ── Mantenimiento ───────────────────────────────────────────────────
    def purgar(self, retencion_dias: Optional[int] = None) -> int:
        """Borra lo anterior a la retención. Devuelve las filas borradas.

        13 meses por defecto: es lo que hace falta para comparar un mes contra
        el mismo mes del año pasado.
        """
        dias = RETENCION_DIAS if retencion_dias is None else int(retencion_dias)
        if dias <= 0:
            return 0                    # 0 o negativo = no purgar nunca
        limite = jornada_de(time.time() - dias * 86400)
        # `tienda_capturas` NO se purga aquí a propósito: es un espejo del
        # disco, no un histórico. Mientras la foto exista, borrar su fila solo
        # conseguiría que la galería la perdiera de vista hasta la siguiente
        # reconciliación, que volvería a insertarla. Las fotos viejas se
        # gestionan con su propia política (hoy: no se tocan).
        try:
            with self._conectar() as con:
                n = con.execute('DELETE FROM tienda_visitas WHERE jornada < ?',
                                (limite,)).rowcount or 0
                n += con.execute('DELETE FROM tienda_eventos WHERE jornada < ?',
                                 (limite,)).rowcount or 0
                n += con.execute(
                    'DELETE FROM tienda_clasificaciones WHERE jornada < ?',
                    (limite,)).rowcount or 0
            if n:
                logger.info('tienda_almacen: purgadas %d filas anteriores a %s',
                            n, limite)
            return int(n)
        except Exception as exc:                            # noqa: BLE001
            self.fallos += 1
            logger.warning('tienda_almacen: purga fallida: %s', exc)
            return 0


# ── Singleton del proceso ────────────────────────────────────────────────
# Un solo escritor por proceso: todas las cámaras y todos los procesadores
# comparten esta instancia, así que el lote sale ordenado y la base no ve
# escrituras concurrentes desde el bucle de inferencia.

_almacen: Optional[AlmacenTienda] = None
_lock_singleton = threading.Lock()


def get_almacen() -> AlmacenTienda:
    """El almacén del proceso. Purga una sola vez, al crearlo."""
    global _almacen
    with _lock_singleton:
        if _almacen is None:
            _almacen = AlmacenTienda()
            _almacen.purgar()
            atexit.register(_volcar_al_salir)
        return _almacen


def _volcar_al_salir() -> None:
    """Que un Ctrl+C no se lleve los últimos segundos de conteo."""
    if _almacen is not None:
        _almacen.volcar(force=True)


def reiniciar_para_pruebas(ruta: Optional[Path] = None,
                           periodo_volcado_s: float = 0.0) -> AlmacenTienda:
    """Sustituye el singleton (solo para las pruebas: `tmp_path` y sin espera).

    Con `periodo_volcado_s = 0` cada anotación se escribe al momento, que es lo
    que quiere una prueba; en producción el valor real amortigua.
    """
    global _almacen
    with _lock_singleton:
        _almacen = AlmacenTienda(ruta=ruta, periodo_volcado_s=periodo_volcado_s)
    # El índice de capturas cachea en memoria lo que hay en la base: si no se
    # le avisa del cambio, una prueba arrastra el índice de la anterior y el
    # fallo aparece en OTRA prueba. Import local: `tienda_capturas` importa
    # este módulo.
    try:
        from . import tienda_capturas
        tienda_capturas.reiniciar_para_pruebas()
    except Exception:                                       # noqa: BLE001
        pass
    return _almacen
