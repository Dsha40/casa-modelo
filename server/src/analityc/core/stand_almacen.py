"""Persistencia del stand de degustación (HITO 2).

## Por qué SQLite y no MongoDB

La especificación pedía colecciones MongoDB. La verificación del HITO 0
encontró que **MongoDB no existe** en este despliegue: no hay servicio, nadie
escucha en 27017, `mongod` no está en el PATH y el repo no tiene una sola
referencia a `pymongo`. Montarlo habría añadido un servicio nuevo que mantener
para un módulo que corre en la misma máquina que el pipeline.

SQLite ya es el precedente del repo (`estacionamiento/placas.db`,
`vigilante_amazonas/db/vigilante.db`), da índices reales para el dashboard y no
añade infraestructura. Las cinco colecciones pasan a cinco tablas con el mismo
esquema; los campos anidados (trayectoria, payload, matriz del heatmap) se
guardan como JSON en columnas TEXT, que es exactamente lo que Mongo haría por
dentro y SQLite sabe consultar con `json_extract` si hiciera falta.

## La unidad de verdad

`stand_visitantes` tiene **una fila por persona única**, indexada por
`persistent_id`. Todo conteo sale de ahí. `stand_eventos` es bitácora inmutable
y NO se cuenta: un mismo visitante genera muchos eventos.

Ver `AUDITORIA_STAND.md` §4 para por qué esta distinción es la que evita el
sobreconteo.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_RAIZ_SERVIDOR = Path(__file__).resolve().parents[3]


# ── Vocabulario cerrado ──────────────────────────────────────────────────
# Se declaran como constantes para que un typo no cree una clasificación
# fantasma que luego no cuadre en el embudo.

class Clasificacion:
    """Estado terminal de un visitante. Son EXCLUYENTES."""
    PASO_DE_LARGO = "paso_de_largo"
    SE_DETUVO_SIN_ATENCION = "se_detuvo_sin_atencion"
    ATENDIDO_SIN_DEGUSTAR = "atendido_sin_degustar"
    DEGUSTO = "degusto"

    TODAS = (PASO_DE_LARGO, SE_DETUVO_SIN_ATENCION,
             ATENDIDO_SIN_DEGUSTAR, DEGUSTO)


class TipoEvento:
    """Tipos de la bitácora. Uno por transición de la máquina de estados."""
    ENTRADA_ESCENA = "entrada_escena"
    ENTRADA_ZONA_STAND = "entrada_zona_stand"
    DETENCION = "detencion"
    INICIO_ATENCION = "inicio_atencion"
    FIN_ATENCION = "fin_atencion"
    DEGUSTACION = "degustacion"
    SALIDA_ZONA_STAND = "salida_zona_stand"
    SALIDA_ESCENA = "salida_escena"
    MERGE_IDENTIDAD = "merge_identidad"
    DESCARTE = "descarte"

    TODOS = (ENTRADA_ESCENA, ENTRADA_ZONA_STAND, DETENCION, INICIO_ATENCION,
             FIN_ATENCION, DEGUSTACION, SALIDA_ZONA_STAND, SALIDA_ESCENA,
             MERGE_IDENTIDAD, DESCARTE)


class MotivoSinDemografia:
    """Por qué un visitante quedó sin género/edad.

    Un `null` mudo no distingue «no se pudo» de «no se intentó», y esa
    ambigüedad es justo lo que hizo invisible el fallo original (ver
    AUDITORIA_STAND.md §3). Nunca se escribe null sin motivo.
    """
    SIN_MUESTRAS = "sin_muestras"
    BAJA_CONFIANZA = "baja_confianza"
    TRACK_CORTO = "track_corto"


def ruta_bd() -> Path:
    """Ruta del .db. Overridable por entorno, como el resto del repo."""
    propia = (os.getenv('STAND_BD') or '').strip()
    return (Path(propia) if propia
            else _RAIZ_SERVIDOR / 'output' / 'stand' / 'stand.db')


def _ahora() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _json_o_none(valor: Any) -> Optional[str]:
    if valor is None:
        return None
    return json.dumps(valor, ensure_ascii=False)


def _carga_json(texto: Optional[str]) -> Any:
    if not texto:
        return None
    try:
        return json.loads(texto)
    except (ValueError, TypeError):
        return None


class AlmacenStand:
    """Las cinco tablas del stand, con sus índices."""

    def __init__(self, ruta: Optional[Path] = None) -> None:
        self.ruta = Path(ruta or ruta_bd())
        self._lock = threading.Lock()
        self._preparar()

    @contextmanager
    def _conectar(self):
        """Conexión que se CIERRA de verdad.

        GOTCHA heredado de `estacionamiento_placas.py`: `with
        sqlite3.connect(...) as con` gestiona la TRANSACCIÓN, no la conexión.
        Sin cerrarla se fugan descriptores y en Windows el .db queda
        bloqueado — lo que impide hasta borrar la carpeta en las pruebas.
        """
        con = sqlite3.connect(str(self.ruta), timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    # ── Esquema ─────────────────────────────────────────────────────────
    def _preparar(self) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self._conectar() as con:
            con.execute('''
                CREATE TABLE IF NOT EXISTS stand_sesiones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stand_id TEXT NOT NULL,
                    nombre TEXT,
                    producto TEXT,
                    fecha TEXT,
                    hora_inicio TEXT,
                    hora_fin TEXT,
                    camara_id TEXT,
                    config_snapshot TEXT,
                    estado TEXT DEFAULT 'abierta'
                )''')

            # UNA FILA POR PERSONA ÚNICA. Todo conteo sale de esta tabla.
            con.execute('''
                CREATE TABLE IF NOT EXISTS stand_visitantes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sesion_id INTEGER NOT NULL,
                    persistent_id TEXT NOT NULL,
                    primera_deteccion TEXT,
                    ultima_deteccion TEXT,
                    duracion_total_s REAL DEFAULT 0,
                    genero TEXT,
                    genero_confianza REAL,
                    genero_muestras INTEGER DEFAULT 0,
                    edad_estimada REAL,
                    rango_etario TEXT,
                    edad_muestras INTEGER DEFAULT 0,
                    demografia_valida INTEGER DEFAULT 0,
                    motivo_sin_demografia TEXT,
                    se_detuvo INTEGER DEFAULT 0,
                    dwell_max_s REAL DEFAULT 0,
                    fue_atendido INTEGER DEFAULT 0,
                    ts_atencion TEXT,
                    duracion_atencion_s REAL,
                    degusto INTEGER DEFAULT 0,
                    ts_degustacion TEXT,
                    metodo_deteccion_degustacion TEXT,
                    confianza_degustacion REAL,
                    trayectoria_resumen TEXT,
                    clasificacion TEXT,
                    es_personal INTEGER DEFAULT 0,
                    tracker_ids TEXT,
                    descartado INTEGER DEFAULT 0,
                    motivo_descarte TEXT,
                    UNIQUE (sesion_id, persistent_id)
                )''')

            con.execute('''
                CREATE TABLE IF NOT EXISTS stand_eventos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sesion_id INTEGER NOT NULL,
                    persistent_id TEXT,
                    tipo TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    zona TEXT,
                    payload TEXT,
                    confianza REAL
                )''')

            con.execute('''
                CREATE TABLE IF NOT EXISTS stand_heatmap (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sesion_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    tipo TEXT NOT NULL,
                    grid TEXT,
                    matriz TEXT
                )''')

            con.execute('''
                CREATE TABLE IF NOT EXISTS stand_metricas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sesion_id INTEGER NOT NULL,
                    bucket_ts TEXT NOT NULL,
                    kpis TEXT,
                    por_genero TEXT,
                    por_rango_etario TEXT,
                    calidad_dato TEXT,
                    UNIQUE (sesion_id, bucket_ts)
                )''')

            # Índices pedidos por el HITO 2.
            for ddl in (
                'CREATE INDEX IF NOT EXISTS idx_vis_pid '
                '  ON stand_visitantes(persistent_id)',
                'CREATE INDEX IF NOT EXISTS idx_vis_sesion '
                '  ON stand_visitantes(sesion_id)',
                'CREATE INDEX IF NOT EXISTS idx_vis_clasif '
                '  ON stand_visitantes(clasificacion)',
                'CREATE INDEX IF NOT EXISTS idx_ev_sesion_ts '
                '  ON stand_eventos(sesion_id, ts)',
                'CREATE INDEX IF NOT EXISTS idx_ev_pid '
                '  ON stand_eventos(persistent_id)',
                'CREATE INDEX IF NOT EXISTS idx_hm_sesion_ts '
                '  ON stand_heatmap(sesion_id, ts)',
                'CREATE INDEX IF NOT EXISTS idx_met_sesion '
                '  ON stand_metricas(sesion_id, bucket_ts)',
            ):
                con.execute(ddl)

    # ── Sesiones ────────────────────────────────────────────────────────
    def abrir_sesion(self, stand_id: str, nombre: str = '', producto: str = '',
                     camara_id: str = '',
                     config_snapshot: Optional[Dict] = None) -> int:
        """Abre una jornada y devuelve su id.

        El `config_snapshot` guarda los umbrales VIGENTES al abrir: sin eso,
        recalibrar a media jornada hace incomparables las cifras y nadie sabe
        con qué reglas se midió cada tramo.
        """
        ahora = datetime.now()
        with self._lock, self._conectar() as con:
            # Las jornadas de DÍAS ANTERIORES que quedaron «abiertas» (el
            # servidor cruzó la medianoche o arrancó al día siguiente sin
            # que nadie las cerrara) se cierran aquí, al abrir la nueva:
            # si no, quedaban abiertas para siempre y el reporte las
            # mostraba en curso (5-sep-2026). Fin nominal: 23:59:59 de su día.
            con.execute(
                "UPDATE stand_sesiones SET hora_fin = COALESCE(hora_fin, "
                "'23:59:59'), estado = 'cerrada' WHERE stand_id = ? AND "
                "estado = 'abierta' AND fecha < ?",
                (stand_id, ahora.strftime('%Y-%m-%d')))
            cur = con.execute(
                'INSERT INTO stand_sesiones (stand_id, nombre, producto, '
                'fecha, hora_inicio, camara_id, config_snapshot, estado) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (stand_id, nombre, producto, ahora.strftime('%Y-%m-%d'),
                 ahora.strftime('%H:%M:%S'), camara_id,
                 _json_o_none(config_snapshot), 'abierta'))
            return int(cur.lastrowid)

    def sesion_abierta_de(self, stand_id: str,
                          fecha: Optional[str] = None) -> Optional[int]:
        """Jornada ya abierta de este stand HOY, si la hay.

        Sin esto, cada reinicio del servidor abría una sesión nueva y dejaba
        la anterior colgada: la misma jornada quedaba partida en trozos y los
        conteos de cada tramo no se podían sumar (una persona vista antes y
        después del reinicio cuenta en las dos). Se detectó con 4 sesiones
        abiertas en una tarde.
        """
        dia = fecha or datetime.now().strftime('%Y-%m-%d')
        with self._conectar() as con:
            fila = con.execute(
                "SELECT id FROM stand_sesiones WHERE stand_id = ? "
                "AND fecha = ? AND estado = 'abierta' "
                "ORDER BY id DESC LIMIT 1",
                (str(stand_id), dia)).fetchone()
        return int(fila['id']) if fila else None

    def cerrar_sesion(self, sesion_id: int) -> None:
        with self._lock, self._conectar() as con:
            con.execute(
                "UPDATE stand_sesiones SET hora_fin = ?, estado = 'cerrada' "
                "WHERE id = ?",
                (datetime.now().strftime('%H:%M:%S'), int(sesion_id)))

    def sesiones(self) -> List[Dict[str, Any]]:
        with self._conectar() as con:
            filas = con.execute(
                'SELECT * FROM stand_sesiones ORDER BY id DESC').fetchall()
        return [dict(f) for f in filas]

    # ── Visitantes ──────────────────────────────────────────────────────
    def upsert_visitante(self, sesion_id: int, persistent_id: str,
                         **campos: Any) -> None:
        """Alta o actualización por (sesion_id, persistent_id).

        Upsert y no insert: el visitante se va enriqueciendo a lo largo de su
        paso (primero existe, luego se detiene, luego lo atienden, luego
        degusta). Insertar en cada etapa duplicaría personas, que es
        exactamente el sobreconteo que este módulo existe para evitar.
        """
        permitidos = {
            'primera_deteccion', 'ultima_deteccion', 'duracion_total_s',
            'genero', 'genero_confianza', 'genero_muestras', 'edad_estimada',
            'rango_etario', 'edad_muestras', 'demografia_valida',
            'motivo_sin_demografia', 'se_detuvo', 'dwell_max_s',
            'fue_atendido', 'ts_atencion', 'duracion_atencion_s', 'degusto',
            'ts_degustacion', 'metodo_deteccion_degustacion',
            'confianza_degustacion', 'trayectoria_resumen', 'clasificacion',
            'es_personal', 'tracker_ids', 'descartado', 'motivo_descarte',
        }
        desconocidos = set(campos) - permitidos
        if desconocidos:
            # Explícito: un campo mal escrito se perdería en silencio y el
            # dashboard mostraría un hueco sin explicación.
            raise ValueError(f"campos desconocidos: {sorted(desconocidos)}")

        datos = dict(campos)
        for clave in ('trayectoria_resumen', 'tracker_ids'):
            if clave in datos:
                datos[clave] = _json_o_none(datos[clave])
        for clave in ('demografia_valida', 'se_detuvo', 'fue_atendido',
                      'degusto', 'es_personal', 'descartado'):
            if clave in datos and datos[clave] is not None:
                datos[clave] = int(bool(datos[clave]))

        with self._lock, self._conectar() as con:
            con.execute(
                'INSERT OR IGNORE INTO stand_visitantes '
                '(sesion_id, persistent_id, primera_deteccion) '
                'VALUES (?, ?, ?)',
                (int(sesion_id), str(persistent_id), _ahora()))
            if datos:
                sets = ', '.join(f'{k} = ?' for k in datos)
                con.execute(
                    f'UPDATE stand_visitantes SET {sets} '
                    'WHERE sesion_id = ? AND persistent_id = ?',
                    (*datos.values(), int(sesion_id), str(persistent_id)))

    def visitante(self, sesion_id: int,
                  persistent_id: str) -> Optional[Dict[str, Any]]:
        with self._conectar() as con:
            fila = con.execute(
                'SELECT * FROM stand_visitantes '
                'WHERE sesion_id = ? AND persistent_id = ?',
                (int(sesion_id), str(persistent_id))).fetchone()
        if fila is None:
            return None
        d = dict(fila)
        d['trayectoria_resumen'] = _carga_json(d.get('trayectoria_resumen'))
        d['tracker_ids'] = _carga_json(d.get('tracker_ids'))
        return d

    def personal_de(self, sesion_id: int) -> List[str]:
        """`persistent_id` marcados como personal en la sesión.

        Es la otra mitad del enrolamiento por API: el POST de
        `/api/stand/{sesion}/personal/enrolar` escribe el flag AQUÍ, pero la
        galería que decide `es_personal` durante el procesado
        (`stand_personal.GaleriaPersonal`) vive en memoria del pipeline.
        `stand_puente` llama esto periódicamente para reflejar el flag en la
        galería viva; sin esa relectura el marcado del operador se quedaba en
        la BD, `_localizar_chef` devolvía None y no había una sola atención
        (visto el 7-ago-2026: chef marcado el día 5, cero atenciones después).
        """
        with self._conectar() as con:
            filas = con.execute(
                'SELECT persistent_id FROM stand_visitantes '
                'WHERE sesion_id = ? AND es_personal = 1',
                (int(sesion_id),)).fetchall()
        return [str(f['persistent_id']) for f in filas]

    def visitantes(self, sesion_id: int,
                   incluir_personal: bool = False,
                   incluir_descartados: bool = False) -> List[Dict[str, Any]]:
        """Visitantes de la sesión.

        Por defecto EXCLUYE personal y descartados: son los dos filtros que
        deben aplicarse a cualquier métrica. Quien los quiera (el panel de
        calidad del dato) tiene que pedirlos a propósito.
        """
        sql = 'SELECT * FROM stand_visitantes WHERE sesion_id = ?'
        if not incluir_personal:
            sql += ' AND es_personal = 0'
        if not incluir_descartados:
            sql += ' AND descartado = 0'
        with self._conectar() as con:
            filas = con.execute(sql, (int(sesion_id),)).fetchall()
        salida = []
        for f in filas:
            d = dict(f)
            d['trayectoria_resumen'] = _carga_json(
                d.get('trayectoria_resumen'))
            d['tracker_ids'] = _carga_json(d.get('tracker_ids'))
            salida.append(d)
        return salida

    # ── Eventos ─────────────────────────────────────────────────────────
    def registrar_evento(self, sesion_id: int, tipo: str,
                         persistent_id: Optional[str] = None,
                         zona: Optional[str] = None,
                         payload: Optional[Dict] = None,
                         confianza: Optional[float] = None,
                         ts: Optional[str] = None) -> None:
        """Bitácora inmutable: solo se inserta, nunca se actualiza."""
        if tipo not in TipoEvento.TODOS:
            raise ValueError(f"tipo de evento desconocido: {tipo!r}")
        with self._lock, self._conectar() as con:
            con.execute(
                'INSERT INTO stand_eventos (sesion_id, persistent_id, tipo, '
                'ts, zona, payload, confianza) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (int(sesion_id), persistent_id, tipo, ts or _ahora(), zona,
                 _json_o_none(payload), confianza))

    def eventos(self, sesion_id: int, tipo: Optional[str] = None,
                limite: int = 500) -> List[Dict[str, Any]]:
        sql = 'SELECT * FROM stand_eventos WHERE sesion_id = ?'
        params: List[Any] = [int(sesion_id)]
        if tipo:
            sql += ' AND tipo = ?'
            params.append(tipo)
        sql += ' ORDER BY id DESC LIMIT ?'
        params.append(int(limite))
        with self._conectar() as con:
            filas = con.execute(sql, params).fetchall()
        salida = []
        for f in filas:
            d = dict(f)
            d['payload'] = _carga_json(d.get('payload'))
            salida.append(d)
        return salida

    # ── Heatmap ─────────────────────────────────────────────────────────
    def guardar_heatmap(self, sesion_id: int, tipo: str, grid: Dict,
                        matriz: List[List[float]]) -> None:
        if tipo not in ('trafico', 'permanencia', 'detenciones'):
            raise ValueError(f"tipo de heatmap desconocido: {tipo!r}")
        with self._lock, self._conectar() as con:
            con.execute(
                'INSERT INTO stand_heatmap (sesion_id, ts, tipo, grid, '
                'matriz) VALUES (?, ?, ?, ?, ?)',
                (int(sesion_id), _ahora(), tipo, _json_o_none(grid),
                 _json_o_none(matriz)))

    def ultimo_heatmap(self, sesion_id: int,
                       tipo: str) -> Optional[Dict[str, Any]]:
        with self._conectar() as con:
            fila = con.execute(
                'SELECT * FROM stand_heatmap WHERE sesion_id = ? AND tipo = ? '
                'ORDER BY id DESC LIMIT 1',
                (int(sesion_id), tipo)).fetchone()
        if fila is None:
            return None
        d = dict(fila)
        d['grid'] = _carga_json(d.get('grid'))
        d['matriz'] = _carga_json(d.get('matriz'))
        return d

    # ── Métricas agregadas ──────────────────────────────────────────────
    def guardar_metricas(self, sesion_id: int, bucket_ts: str,
                         kpis: Dict, por_genero: Optional[Dict] = None,
                         por_rango_etario: Optional[Dict] = None,
                         calidad_dato: Optional[Dict] = None) -> None:
        with self._lock, self._conectar() as con:
            con.execute(
                'INSERT INTO stand_metricas (sesion_id, bucket_ts, kpis, '
                'por_genero, por_rango_etario, calidad_dato) '
                'VALUES (?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(sesion_id, bucket_ts) DO UPDATE SET '
                'kpis = excluded.kpis, por_genero = excluded.por_genero, '
                'por_rango_etario = excluded.por_rango_etario, '
                'calidad_dato = excluded.calidad_dato',
                (int(sesion_id), bucket_ts, _json_o_none(kpis),
                 _json_o_none(por_genero), _json_o_none(por_rango_etario),
                 _json_o_none(calidad_dato)))

    def metricas(self, sesion_id: int) -> List[Dict[str, Any]]:
        with self._conectar() as con:
            filas = con.execute(
                'SELECT * FROM stand_metricas WHERE sesion_id = ? '
                'ORDER BY bucket_ts', (int(sesion_id),)).fetchall()
        salida = []
        for f in filas:
            d = dict(f)
            for c in ('kpis', 'por_genero', 'por_rango_etario',
                      'calidad_dato'):
                d[c] = _carga_json(d.get(c))
            salida.append(d)
        return salida

    # ── Conteos del embudo ──────────────────────────────────────────────
    def contadores(self, sesion_id: int) -> Dict[str, int]:
        """Las cuatro cifras del embudo, contadas por persistent_id.

        Monótonas por construcción: quien degustó también fue atendido, se
        detuvo y pasó. Así el embudo nunca se invierte (HITO 7).
        """
        with self._conectar() as con:
            fila = con.execute('''
                SELECT
                    COUNT(*)                      AS pasaron,
                    COALESCE(SUM(se_detuvo), 0)   AS se_detuvieron,
                    COALESCE(SUM(fue_atendido),0) AS atendidos,
                    COALESCE(SUM(degusto), 0)     AS degustaron
                FROM stand_visitantes
                WHERE sesion_id = ? AND es_personal = 0 AND descartado = 0
            ''', (int(sesion_id),)).fetchone()
        return {k: int(fila[k]) for k in
                ('pasaron', 'se_detuvieron', 'atendidos', 'degustaron')}
