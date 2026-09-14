"""
Smoke test del almacén del stand de degustación (HITO 2).

Verifica lo que el criterio de aceptación pide: esquema creado con índices,
y un visitante ficticio que se inserta y se lee. Añade además las dos
garantías de las que depende la confiabilidad del sistema:

  - el conteo del embudo es MONÓTONO (degustaron <= atendidos <=
    se_detuvieron <= pasaron), que es lo que impide un embudo invertido;
  - personal y descartados quedan FUERA de las métricas por defecto.

Sin GPU y sin red: solo SQLite en carpeta temporal.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core.stand_almacen import (  # noqa: E402
    AlmacenStand, Clasificacion, MotivoSinDemografia, TipoEvento)


def _almacen(tmp: Path) -> AlmacenStand:
    return AlmacenStand(ruta=tmp / 'stand.db')


def test_esquema_e_indices_creados():
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        with alm._conectar() as con:
            tablas = {f[0] for f in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            indices = {f[0] for f in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")}

        assert {'stand_sesiones', 'stand_visitantes', 'stand_eventos',
                'stand_heatmap', 'stand_metricas'} <= tablas
        # Los tres índices que exige el HITO 2.
        assert 'idx_vis_pid' in indices
        assert 'idx_ev_sesion_ts' in indices
        assert 'idx_vis_clasif' in indices


def test_visitante_ficticio_ida_y_vuelta():
    """El smoke test literal del criterio de aceptación."""
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(
            stand_id='charcuteria-01', nombre='Stand de prueba',
            producto='Jamón', camara_id='win-iVMS-4200',
            config_snapshot={'dwell_min_s': 3.0})

        alm.upsert_visitante(
            sid, 'p-0001',
            genero='mujer', genero_confianza=0.81, genero_muestras=6,
            edad_estimada=34.0, rango_etario='26-35', edad_muestras=6,
            demografia_valida=True,
            se_detuvo=True, dwell_max_s=12.4,
            fue_atendido=True, duracion_atencion_s=21.0,
            degusto=True, confianza_degustacion=0.88,
            metodo_deteccion_degustacion='pose+vlm',
            clasificacion=Clasificacion.DEGUSTO,
            tracker_ids=[11, 47],
            trayectoria_resumen={'entrada_zona': 'zona_paso',
                                 'distancia_recorrida_m': 5.2})

        v = alm.visitante(sid, 'p-0001')
        assert v is not None
        assert v['genero'] == 'mujer'
        assert v['rango_etario'] == '26-35'
        # Los booleanos vuelven como 0/1 de SQLite.
        assert v['degusto'] == 1 and v['fue_atendido'] == 1
        # Los campos anidados vuelven deserializados, no como texto.
        assert v['tracker_ids'] == [11, 47]
        assert v['trayectoria_resumen']['distancia_recorrida_m'] == 5.2


def test_upsert_no_duplica_a_la_persona():
    """El visitante se enriquece por etapas SIN crear filas nuevas.

    Es la garantía central contra el sobreconteo: si cada etapa insertara,
    una sola persona aparecería cuatro veces.
    """
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1')

        alm.upsert_visitante(sid, 'p-9', se_detuvo=True)
        alm.upsert_visitante(sid, 'p-9', fue_atendido=True)
        alm.upsert_visitante(sid, 'p-9', degusto=True,
                             clasificacion=Clasificacion.DEGUSTO)

        assert len(alm.visitantes(sid)) == 1
        v = alm.visitante(sid, 'p-9')
        assert v['se_detuvo'] == 1 and v['fue_atendido'] == 1
        assert v['degusto'] == 1


def test_campo_desconocido_falla_en_alto():
    """Un campo mal escrito debe reventar, no perderse en silencio."""
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1')
        try:
            alm.upsert_visitante(sid, 'p-1', degustoo=True)   # typo a propósito
        except ValueError as exc:
            assert 'degustoo' in str(exc)
        else:
            raise AssertionError("un campo desconocido pasó desapercibido")


def test_embudo_monotono_y_exclusiones():
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1')

        # Pasó de largo.
        alm.upsert_visitante(sid, 'p-1',
                             clasificacion=Clasificacion.PASO_DE_LARGO)
        # Se detuvo, nadie lo atendió.
        alm.upsert_visitante(
            sid, 'p-2', se_detuvo=True,
            clasificacion=Clasificacion.SE_DETUVO_SIN_ATENCION)
        # Atendido sin degustar.
        alm.upsert_visitante(
            sid, 'p-3', se_detuvo=True, fue_atendido=True,
            clasificacion=Clasificacion.ATENDIDO_SIN_DEGUSTAR)
        # Degustó (acumulativo: también detenido y atendido).
        alm.upsert_visitante(
            sid, 'p-4', se_detuvo=True, fue_atendido=True, degusto=True,
            clasificacion=Clasificacion.DEGUSTO)
        # El chef: NO debe contar en ninguna cifra.
        alm.upsert_visitante(sid, 'chef-1', es_personal=True,
                             se_detuvo=True, fue_atendido=True)
        # Track basura descartado: tampoco cuenta, pero se conserva.
        alm.upsert_visitante(sid, 'p-basura', descartado=True,
                             motivo_descarte='track_corto')

        c = alm.contadores(sid)
        assert c == {'pasaron': 4, 'se_detuvieron': 3,
                     'atendidos': 2, 'degustaron': 1}
        # La invariante que impide un embudo invertido.
        assert (c['degustaron'] <= c['atendidos'] <= c['se_detuvieron']
                <= c['pasaron'])

        # Los excluidos siguen en disco para el panel de calidad del dato.
        assert len(alm.visitantes(sid, incluir_personal=True,
                                  incluir_descartados=True)) == 6


def test_visitante_sin_demografia_igual_cuenta():
    """El fix del hallazgo §3.2 de la auditoría.

    Hoy `_emitir_visitante` descarta a quien no tiene género, y esa persona
    desaparece. Aquí se fija lo contrario: sin demografía se cuenta igual,
    con el motivo explícito.
    """
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1')
        alm.upsert_visitante(
            sid, 'p-sin-demo', se_detuvo=True, fue_atendido=True, degusto=True,
            demografia_valida=False,
            motivo_sin_demografia=MotivoSinDemografia.SIN_MUESTRAS,
            clasificacion=Clasificacion.DEGUSTO)

        c = alm.contadores(sid)
        assert c['degustaron'] == 1, "una degustación sin género se perdió"
        v = alm.visitante(sid, 'p-sin-demo')
        assert v['genero'] is None
        assert v['motivo_sin_demografia'] == 'sin_muestras'


def test_bitacora_de_eventos():
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1')
        alm.registrar_evento(sid, TipoEvento.ENTRADA_ESCENA,
                             persistent_id='p-1', zona='zona_paso')
        alm.registrar_evento(sid, TipoEvento.DEGUSTACION, persistent_id='p-1',
                             confianza=0.9, payload={'metodo': 'vlm'})

        assert len(alm.eventos(sid)) == 2
        deg = alm.eventos(sid, tipo=TipoEvento.DEGUSTACION)
        assert len(deg) == 1
        assert deg[0]['payload'] == {'metodo': 'vlm'}
        assert deg[0]['confianza'] == 0.9

        try:
            alm.registrar_evento(sid, 'tipo_inventado')
        except ValueError:
            pass
        else:
            raise AssertionError("se aceptó un tipo de evento inexistente")


def test_heatmap_y_metricas():
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1')

        alm.guardar_heatmap(sid, 'permanencia',
                            grid={'ancho': 4, 'alto': 2, 'celda_px': 20},
                            matriz=[[0, 1, 2, 3], [4, 5, 6, 7]])
        hm = alm.ultimo_heatmap(sid, 'permanencia')
        assert hm['matriz'][1][3] == 7
        assert hm['grid']['celda_px'] == 20
        assert alm.ultimo_heatmap(sid, 'trafico') is None

        # El agregado del bucket se re-escribe, no se duplica.
        alm.guardar_metricas(sid, '2026-08-05T15:00', kpis={'pasaron': 10})
        alm.guardar_metricas(sid, '2026-08-05T15:00', kpis={'pasaron': 12})
        m = alm.metricas(sid)
        assert len(m) == 1 and m[0]['kpis']['pasaron'] == 12


def test_ciclo_de_sesion():
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        sid = alm.abrir_sesion(stand_id='s1', nombre='Jornada 1')
        assert alm.sesiones()[0]['estado'] == 'abierta'
        alm.cerrar_sesion(sid)
        s = alm.sesiones()[0]
        assert s['estado'] == 'cerrada' and s['hora_fin']


# ── Ejecutable sin pytest ────────────────────────────────────────────────
# Ningun venv del repo tiene pytest instalado (verificado 5-ago-2026), asi
# que la prueba tiene que poder correrse con `python` a secas. Bajo pytest
# funciona igual: este bloque simplemente no se ejecuta.
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


def test_abrir_la_jornada_de_hoy_cierra_las_de_dias_anteriores():
    """Una sesión de ayer que quedó «abierta» (el servidor cruzó la
    medianoche o arrancó hoy sin cerrarla) se cierra al abrir la de hoy,
    con fin nominal 23:59:59; la de hoy no se toca, y las de OTRO stand
    tampoco."""
    with tempfile.TemporaryDirectory() as d:
        alm = _almacen(Path(d))
        with alm._conectar() as con:
            for stand, fecha in (('s1', '2026-09-01'), ('s1', '2026-09-03'),
                                 ('s2', '2026-09-03')):
                con.execute(
                    "INSERT INTO stand_sesiones (stand_id, nombre, producto, "
                    "fecha, hora_inicio, camara_id, config_snapshot, estado) "
                    "VALUES (?, 'n', 'p', ?, '09:00:00', 'cam', NULL, 'abierta')",
                    (stand, fecha))
        nueva = alm.abrir_sesion(stand_id='s1', nombre='n', producto='p',
                                 camara_id='cam', config_snapshot=None)
        filas = {(f['stand_id'], f['fecha']): f for f in alm.sesiones()
                 if f['id'] != nueva}
        assert filas[('s1', '2026-09-01')]['estado'] == 'cerrada'
        assert filas[('s1', '2026-09-01')]['hora_fin'] == '23:59:59'
        assert filas[('s1', '2026-09-03')]['estado'] == 'cerrada'
        assert filas[('s2', '2026-09-03')]['estado'] == 'abierta'   # otro stand
        hoy = [f for f in alm.sesiones() if f['id'] == nueva][0]
        assert hoy['estado'] == 'abierta' and hoy['hora_fin'] is None
        # Y la de hoy se reanuda en vez de duplicarse.
        assert alm.sesion_abierta_de('s1') == nueva
