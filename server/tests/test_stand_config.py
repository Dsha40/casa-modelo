"""
Pruebas del cargador de configuración y zonas del stand (HITO 3).

Lo que se protege aquí es una sola idea: **una zona mal calibrada nunca debe
producir números**. Es peor que no medir, porque las cifras salen creíbles y
son falsas. Por eso todos los caminos de error devuelven zona vacía y
`listo_para_contar()` en False, en vez de caer a un valor por defecto.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / 'server'))

from src.analityc.core import stand_config as sc  # noqa: E402


def _escribir(carpeta: Path, nombre: str, texto: str) -> Path:
    ruta = carpeta / nombre
    ruta.write_text(texto, encoding='utf-8')
    return ruta


def test_carga_la_configuracion_real_del_repo():
    """Los YAML versionados en config/ deben cargar sin romperse.

    NO se afirma nada sobre `habilitado` ni sobre coordenadas concretas: son
    valores de operación que cambian (el flag se enciende, las zonas se
    recalibran). Una prueba atada a ellos falla por motivos que no tienen que
    ver con lo que dice probar — pasó con las tres de este archivo.
    """
    cfg = sc.cargar()
    # El stand_id concreto es un valor de operación (cambió el 7-ago-2026 de
    # charcuteria-01 a finca-dos-aguas-01 al mover la vista de la cámara):
    # solo se afirma que hay uno y que no es el fallback de "no hay config".
    assert cfg.stand_id and cfg.stand_id != 'stand-desconocido'
    assert cfg.get('detencion.dwell_min_s') == 3.0
    assert cfg.get('degustacion.vlm.confianza_min') == 0.7
    # Ruta inexistente -> el defecto, sin excepción.
    assert cfg.get('no.existe.nada', 'defecto') == 'defecto'


def test_las_cinco_zonas_semilla_estan_completas():
    cfg = sc.cargar()
    calibradas = cfg.zonas_calibradas()
    assert all(calibradas.values()), f"zonas vacías: {calibradas}"
    # La línea de conteo son 2 puntos; los polígonos, al menos 3.
    assert len(cfg.zona_normalizada(sc.ZONA_LINEA)) == 2
    for nombre in sc.ZONAS_POLIGONO:
        assert len(cfg.zona_normalizada(nombre)) >= 3


def test_conversion_a_pixeles():
    """Escala con el tamaño del frame: el punto de guardarlas normalizadas.

    Usa zonas propias en vez de las del repo, que se recalibran.
    """
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)
        cfg_p = _escribir(carpeta, 'c.yaml', 'habilitado: true\n')
        z_p = _escribir(carpeta, 'z.yaml',
                        'zonas:\n  zona_stand: [[0.25, 0.5], [0.75, 0.5], '
                        '[0.75, 0.9], [0.25, 0.9]]\n')
        cfg = sc.cargar(cfg_p, z_p)

        poli = cfg.zona_px('zona_stand', 1280, 720)
        assert poli is not None and poli.shape[1] == 2
        assert poli[0][0] == 320            # 0.25 * 1280
        assert poli[0][1] == 360            # 0.50 * 720
        # El mismo poligono en otro tamaño escala solo.
        otro = cfg.zona_px('zona_stand', 640, 360)
        assert otro[0][0] == 160 and otro[0][1] == 180


def test_zona_sin_calibrar_devuelve_none_no_el_frame_entero():
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)
        cfg_p = _escribir(carpeta, 'c.yaml', 'habilitado: true\n')
        z_p = _escribir(carpeta, 'z.yaml', 'zonas:\n  zona_stand: []\n')
        cfg = sc.cargar(cfg_p, z_p)
        assert cfg.zona_px('zona_stand', 1280, 720) is None


def test_puntos_en_pixeles_se_rechazan():
    """Editar a mano y poner píxeles es un error silencioso clásico."""
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)
        cfg_p = _escribir(carpeta, 'c.yaml', 'habilitado: true\n')
        z_p = _escribir(carpeta, 'z.yaml',
                        'zonas:\n  zona_stand: [[100, 200], [800, 200], '
                        '[800, 600]]\n')
        cfg = sc.cargar(cfg_p, z_p)
        assert cfg.zona_normalizada('zona_stand') == []
        ok, motivo = cfg.listo_para_contar()
        assert ok is False and 'zona_stand' in motivo


def test_poligono_con_pocos_puntos_se_rechaza():
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)
        cfg_p = _escribir(carpeta, 'c.yaml', 'habilitado: true\n')
        z_p = _escribir(carpeta, 'z.yaml',
                        'zonas:\n  zona_paso: [[0.1, 0.1], [0.5, 0.1]]\n')
        cfg = sc.cargar(cfg_p, z_p)
        assert cfg.zona_normalizada('zona_paso') == []


def test_listo_para_contar_exige_las_dos_zonas_criticas():
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)
        cfg_p = _escribir(carpeta, 'c.yaml', 'habilitado: true\n')
        cuadro = '[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]'

        # Sin zona_paso: no se puede calcular el denominador "pasaron".
        z1 = _escribir(carpeta, 'z1.yaml', f'zonas:\n  zona_stand: {cuadro}\n')
        ok, motivo = sc.cargar(cfg_p, z1).listo_para_contar()
        assert ok is False and 'zona_paso' in motivo

        # Con las dos, ya se puede medir aunque falten las opcionales.
        z2 = _escribir(carpeta, 'z2.yaml',
                       f'zonas:\n  zona_paso: {cuadro}\n'
                       f'  zona_stand: {cuadro}\n')
        ok, _ = sc.cargar(cfg_p, z2).listo_para_contar()
        assert ok is True


def test_deshabilitado_nunca_cuenta():
    """`habilitado: false` manda sobre cualquier calibración.

    (El pipeline `AmazonasStand` ya no consulta este flag —elegir el modo ES
    habilitarlo—, pero `listo_para_contar` sigue respetándolo.)
    """
    with tempfile.TemporaryDirectory() as d:
        carpeta = Path(d)
        cfg_p = _escribir(carpeta, 'c.yaml', 'habilitado: false\n')
        cuadro = '[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]'
        z_p = _escribir(carpeta, 'z.yaml',
                        f'zonas:\n  zona_paso: {cuadro}\n'
                        f'  zona_stand: {cuadro}\n')
        ok, motivo = sc.cargar(cfg_p, z_p).listo_para_contar()
        assert ok is False and 'deshabilitado' in motivo


def test_avisos_declaran_las_degradaciones():
    cfg = sc.cargar()
    avisos = cfg.avisos()
    # Las 5 zonas están, así que el único aviso vivo es el de homografía.
    assert any('homograf' in a for a in avisos)
    assert cfg.homografia_activa is False
    assert cfg.matriz_homografia() is None


def test_archivos_ausentes_no_revientan():
    cfg = sc.cargar(Path('/no/existe.yaml'), Path('/tampoco.yaml'))
    assert cfg.habilitado is False
    assert cfg.zona_px('zona_stand', 100, 100) is None


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
