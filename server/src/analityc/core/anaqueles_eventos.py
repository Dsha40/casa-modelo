"""Cada visita y agarre de los anaqueles con su hora, y dónde está cada
producto en la foto de referencia (13-oct-2026).

## Para qué

El registro del día (`anaqueles_registro.py`) guarda TOTALES: cuántas visitas
y agarres, por área y por producto. El informe del pasillo (PDF del
dashboard) necesita además la lista de agarres con su HORA y quién los hizo
(para cruzarlo con su ficha y su género), y las cajas de los productos
dibujadas sobre la foto del anaquel. Nada de eso se guardaba.

## Qué guarda

  * `<cámara>/eventos/<fecha>.jsonl`: una línea por visita o agarre, con
    `t` (epoch), `hora`, `tipo`, `area`, `area_nombre`, `producto`, `pid`
    (identidad o `prov:<número>`), `genero` y `rango` del momento.
  * `<cámara>/fotos_ref/productos.json`: por área, la caja del recorte de su
    foto de referencia y los polígonos de sus productos RELATIVOS a ese
    recorte. Se escribe en el mismo instante que la foto, así siempre
    encajan.

Junto al resto de los datos de los anaqueles (`output/anaqueles/`), fuera
de git. Nunca lanza: guardar es un extra, no puede costar el conteo.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .anaqueles_registro import _carpeta_nombre, raiz_salida

logger = logging.getLogger(__name__)

_candado = threading.Lock()


def _carpeta(camera_id: Any) -> Path:
    return raiz_salida() / _carpeta_nombre(camera_id)


def apuntar(carpeta_camara: Path, evento: Dict[str, Any], t: float) -> None:
    """Añade una visita o agarre al archivo del día (fecha local de `t`).
    `carpeta_camara` es la del registro de esa cámara (en las pruebas, una
    temporal: nunca se escribe en los datos reales)."""
    try:
        if carpeta_camara is None:
            return
        carpeta = Path(carpeta_camara) / 'eventos'
        carpeta.mkdir(parents=True, exist_ok=True)
        fila = dict(evento, t=round(float(t), 3))
        ruta = carpeta / f"{time.strftime('%Y-%m-%d', time.localtime(t))}.jsonl"
        with _candado, open(ruta, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(fila, ensure_ascii=False) + '\n')
    except Exception as exc:                                # noqa: BLE001
        logger.debug("anaqueles_eventos: no se apuntó (%s)", exc)


def leer(camera_id: Any, fecha: str) -> Optional[List[Dict[str, Any]]]:
    """Los eventos de un día, o None si ese día no se guardaban."""
    ruta = _carpeta(camera_id) / 'eventos' / f'{fecha}.jsonl'
    if not ruta.is_file():
        return None
    salida = []
    for linea in ruta.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            salida.append(json.loads(linea))
        except ValueError:
            continue
    return salida


def guardar_diseno(carpeta_camara: Path, idx: int, caja: List[int],
                   productos: List[Dict[str, Any]]) -> None:
    """Caja del recorte de la foto del área `idx` y sus productos (polígonos
    relativos a esa caja)."""
    try:
        if carpeta_camara is None:
            return
        carpeta = Path(carpeta_camara) / 'fotos_ref'
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / 'productos.json'
        with _candado:
            datos = {}
            if ruta.is_file():
                try:
                    datos = json.loads(ruta.read_text(encoding='utf-8'))
                except ValueError:
                    datos = {}
            datos[str(int(idx))] = {'caja': [int(v) for v in caja],
                                    'productos': productos}
            tmp = carpeta / 'productos.tmp.json'
            tmp.write_text(json.dumps(datos, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, ruta)
    except Exception as exc:                                # noqa: BLE001
        logger.debug("anaqueles_eventos: diseño no guardado (%s)", exc)


def diseno(camera_id: Any) -> Dict[str, Any]:
    """{área (str): {'caja', 'productos': [{'pos', 'nombre', 'poligono'}]}}"""
    try:
        ruta = _carpeta(camera_id) / 'fotos_ref' / 'productos.json'
        return json.loads(ruta.read_text(encoding='utf-8')) if ruta.is_file() else {}
    except Exception:                                       # noqa: BLE001
        return {}
