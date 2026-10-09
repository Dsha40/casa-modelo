"""src/app/dashboard_stock.py — apartado «Stock» del dashboard (9-oct-2026).

Lo pidió el usuario: ver en el dashboard las alertas de «posible agotado»
como las ve el panel de alertas del cliente, con su foto y los minutos que
lleva cada una.

Dos fuentes, a propósito:

  * **Activas**: el estado VIVO de `anaqueles_stock` (en memoria): producto,
    área, hora y segundos que lleva. Se pierden al reiniciar el servidor
    (las alertas no se guardan en disco).
  * **Historial**: las fotos de evidencia que deja cada alerta en
    `output/anaqueles/<cámara>/stock_alertas/<fecha>_<hora>_<área>_<producto>.jpg`
    (antes | ahora | quién cogió). Sobreviven a los reinicios.

Las fotos son de clientes: se sirven solo por esta ruta, dentro de la
carpeta de cada cámara (sin rutas relativas), igual que las capturas.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_RAIZ = Path('output') / 'anaqueles'
#: `..._cerca.jpg` = aviso «cerca de vaciarse» (9-oct-2026); sin sufijo,
#: «posible agotado».
_FOTO = re.compile(r'^(\d{4}-\d{2}-\d{2})_(\d{6})_(\d+)_(\d+)(_cerca)?\.jpg$')
_MAX_HISTORIAL = 60


def _nombres_productos(camara: Path) -> Dict[tuple, Dict[str, str]]:
    """(área, producto) -> nombres, del registro del día más reciente."""
    nombres: Dict[tuple, Dict[str, str]] = {}
    try:
        dias = sorted(camara.glob('????-??-??.json'))
        if not dias:
            return nombres
        datos = json.loads(dias[-1].read_text(encoding='utf-8'))
        for a, area in (datos.get('areas') or {}).items():
            for p, prod in (area.get('productos') or {}).items():
                nombres[(int(a), int(p))] = {
                    'area': area.get('nombre') or f'Área {int(a) + 1}',
                    'producto': prod.get('nombre') or f'Producto {int(p) + 1}'}
    except Exception:                                       # noqa: BLE001
        pass
    return nombres


def _estados() -> Dict[str, Dict[str, Any]]:
    """Estado vivo de stock por cámara (copia del pipeline)."""
    try:
        from ..analityc.core import anaqueles_stock
    except Exception:                                       # noqa: BLE001
        return {}
    return {cam: anaqueles_stock.estado(cam) or {}
            for cam in anaqueles_stock.camaras_vigiladas()}


def _activas() -> Dict[str, List[Dict[str, Any]]]:
    """Alertas vivas por cámara, cada una con su `tipo`."""
    out = {}
    for cam, estado in _estados().items():
        out[cam] = ([dict(a, tipo='agotado') for a in estado.get('alertas') or []]
                    + [dict(a, tipo='cerca') for a in estado.get('cerca') or []])
    return out


def _bajadas() -> List[Dict[str, Any]]:
    """Bajadas contadas por producto desde la última foto de balda surtida."""
    out = []
    for cam, estado in _estados().items():
        nombres = _nombres_productos(_RAIZ / cam)
        for b in estado.get('bajadas') or []:
            n = nombres.get((b['area'], b['producto_idx']), {})
            out.append({'camara': cam, 'bajadas': b['bajadas'],
                        'area': n.get('area') or f"Área {b['area'] + 1}",
                        'producto': n.get('producto') or f"Producto {b['producto_idx'] + 1}"})
    return sorted(out, key=lambda b: -b['bajadas'])


def resumen_stock(raiz: Optional[Path] = None) -> Dict[str, Any]:
    """Alertas activas y el historial de fotos, más reciente primero."""
    raiz = raiz or _RAIZ
    activas = _activas()
    ahora = time.time()
    historial: List[Dict[str, Any]] = []
    if raiz.is_dir():
        for camara in sorted(p for p in raiz.iterdir() if p.is_dir()):
            nombres = _nombres_productos(camara)
            vivas = activas.get(camara.name, [])
            carpeta = camara / 'stock_alertas'
            if not carpeta.is_dir():
                continue
            for f in carpeta.glob('*.jpg'):
                m = _FOTO.match(f.name)
                if not m:
                    continue
                fecha, hms, area, prod = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                tipo = 'cerca' if m.group(5) else 'agotado'
                try:
                    t = time.mktime(time.strptime(f'{fecha} {hms}', '%Y-%m-%d %H%M%S'))
                except ValueError:
                    continue
                hora = f'{hms[:2]}:{hms[2:4]}:{hms[4:]}'
                viva = next((a for a in vivas if a.get('area') == area
                             and a.get('producto_idx') == prod
                             and a.get('desde') == hora
                             and a.get('tipo') == tipo), None)
                n = nombres.get((area, prod), {})
                historial.append({
                    'tipo': tipo, 'bajadas': (viva or {}).get('bajadas'),
                    'camara': camara.name, 'archivo': f.name, 't': t,
                    'fecha': fecha, 'hora': hora,
                    'area': (viva or {}).get('area_nombre') or n.get('area') or f'Área {area + 1}',
                    'producto': (viva or {}).get('producto') or n.get('producto') or f'Producto {prod + 1}',
                    'cambio': (viva or {}).get('cambio'),
                    'activa': viva is not None,
                    # Segundos que lleva: el del pipeline si sigue viva; si
                    # no, desde la foto (la alerta pudo cerrarse o perderse
                    # en un reinicio: no se sabe cuándo terminó).
                    'lleva_s': int((viva or {}).get('vacio_s', ahora - t)),
                })
    historial.sort(key=lambda h: -h['t'])
    return {'status': 'ok', 'ahora': ahora,
            'activas': [h for h in historial if h['activa']],
            'historial': historial[:_MAX_HISTORIAL],
            'total': len(historial),
            'bajadas': _bajadas()}


@router.get('/dashboard/api/stock')
def api_stock() -> Dict[str, Any]:
    try:
        return resumen_stock()
    except Exception as exc:                                # noqa: BLE001
        return {'status': 'error', 'message': str(exc)}


@router.get('/dashboard/img/stock/{camara}/{archivo}')
def img_stock(camara: str, archivo: str):
    """Foto de evidencia de una alerta. Solo nombres con el formato de
    `anaqueles_stock`, dentro de la carpeta de esa cámara."""
    if not _FOTO.match(archivo) or '/' in camara or '\\' in camara or '..' in camara:
        raise HTTPException(status_code=404)
    ruta = _RAIZ / camara / 'stock_alertas' / archivo
    if not ruta.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(str(ruta), media_type='image/jpeg')
