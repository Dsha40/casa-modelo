"""src/app/api_grabador_personas.py — parar y reanudar el grabador de
personas sin reiniciar el servidor (2-nov-2026).

El usuario lo pidió para probar con vídeos sin que se grabaran:

  * GET  `/api/v1/tienda/grabador`          → cómo está;
  * POST `/api/v1/tienda/grabador/parar`    → deja de grabar;
  * POST `/api/v1/tienda/grabador/reanudar` → vuelve a grabar.

Solo actúa sobre el grabador encendido al arrancar
(`ELDE_GRABAR_PERSONAS`): sin esa variable no hay nada que parar ni reanudar.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from . import grabador_personas

router = APIRouter(prefix='/api/v1/tienda/grabador', tags=['tienda'])

_SIN_CONFIGURAR = {'configurado': False, 'grabando': False,
                   'nota': 'el servidor se arrancó sin ELDE_GRABAR_PERSONAS'}


@router.get('')
def estado() -> Dict[str, Any]:
    g = grabador_personas.grabador()
    return g.estado() if g is not None else dict(_SIN_CONFIGURAR)


@router.post('/parar')
def parar() -> Dict[str, Any]:
    g = grabador_personas.grabador()
    if g is None:
        return dict(_SIN_CONFIGURAR)
    g.parar()
    return g.estado()


@router.post('/reanudar')
def reanudar() -> Dict[str, Any]:
    g = grabador_personas.grabador()
    if g is None:
        return dict(_SIN_CONFIGURAR)
    ok = g.reanudar()
    r = g.estado()
    if not ok:
        r['nota'] = 'no se reanudó: falta disco o se llegó al máximo'
    return r
