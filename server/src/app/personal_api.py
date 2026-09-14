"""src/app/personal_api.py — API REST del personal de tienda.

## Prefijo propio, patrón stand_api/anaqueles_api

`/api/v1` es solo-lectura por regla; este módulo ESCRIBE (altas, bajas,
fotos), así que vive autocontenido en `/api/personal`. Lo consumen por igual
el dashboard de tienda (fetch mismo-origen) y el cliente Qt (requests).

## Fotos en base64, no multipart

Un solo camino para los dos consumidores: JSON `{nombre, rol, foto_b64}`.
El dashboard lo produce con FileReader.readAsDataURL y el cliente con
`base64.b64encode(open(foto).read())`. Se acepta con o sin el prefijo
`data:image/...;base64,`.

## Validación en el borde

La galería rechaza fotos sin rostro claro o con varios rostros y este
módulo convierte ese ValueError en un 400 con el MENSAJE PARA EL OPERADOR
(«usa una foto de frente, con buena luz») — el error debe leerse en el
formulario, no en un log.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..analityc.core.personal_galeria import ROLES, get_galeria

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/personal', tags=['personal'])

#: Tope de la foto decodificada. Una foto de teléfono ronda 2-5 MB.
_MAX_FOTO_BYTES = 10 * 1024 * 1024


@router.get('/')
def listar() -> Dict[str, Any]:
    """El personal enrolado, con la URL de su foto principal."""
    gal = get_galeria()
    gal.recargar_si_cambio()
    fichas = gal.listar()
    for f in fichas:
        f['foto_url'] = (f"/api/personal/{f['id']}/foto/0"
                         if f.get('n_fotos') else None)
    return {'total': len(fichas),
            'activos': sum(1 for f in fichas if f['activo']),
            'roles': list(ROLES),
            'personal': fichas}


@router.post('/enrolar')
def enrolar(cuerpo: Dict[str, Any]) -> Dict[str, Any]:
    """Alta: {nombre, rol: seguridad|pasillero, foto_b64, tipo?}.

    `tipo`: 'rostro' (default) identifica a LA PERSONA por la cara;
    'uniforme' identifica LA VESTIMENTA (bordado/colores, funciona de
    espaldas) — la foto debe ser del uniforme de cuerpo entero, y conviene
    añadir después una segunda foto de la espalda con POST /{id}/foto.
    """
    foto = _decodificar_foto(cuerpo.get('foto_b64'))
    try:
        ficha = get_galeria().enrolar(
            str(cuerpo.get('nombre') or ''),
            str(cuerpo.get('rol') or ''), foto,
            tipo=str(cuerpo.get('tipo') or 'rostro'))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {'ok': True, 'ficha': ficha}


@router.post('/{pid}/foto')
def agregar_foto(pid: str, cuerpo: Dict[str, Any]) -> Dict[str, Any]:
    """Otra foto del mismo empleado: {foto_b64}. Más ángulos = mejor."""
    foto = _decodificar_foto(cuerpo.get('foto_b64'))
    try:
        ficha = get_galeria().agregar_foto(pid, foto)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {'ok': True, 'ficha': ficha}


@router.patch('/{pid}')
def modificar(pid: str, cuerpo: Dict[str, Any]) -> Dict[str, Any]:
    """Cambia activo/nombre/rol. Desactivar NO borra: conserva las fotos
    para reactivar al empleado que vuelve de vacaciones."""
    try:
        ficha = get_galeria().poner(
            pid,
            activo=cuerpo.get('activo'),
            nombre=cuerpo.get('nombre'),
            rol=cuerpo.get('rol'))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {'ok': True, 'ficha': ficha}


@router.delete('/{pid}')
def borrar(pid: str) -> Dict[str, Any]:
    try:
        get_galeria().borrar(pid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {'ok': True}


@router.get('/{pid}/foto/{n}')
def foto(pid: str, n: int):
    ruta = get_galeria().foto_ruta(pid, int(n))
    if ruta is None:
        raise HTTPException(status_code=404,
                            detail=f'sin foto {n} para {pid!r}')
    return FileResponse(ruta, media_type='image/jpeg')


# ── Interno ──────────────────────────────────────────────────────────────
def _decodificar_foto(foto_b64: Any) -> np.ndarray:
    """data-URL o base64 pelado → imagen BGR. 400 con mensaje legible."""
    crudo = str(foto_b64 or '').strip()
    if not crudo:
        raise HTTPException(status_code=400,
                            detail='falta foto_b64 (la foto del empleado)')
    if ',' in crudo and crudo[:5].lower() == 'data:':
        crudo = crudo.split(',', 1)[1]
    # Tope ANTES de decodificar (base64 ≈ 4/3 del binario): decodificar
    # primero materializaba la copia gigante que el tope quería impedir.
    if len(crudo) > _MAX_FOTO_BYTES * 4 // 3 + 1024:
        raise HTTPException(status_code=400,
                            detail='la foto pesa más de 10 MB; usa una '
                                   'más liviana')
    try:
        datos = base64.b64decode(crudo, validate=False)
    except Exception:
        raise HTTPException(status_code=400,
                            detail='foto_b64 no es base64 válido')
    if len(datos) > _MAX_FOTO_BYTES:
        raise HTTPException(status_code=400,
                            detail='la foto pesa más de 10 MB; usa una '
                                   'más liviana')
    import cv2
    img = cv2.imdecode(np.frombuffer(datos, dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400,
                            detail='la foto no se pudo leer como imagen '
                                   '(usa JPG o PNG)')
    return img
