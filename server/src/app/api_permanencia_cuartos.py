"""src/app/api_permanencia_cuartos.py — permanencia por cuarto (30-oct-2026).

GET `/api/v1/tienda/permanencia-cuartos?fecha=AAAA-MM-DD` (por defecto, hoy):
por cada cámara de solo permanencia, cuántas estancias hubo, el tiempo total
y el promedio por estancia. Lo escribe `permanencia_cuarto.py`.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

router = APIRouter(prefix='/api/v1/tienda', tags=['tienda'])


@router.get('/permanencia-cuartos')
def permanencia_cuartos(fecha: Optional[str] = Query(
        None, pattern=r'^\d{4}-\d{2}-\d{2}$')) -> Dict[str, Any]:
    from ..analityc.core import permanencia_cuarto
    dia = fecha or datetime.date.today().isoformat()
    return {'fecha': dia, 'camaras': permanencia_cuarto.leer_dia(dia)}
