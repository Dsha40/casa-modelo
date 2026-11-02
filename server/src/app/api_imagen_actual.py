"""src/app/api_imagen_actual.py — la última imagen que analizó el servidor,
con lo que entendió de ella.

## Para qué (30-oct-2026, rama contructora)

`/api/anaqueles/frame` solo sirve en modo Anaqueles, y en esta rama se usa
el modo Tiendas. Sin ver la imagen, cada «no lo sigue fuera del área» había
que reconstruirlo leyendo el log y las grabaciones. Con esto, quien
diagnostica ve lo mismo que el servidor:

  * el ROI en verde;
  * cada persona seguida, con su número, su identidad (8 caracteres, o
    `prov`), de dónde viene (cara, cuerpo…), si nació dentro o fuera del ROI
    (`origen_identidad.py`) y su género si ya está decidido.

GET `/api/v1/tienda/imagen-actual` (JPEG). `?camara=` elige la cámara si
hay varias. Son fotos de clientes: solo para mirar en el PC del servidor.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v1/tienda', tags=['tienda'])


def _de_la_camara(cam_proc: Any, nombre: str) -> Any:
    """Un atributo del estado de la cámara, esté colgado o guardado."""
    v = getattr(cam_proc, nombre, None)
    if v is not None and (not hasattr(v, '__len__') or len(v)):
        return v
    for estado in (getattr(cam_proc, 'camera_states', None) or {}).values():
        v2 = (estado or {}).get(nombre)
        if v2 is not None:
            return v2
    return v


def dibujar(img: np.ndarray, roi: Any, tracks: dict, demos: Any = None,
            origen: Any = None) -> np.ndarray:
    """Copia de `img` con el ROI y las personas pintadas."""
    import cv2
    out = img.copy()
    if roi is not None and len(roi) >= 3:
        cv2.polylines(out, [np.asarray(roi, np.int32).reshape(-1, 1, 2)],
                      True, (0, 200, 0), 3)
    for tid, tr in (tracks or {}).items():
        if tr.get('class', -1) != -1 or tr.get('box') is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in list(tr['box'])[:4]]
        pid = str(tr.get('persistent_id') or '')
        corto = 'prov' if not pid or pid.startswith('prov:') else pid[:8]
        nace = ''
        if origen is not None and corto != 'prov':
            d = origen.nacio_dentro(pid)
            nace = ' dentro' if d else (' FUERA' if d is False else '')
        genero = ''
        try:
            demo = demos.get_cached(tid) if demos is not None else None
            g = (demo or {}).get('gender')
            genero = f' {g}' if g and g != 'Desconocido' else ''
        except Exception:                                   # noqa: BLE001
            pass
        texto = (f"{tid} {corto} {tr.get('identity_source') or ''}"
                 f"{nace}{genero}")
        color = (0, 165, 255) if corto == 'prov' else (255, 120, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, texto, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


@router.get('/imagen-actual')
def imagen_actual(camara: Optional[str] = None):
    import cv2

    from ..analityc.core import origen_identidad
    from .app import _iter_person_processors

    for _cid, cam_id, cam_proc in _iter_person_processors():
        if camara and str(cam_id) != camara:
            continue
        img = _de_la_camara(cam_proc, 'last_processed_frame')
        if img is None:
            continue
        tracks = _de_la_camara(cam_proc, 'active_tracks') or {}
        try:
            out = dibujar(img, getattr(cam_proc, 'roi_polygon', None), tracks,
                          getattr(cam_proc, '_demographics', None),
                          origen_identidad)
            ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            return Response(content=buf.tobytes(), media_type='image/jpeg',
                            headers={'X-Camara': str(cam_id),
                                     'X-Personas': str(len(tracks))})
        except Exception as exc:                            # noqa: BLE001
            logger.error("imagen-actual: no se pudo dibujar: %s", exc)
    raise HTTPException(status_code=503,
                        detail='ninguna cámara está enviando imágenes ahora')
