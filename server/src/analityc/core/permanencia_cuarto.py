"""src/analityc/core/permanencia_cuarto.py — cuánto se queda la gente en un
cuarto, en las cámaras de «solo permanencia».

## Por qué (30-oct-2026, rama contructora)

Casa modelo: la recámara principal es la cámara de visitas (identidad y
género). El usuario quiere añadir otras cámaras (salón, cocina…) solo para
saber cuánto se queda la gente en cada cuarto, sin visitas y sin que frenen
al servidor. En esas cámaras (`SOLO_PERMANENCIA` en su perfil de cámara)
`tiendas.py` solo detecta y sigue: sin clasificar, sin identidades, sin
fichas ni avisos. Esto apunta las estancias.

## Qué es una estancia

Desde que aparece un seguimiento hasta que deja de verse `CERRAR_S`. Si en
los `COSTURA_S` siguientes aparece otro a menos de un alto de persona de
donde se perdió, es la MISMA estancia (la caja se cortó un momento: detrás
de la cama, de la isla…). Sin identidad: cada estancia cuenta como una
persona. Se descartan:

  * las de menos de `MIN_S` (detecciones sueltas);
  * las de un objeto QUIETO (`QUIETO_S` o más sin moverse más de
    `QUIETO_PX`): el 30-oct, un cuadro de la pared contó como visitante.

## Dónde queda

`output/permanencia_cuartos/<cámara>/<AAAA-MM-DD>.json`: cada estancia
(inicio, fin, segundos) y el resumen del día (número, total, promedio).
Cada 60 s, una línea `PERMANENCIA CUARTO` en el log.
"""
from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Sin verlo este tiempo, el seguimiento se da por perdido.
CERRAR_S = 2.0
#: Un seguimiento nuevo que aparece en este tiempo, cerca, sigue la estancia.
COSTURA_S = 5.0
#: Distancia máxima para seguir la estancia, en altos de persona.
COSTURA_ALTOS = 1.0
#: Estancias más cortas no cuentan.
MIN_S = 2.0
#: Un objeto que no se mueve más de QUIETO_PX en QUIETO_S o más no es gente.
QUIETO_S = 60.0
QUIETO_PX = 15.0
RESUMEN_CADA_S = 60.0

_RAIZ = Path(__file__).resolve().parents[3]


class PermanenciaCuarto:
    """Una por cámara. `observar` en cada imagen; nunca lanza."""

    def __init__(self, camera_id: Any, carpeta: Optional[Path] = None) -> None:
        from .anaqueles_registro import _carpeta_nombre
        self.camera_id = str(camera_id)
        self.carpeta = Path(carpeta) if carpeta else (
            _RAIZ / 'output' / 'permanencia_cuartos' / _carpeta_nombre(camera_id))
        #: tid vivo -> estancia
        self._vivas: Dict[Any, Dict[str, Any]] = {}
        #: estancias recién perdidas que aún pueden seguir (costura)
        self._perdidas: List[Dict[str, Any]] = []
        self._ultimo_resumen = time.time()
        self._candado = threading.Lock()

    def observar(self, tracks: Dict[Any, Dict[str, Any]],
                 ahora: Optional[float] = None) -> None:
        try:
            self._observar(tracks, time.time() if ahora is None else float(ahora))
        except Exception as exc:                            # noqa: BLE001
            logger.debug("permanencia_cuarto: %s", exc)

    def _observar(self, tracks: Dict[Any, Dict[str, Any]], ahora: float) -> None:
        vistos = set()
        for tid, tr in (tracks or {}).items():
            box = tr.get('box')
            if tr.get('class', -1) != -1 or box is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in list(box)[:4]]
            cx, pie, alto = (x1 + x2) / 2.0, y2, max(1.0, y2 - y1)
            vistos.add(tid)
            e = self._vivas.get(tid)
            if e is None:
                e = self._continuar(cx, pie, alto, ahora) or {
                    'inicio': ahora, 'cx_min': cx, 'cx_max': cx,
                    'cy_min': pie, 'cy_max': pie}
                self._vivas[tid] = e
            e['ultimo'], e['cx'], e['pie'], e['alto'] = ahora, cx, pie, alto
            e['cx_min'], e['cx_max'] = min(e['cx_min'], cx), max(e['cx_max'], cx)
            e['cy_min'], e['cy_max'] = min(e['cy_min'], pie), max(e['cy_max'], pie)
        for tid in [t for t, e in self._vivas.items()
                    if t not in vistos and ahora - e['ultimo'] > CERRAR_S]:
            self._perdidas.append(self._vivas.pop(tid))
        for e in [e for e in self._perdidas if ahora - e['ultimo'] > COSTURA_S]:
            self._perdidas.remove(e)
            self._cerrar(e)
        if ahora - self._ultimo_resumen >= RESUMEN_CADA_S:
            self._ultimo_resumen = ahora
            r = self.resumen()
            logger.info("PERMANENCIA CUARTO [%s] hoy: estancias=%d promedio=%.0f s "
                        "total=%.0f s (ahora dentro: %d)", self.camera_id,
                        r['estancias'], r['promedio_s'], r['total_s'],
                        len(self._vivas))

    def _continuar(self, cx: float, pie: float, alto: float,
                   ahora: float) -> Optional[Dict[str, Any]]:
        """La estancia recién perdida que este seguimiento continúa, si la
        hay: aparece en COSTURA_S y a menos de COSTURA_ALTOS altos."""
        mejor = None
        for e in self._perdidas:
            if ahora - e['ultimo'] > COSTURA_S:
                continue
            d = ((cx - e['cx']) ** 2 + (pie - e['pie']) ** 2) ** 0.5
            if d <= COSTURA_ALTOS * max(alto, e['alto']) and (
                    mejor is None or d < mejor[0]):
                mejor = (d, e)
        if mejor is None:
            return None
        self._perdidas.remove(mejor[1])
        return mejor[1]

    def _cerrar(self, e: Dict[str, Any]) -> None:
        dur = e['ultimo'] - e['inicio']
        if dur < MIN_S:
            return
        quieta = (dur >= QUIETO_S
                  and e['cx_max'] - e['cx_min'] <= QUIETO_PX
                  and e['cy_max'] - e['cy_min'] <= QUIETO_PX)
        dia = datetime.datetime.fromtimestamp(e['inicio']).strftime('%Y-%m-%d')
        with self._candado:
            datos = self._leer(dia)
            if quieta:
                datos['quietas_descartadas'] = datos.get('quietas_descartadas', 0) + 1
            else:
                datos['estancias'].append({
                    'inicio': round(e['inicio'], 1), 'fin': round(e['ultimo'], 1),
                    'segundos': round(dur, 1)})
            self._escribir(dia, datos)

    def _ruta(self, dia: str) -> Path:
        return self.carpeta / f'{dia}.json'

    def _leer(self, dia: str) -> Dict[str, Any]:
        try:
            return json.loads(self._ruta(dia).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'camera_id': self.camera_id, 'fecha': dia, 'estancias': []}

    def _escribir(self, dia: str, datos: Dict[str, Any]) -> None:
        segs = [x['segundos'] for x in datos['estancias']]
        datos['resumen'] = {
            'estancias': len(segs), 'total_s': round(sum(segs), 1),
            'promedio_s': round(sum(segs) / len(segs), 1) if segs else 0.0}
        self.carpeta.mkdir(parents=True, exist_ok=True)
        tmp = self._ruta(dia).with_suffix('.tmp')
        tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=1),
                       encoding='utf-8')
        tmp.replace(self._ruta(dia))

    def resumen(self, dia: Optional[str] = None) -> Dict[str, Any]:
        dia = dia or datetime.date.today().isoformat()
        with self._candado:
            r = self._leer(dia).get('resumen') or {}
        return {'estancias': int(r.get('estancias', 0)),
                'total_s': float(r.get('total_s', 0.0)),
                'promedio_s': float(r.get('promedio_s', 0.0))}


_POR_CAMARA: Dict[str, PermanenciaCuarto] = {}
_CANDADO = threading.Lock()


def de(camera_id: Any) -> PermanenciaCuarto:
    with _CANDADO:
        c = _POR_CAMARA.get(str(camera_id))
        if c is None:
            c = _POR_CAMARA[str(camera_id)] = PermanenciaCuarto(camera_id)
        return c


def leer_dia(dia: str) -> List[Dict[str, Any]]:
    """Los resúmenes de ese día de todas las cámaras (para la API)."""
    base = _RAIZ / 'output' / 'permanencia_cuartos'
    out = []
    if not base.is_dir():
        return out
    for cam in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            d = json.loads((cam / f'{dia}.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        out.append({'camera_id': d.get('camera_id', cam.name),
                    **(d.get('resumen') or {}),
                    'quietas_descartadas': d.get('quietas_descartadas', 0)})
    return out
