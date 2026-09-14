"""src/analityc/core/tiendas_reporte.py — el informe consolidado de visitantes.

Extraído de `tiendas.py` (HITO 5) **sin tocar una línea del cuerpo**: el
pipeline eran 3.968 líneas en una sola clase, con la carga de modelos ONNX, el
tracking, la identidad, las fotos, el dibujado y el reporte conviviendo sin
fronteras. La regla de la casa es un archivo por funcionalidad.

Es un MIXIN, no un módulo de funciones sueltas, justamente para no cambiar
comportamiento: los métodos siguen siendo `self.get_analytics_report()`, las
subclases (`PermanenciaArea`, `Anaqueles`, `AmazonasStand`) los heredan igual
y un `super()` desde ellas sigue resolviendo por el MRO.

Qué hay aquí: las tres lecturas del estado de zona que consume el resto del
sistema — el resumen por persona, el informe consolidado (que es lo que se
guarda a JSON y viaja en el metadata) y su persistencia atómica.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _safe_report_name(value: Any) -> str:
    """Sanitiza un id (client_id/camera_id, puede ser UUID) para usarlo como
    nombre de archivo del reporte."""
    s = str(value if value is not None else "default")
    return "".join(c for c in s if c.isalnum() or c in "_-")[:64] or "default"


class ReporteMixin:
    """Informe consolidado de la analítica de visitantes."""

    def get_visitor_stats(self) -> Dict[str, Any]:
        """Resumen de visitantes del area (Hito 5), listo para reporte/overlay.

        Returns:
            dict con globales (unicos, entradas, salidas, aforo) y detalle por
            persona (visitas y tiempo dentro, sumando el intervalo abierto si
            sigue dentro).
        """
        now = time.time()
        personas = []
        for pid, st in self._zone_state.items():
            t_dentro = float(st.get('tiempo_total_dentro', 0.0))
            if st.get('estado') == 'DENTRO' and st.get('entry_time'):
                t_dentro += max(0.0, now - st['entry_time'])
            personas.append({
                'persistent_id': pid,
                'visitas': int(st.get('conteo_entradas', 0)),
                'salidas': int(st.get('conteo_salidas', 0)),
                'estado': st.get('estado', 'FUERA'),
                'tiempo_total_dentro': round(t_dentro, 1),
            })
        return {
            'visitantes_unicos': len(self._unique_entered),
            'total_entradas': int(self._total_entries),
            'total_salidas': int(self._total_exits),
            'aforo_actual': int(getattr(self, '_zone_count', 0)),
            'personas': personas,
        }

    def get_analytics_report(self) -> Dict[str, Any]:
        """Reporte CONSOLIDADO de analitica de visitantes (Hito 8).

        Une las metricas de todos los hitos en una estructura coherente, lista
        para guardar a JSON o pintar en overlay:
          - visitantes_unicos: persistent_id distintos que ENTRARON al area.
          - aforo_actual: personas dentro ahora (sv.PolygonZone).
          - total_entradas / total_salidas: transiciones confirmadas.
          - permanencia_media_s: media del tiempo dentro de los que entraron.
          - distribucion_genero / distribucion_edad: de los visitantes unicos
            (segun su demografia consolidada por persona en la galeria).
          - galeria_total: personas distintas historicas (DB persistente).

        Coherencia garantizada: aforo_actual <= visitantes_unicos y
        total_entradas >= aforo_actual (un dentro implica una entrada).
        """
        vs = self.get_visitor_stats()
        reid = getattr(self, '_reidentifier', None)
        # Demografia por uid (de la galeria; solo de quienes ENTRARON).
        persons_by_uid: Dict[str, Any] = {}
        if reid is not None:
            try:
                persons_by_uid = {p['uuid']: p for p in reid.get_all_persons()}
            except Exception:
                persons_by_uid = {}
        gen_dist: Counter = Counter()
        age_dist: Counter = Counter()
        dwell_vals: List[float] = []
        for p in vs['personas']:
            if int(p.get('visitas', 0)) <= 0:
                continue  # solo los que efectivamente entraron al area
            dwell_vals.append(float(p.get('tiempo_total_dentro', 0.0)))
            rec = persons_by_uid.get(p['persistent_id'])
            if rec:
                gen_dist[rec.get('gender') or 'Desconocido'] += 1
                age_dist[rec.get('age_range') or 'Desconocido'] += 1
        avg_dwell = (sum(dwell_vals) / len(dwell_vals)) if dwell_vals else 0.0
        # AFORO coherente: personas CONFIRMADAS actualmente DENTRO (subconjunto
        # de las que entraron) -> garantiza aforo_actual <= visitantes_unicos.
        # El zone_count geometrico (todos los pies dentro, incl. no confirmados)
        # se reporta aparte como aforo_geometrico (ocupacion cruda honesta).
        aforo_confirmado = sum(
            1 for st in self._zone_state.values()
            if st.get('estado') == 'DENTRO')
        return {
            'visitantes_unicos': vs['visitantes_unicos'],
            'aforo_actual': aforo_confirmado,
            'aforo_geometrico': vs['aforo_actual'],
            'total_entradas': vs['total_entradas'],
            'total_salidas': vs['total_salidas'],
            'permanencia_media_s': round(avg_dwell, 1),
            'distribucion_genero': dict(gen_dist),
            'distribucion_edad': dict(age_dist),
            'galeria_total': (reid.total_unique_persons
                              if reid is not None else 0),
            'timestamp': time.time(),
        }

    def save_analytics_report(self, path: str = None) -> bool:
        """Persiste el reporte consolidado a JSON (escritura atomica). Asi el
        dashboard/reportes sobreviven reinicios del proceso. Defensivo."""
        try:
            import json as _json
            path = path or os.path.join(
                "output", f"analytics_report_{_safe_report_name(self.client_id)}.json")
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(self.get_analytics_report(), f, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except Exception as exc:
            logger.debug("save_analytics_report fallo: %s", exc)
            return False
