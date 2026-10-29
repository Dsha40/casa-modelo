"""Personal por el UNIFORME de Santa Rosa, en SOMBRA (16-oct-2026).

## Por qué

Los agarres, visitas y bajadas de stock falsos del 15-oct eran casi todos del
personal reponiendo (CLAUDE.md: revisión de las 25 bajadas, 22 falsas). Las
dos vías que ya había no sirven con esta cámara:

  - el alta de UNIFORME de `personal_galeria.py` compara la firma OSNet: el
    16-oct marcaba como personal a casi todos los clientes vestidos de azul
    u oscuro (OSNet mira el color, no el logo);
  - la CARA mide 36-58 px: la misma empleada de frente y mirando abajo salía
    como dos personas distintas (parecido -0,03 a 0,16).

Este módulo usa una ResNet-18 AFINADA con fotos marcadas a mano por el
usuario (azul arriba y salpicadura blanca abajo; tandas del 16-oct: 641
fotos de personal y 507 de clientes, más 574 probables clientes). Medido
con la tienda real (iVMS, fichas que el modelo no había visto): con el 90 %,
de 179 que daba como personal, 174 lo eran y 2 eran clientes (de ~2.900).

## Qué hace (modo «sombra»)

Cada `PERSONAL_UNIFORME_CADA` imágenes (escalonado por número de
seguimiento, en tiempo real con `_imagenes`), a cada persona vista en esta
imagen con la caja de al menos `PERSONAL_UNIFORME_ALTO_MIN` px, le pasa el
modelo y guarda su probabilidad. Cuando el seguimiento lleva 2 s sin verse,
escribe UNA línea en `output/uniforme_sombra/<cámara>/<AAAA-MM-DD>.jsonl`:
número, identidad, ficha, votos, media, máximo y si sería personal (al menos
`PERSONAL_UNIFORME_VOTOS_MIN` votos y que la fracción de votos >=
`PERSONAL_UNIFORME_UMBRAL` llegue a `PERSONAL_UNIFORME_MAYORIA`).

En «sombra» NO toca el seguimiento: ni `es_personal` ni ninguna métrica.
Cada 60 s deja una línea `UNIFORME [...]` en el log con lo cerrado y lo que
tarda el modelo.

## Modo «activo» (16-oct-2026, lo pidió el usuario para recoger datos)

Lo mismo, y además, en cuanto un seguimiento cumple la regla (al menos
`VOTOS_MIN` votos y mayoría), se le pone `es_personal = True`, la MISMA marca
que el personal dado de alta en `personal_galeria.py`: sus entradas y salidas
del área, su ficha, su calor y sus visitas y agarres en los anaqueles dejan
de contar, y tras sus agarres no se vigila el stock. La marca no se quita en
lo que dure el seguimiento. Deja «UNIFORME [...] PERSONAL ...» en el log en
el momento de marcar, para vigilarlo en vivo.

Límites conocidos: lo de los primeros ~1,5 s (antes del tercer voto) ya
contó; si el número de seguimiento cambia de persona (11-15 % de las veces),
la marca pasa con él.

## El modelo

`models/personal/uniforme_resnet18.pt`: torchvision resnet18 con
`fc = Linear(512, 1)`. Entrada: la caja de la persona a 128x256 (ancho x
alto), RGB y normalización de ImageNet, EXACTAMENTE como al entrenar. Está
entrenado con fotos de clientes: vive solo en el PC de la tienda y nunca va
a git (`server/models/`). Sin el archivo, el módulo se apaga solo.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_RAIZ = Path(__file__).resolve().parents[3]            # server/
_ANCHO, _ALTO = 128, 256
_MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
_DESV = np.array([0.229, 0.224, 0.225], np.float32)
#: Segundos sin ver un seguimiento para darlo por terminado y apuntarlo.
_FIN_S = 2.0
#: Cada cuánto deja la línea `UNIFORME [...]` en el log.
_RESUMEN_S = 60.0
MODOS = ('apagado', 'sombra', 'activo')
#: Votos del principio que dicen de quién es la ficha (`ficha_protegida`).
PRIMEROS_N = 2
#: Tipo de personal del modelo de 4 salidas (19-oct-2026) -> (rol, nombre).
#: Solo el de camiseta («pasillero») surte; el vigilante y mantenimiento
#: únicamente dejan de contar (lo pidió el usuario).
TIPOS = {1: ('pasillero', 'uniforme Santa Rosa'),
         2: ('vigilante', 'vigilante'),
         3: ('mantenimiento', 'mantenimiento')}


def _cfg(nombre: str, defecto: Any) -> Any:
    try:
        from .analytics.config import AnalyticsConfig
        return getattr(AnalyticsConfig, nombre, defecto)
    except Exception:                                       # noqa: BLE001
        return defecto


def modo() -> str:
    """`ELDE_UNIFORME` manda sobre `PERSONAL_UNIFORME_MODO`."""
    v = (os.getenv('ELDE_UNIFORME') or '').strip().lower()
    if v in MODOS:
        return v
    v = str(_cfg('PERSONAL_UNIFORME_MODO', 'apagado')).strip().lower()
    return v if v in MODOS else 'apagado'


class _Modelo:
    """La ResNet-18 del uniforme, una por proceso. El candado serializa las
    llamadas de las cámaras (la GPU igual lo haría)."""

    def __init__(self, ruta: Path) -> None:
        import torch
        import torchvision
        self._torch = torch
        self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        estado = torch.load(str(ruta), map_location='cpu')
        #: 1 salida = «uniforme sí/no» (hasta el v7b); 4 = el TIPO
        #: (sin uniforme, camiseta, vigilante, mantenimiento; 19-oct-2026).
        self.salidas = int(estado['fc.weight'].shape[0])
        net = torchvision.models.resnet18()
        net.fc = torch.nn.Linear(512, self.salidas)
        net.load_state_dict(estado)
        self._net = net.to(self.dispositivo).eval()
        self._candado = threading.Lock()

    def __call__(self, recortes: Sequence[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.resize(r, (_ANCHO, _ALTO)) for r in recortes])
        x = (x[..., ::-1].astype(np.float32) / 255.0 - _MEDIA) / _DESV
        t = self._torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        with self._candado, self._torch.no_grad():
            z = self._net(t.to(self.dispositivo))
            if self.salidas == 1:
                return self._torch.sigmoid(z.squeeze(1)).float().cpu().numpy()
            return self._torch.softmax(z, 1).float().cpu().numpy()


_lock = threading.Lock()
#: None = sin intentar; False = no disponible; _Modelo = cargado.
_modelo: Any = None


def _obtener_modelo() -> Optional[Callable[[Sequence[np.ndarray]], np.ndarray]]:
    global _modelo
    if _modelo is None:
        with _lock:
            if _modelo is None:
                ruta = _RAIZ / str(_cfg('PERSONAL_UNIFORME_MODELO', ''))
                if not ruta.is_file():
                    logger.info("uniforme: sin modelo en %s; no se mira el uniforme", ruta)
                    _modelo = False
                else:
                    try:
                        _modelo = _Modelo(ruta)
                        logger.info("uniforme: modelo %s cargado en %s (modo %s)",
                                    ruta.name, _modelo.dispositivo, modo())
                    except Exception as exc:                # noqa: BLE001
                        logger.error("uniforme: no se pudo cargar %s: %s", ruta, exc)
                        _modelo = False
    return _modelo or None


def _numero(tid: Any) -> int:
    try:
        return int(tid)
    except (TypeError, ValueError):
        return abs(hash(tid))


class VotoUniforme:
    """Votos por seguimiento de UNA cámara y su registro en sombra."""

    def __init__(self, camera_id: Any, modelo: Optional[Callable] = None,
                 carpeta: Optional[Path] = None) -> None:
        self.camera_id = str(camera_id)
        self._modelo = modelo
        if carpeta is None:
            from .anaqueles_registro import _carpeta_nombre
            carpeta = _RAIZ / 'output' / 'uniforme_sombra' / _carpeta_nombre(camera_id)
        self._carpeta = Path(carpeta)
        self._votos: Dict[Any, Dict[str, Any]] = {}
        #: ficha -> (t, sus 2 primeros votos): los de la persona de la FOTO
        #: (la ficha se abre al empezar el seguimiento). Sobrevive al cierre
        #: del seguimiento: la ficha puede apartarse después por su identidad
        #: (`ficha_protegida`, en sombra desde el 23-oct-2026).
        self._primeros: Dict[str, Any] = {}
        self._desde = time.time()
        self._cerrados = 0
        self._personal = 0
        self._ms: List[float] = []

    def observar(self, frame: np.ndarray, tracks: Dict[Any, Dict[str, Any]],
                 frame_counter: int, cada: int, t: Optional[float] = None,
                 fichas: Optional[Dict[Any, str]] = None) -> List[Any]:
        """Un frame. Devuelve los seguimientos marcados como personal EN ESTE
        frame (solo en modo «activo»; en «sombra» no toca `tracks`)."""
        t = time.time() if t is None else float(t)
        modelo = self._modelo or _obtener_modelo()
        if modelo is None or frame is None:
            return []
        alto_min = int(_cfg('PERSONAL_UNIFORME_ALTO_MIN', 80))
        h, w = frame.shape[:2]
        tocan, recortes = [], []
        for tid, tr in (tracks or {}).items():
            b = tr.get('box')
            if b is None or tr.get('last_seen', frame_counter) != frame_counter:
                continue                        # solo quien se ve en ESTA imagen
            v = self._votos.setdefault(tid, {'p': [], 'tipo': [], 't0': t, 't1': t, 'pid': None, 'ficha': None,
                                             'marcado': None})
            v['t1'] = t
            v['pid'] = tr.get('persistent_id') or v['pid']
            if fichas:
                v['ficha'] = fichas.get(tid) or v['ficha']
            if (frame_counter + _numero(tid)) % max(1, int(cada)):
                continue
            x1, y1, x2, y2 = (int(round(float(c))) for c in b[:4])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if (y2 - y1) < alto_min or (x2 - x1) < 8:
                continue
            tocan.append(tid)
            recortes.append(frame[y1:y2, x1:x2])
        if recortes:
            t0 = time.perf_counter()
            salida = np.asarray(modelo(recortes), dtype=np.float64)
            self._ms.append((time.perf_counter() - t0) * 1000.0)
            if salida.ndim == 2 and salida.shape[1] == len(TIPOS) + 1:
                # Modelo de TIPOS: «es personal» = 1 - P(sin uniforme), y el
                # tipo más probable entre los de personal.
                probs = 1.0 - salida[:, 0]
                tipos = np.argmax(salida[:, 1:], axis=1) + 1
            else:
                probs = salida.reshape(-1)
                tipos = np.ones(len(probs), dtype=int)
            for tid, p, tipo in zip(tocan, probs, tipos):
                self._votos[tid]['p'].append(float(p))
                self._votos[tid]['tipo'].append(int(tipo))
        for tid in tocan:
            v = self._votos[tid]
            if v['ficha']:
                prev = self._primeros.get(v['ficha'])
                lst = prev[1] if prev else list(v['p'][:PRIMEROS_N])
                while len(lst) < PRIMEROS_N and len(v['p']) > len(lst):
                    lst.append(v['p'][len(lst)])
                self._primeros[v['ficha']] = (t, lst)
        if len(self._primeros) > 5000:              # no crecer sin fin
            for k in [k for k, (tt, _l) in self._primeros.items() if t - tt > 7200]:
                self._primeros.pop(k, None)
        marcados = self._marcar(tracks, tocan, t) if modo() == 'activo' else []
        self._cerrar_terminados(t)
        self._resumen(t)
        return marcados

    def _marcar(self, tracks: Dict[Any, Dict[str, Any]], tocan: Sequence[Any],
                t: float) -> List[Any]:
        """Modo «activo»: la marca de personal del pipeline, en cuanto el
        seguimiento cumple la regla. No se quita mientras dure."""
        marcados = []
        for tid in tocan:
            tr = tracks.get(tid)
            v = self._votos[tid]
            if tr is None or tr.get('es_personal') or not self.decision(v['p']):
                continue
            rol, nombre = TIPOS[self.tipo(v)]
            tr['es_personal'] = True
            tr['personal_rol'] = rol
            tr['personal_nombre'] = nombre
            tr['personal_via'] = 'modelo_uniforme'
            v['marcado'] = t
            marcados.append(tid)
            logger.info("UNIFORME [%s] PERSONAL (%s): seguimiento %s (%s) tras %d votos, %.0f %% a favor",
                        self.camera_id, nombre, tid, tr.get('persistent_id'), len(v['p']),
                        100.0 * float(np.mean(np.asarray(v['p']) >= 0.5)))
        return marcados

    @staticmethod
    def tipo(v: Dict[str, Any]) -> int:
        """El tipo que más votó entre los votos de «es personal» (1 si no
        hay: el modelo viejo solo conoce la camiseta)."""
        umbral = float(_cfg('PERSONAL_UNIFORME_UMBRAL', 0.5))
        votos = [tp for p, tp in zip(v['p'], v.get('tipo') or []) if p >= umbral]
        return max(set(votos), key=votos.count) if votos else 1

    def decision(self, probs: Sequence[float]) -> bool:
        """Mayoría de votos (16-oct-2026): con «media >= 0,9» se escapaban 7
        de 12 empleados que trabajaban largo rato (ver config.py)."""
        p = np.asarray(probs, dtype=np.float64)
        return (len(p) >= int(_cfg('PERSONAL_UNIFORME_VOTOS_MIN', 3))
                and float(np.mean(p >= float(_cfg('PERSONAL_UNIFORME_UMBRAL', 0.5))))
                >= float(_cfg('PERSONAL_UNIFORME_MAYORIA', 0.5)))

    def _cerrar_terminados(self, t: float, todos: bool = False) -> None:
        for tid in [k for k, v in self._votos.items() if todos or (t - v['t1']) > _FIN_S]:
            v = self._votos.pop(tid)
            if not v['p']:
                continue
            p = v['p']
            personal = self.decision(p)
            self._cerrados += 1
            self._personal += int(personal)
            fila = {
                'inicio': time.strftime('%H:%M:%S', time.localtime(v['t0'])),
                'fin': time.strftime('%H:%M:%S', time.localtime(v['t1'])),
                'tid': _numero(tid), 'pid': v['pid'], 'ficha': v['ficha'],
                'votos': len(p), 'media': round(float(np.mean(p)), 3),
                'max': round(float(np.max(p)), 3),
                'frac_05': round(float(np.mean(np.asarray(p) >= 0.5)), 3),
                'personal': bool(personal),
                'tipo': TIPOS[self.tipo(v)][1] if personal else None,
                # Cada voto, en orden (19-oct-2026): con la serie se prueba
                # fuera de línea otra regla de votos (4 o 5, otra mayoría)
                # sobre lo que pasó en vivo. Solo resumida no se podía.
                'serie': [round(float(x), 2) for x in p],
                # Modo «activo»: hora en que se marcó en vivo (None = no).
                'marcado': (time.strftime('%H:%M:%S', time.localtime(v['marcado']))
                            if v.get('marcado') else None),
            }
            try:
                self._carpeta.mkdir(parents=True, exist_ok=True)
                ruta = self._carpeta / (time.strftime('%Y-%m-%d', time.localtime(v['t1'])) + '.jsonl')
                with open(ruta, 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps(fila, ensure_ascii=False) + '\n')
            except OSError as exc:
                logger.warning("uniforme[%s]: no se pudo apuntar: %s", self.camera_id, exc)

    def _resumen(self, t: float) -> None:
        if (t - self._desde) < _RESUMEN_S:
            return
        ms = sorted(self._ms)
        logger.info("UNIFORME [%s] %s: %d seguimientos cerrados, %d personal; "
                    "modelo %s ms por llamada (mediana, %d llamadas) en %.0f s",
                    self.camera_id, modo(), self._cerrados, self._personal,
                    f'{ms[len(ms) // 2]:.1f}' if ms else '-', len(ms), t - self._desde)
        self._desde, self._cerrados, self._personal, self._ms = t, 0, 0, []


_por_camara: Dict[str, VotoUniforme] = {}
_lock_camaras = threading.Lock()


def ficha_protegida(camera_id: Any, stem: str, tid: Any = None, via: str = '',
                    motivo: str = '') -> bool:
    """EN SOMBRA (23-oct-2026): ¿se dejaría de apartar esta ficha porque los
    2 primeros votos de SU persona dicen «sin uniforme»? Solo lo apunta en
    `uniforme_sombra/<cámara>/fichas_protegidas_<fecha>.jsonl`; devuelve la
    respuesta pero quien llama no la usa todavía.

    Por qué: en la tanda `uniforme_vivo` (22-23 oct) 25 de 122 fichas
    apartadas eran clientes: 9 se apartaron al marcar a un empleado con la
    identidad del cliente y ~6 porque el número saltó del cliente a un
    empleado. Con esta regla, sobre esos mismos datos: 15 clientes salvados y
    8 empleados que se quedarían como cliente."""
    voto = _por_camara.get(str(camera_id))
    if voto is None:                 # sin votos de esa cámara no hay qué mirar
        return False
    prim = (voto._primeros.get(stem) or (None, []))[1]
    protege = len(prim) >= PRIMEROS_N and all(p < 0.5 for p in prim[:PRIMEROS_N])
    try:
        voto._carpeta.mkdir(parents=True, exist_ok=True)
        with open(voto._carpeta / f"fichas_protegidas_{time.strftime('%Y-%m-%d')}.jsonl", 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'hora': time.strftime('%H:%M:%S'), 'ficha': stem, 'tid': str(tid),
                                 'via': via, 'motivo': motivo,
                                 'primeros': [round(p, 2) for p in prim], 'protegeria': protege},
                                ensure_ascii=False) + '\n')
    except OSError as exc:
        logger.debug("uniforme[%s]: ficha protegida sin apuntar: %s", camera_id, exc)
    return protege


def observar(camera_id: Any, frame: np.ndarray, tracks: Dict[Any, Dict[str, Any]],
             frame_counter: int, cada: int, fichas: Optional[Dict[Any, str]] = None) -> List[Any]:
    """Enganche de `tiendas.py`, después de resolver identidades. Devuelve los
    seguimientos marcados como personal en este frame. Nunca lanza."""
    if modo() == 'apagado':
        return []
    try:
        cam = str(camera_id)
        voto = _por_camara.get(cam)
        if voto is None:
            with _lock_camaras:
                voto = _por_camara.setdefault(cam, VotoUniforme(camera_id))
        return voto.observar(frame, tracks, frame_counter, cada, fichas=fichas)
    except Exception as exc:                                # noqa: BLE001
        logger.warning("uniforme[%s]: fallo al observar: %s", camera_id, exc)
        return []
