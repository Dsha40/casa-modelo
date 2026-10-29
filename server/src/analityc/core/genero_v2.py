"""Género con nuestro modelo v2 sobre la foto de la ficha (27-oct-2026).
ENCENDIDO por defecto; lo pidió el usuario tras dos exámenes limpios.

## Por qué

El clasificador B, cuando duda, tira a «Hombre» y deja sin decidir a ~40 %
de la gente. Medido con etiquetas a mano en dos días que el v2 no vio:

| Examen | Sistema solo | Sistema + v2 (2ª opinión + indecisas) |
|---|---|---|
| vivo 23-oct, 229 fichas | 59 % decididas, 93,4 % | 2ª opinión sola: 96,3 % |
| vídeo 26-oct, 524 fichas | 60 %, 93,4 %; mujeres bien 116 | 87 %, 93,4 %; mujeres bien 184 |

El modelo (ResNet-18 sobre la foto de la persona a 128×256) está FUERA de
git, en `output/etiquetas/genero_resnet18_v2.pt`: se entrenó con fotos de
clientes. Si no está, esto no hace nada.

## Qué hace

Cuando una ficha de cliente lleva `ESPERA_S` sin cambiar (ya pasó la espera
de 5 min a la identidad o el género tardíos):

- el sistema dijo «Hombre» y el v2 dice «Mujer» con >= `U_SEGUNDA`: Mujer;
- el sistema no decidió y el v2 está seguro (>= `U_INDECISA` hacia un lado):
  ese género.

La ficha cambiada lleva `origen_demografia: modelo_genero_v2` y el género
anterior en `genero_sistema`; TODAS las revisadas llevan `genero_v2` (la
probabilidad de Mujer), así no se repiten tras un reinicio. Se anota también
en la base (clasificaciones y, con identidad, la visita), como cuando
converge la demografía.

Un hilo aparte busca las fichas y pasa el modelo (listar la carpeta cuesta
~46 ms con 19.000 fichas); los cambios en disco y en la base los escribe el
hilo de las imágenes en `aplicar`, el mismo que mueve las fichas del
personal, para no pisarse.

## Interruptor

`ELDE_GENERO_V2=0` (o `AnalyticsConfig.GENERO_V2 = False`) lo apaga.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_RAIZ = Path(__file__).resolve().parents[3]
MODELO = _RAIZ / 'output' / 'etiquetas' / 'genero_resnet18_v2.pt'
U_SEGUNDA = 0.90
U_INDECISA = 0.95
ESPERA_S = 360.0          # la espera de la ficha a lo tardío es de 5 min
CADA_S = 30.0
ORIGEN = 'modelo_genero_v2'
_M = np.array([0.485, 0.456, 0.406], np.float32)
_D = np.array([0.229, 0.224, 0.225], np.float32)


def activo() -> bool:
    v = os.environ.get('ELDE_GENERO_V2')
    if v is not None:
        return v.strip().lower() not in ('0', 'false', 'no', 'off', '')
    try:
        from .analytics.config import AnalyticsConfig
        return bool(getattr(AnalyticsConfig, 'GENERO_V2', True))
    except Exception:                                       # noqa: BLE001
        return True


def decidir(genero: Optional[str], p_mujer: float) -> Optional[str]:
    """El género que debe quedar en la ficha, o None si no cambia."""
    if genero == 'Hombre' and p_mujer >= U_SEGUNDA:
        return 'Mujer'
    if genero in (None, '', 'Desconocido'):
        if p_mujer >= U_INDECISA:
            return 'Mujer'
        if p_mujer <= 1.0 - U_INDECISA:
            return 'Hombre'
    return None


class _Modelo:
    def __init__(self, ruta: Path) -> None:
        import torch
        import torchvision
        self._torch = torch
        self.disp = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = torchvision.models.resnet18()
        net.fc = torch.nn.Linear(512, 1)
        net.load_state_dict(torch.load(str(ruta), map_location='cpu'))
        self.net = net.eval().to(self.disp)

    def p_mujer(self, img: np.ndarray) -> float:
        import cv2
        x = ((cv2.resize(img, (128, 256))[..., ::-1].astype(np.float32) / 255 - _M) / _D).transpose(2, 0, 1)
        with self._torch.no_grad():
            t = self._torch.from_numpy(np.ascontiguousarray(x))[None].to(self.disp)
            return float(self._torch.sigmoid(self.net(t))[0, 0])


class Revisor:
    """Busca fichas por revisar y les pasa el modelo, en su propio hilo."""

    def __init__(self, carpeta: Path, modelo: Any = None, espera_s: float = ESPERA_S) -> None:
        self.carpeta = Path(carpeta)
        self.espera_s = float(espera_s)
        self._modelo = modelo
        self._vistas: set = set()
        self.resultados: 'queue.Queue' = queue.Queue()

    def buscar(self, ahora: Optional[float] = None) -> int:
        """Una pasada: encola (stem, p_mujer) de las fichas listas."""
        import cv2
        ahora = time.time() if ahora is None else ahora
        dias = {time.strftime('%Y%m%d', time.localtime(ahora - h * 3600)) for h in (0, 24)}
        n = 0
        for e in os.scandir(self.carpeta):
            nom = e.name
            if (not nom.endswith('.json') or nom.endswith('_cara.json') or nom[:8] not in dias
                    or nom in self._vistas):
                continue
            try:
                if ahora - e.stat().st_mtime < self.espera_s:
                    continue
                meta = json.loads(Path(e.path).read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            self._vistas.add(nom)
            if 'genero_v2' in meta:
                continue
            img = cv2.imread(e.path[:-5] + '.jpg')
            if img is None or img.size == 0:
                continue
            if self._modelo is None:
                self._modelo = _Modelo(MODELO)
            self.resultados.put((nom[:-5], self._modelo.p_mujer(img)))
            n += 1
        return n

    def bucle(self) -> None:
        while True:
            try:
                self.buscar()
            except Exception as exc:                        # noqa: BLE001
                logger.warning('genero_v2: %s', exc)
            time.sleep(CADA_S)


def aplicar_uno(carpeta: Path, stem: str, p: float, almacen: Any = None) -> Optional[str]:
    """Escribe en la ficha la revisión del v2; devuelve el género nuevo si
    cambió. La ficha que ya no está (personal apartado, duplicado) se salta."""
    jp, jpg = Path(carpeta) / f'{stem}.json', Path(carpeta) / f'{stem}.jpg'
    try:
        meta = json.loads(jp.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if 'genero_v2' in meta or not jpg.exists():
        return None
    antes = meta.get('gender')
    nuevo = decidir(antes, p)
    meta['genero_v2'] = round(float(p), 3)
    if nuevo:
        meta['genero_sistema'] = antes
        meta['gender'] = nuevo
        meta['conf_genero'] = round(float(p if nuevo == 'Mujer' else 1.0 - p), 3)
        meta['origen_demografia'] = ORIGEN
        meta.pop('motivo_sin_demografia', None)
        if not meta.get('age_range'):
            meta['age_range'] = 'Desconocido'
    tmp = jp.with_name(jp.name + '.tmp')
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, jp)
    if nuevo and almacen is not None:
        cam, puid = meta.get('device_id') or '', meta.get('person_uuid')
        edad = meta.get('age_range')
        almacen.anotar_clasificacion(cam, stem, nuevo, edad, person_uuid=puid,
                                     solo_cuerpo=bool(meta.get('solo_cuerpo')))
        if puid:
            almacen.anotar_demografia(cam, puid, nuevo, edad)
    return nuevo


_revisor: Optional[Revisor] = None
_lock = threading.Lock()
_avisado = False


def aplicar(proc: Any = None) -> None:
    """Desde el hilo de las imágenes, en cada imagen. Nunca lanza."""
    global _revisor, _avisado
    try:
        if not activo():
            return
        if _revisor is None:
            with _lock:
                if _revisor is None:
                    if not MODELO.exists():
                        if not _avisado:
                            logger.warning('genero_v2: no está el modelo %s; apagado', MODELO)
                            _avisado = True
                        return
                    from .analytics.config import AnalyticsConfig
                    _revisor = Revisor(Path(AnalyticsConfig.CAPTURE_DIR) / 'persons')
                    threading.Thread(target=_revisor.bucle, name='genero_v2', daemon=True).start()
                    logger.info('genero_v2: ACTIVO (2ª opinión >= %.2f, indecisas >= %.2f)',
                                U_SEGUNDA, U_INDECISA)
        from . import tienda_almacen
        for _ in range(20):
            try:
                stem, p = _revisor.resultados.get_nowait()
            except queue.Empty:
                break
            nuevo = aplicar_uno(_revisor.carpeta, stem, p, tienda_almacen.get_almacen())
            if nuevo:
                logger.info('GENERO V2 %s -> %s (p_mujer %.2f)', stem, nuevo, p)
    except Exception as exc:                                # noqa: BLE001
        logger.debug('genero_v2: %s', exc)
