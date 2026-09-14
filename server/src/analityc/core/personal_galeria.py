"""src/analityc/core/personal_galeria.py — personal de tienda enrolado con FOTOS.

## Qué es

La lista del personal del local (seguridad y pasilleros) con su foto, para
que el pipeline los reconozca por rostro y:

  1. los EXCLUYA de todas las métricas de marketing (visitas/agarres de
     anaqueles, heatmaps, demografía, visitantes únicos) — el marketing es
     solo de CLIENTES;
  2. emita las alertas operativas «Seguridad {nombre} en {cámara}» y
     «Empleado {nombre} surte {área}».

## Por qué NO reutiliza `stand_personal.GaleriaPersonal`

Esa galería vive en memoria, anclada a `persistent_id`s del pipeline y a la
BD del stand: enrola señalando a alguien que YA salió en cámara. Aquí el
enrolamiento es al revés — el encargado sube una FOTO desde el dashboard o
el cliente ANTES de que la persona aparezca — así que la identidad nace del
embedding de la foto, se persiste en disco y sobrevive a reinicios.

## Cómo identifica

La foto subida pasa por YuNet (rostro + 5 landmarks) → alineación al patrón
ArcFace 112x112 → `w600k_r50` (la MISMA sesión ONNX del Re-ID si ya existe,
o una propia perezosa). En vivo, el pipeline ya calcula embeddings por
`persistent_id` (FaceReidentifier, fuera del swap por cámara): el
`IdentificadorPersonal` compara esos embeddings contra las fichas con el
umbral 0.75 — el mismo criterio estricto que el anti-chef del stand
(`personal.umbral_reid`), a propósito más duro que el 0.35 del Re-ID:
confundir a un cliente con un empleado lo BORRA del marketing, así que el
falso positivo caro es ese.

## Persistencia

`server/output/personal_tienda/registro.json` (atómico tmp+replace) con los
embeddings dentro (pocas fichas × pocas fotos: cabe de sobra) y las fotos en
`fotos/`. La API y el pipeline comparten proceso y singleton; el mtime del
registro se vigila igualmente por si algo externo lo edita.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ROL_SEGURIDAD = 'seguridad'
ROL_PASILLERO = 'pasillero'
ROLES = (ROL_SEGURIDAD, ROL_PASILLERO)

#: Tipos de ficha (14-ago, tarde): ROSTRO identifica a LA PERSONA (ArcFace);
#: UNIFORME identifica LA VESTIMENTA (OSNet) — el bordado y los colores del
#: uniforme se ven de espaldas, que es como más trabaja un pasillero. Una
#: ficha de uniforme no dice QUIÉN es, dice QUÉ es (seguridad/pasillero), y
#: con eso basta para excluirlo del marketing y alertar.
TIPO_ROSTRO = 'rostro'
TIPO_UNIFORME = 'uniforme'
TIPOS = (TIPO_ROSTRO, TIPO_UNIFORME)

#: Umbral coseno para «este persistent_id ES este empleado». El mismo 0.75
#: del anti-chef del stand: estricto a propósito (ver docstring del módulo).
UMBRAL_PERSONAL = 0.75

#: Umbral coseno OSNet para «esta persona VISTE este uniforme». El Re-ID
#: corporal con la MISMA persona y ropa usa 0.90; el mismo uniforme sobre
#: CUERPOS DISTINTOS da menos, y un cliente con ropa parecida menos aún.
#: 0.82 + racha de 2 coincidencias (en tiendas.py) es el equilibrio: el
#: falso positivo caro sigue siendo borrar a un cliente del marketing.
UMBRAL_UNIFORME = 0.82

#: Confianza mínima del rostro de la FOTO subida. Más alta que la del
#: pipeline (0.55): una foto de enrolamiento borrosa envenena la galería.
_CONF_ROSTRO_FOTO = 0.7

#: Plantilla canónica ArcFace 112x112 (ojo-izq, ojo-der, nariz, boca-izq,
#: boca-der en coordenadas de IMAGEN: izquierda = x menor).
_PLANTILLA_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

_SEGURO = re.compile(r'[^A-Za-z0-9_-]+')


def raiz_personal() -> Path:
    return Path(__file__).resolve().parents[3] / 'output' / 'personal_tienda'


class GaleriaPersonalTienda:
    """Fichas del personal (nombre, rol, fotos, embeddings), persistidas."""

    def __init__(self, raiz: Optional[Path] = None) -> None:
        self._raiz = Path(raiz) if raiz else raiz_personal()
        self._ruta = self._raiz / 'registro.json'
        self._fotos = self._raiz / 'fotos'
        self._lock = threading.RLock()
        #: id -> {'nombre','rol','activo','creado','fotos':[str],
        #:        'embeddings':[[512 floats]]}
        self._fichas: Dict[str, Dict[str, Any]] = {}
        #: Sube con cada cambio: los cachés de identificación se invalidan.
        self.version = 0
        self._mtime = 0.0
        # Modelos perezosos para «foto -> embedding» (solo el enrolamiento
        # los necesita; el pipeline en vivo usa los embeddings del Re-ID).
        # Lock PROPIO de modelos: serializa la carga (dos POST simultáneos
        # veían el estado a medias) y el par setInputSize+detect del YuNet
        # CPU, que NO es thread-safe — dos enrolamientos a la vez daban 500
        # o, peor, landmarks corruptos → embedding basura persistido.
        self._lock_modelos = threading.Lock()
        self._yunet = None
        self._sesion = None
        #: BodyReidentifier SOLO como calculadora de embeddings OSNet para
        #: las fotos de UNIFORME (su galería efímera no se usa aquí).
        self._cuerpo = None
        #: ts del último intento FALLIDO de carga: se reintenta pasado un
        #: rato (p. ej. el pipeline arrancó después y ya hay sesión que
        #: reutilizar) en vez de quedarse en 400 permanente.
        self._modelos_fallo_ts = 0.0
        self._ultimo_stat = 0.0
        self._cargar()

    # ── Enrolamiento ────────────────────────────────────────────────────
    def enrolar(self, nombre: str, rol: str, foto_bgr: np.ndarray,
                tipo: str = TIPO_ROSTRO) -> Dict[str, Any]:
        """Alta de un empleado (rostro) o de un uniforme (vestimenta).

        Lanza ValueError con un mensaje PARA EL OPERADOR si la foto no
        sirve (sin rostro claro, varios rostros, sin modelos).
        """
        nombre = str(nombre or '').strip()[:80]
        rol = str(rol or '').strip().lower()
        tipo = str(tipo or TIPO_ROSTRO).strip().lower()
        if not nombre:
            raise ValueError('falta el nombre (del empleado o del uniforme)')
        if rol not in ROLES:
            raise ValueError(f"rol inválido {rol!r}: usa 'seguridad' o "
                             f"'pasillero'")
        if tipo not in TIPOS:
            raise ValueError(f"tipo inválido {tipo!r}: usa 'rostro' o "
                             f"'uniforme'")
        emb = (self._embedding_de_uniforme(foto_bgr)
               if tipo == TIPO_UNIFORME else
               self._embedding_de_foto(foto_bgr))
        with self._lock:
            pid = _uuid.uuid4().hex[:12]
            archivo = self._guardar_foto(pid, 0, foto_bgr)
            self._fichas[pid] = {
                'nombre': nombre,
                'rol': rol,
                'tipo': tipo,
                'activo': True,
                'creado': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fotos': [archivo],
                'embeddings': [emb.tolist()],
            }
            self._persistir()
            logger.info("personal: enrolado %s (%s, %s) id=%s",
                        nombre, rol, tipo, pid)
            return self._publica(pid)

    def agregar_foto(self, pid: str, foto_bgr: np.ndarray) -> Dict[str, Any]:
        """Otra foto de la MISMA ficha (más ángulos = mejor reconocimiento;
        en uniformes, frente Y espalda — el bordado no siempre va delante)."""
        with self._lock:
            tipo = self._ficha_o_error(pid).get('tipo') or TIPO_ROSTRO
        emb = (self._embedding_de_uniforme(foto_bgr)
               if tipo == TIPO_UNIFORME else
               self._embedding_de_foto(foto_bgr))
        with self._lock:
            ficha = self._ficha_o_error(pid)
            n = len(ficha['fotos'])
            ficha['fotos'].append(self._guardar_foto(pid, n, foto_bgr))
            ficha['embeddings'].append(emb.tolist())
            self._persistir()
            return self._publica(pid)

    def poner(self, pid: str, *, activo: Optional[bool] = None,
              nombre: Optional[str] = None,
              rol: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            ficha = self._ficha_o_error(pid)
            if activo is not None:
                ficha['activo'] = bool(activo)
            if nombre is not None and str(nombre).strip():
                ficha['nombre'] = str(nombre).strip()[:80]
            if rol is not None:
                if rol not in ROLES:
                    raise ValueError(f'rol inválido {rol!r}')
                ficha['rol'] = rol
            self._persistir()
            return self._publica(pid)

    def borrar(self, pid: str) -> None:
        with self._lock:
            ficha = self._ficha_o_error(pid)
            for archivo in ficha.get('fotos') or []:
                try:
                    (self._fotos / archivo).unlink(missing_ok=True)
                except Exception:                           # noqa: BLE001
                    pass
            del self._fichas[pid]
            self._persistir()
            logger.info("personal: borrado id=%s", pid)

    # ── Consulta ────────────────────────────────────────────────────────
    def listar(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._publica(pid) for pid in sorted(
                self._fichas,
                key=lambda p: self._fichas[p].get('nombre', ''))]

    def foto_ruta(self, pid: str, n: int) -> Optional[Path]:
        with self._lock:
            ficha = self._fichas.get(pid)
            if not ficha:
                return None
            fotos = ficha.get('fotos') or []
            if not (0 <= n < len(fotos)):
                return None
            ruta = self._fotos / fotos[n]
            return ruta if ruta.is_file() else None

    def identificar(self, embedding: np.ndarray,
                    umbral: float = UMBRAL_PERSONAL
                    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """El empleado ACTIVO cuyo rostro es este embedding, o None.

        Devuelve `(id, ficha_publica, score)`. Los embeddings están
        L2-normalizados: coseno = producto punto.
        """
        try:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norma = float(np.linalg.norm(v))
            if norma < 1e-6:
                return None
            v = v / norma
        except Exception:                                   # noqa: BLE001
            return None
        return self._mejor_match(v, umbral, TIPO_ROSTRO)

    def identificar_cuerpo(self, embedding: np.ndarray,
                           umbral: float = UMBRAL_UNIFORME
                           ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """El UNIFORME activo que viste esta persona, o None.

        `embedding` es el OSNet 512-d del recorte corporal (el mismo que el
        pipeline ya calcula para el Re-ID corporal)."""
        try:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norma = float(np.linalg.norm(v))
            if norma < 1e-6:
                return None
            v = v / norma
        except Exception:                                   # noqa: BLE001
            return None
        return self._mejor_match(v, umbral, TIPO_UNIFORME)

    def _mejor_match(self, v: np.ndarray, umbral: float, tipo: str
                     ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        mejor: Optional[Tuple[str, float]] = None
        with self._lock:
            for pid, ficha in self._fichas.items():
                if not ficha.get('activo'):
                    continue
                if (ficha.get('tipo') or TIPO_ROSTRO) != tipo:
                    continue
                for e in ficha.get('embeddings') or []:
                    try:
                        s = float(np.dot(v, np.asarray(e, dtype=np.float32)))
                    except Exception:                       # noqa: BLE001
                        continue
                    if s >= umbral and (mejor is None or s > mejor[1]):
                        mejor = (pid, s)
            if mejor is None:
                return None
            return mejor[0], self._publica(mejor[0]), mejor[1]

    def tiene_fichas(self) -> bool:
        with self._lock:
            return bool(self._fichas)

    def tiene_uniformes(self) -> bool:
        """Atajo del bucle de inferencia: sin uniformes activos, el chequeo
        corporal por track debe costar microsegundos."""
        with self._lock:
            return any(f.get('activo')
                       and (f.get('tipo') or TIPO_ROSTRO) == TIPO_UNIFORME
                       for f in self._fichas.values())

    def recargar_si_cambio(self) -> None:
        """Relee el registro si alguien lo tocó desde fuera del proceso.

        El stat va con throttle de 1 s: esto se consulta por track y por
        frame desde el bucle de inferencia, y un stat por consulta era pagar
        disco por vigilar un archivo que casi nunca cambia.
        """
        ahora = time.monotonic()
        if (ahora - self._ultimo_stat) < 1.0:
            return
        self._ultimo_stat = ahora
        try:
            mtime = self._ruta.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            with self._lock:
                self._cargar()

    # ── Foto → embedding ────────────────────────────────────────────────
    def _embedding_de_foto(self, foto_bgr: np.ndarray) -> np.ndarray:
        if foto_bgr is None or getattr(foto_bgr, 'size', 0) == 0:
            raise ValueError('la imagen llegó vacía o ilegible')
        h, w = foto_bgr.shape[:2]
        if min(h, w) < 80:
            raise ValueError('la foto es demasiado pequeña (mínimo 80 px '
                             'de lado)')
        self._asegurar_modelos()
        if self._yunet is None or self._sesion is None:
            raise ValueError('el servidor no tiene los modelos de rostro '
                             'cargables (YuNet/ArcFace); revisa models/')
        rostros = self._detectar(foto_bgr)
        if not rostros:
            raise ValueError('no se detectó ningún rostro claro en la foto: '
                             'usa una foto de frente, con buena luz')
        if len(rostros) > 1:
            raise ValueError(f'la foto tiene {len(rostros)} rostros: sube '
                             f'una foto donde salga SOLO el empleado')
        alineado = self._alinear(foto_bgr, rostros[0])
        emb = self._embedding(alineado)
        if emb is None:
            raise ValueError('no se pudo calcular la firma del rostro; '
                             'prueba con otra foto')
        return emb

    def _embedding_de_uniforme(self, foto_bgr: np.ndarray) -> np.ndarray:
        """Firma OSNet de una foto de UNIFORME (cuerpo entero, de cerca).

        Sin exigencia de rostro: el punto es que el bordado y los colores se
        reconocen también de espaldas. Recomendación al operador: enrolar
        frente Y espalda como dos fotos de la misma ficha.
        """
        if foto_bgr is None or getattr(foto_bgr, 'size', 0) == 0:
            raise ValueError('la imagen llegó vacía o ilegible')
        h, w = foto_bgr.shape[:2]
        if h < 120 or w < 60:
            raise ValueError('la foto del uniforme es demasiado pequeña: '
                             'usa una del cuerpo entero, de cerca')
        self._asegurar_osnet()
        if self._cuerpo is None or not self._cuerpo.is_available:
            raise ValueError('el servidor no tiene el modelo corporal '
                             '(OSNet) cargable; revisa models/')
        emb = self._cuerpo.compute_embedding(foto_bgr)
        if emb is None:
            raise ValueError('no se pudo calcular la firma del uniforme; '
                             'prueba con otra foto')
        return emb

    def _asegurar_osnet(self) -> None:
        """OSNet perezoso: la sesión del pipeline si existe (ORT thread-safe),
        propia si no. Mismo patrón de reintento que los modelos de rostro."""
        with self._lock_modelos:
            if self._cuerpo is not None and self._cuerpo.is_available:
                return
            if (time.monotonic() - self._modelos_fallo_ts) < 30.0:
                return
            sesion = None
            try:
                from .tiendas import sesiones_compartidas_globales
                sesion = (sesiones_compartidas_globales() or {}).get('osnet')
            except Exception:                               # noqa: BLE001
                pass
            if sesion is None:
                try:
                    import onnxruntime as ort

                    from .analytics.config import AnalyticsConfig
                    from .tiendas import _PROJECT_ROOT, _build_onnx_providers
                    ruta = os.path.join(_PROJECT_ROOT,
                                        AnalyticsConfig.BODY_REID_MODEL)
                    if os.path.exists(ruta):
                        sesion = ort.InferenceSession(
                            ruta, providers=_build_onnx_providers())
                        logger.info("personal: sesion OSNet propia creada")
                except Exception as exc:                    # noqa: BLE001
                    logger.error("personal: no se pudo cargar OSNet: %s", exc)
            if sesion is not None:
                try:
                    from .analytics.body_reidentifier import BodyReidentifier
                    self._cuerpo = BodyReidentifier(session=sesion)
                except Exception as exc:                    # noqa: BLE001
                    logger.error("personal: BodyReidentifier falló: %s", exc)
            if self._cuerpo is None or not self._cuerpo.is_available:
                self._modelos_fallo_ts = time.monotonic()

    def _asegurar_modelos(self) -> None:
        """Carga perezosa BAJO LOCK, con reintento.

        La sesión ArcFace (la cara en VRAM) se REUTILIZA del pipeline vivo
        si existe (`sesiones_compartidas_globales`, sesión ORT thread-safe);
        el YuNet es SIEMPRE propio (el cv2 de CPU no es thread-safe y el del
        pipeline puede estar en uso por las cámaras). Un fallo de carga NO
        es definitivo: se reintenta pasado un rato — típico caso «la API
        enroló antes de que arrancara el primer cliente y no había VRAM».
        """
        with self._lock_modelos:
            if self._yunet is not None and self._sesion is not None:
                return
            if (time.monotonic() - self._modelos_fallo_ts) < 30.0:
                return                       # fallo reciente: no martillear
            try:
                if self._sesion is None:
                    from .tiendas import sesiones_compartidas_globales
                    compartidas = sesiones_compartidas_globales() or {}
                    self._sesion = compartidas.get('embedding')
                if self._yunet is None:
                    from .tiendas import _create_yunet
                    self._yunet = _create_yunet('cpu')
                if self._sesion is None:
                    import onnxruntime as ort

                    from .tiendas import (_FACE_EMBEDDING_MODEL,
                                          _build_onnx_providers)
                    if os.path.exists(_FACE_EMBEDDING_MODEL):
                        self._sesion = ort.InferenceSession(
                            _FACE_EMBEDDING_MODEL,
                            providers=_build_onnx_providers())
                        logger.info("personal: sesion ArcFace propia creada "
                                    "(no habia pipeline vivo que reutilizar)")
            except Exception as exc:                        # noqa: BLE001
                logger.error("personal: no se pudieron cargar los modelos "
                             "de rostro: %s", exc)
            if self._yunet is None or self._sesion is None:
                self._modelos_fallo_ts = time.monotonic()

    def _detectar(self, img: np.ndarray) -> List[np.ndarray]:
        """Rostros con confianza suficiente: filas YuNet [x,y,w,h,5*(x,y),score]."""
        h, w = img.shape[:2]
        # YuNet rinde mal con imágenes enormes: se trabaja a lo sumo a 1280.
        escala = 1.0
        if max(h, w) > 1280:
            escala = 1280.0 / max(h, w)
            img = cv2.resize(img, (int(w * escala), int(h * escala)))
            h, w = img.shape[:2]
        try:
            # setInputSize+detect deben ir JUNTOS bajo el lock: el YuNet CPU
            # tiene estado y dos peticiones a la vez lo corrompen.
            with self._lock_modelos:
                self._yunet.setInputSize((w, h))
                _ok, dets = self._yunet.detect(img)
        except Exception as exc:                            # noqa: BLE001
            # cv2.error NO es ValueError: sin esto el operador veía un 500
            # pelado en vez de un mensaje accionable.
            raise ValueError('no se pudo analizar la foto; inténtalo de '
                             'nuevo (o usa otra foto)') from exc
        salida = []
        for fila in (dets if dets is not None else []):
            if float(fila[14]) < _CONF_ROSTRO_FOTO:
                continue
            fila = np.asarray(fila, dtype=np.float32).copy()
            if escala != 1.0:
                fila[:14] /= escala          # de vuelta a la foto original
            salida.append(fila)
        return salida

    @staticmethod
    def _alinear(img: np.ndarray, fila: np.ndarray) -> np.ndarray:
        """Alineación por los 5 landmarks al patrón ArcFace 112x112.

        Los pares ojo/boca se ordenan por X para no depender del orden
        izquierda/derecha del detector (que nombra por el sujeto, no por la
        imagen)."""
        pts = fila[4:14].reshape(5, 2).astype(np.float32)
        ojos = sorted(pts[0:2], key=lambda p: p[0])
        boca = sorted(pts[3:5], key=lambda p: p[0])
        origen = np.array([ojos[0], ojos[1], pts[2], boca[0], boca[1]],
                          dtype=np.float32)
        m, _inl = cv2.estimateAffinePartial2D(
            origen, _PLANTILLA_112, method=cv2.LMEDS)
        if m is None:
            # Sin transformación estable: recorte del bbox como último
            # recurso (peor alineación, pero embedding utilizable).
            x, y, w, h = [int(v) for v in fila[:4]]
            recorte = img[max(0, y):y + h, max(0, x):x + w]
            return cv2.resize(recorte, (112, 112),
                              interpolation=cv2.INTER_CUBIC)
        return cv2.warpAffine(img, m, (112, 112))

    def _embedding(self, cara_112: np.ndarray) -> Optional[np.ndarray]:
        """El mismo preproceso que FaceReidentifier.compute_embedding."""
        try:
            cara = cv2.resize(cara_112, (112, 112),
                              interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - 127.5) / 127.5
            entrada = np.transpose(rgb, (2, 0, 1))[np.newaxis]
            nombre = self._sesion.get_inputs()[0].name
            salida = self._sesion.run(None, {nombre: entrada})[0][0]
            v = np.asarray(salida, dtype=np.float32).reshape(-1)
            norma = float(np.linalg.norm(v))
            if norma < 1e-6:
                return None
            return v / norma
        except Exception as exc:                            # noqa: BLE001
            logger.error("personal: embedding falló: %s", exc)
            return None

    # ── Interno ─────────────────────────────────────────────────────────
    def _publica(self, pid: str) -> Dict[str, Any]:
        f = self._fichas[pid]
        return {'id': pid, 'nombre': f.get('nombre'), 'rol': f.get('rol'),
                'tipo': f.get('tipo') or TIPO_ROSTRO,
                'activo': bool(f.get('activo')),
                'creado': f.get('creado'),
                'n_fotos': len(f.get('fotos') or [])}

    def _ficha_o_error(self, pid: str) -> Dict[str, Any]:
        ficha = self._fichas.get(str(pid))
        if ficha is None:
            raise KeyError(f'no existe el empleado {pid!r}')
        return ficha

    def _guardar_foto(self, pid: str, n: int, foto_bgr: np.ndarray) -> str:
        self._fotos.mkdir(parents=True, exist_ok=True)
        nombre = f"{_SEGURO.sub('_', pid)}_{n}.jpg"
        ok, buf = cv2.imencode('.jpg', foto_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError('no se pudo codificar la foto')
        (self._fotos / nombre).write_bytes(buf.tobytes())
        return nombre

    def _persistir(self) -> None:
        self._raiz.mkdir(parents=True, exist_ok=True)
        datos = {'version': 1, 'fichas': self._fichas}
        tmp = self._ruta.with_suffix('.tmp')
        tmp.write_text(json.dumps(datos, ensure_ascii=False),
                       encoding='utf-8')
        os.replace(tmp, self._ruta)   # atómico
        self.version += 1
        try:
            self._mtime = self._ruta.stat().st_mtime
        except OSError:
            pass

    def _cargar(self) -> None:
        try:
            if not self._ruta.is_file():
                return
            datos = json.loads(self._ruta.read_text(encoding='utf-8'))
            self._fichas = dict(datos.get('fichas') or {})
            self._mtime = self._ruta.stat().st_mtime
            self.version += 1
            if self._fichas:
                logger.info("personal: %d fichas cargadas (%d activas)",
                            len(self._fichas),
                            sum(1 for f in self._fichas.values()
                                if f.get('activo')))
        except Exception as exc:                            # noqa: BLE001
            logger.error("personal: registro ilegible (%s); se arranca "
                         "vacío SIN pisar el archivo", exc)


class IdentificadorPersonal:
    """Caché «persistent_id → ficha de empleado o None» para el pipeline.

    Consulta los embeddings que el Re-ID ya calculó (fuera del swap por
    cámara) y los compara contra la galería. Nunca lanza: ante cualquier
    fallo responde None (la persona cuenta como cliente, que es el error
    barato — el caro es borrar a un cliente del marketing).
    """

    #: Un «no es personal» se reevalúa pronto (el Re-ID acumula mejores
    #: embeddings); un «sí es» dura más (solo lo tumba un cambio de galería).
    TTL_NO_S = 30.0
    TTL_SI_S = 300.0

    def __init__(self, galeria: Optional[GaleriaPersonalTienda] = None) -> None:
        self._galeria = galeria
        self._cache: Dict[str, Tuple[Optional[Dict[str, Any]], float]] = {}
        self._version_vista = -1

    @property
    def galeria(self) -> GaleriaPersonalTienda:
        if self._galeria is None:
            self._galeria = get_galeria()
        return self._galeria

    def ficha_de(self, pid: Any, reidentifier: Any = None
                 ) -> Optional[Dict[str, Any]]:
        """La ficha del empleado si `pid` es personal ACTIVO, o None."""
        try:
            pid = str(pid or '')
            if not pid or pid.startswith('prov:'):
                return None      # sin rostro no hay identificación posible
            gal = self.galeria
            gal.recargar_si_cambio()
            if gal.version != self._version_vista:
                self._cache.clear()
                self._version_vista = gal.version
            version_leida = gal.version
            # Atajo del caso común: SIN personal enrolado, esto corre por
            # track y por frame y debe costar microsegundos (después del
            # chequeo de versión, para que un enrolamiento en vivo entre).
            if not gal.tiene_fichas():
                return None
            ahora = time.time()
            en_cache = self._cache.get(pid)
            if en_cache is not None:
                ficha, ts = en_cache
                ttl = self.TTL_SI_S if ficha is not None else self.TTL_NO_S
                if (ahora - ts) < ttl:
                    return ficha
            ficha = self._resolver(pid, reidentifier)
            # Cachear SOLO si la galería no cambió mientras se resolvía: una
            # resolución en vuelo durante un borrado se colaría bajo la
            # versión nueva y sobreviviría hasta 300 s.
            if (gal.version == version_leida
                    and self._version_vista == version_leida):
                self._cache[pid] = (ficha, ahora)
                if len(self._cache) > 2000:
                    # Poda simple: los pid con offset por cámara son infinitos.
                    for clave in list(self._cache)[:1000]:
                        self._cache.pop(clave, None)
            return ficha
        except Exception:                                   # noqa: BLE001
            return None

    def ficha_por_cuerpo(self, embedding: Any) -> Optional[Dict[str, Any]]:
        """La ficha del UNIFORME que viste esta persona, o None.

        Sin caché por pid: quien llama (tiendas.py) guarda el veredicto en
        el propio track con una racha de 2 coincidencias — una coincidencia
        suelta puede ser un cliente con ropa parecida. Nunca lanza.
        """
        try:
            if embedding is None:
                return None
            gal = self.galeria
            gal.recargar_si_cambio()
            if not gal.tiene_uniformes():
                return None
            resultado = gal.identificar_cuerpo(embedding)
            if resultado is None:
                return None
            _pid, ficha, score = resultado
            return dict(ficha, score=round(score, 3), via='uniforme')
        except Exception:                                   # noqa: BLE001
            return None

    def _resolver(self, pid: str, reidentifier: Any
                  ) -> Optional[Dict[str, Any]]:
        if reidentifier is None:
            return None
        try:
            from .analytics.face_reidentifier import _get_embeddings
        except Exception:                                   # noqa: BLE001
            _get_embeddings = None
        lock = getattr(reidentifier, '_lock', None)
        db = getattr(reidentifier, '_db', None)
        if db is None:
            return None
        embeddings: List[np.ndarray] = []
        try:
            ctx = lock if lock is not None else threading.RLock()
            with ctx:
                rec = db.get(pid)
                if not rec:
                    return None
                if _get_embeddings is not None:
                    embeddings = list(_get_embeddings(rec) or [])
                else:
                    embeddings = list(rec.get('embeddings') or [])
        except Exception:                                   # noqa: BLE001
            return None
        for emb in embeddings:
            resultado = self.galeria.identificar(emb)
            if resultado is not None:
                _pid_ficha, ficha, score = resultado
                ficha = dict(ficha, score=round(score, 3))
                return ficha
        return None


# ── Singletons de proceso ────────────────────────────────────────────────
_galeria: Optional[GaleriaPersonalTienda] = None
_identificador: Optional[IdentificadorPersonal] = None
_singleton_lock = threading.Lock()


def get_galeria() -> GaleriaPersonalTienda:
    global _galeria
    if _galeria is None:
        with _singleton_lock:
            if _galeria is None:
                _galeria = GaleriaPersonalTienda()
    return _galeria


def get_identificador() -> IdentificadorPersonal:
    global _identificador
    if _identificador is None:
        with _singleton_lock:
            if _identificador is None:
                _identificador = IdentificadorPersonal()
    return _identificador
