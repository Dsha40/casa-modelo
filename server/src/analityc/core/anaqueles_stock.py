"""src/analityc/core/anaqueles_stock.py — aviso de «posible agotado» tras un agarre.

## Qué resuelve (5-oct-2026)

Un agarre dice «alguien cogió algo de esta balda», no «se acabó». Para
saberlo hay que MIRAR el sitio del producto cuando ya nadie lo tapa y
compararlo con cómo estaba surtido. Es lo que hace un reponedor: ve a alguien
coger un bote, espera a que se aparte y mira el hueco.

1. **Foto de referencia** («surtido»): el operador la pide con un botón del
   cliente (POST `/api/anaqueles/<cámara>/stock/referencia`). Se toma en el
   primer frame en que NADIE tapa el recuadro del producto. Se guarda en
   disco (`stock_ref/`) y sobrevive a los reinicios. NO se renueva sola: la
   foto del área de `fotos_ref/` se rehace cada 10 min y, con la balda ya
   vacía, compararía «vacío con vacío».
2. **Tras un agarre** en un área, se vigilan TODOS sus productos, no solo el
   que quedó bajo la muñeca: la muñeca no es la mano (hallazgo 25) y el
   producto asignado cambia de una pasada a otra.
3. **Comparar** el recuadro con su referencia, solo con nadie delante.
4. **Avisar** cuando el cambio pasa el umbral `stock_confirmaciones` veces
   seguidas (un reflejo o alguien que roza el borde no basta). La alerta se
   cierra sola cuando el recuadro vuelve a parecerse a la referencia
   (repuesto).

## Cómo se compara (opción «A», elegida por el usuario el 5-oct)

Escala de grises, suavizado, y cada imagen normalizada a media 0 y
desviación 1 DENTRO del polígono: así un cambio de luz que aclara u oscurece
todo el recuadro no cuenta como cambio. La cifra es la diferencia media entre
las dos (0 = idénticas; ~1 = nada que ver). La de la balda que pasa de llena a
lisa sale alta porque la referencia tenía dibujo y la actual no.

La desviación lleva un SUELO (`_STD_MIN`): sin él, un recuadro casi liso
amplificaba el ruido de la compresión JPEG hasta parecer un cambio enorme.

## Lo que NO está medido

**El umbral (`stock_umbral`) NO está calibrado**: no existe todavía ni un
caso real de balda vaciada, solo el vídeo, donde nadie vacía una balda. Cada
medición sale en el log (`STOCK [...] cambio=...`) para ajustarlo con datos.
Tampoco se sabe si, a la distancia de la cámara 3, «quedan dos latas» se
distingue de «quedan cero».
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: (área, posición del producto dentro del área). La misma identidad que el
#: reparto de agarres: la posición, no el nombre (`anaqueles_productos.py`).
Clave = Tuple[int, int]

#: Suelo de la desviación al normalizar (niveles de gris 0-255). Ver la
#: cabecera: sin él, un recuadro liso amplifica el ruido del JPEG.
_STD_MIN = 8.0

#: Un recuadro más chico que esto no se puede comparar con sentido.
_LADO_MIN = 8

#: Margen (px) alrededor del recuadro dentro del cual un cuerpo ya «tapa».
_MARGEN_TAPA = 6

#: Lo que sale de una alerta hacia el cliente y la API (sin el estado interno
#: de la histéresis).
_CAMPOS_ALERTA = ('area', 'producto_idx', 'area_nombre', 'producto',
                  'desde', 'cambio')

# ── Puente con la API (otro hilo) ───────────────────────────────────────────
# La API corre en el hilo del servidor web y el vigilante en el del pipeline.
# Solo se cruzan dos cosas, siempre bajo el candado: la PETICIÓN de foto y una
# COPIA del estado para consultarlo.
_candado = threading.Lock()
_peticiones: Dict[str, float] = {}
_estados: Dict[str, Dict[str, Any]] = {}


def pedir_referencia(camera_id: str) -> bool:
    """Pide la foto «surtido» de todos los productos de la cámara.

    False si esa cámara no está midiendo en modo Anaqueles ahora mismo (no
    habría quién atendiera la petición).
    """
    cam = str(camera_id)
    with _candado:
        if cam not in _estados:
            return False
        _peticiones[cam] = time.time()
        return True


def estado(camera_id: str) -> Optional[Dict[str, Any]]:
    """Copia del último estado publicado por la cámara, o None."""
    with _candado:
        e = _estados.get(str(camera_id))
        return json.loads(json.dumps(e)) if e is not None else None


def camaras_vigiladas() -> List[str]:
    with _candado:
        return sorted(_estados)


def _tomar_peticion(camera_id: str) -> Optional[float]:
    with _candado:
        return _peticiones.pop(str(camera_id), None)


# ── Imagen ──────────────────────────────────────────────────────────────────
def _caja(poligono: np.ndarray, forma: Tuple[int, int]
          ) -> Optional[Tuple[int, int, int, int]]:
    alto, ancho = forma
    x1 = max(0, int(np.floor(poligono[:, 0].min())))
    y1 = max(0, int(np.floor(poligono[:, 1].min())))
    x2 = min(ancho, int(np.ceil(poligono[:, 0].max())) + 1)
    y2 = min(alto, int(np.ceil(poligono[:, 1].max())) + 1)
    if x2 - x1 < _LADO_MIN or y2 - y1 < _LADO_MIN:
        return None
    return x1, y1, x2, y2


def _mascara(poligono: np.ndarray, caja: Tuple[int, int, int, int]
             ) -> np.ndarray:
    x1, y1, x2, y2 = caja
    m = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    pts = np.round(poligono - np.array([x1, y1], dtype=np.float32))
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m > 0


def _normalizada(recorte_bgr: np.ndarray, mascara: np.ndarray
                 ) -> Optional[np.ndarray]:
    """Gris suavizado y normalizado (media 0, desviación 1) en la máscara."""
    if recorte_bgr.ndim == 3:
        gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gris = recorte_bgr
    gris = cv2.GaussianBlur(gris, (5, 5), 0).astype(np.float32)
    valores = gris[mascara]
    if valores.size < 16:
        return None
    media = float(valores.mean())
    desv = max(_STD_MIN, float(valores.std()))
    return (gris - media) / desv


def diferencia(referencia_bgr: np.ndarray, actual_bgr: np.ndarray,
               mascara: np.ndarray) -> Optional[float]:
    """Cambio entre dos recortes del MISMO recuadro. 0 = idénticos.

    None si no se pueden comparar (tamaños distintos o recuadro vacío).
    """
    if referencia_bgr.shape[:2] != actual_bgr.shape[:2] \
            or referencia_bgr.shape[:2] != mascara.shape:
        return None
    a = _normalizada(referencia_bgr, mascara)
    b = _normalizada(actual_bgr, mascara)
    if a is None or b is None:
        return None
    return float(np.abs(a - b)[mascara].mean())


def _tapado(caja: Tuple[int, int, int, int],
            cuerpos: Sequence[Tuple[float, float, float, float]]) -> bool:
    """¿Algún cuerpo pisa el recuadro (con un margen)? Misma regla de
    intersección de cajas que `_guardar_refs` del medidor."""
    x1, y1, x2, y2 = caja
    x1, y1 = x1 - _MARGEN_TAPA, y1 - _MARGEN_TAPA
    x2, y2 = x2 + _MARGEN_TAPA, y2 + _MARGEN_TAPA
    return any(not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2)
               for (bx1, by1, bx2, by2) in cuerpos)


def _firma_poligono(poligono: np.ndarray, forma: Tuple[int, int]) -> List:
    """Lo que tiene que coincidir para que una referencia siga valiendo:
    el recuadro (redondeado) y el tamaño del frame."""
    return [[int(round(float(x))), int(round(float(y)))]
            for x, y in poligono] + [list(forma)]


def _hora(t: float) -> str:
    return time.strftime('%H:%M:%S', time.localtime(t))


def _jpg_b64(img: Optional[np.ndarray]) -> str:
    """JPEG en base64 (la miniatura de la tarjeta de alerta), o ''."""
    try:
        import base64
        if img is None or not getattr(img, 'size', 0):
            return ''
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''
    except Exception:                                       # noqa: BLE001
        return ''


#: Alto de cada panel de la imagen de evidencia (px).
_ALTO_EVIDENCIA = 240


def componer_evidencia(antes: np.ndarray, ahora: np.ndarray,
                       persona: Optional[np.ndarray],
                       nombre: str) -> np.ndarray:
    """ANTES | AHORA | PERSONA, al mismo alto y con su rótulo.

    El recuadro de un producto mide unas decenas de píxeles: se AMPLÍA para
    que se vea el hueco, que es la prueba. Sin persona, solo los dos
    primeros paneles.
    """
    paneles = [(antes, 'ANTES'), (ahora, 'AHORA')]
    if persona is not None and getattr(persona, 'size', 0):
        paneles.append((persona, 'QUIEN AGARRO'))
    piezas = []
    for img, rotulo in paneles:
        h, w = img.shape[:2]
        ancho = max(1, int(round(w * _ALTO_EVIDENCIA / max(1, h))))
        p = cv2.resize(img, (ancho, _ALTO_EVIDENCIA),
                       interpolation=cv2.INTER_CUBIC)
        p = cv2.copyMakeBorder(p, 28, 0, 4, 4, cv2.BORDER_CONSTANT,
                               value=(30, 30, 30))
        cv2.putText(p, rotulo, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        piezas.append(p)
    salida = np.hstack(piezas)
    # Nombre del producto abajo. ASCII: `putText` no pinta tildes (el «??»
    # de los mapas de calor, CLAUDE.md sección 6).
    texto = nombre.encode('ascii', 'replace').decode('ascii')
    pie = np.full((30, salida.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(pie, f'Posible agotado: {texto}', (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 1, cv2.LINE_AA)
    return np.vstack([salida, pie])


#: Lado (px) de la imagen que viaja en la alerta.
_LADO_MINIATURA = 320


def _encajar(img: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """`img` dentro de un lienzo ancho×alto SIN deformarla (bandas grises)."""
    h, w = img.shape[:2]
    escala = min(ancho / max(1, w), alto / max(1, h))
    nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
    p = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    lienzo = np.full((alto, ancho, 3), 30, dtype=np.uint8)
    x, y = (ancho - nw) // 2, (alto - nh) // 2
    lienzo[y:y + nh, x:x + nw] = p
    return lienzo


def componer_miniatura(antes: np.ndarray, ahora: np.ndarray,
                       lado: int = _LADO_MINIATURA) -> np.ndarray:
    """ANTES y AHORA del producto en un CUADRADO: es la foto de la alerta.

    La pidió el usuario el 5-oct en lugar de la persona («no es mejor mandar
    la foto del antes y después»): en un agotado lo que importa es el hueco.
    Cuadrada porque la tarjeta del panel pinta la miniatura en 56×56
    ESTIRÁNDOLA (`setScaledContents`, `panel_alertas.py:237-238`): una
    imagen apaisada salía deformada. Un recuadro apaisado va uno encima del
    otro (así se acerca al cuadrado); uno alto, uno al lado del otro.
    """
    h, w = antes.shape[:2]
    if w >= h:
        mitades = [_encajar(antes, lado, lado // 2),
                   _encajar(ahora, lado, lado - lado // 2)]
    else:
        mitades = [_encajar(antes, lado // 2, lado),
                   _encajar(ahora, lado - lado // 2, lado)]
    for img, rotulo, color in ((mitades[0], 'ANTES', (90, 200, 90)),
                               (mitades[1], 'AHORA', (80, 80, 255))):
        cv2.putText(img, rotulo, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, rotulo, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    salida = np.vstack(mitades) if w >= h else np.hstack(mitades)
    # Raya entre las dos mitades: a 56 px es lo único que las separa.
    if w >= h:
        salida[lado // 2 - 1:lado // 2 + 1, :] = 255
    else:
        salida[:, lado // 2 - 1:lado // 2 + 1] = 255
    return salida


def _duracion(segundos: float) -> str:
    """«45 s», «2 min 05 s», «1 h 03 min»."""
    s = max(0, int(segundos))
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60} min {s % 60:02d} s'
    return f'{s // 3600} h {(s % 3600) // 60:02d} min'


# ── Vigilante ───────────────────────────────────────────────────────────────
class VigilanteStock:
    """Referencias, vigilancia tras agarre y alertas de UNA cámara.

    Nunca lanza hacia fuera: es un extra del medidor y no puede costar el
    conteo de visitas y agarres.
    """

    def __init__(self, camera_id: Any, carpeta: Path,
                 cfg: Dict[str, Any]) -> None:
        self.camera_id = str(camera_id)
        self.cfg = cfg
        self._carpeta = Path(carpeta) / 'stock_ref'
        #: clave -> {'img', 'mascara', 'caja', 'firma', 'nombre', 'ts'}
        self._refs: Dict[Clave, Dict[str, Any]] = {}
        #: clave -> firma con la que ya se intentó cargar su referencia de
        #: disco. Se REINTENTA cuando la firma cambia. Con un solo intento
        #: (hasta el 5-oct, 18:06), una reconexión del cliente dejó las 8
        #: referencias sin cargar para siempre: el primer frame no cuadró y
        #: no se volvió a mirar.
        self._intentos_disco: Dict[Clave, Any] = {}
        #: clave -> {'desde', 'racha_alta', 'racha_baja', 'ultima'}
        self._pendientes: Dict[Clave, Dict[str, Any]] = {}
        #: clave -> {'desde', 'cambio', 'racha_baja', 'ultima', ...}
        self._alertas: Dict[Clave, Dict[str, Any]] = {}
        #: Petición de foto en curso: claves que faltan + hora de la petición.
        self._peticion: Optional[Dict[str, Any]] = None
        #: Sucesos para el feed del panel (los drena el medidor).
        self._sucesos: List[Dict[str, Any]] = []
        #: área -> {'img', 't'}: recorte de la persona del ÚLTIMO agarre.
        #: Solo para la evidencia en disco: la foto de la alerta es el antes
        #: y el después del producto (lo pidió el usuario el 5-oct).
        self._ultima_persona: Dict[int, Dict[str, Any]] = {}
        #: Alertas «posible agotado» para el panel de alertas, con su foto.
        self._alertas_salida: List[Dict[str, Any]] = []
        #: Hora del último frame: con ella sale «lleva X» de cada alerta.
        self._t = 0.0
        self.cambio = True        # hay algo nuevo que emitir al cliente
        self._publicar()

    # ── Entrada desde el medidor ────────────────────────────────────────
    def tras_agarre(self, area: int, t: float,
                    persona: Optional[np.ndarray] = None) -> None:
        """Un agarre confirmado en `area`: vigilar todos sus productos con
        referencia. Un agarre nuevo reinicia la vigilancia del producto.
        `persona` es el recorte de quien agarró, para la alerta."""
        if persona is not None and getattr(persona, 'size', 0):
            self._ultima_persona[int(area)] = {'img': persona, 't': float(t)}
        for clave in list(self._refs):
            if clave[0] != int(area) or clave in self._alertas:
                continue
            self._pendientes[clave] = {'desde': float(t), 'racha_alta': 0,
                                       'racha_baja': 0, 'ultima': -1e12}
        self.cambio = True

    def vigilar(self, frame: np.ndarray,
                productos: Dict[int, List[Tuple[int, str, np.ndarray]]],
                nombres_area: Dict[int, str],
                cuerpos: Sequence[Tuple[float, float, float, float]],
                t: float) -> None:
        """Un frame: atender la petición de foto, medir lo pendiente y lo
        que está en alerta. Nunca lanza."""
        self._t = float(t)
        try:
            self._vigilar(frame, productos, nombres_area, cuerpos, t)
        except Exception as exc:                            # noqa: BLE001
            logger.error("stock[%s]: falló la vigilancia: %s",
                         self.camera_id, exc)
        self._publicar()

    def drenar_sucesos(self) -> List[Dict[str, Any]]:
        sucesos, self._sucesos = self._sucesos, []
        return sucesos

    def drenar_alertas(self) -> List[Dict[str, Any]]:
        """Alertas nuevas de «posible agotado», cada una con `imagen_b64`
        (antes y después, en cuadrado) y `ruta` (la evidencia en disco)."""
        alertas, self._alertas_salida = self._alertas_salida, []
        return alertas

    def bloque(self) -> Dict[str, Any]:
        """Lo que viaja en `metadata['anaqueles']['stock']`.

        `vacio_s` = segundos desde que se detectó, a la hora del último
        frame. Viaja en SEGUNDOS y no en hora de reloj: el servidor de Linux
        va en UTC y el cliente en hora local (CLAUDE.md, sección 1); el panel
        suma lo que pasa desde que lo recibe con su propio reloj.
        """
        return {
            'referencias': len(self._refs),
            'vigilando': len(self._pendientes),
            'esperando_foto': (len(self._peticion['faltan'])
                               if self._peticion else 0),
            'alertas': [dict({k: a[k] for k in _CAMPOS_ALERTA},
                             vacio_s=max(0, int(self._t - a['desde_t'])))
                        for _c, a in sorted(self._alertas.items())],
        }

    # ── Interno ─────────────────────────────────────────────────────────
    def _vigilar(self, frame, productos, nombres_area, cuerpos, t) -> None:
        forma = tuple(int(v) for v in frame.shape[:2])
        vivos: Dict[Clave, Tuple[str, np.ndarray]] = {}
        for area, lista in (productos or {}).items():
            for posicion, nombre, poligono in lista:
                vivos[(int(area), int(posicion))] = (nombre, poligono)

        self._olvidar_desaparecidos(vivos, forma)
        for clave in vivos:
            self._cargar_de_disco(clave, vivos[clave][1], forma)

        if _tomar_peticion(self.camera_id) is not None:
            self._peticion = {'desde': t, 'faltan': set(vivos)}
            self.cambio = True
            logger.info("stock[%s]: pedida la foto de referencia de %d "
                        "productos", self.camera_id, len(vivos))
        if self._peticion is not None:
            self._atender_peticion(frame, vivos, nombres_area, cuerpos,
                                   forma, t)

        for clave in list(self._pendientes):
            self._medir_pendiente(clave, frame, vivos, nombres_area,
                                  cuerpos, t)
        for clave in list(self._alertas):
            self._medir_alerta(clave, frame, vivos, nombres_area, cuerpos, t)

    def _olvidar_desaparecidos(self, vivos, forma) -> None:
        """Un producto quitado o REDIBUJADO pierde su referencia: la foto
        era de otro recuadro. Sus alertas y vigilancias se van con ella."""
        for clave in list(self._refs):
            vivo = vivos.get(clave)
            if vivo is not None and \
                    self._refs[clave]['firma'] == _firma_poligono(vivo[1],
                                                                  forma):
                continue
            self._refs.pop(clave, None)
            self._pendientes.pop(clave, None)
            if self._alertas.pop(clave, None) is not None:
                self.cambio = True
            self._intentos_disco.pop(clave, None)

    def _cargar_de_disco(self, clave: Clave, poligono: np.ndarray,
                         forma) -> None:
        if clave in self._refs:
            return
        firma = _firma_poligono(poligono, forma)
        if self._intentos_disco.get(clave) == firma:
            return                  # ya probado con estos mismos píxeles
        self._intentos_disco[clave] = firma
        try:
            base = self._carpeta / f'{clave[0]}_{clave[1]}'
            meta_ruta = base.with_suffix('.json')
            if not meta_ruta.exists():
                return
            meta = json.loads(meta_ruta.read_text(encoding='utf-8'))
            guardada = meta.get('firma') or []
            if guardada != firma:
                # Frame (alto, ancho) y primer punto de cada una: dice si
                # cambió el tamaño de la imagen o se movió el recuadro.
                logger.info("stock[%s]: la referencia guardada de %s no "
                            "cuadra con el recuadro de ahora (frame %s, "
                            "punto %s; guardada: frame %s, punto %s). Se "
                            "reintenta si el recuadro cambia; si no, hay "
                            "que volver a tomarla", self.camera_id, clave,
                            firma[-1], firma[0],
                            guardada[-1] if guardada else None,
                            guardada[0] if guardada else None)
                return
            img = cv2.imread(str(base.with_suffix('.png')))
            caja = _caja(poligono, forma)
            if img is None or caja is None:
                return
            mascara = _mascara(poligono, caja)
            if img.shape[:2] != mascara.shape:
                return
            self._refs[clave] = {'img': img, 'mascara': mascara,
                                 'caja': caja, 'firma': meta['firma'],
                                 'nombre': meta.get('nombre'),
                                 'ts': float(meta.get('ts') or 0)}
            self.cambio = True
            logger.info("stock[%s]: referencia de %s cargada del disco",
                        self.camera_id, meta.get('nombre') or clave)
        except Exception as exc:                            # noqa: BLE001
            logger.warning("stock[%s]: no se pudo leer la referencia %s: %s",
                           self.camera_id, clave, exc)

    def _atender_peticion(self, frame, vivos, nombres_area, cuerpos,
                          forma, t) -> None:
        pet = self._peticion
        if pet is None:
            return
        for clave in list(pet['faltan']):
            if clave not in vivos:
                pet['faltan'].discard(clave)
                continue
            nombre, poligono = vivos[clave]
            caja = _caja(poligono, forma)
            if caja is None:
                pet['faltan'].discard(clave)
                logger.warning("stock[%s]: %s es demasiado pequeño para "
                               "compararlo", self.camera_id, nombre)
                continue
            if _tapado(caja, cuerpos):
                continue                # se toma cuando se aparten
            x1, y1, x2, y2 = caja
            img = frame[y1:y2, x1:x2].copy()
            ref = {'img': img, 'mascara': _mascara(poligono, caja),
                   'caja': caja, 'firma': _firma_poligono(poligono, forma),
                   'nombre': nombre, 'ts': t}
            self._refs[clave] = ref
            # Con la foto nueva, lo que se vigilaba o avisaba ya no aplica.
            self._pendientes.pop(clave, None)
            vieja = self._alertas.pop(clave, None)
            if vieja is not None:
                self._suceso('repuesto', clave, nombre, nombres_area, t,
                             duracion_s=int(t - vieja['desde_t']))
            self._guardar(clave, ref)
            pet['faltan'].discard(clave)
            self.cambio = True
            logger.info("stock[%s]: referencia tomada de %s (%s)",
                        self.camera_id, nombre,
                        nombres_area.get(clave[0], f'Área {clave[0] + 1}'))
        caduca = float(self.cfg.get('stock_peticion_max_s', 120.0))
        if not pet['faltan'] or (t - pet['desde']) > caduca:
            if pet['faltan']:
                logger.warning("stock[%s]: %d productos siguieron tapados "
                               "%.0f s; sin referencia nueva",
                               self.camera_id, len(pet['faltan']), caduca)
            self._peticion = None
            self.cambio = True

    def _guardar(self, clave: Clave, ref: Dict[str, Any]) -> None:
        try:
            self._carpeta.mkdir(parents=True, exist_ok=True)
            base = self._carpeta / f'{clave[0]}_{clave[1]}'
            ok, buf = cv2.imencode('.png', ref['img'])
            if not ok:
                return
            tmp = base.with_suffix('.tmp')
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, base.with_suffix('.png'))
            base.with_suffix('.json').write_text(json.dumps(
                {'firma': ref['firma'], 'nombre': ref['nombre'],
                 'ts': ref['ts']}, ensure_ascii=False), encoding='utf-8')
        except Exception as exc:                            # noqa: BLE001
            logger.warning("stock[%s]: no se pudo guardar la referencia "
                           "%s: %s", self.camera_id, clave, exc)

    def _cambio_de(self, clave: Clave, frame, cuerpos) -> Optional[float]:
        """Cambio actual frente a la referencia; None si está tapado."""
        ref = self._refs.get(clave)
        if ref is None or _tapado(ref['caja'], cuerpos):
            return None
        x1, y1, x2, y2 = ref['caja']
        return diferencia(ref['img'], frame[y1:y2, x1:x2], ref['mascara'])

    def _emitir_alerta(self, clave: Clave, frame, nombre: str,
                       area_nombre: str, cambio: float, t: float) -> None:
        """La alerta para el panel: su foto es el ANTES y el AHORA del
        producto (5-oct, a petición del usuario; antes era la persona); en
        disco queda además la evidencia antes | ahora | quién agarró, que
        es lo que abre la tarjeta al pulsarla. Nunca lanza."""
        try:
            ref = self._refs[clave]
            x1, y1, x2, y2 = ref['caja']
            ahora = frame[y1:y2, x1:x2].copy()
            persona = self._ultima_persona.get(clave[0])
            img_persona = persona['img'] if persona else None
            ruta = self._guardar_evidencia(clave, ref['img'], ahora,
                                           img_persona, nombre, t)
            self._alertas_salida.append({
                'area': clave[0], 'producto_idx': clave[1],
                'area_nombre': area_nombre, 'producto': nombre,
                'cambio': round(cambio, 3), 'hora': _hora(t),
                'hora_agarre': _hora(persona['t']) if persona else None,
                'imagen_b64': _jpg_b64(componer_miniatura(ref['img'], ahora)),
                'ruta': str(ruta) if ruta else ''})
        except Exception as exc:                            # noqa: BLE001
            logger.error("stock[%s]: no se pudo preparar la alerta de %s: "
                         "%s", self.camera_id, nombre, exc)

    def _guardar_evidencia(self, clave: Clave, antes, ahora, persona,
                           nombre: str, t: float) -> Optional[Path]:
        """`stock_alertas/<fecha>_<hora>_<área>_<producto>.jpg` junto a las
        referencias. Son fotos de clientes: viven en `output/`, fuera de
        git, como las fichas de las personas."""
        try:
            carpeta = self._carpeta.parent / 'stock_alertas'
            carpeta.mkdir(parents=True, exist_ok=True)
            ruta = carpeta / (time.strftime('%Y-%m-%d_%H%M%S',
                                            time.localtime(t))
                              + f'_{clave[0]}_{clave[1]}.jpg')
            img = componer_evidencia(antes, ahora, persona, nombre)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return None
            ruta.write_bytes(buf.tobytes())
            return ruta
        except Exception as exc:                            # noqa: BLE001
            logger.warning("stock[%s]: no se pudo guardar la evidencia: %s",
                           self.camera_id, exc)
            return None

    def _medir_pendiente(self, clave, frame, vivos, nombres_area,
                         cuerpos, t) -> None:
        p = self._pendientes[clave]
        if (t - p['desde']) > float(self.cfg.get('stock_ventana_s', 180.0)):
            # Nadie lo dejó ver lo bastante: se abandona sin opinar.
            self._pendientes.pop(clave, None)
            self.cambio = True
            return
        if (t - p['desde']) < float(self.cfg.get('stock_espera_s', 2.0)):
            return
        if (t - p['ultima']) < float(self.cfg.get('stock_muestreo_s', 1.0)):
            return
        cambio = self._cambio_de(clave, frame, cuerpos)
        if cambio is None:
            return
        p['ultima'] = t
        umbral = float(self.cfg.get('stock_umbral', 0.45))
        n = int(self.cfg.get('stock_confirmaciones', 3))
        if cambio >= umbral:
            p['racha_alta'] += 1
            p['racha_baja'] = 0
        else:
            p['racha_baja'] += 1
            p['racha_alta'] = 0
        nombre = vivos.get(clave, (self._refs[clave]['nombre'], None))[0]
        logger.info("STOCK [%s] %s / %s: cambio=%.3f (umbral %.2f) "
                    "alto %d/%d bajo %d/%d", self.camera_id,
                    nombres_area.get(clave[0], f'Área {clave[0] + 1}'),
                    nombre, cambio, umbral, p['racha_alta'], n,
                    p['racha_baja'], n)
        if p['racha_alta'] >= n:
            self._pendientes.pop(clave, None)
            self._alertas[clave] = {
                'area': clave[0], 'producto_idx': clave[1],
                'area_nombre': nombres_area.get(clave[0],
                                                f'Área {clave[0] + 1}'),
                'producto': nombre, 'desde': _hora(t), 'desde_t': t,
                'cambio': round(cambio, 3), 'racha_baja': 0, 'ultima': t}
            self._suceso('agotado', clave, nombre, nombres_area, t)
            self._emitir_alerta(clave, frame, nombre,
                                self._alertas[clave]['area_nombre'], cambio, t)
            logger.warning("stock[%s]: POSIBLE AGOTADO %s en %s "
                           "(cambio %.3f)", self.camera_id, nombre,
                           nombres_area.get(clave[0], clave[0]), cambio)
            self.cambio = True
        elif p['racha_baja'] >= n:
            self._pendientes.pop(clave, None)      # sigue surtido
            self.cambio = True

    def _medir_alerta(self, clave, frame, vivos, nombres_area,
                      cuerpos, t) -> None:
        a = self._alertas[clave]
        if (t - a['ultima']) < float(self.cfg.get('stock_muestreo_alerta_s',
                                                   5.0)):
            return
        cambio = self._cambio_de(clave, frame, cuerpos)
        if cambio is None:
            return
        a['ultima'] = t
        a['cambio'] = round(cambio, 3)
        # Histéresis: para darlo por repuesto tiene que parecerse BASTANTE
        # más que el umbral de alerta, o una balda en el límite parpadearía.
        repuesto = float(self.cfg.get('stock_umbral', 0.45)) * float(
            self.cfg.get('stock_repuesto_frac', 0.6))
        if cambio < repuesto:
            a['racha_baja'] += 1
        else:
            a['racha_baja'] = 0
        if a['racha_baja'] >= int(self.cfg.get('stock_confirmaciones', 3)):
            self._alertas.pop(clave, None)
            nombre = vivos.get(clave, (a['producto'], None))[0]
            duracion = int(t - a['desde_t'])
            self._suceso('repuesto', clave, nombre, nombres_area, t,
                         duracion_s=duracion)
            logger.info("stock[%s]: %s vuelve a parecerse a su referencia "
                        "(repuesto) tras %s", self.camera_id, nombre,
                        _duracion(duracion))
            self.cambio = True

    def _suceso(self, tipo, clave, nombre, nombres_area, t,
                **extra) -> None:
        area = nombres_area.get(clave[0], f'Área {clave[0] + 1}')
        self._sucesos.append(dict({
            'tipo': tipo, 'area': clave[0], 'nombre': area,
            'producto': nombre, 'ts': _hora(t), 'genero': None,
            'rango_etario': None, 'id_corto': f'stock-{clave[0]}-{clave[1]}'},
            **extra))

    def _publicar(self) -> None:
        """Copia del estado para la API (otro hilo)."""
        copia = dict(self.bloque(), camera_id=self.camera_id,
                     productos_con_referencia=[
                         {'area': c[0], 'producto_idx': c[1],
                          'producto': r['nombre'], 'desde': _hora(r['ts'])}
                         for c, r in sorted(self._refs.items())])
        with _candado:
            _estados[self.camera_id] = copia
