"""Fotos de evidencia de `PermanenciaArea`: quién entró, quién salió, cuánto.

Una alerta que dice «alguien lleva 40 minutos dentro» obliga a ir a mirar la
cámara. Una alerta **con la foto** se resuelve desde el móvil. Este módulo es
el que produce esa foto, y es lo único del modo que escribe imágenes a disco.

## Las tres imágenes

  - **Entrada**: recorte de la persona en el instante en que se confirma que
    entró, con un banner verde (hora + identidad).
  - **Salida**: lo mismo en rojo, con el tiempo total.
  - **Comparativa**: las dos anteriores pegadas lado a lado. Es la que viaja
    en la alerta de salida, porque responde de un vistazo la pregunta que
    importa —«¿es la misma persona que entró?»— sin abrir dos mensajes.

## Por qué se guarda en disco Y se manda en base64

No son lo mismo y hacen falta las dos:

  - el **base64** viaja en la alerta y es lo que ven WhatsApp, Jarvis y el
    panel del cliente; es efímero y va comprimido y reescalado;
  - el **archivo** es el respaldo consultable después («enséñame las entradas
    del martes»), y no depende de que el cliente estuviera conectado.

## Acentos

`cv2.putText` solo dibuja Hershey, que no tiene acentos ni «ñ»: escribir
«Entró al área» pinta «Entr? al ?rea». El texto del banner —solo el del
banner— se translitera antes de dibujarse. La descripción de la alerta, que
va como texto y no como píxeles, conserva sus tildes.

## Retención

Las fotos se agrupan por fecha y se borran las carpetas más viejas que
`retencion_dias`. Sin esto una cámara de puerta llena el disco en semanas.
La purga es perezosa (una vez por hora, en el hilo de inferencia) y nunca
lanza: perder una purga no puede costar un frame.
"""

from __future__ import annotations

import base64
import datetime
import logging
import shutil
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: BGR por momento. Verde = entró, rojo = salió, ámbar = sigue dentro.
COLORES: Dict[str, Tuple[int, int, int]] = {
    'entrada': (80, 175, 76),
    'salida': (60, 60, 220),
    'permanencia': (40, 190, 240),
    'sin_salida': (150, 150, 150),
}

_VACIA = ''


@dataclass
class Evidencia:
    """Una foto guardada. `b64` vacío = no se pudo, y hay que decirlo."""
    ruta: str = _VACIA
    b64: str = _VACIA

    def __bool__(self) -> bool:
        return bool(self.b64)


# ── Texto ────────────────────────────────────────────────────────────────
def sin_acentos(texto: Any) -> str:
    """'Entró al área' -> 'Entro al area'. Solo para lo que se DIBUJA."""
    try:
        crudo = unicodedata.normalize('NFKD', str(texto))
        return crudo.encode('ascii', 'ignore').decode('ascii')
    except Exception:                                       # noqa: BLE001
        return str(texto)


def nombre_seguro(texto: Any, largo: int = 24) -> str:
    """Nombre de cámara utilizable como carpeta en Windows."""
    limpio = ''.join(c for c in sin_acentos(texto)
                     if c.isalnum() or c in '_-')
    return limpio[:largo] or 'cam'


def id_corto(persistent_id: Any) -> str:
    """UUID largo -> 8 caracteres. Es lo que cabe en un banner."""
    crudo = str(persistent_id or '')
    if crudo.startswith('prov:'):
        return crudo[:12]
    return crudo.replace('-', '')[:8] or 'sin-id'


# ── Imagen ───────────────────────────────────────────────────────────────
def recortar(frame: Optional[np.ndarray],
             box: Optional[Sequence[float]],
             margen_x: float = 0.18,
             margen_y: float = 0.12) -> Optional[np.ndarray]:
    """Recorte de la persona con aire alrededor. `None` si no se puede.

    El margen no es estético: un recorte pegado al cuerpo pierde la cabeza en
    cuanto la caja se queda corta, y la cabeza es justo lo que se mira para
    decidir si es la misma persona.
    """
    if frame is None or getattr(frame, 'size', 0) == 0:
        return None
    if box is None:
        return frame
    try:
        alto, ancho = frame.shape[:2]
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        mx, my = int((x2 - x1) * margen_x), int((y2 - y1) * margen_y)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(ancho, x2 + mx), min(alto, y2 + my)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return frame
        recorte = frame[y1:y2, x1:x2]
        return recorte if getattr(recorte, 'size', 0) else frame
    except (TypeError, ValueError, IndexError):
        return frame


def _escalar(img: np.ndarray, max_ancho: int, max_alto: int) -> np.ndarray:
    """Encoge (nunca amplía) para que el base64 no reviente el WebSocket."""
    import cv2
    alto, ancho = img.shape[:2]
    if alto <= 0 or ancho <= 0:
        return img
    factor = min(max_ancho / float(ancho), max_alto / float(alto), 1.0)
    if factor >= 0.999:
        return img
    return cv2.resize(img, (max(1, int(ancho * factor)),
                            max(1, int(alto * factor))),
                      interpolation=cv2.INTER_AREA)


def anotar(img: np.ndarray, titulo: str, lineas: Sequence[str],
           color: Tuple[int, int, int]) -> np.ndarray:
    """Pega una franja de color arriba con el título y el detalle debajo."""
    import cv2
    try:
        alto, ancho = img.shape[:2]
        ancho_min = 260
        if ancho < ancho_min:                # banner ilegible en fotos angostas
            escala = ancho_min / float(ancho)
            img = cv2.resize(img, (ancho_min, max(1, int(alto * escala))),
                             interpolation=cv2.INTER_LINEAR)
            alto, ancho = img.shape[:2]

        alto_titulo = 26
        alto_pie = 18 * len(lineas) + (8 if lineas else 0)
        lienzo = np.zeros((alto + alto_titulo + alto_pie, ancho, 3),
                          dtype=np.uint8)
        lienzo[:] = (28, 28, 28)
        lienzo[0:alto_titulo, :] = color
        lienzo[alto_titulo:alto_titulo + alto, :] = img

        cv2.putText(lienzo, sin_acentos(titulo).upper(), (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        y = alto_titulo + alto + 14
        for linea in lineas:
            cv2.putText(lienzo, sin_acentos(linea), (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1,
                        cv2.LINE_AA)
            y += 18
        return lienzo
    except Exception as exc:                                # noqa: BLE001
        logger.debug("Evidencia: no se pudo anotar (%s); va sin banner", exc)
        return img


#: Lado máximo de las fotos de ESCENA COMPLETA (Inicio/Fin). Más grande que
#: el recorte de persona: aquí lo que importa es ver todo el encuadre.
ANCHO_ESCENA = 900


def escena_b64(frame: Optional[np.ndarray], calidad: int = 80) -> str:
    """El frame COMPLETO, reescalado, en base64. Vacío si no se puede.

    Es la foto de «Inicio» que la estancia guarda al abrir: cuando la persona
    salga, se apila con la escena del cierre. Sin recorte a propósito — el
    formato pedido (10-ago-2026) muestra el área entera con el reloj de la
    propia cámara en pantalla.
    """
    if frame is None or getattr(frame, 'size', 0) == 0:
        return _VACIA
    try:
        return a_b64(_escalar(frame, ANCHO_ESCENA, ANCHO_ESCENA), calidad)
    except Exception:                                       # noqa: BLE001
        return _VACIA


def etiquetar_esquina(img: np.ndarray, texto: str,
                      color: Tuple[int, int, int] = COLORES['entrada']
                      ) -> np.ndarray:
    """Cinta de color con el texto en la esquina inferior izquierda.

    Es el rótulo «Inicio» / «Fin» del formato pedido: sobre la imagen, sin
    banner externo ni líneas de detalle — la hora ya la pinta la cámara.
    """
    import cv2
    try:
        alto, ancho = img.shape[:2]
        alto_cinta = max(28, alto // 16)
        escala = alto_cinta / 40.0
        (ancho_txt, _), _ = cv2.getTextSize(sin_acentos(texto),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            escala, 2)
        salida = img.copy()
        ancho_cinta = min(ancho, ancho_txt + 24 + alto_cinta)
        franja = salida[alto - alto_cinta:alto, 0:ancho_cinta]
        # Degradado horizontal del color al propio video, como en la muestra.
        for x in range(ancho_cinta):
            f = 1.0 - (x / float(max(1, ancho_cinta - 1))) * 0.55
            franja[:, x] = (np.asarray(color, dtype=np.float32) * f
                            + franja[:, x].astype(np.float32) * (1.0 - f))
        cv2.putText(salida, sin_acentos(texto),
                    (10, alto - alto_cinta // 2 + int(8 * escala)),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return salida
    except Exception as exc:                                # noqa: BLE001
        logger.debug("Evidencia: no se pudo etiquetar (%s)", exc)
        return img


def apilar(arriba: Optional[np.ndarray],
           abajo: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Pega dos fotos en columna (Inicio arriba, Fin abajo), igual de anchas."""
    import cv2
    if arriba is None:
        return abajo
    if abajo is None:
        return arriba
    try:
        ancho = max(arriba.shape[1], abajo.shape[1])

        def _al_ancho(img: np.ndarray) -> np.ndarray:
            if img.shape[1] == ancho:
                return img
            factor = ancho / float(img.shape[1])
            return cv2.resize(img, (ancho, max(1, int(img.shape[0] * factor))),
                              interpolation=cv2.INTER_AREA)

        separador = np.zeros((6, ancho, 3), dtype=np.uint8)
        separador[:] = (255, 255, 255)
        return cv2.vconcat([_al_ancho(arriba), separador, _al_ancho(abajo)])
    except Exception as exc:                                # noqa: BLE001
        logger.debug("Evidencia: no se pudo apilar (%s)", exc)
        return abajo


def componer(izquierda: Optional[np.ndarray],
             derecha: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Pega dos fotos ya anotadas lado a lado, igualadas en altura."""
    import cv2
    if izquierda is None:
        return derecha
    if derecha is None:
        return izquierda
    try:
        h = max(izquierda.shape[0], derecha.shape[0])

        def _rellenar(img: np.ndarray) -> np.ndarray:
            if img.shape[0] == h:
                return img
            fondo = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
            fondo[:] = (28, 28, 28)
            fondo[0:img.shape[0], :] = img
            return fondo

        izq, der = _rellenar(izquierda), _rellenar(derecha)
        separador = np.zeros((h, 4, 3), dtype=np.uint8)
        separador[:] = (70, 70, 70)
        return cv2.hconcat([izq, separador, der])
    except Exception as exc:                                # noqa: BLE001
        logger.debug("Evidencia: no se pudo componer (%s)", exc)
        return derecha


def a_b64(img: Optional[np.ndarray], calidad: int = 80) -> str:
    """JPEG en base64. Cadena vacía si no se puede (nunca lanza)."""
    import cv2
    if img is None or getattr(img, 'size', 0) == 0:
        return _VACIA
    try:
        ok, buf = cv2.imencode('.jpg', img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(calidad)])
        return base64.b64encode(buf).decode('ascii') if ok else _VACIA
    except Exception:                                       # noqa: BLE001
        return _VACIA


def de_b64(dato: Any) -> Optional[np.ndarray]:
    """El inverso, para recomponer la foto de entrada al cerrar la estancia."""
    import cv2
    if not dato:
        return None
    try:
        crudo = base64.b64decode(str(dato))
        img = cv2.imdecode(np.frombuffer(crudo, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        return img if img is not None and img.size else None
    except Exception:                                       # noqa: BLE001
        return None


# ── Gestor ───────────────────────────────────────────────────────────────
class GestorEvidencia:
    """Guarda las fotos en disco y devuelve el base64 para la alerta."""

    def __init__(self, carpeta: Path, retencion_dias: int = 30,
                 max_ancho: int = 420, max_alto: int = 640,
                 calidad: int = 80, guardar_en_disco: bool = True) -> None:
        self.carpeta = Path(carpeta)
        self.retencion_dias = max(0, int(retencion_dias))
        self.max_ancho = max(80, int(max_ancho))
        self.max_alto = max(80, int(max_alto))
        self.calidad = min(95, max(40, int(calidad)))
        self.guardar_en_disco = bool(guardar_en_disco)
        self._ultima_purga = 0.0
        self._fallos_disco = 0

    # -- API --------------------------------------------------------------
    def capturar(self, frame: Optional[np.ndarray],
                 box: Optional[Sequence[float]],
                 *, camara: Any, persistent_id: Any, momento: str,
                 titulo: str, lineas: Sequence[str] = ()) -> Evidencia:
        """Recorta, anota, guarda y codifica. Nunca lanza."""
        try:
            recorte = recortar(frame, box)
            if recorte is None:
                return Evidencia()
            recorte = _escalar(recorte, self.max_ancho, self.max_alto)
            anotada = anotar(recorte, titulo, lineas,
                             COLORES.get(momento, COLORES['permanencia']))
            ruta = self._escribir(anotada, camara, persistent_id, momento)
            return Evidencia(ruta=ruta, b64=a_b64(anotada, self.calidad))
        except Exception as exc:                            # noqa: BLE001
            logger.debug("Evidencia: captura fallida (%s)", exc)
            return Evidencia()

    def escenas_inicio_fin(self, entrada_escena_b64: Any,
                           frame_salida: Optional[np.ndarray],
                           *, camara: Any, persistent_id: Any) -> Evidencia:
        """Escena COMPLETA de la entrada sobre la de la salida, rotuladas.

        El formato pedido (10-ago-2026): dos cuadros del área entera, uno
        encima del otro, con la cinta verde «Inicio» / «Fin» en la esquina.
        La hora la enseña el reloj de la propia cámara en cada cuadro.
        """
        try:
            inicio = de_b64(entrada_escena_b64)
            fin = None
            if frame_salida is not None and getattr(frame_salida, 'size', 0):
                fin = _escalar(frame_salida, ANCHO_ESCENA, ANCHO_ESCENA)
            if inicio is not None:
                inicio = etiquetar_esquina(inicio, 'Inicio')
            if fin is not None:
                fin = etiquetar_esquina(fin, 'Fin')
            par = apilar(inicio, fin)
            if par is None:
                return Evidencia()
            ruta = self._escribir(par, camara, persistent_id, 'inicio_fin')
            return Evidencia(ruta=ruta, b64=a_b64(par, self.calidad))
        except Exception as exc:                            # noqa: BLE001
            logger.debug("Evidencia: escenas inicio/fin fallidas (%s)", exc)
            return Evidencia()

    def comparativa(self, entrada_b64: Any, salida: Optional[np.ndarray],
                    *, camara: Any, persistent_id: Any) -> Evidencia:
        """Foto de entrada + foto de salida en una sola imagen."""
        try:
            par = componer(de_b64(entrada_b64), salida)
            if par is None:
                return Evidencia()
            ruta = self._escribir(par, camara, persistent_id, 'comparativa')
            return Evidencia(ruta=ruta, b64=a_b64(par, self.calidad))
        except Exception as exc:                            # noqa: BLE001
            logger.debug("Evidencia: comparativa fallida (%s)", exc)
            return Evidencia()

    # -- Disco ------------------------------------------------------------
    def _escribir(self, img: np.ndarray, camara: Any, persistent_id: Any,
                  momento: str) -> str:
        if not self.guardar_en_disco:
            return _VACIA
        import cv2
        try:
            ahora = datetime.datetime.now()
            destino = (self.carpeta / ahora.strftime('%Y-%m-%d')
                       / nombre_seguro(camara))
            destino.mkdir(parents=True, exist_ok=True)
            nombre = (f"{ahora.strftime('%H%M%S')}_"
                      f"{id_corto(persistent_id)}_{momento}.jpg")
            ruta = destino / nombre
            cv2.imwrite(str(ruta), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.calidad])
            self.purgar()
            return str(ruta)
        except Exception as exc:                            # noqa: BLE001
            # Disco lleno o carpeta sin permisos: se pierde el respaldo, NO
            # la alerta (el base64 ya se generó y viaja igual).
            self._fallos_disco += 1
            if self._fallos_disco in (1, 10, 100):
                logger.warning("Evidencia: no se pudo escribir en %s (%s)",
                               self.carpeta, exc)
            return _VACIA

    def purgar(self, ahora: Optional[float] = None) -> int:
        """Borra las carpetas de fecha más viejas que la retención."""
        if self.retencion_dias <= 0 or not self.guardar_en_disco:
            return 0
        ahora = time.time() if ahora is None else ahora
        if ahora - self._ultima_purga < 3600:
            return 0
        self._ultima_purga = ahora
        borradas = 0
        try:
            limite = (datetime.date.today()
                      - datetime.timedelta(days=self.retencion_dias))
            for hijo in self.carpeta.iterdir():
                if not hijo.is_dir():
                    continue
                try:
                    fecha = datetime.datetime.strptime(hijo.name,
                                                       '%Y-%m-%d').date()
                except ValueError:
                    continue            # carpeta ajena: no es nuestra, no se toca
                if fecha < limite:
                    shutil.rmtree(hijo, ignore_errors=True)
                    borradas += 1
        except FileNotFoundError:
            pass
        except Exception as exc:                            # noqa: BLE001
            logger.debug("Evidencia: purga fallida (%s)", exc)
        if borradas:
            logger.info("Evidencia: %d carpeta(s) de fotos purgadas "
                        "(retención %d días)", borradas, self.retencion_dias)
        return borradas
