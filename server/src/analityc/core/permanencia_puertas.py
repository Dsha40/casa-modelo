"""Varias PUERTAS en una misma cámara: un umbral por sitio, un libro por sitio.

## De qué va (7-sep-2026)

Hasta hoy una cámara tenía UN umbral (`linea_cruce`) y por tanto UN aforo. El
recuadro del cliente ya deja dibujar hasta ocho umbrales con su flecha doble
—`puertas_lineas`—, y cada uno es un SITIO distinto: el almacén, el patio, la
trastienda. Este módulo es el reparto: instancia un detector por puerta,
le da a cada una su propio libro de estancias y guarda todo en UN fichero por
cámara con una sección por puerta.

No hay geometría nueva ni detector nuevo: el eje de la flecha doble ES la
normal del umbral, así que se reusa `DetectorLinea` tal cual (ver A2/A5 del
anexo y el docstring de `permanencia_linea`). Lo único de aquí es el reparto
y la RECONCILIACIÓN en caliente: el operador añade, mueve, renombra y quita
puertas con el vídeo corriendo, y cada frame trae la lista completa.

## Sin puertas dibujadas no pasa NADA

`poner([])` sobre un gestor que nunca tuvo puertas es un no-op total: ni
crea el fichero, ni cierra estancias, ni escribe una línea en el log. Es el
caso de casi todas las cámaras (el cliente manda `puertas_lineas: []` en cada
frame) y el modo de siempre —área, puerta, cruce y línea suelta— tiene que
comportarse exactamente igual que antes de existir este fichero.

## Las cuatro decisiones y el fallo concreto que evita cada una

  1. **`Dict[int, Puerta]` y NO una lista.** Con lista, borrar la puerta 2 de
     4 corre los índices: la gente que estaba dentro de la 3 pasa a contarse
     en la 2 y el aforo de dos sitios queda mal sin un solo error en el log.
     Con dict, el índice 3 sigue siendo el 3 aunque falte el 2.
  2. **La identidad es el ÍNDICE, no el nombre.** Renombrar «Almacén» a
     «Depósito» a media tarde bifurcaría el aforo en silencio: las visitas
     nuevas irían a una sección nueva y las personas que estaban dentro
     quedarían en una sección inalcanzable, dentro de un sitio que ya nadie
     consulta. Es la misma regla que `RegistroAnaqueles.area`.
  3. **Solo se quita la ÚLTIMA** (regla del recuadro). Es lo que hace seguro
     usar el índice como identidad. Quitar una puerta CON GENTE DENTRO no
     borra a nadie a escondidas: sus estancias vuelven de `poner()` ya
     cerradas como `sin_salida` para que el llamador emita el aviso.
  4. **UN fichero con secciones y no uno por puerta.** Con `<cam>__p3.json`,
     borrar la puerta 3 deja un huérfano en disco que RESUCITA en cuanto el
     operador dibuja otra tercera puerta con otra geometría: gente contada
     dentro de un sitio por el que no pasó y que ya no existe. Aquí cada
     escritura vuelca las puertas VIVAS y punto, y las secciones sin puerta
     se cierran como `sin_salida` en la primera reconciliación.

## El fichero (v2), y el v1 que se adopta

    {"version": 2, "guardado_ts": 1757260000.0,
     "puertas": {"0": {"nombre": "Almacen", "abiertas": [<estancia>, ...]},
                 "1": {"nombre": "Patio",   "abiertas": []}}}

Un fichero **v1** —el del libro de UNA puerta, con `abiertas` en la raíz, tal
como lo escribe `RegistroEstancias`— se lee ENTERO como la puerta 0, que es
justo lo que era: la única puerta de esa cámara. La primera escritura sale ya
en v2. Escritura ATÓMICA (temporal + `os.replace`) y como mucho una cada 5 s:
las estancias llevan la foto de entrada en base64 y escribirlas en cada frame
cuesta más que la inferencia.

## La trampa que este módulo NO debe pisar (A10)

`GestorPuertas` va como atributo PLANO del procesador y **fuera** de la lista
que `TiendasProcess._push_state` multiplexa por cámara. Ahí dentro se
pondría a `None` en cada frame, el `_actualizar` reventaría dentro de un
`try` que solo loguea y el modo quedaría MUDO con el vídeo perfecto: es
exactamente el bug del 6 y del 7-ago-2026. No hace falta multiplexar nada,
ya hay una instancia del procesador por cámara.
`tests/test_permanencia_puertas.py` tiene una prueba de fuente que se pone
roja si alguien lo «arregla» metiéndolo en esa lista.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Mapping, Optional, Sequence,
                    Tuple)

from .permanencia_estancias import (CIERRE_SIN_SALIDA, SIN_ATRIBUIR, Estancia,
                                    RegistroEstancias)
from .permanencia_linea import DetectorLinea
from .permanencia_linea import _geometria as _linea_geometria

logger = logging.getLogger(__name__)

#: Tope de umbrales por cámara. El mismo número que valida el contrato
#: (`FrameInference.puertas_lineas`): si aquí fuese otro, un payload legal
#: llegaría al servidor y se quedaría a medio pintar sin decir por qué.
MAX_PUERTAS = 8

#: Versión del fichero de secciones. La 1 es el libro de UNA puerta.
VERSION_LIBRO = 2

#: Cada cuánto, como mucho, se toca el disco.
INTERVALO_GUARDADO_S = 5.0

#: Nombres de los atributos PLANOS del procesador. Se exportan para que
#: `app.py`, `permanencia_area` y la prueba de fuente de A10 hablen todos del
#: mismo texto: si alguien renombra uno a mano, la prueba lo caza.
ATRIBUTO_GESTOR = '_gestor_puertas'
ATRIBUTO_LINEAS = '_puertas_lineas'
ATRIBUTO_NOMBRES = '_puertas_nombres'
ATRIBUTO_ACTIVATE = '_puertas_activate'


def nombre_por_defecto(indice: int) -> str:
    """Cómo se llama una puerta que el operador no bautizó."""
    return f'Puerta {int(indice) + 1}'


def ruta_libro(carpeta_evidencia: Path, archivo_camara: str) -> Optional[Path]:
    """Dónde vive el libro de puertas de una cámara.

    En un subdirectorio propio (`estancias/puertas/<cam>.json`) y NO junto al
    libro de una sola puerta (`estancias/<cam>.json`): con puertas dibujadas
    conviven los dos —el de la cámara sigue siendo el aforo del modo área— y
    dos escritores sobre el mismo fichero se pisarían el libro entero.

    `None` sin nombre de cámara: esta función se llama al conectarse el
    cliente, antes de saber de qué cámara se trata, y sin la guarda cada
    conexión dejaba un `sin-camara.json` que no es de nadie (misma lección
    que `PermanenciaArea._iniciar_permanencia`).
    """
    nombre = str(archivo_camara or '').strip()
    if not nombre:
        return None
    return Path(carpeta_evidencia) / 'estancias' / 'puertas' / f'{nombre}.json'


def clave_dedup(camara: str, tipo: str, pid: Any,
                puerta: Any = None) -> str:
    """La `dedup_clave` de una alerta, con o sin puerta (A8).

    SIN puerta devuelve EXACTAMENTE la clave de siempre
    (`perm|<cam>|<tipo>|<pid>`): sin nada dibujado no cambia ni un carácter
    de lo que ya viaja a WhatsApp y a Jarvis.

    CON puerta, el segmento va DESPUÉS del tipo (`perm|<cam>|<tipo>|p0|<pid>`)
    para no romper a quien filtra por el prefijo `perm|<cam>|<tipo>|`. Y tiene
    que ir: WhatsApp y Jarvis agrupan por esta clave, así que sin el segmento
    dos personas que entran a la vez por puertas distintas la comparten y la
    SEGUNDA muere en el enfriamiento — justo la alerta que demuestra que son
    dos sitios y no uno.
    """
    if puerta is None:
        return f"perm|{camara}|{tipo}|{pid}"
    idx = puerta.indice if isinstance(puerta, Puerta) else int(puerta)
    if idx < 0:
        return f"perm|{camara}|{tipo}|{pid}"     # -1 = aforo de la cámara
    return f"perm|{camara}|{tipo}|p{idx}|{pid}"


# ── Una puerta ──────────────────────────────────────────────────────────
@dataclass
class Puerta:
    """Un umbral con TODO lo suyo: detector, libro, esperas y contadores.

    Nada de esto se comparte entre puertas, y no es un lujo:

      - `DetectorLinea._pasos` está llaveado SOLO por la clave de la persona,
        y `_Paso.lado_firme` es el estado que decide el sentido. Dos puertas
        compartiendo detector se pisarían el `lado_firme` de la MISMA persona
        en el MISMO frame, y el segundo umbral leería el lado del primero.
      - El libro (`registro`) es de la puerta porque una salida por la
        trasera no puede casar con una entrada por la principal: las fundiría
        en una visita a un sitio por el que esa persona no pasó.
      - Las entradas en espera (alertas retenidas por `confirmar_entrada_s`)
        también, o el roce en una puerta tiraría la alerta buena de la otra.

    El libro es de memoria (`ruta=None`, `persistir=False`): quien escribe el
    disco es `LibroPuertas`, en UN fichero con una sección por puerta.
    """

    indice: int
    nombre: str
    linea: List[List[float]]
    detector: DetectorLinea
    registro: RegistroEstancias
    entradas_en_espera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_visitas: int = 0
    suma_duraciones: float = 0.0
    salidas_sin_atribuir: int = 0

    # -- Lecturas ----------------------------------------------------------
    @property
    def dentro_ahora(self) -> int:
        return len(self.registro)

    @property
    def permanencia_media_s(self) -> float:
        """Media de las visitas COMPLETADAS. 0.0 sin ninguna todavía."""
        if self.total_visitas <= 0:
            return 0.0
        return self.suma_duraciones / float(self.total_visitas)

    def __contains__(self, persistent_id: Any) -> bool:
        return str(persistent_id) in self.registro

    # -- Contadores --------------------------------------------------------
    def anotar_visita(self, duracion_s: float) -> None:
        """Una visita que se cerró con su salida vista."""
        self.total_visitas += 1
        self.suma_duraciones += max(0.0, float(duracion_s))

    def anotar_sin_atribuir(self) -> None:
        """Salió alguien y no se supo quién. Se cuenta, no se esconde."""
        self.salidas_sin_atribuir += 1

    # -- Identidad de la estancia -----------------------------------------
    def sellar(self, estancia: Estancia) -> Estancia:
        """Le pone a la estancia POR QUÉ PUERTA fue (`Estancia.puerta`).

        El índice es la identidad y se persiste: sin él, al releer el libro
        tras un reinicio toda la gente caería en el aforo único (-1) y los
        aforos por puerta quedarían a cero con el vídeo perfecto. El nombre
        va al lado solo para mostrar, y por eso se REFRESCA aquí: si el
        operador renombra la puerta a media tarde, quien ya estaba dentro
        tiene que aparecer en el panel bajo el nombre nuevo — no bajo uno
        que ya no existe en ninguna parte.
        """
        estancia.puerta = int(self.indice)
        estancia.puerta_nombre = str(self.nombre)
        return estancia

    # -- Panel -------------------------------------------------------------
    def resumen(self, personas: Optional[Sequence[Any]] = None
                ) -> Dict[str, Any]:
        """La fila de esta puerta dentro de `metadata['permanencia']` (A8).

        `personas` lo arma el llamador (es él quien sabe pintar una fila del
        panel); aquí solo se garantiza la forma y que los números salgan del
        libro de ESTA puerta.
        """
        return {
            'idx': int(self.indice),
            'nombre': str(self.nombre),
            'dentro_ahora': self.dentro_ahora,
            'personas': list(personas or []),
            'visitas_completadas': int(self.total_visitas),
            'permanencia_media_s': round(self.permanencia_media_s, 1),
            'salidas_sin_atribuir': int(self.salidas_sin_atribuir),
        }


# ── El fichero ──────────────────────────────────────────────────────────
def _registro_cambio(registro: RegistroEstancias) -> bool:
    """¿Cambió el libro de esta puerta desde la última escritura?

    Se mira la bandera interna de `RegistroEstancias` a propósito: ese libro
    es de memoria (`persistir=False`), así que su `_sucio` ya no sirve para
    nada suyo y es la única señal de «aquí entró o salió alguien» sin
    envolver `abrir`/`cerrar` en otra capa que se olvidaría de un caso.
    Ante la duda (atributo que no está), se escribe: perder una estancia
    duele más que una escritura de más.
    """
    return bool(getattr(registro, '_sucio', True))


def _registro_limpio(registro: RegistroEstancias, ahora: float) -> None:
    """Marca el libro como ya volcado a disco."""
    try:
        registro._sucio = False                             # noqa: SLF001
        registro._ultimo_guardado = ahora                   # noqa: SLF001
    except Exception:                                       # noqa: BLE001
        pass


class LibroPuertas:
    """El fichero por cámara con una sección por puerta. Nunca lanza."""

    def __init__(self, ruta: Optional[Path] = None,
                 ttl_horas: float = 24.0,
                 persistir: bool = True,
                 intervalo_guardado: float = INTERVALO_GUARDADO_S) -> None:
        self.ruta = Path(ruta) if ruta else None
        self.ttl_s = max(60.0, float(ttl_horas) * 3600.0)
        self.persistir = bool(persistir) and self.ruta is not None
        self._intervalo = max(0.0, float(intervalo_guardado))
        self._ultimo_guardado = 0.0
        #: Versión del último fichero leído (0 = no había). Solo informativa.
        self.version_leida = 0

    # -- Lectura -----------------------------------------------------------
    def cargar(self, ahora: Optional[float] = None
               ) -> Dict[int, Dict[str, Any]]:
        """Secciones del disco: `{indice: {'nombre': str, 'abiertas': [...]}}`.

        Un JSON a medias (corte de luz en mitad de la escritura, aunque sea
        atómica el disco puede mentir) NO puede tumbar el arranque del
        pipeline: se avisa y se empieza vacío.
        """
        secciones: Dict[int, Dict[str, Any]] = {}
        if not self.persistir or self.ruta is None or not self.ruta.is_file():
            return secciones
        try:
            datos = json.loads(self.ruta.read_text(encoding='utf-8'))
        except Exception as exc:                            # noqa: BLE001
            logger.warning("Libro de puertas ilegible (%s): se empieza vacío",
                           exc)
            return secciones
        if not isinstance(datos, dict):
            return secciones

        crudas = datos.get('puertas')
        if isinstance(crudas, dict):
            self.version_leida = int(datos.get('version') or VERSION_LIBRO)
        elif datos.get('abiertas'):
            # v1: el libro de UNA puerta (`RegistroEstancias`). Se adopta
            # entero como la puerta 0 porque eso es lo que era — la única
            # puerta de la cámara. Descartarlo dejaría a la gente que estaba
            # dentro sin libro y sus salidas sin pareja para siempre.
            self.version_leida = 1
            crudas = {'0': {'nombre': str(datos.get('nombre') or ''),
                            'abiertas': datos.get('abiertas')}}
            logger.info("Libro de puertas de %s en formato v1: se adopta "
                        "entero como la puerta 0", self.ruta.name)
        else:
            return secciones

        ahora = time.time() if ahora is None else float(ahora)
        for clave, seccion in crudas.items():
            try:
                idx = int(clave)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= MAX_PUERTAS:
                logger.warning("Libro de puertas: sección %r fuera de rango, "
                               "se ignora", clave)
                continue
            if not isinstance(seccion, dict):
                continue
            abiertas: List[Estancia] = []
            for crudo in (seccion.get('abiertas') or []):
                est = Estancia.desde_dict(crudo) if isinstance(crudo, dict) \
                    else None
                if est is None:
                    continue
                if (ahora - est.entrada_ts) > self.ttl_s:
                    continue                # ya vencida: no resucita
                abiertas.append(est)
            secciones[idx] = {'nombre': str(seccion.get('nombre') or ''),
                              'abiertas': abiertas}
        total = sum(len(s['abiertas']) for s in secciones.values())
        if total:
            logger.info("Puertas: %d estancia(s) seguían abiertas al "
                        "reiniciar en %d puerta(s) de %s",
                        total, len(secciones), self.ruta.name)
        return secciones

    # -- Escritura ---------------------------------------------------------
    def guardar(self, puertas: Mapping[int, Puerta], forzar: bool = False,
                ahora: Optional[float] = None, sucio: bool = False) -> bool:
        """Vuelca las puertas VIVAS. Atómica, amortiguada y muda si falla.

        Solo las vivas: es lo que impide que una puerta borrada resucite.
        `forzar` se usa justo después de quitar una, para que disco y memoria
        no discrepen durante los 5 s del amortiguador.
        """
        if not self.persistir or self.ruta is None:
            return False
        ahora = time.time() if ahora is None else float(ahora)
        cambio = bool(sucio) or any(_registro_cambio(p.registro)
                                    for p in puertas.values())
        if not forzar:
            if not cambio:
                return False
            if (ahora - self._ultimo_guardado) < self._intervalo:
                return False
        try:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            cuerpo = json.dumps({
                'version': VERSION_LIBRO,
                'guardado_ts': ahora,
                'puertas': {
                    str(idx): {
                        'nombre': p.nombre,
                        'abiertas': [e.a_dict()
                                     for e in p.registro.abiertas.values()],
                    }
                    for idx, p in sorted(puertas.items())
                },
            }, ensure_ascii=False)
            # Temporal + `os.replace`: atómico en POSIX y en NTFS. Escribir
            # sobre el fichero bueno deja un JSON a medias si el proceso cae,
            # y ese fichero es el aforo de todas las puertas de la cámara.
            tmp = self.ruta.with_name(self.ruta.name + '.tmp')
            tmp.write_text(cuerpo, encoding='utf-8')
            os.replace(tmp, self.ruta)
            self._ultimo_guardado = ahora
            for p in puertas.values():
                _registro_limpio(p.registro, ahora)
            return True
        except Exception as exc:                            # noqa: BLE001
            logger.debug("No se pudo guardar el libro de puertas: %s", exc)
            return False


# ── El reparto ──────────────────────────────────────────────────────────
class GestorPuertas:
    """Las puertas de UNA cámara: crea, adopta, renombra, mueve y quita.

    Va como atributo PLANO del procesador y FUERA de la lista que
    `TiendasProcess._push_state` multiplexa por cámara (ver A10 arriba).
    """

    def __init__(self, camara: str = '', ruta: Optional[Path] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        self.camara = str(camara or '')
        self.cfg = cfg
        self._ttl_horas = float(cfg.get('estancia_ttl_horas', 24) or 24)
        self._umbral_cuerpo = float(cfg.get('reid_cuerpo_umbral', 0.86)
                                    or 0.86)
        self._max_abiertas = int(cfg.get('max_estancias_abiertas', 200) or 200)
        self._cfg_linea = dict(cfg.get('linea') or {})
        persistir = bool(cfg.get('persistir_estancias', True))
        self.libro = LibroPuertas(
            ruta=ruta, ttl_horas=self._ttl_horas, persistir=persistir,
            intervalo_guardado=float(cfg.get('intervalo_guardado_s',
                                             INTERVALO_GUARDADO_S)))
        #: ÍNDICE -> Puerta. Dict y no lista: ver decisión 1 del módulo.
        self.puertas: Dict[int, Puerta] = {}
        #: Secciones leídas del disco que todavía no reclamó ninguna puerta.
        #: Se consumen al crear la puerta de ese índice (recuperación tras un
        #: reinicio) y las que sobran se cierran como `sin_salida` en la
        #: primera reconciliación con puertas: son gente dentro de un sitio
        #: que el operador ya no tiene dibujado.
        self._por_adoptar: Dict[int, Dict[str, Any]] = self.libro.cargar()
        #: ¿Hubo puertas alguna vez en esta ejecución? Es la guarda que hace
        #: que una cámara SIN puertas no toque nada nunca.
        self._hubo_puertas = False
        self._sucio = False

    # -- Lecturas ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.puertas)

    def __iter__(self) -> Iterator[Puerta]:
        """Siempre por índice: el panel no puede cambiar de orden solo."""
        return iter([self.puertas[i] for i in sorted(self.puertas)])

    def __bool__(self) -> bool:
        return bool(self.puertas)

    @property
    def activo(self) -> bool:
        """Hay al menos una puerta dibujada AHORA."""
        return bool(self.puertas)

    def get(self, indice: Any) -> Optional[Puerta]:
        try:
            return self.puertas.get(int(indice))
        except (TypeError, ValueError):
            return None

    @property
    def total_dentro(self) -> int:
        """Aforo UNIÓN de la cámara: la suma de las puertas.

        La misma persona puede estar dentro de dos puertas a la vez (entró
        por el almacén y por el patio sin que se le viera salir): se cuentan
        las dos estancias, porque son dos sitios y cada uno tiene su
        cronómetro corriendo.
        """
        return sum(p.dentro_ahora for p in self.puertas.values())

    def abiertas(self) -> List[Tuple[Puerta, Estancia]]:
        """Todas las estancias abiertas, con la puerta a la que pertenecen."""
        return [(p, est) for p in self
                for est in list(p.registro.abiertas.values())]

    def buscar_abierta(self, persistent_id: Any
                       ) -> List[Tuple[Puerta, Estancia]]:
        """Dónde está dentro esta persona (puede ser en más de un sitio)."""
        pid = str(persistent_id)
        salida: List[Tuple[Puerta, Estancia]] = []
        for p in self:
            est = p.registro.get(pid)
            if est is not None:
                salida.append((p, est))
        return salida

    def resumen(self, personas: Optional[Mapping[int, Sequence[Any]]] = None
                ) -> List[Dict[str, Any]]:
        """La lista `puertas` de `metadata['permanencia']` (A8)."""
        filas = dict(personas or {})
        return [p.resumen(filas.get(p.indice)) for p in self]

    # -- Reconciliación ----------------------------------------------------
    def poner(self, lineas: Optional[Sequence[Any]] = None,
              nombres: Optional[Sequence[Any]] = None,
              ahora: Optional[float] = None) -> List[Estancia]:
        """Deja las puertas EXACTAMENTE como las manda el frame.

        Args:
            lineas: `puertas_lineas` del payload — hasta `MAX_PUERTAS`
                segmentos `[[x1,y1],[x2,y2]]` en píxeles del frame. La
                POSICIÓN en la lista es el índice, y el índice es la
                identidad de la puerta.
            nombres: `puertas_nombres`. No se valida contra `lineas` (un
                desfase de nombres es cosmético; convertirlo en frame
                descartado sería peor): sobra o falta, cada puerta se queda
                con el nombre que le toque o con el suyo de antes.
            ahora: reloj del frame.

        Returns:
            Las estancias que se quedaron SIN puerta, ya cerradas como
            `sin_salida` (con `salida_ts` e `identidad_por` puestos, igual
            que las de `RegistroEstancias.caducadas`). El llamador solo tiene
            que emitir el aviso:

                for est in gestor.poner(lineas, nombres):
                    self._cerrar_como('sin_salida', est)

            No se dejan caer en silencio: alguien contado dentro de un sitio
            que ya no existe desaparecería del aforo sin una sola línea.
        """
        ahora = time.time() if ahora is None else float(ahora)
        entrantes = self._parsear(lineas, nombres)

        # Cámara sin puertas que nunca las tuvo: no-op TOTAL. Ni fichero, ni
        # cierres, ni log. Es el 99% de los recuadros y el modo de siempre no
        # puede notar que este módulo existe.
        if not entrantes and not self.puertas and not self._hubo_puertas:
            return []

        huerfanas: List[Estancia] = []
        estructura = False

        # 1) Las que ya no vienen. Con dict, quitar la 2 de 4 NO corre los
        #    índices: la gente de la 3 se queda en la 3.
        for idx in sorted(self.puertas):
            if idx in entrantes:
                continue
            huerfanas.extend(self._quitar(idx, ahora))
            estructura = True

        # 2) Crear las nuevas; a las de siempre solo se les mueve la línea y
        #    se les cambia el nombre. NO se toca ni el libro ni los
        #    contadores: renombrar o mover no puede reiniciar un aforo.
        for idx in sorted(entrantes):
            linea, nombre = entrantes[idx]
            puerta = self.puertas.get(idx)
            if puerta is None:
                self.puertas[idx] = self._crear(idx, linea, nombre)
                estructura = True
                continue
            puerta.linea = linea
            if nombre and nombre != puerta.nombre:
                logger.info("Puerta %d de %s renombrada: %s -> %s (el aforo "
                            "no se toca: la identidad es el índice)",
                            idx, self.camara or 'sin cámara',
                            puerta.nombre, nombre)
                puerta.nombre = nombre
                for est in puerta.registro.abiertas.values():
                    puerta.sellar(est)   # el panel, bajo el nombre nuevo
                self._sucio = True

        # 3) Secciones del disco que no reclamó nadie. Solo se resuelven
        #    cuando ya se sabe cómo está el recuadro (hay o hubo puertas):
        #    con la lista vacía en el primer frame no se sabe si el operador
        #    las borró o si el cliente todavía no las mandó, y cerrar el
        #    libro entero por esa duda sería el peor error posible.
        if self.puertas or self._hubo_puertas:
            for idx in sorted(self._por_adoptar):
                if idx in self.puertas:
                    continue
                seccion = self._por_adoptar.pop(idx)
                for est in seccion.get('abiertas') or []:
                    # Se cierra EXACTAMENTE como la de `_quitar` (7-sep-2026):
                    # salida y, sobre todo, `identidad_por = SIN_ATRIBUIR`.
                    # Son dos rutas que producen la MISMA huérfana; si una
                    # deja la identidad de la entrada puesta, el registro
                    # dice de un mismo hecho dos cosas distintas según por
                    # dónde llegó, y el docstring de `poner()` prometía la
                    # de aquí.
                    est.salida_ts = ahora
                    est.identidad_por = SIN_ATRIBUIR
                    huerfanas.append(est)
                if seccion.get('abiertas'):
                    logger.warning(
                        "Puertas [%s]: la sección %d del libro tenía %d "
                        "estancia(s) y ya no hay puerta ahí: se cierran como "
                        "'sin salida'", self.camara or 'sin cámara', idx,
                        len(seccion['abiertas']))
                estructura = True

        if self.puertas:
            self._hubo_puertas = True
        if estructura:
            self._sucio = True
            # Escritura inmediata: el amortiguador de 5 s está para no
            # escribir en cada frame, no para dejar en disco una puerta que
            # el operador acaba de borrar.
            self.guardar(forzar=True, ahora=ahora)
        return huerfanas

    def _parsear(self, lineas: Optional[Sequence[Any]],
                 nombres: Optional[Sequence[Any]]
                 ) -> Dict[int, Tuple[List[List[float]], str]]:
        """`puertas_lineas` -> `{indice: (linea, nombre)}`, sin sorpresas."""
        crudas = list(lineas or [])
        if len(crudas) > MAX_PUERTAS:
            logger.warning("Puertas [%s]: llegaron %d umbrales y el máximo es "
                           "%d: se ignoran los de más",
                           self.camara or 'sin cámara', len(crudas),
                           MAX_PUERTAS)
            crudas = crudas[:MAX_PUERTAS]
        etiquetas = list(nombres or [])
        salida: Dict[int, Tuple[List[List[float]], str]] = {}
        for idx, cruda in enumerate(crudas):
            geo = _linea_geometria(cruda)
            if geo is None:
                # Segmento ilegible (dos puntos iguales, coordenadas basura).
                # Se salta ESE índice y los demás conservan el suyo: colapsar
                # la lista correría las identidades, que es justo el fallo que
                # el dict evita.
                logger.warning("Puertas [%s]: el umbral %d no define un "
                               "segmento, se ignora",
                               self.camara or 'sin cámara', idx)
                continue
            nombre = ''
            if idx < len(etiquetas):
                nombre = str(etiquetas[idx] or '').strip()
            salida[idx] = ([[float(cruda[0][0]), float(cruda[0][1])],
                            [float(cruda[1][0]), float(cruda[1][1])]], nombre)
        return salida

    def _crear(self, indice: int, linea: List[List[float]],
               nombre: str) -> Puerta:
        """Puerta nueva, con lo que hubiera en su sección del disco.

        Adoptar es lo que hace que un reinicio a media tarde no vacíe el
        almacén: la gente que estaba dentro sigue dentro y sus salidas
        encuentran pareja.
        """
        seccion = self._por_adoptar.pop(indice, None) or {}
        registro = RegistroEstancias(
            ruta=None, ttl_horas=self._ttl_horas,
            umbral_cuerpo=self._umbral_cuerpo,
            max_abiertas=self._max_abiertas, persistir=False)
        for est in seccion.get('abiertas') or []:
            # Asignación directa y no `abrir()`: adoptar del disco no es una
            # entrada nueva (no debe tocar `ultima_vista_ts` ni ensuciar el
            # libro para forzar una reescritura idéntica nada más arrancar).
            registro.abiertas[est.persistent_id] = est
        titulo = nombre or str(seccion.get('nombre') or '') \
            or nombre_por_defecto(indice)
        puerta = Puerta(indice=indice, nombre=titulo, linea=linea,
                        detector=DetectorLinea(dict(self._cfg_linea)),
                        registro=registro)
        # Un libro v1 (una sola puerta) trae las estancias con `puerta=-1`,
        # que era el aforo único: se sellan como la puerta 0, que es lo que
        # eran. Sin esto volverían a escribirse con -1 y el aforo de la
        # puerta saldría vacío en el panel.
        for est in registro.abiertas.values():
            puerta.sellar(est)
        logger.info("Puertas [%s]: puerta %d «%s» en pie con %d estancia(s) "
                    "recuperada(s)", self.camara or 'sin cámara', indice,
                    titulo, len(registro))
        return puerta

    def _quitar(self, indice: int, ahora: float) -> List[Estancia]:
        """Saca la puerta y cierra a quien tuviera dentro como `sin_salida`."""
        puerta = self.puertas.pop(indice, None)
        if puerta is None:
            return []
        # Quien tenía la entrada RETENIDA (esperando `confirmar_entrada_s`)
        # no ha sido anunciado todavía. Su visita entera muere en silencio:
        # ni entrada ni salida (7-sep-2026). Emitir solo el «sin salida»
        # mandaba a WhatsApp «entró a las X y nunca se registró su salida»
        # de alguien cuya entrada nadie oyó jamás — un aviso que no se puede
        # ni entender ni comprobar. Es la misma regla que ya aplica el roce
        # que muere antes de confirmarse.
        callados = set(puerta.entradas_en_espera)
        puerta.entradas_en_espera.clear()
        cerradas: List[Estancia] = []
        for pid in list(puerta.registro.abiertas):
            est = puerta.registro.cerrar(pid, ahora, CIERRE_SIN_SALIDA)
            if est is not None and str(pid) not in callados:
                cerradas.append(est)
        self._por_adoptar.pop(indice, None)
        logger.warning("Puertas [%s]: se quitó la puerta %d «%s» con %d "
                       "persona(s) dentro: se cierran como 'sin salida'",
                       self.camara or 'sin cámara', indice, puerta.nombre,
                       len(cerradas))
        return cerradas

    # -- Ciclo por frame ---------------------------------------------------
    def observar(self, ahora: float,
                 visibles: Dict[str, Dict[str, Any]]) -> List[Tuple[Puerta,
                                                                    Any]]:
        """Un `observar` por puerta, con SU línea. `[(puerta, cruce), ...]`.

        Entrar por «Almacén» y salir por «Patio» deja una visita abierta en
        el primero y una salida sin pareja en el segundo. NO es un fallo: son
        dos sitios distintos y esa persona sigue dentro del almacén.
        """
        cruces: List[Tuple[Puerta, Any]] = []
        for puerta in self:
            for cruce in puerta.detector.observar(ahora, visibles,
                                                  puerta.linea):
                cruces.append((puerta, cruce))
        return cruces

    def caducadas(self, ahora: Optional[float] = None
                  ) -> List[Tuple[Puerta, Estancia]]:
        """Estancias pasadas de TTL, puerta a puerta, ya cerradas."""
        ahora = time.time() if ahora is None else float(ahora)
        vencidas: List[Tuple[Puerta, Estancia]] = []
        for puerta in self:
            for est in puerta.registro.caducadas(ahora):
                vencidas.append((puerta, est))
        return vencidas

    def guardar(self, forzar: bool = False,
                ahora: Optional[float] = None) -> bool:
        """Vuelca el libro. Se puede llamar en cada frame: está amortiguado."""
        if not self.puertas and not self._hubo_puertas:
            # Cámara sin puertas: NO se crea el fichero. Un JSON vacío que no
            # es de nadie confunde al mirar el disco (la lección del
            # `sin-camara.json`).
            return False
        escrito = self.libro.guardar(self.puertas, forzar=forzar, ahora=ahora,
                                     sucio=self._sucio)
        if escrito:
            self._sucio = False
        return escrito
