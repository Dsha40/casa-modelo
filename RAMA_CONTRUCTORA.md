# Rama `contructora`: cambios

Registro de lo que se hace en esta rama, del más antiguo al más nuevo. Cada
entrada dice qué cambió, por qué y cómo se comprobó.

## De dónde sale

- **Creada el 30-oct-2026** desde `revision-del-proyecto`, en el commit
  `261dec4` («Grafo del detector: no encender, apaga los de MiVOLO y
  ViTPose»). Tiene todo lo de esa rama hasta ese día.
- **Solo en local**: no está subida a `fork`. `revision-del-proyecto`
  también tiene 15 commits sin subir.
- **Objetivo, por ahora**: clasificar (género y edad) e identificar a la
  gente. Los anaqueles (áreas, agarres, productos, stock) y la pose de la
  muñeca (ViTPose) no se necesitan.

## 30-oct-2026

### Se borró `espera_gpu`

- Qué: `server/src/app/espera_gpu.py`, `server/tests/test_espera_gpu.py` y
  sus 9 líneas en `server/iniciar_servidor_headless.py`.
- Por qué: nunca se commiteó, y la A/B del 29-oct lo descartó. Sin su
  variable (`ELDE_ESPERA_GPU`) no hacía nada.
- Se borró antes de cambiar a esta rama, así que la rama nace sin él.

### Se usa el modo «Tiendas» en vez de «Anaqueles»

- Qué: en el cliente, «Cerrar conexión con el servidor», elegir **Tiendas**
  en «Tipos de inferencias» y volver a encender la IA del recuadro. El
  código no cambia.
- Por qué: todo lo que hace falta está en el modo Tiendas
  (`server/src/analityc/core/tiendas.py`):
  - el clasificador B y la segunda opinión de género;
  - la identidad por la cara;
  - el uniforme del personal;
  - la costura de seguimientos y la revisión de cruces.

  «Anaqueles» es Tiendas más las áreas, los agarres, el stock y la pose.
  ViTPose solo lo carga Anaqueles (`anaqueles.py` crea `PoseMunecas`).
- Qué se gana: el servidor no carga ViTPose (~0,5 GB de GPU por cámara y
  ~18 s de carga). Además se ahorra la pose, que costaba ~20 ms por imagen
  con alguien junto al estante (medido el 6-oct en Anaqueles; en Tiendas,
  sin medir todavía).

### Lo que NO se ha hecho todavía: borrar el código de anaqueles

Se deja dormido, sin usar. No se borra porque varias piezas que sí se usan
toman funciones de `anaqueles_registro.py` (`_carpeta_nombre` y
`raiz_salida`) para saber dónde guardar sus archivos:

- `clasificador_uniforme.py`, `costura_sombra.py`, `personal_mapa.py` y
  `revision_cruces.py`;
- los informes (`informe_pasillo.py` e `informe_pdf.py`) y `api_en_vivo.py`.

Para limpiarlo habría que:

1. pasar esas funciones a un archivo propio;
2. quitar «Anaqueles» del selector y su pestaña del cliente;
3. borrar el resto y pasar las pruebas.

### Funcionando en modo Tiendas (09:29)

- Cámara 4 (`win-iVMS-4200-c4-i58bf7b`), iVMS en vivo: **~17-18 imágenes
  por segundo**, con el servidor tardando ~50 ms por imagen. En Anaqueles
  iban ~13-14, pero esa cifra no sale del mismo tramo, así que solo orienta.
- Sin ViTPose. Los únicos avisos, de los que hay al cerrar una conexión.
- **Fallos vistos, pendientes:**
  1. «Cerrar conexión con el servidor» dio una excepción a las 09:22:20
     (`socket_client.py:157`: el temporizador de reconexión ya estaba
     destruido) y no dejaba cambiar de modo. Hubo que cerrar el cliente.
  2. Tras cerrarlo a la fuerza, el recuadro volvió sin la ventana de iVMS
     elegida: había que volver a elegirla.
  3. La primera imagen de una cámara nueva tardó más de 5 s, porque el
     servidor la estaba preparando. El cliente la dio por perdida, la
     reenvió y el servidor tiró 3 imágenes en esos 30 s. Después se
     normalizó.

### Clasificación e identidad: más intentos, 24 h y aprender la cara

Lo pidió el usuario: aprovechar el tiempo libre (sin pose y con menos que
dibujar) para clasificar e identificar mejor.

**Lo que se encontró leyendo el código:**

- Desde el 5-oct, las cadencias se cuentan en segundos a 7,7 imágenes por
  segundo (`_imagenes`). A 17 imágenes por segundo se clasificaba igual de
  seguido que a 7,7, así que el tiempo ganado no se usaba.
- Cuando un seguimiento ya tenía identidad, sus caras siguientes **no se
  guardaban** (`face_reidentifier.py`, la rama rápida de
  `identify_or_register`). La galería se quedaba con la cara de la primera
  vez.

**Cambios** (todos en `server/src/analityc/core/`):

| Qué | Antes | Ahora | Dónde |
|---|---|---|---|
| Intentar clasificar a cada persona | cada ~12 imágenes a 17 imágenes/s (~0,65 s) | cada 5 imágenes reales (~0,3 s) | `DEMO_CADENCIA_EN_IMAGENES`, `TiendasProcess._cada` |
| Huella del cuerpo | cada ~27 imágenes (~1,6 s) | cada 6 imágenes reales (~0,35 s) | `BODY_REID_EVERY_N`, `BODY_REID_CADENCIA_EN_IMAGENES` |
| Huellas del cuerpo por persona | 5 | 10 | `BODY_REID_MAX_EMB_PER_PERSON` |
| Cuánto recuerda una cara | 30 min | **24 h** desde la última vez que se la vio | `REID_TTL_S` |
| Dónde vive la galería de caras | solo en memoria: un reinicio la borraba | también en disco, `server/output/person_db/persons.pkl` (fuera de git); se reescribe cada 30 s como mucho | `REID_PERSISTIR_CON_TTL`, `REID_GUARDAR_CADA_S` |
| Caras de referencia por persona | 5 | 15 | `REID_MAX_EMBEDDINGS_PER_PERSON` |
| Aprender la cara durante la visita | no | sí (detalle abajo) | `analytics/aprender_cara.py` (nuevo) |
| Buscar en la galería | un bucle de Python (~15 ms con ~2.000 caras, estimado de la medida del 21-sep) | una sola multiplicación de matrices, mismo resultado | `_best_in_gallery` |

**Aprender la cara**: mientras dura la visita, se añade una cara de
referencia más, con cuatro condiciones:

- como mucho una cada segundo y por seguimiento;
- solo de frente o casi (la misma pose que se exige para registrar a
  alguien nuevo);
- solo si se parece a las que ya tiene la persona, con un parecido de 0,40
  o más. Si el número de seguimiento salta a otra persona, su cara no se
  mezcla;
- si ya hay 15, la nueva sustituye a la más parecida a ella, para
  conservar la variedad.

No se entrena ningún modelo: el de caras (ArcFace) ya sabe comparar, y lo
que aprende es la colección de caras de cada persona. El 0,40 está **sin
calibrar**: cada 60 s el log escribe una línea `CARA APRENDE` con cuántas
se añadieron, cuántas se rechazaron y sus parecidos.

**Riesgos conocidos:**

- Con la galería en disco, **dos clientes conectados a la vez** se pisarían
  el archivo; es el fallo que en septiembre llevó a dejarla solo en
  memoria. Con un solo cliente no pasa.
- Si el servidor se cae, se pierden como mucho los últimos 30 s.
- Clasificar y tomar la huella más seguido gasta más servidor: hay que
  medir cuántas imágenes por segundo se pierden.
- La carpeta `server/output/person_db/faces/` ya tenía 1.073 fotos de caras
  (de antes de este cambio). Cada reinicio dejaba huérfanas las fotos de la
  galería en memoria. Está pendiente decidir si se borran.

**Pruebas**: `server/tests/test_identidad_24h.py` (nuevo, 17 pruebas) y
`test_reid_ttl.py` (el defecto pasa a 24 h). Se comprobaron al revés:

- sin aprender, fallan 4;
- sin comprobar el parecido ni la pose, fallan 2.

Suite completa del servidor: **1162 passed, 15 skipped, 1 xfailed**.

**Medida**: base en modo Tiendas, sin estos cambios, desde las 09:39. Los
cambios entran al reiniciar el servidor. Se compararán en el mismo tiempo de
tienda: identificados, clasificados, duplicados e imágenes por segundo.

### Servidor con los cambios (09:59) y grabador de tramos con gente (10:05)

- 09:59: el servidor arranca con los cambios de arriba. El log dice «TTL
  de 1440 min (galería en disco: 0 personas cargadas)». Con la cámara 4
  vacía: 15-18 imágenes por segundo y ~52 ms por imagen, igual que antes.
- **Para medir limpio** hay que pasar las MISMAS imágenes a las dos
  versiones (el usuario: cámaras y ratos distintos no se comparan). Por
  eso se añadió un grabador nuevo: `server/src/app/grabador_personas.py`,
  enganchado en `app.py` (`_full_frame_sync`).
  - Graba solo cuando hay seguimientos activos, más 2 s antes y 3 s
    después, en una carpeta por tramo.
  - Guarda el JPEG tal como llegó del cliente (sin recomprimir) y los
    parámetros de cada imagen en `imagenes.jsonl`.
  - Se enciende con `ELDE_GRABAR_PERSONAS=<carpeta>`; sin la variable no
    hace nada.
  - Para solo al llegar a `ELDE_GRABAR_PERSONAS_MAX_MB` (5000 por defecto)
    o si quedan menos de 6 GB libres.
  - Pruebas: `server/tests/test_grabador_personas.py` (7). Sin el margen de
    antes y después fallan 2.
- 10:05: el servidor arranca grabando en
  `server/output/_diagnostico/grabacion_personas_30oct` (fuera de git). Son
  **fotos de clientes**: se borran con permiso del usuario.
- Falta el reproductor que pase un tramo grabado por el servidor dos veces
  (cambios apagados y encendidos), y un interruptor que apague todos los
  cambios a la vez.

### La cámara 4 no ve caras: identidad por el cuerpo (10:25-10:45)

**El problema** (lo vio el usuario): una persona que entraba y salía contaba
como otra cada vez, y con otro género.

**Lo medido en el log (10:15-10:23):**

- 2 caras aceptadas frente a ~175 rechazos; 154 de ellos por «persona
  demasiado pequeña» (cajas de 44-94 × 107-137 px, el mínimo es 60×140).
- 1 sola identidad creada.
- La imagen es la ventana entera de iVMS (1280×800). Lo que pasa es que la
  cámara está alta y lejos: en una foto del tramo de las 10:21 la persona
  mide ~110×230 px y su cara ~25 px.
- Sin cara no hay identidad: cada entrada es un `prov:<número>`, y el
  género se decide por separado en cada una.

**Decisión del usuario:** «no se va a ver bien la cara, vamos a usar el
cuerpo».

**Cambios:**

- `server/src/analityc/core/identidad_cuerpo.py` (nuevo). Un seguimiento
  sin cara y sin parecido con nadie de la galería del cuerpo junta 3
  huellas.
  - Si se parecen entre sí (coseno mínimo entre pares ≥ 0,80), recibe una
    identidad nueva y la próxima vez que entre se le reconoce por el
    cuerpo.
  - Si un día se le ve la cara, su galería del cuerpo pasa a la identidad
    de la cara.
  - Hasta este día, el cuerpo NUNCA creaba identidades.
  - Interruptores: `BODY_ID_CREAR`, `BODY_ID_MIN_HUELLAS` y
    `BODY_ID_COHERENCIA_MIN` (config.py), o `ELDE_BODY_ID=0`.
  - Cada 60 s el log escribe `CUERPO ID`, con las identidades creadas y el
    parecido de cada búsqueda, para calibrar el 0,80 y el 0,90 del cuerpo.
- `BODY_REID_TTL_S`: de 4 h a 24 h, como la cara.
- `server/src/analityc/core/voto_identidad.py` (nuevo). Un solo género por
  persona: cada seguimiento decidido es un voto, y la visita y el aviso de
  «visitante» se quedan con la mayoría. Antes ganaba el último. Enganchado
  en `tiendas_fotos.py` y en `genero_v2.py`. La tabla de clasificaciones
  sigue guardando el género de cada seguimiento, sin tocar.
- Pruebas: `server/tests/test_identidad_cuerpo.py` (12). Se comprobaron al
  revés: sin la comprobación de que las huellas se parecen, falla 1; con
  «gana el último», falla 1. Suite completa: **1181 passed, 15 skipped,
  1 xfailed**.
- **Riesgo:** dos personas con ropa parecida se juntan (el personal de
  uniforme, sobre todo).

**Antes de encenderlo en vivo**, se reproducen los tramos grabados (16, de
10:15 a 10:34; 4.387 imágenes) fuera de línea con
`scratchpad/reproducir_tramos/reproducir.py`, para ver quién se juntó con
quién y revisarlo en las fotos.

### Calibración con los tramos grabados y cambios de las 11:35

- El usuario confirmó que los tramos de las 10:xx son **una sola
  persona**. Se reprodujeron 18 tramos (10:15-10:39; 54 pasadas por la
  cámara; 686 huellas) y se simularon 108 combinaciones de umbrales
  (`scratchpad/reproducir_tramos/simular.py`).
  - Con 0,90 fijo, la misma persona salía como **11 o más** identidades:
    se parecía a sí misma solo 0,42-0,88.
  - Con **0,70 si se la vio hace menos de 10 min y 0,80 si hace más**,
    salen **2** identidades, con 47 de 54 pasadas reconocidas.
  - Esta prueba no mide si se juntaría a dos personas distintas: en los
    tramos solo había una.
- Cambios (`identidad_cuerpo.buscar`, `config.py`):
  - `BODY_REID_SIMILARITY_THRESHOLD` 0,90 → 0,80.
  - `BODY_REID_UMBRAL_RECIENTE` = 0,70 y `BODY_REID_VENTANA_RECIENTE_S` =
    600.
  - Cada pasada reconocida refresca la galería del cuerpo.
- **El cuerpo solo crea identidades a quien estuvo dentro del cuadro de
  visitas (ROI)**, si está activo (pedido del usuario: la gente de la
  puerta principal no es una visita). Fuera no se crea, aunque sí se
  reconoce a quien ya la tenía.
- **El aviso de «visitante» y las fichas NO dependen del ROI** (leído en
  `tiendas_fotos.py`): salen con cualquiera que se detecte y se clasifique.
- Pruebas: `test_identidad_cuerpo.py` (16; al revés fallan 2). Suite
  completa: **1185 passed, 15 skipped, 1 xfailed**. En vivo desde las 11:35.
- **Ojo:** el ROI de la cámara 4 ocupa casi toda la imagen, de (12,70) a
  (1259,725) en una de 1280×800: así no deja fuera la puerta.

### Solo cuentan como visita las identidades nacidas en el ROI (12:05)

**Lo que quiere el usuario** (aclarado tras un malentendido mío):

- El ROI es grande a propósito: cubre la zona por la que entra la gente
  (la entrada principal queda lejos) para dar tiempo a crear bien su
  identidad.
- Después la gente recorre la casa y entra en cuartos que la cámara no
  ve. Al volver, el reconocimiento puede fallar y crearle otra identidad.
- Esa nace fuera del ROI y **no es una visita**.

**Cambios:**

- Se deshace el filtro de las 11:35: el cuerpo vuelve a crear
  identidades en todas partes.
- `server/src/analityc/core/origen_identidad.py` (nuevo) apunta si cada
  identidad nació dentro o fuera del ROI:
  - por el cuerpo, al crearse;
  - por la cara, si es nueva (`first_seen` de hace menos de 10 s);
  - al pasar de cuerpo a cara, la cara hereda el origen.
- Solo las nacidas dentro cuentan para:
  - las entradas y salidas (`_process_zone_transitions`, que llena
    `tienda_visitas`);
  - los visitantes únicos (`_confirmed_unique`);
  - la fila de la visita (`anotar_demografia`);
  - el aviso de «visitante».
- La tabla de clasificaciones sí guarda a todos.
- Sin origen apuntado (por ejemplo, una cara que viene del disco tras un
  reinicio, porque este registro vive en RAM), la identidad cuenta: es lo
  que pasaba antes.
- Interruptor: `VISITAS_SOLO_NACIDAS_EN_ROI` o `ELDE_VISITAS_SOLO_ROI=0`.
- Pruebas: en `test_identidad_cuerpo.py` (17); al revés falla 1. Suite
  completa: **1186 passed, 15 skipped, 1 xfailed**. En vivo desde las
  12:05.
- Las personas sin identidad (`prov:`) siguen igual que antes: no van a
  las visitas, pero su aviso de «visitante» sí sale si se les decide el
  género.

### Fuera del ROI se dejaba de seguir a la gente (12:43)

**Lo que vio el usuario en vivo (12:22):** dentro del ROI todo iba bien,
pero quien salía de él dejaba de seguirse y de clasificarse. Al volver al
ROI desde dentro de la casa contaba como otra visita. A esa hora el ROI era
la mitad derecha (x 738-1277), la de la entrada.

**Dos causas en `tiendas.py`, de antes de esta rama:**

1. «Filtro ROI»: las detecciones con el centro fuera del ROI se tiraban
   antes de seguirlas.
2. `process_entry_exit_logic` (la contabilidad antigua) **borraba el
   seguimiento** de quien estuvo dentro en cuanto salía
   (`exit_frames_threshold = 1`), y también al «contarlo». Al borrarse,
   perdía su identidad del cuerpo y su marca de «estuvo dentro». A la
   imagen siguiente se recreaba y se volvía a borrar.

**Arreglo:** `ROI_FILTRA_DETECCIONES = False` (config.py). Se sigue a todos
en toda la imagen y nadie se borra por salir del ROI: solo muere quien deja
de verse. El ROI sigue contando las entradas y decidiendo qué identidades
son visitas. Con `True` vuelve todo lo de antes.

**Para diagnosticar sin modo Anaqueles:** `GET /api/v1/tienda/imagen-actual`
(`server/src/app/api_imagen_actual.py`, nuevo). Devuelve la última imagen
analizada con el ROI en verde y cada persona con su número, su identidad,
de dónde viene, si nació dentro o fuera y su género.

**Visto en las grabaciones:**

- A las 12:22 había 3 personas a la vez. El cuerpo juntó a dos de ellas
  (parecido 0,73, con el umbral «reciente» de 0,70). Pendiente: que una
  identidad no pueda tenerla un segundo seguimiento visible a la vez.
- Los recuadros rojos de las imágenes los dibuja iVMS sobre la gente, y
  ensucian las huellas. Conviene apagarlos en iVMS.
- El reloj de la cámara 4 va 1 h por detrás (11:22 a las 12:22).

**Pruebas:** `server/tests/test_roi_no_filtra.py` (5). Suite: **1191
passed, 15 skipped, 1 xfailed**. En vivo desde las 12:43.

### Una persona, una identidad a la vez (13:00) y revisión de calor y permanencia

- **Regla nueva** (`identidad_cuerpo.buscar`, parámetro `ocupadas`): una
  identidad que ya tiene otro seguimiento visible no se le puede dar a un
  segundo seguimiento por el cuerpo. Habría evitado lo de las 12:22, cuando
  se juntaron dos personas distintas con parecido 0,73.
  - Pruebas: 2 nuevas en `test_identidad_cuerpo.py`; quitando la regla
    falla 1. Suite: **1193 passed**. En vivo desde las 13:00.
- **Mapa de calor: funciona.**
  - El de la cámara 4 acumula: 5.137 muestras a las 12:05 y 5.755 a las
    12:43.
  - Se vuelca a disco en cada reinicio.
  - `/dashboard/img/heatmap/<cámara>.png` lo devuelve bien.
  - Hasta las 12:43 solo sumaba a la gente dentro del ROI; desde entonces,
    a todos.
  - Las leyendas siguen saliendo con «??» (fallo cosmético, de antes).
- **Permanencia: funciona, pero mide el tiempo DENTRO DEL ROI**, no en la
  casa. Ejemplo: `f2b6103d`, 4 entradas, 80 s en total. Con el ROI sobre la
  entrada, es el tiempo en la zona de la entrada.

### Permanencia en toda la imagen (13:07)

- El usuario pidió que la permanencia no se mida dentro del ROI sino en
  toda la imagen.
- `PERMANENCIA_TODA_LA_IMAGEN = True` (config.py), en
  `_process_zone_transitions`:
  - «dentro» = se le ve en cualquier parte de la imagen;
  - entrada = se le empieza a ver; salida = se le deja de ver;
  - la permanencia es el tiempo visible en la cámara.
- El ROI solo decide quién es visita (`origen_identidad`).
- PermanenciaArea, que cronometra su propia zona, no cambia
  (`_zona_toda_la_imagen = False`).
- Si alguien entra en un cuarto que la cámara no ve, cuenta como salida, y
  al volver como otra entrada de la misma persona (si se le reconoce). El
  tiempo en el cuarto no suma.
- Pruebas: `server/tests/test_permanencia_toda_imagen.py` (4), que hacen
  funcionar la máquina de verdad: con la opción apagada, quien está fuera
  del ROI no entra. Suite: **1197 passed**. En vivo desde las 13:07.

### Primera prueba buena en vivo (13:15) y un error mío con la base

- **Lo que pasó:** de 13:07 a 13:26, las visitas se guardaron en la
  **base de pruebas** (`scratchpad/tienda_pruebas.db`), no en
  `server/output/tienda.db`. Causa: lancé la suite y el reinicio en el
  mismo comando de PowerShell, y el servidor heredó `ELDE_TIENDA_DB`.
- **Arreglo:** el servidor se reinició solo a las 13:26 y los scripts de
  reinicio ya quitan esa variable.
- **Lo que hay en la base de pruebas** (prueba del usuario a las 13:15,
  «funcionó con uno»):

  | Identidad | Primera | Última | Entradas/salidas | Permanencia | Género |
  |---|---|---|---|---|---|
  | `acb02e06` | 13:15:10 | 13:21:43 | 2 / 2 | 7,8 s | Mujer |
  | `4b8d393f` | 13:15:41 | 13:15:45 | 1 / 1 | 4,5 s | — |
  | `6c565f5f` | 13:16:04 | 13:16:07 | 1 / 1 | 2,6 s | — |

  Las tres nacieron dentro del ROI. `acb02e06` (cuerpo) se convirtió en la
  cara `4b8d393f` a las 13:15:40: es la misma persona, pero quedó como dos
  filas, porque las visitas ya guardadas con la identidad del cuerpo no se
  pasan a la de la cara (pendiente).
- **Comprobado al reproducir un tramo fuera de línea:** la máquina de
  entradas y salidas y la permanencia en toda la imagen funcionan: entrada
  al aparecer, salida al dejar de verse, 12,7 s.

### Revisado en imagen (13:15-13:16)

- **Cómo revisar:** con las imágenes, no solo con el log
  (`/api/v1/tienda/imagen-actual` en vivo y los tramos grabados para lo ya
  pasado). Lo pidió el usuario.
- 13:15: un hombre de camisa verde. Su identidad de cuerpo `acb02e06` pasó
  a su cara `4b8d393f`. Es la misma persona: **bien unida**.
- 13:16: una niña de blanco. Identidad nueva `6c565f5f`: es **otra
  persona**, así que **bien** (lo confirmó el usuario).
- **Error de género:**
  - El clasificador B dijo «Hombre» del hombre de verde, y acertó.
  - A las 13:21:43, la segunda opinión (`genero_v2.py`, ResNet-18 propia,
    encendida por defecto desde el 27-oct) lo cambió a **«Mujer»**, con
    0,992 de seguridad.
  - Ese modelo se entrenó y se validó con fotos del supermercado (cámaras
    2 y 3). En la casa modelo (cámara alta, otra luz y otra gente) no está
    validado.
  - Ficha: `20261030_131509_iVMS-42004_t100002.json`.
- **Decisión del usuario (13:48):** nuestro modelo solo decide cuando el
  principal NO decidió. Cambios:
  - `GENERO_V2_SEGUNDA_OPINION = False` (config.py) y
    `genero_v2.segunda_opinion()`;
  - `ELDE_GENERO_V2_SEGUNDA=1` la vuelve a encender;
  - las indecisas (≥ 0,95) siguen igual.
- Pruebas: `test_genero_v2.py` (7; la de las 13:15, nueva). Con el defecto
  de antes falla 1. Suite: **1198 passed**. En vivo desde las 13:48.
- La ficha del hombre de verde ya está escrita como «Mujer»; el cambio
  vale para las fichas nuevas.

### «Se queda pegado cuando llega gente» (14:22)

- **Lo medido:** la PRIMERA persona tras cada reinicio dejaba el servidor
  sin responder ~20-25 s.
  - Avisos del cliente «sin respuesta en 5 s» a las 13:14:36, 13:29:35 y
    13:56:44.
  - Huecos de 4,2-4,8 s en los tramos grabados de esa primera persona.
  - Mientras tanto, la gente cruzaba el ROI sin verse: a las 13:56 entraron
    tres personas y sus identidades nacieron fuera, así que no contaron.
- **No era la grabación:** el grabador solo cuesta ~0,2-0,3 s al empezar
  cada tramo, cuando escribe los 2 s previos.
- **Causa** (cProfile con el tramo de las 13:56, en frío): la primera
  llamada de cada sesión ONNX.
  - OSNet 29,2 s; YuNet 6,3 s; MiVOLO 1,9 s, al grabar su grafo.
  - Con torch cargado antes que onnxruntime (como en el servidor), la
    primera llamada de una sesión nueva de OSNet tarda ~26 s; sin torch,
    1,3 s.
  - No es la búsqueda de cuDNN (con HEURISTIC, igual) ni la caché de CUDA.
  - Es por sesión: una segunda sesión del mismo modelo vuelve a tardar lo
    mismo.
  - La causa exacta dentro de las librerías no se ha averiguado.
- **Arreglo:** `server/src/analityc/core/calentar_modelos.py` (nuevo). Al
  crear el procesador (arranque o reconexión del cliente), un hilo llama
  una vez con ceros a cada sesión ONNX.
  - En frío, la primera imagen con persona pasa de **38 s a 1,2 s** (lo que
    queda es el grafo de MiVOLO).
  - En vivo (14:21): «CALENTAR: listo en 33,9 s», a la vez que se carga el
    detector.
  - Interruptor: `CALENTAR_MODELOS` o `ELDE_CALENTAR=0`.
- **Sigue igual:** durante el primer minuto tras un reinicio (modelos
  cargando) no se procesa a nadie.

### Esperar en vez de crear otra identidad (14:22)

- 13:35: una mujer medio tapada por el panel de la derecha se partió en dos
  seguimientos. El segundo no podía recibir su identidad (la tenía el
  primero, visible), así que se creó otra (`4f2abe01`) con una ficha
  sobrante.
- Ahora `identidad_cuerpo.buscar(..., info=)` avisa si la huella se parece
  a una identidad OCUPADA por encima del umbral. El seguimiento queda en
  espera (`_espera_ocupada`) y no crea otra. Cuando el otro número
  desaparece, se enlaza con esa identidad.
- Riesgo: dos personas distintas con ropa muy parecida, a la vez en la
  imagen: la segunda se queda sin identidad mientras dure.
- Pruebas: `test_calentar_modelos.py` (5) y 2 nuevas en
  `test_identidad_cuerpo.py`; al revés fallan 3. Suite: **1205 passed**.

### Nuestro modelo de género solo revisa lo que sirve (15:00)

- A las 14:04 se guardó una visita de `7a846898`, identidad nacida fuera
  del ROI. La escribió `genero_v2` al decidir el género de su ficha 6 min
  después, sin mirar el origen.
- El usuario: si no es una visita, revisarla es cómputo perdido. Ahora
  `genero_v2.hace_falta(meta)` solo deja pasar el modelo si:
  - la ficha tiene identidad;
  - esa identidad cuenta como visita (nació dentro del ROI);
  - el resultado puede cambiar algo: género sin decidir o, con la segunda
    opinión encendida, «Hombre».
- Con la segunda opinión apagada, antes se pasaba el modelo también a las
  fichas ya decididas, y el resultado no se usaba.
- Pruebas: 1 nueva en `test_genero_v2.py`; al revés falla. Suite: **1206
  passed**. En vivo desde las 15:00.
- La fila de `7a846898` de las 14:04 (0 entradas, sin eventos) se borró
  de la base real a las 15:0x, a petición del usuario.
- Informe del día para el usuario:
  `informes/informe_2026-10-30_casa_modelo.md` (local).

### Laboratorio de seguimiento e identidad (15:30-16:40)

- `scratchpad/lab/`:
  - `volcado.py` pasa todas las grabaciones del día por el detector y la
    huella del cuerpo del servidor (13.891 detecciones);
  - `simular.py` repite seguimiento, costura e identidad con variantes;
  - `calibrar.py` mide cuánto se parece la misma persona y cuánto personas
    distintas.
- Lo real lo dio el usuario: A 1, C 3, D 2, E 1, F 3 personas.
- **Resultado:** el modelo del cuerpo (OSNet market1501) no separa a las
  personas en esta cámara.
  - Por huella: misma persona, mediana 0,73; distintas, 0,62.
  - Con 0,80, el 1,2 % de las huellas de personas distintas pasa, y como
    se prueba cada huella, se acaban juntando.
  - Con la mediana de 5 huellas y 0,85 no junta a nadie, pero la empleada
    de A sale en 13 identidades.
  - Ninguna variante (completitud de la caja, galería con huellas tapadas,
    solape de ByteTrack 0,9 o 0,95, costura de 6 s) arregla esto. El 0,9
    parecía bueno, pero en la imagen la identidad de la empleada de A
    quedaba puesta a 5 personas.
  - La mayoría de los cortes son reales: la gente sale de la imagen y
    vuelve.
- **Decisión:** no cambiar nada en vivo. Siguiente paso: probar
  `osnet_x1_0_dukemtmcreid.pt` (ya está en models/osnet) y afinar el
  modelo del cuerpo con datos de esta cámara. `torchreid` está instalado.
- Tarde (16:17-16:28), revisado en las fotos:
  - **un cuadro de la pared contó como visita** («Mujer», 20 entradas);
  - el señor de camisa granate no tuvo identidad;
  - la niña del vestido azul recibió la identidad del señor de la chaqueta
    reflectante.
  - Arreglo pendiente del usuario: zonas a ignorar sobre el cuadro y el
    espejo.
- Vídeos para mirar: `server/output/_diagnostico/videos_30oct/` (mp4 por
  tramo, con la hora del PC).

### Recámara principal y cámaras de solo permanencia (18:14)

- El usuario movió la cámara a la **recámara principal**: es ahora la
  cámara de visitas. Llega con el MISMO nombre (`win-iVMS-4200-c4-...`) y
  con el ROI del salón: hay que redibujarlo.
- Otras cámaras: **solo permanencia** (`SOLO_PERMANENCIA: true` en su
  perfil de `server/config/camera_profiles/`).
  - Solo detecta y sigue: sin clasificar, sin identidades, sin fichas,
    avisos ni visitas.
  - Apunta las estancias en `server/src/analityc/core/permanencia_cuarto.py`
    (nuevo).
  - Un corte de hasta 5 s cerca del sitio donde se perdió la persona es la
    misma estancia.
  - Se descartan las estancias de menos de 2 s y los objetos quietos.
  - Escribe en `output/permanencia_cuartos/<cámara>/<fecha>.json`, con un
    resumen: estancias, total y promedio.
  - Se consulta en `GET /api/v1/tienda/permanencia-cuartos?fecha=`.
  - Mapa de calor del cuarto, sí.
- Probado con un tramo real: 0 identidades, 0 clasificaciones, estancias
  guardadas.
- Pruebas: `test_permanencia_cuarto.py` (7); al revés fallan 2. Suite:
  1212 passed. `test_tiempos_etapas` falló una vez con el PC cargado y
  pasa sola 3 de 3 (mide tiempos reales).
- En vivo desde las 18:14, con el grabador encendido.

### Fin de semana 30-oct a 1-nov y perfiles de c2 y c3 (2-nov)

- El servidor estuvo encendido del 30-oct a las 18:14 al 1-nov a las
  09:22, cuando se reinició el PC.
- Tres cámaras (c2, c3 y c4 = recámara), a ~3,7 imágenes por segundo cada
  una.
- Visitas el sábado 31 (el 30 fue viernes): c2 24, c3 24, c4 22.
- 123 tramos grabados (2,95 GB en total).
- c2 y c3 aún NO eran de solo permanencia: contaron visitas, identidades y
  géneros. Una persona que pasa por varias cámaras cuenta en cada una, así
  que no se suman.
- c2 y c3 tenían los perfiles del supermercado (c2, el stand de
  degustación; c3, el pasillo del café con un mínimo de 150 px). El 2-nov
  se sustituyeron por `SOLO_PERMANENCIA: true` y nada más.
- Comprobado con `_load_camera_profile`: c2 y c3 en solo permanencia con
  los valores globales; c4 sin perfil (todo global).
- Se aplica al arrancar el servidor.

### Una foto por persona, no por zona de la imagen (2-nov, 10:51)

- **Problema:** 9 de las 33 visitas de c4 del 30 y 31-oct no tenían foto
  de c4. La causa principal es el dedup espacial del supermercado: una
  foto por cuadrito de la imagen cada 60 s. En la recámara todos entran
  por la misma puerta, así que quien entraba detrás del primero se quedaba
  sin foto. Medido: en 5 de 7 visitas sin foto del sábado hubo otra foto
  de c4 entre 10 y 40 s antes. Las otras 2 (11:33 y 17:46) no tenían foto
  previa: causa sin aclarar.
- **Cambio:** `CAPTURE_UNA_POR_PERSONA = True` (`config.py`, usado en
  `tiendas_fotos.py`):
  - con identidad confirmada, se hace la foto aunque la zona esté cubierta;
    una vez por identidad, como antes;
  - sin identidad todavía, se reintenta en las imágenes siguientes en vez
    de descartarse;
  - quien nunca recibe identidad sigue con el dedup de zona;
  - `False` vuelve a lo de antes.
- **Prueba:** `test_una_foto_por_persona.py` (5). Verificada al revés: con
  el código anterior fallan las 2 del problema.
- Servidor reiniciado a las 10:51 con el cambio.
- **Ojo al revisar fotos:** la identidad es común a las tres cámaras. Para
  ver las fotos de c4 hay que filtrar por `camera = 'iVMS-4200 (4)'` en
  `tienda_capturas`; si no, salen fotos de c2 y c3.
- El reloj de la cámara (rótulo de iVMS) va 1 h por detrás del PC.

### Parar y reanudar la grabación sin reiniciar (2-nov, 10:59)

- Lo pidió el usuario para probar con vídeos sin grabarlos.
- `api_grabador_personas.py` (nuevo) y `grabador_personas.py`
  (`parar`, `reanudar`, `estado`):
  - `GET /api/v1/tienda/grabador`: cómo está;
  - `POST /api/v1/tienda/grabador/parar`: cierra el tramo abierto y deja
    de grabar;
  - `POST /api/v1/tienda/grabador/reanudar`: vuelve a grabar, salvo si se
    paró por estar lleno o falta disco.
- Un reinicio vuelve a lo que diga `ELDE_GRABAR_PERSONAS`.
- Pruebas: 6 nuevas en `test_grabador_personas.py` (18 en total con las
  de la foto por persona).
- Servidor reiniciado a las 10:59; comprobado que responde `grabando: true`.

## Ideas para más adelante (sin hacer)

### Duración total de la visita, incluidos los cuartos que la cámara no ve

- **Qué:** medir la visita desde la PRIMERA vez que se ve a la persona hasta
  la ÚLTIMA, aunque entre medias haya estado en cuartos que la cámara no ve.
  Hoy (`PERMANENCIA_TODA_LA_IMAGEN`) solo se suma el tiempo visible: cada
  vez que entra en un cuarto cuenta como salida, y ese rato no suma.
- **Por qué interesa:** en la casa modelo la gente recorre cuartos fuera de
  cámara. El tiempo visible se queda corto frente a lo que de verdad duró la
  visita.
- **Cómo se podría hacer:** se guarda la hora de la primera y de la última
  vez que se vio a cada identidad, como un dato aparte de la permanencia
  actual (que se queda igual). Ya existe algo parecido: `tienda_visitas`
  tiene `ts_primera` y `ts_ultima` por persona, jornada y cámara.
- **Lo que depende de esto:** que al volver del cuarto se la reconozca como
  la misma persona (identidad por el cuerpo). Si se le crea otra identidad,
  esa nace fuera del ROI y no cuenta: la visita se quedaría en la hora en
  que entró al cuarto.
- **Por decidir:** cuánto tiempo sin verla se considera que ya se fue de la
  casa y empieza otra visita (por ejemplo, 30 min).

### Estado que hereda la rama

- **Grafo de CUDA del detector**: apagado por defecto. No hay que encenderlo
  hasta arreglar `GrafosPorLote._grabar` (ver la cabecera de
  `detector_grafo.py`). Si no, desactiva los grafos de MiVOLO y ViTPose.
