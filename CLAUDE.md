# CLAUDE.md — ELDE Tiendas (contexto de trabajo)

Traspaso de las sesiones del 16-18 sep, 21-22 sep y 23-sep 2026. **Este
archivo SÍ se commitea**: vive en la rama para que la siguiente sesión lo lea
sin que el usuario tenga que pegarlo.

Rama de trabajo actual: **`fix/gender`** (publicada), que cuelga de
`claude/pensive-galileo-aymjr3` → `fix/eliminar_vigilantes_amazonas` (no de
`main`). Al cambiarte a ella heredas todo lo anterior.

**El 23-sep se midió por primera vez sobre PRODUCCIÓN REAL** (14 h de log y la
base del usuario), y eso corrigió conclusiones que venían de muestras cortas.
Donde este archivo dé una cifra, mira su fecha: las del 23-sep mandan.

---

## EL MODO DE TRABAJO CAMBIÓ (21-sep) — léelo antes que nada

Las sesiones del 16-18 sep corrían en el Windows del usuario, sin acceso al
servidor. **Ya no.** Ahora Claude corre en un **contenedor remoto con el
repositorio clonado**, y eso cambia casi todo lo operativo:

| | Antes (16-18 sep) | Ahora (21-22 sep) |
|---|---|---|
| Acceso al repo | solo Windows | contenedor propio, efímero |
| Suite de pruebas | «aquí no corre» | **corre entera**: 767 pasan |
| Cambios | el usuario los copiaba a mano | Claude **commitea y publica** |
| El usuario | pegaba logs | hace `git pull` y pega logs |

**Lo que NO cambió:** el servidor de inferencia sigue corriendo en el Linux
del usuario (`/home/D-arias365/proyectos/ELDE-TIENDAS`, contenedor
`51b31f634ba9`), Claude **no tiene acceso a él**, y el usuario **no puede usar
Claude y el servidor a la vez**. Toda validación en vivo llega como log pegado.

### Cómo montar el entorno de pruebas en el contenedor

No viene nada instalado. Esto deja la suite **entera en verde** (el índice de
PyTorch está bloqueado por el proxy; **torch hay que sacarlo de PyPI**):

```bash
# La version de OpenCV NO es opcional: ver abajo.
pip install numpy pytest "opencv-contrib-python-headless==4.12.0.88" fastapi \
            msgpack supervision torch httpx onnxruntime rfdetr psutil \
            py-cpuinfo python-multipart reportlab timm torchreid gdown \
            tensorboard
# Los pesos van por Git LFS y git-lfs NO viene en el contenedor:
apt-get install -y git-lfs && git lfs install --local && git lfs pull
cd server && python3 -m pytest tests -q
```

**Línea base medida el 23-sep: `792 passed, 15 skipped, 0 failed`.**

La línea base vieja («767 passed, 4 failed, los 4 son del contenedor») **está
obsoleta**: 3 fallos eran por `reportlab` y 1 por los pesos de Git LFS. Con
esas dos cosas instaladas **no hay ningún fallo aceptable**. Si ves uno, es
tuyo.

**Dos trampas que costaron horas el 23-sep:**

1. **`opencv-contrib-python-headless` SIN versión te da la 5.x, y eso tumba
   TODO el stack facial**, no solo lo que quita. `tiendas_modelos.py:278` carga
   todos los modelos dentro de **un solo `try/except`**, y el primer `readNet`
   de los modelos Caffe viejos revienta en OpenCV 5 («Caffe importer has been
   removed») llevándose por delante YuNet, InsightFace, ArcFace y OSNet. El
   síntoma es demoledor: el servidor arranca «bien» y no reconoce a nadie.
   Comprobación: `p._reidentifier.is_available` y `p._demographics.is_available`
   deben dar `True`.
2. **Sin los pesos de Git LFS no hay pipeline.** Se descargan sin problema
   desde el contenedor; lo que falta es el binario `git-lfs`.

Comprobación oficial del README (debe imprimir la línea, sin reventar):

```bash
python3 -c "import cv2, torch, onnxruntime, rfdetr, transformers, supervision, torchreid; assert hasattr(cv2,'ximgproc'); print('cv2', cv2.__version__)"
```

`verificar_entorno.py` sale con **código 1 en el contenedor** (no hay GPU) y
con **0 en el Linux del usuario**. No es un fallo: el script hace su trabajo.

---

## Qué es este repositorio (léelo antes de tocar nada)

**Es un maquetado de otro repo.** Se sacó del monorepo ELDE (vigilante
perimetral, estacionamiento, comedor, stand de degustación y cuatro clientes
de escritorio) y **se está ajustando a un proyecto nuevo de tiendas**. Por eso:

- Mucho código heredado **probablemente no funcione aquí, y es esperable**: lo
  que depende de `vigilante_amazonas`, WhatsApp, Jarvis, los modos Stand y
  Permanencia, los DVR, o los tests que nombran a otros clientes. Que algo
  heredado falle **no es una regresión**.
- Antes de arreglar o reconstruir algo heredado, **pregunta si el proyecto
  nuevo lo necesita**. No lo reconstruyas sin que te lo pidan.
- Lo que se usa hoy: el cliente `clients/tienda` capturando ventanas de
  iVMS-4200, el modo `Tiendas`, el servidor en Linux y `/dashboards/tienda/`.

### El caso de uso real

**Un pasillo de anaqueles de supermercado.** Pasa mucha gente. La gente
**mira los productos, no la cámara**, y las caras del fondo miden ~20 px.

**Estado real del montaje (23-sep) — no asumas que las cuatro funcionan:**

| Cámara | `camera_id` | Estado | Personas en base |
|---|---|---|---:|
| 1 | `win-iVMS-4200-i58bf7b` | estable | 203 |
| 2 | `win-iVMS-4200-c2-i58bf7b` | estable | 89 |
| 3 | `win-iVMS-4200-c4-i58bf7b` | **intermitente**: se cae y vuelve | 19 |
| 4 | — | **no funciona** (error `OpenSDK 105` de reconexión) | 0 |
| prueba | `win-HD_CCTV_Camera_...` | vídeo de YouTube, NO es una cámara | 3 |

Las 19 personas de la c4 frente a 203 y 89 son la huella de la intermitencia.

**Las imágenes van DESFASADAS entre cámaras** (observado por el usuario, no
medido). Importa: enlazar a la misma persona entre dos cámaras es más fácil si
ambas la ven a la vez. Es candidato a explicar por qué solo 47 de 292
apariciones se enlazan, pero **no está comprobado**.

El vídeo de prueba está doblemente comprimido y **rechaza casi todas las
caras** (rechazos por nitidez: 26,8 % ahí frente a 3,1 % en la cámara real).
Si el usuario pega un log, **pregunta de qué fuente es antes de concluir**.

---

## LO PRIMERO: lee estos archivos tú mismo

**Es de suma importancia.** Al empezar la sesión, lee **completos y con tus
propias herramientas** estos cuatro archivos. **No los delegues en un agente**
y no te conformes con un resumen ajeno: el usuario lo exige. Verificado el
22-sep que los cinco existen.

1. `README.md` (raíz)
2. `clients/tienda/README.md`
3. `clients/tienda/EQUIPO.md`
4. `GUIA.md`

Según el tema, lee también tú (no un agente) `server/README_DEMOGRAFIA.md`.

**Para lo demás puedes usar agentes**: búsquedas amplias, explorar muchos
archivos. Lo que devuelvan es un resumen: antes de afirmarlo, verifícalo tú
con `archivo:línea`.

---

## Cómo trabajar con este usuario

- **En español.** Si pide «sin tecnicismos», explica con palabras llanas.
- **«Sin alucinaciones» es su exigencia n.º 1.** Separa siempre lo *verificado*
  (con `archivo:línea` o una medida) de lo *deducido* y de lo que *no sabes*.
  Mide antes de afirmar. **En la sesión del 21-sep hubo que retractar cuatro
  afirmaciones** (ver «Errores cometidos» abajo): él las detecta.
- **Explica el procedimiento COMPLETO antes de los comandos**, en orden y con
  los pasos numerados. El 22-sep se quejó, con razón, de recibir un `grep` con
  un `<tu-log>` sin explicar y un `mv` suelto al final sin contexto.
- **Avísale cuando un comando vaya a bloquear la terminal.** Un
  `python servidor.py | grep ...` deja el servidor en primer plano y parece
  colgado. Ya pasó.
- **Los procesos largos asustan.** Si la suite tarda 2 minutos, dilo antes.
  El 21-sep interrumpió una ejecución creyendo que estaba trabado.
- **Respeta los nombres que elija** (renombró `fallo_pipeline` a
  `fallo_imagen_IA_pipeline`).
- **Nunca uses credenciales pegadas en el chat.** Ya pegó un token de GitHub
  (lo revocó). Que se autentique él.
- **PIDE PERMISO ANTES DE CADA COMMIT Y DE CADA PUSH.** Lo exigió el 22-sep,
  después de que una sesión commiteara y publicara dos arreglos por su cuenta
  amparándose en la línea que había aquí («los push los haces tú ahora»). Su
  motivo: no quiere que se le llene el historial de GitHub. Puedes **editar**
  los archivos libremente; lo que no haces sin su visto bueno es fijar el
  cambio. El procedimiento, cada vez:
  1. enséñale **qué archivos** tocas y el resumen del diff,
  2. enséñale **el mensaje de commit completo**, tal y como iría,
  3. **espera su respuesta**. Sin ella, ni `commit` ni `push`.
- Si tienes que reescribir un commit ya publicado (`--amend`, rebase),
  **avísale** además de pedir permiso: él tendrá que hacer
  `git reset --hard origin/<rama>` en vez de un `pull` normal.

### Convenciones del repositorio

- **Los comentarios explican el *porqué*** (qué se rompió, cuándo, cómo se
  midió). Es el estándar, no ruido. Mira cualquier fichero.
- **Una funcionalidad, un fichero.**
- **Commits en español**, asunto corto en imperativo y cuerpo con el porqué y
  cómo se comprobó. El usuario pide además que el commit **liste los archivos
  tocados y lleve la fecha**.
- `app.py`, `conftest.py` y `pyrightconfig.json` se comparten con un monorepo
  donde **sí** existe `vigilante_amazonas`: los cambios tienen que funcionar
  con y sin ese paquete.
- Los arneses de medición van al **scratchpad, nunca al repo**, y nunca se
  toca `output/`.

---

## Errores cometidos (no los repitas)

Cinco de las sesiones del 21-22 sep y dos del 23, todos detectados y
corregidos:

1. **`git checkout -- <archivo>` sobre trabajo sin commitear lo BORRA.** Se
   usó para deshacer una línea de prueba y se llevó el cambio entero. Si
   necesitas revertir una modificación temporal, guarda el contenido antes o
   usa un script que reponga exactamente lo que quitaste.
2. **`por_id={}` del log de arranque NO es de los perfiles de cámara.** Es de
   los perfiles de **ángulo** (`_camera_angle_by_id`). Se citó como prueba de
   que el perfil no cargaba; la conclusión era correcta, la evidencia no.
3. **`COUNT(*)` sobre `tienda_visitas` cuenta FILAS, no personas.** La tabla
   tiene `UNIQUE (jornada, camera_id, persistent_id)`: una persona en dos
   cámaras son **dos filas**. Para personas, `COUNT(DISTINCT persistent_id)`.
4. **Las cifras de `CALIDAD FACIAL` son acumuladas**, no instantáneas (ver
   abajo). Se compararon dos líneas de sesiones distintas como si fueran
   mediciones independientes.

5. **Commitear y publicar sin preguntar.** Ocurrió el 22-sep con dos arreglos
   correctos y probados: el problema no fue el código, fue el procedimiento.
   El `CLAUDE.md` decía «los `git push` sí los haces tú ahora» y se tomó por
   permiso permanente. **No lo es.** Ver la regla, con sus pasos, en «Cómo
   trabajar con este usuario».

6. **Recomendar el modelo PAR fiándose de una telemetría rota.** El indicador
   `rama_corporal_activa` decía `False` mientras la rama corporal había
   aceptado 235 muestras, y el contador `body_sin_modelo` sugiere «falta un
   modelo» cuando cuenta «el modelo corrió y no dio resultado válido». Con esos
   dos se le dijo al usuario que generara un PAR **que el código nunca habría
   usado**. Arreglado en `576e53a`. Lección: **una bandera de telemetría es
   código como el resto y puede estar desactualizada**; contrástala con un
   contador que no dependa de ella.
7. **Concluir «el muro es la pose, 48 %» desde 51 segundos de vídeo.** Con 14 h
   de producción el reparto real es **37,7 % tamaño y 13,6 % pose**. El tramo
   corto tenía gente pasando cerca. **No saques repartos porcentuales de
   muestras de minutos**: saca el dato del log acumulado, que cubre jornadas.

Y uno de método que sí salió bien y conviene repetir: **verificar los tests al
revés**. Desactiva el arreglo y comprueba que la prueba falla; luego
restáuralo y comprueba que pasa. Un test que pasa siempre no demuestra nada.

---

## Entorno

| | |
|---|---|
| **Contenedor de Claude** | Repo clonado, efímero. Sin dependencias: instálalas (arriba). Lo que no se commitea, se pierde |
| **Linux del usuario** | Contenedor Docker `51b31f634ba9`, repo en `/home/D-arias365/proyectos/ELDE-TIENDAS`, venv `.venv`. **Ahí corre el servidor. No tienes acceso** |
| **Windows del usuario** | Cliente de escritorio + edición |
| **Limitación** | El usuario **no puede usar Claude y el servidor a la vez** |

### Dónde están los logs y los datos (verificado el 22-sep)

| Qué | Dónde |
|---|---|
| **Log del servidor** | `server/logs/server.log` — rota a 5 MB, 3 copias. Lo fija `iniciar_servidor_headless.py:52-59` (`raiz_proyecto=server/`, `client_type='server'`) |
| Nivel del log | `ELDE_LOG_NIVEL=DEBUG\|INFO\|...` (defecto INFO), `ELDE_LOG_DIR`, `ELDE_LOG_MB`, `ELDE_LOG_COPIAS` |
| **Base de datos** | `server/output/tienda.db` (`ELDE_TIENDA_DB`). Tablas `tienda_visitas`, `tienda_eventos`, `tienda_capturas` |
| Galería de rostros | `server/output/person_db/` — **ya no se usa con la política por defecto** (ver TTL) |
| Heatmaps | `server/output/heatmap/<device_id>.png` — **el 21-sep la carpeta no existía**: nunca se ha generado uno |
| Registro de dispositivos | `server/output/registro_dispositivos.json` — la lista de `device_id` vistos |
| Config del cliente | `%LOCALAPPDATA%\tienda_view\tienda_view\config.json` — **cifrada a propósito, no la descifres**. Ahí está `instalacion.id` |

---

## Cómo funciona (recorrido de un frame)

1. `clients/tienda/src/main.py` elige servidor
   (`elde_core/red/arranque_servidor.py`) y abre `gui/windows_main.py`.
2. Recuadro = `gui/components/render_box/render_box.py`. Fuente: una ventana
   (`elde_core/capture/capture_worker.py`) o un DVR (`rtsp_worker.py`).
3. Con la IA encendida hay **un frame en vuelo por recuadro**.
4. `camera_id` = `elde_core/ui/identidad_camara.py`, función `device_id()`:
   ventana `win-<título>[-cN]-i<instalación>`; DVR `dvr-<serie>-<canal>`; si
   no, `box-N`.
5. Servidor `server/src/app/app.py`: contrato, cola del último frame **por
   cámara** (`ws_drenaje.py`, reparto round-robin), `_full_frame_sync` →
   `process_image_sync` → `TiendasProcess.process_frame`.
6. Pipeline: RF-DETR (`deteccion_compartida.py`) → ByteTrack **por cámara** →
   identidad (`analytics/face_reidentifier.py`) → entradas/salidas con
   histéresis → demografía (`analytics/demographics.py`) → `tienda_almacen.py`
   (SQLite, por jornada).
7. Respuesta: el mismo mensaje con `data` sustituido. En modo directo solo
   viajan `metadata.detections` y el cliente las pinta.
8. Lectura: `/dashboards/tienda` ← `/api/v1/tienda/*` (`api_tienda.py`);
   `/dashboard` antiguo (`dashboard.py`).

### Mecanismos que conviene saber

- **Estado por cámara:** `_push_state`/`_pop_state` (`tiendas.py:1109`,
  `:1127`) cuelgan el estado **solo mientras dura un frame**. Leer
  `proc._zone_state`, `_total_entries`, etc. desde fuera da el estado vacío.
- **Perfiles por cámara:** `config/camera_profiles/<camera_id>.json`.
  **Desde el 21-sep ya no se busca por nombre exacto**: `_resolver_perfil`
  (`demographics.py:100`) prueba el id completo y luego el id sin el sufijo de
  instalación. El nombre del fichero se **sanitiza a `[alnum_-]` y se trunca a
  64** (`_ruta_perfil`, `demographics.py:91`): el punto de `.net` desaparece y
  los ids largos se cortan. **No lo deduzcas a ojo, calcúlalo.**
- **Identidad multicámara:** la galería facial es **una sola compartida** entre
  las cámaras de una conexión. `camera_track_namespace`
  (`face_reidentifier.py:863`) da un offset por cámara (100000, +100M…) para
  que los `track_id` locales de ByteTrack no colisionen. **La cara crea
  identidades (umbral 0,35); el cuerpo solo re-enlaza; sin cara queda `prov:`.**
- **Un procesador por cámara**, con su propio tracking y contadores, pero
  comparten conexión, detector y memoria de caras. Los frames se procesan
  **por turnos**: los FPS se reparten.
- **Géneros:** `demographics.py:176` define `_GENDER_LIST = ['Hombre','Mujer']`.

---

## Qué se resolvió

### Sesiones 16-18 sep — PR `fix/eliminar_vigilantes_amazonas` (`ae11664` + `e01ecde`)

**Imagen congelada en modo IA.** Dos fallos encadenados:
`deteccion_compartida.py` importaba `vigilante_amazonas` (ausente aquí) → el
detector nunca se construía; y los pesos de `models/` eran punteros de Git LFS
(`git lfs install` + `git lfs pull`). Lo volvía silencioso que `tiendas.py`
atrapaba el error y `app.py` respondía `success` sin `detections`.

Arreglado con `ajustes_rfdetr()`, `_pesos_vivos()`, el nuevo
`src/app/fallo_imagen_IA_pipeline.py` y 10 tests.

### Sesión 21-22 sep — rama `claude/pensive-galileo-aymjr3`

**`2644a68` — Caducar la galería facial por TTL y dejarla en RAM.**

La galería venía del stand, donde interesaba recordar mucho tiempo:
`REID_RESET_POLICY` era `"never"` y el fichero solo crecía. Medido en este
repo: `_best_in_gallery` recorre toda la galería en un bucle de Python, y pasa
de **3,6 ms con 100 personas a 67,6 ms con 2 000 y 170,6 ms con 5 000** —
contra un presupuesto de 66 ms por frame a 15 fps, y pagándose **una vez por
cada cara del frame**. Con ~200 personas/hora se llega a ~2 400 en un día.

- `REID_RESET_POLICY` admite `"ttl"` y **es el defecto**; `REID_TTL_S = 1800`.
  Por entorno: `ELDE_REID_RESET_POLICY`, `ELDE_REID_TTL_S`.
- `_caducar_locked` purga por `last_seen` (no `first_seen`: quien sigue en el
  pasillo no caduca), borra el jpg y limpia `_track_to_person`.
- Con TTL **la galería no toca disco** y no se registra `atexit`.
- Cierra un fallo reproducido: `_build_processor` corre **una vez por conexión
  websocket**, así que dos clientes tenían dos galerías escribiendo el mismo
  pickle; `_sync_external_deletions` solo propaga borrados, de modo que cada
  guardado sobrescribía a la otra y **al apagar el fichero quedaba VACÍO**.
- 18 pruebas nuevas (`server/tests/test_reid_ttl.py`), que incluyen la primera
  cobertura del namespace de `track_id` por cámara.

**Confirmado en producción:** `REID TTL: caducaron 2 personas (ttl=60s)`.

**`5ab054e` — Resucitar los perfiles de cámara huérfanos.**

`win-iVMS-4200.json` y `win-iVMS-4200-c2.json` se calibraron el 7-8 de agosto
con mediciones reales. El 9-sep `device_id()` estrenó el sufijo de instalación
(`-i<marca>`) y **dejaron de cargarse**: el cliente pedía
`win-iVMS-4200-i58bf7b` y el fichero se llamaba `win-iVMS-4200.json`.

`_resolver_perfil` reintenta sin el sufijo. 10 pruebas nuevas
(`server/tests/test_perfil_camara_instalacion.py`).

**Confirmado en producción (22-sep 14:16):**
`Perfil demografico camara win-iVMS-4200-c2-i58bf7b aplicado
(win-iVMS-4200-c2.json)` con 13 claves. Y la telemetría pasó a decir
`umbral=16.0` donde antes decía `25.0`.

---

### Sesión 23-sep — rama `fix/gender`

**`f7af573` — Conectar el voto de género, que nunca contaba un voto.**

`_maybe_update_demographics` agrega el género por voto ponderado, pero el
filtro pedía `('Hombres','Mujeres','Ninos')` —el vocabulario de
`result['category']`, que sirve para PINTAR— y `demographics.py` manda
`'Hombre'/'Mujer'`. La condición no se cumplía nunca: `votos_genero` quedaba
vacío y la consolidación por mayoría **no llegaba a ejecutarse**.

El efecto no era solo perder la mayoría: sin votos el género quedaba
**CONGELADO en la primera lectura** (la peor: de lejos y de perfil), y encima
se propagaba, porque `demographics.py:1371` impone a propósito el valor
guardado al reconocer a alguien. **La edad no tenía ese filtro y por eso sí se
corregía sola.**

Se cambió el vocabulario del FILTRO, no la llamada: pasar `category` habría
metido un tercer valor en un campo de dos que leen **~30 sitios** (dashboard,
informe PDF, la columna `genero` de `tienda_visitas`) y habría mezclado
'Hombre' con 'Hombres' en la base. Va en la constante `_GENEROS_VOTABLES`
porque fue la duplicación la que permitió que se desincronizara. 11 pruebas.

**`576e53a` — Contar MiVOLO v2 en la bandera de la rama corporal.**

`rama_corporal_activa` miraba solo `_par` y el MiVOLO ONNX viejo, no
`_estimador_cuerpo` (MiVOLO v2), que desde el Hito 4 es **la ruta principal** y
la que `_infer_body_only` prueba PRIMERO. 4 pruebas, una de ellas fija ese
ORDEN sobre el código: si alguien lo invirtiera, generar el PAR volvería a
tener sentido y la recomendación de no hacerlo dejaría de ser cierta.

**Confirmado en producción (23-sep):** `REID TTL: caducaron 1 personas
(ttl=1800s)` — con el TTL REAL de 30 min, no con los 60 s de prueba. Eso cierra
el pendiente «validar el TTL con los 30 minutos reales».

---

## Lo que se aprendió midiendo

Ordenado de menos a más fiable: primero lo de muestras cortas (21-22 sep),
después **lo medido sobre producción el 23-sep, que es lo que manda**.

### La telemetría `CALIDAD FACIAL` es la métrica buena

```
CALIDAD FACIAL | aceptadas=28 blur_min=37.7 (umbral=25.0) cero_borrosas=True |
rechazos={'bbox_pequeno': 59, 'ensemble_tta': 43, 'body_sin_modelo': 21,
          'sin_cara': 20, 'pose': 13, 'calidad': 6, 'body_conf_baja': 1}
```

Cómo leerla, con las trampas:

- **Es acumulada desde que arrancó el procesador**, no instantánea. Para
  comparar sesiones, usa **la última línea de cada una**.
- **Es por cámara** (cada `cam_proc` tiene su clasificador) pero **la línea no
  dice qué cámara es**. Con varios recuadros abiertos no sabrás cuál es cuál.
  `blur_min` sirve de huella: valores parecidos = misma cámara.
- `umbral=` refleja el perfil aplicado. **Si dice 16.0, el perfil está
  cargado; si dice 25.0, no.** Es la comprobación más rápida.
- Comando: `grep "CALIDAD FACIAL" server/logs/server.log | tail -3`

### El cuello de botella NO es la nitidez, es el tamaño de la cara

`DEMO_MIN_BLUR_VAR` se mide sobre la cara **ya alineada a 112×112**
(`_align_face_arcface` → `warpAffine`), no sobre el píxel nativo. Reproducido
con la foto real del pasillo: la misma cara escalada da

```
cara nativa:  60px → blur 206 |  35px → 34,6 |  28px → 16,1
              25px →  10,2    |  21px →  6,6 |  18px →  3,8
```

Los valores del log (4,5 – 16,5 en el vídeo de prueba) encajan clavados con
caras de 18-28 px. **El umbral de 25 equivale a exigir ~32-35 px de cara
nativa.** No mide nitidez: mide tamaño disfrazado.

Y por debajo hay un muro que ningún perfil salva — `DEMO_ADAPTIVE_BRACKETS`
(`config.py`): **`<25 px nativos se rechaza directo`**.

La foto real del pasillo da **blur 2 915 en la cara nativa** y ~310 tras
alinear: la cámara real está bien. El problema es la distancia, no la imagen.

### LO QUE MANDA: 14 h de producción (23-sep)

Log de `00:10` a `14:28`, 13.082 líneas, 18.800 intentos sobre dos cámaras.
**Tasa de aceptación: 8 %.** Esto sustituye a cualquier reparto sacado de
muestras de minutos.

| Motivo | Casos | % |
|---|---:|---:|
| **tamaño** (`bbox_pequeno` + `body_bbox_pequeno`) | **6.523** | **37,7 %** |
| `sin_cara` | 3.357 | 19,4 % |
| `pose` | 2.355 | 13,6 % |
| `body_sin_modelo` (NO es «falta el modelo») | 1.904 | 11,0 % |
| `calidad` | 1.339 | 7,7 % |
| `ensemble_tta` | 935 | 5,4 % |
| `body_conf_baja` | 872 | 5,0 % |

**Cuando falla la pose, SIEMPRE es giro horizontal:** perfil 3.161 + girada 970
+ yaw 1.565 = 5.696, contra pitch 14 y roll 0. Arriba/abajo e inclinación no
cuentan. `DEMO_DEBUG_REJECTIONS` está activo, así que el log guarda el motivo
de cada cara: `grep -c 'RECHAZO eye_ratio' server/logs/server.log`.

Ojo al encadenar números: una cara rechazada por perfil **cae luego a la rama
corporal** y solo si esa también falla cuenta como `sin_cara`
(`demographics.py:1180-1186`). No sumes pose + sin_cara.

### LA CAUSA Nº 1: el listón, no la imagen (23-sep)

Cada captura guarda por qué no se pudo estimar el perfil. Sobre 1.343 fichas:

| Motivo | Casos | % |
|---|---:|---:|
| **`muestras_insuficientes`** | **718** | **53 %** |
| `sin_rostro` | 489 | 36 % |
| `calidad_insuficiente` | 67 | 5 % |
| `rostro_muy_pequeno` | 52 | 4 % |

`muestras_insuficientes` = se le vio la cara, se tomaron muestras y **nunca
llegaron al listón**: `DEMO_MIN_SAMPLES=5`, `DEMO_MIN_CONFIDENCE=0.85`,
`DEMO_MIN_MARGIN=0.35`, `DEMO_COMMIT_AGREE_RATIO=0.90` (`config.py`, en modo
«precisión máxima»; el perfil de cámara tiene PROHIBIDO tocarlos).

**Es la palanca más rentable del proyecto: la causa mayoritaria y la única que
se mueve cambiando números.** No se ha probado todavía.

### EL FALLO MÁS GORDO: el panel y la base no exigen lo mismo (23-sep)

En `tiendas_fotos.py:552-566`, la alerta al cliente se emite **sin condición**
y el guardado en base solo **`if _pid_demo`** (el `person_uuid`). Y
`person_uuid` solo se rellena por la vía de re-identificación facial
(`demographics.py:911` y `:1378`).

Resultado medido sobre 1.474 capturas:

```
con género .... 131      →   con identidad ...  12
                             SIN identidad ... 119   ← se pierden
```

**Se pierde el 91 % de lo clasificado.** El panel enseña 131 y los informes
salen de 12. **Por qué falta la identidad NO se ha medido**: el campo
`origen_demografia` sale vacío en las 131 fichas.

Decisión pendiente del usuario: `tienda_visitas` exige `persistent_id`, así que
guardar a los 119 obliga a elegir entre inventarles uno (y perder el conteo
único) o guardarlos aparte como «visita sin identificar».

### Identidad entre cámaras: funciona a escala (23-sep)

| Vista por | Personas |
|---|---:|
| 1 cámara | 245 |
| 2 cámaras | 43 |
| 3 cámaras | 4 |

**47 personas enlazadas entre cámaras**, no 1 como el 22-sep. La parte difícil
funciona. Y por eso mismo **el doble conteo ya está medido**: sumar por cámara
da **314**, unir identidades da **261** → **53 personas de más, un 20 %**.
Eso cierra los hallazgos 1 y 13, que llevaban desde el 18-sep sin medirse.

**Ojo con `tienda_capturas`:** es una **caché perezosa** que solo se llena al
abrir el panel de capturas. Si sale vacía no significa que no haya capturas.
La fuente de verdad son los sidecars JSON en disco.

**Límite estructural:** para *registrar* hace falta cara semi-frontal
(`REID_MAX_YAW_FOR_REGISTRATION = 20°`). En un pasillo la gente mira el
producto.

## Hallazgos pendientes (18-sep, revisados el 23-sep)

Antes de citar uno, **comprueba que el código sigue igual**. Los que llevan
✅ se resolvieron; los demás siguen vivos con sus líneas actualizadas.

1. **Persona en dos cámaras — MEDIDO el 23-sep, sigue sin arreglar.**
   `tienda_almacen.demografia` (`:568`) suma por cámara en vez de unir. El
   coste real: **314 frente a 261, 53 personas de más, 20 %**. Ningún test lo
   cubre. Va de la mano del 13.
2. **Jarvis:** `render_box.py` rehace la alerta sin `person_uuid` ni
   `camera_name`, y `jarvis_forwarder.py` agrupa todas con `('obj','?:?')`.
   Solo importa si usan Jarvis.
3. **DVR por IP arrastrados:** `camera_id` `dvr-sin_nombre-<canal>`, igual en
   equipos distintos. No afecta a las ventanas de iVMS.
4. **Panel Anaqueles:** `anaqueles_panel.py:55-56` traduce `male/female` y le
   llega `Hombre/Mujer`, así que el feed dice «Persona». *Verificado 22-sep.*
5. **«Local: el del pie»:** ese selector ya no existe y
   `selected_establishment` no se rellena (`jarvis_forwarder.py:230`). Sin
   local elegido no se envía nada a Jarvis. *Verificado 22-sep.*
6. **Errores que el cliente nunca ve:** `_send_error` responde sin `event` y
   `socket_client.py` lo descarta. En la ruta por ventana no hay reintento.
7. **`status: "error"`:** el cliente imprime el aviso en cada frame (~11/s) y
   no repinta. Aplazado por el usuario.
8. **Votos de género por persona:** `face_reidentifier.py:754` y `:816` solo
   aceptan `'Hombres'/'Mujeres'/'Ninos'`, y `demographics.py:176` define
   `['Hombre','Mujer']`. *El desajuste de literales sigue en el código
   (verificado 22-sep); no se ha comprobado en ejecución qué valor llega.*
9. **Menores:** `main.spec` pide `core.run_controller` (no existe);
   `get_and_test.py` importa `models.dvr_storage` (está en `elde_core/dvr/`).
10. **Aforo en vivo (deducido):** `api_tienda.py` lee el estado del procesador
    fuera de un frame (ver «Estado por cámara»). Se espera 0 o parpadeo.
11. ✅ **Perfiles de cámara huérfanos** — resuelto en `5ab054e` y confirmado en
    producción.
12. **Modo Stand** (el cliente de tienda no lo ofrece): `amazonas_stand.py` lee
    `_demographics` después del pop. Se espera que no confirme degustaciones.

### Hallazgos nuevos del 21-22 sep

13. **`dashboard.py:244` — MEDIDO el 23-sep, sigue sin arreglar.** Sin
    clientes conectados hace `visitantes = sum(...)` en vez de unir UUIDs. La
    rama en vivo sí lo hace bien (`:228`, `:238`). Mismo 20 % del hallazgo 1.
    La API `/api/v1/tienda` **sí lo hace bien** (`tienda_almacen.py:465`,
    `COUNT(DISTINCT persistent_id)`): el que falla es el dashboard viejo.
14. **`_best_in_gallery` es O(N) en Python puro** y `_cosine_similarity`
    renormaliza vectores que ya vienen normalizados. Un producto matricial da
    **250-300× más rápido**. Con el TTL ya no urge.
15. ✅ **`ERROR` rojo en cada arranque** — resuelto en `eb10ef5`. Era: `estacionamiento_planificador` no existe
    en este corte y `app.py:150` lo saca con `logger.exception` (traceback
    completo), cuando `vigilante_amazonas` y `perimetral_nocturna` salen como
    un `WARNING` de una línea. Cosmético, pero asusta.
16. ✅ **`REID TTL: caducaron N` era `logger.debug`** — resuelto en `2dd7267`. Era:
    (`face_reidentifier.py:449`): no se ve con nivel INFO.
17. **`GET /vlm/model` ×16 en un segundo** desde el cliente. Polling agresivo.
18. ❌ **OBSOLETO — NO generes el modelo PAR.** `_infer_body_only`
    (`demographics.py:1803-1815`) prueba `_estimador_cuerpo` (MiVOLO v2)
    PRIMERO y solo mira `_par` si aquel falta. MiVOLO v2 **está instalado y
    activo** (`disponible: True, backend: pytorch, dispositivo: cuda`), así que
    el PAR nunca se usaría. `setup_modelos.py` y el contador `body_sin_modelo`
    siguen diciendo lo contrario: **están desactualizados, no les hagas caso**.

### Hallazgos nuevos del 23-sep

19. **El panel y la base no exigen lo mismo → se pierde el 91 %.** Ver arriba,
    «EL FALLO MÁS GORDO». Es el de mayor impacto del proyecto.
20. **Las capturas se escriben en una ruta y se leen en otra.**
    `AnalyticsConfig.CAPTURE_DIR` es RELATIVA: quien escribe
    (`tiendas_fotos.py:126`) la usa tal cual, contra el directorio de trabajo;
    quien lee (`tienda_capturas.py:73-78`) la ancla a `server/`. Como el
    arranque **no hace `chdir`**, las fotos acaban en `<raíz>/output/captures/`
    y se buscan en `server/output/captures/`. Medido: **2.948 ficheros en la
    primera, la segunda no existe**. El panel de Capturas del cliente marca
    **0** con 1.474 capturas en disco, y `tienda_capturas` nunca se llena.
21. **Un `try/except` gigante tumba todo el stack facial.**
    `tiendas_modelos.py:278` carga Caffe, YuNet, InsightFace, ArcFace y OSNet
    dentro del mismo bloque. Cualquier fallo en el primero se lleva a los
    demás. Con OpenCV 5 pasa de verdad. Conviene separar el legado del resto.
22. **Tres sitios afirman que la rama corporal está apagada y mienten:**
    `setup_modelos.py` (su resumen), el contador `body_sin_modelo` (cuenta
    «corrió y no valió») y `rama_corporal_activa` (✅ arreglado en `576e53a`).
    Los dos primeros siguen vivos.
23. **`site_id` sin configurar.** El log dice `server/sitio-unico`. No rompe
    nada, pero sin él los dashboards no pueden comparar sucursales.

---

## Pendientes, por prioridad

1. **Aflojar el listón de consolidación y MEDIRLO.** Es la causa nº 1 (53 %) y
   la única que se mueve sin comprar nada. El arreglo del voto (`f7af573`) hace
   este cambio más seguro que antes, porque ahora el sistema sí puede
   corregirse con las lecturas siguientes. Se puede medir con el vídeo del
   usuario antes de tocar la tienda. **Nadie lo ha probado todavía.**
2. **Decidir qué hacer con los 119 que se pierden** (hallazgo 19). Bloqueado
   por el usuario: hay que elegir qué significa «visitante».
3. **La ruta de las capturas** (hallazgo 20). Pequeño y desbloquea el panel de
   Capturas, que hoy no sirve.
4. **El doble conteo** (hallazgos 1 y 13). Ya medido: 53 personas, 20 %.
5. **Decisiones suyas, sin prisa:**
   - **La semántica del número.** Con TTL se cuentan **recorridos**, no
     personas del día. El dashboard sigue diciendo «visitantes únicos».
   - **El destino de la rama `fix/gender`:** ¿la integra, se abre un PR?
6. **Cosas del montaje, no del código:** dos cámaras caídas, una intermitente,
   el desfase de imagen y la calibración (el perfil que se aplica hoy se hizo
   en agosto mirando OTRA escena, el stand de charcutería).
7. **Aplazados por el usuario** (no los empieces sin que lo pida): vídeo que no
   se congela si la IA falla (hallazgo 7), y qué contar entre cámaras.
8. **Escáner de ventanas**, pendiente de aprobación. Toca `packages/elde_core`:
   va en rama propia.

### Sin medir, si hace falta tirar del hilo

- **Por qué 119 personas no tienen identidad.** `origen_demografia` sale vacío.
- **Si el desfase entre cámaras** explica que solo 47 de 292 se enlacen.
- **Cuánto mejora aflojar el listón**, midiéndolo con el vídeo.

### Cómo medir sin el servidor del usuario (probado el 23-sep)

El pipeline REAL corre en el contenedor: `TiendasProcess(client_id=...)` y
`proc.process_frame(frame, camera_id=...)`. En CPU va a ~0,65 s/frame. **Usa el
`camera_id` real** (`win-iVMS-4200-i58bf7b`) o no se aplica el perfil. Y lee el
informe de calidad del clasificador **DE LA CÁMARA**
(`proc.camera_states[cam]['_demographics']`), no el de arriba: ese no trabaja.
Haz `os.chdir()` al scratchpad para que nada escriba en el repo.

---

## Cosas que no debes olvidar

- **Cada botón de IA es independiente**: una cámara con la IA apagada no envía
  nada. Y los FPS se reparten entre las cámaras activas.
- **iVMS:** sus ventanas extra se ocultan si se minimiza la principal.
- **No actualices `supervision` por encima de 0.30.1:** `sv.ByteTrack`
  desaparece en la 0.31.
- **El reenvío a WhatsApp está apagado** sin `vigilante_amazonas`. No lo
  reconstruyas sin que te lo pidan.
- **Nunca commitees** `server/models/`, `output/`, `logs/`, el `.env` ni el
  `venv`. Usa `git add <archivos>` explícitos, nunca `git add .`.
- **`opencv-contrib-python` TIENE que ser la 4.12.0.88.** En la 5.x se cae
  todo el stack facial en silencio (ver «Cómo montar el entorno»).
- **Los avisos de `mivolo.onnx` y `par_gender.onnx` «no presente» son RUIDO.**
  MiVOLO v2 carga por otra ruta (`estimador_edad_genero.py`) y la rama corporal
  queda activa. **No generes esos modelos**: el código no los usaría.
- **Antes de creerte una bandera de telemetría, contrástala.** El 23-sep
  `rama_corporal_activa` decía `False` con 235 muestras corporales aceptadas.
  Si un indicador y un contador se contradicen, uno de los dos está viejo.
- **`server/README.md` NO está en la lista de lectura obligatoria y debería.**
  Ahí está la tabla de modelos esenciales contra opcionales y el peso de cada
  rama en la votación (facial 1.0, corporal 0.5).
