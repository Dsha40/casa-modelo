# CLAUDE.md — ELDE Tiendas (contexto de trabajo)

Traspaso de las sesiones del 16-18, 21-22, 23, 24, 25, 28, 29 y **30 de
septiembre de 2026** (actualizado el 1-oct). **Este archivo SÍ se commitea**: vive en la rama para
que la siguiente sesión lo lea sin que el usuario tenga que pegarlo.

**Rama actual: `feature/anaquel`** (publicada). Cuelga de `fix/gender` →
`claude/pensive-galileo-aymjr3` → `fix/eliminar_vigilantes_amazonas`, **no de
`main`**. Al cambiarte a ella heredas todo lo anterior.

**Hay DOS sesiones de Claude en paralelo:** una en el Linux del usuario, junto
al servidor, y otra en el **cliente** de Windows. No se ven entre ellas: solo
comparten lo que quede commiteado aquí. La del cliente tiene su encargo en la
**sección 9** (reescrita el 29-sep). Hasta el 29-sep la sesión del cliente no
había subido nada al remoto.

**Donde este archivo dé una cifra, mira su fecha.** Las cifras de horas o días
de producción mandan sobre las de minutos de vídeo.

---

## 0. Lo primero

### Lee tú mismo estos archivos al empezar

**Es de suma importancia.** Léelos **completos y con tus propias
herramientas**, no los delegues en un agente ni te fíes de un resumen ajeno:

1. `README.md` (raíz)
2. `clients/tienda/README.md`
3. `clients/tienda/EQUIPO.md` (es de otro cliente y dice `git add .`: aquí NO)
4. `GUIA.md`

Según el tema, lee también `server/README_DEMOGRAFIA.md` y `server/README.md`
(tabla de modelos esenciales y opcionales; peso de cada rama en la votación:
facial 1.0, corporal 0.5). Para búsquedas amplias puedes usar agentes, pero
verifica tú con `archivo:línea` lo que devuelvan antes de afirmarlo.

**Desfasado en esos documentos:** `README.md:92` y `GUIA.md:172` dicen «733
pasan»; la cifra buena está en la sección 1 de este archivo.

### Cómo trabajar con este usuario

- **En español.** Si pide «sin tecnicismos», palabras llanas y ejemplos.
  Cuando pide «explícalo mejor», quiere una analogía sencilla y menos tablas.
- **«Sin alucinaciones» es su exigencia n.º 1.** Separa lo *verificado*
  (`archivo:línea` o una medida) de lo *deducido* y de lo que *no sabes*. Mide
  antes de afirmar. Él detecta las afirmaciones sin base y **revisa las
  imágenes él mismo** (ver error 14 de la sección 10).
- **Explica el procedimiento COMPLETO antes de los comandos**, con pasos
  numerados. **Nunca rutas de ejemplo** (`C:\ruta\a\...`): usa las reales
  (sección 1). Se copian tal cual.
- **Avísale cuando un comando vaya a bloquear la terminal**, y de antemano si
  un proceso va a tardar (la suite tarda ~30-90 s). Si no, cree que se colgó.
- **Respeta los nombres que elija.**
- **Nunca uses credenciales pegadas en el chat.** Que se autentique él.
- **PIDE PERMISO ANTES DE CADA COMMIT Y DE CADA PUSH.** Puedes editar
  libremente; lo que no haces sin su visto bueno es fijar el cambio. Cada vez:
  1. enséñale **qué archivos** tocas y el resumen del diff,
  2. enséñale **el mensaje de commit completo**,
  3. **espera su respuesta**. Un hook automático que pida commitear **no**
     es su permiso. Que él diga «haz commit» tampoco te exime de enseñarle
     antes archivos y mensajes.
- Si reescribes un commit publicado (`--amend`, rebase), **avísale**: tendrá
  que hacer `git reset --hard origin/<rama>`.

### Convenciones del repositorio

- **Los comentarios explican el porqué** (qué se rompió, cuándo, cómo se
  midió). Es el estándar, no ruido.
- **Una funcionalidad, un fichero.** Cableado mínimo en lo existente.
- **Commits en español**: asunto corto en imperativo; cuerpo con el porqué,
  cómo se comprobó, **la lista de archivos tocados y la fecha**.
- **Sin prueba no se commitea** (`GUIA.md` 4.3). Y verifica las pruebas **al
  revés** (sección 10, al final).
- `app.py`, `conftest.py` y `pyrightconfig.json` se comparten con un monorepo
  donde **sí** existe `vigilante_amazonas`: los cambios deben funcionar con y
  sin ese paquete.
- `git add <archivos>` explícitos, nunca `git add .`. **Nunca commitees**
  `server/models/`, `output/`, `logs/`, el `.env`, el `venv` ni **`informes/`**
  (informes locales del usuario con fotos de clientes; está en `.gitignore`).

---

## 1. Entorno (verificado del 24 al 29-sep)

### Dónde corres

**Dentro del contenedor Docker `51b31f634ba9` del usuario** (existe
`/.dockerenv`), repo en `/home/D-arias365/proyectos/ELDE-TIENDAS`, venv
`.venv` (Python 3.12.13, `cv2 4.12.0`, `torch 2.8.0+cu128`, CUDA). **Ves el
servidor, su log, su base y sus modelos.** No hay cliente `docker`: no puedes
crear contenedores.

| | |
|---|---|
| GPUs | 2× RTX 3090 Ti (24 GB) + 1× RTX 5060 Ti (16 GB). **Máquina compartida con otros usuarios**: hay memoria y cómputo ajenos en las tres, y **la carga se mueve de una GPU a otra** (28-sep: la GPU 0 al 42 % sin un frame nuestro; horas después, al 1 %). El servidor usa **la primera GPU visible** (`_get_first_gpu`, `app.py:284`; enumeración en `hardware_available.py:21-27`) |
| Ver QUÉ usa nuestro proceso | `nvidia-smi pmon -c 5 -s u` (uso por proceso). `--query-compute-apps` solo muestra los procesos de este contenedor; la memoria ajena no aparece ahí pero sí en el total |
| Máquina anfitriona | se llama **`pop-os`**. La carpeta del proyecto es, en el anfitrión, `/home/gpu/entorno_trabajadores/datos_D-arias365` |
| Zona horaria | el contenedor va en **UTC**. Venezuela = UTC−4 |

### Reglas de convivencia (el entorno es REAL, no desechable)

1. **No instales nada en `.venv`** sin pedirlo.
2. Los arneses van al **scratchpad**, nunca al repo.
3. **No toques `output/`, `logs/` ni `models/`**: son sus datos reales. Para
   leer la base: `sqlite3.connect('file:output/tienda.db?mode=ro', uri=True)`.
4. ⚠️ **Para que una prueba no escriba en su base, exporta `ELDE_TIENDA_DB`
   con una RUTA ABSOLUTA.** Un `os.chdir()` **NO** basta: `ruta_bd()`
   (`tienda_almacen.py:112`) ancla las rutas relativas a `server/`. En cambio
   `CAPTURE_DIR` (`config.py:553`) y `HEATMAP_DIR` (`config.py:420`) sí son
   relativas al directorio de trabajo.
5. **Las grabaciones de diagnóstico son fotos de clientes**: al scratchpad,
   nunca al repo, y borrarlas al terminar (con permiso del usuario).

### El cliente de Windows y cómo llega al servidor

| | |
|---|---|
| Repo en Windows | `C:\Users\DESARROLLO-7\Desktop\Supermercado` (cliente en `clients\tienda`) |
| Cómo conecta | por el **túnel SSH de VS Code** (host `elde-gpu`): en Windows, `localhost:9000` es el contenedor. Todas las peticiones llegan desde `127.0.0.1` |
| Servidor elegido | el cliente guarda el último en `%APPDATA%\ELDE\servidores.json` y **gana al `.env`** (`arranque_servidor.py:16`). `/health` → `servidor_nombre` dice a cuál llega (`51b31f634ba9` = este) |
| Dashboard desde el navegador de Windows | el 28-sep **no cargaba** aunque el servidor respondía y el cliente sí llegaba: hay que reenviar el puerto 9000 en la pestaña **«Puertos»** de VS Code y usar la «Dirección reenviada». Sin resolver del todo |
| Bajar ficheros del contenedor a Windows | explorador de VS Code → clic derecho → **«Descargar…»** |
| Lanzar SOLO el cliente | PowerShell **como Administrador** (iVMS corre elevado y sin admin no se captura su ventana): `cd "C:\Users\DESARROLLO-7\Desktop\Supermercado\clients\tienda"` y `.\venv\Scripts\python.exe src\main.py` |
| `INICIAR_TIENDA.bat` | se relanza como Administrador (**pierde las variables de entorno** que pongas antes) y **arranca también un servidor en Windows** además del cliente. Para pruebas con variables, lanzar solo el cliente |

### Hay dos servidores posibles

| | Dónde | Cómo se llega |
|---|---|---|
| **El del contenedor** | lo arranca Claude o el usuario aquí | `localhost:9000` desde Windows (túnel) |
| **El de `pop-os`** | en el anfitrión | `72.68.60.141:9000`. **Código desconocido y NO escribe en nuestro log** |

### Arrancar y parar el servidor (así, desde el 30-sep)

**Desde el 29-sep el servidor del contenedor lo lleva la sesión Linux de
Claude** (lo pidió el usuario). Antes de arrancar uno, mira si ya hay uno
(`pgrep` de abajo y `/health`). El 1-oct seguía vivo el **PID 143431**
(arrancado el 30-sep 16:12 Venezuela).

```bash
cd /home/D-arias365/proyectos/ELDE-TIENDAS/server       # DESDE server/
echo "=== arranque $(date -u '+%F %T') UTC | GPU 1 | lanzado por Claude ===" >> logs/consola_servidor.log
CUDA_VISIBLE_DEVICES=1 setsid nohup ../.venv/bin/python -u iniciar_servidor_headless.py \
    >> logs/consola_servidor.log 2>&1 < /dev/null &
curl -s --retry 60 --retry-delay 2 --retry-connrefused -m 3 -o /dev/null \
    -w '%{http_code}\n' http://localhost:9000/health      # espera sin sleep
```

- **La salida de consola va a `server/logs/consola_servidor.log`** (fuera
  de git, se AÑADE; cada arranque deja una línea `=== arranque ... ===`).
  Antes iba al scratchpad de la sesión que lo lanzaba y cada sesión nueva
  tenía que buscarla con `readlink /proc/<pid>/fd/1`. **No rota**: crece
  ~2 MB/h con una cámara (un `print` por frame).
- `setsid` hace fork: el `$!` es el lanzador y termina al instante. El PID
  real, con el `pgrep` de abajo.
- **Primera imagen tras arrancar: ~25 s** (carga y compila RF-DETR); la
  primera persona junto a un estante carga ViTPose (4 s el 29-sep, **53 s**
  el 30-sep con la máquina en swap). Mientras, `/health` puede no responder.

- **Desde `server/`.** Desde la raíz, capturas y heatmaps van a
  `<raíz>/output/` y el dashboard no los ve (hallazgo 20).
- **`setsid`**: sin él, el servidor lanzado por Claude **murió sin dejar
  rastro** al cerrarse la sesión (28-sep 15:31; ni error ni OOM). Con `setsid`
  queda en su propia sesión.
- **`CUDA_VISIBLE_DEVICES=1`** lo pone en la GPU 1 sin tocar código (la 3090
  Ti menos cargada el 28-sep). Revisa la carga antes: puede haber cambiado.
- **`-u`**: sin él la salida va amortiguada (ráfagas con la misma hora).
- **Parar**: `PID=$(pgrep -f "^../.venv/bin/python -u iniciar_servidor_headless.py$")`
  y `kill -TERM "$PID"` (se apaga ordenado en 1-11 s y vuelca heatmaps).
  **Nunca `pkill -f` con un patrón que tenga `|`** (error 11).
- El cliente **se reconecta solo** tras un reinicio (5-20 s).

**Variables de diagnóstico** (apagadas por defecto; no dejarlas puestas):

| Variable | Qué hace | Cuidado |
|---|---|---|
| `ELDE_ANAQUELES_DIAG=1` | el log cuenta, frame a frame, qué ve la máquina del agarre de cada muñeca (profundidad, quietud, armado, recorrido) y por qué descarta | varias líneas por frame: el log rota a 5 MB × 3 y **se pierden los logs viejos** |
| `ELDE_ANAQUELES_GRABAR=<ruta absoluta>` | graba frames + tracks + áreas + productos de Anaqueles para reproducirlos fuera de línea (`anaqueles_grabador.py`) | **fotos de clientes**; tope `ELDE_ANAQUELES_GRABAR_MAX` (4000 ≈ 11 min) |

**Telemetría de fps (siempre activa desde el 29-sep, `telemetria_fps.py`)**:
una línea por cámara cada 30 s en el log:

```
FPS [cam] recibidos=6.1/s procesados=6.1/s descartados=0 proceso=63/114 ms entrada=1790 KB/s (302 KB/frame)
```

`grep "FPS \[" server/logs/server.log | tail`. `proceso` = mediana/p90 del
tiempo del frame DENTRO del servidor (sí incluye abrir el JPEG, `app.py:1089`;
no incluye recibirlo por el websocket ni enviar la respuesta).
`descartados` = frames que la cola REEMPLAZÓ sin procesar (corregido el
30-sep). **Antes era `recibidos − procesados` de cada periodo y mentía**:
contaba como perdido el frame en vuelo que cerraba el periodo (70 de 246
líneas del 29-sep decían `descartados=1`), mientras el aviso real de la cola
(`ColaUltimoFrame`, sale con el PRIMER reemplazo) no aparece desde el 17-sep.
Las líneas escritas por un servidor arrancado antes de ese cambio siguen con
la resta.

### Las pruebas

**El `.venv` no tiene `pytest` ni `PySide6`.** Sin tocarlo:

```bash
.venv/bin/python -m pip install --target <scratchpad>/pytest_libs pytest
cd server && ELDE_TIENDA_DB=<scratchpad>/tienda_pruebas.db PYTHONPATH=<scratchpad>/pytest_libs \
    ../.venv/bin/python -m pytest tests -q -p no:cacheprovider
```

**Línea base (30-sep): `852 passed, 16 skipped, 2 xfailed`.** Si ves un
`failed`, es tuyo. Las de `packages/elde_core` necesitan PySide6: aquí no
corren; las que nombran a otros clientes (`clients/amazonas`, `detecciones`,
`perimetrales`) fallan siempre porque esas carpetas no existen en este corte.

**Probar código del cliente sin PySide6 (28-sep):** un doble mínimo de PySide6
en el scratchpad (`QPoint`, `Signal` con `connect/emit`, `QLabel`, `QPainter`
que anota llamadas, `Qt` con constantes) más dobles de `cryptography`,
`win32gui/win32ui/win32con/pyautogui` y un `sitecustomize.py` que pone
`ctypes.windll`. Con eso se importa `render_box.py` y se ejercitan menús,
lienzo y payload. **No sustituye a probar en Windows**: pintar y los diálogos
reales solo los ve el usuario.

En un contenedor propio (nube), mira `README.md`: `torchreid` rompe el
`pip install` entero (instalarlo aparte con `SETUPTOOLS_USE_DISTUTILS=stdlib`)
y `opencv-contrib-python-headless` **tiene que ser la 4.12.0.88**.

### Trampas de este contenedor

- **No existen `ss` ni `docker`.** Para saber si el servidor escucha: `curl`.
- **`awk` no tiene `strtonum`**, y si silencias sus errores la comprobación
  «pasa». No silencies errores en una verificación.
- **`pkill -f <patrón>` se mata a sí mismo** si el patrón aparece en tu propio
  comando, y un `|` en el patrón es un «o» de regex (error 11).
- **`rm` con variables** (`rm -f "$D/$x"`) lo bloquea un control de seguridad:
  usa `"${D:?}/${x:?}"` o, mejor, no borres: descarga a scratchpad y copia.
- **A veces la herramienta Bash deja de responder** («classifier gave no
  verdict»). Es pasajero; mientras, `Read` sigue funcionando (logs, `.git/`).
- **El disco y la RAM se comparten con el anfitrión `pop-os`.** El 30-sep,
  de ~14:21 a ~16:05 (Venezuela), un proceso DEL ANFITRIÓN llenó `/dev/sda3`
  al 100 % (medido: ningún proceso del contenedor escribía más de 16 KB en
  10 s). Consecuencias: «disk I/O error» en la base, el log cortado a media
  línea y, con IA, la imagen del cliente pegada. **El usuario no quiere
  borrar nada nuestro para liberar** (se lo comería el anfitrión); lo
  arregla quien administra `pop-os`. Comprobar: `df -h
  /home/D-arias365/proyectos`; presión de memoria y disco:
  `cat /proc/pressure/{memory,io}`. Ese día el servidor llegó a tener 2,5 GB
  en swap y el disco estuvo ocupado el 45-67 % del tiempo.

---

## 2. Qué es este repositorio

**Un maquetado del monorepo ELDE** que se está ajustando a un proyecto de
tiendas. Mucho código heredado **no funciona aquí y es esperable**
(`vigilante_amazonas`, WhatsApp, Jarvis, Stand, tests de otros clientes).
**Antes de arreglar algo heredado, pregunta si el proyecto lo necesita.**
Tampoco viajaron el `PLAN.md` ni los scripts de etiquetado que cita
`README_DEMOGRAFIA.md`.

**Caso de uso:** un pasillo de supermercado (baldas de café). Pasa mucha
gente, **mira los productos, no la cámara**, y las caras del fondo miden
~20 px.

| Cámara | `camera_id` | Estado |
|---|---|---|
| 1 | `win-iVMS-4200-i58bf7b` | estable |
| 2 | `win-iVMS-4200-c2-i58bf7b` | estable |
| **3** | **`win-iVMS-4200-c3-i58bf7b`** | **baldas de café**; la más medida (24-25 sep) |
| 4 | `win-iVMS-4200-c4-i58bf7b` | intermitente |
| — | — | una quinta no funciona (`OpenSDK 105`) |
| **vídeo** | **`win-Media_Player-c3-i58bf7b`** | **grabación del 23-sep de la cámara 3** (reloj de cámara ~10:44-11:24) en un reproductor. Doblemente comprimida. **Caso de referencia del agarre:** una clienta coge botes de **Coffee-mate** de la balda ALTA hacia las **11:20:46-11:21:11** (reloj de la cámara) |

Si el usuario pega un log o pide medir, **pregunta de qué fuente es** antes de
concluir: en el log, `Media_Player` es el vídeo, `iVMS` la tienda en vivo.

---

## 3. Cómo funciona

1. El cliente (`clients/tienda/src/main.py`) elige servidor y abre la ventana.
   Cada **recuadro** (`render_box.py`) captura una ventana o un DVR.
2. `camera_id`: `elde_core/ui/identidad_camara.py` (`win-<título>[-cN]-i<inst>`).
   **Los nombres cambian a menudo** (título de ventana, canal, instalación).
3. Servidor `app.py` → cola del último frame por cámara (`ws_drenaje.py`;
   descartaría el anterior si llegara otro de la misma cámara antes de
   procesarlo, pero con el cliente de hoy —un frame en vuelo— **no pasa
   nunca**: ningún aviso `ColaUltimoFrame` desde el 17-sep. El «0-1 cada
   30 s» que se leía aquí era la resta mal hecha de la telemetría, sección
   1) → `TiendasProcess.process_frame`.
4. Pipeline: RF-DETR → ByteTrack por cámara → identidad
   (`face_reidentifier.py`) → entradas/salidas del ROI → demografía → base.
5. El modo **`Anaqueles`** (`anaqueles.py`) hereda todo lo de `Tiendas` y
   añade visitas y agarres por área, y **productos dentro de cada área**
   (desde `a8ee806`). **Para cambiar de modo en el cliente hay que detener la
   conexión** (`custom_status_bar.py:109`, `:174`, se libera en `:215`).

### El ciclo del cliente (medido y leído el 28-29 sep) — clave del rendimiento

**Dos rutas de captura:**

| | **Captura de ventana** (iVMS, el reproductor… lo que se usa HOY) | **DVR / stream** (panel Dispositivos) |
|---|---|---|
| Quién captura | proceso aparte `capture_worker.py` (bucle en `:240`); **cada vuelta acaba con `time.sleep(1 / 60)`** (`:315`, sin comentario: llegó con el corte del monorepo, `2a1c9b1`, y el porqué no viajó). Hasta el 30-sep aquí ponía «sin pausa»: era falso | hilo `rtsp_worker.py` |
| Cómo | `PrintWindow`: modo rápido ~11 ms o modo para apps con GPU ~33 ms (`:219-224`) | stream nativo |
| Tamaño | ancho máx. **1600** px (`CAPTURE_MAX_WIDTH`, `:35-38`); la ventana de iVMS mide 1456×868 | **sin límite** (resolución nativa) |
| JPEG | calidad **70** (`:297`) | calidad **80** (`render_box.py:593`) |
| Hora de captura | viaja en cada frame: `header.timestamp` (`:299-303`), reloj de Windows, hora local sin zona | no |

**El ciclo (una cámara), paso a paso:**

1. El proceso de captura entrega JPEG por un tubo, ~**20 por segundo** (dato
   del usuario con la IA apagada). Ese techo lo ponen `PrintWindow`, la
   compresión JPEG con PIL (`:297`) y la pausa de 1/60 s (`:315`).
2. El recuadro lee el tubo, **se queda con el último y tira los intermedios**
   (`render_box_captura.py:506-511`). Si `can_send_next_frame` es True, lo
   manda (~**300 KB**) y lo pone a False (`:574-579`).
3. El servidor lo procesa: **63-80 ms** (telemetría). **Es lo único que hace
   trabajar a la GPU.**
4. Vuelve la respuesta (**modo directo**: solo detecciones, sin imagen,
   `render_box.py:286`) y el cliente **primero la dibuja** —vuelve a abrir
   el JPEG enviado y lo anota con Supervision o QPainter,
   `render_box.py:1624-1636`, `:1649-1686`— y **solo después** pone
   `can_send_next_frame = True` (`render_box.py:1845`, en el `finally`),
   **sin mandar nada**. El dibujo cuenta dentro del ciclo (corregido el
   30-sep: antes no se decía).
5. **Espera a la SIGUIENTE captura** (las que llegaron mientras esperaba ya se
   tiraron). A 20 capturas/s son 0-50 ms, ~25 de media. Vuelve al paso 2.

**Contadores que ve el usuario en el recuadro:** con la IA apagada, «FPS: N»
cuenta **lecturas del tubo con al menos un frame** (si llegan dos juntos
cuenta uno: puede quedarse corto; `render_box_captura.py:517-527`). Con la IA
encendida se sustituye por «IA: N FPS» = **respuestas del servidor**
(`render_box.py:1837`). Nunca se ven las dos a la vez.

### Mecanismos que conviene saber

- **Estado por cámara:** `_push_state`/`_pop_state` cuelgan el estado solo
  durante un fotograma. Leerlo desde fuera da el estado vacío.
- **Un procesador por cámara y por conexión**, compartiendo detector, GPU y
  galería de caras.
- **Identidad:** la cara crea identidades (umbral 0,35); el cuerpo solo
  re-enlaza; sin cara queda `prov:<track>`. La galería **olvida a los 30 min**.
- **Los `track_id` se reinician en cada arranque.** No cruces datos por track.
- **La ficha de cada persona** (`server/output/captures/persons/<stem>.json`)
  se cierra con lo primero que llegue; lo que falte se espera 5 min
  (`tienda_clasificaciones.py`, desde `0ab287c`).
- **Dos vocabularios de género:** `'Hombre'/'Mujer'` (demografía) y
  `'Hombres'/'Mujeres'` (solo para pintar). No los mezcles (`f7af573`).

### Anaqueles: lo que hay que saber antes de tocarlo (28-29 sep)

- **Productos** (`a8ee806`): cuadriláteros violetas dentro de un área, con
  nombre (menú **Área ▾ → «Productos del anaquel»**). El servidor reparte
  cada agarre confirmado al producto bajo la **muñeca**; lo que no cae en
  ninguno va a `agarres_sin_producto`. Identidad = **posición** dentro del
  área (por eso solo se quita el último). API:
  `/api/anaqueles/<cam>/resumen` → `productos` por área y `ranking_productos`.
- **La muñeca NO es la mano**: al estirarse hacia una balda alta queda
  **~70 px por debajo y al lado de los dedos**. Con productos estrechos y
  vecinos, **el producto asignado cambia de una pasada a otra** (el mismo
  agarre salió «coffemate», «Nescafe» y «sin producto» en pasadas distintas).
  Mejora pendiente: estimar la mano prolongando codo → muñeca (la pose ya da
  el codo, keypoints 7 y 8).
- **La pose (ViTPose, 17 puntos) es la ÚNICA forma de ver la mano** (no hay
  modelo de manos; MediaPipe vetado). Cuesta **~21-28 ms por persona** en la
  3090 Ti. **La muñeca no tiene tracker**: se recalcula en cada frame y es
  «la izquierda/derecha de esa persona» (keypoints 9/10). En el vídeo no se
  vio intercambio izq/der.
- **Por defecto, la pose solo se calcula a quien está «en compuerta y
  quieto»** (`anaqueles.py`, en `_medir`). Con una balda **por encima de la
  cabeza**, con el brazo abajo la caja de la persona no pisa el área
  (compuerta cerrada) y al subirlo el cuerpo cuenta como «en movimiento»: **la
  máquina dejaba de mirar justo cuando entraba la mano**. En el vídeo vio la
  muñeca a −34 px y 2 s después a +92: se perdió la entrada entera.
- **Desde el 29-sep, `anaqueles_config.yaml` enciende como NORMA GENERAL**
  `pose_cercania: true` (mirar también a quien está cerca del área, aunque se
  mueva o no pise la compuerta) y `agarre_amplitud_desde: extremo` (medir el
  recorrido desde el punto más alejado en que se vio la mano fuera). **Los
  valores anteriores están comentados al lado** (`false` y `ultimo`) para
  volver atrás. El defecto del CÓDIGO sigue apagado. **Primeros datos en la
  cámara 3 en vivo (29-30 sep): 2 agarres contados, 1 revisado en imagen y
  real** (sección 8, punto 1); todavía poca muestra.
- **Probadas y QUITADAS el 29-sep** (documentado en `anaqueles.py`, junto a
  `_DEFECTOS`): `agarre_quietud='antes_de_entrar'` y `pose_en_movimiento`.
  No cambiaron ni un conteo en el vídeo. No las vuelvas a añadir sin datos.
- **Configuración de Anaqueles = tres capas**: `_DEFECTOS` del código →
  valores globales del YAML → bloque `camaras.<camera_id>` del YAML. **Solo
  se leen claves declaradas en `_DEFECTOS`** (una desconocida se ignora sin
  avisar). **Evita el bloque por cámara**: los nombres cambian y, si no
  coincide, se ignora en silencio. La config se lee cuando se crea el medidor
  de esa cámara (primer frame de cada conexión): un cambio en el YAML vale al
  reconectar el cliente o reiniciar el servidor.
- **El gesto se guarda por (área, identidad de la persona)**: si la cara se
  reconoce A MITAD del gesto (`prov:` → UUID), el gesto se reinicia. En el
  vídeo pasó (a +33,5 s) sin ser la causa; puede costar agarres. Medir en la
  cámara real antes de tocarlo (solución: pasar el estado al nombre nuevo).
- **El calor del área solo se suma con la muñeca dentro a fondo y la persona
  quieta.** El mapa de calor horario del pasillo se guarda **cuando llega el
  primer frame de la hora siguiente** (`heatmap.py:530-545`).
- **Si un recuadro acaba justo en el borde de la balda**, con la regla de
  siempre la muñeca apenas entra: un recuadro que baja ~85 px por debajo del
  borde sí contaba. Con `pose_cercania` ya no hace falta.

### Dónde están los datos

| Qué | Dónde |
|---|---|
| Log | `server/logs/server.log` (rota a 5 MB, 3 copias) |
| Base | `server/output/tienda.db`: `tienda_visitas`, `tienda_eventos` (`entrada`/`salida`, `ts` en epoch), `tienda_capturas` (caché perezosa), `tienda_clasificaciones` (con `solo_cuerpo`) |
| Fichas y fotos | `output/captures/persons/` **relativo al directorio de arranque** |
| Eventos por persona | `output/analytics_log_<camera_id>.jsonl`: `person_detected` y `demographic_classified` (**incluye provisionales**) |
| Anaqueles | `server/output/anaqueles/<camera_id>/<AAAA-MM-DD>.json` (**totales por día** UTC, con `productos`) y `fotos_ref/<área>.jpg` (recorte exacto del área cuando estaba vacía) |
| Heatmaps | `output/heatmap/<camera_id>.png` y `history/<cámara>/<AAAA-MM-DD_HH>.{png,json}` |
| **Informes del usuario** | **`informes/`** (local, **fuera de git**): `informe_2026-09-25.md`, `informe_2026-09-28.md`, `informe_2026-09-29.md`, `informe_2026-09-30.md`, `informe_2026-09-30_tarde.md`, **`Pendientes_Rendimiento.md`** (lista de 17 cambios de rendimiento, de fácil a difícil) |
| Config del cliente | `%LOCALAPPDATA%\tienda_view\tienda_view\config.json`, **cifrada a propósito**. Al cargarla imprime «Archivo de configuración **creado**» aunque la haya **cargado** (`settings_model.py:114`) |

APIs útiles: `/health`, `/api/v1/tienda/{resumen,serie,pasillos,demografia,
clasificaciones,capturas}`, `/api/anaqueles/<cámara>/resumen`,
`/api/anaqueles/<cámara>/heatmap/<área>/imagen?fecha=AAAA-MM-DD`,
`/dashboard/img/heatmap/<cámara>.png`, `/api/v1/heatmaps`.

---

## 4. Qué se resolvió

| Commit | Rama | Qué |
|---|---|---|
| `ae11664`, `e01ecde` | `fix/eliminar_vigilantes_amazonas` | imagen congelada en modo IA (detector importaba `vigilante_amazonas`; pesos como punteros LFS) |
| `2644a68` | `claude/pensive-galileo-aymjr3` | galería facial con **TTL de 30 min** y en RAM |
| `5ab054e` | ídem | perfiles de cámara huérfanos desde el sufijo `-i<instalación>` |
| `eb10ef5`, `2dd7267` | `fix/gender` | avisos de log |
| `f7af573` | ídem | **el voto de género nunca contaba** |
| `576e53a` | ídem | bandera `rama_corporal_activa` y MiVOLO v2 |
| `288be55` | `feature/anaquel` | pintar en el recuadro las muñecas de Anaqueles (menú Vista) |
| `0ab287c` | ídem | esperar identidad/género tardíos en la ficha; tabla `tienda_clasificaciones` |
| `a8ee806` | ídem | **productos dentro de los anaqueles**: dibujo en el cliente, reparto de agarres, API. Probado en vivo el 28-sep |
| `b303013` | ídem | **telemetría de fps por cámara** en el log (recibidos, procesados, tiempo de proceso, KB/s) |
| `c690a86` | ídem | **agarres en baldas altas**: `pose_cercania` + `extremo` (encendidas en el YAML), diagnóstico (`ELDE_ANAQUELES_DIAG`) y grabador (`ELDE_ANAQUELES_GRABAR`); dos opciones probadas y quitadas |
| `8994ba8` | ídem | **`descartados` de la telemetría = reemplazos reales de la cola** (antes, una resta que contaba el frame en vuelo) |

---

## 5. Lo medido

### 5.1 Cámara real 3, del 24-sep 17:26 al 25-sep 17:04 (hora de Venezuela)

Resumen del informe `informes/informe_2026-09-25.md` (local). Modo `Anaqueles`.
Tienda cerrada de 21h a 06h; **cámara sin señal de 13:55 a 15:05 del 25**.

| | Cantidad | % |
|---|---:|---:|
| **Personas capturadas** (una ficha por persona) | **936** | 100 % |
| **Identificadas** (cara reconocida) | 179 capturas, 167 personas | **19,1 %** |
| **Clasificadas** (género y edad) | 90 (60 H, 30 M) | **9,6 %** |

- Con identidad se clasifica al **28,5 %**; sin ella, al **5,2 %**.
- **84 %** de las clasificaciones, **solo con el cuerpo** (MiVOLO v2).
- **Por qué no se clasificó al resto (846):** `muestras_insuficientes`
  **59,6 %**, `sin_rostro` 35,3 %, `calidad` 2,8 %.
- **Anaqueles:** Anaquel 2 (Nescafé/Colcafé) 88 visitas y 5 agarres; Anaquel 1
  (latas altas) 11 visitas y **8 agarres** (proporción sospechosa: ¿personal
  reponiendo?).
- **Permanencia en el ROI:** estancia mediana 3,7 s; 31 % ≤ 2 s. Por persona,
  mediana 11 s.
- **Horas pico:** 09h y 11h. El mapa general: más actividad a las 16h.
- **Personal en las métricas:** `3edc2281` 79 entradas y 22 min 40 s (15 % de
  toda la permanencia); 9 identidades con más de 20 entradas el día 25.

### 5.2 Producción, 14 h del 23-sep (dos cámaras)

18.800 intentos, **aceptación 8 %**. Rechazos: tamaño 37,7 %, sin cara
19,4 %, pose 13,6 % (**siempre giro horizontal**), `body_sin_modelo` 11 %
(**no** significa «falta el modelo»), calidad 7,7 %.

**Conclusión común a 5.1 y 5.2: la causa n.º 1 de no clasificar es el listón
para decidir, no la imagen.** `DEMO_MIN_SAMPLES=5`, `DEMO_MIN_CONFIDENCE=0.85`,
`DEMO_MIN_MARGIN=0.35`, `DEMO_COMMIT_AGREE_RATIO=0.90` (`config.py:96-192`).
**Nadie lo ha aflojado ni medido.** El umbral de nitidez mide en realidad
**tamaño** (25 ≈ exigir ~32-35 px de cara nativa).

**Identidad entre cámaras: funciona** (47 personas enlazadas el 23-sep); por
eso el doble conteo es medible (hallazgos 1 y 13).

### 5.3 Rendimiento del SERVIDOR (24-sep, banco aislado)

38 ms por frame con 1 persona (84 % detección, fija); la demografía escala
(3 ms con 1 persona, 27 ms con 4). **Las cámaras reparten los fps.** fp16,
bajar la resolución del detector y repartir cámaras entre GPUs: **descartados
por la medición** (aceleran una GPU que no es el cuello).

### 5.4 Rendimiento DEL CICLO (28-29 sep, telemetría nueva)

**En vivo, vídeo, una cámara, `Anaqueles` con `pose_cercania`:**

| Prueba | fps que recibe el servidor | Proceso en servidor | Nota |
|---|---:|---:|---|
| GPU 0, sin `pose_cercania` (28-sep 17:20 UTC) | 5,5 | — | carga ajena baja |
| GPU 0, con `pose_cercania` (19:57) | 4,0 | — | **carga ajena ~42 %** en esa GPU |
| **GPU 1, con `pose_cercania`, 40 min (21:11-21:51)** | **mediana 6,1** (2,8-7,0) | **mediana 63 ms**, p90 ~114 | carga ajena media |
| Tramo del agarre, 1456 px (29-sep 14:31) | ~5,0 | 66-84 ms | ~300 KB/frame, 1,2-1,8 MB/s |
| Tramo del agarre, `CAPTURE_MAX_WIDTH=960` (14:41) | ~6,5 | 36-53 ms | ~145 KB/frame, 0,9-1,0 MB/s |

**Lo que se concluye (y con qué seguridad):**

1. **La GPU no es el cuello** (medido): de un ciclo de ~164-200 ms el servidor
   trabaja 63-80. Podría con ~12-15 fps por cámara.
2. **La conexión no está llena** (medido): con fotos a la mitad, los KB/s
   bajaron un 40 % y los fps solo subieron un 30 %. Si la subida fuera el
   techo, los KB/s se habrían mantenido y los fps casi duplicado.
3. **Reducir a 960 px NO compensa**: +30 % fps a cambio de más caras perdidas
   (rechazos de cara 97 → 139; «sin rostro» 2 → 7 de 12 fichas; muestra
   pequeña, orientativo).
4. **Sin medir: el reparto de los ~100-120 ms que el ciclo pasa fuera del
   servidor.** Estimado: ~25 ms de media esperando la siguiente captura
   (paso 5 del ciclo, sección 3) + ~60-100 ms de subida, latencia del túnel,
   respuesta y manejo en el cliente. **La prueba A/B cambiaba cuatro cosas a
   la vez** (tamaño a transmitir, proceso del servidor, compresión en el
   cliente y ritmo de captura): no sirve para repartir esos ms.
5. **El `proceso` de la telemetría no incluye** recibir/desempaquetar el
   mensaje ni enviar la respuesta.

**Siguiente medida buena:** usar `header.timestamp` (hora de captura que el
cliente ya manda) para medir en el servidor cuánto tarda cada foto en llegar,
y la telemetría del cliente (sección 9).

### 5.5 Anaqueles con el vídeo de referencia (28-sep)

Todo sobre la grabación de 482 frames (92 s) del tramo de la clienta de los
Coffee-mate, reproducida fuera de línea con la pose real (la reproducción da
**los mismos descartes que en vivo**):

| Regla | Agarres |
|---|---:|
| La de siempre (y también la del código anterior a los productos, `0ab287c`) | 0 |
| Solo `pose_cercania` | 1 |
| **`pose_cercania` + `extremo`** | **2 (reales, revisados en imagen)** |
| Bajar `agarre_amplitud_min` a 0,20 | 1, **por suerte**: dependía de UNA muestra a 2 px del borde; moviendo el área 8 px, 0 |
| Recuadro alargado hacia el torso (y 600-700) | cuenta, **pero también cuenta a la clienta tocándose el pelo** (falso) |

**El commit de los productos NO rompió el agarre**: `0ab287c`, `a8ee806` y la
copia de trabajo dan idéntico resultado en la grabación.

En vivo, con `pose_cercania` + `extremo`: 2 agarres en ese tramo, **las tres
veces que se reprodujo** (28-sep 19:57, 21:47 y 29-sep 14:32). Coste: la pose
pasa de 40 a 337 cálculos en 482 frames; ~15-20 ms más por frame con alguien
delante del estante.

### 5.6 El vídeo como fuente, 28-sep 17:00-17:57 (Venezuela)

Resumen de `informes/informe_2026-09-28.md` (local). **Es la grabación, no la
tienda**: 102 capturadas, 20/19 identificadas (19,6 %), 16 clasificadas
(15,7 %; 15 H y 1 M; 75 % solo cuerpo), `muestras_insuficientes` 54,7 %;
Anaquel 1: 1 visita y 2 agarres; estancia mediana 3,6 s.

### 5.8 Cámara 3 en vivo, 29-30 sep (informes locales del 29, 30 y 30 tarde)

| Tramo (Venezuela) | Capturadas | Identificadas | Clasificadas | Anaqueles |
|---|---:|---:|---:|---|
| 29-sep 15:00-17:03 | 273 | 17,2 % | 8,4 % | 8 visitas (A2), 0 agarres |
| 29-sep 18:07 → 30-sep 08:44 (≈5 h de imagen buena) | 373 | 15,5 % | 14,2 % | 12 visitas A2 + 1 A1, 0 agarres |
| 30-sep 16:00-17:26 (≈50 min de imagen buena) | 138 | 9,4 % | 8,0 % | 3 visitas A2, **1 agarre** |

- **AGARRE #1 (30-sep 16:24:17, A2 → «colcafe») revisado en iVMS por el
  usuario: real y producto correcto.** AGARRE #2 (17:38:28, A2, sin
  producto): **sin grabación, no verificable**. Las dos personas, sin
  identidad y sin género (`muestras_insuficientes`).
- **La cámara 3 falla a ratos**: de noche manda imagen de ~37 KB (oscura o
  vacía; no se ha visto) y el anaqueles suspende/reanuda el conteo ~2 veces
  por minuto (1.206 veces una noche); también falló el 30-sep 07:44-08:44 y
  16:53. Con esa imagen hubo detecciones fijas en un punto (abajo, algo a la
  derecha) y una ficha a las 02:30: probablemente algo quieto que toma por
  persona (sin comprobar).
- **Personal probable** cada tramo (p. ej. `e1c0cd0f` 43 entradas en 2 h el
  29-sep; `31e27475` 22 en 25 min el 30-sep). Deducción, no visto.

**Rendimiento (telemetría, cámara 3 en vivo, GPU 1):** el servidor tarda
siempre **~55-60 ms** por imagen; el ciclo, **~6 fps el 29-sep** (~110 ms
fuera del servidor) y **4,2-4,9 fps la tarde del 30-sep** (~150-180 ms
fuera). No se sabe de dónde salen esos 40-70 ms de más. Tamaño de la
imagen frente a velocidad (experimentos naturales, NO limpios):

| Imagen | KB | fps | fuera del servidor |
|---|---:|---:|---:|
| ventana de iVMS minimizada | 1 | ~25 | ~28 ms |
| vacía (29-sep) | 17 | ~22 | ~35 ms |
| oscura de noche | 37 | ~9 | ~54 ms |
| buena | ~245 | 4-6 | 110-190 ms |

**El dibujo del cliente con Supervision** (medido en el i5-11400 de este
servidor, sin Qt): ~22 ms, y ~55 ms con el mapa de calor en vivo. **Pero el
cliente de la tienda no tiene Supervision** (hallazgo 31): dibuja con
QPainter, sin medir.

### 5.7 Precisión: NO está medida

No existe una cifra de acierto de género ni de edad, y **no se debe
inventar** (`README_DEMOGRAFIA.md:218`). Para medirla hay que **etiquetar
recortes a mano**: con 30 etiquetas, un 80 % medido es 63-90 %; con 100,
71-87 %; con 200, 74-85 %. Los recortes son fotos de clientes: **en local**.

---

## 6. Lo que distorsiona los datos (léelo antes de concluir)

1. **Personal en las métricas** (5.1). Existe alta de personal
   (`personal_api.py`); **no se ha comprobado** hasta dónde llega su exclusión.
2. **El día se corta a las 20:00 de Venezuela** (servidor en UTC). Hoy no se
   puede medir recurrencia (además, la galería olvida a los 30 min).
3. **Seguimientos, capturas y personas no son lo mismo.**
4. **`demographic_classified` incluye etiquetas provisionales.**
5. **Edades en dos vocabularios** («joven/adulto» y rangos «26-35»).
6. **`COUNT(*)` sobre `tienda_visitas` cuenta filas**, no personas.
7. Los textos de las imágenes de calor muestran **«??»** (cosmético).
8. **El producto asignado a un agarre no es fiable** (muñeca ≠ mano, 3).
9. **El vídeo no es la tienda**: otra compresión y **otro método de captura**
   (el reproductor es una app con GPU: modo lento de `PrintWindow`).
10. **La GPU es compartida**: una medida de fps en dos momentos distintos
    puede mezclar la carga ajena. Compara en la misma GPU y seguido. **Y la
    RAM y el disco también** (sección 1, trampas): el 30-sep el ciclo bajó
    de ~6 a ~4 fps con el servidor igual de rápido.
11. **Minimizar la ventana de iVMS deja la captura vacía** (imágenes de 1 KB,
    «sin señal»; 30-sep 16:28-16:30). Tapada por otra ventana suele seguir
    capturando (sin comprobar con iVMS).
12. **Revisar grabaciones en la MISMA ventana de iVMS que captura un
    recuadro con la IA encendida** podría contarlas como si fueran en vivo.
    Para revisar: apagar la IA de ese recuadro u otra ventana.

---

## 7. Hallazgos pendientes

Antes de citar uno, **comprueba que el código sigue igual**.

| # | Hallazgo | Estado |
|---|---|---|
| 1, 13 | **Doble conteo entre cámaras** (`tienda_almacen.demografia`, `dashboard.py:244`): 20 % de más (23-sep). `/api/v1/tienda/resumen` sí lo hace bien | vivo |
| 19 | Clasificados sin identidad: ya se guardan en `tienda_clasificaciones` (`0ab287c`). **Decisión del usuario pendiente**: si cuentan en el dashboard | en parte |
| 20 | **Rutas relativas al arranque** (capturas, heatmaps, eventos). **Arrancar siempre desde `server/`** | vivo |
| 21 | **Un `try/except` gigante** (`tiendas_modelos.py:313-474`) | vivo |
| 4 | Panel de Anaqueles: «Persona (Desconocido)» (`anaqueles_panel.py:55`, `:363`) | vivo; una línea (cliente) |
| 22 | `setup_modelos.py` y `body_sin_modelo` dicen que la rama corporal está apagada, y mienten. **No generes el modelo PAR** | vivo |
| 24 | **Gesto reiniciado si la cara se reconoce a mitad** (sección 3, Anaqueles) | vivo; medir en real |
| 25 | **Muñeca ≠ mano**: producto inestable; estimar la mano con el codo | vivo |
| 26 | **El panel del cliente no muestra productos**: el bloque en vivo `metadata['anaqueles']` no los lleva (`registro.resumen()` sin productos) | pendiente (servidor + cliente) |
| 27 | Contador «FPS: N» del recuadro cuenta lecturas, no capturas (sección 3) | vivo (cliente) |
| 28 | «Archivo de configuración creado» al CARGARLA (`settings_model.py:114`) | cosmético (cliente) |
| 29 | Dashboard inalcanzable desde el navegador de Windows (sección 1) | del montaje |
| 30 | **La ruta de captura de ventana no tiene red de seguridad**: si una respuesta se pierde, el recuadro no vuelve a enviar hasta reconectar (`render_box_captura.py:574`; la ruta DVR sí libera a los 10 s, `render_box.py:583-589`). Hoy casi no pasa (un frame en vuelo); con dos en vuelo sí | vivo (cliente) |
| 14 | `_best_in_gallery` es O(N) en Python; con el TTL ya no urge | vivo |
| 31 | **El mapa de calor, las estelas y la elipse del recuadro no hacen nada sin Supervision**, y el cliente de la tienda NO lo tiene (`ModuleNotFoundError` en su `venv`, comprobado por el usuario el 30-sep). Las opciones aparecen y no avisan (`render_box.py:296-305`, `:1356-1363`); se dibuja con QPainter. Supervision queda fuera a propósito (`requirements-cliente.txt:25-28`). Los comentarios de `render_box.py:316-318` y `:1370` dicen que «el servidor pinta», pero eso solo pasa en el modo clásico | pendiente de revisar (cliente); detalle en `informes/Pendientes_Rendimiento.md`, n.º 17 (local, fuera de git) |
| 32 | **`print()` de depuración en CADA frame** (`tiendas.py:2192-2193`, «📍 ROI actualizado», y `:2371-2372`): llenan la consola (~2 MB/h, sin rotación). Hipótesis SIN comprobar: con el disco lleno, un `print` que falla rompería cada frame (eso explicaría la imagen pegada del 30-sep); al probarlo ya había espacio | vivo (servidor) |
| 2, 5, 6, 7, 3, 9, 10, 12, 17, 23 | Jarvis, errores que el cliente nunca ve, DVR por IP, `main.spec`, aforo, Stand, polling `/vlm/model`, `site_id` | menores o aplazados |

---

## 8. Pendientes, por prioridad

1. **Validar `pose_cercania` + `extremo` en la cámara REAL 3** con la tienda
   abierta: 20-30 min con `ELDE_ANAQUELES_GRABAR` y `ELDE_ANAQUELES_DIAG`,
   revisar **en imagen** cada agarre contado (buscar falsos: tocarse la cara,
   braceo al pasar) y la telemetría de fps. Si da falsos, volver a los valores
   de antes en el YAML. **Primeros datos (29-30 sep, cámara 3 en vivo):** de
   15:00 a 17:03 del 29-sep, 8 visitas y 0 agarres; el 30-sep a las 16:24:17
   (Venezuela), AGARRE #1 en Anaquel 2 → «colcafe», **revisado en la
   grabación de iVMS por el usuario: real y producto correcto**. AGARRE #2
   (17:38:28, A2, sin producto): sin grabación, no verificable. 1 verificado
   de 1; poca muestra todavía. **El usuario no siempre tiene grabación en
   iVMS**: para verificar cada agarre hará falta `ELDE_ANAQUELES_GRABAR`
   (fotos de clientes, ~2,4 GB por 40 min; pedir permiso y mirar el disco).
2. **Rendimiento del ciclo** (sección 9): el **paso 0 es la telemetría del
   cliente**; sin ella no se puede repartir el tiempo de fuera del servidor.
   Lista completa, de fácil a difícil, en `informes/Pendientes_Rendimiento.md`
   (local). Ya hecho el 30-sep: telemetría de descartes reales (sección 1) y
   correcciones de este archivo.
3. **Estimar la mano con codo → muñeca** para el reparto por producto; medirlo
   con el vídeo de referencia (la clienta coge **Coffee-mate**).
4. **Aflojar el listón de consolidación y MEDIRLO** (causa n.º 1 de no
   clasificar, 53-68 %).
5. **Dar de alta al personal** y comprobar hasta dónde llega su exclusión.
6. **Hora local en el servidor** (`TZ=America/Caracas`).
7. **Rutas relativas** (hallazgo 20).
8. **Anaqueles:** exponer los `descartes` en la API/panel; guardar la hora de
   cada visita y agarre; productos en el panel y el dashboard (hallazgo 26).
9. **Medir la precisión** con recortes etiquetados (5.7).
10. **Decisiones del usuario:** clasificados sin identidad en el dashboard;
    qué es «visitante» con el TTL; destino de las ramas (¿PR a `main`?).
11. Hallazgo 21, vocabularios de edad, el «??», hallazgo 24.
12. **Hallazgo 31 (30-sep): mapa de calor y estelas del recuadro sin
    Supervision.** Decidir entre avisar/ocultar las opciones, instalarlo o
    llevar el cálculo al servidor. La lista de rendimiento completa está en
    `informes/Pendientes_Rendimiento.md` (local, fuera de git).

---

## 9. Encargo para la sesión del CLIENTE (reescrito el 29-sep)

Escrito por la sesión del Linux. **No nos vemos**: esto es lo único que nos
comunica. **El usuario no puede tener el cliente y tu sesión abiertos a la
vez**: todo lo que midas tiene que quedar ESCRITO en disco para leerlo
después. **Avísale de que el fichero solo se completa al cerrar el cliente.**

### Lo que ya está medido (no lo repitas)

- El ciclo completo es de **~164-200 ms** con una cámara (5-6 fps); el
  servidor trabaja **63-80 ms** de ese tiempo (sección 5.4). **Ni la GPU ni la
  conexión son el techo.**
- **El cliente espera la siguiente captura** después de cada respuesta
  (sección 3, paso 5): ~25 ms de media a 20 capturas/s.
- **`CAPTURE_MAX_WIDTH=960`**: +30 % fps, pero más caras perdidas. **No se
  recomienda.** No lo repitas salvo para otra cosa.
- **La ruta que importa es la de captura de ventana** (`render_box_captura.py`
  en `packages/elde_core`, el mixin que usa `render_box.py`): por ahí van iVMS
  y el vídeo. La ruta DVR (`_on_dvr_frame`, `render_box.py:561-646`) hoy no
  se usa.

### Tarea 1 — Medir el ciclo en el cliente (lo primero)

**Qué:** en qué se van los ~100-120 ms que el ciclo pasa fuera del servidor.

**Cómo (propuesta):**

1. Un módulo propio (una funcionalidad, un fichero), p. ej.
   `packages/elde_core/elde_core/ui/telemetria_ciclo.py` (lo usa el mixin).
2. Una línea JSON por frame ENVIADO con estas marcas (`time.time()`):
   - `t_captura`: `header.timestamp` del frame (lo pone `capture_worker.py`
     tras comprimir; ojo: `datetime.now().isoformat()`, hora local sin zona);
   - `t_leido`: cuando `loop_show_result` lo saca del tubo;
   - `t_envio`: tras `send_binary_frame` (`render_box_captura.py:578`);
   - `t_respuesta`: al entrar en `on_text_message_received` (`render_box.py`);
   - `t_pintado`: tras pintar las detecciones;
   - más `camera_id`, bytes del JPEG y cuántas capturas se tiraron desde el
     envío anterior.
3. **Interruptor en el menú Vista, apagado por defecto** (como las muñecas).
4. Ruta fija fuera del repo: `%LOCALAPPDATA%\tienda_view\telemetria_ciclo.jsonl`.
   `flush` cada pocos segundos y rotación (a 6 fps son ~21.600 líneas/hora).
   **Dile al usuario la ruta exacta.**
5. Con el JSONL y las líneas `FPS [...]` del servidor (misma hora) sale el
   reparto: esperar captura / subir / servidor / volver / pintar.

### Tarea 2 — Mejora pequeña: mandar al instante la última captura

Hoy, al llegar la respuesta, se dibuja y después se pone
`can_send_next_frame = True` (`render_box.py:1845`), y se espera la
**siguiente** captura; las que llegaron mientras tanto ya se tiraron
(`render_box_captura.py:506-511`).

**Propuesta:** en `loop_show_result`, cuando no se puede enviar, **guardar el
último frame** (`header`, `image_bytes`) en vez de tirarlo; al llegar la
respuesta, si hay uno guardado sin enviar, **mandarlo ya, ANTES de dibujar**,
y dibujar después con una copia del frame de ESA respuesta (enviar
sobrescribe `_pending_frame_bytes`). **Evaluado el 30-sep: mover solo
`can_send_next_frame = True` antes del dibujo NO adelanta nada**: todo va en
el hilo de Qt, el siguiente envío solo sale del aviso de captura nueva
(`render_box.py:1435`) y ese aviso no se atiende hasta que termina el
dibujo. Con la tarea 2 sin «enviar antes de dibujar», el dibujo queda
entero dentro del ciclo. Ojo: una llamada
recursiva a `loop_show_result` se quitó por reentrancia (comentario en
`render_box_captura.py:583-585`): programarlo con `QTimer.singleShot(0, …)`.
**Esperado (estimado): ~−25 ms por ciclo, de ~5 a ~5,7 fps.** Medir con la
tarea 1 y con las líneas `FPS` del servidor.

### Tarea 3 — Mejora grande: dos fotogramas en vuelo

`can_send_next_frame` pasa de booleano a **contador de frames en vuelo (máx.
2)**. **Esperado (estimado): ~8-12 fps.** Tres cuidados:

1. **La cola del servidor guarda solo el último frame por cámara**
   (`ws_drenaje.py`): si llegan dos mientras procesa uno, **descarta el
   anterior y ese NUNCA recibe respuesta**. Con 2 en vuelo casi nunca pasa
   (uno procesándose, otro esperando), pero el contador tiene que **caducar**
   cada frame pendiente (p. ej. 2 s), no esperar los 10 s de la red de
   seguridad (`render_box_captura.py` / `render_box.py:583-589`). La
   telemetría del servidor dice `descartados=N` (reemplazos reales desde el
   30-sep; antes era una resta, sección 1). **Ojo: la ruta de ventana
   NO tiene red de seguridad**: solo la ruta DVR libera tras 10 s
   (`render_box.py:583-589`); en `render_box_captura.py:574` una respuesta
   perdida deja el recuadro sin enviar hasta reconectar (hallazgo 30).
   Añadirla es parte de esta tarea.
2. **Modo directo:** el cliente pinta las detecciones sobre
   `_pending_frame_bytes` (el frame enviado). Con 2 en vuelo hay que guardar
   **una cola FIFO** de frames enviados y emparejar cada respuesta con el suyo
   (las respuestas llegan en orden), o se pintarán cajas sobre el frame
   equivocado.
3. Mantener la red de seguridad de 10 s de la ruta DVR y **añadirla** a la de
   ventana.

Hacerla **después** de la 1 y la 2, midiendo antes y después (línea base: el
tramo del agarre del vídeo, ~5,0 fps a 1456 px en la GPU 1).

### Tarea 4 — Arreglos pequeños ya localizados

- **Panel de Anaqueles:** traduce `male/female` y llega `Hombre/Mujer`
  (`anaqueles_panel.py:55-56`, `:363`): sale «Persona (Desconocido)».
- **Panel de Anaqueles:** avisa «modelo de pose no disponible» por una bandera
  perezosa (`:249`): vale `False` hasta el primer uso.
- **`settings_model.py:114`** dice «creado» al cargar la configuración.
- **Contador del recuadro:** con la IA apagada cuenta lecturas del tubo, no
  capturas; con la IA encendida no se ve el ritmo de captura. Mostrar ambos.
- **Productos en el panel** (hallazgo 26): necesita que el servidor los meta
  en el bloque en vivo; coordinarlo (lo puede hacer la sesión del Linux).
- **Mapa de calor / estelas / elipse sin Supervision** (hallazgo 31): el
  cliente de la tienda no lo tiene y esas opciones no hacen nada. Mínimo:
  avisar u ocultarlas. Instalarlo haría el dibujo más pesado y choca con
  `opencv-contrib-python`.

### Lo que la sesión del Linux averiguó del cliente el 30-sep (leyendo código)

- La captura pone **`time.sleep(1 / 60)` en cada vuelta**
  (`capture_worker.py:315`, sin comentario). Candidata a quitar o reducir,
  midiendo antes y después.
- **Todo el recorrido de las imágenes va en el hilo de la ventana**, para
  todas las cámaras (lectura del tubo, envío, respuesta, dibujo).
- **Cada recuadro recibe todas las respuestas** y descarta las ajenas
  (`render_box.py:368`, `:1767-1776`).
- El dibujo sin Supervision (`render_box.py:1653-1686`): abrir el JPEG con
  Qt, rectángulos, escalar con suavizado. **Sin medir**: inclúyelo en la
  telemetría (`t_pintado`).

### Cómo probar con el usuario

1. Él hace `git pull` en `C:\Users\DESARROLLO-7\Desktop\Supermercado`
   (`git fetch origin`, `git switch feature/anaquel`, `git pull --ff-only`) y
   comprueba el hash con `git log --oneline -1`.
2. El núcleo `elde_core` va en modo editable (README del cliente): no hace
   falta reinstalar; comprobarlo con
   `clients\tienda\venv\Scripts\python.exe -c "import elde_core; print(elde_core.__file__)"`.
3. Para variables de entorno, lanzar **solo el cliente** desde PowerShell de
   Administrador (sección 1), no el `.bat`.
4. Pídele **la hora** de cada prueba: el servidor escribe en UTC y él vive en
   UTC−4.

---

## 10. Errores cometidos (no los repitas)

1. **`git checkout -- <archivo>` sobre trabajo sin commitear lo BORRA.**
2. **`por_id={}` del log de arranque es de los perfiles de ÁNGULO.**
3. **`COUNT(*)` cuenta filas, no personas.**
4. **`CALIDAD FACIAL` es acumulada.**
5. **Commitear y publicar sin preguntar** (22-sep).
6. **Fiarse de una telemetría rota** (el PAR, `pose_disponible`). Contrasta
   cada bandera con un contador.
7. **Sacar repartos de minutos de vídeo.**
8. **Escribir en este archivo una regla sin comprobarla** (el `chdir`).
9. **Una verificación que no verificaba nada** (error silenciado). El 28-sep,
   otra: `curl ... | head || echo APAGADO` nunca imprime APAGADO (el `||`
   mira a `head`).
10. **Cruzar tracks por número entre sesiones.**
11. **`pkill -f` con `|` en el patrón** (28-sep): el `|` es un «o» de regex,
    el patrón casó con procesos ajenos y con el propio comando, y el servidor
    quedó apagado sin arrancar el nuevo. Parar siempre por PID exacto.
12. **Afirmar desde un comentario del código** que un campo nuevo tiraba el
    100 % de los frames: `validacion_contrato.py:137-150` los tolera desde el
    7-sep. Lee el código que decide, no el comentario de otro fichero.
13. **Concluir de una prueba que cambia varias cosas a la vez** (29-sep, A/B
    de `CAPTURE_MAX_WIDTH`) y dar cifras restando medias con ruido. Y
    **proponer «el listón al 20 %» sin probar su robustez**: dependía de una
    muestra a 2 px del borde. Perturba la entrada (mover el área unos px)
    antes de fiarte de un resultado.
14. **Mezclar en una frase un resultado de simulación con la realidad**: se
    dijo «la mano en el pelo cuenta» sin aclarar que era con un área
    ARTIFICIALMENTE alargada; el usuario buscó el frame y vio que con su área
    no. Y se dijo «revisados en imagen» de agarres cuyas imágenes eran de
    otra pasada. Di siempre de qué variante y de qué pasada sale cada imagen.
15. **Afirmar la causa antes de medirla**: «la caja se ensancha con el brazo»
    (falso: crecía en ALTO) y «el cuello es la subida por el túnel» (la A/B
    dijo que no). Formula como hipótesis y mide.
16. **Dar rutas de ejemplo en comandos** (`C:\ruta\a\...`): el usuario las
    copia tal cual.
17. **Leer `descartados` de la telemetría como frames perdidos** (29-30 sep):
    era la resta recibidos − procesados de cada periodo, y acabó escrito en
    este archivo y en dos informes. Se deshizo mirando la fuente: el aviso
    `ColaUltimoFrame` no salía desde el 17-sep. Antes de interpretar un
    contador, lee cómo se calcula (`telemetria_fps.py`) y contrástalo con
    otra señal, como pide el error 6.
18. **Proponer un cambio sin mirar el modelo de hilos** (30-sep): se apuntó
    «desbloquear el envío antes de dibujar» como mejora fácil; en un solo
    hilo de Qt no adelanta nada (sección 9, tarea 2). Y se citaron líneas de
    una función que nadie llama (`update_tracks`, en vez de
    `_update_tracks_botsort`): comprueba que el código citado se ejecuta.

Método que sí funciona: **verificar las pruebas al revés** (desactiva el
arreglo en memoria, comprueba que falla, restáuralo) y **reproducir
grabaciones reales fuera de línea** con el grabador: la reproducción del
28-sep dio los mismos descartes que en vivo y permitió comparar reglas sobre
los mismos datos.

---

## 11. Cosas que no debes olvidar

- **Cada botón de IA es independiente**; una cámara con la IA apagada no envía.
- **iVMS:** sus ventanas extra se ocultan si se minimiza la principal.
- **No actualices `supervision` por encima de 0.30.1.**
- **WhatsApp está apagado** sin `vigilante_amazonas`.
- Los avisos de `mivolo.onnx` y `par_gender.onnx` «no presente» son **ruido**.
- **MediaPipe está vetado.** El agarre usa las muñecas de ViTPose.
- **No se usa ningún módulo de OpenCV contrib.**
- **`informes/` es del usuario y no se commitea** (fotos de clientes). Lo
  útil de los informes está resumido en 5.1 y 5.6.
