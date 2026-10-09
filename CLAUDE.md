# CLAUDE.md — ELDE Tiendas (contexto de trabajo)

Traspaso de las sesiones del 16-18, 21-22, 23, 24, 25, 28, 29 y **30 de
septiembre de 2026** (actualizado el 1-oct; **secciones 12, 13, 14 y 15
añadidas el 2, el 5, el 6 y el 9-oct por la sesión del CLIENTE**). **Este archivo SÍ se
commitea**: vive en la rama para que la siguiente sesión lo lea sin que el
usuario tenga que pegarlo.

**Rama actual: `feature/anaquel`** (publicada). Cuelga de `fix/gender` →
`claude/pensive-galileo-aymjr3` → `fix/eliminar_vigilantes_amazonas`, **no de
`main`**. Al cambiarte a ella heredas todo lo anterior.

**Hay DOS sesiones de Claude en paralelo:** una en el Linux del usuario, junto
al servidor, y otra en el **cliente** de Windows. No se ven entre ellas: solo
comparten lo que quede commiteado aquí. La del cliente tiene su encargo en la
**sección 9** (reescrita el 29-sep). El 6-oct la sesión del cliente **subió
sus commits** (los del 5-oct, 13.1, y los del 6-oct, 14.1) a
`fork/feature/anaquel`.

> **¿Eres la próxima sesión del CLIENTE (Windows)? Empieza por la sección
> 15** (7-9 oct): clasificador nuevo oficial, stock nuevo, MUCHO código sin
> commitear (15.0) y lo medido con etiquetas a mano. La 14 (6-oct), la 13
> (2 y 5-oct) y la 12 (1-2 oct) siguen valiendo como contexto; los
> pendientes vigentes están en 15.10.

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

### Hay tres servidores posibles

| | Dónde | Cómo se llega |
|---|---|---|
| **El del contenedor** | lo arranca Claude o el usuario aquí | `localhost:9000` desde Windows (túnel) |
| **El de `pop-os`** | en el anfitrión | `72.68.60.141:9000`. **Código desconocido y NO escribe en nuestro log** |
| **El del PC del cliente** (desde el 1-oct) | Windows `DESKTOP-9LHHJI2`: Quadro RTX 4000 8 GB, i5-6500 (4 hilos), 11,9 GB de RAM | `localhost:9000` en el propio PC, sin túnel. Arrancar desde `server\` con `.\venv\Scripts\python.exe -u iniciar_servidor_headless.py 9000`. **No usar `INICIAR_TIENDA.bat`: mata lo que escucha en el 9000, túnel de VS Code incluido.** Su log está en Windows, no aquí (según el informe de la sesión del cliente del 1-oct) |

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
`failed`, es tuyo. **En el PC de Windows** (6-oct): `941 passed, 15
skipped, 1 xfailed` con el código de la sección 14 (el 5-oct, 876; **el
9-oct, 964 passed** con el código sin commitear de la sección 15); la
diferencia de skipped/xfailed con Linux ya estaba antes de esos cambios
(sección 13.2). Cliente en Windows (6-oct): `clients/tienda/test` 30
passed; `packages/elde_core` 377 passed y los 72 fallos de siempre (13.2). Las de `packages/elde_core` necesitan PySide6: aquí no
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
| `c2f96e5` | ídem (5-oct) | **imagen del detector preparada en la GPU** (`imagen_a_gpu.py`): 5,22 → 7,71 fps en el PC de la tienda (sección 13.5) |
| `97b4489` | ídem (5-oct) | **reloj fino** (`perf_counter`) para el «proceso» de la telemetría: resuelve el hallazgo 34. Lleva también el CLAUDE.md con la sección 13 |
| `a27ce80` | ídem (5-oct) | **mapa de calor bajo pedido y en JPEG**: del 9,6 % al 0,9 % del hilo de imágenes (13.6) |
| `7cec236` | ídem (5-oct; cambio del 2-oct) | **filtro del fondo** por alto en px, solo en el perfil del vídeo (13.3) |
| `b2602ac` | ídem (5-oct; cambio del 1-oct) | **el escáner de ventanas ignora las del propio cliente** (12.4) |
| `face1ea` | ídem (5-oct; cambio del 1-oct) | **telemetría del ciclo del cliente** (12.4, tarea 1 de la sección 9) |
| `25c9c13` | ídem (5-oct; cambio del 2-oct) | **regla en píxeles** sobre el recuadro (12.4) |
| `3b9269f` | ídem (6-oct; cambio del 5-oct) | **ajustes contados en imágenes, ahora en segundos** (`ritmo_imagenes.py`): agarre, ByteTrack, ROI, fichas, género y track único (14.8) |
| `9b93e45` | ídem (6-oct; cambio del 5-oct) | **tarea 2**: enviar la captura guardada antes de pintar (14.2) |
| `d77ace2` | ídem (6-oct; cambio del 5-oct) | **aviso de «posible agotado»** tras un agarre (`anaqueles_stock.py`, 14.7) |
| `6d4f132` | ídem (6-oct) | interruptor **`stock_activo`** (14.7) |
| `644fdec` | ídem (6-oct) | **pose en la GPU y en lote** (`pose_gpu.py`, 14.5) |
| `004ef1b` | ídem (6-oct) | **agarres falsos en el área de abajo**: misma mano y bajada (14.6) |
| `d34a1d7` | ídem (6-oct) | **cronómetro por etapas** imagen a imagen (`tiempos_etapas.py`, 14.4) |
| `9933637` | ídem (6-oct) | **modo F**: cada imagen sale al socket en el acto, 9,85 → 14,29 fps (14.3) |

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

1. **La GPU no es el cuello** (medido **en Linux**; en el PC de Windows el
   cuello es la CPU, 13.4): de un ciclo de ~164-200 ms el servidor
   trabaja 63-80. Podría con ~12-15 fps por cámara.
2. **La conexión no está llena** (medido): con fotos a la mitad, los KB/s
   bajaron un 40 % y los fps solo subieron un 30 %. Si la subida fuera el
   techo, los KB/s se habrían mantenido y los fps casi duplicado.
   **⚠️ Corregido el 1-oct: «no está llena» NO quiere decir «no frena».**
   Con una sola imagen en vuelo, lo que tarda en subir cada imagen se suma
   entero al ciclo, aunque la conexión pase la mitad del tiempo ociosa. Ver
   5.9: los datos apuntan a que el túnel cuesta sobre todo **por tamaño**.
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

### 5.9 Servidor en el PC del cliente (1-oct) y qué frena el túnel

**Según el informe de la sesión del cliente** (local en Windows:
`informes\informe_2026-10-01_servidor_en_windows.md`; **no medido por la
sesión del Linux**):

- Con el servidor en el mismo PC: **7,3-8,1 fps** con iVMS en vivo (solo
  ~1,5 min) y **5,4 fps** con el vídeo en Media Player (17,5 min, 5.649
  frames sin errores). El servidor tarda **94-125 ms** por imagen: la GPU es
  más lenta (el detector, 39,5 ms de GPU de 46,7) y la CPU va al **98 %**,
  porque la comparten el servidor, el cliente, iVMS y Media Player.
- **Primera medida del cliente por dentro** (telemetría del ciclo): dibujar
  con QPainter **18 ms**, esperar la siguiente captura **10 ms** (no 25),
  captura → cliente 3 ms, enviar 0,6 ms. **El cliente suma ~30 ms.**
- Su reloj del «proceso» iba a saltos de 15,6 ms en Windows (hallazgo 34,
  resuelto en `97b4489`).

**Qué frena el túnel (deducido por la sesión del Linux, 1-oct):** la sesión
del cliente dedujo ~85-105 ms **fijos por viaje**. **No encaja** con la
tabla de 5.8: con imágenes de 1-17 KB, TODO el tiempo fuera del servidor
eran 28-35 ms. Si cada viaje costara 85-105 ms fijos, ni las imágenes
diminutas bajarían de ahí. Lo que encaja es **un coste por tamaño**: subir
cada imagen por el camino hasta `pop-os`, por internet o VPN. Estimación
con 4 puntos, sin prueba limpia: **~0,3 ms por KB, unos 3 MB/s (~25
Mbit/s)**, más unos ~28 ms fijos.

**Cómo decidirlo** (cliente conectado al servidor de Linux por el túnel):
1. **Coste fijo por viaje:** en PowerShell, 20 veces
   `curl.exe -s -o NUL -w "%{time_total}\n" http://localhost:9000/health`.
2. **Coste por tamaño:** telemetría del cliente con `CAPTURE_MAX_WIDTH`
   1600, 960 y 640, en tramos de 3 min seguidos. Ida y vuelta menos el
   «proceso» del servidor de Linux (reloj fino), frente a los KB por imagen.

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
| 25 | **Muñeca ≠ mano**: producto inestable; estimar la mano con el codo. Ejemplo medido el 6-oct en la grabación del vídeo: la muñeca de la clienta cae en x ≈ 229-230 px, justo en la frontera entre los recuadros dibujados de «coffemate» (borde derecho x 224-232) y «Nescafe» (borde izquierdo x 228-235): por eso el mismo agarre sale «coffemate» o «Nescafe» según la pasada | vivo |
| 26 | **El panel del cliente no muestra productos**: el bloque en vivo `metadata['anaqueles']` no los lleva (`registro.resumen()` sin productos) | pendiente (servidor + cliente) |
| 27 | Contador «FPS: N» del recuadro cuenta lecturas, no capturas (sección 3) | vivo (cliente) |
| 28 | «Archivo de configuración creado» al CARGARLA (`settings_model.py:114`) | cosmético (cliente) |
| 29 | Dashboard inalcanzable desde el navegador de Windows (sección 1) | del montaje |
| 30 | **La ruta de captura de ventana no tiene red de seguridad**: si una respuesta se pierde, el recuadro no vuelve a enviar hasta reconectar (`render_box_captura.py:574`; la ruta DVR sí libera a los 10 s, `render_box.py:583-589`). Hoy casi no pasa (un frame en vuelo); con dos en vuelo sí. **Otra forma de perderla (leído el 6-oct):** si el servidor rechaza un mensaje mal formado, contesta con `_send_error` (`app.py:1545-1547`), que NO lleva `event: inference`, y el cliente solo reparte a los recuadros los mensajes con ese `event` (`socket_client.py:316-317`, `:330-331`): el recuadro que lo envió espera para siempre. Ese error NO desbloquea a los demás recuadros (eso se dijo mal el 6-oct y se corrigió). 0 casos en las pasadas del 6-oct | vivo (cliente) |
| 14 | `_best_in_gallery` es O(N) en Python; con el TTL ya no urge | vivo |
| 31 | **El mapa de calor, las estelas y la elipse del recuadro no hacen nada sin Supervision**, y el cliente de la tienda NO lo tiene (`ModuleNotFoundError` en su `venv`, comprobado por el usuario el 30-sep). Las opciones aparecen y no avisan (`render_box.py:296-305`, `:1356-1363`); se dibuja con QPainter. Supervision queda fuera a propósito (`requirements-cliente.txt:25-28`). Los comentarios de `render_box.py:316-318` y `:1370` dicen que «el servidor pinta», pero eso solo pasa en el modo clásico | pendiente de revisar (cliente); detalle en `informes/Pendientes_Rendimiento.md`, n.º 17 (local, fuera de git) |
| 32 | **`print()` de depuración en CADA frame** (`tiendas.py:2192-2193`, «📍 ROI actualizado», y `:2371-2372`): llenan la consola (~2 MB/h, sin rotación). Hipótesis SIN comprobar: con el disco lleno, un `print` que falla rompería cada frame (eso explicaría la imagen pegada del 30-sep); al probarlo ya había espacio | vivo (servidor) |
| 33 | **Cada cámara carga SU PROPIA copia de ViTPose** (comprobado el 1-oct leyendo el código): cada `PoseMunecas` pide motor (`anaqueles_pose.py:123`) y `crear_motor_pose` crea uno nuevo cada vez (`pose_motor.py:174-199`), que carga su modelo. Con N cámaras, N copias en la GPU y un parón de carga por cámara (18 s en Windows; 4-53 s en Linux). El detector sí se comparte (`deteccion_compartida.py`) | vivo (servidor); arreglo: compartir el motor como el detector |
| 35 | **Una persona entra y sale del ROI 8 veces en 9 min** en el mismo tramo del vídeo, a 9,85 y a 14,29 fps (6-oct, 14.8): no depende de los fps. Parece alguien parado en el borde del ROI [sin comprobar] | vivo (servidor) |
| 36 | **El detector tarda ~6 ms más en las imágenes en que se calcula la pose** (30,8 → 36,8 ms de mediana, 14.4), aunque va antes que la pose y no depende de ella. Sin explicar | vivo (servidor) |
| 37 | **La espera de 10 s entre agarres** (`agarre_cooldown_s`) junta las veces que alguien mete la mano: la clienta del vídeo metió la mano 8 veces en la balda alta (contadas por el usuario el 6-oct) y salen 2-3 agarres. **Decisión del usuario pendiente:** contar cada vez que mete la mano o una vez por visita con intención de coger | pendiente (decisión) |
| 38 | **El Anaquel 1 da 2 o 3 agarres** según la pasada en el tramo de la clienta (y según se mueva el área 8 px, 14.6): sensible a dónde está dibujado | vivo; sin investigar |
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
2. **Rendimiento del ciclo:** la telemetría del cliente existe (12.4) y el
   reparto está medido imagen a imagen en el PC de Windows (14.4). Hecho el
   5-oct: imagen del detector en la GPU (`c2f96e5`), reloj fino (`97b4489`)
   y mapa de calor bajo pedido (13.6). Hecho el 6-oct: tarea 2, pose en la
   GPU y en lote, y modo F (de 7,7 a 14,3 fps en el vídeo, 14.2). Lo que
   queda, en 14.11.
   Lista vieja, de fácil a difícil, en `informes/Pendientes_Rendimiento.md`
   (local, en Linux).
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
13. **Del 1-oct:**
    - **Compartir ViTPose entre cámaras** (hallazgo 33).
    - **La prueba de qué frena el túnel** (5.9).
    - En el PC de Windows: **vigilar la RAM** (llegó a 420 MB libres) **y el
      disco** (bajó de 21 a 13 GB libres). El 5-oct: el disco baja 2-3,5 GB
      al cargar ViTPose y vuelve al apagar (13.9).
    - **Antes de proponer TensorRT o fp16:** `ENABLE_TENSORRT` solo afecta a
      los modelos ONNX (caras, ropa, edad; `tiendas_modelos.py:110-113`),
      **no al detector RF-DETR**, que va en PyTorch. Y el fp16 del detector
      solo se lee de la configuración de `vigilante_amazonas`
      (`deteccion_compartida.py:66-86`): aquí hace falta cambiar código.

---

## 9. Encargo para la sesión del CLIENTE (reescrito el 29-sep)

Escrito por la sesión del Linux. **No nos vemos**: esto es lo único que nos
comunica. **El usuario no puede tener el cliente y tu sesión abiertos a la
vez**: todo lo que midas tiene que quedar ESCRITO en disco para leerlo
después. **Avísale de que el fichero solo se completa al cerrar el cliente.**

### Lo que ya está medido (no lo repitas)

- El ciclo completo es de **~164-200 ms** con una cámara (5-6 fps); el
  servidor trabaja **63-80 ms** de ese tiempo (sección 5.4). **Ni la GPU ni la
  conexión son el techo** (en Linux; en Windows, la CPU: 13.4).
- **El cliente espera la siguiente captura** después de cada respuesta
  (sección 3, paso 5): ~25 ms de media a 20 capturas/s.
- **`CAPTURE_MAX_WIDTH=960`**: +30 % fps, pero más caras perdidas. **No se
  recomienda.** No lo repitas salvo para otra cosa.
- **La ruta que importa es la de captura de ventana** (`render_box_captura.py`
  en `packages/elde_core`, el mixin que usa `render_box.py`): por ahí van iVMS
  y el vídeo. La ruta DVR (`_on_dvr_frame`, `render_box.py:561-646`) hoy no
  se usa.

### Tarea 1 — Medir el ciclo en el cliente (lo primero)

**Estado: hecha por la sesión del cliente el 1-oct, commiteada el 5-oct
(`face1ea`, sin push).**
Resultados en 5.9. Datos en `%LOCALAPPDATA%\tienda_view\telemetria_ciclo.jsonl`
del PC del cliente.

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

**Estado: hecha el 5-oct (14.1).** Medido en el vídeo: 7,71 → 9,86 fps. Su
mejora la remató el **modo F** del 6-oct (14.3): sin él, la imagen enviada
esperaba en el búfer de Qt a que terminara el dibujo. Lo de abajo es el
encargo original.

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
**Esperado, con lo medido el 1-oct (5.9): ~−28 ms por ciclo**: ~10 ms de
esperar la captura y ~18 ms de dibujo, si se envía antes de dibujar. Medir
con la tarea 1 y con las líneas `FPS` del servidor.

### Tarea 3 — Mejora grande: dos fotogramas en vuelo

**Estado (6-oct): sin hacer, y ahora probablemente rinde poco** (deducido,
sin medir): con la tarea 2 y el modo F el cliente ya pinta mientras el
servidor trabaja, y el ciclo (70 ms de media) es casi el tiempo del
servidor (65 ms; 14.2). Una segunda imagen en vuelo esperaría en la cola
del servidor.

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

---

## 12. Sesión del CLIENTE en Windows (1-2 oct)

Escrito el 2-oct por la sesión del cliente **para la siguiente sesión del
cliente, aquí en Windows**. Todo está medido o leído en el PC
`DESKTOP-9LHHJI2`, salvo lo marcado como **deducido**. **El código de esta
sección NO está commiteado todavía** (lista de archivos en 12.4). Los
informes citados viven en `informes\` **de Windows** (fuera de git).

### 12.0 Para empezar (próxima sesión del cliente)

1. **Lee la sección 0** y los cuatro archivos obligatorios, como siempre.
2. **Tus herramientas de la sesión anterior** ya no están en tu scratchpad:
   se copiaron a `informes\herramientas_cliente\`.

   | Herramienta | Qué hace |
   |---|---|
   | `apagar_servidor.ps1` + `ctrl_c.py` | Apagar el servidor en orden |
   | `correr_tests.py` + `stub\pytest.py` | Correr pruebas sin pytest (`$env:PYTHONPATH` apuntando a `stub`) |
   | `analizar_telemetria.py` | Analizar la telemetría (`--desde HH:MM --hasta HH:MM`) |
   | `monitor_recursos.ps1` | Registrar CPU, RAM y GPU |
   | `estado_manana.ps1`, `cobertura.py` | Ver qué hay encendido y qué cubren los logs |
   | `noche_por_hora.py`, `cifras_noche.py` | Las cifras de los informes |
   | `cliente_prueba.py` | Un cliente de mentira |
   | `banco_detector_gpu.py` | El banco del detector |
3. **Mira qué hay encendido:** `/health` en `http://127.0.0.1:9000`
   (`servidor_nombre = DESKTOP-9LHHJI2` es el servidor de este PC). El
   cliente corre elevado: desde tu sesión **no puedes cerrarlo**; pídeselo al
   usuario.
4. **Lanzar el cliente como Administrador desde tu sesión:**
   `Start-Process ...\clients\tienda\venv\Scripts\python.exe -ArgumentList 'src\main.py' -WorkingDirectory ...\clients\tienda -Verb RunAs`.
   Sale el aviso de Windows y el usuario tiene que aceptarlo; la herramienta
   puede quedarse esperando hasta entonces.
5. **El usuario trabaja con la VPN puesta** para poder hablar contigo. Con
   ella no se llega a pop-os directamente (12.5), pero el servidor de este PC
   no la necesita.

### 12.1 El servidor ya corre en el PC de la tienda

- **`server\venv` rehecho el 1-oct.** El anterior heredaba el Python global
  (`include-system-site-packages = true`), con torch 2.9.1 **solo CPU** y sin
  `rfdetr`; quedó apartado en `server\venv_viejo_cpu`. El nuevo: Python
  3.12.10, torch 2.8.0+cu128, onnxruntime-gpu 1.23.2, `opencv-contrib-python`
  4.12.0.88 y `elde_core` editable. `verificar_entorno.py` sale con código 0.
- **`easyocr` metió `opencv-python-headless` 5.0.0.93**: hubo que quitarlo,
  como avisa el README. Si no, se cae el stack facial.
- **Los pesos de Git LFS** (`rf-detr-medium.pth` y el de ViTPose) estaban
  borrados en esa copia: se repusieron con `git restore` (386 y 328 MB).
- **torch con internet malo:** pip falló dos veces (tope de 10 min de la
  herramienta y cortes de red). Lo que funcionó: bajar el wheel con
  `curl -C - --retry`, que reanuda; comprobar su SHA-256 contra el índice de
  PyTorch; e instalarlo desde el disco con `--no-deps`. Después,
  `-r requirements` ya lo da por instalado. Con `--find-links`, pip lo volvió
  a bajar de internet.
- **Arrancar:** desde `server\`,
  `.\venv\Scripts\python.exe -u iniciar_servidor_headless.py 9000`, con la
  salida a `server\logs\consola_*.log`. `/health` responde en 6-16 s; la
  primera imagen tarda ~20 s (carga de RF-DETR); con la primera persona junto
  a un estante, ViTPose tarda ~18 s **por cámara** (hallazgo 33, visto
  también en la memoria de GPU: +0,5 GB con cada cámara nueva).
- **Parar en orden en Windows** (lo que en Linux es `kill -TERM`): mandar
  Ctrl+C a la consola del proceso desde otro python, con `AttachConsole` y
  `GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0)`. Se apaga en ~3 s y vuelca el
  mapa de calor de la hora en curso («Application shutdown complete»).
  `Stop-Process` lo mata sin volcar nada.
- **Va en hora LOCAL**, no en UTC: el mapa de calor `…_16` son las 16h, y la
  jornada del anaqueles cambia a medianoche de Venezuela.
- **Recursos con una cámara:** el servidor ocupa 3-3,6 GB de RAM (de 11,9) y
  2,2-2,7 GB de GPU (de 8). **La CPU (i5-6500, 4 hilos) va al 92-98 %**,
  porque la comparten el servidor, el cliente, iVMS y Media Player; la GPU,
  al 26-28 %.

### 12.2 Rendimiento medido (una cámara)

| Fuente | fps | Proceso en el servidor | Fuera del servidor |
|---|---:|---:|---:|
| Vídeo en Media Player (1-oct 13:05-13:23, 5.649 imágenes) | 5,4 | 110-125 ms | ~45-60 ms |
| **iVMS en vivo, cámara 3** (1-oct 16:46-17:31, 17.270 imágenes) | **6,8** | **94 ms** | **~30 ms** |
| iVMS en vivo, noche del 1 al 2-oct | 7,3-7,5 de día · 8,2 de noche (imagen oscura, 38 KB) | 94 · 79 ms | — |

- **El cliente por dentro** (telemetría, iVMS en vivo): cada ciclo dura
  122-125 ms. De ellos, la ida y vuelta al servidor son 97 ms, dibujar 14-15,
  esperar la siguiente captura 7-9, y el resto ~1. **El cliente suma ~25 ms.**
- **El detector solo** (banco aislado con el perfilador de torch):
  `detectar()` tarda 46,7 ms, de ellos **39,5 ms de GPU** y 7,2 de CPU. En la
  3090 Ti, el detector completo tarda 32-36 ms.
- **La CPU ocupada por otros programas** (Media Player) añade 15-30 ms al
  proceso: 94 → 110-125 ms, con ±15 ms de margen por el reloj (hallazgo 34,
  resuelto el 5-oct en `97b4489`).
- **Lectura:** en este PC el cuello es el procesador compartido, más una GPU
  más lenta. Con el túnel de Linux era la subida de cada imagen (5.9). Los
  fps salen parecidos (~5-7).

### 12.3 Resultados de la cámara 3 con el servidor en el PC

Informes `informe_2026-10-01_tarde.md` e `informe_2026-10-02.md` (Windows).

| Tramo (Venezuela) | Capturadas | Identificadas | Clasificadas | Anaqueles |
|---|---:|---:|---:|---|
| 1-oct 16:46-17:31 | 89 | 11,2 % | 5,6 % | 1 visita A2, 0 agarres |
| 1-oct 18:02-20:48 | 397 | 17,1 % | 10,8 % | 11 visitas A2, **2 agarres** |
| 2-oct 06:54-09:12 | 185 | 15,7 % | 7,0 % | 7 visitas A2, 0 agarres |

- **Los dos agarres, sin revisar en iVMS:** 1-oct **18:34:16**, A2 →
  **colcafe** (identidad `eaee3867`, mujer adulta según su ficha); y
  **19:00:45**, A2, **sin producto** (`7994b019`, hombre de 26-35).
- **El código corto del log** («visita #3 … (f93e)») son los 4 últimos
  caracteres del pid (`_corto`, `anaqueles.py:1771`). Si es hexadecimal, es
  una identidad y se encuentra en `tienda_capturas.person_uuid`; si son
  cifras, es un `prov:<track>`.
- **El anaqueles apunta el género del MOMENTO de la visita** (`_rasgos`,
  `anaqueles.py:1338` y `:1349`). La ficha puede decidirlo después: por eso
  la mayoría de las visitas sale «Desconocido».
- **Personal probable** (deducido, no visto): `b2a68313`, 49 entradas en
  2 h 27 min; `01c833f0`, el 14 % de toda la permanencia; `b5ef8905`,
  reconocida 20 veces.
- **Noche:** 1.175 seguimientos que solo dejaron 11 fichas, todas «sin
  rostro», con la imagen a oscuras; y el anaqueles suspende el conteo cada
  30 s (1.210 veces).

### 12.4 Cambios en el código (commiteados el 5-oct, sin push: 13.1), todos con pruebas

1. **Telemetría del ciclo** (tarea 1 de la sección 9):
   `packages/elde_core/elde_core/ui/telemetria_ciclo.py` (nuevo), enganches
   en `render_box_captura.py` y `render_box.py`, e interruptor en el menú
   Vista (`telemetria_ciclo_boolean`, guardado desde `windows_main.py`).
   Escribe en `%LOCALAPPDATA%\tienda_view\telemetria_ciclo.jsonl` y vacía a
   disco cada 2 s, así que **se puede leer en vivo**. Rota a 20 MB con una
   copia, y **de noche se queda corto**: hay que ampliarlo. Marcas:
   `t_captura`, `t_recibido` (el `t_leido` del encargo), `t_envio`,
   `t_respuesta`, `t_paneles` y `t_pintado`, más `descartados`, `bytes`,
   `personas` y `estado`. 10 pruebas.
2. **Escáner de ventanas:** `windows_detector.py` (`es_del_propio_proceso`) y
   `window_monitor.py` ahora descartan las ventanas del propio proceso. **La
   causa:** la lista de ventanas a ignorar se arma al importar el módulo,
   antes de que el cliente tenga ventanas. Los menús de Qt (título «python»)
   se tomaban por fuentes de captura, y su miniatura les hacía
   `BringWindowToTop` con 0,2 s de pausa: por eso «Añadir producto…» no abría
   su diálogo. 2 pruebas, con ventanas reales de Windows.
3. **Regla manual en píxeles:**
   `packages/elde_core/elde_core/ui/regla_pixeles.py` (nuevo), más
   `interactive_image_label.py` (el ratón y el pintado van ANTES de las
   salidas tempranas, para que funcione sin zonas activas) y el menú Vista en
   `render_box.py`. Mide en píxeles de la imagen que se manda
   (`x * ancho / 1000`, la misma conversión que las zonas). No se guarda. 5
   pruebas.
4. **Descartar a la gente del fondo:**
   `server/src/analityc/core/filtro_fondo.py` (nuevo); `PERSON_MIN_HEIGHT_PX`
   en `config.py` (global 0, apagado); la llamada en `tiendas.py`, entre
   `detectar_sv` y ByteTrack; el perfil
   `server/config/camera_profiles/win-Media_Player-c3.json` (**235, solo para
   el vídeo**, medido por el usuario con la regla); y el README de los
   perfiles. Cada 60 s deja en el log
   `FONDO [cam] descartadas=N de M detecciones`. 7 pruebas. En vivo (2-oct
   14:10-14:20) descarta entre el 30 y el 100 % de las detecciones, y los fps
   pasan de 5,4 a ~6,0 (comparación NO limpia: otro tramo del vídeo).

**Cómo se probó:** las pruebas de `elde_core`, con el venv del cliente; ese
venv no tiene pytest, así que se usó un mini-ejecutor y un doble de pytest en
el scratchpad. Las del servidor, con `server\venv`. **No se ha corrido la
suite entera del servidor en Windows.**

### 12.5 Trampas de Windows

- **NTFS no actualiza el tamaño ni la fecha de un fichero abierto:** el
  listado de la carpeta engaña (unos logs parecían «parados» y se estaban
  escribiendo). Hay que leer el contenido.
- **Un proceso elevado** (el cliente, lanzado como Administrador) no se puede
  cerrar desde una sesión normal, ni se puede leer su línea de órdenes.
- **Con la VPN (Proton) puesta**, cualquier conexión directa a `72.68.60.141`
  (puerto 9000, SSH 2223, ping) da `WinError 10013`.
- **En una prueba de Qt, no fuerces `QT_QPA_PLATFORM=offscreen`:** la
  QApplication es una por proceso, y rompe las pruebas que necesitan
  ventanas reales.
- Los scripts de medida (analizador de la telemetría y registrador de CPU,
  RAM y GPU) están en `informes\datos_2026-10-01\` de Windows.

### 12.6 Lista de revisión del 2-oct (estado al 5-oct)

**Hecho** (detalle en la sección 13):

- **Revisión del filtro del fondo:** bien colocado, antes de ByteTrack. La
  suite completa pasa (13.2) y la A/B sobre el mismo trozo del vídeo está
  hecha (13.3). Los anaqueles no cambian con el filtro.
- **Este archivo frente al remoto:** comprobado el 5-oct. El commit de la
  sesión Linux del 1-oct no llegó nunca (13.0).

- **Los commits:** hechos el 5-oct, uno por funcionalidad (13.1). Falta el
  push, cuando el usuario lo diga.

**Pendiente** (también en 13.10):

1. **Riesgos del filtro sin comprobar:** alguien cerca pero tapado por una
   balda, o agachado, mide menos que el umbral y se pierde. Se mirarán con
   las fotos de las fichas pequeñas (13.3).
2. **El umbral de la cámara 3 en vivo:** medirlo con la regla en iVMS y crear
   `win-iVMS-4200-c3.json`. **El 235 es del vídeo.**
3. **«Escena cambió / se reanuda» en el mismo segundo:** visto el 1-oct
   (17:26:37 y 17:29:32, con la ventana minimizada, y siguió contando sobre
   imágenes vacías) y el 2-oct (15:46-15:47). El 30-sep marcó «sin señal».
   Sin mirar el código.
4. **La telemetría del cliente:** ampliar la rotación, que el 5-oct rotó en
   mitad de una pasada. Si se quiere, pasar `t_recibido` a `t_leido`.
5. **Los 2 agarres del 1-oct,** por verificar en iVMS.
6. **La prueba del túnel** (5.9). Hay que apagar antes el servidor de
   Windows, que ocupa `localhost:9000`.

### 12.7 Errores de la sesión del cliente (no los repitas)

1. **Deducir de la A/B que «el túnel cuesta 85-105 ms fijos por viaje»:**
   era falso, lo corrigió la sesión del Linux (5.9).
2. **Ofrecer TensorRT «para el detector»:** `ENABLE_TENSORRT` solo cambia los
   modelos ONNX.
3. **Dimensionar la rotación de la telemetría con los fps de día:** de noche
   llegan más imágenes por segundo y se llenó antes.
4. **Decir «el log no se escribe» mirando el listado de Windows** (12.5).
5. **Escribir una prueba que forzaba `offscreen`** y rompía otra (12.5).

---

## 13. Sesión del CLIENTE en Windows (2 y 5 oct)

Escrito el 5-oct por la sesión del cliente, para la siguiente. Medido en el
PC `DESKTOP-9LHHJI2` salvo lo marcado como **deducido** o **sin comprobar**.
El servidor de Windows escribe su log en **hora local** (Venezuela).

### 13.0 Para empezar

1. **El usuario NO quiere informes nuevos ni que se actualicen los de
   `informes\` sin pedirlo.** Da los resultados en el chat. Tampoco copies
   datos a `informes\` sin preguntar: es su carpeta.
2. **El remoto de esta rama es `fork`** (`Dsha40/ELDE-TIENDAS`), no
   `origin`. El 5-oct, `fork/feature/anaquel` seguía en `8100b95`: el commit
   de la sesión Linux del 1-oct **nunca se subió**. Si se sube, chocará con
   el CLAUDE.md de aquí.
3. **Herramientas nuevas** en `informes\herramientas_cliente\` (además de
   las de 12.0):

   | Herramienta | Qué hace |
   |---|---|
   | `medir_cpu.ps1` | Espera a que lleguen imágenes y mide 60 s: py-spy del servidor **sin pausarlo** + CPU de cada proceso del PC. py-spy ya está instalado en `pyspy\` al lado |
   | `etapas_pyspy.py` | Reparte una medida de py-spy por etapas del pipeline |
   | `bench_heatmap.py` | Cronometra pintar/guardar el mapa de calor y abrir un JPEG |
   | `tramos.py` | Tramos en que el cliente envió imágenes (telemetría). Lleva la fecha del 2-oct escrita: cámbiala |
   | `contar_pasadas.py`, `recursos_pasadas.py`, `altos_fichas.py` | Conteos, recursos y alto de las fichas por ventana: `"A=HH:MM:SS-HH:MM:SS" ...` |

4. **pytest en Windows:** `server\venv` no lo tiene. Se instala aparte con
   `--target` en el scratchpad y se usa con `$env:PYTHONPATH`, más
   `$env:ELDE_TIENDA_DB` con ruta absoluta.
5. **Reiniciar el servidor:** `apagar_servidor.ps1` (Ctrl+C, vuelca el mapa
   de calor) y arrancar desde `server\` con `Start-Process` y
   `-RedirectStandardOutput/-Error`. **Antes de una A/B, ten la pose cargada**
   (~1 min de vídeo hacia las 11:20:30; el log dice «ViTPose cargado»): su
   carga congela ~18 s.
6. **Antes de dejar el sistema solo, comprueba que llegan imágenes** (líneas
   `FPS [` en el log del último minuto), no solo que `/health` responde (13.9).

### 13.1 Estado del código (5-oct, ~13:00)

**Subidos el 6-oct** (14.1). El 5-oct quedaron commiteados en local: 7
commits por delante de `fork/feature/anaquel`, que seguía en `8100b95`. Uno
por funcionalidad, con la fecha REAL del cambio en el asunto. Cada uno se
comprobó solo, con la suite sobre una copia limpia de ese commit.

| Commit | Qué |
|---|---|
| `c2f96e5` | imagen del detector preparada en la GPU (13.5) |
| `97b4489` | reloj fino + este CLAUDE.md con la sección 13 |
| `a27ce80` | mapa de calor bajo pedido y en JPEG (13.6) |
| `7cec236` | filtro del fondo (13.3) |
| `b2602ac` | escáner de ventanas, con el arreglo de su prueba (13.2) |
| `face1ea` | telemetría del ciclo del cliente |
| `25c9c13` | regla en píxeles |

- `config.py` (filtro + mapa) y `render_box.py` (telemetría + regla) se
  repartieron: cada commit lleva solo su parte.
- **Suite en copias limpias:** servidor 876 passed; cliente, los mismos 72
  fallos de siempre y ninguno nuevo. Dos pruebas de la baliza se saltan si
  el servidor está encendido (ocupa el puerto 9010).
- **Las correcciones de este CLAUDE.md tras los commits están sin commitear**
  (el usuario lo pidió así).
- **El servidor de Windows corre con todo eso** (arrancado el 5-oct a las
  11:18:58, modo G y mapa bajo pedido).

### 13.2 Pruebas en Windows

- **Servidor:** con el último commit del 1-oct (`8100b95`), 854 / 15 / 1. Con
  los cambios del 2-oct, 861. Hoy, **876 passed**. La diferencia con Linux
  (852 / 16 / 2) ya estaba con `8100b95`: es de Windows.
- **Cliente (`packages/elde_core`):** el venv del cliente es **Python
  3.14.7**. Dan **72 fallos con y sin cambios**: son las pruebas que buscan
  `clients/amazonas`, `detecciones` y `perimetrales`, que no están en este
  repo.
- **La prueba del escáner fallaba en la pasada completa:** cinco pruebas
  antiguas (`test_flecha_direccion.py:20` y otras) ponen Qt en `offscreen`
  antes, y sin pantalla no hay ventanas reales. Arreglo: si Qt va
  `offscreen`, se salta. Sola pasa.

### 13.3 Filtro del fondo: A/B sobre el mismo trozo del vídeo (2-oct)

A = con filtro (15:49:36-15:58:11), B2 = sin filtro (17:18:42-17:27:17). Son
**515 s del mismo trozo**: las visitas y los agarres caen en el mismo segundo
del vídeo, con un desfase fijo.

| | A, con filtro | B2, sin filtro |
|---|---:|---:|
| Personas seguidas | 24 | 57 |
| Fichas | 13 (todas ≥ 288 px) | 23 (14 de 56-187 px) |
| Fichas «sin rostro» | 2 | 10 |
| Entradas al ROI | 8 | 19 |
| **Visitas / agarres** | **4 / 3** | **4 / 3** |
| fps | 5,31 | 5,10 |

- El filtro quita el 58 % de los seguimientos y el 43 % de las fichas, y no
  toca los anaqueles. Apenas acelera (~4 %).
- **Pendiente: mirar las fotos** de las 14 fichas pequeñas de la B2 (creadas
  entre las 17:18:42 y las 17:27:17), para confirmar que son gente del fondo.
- 14 fichas pequeñas **no son 14 personas del fondo**: sin filtro, quien
  entra por el fondo del pasillo arranca su ficha cuando aún se ve pequeño.
- **La primera B (17:03-17:08) era otro trozo del vídeo.** Su «13 % más de
  fps» con filtro era falso.
- El 235 es solo del vídeo. La cámara 3 en iVMS **no tiene filtro**.

### 13.4 Dónde se va la CPU

**Hardware** (leído el 5-oct):

| Pieza | Qué es | ¿Frena? |
|---|---|---|
| Procesador | i5-6500, **4 núcleos / 4 hilos**, 3,2 GHz (2015) | **Sí, es el cuello principal** |
| Memoria | 12 GB en **dos módulos distintos** (4 + 8 GB), a 1600 | Justa: con todo abierto quedan ~1,5-2,4 GB libres |
| GPU | Quadro RTX 4000, 8 GB, PCIe gen3 x8 | No: va al 25-32 % |
| Disco | SSD SATA de 224 GB, ~18 GB libres | No por velocidad |

**En todo el PC** (2-oct, vídeo, código viejo): CPU al 99 %. El servidor se
lleva el 50 %, la captura del cliente el 9,5 %, el cliente el 6 %, `dwm` el
5,6 % y el editor ~6,9 %. Las consultas WMI de las propias medidas, ~3 %.

**Hilo de las imágenes del servidor**, con py-spy **sin pausar**:

| Etapa | 2-oct (código viejo) | 5-oct (modo G + mapa bajo pedido) |
|---|---:|---:|
| Detector | 64,4 % (31,4 % era preparar la imagen en la CPU) | 43,9 % |
| Pose | no salió (sin gente en el estante) | 16,8 % |
| Caras e identidad (ONNX) | 7,0 % | 9,6 % |
| Abrir el JPEG que llega | 6,7 % | 9,3 % |
| Mapa de calor | 9,1 % | **0,9 %** |
| Comprobar la escena | 3,7 % | 5,7 % |
| MiVOLO | 3,7 % | 5,0 % |

⚠️ **py-spy con `--native` pausa al servidor en cada muestra.** Repartió mal
el tiempo (dio un 40 % falso al mapa de calor) y dejó al servidor 47 s sin
responder. **Usa siempre `--nonblocking`** y contrasta con un cronómetro.

### 13.5 Imagen del detector en la GPU (`c2f96e5`): A/B G frente a C

G = imagen preparada en la GPU (09:54:35-10:03:49), C = interruptor
`ELDE_DETECTOR_PREPARAR_EN_GPU=0` (10:10:36-10:19:50). Mismos 554 s de vídeo
y con filtro.

| | **G** | C |
|---|---:|---:|
| fps | **7,71** | 5,22 |
| Ciclo completo | 104 ms | 164 ms |
| Servidor por imagen (mediana de los tramos de 30 s) | **57 ms** | 106 ms |
| CPU del servidor / CPU total | **16,5 % / 70 %** | 55 % / 100 % |
| Tiempo con la CPU ≥ 95 % | 26 % | 77 % |
| Agarres / visitas / fichas | 3 / 3 / 12 | 3 / 4 / 12 |

- **Banco fuera de línea:** detecciones idénticas (diferencia 0,0) en 12
  imágenes con 84 personas.
- **El modo G es el de por defecto del servidor**, viva donde viva el
  cliente. Solo vuelve atrás con el interruptor, sin CUDA o si falla.
- La C reproduce las ~5,3 fps del 2-oct: la mejora es del cambio.

### 13.6 Mapa de calor bajo pedido y en JPEG (`a27ce80`)

- **Antes:** cada 5 s, dentro del procesado de una imagen, se pintaba el mapa
  a 1456×868 (~174 ms) y se guardaba en PNG (~258 ms). Eso era el ~9,6 % del
  hilo y producía un tirón de ~430 ms.
- **Ahora:**
  - el procesado solo guarda el JSON, el fondo cada 60 s y el estado;
  - `heatmap_bajo_pedido.py` pinta cuando alguien pide
    `/dashboard/img/heatmap/<cámara>.png` (JPEG por dentro, misma dirección),
    `/api/v1/heatmaps` o el PDF;
  - las cámaras se listan por su `.json`;
  - el histórico por horas sigue en PNG;
  - `HEATMAP_PNG_PERIODICO = True` vuelve a lo de antes.
- **En vivo:** el mapa pasa del 9,6 % al **0,9 %** del hilo. Pedir la imagen
  cuesta **158 ms** y, desde la caché de 5 s, **5-20 ms**. El PNG ya no se
  reescribe.
- 8 pruebas nuevas (`test_heatmap_bajo_pedido.py`), verificadas al revés. Dos
  pruebas de `test_heatmap_persistencia.py` cambiadas a propósito: el volcado
  deja JSON y estado, no PNG.

### 13.7 La captura del cliente (5-oct, leído y medido)

`capture_worker.py` captura **siempre**, se envíe o no: ~17-18 capturas por
segundo con Media Player, de las que se usan ~5. **Se tiran 2,3 por cada
imagen enviada.**

| Paso | Coste |
|---|---:|
| `PrintWindow` modo 0 (app normal) | 11,5 ms |
| `PrintWindow` modo 2 (app que dibuja con GPU, como Media Player; iVMS sin medir) | 33,1 ms |
| JPEG calidad 70 con PIL (OpenCV, 5,2 ms: no compensa cambiarlo) | 5,6 ms |
| Firma anti-congelada | 6,3 ms, 1 de cada 30 |

**Ideas, sin hacer:**
1. **Capturar bajo pedido, con una de reserva.** Ahorro estimado: ~6 % del PC.
2. **Leer el stream de la cámara directamente** (ruta DVR/RTSP del panel
   Dispositivos), en vez de copiar la ventana de iVMS. Hoy la imagen se
   descomprime en iVMS, se pinta, se copia, se vuelve a comprimir y el
   servidor la descomprime otra vez.
3. **Mandar la imagen sin comprimir** cuando cliente y servidor comparten
   PC. Ojo: el formato del mensaje lo usan otros clientes (GUIA 4.7).

**Pendiente:** medir con iVMS abierto (no minimizado) qué modo de copia
necesita y si el panel Dispositivos conecta con la cámara 3.

### 13.8 La pose cuando hay gente en el estante

Con el modo G, el servidor tarda **57 ms con 0 personas**, 64 con 1, 82 con
2, 109 con 3 y 162 con 4. Con la clienta junto al estante, 95-135 ms
(4,4-6,6 fps). Con la misma gente lejos del estante, 62-65 ms.

En py-spy (5-oct 10:50, sin la carga del modelo), la pose es el 14,7 % del
hilo:

| Paso | % del hilo |
|---|---:|
| Preparar el recorte (CPU, preparador de Hugging Face) | **7,1 %** |
| El modelo (GPU) | 4,6 % |
| Posproceso (scipy, CPU) | 2,1 % |
| Dar la vuelta a los colores | 0,7 % |

Candidato a llevar a la GPU, pero **más delicado que el detector**: no
saldría idéntico, y un píxel de muñeca puede cambiar un agarre. Hay que
probarlo con pruebas de puntos, con el tramo de la clienta y con una A/B.

### 13.9 Otros hallazgos

- **Fin de semana del 2 al 5 de oct: ningún dato.** El cliente estuvo
  conectado 63 h y sus paneles consultaron al servidor ~36.000 veces, pero
  ningún recuadro envió imágenes. Según el usuario, las cámaras no daban
  señal. La sesión no comprobó, antes de que se fuera, que llegaban
  imágenes.
- **El disco libre baja 2-3,5 GB cada vez que se carga ViTPose** y vuelve al
  apagar el servidor. Probablemente es el archivo de paginación [sin
  comprobar].
- **La telemetría del cliente rota a 20 MB y se queda corta:** el 5-oct
  rotó en mitad de la pasada G. Hay que ampliarla, y para analizar hay que
  leer también `telemetria_ciclo.jsonl.1`.
- **«Escena cambió / se reanuda» en el mismo segundo**, otra vez el 2-oct a
  las 15:46-15:47, mientras el usuario se movía por el vídeo. Sin mirar el
  código.
- **La clienta del Coffee-mate** (11:21:05 de cámara, aprox.) da **3
  agarres** en todas las pasadas; la referencia de 5.5 decía 2. El producto
  cambia de una pasada a otra (hallazgo 25).
- Dashboard, API v1 y PDF dependían del PNG del mapa de calor (13.6).

### 13.10 Pendientes, por orden

**Lista del 5-oct; la vigente es la 14.11.**

1. ~~**Push** de los 7 commits (13.1) y de las correcciones de este
   CLAUDE.md~~: hecho el 6-oct (14.1).
2. **Mirar las fotos de las 14 fichas pequeñas** (13.3).
3. **iVMS:** el modo de copia y el stream directo de la cámara 3 (13.7), en
   cuanto tenga señal.
4. ~~**Pose en la GPU** (13.8), con su A/B~~: hecho el 6-oct (14.5).
5. **Captura bajo pedido** (13.7, idea 1).
6. **Caras (9,6 %) y abrir el JPEG (9,3 %):** medir dentro qué parte es CPU.
7. **Compartir ViTPose entre cámaras** (hallazgo 33). Importa con varias
   cámaras.
8. **Telemetría del cliente:** ampliar la rotación.
9. Del CLAUDE.md y los informes del 2-oct, sin empezar:
   - umbral del filtro para la cámara 3 en iVMS;
   - «escena cambió / se reanuda»;
   - verificar los 2 agarres del 1-oct;
   - dar de alta al personal;
   - ignorar la imagen oscura de noche;
   - panel de Anaqueles «Persona (Desconocido)»;
   - «configuración creada» al cargar;
   - doble conteo entre cámaras;
   - el listón de la demografía;
   - mano con codo → muñeca;
   - la prueba del túnel.
10. **Hardware** (decisión del usuario): si hiciera falta, primero un
    procesador con más núcleos; la RAM, dos módulos iguales de 8 GB.

### 13.11 Errores de esta sesión (no los repitas)

1. **Dar el 40 % del mapa de calor con py-spy `--native`** (pausa al
   servidor). Usa `--nonblocking` y contrasta.
2. **Comparar pasadas de trozos distintos del vídeo** (el «13 %» del
   filtro). Antes de comparar, comprueba que los mismos sucesos caen en el
   mismo segundo del vídeo, y apunta la hora de cámara **antes** de cada
   pasada.
3. **Contar mal y afirmar sin mirar:** «3 visitas» eran 4; los agarres «de
   la clienta» antes de tener la hora de cámara; «vídeo parado» cuando solo
   se sabía que no llegaban imágenes.
4. **Atribuir al detector una línea genérica de torch** (`_call_impl`) que
   también usan otros modelos. Reparte por la marca más interna de nuestro
   código.
5. **No comprobar, antes de dejar el sistema solo, que llegaban imágenes.**
6. **Meter medidas (py-spy, bancos) en medio de una pasada:** ensucian sus
   fps.
7. **Actualizar el informe varias veces sin que el usuario lo pidiera.**

---

## 14. Sesión del CLIENTE en Windows (6 oct)

Escrito el 6-oct por la sesión del cliente, para la siguiente. Medido en el
PC `DESKTOP-9LHHJI2` con el **vídeo de la cámara 3 en Media Player** (no con
iVMS en vivo: no hubo señal), salvo lo marcado como **deducido** o **sin
comprobar**. Horas del log del servidor de Windows: locales (Venezuela).

### 14.0 Para empezar

1. Lo de 13.0 sigue valiendo: informes solo cuando el usuario los pida, el
   remoto es `fork`, pytest aparte con `--target`.
2. **Interruptores nuevos:**

   | Interruptor | Dónde | Qué hace | Por defecto |
   |---|---|---|---|
   | `ELDE_MODO_F=0` | una línea en `clients/tienda/.env` (vale con `INICIAR_TIENDA.bat`) o variable en la consola del cliente | apaga el modo F (14.3) | encendido |
   | `ELDE_POSE_PREPARAR_EN_GPU=0` | variable del servidor | pose como antes: persona a persona y preparada en la CPU (14.5) | encendido |
   | `stock_activo: false` | `server/config/anaqueles_config.yaml` | apaga el aviso de stock (14.7) | encendido |
   | `ELDE_TIEMPOS_ETAPAS=<ruta absoluta .jsonl>` | variable del servidor | cronómetro por etapas (14.4) | apagado: es diagnóstico |

3. **Si el log del servidor dice `tiempos_etapas: CRONOMETRANDO`**, el
   cronómetro está encendido (una línea por imagen en su `.jsonl`).
   Reinicia el servidor sin la variable cuando acabes.
4. **Pasadas A/B del vídeo:** pose cargada antes (hacia las 11:20:30 de
   cámara; el log dice «ViTPose cargado») y luego 10 min desde las
   11:17:00. **Alinea las pasadas por los agarres y visitas del log**
   (desfase fijo entre pasadas), no por la hora que diga el usuario: a
   veces empieza antes o se pasa de los 10 min.
5. **Una grabación de diagnóstico se reproduce igual que en vivo:** la del
   6-oct (`ELDE_ANAQUELES_GRABAR`, 11:06) dio fuera de línea los mismos
   agarres en los mismos segundos (11:07:04, 11:07:18, 11:07:24). Sirve para
   probar reglas del agarre sin pedir pasadas al usuario. Son fotos de
   clientes: fuera del repo, y preguntar antes de grabar y para borrar.
6. **Herramientas de esta sesión** en
   `informes\herramientas_cliente\sesion_2026-10-06\`, explicadas en su
   `LEEME_herramientas_2026-10-06.md`: reproductor de la grabación
   (`reproducir.py`, `robustez.py`), análisis del cronómetro y del ciclo
   (`analizar_tiempos.py`, `ciclo_completo.py`, `comparar_flush.py`…) y
   validación de la pose. Llevan horas del 6-oct escritas: cámbialas.

### 14.1 Commits del 6-oct

Subidos a `fork/feature/anaquel` junto con los 7 del 5-oct (13.1). Uno por
funcionalidad, con la fecha real del cambio en el asunto; cada uno se
comprobó solo, con la suite sobre una copia limpia de ese commit.
`anaqueles.py` (segundos, stock, interruptor del stock, pose en lote y
agarres falsos), el YAML y `test_anaqueles_stock.py` se repartieron por
partes: cada commit lleva solo la suya.

| Commit | Qué |
|---|---|
| `3b9269f` | ajustes en segundos (14.8), cambio del 5-oct |
| `9b93e45` | tarea 2 (sección 9), cambio del 5-oct |
| `d77ace2` | aviso de «posible agotado» (14.7), cambio del 5-oct |
| `6d4f132` | interruptor `stock_activo` (14.7) |
| `644fdec` | pose en la GPU y en lote (14.5) |
| `004ef1b` | agarres falsos en el área de abajo (14.6) |
| `d34a1d7` | cronómetro por etapas (14.4) |
| `9933637` | modo F (14.3) |
| (el último) | este CLAUDE.md |

### 14.2 Rendimiento: de 7,7 a 14,3 imágenes por segundo

Imágenes respondidas por segundo en el cliente, vídeo, una cámara:

| Pasada | Cuándo | Qué llevaba | img/s |
|---|---|---|---:|
| G | 5-oct 09:54 | imagen del detector en la GPU (13.5) | 7,71 |
| T2 | 5-oct 14:38 | + tarea 2 | 9,86 |
| S1 | 6-oct 09:13 | + segundos + stock | 10,31 |
| B | 6-oct 10:04 | + pose en la GPU y en lote | 10,14 |
| antes del F | 6-oct 12:04-12:13 | + agarres falsos + cronómetro | 9,85 |
| **F** | 6-oct 12:45-12:54 | + **modo F** | **14,29** |

- G, T2, S1 y B: mismo tramo de 554 s. «Antes del F» y F: mismo tramo de
  535 s (algo distinto). No compares filas de tramos distintos.
- **En el agarre** (los mismos 30 s, con ~2 personas): 7,0 img/s sin el
  modo F y **11,6** con él.
- **Ciclo con el modo F** (media, ms): 70 de envío a envío; el servidor 65;
  la ida ≈ 0; la vuelta 3,8; el dibujo, 18,8, ya va en paralelo con el
  servidor. **El cuello ahora es el servidor** (14.4).

### 14.3 Modo F: cada imagen sale al socket en el acto

Lo explica, con cómo llevarlo a otro proyecto, la cabecera de
`packages/elde_core/elde_core/transport/socket_client.py`. Resumen:

- **El problema:** `QWebSocket.sendBinaryMessage` deja la imagen en el búfer
  de Qt, que sale cuando el hilo vuelve a su bucle. Con la tarea 2 el
  recuadro envía y JUSTO DESPUÉS pinta (~18 ms): la imagen esperaba a que
  acabara el dibujo. Medido con el cronómetro del servidor y la telemetría
  del cliente (mismo PC, mismo reloj): 27 ms de mediana de «envío» a «el
  servidor empieza»; la única imagen enviada sin pintar detrás, 0,8 ms.
- **El arreglo:** `self.client.flush()` tras cada envío.
- **Resultado, mismo tramo de 535 s** (1,42 frente a 1,43 personas de
  media; el servidor, 65,9 frente a 65,2 ms): ida 31,4 → ≈0 ms; ciclo
  101,5 → 70,0 ms; 9,85 → 14,29 img/s. Enviar cuesta 0,40 → 0,70 ms
  (mediana): `flush()` no espera a la red. 0 errores en 7.643 imágenes y 0
  avisos en el log del servidor.
- **Interruptor:** `ELDE_MODO_F=0` (14.0). El log del cliente
  (`clients/tienda/logs/tienda.log`) dice al arrancar si está encendido.
- **Sin probar:** iVMS en vivo, varias cámaras a la vez y el servidor de
  Linux por el túnel (ahí la subida es más lenta y debería ayudar más:
  deducido).
- `send_frame` (envío por texto/JSON) NO lleva el modo F: nadie lo usa en
  este repo; un cliente del monorepo que lo use necesitaría la misma línea.

### 14.4 Cronómetro por etapas (`tiempos_etapas.py`)

- Con `ELDE_TIEMPOS_ETAPAS=<ruta .jsonl>` el servidor envuelve al arrancar
  (`lifespan` de `app.py`) 21 funciones y escribe una línea por imagen: los
  ms PROPIOS de cada etapa (la de dentro se resta a la de fuera), las
  personas, a cuántas se calculó la pose y el total. Sin la variable no toca
  nada.
- **Fiable:** en cada imagen, la suma de etapas = el total (0,03 ms como
  mucho), y el total coincide con la telemetría `FPS [` en cada tramo de
  30 s (±1,6 ms).
- **Medias por imagen** (pasada de las 12:02, 6.293 imágenes; con varias
  personas, a la pose se le calcula a la que está junto al estante):

  | Personas | Pose a | Imágenes | Total | Detector | JPEG | Escena | Pose | Caras (YuNet) | Edad/género | Cuerpo |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 0 | — | 1.280 | 49,5 | 31,0 | 9,5 | 4,6 | 0 | 0 | 0 | 0 |
  | 1 | — | 2.110 | 59,0 | 31,7 | 9,8 | 4,5 | 0 | 2,0 | 2,5 | 0,8 |
  | 1 | 1 | 490 | 82,7 | 37,5 | 9,9 | 4,4 | 20,2 | 2,2 | 0,2 | 1,2 |
  | 2 | 1 | 640 | 93,7 | 37,5 | 9,9 | 4,5 | 19,9 | 6,2 | 5,4 | 2,1 |
  | 3 | 1 | 164 | 110,8 | 39,6 | 10,3 | 4,8 | 20,6 | 12,6 | 9,4 | 3,5 |

- **Lo fijo de cada imagen:** detector ~31 ms, abrir el JPEG ~9,5, comprobar
  la escena ~4,5. **Con alguien junto al estante:** la pose ~20 ms, y el
  detector ~6 ms más (hallazgo 36). **Las etapas intermitentes** (caras,
  edad/género) suben con la gente y dan los picos: de las imágenes de
  100-200 ms (7,8 %), la etapa más larga es edad/género en 246, el detector
  en 116, las caras en 112 y la pose en 13.
- **Da medias Y medianas:** en la mediana las etapas intermitentes salen 0
  (el usuario pidió las medias, y con razón).

### 14.5 Pose en la GPU y en lote (`pose_gpu.py`)

- **Qué:** los recortes se preparan en la GPU con la MISMA fórmula que el
  preparador de Hugging Face (recuadro ×1,25 a la proporción 192/256,
  bilineal, 0 fuera de la imagen, redondeo a `uint8`, normalización), y
  todas las personas de una imagen van en una sola llamada al modelo
  (`keypoints_lote` en `pose_motor.py`, `munecas_lote` en
  `anaqueles_pose.py`). El posproceso sigue siendo el de Hugging Face.
- **Dos fallos encontrados al validar:** en el borde, `grid_sample` mezclaba
  con negro donde scipy pone negro (35-60 niveles de más en una franja); y
  una fórmula «equivalente» cambiaba la última cifra y movía una fila de
  borde. Se calculan las coordenadas igual que scipy (la matriz float32
  invertida en float64).
- **Validado con 298 fotos de fichas** (solo leídas en memoria, con permiso
  del usuario): entrada al modelo igual (≤ 1 nivel de gris); muñecas a
  ≤ 0,23 px. En lote (4 personas por imagen, 400 fotos): las mismas 592
  muñecas, a ≤ 0,11 px.
- **Banco:** 1 persona 24,9 → 20,1 ms; 2: 50,7 → 23,3; 3: 74,5 → 26,8; 4:
  100,6 → 30,4.
- **En vivo (pasada B):** los mismos 3 agarres y la misma visita (el tercer
  agarre, 4 s antes que en la S1; el producto asignado cambió, como de
  costumbre: hallazgo 25), pero en el agarre el servidor solo baja de 82-88
  a 77-80 ms: casi siempre se
  calcula la pose a UNA persona. Se estimó «7 → 9-10 fps en el agarre» sin
  medir dentro del servidor, y no se cumplió (14.12).
- Sin CUDA, con el interruptor o si falla, vuelve a persona a persona en la
  CPU. Cada cámara sigue cargando su propio ViTPose (hallazgo 33).

### 14.6 Agarres falsos en el área de abajo

- **Síntoma:** en el tramo de la clienta salían agarres en el Anaquel 2
  (abajo). El usuario revisó el vídeo: mete la mano 8 veces, todas en la
  balda alta.
- **Diagnóstico** (`ELDE_ANAQUELES_DIAG` + grabación del 6-oct a las 11:06,
  reproducida fuera de línea), dos causas:
  1. **Misma mano:** la pose vio una mano y le puso las dos etiquetas. La
     «izquierda», que colgaba al costado (y ≈ 800), apareció encima de la
     derecha (a 1-3 px, confianza 0,30-0,63): 443 px de «recorrido» hacia
     el Anaquel 2.
  2. **Bajada:** la derecha volvía de DENTRO del Anaquel 1 y se quedaba a la
     cintura, que con el ángulo de la cámara cae dentro del Anaquel 2. Para
     él, la mano venía «de fuera».
- **Reglas** (`anaqueles.py`): dos muñecas a menos de max(8 px, 5 % del
  ancho de la persona) son una sola mano y se queda la de más confianza
  (`_una_mano_si_coinciden`). Si el origen del gesto estaba DENTRO de un
  área que queda encima, se descarta (`agarre_bajada`). Subir no cambia:
  los agarres buenos de arriba salen de la cintura, que está en el área de
  abajo. «Encima» = centro vertical de la caja del área más arriba en la
  imagen; con una cámara cenital eso podría no ser la balda alta [sin
  comprobar].
- **Validado con la grabación:**

  | Variante | Anaquel 1 | Anaquel 2 (falsos) |
  |---|---:|---:|
  | Antes | 2 | 1 |
  | Con las dos reglas | 2 | 0 |
  | Con una sola de las dos | 2 | 1 |
  | Áreas movidas 8 px arriba o abajo | 2 | 0 |

  **En vivo** (11:44): 2 agarres en el Anaquel 1 y 0 en el 2.
- **Coste:** si alguien coge de arriba y luego de abajo sin bajar antes la
  mano, el de abajo no cuenta hasta que la mano salga de las dos áreas.
- **Descartado:** una ventana de 1 s tras salir de arriba (la propuso el
  usuario): la mano tardaba ~3,4 s en asentarse abajo y no cambió nada.
- `test_V3_transito_apilado_cuenta_solo_donde_se_asienta` tenía la balda
  «alta» dibujada más abajo que la «baja», que con la regla nueva es una
  bajada: se corrigió la geometría; comprueba lo mismo.

### 14.7 Aviso de «posible agotado» (`anaqueles_stock.py`)

- **Origen:** se lo pidieron al usuario a última hora; se escribió el 5-oct
  por la tarde, fuera de esta sesión, y quedó sin commitear. Se commiteó el
  6-oct tal cual, con su interruptor aparte.
- **Qué hace:** con un botón del cliente se toma la foto «surtido» de cada
  producto (POST `/api/anaqueles/<cámara>/stock/referencia`; se toma cuando
  nadie lo tapa). Tras un agarre se vigilan TODOS los productos del área
  (la muñeca no es la mano), comparando el recuadro con su foto cuando
  nadie lo tapa. Tres medidas seguidas por encima del umbral dan la alerta
  (panel de Anaqueles y columna «Stock» del panel de alertas, con la foto
  antes y después), y se cierra sola si vuelve a parecerse a la foto.
  Evidencias en `output/anaqueles/<cámara>/stock_alertas/`: fotos de
  clientes, fuera de git.
- **El umbral (0,45) NO está calibrado:** no hay ni una balda vaciada de
  verdad. En el vídeo saltó «POSIBLE AGOTADO coffemate» (09:17:14 y
  10:22:06) y «repuesto» a las 10:21:48, probablemente al volver a poner el
  vídeo desde antes del agarre [deducido]; sin comprobar si el hueco es
  real.
- **Coste medido:** 0,1 ms por imagen sin nada que vigilar, 0,4 vigilando 8
  productos y 2 ms forzado a comparar los 8 en cada imagen. **No era lo que
  bajaba los fps** el 5-oct: minuto a minuto, la ida y vuelta subía con la
  gente en pantalla (~80 ms con 1 persona, 120-160 con 2-3), también antes
  de existir el stock. El cronómetro (14.4) lo repartió después: pose y
  detector con alguien junto al estante, y los picos de edad/género y caras.

### 14.8 Ajustes contados en imágenes, ahora en segundos (`ritmo_imagenes.py`)

- **Por qué:** con más fps, un umbral en imágenes dura menos tiempo (el
  usuario lo preguntó el 5-oct: ROI, fichas, género, track único).
- **Cómo:** cada umbral exige lo MAYOR entre sus imágenes de siempre y sus
  segundos a los fps medidos de la cámara (`imagenes_para`). Con
  `FPS_REFERENCIA = 7,7` (la pasada G), a ~5-8 fps todo queda igual que
  antes; solo se endurece al subir los fps. Cubre el agarre
  (`agarre_min_s`, `agarre_rearme_s`), ByteTrack (`BYTETRACK_LOST_S`,
  `BYTETRACK_MIN_CONSECUTIVE_S`), la histéresis y salidas del ROI, el
  anti-duplicado, la cadencia de clasificación, la re-identificación por
  cuerpo y las fichas.
- **Resultado:** con más fps las entradas al ROI no se inflan (31 a 9,85
  fps → 24 a 14,29, mismo tramo). Pero las entradas repetidas siguen
  (hallazgo 35): la idea del 5-oct de que las causaban los umbrales en
  imágenes no se sostiene (G 11 entradas, T2 29, S1 31).

### 14.9 Varias cámaras (revisado leyendo el código el 6-oct)

- Cada recuadro recibe TODAS las respuestas (`render_box.py:377`) y descarta
  las ajenas ANTES del envío de la tarea 2 (`:1868-1877` frente a `:1893`).
  Si no fuera así, B enviaría con una imagen pendiente, el servidor tiraría
  la vieja (`ws_drenaje.py:102-119`) y B esperaría para siempre (hallazgo
  30). Lo fija `test_la_respuesta_de_otra_camara_no_hace_enviar`, con dos
  recuadros reales, verificada al revés.
- Las respuestas de error al procesar llevan su destinatario: el servidor
  reutiliza el sobre del mensaje (`ws_drenaje.py:208-211`, `:238-239`).
- El modo F no mezcla cámaras: solo adelanta lo ya encolado.
- **No hubo ninguna pasada con varias cámaras.**

### 14.10 Probado y descartado

- **Abrir el JPEG en la GPU** (nvJPEG): 8,1 → 5,8 ms contando bajar la
  imagen a la CPU (el resto del sistema la necesita ahí), y la imagen sale
  distinta (88 % de los píxeles, 3,9 niveles de media, hasta 36). No
  compensa.
- **Agrupar la pose sin prepararla en la GPU:** con 1 persona, peor (39
  frente a 26 ms).
- **La ventana de 1 s para la bajada** (14.6).

### 14.11 Pendientes, por orden

1. **Probar el modo F y la tarea 2 con iVMS en vivo**, con **varias
   cámaras**, y con el servidor de Linux por el túnel.
2. **Decisión del usuario: hallazgo 37** (contar cada vez que mete la mano
   o una vez por visita).
3. **Hallazgo 36:** los ~6 ms de más del detector con pose.
4. **Edad/género y caras:** sus picos de 30-80 ms (14.4). Ver por qué se
   juntan en las mismas imágenes y si se pueden repartir.
5. **Hallazgo 35:** las entradas repetidas al ROI.
6. **Hallazgo 30:** un tiempo máximo de espera en la ruta de ventana.
7. **Stock:** calibrar el umbral con una balda vaciada de verdad.
8. **Mano con codo → muñeca** (hallazgo 25), o redibujar los recuadros de
   «coffemate» y «Nescafe».
9. **Compartir ViTPose entre cámaras** (hallazgo 33).
10. **Telemetría del cliente:** ampliar la rotación.
11. Lo que sigue abierto de 13.10: fotos de las fichas pequeñas, modo de
    copia y RTSP de iVMS, captura bajo pedido, y la lista del punto 9.

### 14.12 Errores de esta sesión (no los repitas)

1. **Estimar sin medir dentro del servidor:** «la pose es lo que frena el
   agarre, 7 → 9-10 fps» (casi siempre se calcula a una persona, la ganancia
   fue de 6-8 ms) y «abrir el JPEG en la GPU ahorra 6-8 ms» (fueron 2,3 ms
   y otra imagen). Mide antes de proponer.
2. **Dar solo medianas** de etapas que trabajan en unas imágenes sí y en
   otras no: salen 0 y esconden los picos.
3. **Afirmar sin leer el código** que un error del servidor «sin cámara»
   desbloqueaba a todos los recuadros: nunca les llega.
4. **Diseñar una regla antes de mirar la trayectoria:** la ventana de 1 s
   para la bajada no hacía nada; el diagnóstico imagen a imagen enseñó la
   mano duplicada.
5. **Calcular de memoria el tamaño de una grabación:** dije ~150 MB y
   fueron 601 MB (1.200 imágenes a calidad 95).
6. **Leer y reescribir un fichero UTF-8 con `Get-Content`/`Set-Content`**
   de PowerShell 5.1: rompió los acentos. Usa las herramientas de edición
   o `-Encoding utf8` también al leer.

---

## 15. Sesión del CLIENTE en Windows (7, 8 y 9 oct)

Escrito el 9-oct por la sesión del cliente, para la siguiente. Medido en el
PC `DESKTOP-9LHHJI2` con **iVMS en vivo (cámara 3)** salvo lo marcado como
**deducido**. Horas locales (Venezuela). Informes del usuario de estos días
(local, fuera de git): `informe_2026-10-07.md`, `_07_dia.md`,
`_08_noche.md`, `_08_dia.md` e `informe_2026-10-09.md`.

### 15.0 Para empezar: TODO esto está SIN COMMITEAR

`git status` el 9-oct: el código de la sección 14 que no se subió (grafos
de CUDA, IDs de seguimiento, perfil de la cámara 3) más todo lo de esta
sección. **Nada commiteado ni subido.** Al commitear, uno por
funcionalidad, enseñando antes archivos y mensajes (sección 0):

| Funcionalidad | Archivos |
|---|---|
| Grafos de CUDA | `grafo_cuda.py`, `estimador_edad_genero.py`, `pose_motor.py`, `test_grafo_cuda.py` |
| IDs de seguimiento en el recuadro | `etiqueta_seguimiento.py`, `render_box.py` (parte), `test_etiqueta_seguimiento.py`, `test_envio_antes_de_pintar.py` |
| Perfil de la cámara 3 (filtro 220 px) | `server/config/camera_profiles/win-iVMS-4200-c3.json`, `test_filtro_fondo.py` |
| Red de seguridad del recuadro (15.1) | `render_box_captura.py`, `test_respuesta_perdida.py` |
| Clasificador B oficial (15.4) | `clasificador_b.py`, `tiendas.py`, `config.py`, `test_clasificador_b.py` |
| Clasificación en sombra (diagnóstico) | `clasificacion_sombra.py`, `app.py` (parte) |
| Stock en el dashboard (15.7) | `dashboard_stock.py`, `dashboard.py`, `app.py` (parte), `test_dashboard_stock.py` |
| Stock: fotos por producto, diaria, bajadas (15.7) | `anaqueles_stock.py`, `anaqueles.py`, `anaqueles_api.py`, `render_box.py` (parte), `test_anaqueles_stock_bajadas.py`, `test_anaqueles_stock.py` |

**Pruebas el 9-oct:** servidor `964 passed, 15 skipped, 1 xfailed`;
`clients/tienda/test` 31 passed; las de `elde_core` tocadas, en verde.

**Estado al cerrar (9-oct):**
- servidor arrancado a las 12:14 en su modo normal: clasificador B, sin
  sombra, sin cronómetro y sin grabador. Script de reinicio:
  `reiniciar_oficial.ps1` del scratchpad. El servidor viejo a veces no se
  apaga: mirar procesos tras reiniciar;
- **desde las 09:23 el usuario tiene la IA apagada** a propósito: nada de
  lo de hoy se ha visto aún con gente;
- **el programa de la tienda no se ha reiniciado** desde el 8-oct: no lleva
  todavía la red de seguridad (15.1) ni el menú de fotos por producto.

**Interruptores nuevos:**

| Interruptor | Qué hace | Por defecto |
|---|---|---|
| `ELDE_CLASIFICADOR=clasico` (o `DEMO_CLASIFICADOR` en `config.py`) | vuelve al clasificador anterior | `b` |
| `ELDE_CLASIFICACION_SOMBRA=<ruta .jsonl>` | clasifica también con B a la sombra y lo apunta aparte | apagado |
| `stock_umbral` / `stock_umbral_bajada` / `stock_bajadas_aviso` / `stock_reinicio_diario` (`_DEFECTOS` de `anaqueles.py`) | stock (15.7) | 0,90 / 0,45 / 8 / `07:30` |

**Material en el scratchpad** (fotos de clientes: borrar al terminar, con
permiso; el scratchpad es de la sesión y puede no seguir ahí):
`banco_clasificacion\` (banco, variantes, grabaciones del 7-oct y del 8-oct
18:00-20:05, ~1,5 GB cada una, y las etiquetas a mano `etiquetas.csv`, 201,
y `etiquetas_8oct.csv`, 283), `grabacion_agarre\` (601 MB),
`tiempos_etapas.jsonl` (619 MB), `sombra_8oct.jsonl`,
`telemetria_noche_8oct\`, `imagenes_fijas\`.

### 15.1 El corte del 8-oct 05:45: hallazgo 30 en producción

- **3 h 27 min sin datos** (05:45:20 → 09:12), incluida la apertura.
- El servidor procesó bien su última imagen. Con py-spy: **ningún hilo
  atascado**; el servidor esperaba imágenes y el recuadro esperaba la
  respuesta de la última: **se perdió un mensaje** y nadie dejó error. No
  se sabe qué lado lo perdió.
- **Arreglo (sin commitear):** `ESPERA_RESPUESTA_MAX_S = 5.0` en
  `render_box_captura.py`: a los 5 s sin respuesta la ruta de ventana da la
  imagen por perdida, lo apunta en `tienda.log` y vuelve a enviar. 4
  pruebas, verificadas al revés.
- La telemetría del cliente solo escribe el ciclo al terminarlo: su «todo
  ok» final no descarta una imagen en vuelo sin respuesta.

### 15.2 iVMS: la imagen fija y las imágenes repetidas

- **8-oct 11:42-11:58** iVMS dejó de dar vídeo: llegaba siempre la misma
  imagen (~42 KB). **Es la misma de las madrugadas.** Hipótesis sin
  comprobar: «la noche oscura» de los informes sería iVMS sin vídeo. Copia
  en `imagenes_fijas\` del scratchpad para que el usuario la mire.
- **El 41-64 % de las imágenes enviadas son copias exactas de la anterior**
  (tarde del 8-oct): iVMS refresca a ~9 img/s y el cliente copia a ~18. El
  servidor procesa la mitad dos veces. Pendiente: no enviar repetidas.

### 15.3 El PC sin memoria (8-oct ~10:20)

Quedaban 0,7 GB: servidor 5 GB, programa de la tienda 1,85 GB y su
capturador 545 MB, los dos creciendo desde el corte [deducido: por el
recuadro atascado]. El usuario reinició el PC y liberó disco (39,5 GB
libres). Vigilar la memoria del cliente.

### 15.4 Clasificación: banco, etiquetas y el clasificador B

**Por qué casi todo era «muestras insuficientes»** (leído en el código):
exige 5 lecturas con 90 % de acuerdo; busca la cara hasta 3 veces por
intento; InsightFace tira la lectura si no coinciden sus variantes; el
cuerpo no se intenta fuera de proporción 1,2-4,0; el jurado va por número
de seguimiento y se tira a los 0,65 s sin ver a la persona.

**Banco aislado** (scratchpad, fuera del repo): un `sitecustomize.py` graba
recortes; `banco.py` los pasa por el clasificador y sus variantes; el
usuario etiquetó a mano. Grabación del 7-oct, 201 etiquetas:

| Variante | Decididas | Acierto |
|---|---:|---:|
| Clásico | 41 % | 88,5 % |
| **B sin herencia** | **80 %** | **90,6 %** |

**En vivo, en sombra** (8-oct 18:00-20:05, 251 etiquetas): clásico 27 % y
87 %; **B 63 % y 94,9 %**; mujeres bien clasificadas 15 frente a 62. **8 de
los 9 errores del clásico eran géneros heredados** del reconocimiento de
caras.

**B** (`clasificador_b.py`): una búsqueda de cara por intento; vota solo
MiVOLO (cara sin alinear + cuerpo); cuerpo de proporción 1,0-5,0; decide por
evidencia (log-odds, umbral ln 19, descuento 0,5, tope 3, mínimo 2
lecturas, máx. 25 % de desacuerdo); **no hereda el género** del Re-ID (que
sigue sirviendo para la identidad); sin histéresis; cadencia
`DEMO_RECLASSIFY_EVERY_B = 5`. **Oficial desde el 9-oct 10:34**, sin
probar aún con gente.

**Descartado al medir:** no tirar las caras dudosas del TTA, jurado de 9 s,
coser trozos de seguimiento.

### 15.5 Por qué al anochecer se clasificaba la mitad

No faltaban lecturas: hubo más. Los jurados mezclaban **cara (InsightFace)
y cuerpo (MiVOLO)**, que se contradicen, y la regla del 90 % bloqueaba. El
usuario descartó la luz. Con B vota un solo modelo.

### 15.6 Seguimiento (ByteTrack)

- **Probado y quitado:** `BYTETRACK_MATCH_THRESH` 0,9 (7-oct): en vivo dos
  personas se intercambiaron el número.
- **Con el ajuste normal, el número cambia de persona en el 11-15 %** de
  los etiquetados (30 de 201, 32 de 283). **Siguiente problema grande.**
  Idea: cortar el jurado cuando la huella del cuerpo cambie de golpe.

### 15.7 Stock

- **Dashboard** (`dashboard_stock.py`): «Stock — posible agotado» con las
  alertas activas (contador «lleva X»), el historial con fotos y las
  bajadas. **Las alertas activas viven en memoria: un reinicio las borra.**
- **Tres formas de foto de referencia:** manual de todos o de **un
  producto** (`?area=&producto=`, submenú del cliente); **diaria a las
  7:30** (apuntada en `stock_ref/ultima_diaria.txt`; la del 9-oct se dio
  por hecha); y **rehecha tras cada bajada** confirmada.
- **Bajada** = cambio ≥ 0,45 tres veces seguidas tras un agarre; se cuenta y
  cada 8 avisa «Cerca de vaciarse» (`..._cerca.jpg`). **Posible agotado** ≥
  0,90 (antes 0,45). Umbrales elegidos por el usuario viendo las fotos;
  **sin calibrar**.
- **7 alarmas en vivo el 7-8 oct** (cambio 0,46-0,84: con 0,90 ninguna
  habría saltado). **Ninguna se cerró por reposición**: todas las borró un
  reinicio.
- Las pruebas de stock usan umbrales propios.

### 15.8 Otros datos

- Filtro del fondo de la cámara 3: **220 px** desde el 9-oct (antes 200).
- Agarres del 8-oct tarde: 4 de 3 personas; el clásico no decidió a
  ninguna; la sombra, a 1 (coincide con la etiqueta).
- Desde el filtro del fondo: identificadas 30-35 %, clasificadas 21-26 %
  (clásico).

### 15.9 Errores de esta sesión (no los repitas)

1. **Hipótesis dadas como causa** («detecciones débiles», «la luz»): medir
   antes.
2. **Un vigilante que copiaba un archivo inexistente** (el mapa de calor se
   pinta bajo pedido; el fondo está en `heatmap/bg/`).
3. **Rutas de Windows con `sed` o escapes** (`\a`, `\7`) rompieron scripts:
   usar la herramienta de escritura.
4. **No comprobar que el servidor viejo se había apagado.**

### 15.10 Pendientes, por orden

1. **Commitear** lo de 15.0 (con permiso, uno por funcionalidad).
2. **Ver B y el stock nuevo con gente**; reiniciar el programa de la tienda.
3. **Cambios de persona en el seguimiento** (15.6).
4. **No enviar imágenes repetidas** (15.2).
5. **Guardar las alertas de stock en disco**; calibrar umbrales.
6. **Mirar la imagen fija de la noche** con el usuario.
7. Borrar, con permiso, grabaciones y cronómetro (15.0).
8. Lo abierto de 14.11.
