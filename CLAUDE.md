# CLAUDE.md — ELDE Tiendas (contexto de trabajo)

Traspaso de las sesiones del 16-18 y 21-22 sep 2026. **Este archivo SÍ se
commitea** (antes era local): vive en la rama para que la siguiente sesión lo
lea sin que el usuario tenga que pegarlo.

Rama de trabajo actual: **`claude/pensive-galileo-aymjr3`**, que cuelga de
`fix/eliminar_vigilantes_amazonas` (no de `main`).

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

No viene nada instalado. Esto bastó el 21-sep para correr la suite completa
(el índice de PyTorch está bloqueado por el proxy; **torch hay que sacarlo de
PyPI**):

```bash
pip install numpy pytest opencv-contrib-python-headless fastapi msgpack \
            supervision torch httpx onnxruntime rfdetr psutil py-cpuinfo \
            python-multipart
cd server && python3 -m pytest tests -q
```

**Línea base medida el 22-sep: `767 passed, 15 skipped, 4 failed`.** Los 4
fallos son **del contenedor, no del código**, y salen igual antes y después de
cualquier cambio:

- `test_informe_pdf.py` (3) → `ModuleNotFoundError: No module named 'reportlab'`
- `test_pose_motor.py` (1) → `AssertionError: el modelo local debió cargar sin
  red` (faltan los pesos de Git LFS)

Compáralos siempre contra esa línea base antes de dar un cambio por bueno.
`conftest._NO_APLICAN` salta 5 pruebas de otros verticales; las otras 10 del
total de 15 vienen de otros `skip` del repositorio.

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

### El caso de uso real (contexto que faltaba en el doc anterior)

**Un pasillo de anaqueles de supermercado**, con **dos cámaras del mismo
pasillo funcionando a la vez** (`win-iVMS-4200-i58bf7b` y
`win-iVMS-4200-c2-i58bf7b`). Pasa mucha gente. Eso importa para las
decisiones de diseño: la gente **mira los productos, no la cámara**, y las
caras del fondo del pasillo miden ~20 px.

Hay además una tercera «cámara» que es **un vídeo de YouTube de prueba**
(`win-HD_CCTV_Camera_video_3MP_4MP_iProx_CCTV_HDCCTVCameras.net_retail_store-i58bf7b`).
No la confundas con una cámara real: su imagen está doblemente comprimida y
**rechaza prácticamente todas las caras**. Medido el 21-sep: en ese vídeo los
rechazos por `calidad` (nitidez) eran el **26,8 %**; en la cámara real, el
**3,1 %**. Si el usuario pega un log, pregunta de qué fuente es antes de
concluir nada.

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
- **Los `git push` sí los haces tú ahora.** Si tienes que reescribir un commit
  ya publicado (`--amend`, rebase), **avísale**: él tendrá que hacer
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

Cuatro de la sesión del 21-22 sep, todos detectados y corregidos:

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

## Lo que se aprendió midiendo (21-22 sep)

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

### Efecto medido del perfil (22-sep, cámara real, muestras pequeñas)

| Motivo | Antes | Después |
|---|---:|---:|
| `bbox_pequeno` | 30,9 % | **14,9 %** ↓ |
| `sin_cara` | 10,5 % | 19,3 % ↑ |
| `body_conf_baja` | 0,5 % | 10,5 % ↑ |
| `ensemble_tta` | 22,5 % | 23,7 % (igual) |
| **aceptación** | 14,7 % | 9,6 % |

**El perfil hace lo que promete** (el rechazo por bbox cae a la mitad), pero
**el embudo se desplaza**: la gente que antes se cortaba al principio ahora
llega más lejos y cae igual, porque no tiene cara utilizable. Las muestras son
de minutos y con tráfico distinto: **indicativo, no concluyente**.

### Identidad entre cámaras: CONFIRMADA en producción

```
2026-09-22   vista en 2 camara(s): 1 personas
```

Una persona real reconocida por las dos cámaras del pasillo con el mismo
`persistent_id`. Consulta para reproducirlo:

```sql
SELECT jornada, camaras, COUNT(*) FROM (
  SELECT jornada, persistent_id, COUNT(DISTINCT camera_id) AS camaras
  FROM tienda_visitas WHERE camera_id LIKE 'win-iVMS%'
  GROUP BY jornada, persistent_id) GROUP BY jornada, camaras;
```

**Límite estructural del caso de uso:** para *registrar* a alguien hace falta
una cara semi-frontal (`REID_MAX_YAW_FOR_REGISTRATION = 20°`). En un pasillo
de anaqueles la gente mira los productos. Lo que movería la aguja de verdad es
**una cámara a la altura de la cara en la entrada del pasillo** que haga el
registro, dejando a las demás solo re-identificar (mucho más tolerante).

---

## Hallazgos pendientes (18-sep, revisados el 22-sep)

Antes de citar uno, **comprueba que el código sigue igual**. Los que llevan
✅ se resolvieron; los demás siguen vivos con sus líneas actualizadas.

1. **Persona en dos cámaras:** visitantes únicos 1 (bien), entradas 2, género y
   edad globales 2 (mal: `tienda_almacen.demografia`, `tienda_almacen.py:568`,
   suma por cámara). **Ahora hay un caso real en la base** (22-sep) para
   medirlo. Ningún test lo cubre.
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

13. **`dashboard.py:244`:** sin clientes conectados, la rama de disco hace
    `visitantes = sum(...)` sobre los informes por cámara en vez de unir
    UUIDs → **duplica a quien cruza dos cámaras**. La rama en vivo sí lo hace
    bien (`:228` `unicos |= ...`, `:238` `len(unicos)`). Con las dos cámaras
    del pasillo, este bug ya tiene un caso real que lo dispara.
14. **`_best_in_gallery` es O(N) en Python puro** y `_cosine_similarity`
    renormaliza vectores que ya vienen normalizados. Un producto matricial da
    **250-300× más rápido**. Con el TTL ya no urge.
15. **`ERROR` rojo en cada arranque:** `estacionamiento_planificador` no existe
    en este corte y `app.py:150` lo saca con `logger.exception` (traceback
    completo), cuando `vigilante_amazonas` y `perimetral_nocturna` salen como
    un `WARNING` de una línea. Cosmético, pero asusta.
16. **`REID TTL: caducaron N` es `logger.debug`**
    (`face_reidentifier.py:449`): no se ve con nivel INFO.
17. **`GET /vlm/model` ×16 en un segundo** desde el cliente. Polling agresivo.
18. **Modelo PAR ausente:** `models/classifiers/par_gender.onnx` no está, y
    `body_sin_modelo` fue el **6-11 %** de los rechazos. `scripts/setup_modelos.py`
    lo genera (verificado que el script existe; **no se ha ejecutado**).

---

## Pendientes, por prioridad

1. **Hallazgo 13** (`dashboard.py` suma en vez de unir). Es un bug de conteo
   real y el caso que lo dispara ya está en la base del usuario.
2. **Generar el modelo PAR** (hallazgo 18). Lo hace el usuario en su Linux;
   ataca directamente el cuello de botella del pasillo.
3. **Hallazgos 15 y 16** (el `ERROR` del arranque y el log del TTL a INFO).
   Cinco minutos cada uno, ya ofrecidos al usuario.
4. **Decisiones suyas, sin prisa:**
   - **La semántica del número.** Con TTL se cuentan **recorridos**, no
     personas del día: quien vuelve pasados 30 min es un `persistent_id`
     nuevo. El dashboard sigue diciendo «visitantes únicos». Conviene
     renombrar la columna.
   - **El panel de personas** ahora muestra la ventana viva de 30 min, no un
     histórico. Está pendiente de que lo vea y lo apruebe.
   - **El destino de la rama:** ¿la integra él a `fix/...` o se abre un PR?
5. **Validar el TTL con los 30 minutos reales.** Solo se comprobó con
   `ELDE_REID_TTL_S=60`.
6. **Aplazados por el usuario el 18-sep** (no los empieces sin que lo pida):
   vídeo que no se congela si la IA falla (hallazgo 7), y qué contar
   exactamente entre cámaras (hallazgo 1).
7. **Escáner de ventanas**, propuesto y pendiente de aprobación. Toca
   `packages/elde_core` (compartido): va en rama propia.

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
- **Los avisos de `mivolo.onnx` y `par_gender.onnx` «no presente»** son de
  modelos opcionales. MiVOLO v2 (PyTorch) sí carga y la rama corporal queda
  activa; el aviso del `.onnx` es de otra ruta.
