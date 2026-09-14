# Guía de trabajo — ELDE Tiendas

Cómo funciona el sistema, cómo se conectan sus piezas y cómo se trabaja en
este repositorio. Para montar el entorno y arrancar, mira antes el
[README](README.md).

---

## 1. Qué hace, en un párrafo

Una cámara de la tienda graba. El **cliente de escritorio** se conecta al DVR,
saca los fotogramas y se los manda al **servidor** por websocket. El servidor
los procesa en la GPU —detecta personas, les estima género y rango de edad,
las re-identifica para no contarlas dos veces, acumula el mapa de calor y
aplica las reglas del modo activo— y devuelve al cliente el fotograma anotado
y un bloque de metadata. Esa misma metadata alimenta los dashboards web.

El cliente **no decide nada de analítica**: dibuja, manda y pinta lo que le
devuelven. Toda la inteligencia vive en el servidor. Tenlo presente antes de
mover lógica de sitio.

---

## 2. Cómo se conecta todo

```
   ┌─────────┐   RTSP/ISAPI/SDK   ┌──────────────┐   ws://…:9000/ws/{modo}   ┌──────────┐
   │   DVR   │ ─────────────────► │   CLIENTE    │ ────────────────────────► │ SERVIDOR │
   │ cámaras │                    │ clients/     │ ◄──────────────────────── │ server/  │
   └─────────┘                    │   tienda     │   fotograma + metadata    └────┬─────┘
                                  └──────────────┘                                │
                                         ▲                                        │ HTTP
                                         │      baliza UDP 9010                    ▼
                                         └──────── «¿quién es servidor?» ──► http://…:9000
                                                                              /dashboards/
                                                                              /api/v1
                                                                              /health
```

### 2.1 Del DVR al cliente

El cliente habla con **Hikvision** (HTTP/ISAPI y SDK), **Dahua** (HTTP y SDK) y
**Hik-Connect** (la nube). Cada uno es un módulo en
[`clients/tienda/src/core/dvr/`](clients/tienda/src/core/dvr/) sobre una base
común (`base.py`), así que añadir una marca es escribir un módulo, no tocar los
demás.

Un detalle que vas a encontrarte: **una conexión al DVR puede traer 16 cámaras**.
El cliente abre un recuadro por canal, pero el servidor recibe todos esos
fotogramas por la **misma** conexión websocket. Por eso todo lo que guarde
estado en el servidor se llavea por `camera_id` y nunca por el nombre de la
ventana (todas se llaman igual, «iVMS-4200»). Si mezclas los estados, el
síntoma no es un error: es que «no detecta».

### 2.2 Del cliente al servidor

El cliente **no lleva la IP escrita**. La busca:

1. Responde la **baliza UDP del 9010**: el servidor dice quién es, en qué
   puerto escucha, cuántos clientes tiene y qué GPU usa.
2. Si no hay baliza, escanea el puerto 9000 de la subred (más lento).
3. Y siempre manda lo más explícito: `--servidor <host>` en la línea de
   órdenes, luego lo que fijó el selector, luego **lo que el operador eligió la
   última vez**, luego el `.env`, y si nada de eso responde, pregunta.

Ese orden vive en
[`packages/elde_core/elde_core/red/arranque_servidor.py`](packages/elde_core/elde_core/red/arranque_servidor.py)
y está comentado ahí. **Ningún cliente autoelige un servidor descubierto**: la
baliza solo llena la lista.

### 2.3 El `.env` del cliente

No se versiona. Va en `clients/tienda/.env`:

| clave | |
|---|---|
| `server_ws_url` | **obligatoria**. `ws://192.168.1.50:9000/ws`. Si falta, el cliente aborta con un mensaje claro en vez de quedarse colgado |
| `site_id` | qué local es. Sin ella vale `sitio-unico` y los dashboards no pueden comparar sucursales |
| `jarvis_url` / `jarvis_email` / `jarvis_password` | opcionales, para enviar novedades a Jarvis. Si faltan, el interruptor del pie se apaga solo y el tooltip dice cuál falta |

La URL HTTP **se deriva** de la del websocket (`server_http_url`), a propósito:
cuando eran dos ajustes, acababan apuntando a servidores distintos.

### 2.4 El mensaje

El endpoint es **`/ws/{type_inference}`**, donde `type_inference` es el modo.
Los tres que ofrece este cliente:

| modo | qué hace |
|---|---|
| `Tiendas` | el de siempre: personas, género, edad, conteo único, mapa de calor |
| `Anaqueles` | visitas y agarres por nivel del estante, con su heatmap por área |
| `PermanenciaArea` | cronómetro por persona dentro del ROI, con alertas |

Son **el mismo string** en el cliente y en el servidor
([`custom_status_bar.py`](clients/tienda/src/gui/components/custom_status_bar.py)
↔ `_build_processor`). El servidor acepta además los nombres antiguos
(`Personal de Amazonas`) para no invalidar instalaciones viejas.

El sobre del mensaje está definido una sola vez, en
[`packages/elde_core/elde_core/contracts/envelope.py`](packages/elde_core/elde_core/contracts/envelope.py),
y lo importan **los dos lados**: `client_type` (qué aplicación habla),
`pipeline` (qué modo pide), `device_id` (qué cámara, estable entre reinicios),
`event_version` y `timestamp_utc`. El servidor valida contra ese contrato y
lleva la cuenta en `/health` → `contrato`.

Los fotogramas viajan en binario (msgpack). Si el cliente empuja más rápido de
lo que el servidor drena, hay contrapresión: eso lo gestiona
`ws_drenaje.py`, y el keepalive del websocket está subido a 60 s justo por eso
—con los 20 s de fábrica se cortaban conexiones **vivas** cada minuto—.

### 2.5 Puertos

| | |
|---|---|
| **9000/TCP** | websocket de inferencia, dashboards, `/api/v1`, `/health` |
| **9010/UDP** | baliza de descubrimiento |

---

## 3. Cómo se usa

1. **Arranca el servidor**: `INICIAR_SERVIDOR.bat`. La primera vez tarda ~1
   min en cargar modelos. Comprueba `http://localhost:9000/health`.
2. **Arranca el cliente**: `INICIAR_TIENDA.bat`. Busca el servidor solo.
3. **Conecta el DVR** desde el panel de dispositivos y abre los canales que
   quieras como recuadros.
4. **Dibuja las zonas en el recuadro.** Aquí va una regla del proyecto que no
   es negociable: **las zonas las dibuja el cliente, no un YAML**. El área, la
   línea de cruce, la flecha de dirección y los niveles del anaquel se marcan
   con el ratón sobre el vídeo y viajan en el mensaje. Si te ves tentado de
   añadir una zona a un fichero de configuración, es señal de que vas por el
   camino equivocado.
5. **Mira los resultados** en `http://localhost:9000/dashboards/tienda/`.

El `.bat` de este repositorio trae, a propósito, el envío de WhatsApp apuntando
a un puerto local cerrado: **una máquina de desarrollo no debe mandar alertas a
los grupos de verdad**. Si necesitas probar el envío, cámbialo sabiendo a dónde
va.

---

## 4. Metodología de trabajo

### 4.1 Un archivo por funcionalidad

Cada modo o característica nueva vive en **su propio módulo**, con el cableado
mínimo en los que ya existen. Mira `core/`: `anaqueles.py`, `anaqueles_pose.py`,
`anaqueles_registro.py`, `permanencia_area.py`, `permanencia_cruce.py`,
`permanencia_linea.py`… No es casualidad, es la norma. Un fichero que crece
hasta hacer tres cosas se parte.

### 4.2 Los comentarios explican el porqué

Abre cualquier fichero del repositorio y verás comentarios largos con fechas,
medidas y el síntoma que provocó el cambio. Eso **es el estándar**, no ruido:

> «Reproducido sintéticamente: closeCode=1011 a los 51 s con 427 fotogramas
> enviados y 3 respuestas.»

Un comentario que dice *qué* hace la línea siguiente sobra. Uno que dice *por
qué está así y qué pasó cuando no lo estaba* vale oro, porque evita que alguien
lo «simplifique» dentro de seis meses. Cuando arregles algo, escribe el porqué.

### 4.3 Las pruebas son el contrato

```powershell
cd server
venv\Scripts\python.exe -m pytest tests -q
```

**733 pasan, 15 se saltan.** Que sigan pasando es la condición para dar un
cambio por bueno. Muchas fijan comportamientos que costaron caro de encontrar
—mira las fechas—, así que una prueba que falla no es «una prueba vieja»: es
un aviso.

Si arreglas un fallo, **añade la prueba que lo habría cazado**. Si cambias un
comportamiento a propósito, cambia la prueba y explica por qué en el commit.

### 4.4 Ojo con la clase base

`TiendasProcess` (en `tiendas.py`) es la **base** de `Anaqueles`,
`PermanenciaArea` y `AmazonasStand`. Un cambio ahí toca los cuatro modos. Antes
de darlo por bueno, corre las pruebas de los cuatro, no solo las de tienda.

Y dos cosas de esa clase que ya han mordido:

- `get_camera_processor(camera_id)` da un procesador **por cámara** con estado
  aislado, pero los modelos pesados se comparten. Si añades estado, va ahí
  dentro, llaveado por cámara.
- El estado por cámara se intercambia alrededor de `process_frame`. Quien
  trabaje **después** de esa llamada se encuentra los `active_tracks` vacíos.

### 4.5 Ramas y entrega

Una rama por tarea, y los cambios se entregan por **Pull Request**:

```bash
git checkout -b arreglo/lo-que-sea
git commit
git push -u origin arreglo/lo-que-sea
```

Los commits se escriben **en español**, con un asunto corto en imperativo y un
cuerpo que cuente el porqué y cómo se comprobó. Mira `git log` para ver el tono.

### 4.6 Git: cuatro cosas que muerden desde Windows

1. **Git LFS**: `git lfs install` **antes** de clonar. Instalar el binario no
   registra los filtros, y sin ellos los pesos llegan como ficheros de 130
   bytes con el nombre correcto — todo parece bien hasta que nada detecta.
2. **Bit ejecutable**: NTFS no lo tiene, así que un `.sh` creado en Windows
   llega al repositorio sin él y en Linux responde `Permission denied`.
3. **Finales de línea**: los fija `.gitattributes` (`LF` para todo lo que
   ejecuta Linux, `CRLF` para los `.bat`). **No toques `core.autocrlf`.**
4. **No subas nunca** el `venv`, el `.env`, los `output/`, los `logs/` ni las
   bases de datos. El `.gitignore` ya lo cubre; si algo se cuela, avisa antes
   de commitear.

### 4.7 Qué no tocar sin preguntar

- El **contrato** (`elde_core/contracts/`): lo comparten cliente y servidor, y
  romperlo rompe las dos puntas a la vez.
- La forma del **payload del websocket**: hay otros clientes fuera de este
  repositorio que hablan el mismo protocolo.
- Los **nombres de los modos**: son strings compartidos, y hay instalaciones
  con el nombre antiguo guardado.

---

## 5. Si algo no funciona

| síntoma | mira primero |
|---|---|
| El cliente ve el servidor «offline» | ¿lo lanzaste con `venv\Scripts\python.exe`? Con el Python global sube un servidor a medias que ocupa el 9000 |
| No detecta nada, sin errores | ¿los pesos son de verdad? `server/models/base/rf-detr-medium.pth` debe pesar ~387 MB, no 4 KB |
| Todo va lentísimo | ¿`onnxruntime` (CPU) pisando a `onnxruntime-gpu`? `server/scripts/verificar_entorno.py` lo detecta y sale con código ≠ 0 |
| `cv2` sin módulos | `opencv-python-headless` pisó al de contrib. Está en el README, paso obligatorio |
| Detecciones de cámaras mezcladas | algo guarda estado por nombre de ventana en vez de por `camera_id` |
| El servidor avisa de módulos ausentes | normal: son los verticales que no viven en este repositorio |
