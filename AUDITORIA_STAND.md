# AUDITORÍA — Stand de degustación (HITO 1)

Rama: `feature/stand-degustacion` · Fecha: 5-ago-2026 · Sin modificar código.

Objetivo de este documento: saber exactamente qué hay antes de tocar nada, y
responder con evidencia las dos preguntas que deciden si el sistema será
confiable — **por qué la demografía sale nula** y **cómo se cuenta hoy**.

---

## 1. Mapa del archivo de inferencia

`server/src/analityc/core/person_amazona_inference.py` — **3620 líneas**, una
clase principal y ~85 métodos.

| Bloque | Líneas | Contenido |
|---|---|---|
| Constantes y rutas de modelos | 36–80 | Caffe, InsightFace, MiVOLO, ArcFace, PAR, OSNet |
| Providers ONNX | 82–115 | GPU + flag TensorRT |
| `_GpuYuNet` | 117–216 | Detector facial YuNet en GPU vía onnxruntime |
| `_AutoAngleDetector` | 249–308 | Deduce el ángulo de cámara por relación alto/ancho |
| `PersonAmazonas.__init__` | 316–609 | Tracker, Re-ID, zonas, demografía, analítica |
| Utilidades geométricas | 746–805 | IoU, centro, punto-en-polígono |
| Perfiles por cámara | 822–905 | Overrides por `camera_id` |
| **Demografía (init)** | 906–1103 | Carga de los 6 modelos, sesiones compartidas |
| **Demografía (uso)** | 1104–1154 | `_classify_person` |
| Alertas y fotos de personal | 1155–1327 | Legado del modo «staff» |
| Estado multi-cámara | 1328–1473 | `_get_cam_state`, `get_camera_processor` |
| Tracking propio (IoU) | 1528–2017 | Ruta legada |
| **Tracking BoT-SORT** | 2018–2137 | Ruta actual |
| **Identidad persistente** | 2138–2219 | `_resolve_track_identities` |
| **Transiciones de zona (CONTEO)** | 2238–2321 | `_process_zone_transitions` |
| Estadísticas y reporte | 2322–2427 | `get_visitor_stats`, `get_analytics_report` |
| **Capturas** | 2428–2873 | Recorte, anotación, dedup, sellado de motivo |
| **Emisión al cliente** | 2874–2953 | `_emitir_visitante`, `_drenar_alertas` |
| Dibujo del overlay | 3007–3127 | `draw_detections` |
| **`process_frame`** | 3128–3492 | Orquestador del frame |
| Ajustes en caliente | 3493–3619 | ROI, reset, stats |

### Flujo de un frame

```
process_frame(image, roi, camera_id, heatmap_activate)
  ├─ _get_cam_state(camera_id) / _push_state       estado por cámara
  ├─ detector de personas (clase 0)
  ├─ _update_tracks_botsort(detections)            tracker_id
  ├─ _resolve_track_identities(frame)              tracker_id -> persistent_id (Re-ID)
  ├─ _classify_person(frame, bbox, track_id)       demografía por muestra
  ├─ _process_zone_transitions()                   ← AQUÍ SE CUENTA
  ├─ _capture_established_tracks(frame)            foto + sidecar JSON
  ├─ _update_captures_demographics()               consolida demografía, dedup, emite
  ├─ _run_analytics(frame)                         heatmap
  ├─ draw_detections(...)                          overlay
  └─ return (frame_anotado, metadata)              metadata['alerts'] ← _drenar_alertas
```

> El detector era YOLO26m cuando se escribió esta auditoría (5-ago-2026). El
> **3-sep-2026** se retiró `ultralytics` (AGPL): hoy detecta el RF-DETR
> compartido (`core/deteccion_compartida.py`) y el resto del flujo es el
> mismo, porque las detecciones siguen llegando con `class_id` 0 = persona.

---

## 2. Qué ya existe y es reutilizable

Bastante más de lo esperado. **No hay que construir el pipeline desde cero.**

| Pieza | Estado | Dónde | Reutilizable para |
|---|---|---|---|
| Detector de personas | ✅ producción | `_initialize_model` :692 | Base de todo |
| BoT-SORT / ByteTrack | ✅ producción | `_update_tracks_botsort` :2018 | `tracker_id` |
| Re-ID facial (ArcFace) | ✅ producción | `_resolve_track_identities` :2138 | `persistent_id` |
| Re-ID corporal (OSNet) | ✅ cargado | `_osnet_session` :922 | Identidad sin rostro |
| Demografía 2 ramas | ✅ existe | `analytics/demographics.py` (2438 líneas) | HITO 5 |
| **Telemetría demográfica** | ✅ **ya instrumentada** | `_veredicto_final` :2271 | HITO 5 + panel de calidad |
| Zonas poligonales | ✅ producción | `_get_zone` :644, `sv.PolygonZone` | HITO 3 |
| Máquina de estados de zona | ✅ producción | `_process_zone_transitions` :2238 | Base del HITO 7 |
| Heatmap | ✅ existe | `analytics/heatmap.py` | HITO 9 |
| Emisión al cliente | ✅ producción | `_emitir_visitante` :2874 | HITO 11 |
| VLM Qwen2.5-VL | ✅ 3B y 7B en caché HF | `tools/bench_qwen_vlm.py` | HITO 8 etapa 2 |
| **Pose** | ❌ **no existe** | — | **Falta para HITO 8 etapa 1** |

**Enum `MotivoFinal`** ya existe con las causas de fallo tipificadas
(`SIN_MODELO`, `ROSTRO_MUY_PEQUENO`, `CALIDAD_INSUFICIENTE`, `SIN_ROSTRO`,
`MUESTRAS_INSUFICIENTES`, `TRACK_NO_CERRADO`, `CLASIFICADO_SOLO_CUERPO`,
`HEREDADO_REID`, `EXCEPCION_ESTIMADOR`). El campo `motivo_sin_demografia` del
HITO 2 **ya está implementado** — no hay que inventarlo.

---

## 3. Diagnóstico del null-rate demográfico

### La causa raíz NO es un bug

El módulo fue **endurecido deliberadamente** a precisión sobre recall. El propio
código conserva anotado el valor anterior de cada umbral, en
`server/src/analityc/core/analytics/config.py`:

| Umbral | Línea | Hoy | `[PERMISIVO]` | Factor |
|---|---|---|---|---|
| `DEMO_MIN_BLUR_VAR` | :154 | 25.0 | 3.0 | **×8.3** |
| `DEMO_MIN_FACE_SIZE` | :151 | 40 px | 14 px | ×2.9 |
| `DEMO_MIN_MARGIN` | :102 | 0.35 | 0.12 | ×2.9 |
| `DEMO_MIN_PERSON_BBOX_H` | :214 | 140 px | 80 px | ×1.75 |
| `DEMO_MIN_CONFIDENCE` | :100 | 0.85 | 0.55 | ×1.5 |
| `DEMO_MIN_SAMPLES` | :96 | 5 | 3 | ×1.67 |

Más dos interruptores que amplifican el efecto:

- `DEMO_REQUIRE_ENSEMBLE_AGREEMENT = True` (:267) — si dos clasificadores
  discrepan en género, el resultado es `Desconocido`. Comentario textual del
  código: *"Cero falsos positivos"*.
- `DEMO_SHOW_PROVISIONAL = False` (:196) — solo se publica lo *committed*.

**Conclusión:** el null-rate alto es el comportamiento *diseñado* de una
calibración precision-first. El HITO 5 no es «reparar código roto», es
**reubicar el punto de operación** para esta cámara y este caso de uso.

### Dónde se pierde exactamente

Dos pérdidas distintas, en dos capas:

**(a) Pérdida en la inferencia** — `_sample_once`, `demographics.py:1084`:

```python
if (person_w < self._cfg.DEMO_MIN_PERSON_BBOX_W or
        person_h < self._cfg.DEMO_MIN_PERSON_BBOX_H):
```
Con 60×140 px mínimos, una persona al fondo del pasillo nunca genera muestra.

**(b) Pérdida en la emisión** — `person_amazona_inference.py:2894`:

```python
if not genero or genero.lower() == 'desconocido':
    return              # sin demografia util no hay nada que contar
```

**Esta es la línea más importante de la auditoría.** Un visitante sin
demografía **no genera evento alguno**: no es que se cuente mal, es que
*desaparece* del panel del cliente y del WhatsApp. Para el stand esto es
inaceptable — una persona que degustó cuenta aunque no sepamos su género.
El HITO 5/7 debe desacoplar «contar» de «saber quién era».

### Números: no los tengo, y no los voy a inventar

La spec pide cuántos crops llegan a cada rama y cuántos se descartan por cada
condición. Esos contadores existen (`rechazos_pose`, `rechazos_calidad`,
`rechazos_rostro_pequeno`, `intentos_sin_rostro`, `mejor_ancho_rostro_px`) y se
vuelcan a `output/telemetria_demografica.jsonl`.

**Ese archivo se borró hoy**, en la limpieza de la base de datos de tienda
solicitada antes de este trabajo. Los datos que quedaban eran además del
27-jul, de otra cámara y otra calibración: no habrían sido representativos del
stand. Los números reales se obtienen ejecutando el pipeline sobre el vídeo del
stand, y es lo primero que hará el HITO 5.

---

## 4. Riesgo de doble conteo

**El conteo actual NO es por detección ni por `tracker_id`: es por
`persistent_id`.** `_process_zone_transitions` (:2238) con:

- `ZONE_ENTRY_REQUIRES_CONFIRMED` — los `prov:` (provisionales, sin rostro) no
  disparan transición (:2266–2270).
- `ZONE_HYSTERESIS_FRAMES` — N frames consecutivos antes de confirmar (:2303).
- `self._unique_entered` — **set** de `persistent_id` (:2309).

### Pero hay dos trampas reales

**Trampa 1 — dos métricas que se parecen y no son lo mismo.** Conviven:

| Campo | Semántica | Se infla al reentrar |
|---|---|---|
| `visitantes_unicos` = `len(_unique_entered)` | personas únicas | **No** (set) |
| `total_entradas` = `_total_entries` | transiciones FUERA→DENTRO | **Sí, +1 por reentrada** |

Un sobreconteo tipo ~314% es exactamente lo que se obtiene al leer
`total_entradas` creyendo que son personas. **Para el stand, el denominador
"pasaron" debe salir de `persistent_id` únicos, nunca de `total_entradas`.**

**Trampa 2 — parpadeo del tracker.** En :2276 toda persona sin track activo se
observa como FUERA. Si el tracker la pierde unos frames (oclusión por el
mostrador o por otra persona — frecuente en este encuadre), se cuenta una
salida y luego una reentrada. La histéresis lo mitiga; no lo elimina. Es un
riesgo a medir en el HITO 13, no a dar por resuelto.

**Trampa 3 — el chef.** Hoy **no existe** ningún mecanismo que excluya al
personal del conteo. El chef está en el encuadre toda la jornada: es el track
más persistente de la escena y hoy cuenta como visitante. El HITO 6 es
obligatorio antes de que cualquier cifra sea creíble.

---

## 5. Cliente Amazonas

`clients/amazonas/` — 54 archivos Python. Modo multimodo (`Hummus`,
`Autolavado`, `Perimetrales`, `PerimetralesMultiCam`,
`Personal de Amazonas`).

### Cómo recibe datos hoy — dos canales independientes

**Canal A — WebSocket** (`ws://<servidor>:9000/ws`, definido en
`clients/amazonas/.env`). Devuelve el frame anotado y `metadata`, que incluye
`alerts` alimentado por `_drenar_alertas` (:2943).

**Canal B — HTTP polling de archivos.** `CapturasSidebar`
(`gui/components/sidebar/capturas_sidebar.py:170`) **no consume el WebSocket**:
lee los JPG+JSON de captura del servidor por HTTP (`_leer_capturas` :345,
`_servidor_http` :471, `refrescar` :386) y cuenta géneros con
`_contar_generos` (:321).

Esto importa: el panel de personas del cliente Amazonas **hoy se alimenta de
ficheros, no de eventos**. Su propio encabezado documenta que el panel «Alertas
IA» se retiró por quedar permanentemente vacío.

### Dónde encajan los recuadros en vivo (HITO 11)

- Las dos tarjetas (`PERSONAS ATENDIDAS` / `PERSONAS QUE DEGUSTARON`) van sobre
  `_construir_totales` (:277), que ya es la fila de totales del sidebar y ya
  tiene el patrón de etiqueta con color (`_etiqueta` :296).
- La fuente debe ser el **WebSocket** (decisión de infraestructura tomada en el
  HITO 0), no el polling de ficheros: los ficheros solo existen para personas
  con captura, y el conteo de atendidos/degustaciones debe incluir a quien no
  tiene demografía.
- `_vaciar_detecciones` (:417) ya existe y da el patrón para resetear contadores.

---

## 6. Decisiones de infraestructura (cerradas en HITO 0)

La spec asumía tres piezas que **no existen en este repo**. Resueltas así:

| Spec original | Realidad | Decisión |
|---|---|---|
| MongoDB, colecciones `stand_*` | Sin servicio, sin `pymongo`, cero referencias | **SQLite**, mismas entidades como tablas |
| Socket.IO namespace `/stand` | El transporte es WebSocket de FastAPI | **Reutilizar el WebSocket existente** |
| Dashboard con `VITE_*` | Dashboards en JS vanilla, sin bundler | El bug de `VITE_*` **no aplica** |

---

## 7. Riesgos abiertos

1. **El chef no está excluido** — invalida toda cifra hasta el HITO 6.
2. **Sin modelo de pose** — lo era al escribir esto; se resolvió por otro
   camino. La pose del HITO 8 la da **ViTPose-base**
   (`server/models/pose/vitpose-base-simple/`, `core/pose_motor.py`), que
   sustituyó a `yolo11m-pose.pt` el 3-sep-2026. **No hay que descargar
   ningún `.pt` de YOLO.**
3. **Oclusión por el mostrador** — el encuadre corta a las personas por la
   cintura en la zona del stand; `DEMO_MIN_PERSON_BBOX_H = 140 px` puede
   descartarlas justo donde más importa medir.
4. **Fichero de inferencia compartido** — el mismo `PersonAmazonas` sirve a
   Tienda. Todo cambio va detrás de `MODULO_STAND_ENABLED` (regla 2).
5. **Sin ground truth todavía** — ninguna cifra es defendible hasta el HITO 13.
