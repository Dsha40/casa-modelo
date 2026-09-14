# Stand de degustación — sistema de medición

Rama `feature/stand-degustacion`. Convierte el stand de la charcutería
(Camera 04) en un sistema medible de punta a punta: cuántas personas pasaron,
cuántas se detuvieron, cuántas atendió el chef y cuántas **realmente
degustaron** el producto.

> **Estado honesto:** toda la lógica está implementada y probada (156 pruebas
> en verde), pero **ninguna cifra ha sido validada contra vídeo real
> todavía**. El HITO 13 está pendiente y necesita conteo manual humano. Hasta
> entonces, los números que produzca el sistema son plausibles, no
> defendibles. Ver [Limitaciones](#limitaciones-conocidas).

---

## 1. Arquitectura

Diez módulos nuevos, uno por responsabilidad. Ninguno de ellos hace
inferencia: reciben lo que el pipeline ya calcula (embeddings, keypoints,
muestras demográficas) y **deciden**. Por eso toda la lógica —que es donde
nacen los errores de conteo— se prueba sin GPU y en milisegundos.

```
                      ┌──────────────────────────────┐
   frame ─► RF-DETR ─►│ tiendas.py (ex person_amazona)│  (sin cambios)
                      └──────────────┬───────────────┘
                                     │ detecciones + embeddings + crops
                                     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  stand_config.py     umbrales + zonas (0..1, normalizadas)    │
   ├──────────────────────────────────────────────────────────────┤
   │  stand_identidad.py  tracker_id ──► persistent_id             │
   │                      anti-doble-conteo, galería con TTL       │
   ├──────────────────────────────────────────────────────────────┤
   │  stand_personal.py   ¿es el chef? ──► EXCLUIDO de métricas    │
   ├──────────────────────────────────────────────────────────────┤
   │  stand_demografia.py género/edad por PERSONA, no por frame    │
   ├──────────────────────────────────────────────────────────────┤
   │  stand_estados.py    DETECTADO ─► ... ─► DEGUSTÓ              │
   │  stand_degustacion.py  pose (barata) ──► VLM (caro, en cola)  │
   ├──────────────────────────────────────────────────────────────┤
   │  stand_heatmap.py    3 capas segmentadas                      │
   └────────────────┬─────────────────────────────┬───────────────┘
                    │                             │
       stand_almacen.py (SQLite)        stand_emisor.py
       5 tablas, 1 fila/persona         metadata['stand']
                    │                             │
        stand_metricas.py                         ▼
        KPIs · embudo · cruce            WebSocket ──► StandPanel
                    │                              (cliente Amazonas)
                    ▼
        src/app/stand_api.py  ──►  /dashboards/stand/
```

### Las tres ideas que sostienen la confiabilidad

1. **Se cuenta por `persistent_id`, nunca por detección ni por `tracker_id`.**
   Reforzado en la base de datos con `UNIQUE (sesion_id, persistent_id)`: una
   persona no puede duplicarse aunque la lógica se equivoque.
2. **Contar y conocer están desacoplados.** Un visitante sin demografía se
   cuenta igual, con su `motivo_sin_demografia`. Antes, `_emitir_visitante`
   descartaba entero a quien no tuviera género (ver `AUDITORIA_STAND.md` §3.2).
3. **Los estados son acumulativos.** Quien degustó cuenta también como
   atendido, detenido y pasante. El embudo no puede invertirse.

---

## 2. Cómo levantarlo

### 2.1 Habilitar el módulo

Está **apagado por defecto**. Con `habilitado: false`, Tienda y Perimetrales
se comportan exactamente como antes de este trabajo.

```yaml
# server/config/stand_config.yaml
habilitado: true
```

### 2.2 Calibrar las zonas (obligatorio)

Sin `zona_paso` y `zona_stand` el módulo **no cuenta y lo dice en el log**.
Nunca asume una zona por defecto: un polígono inventado produce cifras
creíbles y falsas, que es el peor resultado posible.

```bat
server\venv\Scripts\python.exe server\tools\calibrar_stand.py --pantalla
```

**Usa `--pantalla`, no `--ventana`.** iVMS-4200 corre **elevado**: `PrintWindow`
sobre su HWND devuelve `None` y `SetWindowPos` da «Acceso denegado», así que
un proceso sin privilegios no puede capturarla ni traerla al frente. Con otras
ventanas `--ventana` funciona; con esta no.

`--pantalla` lee `GetWindowRect` (que sí responde) y recorta ese rectángulo del
escritorio: mismo encuadre que captura el cliente, sin necesitar privilegios.
Da una cuenta atrás de 5 s para que pongas iVMS delante — **si queda tapada se
calibra la ventana que esté encima**. El editor te muestra el frame antes de
dibujar: si no ves la cámara, sal con `q` y repite.

Controles: `1`–`5` elegir zona · clic añadir punto · `u` deshacer · `r` vaciar
· `h` homografía · `s` guardar · `q` salir.

Escribe `config/stand_zonas.yaml` (coordenadas **normalizadas 0..1**, para que
sobrevivan a que la ventana cambie de tamaño) y
`config/stand_zonas_verificacion.png` para comprobarlo a ojo.

Alternativas si no puedes capturar la ventana:

```bat
REM Sobre un PNG/JPG que tengas
server\venv\Scripts\python.exe server\tools\calibrar_stand.py --imagen frame.png
REM Sobre un RTSP directo
server\venv\Scripts\python.exe server\tools\calibrar_stand.py --rtsp rtsp://...
REM Solo re-dibujar la verificación de lo ya guardado
server\venv\Scripts\python.exe server\tools\calibrar_stand.py --ventana --solo-verificar
```

### 2.3 Arrancar

```bat
INICIAR_TIENDA.bat                 REM servidor + dashboard + cliente
```

- Contadores en vivo: panel derecho del **cliente Amazonas**
- Dashboard analítico: **http://localhost:9000/dashboards/stand/**
- API: **http://localhost:9000/api/stand/sesiones**

---

## 3. Calibrar un stand NUEVO

1. Copia `stand_config.yaml` y ajusta el bloque `stand:` (`stand_id`,
   `nombre`, `producto`, `camara_id`).
2. Ejecuta `calibrar_stand.py` sobre la cámara del stand nuevo y dibuja las
   cinco zonas. Criterios:
   - **`zona_paso`** — el corredor. Define el denominador «pasaron». Si lo
     dibujas demasiado grande, la tasa de captación sale artificialmente baja.
   - **`zona_stand`** — el área de influencia frente al mostrador.
   - **`zona_atencion`** — solo donde ocurre la interacción cliente↔chef.
   - **`zona_personal`** — detrás del mostrador. De aquí sale la
     auto-propuesta de personal.
   - **`linea_conteo`** — cruzando el corredor, con dirección.
3. Si el ángulo lo permite, marca la **homografía** (`h`, 4 puntos del suelo
   con medidas reales). Sin ella los umbrales de distancia van en píxeles y
   dependen de la perspectiva — limitación documentada abajo.
4. Corre el HITO 13 (§5) antes de dar por buena ninguna cifra.

---

## 4. Runbook de fallas

### Los contadores del cliente están congelados

| Comprobar | Cómo | Qué significa |
|---|---|---|
| ¿Llega el bloque? | El panel muestra `● sin fps` | El servidor no emite: módulo deshabilitado o zonas sin calibrar |
| ¿El módulo cuenta? | `GET /api/stand/{id}/kpis` → `configuracion.listo_para_contar` | Si es `false`, el motivo viene en el mismo campo |
| ¿WebSocket vivo? | Log del cliente `logs/amazonas.log` | Busca `websocket desconectado` |
| ¿Sesión abierta? | `GET /api/stand/sesiones` → `estado` | Sin sesión abierta no hay nada que contar |

**El panel no acumula por diseño**: pinta totales absolutos. Si el número no
cambia es que el servidor manda el mismo, no que el cliente se haya perdido.

### El null-rate demográfico sube

1. `GET /api/stand/{id}/calidad` → `null_rate_demografia` y `sin_demografia`.
2. Mira `descartes_por_condicion` en `stand:salud`: dice **dónde** se pierde
   (`bbox_pequeno`, `borroso`, `rostro_pequeno`, `sin_rostro`).
3. Según la condición dominante, en `stand_config.yaml → demografia`:
   - `bbox_pequeno` → la gente está lejos. Sube `deteccion.imgsz` o baja los
     mínimos de bbox.
   - `borroso` → motion blur. Revisa iluminación y obturador de la cámara
     antes de tocar umbrales.
   - `sin_rostro` → normal en este encuadre; la rama corporal debe cubrirlo.
     Verifica que `demografia.genero.confianza_min` no esté demasiado alto.

**Recuerda:** el null-rate alto **no es un bug**. Los `DEMO_*` globales están
calibrados a precisión máxima a propósito (`DEMO_MIN_BLUR_VAR` ×8.3 sobre el
valor permisivo). El stand usa sus propios umbrales, más permisivos.

### El FPS cae

| Sospechoso | Comprobación | Mitigación |
|---|---|---|
| Cola del VLM | `stand:salud` → `vlm_en_cola` creciendo | Baja `degustacion.vlm.max_por_minuto` |
| Pose sobre todo el mundo | `frames_pose` desproporcionado | Verifica `ventana_post_atencion_s`: solo debe correr sobre atendidos |
| Heatmap | Payload de 30 s muy grande | Sube `heatmap.celda_px` (menos celdas) |
| Detector | — | Baja `deteccion.imgsz` de 960 a 640 |

Medido en esta máquina: pose = **41 fps / 0.11 GiB** con RTX 5060 Ti.

### La GPU se queda sin memoria

1. `nvidia-smi` — con el pipeline y el VLM caben en 15.9 GiB (medido: 14.56
   GiB libres solo con pose cargada).
2. El VLM es el consumidor grande. Si compite, usa **Qwen2.5-VL-3B** en vez
   del 7B: ambos están en la caché de HuggingFace.
3. Baja `degustacion.vlm.max_por_minuto`: la cola **encola, nunca descarta**,
   así que bajar el ritmo retrasa la confirmación pero no pierde degustaciones.

### Sospecho sobreconteo

1. `GET /api/stand/{id}/calidad` → `tracks_descartados` y `fusiones`.
2. Muchas fusiones = el Re-ID está partiendo personas. Baja
   `reid.umbral_merge` (0.78 → 0.74).
3. Pocas fusiones con gente entrando y saliendo = probable sobreconteo. Sube
   `conteo.min_frames_track`.
4. **Nunca leas `total_entradas` como si fueran personas**: se infla con cada
   reentrada. El conteo bueno es `pasaron` (ver `AUDITORIA_STAND.md` §4).

---

## 5. Validación contra ground truth (HITO 13) — **PENDIENTE**

Sin esto los números no son confiables. Antecedente directo: una validación
previa reveló sobreconteo del ~314 % por falta de deduplicación.

```bat
REM 1. Graba 30 min de operación real en tráfico medio.
REM 2. Cuenta a mano y rellena la plantilla:
server\venv\Scripts\python.exe server\tools\validar_stand.py --plantilla mi_conteo.csv

REM 3. Corre el pipeline sobre el MISMO vídeo, y compara:
server\venv\Scripts\python.exe server\tools\validar_stand.py --sesion 3 --manual mi_conteo.csv
```

La herramienta emite la tabla comparativa con error absoluto y porcentual, y
marca en rojo lo que excede el umbral:

| Métrica | Error máximo tolerado |
|---|---|
| Personas únicas | ±10 % |
| Se detuvieron | ±15 % |
| Atendidos | ±15 % |
| Degustaron | ±20 % |
| Género (accuracy) | ≥ 85 % |
| Rango etario (±1 bucket) | ≥ 70 % |

Si alguna falla: ajusta `stand_config.yaml`, repite, y **documenta cada
iteración con los valores probados**. No entregues el sistema hasta pasar.

---

## 6. Limitaciones conocidas

Lo que el sistema **no** hace bien, y bajo qué condiciones. Documentado a
propósito: un usuario que conoce los límites decide mejor que uno que confía
de más.

### Sin validar todavía

- **Ninguna cifra tiene error medido.** Los umbrales del Anexo A son un punto
  de partida razonable, no valores calibrados para esta cámara.
- El mapa de calor está probado mecánicamente, pero **nadie ha confirmado
  visualmente** que la capa de permanencia concentre calor frente al mostrador.

### De la cámara y el encuadre

- **Ojo de pez marcado.** Una homografía plana tendría error alto en los
  bordes, así que está desactivada y los umbrales de distancia van en
  píxeles: **dependen de la perspectiva**. Una persona al fondo se mueve
  menos píxeles que una cercana a la misma velocidad real.
- **Oclusión por el mostrador.** El encuadre corta a las personas por la
  cintura en la zona del stand, justo donde más importa medir.
- **Grupos densos.** Dos personas muy juntas pueden fusionarse en un track;
  el Re-ID lo corrige al separarse, pero durante la oclusión el conteo va bajo.
- **Contraluz** desde el pasillo del fondo degrada la rama facial. La corporal
  aguanta mejor, pero con edad más gruesa.

### De la detección de degustación

- **No detecta el objeto de la muestra.** La etapa 3 (vasito/bandeja) quedó
  fuera: el dataset ENTREGA-AMAZONAS no está en este equipo. Sin esa tercera
  señal, una degustación con pose ambigua depende solo del VLM.
- **Sin VLM disponible no se confirma nada.** Es deliberado: la pose sola
  confunde comer con rascarse la nariz o hablar por teléfono.
- La ventana de 90 s tras la atención acota el coste, pero **pierde a quien se
  aleja y prueba el producto más tarde**.

### De la demografía

- Precisión y recall **no medidos** en esta cámara.
- Sin rostro, la edad es inherentemente gruesa.
- Un `null` con motivo es un resultado válido y esperado, no un fallo.

### De identidad

- La galería es **efímera** (TTL 45 min, regla de privacidad): quien vuelva
  al día siguiente cuenta como persona nueva. Es intencionado.
- El umbral de la zona gris (0.65–0.78) con ventana de 120 s es un
  compromiso: fuera de esa ventana, una reaparición cuenta como persona nueva.

### De privacidad

- **No se almacenan rostros ni recortes de rostro en disco.** Solo embeddings
  con TTL y atributos agregados.
- El feed del cliente muestra hora, género y rango — **sin fotos**.

---

## 7. Mapa de archivos

| Archivo | Qué hace |
|---|---|
| `server/config/stand_config.yaml` | Todos los umbrales, comentados con su rango |
| `server/config/stand_zonas.yaml` | Zonas calibradas (**generado**, no editar a mano) |
| `server/src/analityc/core/stand_config.py` | Carga y valida umbrales + zonas |
| `server/src/analityc/core/stand_almacen.py` | SQLite: 5 tablas, 7 índices |
| `server/src/analityc/core/stand_identidad.py` | `persistent_id` y anti-doble-conteo |
| `server/src/analityc/core/stand_demografia.py` | Agregación por persona, null-rate |
| `server/src/analityc/core/stand_personal.py` | Chef: enrolamiento y exclusión |
| `server/src/analityc/core/stand_estados.py` | Máquina de estados del visitante |
| `server/src/analityc/core/stand_degustacion.py` | Pose + VLM + cola |
| `server/src/analityc/core/stand_heatmap.py` | 3 capas segmentadas |
| `server/src/analityc/core/stand_metricas.py` | KPIs, embudo, cruce, temporal |
| `server/src/analityc/core/stand_emisor.py` | Bloque `metadata['stand']` |
| `server/src/app/stand_api.py` | 10 endpoints REST |
| `server/tools/calibrar_stand.py` | Herramienta de calibración de zonas |
| `server/tools/validar_stand.py` | Comparación contra ground truth (HITO 13) |
| `clients/amazonas/.../stand_panel.py` | Recuadros en vivo + marcar personal |
| `dashboards/stand/` | Dashboard analítico (8 secciones) |

**Código existente modificado** (todo aditivo, detrás del flag):

| Archivo | Cambio |
|---|---|
| `server/src/app/app.py` | +8 líneas: registro del router |
| `clients/amazonas/src/main.py` | +22/−2: monta el panel en el lateral |
| `clients/amazonas/.../render_box.py` | +11: señal `stand_received` |
| `dashboards/index.html` | +5: enlace al dashboard |

---

## 8. Pruebas

```bat
REM Servidor (137 pruebas)
for %f in (server\tests\test_stand_*.py) do server\venv\Scripts\python.exe %f

REM Cliente (19 pruebas)
clients\amazonas\venv\Scripts\python.exe clients\amazonas\test\test_stand_panel.py
```

Ningún venv del repo tiene pytest, así que cada archivo trae su propio
ejecutor y corre con `python` a secas. Bajo pytest funcionan igual.

| Archivo | Pruebas |
|---|---|
| `test_stand_almacen.py` | 9 |
| `test_stand_api.py` | 25 |
| `test_stand_config.py` | 10 |
| `test_stand_degustacion.py` | 24 |
| `test_stand_demografia.py` | 14 |
| `test_stand_estados.py` | 15 |
| `test_stand_heatmap.py` | 17 |
| `test_stand_identidad.py` | 12 |
| `test_stand_personal.py` | 11 |
| `test_stand_panel.py` (cliente) | 19 |
| **Total** | **156** |

---

## 9. Documentos relacionados

- **`AUDITORIA_STAND.md`** — auditoría del código existente, causa raíz del
  null-rate con número de línea, y el análisis del riesgo de doble conteo.
- **`CONTRATO_STAND.md`** — esquema de las 5 tablas, contrato del WebSocket y
  rutas REST.
