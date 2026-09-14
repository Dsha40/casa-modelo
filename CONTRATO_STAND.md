# CONTRATO — Stand de degustación (HITO 2)

Rama `feature/stand-degustacion`. Este documento fija las formas de mensaje y
las rutas. **Nada de lo que está aquí se inventa sobre la marcha**: si el
pipeline necesita un campo que no esté listado, se añade aquí primero.

---

## 1. Persistencia — SQLite

Archivo: `server/output/stand/stand.db` (override: `STAND_BD`).
Implementación: `server/src/analityc/core/stand_almacen.py`.

La spec pedía MongoDB; el HITO 0 verificó que **no existe** en este despliegue
(sin servicio, sin `pymongo`, nadie en 27017). Se traduce 1:1 a cinco tablas.
Los campos anidados van como JSON en columnas TEXT.

| Tabla | Filas | Papel |
|---|---|---|
| `stand_sesiones` | una por jornada | Guarda `config_snapshot` con los umbrales vigentes al abrir |
| `stand_visitantes` | **una por persona única** | **Unidad de verdad del conteo** |
| `stand_eventos` | muchas por persona | Bitácora inmutable; **nunca se cuenta** |
| `stand_heatmap` | snapshot cada 30 s | 3 tipos: `trafico`, `permanencia`, `detenciones` |
| `stand_metricas` | una por bucket 15 min | Agregados precalculados |

Índices: `idx_vis_pid`, `idx_vis_sesion`, `idx_vis_clasif`, `idx_ev_sesion_ts`,
`idx_ev_pid`, `idx_hm_sesion_ts`, `idx_met_sesion`.

**Restricción clave:** `UNIQUE (sesion_id, persistent_id)` en
`stand_visitantes`. Es lo que hace estructuralmente imposible que una persona
aparezca dos veces — el sobreconteo deja de depender de que la lógica sea
correcta y pasa a depender de la base de datos.

### Vocabulario cerrado

Constantes en `stand_almacen.py`, no literales sueltos:

- `Clasificacion`: `paso_de_largo` · `se_detuvo_sin_atencion` ·
  `atendido_sin_degustar` · `degusto`
- `TipoEvento`: `entrada_escena` · `entrada_zona_stand` · `detencion` ·
  `inicio_atencion` · `fin_atencion` · `degustacion` · `salida_zona_stand` ·
  `salida_escena` · `merge_identidad` · `descarte`
- `MotivoSinDemografia`: `sin_muestras` · `baja_confianza` · `track_corto`

`registrar_evento` y `upsert_visitante` **lanzan `ValueError`** ante un tipo o
un campo desconocido: un typo revienta en el acto en vez de perderse en
silencio y dejar un hueco inexplicable en el dashboard.

---

## 2. Emisión en vivo — WebSocket existente

La spec pedía Socket.IO con namespace `/stand`. El HITO 0 verificó que el
transporte real cliente↔servidor es el **WebSocket de FastAPI**
(`ws://<servidor>:9000/ws`, endpoint `/ws/{type_inference}` en
`server/src/app/app.py:733`). Se reutiliza: el cliente ya está conectado, y no
se añade un segundo canal con su propia reconexión.

Los eventos viajan dentro de `metadata`, en una clave nueva `stand`. Los
clientes que no la conocen la ignoran (leen con `.get()`), así que **Tienda y
Perimetrales no se enteran**.

```jsonc
metadata: {
  "detections": [...],        // sin cambios
  "alerts": [...],            // sin cambios
  "stand": {                  // ← NUEVO, solo si MODULO_STAND_ENABLED
    "sesion_id": 7,           // en todo mensaje: el cliente construye con él
                              // la URL del POST de enrolamiento y del GET
                              // de resincronización
    "tracks_activos": [       // cada 2 s — alimenta "Marcar como personal"
      { "persistent_id": "p-4f2a91c3",
        "genero": "hombre", "rango_etario": "36-50",
        "zona": "zona_personal", "es_personal": false, "dwell_s": 612.4 }
    ],
    "contadores": {           // cada 2 s
      "pasaron": 128, "se_detuvieron": 74,
      "atendidos": 47, "degustaron": 31,
      "tasas": { "captacion": 0.578, "atencion": 0.635,
                 "degustacion": 0.660, "global": 0.242 }
    },
    "ultimos_eventos": [      // feed del cliente, máx. 10
      { "tipo": "degustacion", "ts": "15:28:12",
        "genero": "mujer", "rango_etario": "26-35",
        "confianza": 0.88 }
    ],
    "salud": {                // cada 10 s
      "fps": 18.4, "latencia_ms": 220, "gpu_mem_mb": 4100,
      "null_rate_demografia": 0.12, "tracks_activos": 6,
      "vlm_encoladas": 0
    },
    "heatmap": {              // cada 30 s, solo si el cliente lo pide
      "tipo": "permanencia",
      "grid": { "ancho": 64, "alto": 36, "celda_px": 20 },
      "matriz": [[...]]
    }
  }
}
```

**Frecuencias** (throttling en el emisor, no en el cliente): contadores 2 s ·
tracks 2 s · salud 10 s · heatmap 30 s · `ultimos_eventos` al confirmarse
cada uno.

**`tracks_activos` lleva solo lo mínimo** para señalar a una persona en una
lista: ni embeddings, ni recortes, ni trayectoria. Enviar más engordaría el
payload cada 2 s y sacaría datos biométricos del servidor sin motivo (regla 7).
Los `prov:` se excluyen: una identidad sin confirmar cambia de id al
confirmarse, y enrolarla apuntaría a nadie. Tope de 20 por payload.

**Reconexión:** el cliente NO acumula contadores localmente. Cada payload trae
el total absoluto, así que una reconexión se autocorrige sola con el siguiente
mensaje. Al arrancar pide el estado por REST (`GET /api/stand/{sesion}/kpis`).

---

## 3. REST

Bajo el mismo servidor FastAPI del puerto 9000.

| Método | Ruta | Devuelve |
|---|---|---|
| GET | `/api/stand/sesiones` | Lista de jornadas |
| GET | `/api/stand/{sesion}/kpis` | Las 4 cifras + las 4 tasas |
| GET | `/api/stand/{sesion}/embudo` | Escalones con caída absoluta y % |
| GET | `/api/stand/{sesion}/demografia` | Distribuciones + cruce género × edad |
| GET | `/api/stand/{sesion}/temporal?bucket=15m` | Serie temporal |
| GET | `/api/stand/{sesion}/heatmap?tipo=` | `trafico` · `permanencia` · `detenciones` |
| GET | `/api/stand/{sesion}/visitantes?export=csv` | Tabla filtrable / CSV |
| GET | `/api/stand/{sesion}/calidad` | Panel de calidad del dato |
| POST | `/api/stand/{sesion}/personal/enrolar` | `{persistent_id}` → marca como chef |

`/calidad` no estaba en la spec y se añade por la regla final: ninguna cifra se
presenta como definitiva sin su panel de calidad al lado.

---

## 4. KPIs

Calculados sobre `stand_visitantes` **excluyendo `es_personal` y
`descartado`** (el filtro es el valor por defecto de `visitantes()`, hay que
pedir explícitamente lo contrario).

```
pasaron          = COUNT(*)
se_detuvieron    = SUM(se_detuvo)
atendidos        = SUM(fue_atendido)
degustaron       = SUM(degusto)

tasa_captacion   = se_detuvieron / pasaron
tasa_atencion    = atendidos / se_detuvieron
tasa_degustacion = degustaron / atendidos
tasa_global      = degustaron / pasaron
```

**Invariante verificada en prueba:**
`degustaron ≤ atendidos ≤ se_detuvieron ≤ pasaron`.
Se garantiza porque los estados son **acumulativos**: al marcar `degusto` se
marcan también `fue_atendido` y `se_detuvo`. Un embudo invertido sería un bug,
y la prueba `test_embudo_monotono_y_exclusiones` lo detecta.

Denominador cero → la tasa es `null`, **nunca 0**. Una tasa de conversión del
0% y «todavía no ha pasado nadie» son cosas distintas y no pueden mostrarse
igual.

---

## 5. Cómo probarlo

```bat
server\venv\Scripts\python.exe server\tests\test_stand_almacen.py
```

Ningún venv del repo tiene pytest (verificado 5-ago-2026), así que la prueba
trae su propio ejecutor y corre con `python` a secas. Bajo pytest funciona
igual si algún día se instala.
