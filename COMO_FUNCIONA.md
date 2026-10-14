# Cómo funciona el proyecto (estado al 14-oct-2026)

Este documento explica cómo trabaja hoy el sistema de analítica de tienda:
qué hace cada pieza, por dónde pasa cada imagen, qué decide el servidor y
qué ajustes lo gobiernan. Recoge los cambios de octubre de 2026 (stock,
agarres confirmados por bajada, áreas a ignorar, filtro con memoria,
identidad, cortes de iVMS e informe PDF). Para instalar, mira `README.md`,
`clients/tienda/README.md` y `GUIA.md`.

Cada cifra que aparece aquí está en el código o se midió. Lo que no está
comprobado se dice.

---

## 1. Qué es

Un sistema que mira un **pasillo de supermercado** (baldas de café) con las
cámaras de la tienda y cuenta:

- **cuánta gente pasa**, quién entra en el área de conteo y cuánto se queda;
- **género y edad aproximada** de quien se puede clasificar;
- **visitas y agarres** en cada anaquel y en cada producto dibujado;
- **bajadas de stock** (falta algo en la balda) con foto antes y después;
- **mapas de calor** del pasillo y de cada anaquel.

Lo ven en el **dashboard** (`/dashboards/tienda/`) y en el **informe PDF**
(botón «Exportar PDF»).

### Las tres partes

| Parte | Dónde | Qué hace |
|---|---|---|
| **Cliente** (Windows) | `clients/tienda/` y `packages/elde_core/` | Copia la ventana de iVMS (o el vídeo), deja dibujar las zonas y envía cada imagen al servidor |
| **Servidor** | `server/src/` | Detecta y sigue a las personas, decide identidad, género, visitas, agarres y stock, y guarda todo |
| **Dashboard e informe** | `dashboards/tienda/`, `server/src/app/` | Lo enseñan: páginas, API y PDF |

Las cámaras: la **cámara 3** (`win-iVMS-4200-c3-i58bf7b`) es la de las baldas
de café. Para probar se usa una **grabación de esa cámara** en Media Player
(`win-Media_Player-c3-i58bf7b`).

---

## 2. El recorrido de una imagen

### 2.1 En el cliente

1. `capture_worker.py` copia la ventana (iVMS o el reproductor) y la
   comprime en JPEG (calidad 70, ancho máximo 1600 px).
2. El recuadro (`render_box.py` + `render_box_captura.py`) se queda con la
   última captura y la envía por websocket con todo lo dibujado encima, en
   píxeles de la imagen enviada: área de conteo (ROI), áreas de anaquel,
   productos y **áreas a ignorar** (`areas_ignorar.py`).
3. **Modo F** (`socket_client.py`): cada imagen sale al socket en el acto,
   sin esperar a que termine el dibujo. Llevó el vídeo de 9,85 a 14,29
   imágenes por segundo.
4. **Red de seguridad**: si una respuesta no llega en 5 s
   (`ESPERA_RESPUESTA_MAX_S`), el recuadro la da por perdida y vuelve a
   enviar. El 8-oct se perdieron 3 h 27 min por una respuesta perdida.
5. Al llegar la respuesta, el cliente dibuja cajas, muñecas, productos y
   alertas.

### 2.2 En el servidor

`server/src/app/app.py` recibe la imagen. `ws_drenaje.py` guarda solo la
última de cada cámara, y el procesador de esa cámara la trata. El modo
`Tiendas` (`tiendas.py`) hace lo común y el modo `Anaqueles`
(`anaqueles.py`) añade anaqueles y stock.

```
imagen
  -> ¿corte de iVMS? (cortes_video; pausa OPCIONAL, apagada)
  -> detector de personas RF-DETR (deteccion_compartida; imagen preparada en la GPU)
  -> filtro del fondo: MARCA a quien mide menos de X px de alto (filtro_fondo)
  -> ByteTrack: da un NÚMERO de seguimiento a cada persona
  -> filtro con memoria: quita a quien NUNCA valió (filtro_memoria)
  -> áreas a ignorar: quita a quien queda con la caja ENTERA dentro (zonas_ignorar)
  -> entradas/salidas del área de conteo y permanencia (tienda_almacen)
  -> identidad (cara y cuerpo) y demografía (clasificador B)
  -> ficha de la persona (output/captures/persons/)
  -> mapa de calor del pasillo
  -> [Anaqueles] pose (muñecas), visitas, agarres, stock
  -> respuesta al cliente
```

#### Detector y seguimiento

- **RF-DETR** detecta personas. La imagen se prepara en la GPU
  (`imagen_a_gpu.py`; en el PC de la tienda pasó de 5,22 a 7,71 imágenes
  por segundo).
- **ByteTrack** (`sv.ByteTrack`) da un **número** a cada persona y lo
  mantiene mientras la ve. Recuerda a quien deja de ver durante
  `BYTETRACK_LOST_S` = 9 s, y exige verla `BYTETRACK_MIN_CONSECUTIVE_S` =
  0,35 s antes de darle número. Estos ajustes van en **segundos**, no en
  imágenes (`ritmo_imagenes.py`), para que no cambien con la velocidad.
- **Los números se reinician** cada vez que arranca el servidor.

#### Filtros de la gente que no cuenta

Tres capas, en este orden:

| Capa | Archivo | Regla |
|---|---|---|
| Filtro del fondo | `filtro_fondo.py` | Marca como «no vale» a quien mide menos de `PERSON_MIN_HEIGHT_PX` de alto. Va por cámara: **50 px** en el vídeo y **150 px** en la cámara 3 en vivo (`server/config/camera_profiles/`). Global: 0, apagado |
| Filtro con memoria | `filtro_memoria.py` | ByteTrack sigue a todos, pero detrás se quita a quien **nunca** valió. Así, quien rondaba el umbral (al agacharse su caja baja un instante) ya no se parte en dos números |
| Áreas a ignorar | `zonas_ignorar.py` | Se dibujan en el recuadro. Quita, **sin memoria**, a quien tiene su caja **entera** dentro; con un trozo fuera, cuenta. Desde el 14-oct va después del seguimiento: con memoria se colaba gente del fondo que antes había pasado por delante |

Cada minuto el log escribe una línea por capa: `FONDO [...]`,
`MEMORIA [...]` e `IGNORAR [...]`, con cuántas detecciones quitó.

#### Identidad

Una **identidad** es «quién es esta persona», más allá de su número.

- **Por la cara** (`face_reidentifier.py`): huella de la cara (ArcFace).
  Se compara con las caras guardadas y, si el parecido llega a
  `REID_SIMILARITY_THRESHOLD` = 0,35, es esa persona. Si no se parece a
  nadie, se crea una identidad nueva, solo con caras casi de frente. La
  galería **olvida** a cada persona 30 min después de su última aparición
  (`REID_TTL_S` = 1800) y vive en memoria.
- **Por el cuerpo** (OSNet): **solo vuelve a enlazar** a una identidad que
  ya existe por la cara, por ejemplo de espaldas. Nunca crea identidades.
  En este pasillo distingue mal.
- Sin cara ni cuerpo, la persona queda como `prov:<número>`.
- **Conservar la identidad** (14-oct, `REID_CONSERVAR_IDENTIDAD` = True):
  si un número que ya era alguien enseña una cara que no se parece a nadie,
  **conserva su identidad** en vez de estrenar otra. El 14-oct, al menos 5
  de 123 identidades nuevas eran ese caso. Si la cara sí se parece a OTRA
  identidad, se reasigna. Deja `REID conserva ...` en el log.
- **Cara propia** (14-oct, `DEMO_B_CARA_PROPIA` = True): de las caras que
  hay cerca de una persona solo vale la que cae **en su cabeza**, no la más
  nítida, que a veces era la del vecino. Medido con 251 personas
  etiquetadas a mano: la identidad que más personas juntaba bajó de 10 a 5
  números; el acierto de género subió de 93,0 a 94,2 %; las personas con
  identidad bajaron de 36 a 33 %.

#### Género y edad: el clasificador B

`clasificador_b.py` es el oficial desde el 9-oct (`DEMO_CLASIFICADOR` =
`b`; `ELDE_CLASIFICADOR=clasico` vuelve al anterior).

- Mira a cada persona cada ~0,6 s y lee el género con **MiVOLO** (cara sin
  alinear + cuerpo). Una búsqueda de cara por intento.
- **Decide por evidencia**: suma lo que dice cada lectura y decide cuando
  llega al **90 % de seguridad si alguna lectura tuvo cara**
  (`DEMO_B_SEGURIDAD_CON_CARA`) o al **95 % si solo hubo cuerpo**
  (`DEMO_B_SEGURIDAD`). Pide al menos 2 lecturas y no más de un 25 % en
  contra.
- **No hereda** el género de la identidad: cada número se decide con lo
  que se ve.
- **Guardar el jurado 9 s** (`demografia_aplazada.py`): si ByteTrack pierde
  a alguien, sus lecturas no se borran hasta pasar `BYTETRACK_LOST_S`. Si
  vuelve antes, sigue donde estaba.
- Medido con etiquetas a mano (8-oct, en sombra): decidió el 63 % con un
  acierto del 94,9 %.

#### Fichas y base de datos

- **Ficha**: una por persona seguida, con su foto
  (`output/captures/persons/<stem>.json` y `.jpg`). Se cierra con lo
  primero que llegue (identidad y género); lo que falte se espera 5 min.
- **Base** `output/tienda.db` (`tienda_almacen.py`): entradas y salidas
  (`tienda_eventos`), visitas (`tienda_visitas`), índice de fichas
  (`tienda_capturas`, que se pone al día al consultarlo) y clasificaciones
  (`tienda_clasificaciones`).

#### Mapa de calor del pasillo

`heatmap.py` acumula dónde pisa la gente (los pies). La imagen se pinta
**bajo pedido**, cuando alguien la pide en el dashboard o el informe. El
**fondo** se guarda cada 60 s con el pasillo vacío. Hay histórico por
horas en `output/heatmap/history/`.

---

## 3. Anaqueles: visitas, agarres y stock

Solo en el modo **Anaqueles**. El operador dibuja en el recuadro las
**áreas de anaquel** y, dentro, los **productos** (menú «Área ▾»).

### 3.1 La mano

- **ViTPose** da 17 puntos del cuerpo; se usan las **muñecas**. Corre en la
  GPU y en lote (`pose_gpu.py`). La primera vez que hay alguien junto a un
  estante tarda ~1 minuto en cargar, y el recuadro se queda congelado
  mientras tanto.
- Se calcula a quien está en el área **o cerca** (`pose_cercania: true`
  en `server/config/anaqueles_config.yaml`), para no perder la mano que
  sube a la balda alta.
- **La muñeca no es la mano**: queda unos 70 px por debajo de los dedos.
  Con productos estrechos y vecinos, el producto bajo la muñeca no es
  fiable.

### 3.2 Visita

Una **visita** es una muñeca **dentro del área** 0,5 s seguidos
(`visita_muneca_s`), una por estancia de la mano. La misma persona no repite
visita en la misma área antes de 120 s (`visita_cooldown_s`).

### 3.3 Agarre: gesto + bajada

1. **Gesto**: la muñeca entra en el área y sale. Hay reglas contra falsos,
   como la misma mano vista dos veces o la mano que baja desde la balda de
   arriba. El gesto deja el agarre **«por confirmar»**.
2. **Confirmación**: tras el gesto, el stock vigila los productos de esa
   área. Si ve que a uno **le falta algo** (una bajada o un posible
   agotado), el agarre **cuenta** y se apunta a **ese producto**, no al que
   caía bajo la muñeca.
3. Si la vigilancia termina sin bajada, se guarda como
   **`agarres_sin_confirmar`**: no suma, pero queda en el archivo del día.
4. `agarre_requiere_bajada: false` vuelve a contar el gesto solo, como
   antes. Entre dos agarres de la misma persona pasan al menos 10 s
   (`agarre_cooldown_s`).

Cada visita y cada agarre se apuntan **con su hora**, persona y género en
`output/anaqueles/<cámara>/eventos/<fecha>.jsonl` (`anaqueles_eventos.py`).
Los totales del día van en `output/anaqueles/<cámara>/<fecha>.json`.

### 3.4 Stock (`anaqueles_stock.py`)

Compara cada producto con su **foto de referencia** (balda surtida):

- **Foto de referencia**: a mano (todos o un producto, menú del cliente),
  **cada día a las 07:30** (`stock_reinicio_diario`) y **rehecha tras cada
  bajada**. Solo se toma cuando nadie tapa el producto.
- Tras un agarre se vigila el área hasta **180 s** (`stock_ventana_s`).
- **Bajada**: una medida con cambio **≥ 0,50** (`stock_umbral_bajada`,
  `stock_confirmaciones_bajada` = 1). Se cuenta y cada **8** bajadas avisa
  **«cerca de vaciarse»** (`stock_bajadas_aviso`).
- **Posible agotado**: cambio **≥ 0,90**, tres veces seguidas
  (`stock_umbral`, `stock_confirmaciones`). Se cierra solo si el producto
  vuelve a parecerse a su foto.
- Cada bajada y cada alerta deja su foto **antes | ahora | quién** y un
  `.json` con el cambio en `output/anaqueles/<cámara>/stock_alertas/`.
- **Los umbrales no están calibrados** con una balda vaciada de verdad.
- **Las alertas activas viven en memoria**: un reinicio las borra; las
  fotos se quedan.

---

## 4. Los cortes de vídeo de iVMS

iVMS pierde a menudo el vídeo de la cámara 3. Mientras, manda **una imagen
gris fija** con el texto «Reconnecting… Playing video failed (Error code:
OpenSDK StringError[105])». Del 12-oct a las 17h al 13-oct a las 13h hubo
118 cortes, de una mediana de 29 s. El 13-oct por la tarde estuvo así horas.

- **`cortes_video.sin_video`** (SIEMPRE activo): reconoce la imagen lisa
  (desviación del centro < 10; el vídeo real da 47-59 y la gris, 3-6). Con
  ella, **el stock no mide ni toma fotos**, no se guarda la **foto de
  referencia del anaquel** ni el **fondo del mapa de calor**. Antes del
  13-oct produjo dos bajadas falsas que confirmaron un agarre.
- **Pausa del seguimiento** (OPCIONAL, apagada; `ELDE_CONGELAR_CORTES=1`):
  si llega la misma imagen exacta 0,4 s o más, el servidor responde sin
  procesar y no corre el reloj de nadie, así que nadie pierde su número por
  el corte.
- Cuando no hay imagen, **nada se cuenta**: hay que revisar el problema en
  iVMS o en el grabador de la tienda.

---

## 5. Dashboard e informe

### Dashboard (`/dashboards/tienda/`)

Secciones: Resumen, Curva, Anaqueles, **Stock** (bajadas, «cerca de
vaciarse» y posibles agotados con su foto ampliable), Personal, Quién
visita, Mapas de calor, Pasillos, Capturas y Cámaras. Arriba se elige el
**periodo** y la **cámara**.

### Informe PDF: «Informe del pasillo»

El botón **«Exportar PDF»** genera el informe del **periodo y la cámara**
que se están viendo (`informe_pasillo.py`):

- cabecera, resumen en cuatro tarjetas, tráfico, demografía, afluencia por
  hora, anaqueles (agarres por producto y género), **bajadas de stock**
  (antes | ahora, **sin la foto del cliente**), mapas de calor por anaquel
  con sus productos, permanencia y mapa general;
- se arma como página HTML y la **imprime Edge o Chrome** sin ventana, en
  5-6 s;
- **sin navegador** (por ejemplo, el servidor de Linux) o si falla, sale el
  informe clásico de un día (`informe_pdf.py`);
- **no muestra** las fotos de stock medidas sobre la imagen gris de un
  corte, ni cuenta los agarres que solo confirmaron ellas, y lo dice en una
  nota;
- se salta las horas del mapa de calor pintadas sobre la imagen gris.

API: `/api/informes/tienda.pdf?desde=AAAA-MM-DD&hasta=AAAA-MM-DD&camara=...`
(`fecha=` sigue valiendo para un día; `formato=clasico` pide el de antes) y
`/api/informes/tienda.html` (la misma página).

---

## 6. Interruptores y ajustes

| Interruptor | Dónde | Por defecto | Qué hace |
|---|---|---|---|
| `PERSON_MIN_HEIGHT_PX` | perfil de cada cámara (`server/config/camera_profiles/`) | 0 global; 50 vídeo; 150 cámara 3 | Alto mínimo para contar |
| `REID_CONSERVAR_IDENTIDAD` / `ELDE_REID_CONSERVAR=0` | `config.py` / variable | encendido | Un número con identidad no estrena otra |
| `DEMO_B_CARA_PROPIA` / `ELDE_CARA_PROPIA=0` | `config.py` / variable | encendido | Solo la cara de la cabeza de esa persona |
| `DEMO_CLASIFICADOR` / `ELDE_CLASIFICADOR=clasico` | `config.py` / variable | `b` | Clasificador de género |
| `CONGELAR_EN_CORTES` / `ELDE_CONGELAR_CORTES=1` | `config.py` / variable | apagado | Pausa del seguimiento en los cortes de iVMS |
| `agarre_requiere_bajada` | `anaqueles.py` / YAML | encendido | El agarre se confirma con una bajada |
| `visita_por_muneca` | `anaqueles.py` / YAML | encendido | La visita es la muñeca dentro |
| `stock_activo` | YAML | encendido | Vigilancia de stock |
| `ELDE_MODO_F=0` | cliente | encendido | Envío inmediato de cada imagen |
| `ELDE_DETECTOR_PREPARAR_EN_GPU=0`, `ELDE_POSE_PREPARAR_EN_GPU=0` | servidor | encendidos | Volver a preparar en la CPU |
| `ELDE_ANAQUELES_GRABAR=<ruta>` | servidor | apagado | Graba imágenes y seguimientos para reproducirlos (fotos de clientes) |
| `ELDE_ANAQUELES_DIAG=1`, `ELDE_TIEMPOS_ETAPAS=<ruta>` | servidor | apagados | Diagnóstico del agarre y cronómetro por etapas |

La configuración de Anaqueles se lee en tres capas: `_DEFECTOS` del código,
luego `anaqueles_config.yaml` y luego su bloque por cámara. Solo vale una
clave que exista en `_DEFECTOS`. Un cambio en el YAML vale al reconectar el
cliente o reiniciar el servidor.

---

## 7. Dónde están los datos

Todo bajo `server/output/` (**fuera de git**; son datos y fotos de clientes):

| Qué | Dónde |
|---|---|
| Base | `tienda.db` |
| Fichas y fotos | `captures/persons/` |
| Eventos por persona | `analytics_log_<cámara>.jsonl` |
| Anaqueles | `anaqueles/<cámara>/`: `<fecha>.json`, `eventos/`, `fotos_ref/`, `stock_ref/`, `stock_alertas/` |
| Mapas de calor | `heatmap/`: `bg/`, `history/`, `state/` |
| Log | `server/logs/server.log` (rota a 5 MB, 3 copias) y `logs/consola_*.log` |

Tampoco se suben `informes/` (informes locales con fotos) ni `CLAUDE.md`
(notas locales de trabajo).

### Líneas del log que conviene vigilar

| Línea | Qué dice |
|---|---|
| `FPS [cámara] recibidos=… procesados=… proceso=…` | Cada 30 s: velocidad y tiempo del servidor por imagen. `KB/frame` de ~40 = imagen gris de iVMS |
| `FONDO`, `MEMORIA`, `IGNORAR` | Cada minuto: lo que quitó cada filtro |
| `AGARRE #n en … confirmado por …` | Un agarre que cuenta |
| `gesto de agarre … SIN CONFIRMAR` | Un gesto sin bajada |
| `BAJADA n de …`, `POSIBLE AGOTADO` | Stock |
| `NEW persona=…`, `Re-IDENT …`, `REID conserva …` | Identidades |

---

## 8. Arrancar, parar y probar

- **Servidor**, desde `server\`:
  `.\venv\Scripts\python.exe -u iniciar_servidor_headless.py 9000`. Desde
  otra carpeta, las capturas y mapas acaban en otro sitio. La primera
  imagen tarda ~20 s, y la primera persona junto a un estante ~1 min más
  (carga de la pose).
- **Parar en orden** (para que guarde el mapa de calor y el día de
  anaqueles): Ctrl+C a su consola. Matarlo sin más no guarda.
- **Cliente**, como Administrador (iVMS corre elevado), desde
  `clients\tienda`: `.\venv\Scripts\python.exe src\main.py`.
- **Pruebas del servidor**: `server\venv` no trae pytest; se instala aparte
  (`pip install --target <carpeta> pytest`) y se corre desde `server\` con
  `PYTHONPATH` apuntando a esa carpeta y `ELDE_TIENDA_DB` a una base de
  prueba con **ruta absoluta**, para no tocar la real. El 14-oct: **1029
  pasan**. En `packages/elde_core` fallan siempre 72 pruebas que buscan
  otros clientes que no están en este repositorio.

---

## 9. Límites conocidos

- **Cortes de iVMS**: sin vídeo no se cuenta nada. Es un problema de la
  cámara o del grabador, no del código.
- **Agacharse en la balda de abajo**: queda en el borde inferior de la
  imagen y la persona desaparece 4-10 s, así que al levantarse recibe **otro
  número**. Si se le ve la cara, recupera su identidad; de espaldas, no. Se
  probó un «re-enganche» de números (recuperaba pérdidas de menos de 2 s) y
  se descartó.
- **Dos cajas sobre una persona sola**: a veces el detector la ve dos veces
  (cuerpo entero y medio cuerpo). Quitar la caja de dentro también quitaba
  a personas reales que estaban juntas: descartado.
- **Caras pequeñas** (mediana ~26 px), porque la cámara mira desde arriba y
  la gente mira los productos: muchas personas no se identifican y algunas
  identidades juntan a varias. La «cara propia» lo reduce. Lo que más
  ayudaría es más resolución (leer el vídeo de la cámara directamente, no
  la ventana de iVMS) o una cámara de frente.
- **Producto del agarre**: lo da la bajada de stock, que es fiable si la
  foto de referencia es buena. Los umbrales de stock no están calibrados.
- **Personal de la tienda**: hay alta de personal (`personal_api.py`), pero
  no se ha comprobado hasta dónde llega su exclusión. Los empleados que
  reponen se cuelan en visitas, entradas y agarres.
- **Seguimientos, fichas y personas no son lo mismo**: una persona puede
  dejar varias fichas si pierde su número. Las cifras de personas son una
  aproximación por arriba.
