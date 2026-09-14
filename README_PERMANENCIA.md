# Permanencia — cronómetro de personas en un área o en una puerta

Modo `PermanenciaArea` del servidor + panel del cliente **Detecciones IA**.
Responde, por persona: *entró a esta hora, esta es su foto, estuvo este tiempo,
salió a esta otra y esta es la foto de la salida* — y lo avisa por WhatsApp y
por Jarvis.

> **Estado honesto:** 450 pruebas en verde entre servidor, núcleo y los cuatro
> clientes, pero **nada validado contra vídeo real todavía**. Los umbrales de la
> puerta (`frames_confirmacion`, `segundos_ausencia`) están puestos con criterio,
> no medidos en tu escena. Ver [Limitaciones](#limitaciones-conocidas).

---

## 1. Los dos escenarios

El área vigilada es siempre el **ROI amarillo que dibujas en el recuadro**. No
hay que calibrar nada aparte. Lo que cambia es cómo se interpreta:

| `modo` | El ROI es… | Mientras dura la permanencia… |
|---|---|---|
| `area` | la zona vigilada | la persona **está a la vista** |
| `puerta` | el vano de la puerta | la persona **no se ve**: se metió |

### Por qué la puerta no se podía resolver como el área

`TiendasProcess` ya mantiene una máquina DENTRO/FUERA por persona, con
histéresis. En un área eso basta. En una puerta diría «salió» en el instante
en que la persona cruza y desaparece del encuadre, y **el cronómetro marcaría
cero siempre**.

El sentido del cruce se deduce del ciclo de vida del seguimiento:

```
  nace FUERA del vano  ─►  sus últimas apariciones son DENTRO  ─►  desaparece
  ════════════════════════════════════ ENTRÓ ═════════════════════════════════

  nace DENTRO del vano  ─►  se aleja hasta salir del vano
  ════════════════════════════════════ SALIÓ ═════════════════════════════════
```

Es simétrico y **no necesita saber qué lado es «adentro»**, que es justo lo que
la cámara no puede ver.

**Lo que se niega a decidir:** quien se queda parado en el vano sin llegar a
salir de él nació dentro (parecería salida) y muere dentro (parecería entrada).
Las dos lecturas son igual de defendibles, así que no se reporta ninguna.

### La flecha: de inferir a medir

En el recuadro, con `PermanenciaArea` activo, el menú de ROI ofrece **«Flecha
de dirección»**. Se arrastra **una sola flecha**, de fuera hacia dentro:

```
   ┌──────────────────────────────┐
   │       ░░░  PUERTA  ░░░       │
   │   ┌──────────────────────┐   │
   │   │   ●━━━━━━━━━━━━━━▶   │   │
   │   │   ▲              ▲   │   │
   │   └───┼──────────────┼───┘   │
   │    SALIDA        ENTRADA     │
   │    (rojo)         (verde)    │
   └──────────────────────────────┘
```

Una puerta tiene **un umbral**: si marcas el sentido de entrada, el de salida
es el contrario por definición. Dos flechas independientes no aportan
información nueva y sí un estado incoherente posible —dibujarlas casi
paralelas—, en el que el sistema tendría que elegir una y contradecirte sin
avisar.

Con la flecha, la dirección deja de deducirse de la ausencia y pasa a medirse:
la proyección del desplazamiento de la persona sobre el vector. La flecha
**confirma y desempata**, no sustituye:

| Situación | Sin flecha | Con flecha |
|---|---|---|
| Muere en el vano viniendo de fuera | entrada | entrada **confirmada** |
| Muere en el vano pero venía alejándose | entrada (falsa) | **descartada** |
| Nace en el vano y se aleja | salida | salida **confirmada** |
| Nace en el vano y se adentra | salida (falsa) | **descartada** |
| Nace y muere en el vano, con recorrido | *se calla* | **se decide** |
| Nace y muere en el vano, sin recorrido | *se calla* | *se calla* (o al árbitro) |

El umbral va en **alturas de la propia persona**, no en píxeles: el mismo
ajuste vale para 720p y para 4K, y para quien cruza al fondo o en primer plano.

Cada alerta dice cómo se decidió, en `direccion_por`: `flecha` (una medida),
`ciclo_de_vida` (una inferencia) o `vlm` (una opinión). No se disfrazan entre
sí.

Si no dibujas la flecha, no se rompe nada: se queda la heurística de siempre.

### El umbral: cuando la trayectoria se curva

La flecha se dibuja **a lo largo** del sentido de caminata y proyecta sobre ella
el desplazamiento acumulado. Eso pide medio cuerpo de avance en línea recta, y
hay una cámara donde no lo hay: el gran angular apuntando a una esquina (la
puerta de Kitchen Productions, 10-ago-2026). Ahí la gente entra **curvándose**
alrededor del muro: parte del recorrido va en contra de la flecha aunque estén
entrando, y el tracker suele perderlos en el marco antes de acumular el umbral.

Una puerta **es** una línea: su umbral en el suelo. Dibujarlo cambia la
pregunta de «¿cuánto avanzó y hacia dónde?» a «¿de qué lado estaba y de qué
lado está?» —binaria, local, y resuelta *en* el cruce sin importar hacia dónde
curve la trayectoria después.

Ese umbral se puede dibujar con **dos utensilios que producen el mismo dato**:

```
      ─ Línea de cruce              ⇅ Puerta direccional
   ┌──────────────────────┐      ┌──────────────────────┐
   │  ░░░  PUERTA  ░░░    │      │  ░░░  PUERTA  ░░░    │
   │      ┌ FUERA ┐       │      │        ▲ SALIDA      │
   │  ●━━━━━━━━━━━━━━●    │      │  ●━━━━━┿━━━━━━━━●    │
   │      └ DENTRO ┘      │      │        ▼ ENTRADA     │
   │          ↓           │      │                      │
   └──────────────────────┘      └──────────────────────┘
     lados: DENTRO / FUERA         sentidos: ENTRADA / SALIDA
```

Son **la misma geometría**: dos puntos sobre el vano, con DENTRO a la derecha
caminando de P1 a P2. Viajan por el mismo campo (`linea_cruce`) y los lee el
mismo detector (`permanencia_linea.py`). Lo único que cambia es el rótulo.

Que haya dos no es indecisión, es que **rotulan en dos idiomas distintos** y el
del operador es el segundo. Quien calibra una puerta piensa «por aquí entran»,
no «este lado es el interior»; obligarle a traducir esa frase a un lado es
donde se cuela el error más caro y más frecuente de este modo —el umbral al
revés, que cuenta las entradas como salidas sin que nada falle ni avise. La
puerta direccional pone la flecha doble encima: cabeza verde ENTRADA, roja
SALIDA, y se comprueba de un vistazo desde el otro lado de la sala.

Es también la razón de que la puerta **no estrene detector**. Su eje es
exactamente la normal de la línea, así que hereda `permanencia_linea.py` entero
—histéresis, desempates de vano, cooldown— en vez de duplicar esos tres
mecanismos y, con ellos, los errores que costó cazar.

Doble clic sobre cualquiera de los dos lo invierte sin moverlo del sitio: es el
arreglo de un segundo cuando quedó al revés. La alerta lleva
`umbral_dibujado_con` (`linea` o `puerta`) para que el registro conteste qué
tenía delante quien lo dibujó.

**Precedencia**: si llegan la línea (o la puerta) *y* la flecha, gana el
umbral. Y con el modo `area`, dibujar cualquiera de los dos activa solo la
lectura de cruce para esa cámara; borrarlo devuelve el área clásica.

Los **cuatro clientes** ofrecen la puerta direccional (botón «⇅ Puerta» en la
barra del recuadro y entrada en el menú de zonas). La flecha y la línea siguen
solo en Detecciones IA, que es donde se calibraron.

Para que el botón no fuera un adorno, **Tienda y Perimetrales estrenan también
`PermanenciaArea` en su selector del pie** (7-sep-2026): es el único modo que
lee el umbral, y un utensilio que se dibuja pero que ningún modo mira deja al
operador calibrando algo que no clasifica un solo paso, sin forma de
enterarse. Va el último de la lista y **no cambia el modo por defecto de
ninguna instalación**: quien no abra el desplegable no nota nada. Amazonas y
Detecciones ya lo ofrecían.

Un aviso que vale para los cuatro: la puerta se dibuja **por cámara** y se
guarda en los ajustes de esa cámara, igual que el ROI. Cambiar el canal que
muestra la ventana de iVMS-4200 no mueve la puerta, así que un umbral
calibrado sobre otra escena seguirá ahí clasificando pasos que no existen.

### El árbitro: dónde SÍ entra un modelo de visión

Para el residuo —nace y muere en el vano sin recorrido suficiente— hay un
árbitro opcional (`arbitro_vlm: true`). **No decide la dirección**: eso es una
proyección, cuesta nanosegundos y es reproducible; pedírsela a un VLM sería
cambiar una medida por una opinión que además tarda decenas de segundos en
llegar cuando el cruce dura uno.

Lo que hace es mirar **tres momentos del mismo track en orden temporal**, con
la persona recuadrada, y responder qué hizo entre ellos. El movimiento sí está
en esa tira; en un fotograma suelto, no.

Corre en su propio hilo con cola acotada: el vídeo nunca lo espera, y lo que no
cabe se descarta y se cuenta. El veredicto abre o cierra la estancia con la
hora **original** del cruce, no con la de la respuesta. Apagado por defecto:
sin activarlo el modelo ni se carga.

---

## 2. Cómo se reconoce a quien sale

Por orden. La alerta **siempre dice cuál se usó**, en `identidad_por`:

| Vía | Cuándo | Fiabilidad |
|---|---|---|
| `rostro` | el `persistent_id` de la galería facial coincide | alta — esa galería es persistente en disco y sobrevive a las horas que la persona pasó dentro |
| `cuerpo` | embedding OSNet guardado al entrar vs. el de quien sale | media — umbral alto (`0.86`) porque la ropa discrimina menos que la cara |
| `zona` | hay **exactamente una** estancia abierta | alta por descarte: quien sale solo puede ser esa persona |
| `no atribuida` | ninguna de las anteriores | **se avisa igual, sin tiempo de permanencia**. No se inventa una duración |

El libro de estancias abiertas se **persiste en disco por cámara**. Un reinicio
del servidor a media tarde no vacía el almacén.

---

## 3. Arquitectura

Cuatro módulos, uno por responsabilidad. Ninguno hace inferencia: reciben lo
que el pipeline ya calculó y **deciden**. Por eso toda la lógica —que es donde
nacen los errores de conteo— se prueba sin GPU y en milisegundos.

```
                    ┌────────────────────────────────┐
  frame ─► RF-DETR ─► tiendas.py (ex person_amazona) │  (detección, ByteTrack,
                    │  + Re-ID facial y corporal     │   demografía, zonas)
                    └───────────────┬────────────────┘
                                    │ active_tracks + persistent_id + _zone_state
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  permanencia_area.py      orquesta; modo area | puerta          │
   ├─────────────────────────────────────────────────────────────────┤
   │  permanencia_puerta.py    ¿cruzó?, ¿hacia dónde?                │
   │  permanencia_estancias.py quién está dentro (+ disco, + Re-ID)  │
   │  permanencia_evidencia.py las fotos (recorte, banner, base64)   │
   └───────────────┬─────────────────────────────────────────────────┘
                   │
       ┌───────────┴────────────┬──────────────────────┐
       ▼                        ▼                      ▼
  metadata['alerts']    metadata['permanencia']   detections[].label
   WhatsApp + Jarvis     panel del cliente        cronómetro en el vídeo
```

| Archivo | Líneas | Qué decide |
|---|---|---|
| `server/src/analityc/core/permanencia_area.py` | ~800 | el ciclo entrar → cronometrar → salir, y las alertas |
| `server/src/analityc/core/permanencia_puerta.py` | ~380 | si un track cruzó el vano y en qué sentido |
| `server/src/analityc/core/permanencia_estancias.py` | ~330 | quién sigue dentro, con qué respaldo de identidad |
| `server/src/analityc/core/permanencia_evidencia.py` | ~300 | recorte, banner, comparativa, disco y retención |
| `server/src/analityc/core/permanencia_arbitro.py` | ~340 | el VLM que arbitra lo que la geometría no resuelve |
| `server/src/analityc/core/permanencia_cruce.py` | ~230 | el paso visible del ROI, clasificado por la flecha |
| `server/src/analityc/core/permanencia_linea.py` | ~420 | de qué lado a qué lado se pasó el umbral |
| `packages/elde_core/elde_core/ui/flecha_direccion.py` | ~200 | la flecha «hacia adentro»: modelo, pintado y vector |
| `packages/elde_core/elde_core/ui/linea_cruce.py` | ~130 | el umbral rotulado DENTRO / FUERA |
| `packages/elde_core/elde_core/ui/puerta_direccional.py` | ~180 | el mismo umbral rotulado ENTRADA / SALIDA |
| `packages/elde_core/elde_core/ui/render_box_puerta.py` | ~200 | el cableado de la puerta, común a los cuatro clientes |
| `clients/detecciones/src/gui/components/permanencia_panel.py` | ~460 | el panel en vivo y el guardado local de fotos |

---

## 4. Las fotos

Tres momentos, tres imágenes:

- **Entrada** — banner verde con la hora y la identidad.
- **Salida** — banner rojo con el tiempo total.
- **Comparativa** — las dos anteriores lado a lado. Es la que viaja en la
  alerta de salida, porque responde de un vistazo la pregunta que importa:
  *¿es la misma persona que entró?*

Cada una se guarda **dos veces, a propósito**:

| Dónde | Para qué |
|---|---|
| `server/output/permanencia/<fecha>/<cámara>/` | respaldo consultable después; no depende de que el cliente estuviera conectado |
| `clients/detecciones/capturas/permanencia/<fecha>/` | lo que el operador tiene a mano en su propia máquina, con un `.json` al lado |

Las dos carpetas se purgan solas a los `retencion_dias` (30 por defecto). Una
cámara de puerta llena el disco en semanas sin eso.

---

## 5. Alertas

Salen por la cola de siempre (`metadata['alerts']`), así que **WhatsApp y
Jarvis no necesitaron configuración nueva**.

| `tipo_permanencia` | Cuándo | Lleva tiempo |
|---|---|---|
| `entrada` | se confirma la entrada | no |
| `salida` | se confirma la salida | sí, total |
| `permanencia` | supera `umbral_permanencia_s` sin irse | sí, acumulado |
| `sin_salida` | pasa el TTL sin que se vea salir | sí, **como cota mínima** |
| `no_atribuida` | salió alguien sin emparejar | **no**, y lo dice |

### El arreglo que hizo falta: deduplicar por PERSONA

WhatsApp y Jarvis agrupan por `(cámara, tipo de evento)` durante 30 s. Con esa
regla, **dos personas entrando por la misma puerta con pocos segundos de
diferencia daban un solo aviso**, y el tiempo de la segunda no llegaba a nadie.

Se añadió un campo opcional `dedup_clave` que el pipeline puede fijar; los dos
reenviadores lo respetan si viene. Permanencia lo manda con la persona dentro
(`perm|<cámara>|<tipo>|<id>`). **Ningún otro pipeline cambia de
comportamiento**: sin ese campo, la política de siempre.

> Queda un tope que ninguna clave puede saltarse:
> `WHATSAPP_INTERVALO_GLOBAL_SEG`, el límite global del emisor. En una ráfaga
> de entradas simultáneas, WhatsApp seguirá espaciando los envíos. Jarvis y el
> panel del cliente sí las reciben todas.

---

## 6. El cliente

Panel acoplado bajo el de Alertas IA, en **Detecciones IA**:

```
┌─────────────────────────────────────┐
│  ⏱  Permanencia                     │
├──────────────────┬──────────────────┤
│ DENTRO AHORA     │ VISITAS COMPLET. │
│       3          │       47         │
│ 2 cámaras·puerta │ media 12:30      │
├──────────────────┴──────────────────┤
│  AHORA DENTRO                       │
│ ▎7d4f1a2b  desde 14:02 · H · 25-32  │
│                             8:14    │
│ ▎prov:1187 desde 14:09              │
│            fuera de cámara   1:07   │
├─────────────────────────────────────┤
│  ÚLTIMOS SUCESOS                    │
│ [foto] Salida por la puerta         │
│        Salió tras 1 h 12 min…       │
│        14:02 · CAM4 · ident. cuerpo │
└─────────────────────────────────────┘
```

Tres decisiones que no son obvias:

1. **El reloj corre en el cliente.** Si la cámara baja a 3 FPS los segundos se
   congelarían justo cuando más se miran.
2. **No resta marcas de tiempo del servidor.** Usa los segundos que el
   servidor ya calculó más lo transcurrido *aquí* (`time.monotonic()`). Los dos
   relojes no están sincronizados; restarlos daría permanencias de años.
3. **Una cámara callada más de 20 s deja de sumar.** Enseñar el aforo de hace
   media hora como si fuera de ahora es peor que no enseñar nada.

Además, en modo directo —el de este cliente— **la caja de cada persona lleva su
cronómetro pegado a la etiqueta**, sobre el vídeo.

---

## 7. Puesta en marcha

1. **Arranca el servidor** y el cliente `Detecciones IA` (`SELECTOR.bat`).
2. En el recuadro de la cámara, elige el modo **`PermanenciaArea`**.
3. **Dibuja el ROI**: la zona vigilada, o el **vano de la puerta**.
4. Si es una puerta, enciende **«Flecha de dirección»** en el menú de ROI del
   recuadro y arrástrala de fuera hacia dentro. Es opcional, pero es lo que
   convierte la dirección en una medida.
5. Y dilo en `server/config/permanencia_config.yaml`:

   ```yaml
   camaras:
     "Puerta Almacen":          # el nombre EXACTO que manda el cliente
       modo: puerta
       umbral_permanencia_s: 3600
       estancia_ttl_horas: 12
   ```

6. Enciende **«Enviar por WhatsApp»** y/o **«Enviar a Jarvis»** en el pie.

Todos los umbrales están en ese YAML, comentados uno a uno. Si lo borras, el
modo sigue funcionando con los mismos valores por defecto.

---

## 8. Limitaciones conocidas

1. **Sin validar contra vídeo real.** Es la limitación que manda. Los umbrales
   de la puerta son criterio, no medida. Antes de fiarse de una cifra hay que
   contrastar una franja horaria con conteo manual.

2. **El Re-ID corporal caduca con la ropa.** El respaldo por cuerpo vale para
   la misma jornada. Alguien que entra un lunes y sale el martes solo se
   empareja por rostro; si tampoco lo hay, sale como `no atribuida`.

3. **La identidad provisional no sobrevive al día.** Quien entra sin que se le
   vea la cara recibe un `prov:<track>` y su respaldo es solo el cuerpo.

4. **El ROI que cubre todo el pasillo pierde entradas.** El detector exige
   haber visto a la persona *fuera* del vano. Si el polígono abarca toda la
   escena eso no pasa nunca: `puerta.exigir_fuera_de_zona: false`. **Con la
   flecha dibujada esto deja de importar**: la dirección la decide el
   desplazamiento.

4b. **La flecha es plana y la escena no.** Se proyecta sobre el plano de la
   imagen, así que en una puerta vista casi de frente —donde entrar y salir
   se ven como el mismo movimiento diminuto— la proyección será pequeña y
   muchos cruces caerán por debajo del umbral. Ahí conviene bajar
   `desplazamiento_min_altura`, o encender el árbitro, o recolocar la cámara.

5. **Dos personas cruzando el vano a la vez, abrazadas o muy juntas**, pueden
   compartir un solo track. El sistema contará una.

6. **`estancia_ttl_horas` cierra por reloj, no por evidencia.** Lo dice en la
   alerta (`duracion_es_minima`), pero esa duración es una cota inferior, no un
   dato.

---

## 9. Pruebas

```bat
server\venv\Scripts\python.exe server\tests\test_permanencia_area.py
server\venv\Scripts\python.exe server\tests\test_permanencia_puerta.py
server\venv\Scripts\python.exe server\tests\test_permanencia_estancias.py
server\venv\Scripts\python.exe server\tests\test_permanencia_evidencia.py
server\venv\Scripts\python.exe server\tests\test_permanencia_arbitro.py
clients\detecciones\venv\Scripts\python.exe packages\elde_core\tests\test_flecha_direccion.py
clients\detecciones\venv\Scripts\python.exe clients\detecciones\test\test_permanencia_panel.py
```

No hace falta pytest: cada archivo trae su propio ejecutor.

El árbitro se prueba **sin GPU y sin modelo**: se sustituye el router
multimodal por un doble. Lo que se comprueba es la política —cola acotada,
respuestas ilegibles descartadas, «ninguno» que no se convierte en un cruce,
y que un fallo del modelo no mate el hilo—, que es donde están los errores
caros.

---

## 10. Un fallo que apareció por el camino

`app.py` no procesa el frame con el objeto que construye `_build_processor`,
sino con el que devuelve `get_camera_processor(camera_id)`. Y ese método
instanciaba **`TiendasProcess` con la clase incrustada** en vez de `type(self)`.

Consecuencia: en cualquier modo derivado (`PermanenciaArea`, **y también
`AmazonasStand`**) el procesador por cámara salía siendo el padre. Se veía el
vídeo y las detecciones con normalidad, y **no se emitía un solo evento del
modo**. Un fallo silencioso: ningún error en el log, solo ausencia de datos.

### Alcance real, comprobado

`server/output/stand/stand.db` existe y está **vacía: cero sesiones, cero
eventos**. El stand nunca llegó a medir nada, así que no hay cifras contaminadas
que purgar — pero tampoco hay nada validado. **Hay que probarlo desde cero.**

### La segunda trampa del mismo sitio: el estado por cámara

`TiendasProcess` multiplexa varias cámaras sobre **un mismo objeto**.
`_push_state` mueve el estado de esa cámara a los atributos de `self` al
entrar en `process_frame`, y `_pop_state` **restaura los de antes** al salir.
En esa lista están `active_tracks` y `_zone_state`.

Todo lo que trabaje *después* de `super().process_frame()` —que es donde vive
este módulo, y también `AmazonasStand._medir_stand`— leía por tanto dos
diccionarios vacíos que no eran de nadie. El cronómetro veía cero personas
aunque el pipeline acabara de seguir a una. Sin excepción y sin log: silencio.

**Lo que lo delató:** `_zone_count` NO está en la lista de intercambio, así que
sobrevive al pop. Ver `aforo=1` mientras `_zone_state` estaba vacío es
imposible salvo que existan dos estados distintos.

El arreglo lee `camera_states[camera_id]` y reapunta a esos **mismos**
diccionarios antes de trabajar. Y lo hace a pelo, sin pasar por
`_get_cam_state()`: ese método no es un lector inocente —crea el estado si
falta y, si le falta una clave, lo migra reponiendo `_zone_state` a vacío—.

Es la razón de que `stand.db` siguiera vacía después de arreglar
`get_camera_processor`: eran **dos** fallos encadenados en el mismo punto.

### El mismo agujero, en el cierre

El `finally` de la desconexión llamaba a `cleanup()` sobre el procesador de
arriba, que es justo el que nunca ve un frame. El que tiene el estado vivo es
el de la cámara. Ahora `_cerrar_procesadores()` recorre los dos niveles.

Con eso, `PermanenciaArea.cerrar_permanencia()` vuelca el libro de estancias al
desconectar. Sin ese volcado, apagar el cliente justo después de que alguien
entrara por la puerta perdía su estancia (el guardado normal va limitado a uno
cada 5 s), y al reconectar su salida habría salido como «no atribuida».

### Lo que NO se cerró, a propósito

`AmazonasStand.cerrar_sesion_stand()` **no** se enganchó a la desconexión, y su
docstring decía en falso que el servidor la llamaba. Cerrar la jornada en cada
desconexión partiría el día en trozos cada vez que el cliente reconecta por un
corte de red — que es el fallo de las cuatro sesiones colgadas que ya se
arregló con `sesion_abierta_de`. Queda como gancho para cerrar la jornada a
propósito, cuando exista un disparador explícito.

Tres pruebas vigilan esto: la línea de `type(self)`, que el cierre recorra los
procesadores por cámara, y que el libro sobreviva a la desconexión.
