/* dashboards/stand/app.js — dashboard analitico del stand de degustacion
 * (restyle 14-ago-2026 sobre el patron de dashboards/tienda).
 *
 * Lee de /api/stand con rutas RELATIVAS, igual que el resto de dashboards
 * hace con /api/v1: ni host ni puerto escritos, porque la sirve el mismo
 * servidor que responde la API.
 *
 * `shared/api.js` NO se toca: su helper `API.leer` apunta a /api/v1 y lo
 * comparten cuatro dashboards. Aqui hay un helper propio de cuatro lineas
 * para /api/stand — mas barato y mas seguro que cambiar codigo compartido.
 *
 * ## Dos reglas que gobiernan como se pinta todo esto
 *
 * 1. `null` NO es 0. Una tasa desconocida se muestra como raya. Confundirlas
 *    hace leer «el stand convierte al 0%» donde en realidad pone «todavia no
 *    ha pasado nadie».
 * 2. El panel de calidad del dato se pinta SIEMPRE, aunque este limpio. Su
 *    ausencia haria pensar que las cifras son mas firmes de lo que son.
 *
 * ## Mapa de calor: propio del stand, no el comun
 *
 * Los mapas de esta pagina NO salen de /api/v1/heatmaps (PNGs por camara)
 * sino de /api/stand/{sesion}/heatmap (matriz numerica de 3 capas). Por eso
 * aqui no aplica pintarMapasTodos(): se pintan con dibujarCalor() de
 * shared/util.js y un clic los amplia en el lightbox con cifras, igual que
 * las rejillas de anaqueles en tienda.
 */
'use strict';

/* fmt, abrirLightbox y dibujarCalor vienen de shared/util.js (cargado antes
 * que este archivo). */

const StandAPI = {
  async leer(ruta) {
    const r = await fetch(`/api/stand/${ruta}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`/api/stand/${ruta} respondio ${r.status}`);
    return r.json();
  },
};

/* ── Formato ─────────────────────────────────────────────────────────── */
const pct = v => (v === null || v === undefined) ? '—' : `${(v * 100).toFixed(1)}%`;
const num = v => fmt(v);           // localizado; null/undefined siguen siendo raya
const seg = v => (v === null || v === undefined) ? '—' : `${Number(v).toFixed(1)} s`;
const esc = s => UI.esc(s ?? '—');

const ETIQUETA_CLASIF = {
  paso_de_largo: 'pasó de largo',
  se_detuvo_sin_atencion: 'se detuvo, sin atención',
  atendido_sin_degustar: 'atendido, no degustó',
  degusto: 'degustó',
};

const ETIQUETA_MAPA = {
  permanencia: 'permanencia (dónde se quedan)',
  trafico: 'tráfico (por dónde pasan)',
  detenciones: 'detenciones (dónde se anclan)',
};

let SESION = null;
let MAPA_TIPO = 'permanencia';

/* ── 1 · KPIs + comparativa con sesiones anteriores ──────────────────── */
function pintarKpis(k, promedio) {
  UI.kpis({
    pasaron: num(k.pasaron),
    se_detuvieron: num(k.se_detuvieron),
    atendidos: num(k.atendidos),
    degustaron: num(k.degustaron),
    tasa_captacion: pct(k.tasa_captacion),
    tasa_atencion: pct(k.tasa_atencion),
    tasa_degustacion: pct(k.tasa_degustacion),
    tasa_global: pct(k.tasa_global),
  });

  for (const el of document.querySelectorAll('[data-cmp]')) {
    const clave = el.dataset.cmp;
    const actual = k[clave];
    const media = promedio ? promedio[clave] : null;
    if (actual === null || actual === undefined ||
        media === null || media === undefined || media === 0) {
      el.textContent = '';
      el.className = 'comparativa';
      continue;
    }
    const delta = (actual - media) / Math.abs(media);
    el.textContent = `${delta >= 0 ? '▲' : '▼'} ${Math.abs(delta * 100).toFixed(0)}% vs media`;
    el.className = `comparativa ${delta >= 0 ? 'sube' : 'baja'}`;
  }
}

/** Media de las sesiones ANTERIORES a la mostrada (maximo 10, por coste). */
async function promedioAnteriores(sesiones, actualId) {
  const previas = sesiones.filter(s => s.id < actualId).slice(0, 10);
  if (!previas.length) return null;
  const claves = ['pasaron', 'se_detuvieron', 'atendidos', 'degustaron',
                  'tasa_captacion', 'tasa_atencion', 'tasa_degustacion',
                  'tasa_global'];
  const acum = {}, cuenta = {};
  for (const s of previas) {
    let k;
    try { k = await StandAPI.leer(`${s.id}/kpis`); } catch { continue; }
    for (const c of claves) {
      if (k[c] === null || k[c] === undefined) continue;
      acum[c] = (acum[c] || 0) + k[c];
      cuenta[c] = (cuenta[c] || 0) + 1;
    }
  }
  const media = {};
  for (const c of claves) media[c] = cuenta[c] ? acum[c] / cuenta[c] : null;
  return media;
}

/* ── 2 · Embudo ──────────────────────────────────────────────────────── */
function pintarEmbudo(e) {
  const cont = document.getElementById('embudo');
  const escalones = e.escalones || [];
  if (!escalones.length) { cont.innerHTML = '<div class="vacio">sin datos todavía</div>'; return; }
  const tope = escalones[0].valor || 1;

  cont.innerHTML = escalones.map(s => {
    const ancho = tope ? Math.max(2, 100 * s.valor / tope) : 2;
    const peor = e.mayor_perdida === s.clave;
    const caida = s.caida_absoluta === null || s.caida_absoluta === undefined
      ? '' : `−${fmt(s.caida_absoluta)} (${pct(s.caida_pct)})`;
    return `<div class="escalon ${peor ? 'peor' : ''}">
      <span>${esc(s.etiqueta)}</span>
      <div class="pista"><div class="relleno" style="width:${ancho.toFixed(1)}%">${fmt(s.valor)}</div></div>
      <span class="caida">${caida}</span>
    </div>`;
  }).join('');

  const aviso = document.getElementById('aviso-perdida');
  const peor = escalones.find(s => s.clave === e.mayor_perdida);
  aviso.innerHTML = peor
    ? `<div class="aviso-perdida">Mayor pérdida en <strong>${esc(peor.etiqueta)}</strong>:
       se caen <strong>${fmt(e.mayor_perdida_absoluta)}</strong> personas
       (${pct(peor.caida_pct)} del escalón anterior). Aquí es donde se está fallando.</div>`
    : '';
}

/* ── 3 · Demografia ──────────────────────────────────────────────────── */
function pintarDemografia(d) {
  UI.barras('dist-genero', d.distribucion_genero);
  UI.barras('dist-edad', d.distribucion_rango_etario);

  const cruce = d.cruce_genero_edad || {};
  const generos = Object.keys(cruce).sort();
  const rangos = [...new Set(generos.flatMap(g => Object.keys(cruce[g])))].sort();
  const cont = document.getElementById('cruce');

  if (!generos.length || !rangos.length) {
    cont.innerHTML = '<div class="vacio">sin demografía válida todavía</div>';
    return;
  }
  const cab = ['<th></th>', ...rangos.map(r => `<th>${esc(r)}</th>`)].join('');
  const filas = generos.map(g => {
    const celdas = rangos.map(r => {
      const c = cruce[g][r];
      if (!c) return '<td class="celda">—</td>';
      return `<td class="celda"><span class="tasa">${pct(c.tasa_degustacion)}</span>
              <br><span class="n">${fmt(c.degustaron)}/${fmt(c.atendidos)}</span></td>`;
    }).join('');
    return `<tr><th style="text-align:left">${esc(g)}</th>${celdas}</tr>`;
  }).join('');
  cont.innerHTML = `<table class="cruce"><thead><tr>${cab}</tr></thead><tbody>${filas}</tbody></table>`;
}

/* ── 4 · Temporalidad ────────────────────────────────────────────────── */
function pintarSerie(serie) {
  const cont = document.getElementById('serie');
  if (!serie.length) { cont.innerHTML = '<div class="vacio">sin actividad registrada</div>'; return; }
  const tope = Math.max(...serie.map(b => b.pasaron)) || 1;

  cont.innerHTML = serie.map(b => {
    const alto = n => `${Math.round(120 * n / tope)}px`;
    return `<div class="col" title="${esc(b.bucket)} · pasaron ${fmt(b.pasaron)}, atendidos ${fmt(b.atendidos)}, degustaron ${fmt(b.degustaron)}">
      <div class="seg" style="height:${alto(b.pasaron)};background:#3d9dff"></div>
      <div class="seg" style="height:${alto(b.atendidos)};background:#ffb340"></div>
      <div class="seg" style="height:${alto(b.degustaron)};background:#00d4aa"></div>
    </div>`;
  }).join('');
}

/* ── 5 · Mapa de calor (matriz de /api/stand, dibujarCalor + lightbox) ── */
function pintarMapa(datos) {
  const cont = document.getElementById('mapa-cont');
  const nota = document.getElementById('mapa-nota');
  cont.innerHTML = '';

  if (datos.vacio || !datos.matriz || !datos.matriz.length) {
    cont.innerHTML = '<div class="vacio">Todavía no hay mapa: se genera solo mientras el pipeline procesa.</div>';
    nota.textContent = '';
    return;
  }

  // La matriz llega normalizada 0..1. Se escala a 0–100 ANTES de dibujar:
  // los colores relativos no cambian (dibujarCalor normaliza contra el tope
  // de la propia rejilla) y las cifras de la vista ampliada pasan a ser
  // porcentajes legibles en vez de 0/1.
  const grid = datos.matriz.map(f => f.map(v =>
    Math.round(100 * Math.max(0, Math.min(1, v)))));
  const filas = grid.length, cols = grid[0].length;

  const mini = dibujarCalor(grid, Math.max(4, Math.floor(640 / cols)));
  mini.title = 'clic para ampliar el mapa de calor';
  cont.appendChild(mini);
  mini.addEventListener('click', () => abrirLightbox({
    titulo: `Mapa de calor — ${ETIQUETA_MAPA[MAPA_TIPO] || MAPA_TIPO}`,
    sub: `sesión #${SESION} · rejilla ${cols}×${filas} · snapshot ${datos.ts ?? '—'}`,
    nodo: dibujarCalor(grid, Math.max(24, Math.floor(1100 / cols)), true),
    pie: 'Cada celda es una zona del plano de la cámara; el número es la intensidad relativa (0–100) dentro de la sesión. Naranja más intenso = más actividad.',
  }));

  nota.textContent = `Rejilla ${cols}×${filas} · celda ${datos.grid?.celda_px ?? '?'} px · snapshot ${datos.ts ?? '—'}`;
}

/* ── 6 · Rendimiento del chef ────────────────────────────────────────── */
function pintarChef(kpis, perdido, sesion) {
  const set = (clave, valor) => {
    const el = document.querySelector(`[data-chef="${clave}"]`);
    if (el) el.textContent = valor;
  };
  set('atenciones', num(kpis.atendidos));
  set('media', seg(kpis.tiempo_medio_atencion_s));
  set('cierre', pct(kpis.tasa_degustacion));
  set('perdidos', num(perdido.personas));
  set('perdidos_tasa', pct(perdido.tasa_sobre_detenidos));

  // atenciones/hora se deriva de la duracion de la jornada. Si la sesion
  // sigue abierta no hay hora_fin y no se inventa: se muestra raya.
  let porHora = null;
  if (sesion && sesion.hora_inicio && sesion.hora_fin) {
    const aSeg = h => { const [H, M, S] = h.split(':').map(Number); return H * 3600 + M * 60 + (S || 0); };
    const dur = aSeg(sesion.hora_fin) - aSeg(sesion.hora_inicio);
    if (dur > 60) porHora = kpis.atendidos / (dur / 3600);
  }
  set('por_hora', porHora === null ? '—' : porHora.toFixed(1));

  document.getElementById('chef-nota').textContent = porHora === null
    ? 'Atenciones/hora requiere la sesión cerrada: mientras siga abierta, la jornada no tiene duración conocida.'
    : `Tráfico perdido = personas que se detuvieron y se fueron sin ser atendidas. Si sube, o el chef no da abasto o no está viendo a quien se acerca.`;
}

/* ── 7 · Tabla de visitantes ─────────────────────────────────────────── */
async function pintarVisitantes() {
  const clasif = document.getElementById('f-clasificacion').value;
  const genero = document.getElementById('f-genero').value;
  const q = new URLSearchParams();
  if (clasif) q.set('clasificacion', clasif);
  if (genero) q.set('genero', genero);

  const cola = q.toString() ? `?${q}` : '';
  const { visitantes } = await StandAPI.leer(`${SESION}/visitantes${cola}`);

  document.getElementById('btn-csv').href =
    `/api/stand/${SESION}/visitantes?export=csv${q.toString() ? '&' + q : ''}`;

  UI.tabla('tabla-visitantes', visitantes, [
    ['primera_deteccion', 'entró'],
    ['genero', 'género', v => esc(v)],
    ['rango_etario', 'edad', v => esc(v)],
    ['dwell_max_s', 'permanencia', v => seg(v)],
    ['duracion_atencion_s', 'atención', v => (v ? seg(v) : '—')],
    ['clasificacion', 'resultado', v => esc(ETIQUETA_CLASIF[v] || v)],
    ['motivo_sin_demografia', 'sin demografía', v => v ? esc(v) : ''],
  ]);
}

/* ── 8 · Calidad del dato ────────────────────────────────────────────── */
function pintarCalidad(q) {
  const zonas = q.zonas_calibradas || {};
  const sinCalibrar = Object.entries(zonas).filter(([, ok]) => !ok).map(([n]) => n);
  const motivos = Object.entries(q.descartes_por_motivo || {});

  document.getElementById('calidad-cuerpo').innerHTML = `
    <div class="kpis" style="margin:0">
      <div class="kpi"><div class="valor">${pct(q.null_rate_demografia)}</div><div class="nombre">null-rate demografía</div></div>
      <div class="kpi"><div class="valor">${num(q.sin_demografia)}</div><div class="nombre">sin género/edad</div></div>
      <div class="kpi"><div class="valor">${num(q.tracks_descartados)}</div><div class="nombre">tracks descartados</div></div>
      <div class="kpi"><div class="valor">${num(q.personal_excluido)}</div><div class="nombre">personal excluido</div></div>
    </div>
    ${motivos.length ? `<h3 style="margin-top:14px">Descartes por motivo</h3>
      <ul>${motivos.map(([m, n]) => `<li>${esc(m)}: <strong>${fmt(n)}</strong></li>`).join('')}</ul>` : ''}
    ${sinCalibrar.length ? `<div class="aviso-perdida" style="background:rgba(255,179,64,.12);border-color:#ffb340">
        Zonas SIN calibrar: <strong>${sinCalibrar.map(esc).join(', ')}</strong>.
        Las cifras de esta sesión no son comparables con otras.</div>` : ''}
    ${q.listo_para_contar === false ? `<div class="aviso-perdida">
        El módulo no estaba en condiciones de contar: ${esc(q.motivo_configuracion)}</div>` : ''}
    ${(q.avisos_configuracion || []).length ? `<h3 style="margin-top:14px">Degradaciones conocidas</h3>
      <ul>${q.avisos_configuracion.map(a => `<li>${esc(a)}</li>`).join('')}</ul>` : ''}`;
}

/* ── Orquestacion ────────────────────────────────────────────────────── */
async function cargarSesion(id, sesiones) {
  SESION = id;
  const sesion = sesiones.find(s => s.id === id) || null;

  const [k, e, d, t, perdido, q, mapa] = await Promise.all([
    StandAPI.leer(`${id}/kpis`),
    StandAPI.leer(`${id}/embudo`),
    StandAPI.leer(`${id}/demografia`),
    StandAPI.leer(`${id}/temporal?bucket=15m`),
    StandAPI.leer(`${id}/trafico-perdido`),
    StandAPI.leer(`${id}/calidad`),
    StandAPI.leer(`${id}/heatmap?tipo=${MAPA_TIPO}`),
  ]);

  pintarKpis(k, await promedioAnteriores(sesiones, id));
  pintarEmbudo(e);
  pintarDemografia(d);
  pintarSerie(t.serie || []);
  pintarMapa(mapa);
  pintarChef(k, perdido, sesion);
  pintarCalidad(q);
  await pintarVisitantes();
}

async function arrancar() {
  const marca = document.querySelector('.refresco');
  try {
    const { sesiones } = await StandAPI.leer('sesiones');
    const selector = document.getElementById('selector-sesion');

    if (!sesiones.length) {
      selector.innerHTML = '<option>sin sesiones todavía</option>';
      document.getElementById('embudo').innerHTML =
        '<div class="vacio">Aún no hay ninguna jornada registrada. Se crea sola al arrancar el pipeline del stand.</div>';
      return;
    }

    // El refresco periodico reconstruye el <select>: se conserva la sesion
    // que el usuario estaba mirando (si sigue existiendo) en vez de saltar
    // a la primera de la lista cada 30 s.
    const previa = selector.value;
    selector.innerHTML = sesiones.map(s =>
      `<option value="${s.id}">#${s.id} · ${esc(s.fecha)} ${esc(s.hora_inicio)} · ${esc(s.nombre || s.stand_id)}${s.estado === 'abierta' ? ' (en curso)' : ''}</option>`
    ).join('');
    if (previa && sesiones.some(s => String(s.id) === previa)) {
      selector.value = previa;
    }

    const recargar = () => cargarSesion(Number(selector.value), sesiones);
    selector.onchange = recargar;
    document.getElementById('f-clasificacion').onchange = pintarVisitantes;
    document.getElementById('f-genero').onchange = pintarVisitantes;
    document.getElementById('f-mapa-tipo').onchange = async ev => {
      MAPA_TIPO = ev.target.value;
      pintarMapa(await StandAPI.leer(`${SESION}/heatmap?tipo=${MAPA_TIPO}`));
    };

    await recargar();
    if (marca) marca.textContent = `actualizado ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    if (marca) marca.textContent = `error: ${err.message}`;
    console.error(err);
  }
}

/* ── Guías «¿Cómo leer esta sección?» (14-ago) ─────────────────────────
 * Nada se da por sabido: cada bloque define sus términos con un dibujo.
 * Contenido ESTÁTICO del dashboard (aquí no viaja ningún dato del servidor). */
function montarGuias() {
  montarGuia('jornada', { ilu: 'pasillos', puntos: [
    '<b>Sesión</b>: una jornada de trabajo del stand (una fecha con hora de inicio y fin). En el selector de arriba se puede revisar cualquier jornada pasada; «(en curso)» es la de hoy, que todavía está sumando.',
    'Las cuatro primeras cifras cuentan PERSONAS y cada una es parte de la anterior: <b>pasaron</b> por delante del stand, <b>se detuvieron</b> a mirar, fueron <b>atendidas</b> por el chef y <b>degustaron</b> el producto.',
    'Las tasas miden qué tan bien se convierte cada paso: <b>captación</b> (de los que pasan, % que se detiene), <b>atención</b> (de los detenidos, % atendido), <b>degustación</b> (de los atendidos, % que prueba) y <b>conversión global</b> (de TODOS los que pasaron, % que terminó probando).',
    'La flecha <b>▲/▼ vs media</b> compara con el promedio de las sesiones anteriores (hasta 10): verde = mejor que lo habitual en este stand, rojo = peor.',
    'Una <b>raya (—)</b> significa «sin dato todavía», no cero: en una sesión recién abierta es lo normal.',
  ]});
  montarGuia('embudo-sec', { ilu: 'embudo', abierta: true, puntos: [
    'El <b>embudo</b> cuenta la jornada por pasos: TODOS entran por arriba (pasaron por delante) y en cada escalón quedan menos. Que se estreche es normal; lo importante es <b>dónde</b> se estrecha más.',
    '<b>Paso</b>: la cámara vio cruzar a la persona por delante del stand. <b>Detención</b>: se paró unos segundos frente a él — la primera señal de interés.',
    '<b>Atención</b>: el chef llegó a interactuar con esa persona. <b>Degustación</b>: terminó probando el producto — la meta de todo el stand.',
    'A la derecha de cada escalón va la <b>caída</b>: cuántas personas (y qué %) se perdieron respecto al escalón anterior. El escalón <b style="color:#ff5c5c">en rojo</b> es la mayor fuga de la jornada.',
    'Cada fuga sugiere un arreglo distinto: si se pierden en la <b>detención</b>, el stand no llama la atención (ubicación, cartel, muestra a la vista); si es en la <b>atención</b>, el chef no da abasto o no aborda; si es en la <b>degustación</b>, el argumento o el producto no convencen.',
  ]});
  montarGuia('demografia-sec', { ilu: 'demografia', puntos: [
    '<b>Género y edad</b> se estiman a partir del rostro que capta la cámara. Solo entran caras vistas con claridad: es normal que aquí haya menos personas que en el tráfico total.',
    'Las barras dicen QUIÉN pasa por el stand; la tabla de abajo dice a quién le CONECTA el producto: la <b>tasa de degustación</b> de cada casilla género × edad (de los atendidos de ese segmento, % que degustó).',
    'Entre paréntesis va el detalle <b>degustaron/atendidos</b>. Ojo con las casillas de pocos casos: un 100% sobre 2 personas no prueba nada; fíjate en los segmentos con volumen.',
    'Decisión que permite: a qué público dirigir la promoción, o qué variante ofrecer, según quién de verdad termina probando.',
    'Límite honesto: de noche, a contraluz o de espaldas no hay rostro utilizable; esas personas cuentan en el embudo pero no aquí (el panel de calidad dice cuántas fueron).',
  ]});
  montarGuia('temporal-sec', { ilu: 'permanencia', puntos: [
    'Cada columna es un bloque de <b>15 minutos</b> de la jornada; al pasar el ratón por encima aparecen las cifras exactas de esa franja.',
    'Colores: <b style="color:#3d9dff">azul</b> = pasaron, <b style="color:#ffb340">naranja</b> = atendidos, <b style="color:#00d4aa">verde</b> = degustaron. Mucho azul con poco naranja = pasó gente y casi nadie fue atendido en esa franja.',
    'Decisión que permite: a qué horas reforzar el stand (más producto, una segunda persona) y a qué horas no rinde tenerlo montado.',
    'Una franja vacía no siempre es «no vino nadie»: si la cámara estuvo caída en ese rato tampoco hay datos (el panel de calidad lo delata).',
  ]});
  montarGuia('mapa-sec', { ilu: 'heatmap', puntos: [
    'Es el plano de la cámara dividido en celdas, con <b>colores cálidos</b> (naranja) donde hubo más actividad. Tres capas para elegir: <b>permanencia</b> (dónde se queda la gente), <b>tráfico</b> (por dónde pasa) y <b>detenciones</b> (dónde se ancla frente al stand).',
    'La gracia está en comparar capas: una zona con mucho <b>tráfico</b> y pocas <b>detenciones</b> es gente que pasa sin mirar — buen sitio para reubicar la mesa o poner un cartel.',
    'Clic en el mapa para ampliarlo con cifras. El número de cada celda es <b>intensidad relativa (0–100)</b> dentro de esta sesión, no un conteo de personas: el 100 es solo la celda más caliente de la propia sesión.',
    'Por eso mismo, dos mapas de sesiones distintas <b>no se comparan por color</b>: cada uno se normaliza contra su propio máximo.',
  ]});
  montarGuia('chef-sec', { ilu: 'personal', puntos: [
    '<b>Atención</b>: el rato en que el chef interactúa con una persona que se detuvo. Aquí se mide cuántas hizo, a qué ritmo (<b>atenciones/hora</b>) y su <b>duración media</b>.',
    '<b>Tasa de cierre</b>: de los que atendió, % que terminó degustando — mide la eficacia del discurso, no la del stand.',
    '<b>Tráfico perdido</b>: personas que SE DETUVIERON (ya habían mostrado interés) y se fueron <b>sin ser atendidas</b>: clientes calientes que se escaparon. Si sube, o el chef no da abasto o no ve a quien se le acerca.',
    '<b>Atenciones/hora</b> necesita saber cuánto duró la jornada: mientras la sesión siga abierta se muestra una raya, no un número inventado.',
    'Decisión que permite: con tráfico perdido alto y duraciones largas por atención, hace falta una segunda persona en las horas pico.',
  ]});
  montarGuia('visitantes-sec', { ilu: 'capturas', puntos: [
    'Cada fila es <b>una persona</b> que pasó por el stand, con su historia: cuándo entró, cuánto se quedó (<b>permanencia</b>), cuánto duró su atención y en qué terminó.',
    'El <b>resultado</b> resume su recorrido: <i>pasó de largo</i>, <i>se detuvo sin atención</i> (interés desaprovechado), <i>atendido, no degustó</i> o <i>degustó</i>.',
    'La columna <b>«sin demografía»</b> explica POR QUÉ una persona no tiene género/edad (de espaldas, rostro poco visible…): el hueco viene con motivo, no se esconde.',
    'Los filtros de arriba acotan la tabla (p. ej. solo quienes degustaron) y <b>⬇ exportar CSV</b> descarga exactamente lo filtrado para abrirlo en Excel.',
  ]});
  montarGuia('calidad-sec', { ilu: 'salud', puntos: [
    'Este panel dice cuánto hay que <b>fiarse</b> de todo lo de arriba. Se muestra siempre, incluso cuando está limpio: su ausencia haría creer que las cifras son más firmes de lo que son.',
    '<b>Null-rate de demografía</b>: % de visitantes sin género/edad. Si es alto, el tráfico sigue siendo válido, pero la sección de demografía representa solo a una parte de la gente.',
    '<b>Tracks descartados</b>: detecciones que el sistema decidió NO contar (reflejos, cruces fugaces, duplicados). Descartar es honestidad, no fallo: evita inflar el embudo.',
    '<b>Personal excluido</b>: las veces que el chef o el personal enrolado fue reconocido y sacado de las cifras — sin esto, el propio equipo inflaría el tráfico del stand.',
    '<b>Zonas SIN calibrar</b> es el aviso serio: si las zonas de paso/detención/atención no están dibujadas sobre la imagen real de ESTA cámara, el sistema cuenta con criterios genéricos y la cifra <b>no es comparable</b> con otras sesiones ni sirve de referencia. Antes de sacar conclusiones, calibrar.',
    'Si aparece «el módulo no estaba en condiciones de contar», la sesión entera es solo orientativa: el motivo exacto sale al lado.',
  ]});
}
montarGuias();

arrancar();
// La sesion en curso cambia sola; 30 s es suficiente para un dashboard de
// analisis (los contadores en vivo estan en el cliente, no aqui).
setInterval(arrancar, 30000);
