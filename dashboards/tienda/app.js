/* dashboards/tienda/app.js — dashboard de producto de tienda (rediseño 13-ago-2026).
 *
 * Marketing y consumo sobre /api/v1 + /api/anaqueles + /api/personal.
 * Principios del rediseño:
 *
 *  - Los mapas de calor se muestran TODOS (también los de cámaras sin
 *    registrar como tienda, marcados): antes se filtraban contra el dominio
 *    y los huérfanos desaparecían sin aviso.
 *  - Cualquier imagen o rejilla se AMPLÍA con un clic (lightbox propio).
 *  - Ninguna imagen rota en silencio: las miniaturas de capturas caen a la
 *    foto completa y, si tampoco existe, a un aviso legible.
 *  - Cada sección del HTML dice qué pregunta de marketing responde.
 *
 * NOTA de producto: el evento «empleado repone anaquel» NO se pinta porque el
 * servidor dejó de emitir la analítica de retail en la simplificación del
 * 27-jul-2026. Su sucesor real es el modo Anaqueles (visitas/agarres).
 */
'use strict';

/* fmt, IMAGEN_ROTA, abrirLightbox, pintarMapasTodos y dibujarCalor vienen de
 * shared/util.js (cargado antes que este archivo). */

const TOPE_CAPTURAS = 500;          // límite máximo por página de capturas
let mostrar = 24;

/* ── Periodo elegido: gobierna TODAS las secciones ────────────────────
 * Antes la página no sabía de fechas: los KPIs eran «lo que llevara viva la
 * sesión del servidor» y las franjas horarias mezclaban todos los días. Ahora
 * hay una ventana explícita y el servidor la respeta (HITO 2).
 * `hoy` se calcula en LOCAL, que es el día que vive la tienda. */
const PERIODO = { desde: null, hasta: null, camara: '', etiqueta: 'hoy' };

function iso(fecha) {
  const z = n => String(n).padStart(2, '0');
  return `${fecha.getFullYear()}-${z(fecha.getMonth() + 1)}-${z(fecha.getDate())}`;
}

function diasAtras(n) {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return iso(d);
}

/** Traduce un chip («7d») a la ventana concreta. */
function ventanaDe(etiqueta) {
  const hoy = iso(new Date());
  if (etiqueta === 'ayer') return { desde: diasAtras(1), hasta: diasAtras(1) };
  if (etiqueta === '7d') return { desde: diasAtras(6), hasta: hoy };
  if (etiqueta === '30d') return { desde: diasAtras(29), hasta: hoy };
  return { desde: hoy, hasta: hoy };
}

/** `?desde=…&hasta=…&camara=…` para los endpoints de /api/v1/tienda. */
function consulta(extra = {}) {
  const p = new URLSearchParams({ desde: PERIODO.desde, hasta: PERIODO.hasta });
  if (PERIODO.camara) p.set('camara', PERIODO.camara);
  for (const [k, v] of Object.entries(extra)) p.set(k, v);
  return p.toString();
}

/** Un periodo de un solo día: lo que necesitan anaqueles y el PDF, que
 *  trabajan por jornada. */
function esUnDia() { return PERIODO.desde === PERIODO.hasta; }

/* ── Capturas ─────────────────────────────────────────────────────────── */
function tarjetaCaptura(c) {
  const quien = [c.gender, c.age_range].filter(v => v && v !== 'Desconocido').join(' · ')
    || 'sin demografía';
  // Escrita con `html` (shared/util.js): escapa TODAS las interpolaciones,
  // también las URL, sin tener que acordarse de `UI.esc` en cada una. Es la
  // referencia para lo que se escriba de aquí en adelante.
  //
  // La cadena de respaldo (miniatura → foto completa → placeholder) ya no
  // lleva JavaScript en un atributo: solo declara `data-respaldo` y de lo
  // demás se ocupa el oyente único de la página.
  return html`<a class="item" href="${c.url}" data-titulo="${quien}"
             data-sub="${(c.camera || '—') + ' · ' + (c.timestamp || '')}">
    <img loading="lazy" src="${c.url_miniatura}" alt="captura de persona"
         data-respaldo="${c.url}">
    <div class="pie">
      <b>${quien}</b>
      ${c.visitas ? html`<span class="pill ok">${c.visitas} visitas</span>` : ''}
      <small>${(c.camera || '—') + ' · ' + (c.timestamp || '')}</small>
      ${c.revisado_por_vlm ? html`<small>revisado por VLM 🤖</small>` : ''}
    </div></a>`;
}

/* ── Curva por hora ───────────────────────────────────────────────────
 * Sustituye a las «franjas» que se calculaban contando CAPTURAS por hora
 * sobre las 500 más recientes, sin mirar la fecha: un martes flojo heredaba
 * el pico del sábado, y además costaba 500 capturas por la red cada 15 s.
 * Ahora son las ENTRADAS reales al área, servidas ya agregadas. */
function pintarCurva(serie, cobertura) {
  const el = document.getElementById('curva');
  const nota = document.getElementById('curva-nota');
  const tramos = serie.tramos || [];
  const max = Math.max(1, ...tramos.map(t => t.entradas));

  if (!serie.total_entradas) {
    // «Sin datos» y «cero visitas» NO son lo mismo y no pueden pintarse igual:
    // una barra plana se lee como una caída de tráfico que nunca ocurrió.
    el.innerHTML = `<div class="vacio">${cobertura.length
      ? 'el área de conteo no registró ninguna entrada en este periodo'
      : 'no se midió nada en este periodo — ninguna cámara de tienda transmitió'}</div>`;
    nota.textContent = '';
    return;
  }

  el.innerHTML = tramos.map(t => {
    const alto = Math.round(100 * t.entradas / max);
    const clase = t === serie.punta ? ' punta'
      : (serie.punta && t.hora === serie.punta.hora ? ' punta' : '');
    return `<div class="hora${clase}" title="${UI.esc(t.hora)}:00 — ${fmt(t.entradas)} entradas de ${fmt(t.personas)} personas">
      <div class="barra-h" style="height:${Math.max(2, alto)}%"></div>
      <div class="etq">${UI.esc(t.hora)}</div>
    </div>`;
  }).join('');

  const p = serie.punta;
  const v = serie.valle;
  nota.innerHTML = p
    ? `Hora punta: <b>${UI.esc(p.hora)}:00</b> (${fmt(p.entradas)} entradas)`
      + (v && v.hora !== p.hora
        ? ` · hora más floja con movimiento: <b>${UI.esc(v.hora)}:00</b> (${fmt(v.entradas)})` : '')
      + ` · total del periodo: <b>${fmt(serie.total_entradas)}</b>`
    : '';
}

/* ── Anaqueles ────────────────────────────────────────────────────────── */
async function anaquelesJson(ruta) {
  const r = await fetch(`/api/anaqueles/${ruta}`, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${r.status} en /api/anaqueles/${ruta}`);
  return r.json();
}

async function pintarAnaqueles() {
  const cont = document.getElementById('anaqueles');
  const VACIO = '<div class="vacio">sin datos de anaqueles — se llenan cuando una cámara transmite en modo Anaqueles con las áreas dibujadas</div>';

  // Anaqueles trabaja por JORNADA: con un rango de varios días se muestra el
  // último, y se dice. Inventar una suma de días con contadores absolutos
  // sería justo el tipo de cifra que este dashboard no quiere.
  const fecha = PERIODO.hasta;
  let datos;
  try {
    // UNA petición para todas las cámaras (antes: 1 + 2N cada 30 s).
    datos = await anaquelesJson(`resumen?fecha=${encodeURIComponent(fecha)}`);
  } catch (e) { return; }   // fallo de RED o servidor sin el modo: se conserva la vista

  const bloques = [];
  let estrella = null;      // el área con más agarres de todas las cámaras

  for (const res of datos.camaras || []) {
    if (!res.areas || !res.areas.length) continue;
    const cam = res.camera_id;
    if (PERIODO.camara && String(cam) !== PERIODO.camara) continue;

    const topAgarres = Math.max(1, ...res.areas.map(a => a.agarres));
    const topVisitas = Math.max(1, ...res.areas.map(a => a.visitas));

    const filas = [];
    for (const [i, a] of (res.ranking || []).entries()) {
      const detalle = res.areas.find(x => x.idx === a.idx) || {};
      if (!estrella || a.agarres > estrella.agarres) {
        estrella = { nombre: a.nombre, agarres: a.agarres };
      }
      const tasa = a.visitas ? `${(100 * a.agarres / a.visitas).toFixed(0)}%` : '—';
      const genero = generoDominante(detalle.por_genero);
      filas.push({ a, i, grid: detalle.grid, tasa, genero });
    }

    const idCam = `anq-${bloques.length}`;
    bloques.push({
      idCam, cam, res,
      html: `<div class="anq-camara">
        <h3>📷 ${UI.esc(cam)}
          <span class="fecha">${UI.esc(res.fecha || fecha)}</span>
          <span class="totales"><b>${fmt(res.total_agarres)}</b> agarres · ${fmt(res.total_visitas)} visitas</span>
        </h3>
        ${filas.map(({ a, i, tasa, genero }) => `
          <div class="anq-fila ${i === 0 && a.agarres > 0 ? 'top' : ''}">
            <div class="anq-nombre"><span class="pos">${i + 1}</span><b>${UI.esc(a.nombre)}</b>
              ${genero !== '—' ? `<small style="color:var(--apagado)"> · ${UI.esc(genero)}</small>` : ''}</div>
            <div class="anq-medidas">
              <div class="anq-medida agarres"><span>agarres</span>
                <div class="pista"><div class="relleno" style="width:${(100 * a.agarres / topAgarres).toFixed(1)}%"></div></div>
                <span class="cifra">${fmt(a.agarres)}</span></div>
              <div class="anq-medida visitas"><span>visitas</span>
                <div class="pista"><div class="relleno" style="width:${(100 * a.visitas / topVisitas).toFixed(1)}%"></div></div>
                <span class="cifra">${fmt(a.visitas)}</span></div>
            </div>
            <div class="anq-conversion" title="agarres ÷ visitas"><b>${tasa}</b>conversión</div>
            <div class="anq-heat" data-cam="${idCam}" data-idx="${a.idx}" title="clic para ampliar el mapa de calor"></div>
          </div>`).join('')}
      </div>`,
      filas,
    });
  }

  UI.kpis({ anaquelEstrella: estrella && estrella.agarres > 0 ? estrella.nombre : null });

  // Repintar SIEMPRE que la API respondió: tras la medianoche el resumen de
  // hoy viene vacío y sin esto la sección se quedaba congelada mostrando ayer.
  if (!bloques.length) {
    const sinMedir = (datos.sin_datos || []).length;
    cont.innerHTML = sinMedir
      ? `<div class="vacio">ninguna de las ${sinMedir} cámara(s) con anaqueles midió el ${UI.esc(fecha)}</div>`
      : VACIO;
    return;
  }
  const aviso = esUnDia() ? ''
    : `<div class="vacio">los anaqueles se cuentan por jornada: se muestra la del <b>${UI.esc(fecha)}</b>, no el rango completo</div>`;
  cont.innerHTML = aviso + bloques.map(b => b.html).join('');

  // Los canvas se insertan tras el innerHTML (un canvas no sobrevive a la
  // serialización) y el clic abre la versión grande con cifras.
  for (const b of bloques) {
    for (const { a, grid } of b.filas) {
      const hueco = cont.querySelector(`.anq-heat[data-cam="${b.idCam}"][data-idx="${a.idx}"]`);
      if (!hueco) continue;
      if (grid && grid.some(f => f.some(v => v > 0))) {
        hueco.appendChild(dibujarCalor(grid, 5));
        hueco.addEventListener('click', () => {
          const titulo = `Mapa de calor — ${a.nombre}`;
          const sub = `${b.cam} · ${fmt(a.agarres)} agarres · ${fmt(a.visitas)} visitas`;
          // Primero la versión BUENA: el calor compuesto sobre la foto
          // real del estante (el servidor la toma cuando el área está
          // vacía). Si aún no existe, la rejilla de siempre.
          const img = new Image();
          img.onload = () => abrirLightbox({
            titulo, sub, nodo: img,
            pie: 'El calor está compuesto sobre la <b>foto real del estante</b> (tomada con el área vacía): las zonas cálidas son donde más meten la mano. La escala está grabada abajo en la imagen.',
          });
          img.onerror = () => abrirLightbox({
            titulo, sub, nodo: dibujarCalor(grid, 30, true),
            pie: 'Cada celda es una zona del área dibujada; el número es el peso acumulado (segundos de mano dentro). Naranja más intenso = más interacción. (La foto real del estante aparecerá sola cuando el área lleve unos minutos vacía.)',
          });
          img.src = `/api/anaqueles/${encodeURIComponent(b.cam)}/heatmap/${a.idx}/imagen?fecha=${encodeURIComponent(fecha)}&t=${Date.now()}`;
        });
      } else {
        hueco.innerHTML = '<div class="sin">sin calor aún</div>';
        hueco.style.cursor = 'default';
      }
    }
  }
}

function generoDominante(porGenero) {
  const pares = Object.entries(porGenero || {});
  if (!pares.length) return '—';
  pares.sort((a, b) => b[1] - a[1]);
  const [g, n] = pares[0];
  const legible = { male: 'hombres', female: 'mujeres',
                    Male: 'hombres', Female: 'mujeres' }[g] || g;
  return `${legible} (${n})`;
}

/* ── Personal de la tienda (/api/personal) ────────────────────────────── */
/* API que ESCRIBE, fuera de /api/v1: fetch directo, como /api/anaqueles.
 * El personal enrolado queda excluido del marketing y dispara las alertas
 * «Seguridad en pasillo» / «Empleado surte anaquel». */

const PERS_ETIQUETA_ROL = { seguridad: '🛡️ Seguridad', pasillero: '📦 Pasillero' };
let _persFichas = [];        // último listado, para resolver acciones por id
let _persMontado = false;    // el formulario se monta UNA vez y no se repinta

async function personalApi(ruta, metodo = 'GET', cuerpo = null) {
  const opciones = { method: metodo, cache: 'no-store' };
  if (cuerpo !== null) {
    opciones.headers = { 'Content-Type': 'application/json' };
    opciones.body = JSON.stringify(cuerpo);
  }
  const r = await fetch(`/api/personal/${ruta}`, opciones);
  let datos = null;
  try { datos = await r.json(); } catch (e) { /* respuesta sin JSON */ }
  if (!r.ok) {
    // El `detail` del 400 viene redactado para el operador («usa una foto de
    // frente, con buena luz…»): se muestra tal cual, no un código de error.
    const e = new Error((datos && datos.detail)
      ? datos.detail : `/api/personal/${ruta} respondió ${r.status}`);
    e.status = r.status;   // el que llama distingue 404 (sin API) de lo demás
    throw e;
  }
  return datos;
}

function leerComoDataURL(archivo) {
  return new Promise((resolver, rechazar) => {
    const lector = new FileReader();
    lector.onload = () => resolver(lector.result);   // data-URL entero: la API lo acepta
    lector.onerror = () => rechazar(new Error('no se pudo leer el archivo elegido'));
    lector.readAsDataURL(archivo);
  });
}

/** Valida tamaño y tipo ANTES de readAsDataURL: un archivo gigante congela
 *  la pestaña al codificarse en base64 y el servidor lo rechazaría igual.
 *  Compartido por enrolar y «+ foto». Devuelve false (con aviso) si no vale. */
function validarImagen(archivo) {
  if (archivo.size > 12 * 1024 * 1024) {
    avisoPersonal(`la foto pesa ${(archivo.size / (1024 * 1024)).toFixed(1)} MB `
      + 'y el máximo es 10 MB; usa una más liviana');
    return false;
  }
  if (archivo.type && !archivo.type.startsWith('image/')) {
    avisoPersonal('elige una imagen (JPG o PNG)');
    return false;
  }
  return true;
}

/** Abre el selector de archivos y resuelve con el elegido (null si cancela). */
function elegirImagen() {
  return new Promise(resolver => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    let resuelto = false;
    const terminar = () => {
      if (resuelto) return;      // la promesa se asienta UNA sola vez
      resuelto = true;
      resolver(input.files[0] || null);
    };
    input.addEventListener('change', terminar);
    input.addEventListener('cancel', terminar);
    // No todos los navegadores emiten 'cancel' en <input type=file>: al
    // cerrar el diálogo la ventana recupera el foco, y el retardo deja
    // aterrizar primero el 'change' si sí se eligió archivo. Sin esto la
    // promesa quedaba colgada y el botón «+ foto» deshabilitado para siempre.
    window.addEventListener('focus', () => setTimeout(terminar, 300), { once: true });
    input.click();
  });
}

/** Mensaje junto al formulario. tipo: 'error' (rojo) u 'ok' (verde). */
function avisoPersonal(texto, tipo = 'error') {
  const el = document.getElementById('pers-aviso');
  if (!el) return;
  el.textContent = texto;
  el.className = 'pers-aviso' + (texto ? ` ${tipo}` : '');
}

function tarjetaPersonal(p) {
  const rol = PERS_ETIQUETA_ROL[p.rol] || p.rol || '—';
  const esUniforme = p.tipo === 'uniforme';
  const foto = p.foto_url ? `${p.foto_url}?n=${p.n_fotos || 0}` : IMAGEN_ROTA;
  const alta = p.creado ? new Date(p.creado) : null;
  return `<div class="pers-tarjeta ${p.activo ? '' : 'inactivo'}" data-id="${UI.esc(p.id)}">
    <img class="pers-foto" src="${UI.esc(foto)}" alt="foto de ${UI.esc(p.nombre)}"
         loading="lazy" title="clic para ampliar">
    <div class="pers-datos">
      <b>${esUniforme ? '👕 ' : ''}${UI.esc(p.nombre)}</b>
      <span class="pers-rol">${UI.esc(rol)}</span>
      <span><span class="pill ${p.activo ? 'ok' : ''}">${p.activo ? 'activo' : 'inactivo'}</span>${esUniforme
        ? ' <span class="pill" title="identifica la vestimenta (bordado y colores), no un rostro">uniforme</span>' : ''}</span>
      <small>${fmt(p.n_fotos)} foto${p.n_fotos === 1 ? '' : 's'}${alta && !isNaN(alta)
        ? ` · alta ${alta.toLocaleDateString()}` : ''}</small>
    </div>
    <div class="pers-acciones">
      <button class="boton" data-accion="foto" title="añadir otra foto del mismo empleado — más ángulos = mejor re-identificación">+ foto</button>
      <button class="boton" data-accion="activo" title="desactivar NO borra: conserva las fotos para cuando vuelva">${p.activo ? 'Desactivar' : 'Activar'}</button>
      <button class="boton peligro" data-accion="borrar">Borrar</button>
    </div>
  </div>`;
}

function montarPersonal(cont, roles) {
  const opciones = (roles && roles.length ? roles : Object.keys(PERS_ETIQUETA_ROL))
    .map(r => `<option value="${UI.esc(r)}">${UI.esc(PERS_ETIQUETA_ROL[r] || r)}</option>`)
    .join('');
  cont.innerHTML = `
    <div class="tarjeta pers-alta">
      <form id="pers-form" class="pers-form">
        <input type="text" id="pers-nombre" placeholder="nombre del empleado"
               maxlength="60" autocomplete="off">
        <select id="pers-rol">${opciones}</select>
        <select id="pers-tipo" title="rostro: la foto identifica a la persona · uniforme: identifica la vestimenta (funciona de espaldas)">
          <option value="rostro">🙂 Rostro (la persona)</option>
          <option value="uniforme">👕 Uniforme (la vestimenta)</option>
        </select>
        <input type="file" id="pers-archivo" accept="image/*"
               title="rostro: foto de frente, con buena luz · uniforme: cuerpo entero">
        <button type="submit" class="boton" id="pers-enrolar">Enrolar</button>
        <button type="button" class="boton" id="pers-refrescar" title="recargar la lista">↻</button>
        <span class="pers-cuenta" id="pers-cuenta"></span>
      </form>
      <div class="pers-aviso" id="pers-aviso"></div>
    </div>
    <div class="pers-lista" id="pers-lista"></div>`;

  document.getElementById('pers-form').addEventListener('submit', e => {
    e.preventDefault();
    enrolarPersonal().catch(console.error);
  });
  // Con tipo uniforme el «nombre» describe la PRENDA, no a una persona: el
  // placeholder lo dice antes de que el operador escriba un nombre propio.
  document.getElementById('pers-tipo').addEventListener('change', e => {
    document.getElementById('pers-nombre').placeholder =
      e.target.value === 'uniforme'
        ? 'nombre del uniforme (ej. Uniforme de seguridad)'
        : 'nombre del empleado';
  });
  document.getElementById('pers-refrescar').addEventListener('click', () => {
    avisoPersonal('');
    pintarPersonal().catch(console.error);
  });
  document.getElementById('pers-lista').addEventListener('click', e => {
    accionPersonal(e).catch(console.error);
  });
}

async function enrolarPersonal() {
  const nombre = document.getElementById('pers-nombre').value.trim();
  const rol = document.getElementById('pers-rol').value;
  const tipo = document.getElementById('pers-tipo').value;
  const archivo = document.getElementById('pers-archivo').files[0];
  const esUniforme = tipo === 'uniforme';
  if (!nombre) {
    avisoPersonal(esUniforme
      ? 'escribe el nombre del uniforme (ej. Uniforme de seguridad)'
      : 'escribe el nombre del empleado');
    return;
  }
  if (!archivo) {
    avisoPersonal(esUniforme
      ? 'elige una foto del uniforme (de cuerpo entero, que se vean bordado y colores)'
      : 'elige una foto del empleado (de frente y con buena luz)');
    return;
  }
  if (!validarImagen(archivo)) return;
  const boton = document.getElementById('pers-enrolar');
  boton.disabled = true;
  boton.textContent = 'Enrolando…';
  avisoPersonal('');
  try {
    const foto_b64 = await leerComoDataURL(archivo);
    await personalApi('enrolar', 'POST', { nombre, rol, tipo, foto_b64 });
    document.getElementById('pers-form').reset();
    // reset() vuelve a «rostro»: el placeholder debe acompañarlo.
    document.getElementById('pers-nombre').placeholder = 'nombre del empleado';
    avisoPersonal(esUniforme
      ? `${nombre} quedó enrolado: quien lo vista no cuenta en el marketing (sube también la espalda con «+ foto»)`
      : `${nombre} quedó enrolado: desde ahora no cuenta en el marketing`, 'ok');
    await pintarPersonal();
  } catch (e) {
    avisoPersonal(e.message);        // p. ej. «no se detectó ningún rostro claro…»
  } finally {
    boton.disabled = false;
    boton.textContent = 'Enrolar';
  }
}

async function accionPersonal(e) {
  const tarjeta = e.target.closest('.pers-tarjeta');
  if (!tarjeta) return;
  const p = _persFichas.find(x => x.id === tarjeta.dataset.id);
  if (!p) return;

  // Clic en la foto: se amplía en el lightbox, con enlaces a TODAS sus fotos.
  if (e.target.classList.contains('pers-foto')) {
    if (!p.foto_url) return;
    const enlaces = Array.from({ length: p.n_fotos || 0 }, (_, n) =>
      `<a href="/api/personal/${encodeURIComponent(p.id)}/foto/${n}" target="_blank" rel="noopener">foto ${n + 1}</a>`);
    abrirLightbox({
      titulo: p.nombre,
      sub: `${PERS_ETIQUETA_ROL[p.rol] || p.rol} · ${p.activo ? 'activo' : 'inactivo'}`,
      src: p.foto_url,
      pie: enlaces.length > 1 ? `todas sus fotos: ${enlaces.join(' · ')}` : '',
    });
    return;
  }

  const btn = e.target.closest('button[data-accion]');
  if (!btn) return;
  const textoOriginal = btn.textContent;
  btn.disabled = true;
  try {
    if (btn.dataset.accion === 'activo') {
      await personalApi(encodeURIComponent(p.id), 'PATCH', { activo: !p.activo });
      avisoPersonal(p.activo
        ? `${p.nombre} desactivado: vuelve a contar como cliente (sus fotos se conservan)`
        : `${p.nombre} activado: excluido del marketing otra vez`, 'ok');
      await pintarPersonal();
    } else if (btn.dataset.accion === 'borrar') {
      if (!confirm(`¿Borrar a «${p.nombre}» del personal?\n`
        + `Se borran también sus ${p.n_fotos} foto${p.n_fotos === 1 ? '' : 's'} `
        + 'y volverá a contar como cliente en el marketing.')) return;
      await personalApi(encodeURIComponent(p.id), 'DELETE');
      avisoPersonal(`${p.nombre} borrado del personal`, 'ok');
      await pintarPersonal();
    } else if (btn.dataset.accion === 'foto') {
      const archivo = await elegirImagen();
      if (!archivo || !validarImagen(archivo)) return;
      btn.textContent = 'Subiendo…';
      const foto_b64 = await leerComoDataURL(archivo);
      await personalApi(`${encodeURIComponent(p.id)}/foto`, 'POST', { foto_b64 });
      avisoPersonal(`foto añadida a ${p.nombre}`, 'ok');
      await pintarPersonal();
    }
  } catch (err) {
    avisoPersonal(`${p.nombre}: ${err.message}`);
  } finally {
    // Tras repintar el botón original ya no está en el DOM: esto es inocuo.
    btn.disabled = false;
    btn.textContent = textoOriginal;
  }
}

/** Carga al inicio y tras cada operación. NO va en refrescar(15): el personal
 *  cambia poco y el botón ↻ cubre el resto. */
async function pintarPersonal() {
  const cont = document.getElementById('personal');
  let datos;
  try {
    datos = await personalApi('');
  } catch (e) {
    if (_persMontado) {
      // Con la sección ya montada, el fallo de un refresco se cuenta junto
      // al formulario en vez de morir en un return mudo.
      avisoPersonal(e.message);
      return;
    }
    if (e.status === 404) {
      // Servidor anterior sin /api/personal: la sección lo dice con claridad
      // y el resto del dashboard sigue funcionando igual.
      cont.innerHTML = '<div class="vacio">esta sección se activa con la próxima actualización del servidor</div>';
    } else {
      // Fallo TRANSITORIO (red caída, servidor arrancando): antes dejaba la
      // sección muerta para siempre; ahora reintenta sola hasta montarse.
      cont.innerHTML = '<div class="vacio">sin conexión con /api/personal — reintentando…</div>';
      setTimeout(() => { pintarPersonal().catch(console.error); }, 15000);
    }
    return;
  }
  if (!_persMontado) {
    montarPersonal(cont, datos.roles);
    _persMontado = true;
  }
  _persFichas = datos.personal || [];
  document.getElementById('pers-cuenta').textContent = datos.total
    ? `${fmt(datos.total)} enrolado${datos.total === 1 ? '' : 's'} · ${fmt(datos.activos)} activo${datos.activos === 1 ? '' : 's'}`
    : '';
  document.getElementById('pers-lista').innerHTML = _persFichas.length
    ? _persFichas.map(tarjetaPersonal).join('')
    : '<div class="vacio">nadie enrolado todavía — da de alta al primer empleado con el formulario</div>';
}

/* ── Pintura principal ────────────────────────────────────────────────── */

/** Lo último pintado, para que «⬇ CSV» exporte EXACTAMENTE lo que se ve. */
const ULTIMO = { resumen: null, serie: null, pasillos: null, demo: null };

/** Variación de un KPI contra el periodo anterior, con su flecha y su color.
 *  `null` no es 0 %: es «no hay con qué comparar», y se dice con palabras. */
function pintarDeltas(comparativa) {
  const variacion = (comparativa && comparativa.variacion_pct) || {};
  for (const el of document.querySelectorAll('[data-delta]')) {
    if (!el.dataset.delta) { el.textContent = ''; continue; }
    const v = variacion[el.dataset.delta];
    if (v === null || v === undefined) {
      el.className = 'delta';
      el.textContent = comparativa ? 'sin base de comparación' : '';
      continue;
    }
    const signo = v > 0 ? '▲' : (v < 0 ? '▼' : '=');
    el.className = 'delta ' + (v > 0 ? 'sube' : (v < 0 ? 'baja' : ''));
    el.textContent = `${signo} ${Math.abs(v).toFixed(1)} % vs periodo anterior`;
  }
}

/** Aviso de cobertura: qué jornadas del periodo se midieron de verdad.
 *  Un día con el servidor apagado NO es un día de cero visitas. */
function pintarCobertura(ventana) {
  const el = document.getElementById('cobertura');
  const medidas = (ventana.jornadas_con_datos || []).length;
  const total = ventana.dias || 1;
  if (medidas === total) {
    el.className = 'cobertura ok';
    el.textContent = total === 1 ? 'día medido' : `${total} días medidos`;
  } else if (medidas === 0) {
    el.className = 'cobertura mal';
    el.textContent = 'sin datos en este periodo (nadie estaba midiendo)';
  } else {
    el.className = 'cobertura aviso';
    el.textContent = `medidos ${medidas} de ${total} días — el resto no es cero, es que no se midió`;
  }
}

function pintarDemoPasillo(demo) {
  const el = document.getElementById('demo-pasillo');
  const filas = demo.por_camara || [];
  if (!filas.length) {
    el.innerHTML = '<div class="vacio">todavía sin rostros claros por pasillo</div>';
    return;
  }
  const dominante = dist => {
    const pares = Object.entries(dist || {}).sort((a, b) => b[1] - a[1]);
    return pares.length ? `${UI.esc(pares[0][0])} <small>(${pares[0][1]})</small>` : '—';
  };
  el.innerHTML = `<table><thead><tr><th>pasillo</th><th>género</th><th>edad</th></tr></thead><tbody>${
    filas.map(f => `<tr><td>${UI.esc(f.camera_name || f.camera_id)}</td>
      <td>${dominante(f.genero)}</td><td>${dominante(f.edad)}</td></tr>`).join('')
  }</tbody></table>`;
}

async function pintar() {
  const q = consulta();
  const rango = `desde=${PERIODO.desde}&hasta=${PERIODO.hasta}`;
  const [resumen, serie, pasillos, demo, capturas, dispositivos, heatmaps] =
    await Promise.all([
      API.leer(`tienda/resumen?${q}`),
      API.leer(`tienda/serie?${q}`),
      API.leer(`tienda/pasillos?${rango}`),
      API.leer(`tienda/demografia?${q}`),
      API.leer(`tienda/capturas?${consulta({ limite: Math.min(mostrar, TOPE_CAPTURAS) })}`),
      API.leer('dispositivos?client_type=tienda'),
      API.leer('heatmaps'),
    ]);
  Object.assign(ULTIMO, { resumen, serie, pasillos, demo });

  const k = resumen.kpis || {};
  const top = pasillos.mas_concurrido;
  UI.kpis({
    visitantes: fmt(k.visitantes_unicos),
    entradas: fmt(k.entradas),
    // El aforo de un periodo pasado es null, no 0: nadie sabe cuánta gente
    // había dentro el martes a esta hora.
    aforo: (k.aforo_actual === null || k.aforo_actual === undefined)
      ? null : fmt(k.aforo_actual),
    permanencia: k.permanencia_media_s != null
      ? `${Number(k.permanencia_media_s).toFixed(1)} s` : null,
    masConcurrido: top ? (top.camera_name || top.camera_id) : null,
  });
  pintarDeltas(resumen.comparativa);
  pintarCobertura(resumen.ventana || {});
  pintarCurva(serie, (resumen.ventana || {}).jornadas_con_datos || []);

  UI.barras('genero', demo.genero);
  UI.barras('edad', demo.edad);
  pintarDemoPasillo(demo);

  UI.tabla('pasillos', pasillos.ranking, [
    ['camera_name', 'pasillo / cámara', (v, f) => UI.esc(v || f.camera_id)
      + (f.registrada ? '' : ' <span class="pill aviso">sin registrar</span>')],
    ['entradas', 'entradas', v => UI.esc(fmt(v))],
    ['variacion_pct', 'vs anterior', (v, f) => (v === null || v === undefined)
      ? ((f.entradas_previas === null || f.entradas_previas === undefined)
        ? '<span class="pill">sin base</span>' : '=')
      : `<span class="delta ${v > 0 ? 'sube' : (v < 0 ? 'baja' : '')}">${
        v > 0 ? '▲' : (v < 0 ? '▼' : '=')} ${Math.abs(v).toFixed(1)} %</span>`],
    ['visitantes_unicos', 'visitantes', v => UI.esc(fmt(v))],
    ['permanencia_media_s', 'permanencia media',
      v => v ? `${Number(v).toFixed(1)} s` : '—'],
  ]);

  const galeria = document.getElementById('capturas');
  const filas = capturas.capturas || [];
  galeria.innerHTML = filas.length
    ? filas.map(tarjetaCaptura).join('')
    : '<div class="vacio">sin capturas en este periodo — aparecen cuando la cámara de tienda transmite con demografía</div>';
  // Clic en una captura: se amplía en el lightbox en vez de navegar.
  for (const item of galeria.querySelectorAll('a.item')) {
    item.addEventListener('click', e => {
      e.preventDefault();
      abrirLightbox({
        titulo: item.dataset.titulo,
        sub: item.dataset.sub,
        src: item.getAttribute('href'),
        pie: html`<a href="${item.getAttribute('href')}" target="_blank" rel="noopener">abrir la foto original en otra pestaña</a>`,
      });
    });
  }
  const boton = document.getElementById('mas');
  const agotadas = filas.length >= (capturas.total ?? filas.length);
  boton.disabled = agotadas || mostrar >= TOPE_CAPTURAS;
  document.getElementById('mas-nota').textContent = agotadas
    ? '' : `${filas.length} de ${capturas.total}`;

  // Mapas de calor: TODOS (el filtro es del usuario, no del código).
  const propios = new Set(dispositivos.dispositivos.map(d => d.device_id));
  pintarMapasTodos('heatmaps', 'filtros-mapas', heatmaps.heatmaps || [],
                   propios, 'De tienda');

  UI.tabla('dispositivos', dispositivos.dispositivos, [
    ['device_id', 'dispositivo'],
    ['camera_name', 'cámara'],
    ['pipelines', 'pipelines', v => UI.esc((v || []).join(', ') || '—')],
    ['frames', 'frames', v => UI.esc(fmt(v))],
    ['ultima_vez', 'última vez', v => UI.fecha(v)],
  ]);
}

/* ── Controles de periodo y cámara ────────────────────────────────────── */

/** Aplica la ventana y REPINTA todo. Es el único camino, a propósito: así
 *  ninguna sección puede quedarse enseñando un periodo distinto del que dice
 *  la cabecera. */
function aplicarPeriodo(etiqueta, rango = {}) {
  if (etiqueta) {
    PERIODO.etiqueta = etiqueta;
    Object.assign(PERIODO, ventanaDe(etiqueta));
  } else {
    PERIODO.etiqueta = 'rango';
    if (rango.desde) PERIODO.desde = rango.desde;
    if (rango.hasta) PERIODO.hasta = rango.hasta;
    if (PERIODO.hasta < PERIODO.desde) {          // invertido: se endereza
      const v = PERIODO.desde; PERIODO.desde = PERIODO.hasta; PERIODO.hasta = v;
    }
  }
  document.getElementById('f-desde').value = PERIODO.desde;
  document.getElementById('f-hasta').value = PERIODO.hasta;
  for (const chip of document.querySelectorAll('.chip[data-periodo]')) {
    chip.classList.toggle('activo', chip.dataset.periodo === PERIODO.etiqueta);
  }
  // El PDF es por jornada: se le pasa el último día del periodo.
  const pdf = document.getElementById('btn-pdf');
  if (pdf) pdf.href = `/api/informes/tienda.pdf?fecha=${encodeURIComponent(PERIODO.hasta)}`;

  mostrar = 24;
  pintar().catch(console.error);
  pintarAnaqueles().catch(console.error);
}

async function pintarFiltroCamaras() {
  let datos;
  try { datos = await API.leer('tienda/camaras'); } catch (e) { return; }
  const sel = document.getElementById('f-camara');
  const elegido = PERIODO.camara;
  sel.innerHTML = '<option value="">todas las cámaras</option>'
    + (datos.camaras || []).map(c =>
      `<option value="${UI.esc(c.camera_id)}">${UI.esc(c.camera_name || c.camera_id)}${
        c.registrada ? '' : ' (sin registrar)'}</option>`).join('');
  sel.value = elegido;
}

function montarControles() {
  Object.assign(PERIODO, ventanaDe('hoy'));
  document.getElementById('f-desde').value = PERIODO.desde;
  document.getElementById('f-hasta').value = PERIODO.hasta;
  for (const chip of document.querySelectorAll('.chip[data-periodo]')) {
    chip.addEventListener('click', () => aplicarPeriodo(chip.dataset.periodo));
  }
  document.getElementById('f-desde').addEventListener('change', e =>
    aplicarPeriodo(null, { desde: e.target.value }));
  document.getElementById('f-hasta').addEventListener('change', e =>
    aplicarPeriodo(null, { hasta: e.target.value }));
  document.getElementById('f-camara').addEventListener('change', e => {
    PERIODO.camara = e.target.value;
    aplicarPeriodo(PERIODO.etiqueta === 'rango' ? null : PERIODO.etiqueta,
                   { desde: PERIODO.desde, hasta: PERIODO.hasta });
  });
  document.getElementById('btn-csv').addEventListener('click', exportarCSV);
}

/* ── Exportar ─────────────────────────────────────────────────────────── */
/** CSV de lo que se está viendo. Se arma en el navegador con los datos ya
 *  pintados: ni una petición más, y lo exportado es EXACTAMENTE lo mostrado.
 *  Separador `;` y BOM porque el destino real de esto es Excel en español. */
function exportarCSV() {
  if (!ULTIMO.resumen) return;
  const esc = v => {
    const s = (v === null || v === undefined) ? '' : String(v);
    return /[";\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const filas = [];
  const seccion = (titulo, cabecera) => {
    if (filas.length) filas.push([]);
    filas.push([titulo]);
    filas.push(cabecera);
  };

  const k = ULTIMO.resumen.kpis || {};
  const comp = (ULTIMO.resumen.comparativa || {}).kpis || {};
  seccion(`Periodo ${PERIODO.desde} a ${PERIODO.hasta}`
    + (PERIODO.camara ? ` (camara ${PERIODO.camara})` : ''),
    ['indicador', 'valor', 'periodo anterior']);
  const KPIS = [['visitantes unicos', 'visitantes_unicos'],
                ['entradas', 'entradas'], ['salidas', 'salidas'],
                ['permanencia media (s)', 'permanencia_media_s']];
  for (const par of KPIS) filas.push([par[0], k[par[1]], comp[par[1]]]);

  seccion('Curva por hora', ['hora', 'entradas', 'salidas', 'personas distintas']);
  for (const t of (ULTIMO.serie || {}).tramos || []) {
    filas.push([t.hora === undefined ? t.jornada : t.hora,
                t.entradas, t.salidas, t.personas]);
  }

  seccion('Pasillos', ['camara', 'nombre', 'entradas', 'entradas periodo anterior',
                       'variacion %', 'visitantes unicos', 'permanencia media (s)']);
  for (const f of (ULTIMO.pasillos || {}).ranking || []) {
    filas.push([f.camera_id, f.camera_name, f.entradas, f.entradas_previas,
                f.variacion_pct, f.visitantes_unicos, f.permanencia_media_s]);
  }

  seccion('Demografia', ['ambito', 'genero o edad', 'personas']);
  for (const par of Object.entries((ULTIMO.demo || {}).genero || {})) {
    filas.push(['global', par[0], par[1]]);
  }
  for (const par of Object.entries((ULTIMO.demo || {}).edad || {})) {
    filas.push(['global', par[0], par[1]]);
  }
  for (const f of (ULTIMO.demo || {}).por_camara || []) {
    for (const par of Object.entries(f.genero || {})) {
      filas.push([f.camera_name || f.camera_id, par[0], par[1]]);
    }
  }

  const csv = '﻿' + filas.map(f => f.map(esc).join(';')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `tienda_${PERIODO.desde}_a_${PERIODO.hasta}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

async function pintarPaneles() {
  const enlaces = await API.paneles();
  const partes = [];
  // El dashboard propio de tienda (9030) se retiró el 31-jul-2026: ESTA
  // página es su sustituta, así que no se enlaza a sí misma.
  if (enlaces.visitantes) partes.push(html`<a href="${enlaces.visitantes}" target="_blank" rel="noopener">👥 Analítica de visitantes (detalle, capturas, VLM)</a>`);
  document.getElementById('paneles').innerHTML = partes.join('');
}
/* ── Guías «¿Cómo leer esta sección?» (14-ago) ─────────────────────────
 * Nada se da por sabido: cada bloque explica sus términos con un dibujo.
 * Contenido ESTÁTICO del dashboard (no viaja dato del servidor aquí). */
function montarGuias() {
  montarGuia('curva-sec', { ilu: 'pasillos', puntos: [
    'Cada barra es una <b>hora del día</b> y su altura son las <b>entradas reales</b> al área de conteo en ese tramo, sumando los días del periodo elegido.',
    'Antes esta sección contaba FOTOS por hora sobre las 500 capturas más recientes, mezclando todos los días: un martes flojo heredaba el pico del sábado. Ya no.',
    'La <b>hora punta</b> es la de más entradas: ahí conviene tener más personal y las degustaciones. La hora floja es la buena para reponer sin estorbar.',
    '«No se midió» y «cero entradas» son cosas distintas y se dicen distinto: si nadie estaba grabando, la curva lo avisa en vez de pintar una barra plana.',
  ]});
  montarGuia('hoy', { ilu: 'pasillos', puntos: [
    'Todo lo de esta página responde al <b>periodo</b> elegido arriba (hoy, ayer, 7 o 30 días, o un rango a mano) y a la <b>cámara</b> elegida.',
    'Debajo de cada cifra va su <b>variación contra el periodo anterior de la misma duración</b>: hoy contra ayer, esta semana contra la anterior. Comparar 7 días contra 30 haría que todo «cayera» siempre.',
    'Cuando no hay periodo anterior con datos, dice <b>«sin base de comparación»</b> en vez de inventarse un porcentaje.',
    '<b>Visitantes únicos</b>: personas DISTINTAS del día. El sistema reconoce el rostro: si la misma persona entra y sale tres veces, cuenta <b>una</b>.',
    '<b>Entradas totales</b>: cada cruce del área de conteo suma, aunque sea la misma persona. Por eso siempre es mayor o igual que los visitantes.',
    '<b>Personas ahora</b>: cuántas hay dentro del área en este momento.',
    '<b>Permanencia media</b>: cuánto tiempo se queda la gente, en promedio. Más tiempo = más oportunidad de venta.',
    'El <b>personal de la tienda no suma aquí</b>: todas las cifras son solo de clientes.',
  ]});
  montarGuia('anaqueles-sec', { ilu: 'estante', abierta: true, puntos: [
    'Cada cámara vigila un estante dividido en <b>niveles numerados</b> (Anaquel 1, 2, 3…). Los números no se repiten entre cámaras: el "Anaquel 7" es uno solo en toda la tienda.',
    '<b>Visita</b>: una persona <i>se detuvo</i> frente a ese nivel al menos ~1.5 s. Pasar caminando por el pasillo NO cuenta. Es la señal de que el producto <i>llama la atención</i>.',
    '<b class="naranja">Agarre</b>: con el cuerpo quieto, la mano viajó desde fuera hasta dentro del nivel (un gesto real). Estar parado cerca, el braceo al caminar o el temblor de la mano NO cuentan. Es la señal fuerte: <i>lo tocó</i>.',
    '<b>Conversión</b> = agarres ÷ visitas. Un nivel con muchas visitas y pocas conversiones atrae pero no vende: candidato a cambiar el producto o el precio.',
    'El <b>mapa de calor</b> de cada nivel (clic para ampliarlo) muestra EN QUÉ PARTE del nivel meten la mano: naranja más intenso = más manos.',
    'La misma persona no infla los números: hay tiempos de espera entre visitas y agarres, y el <b>personal</b> (seguridad y pasilleros enrolados) queda <b>excluido</b>.',
  ]});
  montarGuia('personal-sec', { ilu: 'uniforme', puntos: [
    'Aquí se da de alta al personal para que el sistema <b>no lo confunda con clientes</b>: sin esto, un pasillero surtiendo inflaría visitas, agarres y mapas de calor.',
    '<b>🙂 Rostro</b>: identifica a la persona por la cara (foto de frente, con buena luz, una sola persona en la foto).',
    '<b>👕 Uniforme</b>: identifica la vestimenta (bordado y colores) y funciona <b>también de espaldas</b>. Sube el frente y la espalda del uniforme con «+ foto».',
    'Al reconocerlos, además avisa: <i>«Seguridad en el pasillo»</i> y <i>«Empleado surte el Anaquel N»</i>.',
    '<b>Desactivar</b> (vacaciones) conserva las fotos; <b>Borrar</b> las elimina del servidor.',
  ]});
  montarGuia('quien', { ilu: 'demografia', puntos: [
    '<b>Género y edad</b> se estiman con el rostro. Solo entran caras vistas con claridad: es normal que haya menos personas aquí que visitantes totales.',
    'Se cuentan <b>personas distintas</b> del periodo, no fotos: quien viene el lunes y el martes cuenta una vez.',
    '<b>Perfil por pasillo</b>: qué público domina en cada cámara. Esto era imposible hasta ahora, porque la foto solo guardaba el nombre de la ventana y cuatro cámaras pueden llamarse igual.',
    'Precisión antes que cantidad: si el sistema no está seguro de un rostro, prefiere no contarlo.',
  ]});
  montarGuia('mapas', { ilu: 'heatmap', puntos: [
    'Cada mapa acumula <b>por dónde pisa la gente</b> en esa cámara: colores cálidos = zonas más transitadas. Es la foto del recorrido real de los clientes.',
    'Se muestran <b>todas</b> las cámaras con mapa. «Sin registrar» significa que esa cámara aún no está dada de alta como de tienda, pero su mapa existe y no se oculta.',
    '<b>🕐 Histórico</b>: el mapa de cada hora cerrada, para comparar mañana vs tarde.',
    'Clic en cualquier mapa para ampliarlo. El <b>personal enrolado no calienta</b> estos mapas.',
  ]});
  montarGuia('pasillos-sec', { ilu: 'pasillos', puntos: [
    'Ranking de cámaras (una por pasillo) por <b>entradas</b> y <b>permanencia</b> en el periodo elegido.',
    'El pasillo más concurrido es el mejor escaparate: lo que se quiere vender debería estar ahí.',
    'La columna <b>vs anterior</b> dice quién sube y quién baja respecto al periodo anterior de la misma duración. Un pasillo que cae dos periodos seguidos es una señal, no un ruido.',
    'Suma <b>todas las sesiones</b> del rango: antes se leía un solo informe, así que el pasillo que se desconectaba a media mañana aparecía con la mitad de su tráfico.',
    '«Sin registrar» es una cámara que tiene datos pero no está dada de alta como de tienda: no se oculta, se marca.',
  ]});
  montarGuia('capturas-sec', { ilu: 'capturas', puntos: [
    'Las fotos que el sistema tomó de cada persona detectada: son la base de la demografía. Una persona = una foto por visita (no se duplica).',
    'Clic en una foto para ampliarla. Si una foto ya no existe en el servidor, se avisa en su lugar (no queda un hueco roto).',
    'El <b>buscador con IA</b> de abajo responde preguntas en lenguaje normal sobre las capturas («¿alguien lleva mochila?»). Ojo: de noche las cámaras pierden color.',
  ]});
  montarGuia('camaras-sec', { ilu: 'camara', puntos: [
    'Qué cámaras han transmitido, con qué modo de análisis y cuándo fue su último frame.',
    'Una cámara con «última vez» vieja está apagada o desconectada: sus métricas dejaron de crecer desde esa hora.',
  ]});
}
montarGuias();

montarControles();                 // fija el periodo ANTES del primer pintado
pintarFiltroCamaras().catch(console.error);
pintarPaneles().catch(console.error);
pintarPersonal().catch(console.error);

document.getElementById('mas').addEventListener('click', () => {
  mostrar = Math.min(mostrar + 24, TOPE_CAPTURAS);
  pintar().catch(console.error);
});

refrescar(pintar, 15);
// Los anaqueles cambian despacio (contadores del día): cada 30 s basta y no
// compite con el refresco principal. Bucle autoencadenado, no setInterval:
// una pasada lenta no puede solaparse con la siguiente.
(async function cicloAnaqueles() {
  try { await pintarAnaqueles(); } catch (e) { console.error(e); }
  setTimeout(cicloAnaqueles, 30000);
})();
