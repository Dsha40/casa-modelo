/* dashboards/shared/util.js — piezas del rediseño (13-ago-2026), comunes a
 * TODOS los dashboards. Se carga después de api.js y antes del app.js de
 * cada página.
 *
 *  - Lightbox: cualquier imagen, canvas o HTML se amplía con un clic. El
 *    markup se inyecta solo (`montarLightbox`): las páginas no lo declaran.
 *  - IMAGEN_ROTA + cadenas de respaldo: ninguna imagen rota en silencio.
 *  - pintarMapasTodos: la galería de mapas de calor muestra SIEMPRE todos
 *    los mapas del servidor (el filtro es un chip del usuario, no un
 *    descarte del código — antes los huérfanos desaparecían sin aviso).
 *  - dibujarCalor: rejillas numéricas como canvas nítido de un solo tono.
 */
'use strict';

const fmt = n => (n === null || n === undefined || isNaN(Number(n)))
  ? '—' : Number(n).toLocaleString('es');

/* Placeholder legible para una imagen que el servidor ya no tiene. */
const IMAGEN_ROTA = 'data:image/svg+xml,' + encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180">
     <rect width="100%" height="100%" fill="#0a0d12"/>
     <text x="50%" y="46%" fill="#8b949e" font-family="sans-serif" font-size="13"
           text-anchor="middle">imagen no disponible</text>
     <text x="50%" y="62%" fill="#484f58" font-family="sans-serif" font-size="11"
           text-anchor="middle">el archivo ya no está en el servidor</text>
   </svg>`);

/* ── Plantillas que escapan solas ─────────────────────────────────────
 * El HTML de estas páginas se arma concatenando cadenas y el escape era
 * MANUAL (`UI.esc` a mano en cada interpolación). Funciona mientras nadie se
 * olvide, que es una garantía que dura lo que dura la siguiente sección.
 *
 *     html`<b>${nombre}</b>`            <- `nombre` va escapado, siempre
 *     html`<div>${html.crudo(trozo)}</div>`  <- lo ya-HTML, marcado a mano
 *
 * Las listas se unen solas, así que `${filas.map(f => html`<tr>…</tr>`)}`
 * hace lo que uno espera. El código viejo sigue funcionando igual: esto se
 * añade, no sustituye a nada de golpe. */
function html(trozos, ...valores) {
  let salida = trozos[0];
  for (let i = 0; i < valores.length; i++) {
    salida += _valorSeguro(valores[i]) + trozos[i + 1];
  }
  return _marcado(salida);
}

/** Marca un texto como «esto YA es HTML, no lo vuelvas a escapar».
 *
 *  Se devuelve un objeto String y no una cadena normal porque hacía falta
 *  distinguir lo ya-escapado de lo que aún hay que escapar: sin esa marca,
 *  anidar plantillas —que es el patrón más común aquí,
 *  `${filas.map(f => html`<tr>…</tr>`)}`— escapaba el HTML interior y la
 *  página salía con `&lt;tr&gt;` a la vista. Un objeto String se comporta
 *  como una cadena en todo lo que hacemos con él (`innerHTML =`, `+`,
 *  `join('')`), así que quien lo use no tiene que saber que existe. */
function _marcado(texto) {
  const marcado = new String(texto);       // eslint-disable-line no-new-wrappers
  marcado.__esHtml = true;
  return marcado;
}

html.crudo = texto => _marcado(texto);

function _valorSeguro(v) {
  if (v === null || v === undefined || v === false) return '';
  if (Array.isArray(v)) return v.map(_valorSeguro).join('');
  if (v && v.__esHtml) return String(v);   // ya es HTML: pasa entero
  return UI.esc(v);
}

/* ── Imágenes rotas: UN oyente para toda la página ────────────────────
 * Antes cada <img> llevaba su propio `onerror=` en línea, y dentro iba el
 * data-URI del placeholder ENTERO: 636 caracteres repetidos por imagen, que
 * con 500 capturas son ~311 KB de atributo duplicado en el DOM. Además era la
 * construcción más frágil del código: JavaScript dentro de una cadena, dentro
 * de un atributo, dentro de otra cadena.
 *
 * Ahora las imágenes solo DECLARAN qué quieren:
 *   data-respaldo="…"    -> si falla, se intenta esa otra fuente (una vez)
 *   data-rota-texto="…"  -> si falla, el hueco se sustituye por ese aviso
 *   (nada)               -> si falla, el placeholder legible de siempre
 *
 * El evento `error` NO burbujea: hay que escucharlo en fase de captura. */
function montarRespaldoDeImagenes() {
  if (window.__respaldoImagenesMontado) return;
  window.__respaldoImagenesMontado = true;
  document.addEventListener('error', e => {
    const img = e.target;
    if (!img || img.tagName !== 'IMG' || img.dataset.rota === 'si') return;
    if (img.dataset.respaldo) {
      const respaldo = img.dataset.respaldo;
      delete img.dataset.respaldo;      // un solo reintento, nunca un bucle
      img.src = respaldo;
      return;
    }
    img.dataset.rota = 'si';
    if (img.dataset.rotaTexto) {
      const hueco = img.parentElement;
      if (hueco) hueco.innerHTML = `<div class="img-rota">${UI.esc(img.dataset.rotaTexto)}</div>`;
      return;
    }
    img.src = IMAGEN_ROTA;
  }, true);
}

/* ── Lightbox ─────────────────────────────────────────────────────────── */
function montarLightbox() {
  if (document.getElementById('lightbox')) return;
  const div = document.createElement('div');
  div.className = 'lightbox';
  div.id = 'lightbox';
  div.innerHTML = `
    <div class="lb-caja">
      <div class="lb-cab">
        <h3 id="lb-titulo"></h3>
        <span class="lb-sub" id="lb-sub"></span>
        <button class="lb-cerrar" id="lb-cerrar">✕ cerrar (Esc)</button>
      </div>
      <div class="lb-cuerpo" id="lb-cuerpo"></div>
      <div class="lb-pie" id="lb-pie"></div>
    </div>`;
  document.body.appendChild(div);
  document.getElementById('lb-cerrar').addEventListener('click', cerrarLightbox);
  div.addEventListener('click', e => {
    if (e.target.id === 'lightbox') cerrarLightbox();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') cerrarLightbox();
  });
}

/** abrirLightbox({titulo, sub, src | nodo, pie}). `nodo` admite canvas/HTML. */
function abrirLightbox({ titulo = '', sub = '', src = null, nodo = null, pie = '' }) {
  montarLightbox();
  document.getElementById('lb-titulo').textContent = titulo;
  document.getElementById('lb-sub').textContent = sub;
  document.getElementById('lb-pie').innerHTML = pie;
  const cuerpo = document.getElementById('lb-cuerpo');
  cuerpo.innerHTML = '';
  if (nodo) {
    cuerpo.appendChild(nodo);
  } else if (src) {
    const img = document.createElement('img');
    img.src = src;
    img.alt = titulo;
    img.onerror = () => { img.onerror = null; img.src = IMAGEN_ROTA; };
    cuerpo.appendChild(img);
  }
  document.getElementById('lightbox').classList.add('abierto');
}

function cerrarLightbox() {
  const lb = document.getElementById('lightbox');
  if (!lb) return;
  lb.classList.remove('abierto');
  document.getElementById('lb-cuerpo').innerHTML = '';
}

/* ── Galería de mapas de calor: TODOS, con filtro y ampliación ────────── */
let _filtroMapas = 'todas';        // 'todas' | 'dominio' | 'otras'

/**
 * Pinta TODOS los mapas de /api/v1/heatmaps con chips de filtro y zoom.
 *  contId    – id del contenedor .heatmaps
 *  filtrosId – id del contenedor de chips (null = sin chips, muestra todo)
 *  mapas     – filas de /api/v1/heatmaps (SIN filtrar)
 *  propios   – Set de device_id del dominio de la página
 *  dominio   – etiqueta del chip del dominio (p. ej. 'De tienda')
 */
function pintarMapasTodos(contId, filtrosId, mapas, propios, dominio = 'Del dominio') {
  const cont = document.getElementById(contId);
  const filtros = filtrosId ? document.getElementById(filtrosId) : null;

  const conDominio = (mapas || []).map(h => ({
    ...h, esPropio: propios.has(h.device_id),
  }));
  const visibles = !filtros ? conDominio : conDominio.filter(h =>
    _filtroMapas === 'todas' ? true
      : _filtroMapas === 'dominio' ? h.esPropio : !h.esPropio);

  if (filtros) {
    const otras = conDominio.filter(h => !h.esPropio).length;
    filtros.innerHTML = `
      <button class="chip ${_filtroMapas === 'todas' ? 'activo' : ''}" data-f="todas">Todas (${conDominio.length})</button>
      <button class="chip ${_filtroMapas === 'dominio' ? 'activo' : ''}" data-f="dominio">${UI.esc(dominio)} (${conDominio.length - otras})</button>
      <button class="chip ${_filtroMapas === 'otras' ? 'activo' : ''}" data-f="otras">Otras cámaras (${otras})</button>
      <span class="cuenta">${visibles.length} mapa${visibles.length === 1 ? '' : 's'} en pantalla — ninguno se oculta</span>`;
    for (const btn of filtros.querySelectorAll('.chip')) {
      btn.addEventListener('click', () => {
        _filtroMapas = btn.dataset.f;
        pintarMapasTodos(contId, filtrosId, mapas, propios, dominio);
      });
    }
  }

  if (!visibles.length) {
    cont.innerHTML = '<div class="vacio">sin mapas de calor en este filtro — se generan solos mientras las cámaras transmiten</div>';
    return;
  }

  cont.innerHTML = visibles.map(h => `
    <figure>
      <div class="zona-img" data-src="${h.url}?t=${h.modificado}"
           data-titulo="${UI.esc(h.camera_name || h.device_id)}"
           data-sub="mapa de calor acumulado">
        <img src="${h.url}?t=${h.modificado}" alt="mapa de calor ${UI.esc(h.device_id)}"
             loading="lazy"
             data-rota-texto="este mapa aún no tiene imagen en el servidor">
      </div>
      <div class="etiquetas">
        ${h.huerfano ? '<span class="pill aviso">sin registrar</span>' : ''}
        ${!h.esPropio && !h.huerfano ? '<span class="pill">otro dominio</span>' : ''}
      </div>
      <figcaption>
        <b>${UI.esc(h.camera_name || h.device_id)}</b>
        ${h.horas_historico
          ? `<button class="boton btn-historial" data-id="${UI.esc(h.device_id)}" data-nombre="${UI.esc(h.camera_name || h.device_id)}">🕐 histórico (${h.horas_historico} h)</button>`
          : '<small style="color:var(--apagado)">histórico: aún sin horas cerradas</small>'}
      </figcaption>
    </figure>`).join('');

  for (const zona of cont.querySelectorAll('.zona-img')) {
    zona.addEventListener('click', () => abrirLightbox({
      titulo: zona.dataset.titulo,
      sub: zona.dataset.sub,
      src: zona.dataset.src,
      pie: `<a href="${zona.dataset.src}" target="_blank" rel="noopener">abrir el PNG original en otra pestaña</a>`,
    }));
  }
  for (const btn of cont.querySelectorAll('.btn-historial')) {
    btn.addEventListener('click', () =>
      abrirHistoricoMapa(btn.dataset.id, btn.dataset.nombre));
  }
}

async function abrirHistoricoMapa(id, nombre) {
  const nodo = document.createElement('div');
  nodo.className = 'heatmaps';
  nodo.style.width = '100%';
  nodo.innerHTML = '<div class="vacio">cargando histórico…</div>';
  abrirLightbox({
    titulo: `Histórico por horas — ${nombre}`,
    sub: 'un mapa por cada hora cerrada',
    nodo,
  });
  try {
    const { horas } = await API.leer(`heatmaps/${encodeURIComponent(id)}/historico`);
    nodo.innerHTML = horas.map(x => `
      <figure>
        <div class="zona-img"><img src="${x.url}" alt="${UI.esc(x.stamp)}" loading="lazy"></div>
        <figcaption><b>${UI.esc(x.stamp.replace('_', ' · '))} h</b>
          ${x.muestras ? `<small>${UI.esc(x.muestras)} muestras</small>` : ''}</figcaption>
      </figure>`).join('') || '<div class="vacio">sin horas guardadas todavía</div>';
  } catch (e) {
    nodo.innerHTML = `<div class="error-carga">no se pudo leer el histórico: ${UI.esc(e.message)}</div>`;
  }
}

/* ── Rejillas numéricas como canvas (anaqueles y afines) ──────────────── */
/** Rejilla de calor nítida, un solo tono (naranja), celdas con separación.
 *  `conCifras` pinta el valor sobre las celdas más calientes (ampliado). */
function dibujarCalor(grid, escala, conCifras = false) {
  const filas = grid.length, cols = grid[0]?.length || 0;
  const cv = document.createElement('canvas');
  cv.width = cols * escala;
  cv.height = filas * escala;
  const ctx = cv.getContext('2d');
  const tope = Math.max(1, ...grid.map(f => Math.max(...f)));
  for (let f = 0; f < filas; f++) {
    for (let c = 0; c < cols; c++) {
      const v = grid[f][c];
      if (v <= 0) continue;
      ctx.fillStyle = `rgba(255, 140, 0, ${(0.15 + 0.85 * (v / tope)).toFixed(3)})`;
      ctx.fillRect(c * escala, f * escala, escala - 1, escala - 1);
    }
  }
  if (conCifras && escala >= 22) {
    ctx.font = `${Math.round(escala * 0.42)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let f = 0; f < filas; f++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[f][c];
        if (v < tope * 0.5) continue;      // solo las celdas más calientes
        ctx.fillStyle = '#1a0e00';
        ctx.fillText(String(Math.round(v)), c * escala + escala / 2,
                     f * escala + escala / 2);
      }
    }
  }
  return cv;
}

/* El oyente de imágenes rotas se engancha al cargar: es de documento, no
 * depende de que exista ninguna sección, y así ninguna página tiene que
 * acordarse de pedirlo. */
montarRespaldoDeImagenes();
