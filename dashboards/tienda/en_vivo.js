/* dashboards/tienda/en_vivo.js — la página «En vivo» (22-oct-2026).
 *
 * Pide /api/v1/en-vivo cada 3 s (server/src/app/api_en_vivo.py) y lo pinta:
 * si la cámara manda vídeo o la imagen gris de un corte de iVMS, la
 * velocidad de la última hora, la gente, los agarres confirmados, sin
 * confirmar y esperando su bajada, el procesador y la memoria del PC por
 * programa (23-oct-2026) y la temperatura de la GPU.
 *
 * Los «hace X s» se calculan con la hora del SERVIDOR (`ahora` de la
 * respuesta), no con la del navegador: así no dependen del reloj de quien
 * mira. Las gráficas se dibujan con el ancho real de su hueco (SVG a mano,
 * como el resto de los dashboards, que no cargan librerías), para que las
 * letras no encojan en un móvil.
 */
'use strict';

const REFRESCO_S = 3;

/** Más de esto entre dos periodos de 30 s corta la línea: no llegaban imágenes. */
const HUECO_S = 90;

/** La Quadro RTX 4000 del PC de la tienda iba a 82-83 °C, su «temperatura
 *  objetivo», el 21-oct (comentario de vigilante_gpu.ps1). Desde ahí, aviso. */
const TEMP_AVISO_C = 83;

/** Procesador: aviso desde el 85 % y peligro desde el 95 %, el listón con el
 *  que se midió «el tiempo con la CPU ≥ 95 %» el 5-oct (CLAUDE.md 13.5). */
const CPU_AVISO = 85, CPU_MAL = 95;

/** Memoria libre: el PC se quedó sin memoria el 8-oct con 0,7 GB libres, y el
 *  1-oct llegó a 420 MB. Peligro por debajo de 1 GB, aviso por debajo de 2. */
const RAM_MAL_MB = 1024, RAM_AVISO_MB = 2048;

const esc = v => UI.esc(v ?? '');

function num(v, decimales = 0) {
  if (v === null || v === undefined || isNaN(Number(v))) return '—';
  return Number(v).toLocaleString('es', {
    minimumFractionDigits: decimales, maximumFractionDigits: decimales });
}

function hora(epoch, conSegundos = true) {
  if (!epoch) return '—';
  const opciones = { hour: '2-digit', minute: '2-digit' };
  if (conSegundos) opciones.second = '2-digit';
  return new Date(epoch * 1000).toLocaleTimeString('es', opciones);
}

function hace(segundos) {
  if (segundos === null || segundos === undefined || isNaN(segundos)) return '';
  const s = Math.max(0, Math.round(segundos));
  if (s < 60) return `hace ${s} s`;
  const min = Math.floor(s / 60);
  if (min < 60) return `hace ${min} min`;
  return `hace ${Math.floor(min / 60)} h ${String(min % 60).padStart(2, '0')} min`;
}

/** `win-iVMS-4200-c3-i58bf7b` → «iVMS · cámara 3». Si no se reconoce, el id. */
function nombreCamara(id) {
  const canal = /-c(\d+)-i[0-9a-z]+$/i.exec(id);
  const fuente = /iVMS/i.test(id) ? 'iVMS'
    : /Media_Player/i.test(id) ? 'Media Player'
    : /VLC/i.test(id) ? 'VLC' : null;
  if (!fuente) return id;
  return canal ? `${fuente} · cámara ${canal[1]}` : fuente;
}

// ── Gráficas ─────────────────────────────────────────────────────────

/** Huecos de gráfica de este refresco: id → {puntos, opciones}. */
const graficas = new Map();

function hueco(puntos, opciones) {
  const id = `g${graficas.size}`;
  graficas.set(id, { puntos, opciones });
  return `<div class="ev-grafica" data-grafica="${id}"></div>`;
}

/** Paso «redondo» de la escala: 1, 2 o 5 por una potencia de 10. */
function pasoRedondo(bruto) {
  const pot = Math.pow(10, Math.floor(Math.log10(bruto)));
  const f = bruto / pot;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pot;
}

function svgLinea(ancho, puntos, { color, decimales = 0, unidad = '', ahora, titulo, max: tope }) {
  const validos = puntos.filter(p => p.v !== null && p.v !== undefined && !isNaN(p.v));
  if (!validos.length) return '<div class="vacio">sin datos</div>';
  const W = Math.max(260, ancho), H = 150;
  const iz = 40, de = 62, ar = 10, ab = 22;          // márgenes
  const t1 = ahora, t0 = ahora - 3600;               // la última hora
  const valores = validos.map(p => p.v);
  // `tope`: una escala fija (el procesador, de 0 a 100 %); si no, la de los datos.
  const paso = pasoRedondo(tope ? tope / 4 : (Math.max(...valores) || 1) / 3);
  const max = tope || Math.max(paso, Math.ceil(Math.max(...valores) / paso) * paso);
  const x = t => iz + (Math.min(Math.max(t, t0), t1) - t0) / (t1 - t0) * (W - iz - de);
  const y = v => ar + (1 - v / max) * (H - ar - ab);

  let svg = '';
  for (let v = 0; v <= max + paso / 2; v += paso) {
    svg += `<line class="rejilla" x1="${iz}" x2="${W - de}" y1="${y(v)}" y2="${y(v)}"/>`;
    svg += `<text class="eje" x="${iz - 6}" y="${y(v) + 4}" text-anchor="end">${num(v, paso < 1 ? 1 : 0)}</text>`;
  }
  for (const [t, ancla] of [[t0, 'start'], [(t0 + t1) / 2, 'middle'], [t1, 'end']]) {
    svg += `<text class="eje" x="${x(t)}" y="${H - 5}" text-anchor="${ancla}">${hora(t, false)}</text>`;
  }
  // Un hueco sin imágenes corta la línea en vez de unir los dos lados.
  let d = '', previo = null;
  for (const p of validos) {
    if (p.t < t0) continue;
    d += (previo && p.t - previo.t <= HUECO_S ? 'L' : 'M') + `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`;
    previo = p;
  }
  const fin = validos[validos.length - 1];
  svg += `<path class="linea" d="${d}" style="stroke:${color}"/>`;
  svg += `<circle class="punto-final" cx="${x(fin.t)}" cy="${y(fin.v)}" r="4" style="fill:${color}"/>`;
  svg += `<text class="valor-final" x="${x(fin.t) + 8}" y="${y(fin.v) + 4}" style="fill:${color}">${num(fin.v, decimales)}${unidad}</text>`;
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(titulo)}">${svg}</svg>`;
}

function dibujarGraficas() {
  for (const el of document.querySelectorAll('[data-grafica]')) {
    const g = graficas.get(el.dataset.grafica);
    if (g) el.innerHTML = svgLinea(el.clientWidth, g.puntos, g.opciones);
  }
}

// ── Una cámara ───────────────────────────────────────────────────────

function estadoCamara(cam, ahora) {
  const v = cam.video;
  if (!v) return '<span class="ev-estado parada">SIN DATOS</span>';
  const textos = { video: 'VÍDEO', gris: 'IMAGEN GRIS · corte de iVMS',
                   vacia: 'CAPTURA VACÍA · ventana minimizada', parada: 'SIN IMÁGENES' };
  let desde;
  if (v.estado === 'parada') {
    desde = v.desde
      ? `la última llegó a las <b>${hora(v.desde)}</b> (${hace(ahora - v.desde)})`
      : 'no ha llegado ninguna desde que arrancó el servidor';
  } else {
    // La primera vez que se mira una cámara no se sabe cuándo empezó su
    // estado: «al menos desde» (api_en_vivo.py, «Desde cuándo»).
    desde = `${v.cambio_visto ? 'desde' : 'al menos desde'} las <b>${hora(v.desde)}</b> (${hace(ahora - v.desde)})`;
    if (v.ultima_imagen) desde += ` · última imagen ${hace(ahora - v.ultima_imagen)}`;
  }
  let aviso = '';
  if (v.estado === 'gris') {
    aviso = '<div class="ev-aviso">iVMS perdió el vídeo y manda su pantalla gris: mientras dure no se cuenta nada. '
      + 'La velocidad de este rato no se puede comparar con la de vídeo, porque la imagen gris pesa poco.</div>';
  } else if (v.estado === 'vacia') {
    aviso = `<div class="ev-aviso mal">Llegan imágenes vacías (${num(v.ultima_imagen_kb, 1)} KB): `
      + 'pasa cuando la ventana de iVMS está minimizada. Restáurala; mientras tanto no se cuenta nada.</div>';
  } else if (v.estado === 'parada') {
    aviso = cam.conectada
      ? '<div class="ev-aviso mal">No llegan imágenes: ¿IA apagada en el recuadro, o el programa de la tienda colgado?</div>'
      : '<div class="ev-aviso mal">La cámara ya no está conectada al servidor.</div>';
  }
  return `<span class="ev-estado ${esc(v.estado)}">${textos[v.estado] || esc(v.estado)}</span>`
    + `<span class="ev-desde">${desde}</span>${aviso}`;
}

function kpi(valor, nombre, sub = '') {
  return `<div class="kpi"><div class="valor">${valor}</div>`
    + `<div class="nombre">${esc(nombre)}</div><div class="ev-sub">${sub}</div></div>`;
}

function kpisCamara(cam, ahora) {
  const u = cam.velocidad?.ultimo;
  const g = cam.gente_ahora;
  return '<div class="kpis">'
    + kpi(u ? num(u.procesados_s, 1) : '—', 'imágenes por segundo',
          u ? `recibidas ${num(u.recibidos_s, 1)}/s · descartadas ${num(u.descartados)}` : 'aparece a los 30 s de llegar imágenes')
    + kpi(u?.proceso_ms_mediana != null ? `${num(u.proceso_ms_mediana)} ms` : '—', 'en el servidor por imagen',
          u ? `mediana · el 90 % por debajo de ${num(u.proceso_ms_p90)} ms` : '')
    + kpi(u?.kb_por_frame != null ? `${num(u.kb_por_frame)} KB` : '—', 'por imagen',
          u ? `últimos 30 s, cerrados ${hace(ahora - u.t)}` : '')
    + kpi(g ? num(g.en_imagen) : '—', 'personas en imagen', 'seguidas ahora')
    + kpi(g?.dentro_area != null ? num(g.dentro_area) : '—', 'dentro del área de conteo', 'ahora')
    + '</div>';
}

function tarjetaVelocidad(cam, ahora) {
  const h = cam.velocidad?.historial || [];
  let cuerpo;
  if (!h.length) {
    cuerpo = '<div class="vacio">Todavía no hay ningún periodo de 30 s cerrado: aparece a los 30 s de llegar imágenes.</div>';
  } else {
    cuerpo = '<div class="ev-grafica-titulo">Imágenes procesadas por segundo</div>'
      + hueco(h.map(p => ({ t: p.t, v: p.procesados_s })),
              { color: 'var(--acento)', decimales: 1, ahora, titulo: 'Imágenes por segundo, última hora' })
      + '<div class="ev-grafica-titulo">Milisegundos en el servidor por imagen (mediana)</div>'
      + hueco(h.map(p => ({ t: p.t, v: p.proceso_ms_mediana })),
              { color: 'var(--ev-ms)', unidad: ' ms', ahora, titulo: 'Milisegundos por imagen, última hora' });
  }
  return `<div class="tarjeta"><h3>Velocidad, última hora</h3>
    <p class="nota">Un punto cada 30 s, el mismo que la línea <code>FPS [...]</code> del log. Un hueco en la línea es un rato sin imágenes.
    Se borra al reiniciar el servidor.</p>${cuerpo}</div>`;
}

function tarjetaAgarres(cam, ahora) {
  const a = cam.agarres;
  let cuerpo;
  if (cam.modo && cam.modo !== 'Anaqueles') {
    cuerpo = `<div class="vacio">Esta cámara está en modo ${esc(cam.modo)}: los agarres solo se cuentan en modo Anaqueles.</div>`;
  } else if (!a) {
    cuerpo = '<div class="vacio">Sin datos: la cámara no está conectada.</div>';
  } else if (!a.areas.length) {
    cuerpo = `<div class="vacio">Hoy (${esc(a.fecha)}) aún no hay visitas ni agarres en ningún anaquel.</div>`;
  } else {
    const t = { visitas: 0, confirmados: 0, sin_confirmar: 0, esperando: 0 };
    const filas = a.areas.map(ar => {
      for (const k in t) t[k] += ar[k] || 0;
      const espera = ar.esperando
        ? `<span class="ev-espera">${num(ar.esperando)} · lleva ${num(ar.esperando_max_s)} s de ${num(a.ventana_s)}</span>`
        : '<span class="sin">—</span>';
      return `<tr><td>${esc(ar.nombre)}</td><td class="num">${num(ar.visitas)}</td>
        <td class="num confirmados">${num(ar.confirmados)}</td><td class="num sin">${num(ar.sin_confirmar)}</td>
        <td>${espera}</td></tr>`;
    }).join('');
    const ultimos = (a.ultimos_confirmados || []).map(u =>
      `<li><b>${esc(u.hora)}</b>${esc(u.area)} · ${u.producto ? esc(u.producto) : 'sin producto'}</li>`).join('');
    cuerpo = `<table class="ev-agarres"><thead><tr><th>Anaquel</th><th class="num">Visitas</th>
        <th class="num">Confirmados</th><th class="num">Sin confirmar</th><th>Esperando bajada</th></tr></thead>
      <tbody>${filas}<tr class="total"><td>Total</td><td class="num">${num(t.visitas)}</td>
        <td class="num confirmados">${num(t.confirmados)}</td><td class="num">${num(t.sin_confirmar)}</td>
        <td>${t.esperando ? `<span class="ev-espera">${num(t.esperando)}</span>` : '—'}</td></tr></tbody></table>
      <div class="ev-ultimos">Últimos confirmados${ultimos ? `<ul>${ultimos}</ul>` : ': ninguno hoy.'}</div>`;
  }
  const ventana = a ? num(a.ventana_s) : '180';
  return `<div class="tarjeta"><h3>Agarres de hoy${a ? ` <small class="sin">(${esc(a.fecha)})</small>` : ''}</h3>
    <p class="nota"><b class="confirmados">Confirmado</b>: tras el gesto, el stock vio que faltaba algo en la balda; es el que suma.
    <b>Sin confirmar</b>: pasaron ${ventana} s de vigilancia sin ver una bajada; no suma.
    <b>Esperando bajada</b>: gestos de ahora mismo, aún sin decidir (un reinicio del servidor los pierde).</p>${cuerpo}</div>`;
}

function tarjetaGente(cam) {
  const ahora = cam.gente_ahora, hoy = cam.gente_hoy;
  const cifra = (valor, texto, clase = '') => `<div class="${clase}"><b>${valor}</b>${esc(texto)}</div>`;
  let cifras = '';
  cifras += cifra(ahora ? num(ahora.en_imagen) : '—', 'en imagen ahora', 'ahora');
  cifras += cifra(ahora?.dentro_area != null ? num(ahora.dentro_area) : '—', 'dentro del área ahora', 'ahora');
  cifras += cifra(hoy ? num(hoy.entradas) : '—', 'entradas al área hoy');
  cifras += cifra(hoy ? num(hoy.visitantes_unicos) : '—', 'visitantes únicos hoy');
  cifras += cifra(hoy?.permanencia_media_s ? UI.duracion(hoy.permanencia_media_s) : '—', 'permanencia media');
  let barras = '<div class="vacio">sin datos de hoy</div>';
  const c = hoy?.clasificadas;
  if (c && c.total) {
    const orden = Object.entries(c.genero).sort((p, q) => q[1] - p[1]);
    barras = '<div class="barras">' + orden.map(([g, n]) =>
      `<div class="barra"><span>${esc(g)}</span><div class="pista"><div class="relleno" style="width:${(100 * n / c.total).toFixed(1)}%"></div></div>`
      + `<span class="cifra">${num(n)}</span></div>`).join('') + '</div>';
  } else if (c) {
    barras = '<div class="vacio">Hoy aún no se ha clasificado a nadie.</div>';
  }
  return `<div class="tarjeta ev-gente"><h3>Gente</h3>
    <p class="nota">«Ahora» se lee de la imagen en curso; «hoy», de la base de la tienda (puede ir 10 s atrasado).</p>
    <div class="ev-cifras">${cifras}</div>
    <div class="ev-grafica-titulo">Fichas clasificadas hoy por género${c ? ` (${num(c.total)})` : ''}</div>
    <p class="nota">Una ficha por persona seguida: quien pierde su número y vuelve deja otra.</p>${barras}</div>`;
}

function bloqueCamara(cam, ahora) {
  const modo = cam.modo ? ` · modo ${esc(cam.modo)}` : '';
  return `<section class="ev-camara">
    <div class="ev-cabeza"><h2>${esc(nombreCamara(cam.camera_id))}<small>${esc(cam.camera_id)}${modo}</small></h2>
      ${estadoCamara(cam, ahora)}</div>
    ${kpisCamara(cam, ahora)}
    <div class="ev-rejilla">${tarjetaAgarres(cam, ahora)}${tarjetaVelocidad(cam, ahora)}${tarjetaGente(cam)}</div>
  </section>`;
}

// ── Procesador y memoria ─────────────────────────────────────────────

/** Una barra por programa; `total` es lo que llena la barra entera. */
function barrasProgramas(programas, campo, total, formato) {
  return '<div class="barras ev-programas">' + programas.map(p => {
    const v = p[campo] || 0;
    const ancho = total ? Math.min(100, 100 * v / total).toFixed(1) : 0;
    return `<div class="barra${p.clave === 'resto' ? ' resto' : ''}"><span>${esc(p.nombre)}</span>`
      + `<div class="pista"><div class="relleno" style="width:${ancho}%"></div></div>`
      + `<span class="cifra">${p.procesos === 0 ? 'cerrado' : formato(v)}</span></div>`;
  }).join('') + '</div>';
}

function pintarPC(pc, ahora) {
  const el = document.getElementById('pc');
  if (!pc || !pc.ultima) {
    el.innerHTML = '<div class="vacio">Sin datos todavía: el servidor empieza a medir 5 s después de arrancar '
      + '(y no mide si le falta la librería psutil).</div>';
    return;
  }
  const u = pc.ultima;
  const gb = mb => num(mb / 1024, 1);
  const claseCpu = u.cpu_pct >= CPU_MAL ? 'mal' : u.cpu_pct >= CPU_AVISO ? 'aviso' : 'ok';
  const claseRam = u.ram_libre_mb < RAM_MAL_MB ? 'mal' : u.ram_libre_mb < RAM_AVISO_MB ? 'aviso' : 'ok';
  const sinHistorial = '<div class="vacio">El primer punto sale al cerrarse el primer minuto.</div>';
  const graficaCpu = pc.historial.length
    ? hueco(pc.historial.map(p => ({ t: p.t, v: p.cpu_pct })),
            { color: 'var(--acento)', unidad: ' %', ahora, max: 100, titulo: 'Procesador del PC, última hora' })
    : sinHistorial;
  const graficaRam = pc.historial.length
    ? hueco(pc.historial.map(p => ({ t: p.t, v: p.ram_libre_mb / 1024 })),
            { color: 'var(--ev-ram)', decimales: 1, unidad: ' GB', ahora, titulo: 'Memoria libre, última hora' })
    : sinHistorial;
  el.innerHTML = `<div class="ev-rejilla">
    <div class="tarjeta ev-pc"><h3>Procesador</h3>
      <div class="ev-grande ${claseCpu}">${num(u.cpu_pct)} %</div>
      <p class="nota">del PC entero (${num(u.nucleos)} hilos), medido ${hace(ahora - u.t)}</p>
      ${barrasProgramas(u.programas, 'cpu_pct', 100, v => `${num(v, 1)} %`)}
      <div class="ev-grafica-titulo">Procesador del PC, última hora (media de cada minuto)</div>${graficaCpu}
    </div>
    <div class="tarjeta ev-pc"><h3>Memoria (RAM)</h3>
      <div class="ev-grande ${claseRam}">${gb(u.ram_libre_mb)} GB libres</div>
      <p class="nota">en uso ${gb(u.ram_usada_mb)} de ${gb(u.ram_total_mb)} GB, medido ${hace(ahora - u.t)}</p>
      ${barrasProgramas(u.programas, 'ram_mb', u.ram_total_mb, v => `${gb(v)} GB`)}
      <div class="ev-grafica-titulo">Memoria libre, última hora (GB)</div>${graficaRam}
    </div></div>`;
}

// ── GPU ──────────────────────────────────────────────────────────────

function pintarGPU(gpu, ahora) {
  const el = document.getElementById('gpu');
  if (!gpu) {
    el.innerHTML = '<div class="vacio">Sin datos: no existe <code>server\\logs\\gpu_temp.csv</code> o está vacío '
      + '(el vigilante del servidor no está corriendo).</div>';
    return;
  }
  const u = gpu.ultima;
  const clase = u.por_calor ? 'mal' : (u.temp_c >= TEMP_AVISO_C ? 'aviso' : 'ok');
  const limitacion = u.por_calor
    ? '<span class="pill error">frenando por calor</span>'
    : (u.motivos.length ? esc(u.motivos.join(', ')) : 'ninguna');
  const viejo = ahora - u.t > 180
    ? '<div class="ev-aviso mal">La última medida tiene más de 3 min: el vigilante parece parado.</div>' : '';
  const dato = (texto, valor) => `<div>${esc(texto)}: <b>${valor}</b></div>`;
  el.innerHTML = `<div class="tarjeta"><div class="ev-gpu"><div>
      <div class="ev-temp ${clase}">${num(u.temp_c)} °C</div>
      <div class="ev-gpu-datos">
        ${dato('uso', `${num(u.uso_pct)} %`)}${dato('consumo', `${num(u.consumo_w)} W`)}
        ${dato('reloj', `${num(u.reloj_mhz)} MHz`)}${dato('ventilador', `${num(u.ventilador_pct)} %`)}
        ${dato('memoria', `${num(u.memoria_mb)} MB`)}<div>limitación: ${limitacion}</div>
        <div>medida a las <b>${hora(u.t)}</b> (${hace(ahora - u.t)})</div></div>${viejo}</div>
      <div><div class="ev-grafica-titulo">Temperatura, última hora (°C)</div>
        ${hueco(gpu.historial.map(p => ({ t: p.t, v: p.temp_c })),
                { color: 'var(--ev-temp)', unidad: ' °C', ahora, titulo: 'Temperatura de la GPU, última hora' })}
        <div class="ev-grafica-titulo">Uso, última hora (%)</div>
        ${hueco(gpu.historial.map(p => ({ t: p.t, v: p.uso_pct })),
                { color: 'var(--acento)', unidad: ' %', ahora, titulo: 'Uso de la GPU, última hora' })}</div>
    </div></div>`;
}

// ── Bucle ────────────────────────────────────────────────────────────

async function pintar() {
  if (document.hidden) return;              // pestaña oculta: no cargar al servidor
  const d = await API.leer('en-vivo');
  graficas.clear();
  const cont = document.getElementById('camaras');
  cont.innerHTML = d.camaras.length
    ? d.camaras.map(c => bloqueCamara(c, d.ahora)).join('')
    : '<div class="vacio">Ninguna cámara ha mandado imágenes desde que arrancó el servidor: '
      + 'enciende la IA de un recuadro en el programa de la tienda.</div>';
  pintarPC(d.pc, d.ahora);
  pintarGPU(d.gpu, d.ahora);
  dibujarGraficas();
}

refrescar(pintar, REFRESCO_S);
