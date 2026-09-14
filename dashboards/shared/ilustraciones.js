/* dashboards/shared/ilustraciones.js — pictogramas y guías (14-ago-2026).
 *
 * Para que CUALQUIER persona entienda qué mide cada sección sin saber nada
 * del sistema: cada bloque puede montar una guía «¿Cómo leer esto?» con un
 * dibujo (SVG inline, sin internet) y explicaciones en cristiano.
 *
 * Uso desde el app.js de cada dashboard, DESPUÉS del <p class="porque">:
 *   montarGuia('anaqueles-sec', { ilu: 'estante', abierta: true, puntos: [
 *     '<b>Visita</b>: alguien se acercó al nivel…', … ] });
 *
 * Los SVG usan currentColor (hereda el gris de texto suave) y tres clases
 * de acento: .acc (color del dominio), .warm (naranja interacción) y
 * .ok (verde). Los estilos viven en shared/moderno.css (.guia, .ilu).
 */
'use strict';

const ILU = {
  /* Estante con niveles numerados y una mano entrando al nivel 2. */
  estante: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 12v66M98 12v66"/>
    <path d="M22 32h76M22 54h76M22 76h76"/>
    <rect x="30" y="20" width="10" height="11" rx="1"/>
    <rect x="44" y="22" width="8" height="9" rx="1"/>
    <rect x="72" y="66" width="9" height="9" rx="1"/>
    <rect x="85" y="64" width="8" height="11" rx="1"/>
    <path class="warm" d="M4 45h34m0 0-8-6m8 6-8 6"/>
    <rect class="warm" x="42" y="42" width="10" height="11" rx="1"/>
    <circle class="acc" cx="110" cy="27" r="8"/><text x="110" y="31" font-size="10" text-anchor="middle" fill="currentColor" stroke="none">1</text>
    <circle class="acc" cx="110" cy="49" r="8"/><text x="110" y="53" font-size="10" text-anchor="middle" fill="currentColor" stroke="none">2</text>
    <circle class="acc" cx="110" cy="71" r="8"/><text x="110" y="75" font-size="10" text-anchor="middle" fill="currentColor" stroke="none">3</text>
  </svg>`,

  /* Izquierda: persona que solo MIRA (visita). Derecha: mano que ENTRA
     al estante (agarre). */
  visita_agarre: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M58 8v74" stroke-dasharray="3 5"/>
    <circle cx="22" cy="26" r="7"/>
    <path d="M22 33v22m0-16-9 8m9-8 9 8m-9 8-8 14m8-14 8 14"/>
    <path class="acc" d="M38 30c5-3 5 3 0 0Z M36 26q6-4 8 2" stroke-dasharray="2 3"/>
    <path d="M44 14h8v62" />
    <circle cx="86" cy="24" r="7"/>
    <path d="M86 31v24m0-18 16 6m-16 2-8 22m8-22 8 22"/>
    <path d="M112 12v66m-8-52h8m-8 22h8m-8 22h8"/>
    <path class="warm" d="M102 37l8 3m-8-3 2 6"/>
    <rect class="warm" x="103" y="42" width="7" height="8" rx="1"/>
  </svg>`,

  /* Mapa de calor: encuadre de cámara con manchas calientes. */
  heatmap: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <rect x="8" y="12" width="104" height="66" rx="6"/>
    <circle class="warm" cx="44" cy="46" r="16" opacity=".45"/>
    <circle class="warm" cx="44" cy="46" r="9" opacity=".8"/>
    <circle class="warm" cx="44" cy="46" r="3.5" fill="#ffa028" stroke="none"/>
    <circle class="warm" cx="84" cy="34" r="9" opacity=".5"/>
    <circle class="warm" cx="84" cy="34" r="3" fill="#ffa028" stroke="none" opacity=".8"/>
    <circle class="warm" cx="70" cy="62" r="6" opacity=".4"/>
    <path d="M14 18l8 5m-8-2v-3h3"/>
  </svg>`,

  /* Persona con credencial + escudo: el personal enrolado. */
  personal: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="46" cy="24" r="10"/>
    <path d="M26 74c0-14 9-22 20-22s20 8 20 22"/>
    <rect class="acc" x="40" y="56" width="12" height="9" rx="1.5"/>
    <path class="acc" d="M43 60h6"/>
    <path class="ok" d="M88 30c6 3 10 3 14 0v14c0 10-7 15-7 15s-7-5-7-15Z"/>
    <path class="ok" d="M91 43l3 3 6-7"/>
  </svg>`,

  /* Camisa de uniforme con bordado en el pecho. */
  uniforme: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M45 16 60 24l15-8 20 12-9 14-8-5v41H42V37l-8 5-9-14Z"/>
    <path d="M52 16c2 6 14 6 16 0"/>
    <path class="acc" d="M50 44l4-7 4 7-7-4h6"/>
    <path class="acc" d="M64 48h14m-14 6h10"/>
  </svg>`,

  /* Tres personas: demografía (género/edad). */
  demografia: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="30" cy="30" r="9"/><path d="M14 74c0-12 7-19 16-19s16 7 16 19"/>
    <circle class="acc" cx="63" cy="26" r="10"/><path class="acc" d="M45 74c0-14 8-21 18-21s18 7 18 21"/>
    <circle cx="95" cy="34" r="7"/><path d="M83 74c0-10 5-16 12-16s12 6 12 16"/>
  </svg>`,

  /* Fotos apiladas con recuadro de rostro. */
  capturas: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <rect x="30" y="8" width="66" height="50" rx="4" transform="rotate(4 63 33)"/>
    <rect x="20" y="24" width="66" height="52" rx="4"/>
    <circle cx="45" cy="46" r="8"/>
    <path d="M31 72c2-9 7-14 14-14s12 5 14 14"/>
    <rect class="acc" x="34" y="34" width="24" height="26" rx="2" stroke-dasharray="4 3"/>
  </svg>`,

  /* Cámara CCTV con campo de visión. */
  camara: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 14v18"/>
    <rect x="14" y="20" width="44" height="20" rx="4" transform="rotate(12 36 30)"/>
    <circle cx="54" cy="36" r="3" fill="currentColor" stroke="none"/>
    <path class="acc" d="M60 40 96 74M60 40l44 14" stroke-dasharray="4 4"/>
    <path class="acc" d="M75 78a40 40 0 0 0 24-14" stroke-dasharray="4 4"/>
  </svg>`,

  /* Pasillo entre góndolas con huellas. */
  pasillos: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 10 34 80M106 10 86 80"/>
    <path d="M14 30h18M14 50h22M14 70h26M106 30H88M106 50H84M106 70H80" opacity=".6"/>
    <path class="acc" d="M56 72c2-2 6-2 8 0m-10-16c2-2 6-2 8 0m-6-16c2-2 6-2 8 0m-6-16c2-2 6-2 8 0" />
    <path class="acc" d="M60 78v-4m0-12v-4m0-12v-4m0-12v-4" stroke-dasharray="2 4"/>
  </svg>`,

  /* Embudo con personas que van quedando. */
  embudo: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 14h88l-26 34v24l-18 8h-8v-32Z" opacity=".7"/>
    <circle cx="34" cy="8" r="4"/><circle cx="60" cy="6" r="4"/><circle class="acc" cx="86" cy="8" r="4"/>
    <circle class="acc" cx="60" cy="40" r="4"/>
    <circle class="ok" cx="60" cy="74" r="4"/>
    <path d="M104 34l8 8m0-8-8 8" opacity=".6"/>
  </svg>`,

  /* Persona + reloj: permanencia. */
  permanencia: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="38" cy="26" r="9"/>
    <path d="M20 76c0-14 8-22 18-22s18 8 18 22"/>
    <circle class="acc" cx="84" cy="42" r="20"/>
    <path class="acc" d="M84 30v12l9 6"/>
  </svg>`,

  /* Campana + foto: alertas con evidencia. */
  alerta: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path class="warm" d="M38 62c-8 0-10-4-7-8 3-3 4-6 4-14 0-11 7-18 15-18s15 7 15 18c0 8 1 11 4 14 3 4 1 8-7 8Z"/>
    <path class="warm" d="M44 66c1 5 11 5 12 0"/>
    <rect x="72" y="34" width="38" height="30" rx="3"/>
    <circle cx="84" cy="46" r="5"/>
    <path d="M76 62c1-6 4-9 8-9s7 3 8 9"/>
  </svg>`,

  /* Escudo con check grande: salud del sistema. */
  salud: `<svg viewBox="0 0 120 90" class="ilu" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M60 8c14 8 26 8 36 4v30c0 24-19 36-36 42C43 78 24 66 24 42V12c10 4 22 4 36-4Z"/>
    <path class="ok" d="M44 44l12 12 22-24"/>
  </svg>`,
};

/**
 * Monta la guía «¿Cómo leer esta sección?» dentro de una sección.
 *  seccionId – id del <section> (la guía se inserta tras el .porque, o al
 *              inicio si no hay).
 *  ilu       – clave de ILU.
 *  puntos    – lista de strings HTML (contenido ESTÁTICO del dashboard, no
 *              datos del servidor) con las definiciones en cristiano.
 *  abierta   – true para que arranque desplegada (conceptos nuevos).
 * Idempotente: si la sección ya tiene guía, no duplica.
 */
function montarGuia(seccionId, { ilu = null, puntos = [], abierta = false } = {}) {
  const seccion = document.getElementById(seccionId);
  if (!seccion || seccion.querySelector('.guia')) return;
  const det = document.createElement('details');
  det.className = 'guia';
  if (abierta) det.open = true;
  det.innerHTML = `
    <summary>❓ Cómo leer esta sección</summary>
    <div class="guia-cuerpo">
      ${ilu && ILU[ilu] ? `<div class="guia-ilu">${ILU[ilu]}</div>` : ''}
      <ul class="guia-texto">${puntos.map(p => `<li>${p}</li>`).join('')}</ul>
    </div>`;
  const porque = seccion.querySelector('.porque');
  if (porque) porque.after(det);
  else seccion.prepend(det);
}
