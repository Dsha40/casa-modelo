/* server/tests/js/prueba_util.js — la logica de shared/util.js, EJECUTADA.
 *
 * `node --check` solo dice que el archivo parsea. Esto corre el codigo de
 * verdad contra un DOM minimo, que es lo unico que caza los fallos que
 * importan aqui. Asi se cazo uno el 19-ago-2026: anidar `html` dentro de
 * `html` doble-escapaba, y anidar es el patron mas usado en los cuatro
 * dashboards — la pagina habria salido con `&lt;tr&gt;` a la vista.
 *
 * La lanza `server/tests/test_dashboard_js.py`; tambien corre sola:
 *     node server/tests/js/prueba_util.js
 */
const fs = require('fs');
const vm = require('vm');
const path = require('path');

const oyentes = {};
const ctx = {
  window: {},
  console,
  document: {
    addEventListener: (tipo, fn, cap) => { oyentes[tipo] = fn; },
    getElementById: () => null,
    createElement: () => ({ style: {}, classList: { add() {} }, appendChild() {} }),
    body: { appendChild() {} },
    querySelectorAll: () => [],
  },
};
ctx.globalThis = ctx;
// UI vive en api.js; aqui basta con su escape.
ctx.UI = { esc: v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])) };

vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', '..', '..', 'dashboards', 'shared', 'util.js'), 'utf8'), ctx);

let fallos = 0;
const comprobar = (nombre, real, esperado) => {
  const ok = String(real) === String(esperado);
  if (!ok) { fallos++; console.log(`  FALLA ${nombre}\n    real:      ${real}\n    esperado:  ${esperado}`); }
  else console.log(`  ok  ${nombre}`);
};

console.log('plantilla `html`:');
const malicioso = '<img src=x onerror=alert(1)>';
comprobar('escapa el valor interpolado',
  ctx.html`<b>${malicioso}</b>`,
  '<b>&lt;img src=x onerror=alert(1)&gt;</b>');
comprobar('escapa tambien dentro de un atributo',
  ctx.html`<a href="${'" onclick="robar()'}">x</a>`,
  '<a href="&quot; onclick=&quot;robar()">x</a>');
comprobar('null y undefined no pintan "null"', ctx.html`[${null}${undefined}]`, '[]');
comprobar('false tampoco', ctx.html`[${false}]`, '[]');
comprobar('las listas se unen solas', ctx.html`${['a', 'b']}`, 'ab');
comprobar('anidar 3 niveles tampoco',
  ctx.html`${ctx.html`${ctx.html`<i>${'<x>'}</i>`}`}`, '<i>&lt;x&gt;</i>');
comprobar('lo marcado como crudo pasa entero',
  ctx.html`${ctx.html.crudo('<b>ya es html</b>')}`, '<b>ya es html</b>');
comprobar('anidar html no doble-escapa',
  ctx.html`<ul>${[ctx.html`<li>${'a&b'}</li>`]}</ul>`, '<ul><li>a&amp;b</li></ul>');

console.log('\noyente de imagenes rotas:');
const alError = oyentes['error'];
if (!alError) { console.log('  FALLA: no se engancho el oyente'); fallos++; }
else {
  const img1 = { tagName: 'IMG', dataset: { respaldo: '/foto.jpg' }, src: '/mini.jpg' };
  alError({ target: img1 });
  comprobar('primero cae a la foto completa', img1.src, '/foto.jpg');
  alError({ target: img1 });
  comprobar('y luego al placeholder', img1.src.slice(0, 19), 'data:image/svg+xml,');
  alError({ target: img1 });
  comprobar('no entra en bucle', img1.dataset.rota, 'si');

  let pintado = null;
  const img2 = { tagName: 'IMG', dataset: { rotaTexto: 'sin mapa <aun>' }, src: '/hm.png',
                 parentElement: { set innerHTML(v) { pintado = v; } } };
  alError({ target: img2 });
  comprobar('el aviso sustituye el hueco y va escapado', pintado,
    '<div class="img-rota">sin mapa &lt;aun&gt;</div>');

  const noImg = { tagName: 'SCRIPT', dataset: {} };
  alError({ target: noImg });
  comprobar('ignora lo que no es una imagen', noImg.src, undefined);
}

console.log(fallos ? `\n${fallos} FALLOS` : '\nTodo OK');
process.exit(fallos ? 1 : 0);
