// Validate standalone JS syntax and the viewport/arm geometry without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'visualize.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const geometry = script.slice(script.indexOf('const base='), script.indexOf('function line('));
const context = vm.createContext({ canvas: { width: 800, height: 600 } });
vm.runInContext(geometry, context);
const result = vm.runInContext(`(() => {
  const states = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0],
    [Math.PI, -Math.PI / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 1, 0]
  ];
  const viewport = fitView(states);
  return { viewport, frames: states.map(s => {
    const p = pointsForState(s);
    return { base: px(base, viewport), elbow: px(p.e, viewport), tip: px(p.t, viewport) };
  }) };
})()`, context);
assert.ok(Object.values(result.viewport).every(Number.isFinite));
assert.ok(result.viewport.scale > 0);
assert.deepEqual(result.frames[0].base, result.frames[1].base);
for (const frame of result.frames) {
  for (const point of Object.values(frame)) assert.ok(point.every(Number.isFinite));
  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  assert.ok(Math.abs(distance(frame.base, frame.elbow) / result.viewport.scale - 1.05) < 1e-9);
  assert.ok(Math.abs(distance(frame.elbow, frame.tip) / result.viewport.scale - 0.95) < 1e-9);
}
console.log('Visualizer syntax, finite viewport, fixed base and both link lengths passed.');
