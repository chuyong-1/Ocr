#!/usr/bin/env node
/**
 * Vendor asset check.
 * ════════════════════════════════════════════════════════════════════
 * The offline story only holds if every vendored library can find the
 * companion files it asks for at runtime. Those names are version-coded
 * (ONNX Runtime 1.26 wants ort-wasm-simd-threaded.mjs; 1.17 wanted
 * ort-wasm.wasm), so an upgraded .js beside a stale .wasm fails silently
 * in the browser and only shows up as "ONNX unavailable" in the badge.
 *
 * This script cross-checks each loader against the files on disk by
 * reading the asset names the loader actually references.
 *
 * Usage: npm run check
 */

'use strict';

const fs   = require('fs');
const path = require('path');

const WWW = path.resolve(__dirname, '..', 'www');
let failures = 0, warnings = 0;

const ok   = (m) => console.log('  \x1b[32m✓\x1b[0m ' + m);
const bad  = (m) => { failures++; console.log('  \x1b[31m✗\x1b[0m ' + m); };
const warn = (m) => { warnings++; console.log('  \x1b[33m!\x1b[0m ' + m); };

const exists = (rel) => fs.existsSync(path.join(WWW, rel));
const read   = (rel) => fs.readFileSync(path.join(WWW, rel), 'utf8');
const sizeOf = (rel) => fs.statSync(path.join(WWW, rel)).size;

function section(title) { console.log('\n' + title); }

/* ── core app files ──────────────────────────────────────────────── */
section('App');
['index.html', 'main.js'].forEach((f) =>
  exists(f) ? ok(f) : bad(`${f} is missing`));

/* ── ONNX Runtime Web ────────────────────────────────────────────── */
section('ONNX Runtime Web');
const ortLoader = ['vendor/onnx/ort.wasm.min.js', 'vendor/onnx/ort.min.js']
  .find(exists);

if (!ortLoader) {
  warn('no ONNX loader vendored — the app will fall back to the CDN');
} else {
  ok(`loader ${path.basename(ortLoader)}`);
  const src = read(ortLoader);
  const version = (src.match(/"(\d+\.\d+\.\d+)"/) || [])[1];
  if (version) ok(`reports version ${version}`);

  // Every ort-wasm-*.mjs the loader can dynamically import must be on disk.
  const needed = [...new Set(src.match(/ort-wasm[a-z0-9.\-]*\.mjs/g) || [])]
    .filter((n) => !n.includes('proxy-worker'));   // proxy mode is disabled
  if (!needed.length) warn('could not determine required wasm artifacts');
  needed.forEach((name) => {
    const mjs  = 'vendor/onnx/' + name;
    const wasm = mjs.replace(/\.mjs$/, '.wasm');
    exists(mjs)  ? ok(name) : bad(`${name} referenced by the loader but missing`);
    exists(wasm) ? ok(path.basename(wasm)) : bad(`${path.basename(wasm)} missing`);
  });

  // Legacy pre-1.18 artifacts: dead weight that no current loader requests.
  ['ort-wasm.wasm', 'ort-wasm-simd.wasm', 'ort-wasm-threaded.wasm']
    .filter((n) => exists('vendor/onnx/' + n))
    .forEach((n) => warn(`${n} is a pre-1.18 artifact no loader requests — safe to delete`));
}

/* ── font classifier: label count must match the model's output ──── */
section('Font classifier');
if (!exists('models/font_classifier.onnx')) {
  bad('models/font_classifier.onnx missing');
} else {
  ok(`models/font_classifier.onnx (${(sizeOf('models/font_classifier.onnx') / 1e6).toFixed(1)} MB)`);
  const buf = fs.readFileSync(path.join(WWW, 'models/font_classifier.onnx'));
  // Trailing graph.output ValueInfo encodes the final dim; read it back so a
  // retrained model with a different class count is caught here, not at runtime.
  const tail = buf.subarray(buf.length - 512);
  const m = tail.toString('latin1').match(/output[\s\S]{0,40}?batch_size\n\x02\x08(.)/);
  const classes = m ? m[1].charCodeAt(0) : null;
  const labels = (read('main.js').match(/const FONT_LABELS = \[([\s\S]*?)\];/) || [])[1];
  const labelCount = labels ? (labels.match(/'/g) || []).length / 2 : null;
  if (classes && labelCount) {
    if (classes === labelCount) ok(`${classes} classes, ${labelCount} labels`);
    else warn(`model emits ${classes} classes but main.js lists ${labelCount} labels ` +
              `— only the first ${Math.min(classes, labelCount)} are reachable`);
  }
}

/* ── Tesseract.js ────────────────────────────────────────────────── */
section('Tesseract.js (offline OCR)');
const tessFiles = {
  'vendor/tesseract/tesseract.min.js':      'main library',
  'vendor/tesseract/worker.min.js':         'worker (opts.workerPath)',
  'vendor/tesseract/tesseract-core.wasm.js':'core glue (opts.corePath)',
  'vendor/tesseract/tesseract-core.wasm':   'core wasm',
  'tessdata/eng.traineddata':               'English language data (opts.langPath)',
};
let tessComplete = true;
for (const [rel, what] of Object.entries(tessFiles)) {
  if (exists(rel)) ok(`${rel} — ${what}`);
  else { tessComplete = false; warn(`${rel} missing (${what}) — OCR will use the CDN`); }
}
if (tessComplete) ok('complete: OCR works with no network');
if (exists('tessdata/eng.traineddata.gz'))
  warn('eng.traineddata.gz present — main.js sets gzip:false and reads the plain file');

/* ── PDF.js / OpenCV ─────────────────────────────────────────────── */
section('PDF.js / OpenCV.js');
[['vendor/pdfjs/pdf.min.js', 'PDF.js'],
 ['vendor/pdfjs/pdf.worker.min.js', 'PDF.js worker'],
 ['vendor/opencv/opencv.js', 'OpenCV.js']]
  .forEach(([rel, what]) => exists(rel) ? ok(`${rel} — ${what}`)
                                        : warn(`${rel} missing (${what}) — falls back to CDN`));

/* ── summary ─────────────────────────────────────────────────────── */
console.log('');
if (failures) {
  console.log(`\x1b[31m${failures} error(s)\x1b[0m, ${warnings} warning(s)\n`);
  process.exit(1);
}
console.log(`\x1b[32mAll required assets present\x1b[0m${warnings ? ` (${warnings} warning(s))` : ''}\n`);
