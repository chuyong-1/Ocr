/**
 * PixelScribe — main.js  (v3.1 · Edge AI Edition)
 * ════════════════════════════════════════════════════════════════════════
 * 100% offline editor. All ML inference runs in the browser via WASM.
 *
 * v3.1 fixes
 * ──────────
 *  • OfflineOCR: local Tesseract.js vendor → CDN fallback (auto)
 *  • Tesseract.js v4 API used (createWorker(lang, oem, opts));
 *    graceful fall-back to v2/v3 API if the loaded build is older
 *  • _processFile: restricted to .json / .jpg / .jpeg / .pdf only,
 *    with clear Toast messages for unsupported types
 *  • Better progress + status feedback during OCR
 *  • PDF: PDF.js renders page 1 → canvas → OCR pipeline
 *
 * Modules
 * ───────
 *  EdgeML          — ONNX Runtime Web font classifier
 *  AppState        — centralised application data store
 *  EditorState     — undo/redo history stack
 *  ScaleEngine     — coordinate scaling between native px and display px
 *  OverlayEngine   — DOM injection of contenteditable text blocks
 *  PropsPanel      — right sidebar typography controls
 *  LayerPanel      — left sidebar layer list
 *  ExportEngine    — off-screen canvas flatten + file download
 *  CanvasView      — zoom / fit-to-window control
 *  Toast           — lightweight notification system
 * ════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ══════════════════════════════════════════════════════════════════════
   ONNX / PDF / vendor path constants
══════════════════════════════════════════════════════════════════════ */
/* ONNX Runtime Web is pinned to 1.26.0 — the version of the vendored
   ort.wasm.min.js AND of ort-wasm-simd-threaded.wasm. These must agree:
   the loader fetches a wasm artifact whose name encodes the build, so a
   mismatched pair 404s at session-create time. */
const ONNX_VERSION     = '1.26.0';
const ONNX_WASM_LOCAL  = './vendor/onnx/';
const ONNX_WASM_CDN    = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/`;
// The artifact the 1.26 wasm-only build actually requests. Probing for this
// (rather than the long-gone `ort-wasm.wasm`) is what makes the local-vs-CDN
// decision correct.
const ONNX_WASM_PROBE  = 'ort-wasm-simd-threaded.mjs';

const PDF_VENDOR_LOCAL = './vendor/pdfjs/';
const PDF_VENDOR_CDN   = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/';
const PDF_WORKER_FILE  = 'pdf.worker.min.js';

// Neighbourhood radius for cv.inpaint — mirrors backend/app.py's INPAINT_RADIUS.
const INPAINT_RADIUS = 12;

/* Text geometry — shared by the on-screen overlay and the export canvas so
   the PNG matches the preview exactly.

   OCR reports a tight *ink* box. CSS positions the *em* box, and with a
   line-height above 1 the em box is pushed down inside the line box by half
   the leading. Subtracting that half-leading from `top` puts the first line's
   em box where the ink box starts, which is also where canvas `textBaseline
   = 'top'` draws. Without this the overlay sits a few px low on screen and
   the export lands somewhere else again. */
const TEXT_LINE_HEIGHT = 1.25;
// Hard line breaks the user typed with Enter: CRLF or LF.
const NEWLINE_RE = /\r?\n/;
const halfLeading = (fontSizePx) => (TEXT_LINE_HEIGHT - 1) * fontSizePx / 2;

const FONT_CLASSIFIER_MODEL_URL = './models/font_classifier.onnx';
const API_BASE_STORAGE_KEY      = 'pixelscribe_api_base';

/* ── Offline OCR constants ─────────────────────────────────────────── */
const OFFLINE_OCR_LANG = 'eng';
const OFFLINE_OCR_MAX_DIM = 2000;

// Local vendor paths (used if present — works fully offline).
// Tesseract.js does NOT resolve these against the document; it hands them
// straight to a Worker whose own base URL is ./vendor/tesseract/. Relative
// paths would therefore resolve twice. absUrl() below makes them absolute.
const OFFLINE_OCR_VENDOR_LOCAL    = './vendor/tesseract/';
const OFFLINE_OCR_TESSERACT_LOCAL = OFFLINE_OCR_VENDOR_LOCAL + 'tesseract.min.js';
const OFFLINE_OCR_WORKER_LOCAL    = OFFLINE_OCR_VENDOR_LOCAL + 'worker.min.js';
const OFFLINE_OCR_CORE_LOCAL      = OFFLINE_OCR_VENDOR_LOCAL + 'tesseract-core.wasm.js';
const OFFLINE_OCR_TESSDATA_LOCAL  = './tessdata/';

// CDN fallbacks — loaded automatically when local files are absent
const OFFLINE_OCR_TESSERACT_CDN = 'https://cdn.jsdelivr.net/npm/tesseract.js@4.1.1/dist/tesseract.min.js';
const OFFLINE_OCR_TESSDATA_CDN  = 'https://tessdata.projectnaptha.com/4.0.0/';

/** Absolute URL for a path relative to the page — required for anything
 *  handed to a Worker, which resolves relative URLs against its own script. */
function absUrl(path) { return new URL(path, document.baseURI).href; }


/* ══════════════════════════════════════════════════════════════════════
   OPENCV.JS READINESS
══════════════════════════════════════════════════════════════════════ */
let cvReady = false;

const AIRuntime = (() => {
  let _onnxReady = false;
  let _onnxInitDone = false;
  let _onnxInitError = null;
  let _cvReadyResolve = null;

  const _cvReadyPromise = new Promise((resolve) => { _cvReadyResolve = resolve; });
  let _onnxInitPromise = null;

  function setOnnxInitPromise(p) { _onnxInitPromise = p; }

  function markCvReady() {
    if (_cvReadyResolve) { _cvReadyResolve(); _cvReadyResolve = null; }
    _refreshUiLock();
  }
  function markOnnxReady()  { _onnxReady = true;  _onnxInitDone = true; _onnxInitError = null; _refreshUiLock(); }
  function markOnnxFailed(err) { _onnxReady = false; _onnxInitDone = true; _onnxInitError = err; _refreshUiLock(); }

  function isCvReady()   { return cvReady; }
  function isOnnxReady() { return _onnxReady; }
  function isReady()     { return cvReady && _onnxReady; }

  async function waitUntilReady(timeoutMs = 120000) {
    const waits = [_cvReadyPromise];
    if (_onnxInitPromise) waits.push(_onnxInitPromise);
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('AI libraries timed out.')), timeoutMs);
    });
    try {
      await Promise.race([
        Promise.all(waits).then(() => {
          if (!cvReady) throw new Error('OpenCV.js is not ready yet.');
          if (!_onnxInitDone) throw new Error('ONNX model still loading.');
          if (!_onnxReady) throw new Error((_onnxInitError && _onnxInitError.message) || 'ONNX failed.');
        }),
        timeout,
      ]);
    } finally { clearTimeout(timer); }
  }

  return { setOnnxInitPromise, markCvReady, markOnnxReady, markOnnxFailed,
           isCvReady, isOnnxReady, isReady, waitUntilReady };
})();

function _aiAlert(context, err) {
  const msg = (err && err.message) ? err.message : String(err);
  console.error(`[PixelScribe AI · ${context}]`, err);
  alert('AI Error: ' + msg);
}

function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src   = src;
    script.async = true;
    script.onload  = () => resolve();
    script.onerror = () => reject(new Error('Failed to load script: ' + src));
    document.head.appendChild(script);
  });
}

async function _resolveLocalOrCdnBase(localBase, cdnBase, probeFile) {
  try {
    const res = await fetch(localBase + probeFile, { method: 'HEAD' });
    if (res.ok) return localBase;
  } catch (_) { /* fall through */ }
  return cdnBase;
}

async function _setOnnxWasmPaths() {
  if (typeof ort === 'undefined' || !ort.env || !ort.env.wasm) return;
  const base = await _resolveLocalOrCdnBase(ONNX_WASM_LOCAL, ONNX_WASM_CDN, ONNX_WASM_PROBE);
  ort.env.wasm.wasmPaths = absUrl(base);
  // Multi-threading needs SharedArrayBuffer, which needs cross-origin
  // isolation (COOP+COEP). A plain static host — and Capacitor's WebView —
  // has neither, and the threaded build then aborts instead of degrading.
  // One thread is ample for 64x64 crops.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy      = false;
  console.info('[EdgeML] ONNX wasm base:', ort.env.wasm.wasmPaths);
}

async function _ensurePdfWorkerSrc() {
  if (typeof pdfjsLib === 'undefined') return;
  const base = await _resolveLocalOrCdnBase(PDF_VENDOR_LOCAL, PDF_VENDOR_CDN, PDF_WORKER_FILE);
  pdfjsLib.GlobalWorkerOptions.workerSrc = absUrl(base + PDF_WORKER_FILE);
}

function _setImportEnabled(enabled) {
  ['btn-import', 'sidebar-upload-btn'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = !enabled;
  });
  const welcome = document.getElementById('welcome-drop');
  if (welcome && !AppState.isLoaded) {
    welcome.style.pointerEvents = enabled ? 'auto' : 'none';
    welcome.style.opacity = enabled ? '1' : '0.55';
  }
}

function _ensureEditorInteractive() {
  const canvasArea = document.getElementById('canvas-area');
  const overlay    = document.getElementById('overlay');
  const welcome    = document.getElementById('welcome-drop');
  if (canvasArea) canvasArea.style.pointerEvents = 'auto';
  if (overlay)    overlay.style.pointerEvents    = 'auto';
  if (welcome)    { welcome.style.display = 'none'; welcome.style.pointerEvents = 'none'; }
}

function _refreshUiLock() {
  _setImportEnabled(true);
  if (AIRuntime.isReady()) {
    _setStatusOk('AI ready — import or tap text to edit');
  } else if (AIRuntime.isCvReady() && !AIRuntime.isOnnxReady()) {
    _setStatusOk('Ready — ONNX offline, editing works');
  } else if (AIRuntime.isCvReady()) {
    _setStatusOk('Ready — tap text blocks to edit');
  }
}

function onOpenCvLoad() {
  window.__cvLoaded = true;
  if (typeof cv !== 'undefined') {
    if (cv.getBuildInformation) _markCvReady();
    else cv.onRuntimeInitialized = _markCvReady;
  }
}

/* index.html declares a stub onOpenCvLoad in <head> so the async opencv.js
   tag always has a handler, even if it wins the race against this file.
   Take over the hook now; init() below adopts whatever the stub already
   recorded. That replay must not run here: marking CV ready touches
   AppState, a const declared further down, and reading it this early
   throws a TDZ ReferenceError that aborts the whole script — which is what
   happened whenever opencv.js was cached and loaded first. */
window.__onCvReady = onOpenCvLoad;

function _markCvReady() {
  if (cvReady) return;
  cvReady = true;
  console.info('[PixelScribe] OpenCV.js WASM ready.');
  _setBadge('badge-cv', 'ready', 'CV ready');
  AIRuntime.markCvReady();
}


/* ══════════════════════════════════════════════════════════════════════
   EDGE ML — ONNX Runtime Web font classifier
══════════════════════════════════════════════════════════════════════ */
const EdgeML = (() => {
  /* Label order must match backend/train_font_classifier.py's FONT_LABELS.
     The checkpoint currently shipped in models/font_classifier.onnx was
     trained on the first five classes only — its output tensor is [N, 5].
     `_labelCount` is read back from the model at load time so a retrained
     model with more classes starts using them without a code change, and a
     shorter one can never index past the end of this array. */
  const FONT_LABELS = [
    'Arial', 'Times New Roman', 'Courier New', 'Calibri', 'Georgia',
    'Verdana', 'Roboto', 'Helvetica', 'Garamond', 'Consolas',
  ];

  /* The model was trained on 256×80 patches (train_font_classifier.py's
     PATCH_W × PATCH_H): dark text, 18–36 px, centred on a light background,
     then squashed to 64×64. OCR hands us a tight ink box that is often 20×
     wider than tall — squashing *that* straight to 64×64 gives the network
     input unlike anything it saw, and it scored at chance. So rebuild a
     training-shaped patch first: scale the ink to the training ink height,
     centre (or left-clip) it in 256×80, normalise to dark-on-light. */
  const PATCH_W = 256, PATCH_H = 80;
  const PATCH_INK_H = 30;          // middle of the 26–36 px plateau measured on rendered lines
  const PATCH_BG = 245, PATCH_INK = 30;

  let _session = null;
  let _labelCount = FONT_LABELS.length;
  const _patchCanvas = document.createElement('canvas');
  _patchCanvas.width = PATCH_W; _patchCanvas.height = PATCH_H;
  const _patchCtx = _patchCanvas.getContext('2d', { willReadFrequently: true });
  const _cropCanvas = document.createElement('canvas');
  _cropCanvas.width = _cropCanvas.height = 64;
  const _cropCtx = _cropCanvas.getContext('2d', { willReadFrequently: true });

  /** Draw the ink box into a training-shaped, dark-on-light 256×80 patch. */
  function _buildPatch(imageEl, x, y, width, height) {
    const s  = PATCH_INK_H / height;
    const dw = Math.min(PATCH_W, width * s);
    const dx = Math.max(0, Math.round((PATCH_W - width * s) / 2));
    const dy = Math.round((PATCH_H - height * s) / 2);
    _patchCtx.fillStyle = '#fff';
    _patchCtx.fillRect(0, 0, PATCH_W, PATCH_H);
    _patchCtx.imageSmoothingQuality = 'high';
    _patchCtx.drawImage(imageEl, x, y, dw / s, height, dx, dy, dw, height * s);

    const img = _patchCtx.getImageData(0, 0, PATCH_W, PATCH_H);
    const px  = img.data;
    const x0 = dx, x1 = Math.min(PATCH_W, Math.ceil(dx + dw));
    const y0 = Math.max(0, dy), y1 = Math.min(PATCH_H, Math.ceil(dy + height * s));
    const inside = [];
    for (let r = y0; r < y1; r++) for (let c = x0; c < x1; c++) {
      const i = (r * PATCH_W + c) * 4;
      inside.push(0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]);
    }
    const sorted = Float32Array.from(inside).sort();
    const q  = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))] || 0;
    const bg = q(0.5), lo = q(0.02), hi = q(0.98);
    // Ink is whichever extreme sits further from the background.
    const ink = (bg - lo) >= (hi - bg) ? lo : hi;
    const span = Math.abs(bg - ink) || 1;

    for (let r = 0; r < PATCH_H; r++) for (let c = 0; c < PATCH_W; c++) {
      const i = (r * PATCH_W + c) * 4;
      let v = PATCH_BG;
      if (r >= y0 && r < y1 && c >= x0 && c < x1) {
        const g = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
        const t = Math.max(0, Math.min(1, Math.abs(g - bg) / span));   // 0 = bg, 1 = ink
        v = PATCH_BG - t * (PATCH_BG - PATCH_INK);
      }
      px[i] = px[i + 1] = px[i + 2] = v; px[i + 3] = 255;
    }
    _patchCtx.putImageData(img, 0, 0);
  }

  async function init() {
    if (_session) { AIRuntime.markOnnxReady(); return; }
    if (typeof ort === 'undefined') {
      const err = new Error('onnxruntime-web not loaded.');
      console.warn('[EdgeML]', err.message);
      _setBadge('badge-onnx', 'error', 'ONNX unavailable');
      AIRuntime.markOnnxFailed(err);
      return;
    }
    try {
      await _setOnnxWasmPaths();
      _setBadge('badge-onnx', 'loading', 'ONNX loading…');
      _session = await ort.InferenceSession.create(FONT_CLASSIFIER_MODEL_URL, { executionProviders: ['wasm'] });
      console.info('[EdgeML] ONNX session ready.');
      _setBadge('badge-onnx', 'ready', 'ONNX ready');
      AIRuntime.markOnnxReady();
    } catch (err) {
      console.warn('[EdgeML] Could not load ONNX model:', err.message);
      _setBadge('badge-onnx', 'error', 'ONNX error');
      _session = null;
      AIRuntime.markOnnxFailed(err);
    }
  }

  async function predictFont(imageEl, x, y, width, height) {
    if (!_session || !width || !height || width <= 0 || height <= 0) return 'Arial';
    try {
      _buildPatch(imageEl, x, y, width, height);
      _cropCtx.imageSmoothingQuality = 'high';
      _cropCtx.drawImage(_patchCanvas, 0, 0, 64, 64);
      const imageData = _cropCtx.getImageData(0, 0, 64, 64);
      const pixels    = imageData.data;
      const grayFloat = new Float32Array(64 * 64);
      for (let i = 0; i < 64 * 64; i++) {
        grayFloat[i] = (0.299 * pixels[i*4] + 0.587 * pixels[i*4+1] + 0.114 * pixels[i*4+2]) / 255.0;
      }
      const tensor     = new ort.Tensor('float32', grayFloat, [1, 1, 64, 64]);
      const inputName  = _session.inputNames[0];
      const outputName = _session.outputNames[0];
      const results    = await _session.run({ [inputName]: tensor });
      const logits     = results[outputName].data;
      const n = Math.min(logits.length, FONT_LABELS.length);
      if (logits.length > FONT_LABELS.length) {
        console.warn(`[EdgeML] Model emits ${logits.length} classes but only ` +
                     `${FONT_LABELS.length} labels are known — extra classes ignored.`);
      }
      _labelCount = n;
      let maxIdx = 0, maxVal = logits[0];
      for (let i = 1; i < n; i++) { if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; } }
      return FONT_LABELS[maxIdx] || 'Arial';
    } catch (err) {
      console.warn('[EdgeML] predictFont error:', err.message);
      return 'Arial';
    }
  }

  return { init, predictFont, FONT_LABELS, labelCount: () => _labelCount };
})();


/* ══════════════════════════════════════════════════════════════════════
   FONT FALLBACK MAP
══════════════════════════════════════════════════════════════════════ */
const FONT_FALLBACK_MAP = {
  'Arial':           '"Arial", "Helvetica Neue", Helvetica, sans-serif',
  'Times New Roman': '"Times New Roman", Times, "Noto Serif", serif',
  'Courier New':     '"Courier New", Courier, "Roboto Mono", monospace',
  'Calibri':         '"Calibri", "Segoe UI", Candara, sans-serif',
  'Georgia':         '"Georgia", Cambria, "Times New Roman", serif',
  'Verdana':         '"Verdana", Geneva, Tahoma, sans-serif',
  'Roboto':          '"Roboto", "Helvetica Neue", Arial, sans-serif',
  'Helvetica':       '"Helvetica Neue", Helvetica, Arial, sans-serif',
  'Garamond':        '"EB Garamond", Garamond, "Times New Roman", serif',
  'Consolas':        '"Consolas", "Roboto Mono", "Courier New", monospace',
};
function resolveFontStack(fontFamily) {
  // An installed font renders as itself — exactly what FontMatcher scored.
  // The fallback map only matters for names this machine doesn't have.
  const generic = FontMatcher.genericFor(fontFamily);
  if (FontMatcher.isInstalled(fontFamily)) return `"${fontFamily}", ${generic}`;
  return FONT_FALLBACK_MAP[fontFamily] || `"${fontFamily}", ${generic}`;
}

/** Full CSS/canvas `font` shorthand for a block at `px`. */
function _fontCss(block, px) {
  return `${block.font_style || 'normal'} ${block.font_weight || 'normal'} ${px}px ` +
         resolveFontStack(block.font_family || 'Arial');
}


/* ══════════════════════════════════════════════════════════════════════
   FONT MATCHER — render-and-compare font recognition
   ─────────────────────────────────────────────────────────────────────
   The ONNX classifier knows five fonts. This needs no training: OCR has
   already told us *what* the text says, so draw that exact string in
   every candidate font installed on this machine (regular, then bold /
   italic for the front-runners), squeeze each rendering to the shape of
   the source ink, and keep the closest. Width-to-height ratio of the whole
   line is the strongest single cue; the glyph overlap separates fonts of
   similar width (Arial vs Helvetica-likes, Times vs Georgia).
══════════════════════════════════════════════════════════════════════ */
const FontMatcher = (() => {
  // [family, generic]. Only the ones actually installed are tried/offered.
  const CANDIDATES = [
    ['Arial', 'sans-serif'], ['Helvetica', 'sans-serif'], ['Helvetica Neue', 'sans-serif'],
    ['Segoe UI', 'sans-serif'], ['Calibri', 'sans-serif'], ['Verdana', 'sans-serif'],
    ['Tahoma', 'sans-serif'], ['Trebuchet MS', 'sans-serif'], ['Century Gothic', 'sans-serif'],
    ['Franklin Gothic Medium', 'sans-serif'], ['Gill Sans MT', 'sans-serif'], ['Gill Sans', 'sans-serif'],
    ['Lucida Sans Unicode', 'sans-serif'], ['Lucida Grande', 'sans-serif'], ['Candara', 'sans-serif'],
    ['Corbel', 'sans-serif'], ['Arial Narrow', 'sans-serif'], ['Bahnschrift', 'sans-serif'],
    ['Avenir', 'sans-serif'], ['Futura', 'sans-serif'], ['Optima', 'sans-serif'],
    ['Roboto', 'sans-serif'], ['Open Sans', 'sans-serif'], ['Lato', 'sans-serif'],
    ['Montserrat', 'sans-serif'], ['Source Sans Pro', 'sans-serif'], ['Noto Sans', 'sans-serif'],
    ['Ubuntu', 'sans-serif'], ['DejaVu Sans', 'sans-serif'], ['Liberation Sans', 'sans-serif'],
    ['Times New Roman', 'serif'], ['Times', 'serif'], ['Georgia', 'serif'], ['Cambria', 'serif'],
    ['Garamond', 'serif'], ['EB Garamond', 'serif'], ['Book Antiqua', 'serif'],
    ['Palatino Linotype', 'serif'], ['Palatino', 'serif'], ['Constantia', 'serif'],
    ['Century Schoolbook', 'serif'], ['Bookman Old Style', 'serif'], ['Baskerville', 'serif'],
    ['Baskerville Old Face', 'serif'], ['Didot', 'serif'], ['Rockwell', 'serif'],
    ['Noto Serif', 'serif'], ['DejaVu Serif', 'serif'], ['Liberation Serif', 'serif'],
    ['Courier New', 'monospace'], ['Consolas', 'monospace'], ['Lucida Console', 'monospace'],
    ['Cascadia Mono', 'monospace'], ['Menlo', 'monospace'], ['Monaco', 'monospace'],
    ['DejaVu Sans Mono', 'monospace'], ['Liberation Mono', 'monospace'],
    ['Impact', 'sans-serif'], ['Arial Black', 'sans-serif'], ['Comic Sans MS', 'cursive'],
    ['Segoe Script', 'cursive'], ['Segoe Print', 'cursive'], ['Brush Script MT', 'cursive'],
    ['Lucida Handwriting', 'cursive'], ['Monotype Corsiva', 'cursive'],
  ];
  const GENERIC = new Map(CANDIDATES);
  const GROUP_LABEL = { 'sans-serif': 'Sans serif', serif: 'Serif', monospace: 'Monospace', cursive: 'Script' };

  const H = 20, MAX_W = 480;   // comparison raster: fixed height, width follows aspect
  const RENDER_PX = 48;
  const VARIANT_SHORTLIST = 6; // families that also get bold / italic tried

  let _available = null;
  const _big = document.createElement('canvas');
  const _bigCtx = _big.getContext('2d', { willReadFrequently: true });
  const _small = document.createElement('canvas');
  const _smallCtx = _small.getContext('2d', { willReadFrequently: true });

  function genericFor(family) { return GENERIC.get(family) || 'sans-serif'; }

  /** A font is installed if naming it changes the width of a test string
   *  against at least one generic fallback. */
  function available() {
    if (_available) return _available;
    const probe = 'mmmmmmmmmmlli WQ@#0Oo,.ygjk';
    const base = {};
    ['monospace', 'serif', 'sans-serif'].forEach((g) => {
      _bigCtx.font = `72px ${g}`; base[g] = _bigCtx.measureText(probe).width;
    });
    // Drop names that are only aliases of a font already listed (Windows
    // maps Helvetica → Arial and Times → Times New Roman): same widths for
    // every probe means the same glyphs, and listing both just doubles up.
    const seen = new Set();
    const signature = (f) => ['Hamburgefonstiv', 'mmmlll', 'WQ@#0'].map((s) => {
      _bigCtx.font = `72px "${f}", monospace`; return _bigCtx.measureText(s).width.toFixed(2);
    }).join('|');
    _available = CANDIDATES.filter(([f]) => ['monospace', 'serif', 'sans-serif'].some((g) => {
      _bigCtx.font = `72px "${f}", ${g}`;
      return Math.abs(_bigCtx.measureText(probe).width - base[g]) > 0.5;
    })).filter(([f]) => {
      const sig = signature(f);
      if (seen.has(sig)) return false;
      seen.add(sig); return true;
    }).map(([f, g]) => ({ family: f, generic: g }));
    // The four web-safe defaults are always offered, even if the probe
    // could not tell them from the generic they alias.
    ['Arial', 'Times New Roman', 'Courier New', 'Georgia'].forEach((f) => {
      if (!_available.some((a) => a.family === f)) _available.push({ family: f, generic: genericFor(f) });
    });
    return _available;
  }

  /** Grouped for the font <select>: [{ label, fonts: [family…] }]. */
  function groups() {
    return Object.keys(GROUP_LABEL).map((g) => ({
      label: GROUP_LABEL[g],
      fonts: available().filter((a) => a.generic === g).map((a) => a.family).sort(),
    })).filter((grp) => grp.fonts.length);
  }

  /** Grey levels (0..1 ink) of a W×H resample of `src`'s (sx,sy,sw,sh). */
  function _resample(src, sx, sy, sw, sh, W, channel) {
    _small.width = W; _small.height = H;
    _smallCtx.imageSmoothingQuality = 'high';
    _smallCtx.clearRect(0, 0, W, H);
    _smallCtx.drawImage(src, sx, sy, sw, sh, 0, 0, W, H);
    const d = _smallCtx.getImageData(0, 0, W, H).data;
    const out = new Float32Array(W * H);
    for (let i = 0; i < out.length; i++) out[i] = d[i * 4 + channel] / 255;
    return out;
  }

  /** Ink of one OCR box, cropped tight to the ink and resampled.
   *  Returns { mask, W, aspect } or null when the box has no clear ink. */
  function _target(px, cols, rows, box) {
    const st = _inkStats(px, cols, rows, box);
    if (!st) return null;
    const x1 = Math.max(0, box.x), y1 = Math.max(0, box.y);
    const x2 = Math.min(cols, box.x + box.w), y2 = Math.min(rows, box.y + box.h);
    const w = x2 - x1, h = y2 - y1;
    if (w < 4 || h < 4) return null;
    const [br, bg, bb] = st.bg, thr = st.threshold;
    const img = new ImageData(w, h);
    let minX = w, maxX = -1, minY = h, maxY = -1;
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const i = ((y1 + r) * cols + (x1 + c)) * 4;
      const d = Math.abs(px[i] - br) + Math.abs(px[i + 1] - bg) + Math.abs(px[i + 2] - bb);
      const v = Math.max(0, Math.min(1, (d - thr * 0.5) / thr));
      const o = (r * w + c) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = Math.round(v * 255); img.data[o + 3] = 255;
      if (v > 0.5) { if (c < minX) minX = c; if (c > maxX) maxX = c; if (r < minY) minY = r; if (r > maxY) maxY = r; }
    }
    if (maxX < 0) return null;
    const tw = maxX - minX + 1, th = maxY - minY + 1;
    const aspect = tw / th;
    const W = Math.max(4, Math.min(MAX_W, Math.round(H * aspect)));
    _big.width = w; _big.height = h;
    _bigCtx.putImageData(img, 0, 0);
    return { mask: _resample(_big, minX, minY, tw, th, W, 0), W, aspect };
  }

  /** `text` drawn in `css`, cropped to its ink box and resampled to W×H. */
  function _render(text, css, W) {
    _bigCtx.font = css;
    const m = _bigCtx.measureText(text);
    const l = m.actualBoundingBoxLeft, r = m.actualBoundingBoxRight;
    const a = m.actualBoundingBoxAscent, d = m.actualBoundingBoxDescent;
    const bw = Math.ceil(l + r), bh = Math.ceil(a + d);
    if (!(bw > 0 && bh > 0)) return null;
    _big.width = bw + 2; _big.height = bh + 2;   // resizing clears and resets state
    _bigCtx.font = css; _bigCtx.fillStyle = '#000'; _bigCtx.textBaseline = 'alphabetic';
    _bigCtx.fillText(text, l + 1, a + 1);
    return { mask: _resample(_big, 1, 1, bw, bh, W, 3), aspect: (l + r) / (a + d) };
  }

  function _score(t, text, family, weight, style) {
    const css = `${style} ${weight} ${RENDER_PX}px "${family}", ${genericFor(family)}`;
    const rr = _render(text, css, t.W);
    if (!rr) return 0;
    let inter = 0, union = 0;
    for (let i = 0; i < t.mask.length; i++) {
      const p = t.mask[i], q = rr.mask[i];
      inter += Math.min(p, q); union += Math.max(p, q);
    }
    const overlap = union ? inter / union : 0;
    return overlap * Math.exp(-2.5 * Math.abs(Math.log(rr.aspect / t.aspect)));
  }

  /** Best matches for one block, or null if its ink can't be isolated.
   *  → { best: {family, weight, style, score}, byFamily: {family: {…}} } */
  function match(px, cols, rows, block) {
    const text = (block.text || '').trim();
    if (!text) return null;
    const t = _target(px, cols, rows, { x: block.x, y: block.y, w: block.ink_w || block.w, h: block.h });
    if (!t) return null;

    const byFamily = {};
    available().forEach(({ family }) => {
      byFamily[family] = { family, weight: 'normal', style: 'normal', score: _score(t, text, family, 'normal', 'normal') };
    });
    Object.values(byFamily).sort((p, q) => q.score - p.score).slice(0, VARIANT_SHORTLIST).forEach((b) => {
      [['bold', 'normal'], ['normal', 'italic'], ['bold', 'italic']].forEach(([weight, style]) => {
        const s = _score(t, text, b.family, weight, style);
        if (s > byFamily[b.family].score) byFamily[b.family] = { family: b.family, weight, style, score: s };
      });
    });
    const best = Object.values(byFamily).sort((p, q) => q.score - p.score)[0];
    return { best, byFamily };
  }

  function isInstalled(family) { return available().some((a) => a.family === family); }

  return { available, groups, genericFor, isInstalled, match };
})();

/* Blocks from OCR carry the source line's baseline. Pinning the rendered
   first line's alphabetic baseline to it keeps replacement text sitting on
   the same line as the original whatever face or size it ends up in. Blocks
   without one (JSON payloads) keep the em-box-at-ink-top placement. */
const _metricsCtx = document.createElement('canvas').getContext('2d');
function _fontAscentDescent(fontFamily, px) {
  _metricsCtx.font = `${px}px ${resolveFontStack(fontFamily || 'Arial')}`;
  const m = _metricsCtx.measureText('Hg');
  const a = m.fontBoundingBoxAscent, d = m.fontBoundingBoxDescent;
  return (isFinite(a) && isFinite(d)) ? { a, d } : { a: 0.9 * px, d: 0.22 * px };
}

/* .txt-block's padding + border in index.html (4px 6px + 1.5px). The field
   is border-box, so its text starts this far inside left/top and has this
   much less width — the field box must be grown by it, or the text sits
   low-right of the source and wraps a line early (and the export, which
   has no padding, disagrees with the screen). Keep in sync with the CSS. */
const FIELD_INSET_X = 7.5;
const FIELD_INSET_Y = 5.5;

/** CSS `top` (display px) of an overlay field whose font size is `fs`. */
function _overlayTop(live, s, fs) {
  if (!isFinite(live.baseline)) return Math.round(live.y * s - halfLeading(fs) - FIELD_INSET_Y);
  // A CSS line box centres the font's content area (ascent + descent) in
  // line-height, so the baseline sits half the leading plus the ascent down.
  const { a, d } = _fontAscentDescent(live.font_family, fs);
  return Math.round(live.baseline * s - ((TEXT_LINE_HEIGHT * fs - (a + d)) / 2 + a) - FIELD_INSET_Y);
}

/** Size an OCR block so its text, in the font it will render in, spans the
 *  same ink width as the source. The ink-height estimate alone overshoots on
 *  logos and mixed-case lines, and a too-large size wraps onto a second line
 *  that lands on the block below.
 *
 *  Compares ink with ink (actualBoundingBox), not with the advance width:
 *  the advance includes side bearings, so it always reads wider than the
 *  ink and shrank wide-bearing faces (Times, Courier) a size too far.
 *
 *  Shrinks freely down to half the height estimate. Grows (up to 1.25×) only
 *  when FontMatcher identified the face — with the wrong face, width says
 *  little about size. JSON blocks (no ink_w) are left as given. */
function _fitSizeToInk(block) {
  if (!block.ink_w || !block.text) return block;
  _metricsCtx.font = _fontCss(block, block.size);
  const m = _metricsCtx.measureText(block.text);
  const inkW = m.actualBoundingBoxLeft + m.actualBoundingBoxRight;
  const measured = isFinite(inkW) && inkW > 0 ? inkW : m.width;
  const fitted = Math.round(block.size * block.ink_w / measured);
  const max = block.font_candidates ? Math.floor(block.size * 1.25) : block.size;
  const size = Math.max(Math.ceil(block.size / 2), Math.min(max, fitted));
  return size === block.size ? block : { ...block, size };
}

/** Two OCR lines (a above b) belong to one paragraph: similar size, no more
 *  than a line's height of space between them, and sharing a left edge or a
 *  centre line. Side-by-side columns fail the alignment test. */
function _sameParagraph(a, b) {
  const r = b.size / a.size;
  if (r < 0.7 || r > 1.4) return false;
  const lineH = Math.max(a.h, b.h);
  if (b.y - (a.y + a.h) > lineH) return false;
  const leftAligned = Math.abs(a.x - b.x) <= lineH;
  const centred = Math.abs((a.x + a.ink_w / 2) - (b.x + b.ink_w / 2)) <= lineH;
  return leftAligned || centred;
}

/** Fonts and sizes are estimated one line at a time, so a paragraph came
 *  out with one line in Courier or a point larger than its neighbours.
 *  Give every line of a paragraph the font most of its text was classified
 *  as, then the (lower) median of the lines' fitted sizes. A line that the
 *  shared size makes wider than its field gets a wider field rather than a
 *  wrap. JSON blocks (no ink_w) and blocks kept as image are left alone. */
function _harmonizeParagraphs(blocks) {
  const out = blocks.slice();
  const order = out.map((_, i) => i)
    .filter((i) => out[i].ink_w && !out[i].keep_image)
    .sort((i, j) => out[i].y - out[j].y);
  const groups = [];
  order.forEach((i) => {
    const g = groups.find((g) => _sameParagraph(out[g[g.length - 1]], out[i]));
    if (g) g.push(i); else groups.push([i]);
  });

  groups.forEach((g) => {
    // With FontMatcher scores for every line, the family that matches the
    // paragraph best *as a whole* wins — and each line keeps its own best
    // weight/style in that family (a bold lead-in line stays bold).
    // Otherwise fall back to a vote weighted by text length.
    const votes = new Map();
    const scored = g.every((i) => out[i].font_scores);
    g.forEach((i) => {
      if (scored) {
        Object.values(out[i].font_scores).forEach((s) => votes.set(s.family, (votes.get(s.family) || 0) + s.score));
      } else {
        votes.set(out[i].font_family, (votes.get(out[i].font_family) || 0) + out[i].text.length);
      }
    });
    const font = [...votes].sort((p, q) => q[1] - p[1])[0][0];
    g.forEach((i) => {
      const v = scored && out[i].font_scores[font];
      out[i] = _fitSizeToInk({ ...out[i], font_family: font,
        ...(v ? { font_weight: v.weight, font_style: v.style } : {}) });
    });
    if (g.length < 2) return;

    const sizes = g.map((i) => out[i].size).sort((p, q) => p - q);
    const size = sizes[(sizes.length - 1) >> 1];
    g.forEach((i) => {
      const b = { ...out[i], size };
      _metricsCtx.font = _fontCss(b, size);
      const need = Math.ceil(_metricsCtx.measureText(b.text).width) + 4;
      if (need > b.w) b.w = Math.min(need, (AppState.imageW || need + b.x) - b.x);
      out[i] = b;
    });
  });
  return out;
}


/* ══════════════════════════════════════════════════════════════════════
   IMAGE PROCESSOR — OpenCV.js Inpainting
══════════════════════════════════════════════════════════════════════ */
const ImageProcessor = (() => {
  /** Point an <img> at new pixels without re-triggering the payload-load
   *  handler that loadPayload() installed on it. */
  function _replaceImageSrcQuietly(imageElement, src) {
    return new Promise((resolve, reject) => {
      const prevOnLoad  = imageElement.onload;
      const prevOnError = imageElement.onerror;
      imageElement.onload = () => {
        imageElement.onload = prevOnLoad; imageElement.onerror = prevOnError;
        ScaleEngine.recompute();
        resolve();
      };
      imageElement.onerror = () => {
        imageElement.onload = prevOnLoad; imageElement.onerror = prevOnError;
        reject(new Error('Inpainted image failed to load.'));
      };
      imageElement.src = src;
    });
  }

  /** OpenCV.js throws a raw C++ exception pointer (a number) rather than an
   *  Error. Turn it into something a Toast can actually show. */
  function _cvError(e) {
    if (e instanceof Error) return e;
    if (typeof e === 'number' && typeof cv !== 'undefined' && cv.exceptionFromPtr) {
      try { return new Error('OpenCV: ' + cv.exceptionFromPtr(e).msg); } catch (_) { /* fall through */ }
    }
    return new Error('OpenCV error (' + e + ')');
  }

  /** Mark the glyph pixels of one OCR box in `mask`, then dilate them to
   *  cover anti-aliased edges. Masking the whole rectangle instead made
   *  inpainting smear anything the text sat on — a stamp, a rule, the edge of
   *  a seal — across the box. Falls back to the rectangle when the box has no
   *  clear ink (e.g. text the same colour as a busy background). Returns
   *  false if the box lies outside the image. */
  function _paintInkMask(src, mask, bbox) {
    const box = { x: Math.round(bbox.x), y: Math.round(bbox.y),
                  w: Math.round(bbox.width), h: Math.round(bbox.height) };
    const x1 = Math.max(0, box.x - INK_PAD), y1 = Math.max(0, box.y - INK_PAD);
    const x2 = Math.min(src.cols, box.x + box.w + INK_PAD), y2 = Math.min(src.rows, box.y + box.h + INK_PAD);
    if (x2 <= x1 || y2 <= y1) return false;

    const stats = _inkStats(src.data, src.cols, src.rows, box);
    if (!stats) {
      cv.rectangle(mask, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(255), -1, cv.LINE_8, 0);
      return true;
    }
    const px = src.data, m = mask.data, [br, bg, bb] = stats.bg;
    for (let r = y1; r < y2; r++) {
      for (let c = x1; c < x2; c++) {
        const i = (r * src.cols + c) * 4;
        const d = Math.abs(px[i] - br) + Math.abs(px[i + 1] - bg) + Math.abs(px[i + 2] - bb);
        if (d >= stats.threshold) m[r * src.cols + c] = 255;
      }
    }
    const k = 2 * Math.max(2, Math.round(box.h * 0.06)) + 1;
    const rect = new cv.Rect(x1, y1, x2 - x1, y2 - y1);
    const roi = mask.roi(rect);
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k));
    try { cv.dilate(roi, roi, kernel); }
    finally { roi.delete(); kernel.delete(); }
    return true;
  }

  async function inpaintRegion(imageElement, bbox) {
    if (!cvReady || typeof cv === 'undefined') throw new Error('OpenCV.js not loaded yet.');
    let src, rgb, mask, dst;
    try {
      src = cv.imread(imageElement);              // always CV_8UC4 (RGBA)

      // cv.inpaint accepts 1- or 3-channel 8-bit input only; handing it the
      // RGBA Mat that imread() returns throws an unsupported-format exception.
      rgb = new cv.Mat();
      cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);

      mask = new cv.Mat(rgb.rows, rgb.cols, cv.CV_8UC1, new cv.Scalar(0));
      if (!_paintInkMask(src, mask, bbox)) throw new Error('Region is outside the image.');

      dst = new cv.Mat();
      cv.inpaint(rgb, mask, dst, INPAINT_RADIUS, cv.INPAINT_TELEA);

      const hiddenCanvas = document.createElement('canvas');
      cv.imshow(hiddenCanvas, dst);               // handles CV_8UC3 → RGBA
      AppState.bgSrc = hiddenCanvas.toDataURL('image/png');
      // Swapping .src re-fires whatever onload loadPayload() left attached,
      // which would rebuild every overlay block from the original payload and
      // silently throw away the user's edits. Detach for this one swap.
      await _replaceImageSrcQuietly(imageElement, AppState.bgSrc);
    } catch (e) {
      throw _cvError(e);
    } finally {
      if (src)  src.delete();
      if (rgb)  rgb.delete();
      if (mask) mask.delete();
      if (dst)  dst.delete();
    }
  }
  /** Erase every given region in one inpaint pass (all rectangles painted
   *  onto a single mask), instead of one full read/inpaint/encode cycle per
   *  block — which is what made auto-cleaning a busy document (dozens of
   *  OCR'd lines) far too slow to run on every upload. Returns a data URL,
   *  or null if OpenCV isn't ready / there's nothing to erase. */
  async function inpaintRegions(imageElement, bboxList) {
    if (!cvReady || typeof cv === 'undefined' || !bboxList || !bboxList.length) return null;
    let src, rgb, mask, dst;
    try {
      src = cv.imread(imageElement);
      rgb = new cv.Mat();
      cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);

      mask = new cv.Mat(rgb.rows, rgb.cols, cv.CV_8UC1, new cv.Scalar(0));
      let any = false;
      bboxList.forEach((bbox) => { if (_paintInkMask(src, mask, bbox)) any = true; });
      if (!any) return null;

      dst = new cv.Mat();
      cv.inpaint(rgb, mask, dst, INPAINT_RADIUS, cv.INPAINT_TELEA);

      const hiddenCanvas = document.createElement('canvas');
      cv.imshow(hiddenCanvas, dst);
      return hiddenCanvas.toDataURL('image/png');
    } catch (e) {
      console.warn('[ImageProcessor] Batch inpaint failed:', _cvError(e).message);
      return null;
    } finally {
      if (src)  src.delete();
      if (rgb)  rgb.delete();
      if (mask) mask.delete();
      if (dst)  dst.delete();
    }
  }

  /** Copy one block's area back from the original image, undoing its erase.
   *  Padded by INK_PAD plus _paintInkMask's dilation radius, which is how far
   *  the erase can reach past the OCR box. */
  async function restoreRegion(imageElement, originalSrc, bbox) {
    const orig = await _loadImageElement(originalSrc);
    const W = imageElement.naturalWidth, H = imageElement.naturalHeight;
    const pad = INK_PAD + Math.max(2, Math.round(bbox.height * 0.06)) + 1;
    const x = Math.max(0, Math.round(bbox.x) - pad), y = Math.max(0, Math.round(bbox.y) - pad);
    const w = Math.min(W, Math.round(bbox.x + bbox.width) + pad) - x;
    const h = Math.min(H, Math.round(bbox.y + bbox.height) + pad) - y;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    if (w > 0 && h > 0) ctx.drawImage(orig, x, y, w, h, x, y, w, h);
    AppState.bgSrc = canvas.toDataURL('image/png');
    await _replaceImageSrcQuietly(imageElement, AppState.bgSrc);
  }

  return { inpaintRegion, inpaintRegions, restoreRegion };
})();

/* Padding (px) around an OCR ink box when sampling and masking — OCR boxes
   clip anti-aliased glyph edges by a pixel or two. */
const INK_PAD = 4;

/**
 * Separate a text box into background and ink from raw RGBA pixels.
 *   bg        — per-channel median of a 2-px ring just outside the padded box
 *   ink       — mean colour of the pixels furthest from bg (the glyph cores)
 *   threshold — L1 distance from bg above which a pixel counts as ink
 * Returns null when the box has no ink clearly distinct from its background.
 */
function _inkStats(px, cols, rows, box) {
  const x1 = Math.max(0, box.x - INK_PAD), y1 = Math.max(0, box.y - INK_PAD);
  const x2 = Math.min(cols, box.x + box.w + INK_PAD), y2 = Math.min(rows, box.y + box.h + INK_PAD);
  if (x2 - x1 < 3 || y2 - y1 < 3) return null;

  const ring = [[], [], []];
  for (let r = y1; r < y2; r++) {
    for (let c = x1; c < x2; c++) {
      if (r >= y1 + 2 && r < y2 - 2 && c >= x1 + 2 && c < x2 - 2) continue;
      const i = (r * cols + c) * 4;
      ring[0].push(px[i]); ring[1].push(px[i + 1]); ring[2].push(px[i + 2]);
    }
  }
  const median = (a) => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
  const bg = ring.map(median);

  const bx1 = Math.max(0, box.x), by1 = Math.max(0, box.y);
  const bx2 = Math.min(cols, box.x + box.w), by2 = Math.min(rows, box.y + box.h);
  const dist = [];
  for (let r = by1; r < by2; r++) {
    for (let c = bx1; c < bx2; c++) {
      const i = (r * cols + c) * 4;
      dist.push(Math.abs(px[i] - bg[0]) + Math.abs(px[i + 1] - bg[1]) + Math.abs(px[i + 2] - bg[2]));
    }
  }
  if (!dist.length) return null;
  const peak = Float32Array.from(dist).sort()[Math.floor(dist.length * 0.98)];
  if (peak < 40) return null;

  // Ink colour = mean of the most common coarse colour among glyph cores.
  // A plain mean of every core pixel turns a two-tone line (an orange-and-blue
  // logo) into a muddy colour that matches neither part.
  const buckets = new Map(); let k = 0;
  for (let r = by1; r < by2; r++) {
    for (let c = bx1; c < bx2; c++, k++) {
      if (dist[k] < peak * 0.6) continue;
      const i = (r * cols + c) * 4;
      const key = ((px[i] >> 5) << 6) | ((px[i + 1] >> 5) << 3) | (px[i + 2] >> 5);
      const b = buckets.get(key) || { n: 0, s: [0, 0, 0] };
      b.n++; b.s[0] += px[i]; b.s[1] += px[i + 1]; b.s[2] += px[i + 2];
      buckets.set(key, b);
    }
  }
  let top = null;
  buckets.forEach((b) => { if (!top || b.n > top.n) top = b; });
  const ink = top.s.map((s) => Math.round(s / top.n));
  return { bg, ink, threshold: Math.max(30, peak * 0.35) };
}

function _rgbToHex([r, g, b]) {
  return '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase();
}


/* ══════════════════════════════════════════════════════════════════════
   OFFLINE OCR — TextDetector (native) → Tesseract.js (CDN / local)
   ─────────────────────────────────────────────────────────────────────
   Fix summary (v3.1):
   • _initInternal now loads Tesseract.js from local vendor first, then
     automatically falls back to the CDN if local files are absent.
   • Uses Tesseract.js v4 API (createWorker(lang, oem, opts)) which
     auto-initialises the language; falls back to v2/v3 API on error.
   • No blocking of the UI while the OCR engine loads.
══════════════════════════════════════════════════════════════════════ */
const OfflineOCR = (() => {
  let _engine      = 'none';
  let _initPromise = null;
  let _worker      = null;
  let _lastError   = null;

  async function init() {
    if (_initPromise) return _initPromise;
    _initPromise = _initInternal();
    return _initPromise;
  }
  function engine()    { return _engine; }
  function isReady()   { return _engine !== 'none'; }
  function lastError() { return _lastError; }

  /** All three local assets must be present; a partial vendor dir is worse
   *  than none, because Tesseract would half-load and then stall. */
  async function _haveLocalTessAssets() {
    const probes = [
      OFFLINE_OCR_WORKER_LOCAL,
      OFFLINE_OCR_CORE_LOCAL,
      OFFLINE_OCR_TESSDATA_LOCAL + OFFLINE_OCR_LANG + '.traineddata',
    ];
    try {
      const results = await Promise.all(
        probes.map((u) => fetch(u, { method: 'HEAD' }).then((r) => r.ok).catch(() => false))
      );
      const ok = results.every(Boolean);
      if (!ok) {
        console.info('[OCR] Local Tesseract assets incomplete:',
          probes.filter((_, i) => !results[i]).join(', '));
      }
      return ok;
    } catch (_) { return false; }
  }

  /* ─── core initialisation ─────────────────────────────────────── */
  async function _initInternal() {
    _lastError = null;

    /* ① Native TextDetector (Chrome Canary / FLAG #text-detection-api) */
    if (typeof TextDetector !== 'undefined') {
      try {
        new TextDetector();           // throws DOMException if not truly supported
        _engine = 'text-detector';
        console.info('[OCR] Using native TextDetector.');
        return;
      } catch (_e) { /* not truly available */ }
    }

    /* ② Tesseract.js ─ local vendor → CDN fallback ──────────────── */
    try {
      let loadedLocally = (typeof Tesseract !== 'undefined');
      if (typeof Tesseract === 'undefined') {
        // Try local first (works fully offline)
        try {
          await _loadScript(OFFLINE_OCR_TESSERACT_LOCAL);
          loadedLocally = (typeof Tesseract !== 'undefined');
          if (loadedLocally) console.info('[OCR] Tesseract.js loaded from local vendor.');
        } catch (_e) { /* no local file — expected */ }

        // CDN fallback
        if (!loadedLocally) {
          console.info('[OCR] Local Tesseract not found — loading from CDN…');
          _setBadge('badge-ocr', 'loading', 'OCR: downloading…');
          await _loadScript(OFFLINE_OCR_TESSERACT_CDN);
        }
      }

      if (typeof Tesseract === 'undefined') {
        throw new Error(
          'Tesseract.js unavailable. ' +
          'Check network access or add vendor files at ./vendor/tesseract/tesseract.min.js'
        );
      }

      const logFn = (m) => {
        if (m && m.status === 'recognizing text') {
          _showProgress(Math.min(75, Math.max(20, Math.round(m.progress * 70))));
        }
      };

      /* Point the worker at whichever asset set actually exists.
         Tesseract.js defaults every one of these to a CDN, so without this
         block the app downloads ~25 MB on each cold start and fails outright
         with no network — which defeats the entire point of the vendor dir. */
      const opts = { logger: logFn };
      if (loadedLocally && await _haveLocalTessAssets()) {
        opts.workerPath = absUrl(OFFLINE_OCR_WORKER_LOCAL);
        opts.corePath   = absUrl(OFFLINE_OCR_CORE_LOCAL);
        opts.langPath   = absUrl(OFFLINE_OCR_TESSDATA_LOCAL);
        // The vendored eng.traineddata is uncompressed; the default `true`
        // would request eng.traineddata.gz and 404.
        opts.gzip       = false;
        opts.cacheMethod = 'none';   // no IndexedDB round-trip for a local file
        console.info('[OCR] Using fully local Tesseract assets.');
      } else {
        opts.langPath = OFFLINE_OCR_TESSDATA_CDN;
        console.info('[OCR] Falling back to Tesseract CDN assets.');
      }

      /* Try Tesseract.js v4 API first (createWorker(lang, oem, opts))
         which auto-loads the language data. If that fails (v2/v3 build),
         fall back to the explicit loadLanguage / initialize sequence. */
      try {
        _worker = await Tesseract.createWorker(OFFLINE_OCR_LANG, 1, opts);
        console.info('[OCR] Tesseract.js v4 worker ready.');
      } catch (_v4Err) {
        console.warn('[OCR] v4 API failed, trying v2/v3 API…', _v4Err.message);
        _worker = await Tesseract.createWorker(opts);
        await _worker.loadLanguage(OFFLINE_OCR_LANG);
        await _worker.initialize(OFFLINE_OCR_LANG);
        console.info('[OCR] Tesseract.js v2/v3 worker ready.');
      }

      // Default PSM (AUTO) tries to reconstruct paragraph/column reading
      // order, which merges unrelated text across columns on forms and
      // invoices (a billing-address column and a shipping-address column
      // at the same height become one garbled "line"). SPARSE_TEXT finds
      // each line on its own instead, which is also what this app wants —
      // one OCR block per editable line, regardless of page layout.
      try { await _worker.setParameters({ tessedit_pageseg_mode: '11' }); }
      catch (psmErr) { console.warn('[OCR] Could not set sparse-text PSM:', psmErr.message); }

      _engine = 'tesseract';
    } catch (err) {
      _engine    = 'none';
      _lastError = err;
      throw err;
    }
  }

  /* ─── public recognise entry point ──────────────────────────── */
  async function recognize(imgEl) {
    if (!_initPromise) await init();
    if (_engine === 'text-detector') return _recognizeWithTextDetector(imgEl);
    if (_engine === 'tesseract')     return _recognizeWithTesseract(imgEl);
    throw (_lastError || new Error('Offline OCR engine not available.'));
  }

  /* ─── helpers ────────────────────────────────────────────────── */
  function _rasterizeForOcr(imgEl) {
    const w = imgEl.naturalWidth  || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;
    const scale = Math.min(1, OFFLINE_OCR_MAX_DIM / Math.max(w, h));
    if (scale >= 1) return { source: imgEl, scale, width: w, height: h };
    const canvas = document.createElement('canvas');
    canvas.width  = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    canvas.getContext('2d').drawImage(imgEl, 0, 0, canvas.width, canvas.height);
    return { source: canvas, scale, width: canvas.width, height: canvas.height };
  }

  function _normalizeBox(x, y, w, h, maxW, maxH) {
    const nx = Math.max(0, Math.min(x, maxW - 1));
    const ny = Math.max(0, Math.min(y, maxH - 1));
    const nw = Math.max(1, Math.min(w, maxW - nx));
    const nh = Math.max(1, Math.min(h, maxH - ny));
    return { x: nx, y: ny, w: nw, h: nh };
  }

  async function _recognizeWithTextDetector(imgEl) {
    const { source, scale } = _rasterizeForOcr(imgEl);
    const detector = new TextDetector();
    const results  = await detector.detect(source);
    const blocks   = results.map((r) => {
      const box = r.boundingBox || { x: 0, y: 0, width: 0, height: 0 };
      const x = Math.round(box.x / scale), y = Math.round(box.y / scale);
      const w = Math.round(box.width / scale), h = Math.round(box.height / scale);
      const norm = _normalizeBox(x, y, w, h, imgEl.naturalWidth, imgEl.naturalHeight);
      return { text: (r.rawValue || r.text || '').trim(), ...norm, confidence: 0.6 };
    }).filter(b => b.text.length > 0);
    const plainText = results.map(r => (r.rawValue || r.text || '').trim()).filter(Boolean).join('\n');
    return { blocks, plainText, engine: 'text-detector' };
  }

  async function _recognizeWithTesseract(imgEl) {
    if (!_worker) throw new Error('Tesseract worker not initialised.');
    const { source, scale } = _rasterizeForOcr(imgEl);
    const { data }   = await _worker.recognize(source);

    // Support both word-level (older builds) and line-level output
    const lines = (data && data.lines && data.lines.length)
      ? data.lines
      : (data && data.words ? data.words : []);

    const blocks = lines.map((line) => {
      const bbox = line.bbox || { x0: 0, y0: 0, x1: 0, y1: 0 };
      const x = Math.round(bbox.x0 / scale);
      const y = Math.round(bbox.y0 / scale);
      const w = Math.round((bbox.x1 - bbox.x0) / scale);
      const h = Math.round((bbox.y1 - bbox.y0) / scale);
      const norm = _normalizeBox(x, y, w, h, imgEl.naturalWidth, imgEl.naturalHeight);
      const bl = line.baseline;
      return {
        text: (line.text || '').trim(),
        ...norm,
        confidence: typeof line.confidence === 'number'
          ? Math.max(0, Math.min(1, line.confidence / 100)) : 0.6,
        ...(bl && bl.has_baseline !== false && isFinite(bl.y0) && isFinite(bl.y1)
          ? { baseline: Math.round((bl.y0 + bl.y1) / 2 / scale) } : {}),
      };
    }).filter(b => b.text.length > 0);

    const plainText = (data && data.text ? data.text.trim() : '');
    return { blocks, plainText, engine: 'tesseract' };
  }

  return { init, recognize, isReady, engine, lastError };
})();


/* ══════════════════════════════════════════════════════════════════════
   APP STATE
══════════════════════════════════════════════════════════════════════ */
const AppState = {
  payload: null, imageW: 0, imageH: 0, blocks: [], bgSrc: '', originalSrc: '',
  liveBlocks: [], activeId: null, scaleFactor: 1, zoomLevel: 1,
  isLoaded: false, ocrText: '', ocrEngine: 'none',
  // What was uploaded, for "Same as upload" export: format key, file name
  // without extension, and (PDF only) page size in points.
  sourceFormat: 'png', sourceName: '', sourcePagePt: null,
  clear() {
    this.payload = null; this.imageW = 0; this.imageH = 0;
    this.blocks = []; this.liveBlocks = []; this.bgSrc = ''; this.originalSrc = '';
    this.activeId = null; this.scaleFactor = 1; this.zoomLevel = 1;
    this.isLoaded = false; this.ocrText = ''; this.ocrEngine = 'none';
  }
};


/* ══════════════════════════════════════════════════════════════════════
   UNDO / REDO
══════════════════════════════════════════════════════════════════════ */
const EditorState = (() => {
  const undoStack = [], redoStack = [];
  const MAX_DEPTH = 80;

  function push(blockId, oldText, newText) {
    if (oldText === newText) return;
    undoStack.push({ blockId, oldText, newText });
    if (undoStack.length > MAX_DEPTH) undoStack.shift();
    redoStack.length = 0;
    _syncButtons();
  }
  function undo() {
    if (!undoStack.length) return;
    const e = undoStack.pop(); redoStack.push(e);
    _applyText(e.blockId, e.oldText); _syncButtons(); Toast.show('Undo', 'info');
  }
  function redo() {
    if (!redoStack.length) return;
    const e = redoStack.pop(); undoStack.push(e);
    _applyText(e.blockId, e.newText); _syncButtons(); Toast.show('Redo', 'info');
  }
  function clear() { undoStack.length = 0; redoStack.length = 0; _syncButtons(); }

  function _applyText(blockId, text) {
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (live && live.el) { live.el.textContent = text; live.currentText = text; }
  }
  function _syncButtons() {
    const u = document.getElementById('btn-undo'), r = document.getElementById('btn-redo');
    if (u) u.disabled = undoStack.length === 0;
    if (r) r.disabled = redoStack.length === 0;
  }
  return { push, undo, redo, clear };
})();


/* ══════════════════════════════════════════════════════════════════════
   SCALE ENGINE
══════════════════════════════════════════════════════════════════════ */
const ScaleEngine = (() => {
  function recompute() {
    const img = document.getElementById('canvas-img');
    if (!img || !AppState.imageW) return;
    const sf = img.getBoundingClientRect().width / AppState.imageW;
    AppState.scaleFactor = (sf > 0 && isFinite(sf)) ? sf : 1;
    if (AppState.liveBlocks.length) OverlayEngine.repositionAll();
  }
  function toDisplay(v) { return Math.round(v * AppState.scaleFactor); }
  function toNative(v)  { return Math.round(v / AppState.scaleFactor); }
  return { recompute, toDisplay, toNative };
})();


/* ══════════════════════════════════════════════════════════════════════
   OVERLAY ENGINE
══════════════════════════════════════════════════════════════════════ */
const OverlayEngine = (() => {
  function _makeId(index) { return `block-${index}-${Date.now()}`; }

  function renderAll(blocks) {
    const overlay = document.getElementById('overlay');
    overlay.innerHTML = '';
    AppState.liveBlocks = [];
    blocks.forEach((block, idx) => {
      const id = _makeId(idx);
      const live = { id, ...block,
        font_family: block.font_family || block.fontFamily || 'Arial',
        currentText: block.text, originalText: block.text, el: null };
      live.el = _createField(live);
      AppState.liveBlocks.push(live);
      overlay.appendChild(live.el);
    });
    LayerPanel.rebuild();
  }

  function _createField(live) {
    const s  = AppState.scaleFactor;
    const el = document.createElement('div');
    el.id              = `field-${live.id}`;
    el.contentEditable = 'true';
    el.className       = 'txt-block';
    el.dataset.blockId = live.id;
    el.spellcheck      = false;
    el.textContent     = live.currentText;

    _layoutField(el, live, s);
    el.style.color      = live.color || '#1A1A1A';
    el.style.fontFamily = resolveFontStack(live.font_family);
    el.style.fontWeight = live.font_weight || 'normal';
    el.style.fontStyle  = live.font_style  || 'normal';
    el.style.lineHeight = String(TEXT_LINE_HEIGHT);
    if (live.bg_color) el.style.backgroundColor = _hexWithAlpha(live.bg_color, 0.0);

    let _textOnFocus = '';
    el.addEventListener('focus', () => { _textOnFocus = el.textContent; _selectAllText(el); _setActive(live.id); });
    el.addEventListener('blur', () => {
      const newText = el.textContent;
      EditorState.push(live.id, _textOnFocus, newText);
      live.currentText = newText;
      const layerEl = document.getElementById(`lyr-text-${live.id}`);
      if (layerEl) layerEl.textContent = `"${newText}"`;
      _setOcrTextFromBlocks(AppState.liveBlocks.map(b => ({ text: b.currentText })));
    });

    let _clickCount = 0;
    el.addEventListener('click', async () => {
      if (live.keep_image) { _setActive(live.id); return; }
      _clickCount++;
      if (_clickCount === 1) {
        _selectAllText(el);
        setTimeout(() => { _clickCount = 0; }, 600);
      } else if (_clickCount === 2) {
        const imgEl = document.getElementById('canvas-img');
        if (imgEl && !live.inpainted) {
          try {
            if (!AIRuntime.isCvReady()) await AIRuntime.waitUntilReady();
            await ImageProcessor.inpaintRegion(imgEl, { x: live.x, y: live.y, width: live.ink_w || live.w, height: live.h });
            live.inpainted = true;
            Toast.show('Original text erased', 'success');
          } catch (err) { Toast.show('Inpaint failed: ' + err.message, 'error'); }
        }
      }
    });
    el.addEventListener('keydown', (e) => { if (e.key === 'Escape') { el.blur(); e.preventDefault(); } });
    el.addEventListener('touchstart', (e) => { e.stopPropagation(); _setActive(live.id); }, { passive: true });
    el.addEventListener('touchend', (e) => {
      e.stopPropagation();
      if (document.activeElement !== el) { el.focus(); _selectAllText(el); }
    }, { passive: true });
    _applyKeepImage(el, live);
    return el;
  }

  /** A block kept as image shows the original pixels underneath: its field
   *  stays selectable (so it can be switched back) but is not editable and
   *  draws no text. */
  function _applyKeepImage(el, live) {
    el.contentEditable = live.keep_image ? 'false' : 'true';
    el.classList.toggle('as-image', !!live.keep_image);
  }

  /** Switch a block between editable text (original erased) and original
   *  image (original pixels restored, no text drawn). */
  async function setKeepImage(blockId, keep) {
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (!live || !!live.keep_image === keep) return;
    const imgEl = document.getElementById('canvas-img');
    const bbox  = { x: live.x, y: live.y, width: live.ink_w || live.w, height: live.h };
    if (keep) {
      await ImageProcessor.restoreRegion(imgEl, AppState.originalSrc, bbox);
    } else {
      if (!AIRuntime.isCvReady()) await AIRuntime.waitUntilReady();
      await ImageProcessor.inpaintRegion(imgEl, bbox);
    }
    live.keep_image = keep;
    live.inpainted  = !keep;
    if (keep && document.activeElement === live.el) live.el.blur();
    _applyKeepImage(live.el, live);
    LayerPanel.rebuild();
    _setActive(blockId);
  }

  /** Place a field so its *content* box covers the block — see FIELD_INSET_X. */
  function _layoutField(el, live, s) {
    const fs = live.size * s;   // fractional: rounding after zoom made lines visibly wider
    el.style.left      = `${Math.round(live.x * s - FIELD_INSET_X)}px`;
    el.style.top       = `${_overlayTop(live, s, fs)}px`;
    el.style.width     = `${Math.round(live.w * s + 2 * FIELD_INSET_X)}px`;
    el.style.minHeight = `${Math.round(live.h * s + 2 * FIELD_INSET_Y)}px`;
    el.style.fontSize  = `${fs}px`;
  }

  function repositionAll() {
    const s = AppState.scaleFactor;
    AppState.liveBlocks.forEach(live => { if (live.el) _layoutField(live.el, live, s); });
  }

  function updateBlockStyle(blockId, { font_family, size, color, font_weight, font_style }) {
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (!live || !live.el) return;
    const s = AppState.scaleFactor;
    if (font_weight !== undefined) { live.font_weight = font_weight; live.el.style.fontWeight = font_weight; }
    if (font_style  !== undefined) { live.font_style  = font_style;  live.el.style.fontStyle  = font_style; }
    if (font_weight !== undefined || font_style !== undefined) _refreshLayerMeta(live);
    if (font_family !== undefined) {
      live.font_family = font_family; live.el.style.fontFamily = resolveFontStack(font_family);
      _refreshLayerMeta(live);
      // A different face has a different ascent; re-pin the baseline.
      live.el.style.top = `${_overlayTop(live, s, live.size * s)}px`;
    }
    if (size !== undefined) {
      live.size = size;
      const fs = size * s;
      live.el.style.fontSize = `${fs}px`;
      live.el.style.top      = `${_overlayTop(live, s, fs)}px`;   // keep the baseline anchored
    }
    if (color !== undefined) {
      live.color = color; live.el.style.color = color;
      const swatch = document.getElementById(`lyr-swatch-${blockId}`);
      if (swatch) swatch.style.background = color;
    }
  }

  function _refreshLayerMeta(live) {
    const meta = document.querySelector(`.layer-item[data-block-id="${live.id}"] .layer-meta`);
    if (meta) meta.textContent = LayerPanel.metaFor(live);
  }

  function _setActive(blockId) {
    AppState.activeId = blockId;
    document.querySelectorAll('.txt-block').forEach(el => el.classList.toggle('selected', el.dataset.blockId === blockId));
    document.querySelectorAll('.layer-item').forEach(el => el.classList.toggle('active', el.dataset.blockId === blockId));
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (live) PropsPanel.populate(live);
  }

  function _selectAllText(el) {
    const range = document.createRange(); range.selectNodeContents(el);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
  }

  function _hexWithAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function setActive(blockId)  { _setActive(blockId); }
  function clearActive() {
    AppState.activeId = null;
    document.querySelectorAll('.txt-block').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.layer-item').forEach(el => el.classList.remove('active'));
    PropsPanel.clear();
  }

  return { renderAll, repositionAll, updateBlockStyle, setActive, clearActive, setKeepImage };
})();


/* ══════════════════════════════════════════════════════════════════════
   PROPS PANEL
══════════════════════════════════════════════════════════════════════ */
const PropsPanel = (() => {
  /** Fill the font <select> with the fonts installed here, grouped. */
  function buildFontOptions() {
    const sel = document.getElementById('prop-font');
    if (!sel) return;
    sel.innerHTML = '';
    FontMatcher.groups().forEach((grp) => {
      const og = document.createElement('optgroup');
      og.label = grp.label;
      grp.fonts.forEach((f) => { const o = document.createElement('option'); o.value = o.textContent = f; og.appendChild(o); });
      sel.appendChild(og);
    });
  }
  function _ensureFontOption(family) {
    const sel = document.getElementById('prop-font');
    if (!sel || [...sel.options].some((o) => o.value === family)) return;
    const o = document.createElement('option'); o.value = o.textContent = family;
    sel.insertBefore(o, sel.firstChild);
  }

  function populate(live) {
    _ensureFontOption(live.font_family || 'Arial');
    _setVal('prop-font',      live.font_family || 'Arial');
    _setVal('prop-size',      live.size || 16);
    _setVal('prop-color',     live.color || '#000000');
    _setVal('prop-color-hex', live.color || '#000000');
    _setPressed('prop-bold',   live.font_weight === 'bold');
    _setPressed('prop-italic', live.font_style === 'italic');
    document.getElementById('swatch-fg').style.background = live.color || '#000000';
    document.getElementById('coord-x').textContent = live.x + ' px';
    document.getElementById('coord-y').textContent = live.y + ' px';
    document.getElementById('coord-w').textContent = live.w + ' px';
    document.getElementById('coord-h').textContent = live.h + ' px';
    document.getElementById('info-original').textContent = live.originalText;
    document.getElementById('info-conf').textContent = live.confidence ? (live.confidence * 100).toFixed(1) + '%' : '—';
    const fontAiEl = document.getElementById('info-font-ai');
    if (fontAiEl) {
      fontAiEl.textContent = (live.font_candidates && live.font_candidates.length)
        ? live.font_candidates.map((c) => `${c.family} ${Math.round(c.score * 100)}%`).join(' · ')
        : (live.font_family || '—');
    }
    _enableAll(true);
    const keepBtn = document.getElementById('btn-keep-image');
    if (keepBtn) keepBtn.textContent = live.keep_image ? 'Make this editable text' : 'Keep as original image';
    const panel = document.getElementById('props-panel');
    if (panel) { panel.classList.add('has-selection'); panel.classList.toggle('is-image', !!live.keep_image); }
  }
  function clear() {
    _enableAll(false);
    ['coord-x','coord-y','coord-w','coord-h'].forEach(id => { document.getElementById(id).textContent = '—'; });
    document.getElementById('info-original').textContent = '—';
    document.getElementById('info-conf').textContent = '—';
    const fontAiEl = document.getElementById('info-font-ai');
    if (fontAiEl) fontAiEl.textContent = '—';
    const panel = document.getElementById('props-panel');
    if (panel) panel.classList.remove('has-selection', 'is-image');
  }
  function toggleBold() {
    const live = AppState.liveBlocks.find(b => b.id === AppState.activeId); if (!live) return;
    const v = live.font_weight === 'bold' ? 'normal' : 'bold';
    OverlayEngine.updateBlockStyle(live.id, { font_weight: v }); _setPressed('prop-bold', v === 'bold');
  }
  function toggleItalic() {
    const live = AppState.liveBlocks.find(b => b.id === AppState.activeId); if (!live) return;
    const v = live.font_style === 'italic' ? 'normal' : 'italic';
    OverlayEngine.updateBlockStyle(live.id, { font_style: v }); _setPressed('prop-italic', v === 'italic');
  }
  function _setPressed(id, on) { const el = document.getElementById(id); if (el) el.setAttribute('aria-pressed', on ? 'true' : 'false'); }
  function applyFont()  { if (!AppState.activeId) return; OverlayEngine.updateBlockStyle(AppState.activeId, { font_family: document.getElementById('prop-font').value }); }
  function applySize()  { if (!AppState.activeId) return; const v = parseInt(document.getElementById('prop-size').value, 10); if (!isNaN(v) && v > 0) OverlayEngine.updateBlockStyle(AppState.activeId, { size: v }); }
  function stepSize(d)  { const i = document.getElementById('prop-size'); const v = parseInt(i.value, 10) + d; if (v >= 1 && v <= 400) { i.value = v; applySize(); } }
  function applyColor() {
    if (!AppState.activeId) return;
    const val = document.getElementById('prop-color').value;
    document.getElementById('prop-color-hex').value = val;
    document.getElementById('swatch-fg').style.background = val;
    OverlayEngine.updateBlockStyle(AppState.activeId, { color: val });
  }
  function applyColorHex() {
    if (!AppState.activeId) return;
    let val = document.getElementById('prop-color-hex').value.trim();
    if (!val.startsWith('#')) val = '#' + val;
    if (!/^#[0-9A-Fa-f]{6}$/.test(val)) return;
    document.getElementById('prop-color').value = val;
    document.getElementById('swatch-fg').style.background = val;
    OverlayEngine.updateBlockStyle(AppState.activeId, { color: val });
  }
  function applyAll() { if (!AppState.activeId) return; applyFont(); applySize(); applyColor(); Toast.show('Typography applied', 'success'); }
  async function toggleKeepImage() {
    const live = AppState.liveBlocks.find(b => b.id === AppState.activeId);
    if (!live) return;
    const keep = !live.keep_image;
    try {
      await OverlayEngine.setKeepImage(live.id, keep);
      Toast.show(keep ? 'Kept as original image' : 'Block is now editable text', 'success');
    } catch (err) { Toast.show('Could not switch block: ' + err.message, 'error'); }
  }
  function _setVal(id, val) { const el = document.getElementById(id); if (el) el.value = val; }
  function _enableAll(enabled) {
    ['prop-font','prop-size','prop-color','prop-color-hex','prop-bold','prop-italic','btn-apply-all','btn-keep-image'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = !enabled; });
    document.querySelectorAll('#stepper-size .step-btn').forEach(btn => { btn.disabled = !enabled; });
  }
  return { buildFontOptions, populate, clear, applyFont, applySize, stepSize, applyColor, applyColorHex,
           applyAll, toggleBold, toggleItalic, toggleKeepImage };
})();


/* ══════════════════════════════════════════════════════════════════════
   LAYER PANEL
══════════════════════════════════════════════════════════════════════ */
const LayerPanel = (() => {
  function rebuild() {
    const list  = document.getElementById('layer-list');
    const empty = document.getElementById('empty-state');
    const count = document.getElementById('layer-count');
    list.innerHTML = '';
    if (!AppState.liveBlocks.length) { list.appendChild(empty); count.textContent = '0'; return; }
    count.textContent = AppState.liveBlocks.length;
    AppState.liveBlocks.forEach((live, i) => {
      const item   = document.createElement('div');
      item.className = 'layer-item'; item.dataset.blockId = live.id;
      item.style.animationDelay = `${i * 18}ms`;
      const swatch = document.createElement('div');
      swatch.className = 'layer-swatch'; swatch.id = `lyr-swatch-${live.id}`;
      swatch.style.background = live.color || '#888';
      const text = document.createElement('div');
      text.className = 'layer-text'; text.id = `lyr-text-${live.id}`;
      text.textContent = `"${live.currentText}"`;
      const meta = document.createElement('div');
      meta.className = 'layer-meta'; meta.textContent = metaFor(live);
      item.append(swatch, text, meta);
      const focusBlock = () => {
        const field = document.getElementById(`field-${live.id}`);
        if (field) { field.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); field.focus(); }
        OverlayEngine.setActive(live.id);
        if (typeof closeMobilePanels === 'function') closeMobilePanels();
      };
      item.addEventListener('click', focusBlock);
      item.addEventListener('touchend', (e) => { e.preventDefault(); focusBlock(); });
      list.appendChild(item);
    });
  }
  /** "Arial · Bold Italic", or "Original image" for a kept block. */
  function metaFor(live) {
    if (live.keep_image) return 'Original image';
    const v = [live.font_weight === 'bold' && 'Bold', live.font_style === 'italic' && 'Italic'].filter(Boolean).join(' ');
    return (live.font_family || 'Arial') + (v ? ` · ${v}` : '');
  }
  return { rebuild, metaFor };
})();


/* ══════════════════════════════════════════════════════════════════════
   EXPORT ENGINE
══════════════════════════════════════════════════════════════════════ */
const ExportEngine = (() => {
  function download() {
    if (!AppState.isLoaded) { Toast.show('No image loaded', 'error'); return; }
    Toast.show('Rendering composite…', 'info');
    _showProgress(10);
    const canvas = document.getElementById('export-canvas');
    canvas.width = AppState.imageW; canvas.height = AppState.imageH;
    const ctx = canvas.getContext('2d');
    const bgImg = new Image(); bgImg.crossOrigin = 'anonymous';
    bgImg.onload = () => {
      ctx.drawImage(bgImg, 0, 0, AppState.imageW, AppState.imageH);
      _showProgress(50);
      AppState.liveBlocks.forEach(live => {
        if (live.keep_image) return;   // original pixels are already in bgImg
        const text = live.el ? live.el.innerText : live.currentText;
        if (live.bg_color && live.bg_color !== 'transparent') { ctx.fillStyle = live.bg_color; ctx.fillRect(live.x, live.y, live.w, live.h); }
        ctx.font = _fontCss(live, live.size);
        ctx.fillStyle = live.color || '#000000';
        // Mirror _overlayTop: OCR blocks pin the alphabetic baseline; the rest
        // pin the em box top ('top'), where the half-leading offset puts it.
        const pinned = isFinite(live.baseline);
        ctx.textBaseline = pinned ? 'alphabetic' : 'top';
        _drawWrappedText(ctx, text, live.x, pinned ? live.baseline : live.y, live.w, live.size * TEXT_LINE_HEIGHT);
      });
      _showProgress(90);
      setTimeout(() => {
        let url = null;
        try {
          const fmt  = _format();
          const blob = fmt === 'pdf'
            ? _canvasToPdfBlob(canvas, AppState.sourcePagePt)
            : _dataUriToBytesBlob(canvas.toDataURL(`image/${fmt}`, JPEG_QUALITY));
          url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.download = `${AppState.sourceName || 'pixelscribe-export'}-edited.${FORMAT_EXT[fmt]}`;
          link.href = url; link.click();
          _showProgress(100); setTimeout(() => _showProgress(0), 600);
          Toast.show(`Exported ${FORMAT_LABEL[fmt]} — check Downloads ✓`, 'success');
        } catch (err) {
          Toast.show('Export failed — CORS issue with image URL', 'error');
          _showProgress(0);
        } finally {
          if (url) setTimeout(() => URL.revokeObjectURL(url), 10000);
        }
      }, 80);
    };
    bgImg.onerror = () => { Toast.show('Could not load background image', 'error'); _showProgress(0); };
    bgImg.src = AppState.bgSrc;
  }

  const JPEG_QUALITY = 0.95;
  const FORMAT_EXT   = { png: 'png', jpeg: 'jpg', webp: 'webp', pdf: 'pdf' };
  const FORMAT_LABEL = { png: 'PNG', jpeg: 'JPG', webp: 'WEBP', pdf: 'PDF' };

  /** The chosen export format; "same" follows the uploaded file's type. */
  function _format() {
    const sel = document.getElementById('export-format');
    const v = sel ? sel.value : 'same';
    return v === 'same' ? (AppState.sourceFormat || 'png') : v;
  }

  /** Name the format on every export button. */
  function syncLabels() {
    const label = FORMAT_LABEL[_format()];
    document.querySelectorAll('.export-label').forEach((el) => {
      el.textContent = el.textContent.replace(/\S+$/, label);
    });
  }

  function _dataUriToBytes(dataUri) {
    const bin = atob(dataUri.slice(dataUri.indexOf(',') + 1));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }
  function _dataUriToBytesBlob(dataUri) {
    const mime = dataUri.slice(5, dataUri.indexOf(';'));
    return new Blob([_dataUriToBytes(dataUri)], { type: mime });
  }

  /** One-page PDF holding the flattened canvas as a JPEG. The page takes the
   *  uploaded PDF's size in points when there was one; otherwise the image
   *  at 96 dpi. */
  function _canvasToPdfBlob(canvas, pagePt) {
    const jpg = _dataUriToBytes(canvas.toDataURL('image/jpeg', JPEG_QUALITY));
    const pw = (pagePt ? pagePt.w : canvas.width  * 0.75).toFixed(2);
    const ph = (pagePt ? pagePt.h : canvas.height * 0.75).toFixed(2);
    const content = `q ${pw} 0 0 ${ph} 0 0 cm /Im0 Do Q`;
    const objs = [
      '<< /Type /Catalog /Pages 2 0 R >>',
      '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pw} ${ph}] ` +
        '/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>',
      [`<< /Type /XObject /Subtype /Image /Width ${canvas.width} /Height ${canvas.height} ` +
        `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpg.length} >>\nstream\n`,
       jpg, '\nendstream'],
      `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    ];
    const enc = new TextEncoder();
    const parts = [], offsets = [];
    let pos = 0;
    const put = (p) => { const b = typeof p === 'string' ? enc.encode(p) : p; parts.push(b); pos += b.length; };
    put('%PDF-1.4\n%\xE2\xE3\xCF\xD3\n');
    objs.forEach((o, i) => {
      offsets.push(pos);
      put(`${i + 1} 0 obj\n`);
      (Array.isArray(o) ? o : [o]).forEach(put);
      put('\nendobj\n');
    });
    const xref = pos;
    put(`xref\n0 ${objs.length + 1}\n0000000000 65535 f \n` +
        offsets.map((o) => String(o).padStart(10, '0') + ' 00000 n \n').join(''));
    put(`trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
    return new Blob(parts, { type: 'application/pdf' });
  }
  /** Wraps on width and honours hard breaks the user typed with Enter.
   *  Returns the y just past the last line drawn. */
  function _drawWrappedText(ctx, text, x, y, maxWidth, lineHeight) {
    for (const paragraph of String(text).split(NEWLINE_RE)) {
      const words = paragraph.split(' ');
      let line = '';
      for (let i = 0; i < words.length; i++) {
        const testLine = line + words[i] + ' ';
        if (ctx.measureText(testLine).width > maxWidth && i > 0) {
          ctx.fillText(line.trim(), x, y); line = words[i] + ' '; y += lineHeight;
        } else line = testLine;
      }
      ctx.fillText(line.trim(), x, y);
      y += lineHeight;
    }
    return y;
  }
  return { download, syncLabels };
})();


/* ══════════════════════════════════════════════════════════════════════
   CANVAS VIEW
══════════════════════════════════════════════════════════════════════ */
/* zoomLevel is display px per image px (1 = 100% = native size). The
   workspace used to be left at the image's natural width at "100%", so a
   large photo or a 3× PDF render opened far bigger than the window. Images
   now open fitted, and stay fitted across window resizes until the user
   zooms by hand. */
const CanvasView = (() => {
  let _fitMode = true;

  function zoom(delta) {
    if (!AppState.isLoaded) return;
    _fitMode = false;
    AppState.zoomLevel = Math.min(4, Math.max(0.1, Math.round((AppState.zoomLevel + delta) * 10) / 10));
    _applyZoom();
  }
  function fitToWindow() {
    if (!AppState.isLoaded) return;
    _fitMode = true;
    AppState.zoomLevel = _fitScale();
    _applyZoom();
  }
  function refit() { if (_fitMode) fitToWindow(); }

  /** Largest scale (≤ 1) at which the whole image fits the canvas area. */
  function _fitScale() {
    const area = document.getElementById('canvas-area');
    if (!area || !AppState.imageW || !AppState.imageH) return 1;
    const cs = getComputedStyle(area);
    const availW = area.clientWidth  - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const availH = area.clientHeight - parseFloat(cs.paddingTop)  - parseFloat(cs.paddingBottom);
    const s = Math.min(availW / AppState.imageW, availH / AppState.imageH, 1);
    return (s > 0 && isFinite(s)) ? s : 1;
  }
  function _applyZoom() {
    const ws = document.getElementById('workspace'); if (!ws) return;
    ws.style.width = `${Math.floor(AppState.imageW * AppState.zoomLevel)}px`;
    const label = document.getElementById('zoom-label');
    if (label) label.textContent = Math.round(AppState.zoomLevel * 100) + '%';
    ScaleEngine.recompute();
  }
  return { zoom, fitToWindow, refit };
})();


/* ══════════════════════════════════════════════════════════════════════
   TOAST
══════════════════════════════════════════════════════════════════════ */
const Toast = (() => {
  const ICONS = { success: '✓', error: '✕', info: 'ℹ' };
  function show(message, type = 'info', duration = 2600) {
    const container = document.getElementById('toast-container'); if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-icon">${ICONS[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('show')));
    setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 250); }, duration);
  }
  return { show };
})();


/* ══════════════════════════════════════════════════════════════════════
   PDF HANDLER — PDF.js renders page 1 → PNG data URI
══════════════════════════════════════════════════════════════════════ */
async function _renderPdfToImage(file) {
  if (typeof pdfjsLib === 'undefined') {
    throw new Error('PDF.js not loaded. Add local vendor files or enable CDN access.');
  }
  await _ensurePdfWorkerSrc();
  const arrayBuffer   = await file.arrayBuffer();
  const loadingTask   = pdfjsLib.getDocument({ data: arrayBuffer });
  const pdfDoc        = await loadingTask.promise;
  const page          = await pdfDoc.getPage(1);
  const SCALE         = 3.0;                           // ≈ 300 DPI
  const viewport      = page.getViewport({ scale: SCALE });
  const offCanvas     = document.createElement('canvas');
  offCanvas.width     = Math.round(viewport.width);
  offCanvas.height    = Math.round(viewport.height);
  const offCtx        = offCanvas.getContext('2d');
  await page.render({ canvasContext: offCtx, viewport }).promise;
  const pt = page.getViewport({ scale: 1 });           // 1 unit = 1 pt
  // PNG, not JPEG: JPEG rings around sharp glyph edges, and inpainting then
  // spreads that halo into the erased letters, leaving pale "ghost" text.
  return { src: offCanvas.toDataURL('image/png'), width: offCanvas.width, height: offCanvas.height,
           pagePt: { w: pt.width, h: pt.height } };
}


/* ══════════════════════════════════════════════════════════════════════
   OPTIONAL PYTHON BACKEND (OCR + inpaint fallback)
══════════════════════════════════════════════════════════════════════ */
function getApiBaseUrl() { return (localStorage.getItem(API_BASE_STORAGE_KEY) || '').trim().replace(/\/$/, ''); }
function setApiBaseUrl(url) {
  const t = (url || '').trim().replace(/\/$/, '');
  if (t) localStorage.setItem(API_BASE_STORAGE_KEY, t);
  else localStorage.removeItem(API_BASE_STORAGE_KEY);
}

function configureOcrServer() {
  const entered = window.prompt(
    'OCR server URL (optional Python-backend fallback):\n\n' +
    '• Same Wi-Fi: http://YOUR_PC_IP:8000\n' +
    '• Android emulator: http://10.0.2.2:8000\n' +
    '• Browser on this PC: http://localhost:8000\n\n' +
    'Leave empty for fully offline mode.',
    getApiBaseUrl() || 'http://192.168.1.100:8000'
  );
  if (entered === null) return;
  setApiBaseUrl(entered);
  if (getApiBaseUrl()) Toast.show('OCR server set: ' + getApiBaseUrl(), 'success', 4000);
  else Toast.show('OCR server cleared — offline mode', 'info');
  if (typeof closeMobileMore === 'function') closeMobileMore();
}

async function _dataUriToBlob(dataUri) { const res = await fetch(dataUri); return res.blob(); }

function _readFileAsDataUri(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = (e) => resolve(e.target.result);
    reader.onerror = () => reject(new Error('FileReader failed.'));
    reader.readAsDataURL(file);
  });
}

function _loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = () => reject(new Error('Image failed to load.'));
    img.src     = src;
  });
}

/** Fraction of the smaller box's area that two boxes share. */
function _overlapRatio(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  if (inter <= 0) return 0;
  return inter / Math.min(a.w * a.h, b.w * b.h);
}

/** Greedy non-max suppression over OCR line boxes. On multi-column layouts
 *  (forms, invoices) OCR sometimes reports two overlapping boxes for text
 *  that's really one line, or a stray box nested inside a bigger one — left
 *  alone, both render as separate editable overlays stacked on the same
 *  spot, which is what made a busy document look "puzzled"/doubled even
 *  after each block's own region was correctly erased. Keep the
 *  higher-confidence box of any pair that overlaps more than 35%. */
function _suppressOverlappingBlocks(blocks, threshold = 0.35) {
  const sorted = [...blocks].sort((a, b) => (b.confidence - a.confidence) || (b.w * b.h - a.w * a.h));
  const kept = [];
  for (const b of sorted) {
    if (!kept.some((k) => _overlapRatio(b, k) > threshold)) kept.push(b);
  }
  return kept.sort((a, b) => (a.y - b.y) || (a.x - b.x));
}

/* Fractions of the em that typical Latin faces put above and below the
   baseline. Used to recover the font size from the ink height OCR reports:
   "COMPLETION" (caps only) and "Jane Sample" (ascenders + descenders) at the
   same size have very different ink boxes, and treating ink height as a
   flat ~1.2 em made all-caps headings come out at roughly 60% size. */
const CAP_HEIGHT_EM = 0.70;
const X_HEIGHT_EM   = 0.50;
const DESCENDER_EM  = 0.21;
const RISES_ABOVE_X_RE   = /[A-Z0-9bdfhiklt!?%&#@\/\\()[\]{}|'"]/;
const DROPS_BELOW_BASE_RE = /[gjpqyQ,;()[\]{}|]/;

function _fontSizeFromInk(text, inkH) {
  const above = RISES_ABOVE_X_RE.test(text) ? CAP_HEIGHT_EM : X_HEIGHT_EM;
  const below = DROPS_BELOW_BASE_RE.test(text) ? DESCENDER_EM : 0;
  return Math.max(6, Math.round(inkH / (above + below)));
}

/** OCR finds "text" in signatures, seals and faint watermarks too. Those come
 *  back with low confidence and few real characters; erasing them and
 *  overlaying the garbage string damages the image for nothing. */
function _isLikelyText(b) {
  const alnum = (b.text.match(/[\p{L}\p{N}]/gu) || []).length;
  if (b.confidence < 0.3) return false;
  if (b.confidence < 0.7 && alnum < 2) return false;
  return alnum > 0;
}

/** Tesseract reads a superscript ordinal suffix ("25ᵗʰ") as a trademark
 *  sign — "25™". Restore the suffix the number actually takes. */
function _fixOcrOrdinals(text) {
  return text.replace(/(\d+)\s?™/g, (_, n) => {
    const d = n % 10, t = n % 100;
    const suf = (t >= 11 && t <= 13) ? 'th' : d === 1 ? 'st' : d === 2 ? 'nd' : d === 3 ? 'rd' : 'th';
    return n + suf;
  });
}

function _finalizeOcrBlocks(blocks, imageW, imageH) {
  const result = (blocks || []).filter(b => _isLikelyText({ text: (b.text || '').trim(),
    confidence: typeof b.confidence === 'number' ? b.confidence : 0.5 })).map(b => {
    // OCR reports the tight ink width of the recognized glyphs. The overlay
    // renders that same text with the browser's own font metrics (often a
    // substituted font, always with box padding on top) — a box sized to
    // exactly match wraps onto a second line and collides with whatever
    // block sits below it. Pad width so single-line text has room to stay
    // on one line.
    const rawW = Math.max(2, Math.round(b.w || 0));
    const w = Math.max(2, Math.min(imageW, Math.round(rawW * 1.15) + 12));
    const h = Math.max(2, Math.round(b.h || 0));
    const x = Math.max(0, Math.min(Math.round(b.x || 0), imageW - w));
    const y = Math.max(0, Math.min(Math.round(b.y || 0), imageH - h));
    const text = _fixOcrOrdinals((b.text || '').trim());
    return {
      text, x, y, w, h,
      // The padded `w` is for the editable overlay only. Erasing and font
      // classification use the real ink width — erasing the padded box
      // reached 15% past the text into whatever graphic sat beside it.
      ink_w: Math.min(rawW, imageW - x),
      color: b.color || '#1A1A1A', bg_color: b.bg_color || 'transparent',
      size: b.size || _fontSizeFromInk(text, h),
      ...(isFinite(b.baseline) ? { baseline: b.baseline } : {}),
      confidence: typeof b.confidence === 'number' ? b.confidence : 0.5,
      // Deliberately left undefined when the OCR engine did not report one:
      // defaulting to 'Arial' here would satisfy the "already known" test in
      // _enrichBlocksWithFonts and the ONNX classifier would never run.
      // OverlayEngine falls back to Arial at render time if it stays unset.
      ...(b.font_family ? { font_family: b.font_family } : {}),
    };
  }).filter(b => b.text.length > 0);

  if (result.length) return _suppressOverlappingBlocks(result);

  // Absolute fallback: one editable placeholder so the image is still usable
  const fallbackW = Math.max(120, Math.round(imageW * 0.7));
  const fallbackH = Math.max(28,  Math.round(imageH * 0.08));
  return [{
    text: 'Edit text', x: Math.round((imageW - fallbackW) / 2), y: Math.round(imageH * 0.1),
    w: fallbackW, h: fallbackH, color: '#1A1A1A', bg_color: 'transparent',
    size: Math.max(14, Math.round(fallbackH * 0.7)), confidence: 0, font_family: 'Arial',
  }];   // placeholder text has no source glyphs to classify — Arial is right here
}

/** Give each OCR block the colour of its own glyphs. OCR reports none, and
 *  the old fixed #1A1A1A turned grey body copy and coloured headings black. */
function _sampleInkColors(imgEl, blocks) {
  if (!blocks.length) return;
  const cols = imgEl.naturalWidth || imgEl.width, rows = imgEl.naturalHeight || imgEl.height;
  const canvas = document.createElement('canvas');
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0);
  const px = ctx.getImageData(0, 0, cols, rows).data;
  blocks.forEach((b) => {
    const stats = _inkStats(px, cols, rows, { x: b.x, y: b.y, w: b.ink_w || b.w, h: b.h });
    if (stats) b.color = _rgbToHex(stats.ink);
  });
}

/* A line this many times the body-text size is a heading at least. If its
   ink is also coloured — or OCR was unsure of it, as it is with stylised
   lettering — it is a logo or wordmark: re-typesetting it in a system font
   only damages it, so it keeps its original pixels and gets no text field. */
const LOGO_SIZE_RATIO     = 1.8;
const LOGO_MIN_SATURATION = 0.35;
const LOGO_MAX_CONFIDENCE = 0.8;

function _saturation(hex) {
  const v = [1, 3, 5].map((i) => parseInt(String(hex).slice(i, i + 2), 16) / 255);
  if (v.some(isNaN)) return 0;
  const max = Math.max(...v), min = Math.min(...v);
  return max === 0 ? 0 : (max - min) / max;
}

function _markLogoBlocks(blocks) {
  if (blocks.length < 3) return;
  const sizes = blocks.map((b) => b.size).sort((p, q) => p - q);
  const body  = sizes[sizes.length >> 1];
  blocks.forEach((b) => {
    if (b.size < LOGO_SIZE_RATIO * body) return;
    if (_saturation(b.color) >= LOGO_MIN_SATURATION || b.confidence < LOGO_MAX_CONFIDENCE) b.keep_image = true;
  });
}

async function _processImageWithOfflineOcr(src, naturalW, naturalH) {
  _showProgress(20);
  const imgEl  = await _loadImageElement(src);
  const imageW = naturalW || imgEl.naturalWidth || imgEl.width;
  const imageH = naturalH || imgEl.naturalHeight || imgEl.height;

  _setStatusBusy(`Running OCR (${OfflineOCR.engine() || 'loading'})…`);
  const ocr    = await OfflineOCR.recognize(imgEl);
  const blocks = _finalizeOcrBlocks(ocr.blocks, imageW, imageH);
  _sampleInkColors(imgEl, blocks);
  _markLogoBlocks(blocks);
  _showProgress(60);

  // Erase the original pixels under each detected line so the editable
  // overlay isn't rendered on top of the still-visible source text — without
  // this every upload looked like double-exposed/"puzzled" text until each
  // block was manually double-clicked to erase it one at a time.
  let bgImage = src;
  const toErase = blocks.filter(b => !b.keep_image);
  if (toErase.length) {
    _setStatusBusy('Cleaning detected text…');
    const cleaned = await ImageProcessor.inpaintRegions(
      imgEl, toErase.map(b => ({ x: b.x, y: b.y, width: b.ink_w || b.w, height: b.h }))
    );
    if (cleaned) bgImage = cleaned;
  }
  _showProgress(70);

  return {
    bg_image:   bgImage, original_image: src, image_w: imageW, image_h: imageH, blocks,
    ocr_text:   ocr.plainText || blocks.map(b => b.text).join('\n'),
    ocr_engine: ocr.engine,
  };
}

async function _buildFallbackPayload(src, naturalW, naturalH) {
  const imgEl  = await _loadImageElement(src);
  const imageW = naturalW || imgEl.naturalWidth || imgEl.width;
  const imageH = naturalH || imgEl.naturalHeight || imgEl.height;
  const blocks = _finalizeOcrBlocks([], imageW, imageH);
  return { bg_image: src, image_w: imageW, image_h: imageH, blocks,
           ocr_text: blocks.map(b => b.text).filter(Boolean).join('\n'), ocr_engine: 'fallback' };
}

function _setOcrText(text) {
  const clean = (text || '').trim(); AppState.ocrText = clean;
  const el = document.getElementById('ocr-text-output'); if (el) el.value = clean;
}
function _setOcrTextFromBlocks(blocks) {
  _setOcrText((blocks || []).map(b => (b && b.text ? String(b.text).trim() : '')).filter(Boolean).join('\n'));
}

async function copyOcrText() {
  const text = AppState.ocrText || ''; if (!text) return;
  try {
    await navigator.clipboard.writeText(text); Toast.show('OCR text copied', 'success');
  } catch (_) {
    const el = document.getElementById('ocr-text-output');
    if (el) { el.focus(); el.select(); document.execCommand('copy'); Toast.show('OCR text copied', 'success'); }
  }
}

async function _processImageWithBackend(blob, filename) {
  const base = getApiBaseUrl();
  if (!base) throw new Error('No OCR server configured. Use More → OCR Server to set your PC URL.');
  const form = new FormData();
  form.append('file', blob, filename || 'upload.jpg');
  form.append('languages', 'en');
  form.append('confidence', '0.25');
  let res;
  try { res = await fetch(`${base}/process-image`, { method: 'POST', body: form }); }
  catch (err) { throw new Error('Cannot reach OCR server at ' + base + '. ' + err.message); }
  if (!res.ok) { const detail = await res.text(); throw new Error(`Server error ${res.status}: ${detail.slice(0, 200)}`); }
  const data = await res.json();
  const blocks = (data.blocks || []).map(b => ({
    text: b.text, x: b.x, y: b.y, w: b.w, h: b.h,
    color: b.color || '#000000', bg_color: b.bg_color,
    size: b.size || Math.max(12, b.h || 16), confidence: b.confidence,
    ...(b.font_family ? { font_family: b.font_family } : {}),   // see _finalizeOcrBlocks
  }));
  return { bg_image: data.image_b64, image_w: data.image_w, image_h: data.image_h, blocks,
           ocr_text: blocks.map(b => b.text).filter(Boolean).join('\n'), ocr_engine: 'backend' };
}

async function _processPdfFile(file) {
  Toast.show('Rendering PDF page 1…', 'info');
  _showProgress(10);
  const { src, width, height, pagePt } = await _renderPdfToImage(file);
  AppState.sourcePagePt = pagePt;
  _showProgress(35);
  try {
    const payload = await _processImageWithOfflineOcr(src, width, height);
    loadPayload(payload);
    closeModal();
    Toast.show(`PDF ready — ${payload.blocks.length} editable block(s)`, 'success', 4000);
  } catch (ocrErr) {
    console.warn('[PDF] Offline OCR failed:', ocrErr);
    if (getApiBaseUrl()) {
      Toast.show('Sending page to OCR server…', 'info');
      _showProgress(55);
      const blob    = await _dataUriToBlob(src);
      const payload = await _processImageWithBackend(blob, file.name.replace(/\.pdf$/i, '.jpg') || 'page.jpg');
      _showProgress(85);
      loadPayload(payload);
      closeModal();
      Toast.show(`PDF ready — ${payload.blocks.length} editable block(s)`, 'success', 4000);
      return;
    }
    const fallback = await _buildFallbackPayload(src, width, height);
    loadPayload(fallback);
    closeModal();
    Toast.show('OCR not available. Image loaded — add blocks manually.', 'info', 6000);
  }
}


/* ══════════════════════════════════════════════════════════════════════
   CORE LOADER
══════════════════════════════════════════════════════════════════════ */
function loadPayload(payload) {
  if (!payload || typeof payload !== 'object') { Toast.show('Invalid payload: must be a JSON object', 'error'); return; }
  if (!payload.bg_image)             { Toast.show('Payload missing "bg_image" field', 'error'); return; }
  if (!Array.isArray(payload.blocks)) { Toast.show('Payload missing "blocks" array', 'error'); return; }

  AppState.payload   = payload;
  AppState.blocks    = payload.blocks;
  AppState.bgSrc     = payload.bg_image;
  AppState.originalSrc = payload.original_image || payload.bg_image;
  AppState.ocrEngine = payload.ocr_engine || AppState.ocrEngine || 'none';
  if (payload.ocr_text) _setOcrText(payload.ocr_text);

  const img = document.getElementById('canvas-img');
  const ws  = document.getElementById('workspace');
  _setStatusBusy('Loading image…');
  _showProgress(20);

  img.onload = async () => {
    AppState.imageW   = payload.image_w || img.naturalWidth;
    AppState.imageH   = payload.image_h || img.naturalHeight;
    AppState.isLoaded = true;
    _showProgress(50);

    document.getElementById('welcome-drop').style.display = 'none';
    ws.style.display = 'block';
    CanvasView.fitToWindow();

    let enrichedBlocks = payload.blocks;
    try {
      const textImg = AppState.originalSrc && AppState.originalSrc !== AppState.bgSrc
        ? await _loadImageElement(AppState.originalSrc) : img;
      enrichedBlocks = await _enrichBlocksWithFonts(textImg, payload.blocks);
    } catch (err) {
      console.warn('[PixelScribe] Font enrichment skipped:', err);
      enrichedBlocks = payload.blocks.map(b => ({ ...b, font_family: b.font_family || b.fontFamily || 'Arial' }));
    }
    // Needs the final font, so only after classification.
    enrichedBlocks = _harmonizeParagraphs(enrichedBlocks);

    ScaleEngine.recompute();
    OverlayEngine.renderAll(enrichedBlocks);
    _ensureEditorInteractive();
    if (!payload.ocr_text) _setOcrTextFromBlocks(enrichedBlocks);
    requestAnimationFrame(() => ScaleEngine.recompute());
    _showProgress(90);

    ['btn-export','btn-export-2','btn-export-mobile','btn-reset'].forEach(id => {
      const btn = document.getElementById(id); if (btn) btn.disabled = false;
    });
    if (typeof syncMobileButtons === 'function') syncMobileButtons();

    _setStatusOk('Ready — click any text on the image to edit it');
    document.getElementById('label-dimensions').textContent = `${AppState.imageW} × ${AppState.imageH} px`;
    document.getElementById('label-blocks').textContent = `${enrichedBlocks.length} block${enrichedBlocks.length !== 1 ? 's' : ''}`;
    _showProgress(100); setTimeout(() => _showProgress(0), 400);

    if (enrichedBlocks.length === 0) Toast.show('Image loaded. Import a JSON file with "blocks" to edit text.', 'info', 5000);
    else Toast.show(`Loaded ${enrichedBlocks.length} text block(s) — tap to edit`, 'success');

    EditorState.clear();
    closeModal();
  };

  img.onerror = () => {
    _setStatusError('Failed to load image'); _showProgress(0);
    Toast.show('Could not load image. Check the URL or try a different file.', 'error');
  };
  img.src = AppState.bgSrc;
}

/** Recognise each block's font from `imgEl`, which must still show the text
 *  — i.e. the original upload, never the erased background. (Classifying the
 *  erased image used to hand the model blank paper, so its pick was noise.)
 *  FontMatcher first; the ONNX classifier only where the matcher can't
 *  isolate the ink. */
async function _enrichBlocksWithFonts(imgEl, blocks) {
  if (!blocks.length) return blocks;
  const cols = imgEl.naturalWidth || imgEl.width, rows = imgEl.naturalHeight || imgEl.height;
  const canvas = document.createElement('canvas');
  canvas.width = cols; canvas.height = rows;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0);
  const px = ctx.getImageData(0, 0, cols, rows).data;

  const enriched = [];
  for (let n = 0; n < blocks.length; n++) {
    const block = blocks[n];
    // Only classify when the source did not tell us the font. A JSON payload
    // that names one is authoritative; OCR output never does.
    const known = block.font_family || block.fontFamily;
    if (known && known !== 'sans-serif') { enriched.push({ ...block, font_family: known }); continue; }
    if (block.keep_image) { enriched.push({ ...block, font_family: 'Arial' }); continue; }   // never rendered as text

    _setStatusBusy(`Matching fonts… ${n + 1}/${blocks.length}`);
    const m = FontMatcher.match(px, cols, rows, block);
    if (m) {
      const ranked = Object.values(m.byFamily).sort((p, q) => q.score - p.score);
      enriched.push({ ...block, font_family: m.best.family, font_weight: m.best.weight,
                      font_style: m.best.style, font_scores: m.byFamily,
                      font_candidates: ranked.slice(0, 3) });
    } else if (AIRuntime.isOnnxReady()) {
      const predicted = await EdgeML.predictFont(imgEl, block.x, block.y, block.ink_w || block.w, block.h);
      enriched.push({ ...block, font_family: predicted });
    } else {
      enriched.push({ ...block, font_family: 'Arial' });
    }
    if (n % 4 === 3) await new Promise((r) => setTimeout(r, 0));   // let the status repaint
  }
  return enriched;
}

function loadImageOnly(src, naturalW, naturalH) {
  loadPayload({ bg_image: src, image_w: naturalW, image_h: naturalH, blocks: [] });
}


/* ══════════════════════════════════════════════════════════════════════
   RESET
══════════════════════════════════════════════════════════════════════ */
function resetAll() {
  if (!AppState.isLoaded) return;
  AppState.liveBlocks.forEach(live => { if (live.el) live.el.textContent = live.originalText; live.currentText = live.originalText; });
  EditorState.clear(); OverlayEngine.clearActive(); PropsPanel.clear(); LayerPanel.rebuild();
  _setOcrTextFromBlocks(AppState.liveBlocks.map(b => ({ text: b.originalText })));
  Toast.show('All edits reset to original', 'info');
}


/* ══════════════════════════════════════════════════════════════════════
   MODAL LOGIC
══════════════════════════════════════════════════════════════════════ */
let _currentTab = 'image';
function openModal()  { document.getElementById('modal-overlay').classList.add('open'); }
function closeModal() { document.getElementById('modal-overlay').classList.remove('open'); }
function switchTab(tab) {
  _currentTab = tab;
  ['json','image','url'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`pane-${t}`).classList.toggle('hidden', t !== tab);
  });
  // Picking a file loads it straight away; "Load" only means something for
  // the URL and JSON tabs.
  const load = document.getElementById('modal-load-btn');
  if (load) load.classList.toggle('hidden', tab === 'image');
}
function loadFromModal() {
  if      (_currentTab === 'json')  _loadFromJSON();
  else if (_currentTab === 'image') Toast.show('Drop or select a file in the file tab', 'info');
  else if (_currentTab === 'url')   _loadFromURL();
}
function _loadFromJSON() {
  const raw = document.getElementById('json-input').value.trim();
  if (!raw) { Toast.show('JSON textarea is empty', 'error'); return; }
  try { loadPayload(JSON.parse(raw)); }
  catch (err) { Toast.show('Invalid JSON: ' + err.message, 'error'); }
}
function _loadFromURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) { Toast.show('URL is empty', 'error'); return; }
  const img = new Image(); img.crossOrigin = 'anonymous';
  img.onload  = () => loadImageOnly(url, img.naturalWidth, img.naturalHeight);
  img.onerror = () => Toast.show('Could not load image from URL', 'error');
  img.src = url; closeModal();
}


/* ══════════════════════════════════════════════════════════════════════
   DRAG & DROP HANDLERS
══════════════════════════════════════════════════════════════════════ */
function handleDragOver(event) {
  event.preventDefault(); event.dataTransfer.dropEffect = 'copy';
  document.getElementById('welcome-drop').classList.add('drag-over');
}
function handleDragLeave() { document.getElementById('welcome-drop').classList.remove('drag-over'); }
function handleFileDrop(event)      { event.preventDefault(); document.getElementById('welcome-drop').classList.remove('drag-over'); const f = event.dataTransfer.files[0]; if (f) _processFile(f); }
function handleModalDrop(event)     { event.preventDefault(); document.getElementById('modal-drop-zone').classList.remove('drag-over'); const f = event.dataTransfer.files[0]; if (f) _processFile(f); }
// Reset .value after reading the file — otherwise re-selecting the same
// file a second time (e.g. retrying after a confusing first result) fires
// no 'change' event at all, since the input's value never changed. From the
// user's side that looks exactly like the click doing nothing.
function handleModalFileInput(evt)  { const f = evt.target.files[0]; evt.target.value = ''; if (f) _processFile(f); }
function handleHiddenFileInput(evt) { const f = evt.target.files[0]; evt.target.value = ''; if (f) _processFile(f); }

const SUPPORTED_IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'];

/**
 * _processFile(file)
 * ─────────────────────────────────────────────────────────────────
 * Routes dropped / selected files:
 *   .json                            → parse and loadPayload()
 *   .jpg .jpeg .png .webp .bmp .gif  → offline OCR → loadPayload()
 *   .pdf                             → PDF.js page 1 → offline OCR → loadPayload()
 *
 * Any other extension shows a clear error toast. Everything the browser can
 * decode into an <img> works here — OCR and the canvas export never see the
 * container format, so restricting this to JPEG only turned away PNGs
 * (by far the most common source for screenshots) for no reason.
 * ─────────────────────────────────────────────────────────────────
 */
async function _processFile(file) {
  if (!file) return;

  const name = file.name.toLowerCase();
  const isJson = file.type === 'application/json' || name.endsWith('.json');
  const isPdf  = file.type === 'application/pdf' || name.endsWith('.pdf');
  const isImage = (file.type.startsWith('image/') && !file.type.includes('svg'))
               || SUPPORTED_IMAGE_EXT.some((ext) => name.endsWith(ext));

  // Browsers can't encode BMP or GIF, so those export as PNG.
  AppState.sourceFormat = isPdf ? 'pdf'
    : (file.type === 'image/jpeg' || /\.jpe?g$/.test(name)) ? 'jpeg'
    : (file.type === 'image/webp' || name.endsWith('.webp')) ? 'webp'
    : 'png';
  AppState.sourceName   = file.name.replace(/\.[^.]+$/, '');
  AppState.sourcePagePt = null;
  ExportEngine.syncLabels();

  /* ── JSON payload ─────────────────────────────────────────────── */
  if (isJson) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try { loadPayload(JSON.parse(e.target.result)); }
      catch (err) { Toast.show('JSON parse error: ' + err.message, 'error'); }
    };
    reader.readAsText(file);
    return;
  }

  /* ── Raster image ─────────────────────────────────────────────── */
  if (isImage) {
    _setStatusBusy('Detecting text…');
    Toast.show('Detecting text…', 'info', 5000);
    _showProgress(10);
    try {
      const src     = await _readFileAsDataUri(file);
      // Ensure OCR is ready (init() is idempotent)
      if (!OfflineOCR.isReady()) {
        _setBadge('badge-ocr', 'loading', 'OCR: initialising…');
        await OfflineOCR.init();
      }
      const payload = await _processImageWithOfflineOcr(src);
      loadPayload(payload);
      closeModal();
      Toast.show(`Found ${payload.blocks.length} text block(s)`, 'success');
    } catch (ocrErr) {
      console.warn('[OCR] Offline OCR failed:', ocrErr);
      if (getApiBaseUrl()) {
        try {
          Toast.show('Running OCR on server…', 'info');
          _showProgress(30);
          const payload = await _processImageWithBackend(file, file.name);
          _showProgress(85); loadPayload(payload); closeModal();
          Toast.show(`Found ${payload.blocks.length} text block(s)`, 'success');
          return;
        } catch (serverErr) { console.error('[OCR] Server failed:', serverErr); _showProgress(0); }
      }
      // Final fallback: show the image without detected blocks
      const src      = await _readFileAsDataUri(file);
      const fallback = await _buildFallbackPayload(src);
      loadPayload(fallback); closeModal();
      Toast.show(
        'OCR could not detect text. Image loaded — add text blocks manually.',
        'info', 7000
      );
    }
    return;
  }

  /* ── PDF ─────────────────────────────────────────────────────── */
  if (isPdf) {
    try {
      await _processPdfFile(file);
    } catch (err) {
      console.error('[PDF] Failed:', err);
      _aiAlert('PDF import', err);
      _showProgress(0);
    }
    return;
  }

  /* ── Unsupported type ────────────────────────────────────────── */
  Toast.show('Unsupported file type — use an image (JPG/PNG/WEBP/BMP), a PDF, or a JSON payload.', 'error', 5000);
}


/* ══════════════════════════════════════════════════════════════════════
   STATUS BAR HELPERS
══════════════════════════════════════════════════════════════════════ */
function _setStatusBusy(msg) { _dot('amber'); _lbl(msg); }
function _setStatusOk(msg)   { _dot('green'); _lbl(msg || 'Ready'); }
function _setStatusError(msg){ _dot('red');   _lbl(msg); }
function _dot(cls) { const el = document.getElementById('dot-backend'); if (el) el.className = 'status-dot ' + cls; }
function _lbl(msg) { const el = document.getElementById('label-backend'); if (el) el.textContent = msg; }
function _showProgress(pct) {
  const track = document.getElementById('progress-track');
  const fill  = document.getElementById('progress-fill');
  if (!track || !fill) return;
  if (pct <= 0) { track.style.display = 'none'; fill.style.width = '0%'; }
  else          { track.style.display = 'block'; fill.style.width = pct + '%'; }
}
function _setBadge(id, state, text) {
  const el = document.getElementById(id); if (!el) return;
  el.className = `badge badge-${state}`; el.textContent = text;
}


/* ══════════════════════════════════════════════════════════════════════
   KEYBOARD SHORTCUTS
══════════════════════════════════════════════════════════════════════ */
document.addEventListener('keydown', (e) => {
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && !e.shiftKey && e.key === 'z') {
    if (document.activeElement.classList.contains('txt-block')) return;
    e.preventDefault(); EditorState.undo(); return;
  }
  if (ctrl && (e.shiftKey && e.key === 'z' || e.key === 'y')) {
    if (document.activeElement.classList.contains('txt-block')) return;
    e.preventDefault(); EditorState.redo(); return;
  }
  if (ctrl && e.key === 'o') { e.preventDefault(); openModal(); return; }
  if (ctrl && (e.key === 's' || e.key === 'e')) { e.preventDefault(); ExportEngine.download(); return; }
  if (e.key === 'Escape') {
    if (document.getElementById('modal-overlay').classList.contains('open')) closeModal();
    else OverlayEngine.clearActive();
    return;
  }
  if (ctrl && (e.key === '=' || e.key === '+')) { e.preventDefault(); CanvasView.zoom(0.1); }
  if (ctrl && e.key === '-') { e.preventDefault(); CanvasView.zoom(-0.1); }
  if (ctrl && e.key === '0') { e.preventDefault(); CanvasView.fitToWindow(); }
});


/* ══════════════════════════════════════════════════════════════════════
   RESIZE OBSERVER
══════════════════════════════════════════════════════════════════════ */
(() => {
  const ro = new ResizeObserver(() => {
    if (AppState.isLoaded) {
      clearTimeout(window._resizeTimer);
      window._resizeTimer = setTimeout(() => { CanvasView.refit(); ScaleEngine.recompute(); }, 60);
    }
  });
  ro.observe(document.getElementById('canvas-area'));
})();


/* ══════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════ */
(async function init() {
  console.info('[PixelScribe] v3.1 Edge AI — initialising…');
  if (window.__cvLoaded) onOpenCvLoad();   // opencv.js beat this file — see the hook above
  PropsPanel.buildFontOptions();
  PropsPanel.clear();
  _setImportEnabled(true);
  _setStatusBusy('Loading AI libraries…');

  // Start OCR init immediately (non-blocking)
  _setBadge('badge-ocr', 'loading', 'OCR initialising…');
  OfflineOCR.init()
    .then(() => {
      const lbl = OfflineOCR.engine() === 'text-detector' ? 'OCR ready (native)' : 'OCR ready (Tesseract)';
      _setBadge('badge-ocr', 'ready', lbl);
      console.info('[OCR] Engine:', OfflineOCR.engine());
    })
    .catch((err) => {
      console.warn('[PixelScribe] Offline OCR init failed:', err.message);
      _setBadge('badge-ocr', 'error', 'OCR unavailable');
    });

  // ONNX font classifier (non-blocking)
  const onnxBoot = EdgeML.init();
  AIRuntime.setOnnxInitPromise(onnxBoot);
  await onnxBoot;
  _setImportEnabled(true);

  // Canvas-area click deselects blocks
  document.getElementById('canvas-area').addEventListener('click', (e) => {
    if (e.target.id === 'canvas-area' || e.target.id === 'workspace' || e.target.id === 'canvas-img') {
      OverlayEngine.clearActive();
    }
  });

  // OpenCV.js fallback polling (Capacitor sometimes misses the onload event)
  const cvInterval = setInterval(() => {
    if (typeof cv !== 'undefined' && typeof cv.imread === 'function' && !cvReady) {
      clearInterval(cvInterval); console.info('[PixelScribe] OpenCV.js ready (interval fallback).');
      _markCvReady();
    }
  }, 200);
  setTimeout(() => clearInterval(cvInterval), 120000);

  try {
    await AIRuntime.waitUntilReady(120000);
    console.info('[PixelScribe] AI runtime fully ready.');
  } catch (err) {
    console.warn('[PixelScribe] AI partial/unavailable:', err.message);
    _setImportEnabled(true);
    _setStatusOk('Ready — import files (AI features limited)');
  }

  _refreshUiLock();

  // Demo mode
  if (new URLSearchParams(window.location.search).get('demo') === '1') _loadDemoPayload();
})();


/* ══════════════════════════════════════════════════════════════════════
   DEMO PAYLOAD  (?demo=1)
══════════════════════════════════════════════════════════════════════ */
function _loadDemoPayload() {
  const demoCanvas = document.createElement('canvas');
  demoCanvas.width = 1200; demoCanvas.height = 800;
  const ctx  = demoCanvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 1200, 800);
  grad.addColorStop(0, '#F8F4EF'); grad.addColorStop(1, '#EDE8DC');
  ctx.fillStyle = grad; ctx.fillRect(0, 0, 1200, 800);
  ctx.strokeStyle = 'rgba(0,0,0,0.04)'; ctx.lineWidth = 1;
  for (let x = 0; x < 1200; x += 60) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,800); ctx.stroke(); }
  for (let y = 0; y < 800;  y += 60) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(1200,y); ctx.stroke(); }
  ctx.fillStyle = '#FFFFFF'; ctx.shadowColor = 'rgba(0,0,0,0.1)';
  ctx.shadowBlur = 20; ctx.shadowOffsetY = 4;
  _roundRect(ctx, 80, 60, 1040, 680, 12); ctx.fill();
  ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
  ctx.fillStyle = '#6C63FF'; _roundRect(ctx, 80, 60, 1040, 8, { tl:12, tr:12, br:0, bl:0 }); ctx.fill();

  loadPayload({
    bg_image: demoCanvas.toDataURL('image/jpeg', 0.92), image_w: 1200, image_h: 800,
    blocks: [
      { text:'PixelScribe Edge AI',  x:120, y:100, w:600, h:64,  color:'#1A1A2E', size:48,  font_family:'Georgia',          confidence:0.99 },
      { text:'100% offline — runs in your browser via WebAssembly.', x:120, y:180, w:800, h:40, color:'#4A4A62', size:22, font_family:'Arial', confidence:0.97 },
      { text:'Tap any block to edit. Drop a JPG or PDF to begin.', x:120, y:240, w:720, h:32, color:'#6C63FF', size:16, font_family:'Courier New', confidence:0.95 },
      { text:'Times New Roman — classified by ONNX in-browser.', x:120, y:290, w:700, h:28, color:'#2A2A38', size:15, font_family:'Times New Roman', confidence:0.94 },
      { text:'Tesseract.js OCR runs offline — no server needed.',   x:120, y:340, w:700, h:26, color:'#444',   size:14, font_family:'Calibri',    confidence:0.91 },
    ]
  });
  Toast.show('Demo loaded — Edge AI · Offline · WASM', 'info', 4000);
}

function _roundRect(ctx, x, y, w, h, r) {
  if (typeof r === 'number') r = { tl:r, tr:r, br:r, bl:r };
  ctx.beginPath();
  ctx.moveTo(x+r.tl,y); ctx.lineTo(x+w-r.tr,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r.tr);
  ctx.lineTo(x+w,y+h-r.br); ctx.quadraticCurveTo(x+w,y+h,x+w-r.br,y+h);
  ctx.lineTo(x+r.bl,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r.bl);
  ctx.lineTo(x,y+r.tl); ctx.quadraticCurveTo(x,y,x+r.tl,y); ctx.closePath();
}