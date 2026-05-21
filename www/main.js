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
const ONNX_WASM_LOCAL = './vendor/onnx/';
const ONNX_WASM_CDN   = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
const PDF_WORKER_LOCAL = './vendor/pdfjs/pdf.worker.min.js';
const PDF_WORKER_CDN   = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

const FONT_CLASSIFIER_MODEL_URL = './models/font_classifier.onnx';
const API_BASE_STORAGE_KEY      = 'pixelscribe_api_base';

/* ── Offline OCR constants ─────────────────────────────────────────── */
const OFFLINE_OCR_LANG = 'eng';
const OFFLINE_OCR_MAX_DIM = 2000;

// Local vendor paths (used if present — works fully offline)
const OFFLINE_OCR_TESSERACT_LOCAL = './vendor/tesseract/tesseract.min.js';

// CDN fallbacks — loaded automatically when local files are absent
const OFFLINE_OCR_TESSERACT_CDN = 'https://cdn.jsdelivr.net/npm/tesseract.js@4.1.1/dist/tesseract.min.js';
const OFFLINE_OCR_TESSDATA_CDN  = 'https://tessdata.projectnaptha.com/4.0.0/';


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
  const base = await _resolveLocalOrCdnBase(ONNX_WASM_LOCAL, ONNX_WASM_CDN, 'ort-wasm.wasm');
  ort.env.wasm.wasmPaths = base;
}

async function _ensurePdfWorkerSrc() {
  if (typeof pdfjsLib === 'undefined') return;
  const base = await _resolveLocalOrCdnBase('./vendor/pdfjs/', 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/', 'pdf.worker.min.js');
  pdfjsLib.GlobalWorkerOptions.workerSrc = base + 'pdf.worker.min.js';
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
  if (typeof cv !== 'undefined') {
    if (cv.getBuildInformation) _markCvReady();
    else cv.onRuntimeInitialized = _markCvReady;
  }
}

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
  const FONT_LABELS = [
    'Arial', 'Times New Roman', 'Courier New', 'Calibri', 'Georgia',
    'Verdana', 'Roboto', 'Helvetica', 'Garamond', 'Consolas',
  ];

  let _session = null;
  const _cropCanvas = document.createElement('canvas');
  _cropCanvas.width = _cropCanvas.height = 64;
  const _cropCtx = _cropCanvas.getContext('2d', { willReadFrequently: true });

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
      _cropCtx.clearRect(0, 0, 64, 64);
      _cropCtx.drawImage(imageEl, x, y, width, height, 0, 0, 64, 64);
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
      let maxIdx = 0, maxVal = logits[0];
      for (let i = 1; i < logits.length; i++) { if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; } }
      return (maxIdx >= 0 && maxIdx < FONT_LABELS.length) ? FONT_LABELS[maxIdx] : 'Arial';
    } catch (err) {
      console.warn('[EdgeML] predictFont error:', err.message);
      return 'Arial';
    }
  }

  return { init, predictFont, FONT_LABELS };
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
  return FONT_FALLBACK_MAP[fontFamily] || `"${fontFamily}", sans-serif`;
}


/* ══════════════════════════════════════════════════════════════════════
   IMAGE PROCESSOR — OpenCV.js Inpainting
══════════════════════════════════════════════════════════════════════ */
const ImageProcessor = (() => {
  async function inpaintRegion(imageElement, bbox) {
    if (!cvReady || typeof cv === 'undefined') throw new Error('OpenCV.js not loaded yet.');
    let src, mask, dst;
    try {
      src  = cv.imread(imageElement);
      mask = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(0));
      const x1 = Math.max(0, bbox.x - 4), y1 = Math.max(0, bbox.y - 4);
      const x2 = Math.min(src.cols, bbox.x + bbox.width + 4);
      const y2 = Math.min(src.rows, bbox.y + bbox.height + 4);
      cv.rectangle(mask, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(255), -1, cv.LINE_8, 0);
      dst = new cv.Mat();
      cv.inpaint(src, mask, dst, 3, cv.INPAINT_TELEA);
      const hiddenCanvas = document.createElement('canvas');
      cv.imshow(hiddenCanvas, dst);
      AppState.bgSrc = hiddenCanvas.toDataURL('image/png');
      imageElement.src = AppState.bgSrc;
    } finally {
      if (src)  src.delete();
      if (mask) mask.delete();
      if (dst)  dst.delete();
    }
  }
  return { inpaintRegion };
})();


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
      if (typeof Tesseract === 'undefined') {
        // Try local first (works fully offline)
        let loadedLocally = false;
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

      /* Try Tesseract.js v4 API first (createWorker(lang, oem, opts))
         which auto-loads the language data. If that fails (v2/v3 build),
         fall back to the explicit loadLanguage / initialize sequence. */
      try {
        _worker = await Tesseract.createWorker(OFFLINE_OCR_LANG, 1, {
          logger:     logFn,
          langPath:   OFFLINE_OCR_TESSDATA_CDN,   // tessdata from CDN; ignored in v4 online
        });
        console.info('[OCR] Tesseract.js v4 worker ready.');
      } catch (_v4Err) {
        console.warn('[OCR] v4 API failed, trying v2/v3 API…', _v4Err.message);
        _worker = await Tesseract.createWorker({ logger: logFn });
        await _worker.loadLanguage(OFFLINE_OCR_LANG);
        await _worker.initialize(OFFLINE_OCR_LANG);
        console.info('[OCR] Tesseract.js v2/v3 worker ready.');
      }

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
      return {
        text: (line.text || '').trim(),
        ...norm,
        confidence: typeof line.confidence === 'number'
          ? Math.max(0, Math.min(1, line.confidence / 100)) : 0.6,
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
  payload: null, imageW: 0, imageH: 0, blocks: [], bgSrc: '',
  liveBlocks: [], activeId: null, scaleFactor: 1, zoomLevel: 1,
  isLoaded: false, ocrText: '', ocrEngine: 'none',
  clear() {
    this.payload = null; this.imageW = 0; this.imageH = 0;
    this.blocks = []; this.liveBlocks = []; this.bgSrc = '';
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

    el.style.left       = `${Math.round(live.x * s)}px`;
    el.style.top        = `${Math.round(live.y * s)}px`;
    el.style.width      = `${Math.round(live.w * s)}px`;
    el.style.minHeight  = `${Math.round(live.h * s)}px`;
    el.style.fontSize   = `${Math.round(live.size * s)}px`;
    el.style.color      = live.color || '#1A1A1A';
    el.style.fontFamily = resolveFontStack(live.font_family);
    el.style.lineHeight = '1.25';
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
      _clickCount++;
      if (_clickCount === 1) {
        _selectAllText(el);
        setTimeout(() => { _clickCount = 0; }, 600);
      } else if (_clickCount === 2) {
        const imgEl = document.getElementById('canvas-img');
        if (imgEl && !live.inpainted) {
          try {
            if (!AIRuntime.isCvReady()) await AIRuntime.waitUntilReady();
            await ImageProcessor.inpaintRegion(imgEl, { x: live.x, y: live.y, width: live.w, height: live.h });
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
    return el;
  }

  function repositionAll() {
    const s = AppState.scaleFactor;
    AppState.liveBlocks.forEach(live => {
      if (!live.el) return;
      live.el.style.left      = `${Math.round(live.x * s)}px`;
      live.el.style.top       = `${Math.round(live.y * s)}px`;
      live.el.style.width     = `${Math.round(live.w * s)}px`;
      live.el.style.minHeight = `${Math.round(live.h * s)}px`;
      live.el.style.fontSize  = `${Math.round(live.size * s)}px`;
    });
  }

  function updateBlockStyle(blockId, { font_family, size, color }) {
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (!live || !live.el) return;
    const s = AppState.scaleFactor;
    if (font_family !== undefined) { live.font_family = font_family; live.el.style.fontFamily = resolveFontStack(font_family); }
    if (size !== undefined)        { live.size = size; live.el.style.fontSize = `${Math.round(size * s)}px`; }
    if (color !== undefined) {
      live.color = color; live.el.style.color = color;
      const swatch = document.getElementById(`lyr-swatch-${blockId}`);
      if (swatch) swatch.style.background = color;
    }
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

  return { renderAll, repositionAll, updateBlockStyle, setActive, clearActive };
})();


/* ══════════════════════════════════════════════════════════════════════
   PROPS PANEL
══════════════════════════════════════════════════════════════════════ */
const PropsPanel = (() => {
  function populate(live) {
    _setVal('prop-font',      live.font_family || 'Arial');
    _setVal('prop-size',      live.size || 16);
    _setVal('prop-color',     live.color || '#000000');
    _setVal('prop-color-hex', live.color || '#000000');
    document.getElementById('swatch-fg').style.background = live.color || '#000000';
    document.getElementById('coord-x').textContent = live.x + ' px';
    document.getElementById('coord-y').textContent = live.y + ' px';
    document.getElementById('coord-w').textContent = live.w + ' px';
    document.getElementById('coord-h').textContent = live.h + ' px';
    document.getElementById('info-original').textContent = `"${live.originalText}"`;
    document.getElementById('info-conf').textContent = live.confidence ? (live.confidence * 100).toFixed(1) + '%' : '—';
    const fontAiEl = document.getElementById('info-font-ai');
    if (fontAiEl) fontAiEl.textContent = live.font_family || '—';
    _enableAll(true);
  }
  function clear() {
    _enableAll(false);
    ['coord-x','coord-y','coord-w','coord-h'].forEach(id => { document.getElementById(id).textContent = '—'; });
    document.getElementById('info-original').textContent = '—';
    document.getElementById('info-conf').textContent = '—';
    const fontAiEl = document.getElementById('info-font-ai');
    if (fontAiEl) fontAiEl.textContent = '—';
  }
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
  function _setVal(id, val) { const el = document.getElementById(id); if (el) el.value = val; }
  function _enableAll(enabled) {
    ['prop-font','prop-size','prop-color','prop-color-hex','btn-apply-all'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = !enabled; });
    document.querySelectorAll('#stepper-size .step-btn').forEach(btn => { btn.disabled = !enabled; });
  }
  return { populate, clear, applyFont, applySize, stepSize, applyColor, applyColorHex, applyAll };
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
      meta.className = 'layer-meta'; meta.textContent = live.font_family || `${live.size}px`;
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
  return { rebuild };
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
        const text = live.el ? live.el.textContent : live.currentText;
        if (live.bg_color && live.bg_color !== 'transparent') { ctx.fillStyle = live.bg_color; ctx.fillRect(live.x, live.y, live.w, live.h); }
        ctx.font = `${live.size}px ${resolveFontStack(live.font_family || 'Arial')}`;
        ctx.fillStyle = live.color || '#000000'; ctx.textBaseline = 'top';
        _drawWrappedText(ctx, text, live.x + 2, live.y + 2, live.w - 4, live.size * 1.25);
      });
      _showProgress(90);
      setTimeout(() => {
        try {
          const link = document.createElement('a');
          link.download = `pixelscribe-export-${Date.now()}.png`;
          link.href = canvas.toDataURL('image/png'); link.click();
          _showProgress(100); setTimeout(() => _showProgress(0), 600);
          Toast.show('Exported — check Downloads ✓', 'success');
        } catch (err) {
          Toast.show('Export failed — CORS issue with image URL', 'error');
          _showProgress(0);
        }
      }, 80);
    };
    bgImg.onerror = () => { Toast.show('Could not load background image', 'error'); _showProgress(0); };
    bgImg.src = AppState.bgSrc;
  }
  function _drawWrappedText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split(' '); let line = '';
    for (let i = 0; i < words.length; i++) {
      const testLine = line + words[i] + ' ';
      if (ctx.measureText(testLine).width > maxWidth && i > 0) { ctx.fillText(line.trim(), x, y); line = words[i] + ' '; y += lineHeight; }
      else line = testLine;
    }
    ctx.fillText(line.trim(), x, y);
  }
  return { download };
})();


/* ══════════════════════════════════════════════════════════════════════
   CANVAS VIEW
══════════════════════════════════════════════════════════════════════ */
const CanvasView = (() => {
  function zoom(delta) {
    if (!AppState.isLoaded) return;
    AppState.zoomLevel = Math.min(4, Math.max(0.1, Math.round((AppState.zoomLevel + delta) * 10) / 10));
    _applyZoom();
  }
  function fitToWindow() { if (!AppState.isLoaded) return; AppState.zoomLevel = 1; _applyZoom(); }
  function _applyZoom() {
    const ws = document.getElementById('workspace'); if (!ws) return;
    ws.style.width = AppState.zoomLevel > 1 ? `${Math.round(AppState.imageW * AppState.zoomLevel)}px` : '';
    const label = document.getElementById('zoom-label');
    if (label) label.textContent = Math.round(AppState.zoomLevel * 100) + '%';
    requestAnimationFrame(() => ScaleEngine.recompute());
  }
  return { zoom, fitToWindow };
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
   PDF HANDLER — PDF.js renders page 1 → JPEG data URI
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
  return { src: offCanvas.toDataURL('image/jpeg', 0.92), width: offCanvas.width, height: offCanvas.height };
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

function _finalizeOcrBlocks(blocks, imageW, imageH) {
  const result = (blocks || []).map(b => {
    const w = Math.max(2, Math.round(b.w || 0));
    const h = Math.max(2, Math.round(b.h || 0));
    const x = Math.max(0, Math.min(Math.round(b.x || 0), imageW - w));
    const y = Math.max(0, Math.min(Math.round(b.y || 0), imageH - h));
    return {
      text: (b.text || '').trim(), x, y, w, h,
      color: b.color || '#1A1A1A', bg_color: b.bg_color || 'transparent',
      size: b.size || Math.max(12, Math.round(h * 0.85)),
      confidence: typeof b.confidence === 'number' ? b.confidence : 0.5,
      font_family: b.font_family || 'Arial',
    };
  }).filter(b => b.text.length > 0);

  if (result.length) return result;

  // Absolute fallback: one editable placeholder so the image is still usable
  const fallbackW = Math.max(120, Math.round(imageW * 0.7));
  const fallbackH = Math.max(28,  Math.round(imageH * 0.08));
  return [{
    text: 'Edit text', x: Math.round((imageW - fallbackW) / 2), y: Math.round(imageH * 0.1),
    w: fallbackW, h: fallbackH, color: '#1A1A1A', bg_color: 'transparent',
    size: Math.max(14, Math.round(fallbackH * 0.7)), confidence: 0, font_family: 'Arial',
  }];
}

async function _processImageWithOfflineOcr(src, naturalW, naturalH) {
  _showProgress(20);
  const imgEl  = await _loadImageElement(src);
  const imageW = naturalW || imgEl.naturalWidth || imgEl.width;
  const imageH = naturalH || imgEl.naturalHeight || imgEl.height;

  _setStatusBusy(`Running OCR (${OfflineOCR.engine() || 'loading'})…`);
  const ocr    = await OfflineOCR.recognize(imgEl);
  const blocks = _finalizeOcrBlocks(ocr.blocks, imageW, imageH);
  _showProgress(70);

  return {
    bg_image:   src, image_w: imageW, image_h: imageH, blocks,
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
    font_family: b.font_family || 'Arial',
  }));
  return { bg_image: data.image_b64, image_w: data.image_w, image_h: data.image_h, blocks,
           ocr_text: blocks.map(b => b.text).filter(Boolean).join('\n'), ocr_engine: 'backend' };
}

async function _processPdfFile(file) {
  Toast.show('Rendering PDF page 1…', 'info');
  _showProgress(10);
  const { src, width, height } = await _renderPdfToImage(file);
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

    let enrichedBlocks = payload.blocks;
    try {
      enrichedBlocks = await _enrichBlocksWithFonts(img, payload.blocks);
    } catch (err) {
      console.warn('[PixelScribe] Font enrichment skipped:', err);
      enrichedBlocks = payload.blocks.map(b => ({ ...b, font_family: b.font_family || b.fontFamily || 'Arial' }));
    }

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

    _setStatusOk(`${AppState.imageW} × ${AppState.imageH}px`);
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

async function _enrichBlocksWithFonts(imgEl, blocks) {
  if (!blocks.length) return blocks;
  if (!AIRuntime.isOnnxReady()) {
    return blocks.map(b => ({ ...b, font_family: b.font_family || b.fontFamily || 'Arial' }));
  }
  const enriched = [];
  for (const block of blocks) {
    if (block.font_family && block.font_family !== 'sans-serif') { enriched.push({ ...block }); continue; }
    const predicted = await EdgeML.predictFont(imgEl, block.x, block.y, block.w, block.h);
    enriched.push({ ...block, font_family: predicted });
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
let _currentTab = 'json';
function openModal()  { document.getElementById('modal-overlay').classList.add('open'); }
function closeModal() { document.getElementById('modal-overlay').classList.remove('open'); }
function switchTab(tab) {
  _currentTab = tab;
  ['json','image','url'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`pane-${t}`).classList.toggle('hidden', t !== tab);
  });
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
function handleModalFileInput(evt)  { const f = evt.target.files[0]; if (f) _processFile(f); }
function handleHiddenFileInput(evt) { const f = evt.target.files[0]; if (f) _processFile(f); }

/**
 * _processFile(file)
 * ─────────────────────────────────────────────────────────────────
 * Routes dropped / selected files. Accepts ONLY:
 *   .json  → parse and loadPayload()
 *   .jpg / .jpeg → offline OCR → loadPayload()
 *   .pdf   → PDF.js render page 1 → offline OCR → loadPayload()
 *
 * Any other extension shows a clear error toast.
 * ─────────────────────────────────────────────────────────────────
 */
async function _processFile(file) {
  if (!file) return;

  const name = file.name.toLowerCase();
  const isJson = file.type === 'application/json' || name.endsWith('.json');
  const isJpeg = file.type === 'image/jpeg' || file.type === 'image/jpg'
               || name.endsWith('.jpg') || name.endsWith('.jpeg');
  const isPdf  = file.type === 'application/pdf' || name.endsWith('.pdf');

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

  /* ── JPG / JPEG image ─────────────────────────────────────────── */
  if (isJpeg) {
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
  Toast.show('Unsupported file type. Only JPG and PDF are allowed.', 'error');
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
      window._resizeTimer = setTimeout(() => ScaleEngine.recompute(), 60);
    }
  });
  ro.observe(document.getElementById('canvas-area'));
})();


/* ══════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════ */
(async function init() {
  console.info('[PixelScribe] v3.1 Edge AI — initialising…');
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