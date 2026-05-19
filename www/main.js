/**
 * PixelScribe — main.js  (v3 · Edge AI Edition)
 * ════════════════════════════════════════════════════════════════════════
 * 100% offline editor. All ML inference runs in the browser via WASM.
 * Zero server round-trips. No Python backend required.
 *
 * New in v3
 * ─────────
 *  EdgeML      — ONNX Runtime Web font classifier (models/font_classifier.onnx)
 *  PDF support — PDF.js renders page 1 of an uploaded PDF to a canvas image
 *  cvReady     — OpenCV.js WASM readiness flag + future inpaint hook
 *
 * Modules
 * ───────
 *  EdgeML          — ONNX session loader + predictFont()
 *  AppState        — centralised application data store
 *  EditorState     — undo/redo history stack
 *  ScaleEngine     — coordinate scaling between native px and display px
 *  OverlayEngine   — DOM injection of contenteditable text blocks
 *  PropsPanel      — right sidebar typography controls
 *  LayerPanel      — left sidebar layer list
 *  ExportEngine    — off-screen canvas flatten + file download
 *  CanvasView      — zoom / fit-to-window control
 *  Toast           — lightweight notification system
 *
 * Expected JSON payload shape
 * ───────────────────────────
 * {
 *   "bg_image":  "data:image/jpeg;base64,…"  // or absolute URL
 *   "image_w":   2400,
 *   "image_h":   3000,
 *   "blocks": [
 *     {
 *       "text":        "Hello World",
 *       "x": 120, "y": 450, "w": 800, "h": 60,
 *       "color":       "#1A1A1A",
 *       "bg_color":    "#F5F0E8",   // optional
 *       "size":        18,
 *       "font_family": "Arial",     // optional — falls back to Arial
 *       "confidence":  0.98         // optional
 *     }
 *   ]
 * }
 * ════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ══════════════════════════════════════════════════════════════════════
  ONNX RUNTIME WASM PATHS (must be set before InferenceSession.create)
  Prefer local assets for offline, fallback to CDN when missing.
══════════════════════════════════════════════════════════════════════ */
const ONNX_WASM_LOCAL = './vendor/onnx/';
const ONNX_WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
const PDF_WORKER_LOCAL = './vendor/pdfjs/pdf.worker.min.js';
const PDF_WORKER_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

/** Relative to www/ — Capacitor local server serves from webDir root. */
const FONT_CLASSIFIER_MODEL_URL = './models/font_classifier.onnx';

/** localStorage key for optional Python OCR backend (EasyOCR + inpaint). */
const API_BASE_STORAGE_KEY = 'pixelscribe_api_base';

/** Offline OCR (Tesseract.js or native TextDetector) config. */
const OFFLINE_OCR_LANG = 'eng';
const OFFLINE_OCR_TESSERACT_BASE = './vendor/tesseract';
const OFFLINE_OCR_TESSDATA_BASE = './tessdata';
const OFFLINE_OCR_MAX_DIM = 2000;

/* ══════════════════════════════════════════════════════════════════════
   OPENCV.JS READINESS
   OpenCV.js loads asynchronously via a <script async> tag in index.html.
   The `cvReady` flag is set true in `onOpenCvLoad()` once the WASM
   Module fires `onRuntimeInitialized`.

   Future pipeline integration points (replacing Python backend):
   ──────────────────────────────────────────────────────────────
   When cvReady === true, the following OpenCV.js calls become available:

   ① Grayscale conversion (replaces cv2.cvtColor):
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

   ② Otsu threshold for text/bg mask (replaces cv2.threshold):
      const mask = new cv.Mat();
      cv.threshold(gray, mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

   ③ Mask dilation (replaces cv2.dilate):
      const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(17, 17));
      cv.dilate(mask, dilated, kernel);

   ④ Inpainting (replaces cv2.inpaint / Telea algorithm):
      cv.inpaint(src, dilated, dst, 12, cv.INPAINT_TELEA);

   These four calls together replicate the Python text-removal pipeline
   entirely in-browser. Implementation target: a future `CVPipeline`
   module that accepts an ImageBitmap + block list and returns a cleaned
   data URI — the result then feeds directly into loadImageOnly().
══════════════════════════════════════════════════════════════════════ */

/** True once OpenCV.js WASM runtime has fully initialised. */
let cvReady = false;

/* ══════════════════════════════════════════════════════════════════════
   AI RUNTIME — gate processing until OpenCV + ONNX are both ready
══════════════════════════════════════════════════════════════════════ */
const AIRuntime = (() => {
  let _onnxReady = false;
  let _onnxInitDone = false;
  let _onnxInitError = null;
  let _cvReadyResolve = null;

  const _cvReadyPromise = new Promise((resolve) => {
    _cvReadyResolve = resolve;
  });

  let _onnxInitPromise = null;

  function setOnnxInitPromise(p) {
    _onnxInitPromise = p;
  }

  function markCvReady() {
    if (_cvReadyResolve) {
      _cvReadyResolve();
      _cvReadyResolve = null;
    }
    _refreshUiLock();
  }

  function markOnnxReady() {
    _onnxReady = true;
    _onnxInitDone = true;
    _onnxInitError = null;
    _refreshUiLock();
  }

  function markOnnxFailed(err) {
    _onnxReady = false;
    _onnxInitDone = true;
    _onnxInitError = err;
    _refreshUiLock();
  }

  function isCvReady() {
    return cvReady;
  }

  function isOnnxReady() {
    return _onnxReady;
  }

  function isReady() {
    return cvReady && _onnxReady;
  }

  async function waitUntilReady(timeoutMs = 120000) {
    const waits = [_cvReadyPromise];
    if (_onnxInitPromise) waits.push(_onnxInitPromise);

    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => {
        reject(new Error('AI libraries timed out loading. Check network/CDN access and try again.'));
      }, timeoutMs);
    });

    try {
      await Promise.race([
        Promise.all(waits).then(() => {
          if (!cvReady) throw new Error('OpenCV.js is not ready yet.');
          if (!_onnxInitDone) throw new Error('ONNX model is still loading.');
          if (!_onnxReady) {
            const detail = _onnxInitError && _onnxInitError.message
              ? _onnxInitError.message
              : 'ONNX session failed to load.';
            throw new Error(detail);
          }
        }),
        timeout,
      ]);
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    setOnnxInitPromise,
    markCvReady,
    markOnnxReady,
    markOnnxFailed,
    isCvReady,
    isOnnxReady,
    isReady,
    waitUntilReady,
  };
})();

/**
 * Mobile-visible AI error (physical device has no easy console).
 * @param {string} context
 * @param {unknown} err
 */
function _aiAlert(context, err) {
  const msg = (err && err.message) ? err.message : String(err);
  console.error(`[PixelScribe AI · ${context}]`, err);
  alert('AI Error: ' + msg);
}

function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load script: ' + src));
    document.head.appendChild(script);
  });
}

async function _resolveLocalOrCdnBase(localBase, cdnBase, probeFile) {
  try {
    const res = await fetch(localBase + probeFile, { method: 'HEAD' });
    if (res.ok) return localBase;
  } catch (err) {
    // Ignore and fall back to CDN.
  }
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

/** Import only — never disable the canvas/overlay (that blocks all editing). */
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

/** After an image + blocks load, ensure nothing invisible blocks taps on text fields. */
function _ensureEditorInteractive() {
  const canvasArea = document.getElementById('canvas-area');
  const overlay = document.getElementById('overlay');
  const welcome = document.getElementById('welcome-drop');

  if (canvasArea) canvasArea.style.pointerEvents = 'auto';
  if (overlay) overlay.style.pointerEvents = 'auto';
  if (welcome) {
    welcome.style.display = 'none';
    welcome.style.pointerEvents = 'none';
  }
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

/**
 * Called by the `onload` attribute of the <script async src="opencv.js"> tag.
 * OpenCV.js sets cv.onRuntimeInitialized internally; we hook into it here.
 */
function onOpenCvLoad() {
  if (typeof cv !== 'undefined') {
    if (cv.getBuildInformation) {
      _markCvReady();
    } else {
      cv.onRuntimeInitialized = _markCvReady;
    }
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
   EDGE ML  — ONNX Runtime Web font classifier
   Model:  ./models/font_classifier.onnx
   Input:  Float32 [1, 1, 64, 64]  grayscale, normalised 0–1
   Output: Float32 [1, 10]          raw logits → argmax → FONT_LABELS[i]
══════════════════════════════════════════════════════════════════════ */
const EdgeML = (() => {

  /**
   * Font label order MUST match the Python training script
   * (backend/train_font_classifier.py → FONT_LABELS list, indices 0-9).
   */
  const FONT_LABELS = [
    'Arial',            // 0
    'Times New Roman',  // 1
    'Courier New',      // 2
    'Calibri',          // 3
    'Georgia',          // 4
    'Verdana',          // 5
    'Roboto',           // 6
    'Helvetica',        // 7
    'Garamond',         // 8
    'Consolas',         // 9
  ];

  /** ONNX InferenceSession — set after init() resolves */
  let _session = null;

  /** Off-screen 64×64 canvas used for crop preprocessing */
  const _cropCanvas = document.createElement('canvas');
  _cropCanvas.width  = 64;
  _cropCanvas.height = 64;
  const _cropCtx = _cropCanvas.getContext('2d', { willReadFrequently: true });

  /**
   * init()
   * ──────
   * Loads font_classifier.onnx using ONNX Runtime Web's WASM backend.
   * Call once at page load; safe to call multiple times (no-ops after first).
   *
   * @returns {Promise<void>}
   */
  async function init() {
    if (_session) {
      AIRuntime.markOnnxReady();
      return;
    }

    if (typeof ort === 'undefined') {
      const err = new Error('onnxruntime-web is not loaded (check CDN script in index.html).');
      console.warn('[EdgeML]', err.message);
      _setBadge('badge-onnx', 'error', 'ONNX unavailable');
      AIRuntime.markOnnxFailed(err);
      return;
    }

    try {
      await _setOnnxWasmPaths();

      _setBadge('badge-onnx', 'loading', 'ONNX loading…');

      _session = await ort.InferenceSession.create(
        FONT_CLASSIFIER_MODEL_URL,
        { executionProviders: ['wasm'] }
      );

      console.info('[EdgeML] ONNX session ready. Input:', _session.inputNames, 'Output:', _session.outputNames);
      _setBadge('badge-onnx', 'ready', 'ONNX ready');
      AIRuntime.markOnnxReady();
    } catch (err) {
      console.warn('[EdgeML] Could not load ONNX model:', err.message);
      _setBadge('badge-onnx', 'error', 'ONNX error');
      _session = null;
      AIRuntime.markOnnxFailed(err);
    }
  }

  /**
   * predictFont(imageEl, x, y, width, height)
   * ──────────────────────────────────────────
   * Crops the bounding box from the provided image element, converts
   * it to a normalised 64×64 grayscale Float32Array tensor, runs
   * inference, and returns the predicted font family name.
   *
   * Falls back to 'Arial' gracefully if the model is unavailable or
   * if the crop is degenerate (zero area, out-of-bounds, etc.).
   *
   * @param {HTMLImageElement|HTMLCanvasElement} imageEl  — source image
   * @param {number} x       — bounding box left   (native image px)
   * @param {number} y       — bounding box top    (native image px)
   * @param {number} width   — bounding box width  (native image px)
   * @param {number} height  — bounding box height (native image px)
   * @returns {Promise<string>}  Font family name from FONT_LABELS
   */
  async function predictFont(imageEl, x, y, width, height) {
    if (!_session) return 'Arial';
    if (!width || !height || width <= 0 || height <= 0) return 'Arial';

    try {
      // ── a. Draw the bounding-box crop scaled to 64×64 ──────────────
      _cropCtx.clearRect(0, 0, 64, 64);
      _cropCtx.drawImage(imageEl, x, y, width, height, 0, 0, 64, 64);

      // ── b. Extract pixel data and convert to grayscale Float32 ─────
      const imageData = _cropCtx.getImageData(0, 0, 64, 64);
      const pixels    = imageData.data;    // RGBA, length = 64*64*4

      const grayFloat = new Float32Array(64 * 64);
      for (let i = 0; i < 64 * 64; i++) {
        const r = pixels[i * 4];
        const g = pixels[i * 4 + 1];
        const b = pixels[i * 4 + 2];
        // BT.601 luminance, normalised to [0, 1]
        grayFloat[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
      }

      // ── c. Build ORT tensor  shape [1, 1, 64, 64]  (NCHW) ──────────
      const tensor = new ort.Tensor('float32', grayFloat, [1, 1, 64, 64]);

      // ── d. Run inference ───────────────────────────────────────────
      // The model's input name may vary; use the first input key dynamically.
      const inputName  = _session.inputNames[0];
      const outputName = _session.outputNames[0];
      const feeds      = { [inputName]: tensor };

      const results  = await _session.run(feeds);
      const logits   = results[outputName].data;  // Float32Array [10]

      // ── e. Argmax → FONT_LABELS ────────────────────────────────────
      let maxIdx = 0;
      let maxVal = logits[0];
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
      }

      const predicted = (maxIdx >= 0 && maxIdx < FONT_LABELS.length)
        ? FONT_LABELS[maxIdx]
        : 'Arial';

      return predicted;

    } catch (err) {
      console.warn('[EdgeML] predictFont error:', err.message);
      return 'Arial';
    }
  }

  return { init, predictFont, FONT_LABELS };
})();


/* ══════════════════════════════════════════════════════════════════════
   FONT FALLBACK MAP — CSS font-family stacks for the 10-class classifier
══════════════════════════════════════════════════════════════════════ */
const FONT_FALLBACK_MAP = {
  'Arial':            '"Arial", "Helvetica Neue", Helvetica, sans-serif',
  'Times New Roman':  '"Times New Roman", Times, "Noto Serif", serif',
  'Courier New':      '"Courier New", Courier, "Roboto Mono", monospace',
  'Calibri':          '"Calibri", "Segoe UI", Candara, sans-serif',
  'Georgia':          '"Georgia", Cambria, "Times New Roman", serif',
  'Verdana':          '"Verdana", Geneva, Tahoma, sans-serif',
  'Roboto':           '"Roboto", "Helvetica Neue", Arial, sans-serif',
  'Helvetica':        '"Helvetica Neue", Helvetica, Arial, sans-serif',
  'Garamond':         '"EB Garamond", Garamond, "Times New Roman", serif',
  'Consolas':         '"Consolas", "Roboto Mono", "Courier New", monospace',
};

function resolveFontStack(fontFamily) {
  return FONT_FALLBACK_MAP[fontFamily] || `"${fontFamily}", sans-serif`;
}


/* ══════════════════════════════════════════════════════════════════════
   IMAGE PROCESSOR — OpenCV.js Inpainting
══════════════════════════════════════════════════════════════════════ */
const ImageProcessor = (() => {
  async function inpaintRegion(imageElement, bbox) {
    if (!cvReady || typeof cv === 'undefined') {
      throw new Error('OpenCV.js is not loaded yet.');
    }

    let src;
    let mask;
    let dst;

    try {
      src = cv.imread(imageElement);
      mask = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(0));

      const x1 = Math.max(0, bbox.x - 4);
      const y1 = Math.max(0, bbox.y - 4);
      const x2 = Math.min(src.cols, bbox.x + bbox.width + 4);
      const y2 = Math.min(src.rows, bbox.y + bbox.height + 4);

      cv.rectangle(mask, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(255), -1, cv.LINE_8, 0);

      dst = new cv.Mat();
      cv.inpaint(src, mask, dst, 3, cv.INPAINT_TELEA);

      const hiddenCanvas = document.createElement('canvas');
      cv.imshow(hiddenCanvas, dst);
      AppState.bgSrc = hiddenCanvas.toDataURL('image/png');
      imageElement.src = AppState.bgSrc;
    } catch (err) {
      _aiAlert('OpenCV inpaint', err);
      throw err;
    } finally {
      if (src) src.delete();
      if (mask) mask.delete();
      if (dst) dst.delete();
    }
  }

  return { inpaintRegion };
})();


/* ══════════════════════════════════════════════════════════════════════
   OFFLINE OCR — TextDetector (native) → Tesseract.js fallback
══════════════════════════════════════════════════════════════════════ */
const OfflineOCR = (() => {
  let _engine = 'none';
  let _initPromise = null;
  let _worker = null;
  let _lastError = null;

  async function init() {
    if (_initPromise) return _initPromise;
    _initPromise = _initInternal();
    return _initPromise;
  }

  function engine() {
    return _engine;
  }

  function isReady() {
    return _engine !== 'none';
  }

  function lastError() {
    return _lastError;
  }

  async function _initInternal() {
    _lastError = null;

    if (typeof TextDetector !== 'undefined') {
      _engine = 'text-detector';
      return;
    }

    try {
      if (typeof Tesseract === 'undefined') {
        await _loadScript(`${OFFLINE_OCR_TESSERACT_BASE}/tesseract.min.js`);
      }
      if (typeof Tesseract === 'undefined') {
        throw new Error('Tesseract.js not found. Add local assets under /vendor/tesseract.');
      }

      _worker = await Tesseract.createWorker({
        logger: (m) => {
          if (m && m.status === 'recognizing text') {
            _showProgress(Math.min(60, Math.max(10, Math.round(m.progress * 60))));
          }
        },
        workerPath: `${OFFLINE_OCR_TESSERACT_BASE}/worker.min.js`,
        corePath: `${OFFLINE_OCR_TESSERACT_BASE}/tesseract-core.wasm.js`,
        langPath: OFFLINE_OCR_TESSDATA_BASE,
      });
      await _worker.loadLanguage(OFFLINE_OCR_LANG);
      await _worker.initialize(OFFLINE_OCR_LANG);

      _engine = 'tesseract';
    } catch (err) {
      _engine = 'none';
      _lastError = err;
      throw err;
    }
  }

  async function recognize(imgEl) {
    if (!_initPromise) await init();
    if (_engine === 'text-detector') return _recognizeWithTextDetector(imgEl);
    if (_engine === 'tesseract') return _recognizeWithTesseract(imgEl);
    throw (_lastError || new Error('Offline OCR engine not available.'));
  }

  function _rasterizeForOcr(imgEl) {
    const w = imgEl.naturalWidth || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;
    const scale = Math.min(1, OFFLINE_OCR_MAX_DIM / Math.max(w, h));
    if (scale >= 1) return { source: imgEl, scale, width: w, height: h };

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
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
    const results = await detector.detect(source);

    const blocks = results.map((r) => {
      const box = r.boundingBox || { x: 0, y: 0, width: 0, height: 0 };
      const x = Math.round(box.x / scale);
      const y = Math.round(box.y / scale);
      const w = Math.round(box.width / scale);
      const h = Math.round(box.height / scale);
      const norm = _normalizeBox(x, y, w, h, imgEl.naturalWidth, imgEl.naturalHeight);
      return {
        text: (r.rawValue || r.text || '').trim(),
        x: norm.x,
        y: norm.y,
        w: norm.w,
        h: norm.h,
        confidence: 0.6,
      };
    }).filter(b => b.text.length > 0);

    const plainText = results
      .map((r) => (r.rawValue || r.text || '').trim())
      .filter(Boolean)
      .join('\n');

    return { blocks, plainText, engine: 'text-detector' };
  }

  async function _recognizeWithTesseract(imgEl) {
    if (!_worker) throw new Error('Tesseract worker not initialised.');
    const { source, scale } = _rasterizeForOcr(imgEl);
    const { data } = await _worker.recognize(source);

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
        x: norm.x,
        y: norm.y,
        w: norm.w,
        h: norm.h,
        confidence: typeof line.confidence === 'number' ? Math.max(0, Math.min(1, line.confidence / 100)) : 0.6,
      };
    }).filter(b => b.text.length > 0);

    const plainText = (data && data.text ? data.text.trim() : '');
    return { blocks, plainText, engine: 'tesseract' };
  }

  return { init, recognize, isReady, engine, lastError };
})();


/* ══════════════════════════════════════════════════════════════════════
   APP STATE — single source of truth
══════════════════════════════════════════════════════════════════════ */
const AppState = {
  payload:      null,
  imageW:       0,
  imageH:       0,
  blocks:       [],
  bgSrc:        '',
  liveBlocks:   [],
  activeId:     null,
  scaleFactor:  1,
  zoomLevel:    1,
  isLoaded:     false,
  ocrText:      '',
  ocrEngine:    'none',

  clear() {
    this.payload     = null;
    this.imageW      = 0;
    this.imageH      = 0;
    this.blocks      = [];
    this.liveBlocks  = [];
    this.bgSrc       = '';
    this.activeId    = null;
    this.scaleFactor = 1;
    this.zoomLevel   = 1;
    this.isLoaded    = false;
    this.ocrText      = '';
    this.ocrEngine    = 'none';
  }
};


/* ══════════════════════════════════════════════════════════════════════
   UNDO / REDO HISTORY
══════════════════════════════════════════════════════════════════════ */
const EditorState = (() => {
  const undoStack = [];
  const redoStack = [];
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
    const entry = undoStack.pop();
    redoStack.push(entry);
    _applyText(entry.blockId, entry.oldText);
    _syncButtons();
    Toast.show('Undo', 'info');
  }

  function redo() {
    if (!redoStack.length) return;
    const entry = redoStack.pop();
    undoStack.push(entry);
    _applyText(entry.blockId, entry.newText);
    _syncButtons();
    Toast.show('Redo', 'info');
  }

  function clear() {
    undoStack.length = 0;
    redoStack.length = 0;
    _syncButtons();
  }

  function _applyText(blockId, text) {
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (live && live.el) {
      live.el.textContent = text;
      live.currentText    = text;
    }
  }

  function _syncButtons() {
    const u = document.getElementById('btn-undo');
    const r = document.getElementById('btn-redo');
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
    const rendered = img.getBoundingClientRect();
    const sf = rendered.width / AppState.imageW;
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
      const live = {
        id,
        ...block,
        font_family:  block.font_family || block.fontFamily || 'Arial',
        currentText:  block.text,
        originalText: block.text,
        el:           null,
      };
      const el = _createField(live);
      live.el  = el;
      AppState.liveBlocks.push(live);
      overlay.appendChild(el);
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

    el.style.left      = `${Math.round(live.x * s)}px`;
    el.style.top       = `${Math.round(live.y * s)}px`;
    el.style.width     = `${Math.round(live.w * s)}px`;
    el.style.minHeight = `${Math.round(live.h * s)}px`;
    el.style.fontSize  = `${Math.round(live.size * s)}px`;
    el.style.color     = live.color || '#1A1A1A';
    el.style.fontFamily = resolveFontStack(live.font_family);
    el.style.lineHeight = '1.25';

    if (live.bg_color) {
      el.style.backgroundColor = _hexWithAlpha(live.bg_color, 0.0);
    }

    let _textOnFocus = '';
    el.addEventListener('focus', () => {
      _textOnFocus = el.textContent;
      _selectAllText(el);
      _setActive(live.id);
    });

    el.addEventListener('blur', () => {
      const newText = el.textContent;
      EditorState.push(live.id, _textOnFocus, newText);
      live.currentText = newText;
      const layerTextEl = document.getElementById(`lyr-text-${live.id}`);
      if (layerTextEl) layerTextEl.textContent = `"${newText}"`;
      _setOcrTextFromBlocks(AppState.liveBlocks.map((b) => ({ text: b.currentText })));
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
            if (!AIRuntime.isCvReady()) {
              await AIRuntime.waitUntilReady();
            }
            await ImageProcessor.inpaintRegion(imgEl, {
              x: live.x, y: live.y, width: live.w, height: live.h
            });
            live.inpainted = true;
            Toast.show('Original text erased', 'success');
          } catch (err) {
            Toast.show('Inpaint failed: ' + err.message, 'error');
          }
        }
      }
    });

    el.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { el.blur(); e.preventDefault(); }
    });

    // Capacitor / Android WebView: touch often does not focus contenteditable via click alone
    el.addEventListener('touchstart', (e) => {
      e.stopPropagation();
      _setActive(live.id);
    }, { passive: true });

    el.addEventListener('touchend', (e) => {
      e.stopPropagation();
      if (document.activeElement !== el) {
        el.focus();
        _selectAllText(el);
      }
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
    if (font_family !== undefined) {
      live.font_family         = font_family;
      live.el.style.fontFamily = resolveFontStack(font_family);
    }
    if (size !== undefined) {
      live.size              = size;
      live.el.style.fontSize = `${Math.round(size * s)}px`;
    }
    if (color !== undefined) {
      live.color           = color;
      live.el.style.color  = color;
      const swatch = document.getElementById(`lyr-swatch-${blockId}`);
      if (swatch) swatch.style.background = color;
    }
  }

  function _setActive(blockId) {
    AppState.activeId = blockId;
    document.querySelectorAll('.txt-block').forEach(el => {
      el.classList.toggle('selected', el.dataset.blockId === blockId);
    });
    document.querySelectorAll('.layer-item').forEach(el => {
      el.classList.toggle('active', el.dataset.blockId === blockId);
    });
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (live) PropsPanel.populate(live);
  }

  function _selectAllText(el) {
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function _hexWithAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function setActive(blockId) { _setActive(blockId); }

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
    document.getElementById('info-conf').textContent =
      live.confidence ? (live.confidence * 100).toFixed(1) + '%' : '—';

    // Show EdgeML font prediction in props panel
    const fontAiEl = document.getElementById('info-font-ai');
    if (fontAiEl) fontAiEl.textContent = live.font_family || '—';

    _enableAll(true);
  }

  function clear() {
    _enableAll(false);
    ['coord-x','coord-y','coord-w','coord-h'].forEach(id => {
      document.getElementById(id).textContent = '—';
    });
    document.getElementById('info-original').textContent = '—';
    document.getElementById('info-conf').textContent = '—';
    const fontAiEl = document.getElementById('info-font-ai');
    if (fontAiEl) fontAiEl.textContent = '—';
  }

  function applyFont() {
    if (!AppState.activeId) return;
    const val = document.getElementById('prop-font').value;
    OverlayEngine.updateBlockStyle(AppState.activeId, { font_family: val });
  }

  function applySize() {
    if (!AppState.activeId) return;
    const val = parseInt(document.getElementById('prop-size').value, 10);
    if (!isNaN(val) && val > 0) OverlayEngine.updateBlockStyle(AppState.activeId, { size: val });
  }

  function stepSize(delta) {
    const input = document.getElementById('prop-size');
    const val   = parseInt(input.value, 10) + delta;
    if (val >= 1 && val <= 400) { input.value = val; applySize(); }
  }

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

  function applyAll() {
    if (!AppState.activeId) return;
    applyFont(); applySize(); applyColor();
    Toast.show('Typography applied', 'success');
  }

  function _setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  }

  function _enableAll(enabled) {
    ['prop-font','prop-size','prop-color','prop-color-hex','btn-apply-all'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = !enabled;
    });
    document.querySelectorAll('#stepper-size .step-btn').forEach(btn => {
      btn.disabled = !enabled;
    });
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

    if (!AppState.liveBlocks.length) {
      list.appendChild(empty);
      count.textContent = '0';
      return;
    }

    count.textContent = AppState.liveBlocks.length;

    AppState.liveBlocks.forEach((live, i) => {
      const item = document.createElement('div');
      item.className       = 'layer-item';
      item.dataset.blockId = live.id;
      item.style.animationDelay = `${i * 18}ms`;

      const swatch = document.createElement('div');
      swatch.className = 'layer-swatch';
      swatch.id        = `lyr-swatch-${live.id}`;
      swatch.style.background = live.color || '#888';

      const text = document.createElement('div');
      text.className   = 'layer-text';
      text.id          = `lyr-text-${live.id}`;
      text.textContent = `"${live.currentText}"`;

      const meta = document.createElement('div');
      meta.className   = 'layer-meta';
      meta.textContent = live.font_family || `${live.size}px`;

      item.append(swatch, text, meta);
      const focusBlock = () => {
        const field = document.getElementById(`field-${live.id}`);
        if (field) {
          field.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          field.focus();
        }
        OverlayEngine.setActive(live.id);
        if (typeof closeMobilePanels === 'function') closeMobilePanels();
      };
      item.addEventListener('click', focusBlock);
      item.addEventListener('touchend', (e) => {
        e.preventDefault();
        focusBlock();
      });

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
    canvas.width  = AppState.imageW;
    canvas.height = AppState.imageH;
    const ctx = canvas.getContext('2d');

    const bgImg = new Image();
    bgImg.crossOrigin = 'anonymous';

    bgImg.onload = () => {
      ctx.drawImage(bgImg, 0, 0, AppState.imageW, AppState.imageH);
      _showProgress(50);

      AppState.liveBlocks.forEach(live => {
        const text  = live.el ? live.el.textContent : live.currentText;
        const font  = live.font_family || 'Arial';
        const color = live.color || '#000000';

        if (live.bg_color && live.bg_color !== 'transparent') {
          ctx.fillStyle = live.bg_color;
          ctx.fillRect(live.x, live.y, live.w, live.h);
        }

        ctx.font         = `${live.size}px ${resolveFontStack(font)}`;
        ctx.fillStyle    = color;
        ctx.textBaseline = 'top';
        _drawWrappedText(ctx, text, live.x + 2, live.y + 2, live.w - 4, live.size * 1.25);
      });

      _showProgress(90);

      setTimeout(() => {
        try {
          const dataUrl = canvas.toDataURL('image/png');
          const link    = document.createElement('a');
          link.download = `pixelscribe-export-${Date.now()}.png`;
          link.href     = dataUrl;
          link.click();
          _showProgress(100);
          setTimeout(() => _showProgress(0), 600);
          Toast.show('Exported — check Downloads ✓', 'success');
        } catch (err) {
          console.error('[ExportEngine] toDataURL failed:', err);
          Toast.show('Export failed — CORS issue with image URL', 'error');
          _showProgress(0);
        }
      }, 80);
    };

    bgImg.onerror = () => {
      Toast.show('Could not load background image for export', 'error');
      _showProgress(0);
    };

    bgImg.src = AppState.bgSrc;
  }

  function _drawWrappedText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split(' ');
    let line = '';
    for (let i = 0; i < words.length; i++) {
      const testLine = line + words[i] + ' ';
      if (ctx.measureText(testLine).width > maxWidth && i > 0) {
        ctx.fillText(line.trim(), x, y);
        line = words[i] + ' ';
        y   += lineHeight;
      } else {
        line = testLine;
      }
    }
    ctx.fillText(line.trim(), x, y);
  }

  return { download };
})();


/* ══════════════════════════════════════════════════════════════════════
   CANVAS VIEW — zoom and fit
══════════════════════════════════════════════════════════════════════ */
const CanvasView = (() => {
  function zoom(delta) {
    if (!AppState.isLoaded) return;
    AppState.zoomLevel = Math.min(4, Math.max(0.1,
      Math.round((AppState.zoomLevel + delta) * 10) / 10
    ));
    _applyZoom();
  }

  function fitToWindow() {
    if (!AppState.isLoaded) return;
    AppState.zoomLevel = 1;
    _applyZoom();
  }

  function _applyZoom() {
    const ws = document.getElementById('workspace');
    if (!ws) return;
    ws.style.width = AppState.zoomLevel > 1
      ? `${Math.round(AppState.imageW * AppState.zoomLevel)}px`
      : '';
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
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-icon">${ICONS[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('show')));
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  return { show };
})();


/* ══════════════════════════════════════════════════════════════════════
   PDF HANDLER — renders page 1 of a PDF to a JPEG data URI via PDF.js
   Replaces the old "upload to server, run Python pipeline" flow.
   Everything runs in-browser; the file never leaves the device.
══════════════════════════════════════════════════════════════════════ */

/**
 * _renderPdfToImage(file)
 * ──────────────────────
 * Reads a PDF File object, renders page 1 at 3× scale (≈300 DPI),
 * and returns a JPEG data URI string.
 *
 * @param {File} file
 * @returns {Promise<{src: string, width: number, height: number}>}
 */
async function _renderPdfToImage(file) {
  if (typeof pdfjsLib === 'undefined') {
    throw new Error('PDF.js is not loaded. Add local PDF.js assets or enable CDN access.');
  }

  // Ensure worker path is configured (prefer local if available)
  await _ensurePdfWorkerSrc();

  // Read the file as an ArrayBuffer
  const arrayBuffer = await file.arrayBuffer();

  // Load the PDF document from the raw binary data
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
  const pdfDoc = await loadingTask.promise;

  // Grab page 1
  const page = await pdfDoc.getPage(1);

  // Render at 3× scale for high-fidelity output (≈300 DPI from a 96 DPI screen PDF)
  const SCALE = 3.0;
  const viewport = page.getViewport({ scale: SCALE });

  // Create an off-screen canvas matching the scaled viewport dimensions
  const offCanvas     = document.createElement('canvas');
  offCanvas.width     = Math.round(viewport.width);
  offCanvas.height    = Math.round(viewport.height);
  const offCtx        = offCanvas.getContext('2d');

  // Render the PDF page onto the canvas
  const renderContext = { canvasContext: offCtx, viewport };
  await page.render(renderContext).promise;

  // Convert to JPEG data URI (quality 0.92 — good balance of size vs fidelity)
  const src = offCanvas.toDataURL('image/jpeg', 0.92);

  return { src, width: offCanvas.width, height: offCanvas.height };
}


/* ══════════════════════════════════════════════════════════════════════
  OPTIONAL PYTHON BACKEND  (OCR + inpaint → editable blocks)
  Used as a fallback when offline OCR is unavailable or insufficient.
══════════════════════════════════════════════════════════════════════ */

function getApiBaseUrl() {
  return (localStorage.getItem(API_BASE_STORAGE_KEY) || '').trim().replace(/\/$/, '');
}

function setApiBaseUrl(url) {
  const trimmed = (url || '').trim().replace(/\/$/, '');
  if (trimmed) localStorage.setItem(API_BASE_STORAGE_KEY, trimmed);
  else localStorage.removeItem(API_BASE_STORAGE_KEY);
}

function configureOcrServer() {
  const entered = window.prompt(
    'OCR server URL (optional fallback):\n\n' +
    '• Phone on same Wi-Fi: http://YOUR_PC_IP:8000\n' +
    '• Android emulator: http://10.0.2.2:8000\n' +
    '• Browser on this PC: http://localhost:8000\n\n' +
    'Leave empty to keep everything offline.',
    getApiBaseUrl() || 'http://192.168.1.100:8000'
  );
  if (entered === null) return;
  setApiBaseUrl(entered);
  if (getApiBaseUrl()) {
    Toast.show('OCR server set: ' + getApiBaseUrl(), 'success', 4000);
  } else {
    Toast.show('OCR server cleared — offline mode', 'info');
  }
  if (typeof closeMobileMore === 'function') closeMobileMore();
}

async function _dataUriToBlob(dataUri) {
  const res = await fetch(dataUri);
  return res.blob();
}

function _readFileAsDataUri(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = () => reject(new Error('FileReader failed.'));
    reader.readAsDataURL(file);
  });
}

function _loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image failed to load for OCR.'));
    img.src = src;
  });
}

function _finalizeOcrBlocks(blocks, imageW, imageH) {
  const result = (blocks || []).map((b) => {
    const w = Math.max(2, Math.round(b.w || 0));
    const h = Math.max(2, Math.round(b.h || 0));
    const x = Math.max(0, Math.min(Math.round(b.x || 0), imageW - w));
    const y = Math.max(0, Math.min(Math.round(b.y || 0), imageH - h));
    return {
      text: (b.text || '').trim(),
      x,
      y,
      w,
      h,
      color: b.color || '#1A1A1A',
      bg_color: b.bg_color || 'transparent',
      size: b.size || Math.max(12, Math.round(h * 0.85)),
      confidence: typeof b.confidence === 'number' ? b.confidence : 0.5,
      font_family: b.font_family || 'Arial',
    };
  }).filter(b => b.text.length > 0);

  if (result.length) return result;

  const fallbackW = Math.max(120, Math.round(imageW * 0.7));
  const fallbackH = Math.max(28, Math.round(imageH * 0.08));
  return [{
    text: 'Edit text',
    x: Math.round((imageW - fallbackW) / 2),
    y: Math.round(imageH * 0.1),
    w: fallbackW,
    h: fallbackH,
    color: '#1A1A1A',
    bg_color: 'transparent',
    size: Math.max(14, Math.round(fallbackH * 0.7)),
    confidence: 0,
    font_family: 'Arial',
  }];
}

async function _processImageWithOfflineOcr(src, naturalW, naturalH) {
  _showProgress(20);
  const imgEl = await _loadImageElement(src);
  const ocr = await OfflineOCR.recognize(imgEl);

  const imageW = naturalW || imgEl.naturalWidth || imgEl.width;
  const imageH = naturalH || imgEl.naturalHeight || imgEl.height;

  const blocks = _finalizeOcrBlocks(ocr.blocks, imageW, imageH);
  _showProgress(70);

  return {
    bg_image: src,
    image_w: imageW,
    image_h: imageH,
    blocks,
    ocr_text: ocr.plainText || blocks.map(b => b.text).join('\n'),
    ocr_engine: ocr.engine,
  };
}

async function _buildFallbackPayload(src, naturalW, naturalH) {
  const imgEl = await _loadImageElement(src);
  const imageW = naturalW || imgEl.naturalWidth || imgEl.width;
  const imageH = naturalH || imgEl.naturalHeight || imgEl.height;
  const blocks = _finalizeOcrBlocks([], imageW, imageH);

  return {
    bg_image: src,
    image_w: imageW,
    image_h: imageH,
    blocks,
    ocr_text: blocks.map((b) => b.text).filter(Boolean).join('\n'),
    ocr_engine: 'fallback',
  };
}

function _setOcrText(text) {
  const clean = (text || '').trim();
  AppState.ocrText = clean;
  const el = document.getElementById('ocr-text-output');
  if (el) el.value = clean;
}

function _setOcrTextFromBlocks(blocks) {
  const text = (blocks || [])
    .map((b) => (b && b.text ? String(b.text).trim() : ''))
    .filter(Boolean)
    .join('\n');
  _setOcrText(text);
}

async function copyOcrText() {
  const text = AppState.ocrText || '';
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    Toast.show('OCR text copied', 'success');
  } catch (err) {
    const textarea = document.getElementById('ocr-text-output');
    if (textarea) {
      textarea.focus();
      textarea.select();
      document.execCommand('copy');
      Toast.show('OCR text copied', 'success');
    }
  }
}

/**
 * POST image bytes to backend /process-image → editor payload with blocks.
 * @param {Blob} blob
 * @param {string} filename
 * @returns {Promise<object>}
 */
async function _processImageWithBackend(blob, filename) {
  const base = getApiBaseUrl();
  if (!base) {
    throw new Error('No OCR server configured. Use More → OCR Server to set your PC URL.');
  }

  const form = new FormData();
  form.append('file', blob, filename || 'upload.jpg');
  form.append('languages', 'en');
  form.append('confidence', '0.25');

  let res;
  try {
    res = await fetch(`${base}/process-image`, { method: 'POST', body: form });
  } catch (err) {
    throw new Error(
      'Cannot reach OCR server at ' + base + '. Same Wi‑Fi? Server running? ' + err.message
    );
  }

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Server error ${res.status}: ${detail.slice(0, 200)}`);
  }

  const data = await res.json();
  const blocks = (data.blocks || []).map((b) => ({
    text: b.text,
    x: b.x,
    y: b.y,
    w: b.w,
    h: b.h,
    color: b.color || '#000000',
    bg_color: b.bg_color,
    size: b.size || Math.max(12, b.h || 16),
    confidence: b.confidence,
    font_family: b.font_family || 'Arial',
  }));

  return {
    bg_image: data.image_b64,
    image_w: data.image_w,
    image_h: data.image_h,
    blocks,
    ocr_text: blocks.map((b) => b.text).filter(Boolean).join('\n'),
    ocr_engine: 'backend',
  };
}

/** PDF page 1 → JPEG → backend OCR (when server URL is set). */
async function _processPdfFile(file) {
  Toast.show('Rendering PDF page 1…', 'info');
  _showProgress(10);

  const { src, width, height } = await _renderPdfToImage(file);
  _showProgress(35);
  try {
    const payload = await _processImageWithOfflineOcr(src, width, height);
    loadPayload(payload);
    closeModal();
    Toast.show(
      `PDF ready — ${payload.blocks.length} editable text block(s)`,
      'success',
      4000
    );
  } catch (err) {
    console.warn('[PDF] Offline OCR failed:', err);
    const apiBase = getApiBaseUrl();
    if (apiBase) {
      Toast.show('Sending page to OCR server…', 'info');
      _showProgress(55);
      const blob = await _dataUriToBlob(src);
      const payload = await _processImageWithBackend(blob, file.name.replace(/\.pdf$/i, '.jpg') || 'page.jpg');
      _showProgress(85);
      loadPayload(payload);
      closeModal();
      Toast.show(
        `PDF ready — ${payload.blocks.length} editable text block(s)`,
        'success',
        4000
      );
      return;
    }

    const fallback = await _buildFallbackPayload(src, width, height);
    loadPayload(fallback);
    closeModal();
    Toast.show(
      'Offline OCR not ready. Added a manual text block instead.',
      'error',
      7000
    );
  }
}


/* ══════════════════════════════════════════════════════════════════════
   CORE LOADER
══════════════════════════════════════════════════════════════════════ */

/**
 * loadPayload(payload)
 * ─────────────────────
 * Accepts a parsed JSON payload and drives the full render pipeline.
 * After the background image loads, EdgeML.predictFont() is invoked
 * for each block that does not already have a font_family specified.
 *
 * NOTE: This function previously called `fetch('/api/process-image')`
 * (the FastAPI backend). That call has been removed entirely.
 * All processing — font classification via ONNX, image conversion via
 * PDF.js, and future inpainting via OpenCV.js — now runs client-side.
 */
function loadPayload(payload) {
  if (!payload || typeof payload !== 'object') {
    Toast.show('Invalid payload: must be a JSON object', 'error');
    return;
  }
  if (!payload.bg_image) {
    Toast.show('Payload missing "bg_image" field', 'error');
    return;
  }
  if (!Array.isArray(payload.blocks)) {
    Toast.show('Payload missing "blocks" array', 'error');
    return;
  }

  AppState.payload = payload;
  AppState.blocks  = payload.blocks;
  AppState.bgSrc   = payload.bg_image;
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
      enrichedBlocks = payload.blocks.map((b) => ({
        ...b,
        font_family: b.font_family || b.fontFamily || 'Arial',
      }));
    }

    ScaleEngine.recompute();
    OverlayEngine.renderAll(enrichedBlocks);
    _ensureEditorInteractive();
    if (!payload.ocr_text) _setOcrTextFromBlocks(enrichedBlocks);

    requestAnimationFrame(() => {
      ScaleEngine.recompute();
    });

    _showProgress(90);

    ['btn-export', 'btn-export-2', 'btn-export-mobile', 'btn-reset'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = false;
    });
    if (typeof syncMobileButtons === 'function') syncMobileButtons();

    _setStatusOk(`${AppState.imageW} × ${AppState.imageH}px`);
    document.getElementById('label-dimensions').textContent =
      `${AppState.imageW} × ${AppState.imageH} px`;
    document.getElementById('label-blocks').textContent =
      `${enrichedBlocks.length} block${enrichedBlocks.length !== 1 ? 's' : ''}`;

    _showProgress(100);
    setTimeout(() => _showProgress(0), 400);

    if (enrichedBlocks.length === 0) {
      Toast.show('Image loaded. Import a JSON file with "blocks" to edit text.', 'info', 5000);
    } else {
      Toast.show(`Loaded ${enrichedBlocks.length} text block(s) — tap to edit`, 'success');
    }
    EditorState.clear();
    closeModal();
  };

  img.onerror = () => {
    _setStatusError('Failed to load image');
    _showProgress(0);
    Toast.show('Could not load image: ' + AppState.bgSrc, 'error');
  };

  img.src = AppState.bgSrc;
}

/**
 * _enrichBlocksWithFonts(imgEl, blocks)
 * ──────────────────────────────────────
 * For each block that lacks a font_family, calls EdgeML.predictFont()
 * to classify the glyph region. Returns a new array of enriched blocks.
 *
 * NOTE: This replaces the server-side FontClassifier in text_pipeline.py
 * (backend/text_pipeline.py, class FontClassifier, method predict()).
 *
 * @param {HTMLImageElement} imgEl
 * @param {Array} blocks
 * @returns {Promise<Array>}
 */
async function _enrichBlocksWithFonts(imgEl, blocks) {
  if (!blocks.length) return blocks;

  if (!AIRuntime.isOnnxReady()) {
    return blocks.map((b) => ({
      ...b,
      font_family: b.font_family || b.fontFamily || 'Arial',
    }));
  }

  const enriched = [];
  for (const block of blocks) {
    if (block.font_family && block.font_family !== 'sans-serif') {
      enriched.push({ ...block });
      continue;
    }

    const predicted = await EdgeML.predictFont(
      imgEl,
      block.x, block.y, block.w, block.h
    );

    enriched.push({ ...block, font_family: predicted });
  }

  return enriched;
}

/**
 * loadImageOnly(src, naturalW, naturalH)
 * ──────────────────────────────────────
 * Loads just an image as background with no text blocks.
 * Used when user drops an image file or a PDF is rendered to JPEG.
 */
function loadImageOnly(src, naturalW, naturalH) {
  const payload = {
    bg_image: src,
    image_w:  naturalW,
    image_h:  naturalH,
    blocks:   []
  };
  loadPayload(payload);
}


/* ══════════════════════════════════════════════════════════════════════
   RESET
══════════════════════════════════════════════════════════════════════ */
function resetAll() {
  if (!AppState.isLoaded) return;
  AppState.liveBlocks.forEach(live => {
    if (live.el) live.el.textContent = live.originalText;
    live.currentText = live.originalText;
  });
  EditorState.clear();
  OverlayEngine.clearActive();
  PropsPanel.clear();
  LayerPanel.rebuild();
  _setOcrTextFromBlocks(AppState.liveBlocks.map((b) => ({ text: b.originalText })));
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
  ['json', 'image', 'url'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`pane-${t}`).classList.toggle('hidden', t !== tab);
  });
}

async function loadFromModal() {
  if      (_currentTab === 'json')  _loadFromJSON();
  else if (_currentTab === 'image') Toast.show('Drop or select a file in the Image tab', 'info');
  else if (_currentTab === 'url')   _loadFromURL();
}

function _loadFromJSON() {
  const raw = document.getElementById('json-input').value.trim();
  if (!raw) { Toast.show('JSON textarea is empty', 'error'); return; }
  try {
    loadPayload(JSON.parse(raw));
  } catch (err) {
    Toast.show('Invalid JSON: ' + err.message, 'error');
  }
}

function _loadFromURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) { Toast.show('URL is empty', 'error'); return; }
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload  = () => loadImageOnly(url, img.naturalWidth, img.naturalHeight);
  img.onerror = () => Toast.show('Could not load image from URL', 'error');
  img.src = url;
  closeModal();
}


/* ══════════════════════════════════════════════════════════════════════
   DRAG & DROP HANDLERS
══════════════════════════════════════════════════════════════════════ */

function handleDragOver(event) {
  event.preventDefault();
  event.dataTransfer.dropEffect = 'copy';
  document.getElementById('welcome-drop').classList.add('drag-over');
}

function handleDragLeave() {
  document.getElementById('welcome-drop').classList.remove('drag-over');
}

function handleFileDrop(event) {
  event.preventDefault();
  document.getElementById('welcome-drop').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (!file) return;
  _processFile(file);
}

function handleModalDrop(event) {
  event.preventDefault();
  document.getElementById('modal-drop-zone').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (!file) return;
  _processFile(file);
}

function handleModalFileInput(event) {
  const file = event.target.files[0];
  if (!file) return;
  _processFile(file);
}

function handleHiddenFileInput(event) {
  const file = event.target.files[0];
  if (!file) return;
  _processFile(file);
}

/**
 * _processFile(file)
 * ──────────────────
 * Routes a dropped/selected File to the correct handler.
 *
 * ┌─────────────────┬──────────────────────────────────────────────────┐
 * │ image/*         │ FileReader → data URI → loadImageOnly()          │
 * │ application/pdf │ PDF.js renders page 1 at 3× → loadImageOnly()   │  ← NEW
 * │ .json           │ FileReader → JSON.parse → loadPayload()          │
 * └─────────────────┴──────────────────────────────────────────────────┘
 *
 * The PDF path previously showed an error toast and required the user to
 * run the Python backend. It now renders entirely in-browser via PDF.js.
 */
async function _processFile(file) {
  if (!file) return;

  // ── JSON payload ──────────────────────────────────────────────────
  if (file.type === 'application/json' || file.name.endsWith('.json')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try { loadPayload(JSON.parse(e.target.result)); }
      catch (err) { Toast.show('JSON parse error: ' + err.message, 'error'); }
    };
    reader.readAsText(file);
    return;
  }

  // ── Raster image (JPEG, PNG, WEBP, BMP) ──────────────────────────
  if (file.type.startsWith('image/')) {
    try {
      const src = await _readFileAsDataUri(file);
      const payload = await _processImageWithOfflineOcr(src);
      loadPayload(payload);
      closeModal();
      Toast.show(`Found ${payload.blocks.length} text block(s)`, 'success');
    } catch (err) {
      console.warn('[OCR] Offline OCR failed:', err);
      if (getApiBaseUrl()) {
        try {
          Toast.show('Running OCR on server…', 'info');
          _showProgress(30);
          const payload = await _processImageWithBackend(file, file.name);
          _showProgress(85);
          loadPayload(payload);
          closeModal();
          Toast.show(`Found ${payload.blocks.length} text block(s)`, 'success');
          return;
        } catch (serverErr) {
          console.error('[OCR] Server failed:', serverErr);
          _aiAlert('image OCR', serverErr);
          _showProgress(0);
        }
      }

      const src = await _readFileAsDataUri(file);
      const fallback = await _buildFallbackPayload(src);
      loadPayload(fallback);
      closeModal();
      Toast.show('Offline OCR not ready. Added a manual text block instead.', 'error', 6000);
    }
    return;
  }

  // ── PDF — page 1 → optional backend OCR for editable blocks ───────
  if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
    try {
      await _processPdfFile(file);
    } catch (err) {
      console.error('[PDF] Failed:', err);
      _aiAlert('PDF import', err);
      _showProgress(0);
    }
    return;
  }

  Toast.show('Unsupported file type: ' + (file.type || file.name), 'error');
}


/* ══════════════════════════════════════════════════════════════════════
   STATUS BAR HELPERS
══════════════════════════════════════════════════════════════════════ */
function _setStatusBusy(msg) {
  const dot = document.getElementById('dot-backend');
  const lbl = document.getElementById('label-backend');
  if (dot) dot.className = 'status-dot amber';
  if (lbl) lbl.textContent = msg;
}

function _setStatusOk(msg) {
  const dot = document.getElementById('dot-backend');
  const lbl = document.getElementById('label-backend');
  if (dot) dot.className = 'status-dot green';
  if (lbl) lbl.textContent = msg || 'Ready';
}

function _setStatusError(msg) {
  const dot = document.getElementById('dot-backend');
  const lbl = document.getElementById('label-backend');
  if (dot) dot.className = 'status-dot red';
  if (lbl) lbl.textContent = msg;
}

function _showProgress(pct) {
  const track = document.getElementById('progress-track');
  const fill  = document.getElementById('progress-fill');
  if (!track || !fill) return;
  if (pct <= 0) { track.style.display = 'none'; fill.style.width = '0%'; }
  else          { track.style.display = 'block'; fill.style.width = pct + '%'; }
}

/**
 * _setBadge(id, state, text)
 * ──────────────────────────
 * Updates the Edge AI status badges shown in the status bar.
 * @param {'badge-onnx'|'badge-cv'} id
 * @param {'badge-onnx'|'badge-cv'|'badge-ocr'} id
 * @param {'loading'|'ready'|'error'} state
 * @param {string} text
 */
function _setBadge(id, state, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `badge badge-${state}`;
  el.textContent = text;
}


/* ══════════════════════════════════════════════════════════════════════
   KEYBOARD SHORTCUTS
══════════════════════════════════════════════════════════════════════ */
document.addEventListener('keydown', (e) => {
  const ctrl = e.ctrlKey || e.metaKey;

  if (ctrl && !e.shiftKey && e.key === 'z') {
    if (document.activeElement.classList.contains('txt-block')) return;
    e.preventDefault();
    EditorState.undo();
    return;
  }
  if (ctrl && (e.shiftKey && e.key === 'z' || e.key === 'y')) {
    if (document.activeElement.classList.contains('txt-block')) return;
    e.preventDefault();
    EditorState.redo();
    return;
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
   INIT — runs once on page load
══════════════════════════════════════════════════════════════════════ */
(async function init() {
  console.info('[PixelScribe] Edge AI Edition — initialising…');

  _setImportEnabled(true);
  _setStatusBusy('Loading AI libraries…');

  _setBadge('badge-ocr', 'loading', 'OCR loading...');
  OfflineOCR.init()
    .then(() => {
      const label = OfflineOCR.engine() === 'text-detector'
        ? 'OCR ready (native)'
        : 'OCR ready';
      _setBadge('badge-ocr', 'ready', label);
    })
    .catch((err) => {
      console.warn('[PixelScribe] Offline OCR init failed:', err.message);
      _setBadge('badge-ocr', 'error', 'OCR missing');
    });

  const onnxBoot = EdgeML.init();
  AIRuntime.setOnnxInitPromise(onnxBoot);
  await onnxBoot;
  _setImportEnabled(true);

  // Wire canvas-area click to deselect blocks
  document.getElementById('canvas-area').addEventListener('click', (e) => {
    if (e.target.id === 'canvas-area' ||
        e.target.id === 'workspace'   ||
        e.target.id === 'canvas-img') {
      OverlayEngine.clearActive();
    }
  });

  // ── Log keyboard shortcuts ──────────────────────────────────────
  console.info(
    '%cKeyboard shortcuts:\n' +
    '  Ctrl+O        — Open / Import\n' +
    '  Ctrl+S/E      — Export PNG\n' +
    '  Ctrl+Z        — Undo\n' +
    '  Ctrl+Shift+Z  — Redo\n' +
    '  Ctrl++/-      — Zoom in/out\n' +
    '  Ctrl+0        — Fit to window\n' +
    '  Escape        — Deselect / close modal',
    'color:#9B95FF;font-family:monospace;font-size:11px'
  );

  // ── Auto-load demo payload if ?demo=1 in URL ────────────────────
  if (new URLSearchParams(window.location.search).get('demo') === '1') {
    _loadDemoPayload();
  }

  // Fallback: Capacitor/WebView sometimes misses onOpenCvLoad / onRuntimeInitialized
  const cvInterval = setInterval(() => {
    if (typeof cv !== 'undefined' && typeof cv.imread === 'function' && !cvReady) {
      clearInterval(cvInterval);
      console.info('[PixelScribe] OpenCV.js ready (interval fallback).');
      _markCvReady();
    }
  }, 200);

  setTimeout(() => clearInterval(cvInterval), 120000);

  try {
    await AIRuntime.waitUntilReady(120000);
    console.info('[PixelScribe] AI runtime ready (OpenCV + ONNX).');
  } catch (err) {
    console.warn('[PixelScribe] AI partial/unavailable:', err.message);
    _setImportEnabled(true);
    if (AIRuntime.isCvReady()) {
      _setStatusOk('Ready — import & edit (some AI features limited)');
    } else {
      _setStatusOk('Ready — import files (AI still loading)');
    }
  }

  _refreshUiLock();
})();


/* ══════════════════════════════════════════════════════════════════════
   DEMO PAYLOAD  (activated via ?demo=1)
══════════════════════════════════════════════════════════════════════ */
function _loadDemoPayload() {
  const demoCanvas    = document.createElement('canvas');
  demoCanvas.width    = 1200;
  demoCanvas.height   = 800;
  const ctx           = demoCanvas.getContext('2d');

  const grad = ctx.createLinearGradient(0, 0, 1200, 800);
  grad.addColorStop(0, '#F8F4EF');
  grad.addColorStop(1, '#EDE8DC');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1200, 800);

  ctx.strokeStyle = 'rgba(0,0,0,0.04)';
  ctx.lineWidth   = 1;
  for (let x = 0; x < 1200; x += 60) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,800); ctx.stroke(); }
  for (let y = 0; y < 800;  y += 60) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(1200,y); ctx.stroke(); }

  ctx.fillStyle   = '#FFFFFF';
  ctx.shadowColor = 'rgba(0,0,0,0.1)';
  ctx.shadowBlur  = 20;
  ctx.shadowOffsetY = 4;
  _roundRect(ctx, 80, 60, 1040, 680, 12);
  ctx.fill();
  ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;

  ctx.fillStyle = '#6C63FF';
  _roundRect(ctx, 80, 60, 1040, 8, { tl:12, tr:12, br:0, bl:0 });
  ctx.fill();

  const demoPayload = {
    bg_image: demoCanvas.toDataURL('image/jpeg', 0.92),
    image_w:  1200,
    image_h:  800,
    blocks: [
      { text:'PixelScribe Edge AI',            x:120, y:100, w:600, h:64,  color:'#1A1A2E', size:48,  font_family:'Georgia',     confidence:0.99 },
      { text:'100% offline — runs in your browser via WebAssembly.', x:120, y:180, w:800, h:40, color:'#4A4A62', size:22, font_family:'Arial', confidence:0.97 },
      { text:'Edit this text. Click any block to select it, then type.', x:120, y:240, w:720, h:32, color:'#6C63FF', size:16, font_family:'Courier New', confidence:0.95 },
      { text:'Font: Times New Roman · Classified by ONNX in-browser',  x:120, y:290, w:700, h:28, color:'#2A2A38', size:15, font_family:'Times New Roman', confidence:0.94 },
      { text:'PDF.js renders uploaded PDFs to a canvas — no upload needed.', x:120, y:340, w:700, h:26, color:'#444', size:14, font_family:'Calibri', confidence:0.91 },
      { text:'OpenCV.js (cvReady flag) will power in-browser inpainting.', x:120, y:385, w:700, h:26, color:'#555', size:14, font_family:'Verdana', confidence:0.88 },
      { text:'Roboto — Google Fonts fallback stack active.', x:120, y:430, w:660, h:26, color:'#3A3A48', size:14, font_family:'Roboto', confidence:0.93 },
      { text:'Helvetica Neue — matched via CSS font-family stack.', x:120, y:475, w:680, h:26, color:'#2E2E3E', size:14, font_family:'Helvetica', confidence:0.90 },
      { text:'Garamond — elegant serif for body text.', x:120, y:520, w:620, h:26, color:'#3E3024', size:15, font_family:'Garamond', confidence:0.87 },
      { text:'Consolas: monospaced → perfect for code snippets.',  x:120, y:570, w:640, h:26, color:'#6B6B88', size:14, font_family:'Consolas', confidence:0.85 },
    ]
  };

  loadPayload(demoPayload);
  Toast.show('Demo loaded — Edge AI · No server · WASM', 'info', 4000);
}

function _roundRect(ctx, x, y, w, h, r) {
  if (typeof r === 'number') r = { tl:r, tr:r, br:r, bl:r };
  ctx.beginPath();
  ctx.moveTo(x + r.tl, y);
  ctx.lineTo(x + w - r.tr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r.tr);
  ctx.lineTo(x + w, y + h - r.br);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r.br, y + h);
  ctx.lineTo(x + r.bl, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r.bl);
  ctx.lineTo(x, y + r.tl);
  ctx.quadraticCurveTo(x, y, x + r.tl, y);
  ctx.closePath();
}