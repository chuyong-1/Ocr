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

/**
 * Called by the `onload` attribute of the <script async src="opencv.js"> tag.
 * OpenCV.js sets cv.onRuntimeInitialized internally; we hook into it here.
 */
function onOpenCvLoad() {
  // cv may already be ready if the script was cached
  if (typeof cv !== 'undefined') {
    if (cv.getBuildInformation) {
      // Already initialised (cached/synchronous load)
      _markCvReady();
    } else {
      // Async WASM compile — wait for the callback
      cv.onRuntimeInitialized = _markCvReady;
    }
  }
}

function _markCvReady() {
  cvReady = true;
  console.info('[PixelScribe] OpenCV.js WASM ready.');
  _setBadge('badge-cv', 'ready', 'CV ready');
}


/* ══════════════════════════════════════════════════════════════════════
   EDGE ML  — ONNX Runtime Web font classifier
   Model:  ../models/font_classifier.onnx
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
    if (_session) return;   // already loaded

    if (typeof ort === 'undefined') {
      console.warn('[EdgeML] onnxruntime-web not loaded — font prediction disabled.');
      _setBadge('badge-onnx', 'error', 'ONNX unavailable');
      return;
    }

    try {
      // Configure ONNX Runtime to use WASM execution provider.
      // The .wasm files are loaded from the same CDN as ort.min.js.
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

      _setBadge('badge-onnx', 'loading', 'ONNX loading…');

      _session = await ort.InferenceSession.create(
        '../models/font_classifier.onnx',
        { executionProviders: ['wasm'] }
      );

      console.info('[EdgeML] ONNX session ready. Input:', _session.inputNames, 'Output:', _session.outputNames);
      _setBadge('badge-onnx', 'ready', 'ONNX ready');
    } catch (err) {
      console.warn('[EdgeML] Could not load ONNX model:', err.message);
      _setBadge('badge-onnx', 'error', 'ONNX error');
      _session = null;
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
    if (!cvReady) {
      throw new Error('OpenCV is not loaded yet.');
    }

    let src = cv.imread(imageElement);
    let mask = new cv.Mat(src.rows, src.cols, cv.CV_8UC1, new cv.Scalar(0));

    // Add 4px padding/dilation to the bounding box
    let x1 = Math.max(0, bbox.x - 4);
    let y1 = Math.max(0, bbox.y - 4);
    let x2 = Math.min(src.cols, bbox.x + bbox.width + 4);
    let y2 = Math.min(src.rows, bbox.y + bbox.height + 4);

    cv.rectangle(mask, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(255), -1, cv.LINE_8, 0);

    let dst = new cv.Mat();
    cv.inpaint(src, mask, dst, 3, cv.INPAINT_TELEA);

    // Render back to a hidden canvas to update the main image source
    let hiddenCanvas = document.createElement('canvas');
    cv.imshow(hiddenCanvas, dst);
    AppState.bgSrc = hiddenCanvas.toDataURL('image/png');
    imageElement.src = AppState.bgSrc;

    // CRITICAL: Prevent WebAssembly memory leaks
    src.delete();
    mask.delete();
    dst.delete();
  }

  return { inpaintRegion };
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
    AppState.scaleFactor = rendered.width / AppState.imageW;
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
    });

    let _clickCount = 0;
    el.addEventListener('click', async () => {
      _clickCount++;
      if (_clickCount === 1) {
        _selectAllText(el);
        setTimeout(() => { _clickCount = 0; }, 600);
      } else if (_clickCount === 2) {
        // Double-click to erase original text via inpainting
        const imgEl = document.getElementById('canvas-img');
        if (imgEl && !live.inpainted) {
          try {
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
      item.addEventListener('click', () => {
        const field = document.getElementById(`field-${live.id}`);
        if (field) { field.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); field.focus(); }
        OverlayEngine.setActive(live.id);
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
    throw new Error('PDF.js is not loaded. Check the CDN script tag in index.html.');
  }

  // Ensure worker path is configured (safe to set more than once)
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

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

  const img = document.getElementById('canvas-img');
  const ws  = document.getElementById('workspace');

  _setStatusBusy('Loading image…');
  _showProgress(20);

  img.onload = async () => {
    AppState.imageW   = payload.image_w || img.naturalWidth;
    AppState.imageH   = payload.image_h || img.naturalHeight;
    AppState.isLoaded = true;

    _showProgress(50);
    ScaleEngine.recompute();

    // ── Edge AI: run font classifier on blocks that lack font_family ──
    //
    // Previously the Python backend (text_pipeline.py FontClassifier)
    // did this. Now EdgeML.predictFont() runs the same ONNX model in
    // the browser via WASM — no server round-trip.
    //
    // We clone the blocks array and enrich each entry with a predicted
    // font_family before handing it to OverlayEngine.renderAll().
    const enrichedBlocks = await _enrichBlocksWithFonts(img, payload.blocks);

    OverlayEngine.renderAll(enrichedBlocks);

    document.getElementById('welcome-drop').style.display = 'none';
    ws.style.display = 'block';

    _showProgress(90);

    ['btn-export', 'btn-export-2', 'btn-reset'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = false;
    });

    _setStatusOk(`${AppState.imageW} × ${AppState.imageH}px`);
    document.getElementById('label-dimensions').textContent =
      `${AppState.imageW} × ${AppState.imageH} px`;
    document.getElementById('label-blocks').textContent =
      `${enrichedBlocks.length} block${enrichedBlocks.length !== 1 ? 's' : ''}`;

    _showProgress(100);
    setTimeout(() => _showProgress(0), 400);

    Toast.show(`Loaded ${enrichedBlocks.length} text block(s)`, 'success');
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

  const enriched = [];
  for (const block of blocks) {
    // If the payload already has a font prediction, trust it.
    if (block.font_family && block.font_family !== 'sans-serif') {
      enriched.push({ ...block });
      continue;
    }

    // Run EdgeML inference on this block's bounding box crop.
    // Falls back to 'Arial' if the model isn't loaded yet.
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

function loadFromModal() {
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
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target.result;
      const img = new Image();
      img.onload = () => loadImageOnly(src, img.naturalWidth, img.naturalHeight);
      img.src = src;
    };
    reader.readAsDataURL(file);
    return;
  }

  // ── PDF — rendered client-side by PDF.js (replaces Python pipeline) ─
  if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
    Toast.show('Rendering PDF page 1 via PDF.js…', 'info');
    _showProgress(15);

    try {
      const { src, width, height } = await _renderPdfToImage(file);
      _showProgress(60);
      loadImageOnly(src, width, height);
      Toast.show(`PDF rendered: ${width}×${height}px`, 'success');
    } catch (err) {
      console.error('[PDF] Render failed:', err);
      Toast.show('PDF render failed: ' + err.message, 'error');
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

  // Wire canvas-area click to deselect blocks
  document.getElementById('canvas-area').addEventListener('click', (e) => {
    if (e.target.id === 'canvas-area' ||
        e.target.id === 'workspace'   ||
        e.target.id === 'canvas-img') {
      OverlayEngine.clearActive();
    }
  });

  // ── Boot EdgeML (ONNX Runtime Web) ──────────────────────────────
  // This replaces the server-side FontClassifier lazy-load in worker.py.
  // The model loads asynchronously and does not block the UI.
  await EdgeML.init();

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

  // ── OpenCV initialization listener (Interval Check) ───────────────
  const cvInterval = setInterval(() => {
    if (typeof cv !== 'undefined' && typeof cv.imread === 'function') {
      clearInterval(cvInterval);
      if (!cvReady) {
        cvReady = true;
        console.info('[PixelScribe] OpenCV.js WASM fully initialized.');
        if (typeof _setBadge === 'function') {
          _setBadge('badge-cv', 'ready', 'CV ready');
        }
      }
    }
  }, 100);
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