/**
 * PixelScribe — main.js
 * ════════════════════════════════════════════════════════════════════════
 * Vanilla JS editor engine for the content-aware image text editor.
 *
 * Modules
 * ───────
 *  AppState        — centralised application data store
 *  EditorState     — undo/redo history stack
 *  ScaleEngine     — coordinate scaling between native px and display px
 *  OverlayEngine   — DOM injection of contenteditable text blocks
 *  PropsPanel      — right sidebar typography controls
 *  LayerPanel      — left sidebar layer list
 *  ExportEngine    — off-screen canvas flatten + file download
 *  CanvasView      — zoom / fit-to-window control
 *  Toast           — lightweight notification system
 *  Modal           — open/load modal logic
 *
 * Expected JSON payload shape
 * ───────────────────────────
 * {
 *   "bg_image":  "/results/cleaned_job123.jpg",  // or data URI or absolute URL
 *   "image_w":   2400,
 *   "image_h":   3000,
 *   "blocks": [
 *     {
 *       "text":        "Hello World",
 *       "x": 120, "y": 450, "w": 800, "h": 60,
 *       "color":       "#1A1A1A",
 *       "bg_color":    "#F5F0E8",   // optional
 *       "size":        18,
 *       "font_family": "Arial",      // optional — falls back to Arial
 *       "confidence":  0.98           // optional
 *     }
 *   ]
 * }
 * ════════════════════════════════════════════════════════════════════════
 */

'use strict';

/* ══════════════════════════════════════════════════════════════════════
   FONT FALLBACK MAP — CSS font-family stacks for the 10-class classifier
   Each classifier label maps to a full CSS fallback chain so that:
   • Fonts render correctly even if the exact family isn't installed
   • Serif → serif, mono → monospace, sans → sans-serif
   • Google Fonts aliases (EB Garamond, Roboto Mono) are included
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

/** Resolve a classifier label to a full CSS font-family stack */
function resolveFontStack(fontFamily) {
  return FONT_FALLBACK_MAP[fontFamily] || `"${fontFamily}", sans-serif`;
}

/* ══════════════════════════════════════════════════════════════════════
   APP STATE — single source of truth
══════════════════════════════════════════════════════════════════════ */
const AppState = {
  /** Raw payload from worker / JSON paste */
  payload: null,          // full parsed JSON object
  imageW:  0,             // native image width  (px)
  imageH:  0,             // native image height (px)
  blocks:  [],            // original blocks array (immutable reference)
  bgSrc:   '',            // resolved bg_image URL or data URI

  /** Live editing state — mutated as user types */
  liveBlocks: [],         // { ...blockData, el: HTMLElement }

  /** UI state */
  activeId:     null,     // currently selected block id
  scaleFactor:  1,        // rendered width / native width
  zoomLevel:    1,        // user-applied zoom on top of fit
  isLoaded:     false,

  /** Reset everything */
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
   UNDO / REDO HISTORY STACK
══════════════════════════════════════════════════════════════════════ */
const EditorState = (() => {
  /** Each entry: { blockId, oldText, newText } */
  const undoStack = [];
  const redoStack = [];
  const MAX_DEPTH = 80;

  function push(blockId, oldText, newText) {
    if (oldText === newText) return;
    undoStack.push({ blockId, oldText, newText });
    if (undoStack.length > MAX_DEPTH) undoStack.shift();
    redoStack.length = 0;     // clear redo on new action
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
   Calculates and applies the ratio between native image pixels
   and the rendered display pixels inside #workspace.
══════════════════════════════════════════════════════════════════════ */
const ScaleEngine = (() => {

  /**
   * Recomputes scaleFactor from the currently rendered <img> size.
   * Must be called after image load and on every resize.
   */
  function recompute() {
    const img = document.getElementById('canvas-img');
    if (!img || !AppState.imageW) return;

    // getBoundingClientRect gives us the *rendered* pixel size
    const rendered = img.getBoundingClientRect();
    AppState.scaleFactor = rendered.width / AppState.imageW;

    // Re-position all live overlays without re-creating them
    if (AppState.liveBlocks.length) {
      OverlayEngine.repositionAll();
    }
  }

  /**
   * Scale a single value from native px → display px
   */
  function toDisplay(v) {
    return Math.round(v * AppState.scaleFactor);
  }

  /**
   * Scale a value back from display px → native px
   */
  function toNative(v) {
    return Math.round(v / AppState.scaleFactor);
  }

  return { recompute, toDisplay, toNative };
})();


/* ══════════════════════════════════════════════════════════════════════
   OVERLAY ENGINE
   Creates, positions, and manages contenteditable text blocks
   over the cleaned canvas image.
══════════════════════════════════════════════════════════════════════ */
const OverlayEngine = (() => {

  /**
   * Build a unique per-run block id
   */
  function _makeId(index) {
    return `block-${index}-${Date.now()}`;
  }

  /**
   * Primary render pass — wipes #overlay and injects all blocks.
   * Called once after a payload is loaded.
   */
  function renderAll(blocks) {
    const overlay = document.getElementById('overlay');
    overlay.innerHTML = '';
    AppState.liveBlocks = [];

    blocks.forEach((block, idx) => {
      const id = _makeId(idx);

      // Store live block data
      const live = {
        id,
        ...block,
        font_family:  block.font_family  || block.fontFamily || 'Arial',
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

  /**
   * Create a single contenteditable div for one block.
   */
  function _createField(live) {
    const s  = AppState.scaleFactor;
    const el = document.createElement('div');

    el.id              = `field-${live.id}`;
    el.contentEditable = 'true';
    el.className       = 'txt-block';
    el.dataset.blockId = live.id;
    el.spellcheck      = false;
    el.textContent     = live.currentText;

    // ── Scaled position ──
    el.style.left      = `${Math.round(live.x * s)}px`;
    el.style.top       = `${Math.round(live.y * s)}px`;
    el.style.width     = `${Math.round(live.w * s)}px`;
    el.style.minHeight = `${Math.round(live.h * s)}px`;

    // ── Typography ──
    el.style.fontSize   = `${Math.round(live.size * s)}px`;
    el.style.color      = live.color || '#1A1A1A';
    el.style.fontFamily = resolveFontStack(live.font_family);
    el.style.lineHeight = '1.25';

    // Optional: soft background hint from bg_color
    if (live.bg_color) {
      el.style.backgroundColor = _hexWithAlpha(live.bg_color, 0.0);
    }

    // ── Focus: record pre-edit text for undo ──
    let _textOnFocus = '';
    el.addEventListener('focus', () => {
      _textOnFocus = el.textContent;
      _selectAllText(el);
      _setActive(live.id);
    });

    // ── Blur: push to undo stack ──
    el.addEventListener('blur', () => {
      const newText = el.textContent;
      EditorState.push(live.id, _textOnFocus, newText);
      live.currentText = newText;
      // Sync layer list text
      const layerTextEl = document.getElementById(`lyr-text-${live.id}`);
      if (layerTextEl) layerTextEl.textContent = `"${newText}"`;
    });

    // ── Click: select all on first click ──
    let _clickCount = 0;
    el.addEventListener('click', () => {
      _clickCount++;
      if (_clickCount === 1) {
        _selectAllText(el);
        setTimeout(() => { _clickCount = 0; }, 600);
      }
    });

    // ── Keyboard: Escape blurs ──
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        el.blur();
        e.preventDefault();
      }
    });

    return el;
  }

  /**
   * Re-positions all existing overlays after a scale change.
   * Does NOT recreate DOM elements — only updates CSS.
   */
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

  /**
   * Updates visual style on a specific block (font, size, color).
   * Used by PropsPanel without re-creating the element.
   */
  function updateBlockStyle(blockId, { font_family, size, color }) {
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (!live || !live.el) return;

    const s = AppState.scaleFactor;

    if (font_family !== undefined) {
      live.font_family          = font_family;
      live.el.style.fontFamily  = resolveFontStack(font_family);
    }
    if (size !== undefined) {
      live.size                = size;
      live.el.style.fontSize   = `${Math.round(size * s)}px`;
    }
    if (color !== undefined) {
      live.color              = color;
      live.el.style.color     = color;
      // Sync layer swatch
      const swatch = document.getElementById(`lyr-swatch-${blockId}`);
      if (swatch) swatch.style.background = color;
    }
  }

  /**
   * Sets the active (selected) block, highlights it, updates props panel.
   */
  function _setActive(blockId) {
    AppState.activeId = blockId;

    // Remove selected class from all
    document.querySelectorAll('.txt-block').forEach(el => {
      el.classList.toggle('selected', el.dataset.blockId === blockId);
    });

    // Highlight layer item
    document.querySelectorAll('.layer-item').forEach(el => {
      el.classList.toggle('active', el.dataset.blockId === blockId);
    });

    // Populate props panel
    const live = AppState.liveBlocks.find(b => b.id === blockId);
    if (live) PropsPanel.populate(live);
  }

  /**
   * Select all text inside a contenteditable element.
   */
  function _selectAllText(el) {
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  /**
   * Convert hex + alpha to rgba string
   */
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
   PROPS PANEL — right sidebar controls
══════════════════════════════════════════════════════════════════════ */
const PropsPanel = (() => {

  const _ids = {
    font:       'prop-font',
    size:       'prop-size',
    color:      'prop-color',
    colorHex:   'prop-color-hex',
    swatch:     'swatch-fg',
    applyAll:   'btn-apply-all',
    coordX:     'coord-x',
    coordY:     'coord-y',
    coordW:     'coord-w',
    coordH:     'coord-h',
    infoOrig:   'info-original',
    infoConf:   'info-conf',
    sizeMinus:  'stepper-size',
  };

  /**
   * Populate panel fields from a live block object.
   */
  function populate(live) {
    _setVal(_ids.font,     live.font_family || 'Arial');
    _setVal(_ids.size,     live.size || 16);
    _setVal(_ids.color,    live.color || '#000000');
    _setVal(_ids.colorHex, live.color || '#000000');
    document.getElementById(_ids.swatch).style.background = live.color || '#000000';

    // Coordinates (native px)
    document.getElementById(_ids.coordX).textContent = live.x + ' px';
    document.getElementById(_ids.coordY).textContent = live.y + ' px';
    document.getElementById(_ids.coordW).textContent = live.w + ' px';
    document.getElementById(_ids.coordH).textContent = live.h + ' px';

    // Info
    document.getElementById(_ids.infoOrig).textContent = `"${live.originalText}"`;
    document.getElementById(_ids.infoConf).textContent =
      live.confidence ? (live.confidence * 100).toFixed(1) + '%' : '—';

    _enableAll(true);
  }

  /**
   * Clear panel (no selection).
   */
  function clear() {
    _enableAll(false);
    ['coordX','coordY','coordW','coordH'].forEach(k =>
      document.getElementById(_ids[k]).textContent = '—'
    );
    document.getElementById(_ids.infoOrig).textContent = '—';
    document.getElementById(_ids.infoConf).textContent = '—';
  }

  // ── Individual apply handlers ──────────────────────────────────────

  function applyFont() {
    if (!AppState.activeId) return;
    const val = document.getElementById(_ids.font).value;
    OverlayEngine.updateBlockStyle(AppState.activeId, { font_family: val });
  }

  function applySize() {
    if (!AppState.activeId) return;
    const val = parseInt(document.getElementById(_ids.size).value, 10);
    if (!isNaN(val) && val > 0) {
      OverlayEngine.updateBlockStyle(AppState.activeId, { size: val });
    }
  }

  function stepSize(delta) {
    const input = document.getElementById(_ids.size);
    const val   = parseInt(input.value, 10) + delta;
    if (val >= 1 && val <= 400) {
      input.value = val;
      applySize();
    }
  }

  function applyColor() {
    if (!AppState.activeId) return;
    const val = document.getElementById(_ids.color).value;
    document.getElementById(_ids.colorHex).value = val;
    document.getElementById(_ids.swatch).style.background = val;
    OverlayEngine.updateBlockStyle(AppState.activeId, { color: val });
  }

  function applyColorHex() {
    if (!AppState.activeId) return;
    let val = document.getElementById(_ids.colorHex).value.trim();
    if (!val.startsWith('#')) val = '#' + val;
    if (!/^#[0-9A-Fa-f]{6}$/.test(val)) return;  // ignore invalid
    document.getElementById(_ids.color).value = val;
    document.getElementById(_ids.swatch).style.background = val;
    OverlayEngine.updateBlockStyle(AppState.activeId, { color: val });
  }

  function applyAll() {
    if (!AppState.activeId) return;
    applyFont();
    applySize();
    applyColor();
    Toast.show('Typography applied', 'success');
  }

  // ── Private helpers ───────────────────────────────────────────────

  function _setVal(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = val;
  }

  function _enableAll(enabled) {
    ['prop-font','prop-size','prop-color','prop-color-hex','btn-apply-all'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = !enabled;
    });
    // Stepper buttons
    document.querySelectorAll('#stepper-size .step-btn').forEach(btn => {
      btn.disabled = !enabled;
    });
  }

  return { populate, clear, applyFont, applySize, stepSize, applyColor, applyColorHex, applyAll };
})();


/* ══════════════════════════════════════════════════════════════════════
   LAYER PANEL — left sidebar
══════════════════════════════════════════════════════════════════════ */
const LayerPanel = (() => {

  function rebuild() {
    const list   = document.getElementById('layer-list');
    const empty  = document.getElementById('empty-state');
    const count  = document.getElementById('layer-count');

    list.innerHTML = '';

    if (!AppState.liveBlocks.length) {
      list.appendChild(empty);
      count.textContent = '0';
      return;
    }

    count.textContent = AppState.liveBlocks.length;

    AppState.liveBlocks.forEach((live, i) => {
      const item = document.createElement('div');
      item.className     = 'layer-item';
      item.dataset.blockId = live.id;
      item.style.animationDelay = `${i * 18}ms`;

      // Colour swatch
      const swatch = document.createElement('div');
      swatch.className = 'layer-swatch';
      swatch.id        = `lyr-swatch-${live.id}`;
      swatch.style.background = live.color || '#888';

      // Text preview
      const text = document.createElement('div');
      text.className = 'layer-text';
      text.id        = `lyr-text-${live.id}`;
      text.textContent = `"${live.currentText}"`;

      // Meta: font family badge
      const meta = document.createElement('div');
      meta.className   = 'layer-meta';
      meta.textContent = live.font_family || `${live.size}px`;

      item.append(swatch, text, meta);

      // Click: focus the corresponding text field
      item.addEventListener('click', () => {
        const field = document.getElementById(`field-${live.id}`);
        if (field) {
          field.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          field.focus();
        }
        OverlayEngine.setActive(live.id);
      });

      list.appendChild(item);
    });
  }

  return { rebuild };
})();


/* ══════════════════════════════════════════════════════════════════════
   EXPORT ENGINE
   Flattens the composite (bg image + edited text) onto a hidden
   <canvas> at full native resolution and triggers a browser download.
   No server round-trips.
══════════════════════════════════════════════════════════════════════ */
const ExportEngine = (() => {

  function download() {
    if (!AppState.isLoaded) {
      Toast.show('No image loaded', 'error');
      return;
    }

    Toast.show('Rendering composite…', 'info');
    _setProgress(10);

    const canvas = document.getElementById('export-canvas');
    canvas.width  = AppState.imageW;
    canvas.height = AppState.imageH;
    const ctx = canvas.getContext('2d');

    // ── Step 1: Draw background image ──────────────────────────────
    const bgImg = new Image();
    bgImg.crossOrigin = 'anonymous';

    bgImg.onload = () => {
      ctx.drawImage(bgImg, 0, 0, AppState.imageW, AppState.imageH);
      _setProgress(50);

      // ── Step 2: Draw each text block at native resolution ──────────
      AppState.liveBlocks.forEach(live => {
        const text     = live.el ? live.el.textContent : live.currentText;
        const nativeSz = live.size;         // already in native px
        const font     = live.font_family || 'Arial';
        const color    = live.color || '#000000';

        // Background fill (optional — respects bg_color if set)
        if (live.bg_color && live.bg_color !== 'transparent') {
          ctx.fillStyle = live.bg_color;
          ctx.fillRect(live.x, live.y, live.w, live.h);
        }

        // Text
        ctx.font         = `${nativeSz}px ${resolveFontStack(font)}`;
        ctx.fillStyle    = color;
        ctx.textBaseline = 'top';

        // Word-wrap: break long text across lines to fit block width
        _drawWrappedText(ctx, text, live.x + 2, live.y + 2, live.w - 4, nativeSz * 1.25);
      });

      _setProgress(90);

      // ── Step 3: Trigger download ──────────────────────────────────
      setTimeout(() => {
        try {
          const dataUrl  = canvas.toDataURL('image/png');
          const link     = document.createElement('a');
          link.download  = `pixelscribe-export-${Date.now()}.png`;
          link.href      = dataUrl;
          link.click();
          _setProgress(100);
          setTimeout(() => _setProgress(0), 600);
          Toast.show('Exported — check Downloads ✓', 'success');
        } catch (err) {
          console.error('[ExportEngine] toDataURL failed:', err);
          Toast.show('Export failed — CORS issue with image URL', 'error');
          _setProgress(0);
        }
      }, 80);
    };

    bgImg.onerror = () => {
      Toast.show('Could not load background image for export', 'error');
      _setProgress(0);
    };

    bgImg.src = AppState.bgSrc;
  }

  /**
   * Draw text with naive word-wrap to fit within maxWidth.
   */
  function _drawWrappedText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split(' ');
    let line    = '';

    for (let i = 0; i < words.length; i++) {
      const testLine  = line + words[i] + ' ';
      const metrics   = ctx.measureText(testLine);
      if (metrics.width > maxWidth && i > 0) {
        ctx.fillText(line.trim(), x, y);
        line = words[i] + ' ';
        y   += lineHeight;
      } else {
        line = testLine;
      }
    }
    ctx.fillText(line.trim(), x, y);
  }

  function _setProgress(pct) {
    const track = document.getElementById('progress-track');
    const fill  = document.getElementById('progress-fill');
    if (!track || !fill) return;
    if (pct <= 0) {
      track.style.display = 'none';
      fill.style.width    = '0%';
    } else {
      track.style.display = 'block';
      fill.style.width    = pct + '%';
    }
  }

  return { download };
})();


/* ══════════════════════════════════════════════════════════════════════
   CANVAS VIEW — zoom and fit control
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

    // The CSS max-width on the img already handles initial fit.
    // We set explicit width as a percentage of native to allow zoom.
    const pct = AppState.zoomLevel * 100;
    ws.style.width = pct > 100 ? `${Math.round(AppState.imageW * AppState.zoomLevel)}px` : '';

    // Update zoom label
    const label = document.getElementById('zoom-label');
    if (label) label.textContent = Math.round(AppState.zoomLevel * 100) + '%';

    // Recompute scale after layout settles
    requestAnimationFrame(() => {
      ScaleEngine.recompute();
    });
  }

  return { zoom, fitToWindow };
})();


/* ══════════════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
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
    // Trigger animation
    requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('show')));

    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  return { show };
})();


/* ══════════════════════════════════════════════════════════════════════
   CORE LOADER — processes a validated payload object
══════════════════════════════════════════════════════════════════════ */

/**
 * loadPayload(payload)
 * ─────────────────────
 * Accepts a parsed JSON payload object and drives the full render pipeline:
 *  1. Stores data in AppState
 *  2. Loads the background image
 *  3. On image load: recomputes scale, injects overlays
 *  4. Updates status bar and enables toolbar buttons
 */
function loadPayload(payload) {
  // ── Validate minimum required fields ──────────────────────────────
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

  console.info('[PixelScribe] Loading payload:', payload);

  // ── Store ──────────────────────────────────────────────────────────
  AppState.payload  = payload;
  AppState.blocks   = payload.blocks;
  AppState.bgSrc    = payload.bg_image;

  // ── Load background image ──────────────────────────────────────────
  const img = document.getElementById('canvas-img');
  const ws  = document.getElementById('workspace');

  _setStatusBusy('Loading image…');
  _showProgress(20);

  img.onload = () => {
    // Use metadata dimensions if provided, otherwise use natural size
    AppState.imageW = payload.image_w || img.naturalWidth;
    AppState.imageH = payload.image_h || img.naturalHeight;
    AppState.isLoaded = true;

    _showProgress(60);

    // Recompute scale factor from rendered size
    ScaleEngine.recompute();

    // Inject text overlays
    OverlayEngine.renderAll(AppState.blocks);

    // Show workspace, hide welcome
    document.getElementById('welcome-drop').style.display = 'none';
    ws.style.display = 'block';

    _showProgress(90);

    // Enable toolbar buttons
    ['btn-export', 'btn-export-2', 'btn-reset'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = false;
    });

    // Update status bar
    _setStatusOk(`${AppState.imageW} × ${AppState.imageH}px`);
    document.getElementById('label-dimensions').textContent =
      `${AppState.imageW} × ${AppState.imageH} px`;
    document.getElementById('label-blocks').textContent =
      `${AppState.blocks.length} block${AppState.blocks.length !== 1 ? 's' : ''}`;

    _showProgress(100);
    setTimeout(() => _showProgress(0), 400);

    Toast.show(`Loaded ${AppState.blocks.length} text block(s)`, 'success');
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
 * loadImageOnly(src, naturalW, naturalH)
 * ──────────────────────────────────────
 * Loads just an image as background with no text blocks.
 * Used when user drops an image file directly.
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

  // Re-render from original blocks (reset text to detected originals)
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

function openModal() {
  document.getElementById('modal-overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

function switchTab(tab) {
  _currentTab = tab;
  ['json', 'image', 'url'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`pane-${t}`).classList.toggle('hidden', t !== tab);
  });
}

function loadFromModal() {
  if (_currentTab === 'json') {
    _loadFromJSON();
  } else if (_currentTab === 'image') {
    Toast.show('Drop or select a file in the Image tab', 'info');
  } else if (_currentTab === 'url') {
    _loadFromURL();
  }
}

function _loadFromJSON() {
  const raw = document.getElementById('json-input').value.trim();
  if (!raw) {
    Toast.show('JSON textarea is empty', 'error');
    return;
  }
  try {
    const payload = JSON.parse(raw);
    loadPayload(payload);
  } catch (err) {
    Toast.show('Invalid JSON: ' + err.message, 'error');
  }
}

function _loadFromURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) {
    Toast.show('URL is empty', 'error');
    return;
  }
  // Load just the image, no text blocks
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => loadImageOnly(url, img.naturalWidth, img.naturalHeight);
  img.onerror = () => Toast.show('Could not load image from URL', 'error');
  img.src = url;
  closeModal();
}


/* ══════════════════════════════════════════════════════════════════════
   DRAG & DROP HANDLERS
══════════════════════════════════════════════════════════════════════ */

/** Prevent default on the canvas area to allow drops */
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
 * Route a dropped/selected File to the correct handler.
 * Supports: image files → loadImageOnly, .json files → loadPayload
 */
function _processFile(file) {
  if (!file) return;

  if (file.type === 'application/json' || file.name.endsWith('.json')) {
    // Read JSON file
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const payload = JSON.parse(e.target.result);
        loadPayload(payload);
      } catch (err) {
        Toast.show('JSON parse error: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);

  } else if (file.type.startsWith('image/')) {
    // Read image file as data URI
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target.result;
      const img = new Image();
      img.onload = () => loadImageOnly(src, img.naturalWidth, img.naturalHeight);
      img.src = src;
    };
    reader.readAsDataURL(file);

  } else {
    Toast.show('Unsupported file type: ' + file.type, 'error');
  }
}


/* ══════════════════════════════════════════════════════════════════════
   STATUS BAR HELPERS
══════════════════════════════════════════════════════════════════════ */
function _setStatusBusy(msg) {
  const dot = document.getElementById('dot-backend');
  const lbl = document.getElementById('label-backend');
  if (dot) { dot.className = 'status-dot amber'; }
  if (lbl) lbl.textContent = msg;
}

function _setStatusOk(msg) {
  const dot = document.getElementById('dot-backend');
  const lbl = document.getElementById('label-backend');
  if (dot) { dot.className = 'status-dot green'; }
  if (lbl) lbl.textContent = msg || 'Ready';
}

function _setStatusError(msg) {
  const dot = document.getElementById('dot-backend');
  const lbl = document.getElementById('label-backend');
  if (dot) { dot.className = 'status-dot red'; }
  if (lbl) lbl.textContent = msg;
}

function _showProgress(pct) {
  const track = document.getElementById('progress-track');
  const fill  = document.getElementById('progress-fill');
  if (!track || !fill) return;
  if (pct <= 0) {
    track.style.display = 'none';
    fill.style.width    = '0%';
  } else {
    track.style.display = 'block';
    fill.style.width    = pct + '%';
  }
}


/* ══════════════════════════════════════════════════════════════════════
   KEYBOARD SHORTCUTS
══════════════════════════════════════════════════════════════════════ */
document.addEventListener('keydown', (e) => {
  const ctrl = e.ctrlKey || e.metaKey;

  // Ctrl+Z — undo
  if (ctrl && !e.shiftKey && e.key === 'z') {
    // Only if not inside a text field
    if (document.activeElement.classList.contains('txt-block')) return;
    e.preventDefault();
    EditorState.undo();
    return;
  }

  // Ctrl+Shift+Z / Ctrl+Y — redo
  if (ctrl && (e.shiftKey && e.key === 'z' || e.key === 'y')) {
    if (document.activeElement.classList.contains('txt-block')) return;
    e.preventDefault();
    EditorState.redo();
    return;
  }

  // Ctrl+O — open modal
  if (ctrl && e.key === 'o') {
    e.preventDefault();
    openModal();
    return;
  }

  // Ctrl+S / Ctrl+E — export
  if (ctrl && (e.key === 's' || e.key === 'e')) {
    e.preventDefault();
    ExportEngine.download();
    return;
  }

  // Escape — close modal or deselect block
  if (e.key === 'Escape') {
    if (document.getElementById('modal-overlay').classList.contains('open')) {
      closeModal();
    } else {
      OverlayEngine.clearActive();
    }
    return;
  }

  // +/- zoom
  if (ctrl && (e.key === '=' || e.key === '+')) {
    e.preventDefault();
    CanvasView.zoom(0.1);
  }
  if (ctrl && e.key === '-') {
    e.preventDefault();
    CanvasView.zoom(-0.1);
  }
  if (ctrl && e.key === '0') {
    e.preventDefault();
    CanvasView.fitToWindow();
  }
});


/* ══════════════════════════════════════════════════════════════════════
   RESIZE OBSERVER — recomputes scale whenever canvas-area or
   workspace changes size (window resize, panel collapse, etc.)
══════════════════════════════════════════════════════════════════════ */
(() => {
  const ro = new ResizeObserver(() => {
    if (AppState.isLoaded) {
      // Debounce to avoid thundering-herd on rapid resize
      clearTimeout(window._resizeTimer);
      window._resizeTimer = setTimeout(() => {
        ScaleEngine.recompute();
      }, 60);
    }
  });

  ro.observe(document.getElementById('canvas-area'));
})();


/* ══════════════════════════════════════════════════════════════════════
   INIT — run once on page load
══════════════════════════════════════════════════════════════════════ */
(function init() {
  console.info('[PixelScribe] Editor initialised. Press Ctrl+O to open a payload.');

  // Expose shortcuts hint in console
  console.info(
    '%cKeyboard shortcuts:\n' +
    '  Ctrl+O  — Open file / JSON\n' +
    '  Ctrl+S  — Export PNG\n' +
    '  Ctrl+Z  — Undo\n' +
    '  Ctrl+Shift+Z — Redo\n' +
    '  Ctrl++/-  — Zoom\n' +
    '  Ctrl+0  — Fit to window\n' +
    '  Escape  — Deselect / close modal',
    'color:#9B95FF;font-family:monospace;font-size:11px'
  );

  // Wire canvas-area click to deselect blocks
  document.getElementById('canvas-area').addEventListener('click', (e) => {
    if (e.target.id === 'canvas-area' || e.target.id === 'workspace' || e.target.id === 'canvas-img') {
      OverlayEngine.clearActive();
    }
  });

  // ── DEMO: auto-load a sample payload if ?demo=1 in URL ────────────
  const params = new URLSearchParams(window.location.search);
  if (params.get('demo') === '1') {
    _loadDemoPayload();
  }
})();


/* ══════════════════════════════════════════════════════════════════════
   DEMO PAYLOAD
   Loads a programmatically generated canvas with sample text blocks
   (uses a solid-colour canvas rendered via the browser's native 2D API).
   Activated via ?demo=1 in the URL.
══════════════════════════════════════════════════════════════════════ */
function _loadDemoPayload() {
  // Generate a demo background image as a data URI
  const demoCanvas = document.createElement('canvas');
  demoCanvas.width  = 1200;
  demoCanvas.height = 800;
  const ctx = demoCanvas.getContext('2d');

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, 1200, 800);
  grad.addColorStop(0, '#F8F4EF');
  grad.addColorStop(1, '#EDE8DC');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1200, 800);

  // Subtle grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.04)';
  ctx.lineWidth = 1;
  for (let x = 0; x < 1200; x += 60) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 800); ctx.stroke();
  }
  for (let y = 0; y < 800; y += 60) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1200, y); ctx.stroke();
  }

  // Card rectangle
  ctx.fillStyle = '#FFFFFF';
  ctx.shadowColor = 'rgba(0,0,0,0.1)';
  ctx.shadowBlur = 20;
  ctx.shadowOffsetY = 4;
  _roundRect(ctx, 80, 60, 1040, 680, 12);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;

  // Accent bar
  ctx.fillStyle = '#6C63FF';
  _roundRect(ctx, 80, 60, 1040, 8, { tl: 12, tr: 12, br: 0, bl: 0 });
  ctx.fill();

  const bgSrc = demoCanvas.toDataURL('image/jpeg', 0.92);

  const demoPayload = {
    bg_image: bgSrc,
    image_w:  1200,
    image_h:  800,
    blocks: [
      {
        text: 'PixelScribe Editor',
        x: 120, y: 100, w: 600, h: 64,
        color: '#1A1A2E',
        bg_color: 'transparent',
        size: 48,
        font_family: 'Georgia',
        confidence: 0.99
      },
      {
        text: 'Offline content-aware text editing — no cloud, no subscriptions.',
        x: 120, y: 180, w: 800, h: 40,
        color: '#4A4A62',
        bg_color: 'transparent',
        size: 22,
        font_family: 'Arial',
        confidence: 0.97
      },
      {
        text: 'Edit this text. Click any block to select it, then type.',
        x: 120, y: 240, w: 720, h: 32,
        color: '#6C63FF',
        bg_color: 'transparent',
        size: 16,
        font_family: 'Courier New',
        confidence: 0.95
      },
      {
        text: 'Font: Times New Roman · Detected with 94.2% confidence',
        x: 120, y: 290, w: 700, h: 28,
        color: '#2A2A38',
        bg_color: 'transparent',
        size: 15,
        font_family: 'Times New Roman',
        confidence: 0.942
      },
      {
        text: 'Use the right panel to change font, size, and colour.',
        x: 120, y: 340, w: 680, h: 26,
        color: '#444',
        bg_color: 'transparent',
        size: 14,
        font_family: 'Calibri',
        confidence: 0.91
      },
      {
        text: 'Export PNG reconstructs full-res composite locally.',
        x: 120, y: 385, w: 660, h: 26,
        color: '#555',
        bg_color: 'transparent',
        size: 14,
        font_family: 'Verdana',
        confidence: 0.88
      },
      {
        text: 'Roboto — loaded via Google Fonts for cross-platform support.',
        x: 120, y: 430, w: 700, h: 26,
        color: '#3A3A48',
        bg_color: 'transparent',
        size: 14,
        font_family: 'Roboto',
        confidence: 0.93
      },
      {
        text: 'Helvetica Neue renders beautifully on macOS and Windows.',
        x: 120, y: 475, w: 700, h: 26,
        color: '#2E2E3E',
        bg_color: 'transparent',
        size: 14,
        font_family: 'Helvetica',
        confidence: 0.90
      },
      {
        text: 'Garamond — elegant serif typeface for body text.',
        x: 120, y: 520, w: 660, h: 26,
        color: '#3E3024',
        bg_color: 'transparent',
        size: 15,
        font_family: 'Garamond',
        confidence: 0.87
      },
      {
        text: 'Consolas: monospaced → code & technical content.',
        x: 120, y: 570, w: 660, h: 26,
        color: '#6B6B88',
        bg_color: 'transparent',
        size: 14,
        font_family: 'Consolas',
        confidence: 0.85
      }
    ]
  };

  loadPayload(demoPayload);
  Toast.show('Demo payload loaded — try editing any block!', 'info', 4000);
}

/** Helper: draw a rounded rectangle path */
function _roundRect(ctx, x, y, w, h, r) {
  if (typeof r === 'number') r = { tl: r, tr: r, br: r, bl: r };
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