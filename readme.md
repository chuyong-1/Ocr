# PixelScribe — Offline Content-Aware Image Text Editor

Remove and replace text in images. OCR, background inpainting and font
classification all run **in your browser** via WebAssembly — no cloud APIs,
no subscriptions, no server required.

```
┌────────────────────────────────────────────────────────────────────┐
│  Import image/PDF  →  Tesseract detects text  →  OpenCV erases it  │
│  →  Editable fields overlay the cleaned image  →  Export PNG       │
└────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
npm start
```

Then open **http://localhost:5173** — or **http://localhost:5173/?demo=1** to
load a sample document without importing anything.

That's it. No Python, no build step, no network access. Node 18+ is the only
requirement, and only to serve the files.

> The app must be served over `http://`, not opened as a `file://` URL.
> Web Workers, `WebAssembly.instantiateStreaming` and the vendor-asset probes
> are all blocked on `file://`.

Verify the vendored WASM assets are complete and self-consistent:

```bash
npm run check
```

---

## Project Structure

```
Ocr/
├── www/                        ← the application (this is what ships)
│   ├── index.html              ← UI, layout and styles
│   ├── main.js                 ← editor logic + edge-AI orchestration
│   ├── models/
│   │   └── font_classifier.onnx     ← 5-class font CNN (canonical copy)
│   ├── tessdata/
│   │   └── eng.traineddata          ← OCR language data (uncompressed)
│   └── vendor/                 ← pinned offline copies of every library
│       ├── onnx/               ← ONNX Runtime Web 1.26.0 (wasm-only)
│       ├── opencv/             ← OpenCV.js (inpainting)
│       ├── pdfjs/              ← PDF.js (PDF → canvas)
│       └── tesseract/          ← Tesseract.js 4.x (OCR)
│
├── server.js                   ← static dev server (correct MIME + HEAD)
├── scripts/check-assets.js     ← vendor integrity check (`npm run check`)
├── Start                       ← bash launcher; `--backend` adds the API
│
├── backend/                    ← OPTIONAL Python server (see below)
│   ├── app.py                  ← single-file FastAPI OCR service
│   ├── server.py               ← jobs API + WebSockets
│   ├── worker.py               ← Celery worker
│   ├── text_pipeline.py        ← EasyOCR + inpaint + font pipeline
│   ├── langgraph_pipeline.py   ← LangGraph orchestration
│   └── train_font_classifier.py ← trains www/models/font_classifier.onnx
│
├── data/                       ← uploads, results, SQLite jobs.db
├── docker-compose.yml          ← full stack: api + worker + redis + nginx
└── capacitor.config.json       ← Android/iOS wrapper (webDir: www)
```

---

## How It Works

Everything below runs client-side. The browser never sends your image anywhere.

| Stage | Engine | Notes |
|-------|--------|-------|
| **1. Import** | PDF.js / native decode | PDF page 1 rendered at ~300 DPI; JPG, PNG, WEBP, BMP, GIF decoded directly |
| **2. OCR** | native `TextDetector` → Tesseract.js 4 | Line-level boxes + confidence. Images over 2000 px are downscaled for OCR, then coordinates scaled back |
| **3. Font ID** | ONNX Runtime Web | 64×64 grayscale crop per block → font CNN. Only runs when the source didn't already name a font |
| **4. Erase** | OpenCV.js `inpaint` (Telea) | Double-click a block to erase the original glyphs behind it |
| **5. Edit** | `contenteditable` overlay | Absolutely positioned over the image, rescaled on resize/zoom |
| **6. Export** | off-screen canvas | Flattened at native resolution, downloaded as PNG |

### Offline guarantees

Every library is vendored under `www/vendor/` and loaded from disk first; the
CDN fallbacks exist purely as a convenience for when a vendor file is absent.
With the network fully blocked the app still boots all three engines, runs
OCR, classifies fonts, inpaints and exports.

The only external request is the Google Fonts stylesheet, which is decorative
— every rule carries a system-font fallback, so blocking it changes nothing
but the typeface of the UI chrome.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+O` | Import |
| `Ctrl+S` / `Ctrl+E` | Export PNG |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| `Ctrl` `+` / `-` / `0` | Zoom in / out / fit |
| `Esc` | Close modal, or deselect the current block |

---

## JSON Payload Format

You can skip OCR entirely and import a payload directly (the "JSON" tab, or
drop a `.json` file). This is also the shape the Python backend returns.

```json
{
  "bg_image": "data:image/png;base64,…",
  "image_w": 1920,
  "image_h": 1080,
  "blocks": [
    {
      "text": "Hello World",
      "x": 120, "y": 45, "w": 280, "h": 38,
      "color": "#2C2C2C",
      "bg_color": "transparent",
      "size": 32,
      "font_family": "Georgia",
      "confidence": 0.987
    }
  ]
}
```

`font_family` is optional — omit it and the ONNX classifier predicts one.
Supply it and it is treated as authoritative, skipping the classifier.

---

## Optional Python Backend

The browser pipeline is self-sufficient. The Python backend exists for heavier
server-side OCR (EasyOCR, 80+ languages, GPU) when you want it.

```bash
./Start --backend          # web app on :5173 + API on :8000
```

Then in the app: **OCR Server** (or **More → OCR Server** on mobile) → enter
`http://localhost:8000`. The app uses it only as a fallback when in-browser
OCR fails.

### `POST /process-image`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | File | — | Image (JPEG/PNG/WEBP/BMP) |
| `languages` | string | `"en"` | Comma-separated EasyOCR codes |
| `confidence` | float | `0.25` | Minimum OCR confidence |

- `GET /health` → `{ "status": "ok", "ocr_loaded": [["en"]] }`
- `GET /docs` → Swagger UI

### Full stack (Celery + Redis + nginx)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| App (nginx) | http://localhost:3000 |
| API | http://localhost:8000 |
| Flower | http://localhost:5555 |

---

## Mobile (Capacitor)

`capacitor.config.json` already points `webDir` at `www`, so the same offline
build ships to Android:

```bash
npm install
npx cap add android
npm run sync
npm run android
```

---

## Retraining the Font Classifier

```bash
python backend/train_font_classifier.py
```

Writes to `www/models/font_classifier.onnx` — the same file the browser loads,
so there is only ever one copy to keep in sync.

The shipped checkpoint has **5 classes** (Arial, Times New Roman, Courier New,
Calibri, Georgia). `FONT_LABELS` in `www/main.js` lists ten names; the code
reads the real class count off the model and uses only as many labels as the
model actually emits, and `npm run check` reports the mismatch. If you retrain
with more classes, extend `FONT_LABELS` to match the training script's list,
in the same order.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Badges stuck on "loading" | Opened as `file://`. Use `npm start`. |
| "ONNX unavailable" | Loader/wasm version mismatch. Run `npm run check`. |
| "OCR unavailable" offline | Incomplete `www/vendor/tesseract/`, or missing `www/tessdata/eng.traineddata`. Run `npm run check`. |
| OCR finds nothing | Low contrast or very small text. The image is downscaled to 2000 px for OCR — try cropping tighter. |
| Export differs from preview | Both use the same geometry; if they diverge, the block's font isn't installed locally and the canvas fell back to another face. |
| Port 5173 in use | `node server.js --port 5174` |

---

## Requirements

- **Web app:** Node 18+ (to serve files) and any modern browser. ~50 MB of vendored WASM.
- **Optional backend:** Python 3.10+, ~500 MB disk (EasyOCR downloads ~200 MB per language on first use).

---

## License

MIT — use freely in personal and commercial projects.
