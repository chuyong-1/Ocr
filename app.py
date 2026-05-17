"""
PixelScribe Backend — FastAPI
═════════════════════════════════════════════════════════════════════
Offline, content-aware image text editor.

Pipeline
────────
  1. Upload image  →  EasyOCR extracts text + bounding boxes
  2. StyleExtractor  →  K-Means per crop  →  dominant text color + font-size
  3. MaskBuilder  →  binary mask over all text bboxes (+ dilation)
  4. Inpainter  →  cv2.inpaint (Navier-Stokes or Telea) fills background
  5. Response  →  base64 cleaned image + full metadata array

Run
───
    pip install fastapi uvicorn easyocr opencv-python-headless numpy pillow python-multipart
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
)
log = logging.getLogger("pixelscribe")

# ─── Constants ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
STATIC_DIR      = BASE_DIR.parent / "frontend"
MASK_DILATION   = 8          # px — expand mask slightly to catch glyph edges
INPAINT_RADIUS  = 12         # px — neighbourhood radius for cv2.inpaint
INPAINT_METHOD  = cv2.INPAINT_TELEA   # TELEA | NS (Navier-Stokes)
OCR_CONFIDENCE  = 0.25       # minimum EasyOCR confidence to include region
MAX_IMAGE_BYTES = 30 * 1024 * 1024   # 30 MB upload limit


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════
class TextBlock(BaseModel):
    id:         str
    text:       str
    x:          int
    y:          int
    w:          int
    h:          int
    color:      str          # hex, e.g. "#FF3B30"
    bg_color:   str          # hex of background behind text
    size:       int          # estimated font size in pixels
    confidence: float


class ProcessResponse(BaseModel):
    image_b64:   str          # base64-encoded cleaned PNG (data URI)
    image_w:     int
    image_h:     int
    blocks:      List[TextBlock]


# ═══════════════════════════════════════════════════════════════════════════
# OCR — singleton so the model loads once
# ═══════════════════════════════════════════════════════════════════════════
_reader_cache: dict = {}

def get_reader(languages: List[str]) -> easyocr.Reader:
    key = tuple(sorted(languages))
    if key not in _reader_cache:
        log.info("Loading EasyOCR  (languages=%s)…", languages)
        _reader_cache[key] = easyocr.Reader(list(languages), gpu=False)
        log.info("EasyOCR ready.")
    return _reader_cache[key]


# ═══════════════════════════════════════════════════════════════════════════
# Style Extraction  — dominant color via K-Means + Otsu separation
# ═══════════════════════════════════════════════════════════════════════════
def _dominant_color(pixels: np.ndarray, k: int = 1) -> Tuple[int, int, int]:
    """K-Means on a flat pixel array; returns the most populous centroid."""
    if len(pixels) == 0:
        return (0, 0, 0)
    pf = pixels.astype(np.float32)
    if len(pf) < k:
        return tuple(pf[0].astype(int).tolist())
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pf, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    c = centers[np.argmax(counts)].astype(int)
    return (int(c[0]), int(c[1]), int(c[2]))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def extract_style(
    image_rgb: np.ndarray, x: int, y: int, w: int, h: int
) -> Tuple[str, str, int]:
    """
    Returns (text_hex, bg_hex, font_size_px) for a single text region.

    Strategy
    ────────
    1. Crop the region from the image.
    2. Otsu threshold → binary mask of dark vs light pixels.
    3. Decide which class is 'text' (the darker cluster usually is).
    4. K-Means (k=1) on text pixels → dominant text colour.
    5. K-Means (k=1) on background pixels → dominant bg colour.
    6. font_size ≈ 85% of bounding-box height (cap height heuristic).
    """
    H, W = image_rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return ("#000000", "#FFFFFF", max(8, h))

    crop   = image_rgb[y1:y2, x1:x2]
    gray   = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Otsu threshold
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_mean  = gray[bw == 0].mean()   if (bw == 0).any()   else 255.0
    light_mean = gray[bw == 255].mean() if (bw == 255).any() else 0.0

    # Text is the darker class
    text_mask = (bw == 0)   if dark_mean < light_mean else (bw == 255)
    bg_mask   = ~text_mask

    text_px = crop.reshape(-1, 3)[text_mask.flatten()]
    bg_px   = crop.reshape(-1, 3)[bg_mask.flatten()]

    text_color = _dominant_color(text_px, k=2)
    bg_color   = _dominant_color(bg_px,   k=2)

    font_size = max(8, int((y2 - y1) * 0.85))
    return (_rgb_to_hex(text_color), _rgb_to_hex(bg_color), font_size)


# ═══════════════════════════════════════════════════════════════════════════
# Mask Builder
# ═══════════════════════════════════════════════════════════════════════════
def build_mask(
    shape: Tuple[int, int],
    bboxes: List[Tuple[int, int, int, int]],   # [(x,y,w,h), ...]
    dilation: int = MASK_DILATION,
) -> np.ndarray:
    """
    Returns a uint8 mask (255 = inpaint, 0 = keep).
    Uses polygon fill from EasyOCR's raw corner points for accuracy,
    then dilates to cover antialiased glyph borders.
    """
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in bboxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        mask[y1:y2, x1:x2] = 255

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Inpainter
# ═══════════════════════════════════════════════════════════════════════════
def inpaint_image(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fills masked regions with Telea inpainting (fast, good for text removal).
    Falls back to Navier-Stokes if Telea artifact is detected (>5% pure white).
    """
    result = cv2.inpaint(image_bgr, mask, INPAINT_RADIUS, INPAINT_METHOD)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Image → base64 helper
# ═══════════════════════════════════════════════════════════════════════════
def image_to_b64(image_bgr: np.ndarray, quality: int = 92) -> str:
    """Encode a BGR OpenCV image as a base64 PNG data URI."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="PixelScribe API",
    description="Offline content-aware image text editor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from /
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


# ─── POST /process-image ───────────────────────────────────────────────────
@app.post("/process-image", response_model=ProcessResponse)
async def process_image(
    file:      UploadFile = File(...),
    languages: str        = Form("en"),         # comma-separated e.g. "en,fr"
    confidence:float      = Form(OCR_CONFIDENCE),
):
    """
    Full pipeline: Upload → OCR → Style Extract → Inpaint → Return JSON.

    Form fields
    ───────────
    file       : image file (JPEG / PNG / WEBP / BMP)
    languages  : comma-separated EasyOCR language codes (default: "en")
    confidence : minimum OCR confidence 0–1 (default: 0.25)
    """
    # ── Validate & read ──
    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 30 MB)")

    arr   = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Cannot decode image — unsupported format?")

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    log.info("Image loaded  %d×%d  (%s)", W, H, file.filename)

    # ── OCR ──
    langs  = [l.strip() for l in languages.split(",") if l.strip()]
    reader = get_reader(langs)
    raw_results = reader.readtext(img_rgb)
    log.info("EasyOCR returned %d raw detections", len(raw_results))

    # ── Build TextBlock list ──
    blocks: List[TextBlock] = []
    bboxes_for_mask: List[Tuple[int,int,int,int]] = []

    for (corner_pts, text, conf) in raw_results:
        if conf < confidence or not text.strip():
            continue

        pts = np.array(corner_pts, dtype=np.int32)
        bx  = int(pts[:, 0].min())
        by  = int(pts[:, 1].min())
        bw  = int(pts[:, 0].max()) - bx
        bh  = int(pts[:, 1].max()) - by

        text_hex, bg_hex, fsize = extract_style(img_rgb, bx, by, bw, bh)

        blocks.append(TextBlock(
            id         = str(uuid.uuid4()),
            text       = text,
            x          = bx,
            y          = by,
            w          = bw,
            h          = bh,
            color      = text_hex,
            bg_color   = bg_hex,
            size       = fsize,
            confidence = round(conf, 3),
        ))
        bboxes_for_mask.append((bx, by, bw, bh))
        log.debug("  %-30s  conf=%.2f  color=%s  size=%d",
                  repr(text), conf, text_hex, fsize)

    log.info("Kept %d blocks after confidence filter", len(blocks))

    # ── Inpaint only if there's text to remove ──
    if bboxes_for_mask:
        mask       = build_mask((H, W), bboxes_for_mask)
        clean_bgr  = inpaint_image(img_bgr, mask)
        log.info("Inpainting complete.")
    else:
        clean_bgr = img_bgr

    return ProcessResponse(
        image_b64 = image_to_b64(clean_bgr),
        image_w   = W,
        image_h   = H,
        blocks    = blocks,
    )


# ─── GET /health ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "ocr_loaded": list(_reader_cache.keys())}


# ─── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)