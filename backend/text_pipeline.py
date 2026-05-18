"""
╔══════════════════════════════════════════════════════════════════╗
║        Text Removal & Replacement Pipeline  (Image / Video)      ║
║  Stack: EasyOCR · LaMa · Stable Diffusion Inpaint · PIL/OpenCV   ║
║  ENHANCED: 10-Class Font Classifier (ONNX) Integration           ║
╠══════════════════════════════════════════════════════════════════╣
║  v3 — Font Classification additions                              ║
║    • FontClassifier class (10-class ONNX model)                  ║
║    • EditorBlock enriched with 'font_family' field               ║
║    • extract_for_editor() decorated with font predictions        ║
║    • Graceful fallback to "sans-serif" on model failure         ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, TypedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 1. FONT CLASSIFIER — 10-CLASS ONNX MODEL
# ─────────────────────────────────────────────────────────────────
class FontClassifier:
    """
    Inference wrapper for 10-class font family ONNX model.
    
    Supported fonts (0-9):
      0: Arial
      1: Times New Roman
      2: Courier New
      3: Calibri
      4: Georgia
      5: Verdana
      6: Roboto
      7: Helvetica
      8: Garamond
      9: Consolas
    
    Model Input:  64×64 grayscale float32 image [0, 1]
    Model Output: [1, 10] logits → argmax → class index
    
    Thread-safe inference with lazy loading. Gracefully degrades to
    "sans-serif" if model unavailable (offline resilience).
    """

    # ── Index-to-name mapping (EXACT ORDER CRITICAL) ──
    DEFAULT_LABELS = [
        "Arial",              # 0
        "Times New Roman",    # 1
        "Courier New",        # 2
        "Calibri",            # 3
        "Georgia",            # 4
        "Verdana",            # 5
        "Roboto",             # 6
        "Helvetica",          # 7
        "Garamond",           # 8
        "Consolas",           # 9
    ]

    def __init__(self, model_path: Optional[str] = None, gpu: bool = False):
        """
        Initialize FontClassifier.
        
        Parameters
        ----------
        model_path : str, optional
            Path to ONNX model file. If None, attempts auto-discovery
            in common locations. Lazy-loaded on first predict() call.
        gpu : bool
            If True, attempt to use CUDA/GPU for inference.
        """
        self.model_path    = model_path
        self.gpu           = gpu
        self._session      = None
        self._input_name   = None
        self._output_name  = None
        self._initialized  = False
        self._init_error   = None

        log.info("FontClassifier initialized (model_path=%s, gpu=%s)",
                 model_path, gpu)

    def _lazy_load(self) -> bool:
        """
        Lazy-load ONNX runtime and model on first inference.
        Returns True if successful, False if model unavailable.
        """
        if self._initialized:
            return self._session is not None

        try:
            import onnxruntime as ort
        except ImportError:
            self._init_error = (
                "onnxruntime not installed. "
                "Install: pip install onnxruntime"
            )
            log.warning(self._init_error)
            self._initialized = True
            return False

        # ── Resolve model path ──
        model_file = self.model_path
        if not model_file:
            candidates = [
                "models/font_classifier.onnx",
                "./font_classifier.onnx",
                "/app/models/font_classifier.onnx",  # Docker path
                str(Path(__file__).parent.parent / "models" / "font_classifier.onnx"),
            ]
            for cand in candidates:
                if Path(cand).exists():
                    model_file = cand
                    break

        if not model_file or not Path(model_file).exists():
            self._init_error = (
                f"Font classifier ONNX model not found. "
                f"Searched: {candidates}"
            )
            log.warning(self._init_error)
            self._initialized = True
            return False

        try:
            # ── Session options for provider selection ──
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3  # Suppress verbose logs

            # ── Provider selection: GPU > CPU ──
            providers = []
            if self.gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            self._session = ort.InferenceSession(
                model_file,
                sess_opts=sess_opts,
                providers=providers,
            )

            # ── Inspect I/O names ──
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            log.info(
                "FontClassifier model loaded (%s) — "
                "input: %s, output: %s, provider: %s",
                model_file,
                self._input_name,
                self._output_name,
                self._session.get_providers()[0] if self._session.get_providers() else "unknown",
            )
            self._initialized = True
            return True

        except Exception as e:
            self._init_error = f"Failed to load ONNX model: {e}"
            log.warning(self._init_error)
            self._initialized = True
            return False

    def predict(self, image_crop: np.ndarray) -> str:
        """
        Predict font family for a text region crop.
        
        Parameters
        ----------
        image_crop : np.ndarray
            Cropped text region (BGR or RGB, any size).
        
        Returns
        -------
        str
            Font family name from DEFAULT_LABELS, or "sans-serif" fallback.
        """
        try:
            # ── Lazy load on first call ──
            if not self._lazy_load():
                log.debug("Font classifier unavailable, using fallback")
                return "sans-serif"

            if self._session is None:
                return "sans-serif"

            # ── Preprocess: resize → grayscale → normalize ──
            try:
                # Handle both BGR (OpenCV) and RGB (PIL) gracefully
                if len(image_crop.shape) == 3:
                    # Convert BGR → RGB → grayscale if needed
                    if image_crop.shape[2] == 3:
                        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
                    elif image_crop.shape[2] == 4:
                        # BGRA → BGR → grayscale
                        gray = cv2.cvtColor(
                            image_crop[:, :, :3], cv2.COLOR_BGR2GRAY
                        )
                    else:
                        gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_crop

                # Resize to model input shape (64×64)
                resized = cv2.resize(gray, (64, 64),
                                    interpolation=cv2.INTER_LANCZOS4)

                # Normalize to [0, 1] float32
                normalized = resized.astype(np.float32) / 255.0

                # Add batch dimension: (64, 64) → (1, 1, 64, 64)
                # Adjust to model's expected input shape if needed
                # Common: [batch, channels, height, width]
                batch = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)

                # ── Inference ──
                outputs = self._session.run(
                    [self._output_name],
                    {self._input_name: batch},
                )

                # ── Parse output ──
                logits = outputs[0]  # Shape: [1, 10]

                if logits.shape[1] != len(self.DEFAULT_LABELS):
                    log.warning(
                        "Model output shape mismatch: expected %d classes, got %d. "
                        "Using fallback.",
                        len(self.DEFAULT_LABELS),
                        logits.shape[1],
                    )
                    return "sans-serif"

                class_idx = int(np.argmax(logits[0]))

                # ── Bounds check (safety) ──
                if class_idx < 0 or class_idx >= len(self.DEFAULT_LABELS):
                    log.warning(
                        "Class index out of bounds: %d (expected 0-%d). "
                        "Using fallback.",
                        class_idx,
                        len(self.DEFAULT_LABELS) - 1,
                    )
                    return "sans-serif"

                font_name = self.DEFAULT_LABELS[class_idx]
                confidence = float(np.max(np.exp(logits[0]) / np.sum(np.exp(logits[0]))))

                log.debug(
                    "Font prediction: %s (confidence: %.3f)",
                    font_name,
                    confidence,
                )

                return font_name

            except Exception as e:
                log.warning(
                    "Error during font prediction preprocessing: %s. "
                    "Using fallback.",
                    e,
                )
                return "sans-serif"

        except Exception as e:
            log.warning(
                "Unexpected error in FontClassifier.predict(): %s. "
                "Using fallback.",
                e,
            )
            return "sans-serif"


# ─────────────────────────────────────────────────────────────────
# 1-A. EDITOR BLOCK TYPE WITH FONT FAMILY
# ─────────────────────────────────────────────────────────────────

class EditorBlock(TypedDict):
    """
    Serialisable dict for one text region returned by extract_for_editor().

    ENHANCED v3: Includes 'font_family' field for 10-class classification.
    
    Designed to be JSON-dumped directly into the metadata file that
    server.py / worker.py write to results/.
    """
    text:        str
    x:           int
    y:           int
    w:           int
    h:           int
    color:       str       # dominant text colour as CSS hex  e.g. "#2C2C2C"
    bg_color:    str       # dominant background colour       e.g. "#F5F0EB"
    size:        int       # estimated font size in CSS px
    confidence:  float     # EasyOCR confidence 0-1
    font_family: str       # predicted font family name, e.g. "Arial"


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an (R, G, B) tuple to a CSS hex string like '#1A2B3C'."""
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ─────────────────────────────────────────────────────────────────
# 2. DATA CLASSES   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
@dataclass
class TextRegion:
    """Holds everything we know about one detected text region."""
    bbox: List[List[int]]          # EasyOCR format: 4 corner points
    text: str
    confidence: float

    # Axis-aligned bounding box (derived)
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    # Style
    font_size:  int = 20
    text_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color:   Tuple[int, int, int] = (255, 255, 255)
    font_family: str = "sans-serif"  # NEW: predicted font family

    def __post_init__(self):
        pts = np.array(self.bbox, dtype=np.int32)
        self.x = int(pts[:, 0].min())
        self.y = int(pts[:, 1].min())
        self.w = int(pts[:, 0].max()) - self.x
        self.h = int(pts[:, 1].max()) - self.y


# ─────────────────────────────────────────────────────────────────
# 3. TEXT DETECTION   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class TextDetector:
    """
    Wraps EasyOCR for text detection.

    Alternative: swap for PaddleOCR by replacing the detect() method —
    PaddleOCR gives better results on CJK / vertical text.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        import easyocr
        langs = languages or ["en"]
        log.info("Loading EasyOCR (languages=%s, gpu=%s)…", langs, gpu)
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def detect(self, image: np.ndarray,
               confidence_threshold: float = 0.4) -> List[TextRegion]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb)
        regions: List[TextRegion] = []
        for bbox, text, conf in results:
            if conf < confidence_threshold:
                continue
            regions.append(TextRegion(bbox=bbox, text=text, confidence=conf))
            log.debug("  Detected %-30s  conf=%.2f", repr(text), conf)
        log.info("Detected %d text region(s).", len(regions))
        return regions


# ─────────────────────────────────────────────────────────────────
# 4. STYLE EXTRACTION   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class StyleExtractor:
    """
    Estimates font size, dominant text colour, and background colour
    for each TextRegion directly from pixel data.
    """

    @staticmethod
    def dominant_color(pixels: np.ndarray, k: int = 2) -> np.ndarray:
        """K-Means on a pixel array; returns the centroid with most votes."""
        if len(pixels) == 0:
            return np.array([0, 0, 0])
        pixels_f = pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels_f, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        counts = np.bincount(labels.flatten())
        return centers[np.argmax(counts)].astype(np.uint8)

    def extract(self, image: np.ndarray, region: TextRegion) -> TextRegion:
        """
        Populates region.font_size, region.text_color, region.bg_color.
        Works on the cropped bounding-box patch.
        """
        x, y, w, h = region.x, region.y, region.w, region.h
        H, W = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return region

        crop_bgr  = image[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # Otsu threshold to separate text pixels from background
        _, mask = cv2.threshold(crop_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_mean  = crop_gray[mask == 0].mean()   if (mask == 0).any()   else 255
        light_mean = crop_gray[mask == 255].mean() if (mask == 255).any() else 0
        if dark_mean < light_mean:
            text_mask = (mask == 0)
            bg_mask   = (mask == 255)
        else:
            text_mask = (mask == 255)
            bg_mask   = (mask == 0)

        text_pixels = crop_rgb[text_mask]
        bg_pixels   = crop_rgb[bg_mask]

        region.text_color = tuple(
            self.dominant_color(text_pixels, k=1).tolist()
        )
        region.bg_color = tuple(
            self.dominant_color(bg_pixels, k=1).tolist()
        )

        # Font-size estimate: bounding-box height ≈ cap height
        region.font_size = max(8, int(h * 0.85))

        log.debug("  Style → font_size=%d  text_color=%s  bg_color=%s",
                  region.font_size, region.text_color, region.bg_color)
        return region


# ─────────────────────────────────────────────────────────────────
# 5. MASK GENERATION   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class MaskGenerator:
    """
    Builds a binary mask (255 = region to inpaint, 0 = keep).
    A slight dilation ensures we erase every pixel of the glyph border.
    """

    def __init__(self, dilation_px: int = 6):
        self.dilation_px = dilation_px

    def generate(self, image_shape: Tuple[int, int],
                 regions: List[TextRegion]) -> np.ndarray:
        H, W = image_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for r in regions:
            pts = np.array(r.bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        if self.dilation_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_px * 2 + 1, self.dilation_px * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask


# ─────────────────────────────────────────────────────────────────
# 6. INPAINTING BACKENDS (cv2, LaMa, SD) — unchanged from v2
# ─────────────────────────────────────────────────────────────────
class CvInpainter:
    """Fast offline inpainting using OpenCV's Telea algorithm."""

    def __init__(self, radius: int = 12,
                 method: int = cv2.INPAINT_TELEA):
        self.radius = radius
        self.method = method

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(image, mask, self.radius, self.method)


class LamaInpainter:
    """Uses simple-lama-inpainting package for higher quality."""

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        log.info("Loading LaMa model…")
        self.lama = SimpleLama()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pil_img  = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        result   = self.lama(pil_img, pil_mask)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


class SDInpainter:
    """Stable Diffusion inpainting for photorealistic results."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting",
                 device: str = "cuda"):
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        log.info("Loading SD inpaint pipeline on %s…", device)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.pipe.safety_checker = None

    def inpaint(self, image: np.ndarray, mask: np.ndarray,
                prompt: str = "seamless background, high quality",
                num_inference_steps: int = 30) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        TARGET = 512

        def _resize(arr, size):
            return cv2.resize(arr, (size, size),
                              interpolation=cv2.INTER_LANCZOS4)

        img_resized  = _resize(image, TARGET)
        mask_resized = _resize(mask,  TARGET)

        pil_img  = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask_resized)

        out = self.pipe(
            prompt=prompt,
            image=pil_img,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
        ).images[0]

        result_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        return cv2.resize(result_bgr, (orig_w, orig_h),
                          interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────
# 7. TEXT RENDERER   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class TextRenderer:
    """Overlays replacement text onto the inpainted image."""

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        candidates = [
            self.font_path,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            if path and Path(path).exists():
                try:
                    return ImageFont.truetype(path, size=size)
                except OSError:
                    pass
        log.warning("No TTF font found — using PIL default bitmap font.")
        return ImageFont.load_default()

    def render(self, image: np.ndarray,
               region: TextRegion, new_text: str) -> np.ndarray:
        pil  = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = self._get_font(region.font_size)

        fsize = region.font_size
        while fsize > 8:
            font = self._get_font(fsize)
            try:
                bbox = font.getbbox(new_text)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw, _ = draw.textsize(new_text, font=font)
            if tw <= region.w:
                break
            fsize -= 1

        try:
            tb = font.getbbox(new_text)
            th = tb[3] - tb[1]
        except AttributeError:
            _, th = draw.textsize(new_text, font=font)
        x_pos = region.x
        y_pos = region.y + max(0, (region.h - th) // 2)

        shadow_offset = max(1, fsize // 18)
        shadow_color  = tuple(max(0, c - 80) for c in region.text_color)
        draw.text((x_pos + shadow_offset, y_pos + shadow_offset),
                  new_text, font=font, fill=shadow_color)
        draw.text((x_pos, y_pos), new_text, font=font, fill=region.text_color)

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────
# 8. ENHANCED extract_for_editor() WITH FONT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────
def extract_for_editor(
    image_bgr:       np.ndarray,
    languages:       List[str]  = None,
    confidence:      float      = 0.40,
    dilation_px:     int        = 8,
    inpaint_radius:  int        = 12,
    gpu:             bool       = False,
    font_classifier: Optional[FontClassifier] = None,
) -> Tuple[np.ndarray, List[EditorBlock]]:
    """
    ENHANCED: Complete single-call pipeline with 10-class font prediction.

    Stages
    ──────
    1. EasyOCR detection — word-level bounding boxes + confidence
    2. StyleExtractor    — K-Means text/bg colour, font-size estimate
    3. FontClassifier    — 10-class ONNX prediction (with fallback)
    4. MaskGenerator     — dilated binary mask over all text regions
    5. CvInpainter       — cv2.inpaint (Telea) fills background cleanly
    6. Build EditorBlock list with font_family, colors, sizes

    Parameters
    ──────────
    image_bgr           OpenCV BGR image array (from cv2.imread)
    languages           EasyOCR language codes, e.g. ["en", "fr"]
    confidence          Minimum OCR confidence threshold (0–1)
    dilation_px         Mask dilation to catch antialiased glyph borders
    inpaint_radius      cv2.inpaint neighbourhood radius (px)
    gpu                 Pass True to use CUDA for EasyOCR/FontClassifier
    font_classifier     FontClassifier instance; if None, creates new

    Returns
    ───────
    cleaned_bgr         The image with all text regions inpainted away
    blocks              List[EditorBlock] — serialisable, JSON-ready
    """
    langs = languages or ["en"]

    # ── Initialize font classifier if not provided ──
    if font_classifier is None:
        font_classifier = FontClassifier(gpu=gpu)

    # ── Stage 1: OCR ────────────────────────────────────────────
    detector = TextDetector(langs, gpu=gpu)
    regions  = detector.detect(image_bgr, confidence_threshold=confidence)

    if not regions:
        log.info("No text detected — returning original image + empty blocks.")
        return image_bgr.copy(), []

    # ── Stage 2: Style per region ────────────────────────────────
    extractor = StyleExtractor()
    for r in regions:
        extractor.extract(image_bgr, r)

    # ── Stage 3: Font classification per region ─────────────────
    for r in regions:
        try:
            crop_bgr = image_bgr[r.y:r.y+r.h, r.x:r.x+r.w]
            if crop_bgr.size > 0:  # Non-empty crop
                font_name = font_classifier.predict(crop_bgr)
                r.font_family = font_name
                log.debug("Font prediction for '%s': %s", r.text, font_name)
            else:
                r.font_family = "sans-serif"
        except Exception as e:
            log.warning(
                "Error predicting font for region '%s': %s. Using fallback.",
                r.text, e
            )
            r.font_family = "sans-serif"

    # ── Stage 4: Mask ────────────────────────────────────────────
    masker = MaskGenerator(dilation_px=dilation_px)
    mask   = masker.generate(image_bgr.shape, regions)

    # ── Stage 5: Inpaint (OpenCV Telea — no model required) ──────
    inpainter  = CvInpainter(radius=inpaint_radius)
    cleaned    = inpainter.inpaint(image_bgr, mask)
    log.info("Inpainting complete  (%d region(s)).", len(regions))

    # ── Stage 6: Build EditorBlock list with fonts ───────────────
    blocks: List[EditorBlock] = []
    for r in regions:
        blocks.append(EditorBlock(
            text        = r.text,
            x           = r.x,
            y           = r.y,
            w           = r.w,
            h           = r.h,
            color       = rgb_to_hex(r.text_color),
            bg_color    = rgb_to_hex(r.bg_color),
            size        = r.font_size,
            confidence  = round(r.confidence, 4),
            font_family = r.font_family,  # NEW: 10-class prediction
        ))

    return cleaned, blocks


# ─────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE ORCHESTRATOR   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class TextPipeline:
    """
    High-level API that wires all stages together for remove/replace mode.
    (Used by the CLI and legacy job-queue path.)
    """

    def __init__(self,
                 languages:   List[str]     = None,
                 inpainter:   str           = "lama",
                 gpu:         bool          = False,
                 dilation_px: int           = 6,
                 font_path:   Optional[str] = None,
                 confidence:  float         = 0.4,
                 font_classifier_path: Optional[str] = None):

        self.detector   = TextDetector(languages, gpu=gpu)
        self.extractor  = StyleExtractor()
        self.masker     = MaskGenerator(dilation_px)
        self.renderer   = TextRenderer(font_path)
        self.confidence = confidence
        self.font_classifier = FontClassifier(font_classifier_path, gpu=gpu)

        if inpainter == "sd":
            device = "cuda" if gpu else "cpu"
            self.inpainter = SDInpainter(device=device)
        elif inpainter == "cv":
            self.inpainter = CvInpainter()
        else:
            self.inpainter = LamaInpainter()

    def process_image(
        self,
        image_bgr:       np.ndarray,
        mode:            str                 = "remove",
        replacement_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[TextRegion]]:
        regions = self.detector.detect(image_bgr, self.confidence)
        if not regions:
            log.info("No text detected — returning original image.")
            return image_bgr.copy(), regions

        for r in regions:
            self.extractor.extract(image_bgr, r)
            # ── Font classification ──
            try:
                crop_bgr = image_bgr[r.y:r.y+r.h, r.x:r.x+r.w]
                if crop_bgr.size > 0:
                    r.font_family = self.font_classifier.predict(crop_bgr)
            except Exception as e:
                log.warning("Font classification error: %s", e)
                r.font_family = "sans-serif"

        mask   = self.masker.generate(image_bgr.shape, regions)
        result = self.inpainter.inpaint(image_bgr, mask)

        if mode == "replace" and replacement_map:
            for r in regions:
                new_text = replacement_map.get(r.text) or \
                           replacement_map.get(r.text.lower())
                if new_text is not None:
                    result = self.renderer.render(result, r, new_text)

        return result, regions

    def remove_text_from_file(self, input_path: str, output_path: str):
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {input_path}")
        result, _ = self.process_image(img, mode="remove")
        cv2.imwrite(output_path, result)
        log.info("Saved → %s", output_path)

    def replace_text_in_file(self, input_path: str, output_path: str,
                              replacement_map: Dict[str, str]):
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {input_path}")
        result, _ = self.process_image(img, mode="replace",
                                       replacement_map=replacement_map)
        cv2.imwrite(output_path, result)
        log.info("Saved → %s", output_path)


# ─────────────────────────────────────────────────────────────────
# 10. VIDEO PROCESSING   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Processes a video file frame-by-frame using the TextPipeline.
    """

    def __init__(self, pipeline: TextPipeline,
                 keyframe_interval: int = 10,
                 use_optical_flow:  bool = True):
        self.pipeline    = pipeline
        self.kf_interval = keyframe_interval
        self.use_flow    = use_optical_flow

    @staticmethod
    def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        map_x = (flow[:, :, 0] + np.arange(w)).astype(np.float32)
        map_y = (flow[:, :, 1] + np.arange(h)[:, None]).astype(np.float32)
        return cv2.remap(mask, map_x, map_y,
                         interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    def process(self, input_video: str, output_video: str,
                mode: str = "remove",
                replacement_map: Optional[Dict[str, str]] = None) -> None:

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_video}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        prev_gray = None
        active_mask: Optional[np.ndarray] = None
        active_regions: List[TextRegion]  = []
        frame_idx = 0

        log.info("Processing video %s …", input_video)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            is_keyframe = (frame_idx % self.kf_interval == 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if is_keyframe or active_mask is None:
                log.info("  Keyframe %d — running OCR + inpaint", frame_idx)
                result, active_regions = self.pipeline.process_image(
                    frame, mode=mode, replacement_map=replacement_map
                )
                if active_regions:
                    active_mask = self.pipeline.masker.generate(
                        frame.shape, active_regions
                    )
                else:
                    active_mask = None
                writer.write(result)

            elif self.use_flow and prev_gray is not None and active_mask is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                warped_mask = self._warp_mask(active_mask, flow)
                result = self.pipeline.inpainter.inpaint(frame, warped_mask)
                if mode == "replace" and replacement_map and active_regions:
                    for r in active_regions:
                        new_text = replacement_map.get(r.text, "")
                        if new_text:
                            result = self.pipeline.renderer.render(
                                result, r, new_text
                            )
                writer.write(result)

            else:
                writer.write(frame)

            prev_gray = gray
            frame_idx += 1
            if frame_idx % 50 == 0:
                log.info("  … %d frames processed", frame_idx)

        cap.release()
        writer.release()
        log.info("Video saved → %s  (%d frames)", output_video, frame_idx)


# ─────────────────────────────────────────────────────────────────
# 11. DEBUG UTILITIES   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
def visualize_detections(image: np.ndarray,
                          regions: List[TextRegion],
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    vis = image.copy()
    if mask is not None:
        overlay = vis.copy()
        overlay[mask > 0] = [0, 0, 200]
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    for r in regions:
        pts = np.array(r.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        label = f"{r.text[:20]}  ({r.confidence:.2f})  font: {r.font_family}"
        cv2.putText(vis, label, (r.x, max(0, r.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
    return vis


def save_debug_pack(image: np.ndarray, result: np.ndarray,
                    mask: np.ndarray, regions: List[TextRegion],
                    out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out_dir}/input.png",      image)
    cv2.imwrite(f"{out_dir}/mask.png",       mask)
    cv2.imwrite(f"{out_dir}/result.png",     result)
    det_vis = visualize_detections(image, regions, mask)
    cv2.imwrite(f"{out_dir}/detections.png", det_vis)
    h = max(image.shape[0], result.shape[0])

    def pad(img, target_h):
        ph = target_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, ph, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))

    montage = np.hstack([pad(image, h), pad(det_vis, h), pad(result, h)])
    cv2.imwrite(f"{out_dir}/montage.png", montage)
    log.info("Debug pack saved in %s/", out_dir)


# ─────────────────────────────────────────────────────────────────
# 12. CLI   (enhanced with font classifier option)
# ─────────────────────────────────────────────────────────────────
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="text_pipeline",
        description="Remove or replace text in images / videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
# Remove all text (OpenCV inpainting — no model download):
  python text_pipeline.py remove --input sign.jpg --output clean.jpg --inpainter cv

# Remove with font classification (10-class ONNX):
  python text_pipeline.py remove --input sign.jpg --output clean.jpg \\
      --font-classifier models/font_classifier.onnx

# Replace specific text:
  python text_pipeline.py replace \\
      --input label.png --output translated.png \\
      --map '{"Hello":"Bonjour"}'

# Editor extraction only (prints JSON blocks with fonts):
  python text_pipeline.py editor --input photo.jpg --output-dir ./out/ \\
      --font-classifier models/font_classifier.onnx
        """
    )
    p.add_argument("mode", choices=["remove", "replace", "editor"])
    p.add_argument("--input",  "-i", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--output-dir", default=None,
                   help="[editor mode] directory to write cleaned image + JSON")
    p.add_argument("--map", "-m", default=None)
    p.add_argument("--languages", nargs="+", default=["en"])
    p.add_argument("--inpainter", choices=["lama", "sd", "cv"], default="cv",
                   help="Inpainter backend (cv = built-in OpenCV, no download)")
    p.add_argument("--font-classifier", default=None,
                   help="Path to 10-class ONNX font classifier model")
    p.add_argument("--gpu",        action="store_true")
    p.add_argument("--dilation",   type=int,   default=8)
    p.add_argument("--confidence", type=float, default=0.4)
    p.add_argument("--font-path",  default=None)
    p.add_argument("--keyframe-interval", type=int, default=10)
    p.add_argument("--debug-dir",  default=None)
    return p


def main():
    import json
    parser = build_cli()
    args   = parser.parse_args()

    # ── Prepare font classifier ──
    font_classifier = None
    if args.font_classifier:
        font_classifier = FontClassifier(
            model_path=args.font_classifier,
            gpu=args.gpu,
        )

    # ── Editor mode: extract + inpaint → JSON metadata ──
    if args.mode == "editor":
        img = cv2.imread(args.input)
        if img is None:
            sys.exit(f"ERROR: Cannot read image: {args.input}")
        cleaned, blocks = extract_for_editor(
            img,
            languages       = args.languages,
            confidence      = args.confidence,
            dilation_px     = args.dilation,
            gpu             = args.gpu,
            font_classifier = font_classifier,
        )
        out_dir = Path(args.output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        img_path  = out_dir / f"cleaned_{stem}.jpg"
        meta_path = out_dir / f"meta_{stem}.json"
        cv2.imwrite(str(img_path), cleaned)
        meta = {"bg_image": str(img_path), "blocks": blocks}
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info("Editor output → %s  |  %s", img_path, meta_path)
        return

    # ── Remove / Replace modes (legacy pipeline) ─────────────────
    pipeline = TextPipeline(
        languages              = args.languages,
        inpainter              = args.inpainter,
        gpu                    = args.gpu,
        dilation_px            = args.dilation,
        font_path              = args.font_path,
        confidence             = args.confidence,
        font_classifier_path   = args.font_classifier,
    )
    replacement_map = json.loads(args.map) if args.map else None
    input_path      = Path(args.input)
    video_exts      = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    if input_path.suffix.lower() in video_exts:
        vp = VideoProcessor(pipeline,
                            keyframe_interval=args.keyframe_interval)
        vp.process(args.input, args.output or "output.mp4",
                   mode=args.mode, replacement_map=replacement_map)
    else:
        img = cv2.imread(args.input)
        if img is None:
            sys.exit(f"ERROR: Cannot read: {args.input}")
        result, regions = pipeline.process_image(
            img, mode=args.mode, replacement_map=replacement_map
        )
        cv2.imwrite(args.output or "output.jpg", result)
        log.info("Result saved → %s", args.output)
        if args.debug_dir:
            mask = pipeline.masker.generate(img.shape, regions)
            save_debug_pack(img, result, mask, regions, args.debug_dir)


if __name__ == "__main__":
    main()