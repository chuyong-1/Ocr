"""
╔══════════════════════════════════════════════════════════════════╗
║        Text Removal & Replacement Pipeline  (Image / Video)      ║
║  Stack: EasyOCR · LaMa · Stable Diffusion Inpaint · PIL/OpenCV   ║
╠══════════════════════════════════════════════════════════════════╣
║  v2 — Content-Aware Editor additions                             ║
║    • EditorBlock TypedDict                                       ║
║    • rgb_to_hex() helper                                         ║
║    • CvInpainter  (cv2.inpaint — no model download required)     ║
║    • extract_for_editor()  ← single-call editor pipeline         ║
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
# 1-A.  NEW ▶  Editor type + hex helper
# ─────────────────────────────────────────────────────────────────

class EditorBlock(TypedDict):
    """
    Serialisable dict for one text region returned by extract_for_editor().

    Designed to be JSON-dumped directly into the metadata file that
    server.py / worker.py write to results/.
    """
    text:       str
    x:          int
    y:          int
    w:          int
    h:          int
    color:      str    # dominant text colour as CSS hex  e.g. "#2C2C2C"
    bg_color:   str    # dominant background colour       e.g. "#F5F0EB"
    size:       int    # estimated font size in CSS px
    confidence: float  # EasyOCR confidence 0-1


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an (R, G, B) tuple to a CSS hex string like '#1A2B3C'."""
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ─────────────────────────────────────────────────────────────────
# 1-B. DATA CLASSES   (unchanged from v1)
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

    def __post_init__(self):
        pts = np.array(self.bbox, dtype=np.int32)
        self.x = int(pts[:, 0].min())
        self.y = int(pts[:, 1].min())
        self.w = int(pts[:, 0].max()) - self.x
        self.h = int(pts[:, 1].max()) - self.y


# ─────────────────────────────────────────────────────────────────
# 2. TEXT DETECTION   (unchanged from v1)
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
# 3. STYLE EXTRACTION   (unchanged from v1)
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
# 4. MASK GENERATION   (unchanged from v1)
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
# 5-A.  NEW ▶  CvInpainter — zero-dependency OpenCV inpainting
# ─────────────────────────────────────────────────────────────────
class CvInpainter:
    """
    Fast, offline inpainting using OpenCV's built-in Telea algorithm.

    Why Telea for text removal?
    ───────────────────────────
    • Works entirely from surrounding pixel gradients — no model download.
    • Performs excellently on flat/lightly-textured backgrounds (documents,
      signage, UI screenshots).
    • Instant inference: < 50 ms even on 4K images.

    Trade-off: complex photographic backgrounds may show haloing.
    For those cases, upgrade to LamaInpainter (simple-lama-inpainting).

    Inpaint radius
    ──────────────
    The neighbourhood radius controls how far beyond the mask boundary
    the algorithm samples.  10-15 px is ideal for most text sizes.
    """

    def __init__(self, radius: int = 12,
                 method: int = cv2.INPAINT_TELEA):
        """
        Parameters
        ----------
        radius : int
            Inpainting neighbourhood radius in pixels.
        method : int
            cv2.INPAINT_TELEA (default, fast) or cv2.INPAINT_NS
            (Navier-Stokes, smoother but slower).
        """
        self.radius = radius
        self.method = method

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : BGR uint8 ndarray
        mask  : uint8 ndarray  (255 = inpaint, 0 = keep)

        Returns
        -------
        BGR uint8 ndarray — cleaned image
        """
        return cv2.inpaint(image, mask, self.radius, self.method)


# ─────────────────────────────────────────────────────────────────
# 5-B. LamaInpainter — higher quality, needs pip package
# ─────────────────────────────────────────────────────────────────
class LamaInpainter:
    """
    Uses the `simple-lama-inpainting` package which wraps the
    official LaMa model with a one-call API.

    Install: pip install simple-lama-inpainting
    """

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        log.info("Loading LaMa model…")
        self.lama = SimpleLama()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pil_img  = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        result   = self.lama(pil_img, pil_mask)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────
# 5-C. SDInpainter — optional Stable Diffusion backend
# ─────────────────────────────────────────────────────────────────
class SDInpainter:
    """
    Uses runwayml/stable-diffusion-inpainting via 🤗 Diffusers.
    Better for complex scenes; slower (GPU recommended).
    """

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
# 6. TEXT RENDERER   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class TextRenderer:
    """
    Overlays replacement text onto the inpainted image.
    """

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
# 7.  NEW ▶  extract_for_editor() — single entry point for the
#            content-aware editor workflow used by worker.py
# ─────────────────────────────────────────────────────────────────
def extract_for_editor(
    image_bgr:   np.ndarray,
    languages:   List[str]  = None,
    confidence:  float      = 0.40,
    dilation_px: int        = 8,
    inpaint_radius: int     = 12,
    gpu:         bool       = False,
) -> Tuple[np.ndarray, List[EditorBlock]]:
    """
    Complete single-call pipeline for the content-aware editor.

    Stages
    ──────
    1. EasyOCR detection — word-level bounding boxes + confidence
    2. StyleExtractor    — K-Means text/bg colour, font-size estimate
    3. MaskGenerator     — dilated binary mask over all text regions
    4. CvInpainter       — cv2.inpaint (Telea) fills background cleanly
    5. Build EditorBlock list with hex colours ready for JSON output

    Parameters
    ──────────
    image_bgr       OpenCV BGR image array (from cv2.imread)
    languages       EasyOCR language codes, e.g. ["en", "fr"]
    confidence      Minimum OCR confidence threshold (0–1)
    dilation_px     Mask dilation to catch antialiased glyph borders
    inpaint_radius  cv2.inpaint neighbourhood radius (px)
    gpu             Pass True to use CUDA for EasyOCR

    Returns
    ───────
    cleaned_bgr     The image with all text regions inpainted away
    blocks          List[EditorBlock] — serialisable, JSON-ready
    """
    langs = languages or ["en"]

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

    # ── Stage 3: Mask ────────────────────────────────────────────
    masker = MaskGenerator(dilation_px=dilation_px)
    mask   = masker.generate(image_bgr.shape, regions)

    # ── Stage 4: Inpaint (OpenCV Telea — no model required) ──────
    inpainter  = CvInpainter(radius=inpaint_radius)
    cleaned    = inpainter.inpaint(image_bgr, mask)
    log.info("Inpainting complete  (%d region(s)).", len(regions))

    # ── Stage 5: Build EditorBlock list ─────────────────────────
    blocks: List[EditorBlock] = []
    for r in regions:
        blocks.append(EditorBlock(
            text       = r.text,
            x          = r.x,
            y          = r.y,
            w          = r.w,
            h          = r.h,
            color      = rgb_to_hex(r.text_color),
            bg_color   = rgb_to_hex(r.bg_color),
            size       = r.font_size,
            confidence = round(r.confidence, 4),
        ))

    return cleaned, blocks


# ─────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE ORCHESTRATOR   (unchanged from v1 — used by CLI)
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
                 confidence:  float         = 0.4):

        self.detector   = TextDetector(languages, gpu=gpu)
        self.extractor  = StyleExtractor()
        self.masker     = MaskGenerator(dilation_px)
        self.renderer   = TextRenderer(font_path)
        self.confidence = confidence

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
# 9. VIDEO PROCESSING   (unchanged from v1)
# ─────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Processes a video file frame-by-frame using the TextPipeline.
    See v1 docstring for temporal consistency strategies.
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
# 10. DEBUG UTILITIES   (unchanged from v1)
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
        label = f"{r.text[:20]}  ({r.confidence:.2f})"
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
# 11. CLI   (unchanged from v1)
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

# Remove with LaMa (higher quality):
  python text_pipeline.py remove --input sign.jpg --output clean.jpg

# Replace specific text:
  python text_pipeline.py replace \\
      --input label.png --output translated.png \\
      --map '{"Hello":"Bonjour"}'

# Editor extraction only (prints JSON blocks):
  python text_pipeline.py editor --input photo.jpg --output-dir ./out/
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

    # ── Editor mode: extract + inpaint → JSON metadata ──────────
    if args.mode == "editor":
        img = cv2.imread(args.input)
        if img is None:
            sys.exit(f"ERROR: Cannot read image: {args.input}")
        cleaned, blocks = extract_for_editor(
            img,
            languages   = args.languages,
            confidence  = args.confidence,
            dilation_px = args.dilation,
            gpu         = args.gpu,
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
        languages   = args.languages,
        inpainter   = args.inpainter,
        gpu         = args.gpu,
        dilation_px = args.dilation,
        font_path   = args.font_path,
        confidence  = args.confidence,
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