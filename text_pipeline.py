"""
╔══════════════════════════════════════════════════════════════════╗
║        Text Removal & Replacement Pipeline  (Image / Video)      ║
║  Stack: EasyOCR · LaMa · Stable Diffusion Inpaint · PIL/OpenCV   ║
╚══════════════════════════════════════════════════════════════════╝

Pipeline stages
───────────────
  1. Text Detection & OCR        (EasyOCR)
  2. Style Extraction            (dominant colour, font-size estimate)
  3. Mask Generation             (dilated binary mask)
  4. Inpainting / Removal        (LaMa via simple-lama-inpainting OR
                                  Stable Diffusion Inpaint via 🤗)
  5. Text Rendering / Replacement (PIL + OpenCV)
  6. Video support               (FFmpeg + temporal-consistency notes)
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
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 1. DATA CLASSES
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
    font_size: int = 20
    text_color: Tuple[int, int, int] = (0, 0, 0)
    bg_color: Tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self):
        pts = np.array(self.bbox, dtype=np.int32)
        self.x = int(pts[:, 0].min())
        self.y = int(pts[:, 1].min())
        self.w = int(pts[:, 0].max()) - self.x
        self.h = int(pts[:, 1].max()) - self.y


# ─────────────────────────────────────────────────────────────────
# 2. TEXT DETECTION
# ─────────────────────────────────────────────────────────────────
class TextDetector:
    """
    Wraps EasyOCR for text detection.

    Alternative: swap for PaddleOCR by replacing the detect() method —
    PaddleOCR gives better results on CJK / vertical text.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        import easyocr  # lazy import so the module loads without GPU
        langs = languages or ["en"]
        log.info("Loading EasyOCR (languages=%s, gpu=%s)…", langs, gpu)
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def detect(self, image: np.ndarray,
               confidence_threshold: float = 0.4) -> List[TextRegion]:
        """
        Run OCR and return a list of TextRegion objects.

        Parameters
        ----------
        image : BGR numpy array (OpenCV format)
        confidence_threshold : discard boxes below this score
        """
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
# 3. STYLE EXTRACTION
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
        # guard against out-of-bounds
        H, W = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return region

        crop_bgr = image[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # ── Otsu threshold to separate text pixels from background ──
        _, mask = cv2.threshold(crop_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Decide which class is "text" by checking which is darker on average
        dark_mean  = crop_gray[mask == 0].mean() if (mask == 0).any() else 255
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
            self.dominant_color(bg_pixels,   k=1).tolist()
        )

        # ── Font-size estimate: bounding-box height ≈ cap height ──
        region.font_size = max(8, int(h * 0.85))

        log.debug("  Style → font_size=%d  text_color=%s  bg_color=%s",
                  region.font_size, region.text_color, region.bg_color)
        return region


# ─────────────────────────────────────────────────────────────────
# 4. MASK GENERATION
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
        """
        Returns a uint8 mask of the same H×W as the image.
        """
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
# 5-A. INPAINTING — LaMa backend
# ─────────────────────────────────────────────────────────────────
class LamaInpainter:
    """
    Uses the `simple-lama-inpainting` package which wraps the
    official LaMa model with a one-call API.

    Install: pip install simple-lama-inpainting
    LaMa paper: https://arxiv.org/abs/2109.07161

    LaMa is ideal for text removal because:
      • Receptive field covers very large regions (Fourier convolutions)
      • Produces coherent textures, tiles, gradients
      • Fast inference (no diffusion loop)
    """

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        log.info("Loading LaMa model…")
        self.lama = SimpleLama()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        image : BGR uint8
        mask  : uint8  (255 = inpaint, 0 = keep)
        returns BGR uint8
        """
        pil_img  = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        result   = self.lama(pil_img, pil_mask)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────
# 5-B. INPAINTING — Stable Diffusion backend (optional / higher-quality)
# ─────────────────────────────────────────────────────────────────
class SDInpainter:
    """
    Uses runwayml/stable-diffusion-inpainting via 🤗 Diffusers.
    Better for complex scenes; slower (GPU recommended).

    Install:
        pip install diffusers transformers accelerate
    Model download is automatic on first run (~5 GB).

    Usage:  --inpainter sd
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
        self.pipe.safety_checker = None   # disable for speed

    def inpaint(self, image: np.ndarray, mask: np.ndarray,
                prompt: str = "seamless background, high quality",
                num_inference_steps: int = 30) -> np.ndarray:
        """
        image  : BGR uint8
        mask   : uint8 (255 = inpaint)
        returns: BGR uint8
        """
        # SD requires 512×512 multiples; we resize, inpaint, resize back
        orig_h, orig_w = image.shape[:2]
        TARGET = 512

        def _resize(arr, size):
            return cv2.resize(arr, (size, size), interpolation=cv2.INTER_LANCZOS4)

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
# 6. TEXT RENDERER
# ─────────────────────────────────────────────────────────────────
class TextRenderer:
    """
    Overlays replacement text onto the inpainted image.

    Font matching strategy
    ──────────────────────
    1. Try to load a specific TTF/OTF by path (--font-path).
    2. Fall back to ImageFont.truetype with a system font name.
    3. Last resort: PIL default bitmap font (looks rough).

    For production use, integrate FontTools + a font-classification
    model (e.g. DeepFont) to auto-match the original typeface.
    """

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path  # may be None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        candidates = [
            self.font_path,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",   # macOS
            "C:/Windows/Fonts/arial.ttf",            # Windows
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
        """
        Places `new_text` inside `region`'s bounding box,
        matching the extracted style.
        """
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        font = self._get_font(region.font_size)

        # ── Auto-fit: shrink font until text fits the bbox width ──
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

        # ── Center text vertically in the region ──
        try:
            tb = font.getbbox(new_text)
            th = tb[3] - tb[1]
        except AttributeError:
            _, th = draw.textsize(new_text, font=font)
        x_pos = region.x
        y_pos = region.y + max(0, (region.h - th) // 2)

        # Optional: slight shadow for legibility
        shadow_offset = max(1, fsize // 18)
        shadow_color  = tuple(max(0, c - 80) for c in region.text_color)
        draw.text((x_pos + shadow_offset, y_pos + shadow_offset),
                  new_text, font=font, fill=shadow_color)
        draw.text((x_pos, y_pos), new_text, font=font, fill=region.text_color)

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────
class TextPipeline:
    """
    High-level API that wires all stages together.

    Modes
    ─────
    • remove  — erase all detected text (clean background only)
    • replace — erase original text, render replacement_map values
    """

    def __init__(self,
                 languages: List[str] = None,
                 inpainter: str = "lama",   # "lama" | "sd"
                 gpu: bool = False,
                 dilation_px: int = 6,
                 font_path: Optional[str] = None,
                 confidence: float = 0.4):

        self.detector  = TextDetector(languages, gpu=gpu)
        self.extractor = StyleExtractor()
        self.masker    = MaskGenerator(dilation_px)
        self.renderer  = TextRenderer(font_path)
        self.confidence = confidence

        if inpainter == "sd":
            device = "cuda" if gpu else "cpu"
            self.inpainter = SDInpainter(device=device)
        else:
            self.inpainter = LamaInpainter()

    # ── Core image processing ──────────────────────────────────

    def process_image(
        self,
        image_bgr: np.ndarray,
        mode: str = "remove",
        replacement_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[TextRegion]]:
        """
        Parameters
        ──────────
        image_bgr      : BGR image array
        mode           : "remove" | "replace"
        replacement_map: { original_text: new_text }  (used when mode="replace")

        Returns
        ───────
        result_bgr : processed image
        regions    : detected TextRegion list (for downstream use)
        """
        regions = self.detector.detect(image_bgr, self.confidence)
        if not regions:
            log.info("No text detected — returning original image.")
            return image_bgr.copy(), regions

        # Style extraction on each region
        for r in regions:
            self.extractor.extract(image_bgr, r)

        # Build combined mask and inpaint once (batch is faster than per-region)
        mask = self.masker.generate(image_bgr.shape, regions)
        result = self.inpainter.inpaint(image_bgr, mask)

        # Text rendering (replace mode only)
        if mode == "replace" and replacement_map:
            for r in regions:
                new_text = replacement_map.get(r.text)
                if new_text is None:
                    # Fuzzy match: try lower-cased key
                    new_text = replacement_map.get(r.text.lower())
                if new_text is not None:
                    result = self.renderer.render(result, r, new_text)

        return result, regions

    # ── Convenience wrappers ───────────────────────────────────

    def remove_text_from_file(self, input_path: str,
                               output_path: str) -> None:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {input_path}")
        result, _ = self.process_image(img, mode="remove")
        cv2.imwrite(output_path, result)
        log.info("Saved → %s", output_path)

    def replace_text_in_file(self, input_path: str, output_path: str,
                              replacement_map: Dict[str, str]) -> None:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {input_path}")
        result, _ = self.process_image(img, mode="replace",
                                       replacement_map=replacement_map)
        cv2.imwrite(output_path, result)
        log.info("Saved → %s", output_path)


# ─────────────────────────────────────────────────────────────────
# 8. VIDEO PROCESSING  (frame-by-frame with temporal smoothing)
# ─────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Processes a video file frame-by-frame using the TextPipeline.

    Temporal Consistency Strategy (Basic)
    ──────────────────────────────────────
    Problem: Inpainting the same background region independently on
    every frame produces flicker because the model's output varies
    slightly each time.

    Solution A — Mask Propagation (implemented here):
      1. Run OCR only on every Nth frame ("keyframe").
      2. Propagate the mask from the keyframe to neighbouring frames
         using optical flow (cv2.calcOpticalFlowFarneback).
      3. Inpaint each frame with the warped mask.
      This keeps background texture locked to the reference keyframe.

    Solution B — ProPainter (recommended for production):
      ProPainter (ICCV 2023) is a video inpainting model that
      explicitly models temporal coherence via a recurrent
      transformer. Feed it the full video + mask sequence.
      GitHub: https://github.com/sczhou/ProPainter
      pip install propainter (unofficial wrapper)

    Solution C — First-Frame Anchoring (simplest):
      Run inpainting only on frame 0, then use homography / optical
      flow to paste the first-frame result onto subsequent frames
      where the background is assumed static (good for signage).
    """

    def __init__(self, pipeline: TextPipeline,
                 keyframe_interval: int = 10,
                 use_optical_flow: bool = True):
        self.pipeline = pipeline
        self.kf_interval = keyframe_interval
        self.use_flow = use_optical_flow

    # ── Optical-flow mask warp ─────────────────────────────────

    @staticmethod
    def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Apply an optical-flow field to a binary mask."""
        h, w = flow.shape[:2]
        map_x = (flow[:, :, 0] + np.arange(w)).astype(np.float32)
        map_y = (flow[:, :, 1] + np.arange(h)[:, None]).astype(np.float32)
        return cv2.remap(mask, map_x, map_y,
                         interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # ── Main processing loop ───────────────────────────────────

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

        prev_gray  = None
        active_mask: Optional[np.ndarray] = None
        active_regions: List[TextRegion]  = []
        frame_idx  = 0

        log.info("Processing video %s …", input_video)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            is_keyframe = (frame_idx % self.kf_interval == 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Keyframe: re-detect text and inpaint ──────────
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

            # ── Inter-frame: warp mask with optical flow ───────
            elif self.use_flow and prev_gray is not None and active_mask is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                warped_mask = self._warp_mask(active_mask, flow)
                # Inpaint with the warped mask (background consistent since
                # LaMa is deterministic given the same conditioning)
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
                # Fallback: no flow data yet
                writer.write(frame)

            prev_gray = gray
            frame_idx += 1
            if frame_idx % 50 == 0:
                log.info("  … %d frames processed", frame_idx)

        cap.release()
        writer.release()
        log.info("Video saved → %s  (%d frames)", output_video, frame_idx)


# ─────────────────────────────────────────────────────────────────
# 9. DIAGNOSTIC / DEBUG UTILITIES
# ─────────────────────────────────────────────────────────────────
def visualize_detections(image: np.ndarray,
                          regions: List[TextRegion],
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draws bounding boxes + OCR text + style swatches for inspection.
    Optionally overlays the inpaint mask in red.
    """
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
    """Saves input, mask, detection vis, and result side-by-side."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{out_dir}/input.png",      image)
    cv2.imwrite(f"{out_dir}/mask.png",       mask)
    cv2.imwrite(f"{out_dir}/result.png",     result)
    det_vis = visualize_detections(image, regions, mask)
    cv2.imwrite(f"{out_dir}/detections.png", det_vis)

    # Horizontal montage
    h = max(image.shape[0], result.shape[0])
    def pad(img, target_h):
        ph = target_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, ph, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))

    panels = [pad(image, h), pad(det_vis, h), pad(result, h)]
    montage = np.hstack(panels)
    cv2.imwrite(f"{out_dir}/montage.png", montage)
    log.info("Debug pack saved in %s/", out_dir)


# ─────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="text_pipeline",
        description="Remove or replace text in images / videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
# Remove all text from an image (LaMa inpainting):
  python text_pipeline.py remove --input sign.jpg --output clean.jpg

# Replace specific text (JSON map):
  python text_pipeline.py replace \\
      --input label.png --output translated.png \\
      --map '{"Hello":"Bonjour","World":"Monde"}'

# Process a video (remove all text, keyframe every 15 frames):
  python text_pipeline.py remove --input ad.mp4 --output clean.mp4 \\
      --keyframe-interval 15

# Use Stable Diffusion inpainter instead of LaMa (needs GPU):
  python text_pipeline.py remove --input sign.jpg --output clean.jpg \\
      --inpainter sd --gpu
        """
    )
    p.add_argument("mode", choices=["remove", "replace"],
                   help="Processing mode")
    p.add_argument("--input",  "-i", required=True, help="Input file path")
    p.add_argument("--output", "-o", required=True, help="Output file path")
    p.add_argument("--map", "-m", default=None,
                   help='JSON replacement map, e.g. \'{"OLD":"NEW"}\'')
    p.add_argument("--languages", nargs="+", default=["en"],
                   help="OCR languages (EasyOCR codes, e.g. en fr de)")
    p.add_argument("--inpainter", choices=["lama", "sd"], default="lama")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU for OCR and inpainting")
    p.add_argument("--dilation", type=int, default=6,
                   help="Mask dilation in pixels (default: 6)")
    p.add_argument("--confidence", type=float, default=0.4,
                   help="OCR confidence threshold (default: 0.4)")
    p.add_argument("--font-path", default=None,
                   help="Path to a .ttf/.otf font for text replacement")
    p.add_argument("--keyframe-interval", type=int, default=10,
                   help="OCR keyframe interval for video (default: 10)")
    p.add_argument("--debug-dir", default=None,
                   help="Save debug visualisations to this directory")
    return p


def main():
    parser = build_cli()
    args   = parser.parse_args()

    # Build pipeline
    pipeline = TextPipeline(
        languages=args.languages,
        inpainter=args.inpainter,
        gpu=args.gpu,
        dilation_px=args.dilation,
        font_path=args.font_path,
        confidence=args.confidence,
    )

    # Parse replacement map
    replacement_map: Optional[Dict[str, str]] = None
    if args.map:
        import json
        replacement_map = json.loads(args.map)

    # Decide: image or video?
    input_path = Path(args.input)
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    if input_path.suffix.lower() in video_exts:
        vp = VideoProcessor(pipeline,
                            keyframe_interval=args.keyframe_interval)
        vp.process(args.input, args.output,
                   mode=args.mode,
                   replacement_map=replacement_map)
    else:
        img = cv2.imread(args.input)
        if img is None:
            sys.exit(f"ERROR: Cannot read image: {args.input}")
        result, regions = pipeline.process_image(
            img, mode=args.mode, replacement_map=replacement_map
        )
        cv2.imwrite(args.output, result)
        log.info("Result saved → %s", args.output)

        if args.debug_dir:
            mask = pipeline.masker.generate(img.shape, regions)
            save_debug_pack(img, result, mask, regions, args.debug_dir)


if __name__ == "__main__":
    main()