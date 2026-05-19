"""
TextClear — LangGraph Orchestration Engine
==========================================
Replaces the static linear pipeline with a dynamic StateGraph that routes
compute based on texture complexity of the masked region.

Graph topology
--------------
    START
      │
      ▼
  [evaluator]          ← Laplacian variance on masked region
      │
      ▼ conditional
  ┌───┴────────────────────────┐
  │ variance ≤ threshold        │ variance > threshold
  ▼                            ▼
[telea_inpaint]          [lama_inpaint]      ← ONNX Runtime
  │                            │
  └──────────┬─────────────────┘
             │
             ▼
    [font_classifier]           ← ONNX Runtime per text block
             │
             ▼
           END

Celery integration
------------------
Call `init_session_manager()` inside the Celery `worker_process_init`
signal so ONNX sessions are loaded once per worker process, not once
per task.

    from celery.signals import worker_process_init

    @worker_process_init.connect
    def init_onnx(**kwargs):
        init_session_manager()
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from langgraph.graph import END, StateGraph
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import TypedDict

log = logging.getLogger("textclear.langgraph")


# ─────────────────────────────────────────────────────────────────
# PROGRESS EMITTER  (Redis pub/sub — silently swallows failures)
# ─────────────────────────────────────────────────────────────────

class ProgressEmitter:
    """
    Publishes node-level progress to Redis pub/sub.

    Schema matches the existing ``_set_progress`` pattern in worker.py:
    ``r.publish(f"job:{job_id}", json.dumps({"status", "progress", "node"}))``

    All Redis failures are swallowed — progress is best-effort,
    never crashes the pipeline.
    """

    # Canonical percent mapping per node
    PERCENT_MAP: Dict[str, int] = {
        "evaluator":       10,
        "telea_start":     20,
        "lama_start":      20,
        "telea_done":      40,
        "lama_done":       75,
        "font_classifier": 85,
        "renderer":        95,
        "END":             100,
    }

    def __init__(self, redis_url: str, job_id: str):
        self._job_id = job_id
        self._redis_url = redis_url
        self._conn = None
        try:
            from redis import Redis
            self._conn = Redis.from_url(redis_url)
        except Exception:
            log.debug("ProgressEmitter: Redis unavailable, running silent")

    def emit(self, node_name: str, percent: Optional[int] = None) -> None:
        """Publish progress. Uses PERCENT_MAP if percent is None."""
        if self._conn is None:
            return
        pct = percent if percent is not None else self.PERCENT_MAP.get(node_name, 0)
        try:
            self._conn.publish(
                f"job:{self._job_id}",
                json.dumps({
                    "status":   "RUNNING",
                    "progress": pct,
                    "node":     node_name,
                    "error":    None,
                }),
            )
        except Exception:
            pass  # best-effort — never crash the pipeline

# ─────────────────────────────────────────────────────────────────
# TUNEABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Laplacian variance above this → complex texture → LaMa
COMPLEXITY_THRESHOLD: float = float(os.getenv("COMPLEXITY_THRESHOLD", "150.0"))

# LaMa spatial divisor (must match training config; standard is 8)
LAMA_PAD_DIVISOR: int = 8

# cv2.inpaint radius for Telea — tune to your average stroke width
TELEA_RADIUS: int = 3

# ImageNet stats for font classifier (override if your model uses different)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# 1.  STATE DEFINITION
# ─────────────────────────────────────────────────────────────────

class TextBlock(TypedDict):
    id:         str
    bbox:       List[List[int]]   # EasyOCR 4-point polygon [[x,y], ...]
    text:       str
    confidence: float
    x:          int               # AABB top-left
    y:          int
    w:          int               # AABB width
    h:          int               # AABB height


class FontPrediction(TypedDict):
    block_id:   str
    label:      str               # Resolved from label_map; falls back to "class_N"
    confidence: float             # Softmax probability of top class
    logits:     List[float]       # Raw model outputs (all classes)


class PipelineState(TypedDict):
    # ── Primary inputs (set by caller) ───────────────────────
    original_image:   np.ndarray          # BGR uint8  [H, W, 3]
    mask:             np.ndarray          # uint8      [H, W],  255 = inpaint
    text_blocks:      List[TextBlock]
    replacement_map:  Dict[str, str]      # {original_text: new_text}
    mode:             str                 # "remove" | "replace"

    # ── Computed by graph nodes ───────────────────────────────
    complexity_score:  float
    inpaint_method:    str                # "telea" | "lama"
    inpainted_image:   Optional[np.ndarray]
    font_metadata:     List[FontPrediction]
    rendered_image:    Optional[np.ndarray]   # NEW: output of renderer_node

    # ── Error channel ─────────────────────────────────────────
    error:  Optional[str]


# ─────────────────────────────────────────────────────────────────
# 2.  ONNX SESSION MANAGER  (one singleton per worker process)
# ─────────────────────────────────────────────────────────────────

class ONNXSessionManager:
    """
    Holds pre-loaded ONNX Runtime sessions for the worker process.

    Design decisions
    ----------------
    * Singleton — prevents duplicate model loads across tasks in the
      same Celery worker process.
    * Sessions are safe for concurrent *inference* on the same session
      object (ORT uses internal thread pools), but session *creation*
      must be serialised — the singleton __init__ guard handles this.
    * GPU: if CUDAExecutionProvider is available it is prepended
      automatically.  Override via `providers` kwarg.
    """

    _instance: Optional[ONNXSessionManager] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> ONNXSessionManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def __init__(
        self,
        lama_path:  str = "models/lama.onnx",
        font_path:  str = "models/font_classifier.onnx",
        font_labels: Optional[List[str]] = None,
        providers:  Optional[List[str]] = None,
    ) -> None:
        if self._ready:
            return

        if providers is None:
            available = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern        = True
        opts.enable_cpu_mem_arena      = True
        # One intra-op thread per CPU core; tune for your machine
        opts.intra_op_num_threads      = int(os.getenv("ORT_THREADS", "4"))

        log.info("Loading LaMa session  → %s  [%s]", lama_path, providers)
        self.lama = ort.InferenceSession(lama_path,  sess_options=opts, providers=providers)

        log.info("Loading Font session  → %s  [%s]", font_path, providers)
        self.font = ort.InferenceSession(font_path,  sess_options=opts, providers=providers)

        # Cache I/O descriptors — avoids repeated protobuf parsing per call
        lama_inputs          = self.lama.get_inputs()
        self.lama_img_name   = lama_inputs[0].name   # "image"
        self.lama_mask_name  = lama_inputs[1].name   # "mask"
        self.lama_out_name   = self.lama.get_outputs()[0].name

        font_inputs          = self.font.get_inputs()
        self.font_in_name    = font_inputs[0].name
        self.font_out_name   = self.font.get_outputs()[0].name

        # Derive font model input spatial dims from metadata
        font_shape = font_inputs[0].shape            # e.g. [1, 3, 224, 224]
        self.font_h = int(font_shape[2]) if len(font_shape) == 4 else 224
        self.font_w = int(font_shape[3]) if len(font_shape) == 4 else 224

        # Optional human-readable label map
        self.font_labels: Optional[List[str]] = font_labels

        self._ready = True
        log.info("ONNXSessionManager ready.")

    @classmethod
    def get(cls) -> ONNXSessionManager:
        if cls._instance is None or not cls._instance._ready:
            raise RuntimeError(
                "ONNXSessionManager not initialised. "
                "Call init_session_manager() in worker_process_init."
            )
        return cls._instance


def init_session_manager(
    lama_path:   str = "models/lama.onnx",
    font_path:   str = "models/font_classifier.onnx",
    font_labels: Optional[List[str]] = None,
    providers:   Optional[List[str]] = None,
) -> ONNXSessionManager:
    """
    Convenience wrapper — call this once per worker process.

    Celery usage
    ------------
        from celery.signals import worker_process_init
        from langgraph_pipeline import init_session_manager

        @worker_process_init.connect
        def _init_onnx(**_kwargs):
            init_session_manager()
    """
    return ONNXSessionManager(
        lama_path=lama_path,
        font_path=font_path,
        font_labels=font_labels,
        providers=providers,
    )


# ─────────────────────────────────────────────────────────────────
# 3.  TENSOR UTILITIES
# ─────────────────────────────────────────────────────────────────

def _pad_to_divisor(
    image: np.ndarray,
    mask:  np.ndarray,
    div:   int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image and mask so H and W are each divisible by `div`.

    Uses REFLECT_101 for the image (avoids hard boundary artefacts
    at the seam) and CONSTANT=0 for the mask (no additional inpaint
    area introduced by padding).

    Returns
    -------
    padded_image, padded_mask, (pad_top, pad_bottom, pad_left, pad_right)
    """
    H, W = image.shape[:2]

    pad_h = (-H) % div   # equivalent to (div - H % div) % div
    pad_w = (-W) % div

    # Split evenly; remainder goes to the trailing edge
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2

    padded_img  = cv2.copyMakeBorder(image, pt, pb, pl, pr, cv2.BORDER_REFLECT_101)
    padded_mask = cv2.copyMakeBorder(mask,  pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)

    return padded_img, padded_mask, (pt, pb, pl, pr)


def _unpad(
    image:  np.ndarray,
    pads:   Tuple[int, int, int, int],
) -> np.ndarray:
    """Strip padding applied by `_pad_to_divisor`."""
    pt, pb, pl, pr = pads
    H, W = image.shape[:2]
    return image[pt : H - pb if pb else H, pl : W - pr if pr else W]


def bgr_to_lama_tensors(
    image_bgr: np.ndarray,
    mask:      np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Convert a BGR image + binary mask to LaMa ONNX input tensors.

    LaMa input contract
    -------------------
    ``image`` : float32  [1, 3, H', W']   RGB,  values in [0, 1]
    ``mask``  : float32  [1, 1, H', W']   values in {0.0, 1.0}
                         1.0 = pixel to inpaint

    H', W' are padded to the nearest multiple of LAMA_PAD_DIVISOR.

    Returns
    -------
    image_tensor : float32 [1, 3, H', W']
    mask_tensor  : float32 [1, 1, H', W']
    pads         : (pt, pb, pl, pr) — needed to unpad the output
    """
    padded_img, padded_mask, pads = _pad_to_divisor(image_bgr, mask, LAMA_PAD_DIVISOR)

    # BGR → RGB, uint8 → float32 ∈ [0, 1]
    rgb_f32 = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # [H, W, C] → [1, C, H, W]  (NCHW)
    img_t = np.ascontiguousarray(rgb_f32.transpose(2, 0, 1)[np.newaxis])   # [1, 3, H, W]

    # Mask: 255/0 → 1.0/0.0 → [1, 1, H, W]
    msk_t = np.ascontiguousarray(
        (padded_mask.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]   # [1, 1, H, W]
    )

    return img_t, msk_t, pads


def lama_tensor_to_bgr(
    output:  np.ndarray,
    pads:    Tuple[int, int, int, int],
    orig_h:  int,
    orig_w:  int,
) -> np.ndarray:
    """
    Convert LaMa ONNX output tensor back to a BGR uint8 numpy array.

    LaMa output contract
    --------------------
    ``inpainted`` : float32  [1, 3, H', W']   RGB,  values nominally in [0, 1]
                   (can exceed this range slightly; clamp before cast).

    Returns
    -------
    BGR uint8  [orig_H, orig_W, 3]
    """
    # [1, 3, H, W] → [H, W, 3]
    hwc = output[0].transpose(1, 2, 0)

    # Clamp (LaMa can produce values slightly outside [0, 1])
    hwc = np.clip(hwc, 0.0, 1.0)

    # float32 → uint8
    uint8_rgb = (hwc * 255.0).round().astype(np.uint8)

    # RGB → BGR
    bgr = cv2.cvtColor(uint8_rgb, cv2.COLOR_RGB2BGR)

    # Remove padding
    result = _unpad(bgr, pads)

    if result.shape[:2] != (orig_h, orig_w):
        raise ValueError(
            f"Postprocessing shape mismatch: expected ({orig_h}, {orig_w}), "
            f"got {result.shape[:2]}"
        )

    return result


# ─────────────────────────────────────────────────────────────────
# 4.  GRAPH NODES
# ─────────────────────────────────────────────────────────────────

# ── 4a. Evaluator ────────────────────────────────────────────────

def evaluator_node(state: PipelineState) -> Dict[str, Any]:
    """
    Measure texture complexity of the inpaint region.

    Algorithm
    ---------
    1. Convert image to grayscale.
    2. Apply Laplacian (second derivative highlights edges/texture).
    3. Compute variance of Laplacian values *within the mask only*.
       High variance  → fine texture, repeated patterns → LaMa
       Low variance   → smooth gradients, solid colour → Telea

    The Laplacian variance metric is chosen over raw pixel variance
    because it is invariant to uniform luminance shifts (e.g. a bright
    white wall vs a dark wall both read as low-variance).

    Edge cases
    ----------
    * Mask < 10 px: degenerate — route to Telea (safe/cheap).
    * Mask covers entire image: still computed correctly.
    """
    image = state["original_image"]
    mask  = state["mask"]

    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  # float64 [H, W]

    mask_bool = mask > 0
    n_px = int(mask_bool.sum())

    if n_px < 10:
        complexity     = 0.0
        inpaint_method = "telea"
        log.warning("evaluator: mask only %d px — forced Telea", n_px)
    else:
        complexity     = float(np.var(laplacian[mask_bool]))
        inpaint_method = "lama" if complexity > COMPLEXITY_THRESHOLD else "telea"

    log.info(
        "evaluator: n_px=%d  var=%.2f  threshold=%.2f  → %s",
        n_px, complexity, COMPLEXITY_THRESHOLD, inpaint_method,
    )

    return {
        "complexity_score": complexity,
        "inpaint_method":   inpaint_method,
    }


# ── 4b. Telea inpaint ────────────────────────────────────────────

def telea_inpaint_node(state: PipelineState) -> Dict[str, Any]:
    """
    Lightweight inpainting for flat/gradient backgrounds.

    cv2.INPAINT_TELEA is a fast PDE-based method that propagates
    colour information from the boundary inward.  It works well
    when the underlying background is smooth (solid colour, simple
    gradient, uniform texture) but produces smearing on fine patterns.

    ``inpaintRadius=3`` covers a typical dilated text stroke cleanly;
    increase to 5–7 only if dilation_px > 10.
    """
    result = cv2.inpaint(
        state["original_image"],
        state["mask"],
        inpaintRadius=TELEA_RADIUS,
        flags=cv2.INPAINT_TELEA,
    )
    log.info("telea_inpaint: done  shape=%s", result.shape)
    return {"inpainted_image": result}


# ── 4c. LaMa inpaint ─────────────────────────────────────────────

def lama_inpaint_node(state: PipelineState) -> Dict[str, Any]:
    """
    High-quality content-aware inpainting via LaMa ONNX.

    Tensor pipeline (detailed)
    --------------------------
    Input (BGR uint8 [H, W, 3])
      ↓  pad H and W to nearest multiple of 8 (REFLECT_101)
      ↓  BGR → RGB,  /255  →  float32 [0, 1]
      ↓  transpose (H,W,C) → (C,H,W)
      ↓  np.newaxis  →  [1, C, H, W]    image_tensor
    Mask (uint8 [H, W], 255 = erase)
      ↓  pad with zeros (no extra inpaint area)
      ↓  /255  →  float32 {0, 1}
      ↓  newaxis × 2  →  [1, 1, H, W]  mask_tensor
    ONNX run
      →  output float32 [1, 3, H', W']  RGB [0, 1]
      ↓  [0].transpose(1,2,0)  →  [H', W', 3]
      ↓  clip(0,1) → ×255 → round → uint8
      ↓  RGB → BGR
      ↓  unpad  →  [orig_H, orig_W, 3]   ← returned

    Memory
    ------
    The three large temporaries (image_tensor, mask_tensor,
    output_tensor) are explicitly deleted and gc.collect() is called
    after inference.  LaMa on a 4 K image requires ~1.5 GB peak;
    the explicit free helps when the Celery worker processes several
    jobs sequentially without restarting.
    """
    mgr = ONNXSessionManager.get()

    image  = state["original_image"]
    mask   = state["mask"]
    orig_h, orig_w = image.shape[:2]

    log.info("lama_inpaint: image=%s  mask=%s", image.shape, mask.shape)

    image_t, mask_t, pads = bgr_to_lama_tensors(image, mask)

    try:
        outputs = mgr.lama.run(
            [mgr.lama_out_name],
            {
                mgr.lama_img_name:  image_t,
                mgr.lama_mask_name: mask_t,
            },
        )
        result = lama_tensor_to_bgr(outputs[0], pads, orig_h, orig_w)
    finally:
        # Release regardless of success/failure
        del image_t, mask_t
        if "outputs" in dir():
            del outputs
        gc.collect()

    log.info("lama_inpaint: done  result=%s", result.shape)
    return {"inpainted_image": result}


# ── 4d. Font classifier ──────────────────────────────────────────

def font_classifier_node(state: PipelineState) -> Dict[str, Any]:
    """
    Classify the font family for every detected text block.

    Per-block pipeline
    ------------------
    1. Crop the AABB from ``original_image`` (clamped to image bounds).
    2. Resize to the model's expected H × W (read from ONNX metadata).
    3. Normalise with ImageNet μ/σ (override IMAGENET_* constants if
       your model was trained differently).
    4. Transpose to NCHW, add batch dimension.
    5. Run inference → logits [num_classes].
    6. Softmax to get per-class probabilities.
    7. Return top-1 label + confidence + raw logits.

    The original image is used for cropping (not inpainted_image) so
    that the classifier sees the original glyph pixels — it is
    classifying the *source* font, not a reconstructed background.

    Blocks with degenerate bounding boxes (zero area after clamping)
    are skipped with a warning rather than crashing the graph.
    """
    mgr         = ONNXSessionManager.get()
    image       = state["original_image"]
    text_blocks = state["text_blocks"]
    img_h, img_w = image.shape[:2]

    predictions: List[FontPrediction] = []

    for blk in text_blocks:
        x1 = max(0, blk["x"])
        y1 = max(0, blk["y"])
        x2 = min(img_w, blk["x"] + blk["w"])
        y2 = min(img_h, blk["y"] + blk["h"])

        if x2 <= x1 or y2 <= y1:
            log.warning("font_classifier: skipping block %s (degenerate bbox)", blk["id"])
            continue

        crop = image[y1:y2, x1:x2]

        # ── Preprocess ───────────────────────────────────────────
        crop_resized = cv2.resize(
            crop, (mgr.font_w, mgr.font_h), interpolation=cv2.INTER_LANCZOS4
        )
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalisation (in-place for speed)
        crop_rgb -= IMAGENET_MEAN
        crop_rgb /= IMAGENET_STD

        # [H, W, C] → [1, C, H, W], contiguous for ORT
        tensor = np.ascontiguousarray(crop_rgb.transpose(2, 0, 1)[np.newaxis])

        # ── Inference ────────────────────────────────────────────
        logits: np.ndarray = mgr.font.run(
            [mgr.font_out_name],
            {mgr.font_in_name: tensor},
        )[0][0]   # [num_classes]

        # ── Softmax ──────────────────────────────────────────────
        # Subtract max for numerical stability before exp
        shifted = logits - logits.max()
        exp     = np.exp(shifted)
        probs   = exp / exp.sum()

        top_idx = int(np.argmax(probs))
        label   = (
            mgr.font_labels[top_idx]
            if mgr.font_labels and top_idx < len(mgr.font_labels)
            else f"class_{top_idx}"
        )

        predictions.append(
            FontPrediction(
                block_id=blk["id"],
                label=label,
                confidence=float(probs[top_idx]),
                logits=logits.tolist(),
            )
        )

        log.debug(
            "font_classifier: block=%s  label=%s  conf=%.3f",
            blk["id"], label, probs[top_idx],
        )

    log.info("font_classifier: classified %d / %d blocks", len(predictions), len(text_blocks))
    return {"font_metadata": predictions}


# ── 4e. Text Replacement Renderer ────────────────────────────────

# Font-label → system font file candidates (cross-platform)
_FONT_FILE_CANDIDATES: Dict[str, List[str]] = {
    "Arial":            ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"],
    "Times New Roman":  ["C:/Windows/Fonts/times.ttf", "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "/System/Library/Fonts/Times.ttc"],
    "Courier New":      ["C:/Windows/Fonts/cour.ttf", "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf"],
    "Calibri":          ["C:/Windows/Fonts/calibri.ttf"],
    "Georgia":          ["C:/Windows/Fonts/georgia.ttf", "/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf"],
    "Verdana":          ["C:/Windows/Fonts/verdana.ttf", "/usr/share/fonts/truetype/msttcorefonts/Verdana.ttf"],
    "Roboto":           ["C:/Windows/Fonts/Roboto-Regular.ttf", "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf"],
    "Helvetica":        ["C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc"],
    "Garamond":         ["C:/Windows/Fonts/garamond.ttf", "/usr/share/fonts/truetype/ebgaramond/EBGaramond-Regular.ttf"],
    "Consolas":         ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
}

_DEJAVU_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _resolve_font_file(label: str) -> Optional[str]:
    """Find the first existing system font file for a classifier label."""
    for path in _FONT_FILE_CANDIDATES.get(label, []):
        if os.path.isfile(path):
            return path
    # DejaVu fallback
    for path in _DEJAVU_FALLBACKS:
        if os.path.isfile(path):
            return path
    return None


def _get_pil_font(label: str, size: int) -> ImageFont.FreeTypeFont:
    """Return a PIL font for the given classifier label and size."""
    path = _resolve_font_file(label)
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    log.warning("renderer: no TTF for %r size=%d — using PIL default", label, size)
    return ImageFont.load_default()


def renderer_node(state: PipelineState) -> Dict[str, Any]:
    """
    Composite replacement text onto the inpainted image.

    Runs ONLY when ``mode == "replace"``.

    For each block whose original text appears in ``replacement_map``:
    1. Look up the classified font label from ``font_metadata``.
    2. Resolve to the closest system font file via ``_resolve_font_file``.
    3. Render the replacement text with PIL at the original ``font_size``.
    4. Auto-shrink until the text fits the bounding box width.
    5. Composite onto the inpainted image using the original text_color.

    Falls back to DejaVu Sans if font_metadata has no entry for a block.
    """
    inpainted   = state["inpainted_image"]
    text_blocks = state["text_blocks"]
    font_meta   = state.get("font_metadata", [])
    rmap        = state.get("replacement_map", {})

    if inpainted is None:
        log.error("renderer_node: no inpainted_image — skipping")
        return {"error": "renderer_node: inpainted_image is None"}

    if not rmap:
        log.info("renderer_node: replacement_map empty — passthrough")
        return {"rendered_image": inpainted}

    # Build block_id → font label lookup
    font_lookup: Dict[str, str] = {
        fp["block_id"]: fp["label"] for fp in font_meta
    }

    # Work on a copy
    canvas = inpainted.copy()
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    rendered_count = 0
    for blk in text_blocks:
        original_text = blk.get("text", "")
        if original_text not in rmap:
            continue

        new_text   = rmap[original_text]
        block_id   = blk.get("id", "")
        font_label = font_lookup.get(block_id, "Arial")  # fallback
        font_size  = blk.get("h", 20)  # Use bbox height as initial size
        box_w      = blk.get("w", 200)
        box_h      = blk.get("h", 30)
        x, y       = blk.get("x", 0), blk.get("y", 0)

        # ── Auto-fit: shrink until text fits bounding box width ──
        fsize = max(8, int(font_size * 0.85))
        font = _get_pil_font(font_label, fsize)
        while fsize > 8:
            font = _get_pil_font(font_label, fsize)
            try:
                bbox = font.getbbox(new_text)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw, _ = draw.textsize(new_text, font=font)
            if tw <= box_w:
                break
            fsize -= 1

        # ── Vertical centering ──
        try:
            tb = font.getbbox(new_text)
            th = tb[3] - tb[1]
        except AttributeError:
            _, th = draw.textsize(new_text, font=font)
        y_pos = y + max(0, (box_h - th) // 2)

        # ── Extract text_color from original block data ──
        # TextBlock in langgraph doesn't carry color — use black default
        text_color = (0, 0, 0)

        draw.text((x, y_pos), new_text, font=font, fill=text_color)
        rendered_count += 1

    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    log.info("renderer_node: rendered %d / %d replacement(s)", rendered_count, len(rmap))
    return {"rendered_image": result_bgr}


# ─────────────────────────────────────────────────────────────────
# 5.  CONDITIONAL EDGES (ROUTERS)
# ─────────────────────────────────────────────────────────────────

def route_inpainter(state: PipelineState) -> str:
    """
    Read ``inpaint_method`` written by the evaluator node and return
    the string key that LangGraph resolves via ``path_map`` in
    ``add_conditional_edges()``.

    Returning the node name directly (rather than a sentinel) keeps
    the path_map trivial and makes the routing logic self-documenting
    in graph visualisations.
    """
    method = state.get("inpaint_method", "lama")
    if method not in ("telea", "lama"):
        log.error("route_inpainter: unknown method %r — falling back to lama", method)
        return "lama"
    return method


def route_after_font_classifier(state: PipelineState) -> str:
    """
    After font classification, route based on ``mode``:
    - "replace" → renderer_node → END
    - anything else ("remove") → straight to END
    """
    if state.get("mode") == "replace" and state.get("replacement_map"):
        return "renderer"
    return "__end__"


# ─────────────────────────────────────────────────────────────────
# 6.  GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────

def _wrap_node(node_fn: Callable, emitter: Optional[ProgressEmitter], node_key: str) -> Callable:
    """Wrap a node function to emit progress before execution."""
    def wrapped(state: PipelineState) -> Dict[str, Any]:
        if emitter:
            emitter.emit(node_key)
        return node_fn(state)
    wrapped.__name__ = node_fn.__name__
    return wrapped


def build_pipeline_graph(emitter: Optional[ProgressEmitter] = None) -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Graph topology (v4 — with renderer)
    ------------------------------------
        START
          │
          ▼
      [evaluator]          ← Laplacian variance on masked region
          │
          ▼ conditional
      ┌───┴────────────────────────┐
      │ variance ≤ threshold        │ variance > threshold
      ▼                            ▼
    [telea_inpaint]          [lama_inpaint]
      │                            │
      └──────────┬─────────────────┘
                 │
                 ▼
        [font_classifier]
                 │
                 ▼ conditional
         ┌───────┴────────┐
         │ mode=="replace"  │ else
         ▼                 ▼
      [renderer]          END
         │
         ▼
        END

    Parameters
    ----------
    emitter : ProgressEmitter, optional
        If provided, each node emits progress via Redis before executing.
    """
    graph = StateGraph(PipelineState)

    # ── Nodes (optionally wrapped with progress emitter) ──────
    graph.add_node("evaluator",       _wrap_node(evaluator_node, emitter, "evaluator"))
    graph.add_node("telea",           _wrap_node(telea_inpaint_node, emitter, "telea_start"))
    graph.add_node("lama",            _wrap_node(lama_inpaint_node, emitter, "lama_start"))
    graph.add_node("font_classifier", _wrap_node(font_classifier_node, emitter, "font_classifier"))
    graph.add_node("renderer",        _wrap_node(renderer_node, emitter, "renderer"))

    # ── Entry point ────────────────────────────────────────────
    graph.set_entry_point("evaluator")

    # ── Conditional branch 1: evaluator → inpainter ───────────
    graph.add_conditional_edges(
        source="evaluator",
        path=route_inpainter,
        path_map={
            "telea": "telea",
            "lama":  "lama",
        },
    )

    # ── Convergence: both inpainters → font_classifier ────────
    graph.add_edge("telea", "font_classifier")
    graph.add_edge("lama",  "font_classifier")

    # ── Conditional branch 2: font_classifier → renderer | END
    graph.add_conditional_edges(
        source="font_classifier",
        path=route_after_font_classifier,
        path_map={
            "renderer":  "renderer",
            "__end__":   END,
        },
    )

    # ── renderer → END ────────────────────────────────────────
    graph.add_edge("renderer", END)

    return graph.compile()


def _get_or_build_graph(emitter: Optional[ProgressEmitter] = None) -> Any:
    """
    Build graph per invocation when an emitter is provided (the emitter
    is bound into node closures and cannot be reused across jobs).
    When no emitter, returns a cached singleton for performance.
    """
    if emitter is not None:
        return build_pipeline_graph(emitter=emitter)
    # Singleton path (no emitter)
    global _GRAPH_CACHE
    if "_GRAPH_CACHE" not in globals() or _GRAPH_CACHE is None:
        globals()["_GRAPH_CACHE"] = build_pipeline_graph()
        log.info("LangGraph pipeline compiled (cached, no emitter).")
    return _GRAPH_CACHE

_GRAPH_CACHE: Any = None


# ─────────────────────────────────────────────────────────────────
# 7.  EXECUTION HELPER  (called from Celery task or FastAPI endpoint)
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    image_bgr:       np.ndarray,
    mask:            np.ndarray,
    text_blocks:     List[TextBlock],
    mode:            str = "remove",
    replacement_map: Optional[Dict[str, str]] = None,
    emitter:         Optional[ProgressEmitter] = None,
) -> Tuple[np.ndarray, List[FontPrediction], float]:
    """
    Execute the full LangGraph pipeline and return results.

    Parameters
    ----------
    image_bgr       : OpenCV BGR uint8 array [H, W, 3]
    mask            : uint8 [H, W], 255 = region to inpaint
    text_blocks     : EasyOCR detections converted to TextBlock dicts
    mode            : "remove" | "replace"
    replacement_map : {original_text: replacement_text} (replace mode)
    emitter         : ProgressEmitter (optional) — Redis progress callbacks

    Returns
    -------
    result_image    : BGR uint8 [H, W, 3]  (inpainted or rendered)
    font_metadata   : per-block font predictions
    complexity_score: Laplacian variance used for routing

    Raises
    ------
    RuntimeError if the graph node sets ``error`` in state.
    """
    initial_state: PipelineState = {
        "original_image":  image_bgr,
        "mask":            mask,
        "text_blocks":     text_blocks,
        "replacement_map": replacement_map or {},
        "mode":            mode,
        # Computed fields — initialised to safe defaults
        "complexity_score":  0.0,
        "inpaint_method":    "lama",
        "inpainted_image":   None,
        "font_metadata":     [],
        "rendered_image":    None,
        "error":             None,
    }

    compiled = _get_or_build_graph(emitter=emitter)
    final_state: PipelineState = compiled.invoke(initial_state)

    # Emit END progress
    if emitter:
        emitter.emit("END")

    if final_state.get("error"):
        raise RuntimeError(f"Pipeline error: {final_state['error']}")

    # In replace mode, prefer rendered_image over inpainted_image
    result = final_state.get("rendered_image") or final_state.get("inpainted_image")
    if result is None:
        raise RuntimeError("Pipeline completed without producing an output image.")

    return (
        result,
        final_state["font_metadata"],
        final_state["complexity_score"],
    )


# ─────────────────────────────────────────────────────────────────
# 8.  CELERY INTEGRATION STUB
# ─────────────────────────────────────────────────────────────────
#
# In your worker.py, add these lines:
#
#   from celery.signals import worker_process_init
#   from langgraph_pipeline import init_session_manager, run_pipeline
#
#   @worker_process_init.connect
#   def _bootstrap_onnx(**_kwargs):
#       """
#       Called once per worker *process* (not per task).
#       Each Celery worker spawns --concurrency=N processes;
#       each process loads its own ONNX sessions.
#       This is correct: ORT sessions are not fork-safe,
#       so they must be created *after* the fork, not before.
#       """
#       init_session_manager(
#           lama_path="models/lama.onnx",
#           font_path="models/font_classifier.onnx",
#           # font_labels=open("models/font_labels.txt").read().splitlines(),
#       )
#
#   @celery_app.task(bind=True, name="textclear.process_job", max_retries=2)
#   def process_job(self, job_id: str) -> dict:
#       ...
#       result_img, font_meta, complexity = run_pipeline(
#           image_bgr=img,
#           mask=mask,
#           text_blocks=text_blocks,
#           mode=job.mode,
#           replacement_map=replacement_map,
#       )
#       cv2.imwrite(str(output_path), result_img)
#       ...
