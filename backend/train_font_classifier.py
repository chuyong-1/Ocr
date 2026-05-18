#!/usr/bin/env python3
"""
train_font_classifier.py
════════════════════════════════════════════════════════════════════════════════
Synthetic Font Classifier — Training Script (10-Class)
Integrates with: chuyong-1/Ocr  (PixelScribe pipeline)

Pipeline
────────
  1. SyntheticDataGenerator   — Pillow text rendering on varied backgrounds
  2. DocumentAugmentor        — OpenCV scan-like augmentations
  3. FontDataset              — PyTorch Dataset wrapping generated samples
  4. FontClassifierCNN        — Lightweight 3-block CNN (CPU-optimised)
  5. Trainer                  — Train/Val loop with live metrics
  6. ONNXExporter             — torch.onnx.export → models/font_classifier.onnx

Output tensor signature (matches text_pipeline.py / app.py expectations):
  Input  → [1, 1, 64, 64]  float32  (normalised 0.0–1.0)
  Output → [1, 10]         float32  (raw logits per font class)

Font label order (index ↔ class):
  0 → Arial            5 → Verdana
  1 → Times New Roman  6 → Roboto
  2 → Courier New      7 → Helvetica
  3 → Calibri          8 → Garamond
  4 → Georgia          9 → Consolas

Usage
─────
  # Install extra deps first (project venv already has torch-free):
  pip install torch torchvision onnx opencv-python-headless pillow numpy

  python train_font_classifier.py [--samples 2000] [--epochs 15] [--batch 64]

  Output: models/font_classifier.onnx
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import string
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Lazy torch imports (deferred so the script gives a clean error if missing)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    sys.exit(
        "[FATAL] PyTorch not found.\n"
        "  pip install torch torchvision\n"
        "Then re-run this script."
    )

try:
    import onnx  # noqa: F401 — just verify presence; export uses torch.onnx
except ImportError:
    sys.exit(
        "[FATAL] onnx package not found.\n"
        "  pip install onnx\n"
        "Then re-run this script."
    )

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("font_classifier")

# ── Constants ─────────────────────────────────────────────────────────────────
FONT_LABELS: List[str] = [
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
NUM_CLASSES: int  = len(FONT_LABELS)   # 10
IMG_SIZE:    int  = 64          # both width and height
PATCH_W:     int  = 256         # synthetic text patch before resize
PATCH_H:     int  = 80

# Candidate system font paths keyed by logical name.
# Ordered: Linux → macOS → Windows
_FONT_CANDIDATES: dict[str, List[str]] = {
    "Arial": [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "Times New Roman": [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
        "/System/Library/Fonts/Times New Roman.ttf",
        "C:/Windows/Fonts/times.ttf",
    ],
    "Courier New": [
        "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Courier New.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ],
    "Calibri": [
        "/usr/share/fonts/truetype/msttcorefonts/Calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",    # fallback
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ],
    "Georgia": [
        "/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",   # fallback
        "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Georgia.ttf",
        "C:/Windows/Fonts/georgia.ttf",
    ],
    "Verdana": [
        "/usr/share/fonts/truetype/msttcorefonts/Verdana.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # fallback
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ],
    "Roboto": [
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/roboto/unhinted/Roboto-Regular.ttf",
        "/usr/share/fonts/google-roboto/Roboto-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Roboto-Regular.ttf",
        "C:/Windows/Fonts/Roboto-Regular.ttf",
    ],
    "Helvetica": [
        "/usr/share/fonts/truetype/msttcorefonts/Helvetica.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # fallback
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "C:/Windows/Fonts/arial.ttf",  # Helvetica → Arial fallback on Windows
    ],
    "Garamond": [
        "/usr/share/fonts/truetype/msttcorefonts/Garamond.ttf",
        "/usr/share/fonts/truetype/ebgaramond/EBGaramond-Regular.ttf",
        "/usr/share/fonts/opentype/ebgaramond/EBGaramond-Regular.otf",
        "/System/Library/Fonts/Supplemental/Garamond.ttf",
        "C:/Windows/Fonts/garamond.ttf",
    ],
    "Consolas": [
        "/usr/share/fonts/truetype/msttcorefonts/Consolas.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",  # fallback
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Consolas.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ],
}

# Google Fonts download fallbacks (TTF direct links) – used only if no
# local font is found. These are open-licensed substitutes.
_FONT_DOWNLOAD_URLS: dict[str, str] = {
    "Arial":           "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Regular.ttf",
    "Times New Roman": "https://github.com/google/fonts/raw/main/ofl/crimsontext/CrimsonText-Regular.ttf",
    "Courier New":     "https://github.com/google/fonts/raw/main/apache/robotomono/static/RobotoMono-Regular.ttf",
    "Calibri":         "https://github.com/google/fonts/raw/main/apache/nunito/static/Nunito-Regular.ttf",
    "Georgia":         "https://github.com/google/fonts/raw/main/ofl/lora/Lora-Regular.ttf",
    "Verdana":         "https://github.com/google/fonts/raw/main/ofl/sourcesans3/static/SourceSans3-Regular.ttf",
    "Roboto":          "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Regular.ttf",
    "Helvetica":       "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Regular.ttf",
    "Garamond":        "https://github.com/google/fonts/raw/main/ofl/ebgaramond/static/EBGaramond-Regular.ttf",
    "Consolas":        "https://github.com/google/fonts/raw/main/apache/robotomono/static/RobotoMono-Regular.ttf",
}

# Sample word pool for richer text variety
_WORD_POOL = (
    "the quick brown fox jumps over lazy dog "
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz "
    "0123456789 Hello World Python OpenCV PyTorch ONNX "
    "PixelScribe TextClear Font Scan Document OCR "
    "Image Processing Neural Network Classifier "
    "alpha beta gamma delta epsilon omega "
).split()

# ── Output directory ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"


# ════════════════════════════════════════════════════════════════════════════
# §1  FONT RESOLVER
# ════════════════════════════════════════════════════════════════════════════

class FontResolver:
    """
    Resolves a logical font name to a loadable TTF path.
    Falls back to downloading an open-licensed substitute if nothing
    is found locally.
    """

    _cache_dir: Path = Path(__file__).parent.parent / ".font_cache"

    @classmethod
    def resolve(cls, name: str) -> Optional[str]:
        """Return a path string for *name*, or None if all fallbacks fail."""
        # 1. Try system paths
        for candidate in _FONT_CANDIDATES.get(name, []):
            if Path(candidate).exists():
                log.debug("  Font %-20s → system  %s", name, candidate)
                return candidate

        # 2. Try local cache from a previous download
        cache_path = cls._cache_dir / f"{name.replace(' ', '_')}.ttf"
        if cache_path.exists():
            log.debug("  Font %-20s → cache   %s", name, cache_path)
            return str(cache_path)

        # 3. Download substitute
        url = _FONT_DOWNLOAD_URLS.get(name)
        if url:
            return cls._download(name, url, cache_path)

        log.warning("  Font %-20s → NOT FOUND (PIL default will be used)", name)
        return None

    @classmethod
    def _download(cls, name: str, url: str, dest: Path) -> Optional[str]:
        cls._cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("  Downloading substitute font for '%s' from:\n    %s", name, url)
        try:
            urllib.request.urlretrieve(url, dest)
            log.info("  ✓ Saved → %s", dest)
            return str(dest)
        except Exception as exc:
            log.warning("  Download failed for '%s': %s", name, exc)
            return None


# ════════════════════════════════════════════════════════════════════════════
# §2  SYNTHETIC DATA GENERATOR
# ════════════════════════════════════════════════════════════════════════════

class SyntheticDataGenerator:
    """
    Renders synthetic text patches using Pillow for each target font.

    Design choices
    ──────────────
    • Random text strings (words, letters, digits) maximise glyph variety.
    • Three background modes (white, off-white, light grey) prevent the
      model from learning background colour as a proxy feature.
    • Font sizes 18–36 px cover the typical on-screen text range.
    • Returns raw uint8 numpy arrays (H=PATCH_H, W=PATCH_W, C=3).
    """

    def __init__(self, font_size_range: Tuple[int, int] = (18, 36)):
        self.font_size_range = font_size_range
        self._pil_fonts: dict[str, list] = {}   # name → list of PIL fonts

        log.info("Resolving fonts…")
        for label in FONT_LABELS:
            path = FontResolver.resolve(label)
            fonts = []
            for size in range(font_size_range[0], font_size_range[1] + 1, 4):
                try:
                    if path:
                        fonts.append(ImageFont.truetype(path, size=size))
                    else:
                        fonts.append(ImageFont.load_default())
                except OSError:
                    fonts.append(ImageFont.load_default())
            self._pil_fonts[label] = fonts
            log.info("  ✓ %-20s  %d font sizes loaded", label, len(fonts))

    # ── Background generators ────────────────────────────────────────────
    @staticmethod
    def _random_background() -> Tuple[int, int, int]:
        """Return an RGB background colour: white, off-white, or light grey."""
        choice = random.random()
        if choice < 0.40:
            return (255, 255, 255)                          # pure white
        elif choice < 0.70:
            v = random.randint(240, 252)
            return (v, v - random.randint(0, 8), v - random.randint(0, 6))  # off-white
        else:
            v = random.randint(200, 235)
            return (v, v, v)                                # light grey

    @staticmethod
    def _random_text() -> str:
        """Return a random word/character string for rendering."""
        mode = random.random()
        if mode < 0.35:
            # Random single word from pool
            return random.choice(_WORD_POOL)
        elif mode < 0.65:
            # Short phrase (2–4 words)
            return " ".join(random.choices(_WORD_POOL, k=random.randint(2, 4)))
        elif mode < 0.82:
            # Random uppercase letters
            return "".join(random.choices(string.ascii_uppercase, k=random.randint(4, 10)))
        else:
            # Mixed alphanumeric
            chars = string.ascii_letters + string.digits
            return "".join(random.choices(chars, k=random.randint(5, 12)))

    # ── Core render ─────────────────────────────────────────────────────
    def generate(self, label: str) -> np.ndarray:
        """
        Render one text patch for *label*.
        Returns: uint8 numpy array  (PATCH_H × PATCH_W × 3)
        """
        bg_color   = self._random_background()
        text_str   = self._random_text()
        font       = random.choice(self._pil_fonts[label])

        img  = Image.new("RGB", (PATCH_W, PATCH_H), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Random dark text colour (very dark, near-black)
        darkness   = random.randint(0, 60)
        text_color = (darkness, darkness, darkness)

        # Centre text in patch
        try:
            bbox = draw.textbbox((0, 0), text_str, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text_str, font=font)   # PIL < 9.2

        x = max(0, (PATCH_W - tw) // 2)
        y = max(0, (PATCH_H - th) // 2)

        draw.text((x, y), text_str, font=font, fill=text_color)

        return np.array(img, dtype=np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# §3  DOCUMENT AUGMENTOR
# ════════════════════════════════════════════════════════════════════════════

class DocumentAugmentor:
    """
    Applies scan-like augmentations via OpenCV to each generated patch.

    Each call applies a randomised chain of:
      ① Slight rotation      (−5°…+5°)
      ② Blur                 (Gaussian or Bilateral)
      ③ Noise                (salt-and-pepper)
      ④ Contrast / brightness jitter
      ⑤ Resize to 64×64 and convert to greyscale float32

    Returns: float32 numpy array  (64 × 64), normalised 0.0–1.0
    """

    # ── ① Rotation ────────────────────────────────────────────────────────
    @staticmethod
    def _rotate(img: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
        angle = random.uniform(-max_angle, max_angle)
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # Fill border with near-white background colour
        bg    = int(img.mean()) if img.mean() > 127 else 240
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(bg, bg, bg),
        )

    # ── ② Blur ────────────────────────────────────────────────────────────
    @staticmethod
    def _blur(img: np.ndarray) -> np.ndarray:
        choice = random.random()
        if choice < 0.40:
            # Gaussian — simulates out-of-focus scan
            k = random.choice([3, 5])
            sigma = random.uniform(0.5, 1.5)
            return cv2.GaussianBlur(img, (k, k), sigma)
        elif choice < 0.65:
            # Bilateral — preserves edges while smoothing flat areas
            d  = random.randint(5, 9)
            sc = random.randint(20, 60)
            ss = random.randint(20, 60)
            return cv2.bilateralFilter(img, d, sc, ss)
        else:
            # No blur
            return img

    # ── ③ Salt-and-pepper noise ───────────────────────────────────────────
    @staticmethod
    def _noise(img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            return img   # 50% chance: skip noise
        out   = img.copy()
        amount = random.uniform(0.002, 0.015)
        n_total = int(amount * img.size)

        # Salt (white pixels)
        coords = [np.random.randint(0, d, n_total // 2) for d in img.shape[:2]]
        out[coords[0], coords[1]] = 255

        # Pepper (black pixels)
        coords = [np.random.randint(0, d, n_total // 2) for d in img.shape[:2]]
        out[coords[0], coords[1]] = 0

        return out

    # ── ④ Contrast / brightness jitter ────────────────────────────────────
    @staticmethod
    def _contrast_jitter(img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.75, 1.25)   # contrast
        beta  = random.uniform(-20, 20)       # brightness
        return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # ── Full augmentation chain ────────────────────────────────────────────
    def augment(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply augmentation chain.
        Input:  uint8 (H×W×3) RGB array
        Output: float32 (64×64) greyscale, normalised 0–1
        """
        img = img_rgb.copy()

        # ① Rotation
        img = self._rotate(img)

        # ② Blur
        img = self._blur(img)

        # ③ Noise
        img = self._noise(img)

        # ④ Contrast/brightness jitter
        img = self._contrast_jitter(img)

        # ⑤ Convert to greyscale, resize to 64×64
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_AREA)

        # Normalise to [0.0, 1.0]
        return gray.astype(np.float32) / 255.0


# ════════════════════════════════════════════════════════════════════════════
# §4  PYTORCH DATASET
# ════════════════════════════════════════════════════════════════════════════

class FontDataset(Dataset):
    """
    Generates *samples_per_class* examples for each of the 5 font classes
    and stores them in-memory as pre-augmented tensors.

    Keeping data in memory (rather than re-generating each epoch) ensures
    fast DataLoader iteration and reproducible train/val splits.
    """

    def __init__(
        self,
        generator:   SyntheticDataGenerator,
        augmentor:   DocumentAugmentor,
        samples_per_class: int = 2000,
    ):
        self.samples: List[Tuple[np.ndarray, int]] = []

        log.info("Generating synthetic dataset  (%d samples × %d classes)…",
                 samples_per_class, NUM_CLASSES)
        t0 = time.time()

        for class_idx, label in enumerate(FONT_LABELS):
            count = 0
            for _ in range(samples_per_class):
                patch = generator.generate(label)
                arr   = augmentor.augment(patch)
                self.samples.append((arr, class_idx))
                count += 1

            log.info("  [%d/%d] %-20s  %d samples generated",
                     class_idx + 1, NUM_CLASSES, label, count)

        random.shuffle(self.samples)
        log.info("Dataset ready  (%d total)  in %.1fs",
                 len(self.samples), time.time() - t0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        arr, label = self.samples[idx]
        # Shape: (1, 64, 64)  — add channel dim
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, label


# ════════════════════════════════════════════════════════════════════════════
# §5  CNN MODEL
# ════════════════════════════════════════════════════════════════════════════

class FontClassifierCNN(nn.Module):
    """
    Lightweight 3-block CNN optimised for CPU inference.

    Architecture overview
    ─────────────────────
    Input: [B, 1, 64, 64]

    Block 1: Conv(1→32, 3×3)  → BN → ReLU → Conv(32→32, 3×3) → BN → ReLU
             → MaxPool(2×2)                                → [B, 32, 30, 30]

    Block 2: Conv(32→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU
             → MaxPool(2×2)                                → [B, 64, 13, 13]

    Block 3: Conv(64→128, 3×3) → BN → ReLU
             → AdaptiveAvgPool(4×4)                        → [B, 128, 4, 4]

    Head:    Flatten → Dropout(0.4) → FC(2048→256) → ReLU
             → Dropout(0.2) → FC(256→10)

    ~1.1 M parameters — fits easily in RAM; ~10 ms per image on CPU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Convolutional blocks ─────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64→30 (floor)
            nn.Dropout2d(p=0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 30→13
            nn.Dropout2d(p=0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),             # → 4×4 regardless of input
        )

        # ── Classification head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        # ── Weight initialisation (Kaiming for conv, Xavier for linear) ──
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════════════════
# §6  TRAINER
# ════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Encapsulates the full train / validation loop.

    Hyperparameters
    ───────────────
    • Optimiser:  Adam (lr=1e-3, weight_decay=1e-4)
    • Scheduler:  CosineAnnealingLR (restarts at T_max=epochs)
    • Loss:       CrossEntropyLoss (logits input)
    • Split:      80 % train / 20 % validation
    """

    def __init__(
        self,
        model:      FontClassifierCNN,
        dataset:    FontDataset,
        epochs:     int = 15,
        batch_size: int = 64,
        lr:         float = 1e-3,
        device:     str = "cpu",
    ):
        self.model  = model.to(device)
        self.device = device
        self.epochs = epochs

        # ── Split dataset ────────────────────────────────────────────────
        n_total = len(dataset)
        n_val   = max(int(n_total * 0.20), NUM_CLASSES)
        n_train = n_total - n_val

        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        log.info("Train: %d  |  Val: %d  |  Batch: %d", n_train, n_val, batch_size)

        # num_workers=0 for Windows compatibility; increase on Linux if slow
        nw = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=(device == "cuda"),
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=(device == "cuda"),
        )

        # ── Optimiser & scheduler ────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=epochs, eta_min=1e-5
        )

    # ── One training epoch ────────────────────────────────────────────────
    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimiser.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping — prevents rare exploding-gradient spikes
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimiser.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    # ── Validation pass ───────────────────────────────────────────────────
    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits       = self.model(imgs)
                loss         = self.criterion(logits, labels)

                total_loss += loss.item() * imgs.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += imgs.size(0)

        return total_loss / total, correct / total

    # ── Per-class accuracy report ─────────────────────────────────────────
    def _class_report(self):
        self.model.eval()
        per_class_correct = [0] * NUM_CLASSES
        per_class_total   = [0] * NUM_CLASSES

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs   = imgs.to(self.device)
                logits = self.model(imgs)
                preds  = logits.argmax(dim=1).cpu()
                for p, t in zip(preds.numpy(), labels.numpy()):
                    per_class_total[t] += 1
                    if p == t:
                        per_class_correct[t] += 1

        log.info("── Per-class validation accuracy ──────────────────────────")
        for idx, label in enumerate(FONT_LABELS):
            n  = per_class_total[idx]
            ok = per_class_correct[idx]
            acc = (ok / n * 100) if n > 0 else 0.0
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            log.info("  [%d] %-20s  %s  %5.1f%%  (%d/%d)",
                     idx, label, bar, acc, ok, n)
        log.info("────────────────────────────────────────────────────────────")

    # ── Full training run ─────────────────────────────────────────────────
    def run(self) -> FontClassifierCNN:
        log.info("═" * 60)
        log.info("  Starting training on device: %s", self.device.upper())
        log.info("  Model parameters: {:,}".format(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ))
        log.info("═" * 60)

        best_val_acc    = 0.0
        best_state_dict = None

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._val_epoch()
            self.scheduler.step()

            lr_now  = self.optimiser.param_groups[0]["lr"]
            elapsed = time.time() - t0

            log.info(
                "Epoch %3d/%d │ "
                "Train loss: %.4f  acc: %5.1f%%  │  "
                "Val loss: %.4f  acc: %5.1f%%  │  "
                "LR: %.2e  │  %.1fs",
                epoch, self.epochs,
                train_loss, train_acc * 100,
                val_loss,   val_acc   * 100,
                lr_now, elapsed,
            )

            # Track best checkpoint (in-memory — no disk write)
            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = {k: v.clone()
                                   for k, v in self.model.state_dict().items()}
                log.info("  ✓ New best val accuracy: %.1f%%", best_val_acc * 100)

        log.info("═" * 60)
        log.info("  Training complete.  Best val accuracy: %.1f%%",
                 best_val_acc * 100)
        log.info("═" * 60)

        # Restore best weights
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            log.info("  Best weights restored.")

        # Per-class breakdown
        self._class_report()

        return self.model


# ════════════════════════════════════════════════════════════════════════════
# §7  ONNX EXPORTER
# ════════════════════════════════════════════════════════════════════════════

class ONNXExporter:
    """
    Exports a trained FontClassifierCNN to ONNX format.

    The exported model is verified with onnx.checker before saving
    so that downstream inference engines receive a valid graph.

    Tensor signature (matches PixelScribe pipeline):
      Input  : "input"   [1, 1, 64, 64]  float32
      Output : "output"  [1, 10]          float32  (raw logits)
    """

    def __init__(self, model: FontClassifierCNN, output_dir: Path = MODELS_DIR):
        self.model      = model
        self.output_dir = output_dir
        self.onnx_path  = output_dir / "font_classifier.onnx"

    def export(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── CRITICAL: ONNX tracing requires model AND dummy input on the
        #   same device.  ONNX/ONNXRuntime targets CPU inference, so we
        #   always move the model to CPU before export regardless of what
        #   device it was trained on.  This does NOT affect the saved
        #   weights — it only changes where the trace graph is built.
        self.model = self.model.cpu()
        self.model.eval()

        # Dummy input — explicitly on CPU to match the model above
        dummy_input = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE,
                                  dtype=torch.float32, device="cpu")

        log.info("Exporting model to ONNX…")
        log.info("  Output path : %s", self.onnx_path)
        log.info("  Input  shape: %s", list(dummy_input.shape))
        log.info("  Model device: %s (moved to CPU for export)",
                 next(self.model.parameters()).device)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(self.onnx_path),
            export_params   = True,
            opset_version   = 17,          # opset 17 supported by onnxruntime ≥1.15
            do_constant_folding = True,    # fold BN into conv for faster inference
            input_names     = ["input"],
            output_names    = ["output"],
            dynamic_axes    = {
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        log.info("ONNX export complete.")

        # ── Verify the ONNX graph is well-formed ────────────────────────
        try:
            import onnx as _onnx
            model_proto = _onnx.load(str(self.onnx_path))
            _onnx.checker.check_model(model_proto)
            log.info("  ✓ ONNX checker: model is valid.")

            # Print graph summary
            size_mb = self.onnx_path.stat().st_size / 1024 / 1024
            log.info("  File size: %.2f MB", size_mb)
            log.info("  Opset:     %d", model_proto.opset_import[0].version)
        except Exception as exc:
            log.warning("  ONNX checker raised: %s  (export may still work)", exc)

        # ── Smoke-test with onnxruntime if available ─────────────────────
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )
            test_input  = np.zeros((1, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            test_output = sess.run(["output"], {"input": test_input})[0]
            log.info("  ✓ ONNXRuntime smoke-test passed. Output shape: %s",
                     list(test_output.shape))
        except ImportError:
            log.info("  (onnxruntime not installed — skipping live inference test)")
        except Exception as exc:
            log.warning("  ONNXRuntime smoke-test raised: %s", exc)

        log.info("═" * 60)
        log.info("  ✓ Model saved to: %s", self.onnx_path.resolve())
        log.info("  Label order: %s", FONT_LABELS)
        log.info("  Input sig:  [1, 1, 64, 64]  float32  (0.0–1.0)")
        log.info("  Output sig: [1, %d]          float32  (logits)", NUM_CLASSES)
        log.info("═" * 60)

        return self.onnx_path


# ════════════════════════════════════════════════════════════════════════════
# §8  INTEGRATION HELPERS  (for use inside text_pipeline.py / app.py)
# ════════════════════════════════════════════════════════════════════════════

def load_font_classifier(
    onnx_path: Optional[str] = None,
) -> "onnxruntime.InferenceSession":
    """
    Convenience loader — call this from text_pipeline.py to get a
    session ready for inference.

    Usage in text_pipeline.py
    ─────────────────────────
        from train_font_classifier import load_font_classifier, predict_font
        _font_clf = load_font_classifier()   # singleton

        region.font_name = predict_font(_font_clf, crop_gray_uint8)

    Parameters
    ──────────
    onnx_path : str or None
        Path to the ONNX file.  Defaults to models/font_classifier.onnx
        relative to this script's directory.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnxruntime  to use load_font_classifier()")

    path = onnx_path or str(MODELS_DIR / "font_classifier.onnx")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"ONNX model not found at {path}\n"
            "Run train_font_classifier.py first to generate it."
        )

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    log.info("FontClassifier loaded from %s", path)
    return sess


def predict_font(
    session: "onnxruntime.InferenceSession",
    crop_gray: np.ndarray,
) -> str:
    """
    Predict the font label for a greyscale crop.

    Parameters
    ──────────
    session   : onnxruntime.InferenceSession returned by load_font_classifier()
    crop_gray : uint8 greyscale numpy array, any size — will be resized to 64×64

    Returns
    ───────
    str  — one of FONT_LABELS, e.g. "Arial"
    """
    resized = cv2.resize(crop_gray.astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    tensor  = resized.astype(np.float32) / 255.0
    tensor  = tensor[np.newaxis, np.newaxis, :, :]   # [1, 1, 64, 64]

    logits  = session.run(["output"], {"input": tensor})[0]   # [1, 10]
    idx     = int(np.argmax(logits, axis=1)[0])
    if idx < 0 or idx >= len(FONT_LABELS):
        return "sans-serif"
    return FONT_LABELS[idx]


# ════════════════════════════════════════════════════════════════════════════
# §9  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_font_classifier",
        description="Synthetic font classifier — train & export to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # Quick test run (fewer samples, fewer epochs):
  python train_font_classifier.py --samples 500 --epochs 5

  # Full production training:
  python train_font_classifier.py --samples 2000 --epochs 20 --batch 128

  # GPU training (if CUDA available):
  python train_font_classifier.py --device cuda
        """,
    )
    p.add_argument("--samples", type=int, default=2000,
                   help="Synthetic samples per font class (default: 2000)")
    p.add_argument("--epochs",  type=int, default=15,
                   help="Training epochs (default: 15)")
    p.add_argument("--batch",   type=int, default=64,
                   help="Mini-batch size (default: 64)")
    p.add_argument("--lr",      type=float, default=1e-3,
                   help="Adam learning rate (default: 0.001)")
    p.add_argument("--device",  type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Training device (default: auto-detect)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--out-dir", type=str, default=str(MODELS_DIR),
                   help="Output directory for .onnx file")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║          PixelScribe — Font Classifier Trainer           ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Samples/class: %-5d   Epochs: %-4d   Batch: %-4d      ║",
             args.samples, args.epochs, args.batch)
    log.info("║  Device: %-10s   Seed:   %-4d                     ║",
             args.device, args.seed)
    log.info("╚══════════════════════════════════════════════════════════╝")

    # ── 1. Build generator + augmentor ──────────────────────────────────
    generator = SyntheticDataGenerator()
    augmentor  = DocumentAugmentor()

    # ── 2. Generate dataset ──────────────────────────────────────────────
    dataset = FontDataset(
        generator          = generator,
        augmentor          = augmentor,
        samples_per_class  = args.samples,
    )

    # ── 3. Build model ───────────────────────────────────────────────────
    model = FontClassifierCNN(num_classes=NUM_CLASSES)

    # ── 4. Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model      = model,
        dataset    = dataset,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        device     = args.device,
    )
    trained_model = trainer.run()

    # ── 5. Export to ONNX ─────────────────────────────────────────────────
    exporter = ONNXExporter(trained_model, output_dir=Path(args.out_dir))
    onnx_path = exporter.export()

    log.info("Done!  ONNX model at: %s", onnx_path)
    log.info("")
    log.info("To use in text_pipeline.py:")
    log.info("  from train_font_classifier import load_font_classifier, predict_font")
    log.info("  session = load_font_classifier()")
    log.info("  label   = predict_font(session, crop_gray_uint8)   # → 'Arial' etc.")


if __name__ == "__main__":
    main()