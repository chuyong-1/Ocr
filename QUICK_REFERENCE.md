# 🎯 QUICK REFERENCE: 10-CLASS FONT CLASSIFIER INTEGRATION

## 📄 FILES TO UPDATE

```
backend/
├── text_pipeline.py          ← COMPLETE REPLACEMENT (see text_pipeline_updated.py)
├── worker.py                 ← UPDATE with FontClassifier init (see worker_updated.py)
├── models/
│   └── font_classifier.onnx  ← PLACE YOUR TRAINED MODEL HERE
└── requirements.txt          ← ADD: onnxruntime
```

---

## 🔧 CODE CHANGES AT A GLANCE

### 1. text_pipeline.py

#### Add FontClassifier class (NEW)
```python
class FontClassifier:
    DEFAULT_LABELS = [
        "Arial", "Times New Roman", "Courier New", "Calibri", "Georgia",
        "Verdana", "Roboto", "Helvetica", "Garamond", "Consolas"
    ]
    
    def __init__(self, model_path=None, gpu=False):
        self._session = None
        self._initialized = False
    
    def _lazy_load(self) -> bool:
        # Loads ONNX model on first inference call
        # Returns False if model unavailable
        # Gracefully continues without it
    
    def predict(self, image_crop: np.ndarray) -> str:
        # Resizes crop to 64×64 float32
        # Runs inference: [1,1,64,64] → [1,10]
        # Returns font name or "sans-serif" on fallback
```

#### Update EditorBlock TypedDict (EXISTING)
```python
class EditorBlock(TypedDict):
    text: str
    x: int
    y: int
    w: int
    h: int
    color: str
    bg_color: str
    size: int
    confidence: float
    font_family: str  # ← ADD THIS LINE
```

#### Update TextRegion dataclass (EXISTING)
```python
@dataclass
class TextRegion:
    # ... existing fields ...
    font_family: str = "sans-serif"  # ← ADD THIS LINE
```

#### Update extract_for_editor() signature (EXISTING FUNCTION)
```python
def extract_for_editor(
    image_bgr: np.ndarray,
    languages: List[str] = None,
    confidence: float = 0.40,
    dilation_px: int = 8,
    inpaint_radius: int = 12,
    gpu: bool = False,
    font_classifier: Optional[FontClassifier] = None,  # ← ADD THIS PARAM
) -> Tuple[np.ndarray, List[EditorBlock]]:
```

#### Inside extract_for_editor(), add font prediction loop (NEW SECTION)
```python
# After StyleExtractor stage, add:
if font_classifier is None:
    font_classifier = FontClassifier(gpu=gpu)

for r in regions:
    try:
        crop_bgr = image_bgr[r.y:r.y+r.h, r.x:r.x+r.w]
        if crop_bgr.size > 0:
            r.font_family = font_classifier.predict(crop_bgr)
    except Exception as e:
        log.warning("Font classification error: %s", e)
        r.font_family = "sans-serif"
```

#### Update EditorBlock creation (EXISTING SECTION)
```python
blocks.append(EditorBlock(
    text = r.text,
    x = r.x,
    y = r.y,
    w = r.w,
    h = r.h,
    color = rgb_to_hex(r.text_color),
    bg_color = rgb_to_hex(r.bg_color),
    size = r.font_size,
    confidence = round(r.confidence, 4),
    font_family = r.font_family,  # ← ADD THIS LINE
))
```

---

### 2. worker.py

#### Inside process_job() task, add font classifier init (NEW SECTION)
```python
# After imports, inside process_job():
from text_pipeline import extract_for_editor, FontClassifier

# ... existing validation code ...

# ← NEW: Initialize FontClassifier
font_classifier = FontClassifier(gpu=False)

# Pass to extract_for_editor
cleaned, blocks = extract_for_editor(
    image_bgr = img,
    languages = languages,
    confidence = 0.40,
    dilation_px = 8,
    gpu = False,
    font_classifier = font_classifier,  # ← ADD THIS
)
```

#### Update meta_payload (EXISTING SECTION)
```python
meta_payload = {
    "bg_image": f"/results/{cleaned_name}",
    "image_w": int(img.shape[1]),
    "image_h": int(img.shape[0]),
    "blocks": blocks,  # Now includes font_family per block
}
```

#### Update return dict (OPTIONAL, for monitoring)
```python
return {
    "status": "DONE",
    "output_path": str(cleaned_path),
    "meta_path": str(meta_path),
    "block_count": len(blocks),
    "fonts": list(set(b.get("font_family") for b in blocks)),  # ← ADD
}
```

---

### 3. requirements.txt

#### Add ONNX Runtime
```txt
# Existing dependencies...
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
easyocr>=1.7.1
opencv-python-headless>=4.9.0
numpy>=1.26.0
Pillow>=10.3.0

# ← NEW: Add this line
onnxruntime>=1.17.0  # For 10-class font classification
```

---

## ✅ IMPLEMENTATION CHECKLIST

- [ ] Place trained model at `backend/models/font_classifier.onnx`
- [ ] Verify model outputs [1, 10] logits with dummy 64×64 input
- [ ] Copy `text_pipeline_updated.py` → `backend/text_pipeline.py`
- [ ] Copy `worker_updated.py` → `backend/worker.py`
- [ ] Update `requirements.txt` with `onnxruntime`
- [ ] Run unit tests (see INTEGRATION_GUIDE.md Phase 1)
- [ ] Run pipeline tests (Phase 2)
- [ ] Run worker tests (Phase 3)
- [ ] Verify `meta_*.json` includes `font_family` field
- [ ] Test graceful fallback (model missing/corrupted)
- [ ] Deploy and monitor logs

---

## 🧪 QUICK TEST

```bash
cd backend/

# Test 1: FontClassifier loads and falls back gracefully
python3 << 'EOF'
from text_pipeline import FontClassifier
import numpy as np

fc = FontClassifier(model_path="models/font_classifier.onnx")
crop = np.random.randint(0, 255, (45, 200, 3), dtype=np.uint8)
font = fc.predict(crop)
print(f"Predicted font: {font}")
assert font in fc.DEFAULT_LABELS or font == "sans-serif"
print("✓ PASS")
EOF

# Test 2: Extract-for-editor returns font_family
python3 << 'EOF'
import cv2
from text_pipeline import extract_for_editor, FontClassifier

# Use your test image
img = cv2.imread("test_images/sample.jpg")
if img is not None:
    fc = FontClassifier(model_path="models/font_classifier.onnx")
    cleaned, blocks = extract_for_editor(img, font_classifier=fc)
    for block in blocks:
        assert "font_family" in block, "Missing font_family!"
        print(f"  {block['text'][:20]:20s} → {block['font_family']}")
    print("✓ PASS")
EOF

# Test 3: JSON serialization
python3 << 'EOF'
import json
from text_pipeline import EditorBlock

block = EditorBlock(
    text="Test", x=0, y=0, w=100, h=20,
    color="#000000", bg_color="#FFFFFF", size=16, confidence=0.9,
    font_family="Arial"
)
json_str = json.dumps([block])
reloaded = json.loads(json_str)
assert "font_family" in reloaded[0], "font_family lost!"
print(f"JSON contains font_family: {reloaded[0]['font_family']}")
print("✓ PASS")
EOF
```

---

## 📊 KEY DESIGN DECISIONS

| Decision | Why | Impact |
|----------|-----|--------|
| Lazy loading in `_lazy_load()` | Keep worker startup fast | First inference: +200-500ms, subsequent: <5ms |
| FontClassifier as optional param | Decoupled from FastAPI | Can run without ONNX model; graceful fallback |
| 64×64 tensor shape | Standard for font classifiers | Handles variable input sizes |
| [1, 10] output shape | 10-class prediction | Bounds check catches training errors |
| Fallback to "sans-serif" | Offline resilience | Pipeline continues even if model missing |
| Dictionary-based EditorBlock | JSON serializable | Direct `json.dumps(blocks)` works |

---

## 🚨 CRITICAL VALIDATIONS

```python
# 1. Exact label order (order = model training order)
assert FontClassifier.DEFAULT_LABELS == [
    "Arial", "Times New Roman", "Courier New", "Calibri", "Georgia",
    "Verdana", "Roboto", "Helvetica", "Garamond", "Consolas"
], "Label order MUST match training!"

# 2. 10-class constraint
assert len(FontClassifier.DEFAULT_LABELS) == 10, "Must have exactly 10 fonts"

# 3. Bounds check in predict()
class_idx = int(np.argmax(logits[0]))
if class_idx < 0 or class_idx >= len(self.DEFAULT_LABELS):
    return "sans-serif"  # ← Safety net

# 4. EditorBlock has font_family
for block in blocks:
    assert "font_family" in block, "Missing field!"

# 5. JSON roundtrip
assert json.loads(json.dumps(blocks))[0]["font_family"] == original_font
```

---

## 🔗 INTEGRATION POINTS

```
FastAPI server (app.py)
    ↓
    Receives image
    ↓
TextPipeline.process_image()  ← Still uses old path for backward compat
    ↓
    (Now includes FontClassifier internally)

---

Celery Worker (worker.py)  ← PRIMARY PATH FOR FONT CLASSIFICATION
    ↓
    Receives job
    ↓
process_job()
    ↓
    Initialize FontClassifier (lazy-loads ONNX)
    ↓
extract_for_editor()  ← Calls font_classifier.predict() per region
    ↓
    Returns blocks WITH font_family
    ↓
meta_{job_id}.json  ← Contains font_family per block
    ↓
Frontend reads metadata
    ↓
User sees predicted fonts in editor
```

---

## 📦 MINIMAL DOCKER EXAMPLE

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY text_pipeline.py worker.py app.py ./
COPY models/ ./models/

# Create dirs
RUN mkdir -p uploads results

EXPOSE 8000
```

---

## 🎓 TROUBLESHOOTING MATRIX

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: onnxruntime` | Not installed | `pip install onnxruntime` |
| All fonts predicted as "sans-serif" | Model file missing | Place at `models/font_classifier.onnx` |
| `IndexError: index 15 out of bounds` | Model outputs wrong shape | Verify model: `sess.get_outputs()[0]` → [1, 10] |
| `font_family` field missing from JSON | Code not updated | Check `EditorBlock` definition |
| Worker crashes on second task | Memory leak in ONNX session | Sessions are reused; add explicit cleanup |
| Inference time > 100ms per region | ONNX provider config | Use `CPUExecutionProvider`, not CUDA |

---

## 📞 SUPPORT REFERENCE

**File locations:**
- Model: `backend/models/font_classifier.onnx`
- Pipeline: `backend/text_pipeline.py` (FontClassifier class at top)
- Worker: `backend/worker.py` (process_job function)

**Key classes:**
- `FontClassifier` — ONNX inference wrapper
- `EditorBlock(TypedDict)` — Serializable text region with fonts
- `extract_for_editor()` — Main entry point that calls FontClassifier

**Configuration:**
- GPU inference: `FontClassifier(gpu=True)`
- Custom model path: `FontClassifier(model_path="/custom/path/model.onnx")`
- Fallback font: Hardcoded to `"sans-serif"` in `.predict()`

---

**Version:** PixelScribe v3.0 (10-class Font Classification)  
**Last Updated:** 2026-05-18
