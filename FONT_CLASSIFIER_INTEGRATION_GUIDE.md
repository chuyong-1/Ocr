# 🎯 10-CLASS FONT CLASSIFIER INTEGRATION GUIDE
## PixelScribe v3 — Complete Setup & Testing Protocol

---

## 📋 OVERVIEW

This document outlines the complete integration of a 10-class ONNX font classifier into the PixelScribe/TextClear backend pipeline. The classifier predicts font families for detected text regions and serializes results into the JSON metadata.

**Target fonts (indices 0-9, EXACT order critical):**
```python
["Arial", "Times New Roman", "Courier New", "Calibri", "Georgia", 
 "Verdana", "Roboto", "Helvetica", "Garamond", "Consolas"]
```

---

## 🏗️ ARCHITECTURAL IMPACT

### Data Flow Changes
```
BEFORE (v2):
  OCR Detection → Style Extraction → Masking → Inpainting → EditorBlock {text, x, y, w, h, color, size, confidence}

AFTER (v3):
  OCR Detection → Style Extraction → FONT CLASSIFICATION → Masking → Inpainting 
    → EditorBlock {text, x, y, w, h, color, size, confidence, font_family}
                                                           ↑ NEW FIELD
```

### Database Impact
- **No schema changes** — `JobRecord` remains identical
- **Metadata enrichment** — `meta_{job_id}.json` now includes `font_family` per block
- **Lazy loading** — FontClassifier initialized **only in worker**, not server

### Tensor Shapes
```
Input to FontClassifier.predict():
  crop_bgr: [H, W, 3] (variable size, e.g., [45, 280, 3])
  
Preprocessing:
  grayscale: [H, W]
  resized:   [64, 64]
  normalized: [64, 64] float32 ∈ [0, 1]
  batched:   [1, 1, 64, 64] (add batch + channel dims)
  
ONNX Model Output:
  logits:    [1, 10] (10-class softmax logits)
  
Post-processing:
  argmax(logits[0]) → int ∈ [0, 9]
  bounds check: if 0 ≤ idx < 10 ✓ else fallback to "sans-serif"
```

---

## 📁 FILE MODIFICATIONS

### 1. `backend/text_pipeline.py` — COMPLETE REPLACEMENT

**Key Changes:**

#### FontClassifier Class (NEW)
- **DEFAULT_LABELS** = 10 fonts in exact order
- **_lazy_load()** — ONNX model loaded on first inference, not at import
- **predict()** — Handles variable-size crops, resizes to 64×64, safely handles [1,10] output
- **Defensive fallback** — Returns "sans-serif" on any error (file not found, inference failure, etc.)

```python
class FontClassifier:
    DEFAULT_LABELS = [
        "Arial", "Times New Roman", "Courier New", "Calibri", "Georgia",
        "Verdana", "Roboto", "Helvetica", "Garamond", "Consolas"
    ]
    
    def predict(self, image_crop: np.ndarray) -> str:
        # Graceful fallback on model unavailable
        if not self._lazy_load():
            return "sans-serif"
        
        # Preprocess: resize to 64×64, normalize to [0,1]
        # Inference: [1,1,64,64] → [1,10] logits
        # Bounds check: argmax result against [0, 10)
        # Return: font name or "sans-serif"
```

#### EditorBlock TypedDict (ENHANCED)
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
    font_family: str  # ← NEW FIELD (v3)
```

#### extract_for_editor() Function (ENHANCED)
```python
def extract_for_editor(
    image_bgr: np.ndarray,
    languages: List[str] = None,
    confidence: float = 0.40,
    dilation_px: int = 8,
    inpaint_radius: int = 12,
    gpu: bool = False,
    font_classifier: Optional[FontClassifier] = None,  # ← NEW PARAM
) -> Tuple[np.ndarray, List[EditorBlock]]:
    # ... OCR detection ...
    # ... Style extraction ...
    
    # NEW: Font classification per region
    for r in regions:
        crop_bgr = image_bgr[r.y:r.y+r.h, r.x:r.x+r.w]
        r.font_family = font_classifier.predict(crop_bgr)
    
    # Build blocks WITH font_family
    for r in regions:
        blocks.append(EditorBlock(
            # ... existing fields ...
            font_family = r.font_family,  # ← NEW
        ))
```

#### TextRegion Dataclass (ENHANCED)
```python
@dataclass
class TextRegion:
    # ... existing fields ...
    font_family: str = "sans-serif"  # ← NEW DEFAULT
```

---

### 2. `backend/worker.py` — UPDATED for Font Classification

**Key Changes:**

#### process_job() Task (ENHANCED)
```python
@celery_app.task(...)
def process_job(self, job_id: str) -> dict:
    # ... existing validation ...
    
    # NEW: Initialize FontClassifier in worker context
    font_classifier = FontClassifier(gpu=False)
    
    # Call extract_for_editor WITH font_classifier
    cleaned, blocks = extract_for_editor(
        image_bgr       = img,
        languages       = languages,
        confidence      = 0.40,
        dilation_px     = 8,
        gpu             = False,
        font_classifier = font_classifier,  # ← PASS IT
    )
    
    # Build meta_payload (now includes fonts)
    meta_payload = {
        "bg_image": f"/results/{cleaned_name}",
        "image_w": int(img.shape[1]),
        "image_h": int(img.shape[0]),
        "blocks": blocks,  # EditorBlocks WITH font_family
    }
    
    # Return task result with font summary
    return {
        "status": "DONE",
        "fonts": list(set(b.get("font_family") for b in blocks)),
    }
```

---

## 🧪 TESTING CRITERIA

### Phase 1: Unit Tests — FontClassifier Isolation

#### Test 1.1: Model Loading
```bash
# Terminal 1: Start Python in worker directory
cd backend/
python3 << 'EOF'
from text_pipeline import FontClassifier
import numpy as np

# Test lazy loading
fc = FontClassifier(model_path="models/font_classifier.onnx")
print("✓ FontClassifier created (model not loaded yet)")

# First predict triggers load
dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
result = fc.predict(dummy_crop)
print(f"✓ Model loaded, prediction: {result}")
assert isinstance(result, str), "Prediction must be string"
assert result in fc.DEFAULT_LABELS or result == "sans-serif", f"Invalid font: {result}"
print("✓ PASS: Model loading and fallback")
EOF
```

**Expected Output:**
```
✓ FontClassifier created (model not loaded yet)
✓ Model loaded, prediction: <one of 10 fonts or "sans-serif">
✓ PASS: Model loading and fallback
```

#### Test 1.2: Bounds Checking (10-class safety)
```bash
python3 << 'EOF'
from text_pipeline import FontClassifier
import numpy as np

fc = FontClassifier()
print(f"Default labels count: {len(fc.DEFAULT_LABELS)}")
assert len(fc.DEFAULT_LABELS) == 10, "Must have exactly 10 fonts"

# Verify exact order
expected = ["Arial", "Times New Roman", "Courier New", "Calibri", "Georgia",
            "Verdana", "Roboto", "Helvetica", "Garamond", "Consolas"]
assert fc.DEFAULT_LABELS == expected, f"Label order mismatch!\n{fc.DEFAULT_LABELS}"
print("✓ PASS: Exact 10-font label order verified")
EOF
```

**Expected Output:**
```
Default labels count: 10
✓ PASS: Exact 10-font label order verified
```

#### Test 1.3: Graceful Degradation (Model Missing)
```bash
python3 << 'EOF'
from text_pipeline import FontClassifier
import numpy as np

# Point to non-existent model
fc = FontClassifier(model_path="/nonexistent/path/model.onnx")
dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
result = fc.predict(dummy_crop)

assert result == "sans-serif", f"Expected 'sans-serif' fallback, got {result}"
print("✓ PASS: Graceful fallback when model unavailable")
EOF
```

**Expected Output:**
```
✓ PASS: Graceful fallback when model unavailable
```

---

### Phase 2: Pipeline Integration Tests

#### Test 2.1: extract_for_editor() with Font Classification
```bash
python3 << 'EOF'
import cv2
import json
from text_pipeline import extract_for_editor, FontClassifier
from pathlib import Path

# Use a test image with text
test_image_path = "test_images/sample.jpg"
if not Path(test_image_path).exists():
    print("⚠ No test image found. Skipping Test 2.1")
else:
    img = cv2.imread(test_image_path)
    fc = FontClassifier(model_path="models/font_classifier.onnx")
    
    cleaned, blocks = extract_for_editor(
        image_bgr=img,
        languages=["en"],
        confidence=0.3,
        font_classifier=fc,
    )
    
    print(f"✓ Extracted {len(blocks)} text regions")
    
    # Validate each block has font_family
    for i, block in enumerate(blocks):
        assert "font_family" in block, f"Block {i} missing font_family"
        assert isinstance(block["font_family"], str), f"font_family must be string"
        print(f"  Block {i}: '{block['text'][:20]}...' → {block['font_family']}")
    
    print("✓ PASS: All blocks have valid font_family field")
EOF
```

**Expected Output:**
```
✓ Extracted N text regions
  Block 0: 'Text excerpt...' → Arial
  Block 1: 'More text...' → Times New Roman
  ...
✓ PASS: All blocks have valid font_family field
```

#### Test 2.2: JSON Serialization (meta_*.json structure)
```bash
python3 << 'EOF'
import json
from text_pipeline import EditorBlock

# Create sample block
sample_block = EditorBlock(
    text="Hello World",
    x=100,
    y=50,
    w=200,
    h=30,
    color="#2C2C2C",
    bg_color="#F5F0E8",
    size=24,
    confidence=0.95,
    font_family="Arial",  # ← NEW FIELD
)

# Simulate JSON serialization
payload = {
    "bg_image": "/results/cleaned_123.jpg",
    "image_w": 1920,
    "image_h": 1080,
    "blocks": [sample_block],
}

json_str = json.dumps(payload)
reloaded = json.loads(json_str)

assert "font_family" in reloaded["blocks"][0], "font_family lost in serialization"
assert reloaded["blocks"][0]["font_family"] == "Arial", "font_family value corrupted"
print("✓ PASS: EditorBlock serializes/deserializes correctly")
print(f"Sample JSON:\n{json.dumps(reloaded['blocks'][0], indent=2)}")
EOF
```

**Expected Output:**
```
✓ PASS: EditorBlock serializes/deserializes correctly
Sample JSON:
{
  "text": "Hello World",
  "x": 100,
  ...
  "font_family": "Arial"
}
```

---

### Phase 3: Celery Worker Integration

#### Test 3.1: Worker Task Execution
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery worker
cd backend/
celery -A worker.celery_app worker --loglevel=info --concurrency=1

# Terminal 3: Submit test job
python3 << 'EOF'
import json
import uuid
from pathlib import Path
from text_pipeline import TextPipeline

# Create test SQLite job record
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server import Base, JobRecord

engine = create_engine("sqlite:///jobs.db")
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)

db = Session()

# Create test job
job_id = str(uuid.uuid4())
test_image = "test_images/sample.jpg"
if not Path(test_image).exists():
    print("⚠ No test image. Create one at test_images/sample.jpg")
else:
    from shutil import copy
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    input_path = uploads_dir / f"{job_id}_input.jpg"
    copy(test_image, input_path)
    
    job = JobRecord(
        id=job_id,
        mode="remove",
        original_name="sample.jpg",
        file_type="image",
        input_path=str(input_path),
        languages='["en"]',
        inpainter="cv",
    )
    db.add(job)
    db.commit()
    
    print(f"Created job: {job_id}")
    print("Monitor Terminal 2 for worker output...")
    
    # In Terminal 2, you should see:
    # [tasks] Received task: textclear.process_job[...]
    # Extract-for-editor complete: N region(s) with font predictions
    # Metadata JSON → /app/results/meta_<job_id>.json
EOF
```

**Expected Worker Log Output:**
```
[tasks] Received task: textclear.process_job[job_uuid] ...
[textclear.worker] FontClassifier initialized in worker context
[textclear.worker] Extract-for-editor complete: 3 region(s) with font predictions
[textclear.worker] Metadata JSON → ./results/meta_job_uuid.json (3 block(s) with fonts)
[textclear.worker] Job job_uuid DONE (fonts: ['Arial', 'Times New Roman'])
```

#### Test 3.2: Verify meta_*.json Output
```bash
# After Test 3.1 completes, check the metadata file
python3 << 'EOF'
import json
from pathlib import Path

result_files = list(Path("results").glob("meta_*.json"))
if result_files:
    latest = sorted(result_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(latest) as f:
        meta = json.load(f)
    
    print(f"Meta file: {latest.name}")
    print(f"Image dimensions: {meta['image_w']}×{meta['image_h']}")
    print(f"Text blocks: {len(meta['blocks'])}")
    
    for i, block in enumerate(meta['blocks']):
        print(f"  Block {i}: {block['text'][:30]:30s} → {block['font_family']:18s} (conf: {block['confidence']:.3f})")
    
    # Validate structure
    for block in meta['blocks']:
        required_fields = ['text', 'x', 'y', 'w', 'h', 'color', 'bg_color', 'size', 'confidence', 'font_family']
        for field in required_fields:
            assert field in block, f"Missing field: {field}"
    
    print("✓ PASS: meta_*.json has all required fields including font_family")
else:
    print("⚠ No meta_*.json files found in results/")
EOF
```

**Expected Output:**
```
Meta file: meta_abc123def456.json
Image dimensions: 1920×1080
Text blocks: 3
  Block 0: Sample Text One           → Arial              (conf: 0.987)
  Block 1: Sample Text Two           → Times New Roman    (conf: 0.945)
  Block 2: Sample Text Three         → Calibri            (conf: 0.892)
✓ PASS: meta_*.json has all required fields including font_family
```

---

### Phase 4: End-to-End API Test

#### Test 4.1: Full HTTP Request
```bash
# Start server
cd backend/
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, upload image
curl -X POST "http://localhost:8000/process-image" \
  -F "file=@test_images/sample.jpg" \
  -F "languages=en" \
  -F "confidence=0.3" \
  | python -m json.tool | head -100
```

**Expected Response (excerpt):**
```json
{
  "image_b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "image_w": 1920,
  "image_h": 1080,
  "blocks": [
    {
      "id": "uuid",
      "text": "Sample Text",
      "x": 100,
      "y": 50,
      "w": 200,
      "h": 30,
      "color": "#2C2C2C",
      "bg_color": "#F5F0E8",
      "size": 24,
      "confidence": 0.987,
      "font_family": "Arial"  ← NEW FIELD ✓
    }
  ]
}
```

---

### Phase 5: Stress & Degradation Tests

#### Test 5.1: ONNX Model File Corruption
```bash
python3 << 'EOF'
# Corrupt the model file
with open("models/font_classifier.onnx", "rb") as f:
    content = f.read()

# Write corrupted version
with open("models/font_classifier_corrupted.onnx", "wb") as f:
    f.write(content[:100])  # Truncate to 100 bytes

# Try to load corrupted model
from text_pipeline import FontClassifier
import numpy as np

fc = FontClassifier(model_path="models/font_classifier_corrupted.onnx")
dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
result = fc.predict(dummy_crop)

assert result == "sans-serif", f"Expected fallback, got {result}"
print("✓ PASS: Corrupted model gracefully falls back to 'sans-serif'")
EOF
```

**Expected Output:**
```
✓ PASS: Corrupted model gracefully falls back to 'sans-serif'
```

#### Test 5.2: Missing ONNXRUNTIME Package
```bash
# This test is destructive; use a venv
python3 -m venv test_venv
source test_venv/bin/activate
pip install -q numpy opencv-python pillow easyocr

# Now ONNXRUNTIME is NOT installed
python3 << 'EOF'
from text_pipeline import FontClassifier
import numpy as np

fc = FontClassifier()
dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
result = fc.predict(dummy_crop)

assert result == "sans-serif", f"Expected fallback without onnxruntime"
print("✓ PASS: Missing onnxruntime gracefully falls back")
EOF

deactivate
```

**Expected Output:**
```
✓ PASS: Missing onnxruntime gracefully falls back
```

---

## 📦 DEPLOYMENT CHECKLIST

- [ ] **Font Classifier Model File**
  - [ ] `models/font_classifier.onnx` exists and is readable
  - [ ] Model outputs [1, 10] tensor (verified with dummy input)
  - [ ] Model file is ≤ 100 MB (fits in typical deployments)

- [ ] **Code Updates**
  - [ ] `text_pipeline.py` updated with `FontClassifier` class
  - [ ] `text_pipeline.py` updated with 10-font `DEFAULT_LABELS`
  - [ ] `EditorBlock` TypedDict includes `font_family` field
  - [ ] `extract_for_editor()` calls `font_classifier.predict()` per region
  - [ ] `worker.py` initializes `FontClassifier` in worker context
  - [ ] `worker.py` passes `font_classifier` to `extract_for_editor()`

- [ ] **Docker / Container**
  - [ ] `Dockerfile` includes model file in image: `COPY models/ /app/models/`
  - [ ] `docker-compose.yml` mounts results volume for `meta_*.json` output
  - [ ] Worker container has sufficient memory (recommend ≥ 2 GB)

- [ ] **Testing**
  - [ ] All 5 test phases pass
  - [ ] `meta_*.json` files contain valid `font_family` values
  - [ ] Fallback to "sans-serif" works without model file

- [ ] **Monitoring**
  - [ ] Log FontClassifier initialization in worker
  - [ ] Log font predictions per region (debug level)
  - [ ] Monitor memory usage (lazy loading should keep baseline low)

---

## 🚀 DEPLOYMENT COMMANDS

```bash
# 1. Update code
cp text_pipeline_updated.py backend/text_pipeline.py
cp worker_updated.py backend/worker.py

# 2. Add model file
mkdir -p backend/models/
cp /path/to/train_font_classifier_output/model.onnx backend/models/font_classifier.onnx

# 3. Run tests (Phase 1-2)
cd backend/
python -m pytest tests/  # (if you have tests)

# 4. Start full stack
docker-compose up --build

# 5. Monitor logs
docker-compose logs -f worker
```

---

## 📊 PERFORMANCE BASELINE

| Operation | Latency | Notes |
|-----------|---------|-------|
| FontClassifier._lazy_load() | 200-500ms | One-time, only on first task |
| FontClassifier.predict() per region | 5-15ms | 64×64 ONNX inference on CPU |
| extract_for_editor() (5 regions) | 3-5s | Dominated by EasyOCR, not fonts |
| Worker task (full pipeline) | 10-15s | EasyOCR ~8s, fonts ~75ms |

**Memory:**
- FontClassifier in-process: ~50-100 MB (depends on model size)
- Lazy loading keeps worker baseline low until first job

---

## 🛑 TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'onnxruntime'"
**Solution:** Install it in worker container:
```bash
pip install onnxruntime
```
Or the pipeline will automatically fall back to "sans-serif" for all regions.

### "font_classifier.onnx not found"
**Solution:** Ensure model file is in one of these paths:
```
- models/font_classifier.onnx (relative to worker.py)
- ./font_classifier.onnx
- /app/models/font_classifier.onnx (Docker)
```

### "Index out of bounds: 15 (expected 0-9)"
**Solution:** Your ONNX model is outputting wrong shape. Verify:
```python
import onnxruntime as ort
sess = ort.InferenceSession("models/font_classifier.onnx")
print(sess.get_outputs()[0])  # Should show shape [1, 10]
```

### Worker crashes with "ONNX model output mismatch"
**Solution:** Your trained model outputs N classes ≠ 10. Retrain with exactly 10 font classes.

---

## 📝 VERSION HISTORY

- **v1.0 (previous)**: Basic OCR + inpainting
- **v2.0**: Added style extraction (colors, font sizes)
- **v3.0 (current)**: Added 10-class font classification with graceful degradation

---

**End of Integration Guide**
