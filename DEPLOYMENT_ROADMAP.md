# 🗺️ IMPLEMENTATION ROADMAP & DEPLOYMENT CHECKLIST
## PixelScribe v3.0 — 10-Class Font Classifier Integration

---

## 📋 DOCUMENT MAP

```
📦 Complete Integration Package
├── 📄 EXECUTIVE_SUMMARY.md
│   └─ Start here: Overview, architecture, key metrics
├── 📄 QUICK_REFERENCE.md
│   └─ Code changes, checklist, troubleshooting
├── 📄 FONT_CLASSIFIER_INTEGRATION_GUIDE.md
│   └─ Comprehensive 5-phase testing protocol (1000+ lines)
├── 🐍 text_pipeline_updated.py
│   └─ COMPLETE REPLACEMENT for backend/text_pipeline.py
├── 🐍 worker_updated.py
│   └─ UPDATED backend/worker.py with FontClassifier
└── 🗺️ THIS FILE
    └─ Deployment roadmap and final checklist
```

**Reading Order:**
1. **EXECUTIVE_SUMMARY.md** (this gives you the big picture)
2. **QUICK_REFERENCE.md** (for code changes at a glance)
3. **FONT_CLASSIFIER_INTEGRATION_GUIDE.md** (to run tests)
4. **Code files** (text_pipeline_updated.py, worker_updated.py)

---

## 🎯 HIGH-LEVEL CHANGES

### What's New

```python
# BEFORE (v2):
EditorBlock = {
    "text": "Hello",
    "x": 100, "y": 50, "w": 200, "h": 30,
    "color": "#2C2C2C",
    "bg_color": "#F5F0E8",
    "size": 24,
    "confidence": 0.95
}

# AFTER (v3):
EditorBlock = {
    "text": "Hello",
    "x": 100, "y": 50, "w": 200, "h": 30,
    "color": "#2C2C2C",
    "bg_color": "#F5F0E8",
    "size": 24,
    "confidence": 0.95,
    "font_family": "Arial"  ← NEW FIELD (from 10-class ONNX)
}
```

### 10 Supported Font Classes

```
Index 0: Arial              Index 5: Verdana
Index 1: Times New Roman    Index 6: Roboto
Index 2: Courier New        Index 7: Helvetica
Index 3: Calibri            Index 8: Garamond
Index 4: Georgia            Index 9: Consolas
```

---

## 🔄 DEPLOYMENT PHASES

### Phase 0: Preparation (15 minutes)

#### Step 0.1: Verify ONNX Model
```bash
# Ensure your trained model exists and has correct shape
python3 << 'EOF'
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/font_classifier.onnx")

# Print shapes
print("Input:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: {inp.shape}")

print("Output:")
for out in sess.get_outputs():
    print(f"  {out.name}: {out.shape}")

# Test inference
dummy_input = np.random.randn(1, 1, 64, 64).astype(np.float32)
output = sess.run(None, {sess.get_inputs()[0].name: dummy_input})
print(f"\nDummy inference output shape: {output[0].shape}")
print("Expected: (1, 10)")

assert output[0].shape == (1, 10), "Model must output [1, 10]!"
EOF
```

**Expected Output:**
```
Input:
  input.1: [1, 1, 64, 64]
Output:
  output.1: [1, 10]

Dummy inference output shape: (1, 10)
Expected: (1, 10)
```

#### Step 0.2: Backup Existing Code
```bash
cd backend/
git commit -am "Pre-font-classifier snapshot"
# or
cp text_pipeline.py text_pipeline.py.backup
cp worker.py worker.py.backup
```

#### Step 0.3: Prepare Model Directory
```bash
mkdir -p backend/models/
cp /path/to/your/trained/model.onnx backend/models/font_classifier.onnx
chmod 644 backend/models/font_classifier.onnx
ls -lh backend/models/font_classifier.onnx
```

---

### Phase 1: Code Deployment (5 minutes)

#### Step 1.1: Replace text_pipeline.py
```bash
cp text_pipeline_updated.py backend/text_pipeline.py
```

**Verify:**
```bash
# Check FontClassifier class exists
grep -n "class FontClassifier:" backend/text_pipeline.py
# Should output: line number where class begins

# Check 10 fonts
grep -A 10 "DEFAULT_LABELS = \[" backend/text_pipeline.py
# Should show all 10 fonts in exact order
```

#### Step 1.2: Replace worker.py
```bash
cp worker_updated.py backend/worker.py
```

**Verify:**
```bash
# Check FontClassifier import in worker
grep "from text_pipeline import.*FontClassifier" backend/worker.py

# Check font_classifier initialization
grep "font_classifier = FontClassifier" backend/worker.py
```

#### Step 1.3: Update requirements.txt
```bash
echo "onnxruntime>=1.17.0" >> backend/requirements.txt
```

**Verify:**
```bash
grep "onnxruntime" backend/requirements.txt
```

---

### Phase 2: Unit Testing (15 minutes)

Run these tests sequentially. Stop if any fails.

#### Test 2.1: FontClassifier Import
```bash
cd backend/
python3 -c "from text_pipeline import FontClassifier; print('✓ Import OK')"
```

#### Test 2.2: Label Count & Order
```bash
python3 << 'EOF'
from text_pipeline import FontClassifier

fc = FontClassifier()
expected = ["Arial", "Times New Roman", "Courier New", "Calibri", "Georgia",
            "Verdana", "Roboto", "Helvetica", "Garamond", "Consolas"]

print(f"Count: {len(fc.DEFAULT_LABELS)} (expected 10)")
assert len(fc.DEFAULT_LABELS) == 10, "Wrong count!"

print(f"Order: {fc.DEFAULT_LABELS == expected}")
assert fc.DEFAULT_LABELS == expected, "Label order mismatch!"

print("✓ Labels verified")
EOF
```

#### Test 2.3: Predict Method (No Model)
```bash
python3 << 'EOF'
from text_pipeline import FontClassifier
import numpy as np

# Point to non-existent model
fc = FontClassifier(model_path="/nonexistent/model.onnx")

# Should not crash, should fall back
dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
result = fc.predict(dummy_crop)

print(f"Prediction without model: '{result}'")
assert result == "sans-serif", f"Expected 'sans-serif', got '{result}'"

print("✓ Fallback works")
EOF
```

#### Test 2.4: EditorBlock TypedDict
```bash
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

assert "font_family" in reloaded[0], "font_family missing!"
assert reloaded[0]["font_family"] == "Arial", "Value corrupted!"

print("✓ EditorBlock serialization OK")
EOF
```

---

### Phase 3: Pipeline Testing (30 minutes)

#### Test 3.1: extract_for_editor() Basic
```bash
python3 << 'EOF'
import cv2
import numpy as np
from text_pipeline import extract_for_editor, FontClassifier

# Create a synthetic image with some "text" (colored boxes)
img = np.ones((100, 300, 3), dtype=np.uint8) * 240  # Light background

# Draw a "text" region (dark area)
cv2.rectangle(img, (50, 30), (250, 70), (0, 0, 0), -1)

fc = FontClassifier(model_path="models/font_classifier.onnx")
cleaned, blocks = extract_for_editor(img, font_classifier=fc)

print(f"Extracted {len(blocks)} block(s)")
for block in blocks:
    assert "font_family" in block, "Missing font_family!"
    print(f"  Text: {block['text'][:20]:20s} Font: {block['font_family']}")

print("✓ extract_for_editor OK")
EOF
```

#### Test 3.2: Verify Output Shape
```bash
python3 << 'EOF'
import cv2
import numpy as np
from text_pipeline import extract_for_editor

img = np.ones((100, 300, 3), dtype=np.uint8) * 240
cv2.rectangle(img, (50, 30), (250, 70), (0, 0, 0), -1)

cleaned, blocks = extract_for_editor(img)

# Cleaned image should same dimensions as input
assert cleaned.shape == img.shape, f"Shape mismatch: {cleaned.shape} vs {img.shape}"

# Blocks should be list of dicts
assert isinstance(blocks, list), "Blocks must be list"
for block in blocks:
    assert isinstance(block, dict), "Each block must be dict"
    assert "font_family" in block, "Missing font_family"

print("✓ Output shape OK")
EOF
```

---

### Phase 4: Worker Testing (30 minutes)

#### Test 4.1: Start Services
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Celery Worker
cd backend/
celery -A worker.celery_app worker --loglevel=info --concurrency=1

# Watch for:
# [tasks] Received task: textclear.process_job[...]
# [textclear.worker] FontClassifier initialized in worker context
```

#### Test 4.2: Submit Test Job
```bash
# Terminal 3: Submit job
python3 << 'EOF'
import json
import uuid
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server import Base, JobRecord

# Setup DB
engine = create_engine("sqlite:///jobs.db")
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
db = Session()

# Create test job
job_id = str(uuid.uuid4())[:8]
input_path = Path("uploads") / f"{job_id}_input.jpg"
input_path.parent.mkdir(exist_ok=True)

# Create dummy image
import cv2
import numpy as np
dummy_img = np.ones((100, 300, 3), dtype=np.uint8) * 240
cv2.rectangle(dummy_img, (50, 30), (250, 70), (0, 0, 0), -1)
cv2.imwrite(str(input_path), dummy_img)

# Create job record
job = JobRecord(
    id=job_id,
    mode="remove",
    original_name="test.jpg",
    file_type="image",
    input_path=str(input_path),
    languages='["en"]',
    inpainter="cv",
)
db.add(job)
db.commit()

print(f"Created job: {job_id}")
print("Check Terminal 2 for worker output...")
print("Wait 10-30 seconds for processing...")

# Poll status
import time
for _ in range(30):
    db.refresh(job)
    print(f"Status: {job.status} | Progress: {job.progress}%")
    if job.status == "DONE":
        print("✓ Job complete!")
        break
    time.sleep(1)
EOF
```

**Expected Worker Log:**
```
[tasks] Received task: textclear.process_job[job-uuid]...
[textclear.worker] FontClassifier initialized in worker context
[textclear.worker] Extract-for-editor complete: N region(s) with font predictions
[textclear.worker] Metadata JSON → ./results/meta_job-uuid.json (N block(s) with fonts)
[textclear.worker] Job job-uuid DONE (fonts: ['Arial', 'Roboto'])
```

#### Test 4.3: Verify meta_*.json
```bash
# After job completes
python3 << 'EOF'
import json
from pathlib import Path

meta_files = list(Path("results").glob("meta_*.json"))
if meta_files:
    latest = sorted(meta_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(latest) as f:
        meta = json.load(f)
    
    print(f"✓ Found: {latest.name}")
    print(f"  Image: {meta['image_w']}×{meta['image_h']}")
    print(f"  Blocks: {len(meta['blocks'])}")
    
    for block in meta['blocks']:
        print(f"    - {block['text'][:20]:20s} → {block.get('font_family', '?')}")
    
    # Validate structure
    for block in meta['blocks']:
        required = ['text', 'x', 'y', 'w', 'h', 'color', 'bg_color', 'size', 'confidence', 'font_family']
        for field in required:
            assert field in block, f"Missing: {field}"
    
    print("✓ All fields present and valid")
else:
    print("⚠ No meta_*.json files found")
EOF
```

---

### Phase 5: Full Integration Test (15 minutes)

#### Test 5.1: Start Full Stack
```bash
cd backend/
docker-compose down  # Clean slate
docker-compose up --build -d
sleep 10

# Check logs
docker-compose logs -f api
```

#### Test 5.2: API Request
```bash
# Create test image
python3 << 'EOF'
import cv2
import numpy as np

img = np.ones((100, 300, 3), dtype=np.uint8) * 240
cv2.rectangle(img, (50, 30), (250, 70), (0, 0, 0), -1)
cv2.imwrite("test_image.jpg", img)
EOF

# Send request
curl -X POST "http://localhost:8000/process-image" \
  -F "file=@test_image.jpg" \
  -F "languages=en" \
  -F "confidence=0.3" \
  | python -m json.tool > response.json

# Check response
python3 << 'EOF'
import json

with open("response.json") as f:
    resp = json.load(f)

print(f"Image: {resp['image_w']}×{resp['image_h']}")
print(f"Blocks: {len(resp['blocks'])}")

for block in resp['blocks']:
    print(f"  {block['text'][:20]:20s} → {block.get('font_family', '?')}")
    assert "font_family" in block, "Missing font_family in response!"

print("✓ API response valid")
EOF
```

---

## ✅ FINAL DEPLOYMENT CHECKLIST

### Pre-Deployment
- [ ] ONNX model trained with 10 exact fonts (Arial, Times New Roman, ...)
- [ ] Model outputs [1, 10] logits verified
- [ ] Model file size < 100 MB
- [ ] `backend/models/font_classifier.onnx` exists and is readable
- [ ] Git commit made (backup of current code)

### Code Updates
- [ ] `text_pipeline_updated.py` copied to `backend/text_pipeline.py`
- [ ] `worker_updated.py` copied to `backend/worker.py`
- [ ] `requirements.txt` has `onnxruntime>=1.17.0`
- [ ] No syntax errors: `python -m py_compile backend/text_pipeline.py`
- [ ] No syntax errors: `python -m py_compile backend/worker.py`

### Unit Tests (Phase 2)
- [ ] Test 2.1: FontClassifier import ✓
- [ ] Test 2.2: Label count & order ✓
- [ ] Test 2.3: Predict fallback (no model) ✓
- [ ] Test 2.4: EditorBlock serialization ✓

### Pipeline Tests (Phase 3)
- [ ] Test 3.1: extract_for_editor() basic ✓
- [ ] Test 3.2: Output shape & structure ✓

### Worker Tests (Phase 4)
- [ ] Test 4.1: Worker starts and receives task ✓
- [ ] Test 4.2: Job completes with DONE status ✓
- [ ] Test 4.3: meta_*.json contains font_family ✓

### Full Integration (Phase 5)
- [ ] Test 5.1: Docker stack starts cleanly ✓
- [ ] Test 5.2: API request returns font_family ✓

### Production Readiness
- [ ] Monitoring: FontClassifier load logged
- [ ] Monitoring: Font predictions logged (debug level)
- [ ] Alerting: ONNX model errors trigger alerts (optional)
- [ ] Monitoring: Worker memory usage tracked
- [ ] Logging: All fallback paths logged at INFO level

### Documentation
- [ ] EXECUTIVE_SUMMARY.md read and understood
- [ ] QUICK_REFERENCE.md bookmarked for troubleshooting
- [ ] FONT_CLASSIFIER_INTEGRATION_GUIDE.md available for debugging
- [ ] Team trained on new font_family field in responses

---

## 🚀 GO-LIVE PROCEDURE

### Step 1: Staging Deployment
```bash
# Deploy to staging environment
docker-compose -f docker-compose.staging.yml up -d

# Run full test suite (Phase 2-5)
python test_suite.py  # Create this if needed

# Verify metrics
curl http://localhost:8000/health
```

### Step 2: Monitoring Setup
```bash
# Setup logs monitoring
docker-compose logs -f worker | grep "FontClassifier\|font_family\|FAILED"

# Setup metrics (optional)
# Track: latency per region, fallback rate, memory
```

### Step 3: Production Deployment
```bash
# Tag release
git tag v3.0-font-classifier
git push --tags

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost:8000/health
docker-compose ps
docker-compose logs -f worker | head -20
```

### Step 4: Post-Deployment Validation
```bash
# Submit test job
# Check meta_*.json in results/
# Verify font_family values are from the 10 fonts list
# Monitor error logs for 24 hours
```

---

## 📊 SUCCESS METRICS

### Must-Pass Criteria
- ✅ All 5 test phases pass without errors
- ✅ No exceptions in worker logs related to FontClassifier
- ✅ meta_*.json files contain valid font_family values
- ✅ Fallback to "sans-serif" works without model file
- ✅ Performance within baseline (Phase 4: 10–15s per job)

### Nice-to-Have Metrics
- 📊 Fallback rate < 5% (most predictions accurate)
- 📊 Average font prediction confidence > 0.85
- 📊 Memory overhead < 200 MB per worker
- 📊 All 10 fonts represented in predictions

---

## 🔗 QUICK LINKS

| Resource | Purpose |
|----------|---------|
| EXECUTIVE_SUMMARY.md | Overview & architecture |
| QUICK_REFERENCE.md | Code changes & troubleshooting |
| FONT_CLASSIFIER_INTEGRATION_GUIDE.md | Detailed testing protocol |
| text_pipeline_updated.py | Core implementation |
| worker_updated.py | Worker integration |

---

## 🆘 EMERGENCY ROLLBACK

If deployment fails or issues arise:

```bash
# Rollback code
cp backend/text_pipeline.py.backup backend/text_pipeline.py
cp backend/worker.py.backup backend/worker.py

# Restart services
docker-compose restart worker api

# Remove model (optional)
rm backend/models/font_classifier.onnx
```

The system will automatically fall back to v2 behavior (no font_family field).

---

## 📞 SUPPORT ESCALATION

| Issue Level | Action | Resource |
|-------------|--------|----------|
| Code question | Check QUICK_REFERENCE.md | Code changes summary |
| Test failure | Follow INTEGRATION_GUIDE.md | Phase-by-phase tests |
| Production issue | Check logs for error signature | QUICK_REFERENCE.md troubleshooting |
| Unknown error | Run full test suite | INTEGRATION_GUIDE.md Phase 5 |

---

## 📝 SIGN-OFF

**Deployment checklist:**
- [ ] All phases passed
- [ ] Team trained
- [ ] Monitoring configured
- [ ] Rollback plan ready
- [ ] Go-live approved

**Date Deployed:** ___________  
**Deployed By:** ___________  
**Verified By:** ___________

---

**End of Deployment Roadmap**

Version: 1.0 | Date: May 18, 2026 | Status: ✅ Ready
