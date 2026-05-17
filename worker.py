"""
TextClear  —  Celery Worker  (v2: Content-Aware Editor)
════════════════════════════════════════════════════════

What changed from v1
─────────────────────
• process_job now calls text_pipeline.extract_for_editor() instead of
  the full TextPipeline.  This gives us word-level style data (color,
  font-size) for every region in a single pass.

• After OCR + inpainting the worker writes TWO artefacts to results/:
    cleaned_{job_id}.jpg   ← background with all text erased
    meta_{job_id}.json     ← structured payload for the frontend editor

• JobRecord.meta_path is set so server.py can stream the payload back
  to the polling client without re-reading the DB entry.

• The LaMa / SD inpainters are no longer used by default; the built-in
  cv2.INPAINT_TELEA backend (CvInpainter) handles all jobs with zero
  model downloads.  Set job.inpainter = "lama" to restore LaMa.

Usage (unchanged)
──────────────────
  celery -A worker.celery_app worker --loglevel=info --concurrency=2
  celery -A worker.celery_app flower --port=5555
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
from celery import Celery
from celery.signals import task_postrun, task_prerun
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

log = logging.getLogger("textclear.worker")

# ─────────────────────────────────────────────────────────────────
# Celery configuration  (unchanged)
# ─────────────────────────────────────────────────────────────────
REDIS_URL  = "redis://localhost:6379/0"

celery_app = Celery(
    "textclear",
    broker  = REDIS_URL,
    backend = REDIS_URL,
    include = ["worker"],
)

celery_app.conf.update(
    task_serializer            = "json",
    result_serializer          = "json",
    accept_content             = ["json"],
    result_expires             = 86400,
    worker_prefetch_multiplier = 1,
    task_acks_late             = True,
    task_track_started         = True,
    task_time_limit            = 3600,
    task_soft_time_limit       = 3000,
)

# ─────────────────────────────────────────────────────────────────
# Database  (mirrors server.py; re-initialised for worker process)
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RESULT_DIR = BASE_DIR.parent / "results"
DB_URL     = f"sqlite:///{BASE_DIR / 'jobs.db'}"

engine   = create_engine(DB_URL, connect_args={"check_same_thread": False})
Session_ = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─────────────────────────────────────────────────────────────────
# Progress helper  (unchanged — writes to DB + Redis pub/sub)
# ─────────────────────────────────────────────────────────────────
def _set_progress(db, job, status: str, progress: int, error: str = None):
    job.status     = status
    job.progress   = str(progress)
    job.updated_at = datetime.utcnow()
    if error:
        job.error_msg = error
    db.commit()

    try:
        from redis import Redis
        r = Redis.from_url(REDIS_URL)
        r.publish(
            f"job:{job.id}",
            json.dumps({"status": status, "progress": progress,
                        "error": error}),
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
# Main Celery task  — v2 content-aware editor pipeline
# ─────────────────────────────────────────────────────────────────
@celery_app.task(
    bind=True,
    name="textclear.process_job",
    max_retries=2,
    default_retry_delay=10,
)
def process_job(self, job_id: str) -> dict:
    """
    Execute the content-aware editor pipeline for a given job_id.

    Stages
    ──────
    1.  Load image from uploads/
    2.  extract_for_editor()   →   cleaned BGR image + EditorBlock list
    3.  Save cleaned_{job_id}.jpg  to results/
    4.  Save meta_{job_id}.json   to results/
    5.  Update JobRecord with output_path + meta_path
    6.  Set status = DONE
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))

    # Lazy import — keeps worker startup fast; avoids loading EasyOCR
    # until a job actually arrives.
    from text_pipeline import extract_for_editor

    db = Session_()
    try:
        from server import JobRecord  # noqa: PLC0415

        job = db.query(JobRecord).filter_by(id=job_id).first()
        if not job:
            return {"status": "FAILED", "error": "Job not found"}

        # ── 1. Validate & read image ─────────────────────────────
        _set_progress(db, job, "RUNNING", 10)

        input_path = Path(job.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Upload missing: {input_path}")

        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"cv2 could not decode: {input_path}")

        languages = json.loads(job.languages or '["en"]')
        _set_progress(db, job, "RUNNING", 20)

        # ── 2. OCR + Style extraction + cv2 inpainting ───────────
        #
        #  extract_for_editor() returns:
        #    cleaned  — BGR ndarray with all text regions filled in
        #    blocks   — List[EditorBlock] with per-region metadata
        #
        #  The inpainter backend is chosen by job.inpainter:
        #    "cv"   → CvInpainter (cv2.INPAINT_TELEA, zero deps)  ← default
        #    "lama" → LamaInpainter (simple-lama-inpainting)
        #    "sd"   → SDInpainter  (Stable Diffusion, GPU)
        #
        #  extract_for_editor always uses CvInpainter internally.
        #  If the job requests lama/sd, we run a second inpaint pass.
        cleaned, blocks = extract_for_editor(
            image_bgr   = img,
            languages   = languages,
            confidence  = 0.40,
            dilation_px = 8,
            gpu         = False,
        )
        _set_progress(db, job, "RUNNING", 65)

        # Optional: upgrade inpaint quality for lama/sd jobs
        if job.inpainter in ("lama", "sd") and blocks:
            cleaned = _upgrade_inpaint(img, blocks, job.inpainter)
            log.info("Upgraded inpainting with backend: %s", job.inpainter)
        _set_progress(db, job, "RUNNING", 80)

        # ── 3. Save cleaned image ────────────────────────────────
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        ext          = input_path.suffix.lower() or ".jpg"
        cleaned_name = f"cleaned_{job_id}{ext}"
        cleaned_path = RESULT_DIR / cleaned_name
        cv2.imwrite(str(cleaned_path), cleaned)
        log.info("Cleaned image → %s", cleaned_path)

        # ── 4. Build and save JSON metadata ──────────────────────
        #
        #  The bg_image URL uses the /results/ static mount defined in
        #  server.py.  The frontend fetches this URL directly to load
        #  the cleaned background image into the editor canvas.
        #
        meta_payload = {
            "bg_image" : f"/results/{cleaned_name}",  # served as static
            "image_w"  : int(img.shape[1]),
            "image_h"  : int(img.shape[0]),
            "blocks"   : blocks,                       # List[EditorBlock]
        }

        meta_path = RESULT_DIR / f"meta_{job_id}.json"
        meta_path.write_text(json.dumps(meta_payload, indent=2),
                             encoding="utf-8")
        log.info("Metadata JSON  → %s  (%d block(s))",
                 meta_path, len(blocks))

        # ── 5. Persist output paths in DB ────────────────────────
        job.output_path = str(cleaned_path)
        job.meta_path   = str(meta_path)   # new column — see server.py
        _set_progress(db, job, "DONE", 100)

        log.info("Job %s DONE", job_id)
        return {
            "status"      : "DONE",
            "output_path" : str(cleaned_path),
            "meta_path"   : str(meta_path),
            "block_count" : len(blocks),
        }

    except Exception as exc:
        log.exception("Job %s FAILED", job_id)
        _set_progress(db, job, "FAILED", 0, error=str(exc))
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"status": "FAILED", "error": str(exc)}
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────
# Optional quality-upgrade helper
# ─────────────────────────────────────────────────────────────────
def _upgrade_inpaint(
    original_bgr: "np.ndarray",
    blocks:       list,
    backend:      str,
) -> "np.ndarray":
    """
    Re-runs inpainting with LaMa or SD on the original image.

    Called only when job.inpainter != "cv".  extract_for_editor() always
    uses CvInpainter for its fast preview; this function replaces that
    result with a higher-quality pass.
    """
    import numpy as np
    from text_pipeline import MaskGenerator, TextRegion, LamaInpainter, SDInpainter

    # Reconstruct minimal TextRegion objects from EditorBlock data
    regions = []
    for b in blocks:
        # Build a rectangular bbox from x/y/w/h
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        corners    = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        regions.append(TextRegion(bbox=corners, text=b["text"], confidence=1.0))

    masker = MaskGenerator(dilation_px=8)
    mask   = masker.generate(original_bgr.shape, regions)

    if backend == "lama":
        inpainter = LamaInpainter()
    else:
        device    = "cpu"
        inpainter = SDInpainter(device=device)

    return inpainter.inpaint(original_bgr, mask)


# ─────────────────────────────────────────────────────────────────
# Celery signals  (unchanged)
# ─────────────────────────────────────────────────────────────────
@task_prerun.connect(sender=process_job)
def task_started(task_id, **kwargs):
    log.info("Task started: %s", task_id)


@task_postrun.connect(sender=process_job)
def task_finished(task_id, retval, **kwargs):
    log.info("Task finished: %s  →  %s", task_id, retval)