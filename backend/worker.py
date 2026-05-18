"""
TextClear  —  Celery Worker  (v3: 10-Class Font Classifier Integration)
════════════════════════════════════════════════════════════════════════

What's new in v3
────────────────
• FontClassifier lazy-loads inside worker context (not FastAPI server).
• extract_for_editor() now returns EditorBlock structures with 'font_family' field.
• meta_{job_id}.json now includes font predictions per text region.
• Graceful fallback to "sans-serif" if ONNX model unavailable.
• No breaking changes to job persistence or WebSocket broadcasting.

Usage
──────
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
RESULT_DIR = BASE_DIR.parent / "data" / "results"
DB_URL     = f"sqlite:///{BASE_DIR.parent / 'data' / 'jobs.db'}"

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
# Main Celery task  — v3 content-aware editor with font classification
# ─────────────────────────────────────────────────────────────────
@celery_app.task(
    bind=True,
    name="textclear.process_job",
    max_retries=2,
    default_retry_delay=10,
)
def process_job(self, job_id: str) -> dict:
    """
    Execute the content-aware editor pipeline with 10-class font classification.

    Stages
    ──────
    1.  Load image from uploads/
    2.  extract_for_editor()   →   cleaned BGR image + EditorBlock list
    3.  EditorBlocks now include 'font_family' field (10-class prediction)
    4.  Save cleaned_{job_id}.jpg  to results/
    5.  Save meta_{job_id}.json   to results/ (with font data)
    6.  Update JobRecord with output_path + meta_path
    7.  Set status = DONE

    Defensive programming ensures graceful degradation if font classifier
    ONNX model is unavailable (falls back to "sans-serif").
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))

    # Lazy import — keeps worker startup fast; avoids loading EasyOCR,
    # FontClassifier, and other heavy ML deps until a job actually arrives.
    from text_pipeline import extract_for_editor, FontClassifier

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

        # ── 2. Initialize FontClassifier (lazy-loads ONNX) ───────
        #
        # The FontClassifier gracefully falls back to "sans-serif" if the
        # ONNX model file isn't found. This ensures the pipeline continues
        # even if the model is missing or corrupted.
        font_classifier = FontClassifier(gpu=False)
        log.info("FontClassifier initialized in worker context")
        _set_progress(db, job, "RUNNING", 30)

        # ── 3. OCR + Style extraction + Font classification ─────
        #
        # extract_for_editor() now:
        #   • Runs EasyOCR for text detection
        #   • Extracts colors and font sizes
        #   • Predicts 10-class font family per region
        #   • Returns EditorBlocks with font_family field
        #
        cleaned, blocks = extract_for_editor(
            image_bgr       = img,
            languages       = languages,
            confidence      = 0.40,
            dilation_px     = 8,
            gpu             = False,
            font_classifier = font_classifier,
        )
        _set_progress(db, job, "RUNNING", 75)

        log.info(
            "Extract-for-editor complete: %d region(s) with font predictions",
            len(blocks),
        )

        # ── 4. Save cleaned image ────────────────────────────────
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        ext          = input_path.suffix.lower() or ".jpg"
        cleaned_name = f"cleaned_{job_id}{ext}"
        cleaned_path = RESULT_DIR / cleaned_name
        cv2.imwrite(str(cleaned_path), cleaned)
        log.info("Cleaned image → %s", cleaned_path)

        # ── 5. Build and save JSON metadata ──────────────────────
        #
        # The meta_payload now includes 'font_family' in each block.
        # Frontend can read and apply these font predictions directly.
        #
        meta_payload = {
            "bg_image" : f"/results/{cleaned_name}",  # served as static
            "image_w"  : int(img.shape[1]),
            "image_h"  : int(img.shape[0]),
            "blocks"   : blocks,  # List[EditorBlock] with font_family
        }

        meta_path = RESULT_DIR / f"meta_{job_id}.json"
        meta_path.write_text(json.dumps(meta_payload, indent=2),
                             encoding="utf-8")
        log.info(
            "Metadata JSON  → %s  (%d block(s) with fonts)",
            meta_path,
            len(blocks),
        )

        # Validation: ensure all blocks have font_family field
        for block in blocks:
            if "font_family" not in block:
                log.warning(
                    "Block '%s' missing font_family field. "
                    "Adding default 'sans-serif'.",
                    block.get("text", "unknown"),
                )
                block["font_family"] = "sans-serif"

        # ── 6. Persist output paths in DB ────────────────────────
        job.output_path = str(cleaned_path)
        job.meta_path   = str(meta_path)
        _set_progress(db, job, "DONE", 100)

        log.info("Job %s DONE  (fonts: %s)", job_id,
                 [b.get("font_family", "?") for b in blocks[:3]])
        return {
            "status"      : "DONE",
            "output_path" : str(cleaned_path),
            "meta_path"   : str(meta_path),
            "block_count" : len(blocks),
            "fonts"       : list(set(b.get("font_family", "sans-serif") for b in blocks)),
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
# Celery signals  (unchanged)
# ─────────────────────────────────────────────────────────────────
@task_prerun.connect(sender=process_job)
def task_started(task_id, **kwargs):
    log.info("Task started: %s", task_id)


@task_postrun.connect(sender=process_job)
def task_finished(task_id, retval, **kwargs):
    log.info("Task finished: %s  →  %s", task_id, retval)