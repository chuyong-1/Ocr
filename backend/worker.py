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
# LangGraph integration: ONNX session bootstrap (once per worker process)
# ─────────────────────────────────────────────────────────────────
from celery.signals import worker_process_init, task_postrun, task_prerun

@worker_process_init.connect
def _bootstrap_onnx(**_kwargs):
    """
    Called once per worker *process* (not per task).
    Loads ONNX sessions (LaMa + FontClassifier) after the fork.
    """
    try:
        from langgraph_pipeline import init_session_manager
        init_session_manager(
            lama_path=str(BASE_DIR.parent / "models" / "lama.onnx"),
            font_path=str(BASE_DIR.parent / "models" / "font_classifier.onnx"),
            font_labels=[
                "Arial", "Times New Roman", "Courier New", "Calibri",
                "Georgia", "Verdana", "Roboto", "Helvetica",
                "Garamond", "Consolas",
            ],
        )
        log.info("ONNX sessions loaded in worker process")
    except Exception as exc:
        log.warning("ONNX bootstrap skipped (non-fatal): %s", exc)


# ─────────────────────────────────────────────────────────────────
# Main Celery task  — v4 LangGraph pipeline integration
# ─────────────────────────────────────────────────────────────────
@celery_app.task(
    bind=True,
    name="textclear.process_job",
    max_retries=2,
    default_retry_delay=10,
)
def process_job(self, job_id: str) -> dict:
    """
    Execute the LangGraph content-aware pipeline with font classification.

    Stages
    ──────
    1.  Load image from uploads/
    2.  Re-detect text with EasyOCR → build TextBlock list
    3.  Generate inpaint mask via MaskGenerator
    4.  run_pipeline() — LangGraph graph (evaluator→inpaint→font→renderer)
    5.  Write result image + meta JSON to results/
    6.  Update DB with output paths
    7.  Set status = DONE
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))

    # Lazy imports — keeps worker startup fast
    from text_pipeline import (
        TextDetector, MaskGenerator, StyleExtractor,
        EditorBlock, rgb_to_hex,
    )
    from langgraph_pipeline import (
        run_pipeline, TextBlock, ProgressEmitter,
    )

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

        # ── 2. EasyOCR detection → TextBlock list ────────────────
        detector = TextDetector(languages, gpu=False)
        from text_pipeline import TextRegion
        regions = detector.detect(img, confidence_threshold=0.40)

        # Extract styles for metadata
        extractor = StyleExtractor()
        for r in regions:
            extractor.extract(img, r)

        # Convert TextRegions → LangGraph TextBlock dicts
        text_blocks: list = []
        for i, r in enumerate(regions):
            text_blocks.append(TextBlock(
                id=f"blk_{i}",
                bbox=r.bbox,
                text=r.text,
                confidence=r.confidence,
                x=r.x, y=r.y, w=r.w, h=r.h,
            ))

        # ── 3. Build mask ────────────────────────────────────────
        masker = MaskGenerator(dilation_px=8)
        mask = masker.generate(img.shape, regions)

        # ── 4. Parse replacement map (if any) ────────────────────
        replacement_map = {}
        mode = getattr(job, "mode", "remove") or "remove"
        if hasattr(job, "replacement_map") and job.replacement_map:
            try:
                replacement_map = json.loads(job.replacement_map)
            except (json.JSONDecodeError, TypeError):
                log.warning("Invalid replacement_map JSON — using remove mode")
                mode = "remove"

        # ── 5. Build ProgressEmitter ─────────────────────────────
        emitter = ProgressEmitter(redis_url=REDIS_URL, job_id=job_id)

        # ── 6. Run LangGraph pipeline ────────────────────────────
        result_img, font_meta, complexity = run_pipeline(
            image_bgr=img,
            mask=mask,
            text_blocks=text_blocks,
            mode=mode,
            replacement_map=replacement_map,
            emitter=emitter,
        )

        log.info(
            "LangGraph pipeline complete: %d blocks, complexity=%.1f, mode=%s",
            len(text_blocks), complexity, mode,
        )

        # ── 7. Save result image ─────────────────────────────────
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        ext          = input_path.suffix.lower() or ".jpg"
        cleaned_name = f"cleaned_{job_id}{ext}"
        cleaned_path = RESULT_DIR / cleaned_name
        cv2.imwrite(str(cleaned_path), result_img)
        log.info("Result image → %s", cleaned_path)

        # ── 8. Build font_metadata lookup for EditorBlocks ───────
        font_lookup = {fp["block_id"]: fp["label"] for fp in font_meta}

        # ── 9. Build and save JSON metadata ──────────────────────
        blocks: list = []
        for i, r in enumerate(regions):
            block_id = f"blk_{i}"
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
                font_family = font_lookup.get(block_id, "sans-serif"),
            ))

        meta_payload = {
            "bg_image" : f"/results/{cleaned_name}",
            "image_w"  : int(img.shape[1]),
            "image_h"  : int(img.shape[0]),
            "blocks"   : blocks,
            "complexity_score": complexity,
            "inpaint_method": "langgraph",
        }

        meta_path = RESULT_DIR / f"meta_{job_id}.json"
        meta_path.write_text(json.dumps(meta_payload, indent=2),
                             encoding="utf-8")
        log.info("Metadata JSON  → %s  (%d blocks)", meta_path, len(blocks))

        # ── 10. Persist output paths in DB ───────────────────────
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
            "complexity"  : complexity,
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