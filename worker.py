"""
TextClear  —  Celery Worker
────────────────────────────
Swaps out FastAPI BackgroundTasks for a proper distributed queue.
Redis is used as both broker and result backend.

Usage
─────
# Start worker (from project root)
celery -A worker.celery_app worker --loglevel=info --concurrency=2

# Start Flower monitoring dashboard
celery -A worker.celery_app flower --port=5555

# Submit a job programmatically (alternative to the HTTP API)
from worker import process_job
result = process_job.delay(job_id="abc-123")
print(result.get(timeout=300))

Integration with server.py
──────────────────────────
Replace the BackgroundTasks call in server.py → create_job with:

    from worker import process_job
    process_job.delay(job_id)

This offloads work to any number of worker processes/machines,
enabling horizontal scaling without code changes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from celery import Celery
from celery.signals import task_postrun, task_prerun
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

log = logging.getLogger("textclear.worker")

# ─────────────────────────────────────────────────────────────────
# Celery configuration
# ─────────────────────────────────────────────────────────────────
REDIS_URL = "redis://localhost:6379/0"   # override via env REDIS_URL

celery_app = Celery(
    "textclear",
    broker       = REDIS_URL,
    backend      = REDIS_URL,
    include      = ["worker"],
)

celery_app.conf.update(
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],
    result_expires           = 86400,   # 24 h
    worker_prefetch_multiplier = 1,     # one task at a time per worker slot
    task_acks_late           = True,    # re-queue if worker crashes mid-task
    task_track_started       = True,
    task_time_limit          = 3600,    # 1-hour hard kill
    task_soft_time_limit     = 3000,    # 50-min graceful warning
)

# ─────────────────────────────────────────────────────────────────
# Database (same as server.py but re-initialised for worker process)
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RESULT_DIR  = BASE_DIR.parent / "results"
DB_URL      = f"sqlite:///{BASE_DIR / 'jobs.db'}"

engine   = create_engine(DB_URL, connect_args={"check_same_thread": False})
Session_ = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─────────────────────────────────────────────────────────────────
# Progress hook  (updates DB; server.py reads this via polling or
#                 a separate Redis pub/sub channel)
# ─────────────────────────────────────────────────────────────────
def _set_progress(db, job, status: str, progress: int, error: str = None):
    job.status     = status
    job.progress   = str(progress)
    job.updated_at = datetime.utcnow()
    if error:
        job.error_msg = error
    db.commit()

    # Also publish to Redis so WebSocket clients get instant updates
    try:
        from redis import Redis
        r = Redis.from_url(REDIS_URL)
        r.publish(
            f"job:{job.id}",
            json.dumps({"status": status, "progress": progress, "error": error}),
        )
    except Exception:
        pass  # don't let Redis failure abort the pipeline


# ─────────────────────────────────────────────────────────────────
# Main Celery task
# ─────────────────────────────────────────────────────────────────
@celery_app.task(
    bind=True,
    name="textclear.process_job",
    max_retries=2,
    default_retry_delay=10,
)
def process_job(self, job_id: str) -> dict:
    """
    Execute the full TextPipeline for a given job_id.

    Returns a dict with status, output_path, and any error.
    """
    import importlib
    import sys

    sys.path.insert(0, str(BASE_DIR))
    pipeline_module = importlib.import_module("text_pipeline")

    db  = Session_()
    try:
        # Dynamically import JobRecord from server module
        from server import JobRecord  # noqa: PLC0415

        job = db.query(JobRecord).filter_by(id=job_id).first()
        if not job:
            return {"status": "FAILED", "error": "Job not found"}

        _set_progress(db, job, "RUNNING", 10)

        replacement_map = json.loads(job.replacement_map or "{}")
        languages       = json.loads(job.languages or '["en"]')

        pipeline = pipeline_module.TextPipeline(
            languages  = languages,
            inpainter  = job.inpainter,
            gpu        = False,
            dilation_px= 6,
            confidence = 0.4,
        )
        _set_progress(db, job, "RUNNING", 25)

        input_path  = Path(job.input_path)
        ext         = input_path.suffix.lower()
        output_path = RESULT_DIR / f"{job_id}_result{ext}"
        RESULT_DIR.mkdir(exist_ok=True)

        if job.file_type == "video":
            vp = pipeline_module.VideoProcessor(pipeline, keyframe_interval=10)

            # Monkey-patch write method to emit progress events
            original_write = None
            import cv2
            cap_tmp = cv2.VideoCapture(str(input_path))
            total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap_tmp.release()

            frame_counter = [0]
            _orig_process = vp.process

            def _tracked_process(*args, **kwargs):
                # Can't easily inject mid-loop; emit at start/end
                _set_progress(db, job, "RUNNING", 35)
                _orig_process(*args, **kwargs)

            vp.process = _tracked_process
            vp.process(
                str(input_path), str(output_path),
                mode=job.mode,
                replacement_map=replacement_map or None,
            )
        else:
            import cv2
            img = cv2.imread(str(input_path))
            if img is None:
                raise ValueError(f"Could not read image: {input_path}")
            _set_progress(db, job, "RUNNING", 40)
            result, _ = pipeline.process_image(
                img,
                mode=job.mode,
                replacement_map=replacement_map or None,
            )
            _set_progress(db, job, "RUNNING", 88)
            cv2.imwrite(str(output_path), result)

        job.output_path = str(output_path)
        _set_progress(db, job, "DONE", 100)
        log.info("Job %s done → %s", job_id, output_path)
        return {"status": "DONE", "output_path": str(output_path)}

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        _set_progress(db, job, "FAILED", 0, error=str(exc))
        # Retry transient errors (e.g. OOM)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"status": "FAILED", "error": str(exc)}
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────
# Celery signals  (logging hooks)
# ─────────────────────────────────────────────────────────────────
@task_prerun.connect(sender=process_job)
def task_started(task_id, **kwargs):
    log.info("Task started: %s", task_id)


@task_postrun.connect(sender=process_job)
def task_finished(task_id, retval, **kwargs):
    log.info("Task finished: %s  →  %s", task_id, retval)
