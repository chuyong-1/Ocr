"""
TextClear API  —  FastAPI backend
─────────────────────────────────
Endpoints
  POST   /api/jobs               Upload file + config → create job
  GET    /api/jobs               List all jobs
  GET    /api/jobs/{id}          Job status + metadata
  GET    /api/jobs/{id}/result   Stream the processed file
  DELETE /api/jobs/{id}          Remove job + files
  WS     /ws/{id}               Real-time progress stream

Job lifecycle:  PENDING → RUNNING → DONE | FAILED
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent                       # → editor/backend/
PROJECT_DIR = BASE_DIR.parent                              # → editor/
UPLOAD_DIR  = PROJECT_DIR / "data" / "uploads"
RESULT_DIR  = PROJECT_DIR / "data" / "results"
DB_URL      = f"sqlite:///{PROJECT_DIR / 'data' / 'jobs.db'}"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("textclear.api")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ─────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────
engine  = create_engine(DB_URL, connect_args={"check_same_thread": False})
Session_ = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    id              = Column(String, primary_key=True)
    status          = Column(String, default="PENDING")   # PENDING|RUNNING|DONE|FAILED
    mode            = Column(String)                       # remove|replace
    original_name   = Column(String)
    file_type       = Column(String)                       # image|video
    input_path      = Column(String)
    output_path     = Column(String, nullable=True)
    replacement_map = Column(Text, default="{}")
    languages       = Column(Text, default='["en"]')
    inpainter       = Column(String, default="lama")
    progress        = Column(String, default="0")
    error_msg       = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow,
                             onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = Session_()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE    = "DONE"
    FAILED  = "FAILED"


class JobOut(BaseModel):
    id:              str
    status:          str
    mode:            str
    original_name:   str
    file_type:       str
    inpainter:       str
    progress:        int
    error_msg:       Optional[str]
    created_at:      datetime
    has_result:      bool

    @field_validator("progress", mode="before")
    @classmethod
    def coerce_progress(cls, v):
        return int(v) if v is not None else 0

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────
# WebSocket connection manager
# ─────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self._sockets: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self._sockets.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        if job_id in self._sockets:
            self._sockets[job_id].remove(ws)

    async def broadcast(self, job_id: str, payload: dict):
        for ws in list(self._sockets.get(job_id, [])):
            try:
                await ws.send_json(payload)
            except Exception:
                pass


manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────
# Pipeline runner  (runs in a thread pool via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────
def _run_pipeline(job_id: str, db_factory) -> None:
    """
    Runs the TextPipeline synchronously (blocking).
    Called via asyncio.to_thread so it doesn't block the event loop.
    Progress callbacks post events that the WebSocket endpoint picks up.
    """
    import importlib
    import sys

    # Lazy import so the server can start even without heavy ML deps
    pipeline_module = importlib.import_module("text_pipeline")

    db: Session = db_factory()
    job: JobRecord = db.query(JobRecord).filter_by(id=job_id).first()
    if not job:
        return

    # helper: persist + broadcast progress
    def _update(status: str, progress: int, error: str = None):
        job.status   = status
        job.progress = str(progress)
        if error:
            job.error_msg = error
        job.updated_at = datetime.utcnow()
        db.commit()
        # fire-and-forget broadcast (we're in a thread, so schedule it)
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(job_id, {
                "status":   status,
                "progress": progress,
                "error":    error,
            }),
            _event_loop,
        )

    try:
        _update("RUNNING", 5)

        replacement_map = json.loads(job.replacement_map) if job.replacement_map else {}
        languages       = json.loads(job.languages)       if job.languages else ["en"]

        p = pipeline_module.TextPipeline(
            languages  = languages,
            inpainter  = job.inpainter,
            gpu        = False,         # set True if GPU available
            dilation_px= 6,
            confidence = 0.4,
        )
        _update("RUNNING", 20)

        input_path  = Path(job.input_path)
        ext         = input_path.suffix.lower()
        output_path = RESULT_DIR / f"{job_id}_result{ext}"

        if job.file_type == "video":
            vp = pipeline_module.VideoProcessor(p, keyframe_interval=10)
            # Wrap vp.process to emit intermediate progress
            _update("RUNNING", 30)
            vp.process(
                str(input_path), str(output_path),
                mode            = job.mode,
                replacement_map = replacement_map or None,
            )
        else:
            import cv2
            img = cv2.imread(str(input_path))
            _update("RUNNING", 40)
            result, _ = p.process_image(
                img,
                mode            = job.mode,
                replacement_map = replacement_map or None,
            )
            _update("RUNNING", 85)
            cv2.imwrite(str(output_path), result)

        job.output_path = str(output_path)
        _update("DONE", 100)
        log.info("Job %s completed → %s", job_id, output_path)

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        _update("FAILED", 0, error=str(exc))
    finally:
        db.close()


# Store event loop reference so threads can schedule coroutines
_event_loop: asyncio.AbstractEventLoop | None = None


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="TextClear API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    log.info("TextClear API ready.")


# ── POST /api/detect ──────────────────────────────────────────
@app.post("/api/detect")
async def detect_text(
    file:      UploadFile = File(...),
    languages: str        = Form('["en"]'),
):
    """
    Run OCR on an uploaded image and return bounding boxes + text.
    Does NOT inpaint — just detection for the interactive editor.
    """
    import tempfile, json as _json
    langs = _json.loads(languages)

    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import cv2 as _cv2
        pipeline_module = __import__("text_pipeline")
        detector  = pipeline_module.TextDetector(langs, gpu=False)
        extractor = pipeline_module.StyleExtractor()
        img = _cv2.imread(tmp_path)
        h, w = img.shape[:2]
        regions = detector.detect(img, confidence_threshold=0.3)
        results = []
        for r in regions:
            extractor.extract(img, r)
            results.append({
                "id":         str(uuid.uuid4()),
                "text":       r.text,
                "confidence": round(r.confidence, 3),
                "bbox":       r.bbox,
                "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                "font_size":  r.font_size,
                "text_color": list(r.text_color),
                "bg_color":   list(r.bg_color),
            })
        return {"regions": results, "image_w": w, "image_h": h}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── POST /api/jobs ─────────────────────────────────────────────
@app.post("/api/jobs", response_model=JobOut, status_code=201)
async def create_job(
    background_tasks: BackgroundTasks,
    file:             UploadFile     = File(...),
    mode:             str            = Form("remove"),
    replacement_map:  str            = Form("{}"),
    languages:        str            = Form('["en"]'),
    inpainter:        str            = Form("lama"),
    db:               Session        = Depends(get_db),
):
    # ── Validate file type ──
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    if mode not in ("remove", "replace"):
        raise HTTPException(400, "mode must be 'remove' or 'replace'")

    # ── Save upload ──
    job_id     = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_type = "video" if ext in VIDEO_EXTENSIONS else "image"

    # ── Persist job ──
    job = JobRecord(
        id              = job_id,
        mode            = mode,
        original_name   = file.filename,
        file_type       = file_type,
        input_path      = str(input_path),
        replacement_map = replacement_map,
        languages       = languages,
        inpainter       = inpainter,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # ── Dispatch async ──
    background_tasks.add_task(
        asyncio.to_thread, _run_pipeline, job_id, Session_
    )

    log.info("Job %s created (%s / %s)", job_id, file_type, mode)
    return _to_out(job)


# ── GET /api/jobs ──────────────────────────────────────────────
@app.get("/api/jobs", response_model=List[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(JobRecord).order_by(JobRecord.created_at.desc()).all()
    return [_to_out(j) for j in jobs]


# ── GET /api/jobs/{id} ─────────────────────────────────────────
@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobRecord).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return _to_out(job)


# ── GET /api/jobs/{id}/result ──────────────────────────────────
@app.get("/api/jobs/{job_id}/result")
def download_result(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobRecord).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "DONE" or not job.output_path:
        raise HTTPException(404, "Result not ready")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(404, "Result file missing")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    safe_name  = f"result_{job.original_name}"
    return FileResponse(path, media_type=media_type,
                        filename=safe_name)


# ── DELETE /api/jobs/{id} ──────────────────────────────────────
@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobRecord).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    for p in [job.input_path, job.output_path]:
        if p and Path(p).exists():
            Path(p).unlink(missing_ok=True)
    db.delete(job)
    db.commit()


# ── WS /ws/{id} ───────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def ws_progress(job_id: str, websocket: WebSocket,
                      db: Session = Depends(get_db)):
    await manager.connect(job_id, websocket)
    try:
        # Send current state immediately on connect
        job = db.query(JobRecord).filter_by(id=job_id).first()
        if job:
            await websocket.send_json({
                "status":   job.status,
                "progress": int(job.progress or 0),
                "error":    job.error_msg,
            })
        # Keep alive until client disconnects
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"ping": True})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(job_id, websocket)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _to_out(job: JobRecord) -> JobOut:
    return JobOut(
        id            = job.id,
        status        = job.status,
        mode          = job.mode,
        original_name = job.original_name,
        file_type     = job.file_type,
        inpainter     = job.inpainter,
        progress      = int(job.progress or 0),
        error_msg     = job.error_msg,
        created_at    = job.created_at,
        has_result    = bool(job.output_path and
                             Path(job.output_path).exists()),
    )


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)