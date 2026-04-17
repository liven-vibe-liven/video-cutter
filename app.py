import os
import sys
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from processor import process_upload, export_video, export_mix_video, JobStatus, jobs, analyze_scenes_with_ai

app = FastAPI(title="Video Scene Editor")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
THUMBNAILS_DIR = Path("thumbnails")
for d in (UPLOAD_DIR, OUTPUT_DIR, THUMBNAILS_DIR):
    d.mkdir(exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.3, ge=0.05, le=0.95),
):
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "Please upload a video file")

    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_cut.mp4"

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 500 MB)")

    with open(input_path, "wb") as f:
        f.write(content)

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="detecting",
        step="Uploaded",
        progress=5,
        input_path=str(input_path),
        output_path=str(output_path),
    )

    asyncio.create_task(process_upload(job_id, threshold))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "status": job.status,
        "step": job.step,
        "progress": job.progress,
        "error": job.error,
        "scenes": job.scenes if job.status in ("ready", "done", "exporting") else [],
    }


@app.get("/thumbnail/{job_id}/{scene_idx}")
async def get_thumbnail(job_id: str, scene_idx: int):
    thumb = THUMBNAILS_DIR / job_id / f"{scene_idx:04d}.jpg"
    if not thumb.exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(str(thumb), media_type="image/jpeg")


@app.post("/export/{job_id}")
async def start_export(job_id: str, keep_indices: list[int] = Body(...)):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id].status not in ("ready", "done"):
        raise HTTPException(400, "Scenes not ready")
    asyncio.create_task(export_video(job_id, keep_indices))
    return {"ok": True}


@app.get("/api-status")
async def api_status():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {"available": bool(key and key.startswith("sk-"))}


@app.post("/analyze/{job_id}")
async def analyze_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.status not in ("ready", "done"):
        raise HTTPException(400, "Scenes not ready")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set")
    result = await analyze_scenes_with_ai(job_id)
    return result


@app.post("/export-mix")
async def start_export_mix(timeline: list[dict] = Body(...)):
    mix_job_id = str(uuid.uuid4())
    output_path = str(OUTPUT_DIR / f"{mix_job_id}_mix.mp4")
    jobs[mix_job_id] = JobStatus(
        job_id=mix_job_id,
        status="exporting",
        step="Starting...",
        progress=5,
        input_path="",
        output_path=output_path,
    )
    asyncio.create_task(export_mix_video(timeline, output_path, mix_job_id))
    return {"job_id": mix_job_id}


@app.get("/download-mix/{job_id}")
async def download_mix(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.status != "done":
        raise HTTPException(400, "Not ready")
    out = Path(job.output_path)
    if not out.exists():
        raise HTTPException(500, "Output file missing")
    return FileResponse(str(out), filename="mix_video.mp4", media_type="video/mp4")


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.status != "done":
        raise HTTPException(400, "Not ready")
    out = Path(job.output_path)
    if not out.exists():
        raise HTTPException(500, "Output file missing")
    return FileResponse(str(out), filename="cut_video.mp4", media_type="video/mp4")
