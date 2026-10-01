import os
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tools.video.minimax_video import MiniMaxVideo


app = FastAPI(
    title="Smart Assistant OpenMontage API",
    version="0.2.0"
)

OUTPUT_DIR = Path("/tmp/smart-assistant-reels")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}


class ReelRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=7000)
    duration: int = Field(default=5, ge=4, le=15)


def check_api_key(x_reel_key: str | None):
    expected = os.environ.get("REEL_API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="REEL_API_KEY is not configured"
        )

    if not x_reel_key or not secrets.compare_digest(
        x_reel_key,
        expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def generate_reel(job_id: str, prompt: str, duration: int):
    jobs[job_id]["status"] = "generating"

    try:
        output_path = OUTPUT_DIR / f"{job_id}.mp4"

        tool = MiniMaxVideo()

        result = tool.execute({
            "prompt": prompt,
            "operation": "text_to_video",
            "model": "MiniMax-H3",
            "duration": duration,
            "resolution": "2K",
            "aspect_ratio": "9:16",
            "output_path": str(output_path),
            "timeout_seconds": 900
        })

        if not result.success:
            jobs[job_id] = {
                "id": job_id,
                "status": "failed",
                "error": result.error
            }
            return

        if not output_path.exists():
            jobs[job_id] = {
                "id": job_id,
                "status": "failed",
                "error": "Video file was not created"
            }
            return

        jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "cost_usd": result.cost_usd,
            "generation_time": result.duration_seconds,
            "download_url": f"/reel/{job_id}/file"
        }

    except Exception as error:
        jobs[job_id] = {
            "id": job_id,
            "status": "failed",
            "error": str(error)
        }


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Smart Assistant OpenMontage API",
        "version": "0.2.0"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy",
        "minimax_ready": bool(
            os.environ.get("MINIMAX_API_KEY")
        )
    }


@app.post("/reel", status_code=202)
def create_reel(
    request: ReelRequest,
    background_tasks: BackgroundTasks,
    x_reel_key: str | None = Header(default=None)
):
    check_api_key(x_reel_key)

    if not os.environ.get("MINIMAX_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="MINIMAX_API_KEY is not configured"
        )

    job_id = uuid4().hex

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "duration": request.duration,
        "aspect_ratio": "9:16"
    }

    background_tasks.add_task(
        generate_reel,
        job_id,
        request.prompt,
        request.duration
    )

    return jobs[job_id]


@app.get("/reel/{job_id}")
def reel_status(
    job_id: str,
    x_reel_key: str | None = Header(default=None)
):
    check_api_key(x_reel_key)

    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Reel job not found"
        )

    return job


@app.get("/reel/{job_id}/file")
def reel_file(
    job_id: str,
    x_reel_key: str | None = Header(default=None)
):
    check_api_key(x_reel_key)

    job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Reel job not found"
        )

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Reel is not ready"
        )

    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video file not found"
        )

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"reel-{job_id}.mp4"
    )
