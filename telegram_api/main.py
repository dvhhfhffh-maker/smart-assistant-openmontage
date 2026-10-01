import json
import os
import secrets
import time
from pathlib import Path
from uuid import uuid4

import requests
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="Smart Assistant OpenMontage API",
    version="0.3.0"
)

OUTPUT_DIR = Path("/tmp/smart-assistant-reels")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KIE_BASE_URL = "https://api.kie.ai"
KIE_MODEL = "bytedance/seedance-2-mini"
KIE_POLL_INTERVAL_SECONDS = 5
KIE_TIMEOUT_SECONDS = 900

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


def kie_headers():
    api_key = os.environ.get("KIE_API_KEY")

    if not api_key:
        raise RuntimeError("KIE_API_KEY is not configured")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def create_kie_task(prompt: str, duration: int) -> str:
    response = requests.post(
        f"{KIE_BASE_URL}/api/v1/jobs/createTask",
        headers=kie_headers(),
        json={
            "model": KIE_MODEL,
            "input": {
                "prompt": prompt,
                "return_last_frame": False,
                "generate_audio": False,
                "resolution": "720p",
                "aspect_ratio": "9:16",
                "duration": duration,
                "web_search": False,
            },
        },
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    task_id = (payload.get("data") or {}).get("taskId")

    if not task_id:
        raise RuntimeError(
            f"Kie.ai did not return taskId: {payload}"
        )

    return task_id


def get_kie_task(task_id: str) -> dict:
    response = requests.get(
        f"{KIE_BASE_URL}/api/v1/jobs/recordInfo",
        headers=kie_headers(),
        params={"taskId": task_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def extract_result_url(data: dict) -> str | None:
    result_json = data.get("resultJson")

    if not result_json:
        return None

    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except json.JSONDecodeError:
            return None

    if not isinstance(result_json, dict):
        return None

    urls = result_json.get("resultUrls") or []
    if urls:
        return urls[0]

    return (
        result_json.get("videoUrl")
        or result_json.get("video_url")
        or result_json.get("url")
    )


def download_video(url: str, output_path: Path):
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def generate_reel(job_id: str, prompt: str, duration: int):
    jobs[job_id]["status"] = "submitting"

    try:
        task_id = create_kie_task(prompt, duration)

        jobs[job_id].update({
            "status": "generating",
            "provider": "kie.ai",
            "model": KIE_MODEL,
            "provider_task_id": task_id,
        })

        deadline = time.monotonic() + KIE_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            payload = get_kie_task(task_id)
            data = payload.get("data") or {}
            state = data.get("state")

            if state == "success":
                result_url = extract_result_url(data)

                if not result_url:
                    raise RuntimeError(
                        f"Kie.ai task succeeded but no result URL was returned: {payload}"
                    )

                output_path = OUTPUT_DIR / f"{job_id}.mp4"
                download_video(result_url, output_path)

                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise RuntimeError("Video file was not created")
                download_token = secrets.token_urlsafe(24)

                jobs[job_id] = {
                    "id": job_id,
                    "status": "completed",
                    "provider": "kie.ai",
                    "model": KIE_MODEL,
                    "provider_task_id": task_id,
                    "credits_consumed": data.get("creditsConsumed"),
                    "generation_time_ms": data.get("costTime"),
                    "download_url": f"/reel/{job_id}/file",
                    "download_token": download_token,
                    "public_download_url": f"/public/reel/{job_id}/{download_token}",
                }
                return

            if state == "fail":
                error_message = (
                    data.get("failMsg")
                    or data.get("failCode")
                    or payload.get("msg")
                    or "Kie.ai generation failed"
                )
                raise RuntimeError(str(error_message))

            jobs[job_id]["provider_state"] = state or "unknown"
            jobs[job_id]["progress"] = data.get("progress")
            time.sleep(KIE_POLL_INTERVAL_SECONDS)

        raise RuntimeError(
            f"Kie.ai generation timed out after {KIE_TIMEOUT_SECONDS} seconds"
        )

    except Exception as error:
        jobs[job_id] = {
            "id": job_id,
            "status": "failed",
            "provider": "kie.ai",
            "model": KIE_MODEL,
            "error": str(error),
        }


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Smart Assistant OpenMontage API",
        "version": "0.3.0",
        "provider": "kie.ai",
        "model": KIE_MODEL,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy",
        "kie_ready": bool(os.environ.get("KIE_API_KEY")),
        "reel_api_ready": bool(os.environ.get("REEL_API_KEY")),
        "provider": "kie.ai",
        "model": KIE_MODEL,
    }


@app.post("/reel", status_code=202)
def create_reel(
    request: ReelRequest,
    background_tasks: BackgroundTasks,
    x_reel_key: str | None = Header(default=None)
):
    check_api_key(x_reel_key)

    if not os.environ.get("KIE_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="KIE_API_KEY is not configured"
        )

    job_id = uuid4().hex

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "duration": request.duration,
        "aspect_ratio": "9:16",
        "provider": "kie.ai",
        "model": KIE_MODEL,
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

@app.get("/public/reel/{job_id}/{download_token}")
def public_reel_file(job_id: str, download_token: str):
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

    expected_token = job.get("download_token", "")

    if not expected_token or not secrets.compare_digest(
        download_token,
        expected_token
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid download token"
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
