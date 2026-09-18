from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mustard.config import get_paths  # noqa: E402
from mustard.io_utils import load_json  # noqa: E402

app = FastAPI(
    title="MUSTARD — Multimodal Sarcasm Detection",
    version="1.0.0",
    description="Text + audio + video sarcasm classifier with calibrated confidence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

paths = get_paths()
keyframes_dir = paths["keyframes"]
keyframes_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/keyframes", StaticFiles(directory=str(keyframes_dir)), name="keyframes")

DEMO_DIR = ROOT / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")

FRONTEND_DIST = ROOT / "app" / "frontend" / "dist"
FRONTEND_PUBLIC = ROOT / "app" / "frontend" / "public"


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    context: str = ""


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict/text")
def predict_text(body: TextRequest):
    from app.backend.inference import get_service

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    return get_service().predict_text(text, body.context or "")


@app.post("/predict/multimodal")
async def predict_multimodal(
    video: UploadFile = File(...),
    transcript: str | None = Form(None),
    context: str | None = Form(None),
):
    from app.backend.inference import get_service

    suffix = Path(video.filename or "clip.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".m4a"}:
        raise HTTPException(415, f"Unsupported file type: {suffix or 'unknown'}")
    data = await video.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > 80 * 1024 * 1024:
        raise HTTPException(413, "File too large (80 MB max)")
    tmp = Path(tempfile.gettempdir()) / f"mustard_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(data)
    try:
        return get_service().predict_multimodal(str(tmp), transcript=transcript, context=context or "")
    finally:
        tmp.unlink(missing_ok=True)


@app.get("/model/metrics")
def model_metrics():
    metrics_path = paths["metrics"] / "metrics.json"
    if metrics_path.exists():
        return load_json(metrics_path)
    return {
        "accuracy": None,
        "macro_f1": None,
        "precision": None,
        "recall": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "ablation_table": {},
        "ece": None,
        "note": "Train the models first (`python train.py && python evaluate.py`).",
    }


@app.get("/examples/demo")
def demo_clips():
    manifest = DEMO_DIR / "manifest.json"
    if manifest.exists():
        return {"items": json.loads(manifest.read_text(encoding="utf-8"))}
    return {"items": []}


@app.get("/examples")
def examples():
    return {
        "items": [
            {
                "title": "Meeting",
                "text": "Oh great, another meeting that could have been an email.",
                "label_hint": "sarcastic",
            },
            {
                "title": "Traffic",
                "text": "I love being stuck in traffic. Best part of my day.",
                "label_hint": "sarcastic",
            },
            {
                "title": "Trust",
                "text": "Sure, I completely trust the guy who just spilled coffee on the server.",
                "label_hint": "sarcastic",
            },
            {
                "title": "Last time",
                "text": "Yeah, because that worked out so well last time.",
                "label_hint": "sarcastic",
            },
            {
                "title": "Dinner",
                "text": "I'm really looking forward to dinner with you tonight.",
                "label_hint": "non_sarcastic",
            },
            {
                "title": "Sheldon (MUStARD++)",
                "text": "I'm just inferring this is a couch because the evidence suggests the coffee table is having a tiny garage sale.",
                "context": "So Penny's a little messy. A little messy? The Mandelbrot set of complex numbers is a little messy. This is chaos.",
                "label_hint": "sarcastic",
            },
        ]
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
elif (ROOT / "app" / "frontend" / "index.html").exists():
    @app.get("/")
    def index():
        return FileResponse(ROOT / "app" / "frontend" / "index.html")
