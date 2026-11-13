"""Website + API: the project's landing page with a working "Try it" analyser.

    python web_app.py                      # http://127.0.0.1:8000
    python web_app.py --checkpoint outputs/best_model.pt --port 8080

Routes
    GET  /               the website (web/index.html)
    GET  /api/status     is a trained model loaded? which classes?
    POST /api/predict    multipart image upload -> probabilities + Grad-CAM overlay
    POST /api/feedback   saves the feedback form to feedback/feedback.jsonl

The page still opens without a trained model (the Try-it box then explains what to do).
Only you, on the computer running this, can read the saved feedback.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from inference import load_checkpoint, predict_image
from utils import get_device, pretty_name

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "web" / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


class Feedback(BaseModel):
    name: str = Field("", max_length=80)
    email: str = Field("", max_length=120)
    topic: str = Field("Suggestion", max_length=40)
    rating: int | None = Field(None, ge=1, le=5)
    message: str = Field(..., min_length=5, max_length=1000)


def _png_data_url(array) -> str:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(checkpoint: str, device_name: str = "auto", feedback_file: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Brain Tumor MRI Classifier", docs_url=None, redoc_url=None)
    device = get_device(device_name)
    feedback_path = Path(feedback_file) if feedback_file else ROOT / "feedback" / "feedback.jsonl"

    state: dict = {"model": None, "ckpt": None, "error": None}
    model_lock = threading.Lock()   # Grad-CAM registers hooks on the shared model: one request at a time
    feedback_lock = threading.Lock()

    try:
        state["model"], state["ckpt"] = load_checkpoint(checkpoint, device)
        print(f"Model loaded: {state['ckpt']['model_name']} on {device} ({checkpoint})")
    except FileNotFoundError:
        state["error"] = "no_model"
        print(f"[notice] No trained model at '{checkpoint}'. The website works; analysis needs: python train.py --data_dir data")
    except Exception as exc:  # noqa: BLE001 - corrupt or incompatible checkpoint
        state["error"] = "bad_model"
        print(f"[warning] Could not load '{checkpoint}': {exc}")

    @app.get("/", include_in_schema=False)
    def index():
        if not INDEX.is_file():
            raise HTTPException(500, "web/index.html is missing.")
        return FileResponse(INDEX, media_type="text/html")

    @app.get("/api/status")
    def status():
        ckpt = state["ckpt"]
        return {
            "ready": state["model"] is not None,
            "reason": state["error"],
            "model": ckpt["model_name"] if ckpt else None,
            "classes": ckpt["classes"] if ckpt else None,
            "device": str(device),
        }

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        if state["model"] is None:
            msg = ("No trained model found. Train one first: python train.py --data_dir data"
                   if state["error"] == "no_model" else "The model file could not be loaded.")
            raise HTTPException(503, msg)
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 10 MB.")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "That file is not a readable image (use JPG or PNG).")

        with model_lock:
            result = predict_image(state["model"], state["ckpt"], image, device, with_cam=True)

        probs = sorted(result["probs"].items(), key=lambda kv: -kv[1])
        return {
            "label": result["label"],
            "label_pretty": pretty_name(result["label"]),
            "confidence": result["confidence"],
            "probs": [{"cls": c, "name": pretty_name(c), "p": p} for c, p in probs],
            "tumor_probability": result["tumor_probability"],
            "tumor_detected": result["tumor_detected"],
            "original": _png_data_url(result["original"]),
            "overlay": _png_data_url(result["overlay"]),
            "model": state["ckpt"]["model_name"],
        }

    @app.post("/api/feedback")
    def save_feedback(item: Feedback):
        record = item.model_dump()
        record["createdAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with feedback_lock:
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="Run the website with the working MRI analyser.")
    p.add_argument("--checkpoint", default="outputs/best_model.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 so only this computer can reach it.")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    app = create_app(args.checkpoint, args.device)
    print(f"\nOpen http://{args.host}:{args.port} in your browser (Ctrl+C to stop)\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
