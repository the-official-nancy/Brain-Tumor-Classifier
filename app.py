"""Web app: this project's website plus a prediction API.

    python app.py                                  # opens http://127.0.0.1:7860
    python app.py --checkpoint outputs/best_model.pt --port 7861

Routes
    GET  /              the website (web/index.html)
    GET  /api/status    is a trained model loaded?
    POST /api/predict   form field "image" -> class probabilities + Grad-CAM (PNG data URLs)
    POST /api/feedback  JSON feedback, appended to outputs/feedback.jsonl on this computer

Everything runs on your own computer and the server only listens on 127.0.0.1 by default.
Do not expose it on a network together with real patient images.
(The previous Gradio demo is still available as app_gradio.py.)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from inference import load_checkpoint, predict_image
from utils import get_device, pretty_name

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # refuse decompression bombs
TOPICS = {"Suggestion", "Bug report", "Question", "Praise", "Other"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def to_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(checkpoint: str, device_name: str = "auto", feedback_file: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    device = get_device(device_name)
    feedback_path = Path(feedback_file) if feedback_file else ROOT / "outputs" / "feedback.jsonl"

    state: dict = {"model": None, "ckpt": None, "message": None}
    try:
        state["model"], state["ckpt"] = load_checkpoint(checkpoint, device)
    except FileNotFoundError:
        state["message"] = (
            "No trained model found yet. Run  python train.py --data_dir data  "
            "and then restart this app."
        )
    except Exception as exc:  # noqa: BLE001 - corrupt / incompatible checkpoint
        state["message"] = f"The model file could not be loaded: {exc}"

    infer_lock = threading.Lock()  # Grad-CAM hooks the shared model, so run one prediction at a time
    feedback_lock = threading.Lock()

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="That file is too large. The limit is 15 MB."), 413

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/api/status")
    def status():
        ready = state["model"] is not None
        return jsonify(
            ready=ready,
            message=state["message"],
            model=state["ckpt"]["model_name"] if ready else None,
            classes=state["ckpt"]["classes"] if ready else None,
            device=str(device),
        )

    @app.post("/api/predict")
    def predict():
        if state["model"] is None:
            return jsonify(error=state["message"]), 503
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose an image file first."), 400
        try:
            image = Image.open(upload.stream)
            image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="That file could not be read as an image. Use JPG, PNG, BMP or TIFF."), 400

        with infer_lock:
            res = predict_image(state["model"], state["ckpt"], image, device, with_cam=True)

        probs = sorted(res["probs"].items(), key=lambda kv: -kv[1])
        return jsonify(
            label=res["label"],
            label_pretty=pretty_name(res["label"]),
            confidence=res["confidence"],
            tumor_probability=res["tumor_probability"],
            tumor_detected=res["tumor_detected"],
            probs=[{"class": c, "name": pretty_name(c), "p": p} for c, p in probs],
            original=to_data_url(res["original"]),
            overlay=to_data_url(res["overlay"]),
        )

    @app.post("/api/feedback")
    def feedback():
        data = request.get_json(silent=True)  # None unless Content-Type is application/json
        if not isinstance(data, dict):
            return jsonify(error="Invalid request."), 400
        message = str(data.get("message", "")).strip()
        if len(message) < 5:
            return jsonify(error="Please write a short message first."), 400
        email = str(data.get("email", "")).strip()[:120]
        if email and not EMAIL_RE.match(email):
            return jsonify(error="That email address does not look right."), 400
        topic = data.get("topic")
        rating = data.get("rating")
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "name": str(data.get("name", "")).strip()[:80],
            "email": email,
            "topic": topic if topic in TOPICS else "Other",
            "rating": rating if isinstance(rating, int) and not isinstance(rating, bool) and 1 <= rating <= 5 else None,
            "message": message[:1000],
        }
        with feedback_lock:
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return jsonify(ok=True)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the brain tumor MRI web app.")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--feedback_file", default=None, help="Default: outputs/feedback.jsonl")
    parser.add_argument("--no_browser", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    app = create_app(args.checkpoint, args.device, args.feedback_file)
    url = f"http://{args.host}:{args.port}"
    if args.host not in ("127.0.0.1", "localhost"):
        print("WARNING: the app is reachable from other computers. Do not use it with real patient images.")
    print(f"\nOpen {url} in your browser (press Ctrl+C to stop).\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
