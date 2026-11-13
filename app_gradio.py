"""Web app: upload an MRI image, get the prediction and a Grad-CAM heat-map.

    python app.py --checkpoint outputs/best_model.pt
Then open http://127.0.0.1:7860 in your browser.
"""
from __future__ import annotations

import argparse

from PIL import Image

from inference import load_checkpoint, predict_image
from utils import get_device, pretty_name

DISCLAIMER = (
    "**Research / educational demo - not a medical device.** "
    "Results must never be used for diagnosis or treatment decisions. "
    "Always consult a qualified clinician."
)


def build_demo(checkpoint: str, device_name: str = "auto"):
    import gradio as gr

    device = get_device(device_name)
    model, ckpt = load_checkpoint(checkpoint, device)

    def infer(image: Image.Image | None):
        if image is None:
            return None, None, "Upload an MRI image to begin."
        result = predict_image(model, ckpt, image, device, with_cam=True)
        scores = {pretty_name(c): p for c, p in result["probs"].items()}
        summary = f"### Prediction: {pretty_name(result['label'])} ({result['confidence'] * 100:.1f}% confidence)"
        if result["tumor_probability"] is not None:
            flag = "tumor pattern detected" if result["tumor_detected"] else "no tumor pattern detected"
            summary += f"\n\nTumor probability: **{result['tumor_probability'] * 100:.1f}%** - {flag}."
        summary += (
            "\n\nThe heat-map shows where the *model* looked; it is not a tumor outline.\n\n" + DISCLAIMER
        )
        return scores, Image.fromarray(result["overlay"]), summary

    with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
        gr.Markdown("# Brain Tumor MRI Classifier\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="MRI image (JPG / PNG)")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                label = gr.Label(label="Class probabilities", num_top_classes=len(ckpt["classes"]))
                cam = gr.Image(label="Grad-CAM heat-map", type="pil")
                text = gr.Markdown()
        btn.click(infer, inputs=inp, outputs=[label, cam, text])
        inp.upload(infer, inputs=inp, outputs=[label, cam, text])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    args = parser.parse_args()
    demo = build_demo(args.checkpoint, args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
