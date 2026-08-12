#!/usr/bin/env python3
"""Deepfake detector — demo frontend.

Upload a face image; the trained detector returns REAL or FAKE with a
confidence score. Runs locally in the browser via Gradio.

    python app.py                                   # uses runs/deepdetect/baseline/best.pt
    python app.py --checkpoint runs/deepdetect/fair/best.pt
    python app.py --share                           # temporary public link

The app reuses the exact preprocessing the model was trained with: the
same face detector and crop, the same resize, the same normalisation —
so what the model sees at demo time matches training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _bootstrap() -> None:
    """Find the `sadf` package whether it sits in src/ or at the repo root."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise ImportError("Could not locate the `sadf` package. Run app.py from the repo.")


_bootstrap()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from sadf.config import _dotify  # noqa: E402
from sadf.data.faces import FaceCropper  # noqa: E402
from sadf.models.detector import build_model  # noqa: E402

# Filled in by main().
_STATE: dict = {}
# Cache of loaded models so switching in the UI is instant after first load.
_CACHE: dict = {}

# Where to look for trained checkpoints, with friendly names.
CHECKPOINT_LOCATIONS = {
    # Race-balanced 140k models (the current study) — listed first so the
    # app defaults to these.
    "ViT": "runs/models140k/vit/baseline/best.pt",
    "Xception": "runs/models140k/xception/baseline/best.pt",
    "Swin": "runs/models140k/swin/baseline/best.pt",
    # Earlier DeepDetect models (kept as fallback if present).
    # "ViT-Small": "runs/models/vit/baseline/best.pt",
    # "Xception": "runs/models/xception/baseline/best.pt",
    # "Swin-Tiny": "runs/models/swin/baseline/best.pt",
    # "Baseline (deepdetect)": "runs/deepdetect/baseline/best.pt",
    # "Fair (deepdetect)": "runs/deepdetect/fair/best.pt",
}


def discover_models() -> dict:
    """Return {display_name: path} for checkpoints that actually exist."""
    found = {name: p for name, p in CHECKPOINT_LOCATIONS.items() if Path(p).exists()}
    return found


def load_model(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _dotify(ckpt["config"])
    model = build_model(cfg, n_groups=len(ckpt["group_vocab"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tf = T.Compose([
        T.Resize((int(cfg.image.size), int(cfg.image.size))),
        T.ToTensor(),
        T.Normalize(cfg.image.mean, cfg.image.std),
    ])
    cropper = FaceCropper(cfg.face.detector, cfg.face.min_face, cfg.face.margin)
    return model, cfg, tf, cropper


def get_model(display_name: str):
    """Load (and cache) a model by its display name."""
    if display_name in _CACHE:
        return _CACHE[display_name]
    path = _STATE["available"][display_name]
    bundle = load_model(path, _STATE["device"])
    _CACHE[display_name] = bundle
    return bundle


def predict(image: "Image.Image", model_name: str = None):
    """Return (label_dict, message) for the Gradio interface."""
    if image is None:
        return {}, "Please upload an image."

    model_name = model_name or _STATE["default_model"]
    model, cfg, tf, cropper = get_model(model_name)
    device = _STATE["device"]
    threshold = float(cfg.train.threshold)

    # Reuse the training-time face gate/crop. Fall back to the whole image
    # if no face is detected, but tell the user (accuracy may be lower).
    rgb = np.array(image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    crop_bgr, found = cropper.crop(bgr)
    if crop_bgr is None:
        crop_bgr, found = bgr, False
    face_img = Image.fromarray(crop_bgr[:, :, ::-1])

    x = tf(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob_fake = float(torch.sigmoid(out["logits"])[0])

    prob_real = 1.0 - prob_fake
    labels = {"FAKE (AI-generated)": prob_fake, "REAL (authentic)": prob_real}

    verdict = "FAKE (AI-generated)" if prob_fake >= threshold else "REAL (authentic)"
    conf = max(prob_fake, prob_real) * 100
    note = "" if found else (
        "\n\n_No face was detected, so the whole image was analysed. "
        "For best results upload a clear, front-facing face photo._"
    )
    msg = (f"### Prediction: **{verdict}**\nConfidence: **{conf:.1f}%**  "
           f"·  _model: {model_name}_{note}")
    return labels, msg


def build_interface():
    import gradio as gr

    choices = list(_STATE["available"].keys())
    default = _STATE["default_model"]

    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown(
            "# Deepfake Face Detector\n"
            "Upload a face image and the model will classify it as **real** "
            "or **AI-generated**, with a confidence score. Use the dropdown to "
            "compare different model architectures on the same image.\n\n"
            f"_device: {_STATE['device']}_"
        )
        with gr.Row():
            with gr.Column():
                model_dd = gr.Dropdown(
                    choices=choices, value=default, label="Model architecture",
                    info="Switch architectures and re-run on the same image",
                )
                inp = gr.Image(type="pil", label="Upload a face image", height=320)
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                bars = gr.Label(num_top_classes=2, label="Class probabilities")

        btn.click(predict, inputs=[inp, model_dd], outputs=[bars, verdict])
        inp.change(predict, inputs=[inp, model_dd], outputs=[bars, verdict])
        model_dd.change(predict, inputs=[inp, model_dd], outputs=[bars, verdict])

        # gr.Markdown(
        #     "---\n_Research demo (COS700). Detectors are trained on a limited "
        #     "dataset; predictions are indicative, not definitive. Skin-tone "
        #     "fairness analysis uses an ITA proxy with known limitations and is "
        #     "an ongoing part of this study._"
        # )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="Force a single checkpoint (otherwise auto-discovers all)")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--share", action="store_true", help="Create a public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        available = {Path(args.checkpoint).parent.parent.name or "model": args.checkpoint}
    else:
        available = discover_models()
        if not available:
            raise SystemExit(
                "No trained checkpoints found. Expected at least one of:\n  " +
                "\n  ".join(CHECKPOINT_LOCATIONS.values()) +
                "\nTrain a model first (scripts/run_models.py --train vit)."
            )

    _STATE.update({
        "available": available,
        "default_model": next(iter(available)),
        "device": device,
    })
    # Warm-load the default so the first prediction isn't slow.
    get_model(_STATE["default_model"])
    print(f"Available models: {list(available)} on {device}. Launching UI ...")
    build_interface().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()