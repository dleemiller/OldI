"""Run Model 1 on a single page and render bbox overlays.

Used both as a qualitative-sanity tool and as the pseudo-labeller for
CVAT pre-populated annotations.

Usage:
  uv run python training/infer_layout.py \\
      --checkpoint training/checkpoints/pretrain_v1/best \\
      --image data/01_pages/oneill_dance/page_0030.png \\
      --out training/eval_out/p30_preds.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS  # noqa: E402

# Visualisation colours, one per class (RGB).
_CLASS_COLORS = {
    "staff":                (231,  76,  60),
    "measure":              ( 52, 152, 219),
    "tune-title":           ( 46, 204, 113),
    "tempo-marking":        (241, 196,  15),
    "tune-number":          (155,  89, 182),
    "composer-attribution": ( 26, 188, 156),
    "footer":               (149, 165, 166),
    "staff-header":         (230, 126,  34),
    "page-number":          (142,  68, 173),
    "text-block":           ( 52,  73,  94),
    "inline-lyrics":        (255, 105, 180),
    "subtitle":             (106,  90, 205),
    "page-title":           ( 39, 174,  96),
}


def load(checkpoint: Path) -> tuple[RTDetrV2ForObjectDetection, AutoImageProcessor]:
    model = RTDetrV2ForObjectDetection.from_pretrained(
        str(checkpoint),
        id2label=MODEL1_ID_TO_CLASS,
        label2id={v: k for k, v in MODEL1_ID_TO_CLASS.items()},
    )
    processor = AutoImageProcessor.from_pretrained(str(checkpoint))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor


def predict(model, processor, image: Image.Image, *, threshold: float = 0.5) -> list[dict]:
    """Return predictions as [{label, score, bbox:[x0,y0,x1,y1]}]."""
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold,
    )[0]
    preds: list[dict] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        cls = MODEL1_ID_TO_CLASS[int(label)]
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        preds.append({
            "label": cls,
            "score": float(score),
            "bbox": [x0, y0, x1, y1],
        })
    return preds


def draw(image: Image.Image, preds: list[dict], out_path: Path) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for p in preds:
        x0, y0, x1, y1 = p["bbox"]
        color = _CLASS_COLORS.get(p["label"], (0, 0, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{p['label']} {p['score']:.2f}"
        draw.rectangle([x0, y0 - 18, x0 + 6 * len(label), y0],
                       fill=color)
        draw.text((x0 + 2, y0 - 16), label, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    model, processor = load(args.checkpoint)
    image = Image.open(args.image).convert("RGB")
    preds = predict(model, processor, image, threshold=args.threshold)
    print(f"predictions (thresh={args.threshold}): {len(preds)}")
    by_class: dict[str, int] = {}
    for p in preds:
        by_class[p["label"]] = by_class.get(p["label"], 0) + 1
    for cls in MODEL1_CLASSES:
        print(f"  {cls}: {by_class.get(cls, 0)}")
    draw(image, preds, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
