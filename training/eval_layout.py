"""Model 1 evaluation on a COCO-format validation set.

Reports mAP@0.5 and per-class AP. Also writes a handful of qualitative
overlays to `--qual-dir` so you can eyeball predictions next to ground
truth.

Usage:
  uv run python training/eval_layout.py \\
      --checkpoint training/checkpoints/pretrain_v1/best \\
      --coco data/deepscoresv2/model1_coco_test.json \\
      --images-root data/deepscoresv2/ds2_dense/images \\
      --qual-dir training/eval_out/qual_pretrain_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS  # noqa: E402
from training.infer_layout import draw as draw_overlay  # noqa: E402


def run_eval(checkpoint: Path, coco_path: Path, images_root: Path,
             *, threshold: float = 0.3,
             qual_dir: Path | None = None, n_qual: int = 8) -> dict:
    from torchmetrics.detection import MeanAveragePrecision

    model = RTDetrV2ForObjectDetection.from_pretrained(
        str(checkpoint),
        id2label=MODEL1_ID_TO_CLASS,
        label2id={v: k for k, v in MODEL1_ID_TO_CLASS.items()},
    )
    processor = AutoImageProcessor.from_pretrained(str(checkpoint))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    coco = json.loads(Path(coco_path).read_text())
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True,
                                  iou_type="bbox")

    qual_dir = Path(qual_dir) if qual_dir else None
    if qual_dir:
        qual_dir.mkdir(parents=True, exist_ok=True)

    n_qual_saved = 0
    for img_meta in coco["images"]:
        img_path = images_root / img_meta["file_name"]
        if not img_path.exists():
            continue
        pil = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]])
        res = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold,
        )[0]

        gt_boxes = []
        gt_labels = []
        for a in anns_by_img.get(img_meta["id"], []):
            x, y, w, h = a["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(a["category_id"])

        metric.update(
            preds=[{
                "boxes": res["boxes"].cpu(),
                "scores": res["scores"].cpu(),
                "labels": res["labels"].cpu(),
            }],
            target=[{
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32)
                if gt_boxes else torch.zeros((0, 4)),
                "labels": torch.tensor(gt_labels, dtype=torch.long)
                if gt_labels else torch.zeros((0,), dtype=torch.long),
            }],
        )

        if qual_dir and n_qual_saved < n_qual:
            preds = [{
                "label": MODEL1_ID_TO_CLASS[int(l)],
                "score": float(s),
                "bbox": [float(v) for v in b.tolist()],
            } for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
            out = qual_dir / f"{img_meta['id']:06d}_preds.png"
            draw_overlay(pil, preds, out)
            n_qual_saved += 1

    out = metric.compute()
    return {
        "mAP": float(out.get("map", float("nan"))),
        "mAP_50": float(out.get("map_50", float("nan"))),
        "mAP_75": float(out.get("map_75", float("nan"))),
        "per_class_AP": {
            MODEL1_CLASSES[i]: float(v)
            for i, v in enumerate(out.get("map_per_class", []).tolist())
            if i < len(MODEL1_CLASSES)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--qual-dir", type=Path, default=None)
    ap.add_argument("--n-qual", type=int, default=8)
    args = ap.parse_args()

    results = run_eval(args.checkpoint, args.coco, args.images_root,
                       threshold=args.threshold,
                       qual_dir=args.qual_dir, n_qual=args.n_qual)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
