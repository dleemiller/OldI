"""TensorBoard callbacks for per-class AP, sample predictions, GPU usage.

All scalars flow through Trainer's own `log()` method so they land in the
SAME tfevents file as the built-in `train/loss`, `eval/loss`, etc. — a
separate `SummaryWriter` would put our scalars in a sibling run that TB
renders disconnected from Trainer's curves.

Images still need their own writer because `Trainer.log()` is scalar-only;
we grab Trainer's TB callback writer lazily on first use so overlay images
go into the same event file too.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback

from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS


def _grab_trainer_tb_writer(trainer) -> "object | None":
    """Return the SummaryWriter from Trainer's TensorBoardCallback."""
    if trainer is None:
        return None
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, TensorBoardCallback) and getattr(cb, "tb_writer", None):
            return cb.tb_writer
    return None


# ─────────────────────── Per-class AP on val subset ───────────────────────


class PerClassAPCallback(TrainerCallback):
    """On each eval, compute mAP + per-class AP and route scalars through
    Trainer.log() so they join the built-in curves."""

    def __init__(
        self,
        val_dataset,
        processor,
        *,
        n_eval_images: int = 100,
        threshold: float = 0.05,
    ) -> None:
        self.val_dataset = val_dataset
        self.processor = processor
        self.n_eval_images = n_eval_images
        self.threshold = threshold
        self.trainer = None  # injected by train_layout.main after Trainer init

    def on_evaluate(self, args, state, control, **kwargs):
        from torchmetrics.detection import MeanAveragePrecision

        model = kwargs["model"]
        device = model.device
        model.eval()
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True,
                                      iou_type="bbox",
                                      max_detection_thresholds=[1, 10, 100])

        total_preds = 0
        total_preds_above_50 = 0
        for i in range(min(self.n_eval_images, len(self.val_dataset))):
            sample = self.val_dataset[i]
            pil = sample["image"]
            anns = sample["annotations"]
            inputs = self.processor(images=pil, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([pil.size[::-1]])
            res = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold,
            )[0]
            total_preds += int(res["scores"].shape[0])
            total_preds_above_50 += int((res["scores"] >= 0.5).sum().item())
            gt_boxes = []
            gt_labels = []
            for a in anns:
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

        results = metric.compute()
        log_payload: dict[str, float] = {
            "val/mAP":     float(results["map"]),
            "val/mAP_50":  float(results["map_50"]),
            "val/mAP_75":  float(results["map_75"]),
            "val/mAR_100": float(results["mar_100"]),
            f"val/num_preds_at_thresh_{self.threshold}": float(total_preds),
            "val/num_preds_above_0.5": float(total_preds_above_50),
        }
        per_class = results.get("map_per_class", None)
        if per_class is not None:
            pc = per_class.tolist() if hasattr(per_class, "tolist") else per_class
            if not isinstance(pc, list):
                pc = [pc]
            classes_with_gt = results.get("classes", None)
            if classes_with_gt is not None and hasattr(classes_with_gt, "tolist"):
                classes_with_gt = classes_with_gt.tolist()
            if isinstance(classes_with_gt, list) and len(classes_with_gt) == len(pc):
                for cid, ap in zip(classes_with_gt, pc):
                    if 0 <= cid < len(MODEL1_CLASSES) and ap >= 0:
                        log_payload[f"val_class_AP/{MODEL1_CLASSES[int(cid)]}"] = float(ap)
            else:
                for cid, ap in enumerate(pc):
                    if cid < len(MODEL1_CLASSES) and ap >= 0:
                        log_payload[f"val_class_AP/{MODEL1_CLASSES[cid]}"] = float(ap)

        if self.trainer is not None:
            self.trainer.log(log_payload)
        # Track for best-checkpoint preservation
        self._last_map = float(results["map"])
        self._last_step = state.global_step
        self._last_epoch = state.epoch


class BestCheckpointCallback(TrainerCallback):
    """After each checkpoint save, if the latest val/mAP (from
    PerClassAPCallback) is a new best, copy that checkpoint dir to
    `<output_dir>/best_by_map/` so it survives save_total_limit pruning.

    Tracks only the single best — writes a small JSON with (epoch, step,
    map) alongside the copied checkpoint."""

    def __init__(self, per_class_cb: "PerClassAPCallback") -> None:
        self.per_class_cb = per_class_cb
        self.best_map = -1.0
        self._best_dir: Path | None = None

    def on_save(self, args, state, control, **kwargs):
        last_map = getattr(self.per_class_cb, "_last_map", None)
        last_step = getattr(self.per_class_cb, "_last_step", None)
        last_epoch = getattr(self.per_class_cb, "_last_epoch", None)
        if last_map is None or last_step is None:
            return
        # The checkpoint that was JUST saved corresponds to the same step as
        # the preceding eval only when eval_strategy="epoch" and save=epoch.
        if last_step != state.global_step:
            return
        if last_map <= self.best_map:
            return
        self.best_map = last_map
        src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        dst = Path(args.output_dir) / "best_by_map"
        if not src.exists():
            return
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        meta = {
            "epoch": last_epoch,
            "global_step": state.global_step,
            "val_mAP": last_map,
        }
        (dst / "best_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[BestCheckpointCallback] new best val/mAP={last_map:.4f} @ epoch {last_epoch} "
              f"(step {state.global_step}) → {dst}")


# ─────────────────────── Prediction overlay grid ───────────────────────


class PredictionOverlayCallback(TrainerCallback):
    """Log N overlay PNGs to TB Images tab on each eval. Uses Trainer's own
    TB writer so images appear alongside the scalar run."""

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

    def __init__(
        self,
        val_dataset,
        processor,
        *,
        n_images: int = 6,
        threshold: float = 0.05,
    ) -> None:
        self.val_dataset = val_dataset
        self.processor = processor
        self.n_images = n_images
        self.threshold = threshold
        self.trainer = None  # injected after Trainer init

    def on_evaluate(self, args, state, control, **kwargs):
        writer = _grab_trainer_tb_writer(self.trainer)
        if writer is None:
            return
        model = kwargs["model"]
        device = model.device
        model.eval()
        step = state.global_step

        from PIL import ImageDraw, ImageFont
        for i in range(min(self.n_images, len(self.val_dataset))):
            sample = self.val_dataset[i]
            pil = sample["image"].convert("RGB").copy()
            inputs = self.processor(images=pil, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([pil.size[::-1]])
            res = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold,
            )[0]
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
            for score, label, box in zip(
                res["scores"].cpu(), res["labels"].cpu(), res["boxes"].cpu()
            ):
                x0, y0, x1, y1 = [float(v) for v in box.tolist()]
                cls = MODEL1_ID_TO_CLASS[int(label)]
                color = self._CLASS_COLORS.get(cls, (0, 0, 0))
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                tag = f"{cls} {float(score):.2f}"
                draw.rectangle([x0, y0 - 18, x0 + 7 * len(tag), y0], fill=color)
                draw.text((x0 + 2, y0 - 16), tag, fill=(255, 255, 255), font=font)
            max_edge = 1024
            if max(pil.size) > max_edge:
                scale = max_edge / max(pil.size)
                new_size = (int(pil.size[0] * scale), int(pil.size[1] * scale))
                pil = pil.resize(new_size, Image.LANCZOS)
            arr = np.array(pil).transpose(2, 0, 1)
            writer.add_image(f"val_overlay/{i:02d}", arr, step)
        writer.flush()


# ─────────────────────── GPU memory ───────────────────────


class GPUMemoryCallback(TrainerCallback):
    """Log peak VRAM (MB) per epoch via Trainer.log()."""

    def __init__(self) -> None:
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available() or self.trainer is None:
            return
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        self.trainer.log({
            "sys/gpu_peak_mb": float(peak_mb),
            "sys/gpu_alloc_mb": float(alloc_mb),
        })
        torch.cuda.reset_peak_memory_stats()
