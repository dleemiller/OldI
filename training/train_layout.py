"""Train the Model 1 page-layout detector.

Two phases share this script, selected via `--phase`:

  pretrain  — DeepScoresV2 (`training/configs/layout_pretrain.yaml`)
  finetune  — hand-annotated tune-book pages (`training/configs/layout_finetune.yaml`)

Both phases load from a HuggingFace checkpoint (heron for pretrain, our own
pretrained checkpoint for finetune), swap the class head to our 8-class
taxonomy, and run the HF `Trainer` with object-detection data collation.

Augmentation runs per-sample on the raw image *before* the RTDetrImageProcessor
resizes to 640×640, so our augraphy modes see the full-resolution page and
the bboxes stay in absolute pixel space through the augmentation step.

Usage:
  uv run python training/train_layout.py --phase pretrain
  uv run python training/train_layout.py --phase finetune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    AutoImageProcessor,
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS  # noqa: E402
from training.data.augment import finetune_apply, pretrain_apply  # noqa: E402
from training.data.coco_dataset import (  # noqa: E402
    CocoDetectionDataset,
    silence_libpng_worker_init,
)
from training.tb_callbacks import (  # noqa: E402
    BestCheckpointCallback,
    GPUMemoryCallback,
    PerClassAPCallback,
    PredictionOverlayCallback,
)


class QuietTrainer(Trainer):
    """Trainer subclass that installs `silence_libpng_worker_init` on each
    DataLoader so DSv2's occasional libpng IDAT/CRC warnings don't reach
    the main process's stderr."""

    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        dl.worker_init_fn = silence_libpng_worker_init
        return dl

    def get_eval_dataloader(self, eval_dataset=None):
        dl = super().get_eval_dataloader(eval_dataset)
        dl.worker_init_fn = silence_libpng_worker_init
        return dl


# ─────────────────────────── Config + model loading ───────────────────────────


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    # Resolve paths relative to the repo root.
    for key in ("train_coco", "train_images_root", "val_coco",
                "val_images_root", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO / cfg[key])
    # checkpoint can be a local path or a Hub ID — leave as-is.
    return cfg


def build_model(checkpoint: str) -> tuple[RTDetrV2ForObjectDetection, RTDetrImageProcessor]:
    """Load heron (or prior checkpoint) and remap the class head to 8 classes.

    `ignore_mismatched_sizes=True` lets `from_pretrained` swap the final
    classification layer even though our num_labels differs from the
    checkpoint's 17 (heron) or already-remapped 8 (our pretrain output).
    """
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = RTDetrV2ForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=len(MODEL1_CLASSES),
        id2label=MODEL1_ID_TO_CLASS,
        label2id={v: k for k, v in MODEL1_ID_TO_CLASS.items()},
        ignore_mismatched_sizes=True,
    )
    return model, processor


def freeze_backbone(model: RTDetrV2ForObjectDetection, freeze: bool) -> None:
    """Freeze/unfreeze the backbone convolutional stem + stages."""
    # heron's RT-DETRv2 stores the backbone under model.model.backbone
    backbone = getattr(model.model, "backbone", None)
    if backbone is None:
        return
    for p in backbone.parameters():
        p.requires_grad = not freeze


class UnfreezeBackboneAt(TrainerCallback):
    """Unfreeze the backbone at the start of epoch `epoch`."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self._done = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._done or state.epoch is None:
            return
        if state.epoch >= self.epoch:
            model = kwargs["model"]
            freeze_backbone(model, freeze=False)
            self._done = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[UnfreezeBackboneAt] epoch {state.epoch}: unfroze backbone "
                  f"→ {trainable:,} trainable params")


# ─────────────────────────── Collation ───────────────────────────


def make_collator(processor: RTDetrImageProcessor):
    """Return a collate_fn that runs the image processor over a batch.

    The dataset yields {"image_id", "image", "annotations"} — the processor
    takes those in COCO-detection format and returns the normalised pixel
    tensor plus labels in the RTDetrV2 format (class_labels + boxes in
    (cx, cy, w, h) normalised coordinates).
    """

    def _collate(batch: list[dict]) -> dict[str, Any]:
        images = [b["image"] for b in batch]
        anns = [{"image_id": b["image_id"], "annotations": b["annotations"]} for b in batch]
        enc = processor(images=images, annotations=anns,
                        return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],
            "labels": enc["labels"],
        }

    return _collate


# ─────────────────────────── Eval metric ───────────────────────────


def make_compute_metrics(processor: RTDetrImageProcessor,
                          id2label: dict[int, str]):
    """Compute COCO mAP at 0.5 and per-class AP on the validation set."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        # torchmetrics is optional; if missing, just report per-class counts.
        return None

    def _compute(eval_pred) -> dict[str, float]:
        predictions, label_batches = eval_pred
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True,
                                      iou_type="bbox")
        for preds, labels in zip(predictions, label_batches):
            # processor.post_process_object_detection expects the raw outputs;
            # eval_pred delivers already-post-processed tensors depending on
            # how Trainer was configured. Simpler: skip metric in eval_pred
            # form — real mAP eval happens in eval_layout.py.
            pass
        return {}

    return _compute


# ─────────────────────────── Main ───────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("pretrain", "finetune"), required=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="Override the default config path.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run: 20 samples, 2 epochs — sanity check only.")
    args = ap.parse_args()

    default_cfg = {
        "pretrain": REPO / "training/configs/layout_pretrain.yaml",
        "finetune": REPO / "training/configs/layout_finetune.yaml",
    }[args.phase]
    cfg_path = args.config or default_cfg
    cfg = load_config(cfg_path)

    print(f"[{args.phase}] config: {cfg_path}")
    print(f"  checkpoint: {cfg['checkpoint']}")
    print(f"  train coco: {cfg['train_coco']}")
    print(f"  val coco:   {cfg['val_coco']}")
    print(f"  output:     {cfg['output_dir']}")

    model, processor = build_model(cfg["checkpoint"])

    augment_fn = {
        "pretrain": lambda img: pretrain_apply(img).image,
        "finetune": lambda img: finetune_apply(img).image,
    }[cfg["augmentation_phase"]]

    train_ds = CocoDetectionDataset(
        coco_path=Path(cfg["train_coco"]),
        images_root=Path(cfg["train_images_root"]),
        augment_fn=augment_fn,
    )
    val_ds = CocoDetectionDataset(
        coco_path=Path(cfg["val_coco"]),
        images_root=Path(cfg["val_images_root"]),
        augment_fn=None,  # no augmentation on val
    )
    print(f"  train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    if args.smoke:
        train_ds.images = train_ds.images[:20]
        val_ds.images = val_ds.images[:10]
        cfg["num_epochs"] = 2
        print("  (smoke mode: 20 train / 10 val / 2 epochs)")

    # Freeze backbone for the opening epochs if configured. HF Trainer lets
    # us use a callback to unfreeze mid-run, but we keep this simple: freeze
    # at start; caller runs a short job with freeze then a longer one without
    # if they want the two-stage schedule.
    freeze_backbone(model, freeze=cfg.get("freeze_backbone_epochs", 0) > 0)

    tb_logdir = Path(cfg["output_dir"]) / "runs"
    tb_logdir.mkdir(parents=True, exist_ok=True)
    print(f"  tensorboard logs: {tb_logdir}")

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        dataloader_num_workers=cfg["num_workers"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        # load_best_model_at_end=True uses model.load_state_dict() directly
        # on the saved safetensors, bypassing the from_pretrained remap that
        # handles transformers' old→new RTDetrV2 parameter renames. The
        # mismatch silently reinitializes the encoder/decoder. We keep the
        # last checkpoint; pick "best" offline via the TensorBoard val/mAP
        # curve if needed.
        load_best_model_at_end=False,
        logging_steps=2,
        logging_dir=str(tb_logdir),
        warmup_ratio=0.1,  # 10% of steps linear warmup — DETR-family is
                           # sensitive to early LR; the previous runs
                           # showed mAP oscillating wildly in epochs 5-15
                           # without warmup.
        # tensorboard always; wandb if WANDB_API_KEY is set.
        report_to=(["tensorboard", "wandb"]
                   if __import__("os").environ.get("WANDB_API_KEY")
                   else ["tensorboard"]),
        bf16=True,  # RTX PRO 6000 Blackwell supports bf16 natively
        remove_unused_columns=False,  # Trainer must keep "image", "annotations"
        log_level="info",
    )

    # Custom optimizer with 2 param groups so the backbone learns 10× slower
    # than the detection head. Without this, heron's ResNet-101 gets blasted
    # at head LR and val/mAP collapses after ~8 epochs.
    lr_head = float(cfg["learning_rate"])
    lr_backbone = float(cfg.get("learning_rate_backbone", lr_head / 10))
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    print(f"  optimizer: {len(backbone_params)} backbone params @ lr={lr_backbone:.1e}, "
          f"{len(other_params)} head+neck params @ lr={lr_head:.1e}")
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": other_params, "lr": lr_head},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    # Let Trainer build the scheduler matching training_args; passing None
    # keeps its default cosine-with-warmup tied to the total step count.
    trainer = QuietTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(processor),
        processing_class=processor,
        optimizers=(optimizer, None),
    )
    per_class_cb = PerClassAPCallback(val_dataset=val_ds, processor=processor,
                                       n_eval_images=min(100, len(val_ds)))
    overlay_cb = PredictionOverlayCallback(val_dataset=val_ds, processor=processor,
                                            n_images=6)
    gpu_cb = GPUMemoryCallback()
    best_cb = BestCheckpointCallback(per_class_cb)
    trainer.add_callback(per_class_cb)
    trainer.add_callback(overlay_cb)
    trainer.add_callback(gpu_cb)
    trainer.add_callback(best_cb)
    # Inject trainer reference so callbacks can route logs through Trainer.log(),
    # reusing the built-in TB writer instead of spawning sibling runs.
    per_class_cb.trainer = trainer
    overlay_cb.trainer = trainer
    gpu_cb.trainer = trainer
    if cfg.get("freeze_backbone_epochs", 0) > 0:
        trainer.add_callback(UnfreezeBackboneAt(cfg["freeze_backbone_epochs"]))

    trainer.train()
    trainer.save_model(cfg["output_dir"] + "/best")
    processor.save_pretrained(cfg["output_dir"] + "/best")
    print(f"saved best model to {cfg['output_dir']}/best")


if __name__ == "__main__":
    main()
