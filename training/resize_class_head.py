"""Resize a trained RT-DETRv2 class head to a new num_labels.

Handles both truncation (new < old — drops trailing rows) and expansion
(new > old — appends fresh rows initialised the same way HF RT-DETRv2
initialises its class head, see reset_parameters).

Used to migrate a checkpoint across taxonomy changes without discarding
already-trained class logits: the rows for classes 0..min(old, new)-1
are preserved bit-for-bit, so Phase B finetune warm-starts from whatever
those rows learned.

Use:
  uv run python training/resize_class_head.py \\
      --in  training/checkpoints/pretrain_v1/best_by_map_7cls \\
      --out training/checkpoints/pretrain_v1/best_by_map_8cls \\
      --new-num-labels 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_ID_TO_CLASS  # noqa: E402


def _fresh_row(n_features: int, *, like: torch.Tensor, is_bias: bool) -> torch.Tensor:
    """Create a new head row initialised in the same distribution as the
    existing rows. RT-DETRv2 uses Xavier init on class_embed weights; we
    approximate by drawing normal with std matching the existing rows."""
    if is_bias:
        # Biases in RT-DETRv2 class heads are initialised to a constant
        # (a focal-loss prior, log((1-pi)/pi) with pi=0.01 ≈ -4.6).
        # Matching the existing mean is a reasonable fallback.
        return torch.full((n_features,), like.mean().item(), dtype=like.dtype)
    # Weights: match existing row std.
    std = like.std().item() or 0.02
    row = torch.randn(n_features, dtype=like.dtype) * std
    return row


def _resize(tensor: torch.Tensor, new_c: int, is_bias: bool,
            feature_dim_axis: int | None) -> torch.Tensor:
    """Resize along class axis (axis 0) to new_c rows."""
    old_c = tensor.shape[0]
    if new_c == old_c:
        return tensor.clone()
    if new_c < old_c:
        return tensor[:new_c].clone()
    # Expand: keep old rows, append new ones
    extras = []
    feat_shape = tensor.shape[1:] if tensor.ndim > 1 else ()
    n_feats = 1
    for d in feat_shape:
        n_feats *= d
    for _ in range(new_c - old_c):
        if is_bias:
            extras.append(_fresh_row(n_feats, like=tensor, is_bias=True).view(feat_shape))
        else:
            extras.append(_fresh_row(n_feats, like=tensor.view(-1, n_feats),
                                      is_bias=False).view(feat_shape))
    new_rows = torch.stack(extras, dim=0)
    return torch.cat([tensor, new_rows], dim=0).clone()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--new-num-labels", type=int, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.in_dir.iterdir():
        if item.name in ("model.safetensors", "config.json"):
            continue
        dest = args.out_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        else:
            shutil.copytree(item, dest, dirs_exist_ok=True)

    cfg = json.loads((args.in_dir / "config.json").read_text())
    old_n = cfg.get("num_labels", len(cfg.get("id2label", {})))
    new_n = args.new_num_labels
    cfg["num_labels"] = new_n
    cfg["id2label"] = {str(i): MODEL1_ID_TO_CLASS[i] for i in range(new_n)}
    cfg["label2id"] = {v: k for k, v in cfg["id2label"].items()}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    sd = load_file(str(args.in_dir / "model.safetensors"))
    edited = {}
    touched = []
    torch.manual_seed(0)
    for k, v in sd.items():
        is_class_head = (
            ("class_embed" in k and ("weight" in k or "bias" in k))
            or ("enc_score_head" in k and ("weight" in k or "bias" in k))
        )
        is_denoising = k == "model.denoising_class_embed.weight"
        if is_class_head and v.shape[0] == old_n:
            is_bias = "bias" in k
            edited[k] = _resize(v, new_n, is_bias=is_bias,
                                 feature_dim_axis=None if is_bias else 1)
            touched.append((k, tuple(v.shape), tuple(edited[k].shape)))
        elif is_denoising and v.shape[0] == old_n + 1:
            # denoising head has shape (C+1, D): C classes + 1 no-object.
            class_rows = v[:old_n]
            no_obj_row = v[old_n : old_n + 1]
            resized = _resize(class_rows, new_n, is_bias=False, feature_dim_axis=1)
            edited[k] = torch.cat([resized, no_obj_row], dim=0).clone()
            touched.append((k, tuple(v.shape), tuple(edited[k].shape)))
        else:
            edited[k] = v

    save_file(edited, str(args.out_dir / "model.safetensors"))

    print(f"num_labels {old_n} → {new_n}")
    for k, old_shape, new_shape in touched:
        print(f"  {k}: {old_shape} → {new_shape}")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
