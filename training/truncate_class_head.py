"""Truncate a trained RT-DETRv2 class head from N classes to M < N.

Surgical fix for when the taxonomy is shrunk after training. Reads the
`model.safetensors` at the given checkpoint, drops rows [M:N] of every
per-layer class-head weight/bias, rewrites `config.json` with the new
`num_labels` / `id2label` / `label2id`, and saves the result to a new dir.

Used once to go from 8-class pretrain (with the now-deleted
`ornamental-element` at slot 7) down to 7-class, preserving the trained
`staff` logits so Phase B finetune doesn't start from random weights for
the one class DSv2 actually supervised.

Use:
  uv run python training/truncate_class_head.py \\
      --in training/checkpoints/pretrain_v1/best_by_map \\
      --out training/checkpoints/pretrain_v1_7cls/best_by_map \\
      --new-num-labels 7
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

from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--new-num-labels", type=int, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Copy everything except model weights + config (we'll rewrite those).
    for item in args.in_dir.iterdir():
        if item.name in ("model.safetensors", "config.json"):
            continue
        dest = args.out_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        else:
            shutil.copytree(item, dest, dirs_exist_ok=True)

    # Load + edit config.json
    cfg = json.loads((args.in_dir / "config.json").read_text())
    old_n = cfg.get("num_labels", len(cfg.get("id2label", {})))
    new_n = args.new_num_labels
    assert new_n < old_n, f"new_num_labels ({new_n}) must be < old ({old_n})"

    cfg["num_labels"] = new_n
    cfg["id2label"] = {str(i): MODEL1_ID_TO_CLASS[i] for i in range(new_n)}
    cfg["label2id"] = {v: k for k, v in cfg["id2label"].items()}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Load + truncate weights
    sd = load_file(str(args.in_dir / "model.safetensors"))
    edited = {}
    dropped = []
    for k, v in sd.items():
        # These tensors have shape [num_labels, ...] or [num_labels]; truncate.
        # `class_embed.N.weight`  shape (C, 256)
        # `class_embed.N.bias`    shape (C,)
        # `enc_score_head.weight` shape (C, 256)
        # `enc_score_head.bias`   shape (C,)
        # `denoising_class_embed.weight`  shape (C+1, 256)  (one extra for no-object)
        is_class_head = (
            ("class_embed" in k and ("weight" in k or "bias" in k))
            or ("enc_score_head" in k and ("weight" in k or "bias" in k))
        )
        is_denoising = k == "model.denoising_class_embed.weight"
        if is_class_head and v.shape[0] == old_n:
            edited[k] = v[:new_n].clone()
            dropped.append((k, tuple(v.shape), tuple(edited[k].shape)))
        elif is_denoising and v.shape[0] == old_n + 1:
            # shape [C+1, D] — keep rows 0..new_n plus the last (no-object).
            edited[k] = torch.cat([v[:new_n], v[old_n : old_n + 1]], dim=0).clone()
            dropped.append((k, tuple(v.shape), tuple(edited[k].shape)))
        else:
            edited[k] = v

    save_file(edited, str(args.out_dir / "model.safetensors"))

    print(f"num_labels {old_n} → {new_n}")
    print(f"truncated tensors:")
    for k, old_shape, new_shape in dropped:
        print(f"  {k}: {old_shape} → {new_shape}")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
