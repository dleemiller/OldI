"""Turn a CVAT COCO export into train/val splits for Phase B finetune.

Assumes:
  - Input dir is an export produced by `training/cvat_export.py`, pointing
    at `.../task-N-<slug>/annotations/instances_default.json`.
  - The images referenced by that COCO live under data/01_pages/<book>/
    (created during ingest) — we symlink them into the output images dir
    under the CVAT-flat filename (<book>__p<NNNN>.png).
  - CVAT exports category ids 1-indexed starting at 1. Model 1 uses
    0-indexed. We subtract 1 on the way out.

Typical use:
  uv run python training/prepare_finetune_data.py \\
      --coco data/annotations/exports/before_round_1/task-3-.../annotations/instances_default.json \\
      --first-n 11 \\
      --out data/annotations/round_1

Creates:
  round_1/train_coco.json
  round_1/val_coco.json
  round_1/images/<book>__p<NNNN>.png  (symlinks into data/01_pages/)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_CLASSES  # noqa: E402


def _book_from_filename(name: str) -> str:
    m = re.match(r"(.+?)__p(\d+)\.png", name)
    return m.group(1) if m else "unknown"


def _page_from_filename(name: str) -> int:
    m = re.match(r".+?__p(\d+)\.png", name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", type=Path, required=True,
                    help="CVAT COCO export file (instances_default.json)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (e.g. data/annotations/round_1).")
    ap.add_argument("--first-n", type=int, default=None,
                    help="Keep only the first N images (by id order). "
                         "Use when the user has annotated the first N frames.")
    ap.add_argument("--min-anns", type=int, default=5,
                    help="Drop images with fewer than this many annotations "
                         "(default 5 — skips unfinished frames).")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--pages-root", type=Path, default=REPO / "data/01_pages",
                    help="Where the original ingested PNGs live.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text())
    id2name = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Subset to first-N (if requested) then drop images with few anns.
    sorted_imgs = sorted(coco["images"], key=lambda im: im["id"])
    if args.first_n is not None:
        sorted_imgs = sorted_imgs[: args.first_n]
    kept_imgs = [im for im in sorted_imgs
                 if len(anns_by_img.get(im["id"], [])) >= args.min_anns]
    dropped = [im for im in sorted_imgs
               if len(anns_by_img.get(im["id"], [])) < args.min_anns]
    print(f"taken: {len(kept_imgs)} / {len(sorted_imgs)} images "
          f"(dropped {len(dropped)} with < {args.min_anns} annotations)")
    for im in dropped:
        print(f"  dropped: {im['file_name']} ({len(anns_by_img.get(im['id'], []))} anns)")

    # Book-stratified split — hold one image per book out for val if possible,
    # otherwise fall back to random stratified sample.
    rng = random.Random(args.seed)
    rng.shuffle(kept_imgs)
    by_book: dict[str, list[dict]] = defaultdict(list)
    for im in kept_imgs:
        by_book[_book_from_filename(im["file_name"])].append(im)

    val_target = max(1, int(round(len(kept_imgs) * args.val_fraction)))
    val_imgs: list[dict] = []
    # Prefer val picks from books that have ≥ 2 images so we don't blank a
    # whole book from train. Fall back to random if necessary.
    books_with_plenty = [b for b, ims in by_book.items() if len(ims) >= 2]
    rng.shuffle(books_with_plenty)
    for b in books_with_plenty:
        if len(val_imgs) >= val_target:
            break
        val_imgs.append(by_book[b][-1])  # last one in this book
    while len(val_imgs) < val_target and len(val_imgs) < len(kept_imgs):
        for im in kept_imgs:
            if im not in val_imgs:
                val_imgs.append(im)
                break
    train_imgs = [im for im in kept_imgs if im not in val_imgs]
    print(f"\ntrain: {len(train_imgs)}, val: {len(val_imgs)}")
    print("val images:")
    for im in val_imgs:
        print(f"  {im['file_name']}")

    # Build output COCO — category ids 0-indexed (CVAT→Model 1 shift by -1)
    def _build(imgs: list[dict]) -> dict:
        img_ids = {im["id"] for im in imgs}
        out_anns = []
        next_id = 1
        for a in coco["annotations"]:
            if a["image_id"] not in img_ids:
                continue
            out_anns.append({
                "id": next_id,
                "image_id": a["image_id"],
                "category_id": a["category_id"] - 1,  # CVAT 1-idx → Model 1 0-idx
                "bbox": a["bbox"],
                "area": a["area"],
                "iscrowd": a.get("iscrowd", 0),
                "segmentation": a.get("segmentation", []),
            })
            next_id += 1
        cats = [{"id": i, "name": name, "supercategory": "page_layout"}
                for i, name in enumerate(MODEL1_CLASSES)]
        return {
            "info": {"description": "OldI Model 1 Phase B"},
            "images": imgs,
            "annotations": out_anns,
            "categories": cats,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir(exist_ok=True)

    # Symlink source images into round_1/images/ under the CVAT flat name.
    for im in kept_imgs:
        fname = im["file_name"]
        dst = images_dir / fname
        if dst.exists():
            continue
        book = _book_from_filename(fname)
        page = _page_from_filename(fname)
        src = args.pages_root / book / f"page_{page:04d}.png"
        if not src.exists():
            print(f"  WARN: source image missing: {src}", file=sys.stderr)
            continue
        dst.symlink_to(src.resolve())

    (args.out / "train_coco.json").write_text(json.dumps(_build(train_imgs), indent=2))
    (args.out / "val_coco.json").write_text(json.dumps(_build(val_imgs), indent=2))
    print(f"\nwrote {args.out / 'train_coco.json'}")
    print(f"wrote {args.out / 'val_coco.json'}")
    print(f"symlinked {len(kept_imgs)} images into {images_dir}")


if __name__ == "__main__":
    main()
