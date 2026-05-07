"""Sample pages across all source books, render to PNG, and (optionally) run
Model 1 inference to produce a CVAT-ready COCO annotation JSON.

Use:
  # Step 1 — with no model yet, just pick pages and render images:
  uv run python training/prelabel_batch.py \\
      --n-pages 50 --out data/prelabel_batches/batch_001

  # Step 2 — when a checkpoint exists, add pre-labels:
  uv run python training/prelabel_batch.py \\
      --n-pages 50 --out data/prelabel_batches/batch_001 \\
      --checkpoint training/checkpoints/pretrain_v1/best --force

The output directory structure is CVAT-importable:

  batch_001/
    images/           <book>_<page>.png     # copied into the batch dir
    annotations.json  COCO 1.0, 8 categories, predictions as `annotations`
    manifest.json     {book, source_pdf, original_page_1indexed} per image

In CVAT, create a new task, upload `images/*.png`, then Actions → Upload
annotations → COCO 1.0, selecting `annotations.json`. CVAT will render the
predicted bboxes as editable boxes for the annotator to review and correct.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import fitz
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oldi.config import CONFIG, book_alias  # noqa: E402
from oldi.stages.s01_ingest import _deskew_in_place, _detect_source_dpi, _render_page  # noqa: E402
from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS  # noqa: E402

# Directory within sheet_pdf/ to skip — contains text essays, not scores.
_SKIP_SUBDIRS = {"related"}

# Fraction of the front/back matter to skip when sampling. Table of
# contents and indexes rarely contain music.
_PAGE_MARGIN_FRACTION = 0.08


def _music_pdfs(sheet_dir: Path) -> list[Path]:
    return sorted(
        p for p in sheet_dir.iterdir()
        if p.suffix.lower() == ".pdf" and p.parent.name not in _SKIP_SUBDIRS
    )


def _stratified_sample(
    pdfs: list[Path],
    n_pages: int,
    *,
    min_per_book: int = 2,
    rng: random.Random,
) -> list[tuple[Path, int]]:
    """Return [(pdf_path, page_1indexed), ...] sampled across books.

    Strategy:
      - Exclude the first/last `_PAGE_MARGIN_FRACTION` of each book.
      - Guarantee at least `min_per_book` pages per book that has enough
        pages.
      - Distribute the remaining budget proportionally to book page count.
      - Pages are drawn without replacement per book.
    """
    # Determine candidate page ranges per book.
    candidates: dict[Path, list[int]] = {}
    for pdf in pdfs:
        with fitz.open(pdf) as doc:
            n = doc.page_count
        lo = max(1, int(n * _PAGE_MARGIN_FRACTION))
        hi = max(lo + 1, n - int(n * _PAGE_MARGIN_FRACTION))
        candidates[pdf] = list(range(lo + 1, hi + 1))  # 1-indexed

    # Min per book
    out: list[tuple[Path, int]] = []
    for pdf, pages in candidates.items():
        k = min(min_per_book, len(pages))
        picked = rng.sample(pages, k)
        for p in picked:
            out.append((pdf, p))
            pages.remove(p)

    # Remaining budget, weighted by remaining-candidate count.
    remaining = n_pages - len(out)
    if remaining > 0:
        weights = [len(candidates[pdf]) for pdf in pdfs]
        total = sum(weights)
        if total == 0:
            return out
        alloc = [int(round(remaining * w / total)) for w in weights]
        # Round-trip correction: sum may be off by ±1 due to rounding.
        while sum(alloc) > remaining:
            alloc[alloc.index(max(alloc))] -= 1
        while sum(alloc) < remaining:
            alloc[alloc.index(min(alloc))] += 1

        for pdf, k in zip(pdfs, alloc):
            k = min(k, len(candidates[pdf]))
            if k <= 0:
                continue
            picked = rng.sample(candidates[pdf], k)
            for p in picked:
                out.append((pdf, p))

    rng.shuffle(out)
    return out[:n_pages]


def _render_sample(
    pdf: Path, page_1indexed: int, *, force: bool
) -> Path:
    """Ensure the page PNG exists at data/01_pages/<book>/page_NNNN.png.

    Reuses the deskew from s01_ingest. Returns the PNG path.
    """
    book = book_alias(pdf)
    out_png = CONFIG.data_dir / "01_pages" / book / f"page_{page_1indexed:04d}.png"
    if out_png.exists() and not force:
        return out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf) as doc:
        src_dpi = _detect_source_dpi(doc, page_1indexed - 1)
        dpi = min(CONFIG.max_render_dpi, max(CONFIG.min_render_dpi, src_dpi))
        _render_page(doc, page_1indexed - 1, dpi, out_png)
    _deskew_in_place(out_png)
    return out_png


def _copy_to_batch(
    source_png: Path, pdf: Path, page_1indexed: int, batch_dir: Path
) -> tuple[str, int, int]:
    """Copy a page PNG into the batch dir with a collision-free filename.

    Returns the flat filename (used in COCO), plus width and height.
    """
    book = book_alias(pdf)
    flat = f"{book}__p{page_1indexed:04d}.png"
    dest = batch_dir / "images" / flat
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source_png, dest)
    with Image.open(dest) as im:
        w, h = im.size
    return flat, w, h


# ─────────────────────────── COCO writer ───────────────────────────


def _empty_coco() -> dict:
    return {
        "info": {"description": "OldI Model 1 pre-label batch"},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "page_layout"}
            for i, name in enumerate(MODEL1_CLASSES)
        ],
    }


def _run_inference(
    images_root: Path, coco: dict, checkpoint: Path, *, threshold: float
) -> None:
    """Populate coco['annotations'] with model predictions."""
    import torch
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(
        str(checkpoint),
        id2label=MODEL1_ID_TO_CLASS,
        label2id={v: k for k, v in MODEL1_ID_TO_CLASS.items()},
    )
    processor = AutoImageProcessor.from_pretrained(str(checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    next_id = 1
    for img_meta in coco["images"]:
        pil = Image.open(images_root / img_meta["file_name"]).convert("RGB")
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold,
        )[0]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            coco["annotations"].append({
                "id": next_id,
                "image_id": img_meta["id"],
                "category_id": int(label),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": float((x1 - x0) * (y1 - y0)),
                "iscrowd": 0,
                "score": float(score),  # CVAT shows this as a confidence attribute
                "attributes": {"source": "model"},
            })
            next_id += 1


# ─────────────────────────── Main ───────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet-dir", type=Path, default=REPO / "sheet_pdf")
    ap.add_argument("--out", type=Path, required=True,
                    help="Batch output directory (will be created).")
    ap.add_argument("--n-pages", type=int, default=50)
    ap.add_argument("--min-per-book", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Trained Model 1 checkpoint. If omitted, the output "
                         "COCO has zero annotations (images only).")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Confidence threshold for pre-labels.")
    ap.add_argument("--force", action="store_true",
                    help="Re-render pages even if data/01_pages/ PNG exists.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pdfs = _music_pdfs(args.sheet_dir)
    print(f"music PDFs: {len(pdfs)}")
    for p in pdfs:
        print(f"  {p.name}")

    samples = _stratified_sample(pdfs, args.n_pages,
                                  min_per_book=args.min_per_book, rng=rng)
    print(f"\nsampled {len(samples)} pages")

    # Render + copy + COCO image entries
    coco = _empty_coco()
    manifest = []
    print("\nrendering / copying:")
    for idx, (pdf, page_1indexed) in enumerate(samples, start=1):
        png = _render_sample(pdf, page_1indexed, force=args.force)
        flat, w, h = _copy_to_batch(png, pdf, page_1indexed, args.out)
        coco["images"].append({
            "id": idx,
            "file_name": flat,
            "width": w,
            "height": h,
        })
        manifest.append({
            "image_id": idx,
            "flat_file_name": flat,
            "book": book_alias(pdf),
            "source_pdf": str(pdf.relative_to(REPO)),
            "original_page_1indexed": page_1indexed,
        })
        print(f"  [{idx:3d}/{len(samples)}] {flat} ({w}x{h})")

    if args.checkpoint is not None:
        print(f"\nrunning inference with {args.checkpoint} (thresh={args.threshold})...")
        _run_inference(args.out / "images", coco, args.checkpoint,
                       threshold=args.threshold)
        n_by_cls: dict[str, int] = {}
        for a in coco["annotations"]:
            cls = MODEL1_CLASSES[a["category_id"]]
            n_by_cls[cls] = n_by_cls.get(cls, 0) + 1
        print(f"predictions: {len(coco['annotations'])} total")
        for cls in MODEL1_CLASSES:
            print(f"  {cls}: {n_by_cls.get(cls, 0)}")
    else:
        print("\n(no checkpoint provided — writing empty-annotation COCO)")

    # Write outputs
    coco_path = args.out / "annotations.json"
    with coco_path.open("w") as f:
        json.dump(coco, f, indent=2)
    manifest_path = args.out / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump({
            "n_pages": len(samples),
            "seed": args.seed,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "threshold": args.threshold if args.checkpoint else None,
            "entries": manifest,
        }, f, indent=2)

    print(f"\nwrote {coco_path}")
    print(f"wrote {manifest_path}")
    print(f"images in {args.out / 'images'}")
    print("\nIn CVAT: create task → upload images/*.png → Actions → "
          "Upload annotations → COCO 1.0 → annotations.json")


if __name__ == "__main__":
    main()
