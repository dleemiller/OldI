"""Augmentation calibration gate.

Blocks Phase A training until augmented DSv2 pages look comparable to their
namesake real tune-book pages. The gate is visual: there's no quantitative
metric that reliably captures "does my augmented synthetic match the real
distribution" — a human eyeball comparison is the ground truth.

Output: a side-by-side PNG grid.
  - One row per augmentation mode (`training.data.augment.ALL_MODES`).
  - Left column: a real crop from the book that motivated the mode.
  - Right columns: 3 augmented DSv2 crops through that mode.

Run:
  uv run python training/calibrate_augment.py \\
      --out training/eval_out/calibration.png

Review the PNG. If the augmented DSv2 crops in any row look clearly more
extreme (or clearly cleaner) than the real crop, tune that mode in
`training/data/augment.py` and re-render. Do not start Phase A until the
grid reads as "synthetic ≈ real" across every row.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.data.augment import ALL_MODES, ModeName, apply_mode  # noqa: E402

# Real reference crops that define each mode. Each entry is
#   (mode, pdf_relative_path, page_1indexed, crop_y_mid, crop_x_mid).
# We render each reference PDF page on the fly and crop a 800×500 strip at
# the given centre so every panel in the grid is the same size.
REAL_REFERENCES: dict[ModeName, tuple[str, int, int, int]] = {
    "oneill_dance_like": ("dance_music_ireland_oneill.pdf", 30, None, None),
    "oneill_music_like": ("music_of_ireland_oneill.pdf", 22, None, None),
    "petrie_like": ("petrie_collection_ancient_music_of_ireland.pdf", 40, None, None),
    "oneill_waifs_like": ("waifs_and_strays_oneill.pdf", 15, None, None),
    "popsel_like": ("popular_selections_from_oneill.pdf", 10, None, None),
    "minstrel_like": ("the_irish_minstrel.pdf", 51, None, None),
    "graves_like": ("irish_song_book_graves.pdf", 51, None, None),
}

PANEL_W, PANEL_H = 800, 500


def _render_real_crop(pdf_path: Path, page_1indexed: int,
                      crop_y: int | None, crop_x: int | None) -> np.ndarray:
    """Return a PANEL_H × PANEL_W grayscale crop from a reference page."""
    import fitz
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_1indexed - 1)
        mat = fitz.Matrix(350 / 72, 350 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    H, W = arr.shape
    cy = crop_y if crop_y is not None else H // 2
    cx = crop_x if crop_x is not None else W // 2
    y0 = max(0, cy - PANEL_H // 2)
    x0 = max(0, cx - PANEL_W // 2)
    y1 = min(H, y0 + PANEL_H)
    x1 = min(W, x0 + PANEL_W)
    crop = arr[y0:y1, x0:x1]
    # Pad if we hit an edge.
    if crop.shape != (PANEL_H, PANEL_W):
        padded = np.full((PANEL_H, PANEL_W), 255, dtype=np.uint8)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded
    return crop


def _random_dsv2_crops(coco_path: Path, images_root: Path,
                        n: int, rng: random.Random) -> list[np.ndarray]:
    """Return n random PANEL_H × PANEL_W crops centred on a staff bbox.

    Seeding each crop on a staff bbox guarantees every panel in the
    calibration grid shows notation rather than blank margin.
    """
    coco = json.loads(coco_path.read_text())
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    image_meta = {img["id"]: img for img in coco["images"]}
    img_ids = [iid for iid in anns_by_img if iid in image_meta]
    rng.shuffle(img_ids)

    crops: list[np.ndarray] = []
    for iid in img_ids:
        if len(crops) >= n:
            break
        img = image_meta[iid]
        pil = Image.open(images_root / img["file_name"]).convert("RGB")
        arr = np.array(pil)
        H, W = arr.shape[:2]
        if H < PANEL_H or W < PANEL_W:
            continue
        # Pick a staff annotation and centre the crop on it.
        ann = rng.choice(anns_by_img[iid])
        x, y, w, h = ann["bbox"]
        cy = int(y + h / 2)
        cx = int(x + w / 2)
        y0 = min(max(0, cy - PANEL_H // 2), H - PANEL_H)
        x0 = min(max(0, cx - PANEL_W // 2), W - PANEL_W)
        crops.append(arr[y0 : y0 + PANEL_H, x0 : x0 + PANEL_W])
    return crops


def _compose_grid(rows: list[tuple[ModeName, np.ndarray, list[np.ndarray]]],
                   out_path: Path) -> None:
    """Write the grid PNG with row labels on the far left."""
    label_w = 220
    n_cols = 1 + max(len(r[2]) for r in rows)  # real + augmented panels
    grid_w = label_w + PANEL_W * n_cols
    grid_h = PANEL_H * len(rows) + 50  # 50 for header

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    # Header
    draw.text((label_w, 15), "real reference",
              fill=(0, 0, 0), font=font)
    draw.text((label_w + PANEL_W + 20, 15), "augmented DSv2 (left = mode-specific)",
              fill=(0, 0, 0), font=font)

    for row_i, (mode, real, augs) in enumerate(rows):
        y = 50 + row_i * PANEL_H
        # Label column
        draw.text((10, y + PANEL_H // 2 - 10), mode, fill=(0, 0, 0), font=font)
        # Real reference
        canvas.paste(Image.fromarray(real, mode="L").convert("RGB"),
                     (label_w, y))
        draw.rectangle([(label_w, y), (label_w + PANEL_W, y + PANEL_H)],
                       outline=(80, 80, 80), width=2)
        # Augmented panels
        for col_i, aug in enumerate(augs):
            x = label_w + PANEL_W * (col_i + 1)
            canvas.paste(Image.fromarray(aug), (x, y))
            draw.rectangle([(x, y), (x + PANEL_W, y + PANEL_H)],
                           outline=(80, 80, 80), width=2)
            draw.text((x + 5, y + 5), f"DSv2·{mode}",
                      fill=(200, 0, 0), font=font_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet-dir", type=Path,
                    default=REPO / "sheet_pdf",
                    help="Directory containing the reference PDFs.")
    ap.add_argument("--coco", type=Path,
                    default=REPO / "data/deepscoresv2/model1_coco_train.json",
                    help="DSv2 COCO (used to pick random augmentation inputs).")
    ap.add_argument("--images-root", type=Path,
                    default=REPO / "data/deepscoresv2/ds2_dense/images",
                    help="DSv2 image directory referenced by the COCO.")
    ap.add_argument("--n-aug-per-mode", type=int, default=3,
                    help="Augmented panels per row.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "training/eval_out/calibration.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Gather DSv2 input crops — enough for the biggest row and shared across
    # modes so the same crops are shown through each mode (better comparison
    # signal than fully independent samples).
    dsv2_crops = _random_dsv2_crops(args.coco, args.images_root,
                                     n=args.n_aug_per_mode, rng=rng)

    rows: list[tuple[ModeName, np.ndarray, list[np.ndarray]]] = []
    for mode in ALL_MODES:
        pdf_rel, page, cy, cx = REAL_REFERENCES[mode]
        real_crop = _render_real_crop(args.sheet_dir / pdf_rel, page, cy, cx)
        augs = [apply_mode(c, mode) for c in dsv2_crops]
        rows.append((mode, real_crop, augs))

    _compose_grid(rows, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
