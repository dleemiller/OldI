"""Render prelabel batch with predicted bboxes overlaid.

Use:
  uv run python training/preview_batch.py \\
      --batch data/prelabel_batches/batch_001 \\
      --out data/prelabel_batches/batch_001/preview

Writes one PNG per image with predicted bboxes drawn on top, grouped by
book so it's easy to scroll and assess per-book quality.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-edge", type=int, default=1600,
                    help="Downscale preview PNGs to this longest edge.")
    args = ap.parse_args()

    out_dir = args.out or (args.batch / "preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads((args.batch / "annotations.json").read_text())
    manifest = json.loads((args.batch / "manifest.json").read_text())
    entries_by_imgid = {e["image_id"]: e for e in manifest["entries"]}
    category_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image id.
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for img_meta in coco["images"]:
        entry = entries_by_imgid[img_meta["id"]]
        book = entry["book"]
        src_png = args.batch / "images" / img_meta["file_name"]
        pil = Image.open(src_png).convert("RGB")
        draw = ImageDraw.Draw(pil)
        anns = sorted(anns_by_img.get(img_meta["id"], []),
                      key=lambda a: a.get("score", 0), reverse=True)
        for a in anns:
            x, y, w, h = a["bbox"]
            cls = category_names[a["category_id"]]
            color = CLASS_COLORS.get(cls, (0, 0, 0))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
            score = a.get("score", 0.0)
            tag = f"{cls} {score:.2f}"
            tag_h = 20
            y_tag_top = max(0, y - tag_h)
            y_tag_bot = y_tag_top + tag_h
            draw.rectangle([x, y_tag_top, x + 8 * len(tag), y_tag_bot],
                           fill=color)
            draw.text((x + 2, y_tag_top + 2), tag,
                      fill=(255, 255, 255), font=font)
        # Downscale
        if max(pil.size) > args.max_edge:
            scale = args.max_edge / max(pil.size)
            new_size = (int(pil.size[0] * scale), int(pil.size[1] * scale))
            pil = pil.resize(new_size, Image.LANCZOS)
        # Save under book-prefixed filename to sort naturally
        out_name = f"{book}__p{entry['original_page_1indexed']:04d}.png"
        pil.save(out_dir / out_name)

    # Summary: predictions per book, per class
    per_book: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_book_nimg: dict[str, int] = defaultdict(int)
    for img_meta in coco["images"]:
        entry = entries_by_imgid[img_meta["id"]]
        per_book_nimg[entry["book"]] += 1
    for a in coco["annotations"]:
        entry = entries_by_imgid[a["image_id"]]
        per_book[entry["book"]][category_names[a["category_id"]]] += 1

    print(f"\npreview images: {len(coco['images'])} → {out_dir}")
    print(f"\n{'book':<45s} {'pages':>6s} {'staff':>8s} {'misc':>6s}")
    for book in sorted(per_book_nimg):
        staff = per_book[book].get("staff", 0)
        misc = sum(v for k, v in per_book[book].items() if k != "staff")
        print(f"{book:<45s} {per_book_nimg[book]:>6d} {staff:>8d} {misc:>6d}")


if __name__ == "__main__":
    main()
