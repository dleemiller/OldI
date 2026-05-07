"""DeepScoresV2 loader + remap to the 8-class Model 1 COCO format.

DSv2 ships a custom annotation JSON (OBBAnns schema — see
https://github.com/yvan674/obb_anns). We parse it directly rather than
installing the obb_anns package (it's a SWIG build and we only need a subset
of its functionality).

Schema (per OBBAnns):
  {
    "info": { ... },
    "categories": {
      "1": { "name": "noteheadBlack", "annotation_set": "deepscores", "color": "..." },
      ...
    },
    "annotation_sets": ["deepscores"],
    "images": [
      { "id": "1", "filename": "lg-184241...png", "width": 1960, "height": 2772,
        "ann_ids": ["12345", "12346", ...] },
      ...
    ],
    "annotations": {
      "12345": { "a_bbox": [x0, y0, x1, y1], "o_bbox": [x0..y3],
                 "cat_id": ["17"],  # list of stringified IDs, one per annotation_set
                 "area": 1234.5, "img_id": "1", "comments": "" },
      ...
    }
  }

We only consume `a_bbox` (axis-aligned) and filter to the DSv2 classes that map
to our Model 1 taxonomy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..classes import DSV2_TO_MODEL1, MODEL1_CLASS_TO_ID, MODEL1_CLASSES


@dataclass
class DSv2Dataset:
    """Parsed DeepScoresV2 annotation file."""

    info: dict
    categories: dict[str, dict]  # "1" → {name, annotation_set, color}
    annotation_sets: list[str]
    images: list[dict]  # [{id, filename, width, height, ann_ids}, ...]
    annotations: dict[str, dict]  # "ann_id" → {a_bbox, o_bbox, cat_id, area, img_id, ...}

    @classmethod
    def load(cls, json_path: Path) -> DSv2Dataset:
        with json_path.open() as f:
            raw = json.load(f)
        return cls(
            info=raw.get("info", {}),
            categories=raw["categories"],
            annotation_sets=raw.get("annotation_sets", ["deepscores"]),
            images=raw["images"],
            annotations=raw["annotations"],
        )

    def category_id_by_name(self, name: str) -> str | None:
        """Return the DSv2 category ID (stringified int) for a class name, or None."""
        for cid, cat in self.categories.items():
            if cat["name"] == name:
                return cid
        return None

    def annotation_set_index(self, annotation_set: str = "deepscores") -> int:
        """DSv2 stores cat_id as a list — one entry per annotation set. We need
        to know which index corresponds to the 'deepscores' set."""
        return self.annotation_sets.index(annotation_set)


def remap_to_coco(
    dsv2: DSv2Dataset,
    *,
    images_root: Path,
    out_path: Path,
    annotation_set: str = "deepscores",
) -> dict:
    """Write a COCO-format JSON with only Model 1 classes.

    Parameters
    ----------
    dsv2 : DSv2Dataset
        Parsed DSv2 annotations.
    images_root : Path
        Directory containing the PNG files referenced by `images[*].filename`.
        Included in the COCO JSON as `file_name` (relative path from wherever
        the training script resolves image paths).
    out_path : Path
        Where to write the COCO JSON.
    annotation_set : str
        Which DSv2 annotation set's cat_id to read. Default "deepscores".

    Returns
    -------
    dict
        Summary: {num_images, num_annotations, per_class_counts}.
    """
    set_idx = dsv2.annotation_set_index(annotation_set)

    # Build DSv2-id → Model 1 class-name map (only for classes we care about).
    dsv2_to_model1_id: dict[str, int] = {}
    for dsv2_name, model1_name in DSV2_TO_MODEL1.items():
        dsv2_id = dsv2.category_id_by_name(dsv2_name)
        if dsv2_id is None:
            continue
        dsv2_to_model1_id[dsv2_id] = MODEL1_CLASS_TO_ID[model1_name]

    if not dsv2_to_model1_id:
        raise ValueError(
            "No DSv2 categories matched Model 1 taxonomy — check DSV2_TO_MODEL1"
        )

    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    per_class: dict[str, int] = dict.fromkeys(MODEL1_CLASSES, 0)
    next_ann_id = 1

    for img in dsv2.images:
        rel_path = img["filename"]
        coco_img = {
            "id": int(img["id"]),
            "file_name": rel_path,
            "width": img["width"],
            "height": img["height"],
        }
        coco_images.append(coco_img)
        for ann_id in img["ann_ids"]:
            ann = dsv2.annotations[ann_id]
            cat_list = ann["cat_id"]
            # cat_id may be ["17"] or a list indexed by annotation set.
            dsv2_cid = cat_list[set_idx] if len(cat_list) > set_idx else cat_list[0]
            if dsv2_cid not in dsv2_to_model1_id:
                continue
            x0, y0, x1, y1 = ann["a_bbox"]
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            if w == 0 or h == 0:
                continue
            model1_id = dsv2_to_model1_id[dsv2_cid]
            coco_anns.append({
                "id": next_ann_id,
                "image_id": int(img["id"]),
                "category_id": model1_id,
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            })
            per_class[MODEL1_CLASSES[model1_id]] += 1
            next_ann_id += 1

    coco_categories = [
        {"id": i, "name": name, "supercategory": "page_layout"}
        for i, name in enumerate(MODEL1_CLASSES)
    ]

    coco = {
        "info": {
            "description": "OldI Model 1 (page layout) — from DeepScoresV2",
            "source_annotation_set": annotation_set,
            "images_root": str(images_root),
        },
        "images": coco_images,
        "annotations": coco_anns,
        "categories": coco_categories,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(coco, f)

    return {
        "num_images": len(coco_images),
        "num_annotations": len(coco_anns),
        "per_class_counts": per_class,
        "out_path": str(out_path),
    }
