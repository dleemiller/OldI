"""COCO-format object-detection dataset for RT-DETRv2 training.

Yields samples in the format the Hugging Face RTDetrV2 model expects from
the processor: raw PIL image + list of annotations. The collator (below)
runs the image processor, which handles resize + normalise and converts
bboxes to the normalised (cx, cy, w, h) format the model takes.

Augmentation is applied to the raw PIL image before the processor runs, so
bbox coordinates remain in absolute pixel space throughout.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# A handful of DSv2 PNGs have corrupted IDAT CRCs / bad adaptive filter
# values. PIL recovers and decodes them cleanly, but libpng writes a
# warning to stderr at the C level on every affected read — polluting the
# training stdout with hundreds of "libpng error: IDAT: CRC error" lines.
# Accept truncated images silently.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@contextlib.contextmanager
def _suppress_c_stderr():
    """Temporarily redirect fd 2 to /dev/null — silences libpng's C warnings."""
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def silence_libpng_worker_init(worker_id: int) -> None:
    """DataLoader `worker_init_fn` — redirects fd 2 to /dev/null permanently
    in each worker so libpng's C-level CRC warnings never reach the main
    process's stderr. Python exceptions are unaffected — they travel via
    the exception mechanism, not stderr writes."""
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


class CocoDetectionDataset(Dataset):
    """Lightweight COCO reader that pairs images with their annotations."""

    def __init__(
        self,
        coco_path: Path,
        images_root: Path,
        *,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        pre_resize_edge: int | None = 640,
    ) -> None:
        """
        Parameters
        ----------
        pre_resize_edge : int | None
            If set, images are resized so their longer edge is this many
            pixels *before* augmentation runs, and bboxes are scaled to match.
            Defaults to 640 (heron's training resolution) to keep augraphy
            cost tractable — without this, each image is a ~2000×3000 canvas
            and augraphy's ink/letterpress ops take ~1 s/image, starving the
            GPU. The HF image processor also resizes to 640 downstream, so
            the pre-resize adds no quality loss; it just moves the resize
            ahead of the slow CPU-bound step.
        """
        raw = json.loads(Path(coco_path).read_text())
        self.images_root = Path(images_root)
        self.images = raw["images"]
        self.categories = raw["categories"]
        self.augment_fn = augment_fn
        self.pre_resize_edge = pre_resize_edge

        # Index annotations by image id for O(1) lookup.
        anns_by_img: dict[int, list[dict]] = {}
        for a in raw["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        self.anns_by_img = anns_by_img

        # Drop images with zero annotations — they contribute no training
        # signal and can destabilise the Hungarian matcher.
        self.images = [img for img in self.images if self.anns_by_img.get(img["id"])]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_meta = self.images[idx]
        img_path = self.images_root / img_meta["file_name"]
        with _suppress_c_stderr():
            pil = Image.open(img_path)
            pil.load()  # force full decode inside the redirect
            pil = pil.convert("RGB")
        W, H = pil.size

        if self.pre_resize_edge is not None:
            scale = self.pre_resize_edge / max(H, W)
            if scale < 1.0:
                new_w = max(1, int(round(W * scale)))
                new_h = max(1, int(round(H * scale)))
                pil = pil.resize((new_w, new_h), Image.BILINEAR)
            else:
                scale = 1.0
        else:
            scale = 1.0

        arr = np.array(pil)

        if self.augment_fn is not None:
            arr = self.augment_fn(arr)

        anns = self.anns_by_img.get(img_meta["id"], [])
        coco_anns = [{
            "id": a["id"],
            "image_id": img_meta["id"],
            "category_id": a["category_id"],
            "bbox": [v * scale for v in a["bbox"]],
            "area": a["area"] * (scale * scale),
            "iscrowd": a.get("iscrowd", 0),
        } for a in anns]

        return {
            "image_id": img_meta["id"],
            "image": Image.fromarray(arr),
            "annotations": coco_anns,
        }
