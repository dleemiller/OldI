"""End-to-end preprocessing orchestrator.

Expects a page PNG that's already been produced by `stages/s01_ingest.py`
(PDF → grayscale PNG, with the existing projection-profile deskew applied
in-place). This stage adds:

  * UVDoc dewarp (optional; skippable per-page via config)
  * A second deskew pass, since dewarp can reintroduce small tilt
  * Sauvola binarisation written alongside as `page_NNNN.bin.png`

The grayscale PNG is rewritten in place if dewarp changed it; the binary
image is a new file. Both live in `data/01_pages/<book>/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..stages.s01_ingest import _deskew_in_place
from ..util.logging import get_logger
from .binarize import binarize_gray
from .dewarp import dewarp_gray

log = get_logger()


def preprocess_page(
    page_png: Path,
    *,
    dewarp: bool = False,
    binary_suffix: str = ".bin.png",
    force: bool = False,
) -> tuple[Path, Path]:
    """Apply [dewarp] + deskew + binarise to an ingested page PNG.

    Parameters
    ----------
    page_png : Path
        Path to the grayscale page PNG written by `s01_ingest`.
    dewarp : bool
        If True, run UVDoc dewarping before deskew. Defaults False: UVDoc is
        trained on camera-captured photos with real page curl, and empirically
        over-corrects our flat scanner output — it compresses axis-aligned
        staff spacing when presented with a page that has no warp to remove.
        Enable only when ingesting camera-captured sources.
    binary_suffix : str
        Filename suffix for the binary output, e.g. `page_0042.bin.png`.
    force : bool
        If False and the binary output already exists, skip.

    Returns
    -------
    (grayscale_path, binary_path)
    """
    if not page_png.exists():
        raise FileNotFoundError(page_png)
    bin_png = page_png.with_suffix(binary_suffix)
    if bin_png.exists() and not force:
        return page_png, bin_png

    gray = np.asarray(Image.open(page_png).convert("L"), dtype=np.uint8)

    if dewarp:
        gray = dewarp_gray(gray)
        Image.fromarray(gray, mode="L").save(page_png)
        # Dewarp can leave a sub-degree tilt; run the cheap classical deskew
        # again to clean it up before downstream stages make any bbox
        # assumptions.
        angle = _deskew_in_place(page_png)
        if abs(angle) > 0.0:
            log.info("preprocess %s: post-dewarp deskew %+.2f°", page_png.name, angle)
            gray = np.asarray(Image.open(page_png).convert("L"), dtype=np.uint8)

    binary = binarize_gray(gray)
    Image.fromarray(binary, mode="L").save(bin_png)
    log.info("preprocess %s: wrote %s", page_png.name, bin_png.name)
    return page_png, bin_png
