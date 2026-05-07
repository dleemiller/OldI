"""Stage 01 — PDF to per-page grayscale PNG.

Detects the embedded image DPI per page (for the common case of a scanned book
with one big JPEG/JBIG2 per page) and renders at max(source_dpi, min_render_dpi),
capped at max_render_dpi. Grayscale to keep file sizes reasonable.

When the scan is tilted (common on these 19th-century sources) we deskew the
page to horizontal. Staff-line detection, crop bounds, and both OMR engines all
assume axis-aligned music, so running a deskew pass here fixes many downstream
misreads — the effort is trivial compared to re-tuning each downstream stage
to be rotation-tolerant.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf
import numpy as np
from PIL import Image

from ..config import CONFIG, Config, book_alias, stage_dir
from ..errors import StageResult
from ..util.logging import get_logger

log = get_logger()


def _detect_skew_angle(gray: np.ndarray, max_angle: float = 4.0, step: float = 0.2) -> float:
    """Return the rotation (in degrees) that best flattens horizontal lines.

    Works by rotating a downsampled binarized copy through a small range of
    angles and picking the one that maximises the variance of the per-row
    sum of dark pixels. On a scanned page of music, horizontal staff lines
    concentrate most of the ink along a few rows; that concentration peaks
    when the page is rotationally aligned with the pixel grid.
    """
    from scipy import ndimage  # lazy — only loaded if we ingest

    # Downsample: 600-pixel-wide working copy is more than enough to resolve
    # tilt to ±0.1°, and keeps the 400-step search fast.
    h, w = gray.shape
    scale = 600.0 / max(w, 1)
    if scale < 1.0:
        small_h = int(h * scale)
        small_w = int(w * scale)
        small = np.asarray(
            Image.fromarray(gray).resize((small_w, small_h), Image.BILINEAR),
            dtype=np.uint8,
        )
    else:
        small = gray

    # Binarise around mid-grey: dark ink → 1, white page → 0. Saves the
    # rotate step from interpolating grey values and gives a sharper peak.
    binary = (small < 128).astype(np.float32)

    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        rotated = ndimage.rotate(binary, angle, reshape=False, order=1, cval=0.0)
        row_sums = rotated.sum(axis=1)
        score = row_sums.var()
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def _deskew_in_place(png_path: Path, threshold_deg: float = 0.3) -> float:
    """Detect tilt on the page PNG and rewrite it rotated if above threshold.

    Returns the applied angle (0.0 if within threshold — no rewrite).
    """
    with Image.open(png_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    angle = _detect_skew_angle(gray)
    if abs(angle) < threshold_deg:
        return 0.0
    from scipy import ndimage
    rotated = ndimage.rotate(gray, angle, reshape=False, order=3, cval=255)
    Image.fromarray(rotated.astype(np.uint8), mode="L").save(png_path)
    return angle


def _detect_source_dpi(doc: fitz.Document, page_idx: int) -> int:
    """Estimate source DPI from the largest embedded image on the page.

    Falls back to 300 when there is no embedded image (rare for scanned sources).
    """
    page = doc.load_page(page_idx)
    page_w_in = page.rect.width / 72.0
    page_h_in = page.rect.height / 72.0
    best = 0
    for img in page.get_images(full=True):
        w, h = img[2], img[3]
        if page_w_in > 0 and page_h_in > 0:
            dpi_w = w / page_w_in
            dpi_h = h / page_h_in
            best = max(best, int(min(dpi_w, dpi_h)))
    return best or 300


def _render_page(doc: fitz.Document, page_idx: int, dpi: int, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    page = doc.load_page(page_idx)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
    pix.save(out_png)


def run_book(pdf: Path, *, cfg: Config = CONFIG, force: bool = False) -> StageResult:
    """Render every page of a PDF to data/01_pages/<book>/page_NNNN.png."""
    book = book_alias(pdf)
    out_dir = stage_dir(1, "pages", book)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[int] = []
    with fitz.open(pdf) as doc:
        for i in range(doc.page_count):
            out_png = out_dir / f"page_{i + 1:04d}.png"
            if out_png.exists() and not force:
                rendered.append(i + 1)
                continue
            src_dpi = _detect_source_dpi(doc, i)
            dpi = min(cfg.max_render_dpi, max(cfg.min_render_dpi, src_dpi))
            _render_page(doc, i, dpi, out_png)
            applied_angle = _deskew_in_place(out_png)
            rendered.append(i + 1)
            if applied_angle:
                log.info("ingest %s page %d: %d dpi, deskew %+.1f° → %s",
                         book, i + 1, dpi, applied_angle, out_png.name)
            else:
                log.info("ingest %s page %d: %d dpi → %s",
                         book, i + 1, dpi, out_png.name)

    return StageResult(ok=True, outputs=[out_dir], meta={"pages": rendered, "book": book})


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    """Single-page entry point. Assumes the PDF has already been rendered by run_book.

    Used only when the pipeline skips stage 01 for some pages but not others.
    """
    out_png = stage_dir(1, "pages", book) / f"page_{page:04d}.png"
    if not out_png.exists():
        raise FileNotFoundError(f"Expected ingested page image: {out_png}. Run `oldi pipeline` on the source PDF.")
    return StageResult(ok=True, outputs=[out_png])
