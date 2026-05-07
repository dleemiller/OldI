"""Image I/O and single-page PDF wrapping helpers."""

from __future__ import annotations

from pathlib import Path

import img2pdf
from PIL import Image


def crop(src_png: Path, bbox: tuple[int, int, int, int], out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as im:
        im.crop(bbox).save(out_png, format="PNG", optimize=True)
    return out_png


def png_to_single_page_pdf(src_png: Path, out_pdf: Path, dpi: int = 400) -> Path:
    """Wrap a single PNG into a one-page PDF for tools that require PDF input (Clarity, Audiveris).

    The PDF page size is set so the embedded image prints at `dpi`. When Clarity
    re-rasterizes with `--pdf-dpi`, matching the two values round-trips the
    exact pixels; mismatches up- or down-sample.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(str(src_png), layout_fun=layout))
    return out_pdf
