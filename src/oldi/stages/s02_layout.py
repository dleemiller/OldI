"""Stage 02 — page layout primitives: text regions (PaddleOCR) + staff systems (YOLO).

We don't need PP-StructureV3's layout analysis — it treats sheet music as one
opaque `image` block. We just need per-text-line bboxes + OCR'd text, which
the lightweight PaddleOCR (text det + rec only) gives us ~5× faster by
skipping orientation, unwarping, and block layout models.
"""

from __future__ import annotations

import json
import os

from PIL import Image

from ..config import CONFIG, Config, stage_dir
from ..errors import StageError, StageResult
from ..util.logging import get_logger
from ..util.staff import detect_staff_systems

log = get_logger()

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise StageError("paddleocr not available", stage="s02_layout", book="?", page=0) from exc
        _OCR = PaddleOCR(
            device="gpu",
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Server detection finds smaller/fainter titles that mobile misses;
            # server recognition is more robust on Irish Gaelic fraktur-ish fonts.
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
        )
    return _OCR


def _extract_text_regions(payload: dict) -> list[dict]:
    """Handle both PaddleOCR v3 (`res`-wrapped) and v4+ (flat) output shapes."""
    inner = payload.get("res", payload) if isinstance(payload, dict) else {}
    rec_boxes = inner.get("rec_boxes") or []
    rec_texts = inner.get("rec_texts") or []
    rec_scores = inner.get("rec_scores") or []
    regions: list[dict] = []
    for i, box in enumerate(rec_boxes):
        try:
            coords = list(box)
        except TypeError:
            continue
        if len(coords) < 4:
            continue
        x0, y0, x1, y1 = (int(v) for v in coords[:4])
        regions.append({
            "bbox": [x0, y0, x1, y1],
            "text": str(rec_texts[i]) if i < len(rec_texts) else "",
            "score": float(rec_scores[i]) if i < len(rec_scores) else 0.0,
        })
    return regions


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    page_png = stage_dir(1, "pages", book) / f"page_{page:04d}.png"
    out_json = stage_dir(2, "layout", book) / f"page_{page:04d}.json"
    if out_json.exists() and not force:
        return StageResult(ok=True, outputs=[out_json])
    if not page_png.exists():
        raise StageError(f"Page PNG missing: {page_png}", stage="s02_layout", book=book, page=page)

    with Image.open(page_png) as im:
        w, h = im.size

    staves = detect_staff_systems(page_png)

    ocr = _get_ocr()
    text_regions: list[dict] = []
    for r in ocr.predict(str(page_png)):
        text_regions = _extract_text_regions(r.json)
        break

    payload = {
        "page": page,
        "width": w,
        "height": h,
        "text_regions": text_regions,
        "staff_systems": [{"bbox": [s.x0, s.y0, s.x1, s.y1]} for s in staves],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    log.info(
        "layout %s page %d: %d text regions, %d staff systems",
        book, page, len(text_regions), len(staves),
    )
    return StageResult(
        ok=True,
        outputs=[out_json],
        meta={"n_text": len(text_regions), "n_staves": len(staves)},
    )
