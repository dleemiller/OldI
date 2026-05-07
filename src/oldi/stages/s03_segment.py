"""Stage 03 — segment a page into per-tune crops.

The reliable signal on these engraved scores is: the small block of text
sitting immediately above a staff system IS the tune title. So we do not
cluster staves by gap size — we anchor on titles.

Algorithm:
    1. Filter text regions to plausible-title candidates (margin / digit /
       low-confidence / oversized filters).
    2. For each candidate, check whether there is a staff system starting
       within a short vertical distance below it, horizontally overlapping.
       If yes, mark as a tune title.
    3. Walk page top-to-bottom. Each title starts a new tune; all staves
       between this title and the next title belong to it. Staves above the
       first title get attached to the first tune (likely a continuation from
       the previous page, flagged in meta.json).
    4. Crop + wrap each tune to a single-page PDF for Clarity-OMR.
"""

from __future__ import annotations

import json
import re

from ..config import CONFIG, Config, stage_dir
from ..errors import StageError, StageResult
from ..util.bbox import pad, union
from ..util.image import crop, png_to_single_page_pdf
from ..util.logging import get_logger
from ..util.staff import StaffSystem

log = get_logger()

_DIGITS_ONLY = re.compile(r"^[\s\-]*\d+[\s\-.]*$")


def _staves_from_json(staff_dicts: list[dict]) -> list[StaffSystem]:
    return [
        StaffSystem(x0=s["bbox"][0], y0=s["bbox"][1], x1=s["bbox"][2], y1=s["bbox"][3])
        for s in staff_dicts
    ]


def _is_plausible_title(tr: dict, page_w: int, page_h: int) -> bool:
    """A title is a sizeable text block with a real word (or Irish Gaelic name).

    Small text above a staff is usually an ornament ("tr", "A", "B"), dynamic
    or tempo mark ("Andante"), or attribution ("F.O'Neill.") — none of those
    should be picked as a tune title.
    """
    x0, y0, x1, y1 = tr["bbox"]
    text = (tr.get("text") or "").strip()
    score = float(tr.get("score", 0.0))
    th = y1 - y0
    tw = x1 - x0

    if _DIGITS_ONLY.match(text):
        return False
    if y1 < page_h * 0.03 or y0 > page_h * 0.97:
        return False
    if score > 0 and score < 0.5:
        return False
    # Real tune titles are typeset large: at 400 dpi, ~60-120 px tall and
    # typically 300+ px wide. At 300 dpi, scale everything down by 0.75.
    min_h = int(page_h * 0.012)   # ≈58 px on a 4800-tall page
    min_w = int(page_w * 0.05)    # ≈175 px on a 3500-wide page
    if th < min_h or tw < min_w:
        return False
    # Reject common non-title directions that still pass the size filters.
    # Case-insensitive, surrounding punctuation stripped, OCR-misread variants
    # (l↔I↔1, 0↔O) normalized.
    norm = text.lower().strip(" .,:;!?")
    norm_ocr = norm.translate(str.maketrans({"i": "l", "1": "l", "0": "o"}))
    if norm in _NOT_TITLES or norm_ocr in _NOT_TITLES:
        return False
    if any(p.match(norm) for p in _NOT_TITLE_PATTERNS):
        return False
    if _looks_like_attribution(text):
        return False
    return True


# Tempo / expression / section marks that PaddleOCR may pick up as large text
# but are not tune titles.
_NOT_TITLES: frozenset[str] = frozenset({
    "andante", "andante con moto", "allegro", "allegretto", "moderato",
    "adagio", "largo", "presto", "vivace", "con moto",
    "slow", "slow and tenderly", "very slow", "slow and expressive",
    "with expression", "tenderly", "playfully", "playfulty",
    "plaintively", "brilliantly", "gracefully", "brightly",
    "1st setting", "2nd setting", "3rd setting",
    "1 setting", "2 setting", "end setting",
    "tr", "cresc", "dim", "ff", "mf", "mp", "pp", "sf",
    "double jigs", "single jigs", "reels", "hornpipes", "airs", "slip jigs",
    "marches", "set dances", "polkas", "slides",
})


# Regex fallbacks for patterns PaddleOCR mangles enough that exact-match misses.
# Applied against the same normalized (lowercase, stripped-of-punctuation) text.
_NOT_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d+\s*(st|nd|rd|th|t|set)?\s*setting$"),   # "1t setting", "2nd setting"
    re.compile(r"^pend\b"),                                   # "pend. 13 inches"
    re.compile(r"^=\s*pend"),                                 # "= Pend. 13"
    re.compile(r"^a\s+run$"),                                 # section label, not a title
    re.compile(r"^\d+\s*inches?$"),                           # metronome tail
    re.compile(r"^(tempo\s+)?(di|da|del|della)\b"),           # "Tempo di Marcia" fragments
    re.compile(r"^page\s+\d+"),                               # running page header
)


_ATTRIB_PATTERNS = (
    "o'neill",  # J.O'Neill, F.O'Neill — composer / collector attributions
    "chief o'neill",
)


def _looks_like_attribution(text: str) -> bool:
    low = text.lower().strip(" .,:;!?")
    return any(p in low for p in _ATTRIB_PATTERNS) and len(low) < 30


def _staff_below(
    tr: dict,
    staves: list[StaffSystem],
    *,
    max_gap_ratio: float = 0.6,
    page_h: int,
) -> StaffSystem | None:
    """Return the first staff system that plausibly belongs to this text region.

    max_gap_ratio is measured relative to the text region's own height: a title
    usually sits within a few title-heights above its music.
    """
    tx0, ty0, tx1, ty1 = tr["bbox"]
    title_h = max(ty1 - ty0, 12)
    max_gap = int(title_h * 6) + 60  # absolute cap
    max_gap = min(max_gap, int(page_h * max_gap_ratio))

    best: tuple[int, StaffSystem] | None = None
    for s in staves:
        if s.y0 <= ty1:
            continue
        if s.y0 - ty1 > max_gap:
            continue
        # Horizontal overlap: at least 25% of title width must sit within the
        # staff's x-range (titles often extend further than the staff).
        overlap_x0 = max(tx0, s.x0)
        overlap_x1 = min(tx1, s.x1)
        overlap = max(0, overlap_x1 - overlap_x0)
        if overlap < (tx1 - tx0) * 0.25 and overlap < 80:
            continue
        if best is None or s.y0 < best[0]:
            best = (s.y0, s)
    return best[1] if best else None


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    page_png = stage_dir(1, "pages", book) / f"page_{page:04d}.png"
    layout_json = stage_dir(2, "layout", book) / f"page_{page:04d}.json"
    out_dir = stage_dir(3, "tunes", book) / f"page_{page:04d}"
    if not layout_json.exists():
        raise StageError(f"layout.json missing: {layout_json}", stage="s03_segment", book=book, page=page)

    payload = json.loads(layout_json.read_text())
    w, h = payload["width"], payload["height"]
    text_regions: list[dict] = payload["text_regions"]
    staves = _staves_from_json(payload["staff_systems"])

    if not staves:
        # Empty tune set. Record the marker so the driver does not repeat work.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".no_music").write_text("no staff systems detected on page\n")
        log.info("segment %s p%d: no staves — skipping (%d text regions)", book, page, len(text_regions))
        return StageResult(ok=True, outputs=[], meta={"n_tunes": 0})

    # Identify title candidates: text region → first staff immediately below.
    candidates: list[tuple[dict, StaffSystem]] = []
    for tr in text_regions:
        if not _is_plausible_title(tr, w, h):
            continue
        s = _staff_below(tr, staves, page_h=h)
        if s is None:
            continue
        candidates.append((tr, s))

    # Group candidates by the staff they point to. The title of each tune is
    # the best-scoring text region in its group. Scoring prefers (a) clean
    # ASCII-English over garbled Gaelic OCR, since PaddleOCR's English model
    # doesn't handle Seanchló script (these books typeset both), then (b)
    # larger area (bold English title > small Irish subtitle > tempo mark).
    def _area(tr: dict) -> int:
        b = tr["bbox"]
        return (b[2] - b[0]) * (b[3] - b[1])

    def _english_score(text: str) -> float:
        """Fraction of characters that are plain ASCII letters or spaces.
        High = looks like English; low = garbled Gaelic OCR or non-Latin."""
        if not text:
            return 0.0
        clean = text.replace(".", "").replace(",", "").replace("'", "")
        if not clean:
            return 0.0
        english_chars = sum(1 for c in clean if c.isascii() and (c.isalpha() or c == " "))
        return english_chars / len(clean)

    def _title_score(tr: dict) -> float:
        """Higher = better title. Combines English-ness (primary) and area."""
        return _english_score(tr["text"]) * 1000 + _area(tr) / 10000.0

    by_staff: dict[int, list[dict]] = {}
    for tr, s in candidates:
        by_staff.setdefault(id(s), []).append({"tr": tr, "staff": s})

    dedup: list[dict] = []
    for _staff_id, group in by_staff.items():
        group.sort(key=lambda g: -_title_score(g["tr"]))
        primary = group[0]
        others = sorted(group[1:], key=lambda g: g["tr"]["bbox"][1])
        dedup.append({
            "title_bbox": primary["tr"]["bbox"],
            "title_text": primary["tr"]["text"],
            "subtitle_bboxes": [o["tr"]["bbox"] for o in others],
            "subtitle_texts": [o["tr"]["text"] for o in others],
            "first_staff": primary["staff"],
        })
    # Sort the tunes by staff y-order so subsequent logic walks top-to-bottom.
    dedup.sort(key=lambda d: d["first_staff"].y0)

    # Walk staves; assign each to the tune whose title immediately precedes it.
    staves_sorted = sorted(staves, key=lambda s: s.y0)
    if not dedup:
        # No titles detected — fall back: one tune per page, all staves.
        tunes: list[dict] = [
            {
                "title_bbox": None,
                "title_text": "",
                "subtitle_bboxes": [],
                "subtitle_texts": [],
                "staves": staves_sorted,
            }
        ]
    else:
        tunes = []
        for i, d in enumerate(dedup):
            start_y = d["title_bbox"][1] if i > 0 else 0
            end_y = dedup[i + 1]["title_bbox"][1] if i + 1 < len(dedup) else h
            tune_staves = [s for s in staves_sorted if start_y <= s.y0 < end_y]
            if not tune_staves:
                continue
            tunes.append({
                "title_bbox": d["title_bbox"],
                "title_text": d["title_text"],
                "subtitle_bboxes": d["subtitle_bboxes"],
                "subtitle_texts": d["subtitle_texts"],
                "staves": tune_staves,
            })
        # Any staves above the first title (page-continuation): attach to tune 0.
        first_title_y = dedup[0]["title_bbox"][1]
        carryover = [s for s in staves_sorted if s.y0 < first_title_y]
        if carryover and tunes:
            tunes.insert(0, {
                "title_bbox": None,
                "title_text": "",
                "subtitle_bboxes": [],
                "subtitle_texts": [],
                "staves": carryover,
                "continuation": True,
            })

    outputs = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Crop the full page width. Previously we trimmed 20 px from each side,
    # which chopped the ending double-barlines on books whose printed margin
    # is narrower than that (e.g. oneill_music p22 THANKSGIVING / SAILING).
    crop_x0 = 0
    crop_x1 = w

    # Compute vertical crop boundaries. Each tune's crop spans from just above
    # its title (or 80 px above its first staff if untitled) down to just
    # above the next tune's title. YOLO's staff bbox is tight around the 5
    # lines and misses note stems, so we add generous stem room (80 px) when
    # the actual boundary would otherwise clip notes.
    n_tunes = len(tunes)
    for idx, t in enumerate(tunes, start=1):
        my_first_staff_y0 = min(s.y0 for s in t["staves"])
        my_last_staff_y1 = max(s.y1 for s in t["staves"])

        # Preferred top: just above my own title/subtitle, else 80 px above
        # my first staff (room for stems).
        my_title_tops: list[int] = []
        if t["title_bbox"] is not None:
            my_title_tops.append(t["title_bbox"][1])
        my_title_tops.extend(b[1] for b in t.get("subtitle_bboxes", []))
        if my_title_tops:
            top_y = max(0, min(my_title_tops) - 15)
        else:
            top_y = max(0, my_first_staff_y0 - 80)
        # Never overlap the previous tune's last staff.
        if idx > 1:
            prev_last_y1 = max(s.y1 for s in tunes[idx - 2]["staves"])
            top_y = max(top_y, prev_last_y1 + 10)
        # Never go below my own first staff.
        top_y = min(top_y, my_first_staff_y0 - 10)

        # Preferred bottom: midway to next tune, clipped just above next title.
        if idx == n_tunes:
            bottom_y = min(h, my_last_staff_y1 + 80)
        else:
            next_tune = tunes[idx]
            next_title_tops: list[int] = []
            if next_tune.get("title_bbox") is not None:
                next_title_tops.append(next_tune["title_bbox"][1])
            next_title_tops.extend(b[1] for b in next_tune.get("subtitle_bboxes", []))
            next_first_y0 = min(s.y0 for s in next_tune["staves"])
            if next_title_tops:
                bottom_y = min(next_title_tops) - 15
            else:
                bottom_y = (my_last_staff_y1 + next_first_y0) // 2
            # Give ourselves at least 30 px below the last staff for stems.
            bottom_y = max(bottom_y, my_last_staff_y1 + 30)
        bottom_y = min(h, bottom_y)

        u = (crop_x0, top_y, crop_x1, min(h, bottom_y))

        tune_dir = out_dir / f"tune_{idx:02d}"
        crop_png = tune_dir / "crop.png"
        crop_pdf = tune_dir / "crop.pdf"
        meta_json = tune_dir / "meta.json"

        if crop_png.exists() and crop_pdf.exists() and meta_json.exists() and not force:
            outputs.append(tune_dir)
            continue

        tune_dir.mkdir(parents=True, exist_ok=True)
        crop(page_png, u, crop_png)
        png_to_single_page_pdf(crop_png, crop_pdf)

        meta_json.write_text(
            json.dumps(
                {
                    "book": book, "page": page, "idx": idx,
                    "title_bbox": t["title_bbox"],
                    "title_text": t["title_text"],
                    "subtitle_bboxes": t["subtitle_bboxes"],
                    "subtitle_texts": t["subtitle_texts"],
                    "staff_bboxes": [[s.x0, s.y0, s.x1, s.y1] for s in t["staves"]],
                    "n_staves": len(t["staves"]),
                    "continuation": t.get("continuation", False),
                    "crop_bbox": list(u),
                },
                indent=2,
            )
        )
        outputs.append(tune_dir)
        log.info(
            "segment %s p%d t%d: %d staves, title=%r crop=%s",
            book, page, idx, len(t["staves"]), t["title_text"], u,
        )

    return StageResult(ok=True, outputs=outputs, meta={"n_tunes": len(tunes)})
