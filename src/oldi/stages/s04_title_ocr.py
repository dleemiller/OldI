"""Stage 04 — per-tune VLM pass: title OCR + key signature detection.

PaddleOCR's English model produces garbled output on Seanchló (old Irish
Gaelic script with dotted consonants ḋ ġ ṁ ṡ ṫ — used by all 19th-century
Irish tune collections). Gemma 4 is a generalist multimodal LLM with strong
multilingual training, including rare historical typography.

We reuse the already-loaded VLM for a second, tightly-scoped task: reading the
key signature from a narrow crop of the first staff's left edge (clef + the
sharps/flats block immediately after it). Both downstream OMR engines misread
key signatures on these scans — Clarity systematically over-reports sharps
and Audiveris silently omits the key on ~30% of tunes — so we treat Gemma's
reading as authoritative downstream in stage 06.

Output per tune:
    tune_NN.txt              — canonical title for display / filenames
    tune_NN.json             — {title, english, gaelic, slug, raw}
    tune_NN.key.json         — {fifths, mode, clef, raw}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import CONFIG, Config, stage_dir
from ..errors import StageError, StageResult
from ..util.logging import get_logger

log = get_logger()

MODEL_ID = "google/gemma-4-31B-it"

_MODEL = None
_PROCESSOR = None


def _get_model():
    """Lazy singleton. Loads ~62 GB into GPU memory on first call (bf16)."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    log.info("loading %s (~62 GB bf16, first load downloads weights)", MODEL_ID)
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    _MODEL.eval()
    return _MODEL, _PROCESSOR


_PROMPT = (
    "This image is a region from an engraved 19th-century book of Irish "
    "traditional music. It MAY contain a tune title (in English, in Irish "
    "Seanchló with dotted consonants ḋ ġ ṁ ṡ ṫ, or both) — but it may also "
    "contain non-title text that was captured alongside: tempo marks "
    "(Allegro, Andante, Moderato), dynamics (p, f, mf, cresc), section labels "
    "(1st Setting, 2nd Setting, a Run), metronome indications (Pend. 13 "
    "inches), genre headers (Double Jigs, Reels, Airs), or page/book "
    "headers.\n\n"
    "Identify only the TUNE'S PROPER NAME. Return a JSON object of the form "
    '{"english": "...", "gaelic": "..."} with either field set to null if '
    "that language is absent. Use Seanchló as the 'gaelic' field, and a "
    "modern English-alphabet title (or Anglicized Irish) as the 'english' "
    'field. If the region only shows tempo, dynamic, section or header text, '
    'return {"english": null, "gaelic": null}. Preserve exact capitalization. '
    "Output only the JSON, no commentary."
)


# Regex + exact-match fallback: if Gemma still returns a tempo/section/dynamic,
# wipe that field. Caseless comparison on the stripped string.
_TITLE_REJECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(pend\.?|metronome)", re.IGNORECASE),          # "Pend. 13 inches"
    re.compile(r"^(allegro|andante|moderato|adagio|largo|presto|vivace|allegretto|con moto|"
               r"slow|very slow|with expression|tenderly|playfully|plaintively|"
               r"brilliantly|gracefully|brightly)\.?$", re.IGNORECASE),
    re.compile(r"^\d+(st|nd|rd|th)?\s*setting\s*\.?$", re.IGNORECASE),
    re.compile(r"^a\s+run\.?$", re.IGNORECASE),
    re.compile(r"^(double\s+jigs?|single\s+jigs?|slip\s+jigs?|reels?|hornpipes?|"
               r"airs?|marches?|set\s+dances?|polkas?|slides?|strathspeys?)\.?$", re.IGNORECASE),
    re.compile(r"^[pfmcsz]+\.?$", re.IGNORECASE),                # stray dynamic markers
    re.compile(r"(waifs\s+and\s+strays|dance\s+music\s+of\s+ireland|"
               r"ancient\s+music\s+of\s+ireland|music\s+of\s+ireland)",
               re.IGNORECASE),
)


def _sanitize_title(value: str | None) -> str | None:
    """Strip obvious non-title lines. Handles multi-line OCR output where the
    book's running header got captured alongside the actual tune title."""
    if not value:
        return None
    kept: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pat.search(line) for pat in _TITLE_REJECT_PATTERNS):
            continue
        if len(line) < 2:
            continue
        kept.append(line)
    if not kept:
        return None
    # Strip a leading tune-number prefix (e.g., "39. The Lamentation Of ..." → "The ...").
    result = " ".join(kept)
    result = re.sub(r"^\d+\.\s*", "", result)
    return result.strip() or None


_KEY_PROMPT = (
    "This image is a tight crop of the LEFT EDGE of a single staff from an "
    "engraved 19th-century Irish tune book. Identify the CLEF and the KEY "
    "SIGNATURE by reasoning carefully about the symbols left to right.\n\n"
    "The staff has 5 horizontal lines. Reading left to right you will see:\n"
    "  1. The CLEF (usually treble — a spiral curl around the G line).\n"
    "  2. The KEY SIGNATURE: zero or more SHARPS (♯ — two short vertical "
    "lines crossed by two slanted lines, looking like the hash '#' symbol) "
    "OR zero or more FLATS (♭ — a small 'b' shape with a pointed loop). "
    "Each symbol sits ON a specific staff line. Sharps and flats are NEVER "
    "mixed in a key signature.\n"
    "  3. The TIME SIGNATURE: two stacked digits, most commonly 6/8, 4/4, "
    "2/4, 3/4, 9/8, 12/8, or the letter C (common time). The digit '6' is a "
    "ROUNDED CURVE with a loop at the bottom — it is NOT a sharp. Do not "
    "confuse the '6' or '4' of a time signature with a sharp symbol.\n"
    "  4. The first notes of the melody. Accidentals attached to INDIVIDUAL "
    "NOTES (a sharp right before a single notehead) are NOT part of the key "
    "signature — ignore them.\n\n"
    "Count ONLY the sharps or flats in block 2 — between the clef curl and "
    "the time-signature digits (or first notehead if no time signature). "
    "Many Irish jigs in G major show exactly ONE sharp followed directly by "
    "'6/8'; this is a common layout that can look like '#6/8'.\n\n"
    "Return a JSON object of the form "
    '{"clef": "treble|bass|alto|tenor", "fifths": N, "mode": "major|minor|unknown", '
    '"reasoning": "short step-by-step description"} '
    "where fifths is the SIGNED count: +N for N sharps, -N for N flats, 0 "
    "for no accidentals, in the range -7..+7. For mode, use 'major' for most "
    "Irish dance tunes; 'minor' only when the title or feel clearly indicates "
    "it; else 'unknown'. Output only the JSON, no commentary."
)


def _run_vlm_raw(image_path: Path, prompt: str, max_new_tokens: int) -> str:
    """Core VLM call: image + prompt → raw string. Used by both title and key detectors."""
    import torch
    from PIL import Image

    model, processor = _get_model()
    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()


def _run_vlm(image_path: Path) -> dict:
    raw = _run_vlm_raw(image_path, _PROMPT, max_new_tokens=160)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"english": raw, "gaelic": None, "raw": raw}
    else:
        payload["raw"] = raw
    return payload


def _run_key_vlm(image_path: Path) -> dict:
    """Ask Gemma to read the clef and key signature. Returns {fifths, mode, clef, raw}.

    fifths is None when the VLM response didn't parse as valid JSON with a
    fifths field in [-7, 7]; callers should then fall back to engine output.
    """
    raw = _run_vlm_raw(image_path, _KEY_PROMPT, max_new_tokens=220)
    out = {"fifths": None, "mode": None, "clef": None, "raw": raw}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return out
    f = payload.get("fifths")
    if isinstance(f, bool):
        # JSON 'true' / 'false' parses to bool (subclass of int); reject.
        pass
    elif isinstance(f, int) and -7 <= f <= 7:
        out["fifths"] = f
    elif isinstance(f, str):
        try:
            n = int(f)
            if -7 <= n <= 7:
                out["fifths"] = n
        except ValueError:
            pass
    mode = payload.get("mode")
    if isinstance(mode, str):
        mode = mode.strip().lower()
        if mode in {"major", "minor", "unknown"}:
            out["mode"] = mode
    clef = payload.get("clef")
    if isinstance(clef, str):
        out["clef"] = clef.strip().lower() or None
    return out


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip().lower()
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:80] or "untitled"


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    tunes_dir = stage_dir(3, "tunes", book) / f"page_{page:04d}"
    out_dir = stage_dir(4, "titles", book) / f"page_{page:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tunes_dir.exists():
        raise StageError(f"tune dir missing: {tunes_dir}", stage="s04_title_ocr", book=book, page=page)

    outputs: list[Path] = []
    from PIL import Image

    page_png = stage_dir(1, "pages", book) / f"page_{page:04d}.png"

    for tune_dir in sorted(tunes_dir.glob("tune_*")):
        idx = int(tune_dir.name.split("_")[1])
        title_txt = out_dir / f"tune_{idx:02d}.txt"
        title_json = out_dir / f"tune_{idx:02d}.json"
        key_json = out_dir / f"tune_{idx:02d}.key.json"

        need_title = force or not (title_txt.exists() and title_json.exists())
        need_key = force or not key_json.exists()

        if not need_title and not need_key:
            outputs.append(title_txt)
            continue

        meta = json.loads((tune_dir / "meta.json").read_text())
        title_bbox = meta.get("title_bbox")

        # --- Title OCR ---
        if need_title:
            payload: dict = {"english": None, "gaelic": None, "raw": ""}

            if title_bbox is not None:
                # Union title + subtitles so the VLM sees both language versions.
                xs = [title_bbox[0]] + [b[0] for b in meta.get("subtitle_bboxes", [])]
                ys = [title_bbox[1]] + [b[1] for b in meta.get("subtitle_bboxes", [])]
                xe = [title_bbox[2]] + [b[2] for b in meta.get("subtitle_bboxes", [])]
                ye = [title_bbox[3]] + [b[3] for b in meta.get("subtitle_bboxes", [])]
                crop_bbox = (min(xs) - 10, min(ys) - 5, max(xe) + 10, max(ye) + 5)

                title_crop = tune_dir / "title.png"
                with Image.open(page_png) as im:
                    clamped = (
                        max(0, crop_bbox[0]), max(0, crop_bbox[1]),
                        min(im.size[0], crop_bbox[2]), min(im.size[1], crop_bbox[3]),
                    )
                    im.crop(clamped).save(title_crop, "PNG")

                try:
                    payload = _run_vlm(title_crop)
                except Exception as exc:
                    log.warning("VLM OCR failed %s p%d t%d: %s", book, page, idx, exc)
                    payload = {
                        "english": meta.get("title_text", ""),
                        "gaelic": None,
                        "raw": f"fallback: {meta.get('title_text', '')}",
                    }

            english = _sanitize_title(payload.get("english"))
            gaelic = _sanitize_title(payload.get("gaelic"))
            title = english or gaelic or f"untitled_p{page}_t{idx}"
            slug = _slug(english or gaelic or title)

            title_txt.write_text(title + "\n")
            title_json.write_text(json.dumps({
                "title": title,
                "english": english,
                "gaelic": gaelic,
                "slug": slug,
                "raw": payload.get("raw", ""),
            }, indent=2, ensure_ascii=False))
            outputs.append(title_txt)
            log.info(
                "title %s p%d t%d: en=%r ga=%r (slug=%s)",
                book, page, idx, english, gaelic, slug,
            )
        else:
            outputs.append(title_txt)

        # --- Key signature detection ---
        # Clarity and Audiveris both misread key signatures on these scans
        # (Clarity over-counts sharps; Audiveris frequently emits no key at
        # all). Ask Gemma directly — it has a much stronger visual prior for
        # the clef-plus-accidentals block.
        if need_key:
            key_payload = {"fifths": None, "mode": None, "clef": None, "raw": ""}
            staff_bboxes = meta.get("staff_bboxes") or []
            if staff_bboxes:
                sx0, sy0, sx1, sy1 = staff_bboxes[0]
                # PP-Structure's staff bbox starts at the first note rather
                # than at the clef, and the gap between clef and first note
                # varies widely between tunes (tune-number labels, wide
                # flourishes on the clef, etc). We anchor the LEFT edge at
                # the tune's crop bbox (which is the page's left margin), so
                # however far right the first note sits, the clef is always
                # included. Rightward we keep a few notes of context so the
                # VLM can tell the key-signature block apart from per-note
                # accidentals.
                crop_x0 = (meta.get("crop_bbox") or [0, 0, 0, 0])[0]
                ksig_box = (
                    max(0, crop_x0 - 10),
                    max(0, sy0 - 50),
                    sx0 + 300,
                    sy1 + 50,
                )
                ksig_png = tune_dir / "ksig.png"
                with Image.open(page_png) as im:
                    clamped_ks = (
                        max(0, ksig_box[0]), max(0, ksig_box[1]),
                        min(im.size[0], ksig_box[2]), min(im.size[1], ksig_box[3]),
                    )
                    im.crop(clamped_ks).save(ksig_png, "PNG")
                try:
                    key_payload = _run_key_vlm(ksig_png)
                except Exception as exc:
                    log.warning("VLM key detect failed %s p%d t%d: %s", book, page, idx, exc)
            key_json.write_text(json.dumps(key_payload, indent=2))
            log.info(
                "key %s p%d t%d: fifths=%s mode=%s clef=%s",
                book, page, idx,
                key_payload.get("fifths"),
                key_payload.get("mode"),
                key_payload.get("clef"),
            )

    return StageResult(ok=True, outputs=outputs)
