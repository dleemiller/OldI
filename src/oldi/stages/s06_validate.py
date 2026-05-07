"""Stage 06 — validate dual-engine MusicXML and pick the winner per tune.

For each tune we have:
    tune_NN.clarity.musicxml    — from Clarity-OMR
    tune_NN.audiveris.musicxml  — from Audiveris

Both are validated with music21 and scored; the better one is copied to the
canonical output `data/06_validated/<book>/page_NNNN/tune_NN.musicxml`. The
`tune_NN.engine` marker records the winner.

Scoring (lower is better; ties broken by preferring Clarity since it was
tuned for traditional Irish fiddle/whistle printing):
    - Parse failure: ∞
    - Per note outside the tin-whistle range: +1
    - Per chord note (monophony violation): +2
    - Very short output (< 4 notes): +50

Post-processing (applied in-place on the stage-05 outputs, before scoring):
    - Multi-part outputs are collapsed to a single monophonic line. Two cases:
      (a) Clarity's phantom grand-staff artifact (part 2 is really the next
      staff line of the same tune — concat measures); (b) Audiveris real grand
      staff over Petrie-style piano arrangements (drop the accompaniment part).
    - Audiveris chord stacks are flattened to the top pitch: the source is
      monophonic, so simultaneous notes are OMR artifacts.
    - Clarity rests are stripped: Clarity's decoder inserts spurious rests
      that render as visible gaps in otherwise continuous dance music.

After picking a winner we call `_enhance_winner` to graft anything the winner
is missing (voltas, repeats, key/time/clef) from the other engine. Clarity has
no volta token at all and never emits repeat barlines, while Audiveris often
fails to detect the key or time signature — each engine covers the other's
blind spot.
"""

from __future__ import annotations

import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import CONFIG, Config, stage_dir
from ..errors import StageError, StageResult
from ..manifest import upsert_tune
from ..util.logging import get_logger

log = get_logger()

ENGINE_PREFERENCE: tuple[str, ...] = ("clarity", "audiveris")

# Key signature fifths (-7..+7) → major-mode tonic. Irish trad repertoire is
# almost always published in a major or modal-of-major key, so we default to
# the major name and only switch to relative-minor naming when the MusicXML
# explicitly says <mode>minor</mode>.
_MAJOR_BY_FIFTHS: dict[int, str] = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C",
    1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}
_RELATIVE_MINOR_OF_MAJOR: dict[str, str] = {
    "Cb": "Abm", "Gb": "Ebm", "Db": "Bbm", "Ab": "Fm", "Eb": "Cm",
    "Bb": "Gm", "F": "Dm", "C": "Am", "G": "Em", "D": "Bm",
    "A": "F#m", "E": "C#m", "B": "G#m", "F#": "D#m", "C#": "A#m",
}
_STEP_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _written_key_from_xml(xml_path: Path) -> str:
    """Return the written key signature as `D`, `Em`, etc. Empty if absent."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError, OSError):
        return ""
    fifths_el = root.find(".//key/fifths")
    if fifths_el is None or fifths_el.text is None:
        return ""
    try:
        n = int(fifths_el.text)
    except ValueError:
        return ""
    mode_el = root.find(".//key/mode")
    mode = (mode_el.text or "").strip().lower() if mode_el is not None and mode_el.text else ""
    major = _MAJOR_BY_FIFTHS.get(n, "")
    if mode == "minor":
        return _RELATIVE_MINOR_OF_MAJOR.get(major, major + "m" if major else "")
    return major


def _pitch_midi(note_el: ET.Element) -> int:
    p = note_el.find("pitch")
    if p is None:
        return -1
    step = (p.findtext("step") or "").strip()
    try:
        oct_ = int(p.findtext("octave") or 0)
        alter = int(p.findtext("alter") or 0)
    except ValueError:
        return -1
    return oct_ * 12 + _STEP_SEMI.get(step, 0) + alter


def _flatten_chords(xml_path: Path) -> int:
    """Keep only the top pitch of each chord stack. Idempotent."""
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return 0
    removed = 0
    for measure in tree.getroot().iter("measure"):
        notes = list(measure.findall("note"))
        i = 0
        while i < len(notes):
            group = [notes[i]]
            j = i + 1
            while j < len(notes) and notes[j].find("chord") is not None:
                group.append(notes[j])
                j += 1
            if len(group) > 1:
                top_idx = max(range(len(group)), key=lambda k: _pitch_midi(group[k]))
                top = group[top_idx]
                chord_el = top.find("chord")
                if chord_el is not None:
                    top.remove(chord_el)
                for k, n in enumerate(group):
                    if k != top_idx:
                        measure.remove(n)
                        removed += 1
            i = j
    if removed:
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return removed


def _mark_pickup_measure(xml_path: Path) -> bool:
    """Tag a short first measure as a pickup (anacrusis) by setting
    `implicit="yes"`. Many Irish trad tunes start on an upbeat, and both OMR
    engines render the lead-in note as a full measure — that throws Verovio's
    line-wrapping off (4 bars on one line, 5 on the next). Marking the pickup
    implicit tells renderers not to count it in line layout. Idempotent.

    Detection: the first measure's total note duration is compared to the mode
    of all other measures. If it's under 80% of the mode, it's a pickup.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return False
    root = tree.getroot()
    measures = list(root.iter("measure"))
    if len(measures) < 3:
        return False

    def measure_duration(m: ET.Element) -> int:
        total = 0
        for n in m.findall("note"):
            if n.find("chord") is not None:
                # Chord siblings share timing — count the primary note only.
                continue
            d = n.find("duration")
            if d is not None and d.text:
                try:
                    total += int(d.text)
                except ValueError:
                    pass
        return total

    durations = [measure_duration(m) for m in measures]
    nonzero = [d for d in durations[1:] if d > 0]
    if not nonzero or durations[0] <= 0:
        return False
    from collections import Counter
    mode_dur = Counter(nonzero).most_common(1)[0][0]
    if mode_dur <= 0:
        return False
    if durations[0] >= mode_dur * 0.80:
        return False

    m0 = measures[0]
    if m0.get("implicit") == "yes":
        return False
    m0.set("implicit", "yes")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return True


def _inject_default_repeats(xml_path: Path) -> dict:
    """Heuristically add `|:` and `:|` barlines to a tune that has no repeats.

    Irish trad repertoire is overwhelmingly in AABB form with each section
    repeated (`|: A :||: B :||`). Both OMR engines fail to detect the thin
    repeat dots reliably. If the tune has zero `<repeat>` elements AND no
    `<ending>` (volta) markers, we inject defaults:

    * `<repeat direction="forward">` on the first full measure (skipping any
      pickup marked implicit).
    * `<repeat direction="backward">` on the last measure.
    * For tunes with >=14 non-pickup measures where the count is even, also
      add a midpoint `:||:` split so the rendered output shows AABB.

    We skip injection when voltas are already present, to avoid colliding with
    Audiveris's detected 1st/2nd-ending structure.

    Returns a summary dict.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return {"added": 0, "reason": "parse_error"}
    root = tree.getroot()
    measures = list(root.iter("measure"))
    if len(measures) < 4:
        return {"added": 0, "reason": "too_short"}

    # Bail if any repeats or voltas already exist.
    for m in measures:
        for b in m.findall("barline"):
            if b.find("repeat") is not None or b.find("ending") is not None:
                return {"added": 0, "reason": "structure_present"}

    # Find the first full measure (skip pickup).
    first_full_idx = 0
    if measures[0].get("implicit") == "yes":
        first_full_idx = 1
    first = measures[first_full_idx]
    last = measures[-1]

    def ensure_barline(measure: ET.Element, location: str) -> ET.Element:
        for b in measure.findall("barline"):
            if b.get("location", "right") == location:
                return b
        b = ET.SubElement(measure, "barline")
        b.set("location", location)
        return b

    def add_repeat(measure: ET.Element, location: str, direction: str) -> None:
        b = ensure_barline(measure, location)
        if b.find("repeat") is not None:
            return
        # Left barlines typically go at position 0; right barlines at end.
        if location == "left" and list(measure).index(b) != 0:
            measure.remove(b)
            measure.insert(0, b)
        rep = ET.SubElement(b, "repeat")
        rep.set("direction", direction)

    added = 0
    add_repeat(first, "left", "forward"); added += 1
    add_repeat(last,  "right", "backward"); added += 1

    split_info: dict = {}
    non_pickup_count = len(measures) - first_full_idx
    if non_pickup_count >= 14 and non_pickup_count % 2 == 0:
        mid = first_full_idx + non_pickup_count // 2
        a_end = measures[mid - 1]
        b_start = measures[mid]
        add_repeat(a_end, "right", "backward"); added += 1
        add_repeat(b_start, "left", "forward"); added += 1
        split_info = {"aabb_split_at": mid}

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return {"added": added, "first_full": first_full_idx, **split_info}


def _strip_rests(xml_path: Path) -> int:
    """Remove <note><rest/></note> elements. Idempotent."""
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return 0
    removed = 0
    for measure in tree.getroot().iter("measure"):
        for note in list(measure.findall("note")):
            if note.find("rest") is not None:
                measure.remove(note)
                removed += 1
    if removed:
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return removed


def _concat_parts_to_first(xml_path: Path) -> int:
    """Collapse multi-part MusicXML into the monophonic melody line.

    Two distinct failure modes lead to multi-part output:

    * **Clarity phantom grand staff** — YOLO stage A used to merge any two
      close staves into a piano-style grand-staff group, so stage B decoded a
      phantom bass clef for what is really the second staff line of the same
      monophonic tune. Here part 2 continues part 1 (different measure counts
      per part, summing to the real tune length) and the fix is to concatenate
      measures. The upstream patch in `vendor/Clarity-OMR/src/models/yolo_stage_a.py`
      prevents this for new runs; this helper defensively repairs cached output.

    * **Audiveris real grand staff** — Petrie-style piano arrangements carry a
      melody on top and accompaniment below, both in treble clef. Each part
      holds the *same* measure count since they sound simultaneously. For our
      tin-whistle goal we keep just the top (melody) part and discard the rest.

    Discrimination: if all parts have equal measure counts we assume a real
    grand staff and drop extras; otherwise we concatenate.

    Returns the number of extra parts removed.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return 0
    root = tree.getroot()
    parts = root.findall("part")
    if len(parts) < 2:
        return 0
    first = parts[0]

    def mnum(m: ET.Element) -> int:
        try:
            return int(m.get("number", "0"))
        except ValueError:
            return 0

    measure_counts = [len(p.findall("measure")) for p in parts]
    parts_are_parallel = len(set(measure_counts)) == 1

    if parts_are_parallel:
        # Real grand staff: drop extras, keep the top (melody) part.
        for extra in parts[1:]:
            root.remove(extra)
    else:
        # Phantom second-staff artifact: concatenate into part 1.
        max_n = max((mnum(m) for m in first.findall("measure")), default=0)
        for extra in parts[1:]:
            em = extra.findall("measure")
            if em:
                # Drop the attributes block from the very first measure of the
                # appended part, otherwise clef/key/time gets re-announced.
                attrs = em[0].find("attributes")
                if attrs is not None:
                    em[0].remove(attrs)
            for m in em:
                max_n += 1
                m.set("number", str(max_n))
                first.append(m)
            root.remove(extra)

    pl = root.find("part-list")
    if pl is not None:
        for sp in list(pl.findall("score-part"))[1:]:
            pl.remove(sp)

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return len(parts) - 1


def _sanitize_metadata(xml_path: Path, tp: dict, book: str | None = None) -> None:
    """Rewrite placeholder titles/composers with the detected tune title.

    Clarity's output is serialised via music21, which injects
    `<movement-title>Music21 Fragment</movement-title>` and
    `<creator type="composer">Music21</creator>` by default. Audiveris leaks
    `<source>/in/crop.pdf</source>` from the container filesystem. Neither
    belongs in a published tune file. We replace with the OCR'd title (English
    preferred, Gaelic fallback) and the book slug as "rights / source".
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return
    root = tree.getroot()
    english = (tp.get("english") or "").strip()
    gaelic = (tp.get("gaelic") or "").strip()
    display = english or gaelic or (tp.get("title") or "").strip()

    # Reset title elements in canonical order: <work>, <movement-title>.
    for tag in ("movement-title", "work"):
        el = root.find(tag)
        if el is not None:
            root.remove(el)
    if display:
        work = ET.Element("work")
        wt = ET.SubElement(work, "work-title")
        wt.text = display
        mt = ET.Element("movement-title")
        mt.text = display
        # Insert in reverse (insert(0,…)) so work ends up before movement-title.
        root.insert(0, mt)
        root.insert(0, work)

    # Replace Audiveris's generic "Voice" / "Voice Oohs" part metadata with a
    # meaningful tin-whistle label. Renderers like Verovio show <part-name>
    # next to the staff; a title-case instrument name is nicer than "Voice".
    pl = root.find("part-list")
    if pl is not None:
        for sp in pl.findall("score-part"):
            pn = sp.find("part-name")
            if pn is not None and (pn.text or "").strip() in {"Voice", "", "Part", "Music21 Part"}:
                pn.text = "Tin Whistle"
            pa = sp.find("part-abbreviation")
            if pa is not None and (pa.text or "").strip() in {"Voice", "", "Part"}:
                pa.text = "TW"
            for si in sp.findall("score-instrument"):
                iname = si.find("instrument-name")
                if iname is not None and (iname.text or "").strip() == "Voice Oohs":
                    iname.text = "Tin Whistle"

    ident = root.find("identification")
    if ident is not None:
        for c in list(ident.findall("creator")):
            if (c.text or "").strip().lower() in {"music21", ""}:
                ident.remove(c)
        for src in list(ident.findall("source")):
            if (src.text or "").startswith("/in/") or src.text in (None, ""):
                ident.remove(src)
        # Audiveris leaks the container's input path into <miscellaneous>.
        misc = ident.find("miscellaneous")
        if misc is not None:
            for mf in list(misc.findall("miscellaneous-field")):
                txt = (mf.text or "").strip()
                if txt.startswith("/in/") or txt.startswith("/data/") or txt == str(xml_path):
                    misc.remove(mf)
            if not list(misc):
                ident.remove(misc)
        if book:
            existing = next((r for r in ident.findall("rights") if r.text and book in r.text), None)
            if existing is None:
                rights = ET.SubElement(ident, "rights")
                rights.text = f"source: {book}"

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _enhance_winner(
    winner_path: Path,
    other_path: Path | None,
    out_path: Path,
    *,
    prefer_other_key: bool = False,
) -> dict:
    """Copy winner to out_path and borrow structure the winner is missing.

    Specifically:

    * If the other engine has `<ending>` or `<repeat>` barlines and the winner
      does not, graft them onto the winner's measures by proportional index.
    * If the winner's first measure lacks `<key>`, `<time>`, or `<clef>` and
      the other engine's first measure has one, copy it over.
    * If ``prefer_other_key`` is set and the other engine has a DIFFERENT key
      from the winner, overwrite the winner's key with the other engine's.
      Used when the scoring winner (Clarity) is empirically worse than the
      runner-up (Audiveris) at reading key signatures — we pick Clarity for
      notes but Audiveris for the signature.
    """
    try:
        w_tree = ET.parse(winner_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        shutil.copy2(winner_path, out_path)
        return {"merged": False, "reason": "winner_parse_error"}
    info: dict = {"merged": False, "inserted": 0, "borrowed_attrs": []}

    o_tree: ET.ElementTree | None = None
    if other_path is not None and other_path.exists():
        try:
            o_tree = ET.parse(other_path)
        except (ET.ParseError, OSError):
            o_tree = None

    w_measures = list(w_tree.getroot().iter("measure"))

    # --- Borrow missing attributes (key / time / clef) from the first measure. ---
    if o_tree is not None and w_measures:
        o_measures = list(o_tree.getroot().iter("measure"))
        if o_measures:
            w_attrs = w_measures[0].find("attributes")
            o_attrs = o_measures[0].find("attributes")
            if w_attrs is None and o_attrs is not None:
                # Fresh copy of the whole attributes block.
                w_measures[0].insert(0, copy.deepcopy(o_attrs))
                info["borrowed_attrs"].append("attributes(all)")
            elif w_attrs is not None and o_attrs is not None:
                for tag in ("key", "time", "clef"):
                    if w_attrs.find(tag) is None and o_attrs.find(tag) is not None:
                        w_attrs.append(copy.deepcopy(o_attrs.find(tag)))
                        info["borrowed_attrs"].append(tag)
                if prefer_other_key:
                    w_key = w_attrs.find("key")
                    o_key = o_attrs.find("key")
                    if w_key is not None and o_key is not None:
                        w_fifths = (w_key.findtext("fifths") or "").strip()
                        o_fifths = (o_key.findtext("fifths") or "").strip()
                        if w_fifths != o_fifths and o_fifths:
                            w_attrs.remove(w_key)
                            w_attrs.append(copy.deepcopy(o_key))
                            info["borrowed_attrs"].append(f"key(override:{w_fifths}→{o_fifths})")

    # --- Graft voltas / repeat barlines from the other engine if missing. ---
    has_winner_voltas = any(
        m.find("barline/ending") is not None or m.find("barline/repeat") is not None
        for m in w_measures
    )
    if o_tree is not None and not has_winner_voltas and w_measures:
        o_measures = list(o_tree.getroot().iter("measure"))
        n_w, n_o = len(w_measures), len(o_measures)
        if n_w and n_o:
            interesting: list[tuple[int, ET.Element]] = []
            for i, m in enumerate(o_measures):
                for b in m.findall("barline"):
                    if b.find("ending") is not None or b.find("repeat") is not None:
                        interesting.append((i, b))
            if interesting:
                for i, bar in interesting:
                    j = 0 if n_o <= 1 else round(i * (n_w - 1) / (n_o - 1))
                    j = max(0, min(j, n_w - 1))
                    target = w_measures[j]
                    new_bar = copy.deepcopy(bar)
                    loc = new_bar.get("location", "right")
                    if loc == "left":
                        target.insert(0, new_bar)
                    else:
                        target.append(new_bar)
                    info["inserted"] += 1
                info["merged"] = True
                info["n_winner"] = n_w
                info["n_other"] = n_o

    w_tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return info


def _read_title(book: str, page: int, idx: int) -> dict:
    title_json = stage_dir(4, "titles", book) / f"page_{page:04d}" / f"tune_{idx:02d}.json"
    if not title_json.exists():
        fallback = f"untitled_p{page}_t{idx}"
        return {"title": fallback, "slug": fallback, "english": None, "gaelic": None}
    return json.loads(title_json.read_text())


def _read_vlm_key(book: str, page: int, idx: int) -> dict | None:
    """Load the VLM's key-signature reading from stage 04, if present."""
    path = stage_dir(4, "titles", book) / f"page_{page:04d}" / f"tune_{idx:02d}.key.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _apply_vlm_key(xml_path: Path, key_info: dict) -> str | None:
    """Overwrite the key signature in ``xml_path`` with the VLM's reading.

    Returns a short `"old→new"` change string when we modified the file,
    or None when nothing changed (VLM unavailable, or already matches).

    The VLM reads key signatures far more reliably than either OMR engine
    here, so we treat its fifths value as authoritative. Mode is applied
    only when the VLM returned `major` or `minor` — `unknown` leaves the
    existing `<mode>` alone.
    """
    fifths = key_info.get("fifths")
    if fifths is None:
        return None
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError, OSError):
        return None
    root = tree.getroot()
    first_measure = next(iter(root.iter("measure")), None)
    if first_measure is None:
        return None

    attrs = first_measure.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        first_measure.insert(0, attrs)

    key_el = attrs.find("key")
    if key_el is None:
        key_el = ET.SubElement(attrs, "key")

    fifths_el = key_el.find("fifths")
    old_fifths = fifths_el.text if fifths_el is not None else None
    if fifths_el is None:
        fifths_el = ET.SubElement(key_el, "fifths")
        # Keep canonical ordering: <fifths> before <mode>.
        key_el.remove(fifths_el)
        key_el.insert(0, fifths_el)
    fifths_el.text = str(fifths)

    mode = (key_info.get("mode") or "").strip().lower()
    if mode in {"major", "minor"}:
        mode_el = key_el.find("mode")
        if mode_el is None:
            mode_el = ET.SubElement(key_el, "mode")
        mode_el.text = mode

    if str(old_fifths) == str(fifths):
        return None
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return f"{old_fifths}→{fifths}"


def _score_musicxml(xml_path: Path, cfg: Config) -> tuple[float, dict]:
    """Return (score, info). Lower score = better output."""
    from music21 import converter, pitch

    try:
        score_obj = converter.parse(str(xml_path))
    except Exception as exc:
        return float("inf"), {"error": f"parse failed: {exc}"}

    parts = list(score_obj.parts)
    if not parts:
        return float("inf"), {"error": "no parts"}

    notes = list(parts[0].recurse().notes)
    if not notes:
        return float("inf"), {"error": "no notes"}

    n_chords = sum(1 for n in notes if n.isChord)
    pitched = [n for n in notes if n.isNote]
    lo = pitch.Pitch(cfg.tin_whistle_min_pitch)
    hi = pitch.Pitch(cfg.tin_whistle_max_pitch)
    out_of_range = sum(1 for p in pitched if p.pitch.midi < lo.midi or p.pitch.midi > hi.midi)

    key = _written_key_from_xml(xml_path)

    score = 0.0
    if len(notes) < 4:
        score += 50
    score += out_of_range
    score += 2 * n_chords

    info = {
        "n_notes": len(notes),
        "n_chords": n_chords,
        "out_of_range": out_of_range,
        "min_pitch": min(pitched, key=lambda n: n.pitch.midi).pitch.nameWithOctave if pitched else "",
        "max_pitch": max(pitched, key=lambda n: n.pitch.midi).pitch.nameWithOctave if pitched else "",
        "key": key,
    }
    return score, info


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    xml_dir = stage_dir(5, "musicxml", book) / f"page_{page:04d}"
    out_dir = stage_dir(6, "validated", book) / f"page_{page:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xml_dir.exists():
        raise StageError(f"xml dir missing: {xml_dir}", stage="s06_validate", book=book, page=page)

    # Discover tune indices by scanning for any of the engine-specific files.
    indices: set[int] = set()
    for xml in xml_dir.iterdir():
        name = xml.name
        for engine in ENGINE_PREFERENCE:
            suffix = f".{engine}.musicxml"
            if name.endswith(suffix):
                try:
                    idx = int(name.removesuffix(suffix).split("_")[1])
                    indices.add(idx)
                except (ValueError, IndexError):
                    continue

    outputs: list[Path] = []

    for idx in sorted(indices):
        out_xml = out_dir / f"tune_{idx:02d}.musicxml"
        if out_xml.exists() and not force:
            outputs.append(out_xml)
            continue

        tp = _read_title(book, page, idx)
        title = tp.get("title", "")
        slug = tp.get("slug", "")

        aud_xml = xml_dir / f"tune_{idx:02d}.audiveris.musicxml"
        cla_xml = xml_dir / f"tune_{idx:02d}.clarity.musicxml"

        # Repair structural artifacts before scoring so the scorer sees the
        # true monophonic content:
        #   - Clarity sometimes emits phantom second parts (grand-staff artifact)
        #   - Audiveris stacks phantom chords on beamed groups
        #   - Clarity inserts spurious rests
        if cla_xml.exists():
            dropped = _concat_parts_to_first(cla_xml)
            if dropped:
                log.info("concat parts %s p%d t%d clarity: folded %d extra part(s)",
                         book, page, idx, dropped)
            n = _strip_rests(cla_xml)
            if n:
                log.info("strip rests %s p%d t%d clarity: removed %d", book, page, idx, n)
        if aud_xml.exists():
            # Audiveris occasionally emits an accompaniment part for airs where
            # the source has lyrics and figured-bass-style annotations. Collapse.
            dropped = _concat_parts_to_first(aud_xml)
            if dropped:
                log.info("concat parts %s p%d t%d audiveris: folded %d extra part(s)",
                         book, page, idx, dropped)
            n = _flatten_chords(aud_xml)
            if n:
                log.info("flatten chords %s p%d t%d audiveris: removed %d", book, page, idx, n)

        # Score both engines (now cleaned).
        scored: list[tuple[float, str, Path, dict]] = []
        for engine in ENGINE_PREFERENCE:
            xml = xml_dir / f"tune_{idx:02d}.{engine}.musicxml"
            if not xml.exists():
                continue
            score, info = _score_musicxml(xml, cfg)
            scored.append((score, engine, xml, info))
            log.info(
                "score %s p%d t%d %s: score=%.1f notes=%s err=%s",
                book, page, idx, engine, score,
                info.get("n_notes"), info.get("error") or "",
            )

        if not scored:
            log.warning("no engines produced output for %s p%d t%d", book, page, idx)
            if con is not None:
                upsert_tune(con, {
                    "book": book, "page": page, "idx": idx,
                    "title": title, "title_slug": slug,
                    "status": "failed", "error": "no engine output",
                })
            continue

        # Stable sort: lowest score wins; Clarity breaks ties.
        scored.sort(key=lambda t: (t[0], ENGINE_PREFERENCE.index(t[1])))
        best_score, best_engine, best_xml, best_info = scored[0]

        if best_score == float("inf"):
            log.warning("all engines failed validation %s p%d t%d: %s",
                        book, page, idx, best_info.get("error"))
            if con is not None:
                upsert_tune(con, {
                    "book": book, "page": page, "idx": idx,
                    "title": title, "title_slug": slug,
                    "status": "review", "error": best_info.get("error", ""),
                })
            continue

        # Write the canonical winner and merge in structural elements the
        # winner is missing (voltas, repeats, key/time/clef attributes) by
        # borrowing from whichever engine detected them.
        other_xml = aud_xml if best_engine == "clarity" else cla_xml
        # Stage 04 writes a VLM-authoritative key reading per tune. When that
        # is present we use it to overwrite the key below — so we no longer
        # need the engine-to-engine key preference path inside _enhance_winner.
        vlm_key = _read_vlm_key(book, page, idx)
        enh = _enhance_winner(
            best_xml,
            other_xml if other_xml.exists() else None,
            out_xml,
            prefer_other_key=(best_engine == "clarity" and (vlm_key is None or vlm_key.get("fifths") is None)),
        )
        if enh.get("inserted"):
            log.info("merge barlines %s p%d t%d: +%d (w=%d m, o=%d m)",
                     book, page, idx, enh["inserted"],
                     enh.get("n_winner", 0), enh.get("n_other", 0))
        if enh.get("borrowed_attrs"):
            log.info("borrow attrs %s p%d t%d from %s: %s",
                     book, page, idx,
                     "audiveris" if best_engine == "clarity" else "clarity",
                     ", ".join(enh["borrowed_attrs"]))

        # VLM-authoritative key override. Gemma reads the clef+sharps/flats
        # block from a dedicated crop and its answer beats either engine.
        if vlm_key is not None:
            change = _apply_vlm_key(out_xml, vlm_key)
            if change is not None:
                log.info("vlm key %s p%d t%d: fifths %s (mode=%s)",
                         book, page, idx, change, vlm_key.get("mode"))

        # Mark a pickup/anacrusis on the first measure when it's clearly
        # shorter than the rest — stops Verovio from laying out one line as 4
        # bars and the next as 5 because the lead-in got counted as a full bar.
        for p in (out_xml, cla_xml, aud_xml):
            if p.exists() and _mark_pickup_measure(p):
                log.info("pickup marked %s p%d t%d: %s", book, page, idx, p.name)

        # Irish trad is AABB with each section repeated. Both engines miss
        # `|:` / `:|` on most tunes; inject defaults when no structure exists.
        inj = _inject_default_repeats(out_xml)
        if inj.get("added"):
            log.info("inject repeats %s p%d t%d: +%d%s",
                     book, page, idx, inj["added"],
                     f" (AABB split at m{inj['aabb_split_at']+1})" if inj.get("aabb_split_at") else "")

        # Strip OMR-engine metadata placeholders and write the real title onto
        # all three files (canonical winner + each engine's stand-alone copy).
        for p in (out_xml, cla_xml, aud_xml):
            if p.exists():
                _sanitize_metadata(p, tp, book=book)

        (xml_dir / f"tune_{idx:02d}.engine").write_text(best_engine + "\n")
        outputs.append(out_xml)

        status = "ok"
        err = ""
        if best_info.get("out_of_range"):
            status = "review"
            err = f"{best_info['out_of_range']} notes out of range"

        # Read the key from the POST-enhancement file so borrowed attributes
        # (e.g. a key signature copied in from the non-winning engine) show up
        # in the catalog row.
        final_key = _written_key_from_xml(out_xml) or best_info.get("key", "")

        if con is not None:
            upsert_tune(con, {
                "book": book, "page": page, "idx": idx,
                "title": title, "title_slug": slug,
                "title_english": tp.get("english"),
                "title_gaelic": tp.get("gaelic"),
                "key": final_key,
                "source_crop": str(stage_dir(3, "tunes", book) / f"page_{page:04d}" / f"tune_{idx:02d}" / "crop.png"),
                "tune_pdf": str(stage_dir(3, "tunes", book) / f"page_{page:04d}" / f"tune_{idx:02d}" / "crop.pdf"),
                "xml_path": str(out_xml),
                "status": status,
                "engine": best_engine,
                "error": err,
            })

        log.info(
            "validate %s p%d t%d: winner=%s (score=%.1f, %d notes, %s..%s, key=%s)",
            book, page, idx, best_engine, best_score,
            best_info.get("n_notes", 0),
            best_info.get("min_pitch", ""),
            best_info.get("max_pitch", ""),
            final_key,
        )

    return StageResult(ok=True, outputs=outputs)
