"""The 13-class Model 1 taxonomy, shared by the loader, training, and eval.

Class IDs are stable and used directly in the COCO annotation JSONs; don't
reorder without migrating every generated annotation file.

History: initially 8 classes with `ornamental-element` at id 7 (dropped —
Irish trad's note-level ornamentation belongs in Model 2, not page-layout).
Went to 7. Added `staff-header` at id 7. Added `page-number` + `text-block`
at ids 8, 9. Added `inline-lyrics` + `subtitle` at ids 10, 11. The
distinctions matter:
  - `page-number` vs `tune-number` vs `footer`: different kinds of numerics /
    running heads.
  - `inline-lyrics` vs `text-block`: lyrics sit under specific notes on a
    staff; text-block is standalone paragraph prose.
  - `subtitle` vs `tune-title`: O'Neill books routinely print both the
    English title and an Irish Gaelic equivalent; labeling both lets us
    preserve bilingual data.
"""

from __future__ import annotations

# Ordered — index == class id. Only `staff` gets DSv2 pretraining signal;
# the other 11 come from hand annotation in CVAT.
MODEL1_CLASSES: tuple[str, ...] = (
    "staff",
    "measure",
    "tune-title",
    "tempo-marking",
    "tune-number",
    "composer-attribution",
    "footer",
    "staff-header",   # clef + key sig + (first staff only) time sig
    "page-number",    # the numeric page number in the corner / header
    "text-block",     # prose commentary / instructions / paragraphs
    "inline-lyrics",  # syllables printed under staff notes (vocal books)
    "subtitle",       # alt / translated title printed alongside tune-title
    "page-title",     # book/chapter banner like "ANCIENT MUSIC OF IRELAND"
)

MODEL1_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(MODEL1_CLASSES)}
MODEL1_ID_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(MODEL1_CLASSES)}

# DeepScoresV2 → Model 1 remap. Only classes listed here survive the filter;
# everything else in DSv2 (symbols, accidentals, text, ...) is page-level
# irrelevant and gets dropped. DSv2's `staff` class was added in V2
# specifically to cover page regions, which is exactly our signal.
DSV2_TO_MODEL1: dict[str, str] = {
    "staff": "staff",
    # NB: no direct `measure` class in DSv2. Derivable from barline positions
    # + staff y-span, but that's a second pass. Phase A pretrains on `staff`
    # only; `measure` comes from CVAT hand annotation in Phase B.
}
