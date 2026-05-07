# Model 1 — Page Layout Detector

Training code for the first of four planned OMR models: a page-level layout
detector (RT-DETRv2 + ResNet-101, heron-initialized) that localises
`staff`, `measure`, `tune-title`, `tempo-marking`, `tune-number`,
`composer-attribution`, `footer`, `staff-header`, `page-number`,
`text-block`, `inline-lyrics`, `subtitle`, and `page-title` on 19th-century
Irish tune-book pages. Note-level ornamentation
(cuts, taps, rolls, crans, slides) is a Model 2 concern, not a page-layout
class.

## Directory map

```
training/
  classes.py             # 8-class taxonomy (order is stable — don't renumber)
  data/
    deepscoresv2.py      # DSv2 loader + remap to 8-class COCO
    augment.py           # 7 augraphy "modes" keyed to real defect profiles
  calibrate_augment.py   # side-by-side grid: augmented DSv2 vs real scans
  # (Training / eval / inference scripts — written incrementally; see plan.)
  eval_out/              # generated calibration grid + inference overlays
  checkpoints/           # .gitignored; model_v{N}.safetensors
configs/
  layout_pretrain.yaml   # Phase A (DSv2)
  layout_finetune.yaml   # Phase B (real annotations)
```

## Prepared data

`data/deepscoresv2/model1_coco_{train,test}.json` — DeepScoresV2 dense
subset remapped to our 8-class schema. Only the `staff` class has
annotations (DSv2 has no measure/title/etc. classes). Train set has 1,362
images / 14,463 staff bboxes; test set has 352 images / 3,849.

To regenerate:
```python
from pathlib import Path
from training.data.deepscoresv2 import DSv2Dataset, remap_to_coco
root = Path("data/deepscoresv2/ds2_dense")
dsv2 = DSv2Dataset.load(root / "deepscores_train.json")
remap_to_coco(dsv2, images_root=root / "images",
              out_path=Path("data/deepscoresv2/model1_coco_train.json"))
```

## Known nuisance: libpng CRC warnings on DSv2 scans

A handful of DSv2 PNGs have malformed IDAT CRCs. libpng recovers and
decodes cleanly — no impact on training — but it writes C-level warnings
to stderr that Python can't fully intercept (tried per-call redirect,
DataLoader worker_init_fn, and re-encoding, none caught every case). Filter
at launch:

```bash
NUMBA_DISABLE_JIT=1 uv run python training/train_layout.py --phase pretrain \
  2> >(grep -v -E 'libpng|IDAT|CRC error|Read Error' >&2)
```

Or strip them when tailing the log file.

## Augmentation modes

Seven named augraphy modes, each keyed to a real-world defect profile we
audited in `sheet_pdf/`:

| Mode | Motivating book | Signature |
|---|---|---|
| `oneill_dance_like` | `dance_music_ireland_oneill.pdf` | Dense heavy-ink letterpress, white paper, speckle |
| `oneill_music_like` | `music_of_ireland_oneill.pdf` | Thin engraving + JPEG compression |
| `petrie_like` | `petrie_collection_ancient_music_of_ireland.pdf` | Uniform aged grey paper + letterpress |
| `oneill_waifs_like` | `waifs_and_strays_oneill.pdf` | Near-pristine modern reprint |
| `popsel_like` | `popular_selections_from_oneill.pdf` | Low-DPI + grey + bleed-through from verso |
| `minstrel_like` | `the_irish_minstrel.pdf` | Blotchy grey paper + heavy blurry ink |
| `graves_like` | `irish_song_book_graves.pdf` | Camera-captured, black surround, facing-page peek (visual-only — excluded from pretrain sampling because it warps content geometry) |

Before kicking off Phase A training, **run the calibration gate**:
```
uv run python training/calibrate_augment.py
```
Review `training/eval_out/calibration.png`. Each row should show the
real-book crop at left being visually similar to the augmented DSv2 crops at
right. Tune mode parameters in `training/data/augment.py` and re-render
until every row reads as "synthetic ≈ real".

## CVAT hand-annotation workflow

The six tune-book-specific classes (`tune-title`, `tempo-marking`,
`tune-number`, `composer-attribution`, `footer`, `ornamental-element`) and
`measure` are not in DSv2 — they have to come from hand annotation. CVAT
is the target tool.

### CVAT project label list

Create a project with these thirteen labels, exact names required:

```json
[
  {"name": "staff",                 "color": "#e74c3c"},
  {"name": "measure",               "color": "#3498db"},
  {"name": "tune-title",            "color": "#2ecc71"},
  {"name": "tempo-marking",         "color": "#f1c40f"},
  {"name": "tune-number",           "color": "#9b59b6"},
  {"name": "composer-attribution",  "color": "#1abc9c"},
  {"name": "footer",                "color": "#95a5a6"},
  {"name": "staff-header",          "color": "#e67e22"},
  {"name": "page-number",           "color": "#8e44ad"},
  {"name": "text-block",            "color": "#34495e"},
  {"name": "inline-lyrics",         "color": "#ff69b4"},
  {"name": "subtitle",              "color": "#6a5acd"},
  {"name": "page-title",            "color": "#27ae60"}
]
```

### Annotation guidance per class

- **`staff`** — tight rectangle around the 5 staff lines **excluding** the
  clef, key, and time signature area. Enclose the barlines at both ends.
- **`staff-header`** — the clef + key signature + (first staff of a tune
  only) time signature block at the very start of each staff. Tight left-
  to-right bbox starting just before the clef, ending at the right edge of
  the last signature glyph (before the first barline/note). Vertically
  matches the staff. On second-and-later staves of the same tune, this is
  just clef + key signature (no time sig).
- **`measure`** — one box per measure, top/bottom = staff top/bottom plus
  small stem margin (~10 px), left = right edge of `staff-header` for
  measure 1, right edge of previous barline for subsequent measures; right
  = that measure's barline. One tune, one staff, multiple measures.
- **`tune-title`** — the primary printed title of a tune. Usually the
  English form when both English and Irish are present. Bbox just the
  characters.
- **`subtitle`** — alternate title printed alongside / below the primary
  title. Most common form: the Irish Gaelic name of a tune (e.g.
  `an beata! moR go tuiTneac` underneath `THE HIGHWAY TO LIMERICK`).
  Other accepted forms: "A song of the mountains", or any secondary
  descriptive title line.
- **`tempo-marking`** — "Allegro", "Reel", "Jig" etc. when they appear
  separate from the title. If the subtitle is the tempo, label it here.
- **`tune-number`** — sequence numbers stamped *with a staff* (leftward of
  the first measure): `42.`, `No. 173`, `No 267`, `643.`. Specifically the
  in-body tune index; NOT page numbers.
- **`composer-attribution`** — "— Carolan", "Trad." Distinct from tune title.
- **`footer`** — running section head on a page: `DOUBLE JIGS`, `HORNPIPES`,
  `REELS` at the top or bottom spanning a group of tunes. Publisher /
  copyright / signature lines at the very bottom of a page also fit here.
  Does NOT include page numbers (separate class) or book-title banners
  (use `page-title`).
- **`page-title`** — book/chapter banner appearing near the top of a page,
  typically centered: `ANCIENT MUSIC OF IRELAND.`, `O'NEILL'S IRISH MUSIC`,
  `THE MUSIC OF IRELAND.`. Tight bbox around the characters. Distinct from
  `footer` (section head like `DOUBLE JIGS`) and `tune-title` (individual
  tune's name).
- **`page-number`** — the actual numeric page number, typically top/bottom
  corner: `116`, `28`, `p. 42`. Tight bbox around just the numeric text.
- **`text-block`** — prose paragraph content: commentary, instructions,
  historical notes. One bbox per coherent paragraph. Used downstream for
  OCR — draw a generous box covering the full paragraph without including
  neighbouring staves, titles, or inline-lyrics.
- **`inline-lyrics`** — syllables printed **under** individual staff notes
  in vocal/song books (the_irish_minstrel, Joyce, minstrelsy,
  irish_song_book_graves). Bbox spans the lyric line tightly — width
  matches the staff, height is the lyric text height (~20-30 px). Distinct
  from text-block because these are per-note aligned sung text, not prose.

### Starter page list

Target 30–50 annotated pages for Phase B, book-stratified. Good starter
batch (already preprocessed in `data/01_pages/`):

- `oneill_dance` pages 10, 30, 50, 70, 90, 110, 150
- `oneill_music` pages 22, 45, 70, 100, 150, 200
- `petrie_ancient` pages 40, 80, 120, 160
- `oneill_waifs` pages 15, 30, 60, 90
- `the_irish_minstrel` pages 30, 60, 90, 120
- `popular_selections_from_oneill` pages 10, 15, 20, 25

### Import CVAT export → training

CVAT's COCO 1.0 export drops `instances_default.json` alongside the images.
A future `training/data/annotations.py` module will read that directly,
strip CVAT's metadata fields, and merge into our 8-class COCO convention.
For now: just export, keep the JSON, don't hand-edit.

### Pseudo-labelling

Once the first ~15 pages are done and there's a baseline finetuned model,
use `training/infer_layout.py` (forthcoming) to pre-populate bboxes for the
next batch. CVAT accepts COCO-format pre-labels — upload them before the
annotator opens the task; they appear as editable boxes they can tweak
rather than draw from scratch.
