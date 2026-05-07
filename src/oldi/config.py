"""Project paths, book aliases, and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHEET_PDF_DIR = PROJECT_ROOT / "sheet_pdf"
DATA_DIR = PROJECT_ROOT / "data"
REVIEW_DIR = PROJECT_ROOT / "review"
VENDOR_DIR = PROJECT_ROOT / "vendor"
CLARITY_DIR = VENDOR_DIR / "Clarity-OMR"
CATALOG_DB = PROJECT_ROOT / "catalog.sqlite"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# Short aliases used in `data/<book>/...` paths. Keep filesystem-safe.
BOOK_ALIASES: dict[str, str] = {
    "dance_music_ireland_oneill.pdf": "oneill_dance",
    "music_of_ireland_oneill.pdf": "oneill_music",
    "popular_selections_from_oneill.pdf": "oneill_popular",
    "waifs_and_strays_oneill.pdf": "oneill_waifs",
    "petrie_collection_ancient_music_of_ireland.pdf": "petrie_ancient",
    "ancient_music_of_ireland.pdf": "bunting_ancient",
    "general_collection_ancient_music_of_ireland.pdf": "bunting_general",
    "old_irish_folk_music_and_songs.pdf": "joyce_old_irish",
    "minstrelsy_of_ireland.pdf": "minstrelsy",
    "balmoral_reel_book.pdf": "balmoral",
    "pocket_companion_ofarrells.pdf": "ofarrell_pocket",
    "irish_dance_folio.pdf": "dance_folio",
    "irish_folk_music_fascinating_hobby.pdf": "folk_hobby",
}


def stage_dir(stage: int, name: str, book: str) -> Path:
    return DATA_DIR / f"{stage:02d}_{name}" / book


def book_alias(pdf_path: Path | str) -> str:
    name = Path(pdf_path).name
    return BOOK_ALIASES.get(name, Path(name).stem)


@dataclass(frozen=True)
class Config:
    project_root: Path = PROJECT_ROOT
    sheet_pdf_dir: Path = SHEET_PDF_DIR
    data_dir: Path = DATA_DIR
    review_dir: Path = REVIEW_DIR
    vendor_dir: Path = VENDOR_DIR
    clarity_dir: Path = CLARITY_DIR
    catalog_db: Path = CATALOG_DB

    # Ingest
    min_render_dpi: int = 300
    max_render_dpi: int = 400

    # Segmentation
    tune_crop_pad_px: int = 40
    music_region_min_aspect: float = 3.0

    # OMR
    clarity_device: str = "cuda"
    clarity_pdf_dpi: int = 400
    clarity_fast: bool = False  # --fast flag only works on CPU; use clarity_beam_width for GPU
    clarity_beam_width: int | None = None  # None → Clarity default 5 (full quality)
    omr_timeout_s: int = 300

    # Validation (tin whistle range, generous to allow later transposition)
    tin_whistle_min_pitch: str = "C4"
    tin_whistle_max_pitch: str = "E6"
    allowed_keys: tuple[str, ...] = field(
        default_factory=lambda: ("D", "G", "A", "C", "Em", "Bm", "Am", "Dm")
    )


CONFIG = Config()
