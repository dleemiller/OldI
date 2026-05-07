"""Stage 08 — MusicXML → PDF via LilyPond inside a Podman container.

Requires the `oldi-lilypond` image built from src/oldi/containers/lilypond.Containerfile.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import CONFIG, Config, stage_dir
from ..errors import StageError, StageResult
from ..manifest import upsert_tune
from ..util.logging import get_logger

log = get_logger()

IMAGE = "oldi-lilypond"


def _lilypond_one(xml_abs: Path, pdf_abs: Path, project_root: Path) -> None:
    # Bind-mount the project root so the container can read the source XML and
    # write the output PDF without us copying files around.
    xml_rel = xml_abs.relative_to(project_root)
    pdf_rel = pdf_abs.relative_to(project_root)
    cmd = [
        "podman", "run", "--rm",
        "-v", f"{project_root}:/work:Z",
        IMAGE,
        f"/work/{xml_rel}",
        f"/work/{pdf_rel}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"lilypond container failed (rc={proc.returncode}):\nstdout={proc.stdout[-500:]}\nstderr={proc.stderr[-500:]}"
        )
    if not pdf_abs.exists():
        raise RuntimeError(f"lilypond succeeded but PDF not found at {pdf_abs}")


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    xml_dir = stage_dir(6, "validated", book) / f"page_{page:04d}"
    out_dir = stage_dir(8, "pdf", book) / f"page_{page:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xml_dir.exists():
        raise StageError(f"validated xml dir missing: {xml_dir}", stage="s08_lily", book=book, page=page)

    outputs: list[Path] = []
    for xml in sorted(xml_dir.glob("tune_*.musicxml")):
        idx = int(xml.stem.split("_")[1])
        out_pdf = out_dir / f"tune_{idx:02d}.pdf"
        if out_pdf.exists() and not force:
            outputs.append(out_pdf)
            continue
        try:
            _lilypond_one(xml.resolve(), out_pdf.resolve(), cfg.project_root)
        except Exception as exc:
            # LilyPond failures are non-fatal for the overall pipeline (Verovio
            # HTML is still usable). Record and continue.
            log.warning("lilypond failed %s p%d t%d: %s", book, page, idx, exc)
            continue

        outputs.append(out_pdf)
        if con is not None:
            upsert_tune(con, {"book": book, "page": page, "idx": idx, "pdf_path": str(out_pdf)})
        log.info("lilypond %s p%d t%d → %s", book, page, idx, out_pdf.name)

    return StageResult(ok=True, outputs=outputs)
