"""Stage 05 — dual-engine OMR: Clarity + Audiveris.

We run both engines on every tune and let stage 06 pick the one whose output
validates cleaner. Producing both is costly (Audiveris ~30-60s per tune) but
gives us a much better chance of an accurate MusicXML for hard pages.

Artifacts per tune:
    data/05_musicxml/<book>/page_NNNN/
        tune_NN.clarity.musicxml       # from Clarity-OMR (primary)
        tune_NN.audiveris.musicxml     # from Audiveris (secondary)
        tune_NN.engine                 # winner: "clarity" | "audiveris"
        tune_NN.musicxml               # symlink/copy of the winner (set by s06)

Clarity path: batched per book (see earlier comment). Audiveris path: one
podman run per tune PDF — slow but parallelizable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pypdf

from ..config import CONFIG, Config, stage_dir
from ..errors import StageError, StageResult
from ..util.image import png_to_single_page_pdf
from ..util.logging import get_logger

log = get_logger()

AUDIVERIS_IMAGE = "oldi-audiveris"
AUDIVERIS_PARALLEL = 4  # concurrent `podman run` invocations


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _clarity_sys_path(cfg: Config) -> None:
    if str(cfg.clarity_dir) not in sys.path:
        sys.path.insert(0, str(cfg.clarity_dir))


def _page_pdf_dir(cfg: Config, book: str) -> Path:
    return cfg.data_dir / "05_musicxml" / book / "_page_pdfs"


def _book_work_dir(cfg: Config, book: str) -> Path:
    return cfg.data_dir / "05_musicxml" / book / "_clarity_work"


def _ensure_page_pdf(page_png: Path, out_pdf: Path) -> Path:
    if not out_pdf.exists():
        png_to_single_page_pdf(page_png, out_pdf, dpi=400)
    return out_pdf


def _build_book_pdf(book: str, pages: list[int], cfg: Config) -> tuple[Path, list[int]]:
    """Merge per-page PDFs into one book PDF for Clarity."""
    page_dir = _page_pdf_dir(cfg, book)
    page_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[int, Path]] = []
    for p in sorted(pages):
        png = stage_dir(1, "pages", book) / f"page_{p:04d}.png"
        if not png.exists():
            continue
        pdf = page_dir / f"page_{p:04d}.pdf"
        _ensure_page_pdf(png, pdf)
        entries.append((p, pdf))
    if not entries:
        raise StageError("no pages found to OMR", stage="s05_omr", book=book, page=0)

    book_pdf = page_dir / "_book.pdf"
    writer = pypdf.PdfWriter()
    for _, pdf in entries:
        for pg in pypdf.PdfReader(str(pdf)).pages:
            writer.add_page(pg)
    with open(book_pdf, "wb") as f:
        writer.write(f)
    return book_pdf, [p for p, _ in entries]


# ---------------------------------------------------------------------------
# Clarity path (batched per book)
# ---------------------------------------------------------------------------


def _run_clarity_on_book(book_pdf: Path, work_dir: Path, cfg: Config) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    dummy_out = work_dir / "book.musicxml"
    beam = cfg.clarity_beam_width if cfg.clarity_beam_width else 5

    cmd = [
        sys.executable,
        "-m", "src.pdf_to_musicxml",
        "--pdf", str(book_pdf),
        "--output-musicxml", str(dummy_out),
        "--project-root", str(cfg.clarity_dir),
        "--work-dir", str(work_dir),
        "--weights", str(cfg.clarity_dir / "info" / "yolo.pt"),
        "--stage-b-checkpoint", str(cfg.clarity_dir / "info" / "model.safetensors"),
        "--beam-width", str(beam),
        "--image-height", "250",
        "--image-max-width", "2500",
        "--length-penalty-alpha", "0.4",
        "--pdf-dpi", str(cfg.clarity_pdf_dpi),
        "--stage-b-device", cfg.clarity_device,
        "--enforce-full-width-crops",
        "--full-width-left-page-edge",
        "--full-width-right-page-edge",
    ]
    log.info("clarity on %s (pages → single book PDF)", book_pdf.name)
    proc = subprocess.run(
        cmd, cwd=str(cfg.clarity_dir), capture_output=True, text=True,
        timeout=cfg.omr_timeout_s * 10,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"clarity failed (rc={proc.returncode}):\n"
            f"stdout={proc.stdout[-800:]}\nstderr={proc.stderr[-800:]}"
        )
    preds = work_dir / "stage_b_predictions.jsonl"
    if not preds.exists():
        raise RuntimeError(f"clarity succeeded but predictions missing: {preds}")
    return preds


def _load_predictions(preds_path: Path) -> list[dict]:
    return [json.loads(l) for l in preds_path.read_text().splitlines() if l.strip()]


def _pred_y_center(p: dict) -> float:
    b = p["bbox"]
    return (b["y_min"] + b["y_max"]) / 2.0


def _clarity_assemble_tune(
    tune_preds: list[dict],
    tune_dir: Path,
    out_xml: Path,
    tune_idx: int,
) -> None:
    from src.cli import run_assemble, run_export  # type: ignore

    renum: list[dict] = []
    for i, p in enumerate(sorted(tune_preds, key=_pred_y_center)):
        q = dict(p)
        q["page_index"] = 0
        q["system_index"] = i
        q["staff_index"] = 0
        q["sample_id"] = f"tune_{tune_idx:02d}:staff_{i:02d}"
        renum.append(q)

    preds_path = tune_dir / "stage_b_subset.jsonl"
    preds_path.write_text("\n".join(json.dumps(r) for r in renum) + "\n")
    assembly_path = tune_dir / "assembled_score.json"
    run_assemble(argparse.Namespace(staff_predictions=preds_path, output_assembly=assembly_path))
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    run_export(argparse.Namespace(assembly_manifest=assembly_path, output_musicxml=out_xml))


_book_predictions_cache: dict[tuple[str, tuple[int, ...]], tuple[Path, list[int]]] = {}


def _get_book_predictions(
    book: str, pages: list[int], cfg: Config, force: bool = False
) -> tuple[Path, list[int]]:
    key = (book, tuple(sorted(pages)))
    if key in _book_predictions_cache and not force:
        return _book_predictions_cache[key]

    work_dir = _book_work_dir(cfg, book)
    preds = work_dir / "stage_b_predictions.jsonl"
    page_list_marker = work_dir / "page_list.json"

    cached_pages: list[int] = []
    if page_list_marker.exists():
        try:
            cached_pages = json.loads(page_list_marker.read_text())
        except Exception:
            cached_pages = []

    need_rerun = force or (not preds.exists()) or (sorted(cached_pages) != sorted(pages))
    if need_rerun:
        book_pdf, page_list = _build_book_pdf(book, pages, cfg)
        _run_clarity_on_book(book_pdf, work_dir, cfg)
        page_list_marker.write_text(json.dumps(page_list))
    else:
        page_list = cached_pages

    _book_predictions_cache[key] = (preds, page_list)
    return preds, page_list


# ---------------------------------------------------------------------------
# Audiveris path (per-tune container run)
# ---------------------------------------------------------------------------


def _audiveris_image_available() -> bool:
    p = subprocess.run(
        ["podman", "image", "exists", AUDIVERIS_IMAGE],
        capture_output=True,
    )
    return p.returncode == 0


def _run_audiveris_one(tune_pdf: Path, out_xml: Path, cfg: Config) -> None:
    """Run Audiveris on a single tune PDF in the oldi-audiveris container.

    Audiveris writes MusicXML (compressed .mxl or .xml) into its output dir;
    we pick the first one and rename to out_xml.
    """
    tmp_out = out_xml.parent / f"{out_xml.stem}.audiveris_work"
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "podman", "run", "--rm",
        "-v", f"{tune_pdf.parent}:/in:Z",
        "-v", f"{tmp_out}:/out:Z",
        AUDIVERIS_IMAGE,
        "-batch", "-export",
        "-output", "/out",
        f"/in/{tune_pdf.name}",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=cfg.omr_timeout_s * 2,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"audiveris rc={proc.returncode}: {proc.stderr[-400:]}"
        )

    produced = (
        sorted(tmp_out.rglob("*.mxl"))
        + sorted(tmp_out.rglob("*.musicxml"))
        + sorted(tmp_out.rglob("*.xml"))
    )
    if not produced:
        raise RuntimeError("audiveris produced no MusicXML")

    # If compressed (.mxl), unzip to raw .musicxml; otherwise just copy.
    first = produced[0]
    if first.suffix == ".mxl":
        import zipfile
        with zipfile.ZipFile(first) as z:
            inner = [n for n in z.namelist() if n.endswith(".xml") and "META-INF" not in n]
            if not inner:
                raise RuntimeError("audiveris .mxl has no xml")
            with z.open(inner[0]) as f:
                out_xml.write_bytes(f.read())
    else:
        shutil.copy2(first, out_xml)


def _run_audiveris_for_tunes(
    book: str,
    tunes: list[tuple[int, int, Path, Path]],  # (page, idx, tune_pdf, out_xml)
    cfg: Config,
) -> None:
    """Run Audiveris in parallel across tunes. Errors are logged, not raised."""
    if not _audiveris_image_available():
        log.warning("audiveris image '%s' not built; skipping Audiveris path", AUDIVERIS_IMAGE)
        return

    def _work(page: int, idx: int, tune_pdf: Path, out_xml: Path) -> tuple[int, int, bool, str]:
        try:
            _run_audiveris_one(tune_pdf, out_xml, cfg)
            return page, idx, True, ""
        except Exception as exc:
            return page, idx, False, str(exc)

    todo = [t for t in tunes if not t[3].exists()]
    if not todo:
        return
    log.info("audiveris on %s: %d tunes (parallel=%d)", book, len(todo), AUDIVERIS_PARALLEL)
    with ThreadPoolExecutor(max_workers=AUDIVERIS_PARALLEL) as ex:
        futures = [ex.submit(_work, *t) for t in todo]
        for fut in as_completed(futures):
            page, idx, ok, err = fut.result()
            if ok:
                log.info("audiveris OK %s p%d t%d", book, page, idx)
            else:
                log.warning("audiveris FAIL %s p%d t%d: %s", book, page, idx, err[:160])


# ---------------------------------------------------------------------------
# Stage entry
# ---------------------------------------------------------------------------


def run_book(
    book: str, pages: list[int], *, cfg: Config = CONFIG, force: bool = False, con=None
) -> StageResult:
    """Run Clarity (batched) AND Audiveris (per-tune, parallel) on all pages."""
    _clarity_sys_path(cfg)
    preds_path, page_list = _get_book_predictions(book, pages, cfg, force=force)
    page_index_to_page = {i: p for i, p in enumerate(page_list)}
    predictions = _load_predictions(preds_path)

    outputs: list[Path] = []
    audiveris_todo: list[tuple[int, int, Path, Path]] = []

    for p_idx, page in page_index_to_page.items():
        tunes_dir = stage_dir(3, "tunes", book) / f"page_{page:04d}"
        out_dir = stage_dir(5, "musicxml", book) / f"page_{page:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not tunes_dir.exists() or (tunes_dir / ".no_music").exists():
            continue

        page_preds = [pr for pr in predictions if pr.get("page_index") == p_idx]
        if not page_preds:
            log.warning("no Clarity predictions for %s p%d", book, page)

        for tune_dir in sorted(tunes_dir.glob("tune_*")):
            idx = int(tune_dir.name.split("_")[1])
            tune_pdf = tune_dir / "crop.pdf"
            clarity_xml = out_dir / f"tune_{idx:02d}.clarity.musicxml"
            audiveris_xml = out_dir / f"tune_{idx:02d}.audiveris.musicxml"
            meta = json.loads((tune_dir / "meta.json").read_text())
            staff_bboxes = meta.get("staff_bboxes") or []

            # ---------- Clarity ----------
            if staff_bboxes and (force or not clarity_xml.exists()):
                y_lo = min(b[1] for b in staff_bboxes) - 15
                y_hi = max(b[3] for b in staff_bboxes) + 15
                tune_preds = [pr for pr in page_preds if y_lo <= _pred_y_center(pr) <= y_hi]
                if tune_preds:
                    try:
                        _clarity_assemble_tune(tune_preds, tune_dir, clarity_xml, tune_idx=idx)
                        outputs.append(clarity_xml)
                        log.info("clarity %s p%d t%d: %d staves → %s",
                                 book, page, idx, len(tune_preds), clarity_xml.name)
                    except Exception as exc:
                        log.warning("clarity assemble failed %s p%d t%d: %s",
                                    book, page, idx, exc)
                else:
                    log.warning("no Clarity preds inside tune bbox %s p%d t%d", book, page, idx)
            elif clarity_xml.exists():
                outputs.append(clarity_xml)

            # ---------- Audiveris (queue) ----------
            if tune_pdf.exists() and (force or not audiveris_xml.exists()):
                audiveris_todo.append((page, idx, tune_pdf, audiveris_xml))

    # Run all Audiveris jobs for this book in parallel.
    _run_audiveris_for_tunes(book, audiveris_todo, cfg)
    outputs.extend(t[3] for t in audiveris_todo if t[3].exists())

    return StageResult(ok=True, outputs=outputs)


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    return run_book(book, [page], cfg=cfg, force=force, con=con)
