"""`oldi` command-line entry point."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG, book_alias
from .errors import StageError
from .manifest import db, stats
from .util.logging import get_logger

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = get_logger()
console = Console()


# The fixed smoketest set: exercises different engravers, layouts, title placements.
SMOKETEST_PAGES: list[tuple[str, list[int]]] = [
    # Dense O'Neill reels (400 dpi, cleanest baseline)
    ("dance_music_ireland_oneill.pdf", [30, 50]),
    # Petrie airs with lyrics — p40 has tunes, p41 is a chapter-text page (zero-music check)
    ("petrie_collection_ancient_music_of_ireland.pdf", [40, 41]),
    # Multi-tune-per-page O'Neill layout
    ("music_of_ireland_oneill.pdf", [22, 23]),
    # Different engraver / smaller book
    ("waifs_and_strays_oneill.pdf", [15]),
]


def _parse_page_range(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def _route_review(exc: StageError) -> None:
    dest = CONFIG.review_dir / exc.book / f"page_{exc.page:04d}_tune_{exc.tune_idx or 0:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "error.json").write_text(
        '{"stage": %r, "message": %r}' % (exc.stage, str(exc))
    )
    for name, path in exc.artifacts.items():
        if path.exists():
            shutil.copy2(path, dest / f"{name}{path.suffix}")
    log.warning("Routed failure to %s", dest)


@app.command()
def pipeline(
    pdf: Path,
    pages: str = typer.Option("", help="Page range, e.g. '1-5,10,12-14'. Empty = all pages."),
    stages: str = typer.Option("1-9", help="Stages to run."),
    force: bool = typer.Option(False, help="Re-run stages even if outputs exist."),
) -> None:
    """Run the pipeline on a single PDF."""
    if not pdf.exists():
        raise typer.BadParameter(f"PDF not found: {pdf}")

    # Imported lazily so `--help` is fast and doesn't pull GPU deps.
    from .stages import s01_ingest, s02_layout, s03_segment, s04_title_ocr
    from .stages import s05_omr, s06_validate, s07_render_verovio, s08_render_lily, s09_gallery

    stage_fns = {
        1: s01_ingest.run,
        2: s02_layout.run,
        3: s03_segment.run,
        4: s04_title_ocr.run,
        5: s05_omr.run,
        6: s06_validate.run,
        7: s07_render_verovio.run,
        8: s08_render_lily.run,
    }
    active = set(_parse_page_range(stages))

    book = book_alias(pdf)
    con = db()

    # Stage 01 figures out the page count and writes page PNGs.
    if 1 in active:
        ingest_result = s01_ingest.run_book(pdf, cfg=CONFIG, force=force)
        available_pages = ingest_result.meta["pages"]
    else:
        available_pages = sorted(
            int(p.stem.split("_")[1]) for p in (CONFIG.data_dir / "01_pages" / book).glob("page_*.png")
        )

    target_pages = _parse_page_range(pages) if pages else available_pages

    # Stages 2-4 run per page (they're cheap and independent).
    for page in target_pages:
        if 2 in active:
            try:
                s02_layout.run(book=book, page=page, cfg=CONFIG, force=force, con=con)
            except StageError as exc:
                _route_review(exc)
                continue
        if 3 in active:
            try:
                r3 = s03_segment.run(book=book, page=page, cfg=CONFIG, force=force, con=con)
            except StageError as exc:
                _route_review(exc)
                continue
            if r3.meta.get("n_tunes", 0) == 0:
                continue
        if 4 in active:
            try:
                s04_title_ocr.run(book=book, page=page, cfg=CONFIG, force=force, con=con)
            except StageError as exc:
                _route_review(exc)
                continue

    # Stage 5 is batched per book (Clarity models load once).
    if 5 in active:
        try:
            s05_omr.run_book(book, list(target_pages), cfg=CONFIG, force=force, con=con)
        except StageError as exc:
            _route_review(exc)

    # Stages 6-8 run per page.
    for page in target_pages:
        for stage in (6, 7, 8):
            if stage not in active:
                continue
            fn = stage_fns[stage]
            try:
                fn(book=book, page=page, cfg=CONFIG, force=force, con=con)
            except StageError as exc:
                _route_review(exc)
                break
            except Exception as exc:
                log.exception("Unexpected error in stage %d for %s page %d: %s", stage, book, page, exc)
                break

    if 9 in active:
        s09_gallery.run(cfg=CONFIG, con=con)


@app.command()
def smoketest(force: bool = typer.Option(False)) -> None:
    """Run the fixed stress-test loop: a few curated pages across 4 books."""
    for pdf_name, pages in SMOKETEST_PAGES:
        pdf = CONFIG.sheet_pdf_dir / pdf_name
        if not pdf.exists():
            log.warning("Missing source PDF, skipping: %s", pdf)
            continue
        log.info("smoketest: %s pages %s", pdf_name, pages)
        page_spec = ",".join(str(p) for p in pages)
        pipeline(pdf=pdf, pages=page_spec, stages="1-8", force=force)

    # One final gallery build across the whole catalog.
    from .stages import s09_gallery
    s09_gallery.run(cfg=CONFIG, con=db())


@app.command("catalog-stats")
def catalog_stats() -> None:
    """Summarize the tune catalog by book/status/engine."""
    rows = stats(db())
    table = Table(title="Tune catalog")
    for col in ("book", "status", "engine", "n"):
        table.add_column(col)
    for r in rows:
        table.add_row(str(r["book"]), str(r["status"]), str(r["engine"] or ""), str(r["n"]))
    console.print(table)


@app.command()
def gallery() -> None:
    """Rebuild data/09_gallery/index.html from catalog.sqlite."""
    from .stages import s09_gallery
    s09_gallery.run(cfg=CONFIG, con=db())


@app.command()
def serve(port: int = typer.Option(8787), host: str = typer.Option("0.0.0.0")) -> None:
    """Serve the `data/` tree over HTTP so you can browse the gallery live.

    Open http://<machine>:<port>/09_gallery/index.html — page refreshes show
    new tunes as the pipeline writes them.
    """
    import http.server
    import socketserver
    import socket

    handler_cls = http.server.SimpleHTTPRequestHandler
    os.chdir(CONFIG.data_dir)
    # Allow reuse after ctrl-C without waiting for TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer((host, port), handler_cls) as httpd:
        hostname = socket.gethostname()
        console.print(f"[bold green]OldI gallery[/bold green] serving {CONFIG.data_dir}")
        console.print(f"  local:   http://localhost:{port}/09_gallery/index.html")
        console.print(f"  lan:     http://{hostname}:{port}/09_gallery/index.html")
        console.print("[dim]Ctrl-C to stop.[/dim]")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[dim]shutting down[/dim]")


if __name__ == "__main__":
    app()
