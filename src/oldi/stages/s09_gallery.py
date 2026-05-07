"""Stage 09 — build an HTML gallery index from catalog.sqlite."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import CONFIG, Config, TEMPLATES_DIR, stage_dir
from ..errors import StageResult
from ..util.logging import get_logger

log = get_logger()


def _rel(from_dir: Path, target: str | None) -> str | None:
    if not target:
        return None
    try:
        return os.path.relpath(Path(target), from_dir)
    except ValueError:
        return None


def run(*, cfg: Config = CONFIG, con=None) -> StageResult:
    gallery_dir = cfg.data_dir / "09_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    out_html = gallery_dir / "index.html"

    from ..manifest import db as _db
    d = con or _db()

    rows = list(d["tunes"].rows_where(order_by="book, page, idx"))

    by_book: dict[str, list[dict]] = defaultdict(list)
    n_ok = n_review = n_failed = 0
    for r in rows:
        entry = dict(r)
        # Rewrite absolute filesystem paths to paths relative to gallery_dir for a portable index.
        for key in ("source_crop", "html_path", "pdf_path", "xml_path", "clarity_html", "audiveris_html"):
            entry[key] = _rel(gallery_dir, r.get(key))
        by_book[r["book"]].append(entry)
        status = r.get("status") or "pending"
        if status == "ok":
            n_ok += 1
        elif status == "review":
            n_review += 1
        elif status == "failed":
            n_failed += 1

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"]))
    html = env.get_template("gallery.html.j2").render(
        by_book=by_book,
        total=len(rows),
        n_ok=n_ok,
        n_review=n_review,
        n_failed=n_failed,
    )
    out_html.write_text(html)
    log.info("gallery → %s (%d tunes)", out_html, len(rows))
    return StageResult(ok=True, outputs=[out_html])
