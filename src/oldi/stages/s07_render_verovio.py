"""Stage 07 — Verovio: render BOTH engines' MusicXML to HTML + SVG for side-by-side comparison."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import CONFIG, Config, TEMPLATES_DIR, stage_dir
from ..errors import StageError, StageResult
from ..manifest import upsert_tune
from ..util.logging import get_logger

log = get_logger()

ENGINES = ("clarity", "audiveris")

_TK = None
_ENV = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"]))


def _toolkit():
    global _TK
    if _TK is None:
        import verovio
        _TK = verovio.toolkit()
        _TK.setOptions(
            {
                "pageHeight": 2970,
                "pageWidth": 2100,
                "scale": 40,
                "adjustPageHeight": True,
                "svgRemoveXlink": True,
                "breaks": "auto",
            }
        )
    return _TK


def _render_one(xml_path: Path, out_html: Path, out_svg: Path, context: dict) -> None:
    tk = _toolkit()
    tk.loadFile(str(xml_path))
    svg = tk.renderToSVG(1)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(svg)
    tmpl = _ENV.get_template("tune.html.j2")
    html = tmpl.render(svg=svg, **context)
    out_html.write_text(html)


def run(
    *,
    book: str,
    page: int,
    cfg: Config = CONFIG,
    force: bool = False,
    con=None,
) -> StageResult:
    xml_dir_validated = stage_dir(6, "validated", book) / f"page_{page:04d}"
    xml_dir_raw = stage_dir(5, "musicxml", book) / f"page_{page:04d}"
    out_dir = stage_dir(7, "html", book) / f"page_{page:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xml_dir_validated.exists():
        raise StageError(f"validated xml dir missing: {xml_dir_validated}", stage="s07_verovio", book=book, page=page)

    outputs: list[Path] = []

    for xml in sorted(xml_dir_validated.glob("tune_*.musicxml")):
        idx = int(xml.stem.split("_")[1])

        engine_marker = xml_dir_raw / f"tune_{idx:02d}.engine"
        winner = engine_marker.read_text().strip() if engine_marker.exists() else ""

        title_json = stage_dir(4, "titles", book) / f"page_{page:04d}" / f"tune_{idx:02d}.json"
        tp = json.loads(title_json.read_text()) if title_json.exists() else {"title": xml.stem}

        crop_png = stage_dir(3, "tunes", book) / f"page_{page:04d}" / f"tune_{idx:02d}" / "crop.png"

        base_ctx = dict(
            book=book, page=page, idx=idx,
            title=tp.get("english") or tp.get("title") or xml.stem,
            title_gaelic=tp.get("gaelic"),
            key="",
            engine=winner,
            musicxml_rel=Path("../../../../06_validated") / book / f"page_{page:04d}" / xml.name,
            pdf_rel=None,
            crop_rel=Path("../../../../03_tunes") / book / f"page_{page:04d}" / f"tune_{idx:02d}" / "crop.png",
        )

        # Render the winner (canonical HTML path).
        out_html = out_dir / f"tune_{idx:02d}.html"
        out_svg = out_dir / f"tune_{idx:02d}.svg"
        if not (out_html.exists() and out_svg.exists()) or force:
            _render_one(xml, out_html, out_svg, base_ctx)
            log.info("verovio %s p%d t%d (winner=%s) → %s", book, page, idx, winner, out_html.name)
        outputs.append(out_html)

        # Also render each engine separately for side-by-side comparison.
        for engine in ENGINES:
            src = xml_dir_raw / f"tune_{idx:02d}.{engine}.musicxml"
            if not src.exists():
                continue
            engine_html = out_dir / f"tune_{idx:02d}.{engine}.html"
            engine_svg = out_dir / f"tune_{idx:02d}.{engine}.svg"
            if engine_html.exists() and engine_svg.exists() and not force:
                continue
            ctx = dict(base_ctx)
            ctx["engine"] = engine
            ctx["musicxml_rel"] = Path("../../../../05_musicxml") / book / f"page_{page:04d}" / src.name
            try:
                _render_one(src, engine_html, engine_svg, ctx)
            except Exception as exc:
                log.warning("verovio %s p%d t%d %s failed: %s", book, page, idx, engine, exc)

        if con is not None:
            upsert_tune(con, {
                "book": book, "page": page, "idx": idx,
                "html_path": str(out_html),
                "clarity_html": str(out_dir / f"tune_{idx:02d}.clarity.html") if (out_dir / f"tune_{idx:02d}.clarity.html").exists() else None,
                "audiveris_html": str(out_dir / f"tune_{idx:02d}.audiveris.html") if (out_dir / f"tune_{idx:02d}.audiveris.html").exists() else None,
            })

    return StageResult(ok=True, outputs=outputs)
