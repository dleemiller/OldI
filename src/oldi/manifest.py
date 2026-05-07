"""SQLite-backed catalog of tunes and their pipeline state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlite_utils

from .config import CONFIG


def db(path: Path | None = None) -> sqlite_utils.Database:
    p = path or CONFIG.catalog_db
    p.parent.mkdir(parents=True, exist_ok=True)
    d = sqlite_utils.Database(p)
    _ensure_schema(d)
    return d


def _ensure_schema(d: sqlite_utils.Database) -> None:
    if "tunes" not in d.table_names():
        d["tunes"].create(
            {
                "book": str,
                "page": int,
                "idx": int,
                "title": str,
                "title_slug": str,
                "key": str,
                "source_crop": str,
                "tune_pdf": str,
                "xml_path": str,
                "html_path": str,
                "pdf_path": str,
                "status": str,   # ok | review | failed | pending
                "engine": str,   # clarity | audiveris
                "error": str,
            },
            pk=("book", "page", "idx"),
        )


def upsert_tune(d: sqlite_utils.Database, row: dict[str, Any]) -> None:
    d["tunes"].upsert(row, pk=("book", "page", "idx"), alter=True)


def stats(d: sqlite_utils.Database) -> list[dict[str, Any]]:
    return list(
        d.query(
            """
            SELECT book, status, engine, COUNT(*) AS n
            FROM tunes
            GROUP BY book, status, engine
            ORDER BY book, status
            """
        )
    )
