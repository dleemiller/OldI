"""Shared stage error + result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class StageError(Exception):
    """Raised by a stage when it cannot produce its output.

    Caught by the pipeline driver, which copies relevant artifacts to review/.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        book: str,
        page: int,
        tune_idx: int | None = None,
        artifacts: dict[str, Path] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.book = book
        self.page = page
        self.tune_idx = tune_idx
        self.artifacts = artifacts or {}


@dataclass
class StageResult:
    ok: bool
    outputs: list[Path] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
