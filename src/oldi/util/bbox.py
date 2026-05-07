"""Bounding-box helpers. BBox convention: (x0, y0, x1, y1) in pixels."""

from __future__ import annotations

BBox = tuple[int, int, int, int]


def pad(b: BBox, px: int, w: int, h: int) -> BBox:
    x0, y0, x1, y1 = b
    return (max(0, x0 - px), max(0, y0 - px), min(w, x1 + px), min(h, y1 + px))


def union(boxes: list[BBox]) -> BBox:
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[2] for b in boxes)
    ys1 = max(b[3] for b in boxes)
    return (xs0, ys0, xs1, ys1)


def aspect(b: BBox) -> float:
    w = b[2] - b[0]
    h = b[3] - b[1]
    return (w / h) if h > 0 else 0.0


def y_center(b: BBox) -> int:
    return (b[1] + b[3]) // 2
