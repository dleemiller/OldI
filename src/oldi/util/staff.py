"""Staff-system detection via Clarity-OMR's YOLOv8 weights (vendor/yolo.pt).

One class: `staff` = one 5-line staff system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import CONFIG


@dataclass
class StaffSystem:
    y0: int
    y1: int
    x0: int
    x1: int


_YOLO = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        weights = CONFIG.vendor_dir / "yolo.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO staff-detector weights missing at {weights}. "
                f"Download with: curl -L -o {weights} "
                f"https://huggingface.co/clquwu/Clarity-OMR/resolve/main/yolo.pt"
            )
        from ultralytics import YOLO
        _YOLO = YOLO(str(weights))
    return _YOLO


def detect_staff_systems(
    image_path: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
) -> list[StaffSystem]:
    model = _get_yolo()
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()

    # Dedupe near-duplicate staves (>0.85 IOU)
    staves = sorted(zip(confs, xyxy), key=lambda s: -s[0])
    kept: list[tuple[float, float, float, float]] = []
    for _, (x0, y0, x1, y1) in staves:
        dup = False
        for px0, py0, px1, py1 in kept:
            iw = max(0.0, min(x1, px1) - max(x0, px0))
            ih = max(0.0, min(y1, py1) - max(y0, py0))
            inter = iw * ih
            area_a = max(1e-6, (x1 - x0) * (y1 - y0))
            area_b = max(1e-6, (px1 - px0) * (py1 - py0))
            if inter / max(1e-6, area_a + area_b - inter) > 0.85:
                dup = True
                break
        if not dup:
            kept.append((x0, y0, x1, y1))

    kept.sort(key=lambda b: (b[1] + b[3]) / 2)
    return [StaffSystem(y0=int(y0), y1=int(y1), x0=int(x0), x1=int(x1)) for x0, y0, x1, y1 in kept]
