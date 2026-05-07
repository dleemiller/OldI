"""Replace a CVAT task's annotations with a COCO 1.0 file.

Use to re-import a previous export (e.g. after migrating CVAT instances)
or to swap the prelabels uploaded by `cvat_setup.py` with hand-corrected
ground truth.

Example:
  uv run python training/cvat_import_annotations.py \\
      --task-id 1 \\
      --coco data/annotations/exports/20260422-215027/\\
task-3-batch_004___all_classes__3e-5_pretrain/annotations/instances_default.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.cvat_setup import _login, _upload_annotations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://192.168.0.125:8080")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="oldi-admin")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--coco", type=Path, required=True,
                    help="Path to COCO 1.0 annotations JSON.")
    args = ap.parse_args()

    assert args.coco.exists(), f"missing {args.coco}"
    print(f"host:    {args.host}")
    print(f"task:    {args.task_id}")
    print(f"coco:    {args.coco}")

    s = _login(args.host, args.user, args.password)
    _upload_annotations(s, args.host, args.task_id, args.coco)
    print(f"\ndone — open {args.host}/tasks/{args.task_id} to review")


if __name__ == "__main__":
    main()
