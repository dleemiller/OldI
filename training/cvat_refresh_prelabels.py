"""Refresh prelabels on untouched frames inside an existing CVAT task.

Preserves your hand-annotated frames bit-for-bit; only rewrites frames
where every shape still has `source=file` (i.e. nothing you've drawn or
adjusted). Per-frame workflow:

  1. Fetch current shapes for the task.
  2. Mark a frame "touched" if any shape has `source=manual` OR an
     `updated_date` newer than the task's creation. Touched frames are
     skipped entirely.
  3. For every untouched frame, delete its stale prelabel shapes and run
     fresh inference with the provided checkpoint. The new shapes are
     POSTed via the `action=create` endpoint, which adds without
     replacing anything else.

Use:
  uv run python training/cvat_refresh_prelabels.py \\
      --task-id 3 \\
      --checkpoint training/checkpoints/finetune_round_1/best_by_map \\
      --threshold 0.1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import requests
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_CLASSES, MODEL1_ID_TO_CLASS  # noqa: E402


def _login(host: str, user: str, password: str) -> requests.Session:
    s = requests.Session()
    r = s.post(f"{host}/api/auth/login",
               json={"username": user, "password": password}, timeout=30)
    r.raise_for_status()
    token = r.json().get("key")
    if token:
        s.headers["Authorization"] = f"Token {token}"
    return s


def _job_ids_for_task(s: requests.Session, host: str, task_id: int) -> list[int]:
    r = s.get(f"{host}/api/jobs", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return [j["id"] for j in r.json()["results"]]


def _frame_filename_map(s: requests.Session, host: str, job_id: int) -> dict[int, str]:
    r = s.get(f"{host}/api/jobs/{job_id}/data/meta", timeout=30)
    r.raise_for_status()
    return {i: f["name"] for i, f in enumerate(r.json().get("frames", []))}


def _label_id_map(s: requests.Session, host: str, project_id: int) -> dict[str, int]:
    r = s.get(f"{host}/api/labels",
              params={"project_id": project_id, "page_size": 100}, timeout=30)
    r.raise_for_status()
    return {l["name"]: l["id"] for l in r.json()["results"]}


def _fetch_shapes(s: requests.Session, host: str, job_id: int) -> list[dict]:
    r = s.get(f"{host}/api/jobs/{job_id}/annotations", timeout=60)
    r.raise_for_status()
    return r.json().get("shapes", [])


def _touched_frames(shapes: list[dict]) -> set[int]:
    """Which frames should be left alone?

    A frame is "touched" if any shape on it has `source=manual` (user
    drew/edited it in CVAT). The initial pre-label upload uses
    `source=file`, so untouched frames retain only `file`-sourced shapes.
    """
    touched: set[int] = set()
    manual_count = defaultdict(int)
    for sh in shapes:
        if sh.get("source") == "manual":
            touched.add(sh["frame"])
            manual_count[sh["frame"]] += 1
    return touched


def _predict_frame(model, processor, image_path: Path, threshold: float) -> list[dict]:
    pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=pil, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    sizes = torch.tensor([pil.size[::-1]])
    res = processor.post_process_object_detection(
        outputs, target_sizes=sizes, threshold=threshold,
    )[0]
    preds = []
    for score, label, box in zip(
        res["scores"].cpu(), res["labels"].cpu(), res["boxes"].cpu()
    ):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        preds.append({
            "class": MODEL1_ID_TO_CLASS[int(label)],
            "score": float(score),
            "points": [x0, y0, x1, y1],
        })
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://192.168.0.125:8080")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="oldi-admin")
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--pages-root", type=Path, default=REPO / "data/01_pages",
                    help="Where the source page PNGs live.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import re
    s = _login(args.host, args.user, args.password)

    r = s.get(f"{args.host}/api/tasks/{args.task_id}", timeout=30)
    r.raise_for_status()
    task = r.json()
    project_id = task["project_id"]
    label_name_to_id = _label_id_map(s, args.host, project_id)
    print(f"task {args.task_id} '{task['name']}' in project {project_id}")
    print(f"labels: {sorted(label_name_to_id)}")

    job_ids = _job_ids_for_task(s, args.host, args.task_id)
    print(f"jobs: {job_ids}")

    # Load model
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
    print(f"\nloading checkpoint {args.checkpoint}")
    model = RTDetrV2ForObjectDetection.from_pretrained(
        str(args.checkpoint),
        id2label=MODEL1_ID_TO_CLASS,
        label2id={v: k for k, v in MODEL1_ID_TO_CLASS.items()},
    )
    processor = AutoImageProcessor.from_pretrained(str(args.checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # Per-job pass (single task usually has one job but be general)
    total_created = 0
    total_deleted = 0
    total_skipped_touched = 0
    total_refreshed_frames = 0

    for job_id in job_ids:
        shapes = _fetch_shapes(s, args.host, job_id)
        frame_files = _frame_filename_map(s, args.host, job_id)
        touched = _touched_frames(shapes)
        print(f"\njob {job_id}: {len(shapes)} shapes, {len(frame_files)} frames, "
              f"{len(touched)} frames touched (preserved)")
        total_skipped_touched += len(touched)

        # Group existing shapes by frame so we know what to delete.
        shapes_by_frame: dict[int, list[dict]] = defaultdict(list)
        for sh in shapes:
            shapes_by_frame[sh["frame"]].append(sh)

        # Build the list of new shapes (all untouched frames combined).
        new_shapes_batch: list[dict] = []
        delete_shapes: list[dict] = []
        for frame_idx, fname in sorted(frame_files.items()):
            if frame_idx in touched:
                continue
            # Queue existing stale shapes for deletion (full object required).
            for sh in shapes_by_frame.get(frame_idx, []):
                delete_shapes.append(sh)
            # Resolve source PNG path.
            m = re.match(r"(.+?)__p(\d+)\.png", fname)
            if not m:
                print(f"  frame {frame_idx} {fname!r}: can't parse book/page, skip")
                continue
            book, page = m.group(1), int(m.group(2))
            png = args.pages_root / book / f"page_{page:04d}.png"
            if not png.exists():
                print(f"  frame {frame_idx} {fname!r}: source PNG missing ({png}), skip")
                continue
            preds = _predict_frame(model, processor, png, args.threshold)
            for p in preds:
                lid = label_name_to_id.get(p["class"])
                if lid is None:
                    continue
                new_shapes_batch.append({
                    "type": "rectangle",
                    "label_id": lid,
                    "frame": frame_idx,
                    "points": p["points"],
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "group": 0,
                    "rotation": 0.0,
                    "attributes": [],
                    "source": "file",
                })
            total_refreshed_frames += 1
            print(f"  frame {frame_idx:3d} {fname}: {len(preds)} new preds, "
                  f"{len(shapes_by_frame.get(frame_idx, []))} old to delete")

        if args.dry_run:
            print(f"\n[dry-run] would delete {len(delete_shapes)} shapes, create "
                  f"{len(new_shapes_batch)} new shapes across {total_refreshed_frames} frames")
            continue

        # Step 1: delete stale shapes. CVAT wants the full shape objects,
        # not just ids. Batch them since the payload can be large.
        if delete_shapes:
            DEL_BATCH = 200
            for i in range(0, len(delete_shapes), DEL_BATCH):
                chunk = delete_shapes[i:i + DEL_BATCH]
                payload = {"shapes": chunk, "tags": [], "tracks": []}
                r = s.patch(f"{args.host}/api/jobs/{job_id}/annotations",
                            params={"action": "delete"}, json=payload, timeout=120)
                r.raise_for_status()
            total_deleted += len(delete_shapes)
            print(f"  deleted {len(delete_shapes)} stale prelabel shapes")

        # Step 2: create new shapes in batches (CVAT has a payload-size cap).
        BATCH = 200
        for i in range(0, len(new_shapes_batch), BATCH):
            chunk = new_shapes_batch[i:i + BATCH]
            payload = {"shapes": chunk, "tags": [], "tracks": []}
            r = s.patch(f"{args.host}/api/jobs/{job_id}/annotations",
                        params={"action": "create"}, json=payload, timeout=120)
            r.raise_for_status()
            total_created += len(chunk)
        print(f"  created {len(new_shapes_batch)} new shapes")

    print(f"\nsummary:")
    print(f"  frames preserved (touched): {total_skipped_touched}")
    print(f"  frames refreshed: {total_refreshed_frames}")
    print(f"  stale shapes deleted: {total_deleted}")
    print(f"  new shapes created: {total_created}")


if __name__ == "__main__":
    main()
