"""Create a CVAT project + task and upload prelabels.

Assumes a CVAT stack is running at `--host` (default http://localhost:8080)
with an admin user already created (see README).

Creates:
  - Project "OldI Model 1 — page layout" with the 7-class Model 1 taxonomy
  - Task "<batch_name>" inside that project, populated with the batch's
    images
  - Uploads the batch's annotations.json as COCO 1.0 pre-labels

Use:
  uv run python training/cvat_setup.py \\
      --batch data/prelabel_batches/batch_003_staff_only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.classes import MODEL1_CLASSES  # noqa: E402

# Distinct colours per class for CVAT UI
LABEL_COLORS = {
    "staff":                "#e74c3c",
    "measure":              "#3498db",
    "tune-title":           "#2ecc71",
    "tempo-marking":        "#f1c40f",
    "tune-number":          "#9b59b6",
    "composer-attribution": "#1abc9c",
    "footer":               "#95a5a6",
    "staff-header":         "#e67e22",
    "page-number":          "#8e44ad",
    "text-block":           "#34495e",
    "inline-lyrics":        "#ff69b4",
    "subtitle":             "#6a5acd",
    "page-title":           "#27ae60",
}


def _login(host: str, user: str, password: str) -> requests.Session:
    s = requests.Session()
    r = s.post(f"{host}/api/auth/login",
               json={"username": user, "password": password}, timeout=30)
    r.raise_for_status()
    token = r.json().get("key")
    if token:
        s.headers["Authorization"] = f"Token {token}"
    return s


def _get_or_create_project(s: requests.Session, host: str, name: str) -> int:
    r = s.get(f"{host}/api/projects", params={"name": name}, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    for p in results:
        if p["name"] == name:
            print(f"reusing project {name!r} (id={p['id']})")
            return p["id"]
    labels = [
        {"name": cls, "color": LABEL_COLORS[cls], "attributes": []}
        for cls in MODEL1_CLASSES
    ]
    r = s.post(f"{host}/api/projects",
               json={"name": name, "labels": labels},
               timeout=30)
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"created project {name!r} (id={pid}) with {len(labels)} labels")
    return pid


def _create_task(s: requests.Session, host: str, project_id: int,
                 name: str, images: list[Path],
                 segment_size: int | None = None) -> int:
    body: dict[str, Any] = {"name": name, "project_id": project_id}
    if segment_size:
        body["segment_size"] = segment_size
    r = s.post(
        f"{host}/api/tasks",
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    task_id = r.json()["id"]
    print(f"created task {name!r} (id={task_id})")

    # Upload images
    files = []
    for i, img in enumerate(images):
        files.append(("client_files[" + str(i) + "]", (img.name, open(img, "rb"), "image/png")))
    data = {
        "image_quality": 95,
        "use_zip_chunks": True,
        "use_cache": True,
        "sorting_method": "natural",
    }
    r = s.post(f"{host}/api/tasks/{task_id}/data", data=data, files=files,
               timeout=600)
    for _, (_, f, _) in files:
        f.close()
    r.raise_for_status()
    print(f"  uploaded {len(images)} images")

    # Poll until task data is ready (server creates frames, manifests, etc.)
    for _ in range(240):
        r = s.get(f"{host}/api/tasks/{task_id}/status", timeout=30)
        r.raise_for_status()
        state = r.json()["state"]
        if state == "Finished":
            break
        if state == "Failed":
            raise RuntimeError(f"task data creation failed: {r.json()}")
        time.sleep(2)
    else:
        raise TimeoutError("task data never became Finished")
    print("  task data ready")
    return task_id


def _upload_annotations(s: requests.Session, host: str, task_id: int,
                         coco_path: Path) -> None:
    """Upload COCO 1.0 annotations to a task via the two-step API.

    CVAT's COCO importer expects 1-indexed `category_id` values and matches
    categories by `name` (not id). Our on-disk COCO uses 0-indexed ids
    (matching Model 1 class ids in `training/classes.py`). We rewrite to a
    temp file with every id shifted +1 before upload.
    """
    import tempfile

    raw = json.loads(coco_path.read_text())
    raw["categories"] = [{**c, "id": c["id"] + 1} for c in raw["categories"]]
    raw["annotations"] = [{**a, "category_id": a["category_id"] + 1}
                          for a in raw["annotations"]]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                      mode="w") as tmp:
        json.dump(raw, tmp)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        files = {"annotation_file": (coco_path.name, f, "application/json")}
        r = s.post(
            f"{host}/api/tasks/{task_id}/annotations",
            params={"format": "COCO 1.0"},
            files=files,
            timeout=120,
        )
    if r.status_code not in (201, 202):
        raise RuntimeError(f"annotation upload failed: {r.status_code} {r.text[:500]}")
    # Poll for completion if async
    rq_id = r.json().get("rq_id")
    if rq_id:
        for _ in range(60):
            rr = s.get(f"{host}/api/requests/{rq_id}", timeout=30)
            rr.raise_for_status()
            status = rr.json().get("status")
            if status == "finished":
                break
            if status == "failed":
                raise RuntimeError(f"annotation import failed: {rr.json()}")
            time.sleep(2)
    print(f"  uploaded annotations from {coco_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://192.168.0.125:8080",
                    help="CVAT base URL. Default matches the CVAT_HOST in "
                         "~/cvat/cvat/.env (LAN IP). Use localhost only if "
                         "CVAT was started without CVAT_HOST override.")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="oldi-admin")
    ap.add_argument("--project-name", default="OldI Model 1 — page layout")
    ap.add_argument("--batch", type=Path, required=True,
                    help="Batch dir containing images/ and annotations.json")
    ap.add_argument("--task-name", default=None,
                    help="Override task name (defaults to batch dir name).")
    ap.add_argument("--segment-size", type=int, default=None,
                    help="Split the task into jobs of this many frames. "
                         "Omit for the CVAT default (one job per task).")
    args = ap.parse_args()

    images_dir = args.batch / "images"
    coco_path = args.batch / "annotations.json"
    assert images_dir.exists(), f"missing {images_dir}"
    assert coco_path.exists(), f"missing {coco_path}"

    images = sorted(images_dir.iterdir())
    task_name = args.task_name or args.batch.name
    print(f"host:     {args.host}")
    print(f"project:  {args.project_name}")
    print(f"task:     {task_name}")
    print(f"images:   {len(images)}")

    s = _login(args.host, args.user, args.password)
    pid = _get_or_create_project(s, args.host, args.project_name)
    tid = _create_task(s, args.host, pid, task_name, images,
                       segment_size=args.segment_size)
    _upload_annotations(s, args.host, tid, coco_path)
    print(f"\ndone — open {args.host}/tasks/{tid} to annotate")


if __name__ == "__main__":
    main()
