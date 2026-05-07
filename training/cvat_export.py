"""Export all CVAT task annotations as COCO 1.0 dumps.

Pulls every task in every project, zips the COCO export, unpacks it to
`data/annotations/exports/<timestamp>/task-<id>-<name>/`. Also writes a
manifest with task metadata so we can trace which annotations came from
which CVAT task version.

Run regularly (or before any destructive operation — recreating tasks,
re-uploading pre-labels, etc.). Git-committing the `exports/` dir
preserves annotation history.

Use:
  uv run python training/cvat_export.py
  uv run python training/cvat_export.py --out data/annotations/exports/before_round_1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent


def _login(host: str, user: str, password: str) -> requests.Session:
    s = requests.Session()
    r = s.post(f"{host}/api/auth/login",
               json={"username": user, "password": password}, timeout=30)
    r.raise_for_status()
    token = r.json().get("key")
    if token:
        s.headers["Authorization"] = f"Token {token}"
    return s


def _export_task(s: requests.Session, host: str, task_id: int,
                  out_dir: Path) -> Path | None:
    """Trigger + download a COCO 1.0 dataset export for one task.

    CVAT v2 uses an async "request" API. We POST to kick off, poll the
    request until finished, then download from the provided URL.
    """
    # Step 1: kick off the export
    r = s.post(
        f"{host}/api/tasks/{task_id}/dataset/export",
        params={"format": "COCO 1.0", "save_images": "false"},
        timeout=30,
    )
    if r.status_code not in (200, 201, 202):
        print(f"  task {task_id}: export start failed {r.status_code} {r.text[:200]}",
              file=sys.stderr)
        return None
    body = r.json()
    rq_id = body.get("rq_id")
    if not rq_id:
        print(f"  task {task_id}: no rq_id returned: {body}", file=sys.stderr)
        return None

    # Step 2: poll until done
    result_url = None
    for _ in range(120):
        rr = s.get(f"{host}/api/requests/{rq_id}", timeout=30)
        rr.raise_for_status()
        data = rr.json()
        status = data.get("status")
        if status == "finished":
            result_url = data.get("result_url")
            break
        if status == "failed":
            print(f"  task {task_id}: export request failed: {data}", file=sys.stderr)
            return None
        time.sleep(2)
    else:
        print(f"  task {task_id}: export timed out", file=sys.stderr)
        return None

    if not result_url:
        print(f"  task {task_id}: no result_url", file=sys.stderr)
        return None

    # Step 3: download the zip
    r = s.get(result_url, stream=True, timeout=120)
    r.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "coco.zip"
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    # Unzip in-place and remove the zip
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://192.168.0.125:8080")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="oldi-admin")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory. Default: "
                         "data/annotations/exports/<timestamp>/")
    args = ap.parse_args()

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out = REPO / "data/annotations/exports" / stamp
    args.out.mkdir(parents=True, exist_ok=True)

    s = _login(args.host, args.user, args.password)

    r = s.get(f"{args.host}/api/tasks", params={"page_size": 500}, timeout=30)
    r.raise_for_status()
    tasks = r.json().get("results", [])
    print(f"found {len(tasks)} task(s); exporting to {args.out}")

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "host": args.host,
        "tasks": [],
    }
    for t in tasks:
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in t["name"])[:80]
        task_dir = args.out / f"task-{t['id']}-{slug}"
        print(f"  task {t['id']} {t['name']!r} → {task_dir.name}")
        exported = _export_task(s, args.host, t["id"], task_dir)
        if exported is None:
            manifest["tasks"].append({"id": t["id"], "name": t["name"],
                                        "status": "failed"})
            continue
        # Count shapes for the manifest
        coco_json = next(task_dir.rglob("*.json"), None)
        n_ann = 0
        if coco_json:
            try:
                data = json.loads(coco_json.read_text())
                n_ann = len(data.get("annotations", []))
            except Exception:
                pass
        manifest["tasks"].append({
            "id": t["id"],
            "name": t["name"],
            "project_id": t.get("project_id"),
            "updated": t.get("updated_date"),
            "status": t.get("status"),
            "mode": t.get("mode"),
            "jobs_state": t.get("status"),
            "n_annotations_in_export": n_ann,
            "export_dir": str(task_dir.relative_to(REPO)
                                if task_dir.is_relative_to(REPO) else task_dir),
        })

    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest to {args.out / 'manifest.json'}")
    total = sum(t.get("n_annotations_in_export", 0) for t in manifest["tasks"])
    print(f"total annotations across all tasks: {total}")


if __name__ == "__main__":
    main()
