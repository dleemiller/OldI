"""Re-encode any DSv2 PNG that libpng complains about, to regenerate its CRCs.

Background: ~handful of images in the `ds2_dense` dump have corrupted IDAT
CRCs or truncated adaptive filter values. PIL recovers and decodes them,
but libpng emits warnings on every read — spamming training stdout.

Reading each image through PIL and re-saving wipes the broken CRC. One-time
operation, in-place (atomic rename), safe to run even while training is in
progress (workers holding the old file in memory continue, subsequent
opens see the clean version).

Use:
  uv run python -m training.data.fix_dsv2_png_crcs \\
      --images-root data/deepscoresv2/ds2_dense/images
"""

from __future__ import annotations

import argparse
import os
import select
import sys
from pathlib import Path

import warnings
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _has_libpng_warning(path: Path) -> bool:
    """Open the PNG and capture any libpng writes to stderr.

    Uses a temp file (not a pipe) so we don't race libpng's flush — the
    pipe approach needed `select()` which returned false-negatives when
    the OS hadn't propagated the data yet.
    """
    import tempfile
    with tempfile.TemporaryFile() as tmp:
        saved = os.dup(2)
        try:
            os.dup2(tmp.fileno(), 2)
            try:
                with Image.open(path) as im:
                    im.load()
                    # Some warnings (IDAT CRC in particular) only fire when
                    # the full image data is consumed via tobytes / np.array.
                    _ = im.tobytes()
            except Exception:
                return True
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.flush()
        tmp.seek(0)
        msg = tmp.read()
    return b"libpng" in msg or b"IDAT" in msg or b"CRC" in msg


def _reencode(path: Path) -> None:
    with Image.open(path) as im:
        im.load()
        # Preserve mode and bit depth implicitly by round-tripping through
        # PIL. Save to .tmp then atomic rename.
        tmp = path.with_suffix(path.suffix + ".tmp")
        im.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-root", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(Path(args.images_root).glob("*.png"))
    print(f"scanning {len(files)} PNGs for libpng warnings...")

    bad: list[Path] = []
    for i, p in enumerate(files, start=1):
        if _has_libpng_warning(p):
            bad.append(p)
        if i % 200 == 0:
            print(f"  {i}/{len(files)} scanned, {len(bad)} flagged so far")
    print(f"flagged: {len(bad)} / {len(files)}")

    fixed = 0
    for p in bad:
        try:
            _reencode(p)
            fixed += 1
        except Exception as exc:
            print(f"  FAILED to reencode {p.name}: {exc}", file=sys.stderr)
    print(f"re-encoded: {fixed}/{len(bad)}")


if __name__ == "__main__":
    main()
