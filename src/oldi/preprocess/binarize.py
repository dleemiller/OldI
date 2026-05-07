"""Adaptive binarization via Sauvola's method.

We keep binarization as a separate artefact alongside the grayscale page: the
detector trains on grayscale (preserves anti-aliasing and ink-density cues)
while downstream classical-CV steps (staff-line projection profile, connected-
component analysis on noteheads) run on the binary.

Sauvola was chosen over Otsu because 19th-century prints have large brightness
gradients (uneven plate inking, scanner illumination); a global threshold
loses staff lines in the dim regions. Sauvola's per-pixel threshold follows
the local mean, which is exactly what we want.
"""

from __future__ import annotations

import numpy as np


def binarize_gray(gray: np.ndarray, *, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Return the Sauvola binarisation of a grayscale page.

    Parameters
    ----------
    gray : (H, W) uint8
        Grayscale page.
    window_size : int
        Sauvola window in pixels. 25 is a reasonable default at 300–400 DPI
        music scans — wider than a staff-line spacing, narrower than a
        notehead row.
    k : float
        Sauvola sensitivity. Lower k → more ink; higher k → cleaner whites.

    Returns
    -------
    (H, W) uint8 with values in {0, 255}, where 255 = paper, 0 = ink.
    """
    from skimage.filters import threshold_sauvola

    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected (H, W) uint8, got shape={gray.shape} dtype={gray.dtype}")
    # window must be odd for Sauvola's sliding window math.
    if window_size % 2 == 0:
        window_size += 1
    threshold = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray > threshold).astype(np.uint8) * 255
    return binary
