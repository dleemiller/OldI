"""Preprocessing pipeline for training data, annotation inputs, and inference.

Order: ingest (s01) → dewarp (UVDoc) → deskew (s01) → binarize (Sauvola).

Dewarp runs before deskew because UVDoc corrects general geometric warp (curl,
perspective, fold), which is a superset of rotation. Running the cheap classical
deskew afterwards catches any residual axis-aligned tilt.
"""

from .binarize import binarize_gray
from .dewarp import dewarp_gray
from .pipeline import preprocess_page

__all__ = ["binarize_gray", "dewarp_gray", "preprocess_page"]
