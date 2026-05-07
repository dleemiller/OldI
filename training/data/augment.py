"""Augmentation pipelines for bridging DeepScoresV2 → 19th-century tune books.

Four augraphy "modes" target the dominant defect patterns we observed across
the actual corpus. Each mode is named after the representative book that
motivated it; the calibration grid (`training/calibrate_augment.py`) should
make every mode look comparable to its namesake.

Observed per-book defect signatures (from audit at 2026-04-22):

  oneill_dance  — dense black letterpress, speckle noise, crisp white paper.
                  Mean lum 223, std 84, ink-fraction 12.6%.
  oneill_music  — thin-stroke engraving, visible JPEG ringing around glyphs,
                  clean white paper. Lum 223, std 70, ink 12.2%. Low-res
                  source JPEG.
  petrie        — aged paper: global luminance drop (mean 180), low contrast
                  (std 44), heavy paper-texture showing through ink. Grey
                  edges on all four sides (gutter not just spine).
  oneill_waifs  — almost pristine modern reprint; very sparse ink (4.6%),
                  bright paper (lum 243). Baseline minimal degradation.

For pretraining (Phase A) we sample uniformly among the four modes so the
detector sees all four profiles. For finetuning (Phase B) we keep only a
light pipeline — real pages already carry the real artefacts; over-augmenting
just makes the train distribution worse than the test distribution.

Custom ops layered on top:
  - staff-line thickness jitter (dilate/erode rows detected as staff lines)
  - mild elastic deformation (α≈20, σ≈5) on the full image — hand-engraving
    irregularity
  - ±2° skew + mild perspective (residual after deskew)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from augraphy import (
    AugraphyPipeline,
    BleedThrough,
    BookBinding,
    Brightness,
    BrightnessTexturize,
    InkBleed,
    Jpeg,
    Letterpress,
    LightingGradient,
    PageBorder,
)

ModeName = Literal[
    "oneill_dance_like",
    "oneill_music_like",
    "petrie_like",
    "oneill_waifs_like",
    "popsel_like",
    "minstrel_like",
    "graves_like",
]
ALL_MODES: tuple[ModeName, ...] = (
    "oneill_dance_like",
    "oneill_music_like",
    "petrie_like",
    "oneill_waifs_like",
    "popsel_like",
    "minstrel_like",
    "graves_like",
)
# Modes safe to sample during training: they only alter pixel values, not
# geometry. `graves_like` uses augraphy's BookBinding/PageBorder which shift
# the original page content within the output canvas; even after resizing
# back to input dims, the content is no longer at its original (x, y), so
# COCO bboxes computed on the input image become invalid. graves_like is
# available for calibration visualisation only.
PRETRAIN_MODES: tuple[ModeName, ...] = (
    "oneill_dance_like",
    "oneill_music_like",
    "petrie_like",
    "oneill_waifs_like",
    "popsel_like",
    "minstrel_like",
)


@dataclass
class AugmentedSample:
    """Result of running an augmentation pipeline on a single image."""

    image: np.ndarray  # (H, W) or (H, W, 3) uint8
    mode: ModeName | Literal["finetune", "identity"]


# ───────────────────────── Per-book mode pipelines ─────────────────────────


def oneill_dance_mode() -> AugraphyPipeline:
    """Heavy-ink letterpress. Clean white paper.

    Matches the 1907 O'Neill engravings: solid dark ink, irregular letterpress
    stipple. Staff lines stay crisp. (Speckle noise was previously supplied by
    augraphy's BadPhotoCopy but that op crashes under numpy 2.x; the
    letterpress stipple gives us the dominant effect anyway.)
    """
    ink_phase = [
        InkBleed(intensity_range=(0.2, 0.4), kernel_size=(3, 3), severity=(0.2, 0.35), p=0.9),
        Letterpress(n_samples=(120, 250), n_clusters=(120, 250),
                    std_range=(1500, 3500), value_range=(180, 245),
                    value_threshold_range=(128, 160), blur=1, p=0.9),
    ]
    paper_phase = []
    post_phase = [
        Brightness(brightness_range=(0.92, 1.08), p=0.5),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


def oneill_music_mode() -> AugraphyPipeline:
    """Thin-stroke engraving + JPEG compression artifacts.

    Matches the low-res 1903 O'Neill JPEG scans on archive.org: thinner ink,
    mild bleed, noticeable JPEG ringing around glyphs.
    """
    ink_phase = [
        InkBleed(intensity_range=(0.1, 0.3), kernel_size=(3, 3), severity=(0.15, 0.25), p=0.7),
    ]
    paper_phase = []
    post_phase = [
        Jpeg(quality_range=(30, 70), p=0.95),
        Brightness(brightness_range=(0.9, 1.1), p=0.4),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


def petrie_mode() -> AugraphyPipeline:
    """Aged paper + letterpress + low contrast.

    Matches the 1855 Petrie collection: page luminance ~180 with std ~44,
    mild paper grain visible through ink, uniform grey cast including edges.
    The texture in the real scans is subtle — we keep BrightnessTexturize
    only, and skip NoiseTexturize which produced a gravel-like result in
    calibration.
    """
    ink_phase = [
        InkBleed(intensity_range=(0.3, 0.5), kernel_size=(5, 5), severity=(0.25, 0.4), p=0.8),
        Letterpress(n_samples=(150, 350), n_clusters=(150, 300),
                    std_range=(2000, 4000), value_range=(180, 230),
                    value_threshold_range=(140, 180), blur=1, p=0.7),
    ]
    paper_phase = [
        BrightnessTexturize(texturize_range=(0.90, 0.98), deviation=0.05, p=0.7),
    ]
    # Dim the paper globally to simulate aged page. Brightness < 1 darkens.
    post_phase = [
        Brightness(brightness_range=(0.75, 0.88), p=1.0),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


def popsel_mode() -> AugraphyPipeline:
    """Low-DPI scan + bleed-through from verso + mild grey paper.

    Matches `popular_selections_from_oneill.pdf`: 100-DPI embedded JPEG, grey
    paper (lum ~207), visible ghost mirror-image text from the back of the
    page showing through above the staves. Distinct from oneill_music_like
    (which has JPEG but no bleed-through) and from petrie_like (which has
    darker, more textured paper).
    """
    ink_phase = [
        InkBleed(intensity_range=(0.2, 0.4), kernel_size=(3, 3), severity=(0.2, 0.35), p=0.6),
    ]
    paper_phase = [
        BleedThrough(intensity_range=(0.1, 0.3), color_range=(64, 180),
                     ksize=(17, 17), sigmaX=1, alpha=0.15,
                     offsets=(30, 50), p=0.9),
    ]
    post_phase = [
        Brightness(brightness_range=(0.85, 0.95), p=1.0),
        Jpeg(quality_range=(35, 65), p=0.8),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


def oneill_waifs_mode() -> AugraphyPipeline:
    """Near-pristine modern reprint — minimal degradation.

    Matches oneill_waifs's clean look: very light brightness drift, trace
    JPEG, no letterpress. Acts as the "easy" anchor mode in the mix.
    """
    ink_phase = []
    paper_phase = []
    post_phase = [
        Brightness(brightness_range=(0.95, 1.05), p=0.5),
        Jpeg(quality_range=(70, 95), p=0.5),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


def minstrel_mode() -> AugraphyPipeline:
    """Heavy-ink blotchy grey paper + soft blur.

    Matches `the_irish_minstrel.pdf` (1880×2888 from 116 DPI source) and
    `general_collection_ancient_music_of_ireland.pdf`: grey paper mean ~180
    but with distinctly uneven discoloration (lighting gradient creates
    darker blotchy patches), and very thick ink that has been slightly
    softened by the low-DPI upscale.
    """
    ink_phase = [
        InkBleed(intensity_range=(0.4, 0.6), kernel_size=(7, 7), severity=(0.3, 0.5), p=0.9),
        Letterpress(n_samples=(120, 220), n_clusters=(120, 220),
                    std_range=(2500, 4500), value_range=(140, 210),
                    value_threshold_range=(128, 180), blur=2, p=0.7),
    ]
    paper_phase = []
    post_phase = [
        Brightness(brightness_range=(0.78, 0.88), p=1.0),
        LightingGradient(mode="gaussian", min_brightness=0, max_brightness=255,
                         transparency=0.75, p=0.7),
        Jpeg(quality_range=(45, 75), p=0.6),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


def graves_mode() -> AugraphyPipeline:
    """Camera-captured book: black surround, facing-page peek, perspective.

    Matches `irish_song_book_graves.pdf`: page ~lum 140 on pale paper, but
    edges drop to ~10-16 because the book was photographed on a black
    surface. Also contains a partial view of the facing page on one side.
    The signature artefact is the sharp transition from lit page to black
    surround at the page edges.
    """
    ink_phase = [
        InkBleed(intensity_range=(0.2, 0.4), kernel_size=(3, 3), severity=(0.2, 0.35), p=0.7),
    ]
    paper_phase = []
    post_phase = [
        # BookBinding draws the book gutter + curls the far edge + shadows it;
        # closest available augraphy op to the "book photographed on black
        # surface" look.
        BookBinding(shadow_radius_range=(30, 90),
                    curve_range_right=(50, 120), curve_range_left=(200, 350),
                    curve_ratio_right=(0.05, 0.12), curve_ratio_left=(0.45, 0.62),
                    enable_shadow=1, p=0.9),
        # Dark page border (the black surround) — varies per page.
        PageBorder(page_border_width_height=(-30, -30),
                   page_border_color=(0, 0, 0),
                   page_border_background_color=(0, 0, 0),
                   page_rotation_angle_range=(-2, 2),
                   same_page_border=1, p=0.5),
        Brightness(brightness_range=(0.80, 0.95), p=0.8),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase, paper_phase=paper_phase, post_phase=post_phase, log=False,
    )


_MODE_FACTORIES: dict[ModeName, callable] = {
    "oneill_dance_like": oneill_dance_mode,
    "oneill_music_like": oneill_music_mode,
    "petrie_like": petrie_mode,
    "oneill_waifs_like": oneill_waifs_mode,
    "popsel_like": popsel_mode,
    "minstrel_like": minstrel_mode,
    "graves_like": graves_mode,
}


# ───────────────────────── Custom geometric + morphological ops ─────────────


def staff_line_jitter(img: np.ndarray, *, p: float = 0.3, rng: random.Random | None = None) -> np.ndarray:
    """Dilate or erode pure staff-line rows by 1 px.

    A "staff line" here is specifically: a row where the longest *continuous*
    horizontal run of dark pixels is a large fraction of the page width. That
    condition is true of real staff lines (spans the whole staff) but not of
    note-stem rows or beamed-note rows (where dark pixels are scattered).

    The earlier heuristic (row_ink_fraction > threshold) fired on every row
    that intersects a notehead cluster, producing horizontal streaks across
    the whole page; the run-length test is much stricter.
    """
    rng = rng or random.Random()
    if rng.random() >= p:
        return img
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 128).astype(np.uint8)
    h, w = gray.shape

    # Longest contiguous dark run per row, via per-row reset-on-gap cumsum.
    # Compute the max run length for each row without a Python loop.
    # For each pixel: run = dark and (left_run + 1)
    runs = np.zeros_like(dark, dtype=np.int32)
    prev = np.zeros(h, dtype=np.int32)
    for x in range(w):
        prev = (prev + 1) * dark[:, x]
        runs[:, x] = prev
    max_run = runs.max(axis=1)
    # Real staff lines usually span ≥60% of the page width.
    candidate = max_run > int(0.6 * w)
    if not candidate.any():
        return img

    action = rng.choice(["dilate", "erode"])
    kernel = np.ones((3, 1), np.uint8)
    out = img.copy()
    gray_out = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if action == "dilate":
        modified = cv2.erode(gray_out, kernel, iterations=1)  # thicker ink
    else:
        modified = cv2.dilate(gray_out, kernel, iterations=1)  # thinner ink
    if out.ndim == 2:
        out[candidate] = modified[candidate]
    else:
        out[candidate, :] = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)[candidate, :]
    return out


def elastic_noteheads(
    img: np.ndarray, *, p: float = 0.3, alpha: float = 20.0, sigma: float = 5.0,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Mild elastic deformation (Simard α≈20, σ≈5) applied to the whole image.

    Simulates the irregularity of hand-engraved glyphs. Kept very mild so
    staff-line parallelism doesn't visibly drift.
    """
    rng = rng or random.Random()
    if rng.random() >= p:
        return img
    shape = img.shape[:2]
    # Random displacement fields, smoothed with a Gaussian then scaled by α.
    dx = (np.random.rand(*shape) * 2 - 1) * alpha
    dy = (np.random.rand(*shape) * 2 - 1) * alpha
    dx = cv2.GaussianBlur(dx, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    dy = cv2.GaussianBlur(dy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def mild_skew_warp(img: np.ndarray, *, p: float = 0.5, max_angle: float = 2.0,
                   rng: random.Random | None = None) -> np.ndarray:
    """±2° rotation + mild perspective.

    We already deskew at ingest time; this covers sub-degree residual tilt
    and occasional mild perspective drift that slips through.
    """
    rng = rng or random.Random()
    if rng.random() >= p:
        return img
    h, w = img.shape[:2]
    angle = rng.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255),
                         flags=cv2.INTER_LINEAR)
    return out


# ───────────────────────── Public pipelines ─────────────────────────


def apply_mode(img: np.ndarray, mode: ModeName) -> np.ndarray:
    """Run a single named augraphy mode + the custom ops on one image.

    Output is always the same spatial dimensions as input. Some augraphy ops
    (BookBinding, PageBorder) grow the canvas to add facing pages or borders;
    we resize back so annotation bboxes computed on the original image remain
    valid. This gives up a little photorealism on `graves_like` but keeps
    training labels correct.
    """
    pipeline = _MODE_FACTORIES[mode]()
    img3 = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    in_h, in_w = img3.shape[:2]
    out = pipeline(img3)
    if out.shape[:2] != (in_h, in_w):
        out = cv2.resize(out, (in_w, in_h), interpolation=cv2.INTER_AREA)
    out = staff_line_jitter(out)
    out = elastic_noteheads(out)
    out = mild_skew_warp(out)
    # mild_skew_warp can very slightly change size if it applied affine; force
    # final size match.
    if out.shape[:2] != (in_h, in_w):
        out = cv2.resize(out, (in_w, in_h), interpolation=cv2.INTER_AREA)
    return out


def pretrain_apply(img: np.ndarray, *, rng: random.Random | None = None) -> AugmentedSample:
    """Sample a random pretrain-safe mode and apply it. Used during Phase A.

    Uniform sample of `PRETRAIN_MODES` so the detector sees every defect
    profile across the epoch. graves_like is excluded — it warps page
    content and would invalidate bboxes.
    """
    rng = rng or random.Random()
    mode = rng.choice(PRETRAIN_MODES)
    return AugmentedSample(image=apply_mode(img, mode), mode=mode)


def finetune_pipeline() -> AugraphyPipeline:
    """Light augmentation for real finetune. Real pages already carry the
    heavy-ink / letterpress / aged-paper artefacts; overdoing augraphy here
    would make the training distribution worse than the test distribution.
    Keeps only brightness drift + mild JPEG."""
    return AugraphyPipeline(
        ink_phase=[],
        paper_phase=[],
        post_phase=[
            Brightness(brightness_range=(0.9, 1.1), p=0.5),
            Jpeg(quality_range=(70, 95), p=0.3),
        ],
        log=False,
    )


def finetune_apply(img: np.ndarray, *, rng: random.Random | None = None) -> AugmentedSample:
    pipeline = finetune_pipeline()
    img3 = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = pipeline(img3)
    out = mild_skew_warp(out, p=0.3, max_angle=1.0, rng=rng)
    return AugmentedSample(image=out, mode="finetune")
