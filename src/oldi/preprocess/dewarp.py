"""UVDoc document dewarping.

UVDoc (PaddlePaddle, Apache-2.0) predicts a 2D Bézier mesh over the input image
and applies the inverse transform to rectify document warp. We run it via the
HuggingFace `transformers` port (first-party in v5.6+), so the model weights
are pulled from `PaddlePaddle/UVDoc_safetensors` on the Hub.

Input to this module is a grayscale page; internally we widen to 3 channels
for UVDoc (trained on RGB photos) and narrow back to grayscale on the way out.

The loader is a lazy singleton: the model stays resident on GPU for the life
of the process so repeated `dewarp_gray` calls amortise the load.
"""

from __future__ import annotations

import numpy as np

_MODEL = None
_PROCESSOR = None
_MODEL_NAME = "PaddlePaddle/UVDoc_safetensors"


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    from transformers import AutoImageProcessor, AutoModel

    _PROCESSOR = AutoImageProcessor.from_pretrained(_MODEL_NAME)
    _MODEL = AutoModel.from_pretrained(_MODEL_NAME, device_map="auto")
    _MODEL.eval()
    return _MODEL, _PROCESSOR


def dewarp_gray(gray: np.ndarray) -> np.ndarray:
    """Rectify document warp on a grayscale page.

    Parameters
    ----------
    gray : (H, W) uint8
        Grayscale page, 255 = paper, 0 = ink.

    Returns
    -------
    (H', W') uint8
        Rectified grayscale page. Output size may differ slightly from input
        depending on UVDoc's predicted mesh, but is close to (H, W).
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected (H, W) uint8, got shape={gray.shape} dtype={gray.dtype}")
    import torch
    from PIL import Image

    model, processor = _load()
    # UVDoc was trained on RGB photos; replicate the grayscale channel so the
    # 3-channel normalisation produces a sensible input.
    pil = Image.fromarray(gray, mode="L").convert("RGB")
    inputs = processor(images=pil, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_document_rectification(
        outputs.last_hidden_state, inputs["original_images"]
    )
    # results is a list; one entry per input image. `images` is (H, W, 3) uint8
    # in BGR order (OpenCV convention) on CPU.
    rect_bgr = results[0]["images"]
    if hasattr(rect_bgr, "cpu"):
        rect_bgr = rect_bgr.cpu().numpy()
    else:
        rect_bgr = np.asarray(rect_bgr)
    # BGR → grayscale via ITU-R BT.601 luma (matches OpenCV's CV_BGR2GRAY).
    # Channel order doesn't matter much for a grayscale-in grayscale-out round
    # trip since all three channels held the same values on the way in; we
    # still use the luma formula for consistency with any future 3-channel
    # input path.
    b, g, r = rect_bgr[..., 0], rect_bgr[..., 1], rect_bgr[..., 2]
    gray_out = (0.114 * b + 0.587 * g + 0.299 * r).round().clip(0, 255).astype(np.uint8)
    return gray_out
