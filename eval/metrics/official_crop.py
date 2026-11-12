"""official_crop.py -- geometry of the official Pixal3D input preprocessing (numpy + PIL only).

Replica of the crop performed by Pixal3DImageTo3DPipeline.preprocess_image (Pixal3D repository,
pixal3d/pipelines/pixal3d_image_to_3d.py, commit 28efad6) for inputs that carry a real alpha channel (every
conditioning render of the evaluation set):

    scale  = min(1, 1024 / max(size));  resize (LANCZOS) when scale < 1
    bbox   = argwhere(alpha > 0.8 * 255)  -> (xmin, ymin, xmax, ymax)
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2);  size = int(max(w, h) * 1.1)
    box    = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    crop   = PIL.Image.crop(box)     # PIL rounds every coordinate: int(round(v)) (banker's rounding)

The generator feeds exactly this crop to MoGe-2 and to the model, so the model's inference-time projection lives in
the crop's image frame.  The evaluation is done in the raw conditioning image frame; `crop_to_raw_ndc` is the exact
affine map between the two normalised image frames (no ground-truth camera involved: a function of the input only).
"""
from __future__ import annotations

import numpy as np

OFFICIAL_MAX_SIZE = 1024
OFFICIAL_ALPHA_THRESHOLD = 0.8 * 255
OFFICIAL_PAD = 1.1
SOURCE = "Pixal3D pixal3d/pipelines/pixal3d_image_to_3d.py preprocess_image (commit 28efad6)"


def official_crop_box(image) -> dict:
    """image: PIL.Image (RGBA with a non-trivial alpha).  Returns the crop geometry in the pixel grid of the (possibly
    down-scaled) image the official code crops from."""
    from PIL import Image

    if image.mode != "RGBA" or bool(np.all(np.array(image)[:, :, 3] == 255)):
        raise ValueError("official_crop_box: input has no usable alpha channel (the official code would run "
                         "background removal; the crop is then not a function of the input alone)")
    max_size = max(image.size)
    scale = min(1, OFFICIAL_MAX_SIZE / max_size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    alpha = np.array(image)[:, :, 3]
    bbox = np.argwhere(alpha > OFFICIAL_ALPHA_THRESHOLD)
    if len(bbox) == 0:
        raise ValueError("official_crop_box: empty alpha mask")
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * OFFICIAL_PAD)
    box = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    x0, y0, x1, y1 = (int(round(v)) for v in box)          # PIL.Image.crop: map(int, map(round, box))
    return {"source_size": [int(image.width), int(image.height)], "resize_scale": float(scale),
            "alpha_bbox": [int(v) for v in bbox], "crop_box": [x0, y0, x1, y1],
            "crop_size": [x1 - x0, y1 - y0], "pad": OFFICIAL_PAD, "alpha_threshold": OFFICIAL_ALPHA_THRESHOLD,
            "source": SOURCE}


def crop_to_raw_ndc(crop_box, source_size) -> dict:
    """Affine map between normalised image frames (x right, y up, both in [-1, 1]):
           x_raw = sx * x_crop + tx,   y_raw = sy * y_crop + ty
    for a crop box (x0, y0, x1, y1) (pixel edges, y down) of an image of size (W, H)."""
    x0, y0, x1, y1 = (float(v) for v in crop_box)
    W, H = (float(v) for v in source_size)
    return {"sx": (x1 - x0) / W, "sy": (y1 - y0) / H,
            "tx": (x0 + x1) / W - 1.0, "ty": 1.0 - (y0 + y1) / H}


IDENTITY_NDC = {"sx": 1.0, "sy": 1.0, "tx": 0.0, "ty": 0.0}
