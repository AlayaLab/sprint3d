"""Narrow-band dual-contouring remesh, optionally at half resolution.

The decoded mesh is remeshed on a narrow band of the unsigned distance field
(Pixal3D's export path does the same at the decoder resolution).
``adaptive_remesh`` lowers the remesh grid to half the decoder resolution for
every object except thin ones: an object counts as thin when its fine-shape
support has many tokens per unit of resolution squared,

    density = hr_tokens / resolution**2 >= 0.025   ->  keep the full grid.

(A branched or thin object has much more surface per bounding volume than a
compact one, so its support is dense relative to resolution**2.)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

DIVISOR_FULL = 1
DIVISOR_HALF = 2


def support_density(hr_tokens: int, resolution: int) -> float:
    if hr_tokens < 0 or resolution <= 0:
        raise ValueError(f"support_density: hr_tokens={hr_tokens!r}, resolution={resolution!r}")
    return hr_tokens / (resolution ** 2)


def remesh_divisor(hr_tokens: int, resolution: int, threshold: float = 0.025, adaptive: bool = True) -> int:
    if not adaptive:
        return DIVISOR_FULL
    return DIVISOR_FULL if support_density(hr_tokens, resolution) >= threshold else DIVISOR_HALF


def remesh_arguments(scale_extent: float, resolution: int, band, divisor: int) -> Tuple[float, int]:
    """(scale, resolution) passed to ``remesh_narrow_band_dc``.

    The domain is padded by 3*band voxels: ``scale = (res + 3*band) / res * extent``.
    For a divided grid the padding is recomputed for the new resolution from the
    padded scale, with the same floating-point expressions as the reference run.
    """
    scale = (resolution + 3 * band) / resolution * scale_extent
    if divisor == DIVISOR_FULL:
        return scale, resolution
    scale_orig = scale * resolution / (resolution + 3 * band)
    resolution2 = resolution // divisor
    scale2 = (resolution2 + 3 * band) / resolution2 * scale_orig
    return scale2, resolution2


def remesh(remeshing, vertices, faces, center, scale_extent: float, resolution: int, band, project_back,
           bvh, divisor: int = DIVISOR_FULL, stats: Dict[str, Any] = None):
    """Run ``remeshing.remesh_narrow_band_dc`` and return ``(vertices, faces)``."""
    scale, res = remesh_arguments(scale_extent, resolution, band, divisor)
    out = remeshing.remesh_narrow_band_dc(vertices, faces, center=center, scale=scale, resolution=res, band=band,
                                          project_back=project_back, verbose=False, bvh=bvh)
    if stats is not None:
        stats.update({"divisor": divisor, "remesh_resolution": res, "remesh_faces": int(out[1].shape[0])})
    return out
