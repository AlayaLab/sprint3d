"""Time grid, segment pairs and per-view noise of the trajectory distillation.

Pixal3D's Euler sampler integrates the noise fraction t from 1 (noise) to 0 (data) on

    t_j = r * s_j / (1 + (r - 1) * s_j),    s = linspace(1, 0, steps + 1)

with ``r = rescale_t``. For a fixed ``r`` the grid of ``k`` steps is the subsequence of the
12-step grid at indices ``0, 12/k, 2*12/k, ...`` whenever ``k`` divides 12, so a student with ``k``
steps can be trained on states the teacher visits on its own 12-step trajectory ("anchors").
The released students use ``r = 3`` and the index grids

    structure: 0, 3, 6, 8, 12   (t = 1.0, 0.9, 0.75, 0.6, 0.0; not uniform)
    texture:   0, 6, 12         (t = 1.0, 0.75, 0.0; k = 2)

Each training sample is one segment ``(i, j)`` of the grid: the student starts from the teacher's
state at index ``i`` and must land, with one Euler step, on the teacher's state at index ``j``.
The state at index 0 is pure noise and is never stored: it is regenerated from a seed.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np

TEACHER_STEPS = 12
RESCALE_T = 3.0
SIGMA_MIN = 1e-5  # Pixal3D's flow-matching floor: x_t = (1 - t) x0 + (sigma_min + (1 - sigma_min) t) eps

# Salts of the per-view noise seeds. They are kept verbatim so that the anchors of the released
# training runs can be regenerated bit for bit (structure: grid index 0 of the structure flow;
# fine shape: grid index 0 of the fine-shape flow, whose final state conditions the texture flow).
STRUCTURE_NOISE_SALT = "ss_traj_distill|ss_traj_x0"
FINE_SHAPE_NOISE_SALT = "lrhr_traj_distill|hr_traj_x0"
TEXTURE_NOISE_SEED = 0  # the texture flow's initial noise: one fixed seed for every view (Pixal3D's own draw)

STRUCTURE_LATENT_CHANNELS = 8
STRUCTURE_LATENT_RESOLUTION = 16
SLAT_CHANNELS = 32


def teacher_t_grid(steps: int = TEACHER_STEPS, rescale_t: float = RESCALE_T) -> np.ndarray:
    """The sampler's own t sequence (float64, descending, ``steps + 1`` values)."""
    t = np.linspace(1.0, 0.0, int(steps) + 1, dtype=np.float64)
    if rescale_t != 1.0:
        t = rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
    return t


def check_grid_indices(indices: Sequence[int], steps: int = TEACHER_STEPS,
                       stored: Sequence[int] | None = None) -> List[int]:
    """Validate a student grid: strictly increasing teacher-grid indices from 0 to ``steps``;
    every interior index must be one of the ``stored`` anchor indices when those are given."""
    idx = [int(i) for i in indices]
    if len(idx) < 2 or idx[0] != 0 or idx[-1] != int(steps):
        raise ValueError(f"grid indices must start at 0 and end at {steps}: {idx}")
    if any(b <= a for a, b in zip(idx[:-1], idx[1:])):
        raise ValueError(f"grid indices must be strictly increasing: {idx}")
    if stored is not None:
        missing = [i for i in idx[1:-1] if i not in set(stored)]
        if missing:
            raise ValueError(f"interior grid indices {missing} have no stored anchors (stored: {list(stored)})")
    return idx


def segment_pairs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """``[(i0, i1), (i1, i2), ...]``: the segments a training sample is drawn from (uniformly)."""
    idx = [int(i) for i in indices]
    return list(zip(idx[:-1], idx[1:]))


def uniform_grid_indices(k: int, steps: int = TEACHER_STEPS) -> List[int]:
    if steps % k:
        raise ValueError(f"k={k} does not divide {steps}")
    return list(range(0, steps + 1, steps // k))


def grid_t_values(indices: Sequence[int], steps: int = TEACHER_STEPS, rescale_t: float = RESCALE_T) -> List[float]:
    grid = teacher_t_grid(steps, rescale_t)
    return [float(grid[i]) for i in indices]


def stable_seed(salt: str, view: str, extra: int = 0) -> int:
    """Reproducible 63-bit seed of one view: sha256 of ``"<salt>|<view>|<extra>"``."""
    raw = f"{salt}|{view}|{int(extra)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % ((1 << 63) - 1)


def cpu_normal(shape: Sequence[int], seed: int):
    """``torch.randn(shape)`` from a dedicated CPU generator (the global RNG is not touched)."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(tuple(int(s) for s in shape), dtype=torch.float32, device="cpu", generator=generator)


def structure_initial_noise(view: str):
    """Dense ``[1, 8, 16, 16, 16]`` noise of the structure flow at t = 1 for one view."""
    r = STRUCTURE_LATENT_RESOLUTION
    return cpu_normal((1, STRUCTURE_LATENT_CHANNELS, r, r, r), stable_seed(STRUCTURE_NOISE_SALT, view))


def fine_shape_initial_noise(view: str, num_tokens: int):
    """``[N, 32]`` noise of the fine-shape flow at t = 1 for one view."""
    return cpu_normal((num_tokens, SLAT_CHANNELS), stable_seed(FINE_SHAPE_NOISE_SALT, view))


def texture_initial_noise(num_tokens: int):
    """``[N, 32]`` noise of the texture flow at t = 1 (the same seed for every view)."""
    return cpu_normal((num_tokens, SLAT_CHANNELS), TEXTURE_NOISE_SEED)


def full_grid_index(resolution: int = STRUCTURE_LATENT_RESOLUTION, device="cpu"):
    """``[R^3, 4]`` int64 rows ``(0, x, y, z)`` of the full grid in C order (row ``x*R*R + y*R + z``).
    The structure anchors store the dense ``[1, C, R, R, R]`` states as ``[R^3, C]`` token
    matrices in this row order."""
    import torch

    r = int(resolution)
    linear = torch.arange(r ** 3, device=device, dtype=torch.int64)
    x = torch.div(linear, r * r, rounding_mode="floor")
    rest = linear.remainder(r * r)
    y = torch.div(rest, r, rounding_mode="floor")
    return torch.stack((torch.zeros_like(linear), x, y, rest.remainder(r)), dim=1)


def full_grid_gather(dense, index):
    """``[1, C, R, R, R]`` -> ``[R^3, C]`` (advanced indexing, as in training)."""
    return dense[index[:, 0], :, index[:, 1], index[:, 2], index[:, 3]]


def full_grid_scatter(tokens, index, resolution: int = STRUCTURE_LATENT_RESOLUTION):
    """``[R^3, C]`` -> ``[1, C, R, R, R]`` (index assignment into zeros, as in training)."""
    r = int(resolution)
    dense = tokens.new_zeros(1, tokens.shape[1], r, r, r)
    dense[index[:, 0], :, index[:, 1], index[:, 2], index[:, 3]] = tokens
    return dense
