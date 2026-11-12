"""Sparse structure: 16^3 latent -> occupied 32^3 voxels.

``structure_student`` on: four deterministic, conditional-only Euler steps of
the LoRA-adapted structure flow on the noise fractions
``t = 1.0, 0.9, 0.75, 0.6, 0.0`` (indices 0, 3, 6, 8, 12 of the 12-step grid
``t' = 3t / (1 + 2t)``), starting from a fixed seeded noise. The dense
forward runs through a static CUDA graph (:mod:`fastpixal3d.runtime.cuda_graph`);
the first graphed rollout of a process is compared with an eager rollout of
the same inputs (``torch.equal``).

A structure adapter whose students.json entry describes an x0 prediction (for
example the entry ``structure_consistency``, selected by
``Config.structure_adapter``; no configuration of this repository selects it)
is sampled as a consistency student instead: it predicts the clean latent at
``t = 1.0, 0.9, 0.75, 0.5`` and re-noises it to the next time with a seeded
CUDA generator between the steps (:func:`consistency_rollout`).

Off: Pixal3D's 12-step guided sampler (guidance 7.5, rescale 0.7, interval
[0.6, 1.0], ``rescale_t`` 5) from the same noise.

The latent is decoded by Pixal3D's sparse-structure decoder; voxels with a
positive logit, max-pooled from 64^3 to 32^3, form the support.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

LATENT_RESOLUTION = 16
SUPPORT_RESOLUTION = 32
LATENT_CHANNELS = 8


def teacher_t_grid(steps: int, rescale_t: float) -> np.ndarray:
    """Noise fractions of Pixal3D's Euler sampler (descending, length steps + 1)."""
    t = np.linspace(1.0, 0.0, int(steps) + 1, dtype=np.float64)
    if rescale_t != 1.0:
        t = rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
    return t


def student_t_values(grid_indices: Sequence[int], rescale_t: float = 3.0, steps: int = 12):
    grid = teacher_t_grid(steps, rescale_t)
    return [float(grid[i]) for i in grid_indices]


def initial_noise(device, seed: int = 42) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((1, LATENT_CHANNELS, LATENT_RESOLUTION, LATENT_RESOLUTION, LATENT_RESOLUTION),
                       dtype=torch.float32, device="cpu", generator=generator).to(device)


def euler_rollout(flow_model, noise_dense: torch.Tensor, cond, t_values: Sequence[float]) -> torch.Tensor:
    """Conditional-only Euler steps ``x <- x - (t - t_next) * v(x, 1000 t)`` on the given grid."""
    sample = noise_dense
    t_values = [float(t) for t in t_values]
    for t, t_prev in zip(t_values[:-1], t_values[1:]):
        t_tensor = torch.tensor([1000.0 * t] * sample.shape[0], device=sample.device, dtype=torch.float32)
        pred_v = flow_model(sample, t_tensor, cond)
        sample = sample - (t - t_prev) * pred_v
    return sample


# Earlier structure student (consistency-distilled): x0 prediction, re-noised between the steps.
TRAJECTORY, CONSISTENCY = "trajectory", "consistency"
SIGMA_MIN = 1e-5


def sampler_of(steps: Dict[str, Any]) -> str:
    """Which rollout a students.json ``steps`` entry describes: ``update`` "euler" -> trajectory
    (Euler steps on a grid), an x0 prediction / consistency update -> consistency."""
    update = str(steps.get("update", "euler")).lower()
    if update == "euler":
        return TRAJECTORY
    if "consistency" in update or "x0" in update:
        return CONSISTENCY
    raise ValueError(f"unknown structure student update {update!r}")


def _rows(dense: torch.Tensor) -> torch.Tensor:
    """[1, C, R, R, R] -> [R^3, C] in the C order of the (x, y, z) grid coordinates."""
    return dense[0].permute(1, 2, 3, 0).reshape(-1, dense.shape[1])


def _dense(rows: torch.Tensor, resolution: int) -> torch.Tensor:
    return rows.reshape(resolution, resolution, resolution, -1).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def add_noise(x0: torch.Tensor, eps: torch.Tensor, t: float) -> torch.Tensor:
    """x_t = (1 - t) x0 + (sigma_min + (1 - sigma_min) t) eps (Pixal3D's forward process)."""
    from .shape import _bcast

    tt = _bcast(t, x0)
    return (1.0 - tt) * x0 + (SIGMA_MIN + (1.0 - SIGMA_MIN) * tt) * eps


@torch.no_grad()
def consistency_rollout(flow_model, noise_dense: torch.Tensor, cond, query_t: Sequence[float],
                        renoise_seed: int = 42) -> torch.Tensor:
    """The consistency student: at each query time the clean latent is predicted from the velocity
    (``x0 = (1 - s) x_t - (s + (1 - s) t) v``, float64); before the next query time it is re-noised
    with fresh noise from a CUDA generator seeded once per call. The latent is handled as the rows of
    the full 16^3 grid in coordinate order (so the noise draws have the shape [4096, 8]); the forward
    pass sees the same values as a dense [1, 8, 16, 16, 16] tensor."""
    from .shape import x0_from_v

    resolution = int(noise_dense.shape[2])
    generator = torch.Generator(device=noise_dense.device)
    generator.manual_seed(int(renoise_seed))
    x = _rows(noise_dense)
    query_t = [float(t) for t in query_t]
    for index, t in enumerate(query_t):
        t1000 = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1) * 1000.0
        v = _rows(flow_model(_dense(x, resolution), t1000, cond))
        x0 = x0_from_v(x, v, t, double=True)
        if index == len(query_t) - 1:
            return _dense(x0, resolution)
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        x = add_noise(x0, eps, query_t[index + 1])
    raise ValueError("consistency_rollout needs at least one query time")


class StructureStudent:
    """Rollout of the structure student (optionally through the static CUDA graph). ``sampler``:
    ``trajectory`` (Euler steps on ``t_values``) or ``consistency`` (x0 prediction at the query times
    ``t_values`` with re-noising, :func:`consistency_rollout`)."""

    def __init__(self, model, t_values: Sequence[float], use_graph: bool = True, verify: bool = True,
                 sampler: str = TRAJECTORY, renoise_seed: int = 42):
        if sampler not in (TRAJECTORY, CONSISTENCY):
            raise ValueError(f"unknown structure sampler {sampler!r}")
        self.model = model
        self.t_values = list(t_values)
        self.use_graph = use_graph
        self.verify = verify
        self.sampler = sampler
        self.renoise_seed = int(renoise_seed)
        self._checked = False

    def _rollout(self, noise: torch.Tensor, cond) -> torch.Tensor:
        if self.sampler == CONSISTENCY:
            return consistency_rollout(self.model, noise, cond, self.t_values, self.renoise_seed)
        return euler_rollout(self.model, noise, cond, self.t_values)

    @torch.no_grad()
    def __call__(self, cond: Dict[str, Any], noise: torch.Tensor, stats: Dict[str, Any] = None) -> torch.Tensor:
        stats = {} if stats is None else stats
        stats["sampler"] = self.sampler
        if not self.use_graph:
            return self._rollout(noise, cond)
        from ..runtime.cuda_graph import graphed_forward

        reference = None
        if self.verify and not self._checked:
            reference = self._rollout(noise, cond)
            torch.cuda.synchronize()
        with graphed_forward(self.model, use_kv=True, strict=True, verify=self.verify) as record:
            out = self._rollout(noise, cond)
        n_forwards = len(self.t_values) - (1 if self.sampler == TRAJECTORY else 0)
        if record.get("replays") != n_forwards or record.get("fallbacks"):
            raise RuntimeError(f"structure CUDA graph did not cover the rollout: {record}")
        if record.get("graph_captured") and record.get("capture_verified") is False:
            raise RuntimeError("structure CUDA graph capture was not verified")
        if reference is not None:
            equal = bool(torch.equal(out, reference))
            stats["graph_selfcheck_equal"] = equal
            self._checked = True
            if not equal:
                raise RuntimeError("graphed structure rollout differs from the eager rollout")
        stats["graph"] = {k: record.get(k) for k in ("graph_captured", "replays", "kv_fills", "kv_reuses")}
        return out


TEACHER_GUIDANCE = {"guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0]}


def _guidance(guidance: bool) -> Dict[str, Any]:
    # guidance_strength 1.0 makes Pixal3D's guided sampler return the conditional prediction
    # (one forward pass per step).
    return dict(TEACHER_GUIDANCE) if guidance else {**TEACHER_GUIDANCE, "guidance_strength": 1.0}


@torch.no_grad()
def teacher_rollout(pipe, cond_full: Dict[str, Any], noise: torch.Tensor, steps: int = 12,
                    rescale_t: float = 5.0, guidance: bool = True) -> torch.Tensor:
    params = dict(getattr(pipe, "sparse_structure_sampler_params", None) or {})
    params.update({"steps": steps, "rescale_t": rescale_t, **_guidance(guidance), "verbose": False})
    result = pipe.sparse_structure_sampler.sample(pipe.models["sparse_structure_flow_model"], noise.clone(),
                                                  cond=cond_full["cond"], neg_cond=cond_full["neg_cond"], **params)
    return result.samples


@torch.no_grad()
def teacher_rollout_on_grid(pipe, cond_full: Dict[str, Any], noise: torch.Tensor, t_values: Sequence[float],
                            guidance: bool = False) -> torch.Tensor:
    """Pixal3D's structure sampler, one ``sample_once`` per pair of consecutive ``t_values``
    (the same-step teacher of the structure student: its grid, conditional prediction only)."""
    sampler = pipe.sparse_structure_sampler
    model = pipe.models["sparse_structure_flow_model"]
    kwargs = {"neg_cond": cond_full["neg_cond"], **_guidance(guidance)}
    sample = noise.clone()
    t_values = [float(t) for t in t_values]
    for t, t_prev in zip(t_values[:-1], t_values[1:]):
        sample = sampler.sample_once(model, sample, t, t_prev, cond_full["cond"], **kwargs).pred_x_prev
    return sample


@torch.no_grad()
def decode_support(decoder, z_dense: torch.Tensor, device, ss_resolution: int = SUPPORT_RESOLUTION) -> torch.Tensor:
    """Dense latent [1, 8, 16, 16, 16] -> int32 support coordinates [M, 4] (b, x, y, z)."""
    decoder = decoder.to(device)
    decoder.requires_grad_(False)
    decoder.eval()
    logits = decoder(z_dense.to(device))
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("sparse-structure decoder output contains NaN or Inf")
    decoded = logits > 0
    res = int(decoded.shape[2])
    if ss_resolution != res:
        ratio = res // ss_resolution
        decoded = F.max_pool3d(decoded.float(), kernel_size=ratio, stride=ratio, padding=0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    return coords.to(device=device, dtype=torch.int32)
