"""Shape latents: coarse (512 cascade level) and fine (1024-1536) structured latents.

Coarse (``coarse_shape_student``): one step of the LoRA-adapted 512 shape
flow at noise fraction 1 from a fixed CPU-seeded noise; the clean latent is
recovered in float64 as ``x0 = (1 - s) x - (s + (1 - s) t) v`` with
``s = 1e-5`` (Pixal3D's flow convention) and de-normalised with the shape
latent statistics. Off: Pixal3D's 12-step guided sampler (seed 0).

Fine support: the coarse latent is upsampled by the shape decoder (x4) and
quantised to the fine grid with Pixal3D's token cap: the requested 1536 grid
is lowered by 128 until fewer than 49,152 voxels remain (never below 1024).

Fine (``fine_shape_student``): one step of the LoRA-adapted fine shape flow
from a per-object seeded noise, same recovery as the coarse step. With
``fold_lora_fine_shape`` the adapter is folded into the bf16 weights at load
time; with ``fa3_fine_shape_flow`` the fine flow runs on FlashAttention-3.
Off: Pixal3D's 12-step guided sampler from the same noise.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

SIGMA_MIN = 1e-5
SHAPE_LR_RESOLUTION = 512
SHAPE_HR_MIN_RESOLUTION = 1024
SHAPE_HR_RESOLUTION_STEP = 128
LATENT_CHANNELS = 32


def _bcast(t, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)
    if t.ndim == 0:
        t = t.reshape(1, 1).expand(x.shape[0], 1)
    elif t.ndim == 1:
        t = t.reshape(-1, 1)
    return t


def x0_from_v(xt: torch.Tensor, v: torch.Tensor, t, double: bool = True) -> torch.Tensor:
    """Clean latent from a velocity prediction (computed in float64, returned in ``xt``'s dtype)."""
    orig = xt.dtype
    if double:
        xt = xt.double()
        v = v.double()
    tt = _bcast(t, xt)
    x0 = (1.0 - SIGMA_MIN) * xt - (SIGMA_MIN + (1.0 - SIGMA_MIN) * tt) * v
    return x0.to(orig)


def student_velocity(model, x, t: float, cond):
    """The student's velocity at noise fraction ``t`` (timestep input ``1000 t``)."""
    t1000 = torch.as_tensor(t, device=x.feats.device, dtype=torch.float32).reshape(-1).to(
        device=x.feats.device, dtype=torch.float32) * 1000.0
    return model(x, t1000, cond, concat_cond=None)


def _sparse(feats, coords):
    from pixal3d.modules import sparse as sp

    return sp.SparseTensor(feats=feats, coords=coords)


# ---------------------------------------------------------------------------
# coarse level
# ---------------------------------------------------------------------------
def coarse_initial_noise(num_tokens: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((int(num_tokens), LATENT_CHANNELS), dtype=torch.float32, device="cpu", generator=generator)


@torch.no_grad()
def coarse_student(model, cond, coords: torch.Tensor, initial_noise_cpu: torch.Tensor, query_t: float,
                   renoise_seed: int, slat_std: torch.Tensor, slat_mean: torch.Tensor):
    """One-step coarse student; returns (de-normalised latent, normalised latent) as sparse tensors."""
    initial = initial_noise_cpu.to(device=coords.device, dtype=torch.float32)
    x = _sparse(initial, coords)
    renoise = torch.Generator(device=x.feats.device)  # unused for a single step (kept for the schedule)
    renoise.manual_seed(int(renoise_seed))
    velocity = student_velocity(model, x, query_t, cond)
    if velocity.feats.shape != x.feats.shape or not torch.equal(velocity.coords.to(x.coords.device), x.coords):
        raise RuntimeError("coarse-shape student changed the latent shape or support")
    x0 = x.replace(x0_from_v(x.feats, velocity.feats, query_t, double=True))
    if not bool(torch.isfinite(x0.feats).all().item()):
        raise RuntimeError("coarse-shape student produced non-finite values")
    device = coords.device
    std = slat_std.to(device=device, dtype=torch.float32)[None]
    mean = slat_mean.to(device=device, dtype=torch.float32)[None]
    return x0.replace(x0.feats.float() * std + mean), x0


@torch.no_grad()
def coarse_teacher(pipe, cond, coords, steps: int, rescale_t: float, seed: int = 0, guidance: bool = True):
    """Pixal3D's coarse-shape sampler (its own noise draw under ``seed``); ``guidance=False``:
    conditional prediction only (guidance strength 1)."""
    device = torch.device(pipe.device)
    params = {"steps": steps, "rescale_t": rescale_t}
    if not guidance:
        params["guidance_strength"] = 1.0
    with fork_cuda_rng(device):
        torch.manual_seed(seed)
        return pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_512"], coords, params)


# ---------------------------------------------------------------------------
# fine support
# ---------------------------------------------------------------------------
def _check_coords(coords: torch.Tensor, name: str, grid_resolution: Optional[int] = None, unique: bool = False):
    if not (torch.is_tensor(coords) and coords.ndim == 2 and int(coords.shape[1]) == 4 and int(coords.shape[0]) > 0):
        raise ValueError(f"{name} must be a non-empty [N, 4] tensor")
    if coords.dtype.is_floating_point:
        raise ValueError(f"{name} must be integer")
    if not bool((coords[:, 0] == 0).all().item()):
        raise ValueError(f"{name}: batch column must be zero")
    if grid_resolution is not None:
        xyz = coords[:, 1:]
        if not bool((xyz >= 0).all().item()) or not bool((xyz < int(grid_resolution)).all().item()):
            raise ValueError(f"{name}: coordinates outside the {grid_resolution} grid")
    if unique and int(torch.unique(coords, dim=0).shape[0]) != int(coords.shape[0]):
        raise ValueError(f"{name}: duplicate coordinates")


def select_fine_support(upsampled_coords, requested_resolution: int = 1536, max_num_tokens: int = 49_152
                        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Quantise the upsampled coarse support to the fine grid with Pixal3D's token cap:
    ``round((c + 0.5) / 512 * (grid - 1))``, de-duplicate, and lower the resolution by 128
    until fewer than ``max_num_tokens`` voxels remain or 1024 is reached."""
    requested_resolution = int(requested_resolution)
    if requested_resolution not in (1024, 1536):
        raise ValueError("requested_resolution must be 1024 or 1536")
    source = getattr(upsampled_coords, "coords", upsampled_coords)
    _check_coords(source, "upsampled coords")
    source = source.to(dtype=torch.int32)
    actual = requested_resolution
    while True:
        grid = actual // 16
        quantized = torch.cat([source[:, :1],
                               ((source[:, 1:].float() + 0.5) / float(SHAPE_LR_RESOLUTION) * float(grid - 1))
                               .round().to(torch.int32)], dim=1)
        coords = torch.unique(quantized, dim=0)
        n_tokens = int(coords.shape[0])
        cap_satisfied = n_tokens < max_num_tokens
        if cap_satisfied or actual == SHAPE_HR_MIN_RESOLUTION:
            break
        actual -= SHAPE_HR_RESOLUTION_STEP
    meta = {"requested_resolution": requested_resolution, "actual_resolution": actual, "grid_resolution": actual // 16,
            "max_num_tokens": max_num_tokens, "n_tokens": n_tokens, "cap_satisfied": bool(cap_satisfied)}
    _check_coords(coords, "fine coords", grid_resolution=meta["grid_resolution"], unique=True)
    return coords, meta


# ---------------------------------------------------------------------------
# fine level
# ---------------------------------------------------------------------------
def fine_initial_noise(num_tokens: int, seed: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    initial = torch.randn((int(num_tokens), LATENT_CHANNELS), dtype=torch.float32, device=device, generator=generator)
    state = generator.get_state().detach().cpu().clone()
    return initial, state


@torch.no_grad()
def fine_student(model, cond, coords: torch.Tensor, initial_noise: torch.Tensor, rng_state: torch.Tensor,
                 query_t: float) -> torch.Tensor:
    """One-step fine student; returns the normalised latent features [N, 32]."""
    if not (math.isfinite(query_t) and 0.0 < query_t <= 1.0):
        raise ValueError("query time must be in (0, 1]")
    if initial_noise.ndim != 2 or int(initial_noise.shape[0]) != int(coords.shape[0]):
        raise ValueError("initial noise must align with the fine support")
    renoise = torch.Generator(device=coords.device)  # unused for a single step (kept for the schedule)
    renoise.set_state(rng_state.detach().cpu())
    x = _sparse(initial_noise.clone(), coords)
    velocity = student_velocity(model, x, query_t, cond)
    if velocity.feats.shape != x.feats.shape:
        raise RuntimeError("fine-shape student changed the latent shape")
    x0 = x.replace(x0_from_v(x.feats, velocity.feats, query_t, double=True))
    if not bool(torch.isfinite(x0.feats).all().item()):
        raise RuntimeError("fine-shape student produced non-finite values")
    return x0.feats


@torch.no_grad()
def fine_teacher(pipe, cond_full, coords, initial_noise, steps: int, rescale_t: float,
                 guidance: bool = True) -> torch.Tensor:
    params = dict(pipe.shape_slat_sampler_params or {})
    params.update({"steps": steps, "rescale_t": rescale_t, "verbose": False})
    if not guidance:
        params["guidance_strength"] = 1.0
    noise = _sparse(initial_noise.clone(), coords)
    result = pipe.shape_slat_sampler.sample(pipe.models["shape_slat_flow_model_1024"], noise, **cond_full, **params)
    return result.samples.feats


def fork_cuda_rng(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return torch.random.fork_rng(devices=[])
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    return torch.random.fork_rng(devices=[index])
