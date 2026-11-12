"""Teacher trajectories of the cache builders: Pixal3D's own samplers on the 12-step grid.

Each function runs one of Pixal3D's samplers (``pipeline.<stage>_sampler.sample``) for 12 Euler steps
with ``rescale_t = 3`` and returns the sampler's result (``.samples`` and ``.pred_x_t``, the states
after each step: ``pred_x_t[s - 1]`` is the state at grid index ``s``). The initial noise is drawn on
the CPU from a dedicated generator (:mod:`fastpixal3d.training.schedule`); everything stays in the
flows' normalised latent spaces.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

STEPS, RESCALE_T = 12, 3.0
STRUCTURE_GUIDANCE = {"guidance_strength": 7.5, "guidance_rescale": 0.7, "guidance_interval": [0.6, 1.0]}
FINE_SHAPE_GUIDANCE = {"guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0]}
TEXTURE_GUIDANCE = {"guidance_strength": 1.0, "guidance_rescale": 0.0, "guidance_interval": [0.6, 0.9]}


def structure_trajectory(pipeline, record: Mapping[str, Any], view: str, device):
    """Guided 12-step trajectory of the structure flow on a structure conditioning record
    (negative conditioning = zeros, Pixal3D's own)."""
    import torch

    from .schedule import structure_initial_noise

    g = record["global"].to(torch.float32).unsqueeze(0).to(device)
    p = record["proj"].to(torch.float32).unsqueeze(0).to(device)
    noise = structure_initial_noise(view).to(device)
    with torch.no_grad():
        return pipeline.sparse_structure_sampler.sample(
            pipeline.models["sparse_structure_flow_model"], noise.clone(), cond={"global": g, "proj": p},
            neg_cond={"global": torch.zeros_like(g), "proj": torch.zeros_like(p)},
            steps=STEPS, rescale_t=RESCALE_T, verbose=False, **STRUCTURE_GUIDANCE)


def fine_shape_trajectory(pipeline, record: Mapping[str, Any], view: str, device):
    """Guided 12-step trajectory of the fine (1024) shape flow on a fine-shape conditioning record."""
    import torch
    from pixal3d.modules.sparse import SparseTensor

    from .schedule import fine_shape_initial_noise

    coords = record["coords"].clone().to(device).int()
    coords[:, 0] = 0
    proj = record["proj"].to(torch.float32).to(device)
    g = record["global"].to(torch.float32).unsqueeze(0).to(device)
    cond = {"global": g, "proj": SparseTensor(feats=proj, coords=coords)}
    neg = {"global": torch.zeros_like(g), "proj": SparseTensor(feats=torch.zeros_like(proj), coords=coords)}
    noise = SparseTensor(feats=fine_shape_initial_noise(view, int(coords.shape[0])).to(device), coords=coords)
    with torch.no_grad():
        return pipeline.shape_slat_sampler.sample(
            pipeline.models["shape_slat_flow_model_1024"], noise, cond=cond, neg_cond=neg,
            steps=STEPS, rescale_t=RESCALE_T, verbose=False, **FINE_SHAPE_GUIDANCE)


def texture_trajectory(pipeline, coords, concat_cond, cond: Dict[str, Any], device):
    """12-step trajectory of the texture flow (conditional only) for a fine support ``coords``, the
    normalised fine-shape latent ``concat_cond`` [N, 32] and the texture conditioning ``cond``."""
    import torch
    from pixal3d.modules.sparse import SparseTensor

    from .schedule import texture_initial_noise

    concat = SparseTensor(feats=concat_cond.to(device), coords=coords)
    noise = concat.replace(feats=texture_initial_noise(int(coords.shape[0])).to(device))
    neg = {k: (v.replace(torch.zeros_like(v.feats)) if hasattr(v, "feats") else torch.zeros_like(v))
           for k, v in cond.items()}
    with torch.no_grad():
        return pipeline.tex_slat_sampler.sample(
            pipeline.models["tex_slat_flow_model_1024"], noise, concat_cond=concat, cond=cond, neg_cond=neg,
            steps=STEPS, rescale_t=RESCALE_T, verbose=False, **TEXTURE_GUIDANCE)
