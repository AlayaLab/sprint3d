"""Training checkpoints.

A checkpoint is one ``torch.save`` of a dict::

    schema, schema_version   format identifiers (structure: version 1, texture: version 2,
                             coarse shape: version 3, fine shape: version 4)
    completed_updates        number of optimizer updates done
    world_size               number of processes of the run
    gen                      {parameter name: float32 CPU tensor}  -- the trained adapters
    ema                      {"shadow": {name: float32 CPU tensor}, "n": updates, "updates": updates}
                             (the consistency trainers ramp the EMA decay with ``n``, see consistency.py)
    opt_gen                  AdamW state_dict (parameters indexed in the insertion order of ``gen``)
    provenance               the settings of the run (rank, alpha, learning rate, grid, data, seed, ...)

Names are ``backbone.model.blocks.<i>.<layer>.lora_{A,B}.weight``. The released weight files hold
``ema["shadow"]`` unchanged; ``fastpixal3d.students.convert`` writes them from a checkpoint.
Files are written to a temporary name and renamed, so a checkpoint is either complete or absent.
"""
from __future__ import annotations
import pickle

import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

STRUCTURE_SCHEMA = ("structure_trajectory_student", 1)
TEXTURE_SCHEMA = ("texture_trajectory_student", 2)
COARSE_SHAPE_SCHEMA = ("coarse_shape_consistency_student", 3)
FINE_SHAPE_SCHEMA = ("fine_shape_consistency_student", 4)
_NAME = re.compile(r"^completed_updates_(\d{12})\.pt$")


def checkpoint_name(updates: int) -> str:
    return f"completed_updates_{int(updates):012d}.pt"


def make_payload(*, schema, params: Mapping[str, torch.nn.Parameter], ema_shadow: Mapping[str, torch.Tensor],
                 optimizer: torch.optim.Optimizer, completed_updates: int, world_size: int,
                 provenance: Mapping[str, Any]) -> Dict[str, Any]:
    names = list(params)
    return {
        "schema": schema[0], "schema_version": schema[1],
        "completed_updates": int(completed_updates), "world_size": int(world_size),
        "gen": {n: p.detach().float().cpu().clone() for n, p in params.items()},
        "ema": {"shadow": {n: ema_shadow[n].detach().float().cpu().clone() for n in names},
                "n": int(completed_updates), "updates": int(completed_updates)},
        "opt_gen": optimizer.state_dict(),
        "provenance": dict(provenance),
    }


def save(payload: Mapping[str, Any], directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / checkpoint_name(payload["completed_updates"])
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(dict(payload), str(tmp))
    os.replace(tmp, path)
    return path


def newest(directory: str | Path) -> Optional[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    found = sorted(p for p in directory.iterdir() if _NAME.match(p.name))
    return found[-1] if found else None


def load(path: str | Path, trust: Optional[bool] = None) -> Dict[str, Any]:
    """Load a checkpoint on the CPU with restricted deserialization by default.

    ``weights_only=True`` admits tensors and plain containers, which is all this project's
    checkpoints hold (parameters, EMA shadow, optimizer state, provenance strings). A file that
    holds other Python objects is unpickled in full only when ``trust`` is true or the environment
    variable ``FASTPIXAL3D_TRUST_CHECKPOINT=1`` is set: full unpickling can execute code, so do that only for
    checkpoints you trust. PyTorch 2.6 or newer is required (CVE-2025-32434)."""
    if trust is None:
        trust = os.environ.get("FASTPIXAL3D_TRUST_CHECKPOINT") == "1"
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if not trust:
            raise RuntimeError(f"{path} holds objects other than tensors and plain containers; set "
                               f"FASTPIXAL3D_TRUST_CHECKPOINT=1 to load it with full unpickling if you trust the file") from error
        return torch.load(str(path), map_location="cpu", weights_only=False)


def restore(payload: Mapping[str, Any], params: Mapping[str, torch.nn.Parameter], ema_shadow: Dict[str, torch.Tensor],
            optimizer: Optional[torch.optim.Optimizer]) -> int:
    """Copy adapters, EMA and optimizer state into a run; returns ``completed_updates``.
    Names and shapes must match exactly. The EMA shadow (saved on the CPU) is moved to the device of
    its parameter."""
    gen = payload["gen"]
    names = list(params)
    if set(gen) != set(names):
        missing, extra = sorted(set(names) - set(gen)), sorted(set(gen) - set(names))
        raise KeyError(f"checkpoint adapters do not match the model: missing={missing[:3]} extra={extra[:3]}")
    with torch.no_grad():
        for n, p in params.items():
            if tuple(gen[n].shape) != tuple(p.shape):
                raise ValueError(f"{n}: checkpoint shape {tuple(gen[n].shape)} != {tuple(p.shape)}")
            p.copy_(gen[n].to(device=p.device, dtype=p.dtype))
    shadow = (payload.get("ema") or {}).get("shadow")
    if isinstance(shadow, Mapping):
        for n in names:
            ema_shadow[n] = shadow[n].float().to(params[n].device).clone()
    elif isinstance(shadow, (list, tuple)):  # positional shadow, in the insertion order of ``gen``
        for n, t in zip(list(gen), shadow):
            ema_shadow[n] = t.float().to(params[n].device).clone()
    if optimizer is not None and "opt_gen" in payload:
        optimizer.load_state_dict(payload["opt_gen"])
    return int(payload.get("completed_updates", 0))
