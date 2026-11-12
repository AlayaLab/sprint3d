"""Attention backend selection for the Pixal3D flow transformers.

Pixal3D reads two process-wide switches at call time: the dense attention
backend (``pixal3d.modules.attention.config.BACKEND``) and the sparse/varlen
one (``pixal3d.modules.sparse.config.ATTN``). The deployed configuration runs
everything on FlashAttention-2 except the fine-shape flow and the texture
flow, which use FlashAttention-3 (Hopper only). :func:`attention_backend`
switches both flags for the duration of a ``with`` block and restores the
previous values afterwards.

FlashAttention-3 needs a Hopper GPU (sm_90) and the ``flash_attn_3``
build of the FlashAttention repository (``flash_attn_interface`` module). On
other GPUs, or with ``Config.attention_fallback == "fa2"``, the FA3 levers run
on FlashAttention-2 instead (same results up to rounding, slower).
"""
from __future__ import annotations

import contextlib
import importlib
from typing import Optional

FA2 = "flash_attn"
FA3 = "flash_attn_3"

_FA3_STATUS: Optional[str] = None


def fa3_status() -> str:
    """``"ok"`` if FlashAttention-3 can be used here, else the reason it cannot."""
    global _FA3_STATUS
    if _FA3_STATUS is None:
        try:
            import torch

            if not torch.cuda.is_available():
                _FA3_STATUS = "no CUDA device"
            elif torch.cuda.get_device_capability() != (9, 0):
                _FA3_STATUS = f"compute capability {torch.cuda.get_device_capability()} is not sm_90"
            else:
                importlib.import_module("flash_attn_interface")
                _FA3_STATUS = "ok"
        except Exception as error:  # ImportError or a broken build
            _FA3_STATUS = f"flash_attn_interface not importable ({type(error).__name__}: {error})"
    return _FA3_STATUS


def fast_backend(fallback: Optional[str] = None) -> str:
    """The backend the FA3 levers use: FA3 when available and not overridden, else FA2."""
    if fallback == "fa2":
        return FA2
    if fallback not in (None, "auto"):
        raise ValueError(f"attention_fallback must be None, 'auto' or 'fa2', got {fallback!r}")
    return FA3 if fa3_status() == "ok" else FA2


def _configs():
    import pixal3d.modules.attention.config as dense_config
    import pixal3d.modules.sparse.config as sparse_config

    return dense_config, sparse_config


def current_backends():
    dense, sparse = _configs()
    return dense.BACKEND, sparse.ATTN


@contextlib.contextmanager
def attention_backend(name: str):
    """Set both Pixal3D attention switches to ``name`` inside the block."""
    dense, sparse = _configs()
    previous = (dense.BACKEND, sparse.ATTN)
    dense.set_backend(name)
    sparse.set_attn_backend(name)
    try:
        yield
    finally:
        dense.set_backend(previous[0])
        sparse.set_attn_backend(previous[1])
