"""Process settings and access to the Pixal3D code and weights for the training scripts.

Call :func:`configure_process` before torch is imported. The settings are those of the training
runs: FlashAttention for Pixal3D's dense and sparse attention, expandable CUDA allocator segments,
and ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` (float32 cuBLAS matmuls may use TF32). The runs did not
set the last variable themselves; the container image they ran in sets it, so it is part of their
numerics. ``tf32_override: false`` removes it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from ..weights import WeightPaths


def configure_process(threads: int = 4, tf32_override: bool = True, deterministic_cublas: bool = False) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(int(threads))
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("TQDM_DISABLE", "1")
    if tf32_override:
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    else:
        os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    if deterministic_cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def weight_paths(model: Optional[str] = None, source: Optional[str] = None) -> WeightPaths:
    """Weight locations from the configuration, falling back to the ``FASTPIXAL3D_*`` variables."""
    return WeightPaths.from_env(pixal3d=model, pixal3d_src=source)


def import_pixal3d(weights: WeightPaths) -> Path:
    """Put the Pixal3D source checkout on ``sys.path`` (``import pixal3d`` / ``import inference``);
    returns the local model directory (downloaded at the pinned revision if given a repository id)."""
    weights.apply_environment()
    source = str(weights.source_dir())
    if source not in sys.path:
        sys.path.insert(0, source)
    return Path(weights.model_dir())


def toolkit_utils(weights: WeightPaths):
    """Pixal3D's ``data_toolkit/utils.py`` (camera and mesh-normalisation helpers)."""
    import importlib.util

    path = weights.source_dir() / "data_toolkit" / "utils.py"
    spec = importlib.util.spec_from_file_location("pixal3d_data_toolkit_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline(weights: WeightPaths, device: str = "cuda"):
    """Pixal3D's full pipeline (``inference.init_pipeline``): flows, decoders and the DINOv3
    conditioning extractors, all on ``device``."""
    model_dir = import_pixal3d(weights)
    cache = os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH")
    import flex_gemm  # noqa: F401  (binds the autotune cache path before Pixal3D's inference.py sets its own)
    import inference

    if cache:
        os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = cache
    os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "0"
    for key in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
        del sys.modules[key]  # the feature upsampler's torch.hub code imports a top-level package named "src"
    from ..auxiliary import use_pinned_auxiliary_models

    use_pinned_auxiliary_models(inference, weights.root_dir())
    return inference.init_pipeline(str(model_dir), device=device, low_vram=False)


def load_moge(weights: WeightPaths, device: str = "cuda"):
    """Pixal3D's ``inference`` module and its MoGe-2 camera estimator from the pinned weights (call after
    :func:`load_pipeline`); the pair the in-the-wild input path of the shape caches needs."""
    import inference

    from ..auxiliary import use_pinned_auxiliary_models

    aux = use_pinned_auxiliary_models(inference, weights.root_dir())
    return inference, inference.load_moge_model(device=device, model_name=aux["moge2_model"])


def deterministic_torch() -> None:
    """Deterministic algorithms, no TF32 in PyTorch's own flags, highest float32 matmul precision
    (the numeric profile the structure-conditioning cache was built with; needs
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` before CUDA starts)."""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
