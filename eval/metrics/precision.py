"""precision.py -- exact FP32 for the measurement instrument.

Some container images export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, so every float32 matmul silently runs in TF32 (inputs
rounded to a 10-bit mantissa).  render.world_to_cam is a GPU matmul on vertex positions, so ground-truth and
true_camera normal maps would be rendered from vertices quantised to ~1e-3 scene units (~0.3-0.5 px at 512^2): the
mask IoU between two renders of the same mesh drops to ~0.995 and flat face normals pick up a few degrees of noise.

`enforce_exact_fp32()` must run before the first CUDA matmul of the process: it removes the PyTorch override, sets
NVIDIA's library-level switch NVIDIA_TF32_OVERRIDE=0 and the PyTorch flags, and returns the state for corpus.json.
`matmul_probe()` measures the float32 GPU matmul error on camera-like magnitudes (~1e-7 when exact, ~1e-3 under TF32).
"""
from __future__ import annotations

import os

MAX_PROBE_ERROR = 1e-5


def enforce_exact_fp32() -> dict:
    state = {"env_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE_before": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
             "env_NVIDIA_TF32_OVERRIDE_before": os.environ.get("NVIDIA_TF32_OVERRIDE")}
    os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch
    state["cuda_initialized_before"] = bool(torch.cuda.is_initialized())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    state.update({"allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                  "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                  "float32_matmul_precision": torch.get_float32_matmul_precision(), "torch": torch.__version__})
    return state


def matmul_probe(device="cuda") -> dict:
    """max |float32 GPU matmul - float64 reference| on camera-like magnitudes; ~1e-7 when exact, ~1e-3 under TF32."""
    import torch
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return {"device": str(device), "max_abs_error": None}
    g = torch.Generator().manual_seed(0)
    v = (torch.rand(20000, 3, generator=g, dtype=torch.float64) - 0.5) + torch.tensor([0.3, -2.7, 0.9], dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.rand(3, 3, generator=g, dtype=torch.float64))
    ref = v @ q
    out = (v.float().to(device) @ q.float().to(device)).double().cpu()
    return {"device": str(device), "max_abs_error": float((out - ref).abs().max())}
