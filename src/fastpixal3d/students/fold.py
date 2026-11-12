"""Folding a LoRA into the base weights.

Fold rule (the one the reported numbers use):

    W_bf16 <- W_bf16 + bf16( (B_fp32 @ A_fp32) * scaling )

i.e. the low-rank product is formed in fp32 on the device that holds the
adapter, cast to the base weight's dtype (bf16), and added in bf16. The
wrapper is then replaced by the bare base layer. Rounding the fp32 delta to
bf16 loses part of it (for the fine-shape student a large part), which is why
folding is a separate, switchable lever for the fine-shape student
(``fold_lora_fine_shape``) and is always applied to the texture student. The
structure and coarse-shape students are never folded (they keep their fp32
adapters).

Note: when the product is computed on a GPU, it follows the process's fp32
matmul precision (``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` makes it TF32), like
every other fp32 matmul in the pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .lora import LoRALinear


def lora_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """fp32 low-rank product ``(B @ A) * scaling``."""
    return (lora_b.float() @ lora_a.float()) * float(scaling)


def fold_into(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """In-place ``weight += delta.to(weight.dtype)``; returns ``weight``."""
    delta = lora_delta(lora_a, lora_b, scaling)
    weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
    return weight


def fold_lora(model: nn.Module) -> int:
    """Fold every :class:`LoRALinear` inside ``model`` into its base layer and put the
    bare base layer back in its place. Returns the number of folded adapters."""
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    with torch.no_grad():
        for name, module in targets:
            base = module.base
            fold_into(base.weight, module.lora_A.weight, module.lora_B.weight, module.scaling)
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], base)
    return len(targets)


def lost_fraction(model_before: nn.Module) -> float:
    """Diagnostic: fraction of the adapters' delta norm lost to bf16 rounding
    (computed on a model that still has its adapters; does not modify it)."""
    total, lost = 0.0, 0.0
    with torch.no_grad():
        for module in model_before.modules():
            if not isinstance(module, LoRALinear):
                continue
            delta = lora_delta(module.lora_A.weight, module.lora_B.weight, module.scaling)
            w = module.base.weight
            folded = (w + delta.to(device=w.device, dtype=w.dtype)).float() - w.float()
            total += float(delta.norm()) ** 2
            lost += float((delta.to(folded.device) - folded).norm()) ** 2
    return (lost / total) ** 0.5 if total > 0 else 0.0
