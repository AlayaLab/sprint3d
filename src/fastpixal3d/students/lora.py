"""Low-rank adapters for the Pixal3D flow transformers.

The students are the Pixal3D flows plus a rank-256 LoRA (alpha 256, i.e.
scaling 1) on every linear layer inside ``blocks`` except the adaLN
modulation layers: the attention projections (``to_qkv``, ``to_q``,
``to_kv``, ``to_out``), the projection-attention ``proj_linear`` and the MLP
layers. The MLP layers are ``SparseLinear`` modules that take a sparse tensor
instead of a dense one; :class:`LoRALinear` handles both call conventions.

Numerics (kept identical to the training code): the adapter runs in the
dtype of its own weights (fp32), on the input cast to that dtype, and its
output is cast back to the base layer's output dtype before the addition.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """``base(x) + B(A(x)) * scaling`` for a dense ``nn.Linear`` or a ``SparseLinear``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, device=None,
                 init: bool = True):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        device = device if device is not None else base.weight.device
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, device=device)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, device=device)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True
        if init:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def _delta(self, feats: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(feats))) * self.scaling

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        if hasattr(x, "feats"):  # SparseLinear: sparse tensor in, sparse tensor out
            d = self._delta(x.feats.to(self.lora_A.weight.dtype)).to(out.feats.dtype)
            return out.replace(out.feats + d)
        d = self._delta(x.to(self.lora_A.weight.dtype)).to(out.dtype)
        return out + d


def is_lora_target(name: str, module: nn.Module) -> bool:
    return (isinstance(module, nn.Linear) and "blocks." in name
            and "adaLN_modulation" not in name and "lora_" not in name)


def lora_target_names(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if is_lora_target(name, module)]


def apply_lora(model: nn.Module, rank: int = 256, alpha: float = 256.0, dropout: float = 0.0,
               init: bool = True) -> List[LoRALinear]:
    """Wrap every target linear layer of ``model`` in place; returns the adapters in module order."""
    adapters: List[LoRALinear] = []
    targets = [(n, m) for n, m in model.named_modules() if is_lora_target(n, m)]
    for name, module in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        leaf = name.rsplit(".", 1)[1]
        wrapped = LoRALinear(module, rank, alpha, dropout, device=module.weight.device, init=init)
        setattr(parent, leaf, wrapped)
        adapters.append(wrapped)
    return adapters


def lora_parameter_names(model: nn.Module) -> List[str]:
    """Names (relative to ``model``) of the adapter weights after :func:`apply_lora`."""
    return [name for name, _ in model.named_parameters() if ".lora_A." in name or ".lora_B." in name]


def bind_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> int:
    """Copy adapter weights (keys relative to ``model``) into the wrapped model. Strict on
    names, shapes and dtypes; returns the number of tensors copied."""
    params = dict(model.named_parameters())
    names = set(lora_parameter_names(model))
    missing = sorted(names - set(state))
    extra = sorted(set(state) - names)
    if missing or extra:
        raise KeyError(f"adapter weights do not match the model: missing={missing[:3]} (+{max(0, len(missing) - 3)}) "
                       f"extra={extra[:3]} (+{max(0, len(extra) - 3)})")
    with torch.no_grad():
        for name in sorted(names):
            source, target = state[name], params[name]
            if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
                raise ValueError(f"adapter tensor {name!r}: {tuple(source.shape)}/{source.dtype} does not match "
                                 f"{tuple(target.shape)}/{target.dtype}")
            target.copy_(source.to(device=target.device))
    return len(names)
