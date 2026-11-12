"""Students for training: a frozen Pixal3D flow plus trainable rank-256 adapters.

Kinds: ``structure`` (dense structure flow), ``texture`` (texture flow with ``concat_cond``), ``coarse_shape``
and ``fine_shape`` (the sparse shape flows; their consistency distillation also uses the frozen flow as the
teacher, see :class:`ShapeStudentNetwork`).

Parameter names. The flow is held as ``backbone.model`` inside the student network, so the adapter
parameters are named ``backbone.model.blocks.<i>.<layer>.lora_{A,B}.weight`` -- the key convention
of the released weights (``fastpixal3d.students.convert``).

Initialisation (kept identical to the runs that produced the released students). The training
script seeds the global generator (``torch.manual_seed(seed)``), builds the flow with
``pixal3d.models.from_pretrained`` (its constructor draws the initial weights before the pretrained
ones are loaded) and then wraps the target layers in module order. Each adapter is created on the
CPU -- both ``nn.Linear`` constructors draw their default initialisation, then ``lora_A`` is
re-drawn (Kaiming uniform, a = sqrt(5)) and ``lora_B`` is zeroed -- and only then moved to the
flow's device. Creating the adapters directly on the GPU would draw from the CUDA generator and
give different (equally valid) initial ``lora_A`` values. ``lora_B = 0`` makes the student equal
to the teacher at step 0.

Numerics of the forward (unchanged from the training runs): the adapters hold float32 weights;
inside the bfloat16 autocast of the training step their matmuls run in bfloat16, and their output
is cast to the base layer's output dtype before it is added.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..students.lora import LoRALinear, is_lora_target

STRUCTURE_SUBMODEL = "ckpts/ss_flow_img_dit_1_3B_64_bf16"
TEXTURE_SUBMODEL = "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
COARSE_SHAPE_SUBMODEL = "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
FINE_SHAPE_SUBMODEL = "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"


def load_base_flow(model_dir: str | Path, submodel: str, device) -> nn.Module:
    """``pixal3d.models.from_pretrained(<model_dir>/<submodel>)``, frozen, in eval mode, on ``device``."""
    from pixal3d import models

    flow = models.from_pretrained(str(Path(model_dir) / submodel))
    flow.to(device)
    flow.requires_grad_(False)
    flow.eval()
    return flow


def attach_adapters(model: nn.Module, rank: int = 256, alpha: float = 256.0, dropout: float = 0.0,
                    init_device: str = "cpu") -> List[LoRALinear]:
    """Wrap every target layer (``nn.Linear`` under ``blocks.`` except the adaLN modulation) in a
    :class:`LoRALinear`, in module order, initialised on ``init_device`` (the CPU, as in training;
    ``"meta"`` only to inspect shapes) and then moved to the layer's device."""
    adapters: List[LoRALinear] = []
    targets = [(name, module) for name, module in model.named_modules() if is_lora_target(name, module)]
    for name, module in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        leaf = name.rsplit(".", 1)[1]
        wrapped = LoRALinear(module, rank, alpha, dropout, device=torch.device(init_device), init=True)
        wrapped = wrapped.to(module.weight.device)
        setattr(parent, leaf, wrapped)
        adapters.append(wrapped)
    return adapters


def enable_block_checkpointing(model: nn.Module) -> int:
    """Activation checkpointing in every transformer block (Pixal3D's own ``use_checkpoint`` path;
    the recomputation is deterministic because the adapters have no dropout)."""
    import torch.utils.checkpoint  # noqa: F401  (the blocks call torch.utils.checkpoint.checkpoint)

    blocks = list(model.blocks)
    for block in blocks:
        block.use_checkpoint = True
    return len(blocks)


def timestep_input(t, batch: int, device) -> torch.Tensor:
    """Pixal3D's timestep input ``1000 * t``, computed as in training: ``t`` cast to float32 first,
    then multiplied by 1000 in float32."""
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1 and batch > 1:
        t = t.expand(batch)
    return t * 1000.0


class _Holder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class StudentNetwork(nn.Module):
    """Frozen flow + adapters; ``trainable()`` returns the adapter parameters by name."""

    def __init__(self, flow: nn.Module, adapters: List[LoRALinear]):
        super().__init__()
        self.backbone = _Holder(flow)
        self.adapters = adapters  # plain list: not registered twice

    def trainable(self) -> Dict[str, nn.Parameter]:
        return {name: p for name, p in self.named_parameters() if p.requires_grad}


class StructureStudentNetwork(StudentNetwork):
    """The structure flow is dense (``[1, 8, 16, 16, 16]``); training samples are kept as the
    ``[4096, 8]`` token matrix of the full 16^3 grid in C order, as stored in the anchors. The
    conversion uses index assignment / advanced indexing like the training runs (the same autograd
    operations, hence the same gradient memory layouts and kernels)."""

    resolution = 16

    def __init__(self, flow: nn.Module, adapters: List[LoRALinear]):
        super().__init__(flow, adapters)
        self._index = {}

    def grid_index(self, device) -> torch.Tensor:
        from .schedule import full_grid_index

        key = str(device)
        if key not in self._index:
            self._index[key] = full_grid_index(self.resolution, device)
        return self._index[key]

    def forward(self, tokens: torch.Tensor, t, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        from .schedule import full_grid_gather, full_grid_scatter

        index = self.grid_index(tokens.device)
        dense = full_grid_scatter(tokens, index, self.resolution)
        velocity = self.backbone.model(dense, timestep_input(t, 1, tokens.device), cond)
        return full_grid_gather(velocity, index)


class TextureStudentNetwork(StudentNetwork):
    """The texture flow takes a sparse latent and the normalised fine-shape latent as ``concat_cond``."""

    def forward(self, x, t, cond, concat_cond):
        return self.backbone.model(x, timestep_input(t, int(x.shape[0]), x.feats.device), cond,
                                   concat_cond=concat_cond)


class ShapeStudentNetwork(StudentNetwork):
    """The coarse- and fine-shape flows take a sparse latent ``[N, 32]`` on the object's support and the
    image conditioning ``{"global": [1, 5, 1024], "proj": SparseTensor [N, 2048]}`` on the same coordinates.
    The same flow with the adapters switched off is the frozen teacher of the consistency distillation
    (:meth:`teacher`); the original code held a second copy of the flow for that, with the same result."""

    def forward(self, x, t, cond):
        return self.backbone.model(x, timestep_input(t, int(x.shape[0]), x.feats.device), cond, concat_cond=None)

    @contextlib.contextmanager
    def adapters_disabled(self):
        previous = [adapter.enabled for adapter in self.adapters]
        for adapter in self.adapters:
            adapter.enabled = False
        try:
            yield
        finally:
            for adapter, enabled in zip(self.adapters, previous):
                adapter.enabled = enabled

    def teacher(self, x, t, cond):
        with self.adapters_disabled():
            return self.forward(x, t, cond)


def build_student(kind: str, model_dir: str | Path, submodel: str, device, *, rank: int = 256, alpha: float = 256.0,
                  dropout: float = 0.0, gradient_checkpointing: bool = True, seed: Optional[int] = None) -> StudentNetwork:
    """Seed (if given) -> base flow -> adapters -> student network, in the order of the training runs."""
    if seed is not None:
        torch.manual_seed(int(seed))
    flow = load_base_flow(model_dir, submodel, device)
    if gradient_checkpointing and kind == "structure":
        enable_block_checkpointing(flow)
    adapters = attach_adapters(flow, rank, alpha, dropout)
    if gradient_checkpointing and kind != "structure":
        enable_block_checkpointing(flow)
    cls = {"structure": StructureStudentNetwork, "texture": TextureStudentNetwork,
           "coarse_shape": ShapeStudentNetwork, "fine_shape": ShapeStudentNetwork}[kind]
    return cls(flow, adapters)
