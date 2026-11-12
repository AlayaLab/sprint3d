"""One-step trajectory loss and the optimizer update.

For a segment ``(i, j)`` of the student grid with noise fractions ``t_i > t_j``:

    v       = student(x_i, t_i)                       (bfloat16 autocast)
    x_pred  = x_i - (t_i - t_j) * v                   (float32; Pixal3D's Euler update)
    loss    = mean((x_pred - x_j)^2)                  (x_j: the teacher's state at index j)

One optimizer update averages the gradients of ``grad_accum`` micro-batches per process (each
micro-batch loss is divided by ``grad_accum`` before ``backward``) and, with several processes,
sums the gradients over processes and divides by their number. The effective batch is
``world_size * grad_accum`` samples; 1 process with ``grad_accum = 4`` and 4 processes with
``grad_accum = 1`` average the same number of independent samples. The update then clips the
global gradient norm, takes one AdamW step and updates the exponential moving average (EMA) of the
adapter weights, which is what is released.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


def euler_step(x: torch.Tensor, v: torch.Tensor, t: float, t_next: float) -> torch.Tensor:
    """``x_{t'} = x_t - (t - t') * v``."""
    return x - (float(t) - float(t_next)) * v


def trajectory_loss(x_i: torch.Tensor, v: torch.Tensor, t_i: float, t_j: float, target: torch.Tensor) -> torch.Tensor:
    pred = euler_step(x_i.float(), v.float(), t_i, t_j)
    return F.mse_loss(pred, target.float())


@dataclass
class MicroBatch:
    """One training sample: the student input ``x_i`` at ``t_i``, the teacher target at ``t_j`` and
    whatever else the student's forward needs (``inputs``)."""

    view: str
    i: int
    j: int
    t_i: float
    t_j: float
    x_i: torch.Tensor
    target: torch.Tensor
    inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateResult:
    loss: float                 # mean of the micro-batch losses
    grad_norm: float            # global norm before clipping
    losses: List[float]


def make_optimizer(params: Mapping[str, torch.nn.Parameter], lr: float, betas: Sequence[float],
                   weight_decay: float) -> torch.optim.Optimizer:
    """AdamW over the adapter parameters in their insertion order (that order indexes the
    optimizer state in a checkpoint)."""
    return torch.optim.AdamW(list(params.values()), lr=float(lr), betas=(float(betas[0]), float(betas[1])),
                             weight_decay=float(weight_decay))


def init_ema(params: Mapping[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().float().clone() for name, p in params.items()}


@torch.no_grad()
def ema_update(shadow: Dict[str, torch.Tensor], params: Mapping[str, torch.nn.Parameter], decay: float) -> None:
    for name, p in params.items():
        shadow[name].mul_(decay).add_(p.detach().float(), alpha=1.0 - decay)


def average_gradients(params: Mapping[str, torch.nn.Parameter], world_size: int) -> None:
    """Sum the gradients over processes and divide by their number (only the adapters have
    gradients, so the frozen base model is not wrapped in DistributedDataParallel)."""
    if world_size <= 1:
        return
    import torch.distributed as dist

    for p in params.values():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def run_update(forward: Callable[[MicroBatch], torch.Tensor], micro_batches: Sequence[Any],
               params: Mapping[str, torch.nn.Parameter], optimizer: torch.optim.Optimizer,
               ema_shadow: Dict[str, torch.Tensor], *, grad_clip: float, ema_decay: float,
               prepare: Optional[Callable[[Any], MicroBatch]] = None, world_size: int = 1,
               autocast_dtype: Optional[torch.dtype] = torch.bfloat16, device_type: str = "cuda") -> UpdateResult:
    """One optimizer update over ``micro_batches`` (``len(micro_batches)`` = gradient accumulation).

    ``prepare(item)`` (optional) turns an item into a :class:`MicroBatch` just before its forward,
    so only one micro-batch is held in memory at a time. ``forward(mb)`` returns the student's
    velocity for ``mb.x_i`` (a ``[N, C]`` tensor) and runs under autocast; the loss is computed
    outside autocast in float32.
    """
    n = len(micro_batches)
    if n < 1:
        raise ValueError("an update needs at least one micro-batch")
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    for item in micro_batches:
        mb = prepare(item) if prepare is not None else item
        with torch.autocast(device_type=device_type, dtype=autocast_dtype or torch.float32,
                            enabled=autocast_dtype is not None):
            v = forward(mb)
        loss = trajectory_loss(mb.x_i, v, mb.t_i, mb.t_j, mb.target)
        (loss / n).backward()
        losses.append(float(loss.detach().item()))
    average_gradients(params, world_size)
    grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), float(grad_clip))
    optimizer.step()
    ema_update(ema_shadow, params, float(ema_decay))
    return UpdateResult(loss=sum(losses) / n, grad_norm=float(grad_norm), losses=losses)


def draw_segments(rng, views: Sequence[str], pairs: Sequence[tuple], count: int) -> List[tuple]:
    """``count`` draws of ``(view, (i, j))`` from ``random.Random`` ``rng``: per draw one
    ``rng.choice(views)`` followed by one ``rng.randrange(len(pairs))`` (this order fixes the
    sample sequence of a seed)."""
    out = []
    for _ in range(int(count)):
        view = rng.choice(views)
        pair = pairs[rng.randrange(len(pairs))]
        out.append((view, pair))
    return out
