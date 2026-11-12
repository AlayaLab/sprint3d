"""One-step consistency distillation (CD) of the coarse- and fine-shape students.

The released coarse-shape and fine-shape students were not trained by the trajectory distillation of the
structure and texture students but by consistency distillation, the ``cd_init`` stage of the original
training code. Per update and per process, one object (its cached support and image conditioning):

    anchor   x0 = the teacher's own 12-step sample on the object's support (Euler, rescale_t 3, guidance
             7.5 / rescale 0.5 / interval [0.6, 1.0]) from a noise drawn once per object from a seed derived
             from the run seed and the object id; computed once per object and kept in CPU memory
    time     n uniform in {1, ..., N - 1}, N = 50:  t_n = 1 - n / N (student),  t_next = t_n - 1 / N (target)
    x_n      = (1 - t_n) x0 + (s + (1 - s) t_n) eps,   eps ~ N(0, 1),  s = 1e-5 (Pixal3D's flow)
    x_next   = x_n - (t_n - t_next) v_teacher(x_n, t_n)          one guided Euler step of the teacher
    target   = x0_from_v(x_next, v_ema(x_next, t_next), t_next)  the student with its EMA weights, no gradient
    pred     = x0_from_v(x_n, v_student(x_n, t_n), t_n)
    loss     = 80 * mean((pred - target)^2)                      in float64

with Pixal3D's ``x0_from_v(x, v, t) = (1 - s) x - (s + (1 - s) t) v`` evaluated in float64 (the same function
the released pipeline uses for the one-step students, :mod:`fastpixal3d.stages.shape`). The student runs
conditional-only; the teacher is the same frozen flow with the adapters disabled. There is no autocast: the
flow's bfloat16 blocks and the float32 adapters run as they are. After the backward the gradients are
averaged over processes, clipped to global norm 10, one AdamW step is taken (learning rate 2e-6, betas
(0, 0.999), weight decay 1e-3) and the EMA of the adapters is updated with the ramped decay
``d = min(0.99, (1 + n) / (10 + n))``, ``n`` = EMA updates so far (0.1 at the first update, 0.99 from the
891st on). The released weights are the EMA.

Data order: the objects with a record are sorted by name and visited in the order of PyTorch's
``DistributedSampler(shuffle=True, drop_last=True, seed=run seed)`` (:func:`distributed_order`), one object
per process per update, re-shuffled at every epoch. Random draws of an update (the time index and the noise)
come from a generator seeded from the run seed, the update number, the process rank and the object id
(:func:`update_seed`), so a resumed run repeats the draws it would have made; the original runs drew them
from the process-global generator instead (not reproducible), see docs/TRAINING.md.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import checkpoint as ckpt
from . import config as C
from .objective import average_gradients, init_ema, make_optimizer
from .schedule import SIGMA_MIN, teacher_t_grid
from ..stages.shape import x0_from_v

NUM_TIMESTEPS = 50          # the uniform time grid of the consistency step (h = 1 / 50)
LOSS_WEIGHT = 80.0          # the loss scale of the original runs (``ode_weight``)
ANCHOR_STEPS = 12           # the teacher's sampling steps for the anchor
ANCHOR_RESCALE_T = 3.0
GUIDANCE = {"guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": (0.6, 1.0)}
EMA_RAMP_OFFSET = 10.0      # d = min(decay, (1 + n) / (EMA_RAMP_OFFSET + n))
LATENT_CHANNELS = 32

_SEED_PERSON = b"slatd.draw.v1"   # kept verbatim: it makes the anchors of the released runs reproducible
_MAX_SEED = 1 << 63


# ------------------------------------------------------------------------------------------ seeds
def draw_seed(run_seed: int, completed_updates: int, rank: int, object_id: str, role: str, draw_index: int) -> int:
    """63-bit seed of one random draw, a blake2b digest of its complete identity (the original code's
    derivation, so that the anchors of the released runs are regenerated bit for bit)."""
    fields = [int(run_seed), int(completed_updates), int(rank), str(object_id), str(role), int(draw_index)]
    payload = json.dumps(fields, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=_SEED_PERSON).digest()
    return int.from_bytes(digest, "big", signed=False) & (_MAX_SEED - 1)


def anchor_seed(run_seed: int, object_id: str) -> int:
    """Seed of the noise the teacher anchor of an object starts from (independent of rank and update)."""
    return draw_seed(run_seed, 0, 0, object_id, "cd_anchor", 0)


def update_seed(run_seed: int, completed_updates: int, rank: int, object_id: str) -> int:
    """Seed of the draws of one update (time index, then noise) on one process."""
    return draw_seed(run_seed, completed_updates, rank, object_id, "cd_update", 0)


def make_generator(device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


# ------------------------------------------------------------------------------------- flow math
def _bcast(t, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)
    if t.ndim == 0:
        t = t.reshape(1, 1).expand(x.shape[0], 1)
    elif t.ndim == 1:
        t = t.reshape(-1, 1)
    return t


def expand_t(t: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Per-sample ``t`` [B] -> per-token [N, 1] through the batch column of ``coords``."""
    return t.reshape(-1)[coords[:, 0].long()].reshape(-1, 1)


def add_noise(x0: torch.Tensor, eps: torch.Tensor, t) -> torch.Tensor:
    """``x_t = (1 - t) x0 + (s + (1 - s) t) eps`` (Pixal3D's forward process)."""
    tt = _bcast(t, x0)
    return (1.0 - tt) * x0 + (SIGMA_MIN + (1.0 - SIGMA_MIN) * tt) * eps


def euler_step(xt: torch.Tensor, v: torch.Tensor, t, t_prev) -> torch.Tensor:
    """``x_{t'} = x_t - (t - t') v``."""
    tt = _bcast(t, xt)
    tp = _bcast(t_prev, xt)
    return xt - (tt - tp) * v


def v_from_x0(xt: torch.Tensor, x0: torch.Tensor, t_tok: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`x0_from_v`: ``v = ((1 - s) x_t - x0) / (s + (1 - s) t)``."""
    return ((1.0 - SIGMA_MIN) * xt - x0) / (SIGMA_MIN + (1.0 - SIGMA_MIN) * t_tok)


def time_pair(n_idx: torch.Tensor, num_timesteps: int = NUM_TIMESTEPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """``n_idx``: float32 tensor [B] with values in {1, ..., N - 1}. Returns ``t_n = 1 - n / N`` (the student's,
    noisier) and ``t_next = max(t_n - 1 / N, 0)`` (the target's, one step cleaner), both float32 [B], computed in
    float32 as the original code did (for ``n = N - 1`` the target time is about 4e-8, not exactly 0)."""
    h = 1.0 / float(num_timesteps)
    t_n = 1.0 - n_idx * h
    t_next = (t_n - h).clamp_min(0.0)
    return t_n, t_next


def draw_time_index(generator: torch.Generator, device, num_timesteps: int = NUM_TIMESTEPS) -> torch.Tensor:
    """One uniform draw of ``n`` in {1, ..., N - 1} as a float32 tensor [1] (the first draw of an update)."""
    return torch.randint(1, int(num_timesteps), (1,), device=device, generator=generator).float()


def draw_noise_like(generator: torch.Generator, feats: torch.Tensor) -> torch.Tensor:
    """The noise of an update (the second draw), shaped like the anchor."""
    return torch.randn(feats.shape, device=feats.device, dtype=feats.dtype, generator=generator)


# --------------------------------------------------------------------------------- conditioning
def neg_cond_like(cond: Mapping[str, Any]) -> Dict[str, Any]:
    """Pixal3D's unconditional context: zeros for ``global`` and a zero-feature sparse tensor for ``proj``."""
    neg = {}
    for key, value in cond.items():
        if hasattr(value, "feats"):
            neg[key] = value.replace(torch.zeros_like(value.feats))
        elif torch.is_tensor(value):
            neg[key] = torch.zeros_like(value)
        else:
            neg[key] = value
    return neg


def _segment_std(feats: torch.Tensor, bidx: torch.Tensor, batch: int) -> torch.Tensor:
    out = feats.new_zeros(batch)
    for b in range(batch):
        mask = bidx == b
        if mask.any():
            out[b] = feats[mask].float().std()
    return out


def guided_velocity(forward: Callable, x, t: torch.Tensor, cond, neg_cond, *, guidance_strength: float = 7.5,
                    guidance_rescale: float = 0.5, guidance_interval: Optional[Tuple[float, float]] = (0.6, 1.0)):
    """The teacher's guided velocity, Pixal3D's classifier-free guidance with rescaling and interval as the
    original training code transcribed it: ``pred = gs v_pos + (1 - gs) v_neg``; the predicted clean latent of
    ``pred`` is rescaled to the per-sample standard deviation of the conditional one and blended with weight
    ``guidance_rescale``; outside ``guidance_interval`` the conditional velocity is used (and the unconditional
    forward is skipped). ``forward(x, t, cond)`` returns a sparse velocity on ``x``'s coordinates."""
    gs = float(guidance_strength)
    if gs == 1.0 and guidance_rescale == 0.0 and guidance_interval is None:
        return forward(x, t, cond)
    coords = x.coords
    bidx = coords[:, 0].long()
    batch = int(x.shape[0])
    t = t.reshape(-1).to(x.feats.device)
    t_tok = expand_t(t, coords)
    v_pos = forward(x, t, cond).feats
    if guidance_interval is not None:
        lo, hi = guidance_interval
        in_interval = (t >= lo) & (t <= hi)
        if not in_interval.any().item():
            return x.replace(v_pos)
    v_neg = forward(x, t, neg_cond).feats
    pred = gs * v_pos + (1.0 - gs) * v_neg
    if guidance_rescale and guidance_rescale > 0.0:
        x0_pos = x0_from_v(x.feats, v_pos, t_tok)
        x0_cfg = x0_from_v(x.feats, pred, t_tok)
        std_pos = _segment_std(x0_pos, bidx, batch)
        std_cfg = _segment_std(x0_cfg, bidx, batch).clamp_min(1e-12)
        ratio = (std_pos / std_cfg)[bidx].reshape(-1, 1)
        x0 = guidance_rescale * (x0_cfg * ratio) + (1.0 - guidance_rescale) * x0_cfg
        pred = v_from_x0(x.feats.double(), x0.double(), t_tok.double()).to(v_pos.dtype)
    if guidance_interval is not None:
        lo, hi = guidance_interval
        gate = ((t >= lo) & (t <= hi)).float()[bidx].reshape(-1, 1)
        pred = gate * pred + (1.0 - gate) * v_pos
    return x.replace(pred)


@torch.no_grad()
def teacher_anchor(forward: Callable, x_noise, cond, neg_cond, *, steps: int = ANCHOR_STEPS,
                   rescale_t: float = ANCHOR_RESCALE_T, guidance: Mapping[str, Any] = GUIDANCE):
    """The teacher's sample from ``x_noise``: ``steps`` guided Euler steps on its own time grid. Returns the
    clean latent (sparse, normalised latent space)."""
    grid = teacher_t_grid(steps, rescale_t)
    x = x_noise
    for i in range(len(grid) - 1):
        t, t_next = float(grid[i]), float(grid[i + 1])
        if t == t_next:
            continue
        t_tensor = torch.full((int(x.shape[0]),), t, device=x.feats.device)
        v = guided_velocity(forward, x, t_tensor, cond, neg_cond, **guidance)
        x = x.replace(euler_step(x.feats, v.feats, t, t_next))
    return x


# ------------------------------------------------------------------------------------------ EMA
def ema_ramp_decay(n: int, decay: float) -> float:
    """The decay of the ``n``-th EMA update (``n`` updates done before it): ``min(decay, (1 + n) / (10 + n))``."""
    return min(float(decay), (1.0 + n) / (EMA_RAMP_OFFSET + n))


@torch.no_grad()
def ema_update_ramp(shadow: Dict[str, torch.Tensor], params: Mapping[str, torch.nn.Parameter], decay: float,
                    n: int) -> float:
    d = ema_ramp_decay(n, decay)
    for name, p in params.items():
        shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
    return d


@contextlib.contextmanager
def ema_weights(params: Mapping[str, torch.nn.Parameter], shadow: Mapping[str, torch.Tensor]):
    """Run with the EMA weights copied into the live adapters (the consistency target), then restore them."""
    with torch.no_grad():
        stored = {name: p.detach().clone() for name, p in params.items()}
        for name, p in params.items():
            p.copy_(shadow[name])
    try:
        yield
    finally:
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(stored[name])


# ----------------------------------------------------------------------------------------- loss
@dataclass
class Sample:
    """One training sample: the object's support and conditioning, its anchor and the draws of the update."""

    object_id: str
    coords: torch.Tensor
    cond: Dict[str, Any]
    neg_cond: Dict[str, Any]
    anchor: Any                 # sparse clean latent x0 (anchor) on ``coords``
    t_n: torch.Tensor           # [1] float32
    t_next: torch.Tensor        # [1] float32
    eps: torch.Tensor           # [N, C] float32
    tokens: int = 0


@dataclass
class UpdateResult:
    loss: float
    grad_norm: float
    ema_decay: float
    info: Dict[str, float] = field(default_factory=dict)


def consistency_loss(student: Callable, teacher: Callable, params: Mapping[str, torch.nn.Parameter],
                     ema_shadow: Mapping[str, torch.Tensor], sample: Sample, *, weight: float = LOSS_WEIGHT,
                     guidance: Mapping[str, Any] = GUIDANCE) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The consistency loss of one sample (gradient only through ``student(x_n, t_n, cond)``)."""
    x0, coords = sample.anchor, sample.coords
    t_n_tok, t_next_tok = expand_t(sample.t_n, coords), expand_t(sample.t_next, coords)
    x_n = x0.replace(add_noise(x0.feats, sample.eps, t_n_tok))
    with torch.no_grad():
        v_t = guided_velocity(teacher, x_n, sample.t_n, sample.cond, sample.neg_cond, **guidance)
        x_next = x_n.replace(euler_step(x_n.feats, v_t.feats, t_n_tok, t_next_tok))
    with ema_weights(params, ema_shadow), torch.no_grad():
        v_tgt = student(x_next, sample.t_next, sample.cond)
        x0_target = x0_from_v(x_next.feats, v_tgt.feats, t_next_tok, double=True)
    v_s = student(x_n, sample.t_n, sample.cond)
    x0_student = x0_from_v(x_n.feats, v_s.feats, t_n_tok, double=True)
    loss = float(weight) * F.mse_loss(x0_student.double(), x0_target.double().detach(), reduction="mean")
    info = {"x0_target_std": float(x0_target.float().std().detach()),
            "x0_student_std": float(x0_student.float().std().detach())}
    return loss, info


def run_update(loss_fn: Callable[[], Tuple[torch.Tensor, Dict[str, float]]], params: Mapping[str, torch.nn.Parameter],
               optimizer: torch.optim.Optimizer, ema_shadow: Dict[str, torch.Tensor], ema_steps: int, *,
               grad_clip: float, ema_decay: float, world_size: int = 1) -> UpdateResult:
    """One optimizer update: loss, backward, gradients averaged over processes, clipped, AdamW, EMA (ramp)."""
    optimizer.zero_grad(set_to_none=True)
    loss, info = loss_fn()
    loss.backward()
    average_gradients(params, world_size)
    grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), float(grad_clip))
    optimizer.step()
    d = ema_update_ramp(ema_shadow, params, float(ema_decay), int(ema_steps))
    return UpdateResult(loss=float(loss.detach().item()), grad_norm=float(grad_norm), ema_decay=d, info=info)


# ------------------------------------------------------------------------------------ data order
def distributed_order(n_items: int, world_size: int, rank: int, seed: int, epoch: int) -> List[int]:
    """The indices one process visits in one epoch: PyTorch's ``DistributedSampler`` with ``shuffle=True``
    and ``drop_last=True`` (``randperm`` seeded with ``seed + epoch``, cut to a multiple of the world size,
    every ``world_size``-th index from ``rank``)."""
    n_items, world_size, rank = int(n_items), int(world_size), int(rank)
    if n_items < world_size:
        raise ValueError(f"{n_items} objects cannot be sharded over {world_size} processes with drop_last")
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    indices = torch.randperm(n_items, generator=generator).tolist()
    if n_items % world_size != 0:
        num_samples = math.ceil((n_items - world_size) / world_size)
    else:
        num_samples = math.ceil(n_items / world_size)
    total = num_samples * world_size
    return indices[:total][rank:total:world_size]


# ------------------------------------------------------------------------------------- the loop
def defaults(student: str, submodel: str, seed: int, gradient_checkpointing: bool, output_dir: str) -> Dict[str, Any]:
    return {
        "model": {"pixal3d": None, "pixal3d_src": None, "student": student, "base_submodel": submodel,
                  "gradient_checkpointing": gradient_checkpointing},
        "lora": {"rank": 256, "alpha": 256.0, "dropout": 0.0},
        "consistency": {"num_timesteps": NUM_TIMESTEPS, "loss_weight": LOSS_WEIGHT, "anchor_steps": ANCHOR_STEPS,
                        "anchor_rescale_t": ANCHOR_RESCALE_T, "guidance_strength": 7.5, "guidance_rescale": 0.5,
                        "guidance_interval": [0.6, 1.0]},
        "data": {"conditioning": None, "objects": None, "object_key": "id", "exclude": None, "max_objects": None},
        "optim": {"updates": 3000, "lr": 2.0e-6, "betas": [0.0, 0.999], "weight_decay": 1.0e-3, "grad_clip": 10.0,
                  "ema_decay": 0.99, "seed": seed},
        "output": {"dir": output_dir, "checkpoint_every": 250, "log_every": 1, "resume": "auto"},
        "runtime": {"device": "cuda", "tf32_override": False, "threads": 4, "max_minutes": None},
    }


PATH_KEYS = ["output.dir", "data.conditioning", "data.objects", "data.exclude"]
OPTIONAL_PATH_KEYS = ["model.pixal3d", "model.pixal3d_src"]


def check(cfg: Mapping[str, Any]) -> None:
    o, c = cfg["optim"], cfg["consistency"]
    C.require(cfg["model"]["student"] in ("coarse_shape", "fine_shape"), "model.student must be coarse_shape or fine_shape")
    C.require(int(o["updates"]) >= 1, "optim.updates must be >= 1")
    C.require(float(o["lr"]) > 0, "optim.lr must be > 0")
    C.require(0.0 <= float(o["ema_decay"]) < 1.0, "optim.ema_decay must be in [0, 1)")
    C.require(int(c["num_timesteps"]) >= 2, "consistency.num_timesteps must be >= 2")
    C.require(int(c["anchor_steps"]) >= 1, "consistency.anchor_steps must be >= 1")
    C.require(float(c["guidance_strength"]) > 0, "consistency.guidance_strength must be > 0")
    interval = c["guidance_interval"]
    C.require(interval is None or (len(interval) == 2 and float(interval[0]) <= float(interval[1])),
              "consistency.guidance_interval must be null or [lo, hi]")
    C.require(int(cfg["output"]["checkpoint_every"]) >= 1, "output.checkpoint_every must be >= 1")
    C.require(int(cfg["lora"]["rank"]) >= 1, "lora.rank must be >= 1")
    C.require(cfg["data"]["conditioning"] is not None, "data.conditioning is not set")


def guidance_of(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    c = cfg["consistency"]
    interval = c["guidance_interval"]
    return {"guidance_strength": float(c["guidance_strength"]), "guidance_rescale": float(c["guidance_rescale"]),
            "guidance_interval": None if interval is None else (float(interval[0]), float(interval[1]))}


def provenance(cfg: Mapping[str, Any], *, world_size: int, n_objects: int, trainer: str) -> Dict[str, Any]:
    o, c, l = cfg["optim"], cfg["consistency"], cfg["lora"]
    return {
        "student": cfg["model"]["student"], "objective": "consistency_distillation",
        "k": 1, "grid_t_values": [1.0, 0.0],  # the released one-step students: one query at t = 1
        "lr": float(o["lr"]), "betas": [float(b) for b in o["betas"]], "weight_decay": float(o["weight_decay"]),
        "lora_rank": int(l["rank"]), "lora_alpha": float(l["alpha"]), "ema_decay": float(o["ema_decay"]),
        "ema_ramp": f"min(decay, (1 + n) / ({EMA_RAMP_OFFSET:g} + n))", "grad_clip": float(o["grad_clip"]),
        "seed": int(o["seed"]), "world_size": int(world_size), "effective_batch_per_update": int(world_size),
        "num_timesteps": int(c["num_timesteps"]), "loss_weight": float(c["loss_weight"]),
        "anchor_steps": int(c["anchor_steps"]), "anchor_rescale_t": float(c["anchor_rescale_t"]),
        "teacher_guidance": guidance_of(cfg), "n_objects": int(n_objects),
        "conditioning_dir": str(cfg["data"]["conditioning"]), "objects": cfg["data"]["objects"], "trainer": trainer,
    }


def train(run, *, schema, pool: Sequence[str], load_object: Callable[[str], Tuple[torch.Tensor, Dict[str, Any]]],
          sparse: Callable[[torch.Tensor, torch.Tensor], Any], provenance: Mapping[str, Any],
          log_fields: Optional[Callable[[Sample], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The update loop. ``run`` is a :class:`fastpixal3d.training.loop.Run` after ``loop.build``;
    ``load_object(object_id)`` returns ``(coords [N, 4] int32, cond)`` on the device; ``sparse(feats, coords)``
    builds the sparse tensor the flow takes. Returns a summary (updates done, mean step time, last loss)."""
    cfg, o, out, c = run.cfg, run.cfg["optim"], run.cfg["output"], run.cfg["consistency"]
    net, params, device = run.net, run.params, run.device
    guidance = guidance_of(cfg)
    optimizer = make_optimizer(params, o["lr"], o["betas"], o["weight_decay"])
    ema = init_ema(params)
    start_step, position = 0, {"epoch": 0, "cursor": 0}
    resume, resume_path = out["resume"], None
    if resume == "auto":
        resume_path = ckpt.newest(out["dir"])
    elif resume not in (None, "none", False):
        resume_path = Path(resume)
    if resume_path is not None:
        payload = ckpt.load(resume_path)
        if int(payload.get("world_size", run.world_size)) != run.world_size:
            raise RuntimeError(f"checkpoint was written by {payload.get('world_size')} processes, this run has "
                               f"{run.world_size}: the data order would change (start a new output.dir)")
        start_step = ckpt.restore(payload, params, ema, optimizer)
        position = dict((payload.get("provenance") or {}).get("data_position") or position)
        if run.is_main:
            print(json.dumps({"event": "resumed", "checkpoint": str(resume_path), "completed_updates": start_step,
                              "data_position": position}), flush=True)
    if not pool:
        raise RuntimeError("the training pool is empty (no records matching the object list)")
    out_dir = Path(out["dir"])
    log_path = out_dir / "loss.jsonl"
    if run.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(C.dump(cfg))
        print(json.dumps({"event": "pool", "objects": len(pool), "world_size": run.world_size, "start": start_step,
                          "data_position": position}), flush=True)

    anchors: Dict[str, torch.Tensor] = {}
    epoch, cursor = int(position["epoch"]), int(position["cursor"])
    order = distributed_order(len(pool), run.world_size, run.rank, int(o["seed"]), epoch)
    if cursor >= len(order):
        epoch, cursor = epoch + 1, 0
        order = distributed_order(len(pool), run.world_size, run.rank, int(o["seed"]), epoch)

    def save(step: int) -> None:
        prov = dict(provenance)
        prov["run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        prov["data_position"] = {"epoch": epoch, "cursor": cursor}
        payload = ckpt.make_payload(schema=schema, params=params, ema_shadow=ema, optimizer=optimizer,
                                    completed_updates=step, world_size=run.world_size, provenance=prov)
        path = ckpt.save(payload, out_dir)
        print(json.dumps({"event": "checkpoint", "path": str(path)}), flush=True)

    def teacher(x, t, cond):
        return net.teacher(x, t, cond)

    def student(x, t, cond):
        return net(x, t, cond)

    def anchor_of(object_id: str, coords: torch.Tensor, cond, neg):
        cached = anchors.get(object_id)
        if cached is None:
            generator = make_generator(coords.device, anchor_seed(int(o["seed"]), object_id))
            eps0 = torch.randn(int(coords.shape[0]), LATENT_CHANNELS, device=coords.device, generator=generator)
            x0 = teacher_anchor(teacher, sparse(eps0, coords), cond, neg, steps=int(c["anchor_steps"]),
                                rescale_t=float(c["anchor_rescale_t"]), guidance=guidance)
            anchors[object_id] = x0.feats.detach().cpu()
            return x0
        return sparse(cached.to(coords.device), coords)

    t_start = time.perf_counter()
    step_times: List[float] = []
    recent: List[float] = []
    last = start_step
    log_every, every = int(out["log_every"]), int(out["checkpoint_every"])
    max_minutes = cfg["runtime"]["max_minutes"]
    result = None
    for step in range(start_step + 1, int(o["updates"]) + 1):
        if max_minutes and (time.perf_counter() - t_start) / 60.0 > float(max_minutes):
            if run.is_main:
                print(json.dumps({"event": "max_minutes_stop", "step": last}), flush=True)
            break
        t0 = time.perf_counter()
        object_id = pool[order[cursor]]
        coords, cond = load_object(object_id)
        neg = neg_cond_like(cond)
        x0 = anchor_of(object_id, coords, cond, neg)
        generator = make_generator(device, update_seed(int(o["seed"]), step - 1, run.rank, object_id))
        n_idx = draw_time_index(generator, device, int(c["num_timesteps"]))
        t_n, t_next = time_pair(n_idx, int(c["num_timesteps"]))
        eps = draw_noise_like(generator, x0.feats)
        sample = Sample(object_id=object_id, coords=coords, cond=cond, neg_cond=neg, anchor=x0, t_n=t_n, t_next=t_next,
                        eps=eps, tokens=int(coords.shape[0]))
        result = run_update(lambda: consistency_loss(student, teacher, params, ema, sample, weight=float(c["loss_weight"]),
                                                     guidance=guidance),
                            params, optimizer, ema, step - 1, grad_clip=float(o["grad_clip"]),
                            ema_decay=float(o["ema_decay"]), world_size=run.world_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cursor += 1
        if cursor >= len(order):
            epoch, cursor = epoch + 1, 0
            order = distributed_order(len(pool), run.world_size, run.rank, int(o["seed"]), epoch)
        step_times.append(time.perf_counter() - t0)
        recent.append(result.loss)
        last = step
        if run.is_main and (step % log_every == 0 or step == start_step + 1):
            row = {"step": step, "loss": result.loss, "loss_mean_recent": sum(recent[-100:]) / min(len(recent), 100),
                   "t_n": float(t_n.item()), "t_next": float(t_next.item()), "object": object_id, "tokens": sample.tokens,
                   "grad_norm": result.grad_norm, "ema_decay": result.ema_decay, **result.info,
                   "epoch": epoch, "step_s": round(step_times[-1], 3), "wall_s": round(time.perf_counter() - t_start, 2)}
            if log_fields is not None:
                row.update(log_fields(sample))
            print(json.dumps(row), flush=True)
            with open(log_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        if run.is_main and (step % every == 0 or step == int(o["updates"])):
            save(step)
    if run.is_main and last > start_step and not (last % every == 0 or last == int(o["updates"])):
        save(last)
    if run.world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return {"updates": last - start_step, "last_step": last,
            "step_s_mean": (sum(step_times) / len(step_times)) if step_times else None,
            "step_s": step_times, "last_loss": result.loss if result is not None else None,
            "anchors": len(anchors)}
