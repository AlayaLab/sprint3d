"""The training loop shared by the structure and texture trainers (:func:`start` and :func:`build` also
serve the consistency trainer of the shape students, :mod:`fastpixal3d.training.consistency`).

Order of operations (the order of the training runs; it fixes the initial adapters and the sample
sequence of a seed):

1. ``torch.manual_seed(seed)``, load the base flow, attach the adapters (:mod:`.student`);
   with several processes the adapters of rank 0 are broadcast to the others;
2. AdamW over the adapters, EMA shadow = copy of the initial adapters; resume if requested;
3. training pool = sorted record names (optionally restricted to a view list);
   ``rng = random.Random(seed + 1000 * rank)``;
4. per update: ``grad_accum`` draws of (view, segment) from ``rng`` (:func:`.objective.draw_segments`),
   one :func:`.objective.run_update`; log every ``log_every`` updates and at the first update;
   checkpoint every ``checkpoint_every`` updates, at the last update and when stopping early.

A resumed run creates ``rng`` from the seed again, so its sample sequence restarts from the
beginning; this is how the released texture student's second training segment was run.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import checkpoint as ckpt
from . import config as C
from .objective import MicroBatch, draw_segments, init_ema, make_optimizer, run_update
from .schedule import check_grid_indices, grid_t_values, segment_pairs, teacher_t_grid


@dataclass
class Run:
    """Everything the student-specific code needs during training."""

    cfg: Dict[str, Any]
    device: Any
    rank: int
    world_size: int
    local_rank: int
    model_dir: Path
    weights: Any
    net: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    grid: Any = None
    pairs: List[tuple] = field(default_factory=list)
    pool: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def common_defaults(submodel: str, grid_indices: Sequence[int], updates: int, grad_accum: int, seed: int,
                    checkpoint_every: int, output_dir: str) -> Dict[str, Any]:
    return {
        "model": {"pixal3d": None, "pixal3d_src": None, "base_submodel": submodel, "gradient_checkpointing": True},
        "lora": {"rank": 256, "alpha": 256.0, "dropout": 0.0},
        "schedule": {"teacher_steps": 12, "rescale_t": 3.0, "grid_indices": list(grid_indices)},
        "optim": {"updates": updates, "grad_accum": grad_accum, "lr": 1.0e-4, "betas": [0.9, 0.999],
                  "weight_decay": 0.0, "grad_clip": 10.0, "ema_decay": 0.99, "seed": seed, "autocast": "bfloat16"},
        "output": {"dir": output_dir, "checkpoint_every": checkpoint_every, "log_every": 10, "resume": "auto"},
        "runtime": {"device": "cuda", "tf32_override": True, "threads": 4, "max_minutes": None},
    }


COMMON_PATH_KEYS = ["output.dir"]
COMMON_OPTIONAL_PATH_KEYS = ["model.pixal3d", "model.pixal3d_src"]


def check_common(cfg: Mapping[str, Any], stored_anchor_steps: Sequence[int]) -> None:
    o, s = cfg["optim"], cfg["schedule"]
    C.require(int(o["updates"]) >= 1, "optim.updates must be >= 1")
    C.require(int(o["grad_accum"]) >= 1, "optim.grad_accum must be >= 1")
    C.require(float(o["lr"]) > 0, "optim.lr must be > 0")
    C.require(0.0 <= float(o["ema_decay"]) < 1.0, "optim.ema_decay must be in [0, 1)")
    C.require(o["autocast"] in ("bfloat16", None), "optim.autocast must be bfloat16 or null")
    C.require(int(cfg["output"]["checkpoint_every"]) >= 1, "output.checkpoint_every must be >= 1")
    C.require(int(cfg["lora"]["rank"]) >= 1, "lora.rank must be >= 1")
    try:
        check_grid_indices(s["grid_indices"], int(s["teacher_steps"]), stored_anchor_steps)
    except ValueError as error:
        raise C.ConfigError(f"schedule.grid_indices: {error}") from None


def start(cfg: Dict[str, Any]) -> Run:
    """Process settings, torch, distributed setup and the Pixal3D import (no model yet)."""
    from .environment import configure_process, import_pixal3d, weight_paths

    configure_process(threads=int(cfg["runtime"]["threads"]), tf32_override=bool(cfg["runtime"]["tf32_override"]))
    import torch

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = str(cfg["runtime"]["device"])
    if device_name.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_name)
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
    weights = weight_paths(cfg["model"]["pixal3d"], cfg["model"]["pixal3d_src"])
    model_dir = import_pixal3d(weights)
    run = Run(cfg=cfg, device=device, rank=rank, world_size=world, local_rank=local_rank,
              model_dir=model_dir, weights=weights)
    s = cfg.get("schedule")  # the trajectory trainers; the consistency trainer has no time grid
    if s:
        run.grid = teacher_t_grid(int(s["teacher_steps"]), float(s["rescale_t"]))
        run.pairs = segment_pairs(s["grid_indices"])
    return run


def build(run: Run, kind: str) -> None:
    """Seeded student construction, broadcast, in the order of the training runs."""
    import torch

    from .student import build_student

    cfg = run.cfg
    run.net = build_student(kind, run.model_dir, cfg["model"]["base_submodel"], run.device,
                            rank=int(cfg["lora"]["rank"]), alpha=float(cfg["lora"]["alpha"]),
                            dropout=float(cfg["lora"]["dropout"]),
                            gradient_checkpointing=bool(cfg["model"]["gradient_checkpointing"]),
                            seed=int(cfg["optim"]["seed"]))
    run.params = run.net.trainable()
    if run.world_size > 1:
        for p in run.params.values():
            torch.distributed.broadcast(p.data, src=0)


def provenance_common(run: Run) -> Dict[str, Any]:
    cfg, s = run.cfg, run.cfg["schedule"]
    idx = [int(i) for i in s["grid_indices"]]
    k = len(idx) - 1
    uniform = idx == list(range(0, int(s["teacher_steps"]) + 1, int(s["teacher_steps"]) // k)) \
        if int(s["teacher_steps"]) % k == 0 else False
    o = cfg["optim"]
    return {
        "k": k, "m": (int(s["teacher_steps"]) // k) if uniform else None,
        "lr": float(o["lr"]), "lora_rank": int(cfg["lora"]["rank"]), "lora_alpha": float(cfg["lora"]["alpha"]),
        "ema_decay": float(o["ema_decay"]), "grad_clip": float(o["grad_clip"]), "seed": int(o["seed"]),
        "grad_accum": int(o["grad_accum"]), "world_size": run.world_size,
        "effective_batch_per_update": run.world_size * int(o["grad_accum"]),
        "grid_indices": idx, "grid_t_values": grid_t_values(idx, int(s["teacher_steps"]), float(s["rescale_t"])),
        "anchor_rescale_t": float(s["rescale_t"]), "n_objects": len(run.pool),
    }


def train(run: Run, *, schema, prepare: Callable[[tuple], MicroBatch], forward: Callable[[MicroBatch], Any],
          provenance: Mapping[str, Any], log_fields: Optional[Callable[[MicroBatch], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The update loop. Returns a summary (updates done, mean step time, last loss)."""
    import torch

    cfg, o, out = run.cfg, run.cfg["optim"], run.cfg["output"]
    optimizer = make_optimizer(run.params, o["lr"], o["betas"], o["weight_decay"])
    ema = init_ema(run.params)
    start_step = 0
    resume = out["resume"]
    resume_path = None
    if resume == "auto":
        resume_path = ckpt.newest(out["dir"])
    elif resume not in (None, "none", False):
        resume_path = Path(resume)
    if resume_path is not None:
        start_step = ckpt.restore(ckpt.load(resume_path), run.params, ema, optimizer)
        if run.is_main:
            print(json.dumps({"event": "resumed", "checkpoint": str(resume_path), "completed_updates": start_step}),
                  flush=True)
    if not run.pool:
        raise RuntimeError("the training pool is empty (no records matching the view list)")
    rng = random.Random(int(o["seed"]) + 1000 * run.rank)
    autocast = torch.bfloat16 if o["autocast"] == "bfloat16" else None
    device_type = "cuda" if run.device.type == "cuda" else "cpu"
    out_dir = Path(out["dir"])
    log_path = out_dir / "loss.jsonl"
    if run.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(C.dump(cfg))
        print(json.dumps({"event": "pool", "views": len(run.pool), "world_size": run.world_size,
                          "grad_accum": int(o["grad_accum"]), "start": start_step}), flush=True)

    def save(step: int) -> None:
        prov = dict(provenance)
        prov["run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = ckpt.make_payload(schema=schema, params=run.params, ema_shadow=ema, optimizer=optimizer,
                                    completed_updates=step, world_size=run.world_size, provenance=prov)
        path = ckpt.save(payload, out_dir)
        print(json.dumps({"event": "checkpoint", "path": str(path)}), flush=True)

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
        items = draw_segments(rng, run.pool, run.pairs, int(o["grad_accum"]))
        prepared: List[MicroBatch] = []

        def prep(item):
            mb = prepare(item)
            prepared.append(mb)
            return mb

        result = run_update(forward, items, run.params, optimizer, ema, prepare=prep,
                            grad_clip=float(o["grad_clip"]), ema_decay=float(o["ema_decay"]),
                            world_size=run.world_size, autocast_dtype=autocast, device_type=device_type)
        if run.device.type == "cuda":
            torch.cuda.synchronize(run.device)
        step_times.append(time.perf_counter() - t0)
        recent.append(result.loss)
        last = step
        if run.is_main and (step % log_every == 0 or step == start_step + 1):
            mb = prepared[-1]
            row = {"step": step, "loss": result.loss,
                   "loss_mean_recent": sum(recent[-log_every:]) / min(len(recent), log_every),
                   "micro_losses": result.losses, "i": mb.i, "j": mb.j, "t_i": mb.t_i, "t_j": mb.t_j,
                   "view": mb.view, "grad_norm": result.grad_norm, "step_s": round(step_times[-1], 3),
                   "wall_s": round(time.perf_counter() - t_start, 2)}
            if log_fields is not None:
                row.update(log_fields(mb))
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
            "step_s": step_times, "last_loss": result.loss if result is not None else None}
