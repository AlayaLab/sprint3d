"""Per-stage wall-clock timing with CUDA synchronisation at every stage boundary.

Each timed stage is bracketed by ``torch.cuda.synchronize()`` so that its
duration includes all GPU work it launched. The same synchronisation points
are part of the measured end-to-end time (they are part of the pipeline's
schedule, not instrumentation added on top of it).

Stage keys (seconds, in execution order):

  preprocess_s, camera_s,
  ss_condition_s, ss_initial_noise_s, ss_rollout_s, ss_decode_support_s,
  lr_condition_s, lr_initial_noise_s, lr_rollout_s,
  hr_upsample_support_s, hr_condition_s, hr_initial_noise_s, hr_rollout_s,
  shape_decode_s, shape_fill_holes_s,
  tex_condition_s, tex_rollout_s, tex_decode_s       (overlapped with the mesh worker)
  glb_finish_s, glb_bake_finalize_s, to_glb_s         (to_glb_s spans mesh + texture + bake)
  export_s, e2e_s

``critical_path_keys`` lists the keys whose sum accounts for the end-to-end
time (the texture keys are inside ``to_glb_s`` when the mesh worker runs).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

CRITICAL_PATH_KEYS = (
    "preprocess_s", "camera_s",
    "ss_condition_s", "ss_initial_noise_s", "ss_rollout_s", "ss_decode_support_s",
    "lr_condition_s", "lr_initial_noise_s", "lr_rollout_s",
    "hr_upsample_support_s", "hr_condition_s", "hr_initial_noise_s", "hr_rollout_s",
    "shape_decode_s", "shape_fill_holes_s",
    "to_glb_s", "export_s",
)


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class StageTimer:
    """Collects stage durations into ``self.times`` (rounded to microseconds)."""

    def __init__(self, sync: Callable[[], None] = cuda_sync):
        self.sync = sync
        self.times: Dict[str, Any] = {}
        self.order: List[str] = []

    def timed(self, key: str, fn: Callable[[], Any]) -> Any:
        self.sync()
        started = time.perf_counter()
        value = fn()
        self.sync()
        self.record(key, time.perf_counter() - started)
        return value

    def record(self, key: str, seconds: float) -> None:
        self.times[key] = round(float(seconds), 6)
        if key not in self.order:
            self.order.append(key)

    def summary(self) -> Dict[str, Any]:
        out = {key: self.times[key] for key in self.order}
        if "e2e_s" in out:
            accounted = sum(float(out[k]) for k in CRITICAL_PATH_KEYS if k in out)
            out["timing_accounted_s"] = round(accounted, 6)
            out["residual_s"] = round(out["e2e_s"] - accounted, 6)
        return out
