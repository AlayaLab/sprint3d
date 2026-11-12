#!/usr/bin/env python3
"""Structure anchor cache: the teacher's 12-step trajectory of the structure flow, per rendered view.

    python train/caches/structure_anchors.py --config train/configs/caches.yaml --dataset train_5k --verify-n 3

For each view with a structure conditioning record (structure_conditioning.py), Pixal3D's own
structure sampler runs 12 Euler steps with ``rescale_t = 3`` (the student's grid, so that the
student's 4-step grid is a subsequence of it) and the teacher's guidance: strength 7.5, rescale
0.7, interval [0.6, 1.0], negative conditioning = zeros. The initial noise ``[1, 8, 16, 16, 16]``
is drawn from a seed derived from the view name (fastpixal3d.training.schedule), so index 0 is
never stored. The states after steps 2, 3, 4, 6, 8, 9, 10 and 12 are stored as bfloat16
``[4096, 8]`` token matrices (full 16^3 grid, C order).

``--verify-n N`` samples the first N built views a second time and checks the relative difference
of the final state (default limit 5e-3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import cachework as W  # noqa: E402

from fastpixal3d.training.teacher import RESCALE_T, STEPS  # noqa: E402
from fastpixal3d.training.teacher import STRUCTURE_GUIDANCE as GUIDANCE  # noqa: E402

SCHEMA = ("structure_anchors", 1)
STORED = (2, 3, 4, 6, 8, 9, 10, 12)


def main(argv=None) -> int:
    args = W.parse_args(__doc__.split("\n\n")[0], argv)
    cfg = W.load(args)
    from fastpixal3d.training.environment import configure_process, load_pipeline, weight_paths

    configure_process(threads=cfg["runtime"]["threads"], tf32_override=cfg["runtime"]["tf32_override"])
    cond_dir = W.output_dir(cfg, "structure_conditioning", args.dataset)
    out_dir = W.output_dir(cfg, "structure_anchors", args.dataset)
    from fastpixal3d.training.records import record_names

    names = record_names(cond_dir)
    if not W.pending(args, names, out_dir):
        print(json.dumps({"event": "nothing_to_do", "out_dir": str(out_dir)}))
        return 0
    import torch

    from fastpixal3d.training.records import load_record
    from fastpixal3d.training.schedule import STRUCTURE_NOISE_SALT, full_grid_gather, full_grid_index, stable_seed
    from fastpixal3d.training.teacher import structure_trajectory

    pipeline = load_pipeline(weight_paths(cfg["pixal3d"]["model"], cfg["pixal3d"]["source"]),
                             device=cfg["runtime"]["device"])
    device = torch.device(pipeline.device)
    index = full_grid_index(16, device)

    def rollout(view: str):
        rec = load_record(cond_dir / f"{view}.pt")
        return structure_trajectory(pipeline, rec, view, device), rec

    def build(view: str):
        ret, rec = rollout(view)
        anchors = {str(s): full_grid_gather(ret.pred_x_t[s - 1], index).detach().to(torch.bfloat16).cpu()
                   for s in STORED}
        return {"schema": SCHEMA[0], "schema_version": SCHEMA[1], "object_id": view, "anchors": anchors,
                "camera": rec.get("camera"),
                "provenance": {"teacher_steps": STEPS, "anchor_rescale_t": RESCALE_T, **GUIDANCE,
                               "stored_steps": list(STORED), "noise_seed": stable_seed(STRUCTURE_NOISE_SALT, view),
                               "conditioning_dir": str(cond_dir)}}

    def verify(view: str, record) -> float:
        ret, _ = rollout(view)
        return W.rel_drift(full_grid_gather(ret.samples, index).cpu(), record["anchors"]["12"])

    W.run(args, out_dir, names, build, verify, tag="structure_anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
