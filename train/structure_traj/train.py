#!/usr/bin/env python3
"""Train the structure student: trajectory distillation of Pixal3D's sparse-structure flow.

    python train/structure_traj/train.py --config train/configs/structure_student.yaml
    python train/structure_traj/train.py --config train/configs/structure_student.yaml \\
        --set optim.updates=3 --set data.max_views=4 --set output.dir=/tmp/structure_smoke   # a short check

The student is Pixal3D's structure flow with rank-256 adapters, run conditional-only with 4 Euler
steps on the grid t = 1.0, 0.9, 0.75, 0.6, 0.0 (indices 0, 3, 6, 8, 12 of the teacher's 12-step
grid, rescale_t 3). It reads two caches per rendered view (docs/TRAINING.md):

    data.anchors        the teacher's 12-step guided trajectory    (train/caches/structure_anchors.py)
    data.conditioning   the image conditioning of the view          (train/caches/structure_conditioning.py)

With ``torchrun --nproc_per_node N`` the processes average their gradients (data parallel); the
released student was trained with one process and gradient accumulation 4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import config as C  # noqa: E402
from fastpixal3d.training import loop  # noqa: E402

# Anchor indices stored by train/caches/structure_anchors.py (index 0 is regenerated from its seed).
STORED_ANCHOR_STEPS = (2, 3, 4, 6, 8, 9, 10, 12)

DEFAULTS = loop.common_defaults(submodel="ckpts/ss_flow_img_dit_1_3B_64_bf16", grid_indices=[0, 3, 6, 8, 12],
                                updates=500, grad_accum=4, seed=20260915, checkpoint_every=250,
                                output_dir="runs/structure_student")
DEFAULTS["data"] = {
    "anchors": None,         # directory of <view>.pt trajectory records
    "conditioning": None,    # directory of <view>.pt conditioning records
    "views": None,           # optional: object list (.csv) or view-name list (one per line); default: all records
    "view_key": "id",        # CSV column that names the object in view names (<key>.viewNN)
    "view_indices": [0, 1],  # render indices per object when `views` is a CSV list
    "max_views": None,       # use only the first N views of the sorted pool (short checks)
}
PATH_KEYS = loop.COMMON_PATH_KEYS + ["data.anchors", "data.conditioning", "data.views"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML configuration (train/configs/structure_student.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override one configuration key, e.g. --set optim.updates=10 (repeatable)")
    p.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    return p.parse_args(argv)


def load(argv=None) -> dict:
    args = parse_args(argv)
    cfg = C.load_config(args.config, DEFAULTS, path_keys=PATH_KEYS,
                        optional_path_keys=loop.COMMON_OPTIONAL_PATH_KEYS, overrides=args.overrides)
    loop.check_common(cfg, STORED_ANCHOR_STEPS)
    C.require(cfg["data"]["anchors"] is not None, "data.anchors is not set")
    C.require(cfg["data"]["conditioning"] is not None, "data.conditioning is not set")
    if args.print_config:
        print(C.dump(cfg))
        raise SystemExit(0)
    return cfg


def main(argv=None) -> dict:
    cfg = load(argv)
    run = loop.start(cfg)
    import torch

    from fastpixal3d.training.checkpoint import STRUCTURE_SCHEMA
    from fastpixal3d.training.objective import MicroBatch
    from fastpixal3d.training.records import load_record, training_pool, views_from_list
    from fastpixal3d.training.schedule import full_grid_gather, full_grid_index, structure_initial_noise

    d = cfg["data"]
    anchors_dir, cond_dir = Path(d["anchors"]), Path(d["conditioning"])
    wanted = views_from_list(d["views"], d["view_key"], d["view_indices"]) if d["views"] else None
    run.pool = training_pool(anchors_dir, wanted, d["max_views"])
    loop.build(run, "structure")
    device, grid, net = run.device, run.grid, run.net
    index = full_grid_index(16, device)

    def prepare(item) -> MicroBatch:
        view, (i, j) = item
        anchors = load_record(anchors_dir / f"{view}.pt")["anchors"]
        rec = load_record(cond_dir / f"{view}.pt")
        cond = {"global": rec["global"].to(torch.float32).unsqueeze(0).to(device),
                "proj": rec["proj"].to(torch.float32).unsqueeze(0).to(device)}
        if i == 0:
            x_i = full_grid_gather(structure_initial_noise(view).to(device), index)
        else:
            x_i = anchors[str(i)].to(device).float()
        target = anchors[str(j)].to(device).float()
        return MicroBatch(view=view, i=i, j=j, t_i=float(grid[i]), t_j=float(grid[j]), x_i=x_i, target=target,
                          inputs={"cond": cond})

    def forward(mb: MicroBatch):
        return net(mb.x_i, mb.t_i, mb.inputs["cond"])

    provenance = loop.provenance_common(run)
    provenance.update({"student": "structure", "anchor_dir": str(anchors_dir), "conditioning_dir": str(cond_dir),
                       "views": d["views"], "trainer": "train/structure_traj/train.py"})
    summary = loop.train(run, schema=STRUCTURE_SCHEMA, prepare=prepare, forward=forward, provenance=provenance)
    if run.is_main:
        print(json.dumps({"event": "done", **{k: v for k, v in summary.items() if k != "step_s"}}), flush=True)
    return summary


if __name__ == "__main__":
    main()
