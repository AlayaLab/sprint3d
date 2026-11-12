#!/usr/bin/env python3
"""Train the texture student: trajectory distillation of Pixal3D's texture flow (k = 2 steps).

    python train/texture_traj/train.py --config train/configs/texture_student.yaml
    python train/texture_traj/train.py --config train/configs/texture_student.yaml \\
        --set optim.updates=3 --set data.max_views=4 --set output.dir=/tmp/texture_smoke   # a short check

The student is Pixal3D's texture flow with rank-256 adapters, run conditional-only with 2 Euler
steps on t = 1.0, 0.75, 0.0 (indices 0, 6, 12 of the teacher's 12-step grid, rescale_t 3). It reads
one record per rendered view (train/caches/texture_anchors.py): the voxel coordinates, the
normalised fine-shape latent the texture flow is conditioned on, the teacher's trajectory states
and the view's camera and render. The image conditioning is recomputed at every step from the
render (DINOv3 features projected on the voxels, Pixal3D's ``get_proj_cond_shape``), which needs
the full Pixal3D pipeline in memory.

The released student was trained in two segments (docs/TRAINING.md); this script is the trainer of
the second segment. ``runtime.max_minutes`` stops cleanly after a checkpoint, and with
``output.resume: auto`` a restarted run continues from the newest checkpoint in ``output.dir``.
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

# Anchor indices stored by train/caches/texture_anchors.py (index 0 is regenerated from its seed).
STORED_ANCHOR_STEPS = (3, 4, 6, 8, 9, 12)

DEFAULTS = loop.common_defaults(submodel="ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16", grid_indices=[0, 6, 12],
                                updates=3000, grad_accum=4, seed=20260912, checkpoint_every=250,
                                output_dir="runs/texture_student")
DEFAULTS["data"] = {
    "anchors": None,         # directory of <view>.pt texture trajectory records
    "image_roots": [],       # dataset roots holding <subset dir>/renders_cond/<sha256>/<view:03d>.png
    "views": None,           # optional: object list (.csv) or view-name list; default: all records
    "view_key": "sha256",    # CSV column that names the object in view names (<key>.viewNN)
    "view_indices": [0, 1],  # render indices per object when `views` is a CSV list
    "max_views": None,       # use only the first N views of the sorted pool (short checks)
    "name": "curated_10k",   # recorded in the checkpoint provenance
}
PATH_KEYS = loop.COMMON_PATH_KEYS + ["data.anchors", "data.image_roots", "data.views"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML configuration (train/configs/texture_student.yaml)")
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
    C.require(isinstance(cfg["data"]["image_roots"], list), "data.image_roots must be a list of directories")
    if args.print_config:
        print(C.dump(cfg))
        raise SystemExit(0)
    return cfg


def main(argv=None) -> dict:
    cfg = load(argv)
    run = loop.start(cfg)
    from fastpixal3d.training.checkpoint import TEXTURE_SCHEMA
    from fastpixal3d.training.environment import load_pipeline
    from fastpixal3d.training.objective import MicroBatch
    from fastpixal3d.training.records import conditioning_image, find_render, load_record, training_pool, views_from_list
    from fastpixal3d.training.schedule import texture_initial_noise

    d = cfg["data"]
    anchors_dir = Path(d["anchors"])
    roots = [Path(r) for r in d["image_roots"]]
    wanted = views_from_list(d["views"], d["view_key"], d["view_indices"]) if d["views"] else None
    run.pool = training_pool(anchors_dir, wanted, d["max_views"])
    loop.build(run, "texture")  # seeded adapters first, as in the training runs
    pipeline = load_pipeline(run.weights, device=str(run.device))  # conditioning extractors (and the rest of Pixal3D)
    from pixal3d.modules import sparse as sp

    device, grid, net = run.device, run.grid, run.net

    def prepare(item) -> MicroBatch:
        view, (i, j) = item
        rec = load_record(anchors_dir / f"{view}.pt")
        coords = rec["coords"].to(device)
        concat = sp.SparseTensor(feats=rec["concat_cond"].to(device), coords=coords)
        render = rec["render"]
        image = conditioning_image(find_render(render, roots).read_bytes(), render.get("image_sha256"))
        cam = rec["camera"]
        cond = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024, [image], coords, camera_angle_x=cam["camera_angle_x"],
            distance=cam["distance"], mesh_scale=cam.get("mesh_scale", 1.0),
            grid_resolution_override=int(rec["grid_res"]))["cond"]
        if i == 0:
            x_i = texture_initial_noise(int(coords.shape[0])).to(device)
        else:
            x_i = rec["anchors"][str(i)].to(device)
        target = rec["anchors"][str(j)].to(device).float()
        return MicroBatch(view=view, i=i, j=j, t_i=float(grid[i]), t_j=float(grid[j]), x_i=x_i, target=target,
                          inputs={"x": sp.SparseTensor(feats=x_i, coords=coords), "cond": cond, "concat": concat,
                                  "tokens": int(coords.shape[0])})

    def forward(mb: MicroBatch):
        return net(mb.inputs["x"], mb.t_i, mb.inputs["cond"], mb.inputs["concat"]).feats

    provenance = loop.provenance_common(run)
    provenance.update({"student": "texture", "anchor_dir": str(anchors_dir), "views": d["views"], "data": d["name"],
                       "conditioning_convention": "raw render, true camera", "trainer": "train/texture_traj/train.py"})
    summary = loop.train(run, schema=TEXTURE_SCHEMA, prepare=prepare, forward=forward, provenance=provenance,
                         log_fields=lambda mb: {"tokens": mb.inputs["tokens"]})
    if run.is_main:
        print(json.dumps({"event": "done", **{k: v for k, v in summary.items() if k != "step_s"}}), flush=True)
    return summary


if __name__ == "__main__":
    main()
