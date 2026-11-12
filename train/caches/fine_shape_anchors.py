#!/usr/bin/env python3
"""Fine-shape anchor cache: the teacher's 12-step trajectory of the fine (1024) shape flow, per view.

    python train/caches/fine_shape_anchors.py --config train/configs/caches.yaml --dataset curated_10k --verify-n 2

For each view with a fine-shape conditioning record (fine_shape_conditioning.py), Pixal3D's shape
sampler runs the fine flow for 12 Euler steps (``rescale_t = 3``, guidance 7.5, rescale 0.5,
interval [0.6, 1.0], negative conditioning = zeros) from ``[N, 32]`` noise drawn from a seed derived
from the view name. The states after steps 3, 4, 6, 8, 9 and 12 are stored as bfloat16, in the
flow's normalised latent space. The final state (step 12) is the fine-shape latent that conditions
the texture flow (texture_anchors.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import cachework as W  # noqa: E402

from fastpixal3d.training.teacher import FINE_SHAPE_GUIDANCE as GUIDANCE  # noqa: E402
from fastpixal3d.training.teacher import RESCALE_T, STEPS  # noqa: E402

SCHEMA = ("fine_shape_anchors", 1)
STORED = (3, 4, 6, 8, 9, 12)


def main(argv=None) -> int:
    args = W.parse_args(__doc__.split("\n\n")[0], argv)
    cfg = W.load(args)
    from fastpixal3d.training.environment import configure_process, load_pipeline, weight_paths

    configure_process(threads=cfg["runtime"]["threads"], tf32_override=cfg["runtime"]["tf32_override"])
    cond_dir = W.output_dir(cfg, "fine_shape_conditioning", args.dataset)
    out_dir = W.output_dir(cfg, "fine_shape_anchors", args.dataset)
    from fastpixal3d.training.records import record_names

    names = record_names(cond_dir)
    if not W.pending(args, names, out_dir):
        print(json.dumps({"event": "nothing_to_do", "out_dir": str(out_dir)}))
        return 0
    import torch

    from fastpixal3d.training.records import load_record
    from fastpixal3d.training.schedule import FINE_SHAPE_NOISE_SALT, stable_seed
    from fastpixal3d.training.teacher import fine_shape_trajectory

    pipeline = load_pipeline(weight_paths(cfg["pixal3d"]["model"], cfg["pixal3d"]["source"]),
                             device=cfg["runtime"]["device"])
    device = torch.device(pipeline.device)

    def rollout(view: str):
        rec = load_record(cond_dir / f"{view}.pt")
        return fine_shape_trajectory(pipeline, rec, view, device), rec

    def build(view: str):
        ret, rec = rollout(view)
        anchors = {str(s): ret.pred_x_t[s - 1].feats.detach().to(torch.bfloat16).cpu().clone() for s in STORED}
        return {"schema": SCHEMA[0], "schema_version": SCHEMA[1], "object_id": view,
                "n_tokens": int(rec["coords"].shape[0]), "anchors": anchors, "camera": rec.get("camera"),
                "provenance": {"teacher_steps": STEPS, "anchor_rescale_t": RESCALE_T, **GUIDANCE,
                               "stored_steps": list(STORED), "latent_space": "normalized",
                               "noise_seed": stable_seed(FINE_SHAPE_NOISE_SALT, view),
                               "actual_resolution": rec.get("actual_resolution"), "grid_res": rec.get("grid_res"),
                               "conditioning_dir": str(cond_dir)}}

    def verify(view: str, record) -> float:
        ret, _ = rollout(view)
        return W.rel_drift(ret.samples.feats.detach().cpu(), record["anchors"]["12"])

    W.run(args, out_dir, names, build, verify, tag="fine_shape_anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
