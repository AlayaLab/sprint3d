#!/usr/bin/env python3
"""Texture anchor cache: the teacher's 12-step texture trajectory, per rendered view.

    python train/caches/texture_anchors.py --config train/configs/caches.yaml --dataset curated_10k --verify-n 2

For each view with a fine-shape conditioning record:

    coords        the fine support (fine_shape_conditioning.py), grid ``grid_res``
    concat_cond   the fine-shape latent the texture flow is conditioned on: the final state (step 12) of
                  the guided fine-shape teacher trajectory, normalised, stored as float32. With
                  ``--fine-shape-source auto`` (default) it is read from the fine-shape anchor record when
                  that exists (fine_shape_anchors.py; rounded to bfloat16 there) and otherwise computed
                  here with the same function (not rounded); ``anchor`` / ``rollout`` force one source.
    cond          ``get_proj_cond_shape`` with the texture extractor on the view's image (raw render,
                  512 x 512 Lanczos, on black) and true camera, ``grid_resolution_override = grid_res``
    trajectory    Pixal3D's texture sampler, 12 Euler steps, ``rescale_t = 3``, guidance strength 1.0
                  (conditional only; rescale 0.0 and interval [0.6, 0.9] have no effect), from ``[N, 32]``
                  noise drawn with seed 0 (Pixal3D's own texture noise)

The states after steps 3, 4, 6, 8, 9 and 12 are stored as float32. The record also carries the
camera, ``grid_res`` and the ``render`` block (subset, sha256, view, image sha256), from which the
texture trainer rebuilds the image conditioning at every step; ``provenance.concat_cond_source`` says
which source the fine-shape latent came from.
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
from fastpixal3d.training.teacher import TEXTURE_GUIDANCE as GUIDANCE  # noqa: E402

SCHEMA = ("texture_anchors", 1)
STORED = (3, 4, 6, 8, 9, 12)


def extra_args(p):
    p.add_argument("--fine-shape-source", choices=("auto", "anchor", "rollout"), default="auto",
                   help="where the fine-shape latent comes from (see the module description)")


def main(argv=None) -> int:
    args = W.parse_args(__doc__.split("\n\n")[0], argv, extra_args)
    cfg = W.load(args)
    from fastpixal3d.training.environment import configure_process, load_pipeline, weight_paths

    configure_process(threads=cfg["runtime"]["threads"], tf32_override=cfg["runtime"]["tf32_override"])
    cond_dir = W.output_dir(cfg, "fine_shape_conditioning", args.dataset)
    shape_dir = W.output_dir(cfg, "fine_shape_anchors", args.dataset)
    out_dir = W.output_dir(cfg, "texture_anchors", args.dataset)
    from fastpixal3d.training.records import record_names

    names = record_names(cond_dir)
    if args.fine_shape_source == "anchor":
        names = sorted(set(names) & set(record_names(shape_dir) if shape_dir.is_dir() else []))
    if not W.pending(args, names, out_dir):
        print(json.dumps({"event": "nothing_to_do", "out_dir": str(out_dir)}))
        return 0
    import torch

    from fastpixal3d.training.records import conditioning_image, find_render, load_record
    from fastpixal3d.training.schedule import TEXTURE_NOISE_SEED
    from fastpixal3d.training.teacher import fine_shape_trajectory, texture_trajectory

    pipeline = load_pipeline(weight_paths(cfg["pixal3d"]["model"], cfg["pixal3d"]["source"]),
                             device=cfg["runtime"]["device"])
    device = torch.device(pipeline.device)
    roots = [Path(cfg["datasets"][args.dataset]["root"])]

    def fine_shape_latent(view: str, hr):
        anchor = shape_dir / f"{view}.pt"
        if args.fine_shape_source in ("auto", "anchor") and anchor.is_file():
            return load_record(anchor)["anchors"]["12"].float(), f"anchor:{anchor}"
        if args.fine_shape_source == "anchor":
            raise FileNotFoundError(f"no fine-shape anchor record for {view}")
        ret = fine_shape_trajectory(pipeline, hr, view, device)
        return ret.pred_x_t[STEPS - 1].feats.detach().float().cpu(), "rollout"

    def rollout(view: str):
        hr = load_record(cond_dir / f"{view}.pt", mmap=True)
        coords_cpu = hr["coords"].clone()
        concat, source = fine_shape_latent(view, hr)
        if concat.shape[0] != coords_cpu.shape[0]:
            raise ValueError(f"fine-shape latent has {concat.shape[0]} tokens, the support {coords_cpu.shape[0]}")
        render, camera, grid = dict(hr["render"]), {k: float(v) for k, v in hr["camera"].items()}, int(hr["grid_res"])
        image = conditioning_image(find_render(render, roots).read_bytes(), render.get("image_sha256"))
        coords = coords_cpu.to(device)
        with torch.no_grad():
            cond = pipeline.get_proj_cond_shape(pipeline.image_cond_model_tex_1024, [image], coords,
                                                camera_angle_x=camera["camera_angle_x"], distance=camera["distance"],
                                                mesh_scale=camera.get("mesh_scale", 1.0),
                                                grid_resolution_override=grid)["cond"]
        ret = texture_trajectory(pipeline, coords, concat.to(device), cond, device)
        return ret, hr, coords_cpu, concat, source, render, camera, grid

    def build(view: str):
        ret, hr, coords_cpu, concat, source, render, camera, grid = rollout(view)
        anchors = {str(s): ret.pred_x_t[s - 1].feats.detach().float().cpu() for s in STORED}
        return {"schema": SCHEMA[0], "schema_version": SCHEMA[1], "object_id": view, "coords": coords_cpu,
                "concat_cond": concat.float(), "anchors": anchors, "camera": camera, "grid_res": grid,
                "n_tokens": int(coords_cpu.shape[0]), "render": render,
                "provenance": {"teacher_steps": STEPS, "anchor_rescale_t": RESCALE_T, **GUIDANCE,
                               "noise_seed": TEXTURE_NOISE_SEED, "stored_steps": list(STORED),
                               "concat_cond": "final state of the fine-shape teacher trajectory (normalised)",
                               "concat_cond_source": source, "actual_resolution": hr.get("actual_resolution"),
                               "conditioning_dir": str(cond_dir)}}

    def verify(view: str, record) -> float:
        ret = rollout(view)[0]
        return W.rel_drift(ret.pred_x_t[STEPS - 1].feats.detach().cpu(), record["anchors"]["12"])

    W.run(args, out_dir, names, build, verify, tag="texture_anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
