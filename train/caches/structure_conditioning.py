#!/usr/bin/env python3
"""Structure conditioning cache: the image conditioning of Pixal3D's structure flow, per rendered view.

    python train/caches/structure_conditioning.py --config train/configs/caches.yaml --dataset train_5k
    python train/caches/structure_conditioning.py --config train/configs/caches.yaml --dataset curated_10k \\
        --shard 0 --nshard 4

For each view ``<key>.viewNN`` of the dataset's object list:

    image    renders_cond/<sha256>/<NNN>.png: RGBA, resized to 512 x 512 with Lanczos when larger
             (the 10k renders are 1024 x 1024; the 5k renders are 512 x 512 and are used as they
             are), composited on black (``rgb * alpha``)
    camera   the render's own camera: field of view and distance from transforms.json, mesh scale
             recomputed from mesh_dumps/<sha256>.pickle (fastpixal3d.training.records.true_camera)
    cond     Pixal3D's ``get_proj_cond_ss([image], camera)``: global [5, 1024] and projected
             [4096, 1024] DINOv3 features, float32, computed with deterministic algorithms and
             PyTorch's own TF32 switches off (the numeric profile the caches were built with)

The record ``{schema, schema_version, global, proj, object_id, camera, render}`` is the conditioning
of the structure student (train/structure_traj) and the input of structure_anchors.py and
fine_shape_conditioning.py. ``render`` records where the image came from and its sha256.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import cachework as W  # noqa: E402

SCHEMA = ("structure_conditioning", 1)
IMAGE_RECIPE = "RGBA render -> 512x512 (PIL Lanczos when larger) -> rgb*alpha on black, uint8"
CAMERA_SOURCE = ("transforms.json: camera_angle_x, distance = |c2w[:3,3]|; mesh_scale = "
                 "(0.49999 / max|transform_mesh(sphere_normalize(mesh dump))|) / sphere_radius")


def main(argv=None) -> int:
    args = W.parse_args(__doc__.split("\n\n")[0], argv)
    cfg = W.load(args)
    from fastpixal3d.training.environment import (configure_process, deterministic_torch, load_pipeline,
                                                  toolkit_utils, weight_paths)

    configure_process(threads=cfg["runtime"]["threads"], tf32_override=cfg["runtime"]["tf32_override"],
                      deterministic_cublas=True)
    ds = W.dataset(cfg, args.dataset)
    rows = W.list_rows(ds)
    out_dir = W.output_dir(cfg, "structure_conditioning", args.dataset)
    if not W.pending(args, sorted(rows), out_dir):
        print(json.dumps({"event": "nothing_to_do", "out_dir": str(out_dir)}))
        return 0
    import torch

    from fastpixal3d.training.records import (composite_on_black, concatenate_mesh_dump, resize_rgba, sha256_bytes,
                                              true_camera)

    weights = weight_paths(cfg["pixal3d"]["model"], cfg["pixal3d"]["source"])
    pipeline = load_pipeline(weights, device=cfg["runtime"]["device"])
    deterministic_torch()
    utils = toolkit_utils(weights)
    root = Path(ds["root"])

    def encode(view: str):
        row = rows[view]
        base = root / row["subset_dir"] if row["subset_dir"] else root
        render_dir = base / "renders_cond" / row["sha256"]
        transforms_bytes = (render_dir / "transforms.json").read_bytes()
        frame = json.loads(transforms_bytes)["frames"][row["view"]]
        name = f"{row['view']:03d}.png"
        if frame.get("file_path") != name:
            raise ValueError(f"transforms.json frame {row['view']} is {frame.get('file_path')!r}, expected {name!r}")
        png = (render_dir / name).read_bytes()
        mesh = concatenate_mesh_dump(pickle.loads((base / "mesh_dumps" / f"{row['sha256']}.pickle").read_bytes()))
        camera = true_camera(frame, mesh, utils)
        rgba, source_size = resize_rgba(png, 512)
        image = composite_on_black(rgba)
        with torch.no_grad():
            cond = pipeline.get_proj_cond_ss([image], camera_angle_x=camera["camera_angle_x"],
                                             distance=camera["distance"], mesh_scale=camera["mesh_scale"])["cond"]
        render = {"subset": row["subset_dir"], "sha256": row["sha256"], "view": row["view"],
                  "image_path": str(render_dir / name), "image_sha256": sha256_bytes(png),
                  "transforms_sha256": sha256_bytes(transforms_bytes), "source_image_size": list(source_size),
                  "image_recipe": IMAGE_RECIPE, "camera_source": CAMERA_SOURCE,
                  "numeric_profile": "deterministic algorithms, PyTorch TF32 switches off"}
        return cond, camera, render

    def build(view: str):
        cond, camera, render = encode(view)
        return {"schema": SCHEMA[0], "schema_version": SCHEMA[1],
                "global": cond["global"].detach().cpu().to(torch.float32).squeeze(0),
                "proj": cond["proj"].detach().cpu().to(torch.float32).squeeze(0),
                "object_id": view, "camera": {k: float(v) for k, v in camera.items()}, "render": render}

    def verify(view: str, record) -> float:
        again = build(view)
        return max(W.rel_drift(again["global"], record["global"]), W.rel_drift(again["proj"], record["proj"]))

    W.run(args, out_dir, sorted(rows), build, verify, tag="structure_conditioning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
