#!/usr/bin/env python3
"""Fine-shape conditioning cache: the teacher's fine (1024) voxel support and its image conditioning.

    python train/caches/fine_shape_conditioning.py --config train/configs/caches.yaml --dataset curated_10k

For each view with a structure conditioning record (structure_conditioning.py), Pixal3D's cascade is
run with fixed seeds, on the record's conditioning and on the same input image and camera:

1. structure: ``torch.manual_seed(42)``, ``pipeline.sample_sparse_structure(cond, 32, 1, {"steps": 12})``
   with the pipeline's default structure sampler settings (rescale_t 5, guidance 7.5 / 0.7 /
   [0.6, 1.0]) -> coarse support (32^3);
2. coarse shape: ``get_proj_cond_shape`` with the 512 extractor on that support (optionally stored as
   a coarse-shape conditioning record), then ``torch.manual_seed(0)``,
   ``pipeline.sample_shape_slat(cond, coarse flow, coords, {"steps": 12})``;
3. fine support: ``shape_slat_decoder.upsample(..., upsample_times=4)`` quantised to a grid of
   ``resolution // 16`` with ``resolution`` = 1024, lowered in steps of 128 (not below 768) while the
   support has 32,768 tokens or more (Pixal3D's own cascade rule);
4. fine conditioning: ``get_proj_cond_shape`` with the 1024 extractor on the fine support.

Record: ``{schema, schema_version, coords [N, 4] int32, global [5, 1024], proj [N, 2048], object_id,
camera, actual_resolution, grid_res, render, provenance}``; ``global`` and ``proj`` are stored as
bfloat16 by default (the shape flows cast their conditioning to bfloat16, so this is lossless for
them; ``--storage-dtype float32`` keeps float32).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import cachework as W  # noqa: E402

SCHEMA = ("fine_shape_conditioning", 1)
COARSE_SCHEMA = ("coarse_shape_conditioning", 1)
SS_SEED, COARSE_SEED, STEPS, SS_RESOLUTION = 42, 0, 12, 32
LR_RESOLUTION, TARGET_RESOLUTION, MIN_RESOLUTION, DECREMENT, MAX_TOKENS, UPSAMPLE = 512, 1024, 768, 128, 32768, 4


def quantize_fine_coords(hr_coords, target=TARGET_RESOLUTION, minimum=MIN_RESOLUTION, max_tokens=MAX_TOKENS):
    """Upsampled coarse coordinates (on the 512 grid) -> unique fine coordinates, the resolution and the
    grid size (``resolution // 16``). The resolution drops by 128 while the support has ``max_tokens``
    tokens or more, but not below ``minimum``."""
    import torch

    resolution = target
    while True:
        grid = resolution // 16
        quant = torch.cat([hr_coords[:, :1],
                           ((hr_coords[:, 1:] + 0.5) / LR_RESOLUTION * (grid - 1)).round().int()], dim=1)
        unique = quant.unique(dim=0)
        if unique.shape[0] < max_tokens or resolution == minimum:
            return unique, resolution, grid
        resolution -= DECREMENT


def extra_args(p):
    p.add_argument("--storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")


def main(argv=None) -> int:
    args = W.parse_args(__doc__.split("\n\n")[0], argv, extra_args)
    cfg = W.load(args)
    from fastpixal3d.training.environment import configure_process, load_pipeline, weight_paths

    configure_process(threads=cfg["runtime"]["threads"], tf32_override=cfg["runtime"]["tf32_override"])
    ss_dir = W.output_dir(cfg, "structure_conditioning", args.dataset)
    out_dir = W.output_dir(cfg, "fine_shape_conditioning", args.dataset)
    coarse_dir = (cfg["caches"].get("coarse_shape_conditioning") or {}).get(args.dataset)
    coarse_dir = Path(coarse_dir) if coarse_dir else None
    from fastpixal3d.training.records import record_names

    names = record_names(ss_dir)
    if not W.pending(args, names, out_dir):
        print(json.dumps({"event": "nothing_to_do", "out_dir": str(out_dir)}))
        return 0
    import torch

    from fastpixal3d.training.records import atomic_torch_save, conditioning_image, find_render, load_record

    weights = weight_paths(cfg["pixal3d"]["model"], cfg["pixal3d"]["source"])
    pipeline = load_pipeline(weights, device=cfg["runtime"]["device"])
    device = torch.device(pipeline.device)
    flow_512 = pipeline.models["shape_slat_flow_model_512"]
    decoder = pipeline.models["shape_slat_decoder"]
    store = torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32
    roots = [Path(cfg["datasets"][args.dataset]["root"])]
    ss_params = {**dict(pipeline.sparse_structure_sampler_params or {}), "steps": STEPS}
    lr_params = {**dict(pipeline.shape_slat_sampler_params or {}), "steps": STEPS}
    provenance = {"seeds": {"sparse_structure": SS_SEED, "shape_lr": COARSE_SEED},
                  "steps": {"sparse_structure": STEPS, "shape_lr": STEPS},
                  "sampler_params": {"sparse_structure": json.loads(json.dumps(ss_params, default=float)),
                                     "shape_lr": json.loads(json.dumps(lr_params, default=float))},
                  "cascade": {"lr_resolution": LR_RESOLUTION, "target_resolution": TARGET_RESOLUTION,
                              "min_resolution": MIN_RESOLUTION, "resolution_decrement": DECREMENT,
                              "upsample_times": UPSAMPLE, "max_num_tokens": MAX_TOKENS},
                  "storage": {"coords_dtype": "int32", "conditioning_dtype": args.storage_dtype}}

    def build(view: str):
        ss = load_record(ss_dir / f"{view}.pt")
        render, camera = dict(ss["render"]), {k: float(v) for k, v in ss["camera"].items()}
        image = conditioning_image(find_render(render, roots).read_bytes(), render.get("image_sha256"))
        cam = {"camera_angle_x": camera["camera_angle_x"], "distance": camera["distance"],
               "mesh_scale": camera["mesh_scale"]}
        g, p = ss["global"].to(device).unsqueeze(0), ss["proj"].to(device).unsqueeze(0)
        cond_ss = {"cond": {"global": g, "proj": p}, "neg_cond": {"global": torch.zeros_like(g), "proj": torch.zeros_like(p)}}
        torch.manual_seed(SS_SEED)
        with torch.no_grad():
            coords = pipeline.sample_sparse_structure(cond_ss, SS_RESOLUTION, 1, {"steps": STEPS})
        if coords.shape[0] == 0:
            raise ValueError("the structure rollout gave an empty support")
        with torch.no_grad():
            cond_lr = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_512, [image], coords, **cam)
        if coarse_dir is not None and not (coarse_dir / f"{view}.pt").exists():
            atomic_torch_save({"schema": COARSE_SCHEMA[0], "schema_version": COARSE_SCHEMA[1],
                               "coords": coords.detach().cpu().int(),
                               "global": cond_lr["cond"]["global"].detach().cpu().to(store).squeeze(0),
                               "proj": cond_lr["cond"]["proj"].feats.detach().cpu().to(store),
                               "object_id": view, "camera": camera, "render": render,
                               "provenance": {"seeds": {"sparse_structure": SS_SEED}, "ss_resolution": SS_RESOLUTION,
                                              "sampler_params": {"sparse_structure": provenance["sampler_params"]["sparse_structure"]}}},
                              coarse_dir / f"{view}.pt")
        torch.manual_seed(COARSE_SEED)
        with torch.no_grad():
            lr_slat = pipeline.sample_shape_slat(cond_lr, flow_512, coords, {"steps": STEPS})
            hr_coords = decoder.upsample(lr_slat, upsample_times=UPSAMPLE)
        coords_hr, resolution, grid = quantize_fine_coords(hr_coords)
        with torch.no_grad():
            cond_hr = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_1024, [image], coords_hr, **cam,
                                                   grid_resolution_override=grid)
        return {"schema": SCHEMA[0], "schema_version": SCHEMA[1],
                "coords": coords_hr.detach().cpu().to(torch.int32),
                "global": cond_hr["cond"]["global"].detach().cpu().to(store).squeeze(0),
                "proj": cond_hr["cond"]["proj"].feats.detach().cpu().to(store),
                "object_id": view, "camera": camera, "actual_resolution": int(resolution), "grid_res": int(grid),
                "render": {**render, "structure_conditioning": str(ss_dir / f"{view}.pt"),
                           "coarse_tokens": int(coords.shape[0])},
                "provenance": provenance}

    def verify(view: str, record) -> float:
        again = build(view)
        if not torch.equal(again["coords"], record["coords"]):
            return float("inf")
        return W.rel_drift(again["proj"], record["proj"])

    W.run(args, out_dir, names, build, verify, tag="fine_shape_conditioning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
