#!/usr/bin/env python3
"""Shape conditioning caches of the 5k set, prepared as Pixal3D prepares an in-the-wild input.

    python train/caches/shape_conditioning_wild.py --config train/configs/caches.yaml --dataset train_5k_images
    python train/caches/shape_conditioning_wild.py --config train/configs/caches.yaml --dataset train_5k_images \\
        --stage coarse --shard 0 --nshard 4

The coarse- and fine-shape students were trained on one conditioning image per object of the 5k set
(``<root>/<subset dir>/<id>/image.png``), taken through the input path of Pixal3D's inference script rather than
the raw-render / true-camera path of the other caches:

1. ``pipeline.preprocess_image``: background removal (RMBG-2.0 when the image has no usable alpha) and crop;
2. camera: MoGe-2 on the prepared image (``inference.get_camera_params_wild_moge``, mesh scale 1);
3. coarse support: ``torch.manual_seed(42)``, the structure conditioning of the prepared image and
   ``pipeline.sample_sparse_structure(cond, 32, 1, {})`` (the pipeline's default sampler settings: 12 steps,
   guidance 7.5 / 0.7 / [0.6, 1.0], rescale_t 5); coarse conditioning: ``get_proj_cond_shape`` with the 512
   extractor on that support;
4. fine support: ``torch.manual_seed(0)``, the coarse flow for 12 steps (default guidance), the shape decoder's
   x4 upsampling, quantised to the grid of resolution 1024, lowered by 128 (not below 768) while 32,768 tokens
   or more remain (the training cap; the released pipeline's inference cap is 49,152 at 1536); fine
   conditioning: ``get_proj_cond_shape`` with the 1024 extractor on the fine support.

Records are named by object id (no view suffix). Coarse: ``{schema, schema_version, coords [N, 4] int32,
global [5, 1024], proj [N, 2048], object_id, camera, render, provenance}``; fine: the same plus
``actual_resolution`` and ``grid_res``. ``global`` and ``proj`` are stored as bfloat16 by default, which is lossless
for the shape flows (they cast their conditioning to bfloat16); ``--storage-dtype float32`` keeps the float32 of
the original caches. Objects whose structure sample is empty are skipped (253 of the 5,243 objects of the
original build, 250 for the coarse cache).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import cachework as W  # noqa: E402
from fastpixal3d.training import config as C  # noqa: E402

COARSE_SCHEMA = ("coarse_shape_conditioning_wild", 1)
FINE_SCHEMA = ("fine_shape_conditioning_wild", 1)
SS_SEED, COARSE_SEED, STEPS, SS_RESOLUTION = 42, 0, 12, 32
IMAGE_RECIPE = ("Pixal3D inference input path: pipeline.preprocess_image (background removal, crop), camera from "
                "MoGe-2 on the prepared image (get_camera_params_wild_moge, mesh_scale 1, extend_pixel 0, 512)")


def _fine_shape_conditioning_module():
    """The sibling builder's support quantisation (one definition for both caches)."""
    path = Path(__file__).resolve().with_name("fine_shape_conditioning.py")
    spec = importlib.util.spec_from_file_location("fastpixal3d_fine_shape_conditioning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extra_args(p):
    p.add_argument("--stage", choices=("coarse", "fine", "both"), default="both",
                   help="which records to write (both: the fine record and, when missing, the coarse one)")
    p.add_argument("--storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--keep-prepared", action="store_true", help="keep the prepared images under <out>/_prepared/")


def object_rows(ds):
    """object id -> {source, subset_dir, image} for every row of the list."""
    import csv

    C.require(ds["list"] is not None, "the dataset has no object list")
    out = {}
    with open(ds["list"], newline="") as handle:
        for row in csv.DictReader(handle):
            key = row[ds["key"]]
            subset = ds["subset_dirs"].get(row.get("source", ""), "")
            base = Path(ds["root"]) / subset if subset else Path(ds["root"])
            out[key] = {"source": row.get("source", ""), "subset_dir": subset, "sha256": row.get("sha256", ""),
                        "image": base / key / ds["image_name"]}
    return out


def main(argv=None) -> int:
    args = W.parse_args(__doc__.split("\n\n")[0], argv, extra_args)
    cfg = W.load(args)
    from fastpixal3d.training.environment import configure_process, load_moge, load_pipeline, weight_paths

    configure_process(threads=cfg["runtime"]["threads"], tf32_override=cfg["runtime"]["tf32_override"])
    ds = W.dataset(cfg, args.dataset)
    rows = object_rows(ds)
    coarse_dir = W.output_dir(cfg, "coarse_shape_conditioning_wild", args.dataset) if args.stage != "fine" else None
    fine_dir = W.output_dir(cfg, "fine_shape_conditioning_wild", args.dataset) if args.stage != "coarse" else None
    out_dir = fine_dir if fine_dir is not None else coarse_dir
    names = sorted(rows)
    if not W.pending(args, names, out_dir):
        print(json.dumps({"event": "nothing_to_do", "out_dir": str(out_dir)}))
        return 0
    import torch
    from PIL import Image

    from fastpixal3d.training.records import atomic_torch_save, sha256_file

    quantize_fine_coords = _fine_shape_conditioning_module().quantize_fine_coords
    weights = weight_paths(cfg["pixal3d"]["model"], cfg["pixal3d"]["source"])
    pipeline = load_pipeline(weights, device=cfg["runtime"]["device"])
    inference, moge = load_moge(weights, device=cfg["runtime"]["device"])
    device = torch.device(pipeline.device)
    store = torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32
    prepared_dir = out_dir / "_prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    ss_params = {**dict(pipeline.sparse_structure_sampler_params or {}), "steps": STEPS}
    lr_params = {**dict(pipeline.shape_slat_sampler_params or {}), "steps": STEPS}
    provenance = {"image_recipe": IMAGE_RECIPE, "seeds": {"sparse_structure": SS_SEED, "shape_lr": COARSE_SEED},
                  "steps": {"sparse_structure": STEPS, "shape_lr": STEPS},
                  "sampler_params": {"sparse_structure": json.loads(json.dumps(ss_params, default=float)),
                                     "shape_lr": json.loads(json.dumps(lr_params, default=float))},
                  "ss_resolution": SS_RESOLUTION, "storage": {"coords_dtype": "int32", "conditioning_dtype": args.storage_dtype}}

    def prepare(object_id: str):
        row = rows[object_id]
        image_path = Path(row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        source = Image.open(image_path)
        try:
            image = pipeline.preprocess_image(source)
        except ValueError:  # a broken alpha channel (empty object box): fall back to background removal on the RGB
            image = pipeline.preprocess_image(source.convert("RGB"))
        prepared = prepared_dir / f"{object_id}.png"
        image.save(prepared)
        try:
            camera = inference.get_camera_params_wild_moge(str(prepared), moge, device=str(device), mesh_scale=1.0,
                                                           extend_pixel=0, image_resolution=512)
        finally:
            if not args.keep_prepared and prepared.exists():
                prepared.unlink()
        camera = {k: float(v) for k, v in camera.items()}
        render = {"source": row["source"], "subset": row["subset_dir"], "sha256": row["sha256"],
                  "image_path": str(image_path), "image_sha256": sha256_file(image_path), "image_recipe": IMAGE_RECIPE,
                  "prepared_size": list(image.size)}
        return image, camera, render

    def coarse_record(object_id: str, image, camera, render):
        torch.manual_seed(SS_SEED)
        with torch.no_grad():
            cond_ss = pipeline.get_proj_cond_ss([image], **camera)
            coords = pipeline.sample_sparse_structure(cond_ss, SS_RESOLUTION, 1, {})
        if coords.shape[0] == 0:
            raise ValueError("the structure sample gave an empty support")
        with torch.no_grad():
            cond_lr = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_512, [image], coords, **camera)
        record = {"schema": COARSE_SCHEMA[0], "schema_version": COARSE_SCHEMA[1],
                  "coords": coords.detach().cpu().to(torch.int32),
                  "global": cond_lr["cond"]["global"].detach().cpu().to(store).squeeze(0),
                  "proj": cond_lr["cond"]["proj"].feats.detach().cpu().to(store),
                  "object_id": object_id, "camera": camera, "render": render,
                  "provenance": {k: v for k, v in provenance.items() if k != "steps"} | {
                      "steps": {"sparse_structure": STEPS},
                      "sampler_params": {"sparse_structure": provenance["sampler_params"]["sparse_structure"]}}}
        return record, coords, cond_lr

    def fine_record(object_id: str, image, camera, render, coords, cond_lr):
        torch.manual_seed(COARSE_SEED)
        with torch.no_grad():
            lr_slat = pipeline.sample_shape_slat(cond_lr, pipeline.models["shape_slat_flow_model_512"], coords, {"steps": STEPS})
            hr_coords = pipeline.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
        coords_hr, resolution, grid = quantize_fine_coords(hr_coords)
        with torch.no_grad():
            cond_hr = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_1024, [image], coords_hr, **camera,
                                                   grid_resolution_override=grid)
        return {"schema": FINE_SCHEMA[0], "schema_version": FINE_SCHEMA[1],
                "coords": coords_hr.detach().cpu().to(torch.int32),
                "global": cond_hr["cond"]["global"].detach().cpu().to(store).squeeze(0),
                "proj": cond_hr["cond"]["proj"].feats.detach().cpu().to(store),
                "object_id": object_id, "camera": camera, "actual_resolution": int(resolution), "grid_res": int(grid),
                "render": {**render, "coarse_tokens": int(coords.shape[0])},
                "provenance": {**provenance, "cascade": {"lr_resolution": 512, "target_resolution": 1024, "min_resolution": 768,
                                                         "resolution_decrement": 128, "upsample_times": 4, "max_num_tokens": 32768}}}

    def build(object_id: str):
        image, camera, render = prepare(object_id)
        coarse, coords, cond_lr = coarse_record(object_id, image, camera, render)
        if coarse_dir is not None and (args.stage == "coarse" or not (coarse_dir / f"{object_id}.pt").exists()):
            if args.stage == "coarse":
                return coarse
            atomic_torch_save(coarse, coarse_dir / f"{object_id}.pt")
        return fine_record(object_id, image, camera, render, coords, cond_lr)

    def verify(object_id: str, record) -> float:
        again = build(object_id)
        if not torch.equal(again["coords"], record["coords"]):
            return float("inf")
        return W.rel_drift(again["proj"], record["proj"])

    W.run(args, out_dir, names, build, verify, tag=f"shape_conditioning_wild[{args.stage}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
