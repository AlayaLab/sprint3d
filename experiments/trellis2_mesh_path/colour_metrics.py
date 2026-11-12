#!/usr/bin/env python3
r"""Base-colour comparison of TRELLIS.2 outputs: the "Delta color" column of the report's TRELLIS.2 table and the
agreement of the base-colour renders (42.7 dB against TRELLIS.2 as released, 43.1 dB between two exports of the same
latents).

    python experiments/trellis2_mesh_path/colour_metrics.py --config eval/config.yaml --list inputs/timing_80.jsonl \
        --reference stock=runs/trellis2_stock --run mesh_path=runs/trellis2_mesh/mesh_path \
        --run mesh_path_adaptive_remesh=runs/trellis2_mesh/mesh_path_adaptive_remesh \
        --run mesh_path_serial=runs/trellis2_mesh/mesh_path_serial --pair mesh_path=mesh_path_serial \
        --out results/trellis2_colour.json
    python experiments/trellis2_mesh_path/summarize_trellis2.py ... --colour results/trellis2_colour.json

The normal-map metrics of eval/ do not see the texture. This script renders the unlit base colour of the products of
every input (``<dir>/<id>/model.glb`` of time_trellis2.py; ``--list``: an input list of make_input_list.py, which
gives ``sha256`` and ``view``) and compares

  vs_input  each product, rendered at the input view, with the conditioning image composited on white. The table's
            Delta color is the paired difference (run - reference) of this PSNR per object (summarize_trellis2.py).
            Relative use only: the input is a shaded render and the products are unlit base colour.
  pairs     two products of the same input at five views: every run against the reference, and the pairs given
            with ``--pair``; the mean over the five views is the agreement of the two exports.

Placement: the reference product is mapped as eval/run_eval.py maps a baseline row: the fixed frame map of
``trellis2`` (eval/frame_maps/baseline_frame_maps.json), then normalised by its own bounding box in ground-truth world
axes (metrics.projection.normalise_own_bbox). Every other product of the same input gets the same transform (the
reference's bounding-box centre and extent), so the products are pixel-registered where their geometry agrees; nothing
is fitted. Views: the ground-truth input camera (``in``), the same camera turned by 90, 180 and 270 degrees about the
world up axis (``y90``, ``y180``, ``y270``) and a view from below (``under``: -60 degrees elevation at the input
azimuth, same distance); 512 x 512, perspective with the input field of view. Rendering: the renderer of
eval/metrics/render.py (nvdiffrast, exact float32), unlit base colour (base-colour texture x baseColorFactor), white
background. Metrics of an image pair: PSNR over the full image, SSIM (metrics.normal_maps.ssim_map, mean over the
image), LPIPS (AlexNet, metrics.normal_maps.LPIPSScorer) and, for two products, the IoU of their masks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "eval") not in sys.path:
    sys.path.insert(0, str(REPO / "eval"))

RES = 512
METHOD = "trellis2"
UNDER_ELEVATION_DEG = -60.0


def named(values: List[str], flag: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"{flag} expects NAME=DIR, got {v!r}")
        name, path = v.split("=", 1)
        if not name or name in out:
            raise SystemExit(f"{flag}: empty or repeated name {name!r}")
        out[name] = Path(path)
    return out


def look_c2w(pos, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Blender camera-to-world matrix of a camera at ``pos`` looking at ``target``."""
    pos, target, up = (np.asarray(x, np.float64) for x in (pos, target, up))
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, u, -f, pos
    return c2w


def views_of(view_frame) -> Dict[str, np.ndarray]:
    from metrics.render import yaw_c2w

    c2w = view_frame.c2w
    cam = np.asarray(c2w[:3, 3], np.float64)
    dist = float(np.linalg.norm(cam))
    az = math.atan2(cam[1], cam[0])
    e = math.radians(UNDER_ELEVATION_DEG)
    up = (0.0, 0.0, 1.0)
    return {"in": c2w, "y90": yaw_c2w(c2w, 90.0, up=up), "y180": yaw_c2w(c2w, 180.0, up=up),
            "y270": yaw_c2w(c2w, 270.0, up=up),
            "under": look_c2w((dist * math.cos(e) * math.cos(az), dist * math.cos(e) * math.sin(az), dist * math.sin(e)))}


def unlit(rz, clip, faces, tex, res: int, device):
    """(image [res, res, 3] in [0, 1] on white, mask [res, res]) of the unlit base colour."""
    import torch

    rast, faces_i = rz.rasterize(clip, faces, res)
    rast = torch.flip(rast, dims=[0])
    mask = rast[..., 3] > 0
    if tex is not None:
        albedo = rz.texture(tex["tex"], rz.interpolate(tex["uv"], rast, faces_i))
        if tex.get("base") is not None:
            albedo = albedo * tex["base"].view(1, 1, 3)
    else:
        albedo = torch.full((res, res, 3), 0.8, device=device)
    return torch.where(mask[..., None], albedo.clamp(0, 1), torch.ones_like(albedo)), mask


def image_metrics(p, g, lpips: Optional[Callable] = None) -> Dict[str, float]:
    from metrics.normal_maps import ssim_map

    mse = float(((p - g) ** 2).mean())
    out = {"tex_psnr": round(10.0 * math.log10(1.0 / max(mse, 1e-12)), 4),
           "tex_ssim": round(float(ssim_map(p.permute(2, 0, 1)[None], g.permute(2, 0, 1)[None]).mean()), 5)}
    if lpips is not None:
        out["tex_lpips"] = round(lpips(p, g), 5)
    return out


def product_glb(directory: Path, ident: str) -> Optional[Path]:
    glb = directory / ident / "model.glb"
    return glb if glb.is_file() and glb.stat().st_size > 0 else None


def compare_input(row: Dict[str, Any], reference: Tuple[str, Path], runs: Dict[str, Path],
                  pairs: List[Tuple[str, str]], ds, rz, device, lpips=None, res: int = RES) -> Dict[str, Any]:
    """All comparisons of one input (one line of the input list)."""
    import torch

    from metrics import baselines, projection
    from metrics.frame import ViewFrame
    from metrics.geometry import load_product_mesh
    from metrics.render import composite_rgba_png, mesh_texture_tensors, perspective_clip, world_to_cam

    ident, sha, view = row["id"], row["sha256"], int(row["view"])
    entry: Dict[str, Any] = {"sha256": sha, "view": view, "pairs": {}, "errors": {}}
    spec = baselines.load_frame_maps()["methods"][METHOD]
    ref_name, ref_dir = reference
    ref_glb = product_glb(ref_dir, ident)
    if ref_glb is None:
        raise RuntimeError(f"reference product of {ident} missing")
    rec = ds.record(sha, view)
    scale = rec.scale_json if Path(rec.scale_json).is_file() else None
    vf = ViewFrame.from_files(rec.transforms_json, view, scale_json=scale, sha256=sha)
    views = views_of(vf)
    gt_in = torch.as_tensor(composite_rgba_png(str(rec.image_png), bg=(255, 255, 255), size=res).astype(np.float32)
                            / 255.0, device=device)
    ref_mesh = load_product_mesh(ref_glb)
    _, nb = projection.normalise_own_bbox(baselines.to_world_axes(spec, vf, np.asarray(ref_mesh.vertices, np.float64)))
    centre, extent = np.asarray(nb["bbox_centre"]), float(nb["bbox_max_extent"])
    entry["placement"] = {"bbox_centre": centre.tolist(), "bbox_max_extent": extent}

    def renders(mesh):
        Vw = baselines.to_world_axes(spec, vf, np.asarray(mesh.vertices, np.float64))
        Vn = torch.as_tensor((Vw - centre) / extent, dtype=torch.float32, device=device)
        Ft = torch.as_tensor(np.asarray(mesh.faces, np.int64), dtype=torch.int64, device=device)
        tex = mesh_texture_tensors(mesh, device)
        images = {}
        for tag, c2w in views.items():
            Vc = world_to_cam(Vn, c2w)
            images[tag] = unlit(rz, perspective_clip(Vc, vf.camera_angle_x), Ft, tex, res, device)
        facts = {"textured": tex is not None, "faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices)),
                 "texture": list(mesh.visual.material.baseColorTexture.size) if tex is not None else None}
        return images, facts

    cache = {}
    cache[ref_name], facts = renders(ref_mesh)
    del ref_mesh
    entry["products"] = {ref_name: facts}
    entry["vs_input"] = {ref_name: image_metrics(cache[ref_name]["in"][0], gt_in, lpips)}
    for name, directory in runs.items():
        glb = product_glb(directory, ident)
        if glb is None:
            entry["errors"][name] = "product missing"
            continue
        mesh = load_product_mesh(glb)
        cache[name], entry["products"][name] = renders(mesh)
        del mesh
        entry["vs_input"][name] = image_metrics(cache[name]["in"][0], gt_in, lpips)
    for x, y in [(name, ref_name) for name in runs] + list(pairs):
        if x not in cache or y not in cache:
            continue
        per_view = {}
        for tag in views:
            (px, mx), (py, my) = cache[x][tag], cache[y][tag]
            m = image_metrics(px, py, lpips)
            m["mask_iou"] = round(float((mx & my).sum()) / max(float((mx | my).sum()), 1.0), 5)
            per_view[tag] = m
        keys = [k for k in ("tex_psnr", "tex_ssim", "tex_lpips", "mask_iou") if k in per_view["in"]]
        entry["pairs"][f"{x}__vs__{y}"] = {"per_view": per_view,
                                           "mean": {k: round(float(np.mean([v[k] for v in per_view.values()])), 5)
                                                    for k in keys}}
    return entry


def read_rows(path) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not all(k in r for k in ("id", "sha256", "view")):
                raise SystemExit(f"{path}: every line needs id, sha256 and view (experiments/baselines/make_input_list.py)")
            rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="input list of make_input_list.py (id, sha256, view)")
    ap.add_argument("--reference", required=True, metavar="NAME=DIR", help="TRELLIS.2 as released (time_trellis2.py)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=DIR", help="another arm (repeatable)")
    ap.add_argument("--pair", action="append", default=[], metavar="A=B",
                    help="also compare two runs with each other (e.g. mesh_path=mesh_path_serial)")
    ap.add_argument("--config", default=None, help="evaluation configuration (dataset.root, dataset.list, models)")
    ap.add_argument("--dataset-root", default=None, help="ground-truth root (overrides dataset.root)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS (no metric model needed)")
    ap.add_argument("--cpu", action="store_true", help="CPU rasteriser of eval/tests (tests only; edge pixels differ)")
    a = ap.parse_args(argv)
    reference = named([a.reference], "--reference")
    runs = named(a.run, "--run")
    ref_name = next(iter(reference))
    if ref_name in runs:
        raise SystemExit(f"{ref_name!r} is both the reference and a run")
    pairs = []
    for p in a.pair:
        x, _, y = p.partition("=")
        if x not in runs or y not in runs or x == y:
            raise SystemExit(f"--pair {p!r}: both names must be distinct --run names")
        pairs.append((x, y))

    import torch

    from metrics.config import apply_environment, load_config
    from metrics.dataset import GTDataset

    cfg = load_config(a.config)
    if a.dataset_root:
        cfg["dataset"]["root"] = str(Path(a.dataset_root).resolve())
    if not cfg["dataset"]["root"]:
        raise SystemExit("give --config with dataset.root or --dataset-root")
    apply_environment(cfg)
    ds = GTDataset(cfg["dataset"]["root"], list_csv=cfg["dataset"]["list"], source_dirs=cfg["dataset"]["source_dirs"])
    if a.cpu:
        sys.path.insert(0, str(REPO / "eval" / "tests"))
        from cpu_raster import CPURaster

        rz, device = CPURaster(), "cpu"
    else:
        from metrics import precision
        from metrics.render import Raster

        precision.enforce_exact_fp32()
        rz, device = Raster("cuda"), "cuda"
    lpips = None
    if not a.no_lpips:
        from metrics.normal_maps import LPIPSScorer

        lpips = LPIPSScorer(device, net="alex")
    rows = read_rows(a.list)[: a.limit or None]
    from metrics import baselines

    out = {"definition": __doc__, "frame_map": {"method": METHOD, **baselines.load_frame_maps()["methods"][METHOD]},
           "res": RES, "reference": ref_name, "runs": list(runs), "pairs": [f"{x}__vs__{y}" for x, y in pairs],
           "lpips": lpips is not None, "rasteriser": "cpu" if a.cpu else "nvdiffrast", "per_input": {},
           "started_unix": time.time()}
    with torch.no_grad():
        for row in rows:
            try:
                entry = compare_input(row, (ref_name, reference[ref_name]), runs, pairs, ds, rz, device, lpips)
            except Exception as error:  # recorded; the next input follows
                entry = {"sha256": row["sha256"], "view": int(row["view"]), "pairs": {},
                         "errors": {"fatal": f"{error!r}\n{traceback.format_exc()[-800:]}"}}
            out["per_input"][row["id"]] = entry
            if device == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"input": row["id"], "pairs": {k: v["mean"] for k, v in entry["pairs"].items()},
                              "errors": list(entry["errors"])}), flush=True)
    out["finished_unix"] = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0 if not any(e["errors"] for e in out["per_input"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
