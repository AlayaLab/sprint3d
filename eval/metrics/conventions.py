"""conventions.py -- the declared conventions of the evaluation and the hashes that key every record.

The evaluation follows the protocol of the Pixal3D paper (arXiv 2605.10922, Sec. 3.2.2 and 4.1):

  * normal maps of the generated mesh and of the ground-truth mesh are rendered in the input-image frame at 512^2;
  * PSNR / SSIM / LPIPS are computed in two regions, both stored for every view:
      overlap  pixels covered by both prediction and ground truth (keys psnr_ov / ssim_ov / lpips_ov),
      full     all 512^2 pixels, each normal map composited on a constant background (keys psnr_full / ...);
    the region shown in the main table is a presentation choice (run_eval.py --table1-region), not a convention;
  * angular errors, threshold accuracies and Mean_B are restricted to the overlap;
  * Pixal3D-family products generated without a ground-truth camera (condition `official`) are rendered with their
    own inference-time projection; external baselines are rendered with the ground-truth camera pose after a fixed
    per-method frame map; products generated with the ground-truth camera form the condition `true_camera`;
  * 3D metrics: every mesh (all methods and the ground truth) is normalised by its own axis-aligned bounding box in
    ground-truth world axes (max extent 1, bbox-centred); no registration, no ground-truth-derived scale.

Every per-view record is keyed by arm|sha256|view|repeat|code:<hash>|conv:<hash>:
  code hash        = sha256 over (relative path, sha256) of run_eval.py and metrics/*.py
  conventions hash = sha256 of the canonical JSON of the effective conventions (this dict with the command-line
                     overrides applied, plus the sha256 of the baseline frame-map file)
so records computed by other code or under other conventions are never reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SUITE_VERSION = "1.0"
SUITE_DIR = Path(__file__).resolve().parent.parent
FRAME_MAP_PATH = SUITE_DIR / "frame_maps" / "baseline_frame_maps.json"

BACKGROUNDS = {"white": [1.0, 1.0, 1.0], "black": [0.0, 0.0, 0.0], "gray": [0.5, 0.5, 0.5]}

CONDITION_LABELS = ("true_camera", "official", "manual_fov", "baseline")
CONDITION_DISPLAY = {
    "true_camera": "with true camera (ground-truth intrinsics / distance / scale)",
    "official": "official inference path (MoGe-estimated camera)",
    "manual_fov": "official inference path (manual FoV)",
    "baseline": "external baseline (fixed frame map, ground-truth camera pose)",
}
CONDITION_SUFFIX = {"true_camera": "@true_camera", "official": "@official", "manual_fov": "@manual_fov"}

CONVENTIONS = {
    "suite_version": SUITE_VERSION,
    "reference": "Pixal3D, arXiv 2605.10922 (Sec. 3.2.2 inference camera, Sec. 4.1 evaluation)",
    "evaluation_image_frame": "the raw conditioning render renders_cond/<sha256>/<view>.png (square), rasterised at 512^2; "
                              "identical for every method and condition",
    "normal_maps": {"space": "camera (x right, y up, -z forward)", "shading": "flat (per-face geometric normal)",
                    "facing": "flipped to face the camera (prediction and ground truth alike)",
                    "encoding": "rgb = n * 0.5 + 0.5", "resolution": 512},
    "image_metrics": {
        "regions": {"overlap": "prediction & ground-truth pixels; PSNR over the overlap, SSIM map averaged over the "
                               "overlap, LPIPS(alex) with both images 0.5-gray outside the overlap (keys psnr_ov / "
                               "ssim_ov / lpips_ov)",
                    "full": "all 512^2 pixels, each normal map composited on the constant background (keys "
                            "psnr_full / ssim_full / lpips_full / lpips_full_vgg)"},
        "background_rgb": BACKGROUNDS["white"],
        "background_name": "white",
        "psnr": "10 log10(1 / MSE), MSE over the region's pixels and the 3 channels, data range 1",
        "ssim": "Wang et al., gaussian 11x11 sigma 1.5, K1 0.01, K2 0.03, data range 1, per channel, 'valid' "
                "convolution, mean of the map (overlap: mean of the map over the overlap)",
        "lpips_main": {"net": "alex", "version": "0.1", "keys": ["lpips_full", "lpips_ov"]},
        "lpips_secondary": {"net": "vgg", "version": "0.1", "keys": ["lpips_full_vgg"]},
    },
    "angular_metrics": {"region": "overlap (prediction & ground truth)", "thresholds_deg": [11.25, 22.5, 30.0],
                        "mean_b": "overlap & pixels within 10 px of the ground-truth mask boundary",
                        "iou": "mask IoU |pred & gt| / |pred | gt|"},
    "conditions": {
        "true_camera": {"display": CONDITION_DISPLAY["true_camera"],
                        "definition": "generated with the ground-truth intrinsics, distance and total scale (no crop); "
                                      "normal map rendered through the analytic frame chain with the ground-truth camera",
                        "row_label": "<arm>@true_camera"},
        "official": {"display": CONDITION_DISPLAY["official"],
                     "definition": "no ground-truth camera: official alpha-bbox crop, FoV estimated by MoGe-2 on the crop, "
                                   "mesh_scale 1.0, distance_from_fov (the official inference.py defaults); normal map "
                                   "rendered with the product's own inference-time projection",
                     "row_label": "<arm>@official"},
        "manual_fov": {"display": CONDITION_DISPLAY["manual_fov"],
                       "definition": "as `official` with a fixed manual FoV",
                       "row_label": "<arm>@manual_fov"},
        "baseline": {"display": CONDITION_DISPLAY["baseline"],
                     "definition": "external method: mesh mapped by the method's fixed frame map, normalised by its own "
                                   "bbox, rendered with the ground-truth camera pose; pose and scale error count as "
                                   "generation error; no ICP",
                     "row_label": "<method display name>"},
    },
    "own_projection": {
        "camera": "pinhole, horizontal FoV camera_angle_x, principal point at the image centre, at (0, 0, distance) of "
                  "the generation frame looking along -z: v_cam = (v_product @ P2D) / mesh_scale - (0, 0, distance)",
        "image_frame": "the official crop of the input image (metrics/official_crop.py); mapped to the raw image frame "
                       "by the exact NDC affine of the crop box (a function of the input image only)",
        "note": "the official code offsets the principal point by +0.5 px at the conditioning resolution; ignored "
                "(<= 0.1 % of the image)",
    },
    "geometry_3d": {
        "normalisation": "own_bbox: v' = (v - bbox_centre) / max_extent, computed per mesh in ground-truth world axes; "
                         "applied to every method's mesh and to the ground-truth mesh alike",
        "axes": "ground-truth world (Z-up frame of the mesh dump); Pixal3D-family and view-aligned products are "
                "rotated there by the ground-truth camera rotation (rotation only), canonical methods by their fixed "
                "axis swap",
        "chamfer": "non-squared L2, symmetric mean, 100k area-weighted surface samples per side (keys *_nb)",
        "f_score_tau": [0.005, 0.01, 0.02],
        "registration": "none",
        "decimation": "none by the evaluation (face counts recorded per row)",
        "voxel_frame": "true_camera rows only: values in the voxel frame of the input view with the ground-truth total "
                       "scale (keys chamfer_vox, f_*_vox) and the 32^3 support metrics",
    },
    "perception": {"clip": "textured methods only (n/a otherwise)", "clip_n": "all methods",
                   "placement": "true_camera: analytic chain; official / baselines: own-bbox normalised world placement"},
    "baselines": {"frame_map_file": "frame_maps/baseline_frame_maps.json",
                  "rule": "one frame map per method, fixed once (never per sample); view-aligned methods: ground-truth "
                          "camera rotation + fixed depth-axis sign; azimuth-aligned methods: gravity-up frame turned to "
                          "the camera azimuth; canonical methods: axis swap",
                  "geometry_source": "shape-stage output (Hunyuan3D-2.1: shape_untextured.glb, never the painted GLB)"},
}


def canonical_condition(label) -> str:
    label = (label or "official").strip()
    if label not in CONDITION_LABELS:
        raise ValueError(f"unknown condition {label!r}; expected one of {CONDITION_LABELS}")
    return label


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_code_files(suite_dir: Path = SUITE_DIR) -> list:
    files = [suite_dir / "run_eval.py"] + sorted((suite_dir / "metrics").glob("*.py"))
    return [f for f in files if f.is_file()]


def code_hash(suite_dir: Path = SUITE_DIR) -> tuple[str, dict]:
    per_file = {str(f.relative_to(suite_dir)): sha256_file(f) for f in metric_code_files(suite_dir)}
    h = hashlib.sha256()
    for rel in sorted(per_file):
        h.update(f"{rel}:{per_file[rel]}\n".encode())
    return h.hexdigest(), per_file


def effective_conventions(background: str = "white", frame_map_path: Path | None = None) -> dict:
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {sorted(BACKGROUNDS)}")
    conv = copy.deepcopy(CONVENTIONS)
    conv["image_metrics"]["background_rgb"] = list(BACKGROUNDS[background])
    conv["image_metrics"]["background_name"] = background
    fm = Path(frame_map_path) if frame_map_path else FRAME_MAP_PATH
    conv["baselines"]["frame_map_sha256"] = sha256_file(fm) if fm.is_file() else None
    return conv


def conventions_hash(conv: dict) -> str:
    return hashlib.sha256(json.dumps(conv, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record_key(row: dict, code_h: str, conv_h: str) -> str:
    return f"{row['arm']}|{row['sha256']}|{row['view']}|{row['repeat']}|code:{code_h[:16]}|conv:{conv_h[:16]}"
