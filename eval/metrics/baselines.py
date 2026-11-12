"""baselines.py -- ingestion of external image-to-3D baselines.

Frame maps are fixed once per method in frame_maps/baseline_frame_maps.json and never chosen per sample (row-vector
convention, v_out = v_in @ M):

  family "canonical"         gravity-canonical Y-up GLB (TRELLIS.2): v_world = v_glb @ matrix with the exporter's axis
                             swap [[1,0,0],[0,0,1],[0,-1,0]] (= (gx, -gz, gy)); the yaw is not searched (pose error is
                             generation error).
  camera-referenced families the GLB is the model's raw frame with x right / y up as seen in the input image:
                             v_axes = v_glb @ diag(1, 1, depth_sign), then
      "view_aligned"         v_world = v_axes @ R_view      (full ground-truth camera rotation)
      "azimuth_aligned"      v_world = v_axes @ R_az        (gravity-up frame turned to the camera azimuth only: rows of
                             R_az = (camera right, world up, right x up); the camera elevation is not baked in)

After the frame map every mesh is normalised by its own bounding box in ground-truth world axes
(projection.normalise_own_bbox) -- the same rule as for Pixal3D-family products and the ground truth -- and rendered
with the ground-truth camera pose.  Geometry comes from the shape-stage file of each method (`geometry_file`),
textures (CLIP only) from `textured_file` when the method has a texture stage.  No decimation by the evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .conventions import FRAME_MAP_PATH
from .frame import ViewFrame

CAMERA_FAMILIES = ("view_aligned", "azimuth_aligned")


def load_frame_maps(path=None) -> dict:
    p = Path(path) if path else FRAME_MAP_PATH
    d = json.loads(p.read_text())
    for name, m in d["methods"].items():
        if m["family"] not in ("canonical", *CAMERA_FAMILIES):
            raise ValueError(f"{name}: unknown family {m['family']}")
        if m["family"] == "canonical" and np.asarray(m["matrix"]).shape != (3, 3):
            raise ValueError(f"{name}: canonical methods need a 3x3 matrix")
        if m["family"] in CAMERA_FAMILIES and m.get("depth_sign") not in (1, -1):
            raise ValueError(f"{name}: a camera-referenced family needs depth_sign +1 or -1")
    return d


def azimuth_rotation(view_frame: ViewFrame) -> np.ndarray:
    """rows = (camera right [horizontal], world up, right x up) in ground-truth world coordinates (Z-up)."""
    right = np.array(view_frame.c2w[:3, 0], np.float64)
    right[2] = 0.0
    n = np.linalg.norm(right)
    if n < 1e-6:
        raise ValueError("camera right vector is vertical: azimuth undefined")
    right /= n
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([right, up, np.cross(right, up)], 0)


def to_world_axes(spec: dict, view_frame: ViewFrame, v_glb) -> np.ndarray:
    """fixed per-method frame map: GLB coordinates -> ground-truth world axes (rotation / axis map only)."""
    v = np.asarray(v_glb, np.float64)
    if spec["family"] == "canonical":
        return v @ np.asarray(spec["matrix"], np.float64)
    if spec.get("depth_sign") not in (1, -1):
        raise ValueError("depth_sign missing")
    v = v @ np.diag([1.0, 1.0, float(spec["depth_sign"])])
    if spec["family"] == "view_aligned":
        return v @ view_frame.R_view
    if spec["family"] == "azimuth_aligned":
        return v @ azimuth_rotation(view_frame)
    raise ValueError(f"unknown family {spec['family']!r}")


def rows_from_baseline_manifest(path, ds, frame_maps: dict) -> list:
    """manifest rows (JSON list or CSV): method, object_id | sha256, view, geometry_glb, [textured_glb], [e2e_s],
    [input_png], [repeat]"""
    from .manifest import read_table
    rows = []
    for r in read_table(path):
        method = r["method"]
        if method not in frame_maps["methods"]:
            raise KeyError(f"baseline method {method!r} has no frame map")
        key = str(r.get("sha256") or r["object_id"])
        sha = ds.resolve(key)
        if sha is None:
            raise KeyError(f"baseline manifest object {key!r} not in the dataset")
        spec = frame_maps["methods"][method]
        rows.append({"arm_label": method, "arm": spec.get("display", method), "method": method, "condition": "baseline",
                     "object_id": ds.object_id(sha), "sha256": sha, "view": int(r["view"]),
                     "repeat": int(r.get("repeat") or 0), "product_glb": r["geometry_glb"],
                     "textured_glb": r.get("textured_glb") or None, "camera": None,
                     "e2e_s": float(r["e2e_s"]) if r.get("e2e_s") not in (None, "") else None,
                     "support_npz": None, "input_png": r.get("input_png") or None})
    return rows
