"""manifest.py -- evaluation rows: one row per (arm, sha256, view, repeat) product.

Product manifest (JSON list or CSV; eval/make_manifest.py writes one from a directory of GLBs):
  object_id      sha256 of the object, its list id, or an unambiguous sha256 prefix
  view           view index of the conditioning render (the list's `view` for the validation set)
  product_glb    the generated GLB
  arm            row label, e.g. pixal3d_official or ours (the condition suffix is appended: ours@official)
  condition      official (default) | manual_fov | true_camera
  camera_angle_x, distance, mesh_scale
                 the inference-time camera of the product (required for official / manual_fov rows): horizontal FoV in
                 radians, camera distance and mesh scale exactly as used by the generator
  e2e_s          optional end-to-end seconds recorded by the generator (reported in the speed appendix only)
  input_png      optional; default = the dataset's conditioning render of (object, view)
  repeat         optional, default 0
  support_npz    optional (true_camera rows): decoded 32^3 structure support of the product
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .conventions import CONDITION_SUFFIX, canonical_condition
from .dataset import GTDataset

CAMERA_KEYS = ("camera_angle_x", "distance", "mesh_scale")


def read_table(path) -> list:
    p = Path(path)
    return json.loads(p.read_text()) if p.suffix == ".json" else list(csv.DictReader(open(p)))


def base_arm(arm: str) -> str:
    return arm.split("@", 1)[0]


def rows_from_manifest(path, ds: GTDataset, default_condition: str = "official") -> list:
    rows = []
    for r in read_table(path):
        key = str(r["object_id"])
        sha = ds.resolve(key)
        if sha is None:
            raise KeyError(f"manifest object {key!r} not in the dataset")
        cond = canonical_condition(r.get("condition") or default_condition)
        if cond == "baseline":
            raise ValueError("baseline rows go into the baseline manifest (they need a frame map)")
        camera = None
        if cond != "true_camera":
            if any(r.get(k) in (None, "") for k in CAMERA_KEYS):
                raise ValueError(f"{cond} row {key} v{r.get('view')} needs its inference camera {CAMERA_KEYS}")
            camera = {k: float(r[k]) for k in CAMERA_KEYS}
        arm = str(r.get("arm") or "product")
        rows.append({"arm_label": arm, "arm": base_arm(arm) + CONDITION_SUFFIX[cond], "condition": cond,
                     "object_id": ds.object_id(sha), "sha256": sha, "view": int(r["view"]),
                     "repeat": int(r.get("repeat") or 0), "product_glb": str(r["product_glb"]), "textured_glb": None,
                     "camera": camera, "e2e_s": float(r["e2e_s"]) if r.get("e2e_s") not in (None, "") else None,
                     "support_npz": r.get("support_npz") or None, "input_png": r.get("input_png") or None,
                     "method": None})
    return rows


def attach_gt(rows: list, ds: GTDataset) -> list:
    for r in rows:
        rec = ds.record(r["sha256"], r["view"])
        r["input_png"] = r.get("input_png") or str(rec.image_png)
        r["transforms_json"] = str(rec.transforms_json)
        r["mesh_dump"] = str(rec.mesh_dump)
        r["gt_support_npz"] = str(rec.support_npz)
        r["scale_json"] = str(rec.scale_json)
        r["dataset_object_id"] = rec.object_id
        r["category"] = rec.category
        r["missing_gt"] = ds.missing(r["sha256"], r["view"], with_support=r["condition"] == "true_camera")
    return rows
