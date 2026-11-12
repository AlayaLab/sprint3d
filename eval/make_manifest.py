#!/usr/bin/env python
"""make_manifest.py -- build the run_eval.py manifest of one arm from a directory of generated GLBs.

Expected layout (one GLB per input of the object list; <id> = the list id or the full sha256, <view> = the view index):
    <dir>/<id>_v<view>.glb   with   <dir>/<id>_v<view>.json
or  <dir>/<id>_v<view>/model.glb   with   <dir>/<id>_v<view>/camera.json
The JSON sidecar holds the inference-time camera used by the generator, at the top level or under "camera":
    {"camera_angle_x": <horizontal FoV, rad>, "distance": <camera distance>, "mesh_scale": <mesh scale>[, "e2e_s": <s>]}
(the official condition renders each product with its own inference-time projection, so the camera is required).

usage: python eval/make_manifest.py --dir GLB_DIR --arm ours [--list eval/lists/validation_400.csv] --out ours.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^(?P<id>[0-9a-f]{8,64})_v(?P<view>\d+)$")
CAMERA_KEYS = ("camera_angle_x", "distance", "mesh_scale")


def sidecar_of(glb: Path) -> Path | None:
    for c in (glb.with_suffix(".json"), glb.parent / "camera.json"):
        if c.is_file():
            return c
    return None


def scan(glb_dir: Path) -> dict:
    """{(id, view): (glb, sidecar)}"""
    out = {}
    for glb in sorted(glb_dir.rglob("*.glb")):
        stem = glb.parent.name if glb.stem == "model" else glb.stem
        m = NAME.match(stem)
        if m:
            out[(m["id"], int(m["view"]))] = (glb, sidecar_of(glb))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of GLBs")
    ap.add_argument("--arm", required=True, help="row label, e.g. ours or pixal3d_official")
    ap.add_argument("--condition", default="official", choices=["official", "manual_fov", "true_camera"])
    ap.add_argument("--list", default=str(Path(__file__).resolve().parent / "lists" / "validation_400.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    found = scan(Path(a.dir))
    by_key = {}
    for (oid, view), v in found.items():
        by_key[(oid, view)] = v
    rows, missing, no_camera = [], [], []
    for r in csv.DictReader(open(a.list)):
        view = int(r["view"])
        hit = by_key.get((r["id"], view)) or by_key.get((r["sha256"], view))
        if hit is None:
            missing.append(f"{r['id']}_v{view}")
            continue
        glb, side = hit
        row = {"object_id": r["sha256"], "view": view, "product_glb": str(glb.resolve()), "arm": a.arm,
               "condition": a.condition, "repeat": 0}
        if side is not None:
            d = json.loads(side.read_text())
            cam = d.get("camera", d)
            row.update({k: float(cam[k]) for k in CAMERA_KEYS if k in cam})
            if d.get("e2e_s") is not None:
                row["e2e_s"] = float(d["e2e_s"])
        if a.condition != "true_camera" and not all(k in row for k in CAMERA_KEYS):
            no_camera.append(f"{r['id']}_v{view}")
            continue
        rows.append(row)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} rows -> {a.out}; {len(missing)} list inputs without a GLB; {len(no_camera)} without a camera sidecar")
    if missing or no_camera:
        print("first missing:", (missing + no_camera)[:5], file=sys.stderr)


if __name__ == "__main__":
    main()
