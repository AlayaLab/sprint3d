#!/usr/bin/env python3
r"""Baseline manifest of eval/run_eval.py (``--baseline-manifest``) from the output directory of one adapter run.

    python experiments/baselines/make_baseline_manifest.py --dir runs/trellis2 --out manifests/trellis2.json
    python eval/run_eval.py --config eval/config.yaml --baseline-manifest manifests/trellis2.json \
        --out results/trellis2 --skip vs_reference

Reads ``<dir>/<id>_v<view>/camera.json`` (written by the adapters next to every output), keeps the inputs of the
list (``--list``: an object list with ``id``, ``sha256`` and ``view`` such as eval/lists/validation_400.csv, the
default, or an input list of make_input_list.py; matched by list id or sha256 and view) and writes one row per
input: ``method``, ``object_id``, ``sha256``, ``view``, ``geometry_glb``, ``textured_glb`` (the files the frame
map names), ``e2e_s`` (``--e2e-key``: ``stage_sum_s`` by default, or ``e2e_s``) and ``input_png``.

Each record must name the frame map that eval/frame_maps/baseline_frame_maps.json holds now for its method (same
family, matrix, depth sign and files); a product generated for another convention is refused, so the evaluation
never maps a mesh with a frame it was not generated for.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_io import SIDECAR, frame_map  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NAME = re.compile(r"^(?P<id>[0-9A-Za-z]+)_v(?P<view>\d+)$")
FRAME_KEYS = ("family", "matrix", "depth_sign", "geometry_file", "textured_file")


def scan(directory: Path) -> Dict[Tuple[str, int], Path]:
    found = {}
    for sidecar in sorted(directory.glob(f"*/{SIDECAR}")):
        m = NAME.match(sidecar.parent.name)
        if m:
            found[(m["id"], int(m["view"]))] = sidecar
    return found


def check_frame_map(record: Dict, sidecar: Path) -> str:
    method = record.get("method")
    if not method:
        raise SystemExit(f"{sidecar}: no method")
    current = frame_map(method)["entry"]
    recorded = record.get("frame_map") or {}
    diff = [k for k in FRAME_KEYS if recorded.get(k) != current.get(k)]
    if diff:
        raise SystemExit(f"{sidecar}: generated for another frame map of {method!r} (differs in {diff}); "
                         f"eval/frame_maps/baseline_frame_maps.json holds the convention the evaluation applies")
    return method


def list_rows(path: Path) -> List[Dict]:
    """``[{id, sha256, view}]`` from an object list (CSV) or an input list of make_input_list.py (JSON lines)."""
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                m = NAME.match(r["id"])
                if not m or "sha256" not in r:
                    raise SystemExit(f"{path}: rows need an id <id>_v<view> and a sha256 (make_input_list.py)")
                rows.append({"id": m["id"], "sha256": r["sha256"], "view": int(m["view"])})
        return rows
    return list(csv.DictReader(open(path)))


def build(directory: Path, list_path: Path, e2e_key: str, method: str = None) -> Tuple[List[Dict], List[str]]:
    found = scan(directory)
    rows, missing = [], []
    for r in list_rows(Path(list_path)):
        view = int(r["view"])
        sidecar = found.get((r["id"], view)) or found.get((r["sha256"], view))
        if sidecar is None:
            missing.append(f"{r['id']}_v{view}")
            continue
        record = json.loads(sidecar.read_text())
        m = check_frame_map(record, sidecar)
        if method and m != method:
            raise SystemExit(f"{sidecar}: method {m!r}, expected {method!r}")
        entry = frame_map(m)["entry"]
        geometry = sidecar.parent / entry["geometry_file"]
        textured = sidecar.parent / entry["textured_file"] if entry.get("textured_file") else None
        for f in (geometry, textured):
            if f is not None and not f.is_file():
                raise SystemExit(f"{f} is missing")
        rows.append({"method": m, "object_id": r["id"], "sha256": r["sha256"], "view": view,
                     "geometry_glb": str(geometry.resolve()), "textured_glb": str(textured.resolve()) if textured else None,
                     "e2e_s": record.get(e2e_key), "input_png": record.get("image")})
    methods = {x["method"] for x in rows}
    if len(methods) > 1:
        raise SystemExit(f"{directory}: products of several methods {sorted(methods)}")
    return rows, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="output directory of one adapter run (one method)")
    ap.add_argument("--list", default=str(REPO / "eval" / "lists" / "validation_400.csv"),
                    help="object list (CSV with id, sha256, view) or input list of make_input_list.py (.jsonl)")
    ap.add_argument("--method", default=None, help="expected method (trellis2, triposg, hunyuan3d21)")
    ap.add_argument("--e2e-key", default="stage_sum_s", choices=("stage_sum_s", "e2e_s"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows, missing = build(Path(args.dir), Path(args.list), args.e2e_key, args.method)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} rows -> {args.out}; {len(missing)} list inputs without a product")
    if missing:
        print("first missing:", missing[:5], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
