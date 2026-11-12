#!/usr/bin/env python3
"""Pixal3D (official): run the released Pixal3D inference script at its defaults on one or many inputs.

    python scripts/run_official.py --image input.png --output out/obj_v00.glb
    python scripts/run_official.py --list inputs.jsonl --out-dir out/          (JSON lines with id and path)

For every input this calls, unmodified, ``run_inference(image_path=<input>, output_path=<out>/<id>.glb,
model_path=<weights>)`` from Pixal3D's ``inference.py`` with all other arguments at their defaults -- the
same call as ``python inference.py --image <input> --output <out>/<id>.glb --model_path <weights>``
(seed 42, 12 steps per stage, MoGe-2 camera, 1,000,000 faces, 4096 x 4096 WebP textures). The models are
loaded once per process; see ``fastpixal3d/official.py`` for exactly what is added around the call.

Outputs per input, next to each other: ``<id>.glb`` and ``<id>.json`` (the inference-time camera, same
format as ``scripts/infer.py``; docs/INSTALL.md "Output format"). One JSON line per input is appended to
``runs.jsonl`` in the output directory (timings, statistics, or the error).

TF32: by default ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` is removed before PyTorch is imported (stock PyTorch
precision, as the reference was measured); ``--tf32 on`` keeps the value of the environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--list", help="JSON lines file with 'id' and 'path' per input (processed in order)")
    ap.add_argument("--output", help="output .glb path (with --image)")
    ap.add_argument("--out-dir", help="output directory (with --list): <id>.glb, <id>.json, runs.jsonl")
    ap.add_argument("--object-id", default=None, help="(--image) default: output stem")
    ap.add_argument("--seed", type=int, default=None, help="default: the script's own default (42)")
    ap.add_argument("--pixal3d", default=None,
                    help="Pixal3D weights: local directory or Hugging Face repo id (default: FASTPIXAL3D_PIXAL3D, "
                         "else <weights dir>/pixal3d, else the pinned revision on Hugging Face)")
    ap.add_argument("--pixal3d-src", default=None, help="Pixal3D source checkout (default: third_party/Pixal3D)")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--torch-home", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="OpenMP / MKL threads")
    ap.add_argument("--tf32", choices=("off", "on"), default="off",
                    help="off: remove TORCH_ALLOW_TF32_CUBLAS_OVERRIDE (default); on: keep the environment's value")
    ap.add_argument("--autotune-cache", default=None,
                    help="FlexGEMM autotune cache file (default: <output directory>/autotune_cache.json, "
                         "started from the Pixal3D checkout's cache file if there is one)")
    ap.add_argument("--resume", action="store_true", help="skip inputs whose GLB and camera file exist")
    args = ap.parse_args(argv)
    if args.image and not args.output:
        ap.error("--image needs --output")
    if args.list and not args.out_dir:
        ap.error("--list needs --out-dir")
    return args


def read_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append((str(row.get("id") or row.get("object_id")), row["path"]))
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.image:
        output = Path(args.output).resolve()
        jobs = [(args.object_id or output.stem, str(Path(args.image).resolve()), output)]
        out_dir = output.parent
    else:
        out_dir = Path(args.out_dir).resolve()
        jobs = [(i, str(Path(p).resolve()), out_dir / f"{i}.glb") for i, p in read_list(args.list)]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = str(Path(args.autotune_cache).resolve()) if args.autotune_cache else None

    from fastpixal3d.official import configure_official_process
    from fastpixal3d.weights import WeightPaths

    weights = WeightPaths.from_env(pixal3d_src=args.pixal3d_src, pixal3d=args.pixal3d, hf_home=args.hf_home,
                                   torch_home=args.torch_home, offline=True if args.offline else None)
    configure_official_process(threads=args.threads, tf32_override=args.tf32 == "on", autotune_cache=cache)
    from fastpixal3d.official import OfficialPixal3D
    from fastpixal3d.pipeline import camera_record

    model = OfficialPixal3D.load(weights, work_dir=out_dir)
    print(f"[official] loaded in {model.info['load_s']:.1f} s ({model.info['model_path']})", flush=True)
    failed = 0
    with open(out_dir / "runs.jsonl", "a") as log:
        for stem, path, output in jobs:
            camera_file = output.with_suffix(".json")
            if args.resume and output.is_file() and camera_file.is_file():
                continue
            record = {"id": stem, "image": path, "glb": str(output), "started_unix": time.time()}
            try:
                result = model.run(path, output, object_id=stem, seed=args.seed)
                camera_file.write_text(json.dumps(camera_record(result, stem, path), indent=1))
                record.update(status="ok", **result)
                print(f"[official] {stem}: e2e {result['timings']['e2e_s']:.2f} s -> {output}", flush=True)
            except Exception as error:  # keep going; the error is recorded
                failed += 1
                record.update(status="error", error=f"{type(error).__name__}: {error}",
                              traceback=traceback.format_exc()[-3000:])
                print(f"[official] {stem}: {record['error']}", flush=True)
            log.write(json.dumps(record, default=str) + "\n")
            log.flush()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
