#!/usr/bin/env python3
"""Image(s) -> textured GLB(s), with per-stage timings.

Single image:

    python scripts/infer.py --image input.png --output out/obj_v00.glb \
        --students /path/to/students --timing-json out/timing.json

Many inputs in one process (JSON lines with ``id`` and ``path``; ``id`` names the outputs):

    python scripts/infer.py --list inputs.jsonl --out-dir out/ --students ...

The GLB uses the export settings of Pixal3D's ``inference.py``: 1,000,000 faces (target) and a
4096 x 4096 texture.

Weights are read from the directory filled by ``scripts/download_weights.py`` (the weights dir:
``$FASTPIXAL3D_WEIGHTS``, by default ``weights/``); ``--students`` and ``--pixal3d`` override two of them.

Outputs per input: ``<stem>.glb`` and ``<stem>.json`` (the inference-time camera, see
docs/INSTALL.md "Output format"). With ``--dump-dir`` the intermediate results used by
``scripts/parity_check.py`` are written to ``<dump-dir>/<stem>/``.

The model is loaded once; ``--warmup`` untimed requests on the first image run before the
timed ones (the first requests of a process include one-off costs such as kernel autotuning
and CUDA graph capture). Stage keys are documented in ``fastpixal3d.runtime.timing``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="input image (RGBA with alpha, or RGB: background removed)")
    src.add_argument("--list", help="JSON lines file with 'id' and 'path' per input (processed in order)")
    ap.add_argument("--output", help="output .glb path (with --image)")
    ap.add_argument("--out-dir", help="output directory (with --list): <id>.glb and <id>.json")
    ap.add_argument("--timing-json", default=None, help="write timings and statistics here")
    ap.add_argument("--dump-dir", default=None, help="also write per-input parity records here")
    ap.add_argument("--students", default=None,
                    help="directory with students.json (default: FASTPIXAL3D_STUDENTS, else <weights dir>/students)")
    ap.add_argument("--student-checkpoints", default=None,
                    help="development: JSON mapping structure/coarse_shape/fine_shape/texture to training checkpoints")
    ap.add_argument("--pixal3d", default=None,
                    help="Pixal3D weights: local directory or Hugging Face repo id (default: FASTPIXAL3D_PIXAL3D, "
                         "else <weights dir>/pixal3d, else the pinned revision on Hugging Face)")
    ap.add_argument("--pixal3d-src", default=None, help="Pixal3D source checkout (default: third_party/Pixal3D)")
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--torch-home", default=None)
    ap.add_argument("--offline", action="store_true", help="do not contact the Hugging Face hub")
    ap.add_argument("--object-id", default=None, help="(--image) seeds the fine-shape noise (default: output stem)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="untimed requests on the first input before the timed ones")
    ap.add_argument("--runs", type=int, default=1, help="(--image) timed requests (the last GLB is kept)")
    ap.add_argument("--disable", action="append", default=[], metavar="LEVER",
                    help="switch a lever off (repeatable); see docs/LEVERS.md")
    ap.add_argument("--fa2", action="store_true", help="use FlashAttention-2 for the FA3 levers (non-Hopper GPUs)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (OpenMP / torch)")
    ap.add_argument("--tf32", choices=("on", "off"), default="on",
                    help="on (default, the measured configuration): float32 cuBLAS matmuls may use TF32 "
                         "(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1); off: full float32")
    ap.add_argument("--autotune-cache", default=None, help="FlexGEMM autotune cache file (seeded copy recommended)")
    ap.add_argument("--work-dir", default=None, help="directory for the preprocessed images (default: output directory)")
    args = ap.parse_args(argv)
    if args.image and not args.output:
        ap.error("--image needs --output")
    if args.list and not args.out_dir:
        ap.error("--list needs --out-dir")
    return args


def read_list(path):
    items = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            items.append((str(row.get("id") or row.get("object_id")), row["path"]))
    return items


def main(argv=None) -> int:
    args = parse_args(argv)
    from fastpixal3d.pipeline import configure_process

    configure_process(threads=args.threads, autotune_cache=args.autotune_cache, tf32_override=args.tf32 == "on")
    from fastpixal3d.config import Config, Levers
    from fastpixal3d.pipeline import FastPixal3D, camera_record
    from fastpixal3d.weights import WeightPaths

    checkpoints = json.loads(Path(args.student_checkpoints).read_text()) if args.student_checkpoints else None
    weights = WeightPaths.from_env(pixal3d_src=args.pixal3d_src, pixal3d=args.pixal3d, students=args.students,
                                   student_checkpoints=checkpoints, hf_home=args.hf_home, torch_home=args.torch_home,
                                   offline=True if args.offline else None)
    config = Config(levers=Levers().with_disabled(args.disable),
                    attention_fallback="fa2" if args.fa2 else None, threads=args.threads)
    if args.image:
        output = Path(args.output)
        stem = args.object_id or output.stem
        jobs = [(stem, args.image, output)] * max(1, args.runs)
        out_dir = output.resolve().parent
    else:
        out_dir = Path(args.out_dir)
        jobs = [(stem, path, out_dir / f"{stem}.glb") for stem, path in read_list(args.list)]
    work_dir = Path(args.work_dir) if args.work_dir else out_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = FastPixal3D.load(config, weights)
    load_s = time.perf_counter() - started
    print(f"[infer] loaded in {load_s:.1f} s (attention backend: {model.info.get('fast_attention_backend')})", flush=True)

    warmups = []
    first_stem, first_path, _ = jobs[0]
    for i in range(args.warmup):
        res = model.run(first_path, work_dir / f"warmup_{i}.glb", object_id=first_stem, seed=args.seed, work_dir=work_dir)
        warmups.append(res["timings"]["e2e_s"])
        print(f"[infer] warmup {i}: e2e {res['timings']['e2e_s']:.3f} s", flush=True)

    runs = []
    for stem, path, output in jobs:
        dump = Path(args.dump_dir) / stem if args.dump_dir else None
        res = model.run(path, output, object_id=stem, seed=args.seed, work_dir=work_dir, dump_dir=dump)
        output.with_suffix(".json").write_text(json.dumps(camera_record(res, stem, path), indent=1))
        res["id"] = stem
        runs.append(res)
        print(f"[infer] {stem}: e2e {res['timings']['e2e_s']:.3f} s, {res['stats']['output_faces']} faces -> {output}",
              flush=True)

    if args.timing_json:
        report = {"export_settings": dataclasses.asdict(config.export), "levers_disabled": sorted(args.disable),
                  "attention": model.info.get("fast_attention_backend"), "load_s": round(load_s, 3),
                  "warmup_e2e_s": warmups, "load_info": model.info, "runs": runs,
                  "timings": runs[-1]["timings"] if runs else None, "environment": environment_record()}
        Path(args.timing_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.timing_json).write_text(json.dumps(report, indent=1, default=str))
    return 0


def environment_record():
    import platform

    import torch

    keys = ("OMP_NUM_THREADS", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF",
            "ATTN_BACKEND", "CUDA_VISIBLE_DEVICES")
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(), "env": {k: os.environ.get(k) for k in keys}}


if __name__ == "__main__":
    sys.exit(main())
