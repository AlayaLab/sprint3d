#!/usr/bin/env python3
"""Dedicated end-to-end timing run (protocol: docs/TIMING.md).

    python scripts/time_e2e.py --method fast --list inputs.jsonl --out-dir runs/fast \
        --students /path/to/students
    python scripts/time_e2e.py --method official --list inputs.jsonl --out-dir runs/official
    python scripts/analyze_timing.py runs/official runs/fast

One process per machine and nothing else on the GPU. The models are loaded once (not timed). The inputs
run one at a time in the order of the list; the first ``--exclude-first`` inputs (default 3) are run and
recorded but marked as warm-up and excluded from the statistics. For the fast method, one additional
untimed request on the first input precedes the list (``--warmup-requests``, default 1; 0 for the
official method), as in the reference measurement.

End-to-end time of an input: from reading the input image to the closed GLB file, with CUDA synchronisation
before the clock starts and before it stops. The process environment (CPU threads, TF32, allocator,
attention backend) is set explicitly before PyTorch is imported and recorded.

Writes to ``--out-dir``:
  run.json        method, export settings, command line, list sha256, machine and environment records, load time,
                  warm-up request times, start and end times
  timing.jsonl    one line per input: index, id, warm-up flag, e2e_s, per-stage timings, statistics
  sampler.jsonl   machine load every ``--sampler-interval`` seconds (separate low-priority process)
  <id>.glb, <id>.json   output and inference-time camera of every input
  dump/<id>/      with ``--dump-dir``: parity records (fast method)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

STATS_KEYS = ("ss_tokens", "hr_tokens", "resolution", "remesh_divisor", "source_faces", "output_faces",
              "output_vertices", "mesh_worker_s", "tex_stream_s", "cuda_peak_allocated_mib", "glb_bytes",
              "prepared_size", "camera", "load_s_this_request", "mesh_path", "mesh_schedule",
              "mesh_overlap_window_s", "mesh_timings")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("fast", "official"), required=True,
                    help="fast: this package; official: Pixal3D's inference script at its defaults")
    ap.add_argument("--list", required=True, help="JSON lines with 'id' and 'path', in the order to run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None,
                    help="(fast) YAML or JSON configuration: levers, sampling (fastpixal3d.config."
                         "config_from_mapping); --disable and --fa2 apply on top of it")
    ap.add_argument("--exclude-first", type=int, default=3, help="inputs marked as warm-up (run, not counted)")
    ap.add_argument("--warmup-requests", type=int, default=None,
                    help="untimed requests on the first input before the list (default: 1 fast, 0 official)")
    ap.add_argument("--threads", type=int, default=8, help="OMP_NUM_THREADS / MKL_NUM_THREADS")
    ap.add_argument("--tf32", choices=("on", "off"), default=None,
                    help="TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (on) or unset (off); default: on for fast, off for "
                         "official (the reference measurement)")
    ap.add_argument("--autotune-cache", default=None,
                    help="FlexGEMM autotune cache file, updated during the run (default: <out-dir>/autotune_cache.json)")
    ap.add_argument("--students", default=None, help="(fast) directory with students.json")
    ap.add_argument("--student-checkpoints", default=None, help="(fast, development) JSON mapping to training checkpoints")
    ap.add_argument("--disable", action="append", default=[], metavar="LEVER", help="(fast) switch a lever off")
    ap.add_argument("--fa2", action="store_true", help="(fast) FlashAttention-2 instead of FlashAttention-3")
    ap.add_argument("--pixal3d", default=None)
    ap.add_argument("--pixal3d-src", default=None)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--torch-home", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="default: 0 (fast), the script's 42 (official)")
    ap.add_argument("--dump-dir", default=None, help="(fast) parity records per input")
    ap.add_argument("--sampler-interval", type=float, default=30.0, help="seconds; 0 disables the load sampler")
    args = ap.parse_args(argv)
    if args.warmup_requests is None:
        args.warmup_requests = 1 if args.method == "fast" else 0
    if args.tf32 is None:
        args.tf32 = "on" if args.method == "fast" else "off"
    return args


def read_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append((str(row.get("id") or row.get("object_id")), str(Path(row["path"]).resolve())))
    if len({i for i, _ in rows}) != len(rows):
        raise SystemExit("duplicate ids in the input list")
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "timing.jsonl").exists():
        raise SystemExit(f"{out_dir / 'timing.jsonl'} exists; use a new output directory")
    rows = read_list(args.list)
    # FlexGEMM autotune cache: the given file, else <out-dir>/autotune_cache.json (for the official method
    # started from a copy of the Pixal3D checkout's cache file, see fastpixal3d/official.py)
    cache = str(Path(args.autotune_cache).resolve()) if args.autotune_cache else None
    if cache is None and args.method == "fast":
        cache = str(out_dir / "autotune_cache.json")

    from fastpixal3d.runtime.environment import machine_record

    machine_at_start = machine_record()   # before anything is loaded
    tf32 = args.tf32 == "on"
    if args.method == "fast":
        from fastpixal3d.pipeline import configure_process

        configure_process(threads=args.threads, autotune_cache=cache, tf32_override=tf32)
    else:
        from fastpixal3d.official import configure_official_process

        configure_official_process(threads=args.threads, tf32_override=tf32, autotune_cache=cache)

    from fastpixal3d.pipeline import camera_record
    from fastpixal3d.runtime.environment import environment_record, start_sampler
    from fastpixal3d.weights import WeightPaths

    checkpoints = json.loads(Path(args.student_checkpoints).read_text()) if args.student_checkpoints else None
    weights = WeightPaths.from_env(pixal3d_src=args.pixal3d_src, pixal3d=args.pixal3d, students=args.students,
                                   student_checkpoints=checkpoints, hf_home=args.hf_home, torch_home=args.torch_home,
                                   offline=True if args.offline else None)
    config = None
    if args.method == "fast":
        from fastpixal3d.config import config_from_mapping, load_config_file

        spec = load_config_file(args.config) if args.config else {}
        config = config_from_mapping(spec)
        config = config.replace(levers=config.levers.with_disabled(args.disable), threads=args.threads,
                                attention_fallback="fa2" if args.fa2 else config.attention_fallback)
    run = {"method": args.method,
           "export_settings": (dataclasses.asdict(config.export) if config is not None
                               else "the defaults of Pixal3D's inference.py"),
           "argv": sys.argv, "list": str(Path(args.list).resolve()),
           "list_sha256": hashlib.sha256(Path(args.list).read_bytes()).hexdigest(), "n_inputs": len(rows),
           "exclude_first": args.exclude_first, "warmup_requests": args.warmup_requests,
           "machine_at_start": machine_at_start, "started_unix": time.time()}
    if config is not None:
        run["config"] = dataclasses.asdict(config)
        if args.config:
            run["config_file"] = {"path": str(Path(args.config).resolve()),
                                  "sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
    sampler = start_sampler(str(out_dir / "sampler.jsonl"), args.sampler_interval) if args.sampler_interval > 0 else None
    try:
        started = time.perf_counter()
        if args.method == "fast":
            from fastpixal3d.pipeline import FastPixal3D

            model = FastPixal3D.load(config, weights)
            seed = 0 if args.seed is None else args.seed

            def request(stem, path, output, dump=None):
                return model.run(path, output, object_id=stem, seed=seed, work_dir=out_dir, dump_dir=dump)
        else:
            from fastpixal3d.official import OfficialPixal3D

            model = OfficialPixal3D.load(weights, work_dir=out_dir)

            def request(stem, path, output, dump=None):
                return model.run(path, output, object_id=stem, seed=args.seed)
        run["load_s"] = round(time.perf_counter() - started, 3)
        run["load_info"] = model.info
        run["environment"] = environment_record()
        print(f"[time_e2e] {args.method}: loaded in {run['load_s']:.1f} s", flush=True)

        warm = []
        first_stem, first_path = rows[0]
        for i in range(args.warmup_requests):
            res = request(first_stem, first_path, out_dir / f"warmup_{i}.glb")
            warm.append(res["timings"]["e2e_s"])
            print(f"[time_e2e] warm-up request {i}: {warm[-1]:.3f} s", flush=True)
        run["warmup_request_e2e_s"] = warm
        (out_dir / "run.json").write_text(json.dumps(run, indent=1, default=str))

        failed = 0
        with open(out_dir / "timing.jsonl", "w") as log:
            for index, (stem, path) in enumerate(rows, start=1):
                output = out_dir / f"{stem}.glb"
                dump = Path(args.dump_dir) / stem if (args.dump_dir and args.method == "fast") else None
                record = {"index": index, "id": stem, "image": path, "warmup": index <= args.exclude_first}
                try:
                    res = request(stem, path, output, dump)
                    output.with_suffix(".json").write_text(json.dumps(camera_record(res, stem, path), indent=1))
                    record.update(status="ok", e2e_s=res["timings"]["e2e_s"], timings=res["timings"],
                                  stats={k: res["stats"].get(k) for k in STATS_KEYS if k in res["stats"]})
                except Exception as error:  # recorded; the run continues with the next input
                    failed += 1
                    record.update(status="error", error=f"{type(error).__name__}: {error}",
                                  traceback=traceback.format_exc()[-3000:])
                record["finished_unix"] = time.time()
                log.write(json.dumps(record, default=str) + "\n")
                log.flush()
                print(f"[time_e2e] {index}/{len(rows)} {stem}: "
                      f"{record.get('e2e_s', record.get('error'))}", flush=True)
        run.update(finished_unix=time.time(), failed=failed, machine_at_end=machine_record())
        (out_dir / "run.json").write_text(json.dumps(run, indent=1, default=str))
        return 0 if failed == 0 else 1
    finally:
        if sampler is not None:
            sampler.terminate()


if __name__ == "__main__":
    sys.exit(main())
