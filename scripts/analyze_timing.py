#!/usr/bin/env python3
"""Summarise timing runs written by scripts/time_e2e.py (protocol: docs/TIMING.md).

    python scripts/analyze_timing.py RUN_DIR [RUN_DIR ...] [--labels A,B] [--exclude-first 3] \
        [--out summary.json] [--csv per_input.csv]

For every run: end-to-end statistics over the measured inputs (the first ``--exclude-first`` inputs of
the list are warm-up and excluded; the same statistics over all inputs are reported too): median, mean,
standard deviation, 10th / 25th / 75th / 90th percentiles (linear interpolation), interquartile range,
minimum, maximum, and 95 % bootstrap confidence intervals of the median and of the mean over inputs
(10,000 resamples of the inputs with replacement, fixed seed 0, percentile interval). Per stage: median,
mean and the bootstrap interval of the mean. For the fast method the stages are also grouped along the
critical path (the texture branch and the mesh-preparation branch overlap; the longer one is counted).

With two or more runs, the first one is the reference and each other run is compared with it on the
inputs present in both: per-input ratio reference / run (median of ratios, geometric mean), ratio of
medians, ratio of means, with bootstrap intervals over inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

N_BOOT = 10_000
SEED = 0

# Critical-path groups of the fast method (stage keys of fastpixal3d.pipeline).
GROUPS = (
    ("image_and_camera", ("preprocess_s", "camera_s")),
    ("structure", ("ss_condition_s", "ss_initial_noise_s", "ss_rollout_s", "ss_decode_support_s")),
    ("coarse_shape", ("lr_condition_s", "lr_initial_noise_s", "lr_rollout_s")),
    ("fine_shape", ("hr_upsample_support_s", "hr_condition_s", "hr_initial_noise_s", "hr_rollout_s")),
    ("shape_decode", ("shape_decode_s", "shape_fill_holes_s")),
    ("bake", ("glb_finish_s", "glb_bake_finalize_s")),
    ("export", ("export_s",)),
)


# ----------------------------------------------------------------------------------------------- statistics
def pct(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def boot_ci(items: Sequence[Any], fn: Callable[[List[Any]], float], n: int = N_BOOT, seed: int = SEED):
    """95 % percentile interval of ``fn`` over ``n`` resamples (with replacement) of ``items``."""
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    k = len(items)
    values = sorted(fn([items[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return [values[int(0.025 * n)], values[int(0.975 * n) - 1]]


def describe(values: Sequence[float], ci: bool = True) -> Dict[str, Any]:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0}
    out = {"n": len(v), "median": st.median(v), "mean": st.mean(v), "stdev": st.stdev(v) if len(v) > 1 else 0.0,
           "p10": pct(v, 0.10), "p25": pct(v, 0.25), "p75": pct(v, 0.75), "p90": pct(v, 0.90),
           "iqr": pct(v, 0.75) - pct(v, 0.25), "min": min(v), "max": max(v)}
    if ci:
        out["ci95_median"] = boot_ci(v, st.median)
        out["ci95_mean"] = boot_ci(v, st.mean)
    return out


def paired(ref: Dict[str, float], other: Dict[str, float], ids: Sequence[str]) -> Dict[str, Any]:
    pairs = [(ref[i], other[i]) for i in ids if ref.get(i) is not None and other.get(i) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    ratios = [a / b for a, b in pairs]
    rom = lambda s: st.median([a for a, _ in s]) / st.median([b for _, b in s])  # noqa: E731
    mor = lambda s: st.median([a / b for a, b in s])  # noqa: E731
    rmean = lambda s: st.mean([a for a, _ in s]) / st.mean([b for _, b in s])  # noqa: E731
    return {"n": len(pairs), "median_of_ratios": mor(pairs), "ratio_of_medians": rom(pairs),
            "ratio_of_means": rmean(pairs), "geomean_ratio": math.exp(st.mean([math.log(r) for r in ratios])),
            "ratio_p10": pct(ratios, 0.1), "ratio_p90": pct(ratios, 0.9),
            "ci95_median_of_ratios": boot_ci(pairs, mor), "ci95_ratio_of_medians": boot_ci(pairs, rom),
            "ci95_ratio_of_means": boot_ci(pairs, rmean)}


# ----------------------------------------------------------------------------------------------- loading
def load_run(directory: Path) -> Dict[str, Any]:
    meta = json.loads((directory / "run.json").read_text()) if (directory / "run.json").is_file() else {}
    rows = [json.loads(line) for line in (directory / "timing.jsonl").read_text().splitlines() if line.strip()]
    return {"dir": str(directory), "meta": meta, "rows": rows}


def critical_path(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Group the stage times of one fast-method input along the critical path."""
    t, s = row.get("timings") or {}, row.get("stats") or {}
    if "to_glb_s" not in t or "e2e_s" not in t:
        return None
    out = {name: sum(float(t.get(k, 0.0)) for k in keys) for name, keys in GROUPS}
    window = float(t["to_glb_s"]) - out["bake"]
    texture, mesh = s.get("tex_stream_s"), s.get("mesh_worker_s")
    out["texture_or_mesh_window"] = window
    if texture is not None and mesh is not None:
        out["critical_branch_is_texture"] = 1.0 if texture >= mesh else 0.0
        out["hidden_branch"] = min(texture, mesh)
    out["other"] = float(t["e2e_s"]) - sum(out[name] for name, _ in GROUPS) - window
    return out


def summarise_run(run: Dict[str, Any], exclude_first: int) -> Dict[str, Any]:
    ok = [r for r in run["rows"] if r.get("status") == "ok"]
    measured = [r for r in ok if int(r["index"]) > exclude_first]
    summary = {"dir": run["dir"], "method": run["meta"].get("method"),
               "export_settings": run["meta"].get("export_settings"),
               "n_inputs": len(run["rows"]), "n_ok": len(ok), "n_failed": len(run["rows"]) - len(ok),
               "excluded_warmup_ids": [r["id"] for r in run["rows"] if int(r["index"]) <= exclude_first],
               "load_s": run["meta"].get("load_s"), "warmup_request_e2e_s": run["meta"].get("warmup_request_e2e_s"),
               "environment": (run["meta"].get("environment") or {}).get("env"),
               "e2e_s": describe([r["e2e_s"] for r in measured]),
               "e2e_s_all_inputs": describe([r["e2e_s"] for r in ok], ci=False)}
    keys = sorted({k for r in measured for k in (r.get("timings") or {}) if k.endswith("_s")})
    summary["stages"] = {}
    for key in keys:
        values = [float(r["timings"][key]) for r in measured if key in (r.get("timings") or {})]
        d = describe(values, ci=False)
        d["ci95_mean"] = boot_ci(values, st.mean)
        summary["stages"][key] = {k: d.get(k) for k in ("n", "median", "mean", "ci95_mean", "p90")}
    groups = [g for g in (critical_path(r) for r in measured) if g is not None]
    if groups:
        summary["critical_path"] = {name: {"median": st.median([g[name] for g in groups if name in g]),
                                           "mean": st.mean([g[name] for g in groups if name in g])}
                                    for name in groups[0]}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="output directories of scripts/time_e2e.py")
    ap.add_argument("--labels", default=None, help="comma-separated names of the runs (default: directory names)")
    ap.add_argument("--exclude-first", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="summary JSON")
    ap.add_argument("--csv", type=Path, default=None, help="per-input end-to-end times")
    args = ap.parse_args(argv)
    labels = args.labels.split(",") if args.labels else [p.name for p in args.runs]
    if len(labels) != len(args.runs):
        ap.error("--labels must name every run")
    runs = [load_run(p) for p in args.runs]
    report = {"exclude_first": args.exclude_first, "bootstrap": {"resamples": N_BOOT, "seed": SEED},
              "runs": {}, "paired": {}}
    e2e = {}
    for label, run in zip(labels, runs):
        report["runs"][label] = summarise_run(run, args.exclude_first)
        e2e[label] = {r["id"]: r["e2e_s"] for r in run["rows"] if r.get("status") == "ok"}
    order = [r["id"] for r in runs[0]["rows"]]
    measured_ids = [r["id"] for r in runs[0]["rows"] if int(r["index"]) > args.exclude_first]
    for label in labels[1:]:
        report["paired"][f"{labels[0]} / {label}"] = paired(e2e[labels[0]], e2e[label], measured_ids)
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        args.out.write_text(text)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "id", "warmup"] + [f"{label}_e2e_s" for label in labels])
            for index, item in enumerate(order, start=1):
                writer.writerow([index, item, int(index <= args.exclude_first)] + [e2e[l].get(item) for l in labels])
    for label in labels:
        d = report["runs"][label]["e2e_s"]
        if d.get("n"):
            print(f"{label}: n={d['n']} median {d['median']:.3f} s (95% CI {d['ci95_median'][0]:.3f}-"
                  f"{d['ci95_median'][1]:.3f}) mean {d['mean']:.3f} s" if d.get("ci95_median") else
                  f"{label}: n={d['n']} median {d['median']:.3f} s mean {d['mean']:.3f} s")
    for name, p in report["paired"].items():
        if p.get("n", 0) >= 3:
            print(f"{name}: ratio of medians {p['ratio_of_medians']:.3f} (95% CI {p['ci95_ratio_of_medians'][0]:.3f}-"
                  f"{p['ci95_ratio_of_medians'][1]:.3f}), median of ratios {p['median_of_ratios']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
