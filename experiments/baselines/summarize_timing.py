#!/usr/bin/env python3
r"""Rows of the report's table of the released pipelines (end-to-end time split into sampling and post-processing).

    python experiments/baselines/summarize_timing.py --run TRELLIS.2=runs/trellis2 \
        --run Hunyuan3D-2.1=runs/hunyuan3d21 --run TripoSG=runs/triposg --exclude-first 3 \
        --out results/baselines_timing.json --markdown results/baselines_timing.md

Each ``--run`` is the output directory of one adapter run on the timing inputs (eval/lists/timing_80.csv through
make_input_list.py, in that order). The statistics cover the inputs recorded with status ok after the first
``--exclude-first`` of the list (the report: 3, which leaves the 77 measured inputs of the dedicated timing run), and
only the inputs present in every run, so that every row is measured on the same inputs.

Per run: median, mean and 90th percentile of the end-to-end time (``--e2e-key``: ``stage_sum_s``, the definition
used for these rows of the report, or ``e2e_s``), and for each part of baseline_io.parts() the median seconds and
the mean over inputs of its share of the input's end-to-end time: sampling (image conditioning and flow or
diffusion sampling), post-processing (= latent decoding and iso-surface extraction + mesh processing and export;
both also reported separately) and other (input preparation and the remainder).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

PARTS = ("sampling", "decoding", "meshwork", "other")


def pct(values: List[float], q: float) -> float:
    v = sorted(values)
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def load(directory: Path, exclude_first: int) -> Dict[str, Dict[str, Any]]:
    path = directory / "timing.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["id"]: r for r in rows if r.get("status") == "ok" and int(r["index"]) > exclude_first}


def summarize(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    e2e = [float(r["e2e_s"] if key == "e2e_s" else r["timings"][key]) for r in rows]
    out: Dict[str, Any] = {"n": len(rows), "median": st.median(e2e), "mean": st.fmean(e2e), "p90": pct(e2e, 0.9)}
    values = {p: [float(r["timings"][f"part_{p}_s"]) for r in rows] for p in PARTS}
    values["post_processing"] = [a + b for a, b in zip(values["decoding"], values["meshwork"])]
    for name, v in values.items():
        out[f"{name}_median_s"] = st.median(v)
        out[f"{name}_share_mean"] = st.fmean(x / t for x, t in zip(v, e2e))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--exclude-first", type=int, default=3)
    ap.add_argument("--e2e-key", default="stage_sum_s", choices=("stage_sum_s", "e2e_s"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)
    runs = {}
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run expects LABEL=DIR, got {item!r}")
        label, path = item.split("=", 1)
        runs[label] = load(Path(path), args.exclude_first)
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        raise SystemExit("no input is recorded in every run")
    for label, rows in runs.items():
        missing = [r for r in rows if "part_sampling_s" not in rows[r]["timings"]]
        if missing:
            raise SystemExit(f"{label}: no part_* times (not written by an adapter of experiments/baselines)")
    report = {"e2e_key": args.e2e_key, "exclude_first": args.exclude_first, "n_common": len(common),
              "rows": {label: summarize([rows[i] for i in sorted(common)], args.e2e_key) for label, rows in runs.items()}}
    lines = ["| method | n | median (s) | mean (s) | p90 (s) | sampling, median s (mean share) | "
             "post-processing, median s (mean share) |", "|---|---|---|---|---|---|---|"]
    for label, s in report["rows"].items():
        lines.append(f"| {label} | {s['n']} | {s['median']:.1f} | {s['mean']:.1f} | {s['p90']:.1f} | "
                     f"{s['sampling_median_s']:.1f} ({100 * s['sampling_share_mean']:.0f}%) | "
                     f"{s['post_processing_median_s']:.1f} ({100 * s['post_processing_share_mean']:.0f}%) |")
    table = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    if args.markdown:
        args.markdown.write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
