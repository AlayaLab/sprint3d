#!/usr/bin/env python3
"""Paired table against a reference row: the one-lever-off ablations and the same-step teachers.

    python experiments/analysis/paired_table.py experiments/configs/ablations --runs runs/ablations \
        --out results/ablations
    python experiments/analysis/paired_table.py experiments/configs/same_step_teacher --runs runs/same_step \
        --out results/same_step_teacher

The first configuration (file order) is the reference (Ours). For every other row: the end-to-end median,
and, paired with the reference on the inputs measured in both, the median of the per-input differences of
the end-to-end time and of the texture sampling loop (``tex_rollout_s``) and of the stage times of the
structure, coarse-shape and fine-shape flows (``*_rollout_s``), each with a 95 % bootstrap interval in
which the inputs of one object (ids ``<object>_v<view>``) are resampled together. The quality metrics of
these rows come from the evaluation suite (eval/), run on the GLBs of the same run directories.
Writes ``paired.csv`` and ``paired.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402

KEYS = (("e2e", "e2e_s"), ("texture_sampling", "tex_rollout_s"), ("structure_sampling", "ss_rollout_s"),
        ("coarse_shape_sampling", "lr_rollout_s"), ("fine_shape_sampling", "hr_rollout_s"))
COLUMNS = (["row", "name", "paper_row", "reference", "n", "n_objects", "e2e_median"]
           + [f"{k}_{x}" for k, _ in KEYS for x in ("delta_median", "ci_lo", "ci_hi")])


def build(table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ref = table[0]
    out = []
    for k, entry in enumerate(table):
        cfg, meas = entry["config"], entry["measured"]
        row: Dict[str, Any] = {"row": k + 1, "name": cfg.get("name", cfg["stem"]), "paper_row": cfg.get("paper_row"),
                               "reference": ref["config"].get("name", ref["config"]["stem"])}
        if meas:
            row["e2e_median"] = st.median(float(r["e2e_s"]) for r in meas.values())
        if k > 0 and meas and ref["measured"]:
            for label, key in KEYS:
                def value(r, key=key):
                    return float(r["e2e_s"]) if key == "e2e_s" else float((r.get("timings") or {}).get(key) or 0.0)
                d = C.paired(ref["measured"], meas, value=value, unit="object")
                row.update(n=d["n"], n_objects=d["n_objects"])
                row[f"{label}_delta_median"] = d["delta_median"]
                if d.get("ci95_median"):
                    row[f"{label}_ci_lo"], row[f"{label}_ci_hi"] = d["ci95_median"]
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude-first", type=int, default=0,
                    help="inputs at the start of the list left out of the statistics (default: none, as the report)")
    args = ap.parse_args(argv)
    rows = build(C.load_table(args.configs, args.runs, args.exclude_first))
    args.out.mkdir(parents=True, exist_ok=True)
    C.write_csv(args.out / "paired.csv", rows, COLUMNS)
    (args.out / "paired.json").write_text(json.dumps(
        {"bootstrap": {"resamples": C.N_BOOT, "seed": C.SEED, "unit": "object", "statistic": "median"},
         "rows": rows}, indent=1, default=str))
    for r in rows:
        if r.get("e2e_delta_median") is not None:
            print(f"{r['name']:<34} median {r['e2e_median']:.3f} s, delta {r['e2e_delta_median']:+.3f} s "
                  f"{C.fmt_ci([r.get('e2e_ci_lo'), r.get('e2e_ci_hi')]) if r.get('e2e_ci_lo') is not None else ''}")
        elif r.get("e2e_median") is not None:
            print(f"{r['name']:<34} median {r['e2e_median']:.3f} s (reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
