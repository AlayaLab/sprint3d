#!/usr/bin/env python3
"""Stage-delta table: how each critical-path stage changes between two configurations.

    python experiments/analysis/stage_deltas.py experiments/configs/cumulative --runs runs/cumulative \
        --out results/cumulative [--unit object] [--pair pixal3d_rerun:gpu_chart_segmentation ...]

For every pair of configurations and every critical-path stage (experiments/analysis/analysis_common.py
``attribute``): the paired difference TO - FROM of that stage's seconds, mean over the inputs measured in
both with a 95 % bootstrap interval (``--unit input``: inputs resampled, the default; ``object``: the
inputs of one object resampled together, as the report's Appendix A), and the median difference; the
end-to-end difference is listed as stage ``all``. Pairs (``--pair FROM:TO`` by name or file stem, or by
default):

* ``adjacent``: every pair of adjacent rows of the chain;
* ``block``: every block of the chain, from the row before its first row to its last row (the report's
  end-to-end anchors: Pixal3D re-run -> mesh path rebuilt -> + flow optimizations -> Ours);
* ``total``: the first row of the chain -> the last;
* ``control``: every control with its row, ordered ``without -> with`` the levers in which they differ.

Writes ``stage_deltas.csv`` and ``stage_deltas.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402

COLUMNS = ["from", "to", "kind", "label", "stage", "stage_label", "n", "delta_mean", "ci_lo", "ci_hi", "delta_median",
           "from_median", "to_median", "unit"]


def default_pairs(table: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
    """``(from, to, kind, label)`` of the default pairs (see the module docstring)."""
    chain, _ = C.split_chain(table)
    names = [C.name_of(e["config"]) for e in chain]
    pairs = [(a, b, "adjacent", "") for a, b in zip(names[:-1], names[1:])]
    for block, label, a, b in C.block_spans(chain):
        pairs.append((names[a], names[b], "block", label))
    if len(names) > 1:
        pairs.append((names[0], names[-1], "total", "whole chain"))
    for pair in C.control_pairs(table):
        pairs.append((C.name_of(pair["without"]["config"]), C.name_of(pair["with"]["config"]), "control",
                      "+".join(pair["levers"])))
    return pairs


def build(table: List[Dict[str, Any]], pairs: List[Tuple[str, str, str, str]], unit: str = "input") -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for entry in table:
        by_name[entry["config"]["stem"]] = entry
        by_name.setdefault(entry["config"].get("name", entry["config"]["stem"]), entry)
    out = []
    for a, b, kind, label in pairs:
        if a not in by_name or b not in by_name:
            raise SystemExit(f"--pair {a}:{b}: unknown configuration")
        ea, eb = by_name[a], by_name[b]
        if not ea["measured"] or not eb["measured"]:
            continue
        base = {"from": a, "to": b, "kind": kind, "label": label, "unit": unit}
        d = C.paired(ea["measured"], eb["measured"], unit=unit)
        if not d["n"]:
            continue
        out.append({**base, "stage": "all", "stage_label": "end to end", "n": d["n"], "delta_mean": d["delta_mean"],
                    "ci_lo": (d.get("ci95_mean") or [None, None])[0], "ci_hi": (d.get("ci95_mean") or [None, None])[1],
                    "delta_median": d["delta_median"], "from_median": d["median_prev"], "to_median": d["median_cur"]})
        attr_a, attr_b = ea["attribution"], eb["attribution"]
        for stage in C.STAGES:
            s = C.paired(attr_a, attr_b, value=lambda cp, stage=stage: float(cp[stage]), unit=unit)
            if not s["n"] or (s["median_prev"] == 0 and s["median_cur"] == 0 and s["delta_mean"] == 0):
                continue
            out.append({**base, "stage": stage, "stage_label": C.STAGE_LABELS[stage], "n": s["n"],
                        "delta_mean": s["delta_mean"], "ci_lo": (s.get("ci95_mean") or [None, None])[0],
                        "ci_hi": (s.get("ci95_mean") or [None, None])[1], "delta_median": s["delta_median"],
                        "from_median": s["median_prev"], "to_median": s["median_cur"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pair", action="append", default=[], metavar="FROM:TO")
    ap.add_argument("--exclude-first", type=int, default=0,
                    help="inputs at the start of the list left out of the statistics (default: none, as the report)")
    ap.add_argument("--unit", choices=("input", "object"), default="input", help="bootstrap resampling unit")
    args = ap.parse_args(argv)
    table = C.load_table(args.configs, args.runs, args.exclude_first)
    pairs = [(*p.split(":", 1), "requested", "") for p in args.pair] if args.pair else default_pairs(table)
    rows = build(table, pairs, args.unit)
    args.out.mkdir(parents=True, exist_ok=True)
    C.write_csv(args.out / "stage_deltas.csv", rows, COLUMNS)
    (args.out / "stage_deltas.json").write_text(json.dumps(
        {"bootstrap": {"resamples": C.N_BOOT, "seed": C.SEED, "unit": args.unit}, "rows": rows}, indent=1, default=str))
    for r in rows:
        if r["stage"] == "all":
            print(f"{r['from']} -> {r['to']} ({r['kind']}{', ' + r['label'] if r['label'] else ''}): {r['delta_mean']:+.3f} s "
                  f"{C.fmt_ci([r['ci_lo'], r['ci_hi']]) if r['ci_lo'] is not None else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
