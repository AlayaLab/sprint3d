#!/usr/bin/env python3
"""Lever table: what each step of the cumulative measurement adds, its output class and the time it saves.

    python experiments/analysis/lever_table.py experiments/configs/cumulative --runs runs/cumulative \
        --out results/cumulative

One line (``kind = step``) per row of the chain after the first, in the order of the measurement, with its
block and the levers it adds (``adds``; the tiled mesh path of row 01 is not a lever and is listed by its
row name), and one line (``kind = control``) per control: the levers in which the control and its row
differ. The effect is the paired difference of the end-to-end time (mean of the per-input differences,
95 % bootstrap interval over inputs, and the median difference) of a step to the previous row, and of a
control pair ``with - without`` (the row with the levers minus the row without them); ``saved_s`` is minus
the mean. ``stage_delta_s`` is the paired difference of the time of the stage the levers act on (e.g. the
texture sampling loop for the texture-flow levers, both shape sampling loops for the shape students), which
can show an effect that a concurrent, longer branch hides from the end-to-end time. Output class, stage and
the report's name of each lever come from :mod:`fastpixal3d.levers`. Writes ``levers.csv`` and
``levers.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import analysis_common as C  # noqa: E402
from fastpixal3d.levers import LEVERS_BY_NAME  # noqa: E402

# Timing key(s) of the stage a lever acts on: by lever where the stage name is not specific enough, else by
# the lever's stage. A step whose levers include one without a key (mesh, UV) gets no stage difference.
LEVER_KEYS = {
    "coarse_shape_student": ("lr_rollout_s",),
    "fine_shape_student": ("hr_rollout_s",),
    "fold_lora_fine_shape": ("hr_rollout_s",),
    "fa3_fine_shape_flow": ("hr_rollout_s",),
}
STAGE_KEYS = {
    "texture": ("tex_rollout_s",),
    "shape": ("hr_rollout_s",),
    "structure": ("ss_rollout_s",),
    "conditioning": ("ss_condition_s", "lr_condition_s", "hr_condition_s", "tex_condition_s"),
    "bake": ("glb_bake_finalize_s",),
    "export": ("export_s",),
}
COLUMNS = ["kind", "row", "name", "paper_row", "block", "from", "to", "levers", "report_name", "output_class",
           "stage", "n", "delta_mean", "delta_ci_lo", "delta_ci_hi", "delta_median", "saved_s", "stage_key",
           "stage_delta_mean", "stage_delta_ci_lo", "stage_delta_ci_hi"]


def stage_time(keys):
    def value(row: Mapping[str, Any]) -> float:
        t = row.get("timings") or {}
        return sum(float(t.get(k) or 0.0) for k in keys)
    return value


def stage_keys(names: Sequence[str]):
    keys: List[str] = []
    for name in names:
        lv = LEVERS_BY_NAME[name]
        found = LEVER_KEYS.get(name) or STAGE_KEYS.get(lv.stage)
        if not found:
            return ()
        keys += [k for k in found if k not in keys]
    return tuple(keys)


def effect(row: Dict[str, Any], prev: Mapping[str, Any], cur: Mapping[str, Any], names: Sequence[str]) -> None:
    if not prev or not cur:
        return
    d = C.paired(prev, cur)
    if not d["n"]:
        return
    row.update(n=d["n"], delta_mean=d["delta_mean"], delta_median=d["delta_median"], saved_s=-d["delta_mean"])
    if d.get("ci95_mean"):
        row.update(delta_ci_lo=d["ci95_mean"][0], delta_ci_hi=d["ci95_mean"][1])
    keys = stage_keys(names) if names else ()
    if keys:
        s = C.paired(prev, cur, value=stage_time(keys))
        row.update(stage_key="+".join(keys), stage_delta_mean=s["delta_mean"])
        if s.get("ci95_mean"):
            row.update(stage_delta_ci_lo=s["ci95_mean"][0], stage_delta_ci_hi=s["ci95_mean"][1])


def describe(names: Sequence[str], fallback: str) -> Dict[str, str]:
    levers = [LEVERS_BY_NAME[a] for a in names]
    if not levers:
        return {"levers": "", "report_name": fallback, "output_class": "", "stage": ""}
    return {"levers": "+".join(names), "report_name": "; ".join(lv.paper_name for lv in levers),
            "output_class": "; ".join(sorted({lv.output for lv in levers})),
            "stage": "; ".join(sorted({lv.stage for lv in levers}))}


def build(table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chain, _ = C.split_chain(table)
    out = []
    for k, entry in enumerate(chain):
        if k == 0:
            continue
        cfg, prev_cfg = entry["config"], chain[k - 1]["config"]
        adds = list(cfg.get("adds") or [])
        paper = (cfg.get("paper_row") or C.name_of(cfg)).lstrip("+ ").strip()
        row: Dict[str, Any] = {"kind": "step", "row": f"{k:02d}", "name": C.name_of(cfg),
                               "paper_row": cfg.get("paper_row"), "block": cfg.get("block") or "",
                               "from": C.name_of(prev_cfg), "to": C.name_of(cfg), **describe(adds, paper)}
        effect(row, chain[k - 1]["measured"], entry["measured"], adds)
        out.append(row)
    for pair in C.control_pairs(table):
        cfg = pair["control"]["config"]
        row = {"kind": "control", "row": cfg["stem"].split("_", 1)[0], "name": C.name_of(cfg),
               "paper_row": cfg.get("paper_row"), "block": "", "from": C.name_of(pair["without"]["config"]),
               "to": C.name_of(pair["with"]["config"]), **describe(pair["levers"], "")}
        effect(row, pair["without"]["measured"], pair["with"]["measured"], pair["levers"])
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
    C.write_csv(args.out / "levers.csv", rows, COLUMNS)
    (args.out / "levers.json").write_text(json.dumps(
        {"bootstrap": {"resamples": C.N_BOOT, "seed": C.SEED, "unit": "input"}, "rows": rows}, indent=1, default=str))
    for r in rows:
        eff = (f"{r['delta_mean']:+.3f} s {C.fmt_ci([r.get('delta_ci_lo'), r.get('delta_ci_hi')])}"
               if r.get("delta_mean") is not None and r.get("delta_ci_lo") is not None else
               (f"{r['delta_mean']:+.3f} s" if r.get("delta_mean") is not None else "(no runs)"))
        label = r["levers"] or r["report_name"]
        print(f"{r['row']:>3} {r['kind']:<7} {label:<60} {r['output_class']:<16} {eff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
