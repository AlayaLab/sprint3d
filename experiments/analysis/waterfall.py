#!/usr/bin/env python3
"""Waterfall of the cumulative measurement: per-row end-to-end medians, deltas, block factors and a figure.

    python experiments/analysis/waterfall.py experiments/configs/cumulative --runs runs/cumulative \
        --out results/cumulative [--figure]

Rows are the chain of the configuration files in file order (the order of the measurement); controls
(``control_of``) are not rows of the waterfall (experiments/analysis/lever_table.py and stage_deltas.py
compare them with their row). The run of a row is ``<runs>/<file stem>`` (experiments/run_arms.py). For
every row (``row`` = its position in the chain, the number of its file):

* block, n, failed inputs, end-to-end median with its 95 % bootstrap interval (inputs resampled), mean, p90;
* the median of every critical-path stage (experiments/analysis/analysis_common.py ``attribute``) and the median
  face count of the GLB;
* the paired difference to the previous row and to the first row: mean and median of the per-input
  differences, each with a bootstrap interval over inputs.

For every block (rows 01-08 mesh post-processing, 09-13 flow engineering, 14-18 distillation: the
``block`` key) and for the whole chain: the speed-up factor ``median(from) / median(to)`` from the row
before the block to its last row and the paired mean difference, with 95 % bootstrap intervals in which
the inputs of one object are resampled together (``--block-unit``; the report's choice).

Writes ``waterfall.csv`` (rows), ``waterfall_blocks.csv`` (blocks and the whole chain) and
``waterfall.json`` (both, plus the medians of the controls); with ``--figure`` also ``waterfall.png`` and
``waterfall.pdf`` (one bar per row: stage medians stacked, the grey rest up to the end-to-end median; the
blocks separated, each labelled with its factor). Rows without a run are kept in the table (n = 0) and
differences across them are not computed.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402

COLUMNS = (["row", "name", "paper_row", "block", "method", "adds", "n", "n_failed", "e2e_median", "e2e_median_ci_lo",
            "e2e_median_ci_hi", "e2e_mean", "e2e_p90", "faces_median"]
           + [f"{s}_median" for s in C.STAGES]
           + ["delta_prev_mean", "delta_prev_ci_lo", "delta_prev_ci_hi", "delta_prev_median", "delta_prev_n",
              "delta_first_mean", "delta_first_ci_lo", "delta_first_ci_hi", "delta_first_median"])
BLOCK_COLUMNS = ["kind", "block", "label", "from", "to", "n", "n_objects", "unit", "from_median", "to_median",
                 "factor", "factor_ci_lo", "factor_ci_hi", "delta_mean", "delta_ci_lo", "delta_ci_hi"]


def build(chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    first = chain[0]["measured"] if chain else {}
    for k, entry in enumerate(chain):
        cfg, meas = entry["config"], entry["measured"]
        e2e = [float(r["e2e_s"]) for r in meas.values()]
        row: Dict[str, Any] = {"row": k, "name": C.name_of(cfg), "paper_row": cfg.get("paper_row"),
                               "block": cfg.get("block") or "", "method": cfg.get("method", "fast"),
                               "adds": "+".join(cfg.get("adds") or []), "n": len(e2e), "n_failed": entry["n_failed"]}
        if e2e:
            ci = C.boot_ci(e2e, np.median)
            faces = [r["stats"].get("output_faces") for r in meas.values() if (r.get("stats") or {}).get("output_faces")]
            row.update(e2e_median=st.median(e2e), e2e_mean=st.mean(e2e), e2e_p90=C.pct(e2e, 0.9),
                       e2e_median_ci_lo=ci[0] if ci else None, e2e_median_ci_hi=ci[1] if ci else None,
                       faces_median=st.median(faces) if faces else None)
            for s in C.STAGES:
                row[f"{s}_median"] = st.median(a[s] for a in entry["attribution"].values())
        if k > 0 and chain[k - 1]["measured"] and meas:
            d = C.paired(chain[k - 1]["measured"], meas)
            row.update(delta_prev_mean=d["delta_mean"], delta_prev_median=d["delta_median"], delta_prev_n=d["n"])
            if d.get("ci95_mean"):
                row.update(delta_prev_ci_lo=d["ci95_mean"][0], delta_prev_ci_hi=d["ci95_mean"][1])
        if k > 0 and first and meas:
            d = C.paired(first, meas)
            row.update(delta_first_mean=d["delta_mean"], delta_first_median=d["delta_median"])
            if d.get("ci95_mean"):
                row.update(delta_first_ci_lo=d["ci95_mean"][0], delta_first_ci_hi=d["ci95_mean"][1])
        out.append(row)
    return out


def build_blocks(chain: List[Dict[str, Any]], unit: str = "object") -> List[Dict[str, Any]]:
    """Speed-up factor and paired mean difference of every block and of the whole chain."""
    spans = [("block", block, label, a, b) for block, label, a, b in C.block_spans(chain)]
    if len(chain) > 1:
        spans.append(("total", "", "whole chain", 0, len(chain) - 1))
    out = []
    for kind, block, label, a, b in spans:
        prev, cur = chain[a]["measured"], chain[b]["measured"]
        rec: Dict[str, Any] = {"kind": kind, "block": block, "label": label, "from": C.name_of(chain[a]["config"]),
                               "to": C.name_of(chain[b]["config"]), "unit": unit, "n": 0}
        if prev and cur:
            f = C.paired_ratio(prev, cur, unit=unit)
            d = C.paired(prev, cur, unit=unit)
            if f["n"]:
                rec.update(n=f["n"], n_objects=f["n_objects"], from_median=f["median_prev"], to_median=f["median_cur"],
                           factor=f["factor"], factor_ci_lo=f["ci95"][0], factor_ci_hi=f["ci95"][1],
                           delta_mean=d["delta_mean"])
                if d.get("ci95_mean"):
                    rec.update(delta_ci_lo=d["ci95_mean"][0], delta_ci_hi=d["ci95_mean"][1])
        out.append(rec)
    return out


def control_medians(controls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for entry in controls:
        cfg, meas = entry["config"], entry["measured"]
        e2e = [float(r["e2e_s"]) for r in meas.values()]
        out.append({"stem": cfg["stem"], "name": C.name_of(cfg), "paper_row": cfg.get("paper_row"),
                    "control_of": cfg["control_of"], "n": len(e2e), "n_failed": entry["n_failed"],
                    "e2e_median": st.median(e2e) if e2e else None, "e2e_mean": st.mean(e2e) if e2e else None})
    return out


# Figure groups: (label, stages). Colours: the first six slots of a validated categorical palette,
# in fixed order; grey (not a series colour) for the rest.
GROUPS = (("structure and shape flows", ("structure", "coarse_shape", "fine_shape")),
          ("texture flow", ("texture",)),
          ("mesh preparation", ("mesh",)),
          ("UV atlas", ("uv",)),
          ("bake and export", ("bake", "export")),
          ("mesh to GLB, not split", ("unsplit",)))
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
REST = "#c9c8c3"
INK, MUTED = "#0b0b0b", "#52514e"


def figure(rows: List[Dict[str, Any]], out: Path, xmax: float = None, blocks: List[Dict[str, Any]] = ()) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("e2e_median") is not None]
    if not rows:
        return
    height = 1.1 + 0.32 * len(rows)
    fig, ax = plt.subplots(figsize=(8.0, height))
    labels, seen = [], set()

    def bar(y, width, left, colour, label):
        ax.barh(y, width, left=left, height=0.62, color=colour, edgecolor="white", linewidth=1.0)
        seen.add(label)

    for y, r in enumerate(rows):
        left = 0.0
        for (label, stages), colour in zip(GROUPS, COLOURS):
            width = sum(float(r.get(f"{s}_median") or 0.0) for s in stages)
            if width > 0:
                bar(y, width, left, colour, label)
                left += width
        rest = float(r["e2e_median"]) - left
        if rest > 0:
            bar(y, rest, left, REST, "image, camera and other")
        text = f"{r['e2e_median']:.1f} s" if r["e2e_median"] >= 10 else f"{r['e2e_median']:.2f} s"
        ax.text(max(float(r["e2e_median"]), left) + 0.6, y, text, va="center", ha="left", fontsize=7.5, color=INK)
        labels.append(r.get("paper_row") or r["name"])
    # blocks: a thin line above the first row of each block and its label with the factor, right-aligned
    factors = {b["block"]: b for b in blocks if b["kind"] == "block"}
    top = max(float(r["e2e_median"]) for r in rows)
    right = xmax or top * 1.12
    for y, r in enumerate(rows):
        block = r.get("block")
        if block and (y == 0 or rows[y - 1].get("block") != block):
            ax.axhline(y - 0.5, color="#d8d7d2", linewidth=0.8, zorder=0)
            b = factors.get(block)
            text = C.BLOCK_LABELS.get(block, block)
            if b and b.get("factor") is not None:
                text += f": {b['factor']:.2f}x"
            ax.text(right, y - 0.42, text, va="top", ha="right", fontsize=7, color=MUTED, style="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("end-to-end time, median over inputs (s); bars: median of each critical-path stage",
                  fontsize=8, color=MUTED)
    ax.set_xlim(0, right)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.6)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch

    order = [label for label, _ in GROUPS] + ["image, camera and other"]
    colours = dict(zip(order, list(COLOURS) + [REST]))
    present = [label for label in order if label in seen]  # fixed order, only the groups that occur
    ax.legend([Patch(facecolor=colours[label], edgecolor="none") for label in present], present,
              loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=min(4, len(present)), fontsize=7, frameon=False,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"waterfall.{suffix}", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="configuration files or directories (file order = row order)")
    ap.add_argument("--runs", required=True, type=Path, help="output directory of experiments/run_arms.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude-first", type=int, default=0,
                    help="inputs at the start of the list left out of the statistics (default: none, as the report)")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--xmax", type=float, default=None, help="right end of the figure's time axis (s)")
    ap.add_argument("--block-unit", choices=("object", "input"), default="object",
                    help="resampling unit of the block factors (default: the inputs of one object together)")
    args = ap.parse_args(argv)
    table = C.load_table(args.configs, args.runs, args.exclude_first)
    chain, controls = C.split_chain(table)
    rows = build(chain)
    blocks = build_blocks(chain, args.block_unit)
    args.out.mkdir(parents=True, exist_ok=True)
    C.write_csv(args.out / "waterfall.csv", rows, COLUMNS)
    C.write_csv(args.out / "waterfall_blocks.csv", blocks, BLOCK_COLUMNS)
    (args.out / "waterfall.json").write_text(json.dumps(
        {"exclude_first": args.exclude_first, "bootstrap": {"resamples": C.N_BOOT, "seed": C.SEED, "unit": "input",
                                                            "block_unit": args.block_unit},
         "rows": rows, "blocks": blocks, "controls": control_medians(controls)}, indent=1, default=str))
    if args.figure:
        figure(rows, args.out, args.xmax, blocks)
    for r in rows:
        if r.get("e2e_median") is not None:
            delta = (f"  delta {r['delta_prev_mean']:+.3f} s" if r.get("delta_prev_mean") is not None else "")
            print(f"{r['row']:>2} {r['name']:<34} n={r['n']:<3} median {r['e2e_median']:.3f} s{delta}")
        else:
            print(f"{r['row']:>2} {r['name']:<34} (no run)")
    for b in blocks:
        if b.get("factor") is not None:
            print(f"   {b['label']:<34} {b['from']} -> {b['to']}: {b['factor']:.2f}x "
                  f"[{b['factor_ci_lo']:.2f}, {b['factor_ci_hi']:.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
