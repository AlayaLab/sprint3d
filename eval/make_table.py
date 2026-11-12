#!/usr/bin/env python
"""make_table.py -- the LaTeX rows of the report's normal-map table from per-object results (no manual editing).

Input: one or more per_object.json files written by run_eval.py ({arm: {sha256: {metric: value}}}).  Two rows are given
with --row "LABEL=ARM": the reference (Pixal3D (official)) first, then the candidate (Ours).  Additional reference rows
are optional and off by default; each is printed only when passed explicitly with --extra-row "LABEL=ARM".  An earlier
version of the report's table had one: Ours at a 200,000-face / 2048^2 export, a setting this repository does not produce (its export is
1,000,000 faces / 4096^2).  Every row must cover the same objects.

Outputs (--out-dir):
  table1.tex              main table: PSNR / SSIM / LPIPS in the overlap region; in every column the better of the
                          first two rows is set in bold; additional rows are not bolded (label in italics)
  table1_full_region.tex  the same rows with PSNR / SSIM / LPIPS on the full image; markers on the candidate rows:
                          $^{\\ast}$ / $^{\\dagger}$ = significantly better / worse than the reference row (paired per-object
                          sign test, Holm-adjusted over the 13 normal-map metrics, p < 0.05)
  table1_values.csv       per row and metric: n, mean, 95% bootstrap CI; per candidate: paired difference vs the
                          reference, its CI, sign-test counts, p and Holm-adjusted p

Statistics (metrics/stats.py): object means (views averaged per object first); percentile bootstrap, 10,000 resamples,
seed 20260918; exact two-sided sign test with ties dropped; Holm over the 13 metrics IoU, PSNR / SSIM / LPIPS in both
regions, Mean, Median, Mean_B and the three accuracies.

The E2E column is not an output of the evaluation: it is the median end-to-end time of the dedicated timing run
(docs/TIMING.md) and is passed with --e2e "LABEL=SECONDS".  The rows of methods without results (--not-reported) are
printed as "not reported".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE))

from metrics import stats as est  # noqa: E402

T1 = ["mask_iou", "psnr_ov", "ssim_ov", "lpips_ov", "psnr_full", "ssim_full", "lpips_full",
      "ang_mean", "ang_median", "ang_mean_b", "acc_11.25", "acc_22.5", "acc_30.0"]
COLS = {"overlap": ["mask_iou", "psnr_ov", "ssim_ov", "lpips_ov", "ang_mean", "ang_median", "ang_mean_b",
                    "acc_11.25", "acc_22.5", "acc_30.0"],
        "full": ["mask_iou", "psnr_full", "ssim_full", "lpips_full", "ang_mean", "ang_median", "ang_mean_b",
                 "acc_11.25", "acc_22.5", "acc_30.0"]}
LOWER = {"lpips_ov", "lpips_full", "ang_mean", "ang_median", "ang_mean_b"}
SCALE = {"mask_iou": 100.0, "acc_11.25": 100.0, "acc_22.5": 100.0, "acc_30.0": 100.0}
DIG = {"mask_iou": 2, "psnr_ov": 2, "psnr_full": 2, "ssim_ov": 3, "ssim_full": 3, "lpips_ov": 3, "lpips_full": 3,
       "ang_mean": 2, "ang_median": 2, "ang_mean_b": 2, "acc_11.25": 2, "acc_22.5": 2, "acc_30.0": 2}
HEAD = {"mask_iou": r"IoU$\uparrow$", "psnr_ov": r"PSNR$\uparrow$", "psnr_full": r"PSNR$\uparrow$",
        "ssim_ov": r"SSIM$\uparrow$", "ssim_full": r"SSIM$\uparrow$", "lpips_ov": r"LPIPS$\downarrow$",
        "lpips_full": r"LPIPS$\downarrow$", "ang_mean": r"Mean$\downarrow$", "ang_median": r"Median$\downarrow$",
        "ang_mean_b": r"Mean$_B\downarrow$", "acc_11.25": r"11.25$^\circ\uparrow$", "acc_22.5": r"22.5$^\circ\uparrow$",
        "acc_30.0": r"30$^\circ\uparrow$"}
MARK = {"better": r"$^{\ast}$", "worse": r"$^{\dagger}$"}
NOT_REPORTED = ["TRELLIS.2", "TripoSG", "Hunyuan3D-2.1"]


def fmt(v, m):
    return "--" if v is None else f"{v * SCALE.get(m, 1.0):.{DIG.get(m, 3)}f}"


def load_per_object(paths) -> dict:
    merged = {}
    for p in paths:
        for arm, objs in json.loads(Path(p).read_text()).items():
            if arm in merged:
                raise SystemExit(f"arm {arm!r} appears in more than one per_object.json")
            merged[arm] = objs
    return merged


def values(objs: dict, m: str) -> dict:
    out = {}
    for o, v in objs.items():
        x = v.get(m)
        if x is not None:
            out[o] = float(x)
    return out


def mean_ci(objs: dict, m: str) -> dict:
    vals = values(objs, m)
    b = est.paired_bootstrap(vals, {o: 0.0 for o in vals})
    return {"n": b["n"], "mean": b["mean_diff"], "lo": b["ci95_lo"], "hi": b["ci95_hi"]} if b else {"n": 0, "mean": None}


def paired(cand: dict, ref: dict) -> dict:
    out = {}
    for m in T1:
        c, r = values(cand, m), values(ref, m)
        b, s = est.paired_bootstrap(c, r), est.sign_test(c, r)
        low = m in LOWER
        out[m] = {"n_obj": b["n"], "mean_diff": b["mean_diff"], "ci_lo": b["ci95_lo"], "ci_hi": b["ci95_hi"],
                  "n_better": s["n_neg"] if low else s["n_pos"], "n_worse": s["n_pos"] if low else s["n_neg"],
                  "n_tie": s["n_tie"], "p_sign": s["p_value"],
                  "direction": "better" if ((b["mean_diff"] > 0) != low) else "worse"}
    for m, p in est.holm({m: out[m]["p_sign"] for m in T1}).items():
        out[m]["p_holm"] = p
        out[m]["marker"] = out[m]["direction"] if p < 0.05 else ""
    return out


def e2e_cell(label, e2e, reference: bool) -> str:
    if label not in e2e:
        return "--"
    return f"{e2e[label]:.1f}" if reference else f"{e2e[label]:.2f}"


def latex_table(rows, means, pr, e2e, region, style, not_reported) -> str:
    cols = COLS[region]
    n_cols = len(cols) + 1
    L = [r"\tablestyle{3.5pt}{1.1}", r"\begin{tabular}{@{}l" + "c" * n_cols + "@{}}", r"\toprule",
         "Method & " + " & ".join(HEAD[c] for c in cols) + r" & E2E (s)$\downarrow$ \\", r"\midrule"]
    if not_reported:
        for b in not_reported:
            L.append(f"{b} & \\multicolumn{{{n_cols}}}{{c}}{{\\textit{{not reported}}}} \\\\")
        L.append(r"\midrule")
    (ref_label, _), (main_label, _) = rows[0], rows[1]
    bold = {}
    if style == "bold":              # better of the first two rows per column (ties: both)
        for c in cols:
            a, b = fmt(means[ref_label][c]["mean"], c), fmt(means[main_label][c]["mean"], c)
            if "--" in (a, b):
                continue
            if a == b:
                bold[c] = {ref_label, main_label}
            else:
                better_ref = (means[ref_label][c]["mean"] < means[main_label][c]["mean"]) == (c in LOWER)
                bold[c] = {ref_label if better_ref else main_label}
        if ref_label in e2e and main_label in e2e:
            bold["e2e"] = {ref_label if e2e[ref_label] < e2e[main_label] else main_label}
    for i, (label, _arm) in enumerate(rows):
        if i == 2:
            L.append(r"\midrule")
        cells = []
        for c in cols:
            s = fmt(means[label][c]["mean"], c)
            if style == "bold" and label in bold.get(c, ()):
                s = r"\textbf{" + s + "}"
            if style == "markers" and i > 0:
                s += MARK.get(pr[label][c]["marker"], "")
            cells.append(s)
        e = e2e_cell(label, e2e, i == 0)
        if style == "bold" and label in bold.get("e2e", ()):
            e = r"\textbf{" + e + "}"
        name = label
        if i == 1:
            name = r"\textbf{" + label + "}"
        elif i >= 2 and style == "bold":
            name = r"\textit{" + label + "}"
        L.append((r"\rowcolor{rowgray}" + "\n" if i == 1 else "") + name + " & " + " & ".join(cells) + f" & {e} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-object", action="append", required=True, help="per_object.json of run_eval.py (repeatable)")
    ap.add_argument("--row", action="append", required=True,
                    help='"LABEL=ARM", exactly twice, in table order: reference, candidate')
    ap.add_argument("--extra-row", action="append", default=[],
                    help='"LABEL=ARM" of an additional reference row below the two (optional, repeatable; default: none)')
    ap.add_argument("--e2e", action="append", default=[], help='"LABEL=SECONDS" from the dedicated timing run')
    ap.add_argument("--not-reported", default=",".join(NOT_REPORTED),
                    help="comma list of method rows printed as 'not reported' ('' for none)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    po = load_per_object(a.per_object)
    if len(a.row) != 2:
        raise SystemExit("give --row exactly twice (reference, candidate); additional rows with --extra-row")
    rows = [tuple(x.split("=", 1)) for x in a.row + a.extra_row]
    for label, arm in rows:
        if arm not in po:
            raise SystemExit(f"arm {arm!r} of row {label!r} not in the per-object files (arms: {sorted(po)})")
    objs = {label: po[arm] for label, arm in rows}
    keys = {label: set(o) for label, o in objs.items()}
    if len({frozenset(k) for k in keys.values()}) != 1:
        raise SystemExit("rows do not cover the same objects: " + ", ".join(f"{k}: {len(v)}" for k, v in keys.items()))
    e2e = {k: float(v) for k, v in (x.rsplit("=", 1) for x in a.e2e)}
    unknown = set(e2e) - {label for label, _ in rows}
    if unknown:
        raise SystemExit(f"--e2e labels not among the rows: {sorted(unknown)}")
    means = {label: {m: mean_ci(o, m) for m in T1} for label, o in objs.items()}
    ref_label = rows[0][0]
    pr = {label: paired(objs[label], objs[ref_label]) for label, _ in rows[1:]}
    not_reported = [s.strip() for s in a.not_reported.split(",") if s.strip()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table1.tex").write_text(latex_table(rows, means, pr, e2e, "overlap", "bold", not_reported))
    (out / "table1_full_region.tex").write_text(latex_table(rows, means, pr, e2e, "full", "markers", not_reported))
    with open(out / "table1_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "arm", "metric", "n_obj", "mean", "ci_lo", "ci_hi", "diff_vs_reference", "diff_ci_lo",
                    "diff_ci_hi", "n_better", "n_worse", "n_tie", "p_sign", "p_holm_13", "marker"])
        for label, arm in rows:
            for m in T1:
                s = means[label][m]
                x = pr.get(label, {}).get(m, {})
                w.writerow([label, arm, m, s["n"], f"{s['mean']:.8g}", f"{s['lo']:.8g}", f"{s['hi']:.8g}"]
                           + [("" if x.get(k) is None else (f"{x[k]:.8g}" if isinstance(x[k], float) else x[k]))
                              for k in ("mean_diff", "ci_lo", "ci_hi", "n_better", "n_worse", "n_tie", "p_sign", "p_holm", "marker")])
    n = len(next(iter(keys.values())))
    print(f"n = {n} objects per row; wrote {out}/table1.tex, table1_full_region.tex, table1_values.csv"
          + ("" if e2e else "; E2E column empty (pass --e2e)"))


if __name__ == "__main__":
    main()
