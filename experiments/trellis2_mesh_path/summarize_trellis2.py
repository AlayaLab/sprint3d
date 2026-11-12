#!/usr/bin/env python3
r"""Rows of the report's TRELLIS.2 table from timing runs of time_trellis2.py and evaluation runs of eval/run_eval.py.

    python experiments/trellis2_mesh_path/summarize_trellis2.py \
        --run stock=runs/trellis2_stock --run mesh_path=runs/trellis2_mesh/mesh_path \
        --run mesh_path_adaptive_remesh=runs/trellis2_mesh/mesh_path_adaptive_remesh \
        --serial runs/trellis2_mesh/mesh_path_serial \
        --eval stock=results/trellis2_stock --eval mesh_path=results/trellis2_mesh_path \
        --eval mesh_path_adaptive_remesh=results/trellis2_mesh_path_adaptive_remesh \
        --colour results/trellis2_colour.json \
        --out results/trellis2_table.json --markdown results/trellis2_table.md

The first ``--run`` is the reference (TRELLIS.2 as released). Per run, over the measured inputs (all inputs by
default; ``--exclude-first`` drops the first ones of the list):

  end-to-end time   median, mean and 90th percentile of ``--e2e-key`` (default ``stage_sum_s``, the definition of the
                    report's table; ``e2e_s`` is the wall clock of docs/TIMING.md)
  post-processing   median of ``post_processing_s`` (everything after the flows on the critical path)
  speed-up          ratio of the medians (reference / run) on the inputs present in both, with a 95 % percentile
                    interval of a bootstrap over objects (both views of an object resampled together; 10,000
                    resamples, seed 20260918); also the median of the per-input ratios and the per-input ratio of
                    the flow times (the generation stages run the same code in both)
  conservative      with ``--serial``: the reference's own generation stages plus the run's post-processing beyond
                    an uncontended texture flow (the serial run's texture-flow time on the same input), i.e. no
                    generation-side difference is credited to the run

``--eval NAME=DIR`` adds the paired per-object differences to the reference (mean over objects, bootstrap 95 % CI
and sign test of eval/metrics/stats.py) of ``--metrics`` (default: full-image normal PSNR and mean angular error)
from the ``per_object.json`` of evaluation runs with the baseline manifest of each arm (make_baseline_manifest.py).
Objects are the input ids without the ``_v<view>`` suffix.

``--colour FILE`` (the output of colour_metrics.py; run names as in ``--run``) adds the base-colour results:

  delta colour      per run, the paired per-object difference (run - reference, views of an object averaged) of the
                    base colour at the input view scored against the input image (PSNR, SSIM, LPIPS), with the
                    bootstrap 95 % CI and sign test of eval/metrics/stats.py: the "Delta color" column
  agreement         per compared pair of products, the mean over objects (views averaged per object) of the mean
                    over the five views, with a 95 % percentile interval of a bootstrap of that mean over objects
                    (10,000 resamples, seed 20260918): the 42.7 dB (mesh path against TRELLIS.2 as released) and
                    43.1 dB (mesh path against its serial export of the same latents) of the report
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
N_BOOT = 10_000
SEED = 20260918


def named(values: List[str], flag: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"{flag} expects NAME=DIR, got {v!r}")
        name, path = v.split("=", 1)
        if name in out:
            raise SystemExit(f"{flag}: {name!r} given twice")
        out[name] = Path(path)
    return out


def object_of(ident: str) -> str:
    return ident.rsplit("_v", 1)[0] if "_v" in ident else ident


def load_rows(directory: Path, exclude_first: int) -> Dict[str, Dict[str, Any]]:
    path = directory / "timing.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return {r["id"]: r for r in rows if r.get("status") == "ok" and int(r["index"]) > exclude_first}


def cluster_indices(ids: List[str], n_boot: int = N_BOOT, seed: int = SEED) -> List[np.ndarray]:
    objects = sorted({object_of(i) for i in ids})
    members = {o: [j for j, i in enumerate(ids) if object_of(i) == o] for o in objects}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(objects), size=(n_boot, len(objects)))
    return [np.concatenate([members[objects[k]] for k in row]) for row in pick]


def interval(stat, indices) -> List[float]:
    values = np.asarray([stat(ix) for ix in indices], np.float64)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def time_of(row: Dict[str, Any], key: str) -> float:
    return float(row["e2e_s"] if key == "e2e_s" else row["timings"][key])


def summarize_run(rows: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    e = np.asarray([time_of(r, key) for r in rows.values()], np.float64)
    post = [float(r["timings"]["post_processing_s"]) for r in rows.values()]
    flows = [float(r["timings"]["flows_s"]) for r in rows.values()]
    return {"n": int(len(e)), "median": float(np.median(e)), "mean": float(e.mean()),
            "p90": float(np.percentile(e, 90)), "post_processing_median": float(np.median(post)),
            "flows_median": float(np.median(flows)), "faces_median": float(np.median(
                [r["stats"].get("output_faces", float("nan")) for r in rows.values()]))}


def uncontended_post(row: Dict[str, Any], key: str, serial: Optional[Dict[str, Any]]) -> float:
    """Post-processing of ``row`` beyond an uncontended texture flow: the stage total minus preprocessing minus the
    flows other than the texture flow minus the serial run's texture-flow time on the same input."""
    t = row["timings"]
    tex_ref = float((serial or row)["timings"].get("sample_tex_slat_s", t.get("sample_tex_slat_s", 0.0)))
    flows_without_tex = float(t["flows_s"]) - float(t.get("sample_tex_slat_s", 0.0))
    return time_of(row, key) - float(t.get("preprocess_image_s", 0.0)) - flows_without_tex - tex_ref


def compare(ref: Dict[str, Dict[str, Any]], run: Dict[str, Dict[str, Any]], key: str,
            serial: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    ids = [i for i in ref if i in run]
    if len(ids) < 2:
        return {"n": len(ids)}
    a = np.asarray([time_of(ref[i], key) for i in ids])
    b = np.asarray([time_of(run[i], key) for i in ids])
    fa = np.asarray([float(ref[i]["timings"]["flows_s"]) for i in ids])
    fb = np.asarray([float(run[i]["timings"]["flows_s"]) for i in ids])
    idx = cluster_indices(ids)
    out = {"n": len(ids), "n_objects": len({object_of(i) for i in ids}),
           "speedup_ratio_of_medians": float(np.median(a) / np.median(b)),
           "speedup_ratio_of_medians_ci95": interval(lambda ix: np.median(a[ix]) / np.median(b[ix]), idx),
           "speedup_median_of_ratios": float(np.median(a / b)),
           "speedup_median_of_ratios_ci95": interval(lambda ix: np.median(a[ix] / b[ix]), idx),
           "flows_ratio_median": float(np.median(fb / fa)),
           "flows_ratio_median_ci95": interval(lambda ix: np.median(fb[ix] / fa[ix]), idx)}
    if serial is not None:
        gen = np.asarray([float(ref[i]["timings"].get("preprocess_image_s", 0.0)) + float(ref[i]["timings"]["flows_s"])
                          + uncontended_post(run[i], key, serial.get(i)) for i in ids])
        out.update(conservative_median=float(np.median(gen)),
                   conservative_speedup=float(np.median(a) / np.median(gen)),
                   conservative_speedup_ci95=interval(lambda ix: np.median(a[ix]) / np.median(gen[ix]), idx))
    return out


def per_object(directory: Path, arm: Optional[str]) -> Dict[str, Dict[str, float]]:
    data = json.loads((directory / "per_object.json").read_text())
    if arm is None:
        if len(data) != 1:
            raise SystemExit(f"{directory}/per_object.json has arms {sorted(data)}; give --eval-arm")
        arm = next(iter(data))
    return data[arm]


def quality(ref_dir: Path, run_dir: Path, metrics: List[str], arm: Optional[str]) -> Dict[str, Any]:
    sys.path.insert(0, str(REPO / "eval"))
    from metrics import stats

    ref, run = per_object(ref_dir, arm), per_object(run_dir, arm)
    out = {}
    for m in metrics:
        cand = {o: v.get(m) for o, v in run.items()}
        base = {o: v.get(m) for o, v in ref.items()}
        boot = stats.paired_bootstrap(cand, base)
        try:
            sign = stats.sign_test(cand, base)
        except ImportError:
            sign = None
        out[m] = {"bootstrap": boot, "sign_test": sign}
    return out


def mean_ci(values: List[float], n_boot: int = N_BOOT, seed: int = SEED) -> List[float]:
    """95 % percentile interval of the mean over a bootstrap of ``values``."""
    d = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [float(lo), float(hi)]


def colour(path: Path, reference: Optional[str] = None) -> Dict[str, Any]:
    """Delta colour against the input image and agreement of products, from colour_metrics.py output."""
    sys.path.insert(0, str(REPO / "eval"))
    from metrics import stats

    data = json.loads(Path(path).read_text())
    ref = reference or data.get("reference") or "official"
    per = data["per_input"]
    out: Dict[str, Any] = {"reference": ref, "delta_vs_input": {}, "agreement": {}}
    arms = sorted({a for e in per.values() for a in e.get("vs_input", {}) if a != ref})
    for arm in arms:
        cand: Dict[str, list] = {}
        base: Dict[str, list] = {}
        for e in per.values():
            vi = e.get("vs_input", {})
            if arm in vi and ref in vi:
                cand.setdefault(e["sha256"], []).append(vi[arm])
                base.setdefault(e["sha256"], []).append(vi[ref])
        res = {}
        for m in ("tex_psnr", "tex_ssim", "tex_lpips"):
            c = {k: float(np.mean([x[m] for x in v])) for k, v in cand.items() if all(m in x for x in v)}
            r = {k: float(np.mean([x[m] for x in v])) for k, v in base.items() if all(m in x for x in v)}
            boot = stats.paired_bootstrap(c, r)
            if boot is None:
                continue
            try:
                sign = stats.sign_test(c, r)
            except ImportError:
                sign = None
            res[m] = {"bootstrap": boot, "sign_test": sign, "reference_mean": float(np.mean(list(r.values()))),
                      "run_mean": float(np.mean(list(c.values())))}
        out["delta_vs_input"][arm] = res
    for pk in sorted({k for e in per.values() for k in e.get("pairs", {})}):
        per_obj: Dict[str, list] = {}
        for e in per.values():
            if pk in e.get("pairs", {}):
                per_obj.setdefault(e["sha256"], []).append(e["pairs"][pk]["mean"])
        res = {"n_objects": len(per_obj)}
        for m in ("tex_psnr", "tex_ssim", "tex_lpips", "mask_iou"):
            v = [float(np.mean([x[m] for x in vals])) for vals in per_obj.values() if all(m in x for x in vals)]
            if v:
                res[m] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                          "ci95": mean_ci(v) if len(v) > 1 else None}
        out["agreement"][pk] = res
    return out


def fmt_ci(v, ci, digits=1, suffix="") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{digits}f}{suffix} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def markdown(report: Dict[str, Any], metrics: List[str]) -> str:
    head = ["configuration", "n", "median (s)", "mean (s)", "p90 (s)", "post-proc. (s)", "speed-up"]
    head += [f"delta {m}" for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in report["runs"].items():
        c = report["paired"].get(name, {})
        q = report["quality"].get(name, {})
        cells = [name, str(r["n"]), f"{r['median']:.1f}", f"{r['mean']:.1f}", f"{r['p90']:.1f}",
                 f"{r['post_processing_median']:.1f}",
                 fmt_ci(c.get("speedup_ratio_of_medians"), c.get("speedup_ratio_of_medians_ci95"), 1, "x")
                 if c.get("speedup_ratio_of_medians") else "--"]
        for m in metrics:
            b = (q.get(m) or {}).get("bootstrap")
            cells.append(fmt_ci(b["mean_diff"], [b["ci95_lo"], b["ci95_hi"]], 3) if b else "--")
        if report.get("colour"):
            b = ((report["colour"]["delta_vs_input"].get(name) or {}).get("tex_psnr") or {}).get("bootstrap")
            cells.append(fmt_ci(b["mean_diff"], [b["ci95_lo"], b["ci95_hi"]], 3) if b else "--")
        lines.append("| " + " | ".join(cells) + " |")
    if report.get("colour"):
        lines[0] = lines[0][:-1] + "| delta colour PSNR (dB) |"
        lines[1] += "---|"
        lines += ["", "| base-colour agreement (5 views) | objects | PSNR (dB) | SSIM | LPIPS | mask IoU |",
                  "|---|---|---|---|---|---|"]
        for pk, a in report["colour"]["agreement"].items():
            cell = lambda m, d: (fmt_ci(a[m]["mean"], a[m]["ci95"], d) if a.get(m, {}).get("ci95")  # noqa: E731
                                 else (f"{a[m]['mean']:.{d}f}" if m in a else "--"))
            lines.append(f"| {pk} | {a['n_objects']} | {cell('tex_psnr', 2)} | {cell('tex_ssim', 4)} | "
                         f"{cell('tex_lpips', 4)} | {cell('mask_iou', 4)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="timing directory of one arm (the first one is the reference)")
    ap.add_argument("--serial", default=None, help="timing directory of the serial arm (conservative estimate)")
    ap.add_argument("--e2e-key", default="stage_sum_s", choices=("stage_sum_s", "e2e_s"))
    ap.add_argument("--exclude-first", type=int, default=0)
    ap.add_argument("--eval", action="append", default=[], metavar="NAME=DIR", help="run_eval.py output of one arm")
    ap.add_argument("--eval-arm", default=None, help="arm key inside per_object.json (default: its only arm)")
    ap.add_argument("--metrics", default="psnr_full,ang_mean")
    ap.add_argument("--colour", type=Path, default=None, help="output of colour_metrics.py")
    ap.add_argument("--colour-reference", default=None,
                    help="name of the reference inside --colour (default: the file's own reference)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)
    runs = named(args.run, "--run")
    evals = named(args.eval, "--eval")
    metrics = [m for m in args.metrics.split(",") if m]
    names = list(runs)
    ref_name = names[0]
    rows = {n: load_rows(d, args.exclude_first) for n, d in runs.items()}
    serial = load_rows(Path(args.serial), args.exclude_first) if args.serial else None
    report: Dict[str, Any] = {"reference": ref_name, "e2e_key": args.e2e_key, "exclude_first": args.exclude_first,
                              "bootstrap": {"resamples": N_BOOT, "seed": SEED, "unit": "object"},
                              "runs": {n: summarize_run(r, args.e2e_key) for n, r in rows.items()},
                              "paired": {n: compare(rows[ref_name], rows[n], args.e2e_key, serial)
                                         for n in names[1:]},
                              "quality": {}}
    if evals:
        if ref_name not in evals:
            raise SystemExit(f"--eval needs the reference {ref_name!r} too")
        for n, d in evals.items():
            if n != ref_name:
                report["quality"][n] = quality(evals[ref_name], d, metrics, args.eval_arm)
    if args.colour:
        report["colour"] = colour(args.colour, args.colour_reference)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    table = markdown(report, metrics)
    if args.markdown:
        args.markdown.write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
