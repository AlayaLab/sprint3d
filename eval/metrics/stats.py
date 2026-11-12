"""stats.py -- aggregation and statistics.

  * views -> object: per-view metrics averaged per object first; objects -> corpus: mean over objects (objects
    weighted equally).
  * paired bootstrap 95 % CI of the per-object delta (candidate - reference): n_boot = 10,000, seed 20260918,
    percentile interval; only objects present in both arms.
  * sign test on per-object deltas (two-sided exact binomial, ties dropped; scipy.stats.binomtest).
  * Holm step-down correction over a family of p-values.
  * flips: count of objects whose value exceeds a threshold (Chamfer to the reference above the noise band; threshold
    supplied by the caller, see vs_reference.flip_threshold_from_noise_band).
"""
from __future__ import annotations

import math

import numpy as np

N_BOOT = 10_000
BOOT_SEED = 20260918


def _finite(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    return a[np.isfinite(a)]


def summarize(values) -> dict:
    v = _finite(values)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}


def views_to_objects(rows, object_key="object_id", metric_keys=None) -> dict:
    """rows: list of dicts with `object_key` and flat metric values. Returns {object: {metric: mean}}
    plus 'n_views'.  Non-finite / None values are skipped per metric."""
    per = {}
    for r in rows:
        per.setdefault(r[object_key], []).append(r)
    out = {}
    for obj, rs in per.items():
        keys = metric_keys or sorted({k for r in rs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        agg = {"n_views": len(rs)}
        for k in keys:
            vals = _finite([r.get(k) for r in rs])
            agg[k] = float(vals.mean()) if len(vals) else None
        out[obj] = agg
    return out


def paired_bootstrap(cand: dict, ref: dict, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05) -> dict | None:
    common = sorted(o for o in cand if o in ref and cand[o] is not None and ref[o] is not None
                    and math.isfinite(cand[o]) and math.isfinite(ref[o]))
    if not common:
        return None
    d = np.asarray([cand[o] - ref[o] for o in common], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "ci95_lo": float(lo), "ci95_hi": float(hi), "n_boot": int(n_boot), "seed": int(seed),
            "significant": bool(lo > 0 or hi < 0)}


def sign_test(cand: dict, ref: dict) -> dict | None:
    from scipy.stats import binomtest
    common = [o for o in cand if o in ref and cand[o] is not None and ref[o] is not None]
    if not common:
        return None
    d = np.asarray([cand[o] - ref[o] for o in common], np.float64)
    d = d[np.isfinite(d)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    ties = int(len(d) - pos - neg)
    n = pos + neg
    p = float(binomtest(pos, n, 0.5, alternative="two-sided").pvalue) if n > 0 else 1.0
    return {"n_pos": pos, "n_neg": neg, "n_tie": ties, "p_value": p}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family {name: p}."""
    items = [(k, v) for k, v in pvals.items() if v is not None and math.isfinite(v)]
    m = len(items)
    if m == 0:
        return {}
    order = sorted(items, key=lambda kv: kv[1])
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adj[k] = running
    return adj


def count_flips(values: dict, threshold: float, higher_is_worse=True) -> dict:
    v = _finite(list(values.values()))
    n = int((v > threshold).sum()) if higher_is_worse else int((v < threshold).sum())
    return {"n_flips": n, "n": int(len(v)), "threshold": float(threshold)}


def compare_arms(per_object_by_arm: dict, ref_arm: str, metrics, higher_is_better: dict, n_boot=N_BOOT,
                 seed=BOOT_SEED, holm_family=True) -> dict:
    """per_object_by_arm: {arm: {object: {metric: value}}}.  Returns per-arm summaries and, for every
    non-reference arm, paired bootstrap + sign test per metric (+ Holm over the metrics of that arm)."""
    out = {"reference": ref_arm, "summary": {}, "paired": {}}
    for arm, objs in per_object_by_arm.items():
        out["summary"][arm] = {m: summarize([o.get(m) for o in objs.values()]) for m in metrics}
    ref = per_object_by_arm.get(ref_arm, {})
    for arm, objs in per_object_by_arm.items():
        if arm == ref_arm:
            continue
        rows, pv = {}, {}
        for m in metrics:
            cand = {o: v.get(m) for o, v in objs.items()}
            base = {o: v.get(m) for o, v in ref.items()}
            bs = paired_bootstrap(cand, base, n_boot=n_boot, seed=seed)
            st = sign_test(cand, base)
            better = None
            if bs is not None and higher_is_better.get(m) is not None:
                better = (bs["mean_diff"] > 0) == bool(higher_is_better[m])
            rows[m] = {"bootstrap": bs, "sign_test": st, "candidate_better_on_mean": better}
            if st is not None:
                pv[m] = st["p_value"]
        if holm_family:
            adj = holm(pv)
            for m, p in adj.items():
                rows[m]["sign_test"]["p_holm"] = p
        out["paired"][arm] = rows
    return out
