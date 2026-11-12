#!/usr/bin/env python
"""Step 6: deterministic, stratified, nested training sets and a disjoint validation split from the keep pool.
Strata: source (keep shares, largest-remainder rounding) x fixed face-count bin (proportional within source).
Order inside a stratum: sha256("trellis500k_curated_v1/selection_v1:" + object sha256), ascending (the prefix is
the seed string of the released lists). Within each stratum the first q_dev objects form the validation set
(validation_400, eval/lists/validation_400.csv); the following objects form the nested training sets matched_5243 = first q_5243, curated_10k = first
q_10k, large_v1 = first q_50k (5243 c 10k c 50k, all disjoint from the validation set).
Output: <work_dir>/sets.pkl."""
import hashlib, json, os, pickle
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEED_STR = "trellis500k_curated_v1/selection_v1:"
SOURCES = ["sketchfab", "3dfuture", "hssd", "abo"]
BINS = [0, 1_000, 5_000, 20_000, 100_000, 500_000, 2_000_001]
BIN_LABELS = ["f<1k", "f1k-5k", "f5k-20k", "f20k-100k", "f100k-500k", "f500k-2M"]
SIZES = {"validation_400": 400, "matched_5243": 5243, "curated_10k": 10_000, "large_v1": 50_000}


def lr_round(weights, total):
    w = np.asarray(weights, dtype=float); raw = w / w.sum() * total
    base = np.floor(raw).astype(int); rem = raw - base
    for i in np.argsort(-rem, kind="stable")[: total - base.sum()]:
        base[i] += 1
    return base


cfg = common.config("nested sets")
o = pickle.load(open(common.work(cfg, "selection.pkl"), "rb"))
o["face_bin"] = pd.cut(o.face_count, BINS, right=False, labels=BIN_LABELS).astype(str)
o["priority"] = o.sha256.map(lambda h: hashlib.sha256((SEED_STR + h).encode()).hexdigest())
pool = o[o.selection == "keep"].copy()
assert (pool.face_bin != "nan").all(), pool[pool.face_bin == "nan"][["subset", "face_count"]].head()
counts = pool.groupby(["subset", "face_bin"]).size().unstack(fill_value=0).reindex(index=SOURCES, columns=BIN_LABELS, fill_value=0)
src_tot = counts.sum(axis=1)
quota = {}
for name, n in SIZES.items():
    per_src = dict(zip(SOURCES, lr_round(src_tot.values, n)))
    q = {}
    for s in SOURCES:
        q[s] = dict(zip(BIN_LABELS, lr_round(counts.loc[s].values, per_src[s])))
    quota[name] = q
# nesting feasibility: 5243 <= 10k <= 50k per cell, dev + 50k <= available
for s in SOURCES:
    for b in BIN_LABELS:
        assert quota["matched_5243"][s][b] <= quota["curated_10k"][s][b] <= quota["large_v1"][s][b], (s, b)
        assert quota["validation_400"][s][b] + quota["large_v1"][s][b] <= counts.loc[s, b], (s, b)
pool = pool.sort_values(["subset", "face_bin", "priority"])
pool["cell_rank"] = pool.groupby(["subset", "face_bin"]).cumcount()
qd = pool.apply(lambda r: quota["validation_400"][r.subset][r.face_bin], axis=1)
pool["in_validation_400"] = pool.cell_rank < qd
for name in ("matched_5243", "curated_10k", "large_v1"):
    qn = pool.apply(lambda r, nm=name: quota[nm][r.subset][r.face_bin], axis=1)
    pool["in_" + name] = (pool.cell_rank >= qd) & (pool.cell_rank < qd + qn)
    assert pool["in_" + name].sum() == SIZES[name], (name, pool["in_" + name].sum())
assert pool.in_validation_400.sum() == 400
assert not (pool.in_validation_400 & pool.in_large_v1).any()
assert (pool.in_matched_5243 <= pool.in_curated_10k).all() and (pool.in_curated_10k <= pool.in_large_v1).all()
pool["reserve_rank_in_cell"] = pool.cell_rank - qd  # >=0 for non-dev; next objects in a cell are the replacement order
o = o.merge(pool[["sha256", "cell_rank", "in_validation_400", "in_matched_5243", "in_curated_10k", "in_large_v1", "reserve_rank_in_cell"]], on="sha256", how="left")
for c in ["in_validation_400", "in_matched_5243", "in_curated_10k", "in_large_v1"]:
    o[c] = o[c].fillna(False).astype(bool)
pickle.dump({"objects": o, "quota": {k: {s: {b: int(x) for b, x in v2.items()} for s, v2 in v.items()} for k, v in quota.items()},
             "pool_counts": counts}, open(common.work(cfg, "sets.pkl"), "wb"))
print(counts.to_string()); print(src_tot.to_dict())
for name in SIZES:
    t = pd.DataFrame(quota[name]).T.reindex(index=SOURCES, columns=BIN_LABELS)
    t["total"] = t.sum(axis=1)
    print("##", name); print(t.to_string())
