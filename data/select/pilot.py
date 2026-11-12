#!/usr/bin/env python
"""Optional: pilot list for visual QA of the rules (renders of a stratified sample). Seeded.
keep: stratified sample of the keep pool (select.py): Sketchfab 150 by first Sketchfab category (proportional),
ABO/HSSD/3D-FUTURE 40 each by aesthetic-score quartile (small sources over-sampled on purpose, QA only).
control_rejected: 30 Sketchfab rejects spread over the HARD families (preferring objects where only that family fires).
Writes <work_dir>/out/pilot_list.csv."""
import os, pickle
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

cfg = common.config("pilot list")
SEED = 20260917
o = pickle.load(open(common.work(cfg, "selection.pkl"), "rb"))
rng = np.random.default_rng(SEED)
cols = ["subset", "sha256", "file_identifier", "raw_path", "shortest_caption", "license", "aesthetic_score", "face_count", "selection", "reject_reason", "soft_flags"]
rows = []
keep = o.selection == "keep"
skc = o[keep & (o.subset == "sketchfab")].copy()
skc["cat0"] = skc.categories.map(lambda c: c[0] if isinstance(c, list) and c else "none")
share = skc.cat0.value_counts(normalize=True)
alloc = (share * 150).round().astype(int).clip(lower=2)
while alloc.sum() > 150:
    alloc[alloc.idxmax()] -= 1
while alloc.sum() < 150:
    alloc[alloc.idxmax()] += 1
for c in sorted(alloc.index):
    pick = skc[skc.cat0 == c].sample(n=int(alloc[c]), random_state=int(rng.integers(1 << 31)))
    for _, r in pick.iterrows():
        rows.append({**{k: r[k] for k in cols[:8]}, "selection": "keep", "reject_reason": "", "soft_flags": r["soft_flags"]})
for sub in ("abo", "hssd", "3dfuture"):
    pool = o[keep & (o.subset == sub)].copy()
    pool["q"] = pd.qcut(pool.aesthetic_score, 4, labels=False)
    for q in range(4):
        pick = pool[pool.q == q].sample(n=10, random_state=int(rng.integers(1 << 31)))
        for _, r in pick.iterrows():
            rows.append({**{k: r[k] for k in cols[:8]}, "selection": "keep", "reject_reason": "", "soft_flags": r["soft_flags"]})
# controls: (family, sub-condition label, mask, n)
rej = (o.subset == "sketchfab") & (o.selection == "reject")
nh = o[[c for c in o.columns if c.startswith("hard_")]].sum(axis=1)
plan = [
    ("scene", "uploader_scene", o.uploader_scene & ~o.scene_head & ~o.ph_scene, 3),
    ("scene", "scene_head", o.scene_head & ~o.uploader_scene, 3),
    ("scene", "scene_phrase", o.ph_scene & ~o.scene_head & ~o.uploader_scene, 2),
    ("scene_context", "context", o.hard_scene_context, 4),
    ("multi_object_set", "multi", o.hard_multi_object_set, 4),
    ("flat_or_text", "flat", o.hard_flat_or_text, 3),
    ("point_cloud", "pc", o.hard_point_cloud, 3),
    ("world_export", "we", o.hard_world_export, 2),
    ("extreme_face_count", "low", o.hard_extreme_face_count & (o.face_count < 12), 2),
    ("extreme_face_count", "huge", o.hard_extreme_face_count & (o.face_count > 2_000_000) & (o.face_count < 5_000_000), 2),
    ("no_caption_sketchfab", "nocap", o.hard_no_caption_sketchfab, 2),
]
used = set()
for fam, lab, m, n in plan:
    pool = o[rej & m & (nh == 1) & ~o.sha256.isin(used)]
    if len(pool) < n:
        pool = o[rej & m & ~o.sha256.isin(used)]
    pick = pool.sample(n=n, random_state=int(rng.integers(1 << 31)))
    for _, r in pick.iterrows():
        used.add(r.sha256)
        rows.append({**{k: r[k] for k in cols[:8]}, "selection": "control_rejected", "reject_reason": r["reject_reason"] + (f" [{lab}]" if fam == "scene" or fam == "extreme_face_count" else ""), "soft_flags": r["soft_flags"]})
P = pd.DataFrame(rows, columns=cols)
P["face_count"] = P["face_count"].map(lambda x: "" if pd.isna(x) else str(int(x)))
P["aesthetic_score"] = P["aesthetic_score"].map(lambda x: f"{x:.4f}")
P["shortest_caption"] = P["shortest_caption"].fillna("")
assert P.sha256.is_unique and len(P) == 150 + 120 + sum(p[3] for p in plan)
P.to_csv(os.path.join(common.out_dir(cfg), "pilot_list.csv"), index=False)
print(P.groupby(["subset", "selection"]).size().to_string())
print(P[P.selection == "control_rejected"][["reject_reason", "shortest_caption", "face_count"]].to_string())
print("soft among sketchfab keeps:", P[(P.selection == "keep") & (P.subset == "sketchfab")].soft_flags.str.split("+").explode().value_counts().to_dict())
