#!/usr/bin/env python
"""Optional audit: caption nearest neighbours (TF-IDF) in the keep pool of every object of a Toys4K evaluation
subset (config.yaml ``inputs.toys4k_audit_list``: CSV with object_id and category), as a list for a later
geometric near-duplicate check. Writes <work_dir>/out/toys4k_caption_neighbours.csv."""
import json, os, pickle
import sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

cfg = common.config("Toys4K caption-neighbour audit")
if not cfg["inputs"]["toys4k_audit_list"]:
    raise SystemExit("inputs.toys4k_audit_list is not set; nothing to audit")
o = pickle.load(open(common.work(cfg, "final_objects.pkl"), "rb"))
toys = pickle.load(open(common.work(cfg, "joined.pkl"), "rb"))["toys4k"]
ev = pd.read_csv(cfg["inputs"]["toys4k_audit_list"])
toys["key"] = toys.file_identifier.str.split("/").str[1]
tcap = toys.set_index("key").captions.to_dict()
keep = o[o.selection == "keep"].reset_index(drop=True)
docs = [" ".join(c) for c in keep.captions]
evd = [" ".join(tcap.get(k) or []) for k in ev.object_id]
vec = TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words="english", dtype=np.float32)
X = vec.fit_transform(docs + evd); Xk, Xe = X[: len(docs)], X[len(docs):]
Sm = (Xe @ Xk.T).toarray()
rows = []
for i, oid in enumerate(ev.object_id):
    for rnk, j in enumerate(np.argsort(-Sm[i])[:10]):
        r = keep.iloc[j]
        rows.append({"eval_object_id": oid, "eval_category": ev.category[i], "eval_shortest_caption": min(tcap.get(oid) or [""], key=len),
                     "rank": rnk + 1, "cos": round(float(Sm[i, j]), 3), "subset": r.subset, "sha256": r.sha256, "raw_path": r.raw_path,
                     "shortest_caption": r.shortest_caption, "in_validation_400": bool(r.in_validation_400), "in_matched_5243": bool(r.in_matched_5243),
                     "in_curated_10k": bool(r.in_curated_10k), "in_large_v1": bool(r.in_large_v1)})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(common.out_dir(cfg), "toys4k_caption_neighbours.csv"), index=False)
print("eval with caption", sum(bool(x) for x in evd), "rows", len(R))
print("cos>=0.8 pairs", int((R.cos >= 0.8).sum()), "in 10k", int(((R.cos >= 0.8) & R.in_curated_10k).sum()), "in validation", int(((R.cos >= 0.8) & R.in_validation_400).sum()))
print(R[R["rank"] == 1].cos.describe().round(3).to_dict())
print(R[(R["rank"] == 1)].sort_values("cos", ascending=False).head(8)[["eval_object_id", "eval_shortest_caption", "cos", "shortest_caption", "in_curated_10k"]].to_string())
