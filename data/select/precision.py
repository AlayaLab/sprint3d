#!/usr/bin/env python
"""Optional: metadata-only precision estimate of the rules from a blind labelling of 100 rejected and 100 kept
Sketchfab objects (sampled with seeds 11/12 from the pools before the last rule revisions, shuffled with seed 13;
the labeller saw name, tags, categories, captions, face and animation counts but no flags). The labels are not
part of the release (config.yaml ``inputs.precision_dir``: blind_key.csv, blind_labels.csv). Uses the final
decisions. Writes <work_dir>/precision/{precision_summary.json, blind_joined.csv}."""
import json, math, os, pickle, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

cfg = common.config("precision estimate")
if not cfg["inputs"]["precision_dir"]:
    raise SystemExit("inputs.precision_dir is not set; the precision estimate needs the blind labels")
PREC = common.work(cfg, "precision")
os.makedirs(PREC, exist_ok=True)
key = pd.read_csv(os.path.join(cfg["inputs"]["precision_dir"], "blind_key.csv"))
lab = pd.read_csv(os.path.join(cfg["inputs"]["precision_dir"], "blind_labels.csv"))
o = pickle.load(open(common.work(cfg, "final_objects.pkl"), "rb")).set_index("sha256")
m = key.merge(lab, on="item_id", how="left")
m["final_selection"] = m.sha256.map(o.selection)
m["final_reason"] = m.sha256.map(o.reject_reason)
m["final_soft"] = m.sha256.map(o.soft_flags)
m["shortest_caption"] = m.sha256.map(o.shortest_caption)
m["name"] = m.sha256.map(o["name"])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(p, 3), round(c - h, 3), round(c + h, 3)


res = {"n_labelled": int(lab.item_id.nunique()), "label_counts": lab.label.value_counts().to_dict()}
for grp, name in (("reject", "rejected"), ("keep", "kept")):
    x = m[m.final_selection == grp]
    dec = x[x.ok != "unclear"]
    bad = int((dec.ok == "no").sum()); good = int((dec.ok == "yes").sum())
    res[name] = {"n": int(len(x)), "unclear": int((x.ok == "unclear").sum()), "labels": x.label.value_counts().to_dict(),
                 ("reject_precision_bad_share" if grp == "reject" else "keep_precision_single_share"):
                 wilson(bad, bad + good) if grp == "reject" else wilson(good, bad + good)}
    if grp == "reject":
        fam = x.assign(fam=x.final_reason.str.split("+").str[0]).groupby("fam").apply(lambda g: f"{int((g.ok=='no').sum())}/{int((g.ok!='unclear').sum())}")
        res[name]["bad_share_by_primary_family"] = fam.to_dict()
    else:
        x2 = x.assign(soft=x.final_soft.fillna("").map(lambda s: s if s else "(none)"))
        res[name]["single_share_by_softflag_presence"] = {
            "no_soft_flag": f"{int(((x2.soft=='(none)') & (x2.ok=='yes')).sum())}/{int(((x2.soft=='(none)') & (x2.ok!='unclear')).sum())}",
            "with_soft_flag": f"{int(((x2.soft!='(none)') & (x2.ok=='yes')).sum())}/{int(((x2.soft!='(none)') & (x2.ok!='unclear')).sum())}"}
json.dump(res, open(os.path.join(PREC, "precision_summary.json"), "w"), indent=1, ensure_ascii=False)
m.to_csv(os.path.join(PREC, "blind_joined.csv"), index=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print("## kept but labelled not single:")
print(m[(m.final_selection == "keep") & (m.ok == "no")][["label", "note", "shortest_caption", "name", "final_soft"]].to_string())
print("## rejected but labelled single:")
print(m[(m.final_selection == "reject") & (m.ok == "yes")][["final_reason", "note", "shortest_caption", "name"]].to_string())
