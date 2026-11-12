#!/usr/bin/env python
"""Step 8: assemble <work_dir>/out/selection_stats.json: funnel per subset and rule, keep shares, set strata and
quotas, overlap with the earlier 5k set, Toys4K overlap counts, and (when present) the precision estimate and the
Toys4K caption-neighbour audit."""
import json, os, pickle
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

cfg = common.config("selection statistics")
STAGE = common.out_dir(cfg)
st = json.load(open(common.work(cfg, "selection_stats.v1.json")))
S = pickle.load(open(common.work(cfg, "sets.pkl"), "rb"))
o = pickle.load(open(common.work(cfg, "final_objects.pkl"), "rb"))
prec_path = common.work(cfg, os.path.join("precision", "precision_summary.json"))
prec = json.load(open(prec_path)) if os.path.exists(prec_path) else None
nn_path = os.path.join(STAGE, "toys4k_caption_neighbours.csv")
nn = pd.read_csv(nn_path) if os.path.exists(nn_path) else None
st["pool_definition"] = "TRELLIS-500K (JeffreyXiang/TRELLIS-500K@288b0611) ObjaverseXL_sketchfab + 3D-FUTURE + HSSD + ABO; ObjaverseXL_github excluded; Toys4k is eval only"
st["fields_used"] = {
    "TRELLIS-500K CSV (all subsets)": ["sha256", "file_identifier", "aesthetic_score (recorded only, no extra threshold)", "captions (all; shortest 3 and full text)"],
    "Sketchfab manifest trellis_sketchfab_objects.csv.gz": ["uid", "sha256 (join key)", "object_path (raw_path)", "face_count", "vertex_count", "glb_size", "license"],
    "Objaverse 1.0 metadata 000-*.json.gz": ["name", "tags[].name", "categories[].name", "animationCount", "user.username", "description (toys4k mention only)"],
    "Objaverse 1.0 lvis-annotations.json.gz": ["joined and recorded (lvis_categories in pickle), not used by rules"],
    "HSSD/ABO GLB header (JSON chunk only)": ["triangle count from index accessors (mode 4)", "vertex count", "file size"],
    "3D-FUTURE model dir": ["raw_model.obj 'f ' line count (polygons)", "raw_model.obj size", "normalized_model.obj existence"],
    "Toys4K": ["TRELLIS Toys4k.csv sha256 + captions", "optional extra Toys4K lists (sha256)", "optional audit list (object_id, category)"],
}
st["face_count_note"] = "Sketchfab: Sketchfab API faceCount (triangles); HSSD/ABO: GLB index accessor triangles (no instancing); 3D-FUTURE: OBJ polygon lines. Bins are used within source only."
st["sets"] = {"seed_string": "trellis500k_curated_v1/selection_v1:", "face_bins": ["f<1k", "f1k-5k", "f5k-20k", "f20k-100k", "f100k-500k", "f500k-2M"],
              "pool_counts_by_source_bin": {s: {b: int(v) for b, v in row.items()} for s, row in S["pool_counts"].iterrows()},
              "quota_by_set_source_bin": S["quota"],
              "nesting": "matched_5243 c curated_10k c large_v1; validation_400 disjoint from all",
              "old_train5k_overlap": {nm: int((o["in_" + nm] & o.in_old_train5k).sum()) for nm in ["validation_400", "matched_5243", "curated_10k", "large_v1"]}}
st["old_train5k_under_v1_rules"] = {s: {"total": int((o.in_old_train5k & (o.subset == s)).sum()), "kept": int((o.in_old_train5k & (o.subset == s) & (o.selection == "keep")).sum()),
                                        "reject_primary_reason": o[o.in_old_train5k & (o.subset == s) & (o.selection == "reject")].primary_reject_reason.value_counts().to_dict()} for s in ["sketchfab", "hssd"]}
st["toys4k_leakage"] = {"sha256_overlap_with_toys4k_lists": int(o.toys4k_sha256.sum()), "name_tag_description_mentions_toys4k": int(o.toys4k_name.sum()),
                        "identical_shortest_and_longest_caption": int(o.toys4k_caption_exact.sum()),
                        "caption_tfidf_cos_ge_0.85_rejected": int((o.toys4k_caption_cos >= 0.85).sum()),
                        "caption_tfidf_cos_0.70_0.85_soft_flag": int(((o.toys4k_caption_cos >= 0.7) & (o.toys4k_caption_cos < 0.85)).sum())}
if nn is not None:
    st["toys4k_leakage"].update({
        "audit_objects_with_trellis_caption": int(nn.groupby("eval_object_id").eval_shortest_caption.first().fillna("").astype(bool).sum()),
        "audit_top1_cos_max": float(nn[nn["rank"] == 1].cos.max()),
        "audit_nn_cos_ge_0.8_in_curated_10k_or_validation": int(((nn.cos >= 0.8) & (nn.in_curated_10k | nn.in_validation_400)).sum())})
if prec is not None:
    st["precision_metadata_blind_llm"] = prec
json.dump(st, open(os.path.join(STAGE, "selection_stats.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: st[k] for k in ["keep_total", "keep_share", "old_train5k_under_v1_rules", "toys4k_leakage"]}, indent=1))
print(st["sets"]["old_train5k_overlap"])
