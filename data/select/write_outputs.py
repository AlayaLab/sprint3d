#!/usr/bin/env python
"""Step 7: write candidates_<subset>.csv (every object with its decision, rule flags and ranks) and the set
lists validation_400.csv, matched_5243.csv, curated_10k.csv, large_v1.csv into <work_dir>/out/, plus
<work_dir>/final_objects.pkl. The released data/lists/*.csv keep only id, sha256, source and licence of these
lists. The overlap with the earlier 5k set is matched by Objaverse uid (Sketchfab) and HSSD id."""
import json, os, pickle
import sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

cfg = common.config("write candidates and set lists")
STAGE = common.out_dir(cfg)
S = pickle.load(open(common.work(cfg, "sets.pkl"), "rb")); o = S["objects"]
# ---------------- old train5k overlap
man = pd.read_csv(cfg["inputs"]["earlier_5k_list"], dtype=str)
old_sk = set(man.id[man.source == "sketchfab"])
old_hssd = set(man.id[man.source == "hssd"])
key = np.where(o.subset == "sketchfab", o.uid.astype(str), o.file_identifier.str.split("/").str[-1].str.replace(".glb", "", regex=False))
o["in_old_train5k"] = [(s == "sketchfab" and k in old_sk) or (s == "hssd" and k in old_hssd) for s, k in zip(o.subset, key)]
# ---------------- candidates
hard_cols = [c for c in o.columns if c.startswith("hard_")]
soft_cols = [c for c in o.columns if c.startswith("soft_") and c != "soft_flags"]
raw_cols = ["no_caption", "scene_head", "scene_head_any3", "building_head", "building_head_2of3", "flat_head", "multi_head", "quant_plural",
            "quant_standalone", "plural_head", "noun_and", "comma_list", "comma_list_scene", "context_noun", "ph_scene", "ph_context",
            "ph_multi_start", "ph_copies", "name_scene", "tag_scene", "name_tag_scene_strong", "name_pack", "tag_pack", "pack_confirmed",
            "pointcloud", "scan_tag", "phone_scan", "world_export", "uploader_scene", "animated", "cat_architecture", "cat_places",
            "toys4k_sha256", "toys4k_name", "toys4k_caption_exact"]
base_cols = ["subset", "sha256", "file_identifier", "raw_path", "uid", "name", "username", "shortest_caption", "n_captions", "head",
             "license", "aesthetic_score", "face_count", "vertex_count", "file_size", "animation_count", "categories",
             "toys4k_caption_cos", "toys4k_caption_nn", "scene_phrase_hits"]
dec_cols = ["selection", "reject_reason", "primary_reject_reason", "soft_flags", "face_bin", "cell_rank", "reserve_rank_in_cell",
            "in_validation_400", "in_matched_5243", "in_curated_10k", "in_large_v1", "in_old_train5k"]
C = o[base_cols + dec_cols + hard_cols + soft_cols + raw_cols].copy()
C["categories"] = C.categories.map(lambda c: "|".join(c) if isinstance(c, list) else "")
for c in hard_cols + soft_cols + raw_cols + ["in_validation_400", "in_matched_5243", "in_curated_10k", "in_large_v1", "in_old_train5k"]:
    C[c] = C[c].fillna(False).astype(bool).astype(int)
for c in ["face_count", "vertex_count", "file_size", "animation_count", "cell_rank", "reserve_rank_in_cell", "n_captions"]:
    C[c] = C[c].map(lambda x: "" if pd.isna(x) else str(int(x)))
C["aesthetic_score"] = C.aesthetic_score.map(lambda x: f"{x:.4f}")
C["toys4k_caption_cos"] = C.toys4k_caption_cos.map(lambda x: f"{x:.3f}")
name_of = {"sketchfab": "ObjaverseXL_sketchfab", "3dfuture": "3D-FUTURE", "hssd": "HSSD", "abo": "ABO"}
for s in ["sketchfab", "3dfuture", "hssd", "abo"]:
    C[C.subset == s].sort_values("sha256").to_csv(os.path.join(STAGE, f"candidates_{name_of[s]}.csv"), index=False)
# ---------------- sets
set_cols = ["subset", "sha256", "file_identifier", "raw_path", "shortest_caption", "license", "aesthetic_score", "face_count", "face_bin",
            "reserve_rank_in_cell", "soft_flags", "uid", "name", "in_old_train5k"]
for nm, extra in [("validation_400", []), ("matched_5243", ["in_curated_10k", "in_large_v1"]), ("curated_10k", ["in_matched_5243", "in_large_v1"]),
                  ("large_v1", ["in_matched_5243", "in_curated_10k"])]:
    x = C[C["in_" + nm] == 1][set_cols + extra].sort_values(["subset", "face_bin", "reserve_rank_in_cell"], key=lambda col: col if col.name != "reserve_rank_in_cell" else col.astype(int))
    x.to_csv(os.path.join(STAGE, f"{nm}.csv"), index=False)
    print(nm, len(x))
# (the caption-neighbour audit of a Toys4K evaluation subset is written by toys4k_neighbours.py)
pickle.dump(o, open(common.work(cfg, "final_objects.pkl"), "wb"))
print("old train5k:", {s: {"total": int((o.in_old_train5k & (o.subset == s)).sum()), "kept": int((o.in_old_train5k & (o.subset == s) & (o.selection == "keep")).sum())} for s in ["sketchfab", "hssd"]})
for nm in ["matched_5243", "curated_10k", "validation_400"]:
    print(nm, "old-train5k overlap", int((o["in_" + nm] & o.in_old_train5k).sum()))
