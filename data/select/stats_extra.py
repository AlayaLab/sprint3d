#!/usr/bin/env python
"""Step 9: add the sub-rule counts (and, when the precision estimate exists, its breakdown per rule source) to
<work_dir>/out/selection_stats.json (updated in place)."""
import json, os, pickle
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

cfg = common.config("selection sub-rule statistics")
STATS = os.path.join(common.out_dir(cfg), "selection_stats.json")
st = json.load(open(STATS))
o = pickle.load(open(common.work(cfg, "final_objects.pkl"), "rb"))
sk = o[o.subset == "sketchfab"]
sc = sk[sk.hard_scene]; cx = sk[sk.hard_scene_context]; ex = o[o.hard_extreme_face_count]
st["rule_subcounts"] = {
    "sketchfab_scene_any": {"scene_head_shortest_caption": int(sc.scene_head.sum()), "strong_scene_phrase": int(sc.ph_scene.sum()),
                            "name_or_tag_strong": int(sc.name_tag_scene_strong.sum()), "uploader_scene_rate": int(sc.uploader_scene.sum()),
                            "uploader_only": int((sc.uploader_scene & ~sc.scene_head & ~sc.ph_scene & ~sc.name_tag_scene_strong).sum()),
                            "uploader_only_by_user_top10": sc[sc.uploader_scene & ~sc.scene_head & ~sc.ph_scene & ~sc.name_tag_scene_strong].username.value_counts().head(10).to_dict()},
    "sketchfab_scene_context_any": {"context_phrase": int(cx.ph_context.sum()), "on_in_ground_noun_or_with_surroundings": int(cx.context_noun.sum()),
                                    "comma_list_with_scene_item": int(cx.comma_list_scene.sum())},
    "multi_object_set_any": o[o.hard_multi_object_set].groupby("subset")[["head_multi_hard", "quant_standalone", "ph_multi_start", "ph_copies", "pack_confirmed"]].sum().astype(int).to_dict(),
    "flat_or_text_head_top20": o[o.hard_flat_or_text]["head"].value_counts().head(20).astype(int).to_dict(),
    "extreme_face_count": {s: {"faces_lt_12": int((g.face_count < 12).sum()), "faces_gt_2M": int((g.face_count > 2_000_000).sum()),
                               "glb_gt_200MB": int((g.glb_size.fillna(0) > 200_000_000).sum())} for s, g in ex.groupby("subset")},
}
joined = common.work(cfg, os.path.join("precision", "blind_joined.csv"))
if os.path.exists(joined) and "precision_metadata_blind_llm" in st:
    b = pd.read_csv(joined).merge(
        o[["sha256", "uploader_scene", "scene_head", "ph_scene", "name_tag_scene_strong", "username"]], on="sha256")
    rej = b[b.final_selection == "reject"]
    upo = rej[rej.uploader_scene & ~rej.scene_head & ~rej.ph_scene & ~rej.name_tag_scene_strong]
    cap = rej[rej.scene_head | rej.ph_scene | rej.name_tag_scene_strong]
    st["precision_metadata_blind_llm"]["rejected_breakdown"] = {
        "scene_by_caption_or_name": f"{int((cap.ok == 'no').sum())}/{len(cap)} judged scene/multi/junk",
        "scene_by_uploader_rule_only": f"{int((upo.ok == 'no').sum())}/{len(upo)} judged scene/multi/junk; of the rest {int(upo[upo.ok == 'yes'].note.fillna('').str.contains('building|tower|structure').sum())} are standalone buildings/structures",
        "non_semantic_families": "extreme_face_count / point_cloud(LiDAR-named mesh scans) / world_export reject by design; 0/8 sampled were judged non-single",
        "rejected_judged_single_that_are_buildings": int(rej[(rej.ok == 'yes')].note.fillna('').str.contains('building|tower|structure|temple|vault|ruin').sum()),
    }
    st["precision_metadata_blind_llm"]["method"] = ("100 rejected + 100 kept Sketchfab objects (seeds 11/12 from the draft-final pools, shuffled seed 13); "
        "labelled blind (no flags shown) by a language model from name/tags/categories/3 shortest + longest caption/face_count/animation_count; "
        "counts use the FINAL decision of each sampled object (95 reject / 105 keep). Rubric: standalone buildings count as single_object. Not ground truth; visual QA on pilot renders pending.")
json.dump(st, open(STATS, "w"), indent=1, ensure_ascii=False)
print(json.dumps(st["rule_subcounts"], indent=1)[:3000])
