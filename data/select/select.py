#!/usr/bin/env python
"""Step 5: keep/reject decision. HARD rule families reject; SOFT families are only recorded (columns
soft_*) for stricter variants. Input: flags_raw.pkl (flags.py), geom_headers.pkl (geometry_headers.py),
future_faces.tsv (future_face_counts.py, optional). Output: <work_dir>/selection.pkl and
<work_dir>/selection_stats.v1.json (funnel per subset and rule)."""
import json, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import TOK, MULTI_HEAD, FLAT_HEAD  # noqa
import common  # noqa: E402

cfg = common.config("keep/reject decision")
d = pickle.load(open(common.work(cfg, "flags_raw.pkl"), "rb")); o = d["objects"].copy(); bad_users = d["bad_users"]
gh = pickle.load(open(common.work(cfg, "geom_headers.pkl"), "rb"))
glb = gh["glb"].set_index("raw_path"); fut = gh["future"].set_index("raw_path")
ns = o.subset != "sketchfab"; sk = ~ns
o.loc[o.subset.isin(["hssd", "abo"]), "face_count"] = o.loc[o.subset.isin(["hssd", "abo"]), "raw_path"].map(glb["face_count"])
o.loc[o.subset.isin(["hssd", "abo"]), "vertex_count"] = o.loc[o.subset.isin(["hssd", "abo"]), "raw_path"].map(glb["vertex_count"])
o["file_size"] = o["glb_size"]
o.loc[o.subset.isin(["hssd", "abo"]), "file_size"] = o.loc[o.subset.isin(["hssd", "abo"]), "raw_path"].map(glb["file_size"])
o.loc[o.subset == "3dfuture", "file_size"] = o.loc[o.subset == "3dfuture", "raw_path"].map(fut["raw_model_obj_size"])
fut_missing = o.raw_path.map(fut["normalized_model_obj_size"]).isna() & (o.subset == "3dfuture")
ff = common.work(cfg, "future_faces.tsv")
if os.path.exists(ff):
    t = pd.read_csv(ff, sep="\t", header=None, names=["uuid", "f", "v"], dtype={"uuid": str})
    t = t.set_index("uuid")
    uu = o.file_identifier.str.split("/").str[-1]
    m = o.subset == "3dfuture"
    o.loc[m, "face_count"] = uu[m].map(t["f"])  # OBJ polygon faces (raw_model.obj); not triangulated
    o.loc[m, "vertex_count"] = uu[m].map(t["v"])
    print("3D-FUTURE face counts joined", int(o.loc[m, "face_count"].notna().sum()))

short_l = o.shortest_caption.str.lower()
first_word = short_l.map(lambda s: (TOK.findall(s) or [""])[0])
HARD_MULTI_HEAD = MULTI_HEAD - {"kit", "kits", "stack", "stacks", "pile", "piles", "furniture", "series", "elements", "array", "arrays"}
HARD_FLAT_HEAD = FLAT_HEAD - {"sign", "signage", "emblem", "letter", "letters", "relief", "reliefs", "carving", "engraving"}
MULTI_PHRASES = {"collection of", "group of", "assortment of", "multiple objects", "various objects", "assorted objects", "set of various",
                 "various poses", "different poses", "multiple poses", "turnaround", "variations of", "different versions",
                 "various versions", "several versions", "various stages", "different stages"}
SCENE_PHRASES_STRONG = {"diorama", "landscape", "terrain", "cityscape", "floating island", "game level", "low poly scene", "lowpoly scene",
                        "room interior", "interior view", "scene with", "scene featuring", "a scene", "the scene", "an environment",
                        "environment with", "in a room"}
CONTEXT_PHRASES = {"on the ground", "on grass", "on a grassy", "on grassy", "on terrain", "on a hill", "on rocky", "with trees", "with a tree",
                   "with fence", "with a fence", "with road", "with a road", "with a pool", "with pool", "with garden", "with a garden",
                   "with a yard", "with yard", "with background", "and trees", "and a tree", "and grass", "and rocks", "and fence",
                   "surrounded by <surroundings>"}
STANDALONE_PLURALS = set("""chairs tables beds sofas couches stools benches cabinets wardrobes boxes crates baskets bottles jars vases pots
candles candlesticks mugs cups plates bowls books trays towels frames pictures paintings posters lamps plants trees rocks stones
figures figurines statues toys balls apples lemons oranges fruits shoes boots sneakers dolls cars vehicles trucks people persons
characters animals buildings houses barrels chests bags suitcases weapons swords guns knives tools bricks blocks cubes spheres
cylinders pillars columns fences""".split())
from rules import QUANT, ADJ  # noqa
def quant_standalone(caps):
    for c in sorted(caps, key=len)[:3]:
        w = TOK.findall(c.lower().replace("-", " - "))
        w = [x for x in w]
        if len(w) >= 2 and w[0] in QUANT and (w[1] in STANDALONE_PLURALS or (len(w) >= 3 and w[1] in ADJ and w[2] in STANDALONE_PLURALS)):
            if "-" not in c.lower().split()[0]:  # "Three-drawer chest", "Two-seater sofa" are single objects
                return True
    return False
o["quant_standalone"] = o.captions.map(quant_standalone)
hits = o.scene_phrase_hits.fillna("").map(lambda s: set(x for x in s.split(";") if x))
# NOTE: scene_phrase_hits was truncated to 4 hits in 02; the families below use set intersection on those hits.
import re
MULTI_START_RX = re.compile(r"^(?:a |an |the )?(?:small |large |big |diverse |varied |mixed |colorful |assorted )?(?:collection|group|assortment|set|variety|pack|series|array|bundle|lineup|trio|duo|selection|pile) of (?!drawers|shelves|stairs|steps|wheels|legs|doors|shelving|cabinets\b)"
                            r"|^(?:multiple|various|assorted|several|different|many|numerous|mixed) (?:objects|items|things|props|elements|models|assets)\b")
COPIES = {"various poses", "different poses", "multiple poses", "turnaround", "variations of", "different versions", "various versions",
          "several versions", "various stages", "different stages"}
o["ph_multi_start"] = o.captions.map(lambda caps: any(MULTI_START_RX.search(c.lower().strip()) for c in caps))
o["ph_copies"] = hits.map(lambda h: bool(h & COPIES))
o["ph_multi"] = o.ph_multi_start | o.ph_copies
o["set_exception"] = o.shortest_caption.str.lower().str.contains(r"\b(?:tv|television|drum|drums|swing|radio)\s+set\b", regex=True)
o["head_multi_hard"] = o["head"].isin(HARD_MULTI_HEAD) & ~o.set_exception
NS_SCENE_EXCLUDE = {"a scene", "the scene", "scene with", "scene featuring", "an environment", "environment with", "in a room"}  # depictions on furniture
o["ph_scene"] = [bool(h & (SCENE_PHRASES_STRONG if sub == "sketchfab" else SCENE_PHRASES_STRONG - NS_SCENE_EXCLUDE)) for h, sub in zip(hits, o.subset)]
o["ph_context"] = hits.map(lambda h: bool(h & CONTEXT_PHRASES))
o["head_flat_hard"] = o["head"].isin(HARD_FLAT_HEAD) & o.flat_head
tags_l = o.tags.map(lambda t: set(x.lower() for x in t) if isinstance(t, list) else set())
name_t = o.name.map(lambda n: set(TOK.findall(str(n).lower())) if isinstance(n, str) else set())
o["name_tag_scene_strong"] = sk & (tags_l.map(lambda t: bool(t & {"diorama", "dioramas", "landscape", "terrain", "cityscape", "floatingisland", "floating-island", "gamelevel", "leveldesign", "level-design"}))
                                    | name_t.map(lambda t: bool(t & {"diorama", "dioramas", "landscape", "terrain", "cityscape", "battlefield"})))
weak_multi_caption = o.plural_head | o.noun_and | o.comma_list
name_pack_strong = name_t.map(lambda t: bool(t & {"pack", "packs", "bundle", "collection", "megapack", "kitbash", "assetpack", "tileset"}))
o["pack_confirmed"] = sk & (name_pack_strong | o.tag_pack) & weak_multi_caption  # name/tag says pack AND the caption lists several things

HARD = {  # family -> boolean Series
    "toys4k_dup": o.toys4k_sha256 | o.toys4k_name | o.toys4k_caption_exact | (o.toys4k_caption_cos >= 0.85),
    "missing_or_broken_file": fut_missing,
    "no_caption_sketchfab": sk & o.no_caption,
    "scene": (o.scene_head | o.ph_scene | o.name_tag_scene_strong | o.uploader_scene) & (sk | (o.scene_head | o.ph_scene)),
    "scene_context": o.ph_context | (sk & o.context_noun) | (sk & o.comma_list_scene) | (sk & o.building_head & (o.context_noun | o.comma_list_scene)),
    "multi_object_set": o.head_multi_hard | o.quant_standalone | o.ph_multi | o.pack_confirmed,
    "flat_or_text": o.head_flat_hard | o.flat_phrase,
    "point_cloud": sk & o.pointcloud,
    "world_export": sk & o.world_export,
    "extreme_face_count": (o.face_count.fillna(1e9) < 12) & sk | (o.face_count.fillna(0) > 2_000_000) | (sk & (o.glb_size.fillna(0) > 200_000_000)),
}
SOFT = {
    "soft_building": o.building_head | o.building_head_2of3,
    "soft_scene_head_in_3_shortest": o.scene_head_any3 & ~o.scene_head,
    "soft_plural_head": o.plural_head,
    "soft_quantifier_plural_parts": o.quant_plural & ~o.quant_standalone,
    "soft_noun_and": o.noun_and,
    "soft_comma_list": o.comma_list,
    "soft_name_tag_scene_weak": o.name_scene | o.tag_scene,
    "soft_name_tag_pack_unconfirmed": (o.name_pack | o.tag_pack) & ~o.pack_confirmed,
    "soft_animated": o.animated,
    "soft_scan": o.scan_tag & ~o.phone_scan,
    "soft_phone_scan": o.phone_scan,
    "soft_very_low_poly_lt100": (o.face_count < 100) & (o.face_count >= 12),
    "soft_heavy_gt1M": (o.face_count > 1_000_000) & (o.face_count <= 2_000_000),
    "soft_category_architecture_places": o.cat_architecture | o.cat_places,
    "soft_no_caption_curated_source": ns & o.no_caption,
    "soft_toys4k_caption_cos_ge_0.7": (o.toys4k_caption_cos >= 0.7) & (o.toys4k_caption_cos < 0.85),
}
H = pd.DataFrame(HARD).fillna(False).astype(bool); S = pd.DataFrame(SOFT).fillna(False).astype(bool)
order = list(HARD.keys())
o["reject_reason"] = H.apply(lambda r: "+".join(k for k in order if r[k]), axis=1)
o["selection"] = np.where(o.reject_reason == "", "keep", "reject")
o["soft_flags"] = S.apply(lambda r: "+".join(k.replace("soft_", "") for k in S.columns if r[k]), axis=1)
o["primary_reject_reason"] = o.reject_reason.str.split("+").str[0]
for k in order:
    o["hard_" + k] = H[k]
for k in S.columns:
    o[k] = S[k]
# ---------------- stats
stats = {"recipe": "selection rules (data/select)", "per_subset": {}}
for sub in ["sketchfab", "3dfuture", "hssd", "abo"]:
    m = o.subset == sub
    x = {"total": int(m.sum()), "keep": int((m & (o.selection == "keep")).sum()), "reject": int((m & (o.selection == "reject")).sum())}
    x["hard_any_count"] = {k: int((m & H[k]).sum()) for k in order}
    x["hard_exclusive_count"] = {k: int((m & H[k] & (H.sum(axis=1) == 1)).sum()) for k in order}
    rem = m.copy(); funnel = []
    for k in order:
        n0 = int(rem.sum()); rem = rem & ~H[k]
        funnel.append({"rule": k, "removed": n0 - int(rem.sum()), "remaining": int(rem.sum())})
    x["funnel_sequential"] = funnel
    x["soft_count_among_keep"] = {k: int((m & (o.selection == "keep") & S[k]).sum()) for k in S.columns}
    stats["per_subset"][sub] = x
keep = o.selection == "keep"
tot = int(keep.sum())
stats["keep_total"] = tot
stats["keep_share"] = {s: round(int((keep & (o.subset == s)).sum()) / tot, 5) for s in ["sketchfab", "3dfuture", "hssd", "abo"]}
stats["bad_uploaders"] = {u: {"n": int(r["size"]), "scene_rate": round(float(r["mean"]), 3)} for u, r in bad_users.iterrows()}
json.dump(stats, open(common.work(cfg, "selection_stats.v1.json"), "w"), indent=1)
pickle.dump(o, open(common.work(cfg, "selection.pkl"), "wb"))
for sub in ["sketchfab", "3dfuture", "hssd", "abo"]:
    x = stats["per_subset"][sub]
    print(sub, x["total"], "keep", x["keep"], "reject", x["reject"], {f["rule"]: f["removed"] for f in x["funnel_sequential"]})
print("keep_total", tot, stats["keep_share"])
