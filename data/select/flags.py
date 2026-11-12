#!/usr/bin/env python
"""Step 4: per-object rule flags (captions, Sketchfab name/tags/categories/uploader, Toys4K overlap).
Input: <work_dir>/joined.pkl (join_metadata.py). Output: <work_dir>/flags_raw.pkl. The keep/reject
decision is made by select.py.

Toys4K overlap (all four signals are recorded; select.py rejects an object if any of them fires):
  toys4k_sha256         the asset sha256 equals that of a Toys4K object (Toys4k.csv of TRELLIS-500K and
                        the optional extra lists of config.yaml)
  toys4k_name           the regular expression ``toys?\\s?4k`` matches the lower-cased Sketchfab name, tags
                        or description ("toy4k", "toys4k", "toys 4k", ...)
  toys4k_caption_exact  the (shortest, longest) caption pair equals that of a Toys4K object
  toys4k_caption_cos    highest TF-IDF cosine similarity between the object's captions and the captions of
                        a Toys4K object (rejected at >= 0.85)
"""
import json, os, pickle, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules import *  # noqa
import common  # noqa: E402

HEAD_EXCEPT = {("kitchen", "island"), ("charging", "station"), ("docking", "station"), ("work", "station"),
               ("weather", "station"), ("soldering", "station"), ("crafting", "station"), ("computer", "tower"),
               ("pc", "tower"), ("dog", "house"), ("bird", "house"), ("doll", "house"), ("toy", "house"), ("cat", "house"),
               ("jenga", "tower"), ("cat", "tower"), ("speaker", "tower"), ("fan", "tower"), ("mouse", "pad"),
               ("play", "mat"), ("yoga", "mat"), ("bath", "mat"), ("gaming", "mat"), ("mouse", "mat"), ("floor", "lamp"),
               ("wall", "clock"), ("wall", "lamp"), ("wall", "shelf"), ("tile", "stove"), ("playing", "cards"),
               ("tarot", "card"), ("sd", "card"), ("graphics", "card"), ("video", "card"), ("credit", "card"),
               ("sim", "card"), ("memory", "card"), ("coffee", "table"), ("picture", "frame"), ("photo", "frame"),
               ("stone", "wall")}
NONSKETCH_FLAT = set("poster posters painting paintings picture pictures photo photos photograph canvas rug rugs carpet carpets mat mats doormat tapestry wallpaper artwork mural sticker decal logo banner plaque".split())
STRONG_PHRASES = ["diorama", "landscape", "terrain", "cityscape", "floating island", "multiple objects", "various objects",
                  "assorted objects", "collection of", "group of", "assortment of", "variety of", "multiple views",
                  "various poses", "different poses", "multiple poses", "turnaround", "point cloud", "room interior",
                  "interior view", "a scene", "the scene", "scene with", "scene featuring"]


PHRASE_RX_SK = [(p, re.compile(r"(?<![a-z-])" + re.escape(p) + r"(?![a-z-])")) for p in SCENE_PHRASES]
PHRASE_RX_NS = [(p, re.compile(r"(?<![a-z-])" + re.escape(p) + r"(?![a-z-])")) for p in STRONG_PHRASES]
SURROUND_RX = re.compile(r"surrounded by (?:[a-z-]+ ){0,3}(?:" + "|".join(sorted(SURROUND_NOUN | CONTEXT_NOUN | {"water", "buildings", "walls", "plants", "flowers"})) + r")\b")


TOYS4K_NAME = re.compile(r"toys?\s?4k")


def mentions_toys4k(name, tags, description):
    """Name rule of the Toys4K deduplication: the expression matches the lower-cased name, tags and description."""
    text = " ".join([str(name)] + [str(t) for t in (tags if isinstance(tags, list) else [])] + [str(description)]).lower()
    return bool(TOYS4K_NAME.search(text))


def head_ctx(caption):
    """head analysis plus the word before the head (for compound exceptions)."""
    head, first, noun_and, n_commas = head_analysis(caption)
    words = TOK.findall(caption.lower())
    prev = ""
    if head in words:
        i = words.index(head)
        prev = words[i - 1] if i > 0 else ""
    return head, prev, first, noun_and, n_commas


def cap_flags(caps, subset):
    f = {}
    if not caps:
        f["no_caption"] = True
        return f, "", ""
    srt = sorted(caps, key=len)
    short = srt[0]
    h, prev, first, noun_and, ncom = head_ctx(short)
    heads3 = [head_ctx(c) for c in srt[:3]]
    exc = lambda hh, pp: (pp, hh) in HEAD_EXCEPT
    f["head"] = h
    f["scene_head"] = (h in SCENE_HEAD and not exc(h, prev))
    f["scene_head_any3"] = any(x[0] in SCENE_HEAD and not exc(x[0], x[1]) for x in heads3)
    f["building_head"] = (h in BUILDING_HEAD and not exc(h, prev))
    f["building_head_2of3"] = sum(x[0] in BUILDING_HEAD and not exc(x[0], x[1]) for x in heads3) >= 2
    f["flat_head"] = (h in FLAT_HEAD and not exc(h, prev))
    f["multi_head"] = (h in MULTI_HEAD) or (first in QUANT and len(TOK.findall(short)) > 1 and is_plural(TOK.findall(short.lower())[1] if len(TOK.findall(short)) > 1 else ""))
    # quantifier followed by a plural noun anywhere in the 3 shortest captions ("Two chairs.", "Set of three vases.")
    q = False
    for c in srt[:3]:  # quantifier at the START of one of the 3 shortest captions ("Two chairs.", "Various fruits.")
        w = TOK.findall(c.lower())
        if len(w) >= 2 and w[0] in QUANT and (is_plural(w[1]) or (len(w) >= 3 and is_plural(w[2]))):
            q = True
    f["quant_plural"] = q
    # comma list whose items include scene/building/surroundings nouns ("House, pool, garden, deck.")
    items = [TOK.findall(x.replace("-", " ")) for x in re.split(r",| and ", short.lower())]
    item_heads = [w[-1] for w in items if w]
    f["comma_list_scene"] = (ncom >= 1) and any((h2 in SCENE_HEAD or h2 in BUILDING_HEAD or h2 in SURROUND_NOUN or h2 in CONTEXT_NOUN) and h2 not in ("interior", "interiors", "background") for h2 in item_heads[1:])
    f["plural_head"] = is_plural(h) and not exc(h, prev)
    f["noun_and"] = noun_and
    f["comma_list"] = ncom >= 2
    low3 = " . ".join(c.lower() for c in srt[:3])
    lowall = " . ".join(c.lower() for c in caps)
    phrases = SCENE_PHRASES if subset == "sketchfab" else STRONG_PHRASES
    hit = [p for p, rx in (PHRASE_RX_SK if subset == "sketchfab" else PHRASE_RX_NS) if rx.search(lowall)]
    if SURROUND_RX.search(low3):
        hit.append("surrounded by <surroundings>")
    f["scene_phrase"] = bool(hit)
    f["scene_phrase_hits"] = ";".join(hit[:4])
    ctx = False
    for c in srt[:3]:
        w = TOK.findall(c.lower())
        ON = ("on", "in", "among", "atop", "amid", "inside")
        WITH = ("with", "and", "near", "beside")
        MOD = ("a", "an", "the", "green", "grassy", "rocky", "small", "large", "snowy", "sandy", "wooden", "stone", "tall", "lush", "some")
        for a, b in zip(w, w[1:]):
            if (a in ON and b in CONTEXT_NOUN) or (a in WITH and b in SURROUND_NOUN):
                ctx = True
        for a, b, cc in zip(w, w[1:], w[2:]):
            if b in MOD and ((a in ON and cc in CONTEXT_NOUN) or (a in WITH and cc in SURROUND_NOUN)):
                ctx = True
    f["context_noun"] = ctx
    f["flat_phrase"] = bool(re.search(r"\b2d\b|\bflat image\b|\bflat plane\b|\btext reading\b|\bthe words?\b|\bletters spelling\b|\bflat illustration\b|\bflat cutout\b|\bflat graphic\b", low3))
    return f, short, srt[-1]


def main():
    cfg = common.config("per-object rule flags")
    t0 = time.time()
    d = pickle.load(open(common.work(cfg, "joined.pkl"), "rb"))
    o, toys = d["objects"].copy(), d["toys4k"]
    # ---------------- caption rules
    recs, shorts, longs = [], [], []
    for caps, sub in zip(o.captions, o.subset):
        f, s, l = cap_flags(caps, sub)
        recs.append(f); shorts.append(s); longs.append(l)
    F = pd.DataFrame(recs).infer_objects().fillna(False)
    for c in F.columns:
        if c not in ("head", "scene_phrase_hits"):
            F[c] = F[c].astype(bool)
    F["head"] = F["head"].replace({False: ""})
    F["scene_phrase_hits"] = F["scene_phrase_hits"].replace({False: ""})
    o = pd.concat([o.reset_index(drop=True), F], axis=1)
    o["shortest_caption"] = shorts
    o["longest_caption"] = longs
    o["n_captions"] = o.captions.map(len)
    print("caption rules", f"{time.time()-t0:.0f}s", flush=True)
    # ---------------- Sketchfab name/tag/category/user rules
    sk = o.subset == "sketchfab"
    def ntoks(row_name, row_tags):
        nt = set(TOK.findall((row_name or "").lower()))
        tt = set(t.lower() for t in (row_tags or [])) | set(w for t in (row_tags or []) for w in TOK.findall(t.lower()))
        return nt, tt
    nts = [ntoks(n, t) if s == "sketchfab" else (set(), set()) for n, t, s in zip(o.name, o.tags, o.subset)]
    o["name_scene"] = [bool(nt & NAME_TAG_SCENE) for nt, tt in nts]
    o["tag_scene"] = [bool(set(t.lower() for t in (tags if isinstance(tags, list) else [])) & TAG_SCENE_STRONG) for tags in o.tags]
    o["name_pack"] = [bool(nt & NAME_TAG_PACK) for nt, tt in nts]
    o["tag_pack"] = [bool(set(t.lower() for t in (tags if isinstance(tags, list) else [])) & TAG_PACK_STRONG) for tags in o.tags]
    o["pointcloud"] = pd.Series([bool((nt | tt) & POINTCLOUD_WORDS) for nt, tt in nts], index=o.index) | o.longest_caption.str.lower().str.contains("point cloud")
    o["scan_tag"] = [bool(tt & SCAN_TAGS) or bool(nt & {"scan", "photogrammetry", "3dscan", "scanned"}) for nt, tt in nts]
    o["phone_scan"] = [bool(tt & PHONE_SCAN) or bool(nt & {"polycam", "scaniverse", "kiri", "trnio", "qlone"}) for nt, tt in nts]
    o["world_export"] = [bool(tt & WORLD_EXPORT) for nt, tt in nts]
    o["animated"] = sk & (o.animation_count.fillna(0) > 0)
    o["low_poly"] = sk & (o.face_count.fillna(0) < 100)
    o["huge"] = sk & ((o.face_count.fillna(0) > 2_000_000) | (o.glb_size.fillna(0) > 200_000_000))
    cat = o.categories.map(lambda c: c if isinstance(c, list) else [])
    o["cat_architecture"] = cat.map(lambda c: "architecture" in c)
    o["cat_places"] = cat.map(lambda c: "places-travel" in c)
    # uploader-level scene rate (Sketchfab): users with >=20 objects of which >=50% carry a scene/building/pack signal
    sceneish = sk & (o.scene_head_any3 | o.building_head | o.scene_phrase | o.name_scene | o.tag_scene | o.context_noun | o.world_export)
    g = pd.DataFrame({"u": o.username, "s": sceneish})[sk].groupby("u").s.agg(["size", "mean"])
    bad_users = g[(g["size"] >= 20) & (g["mean"] >= 0.5)].sort_values("size", ascending=False)
    o["uploader_scene"] = sk & o.username.isin(set(bad_users.index))
    print("bad uploaders", len(bad_users), int(bad_users["size"].sum()), flush=True)
    # ---------------- Toys4K leakage
    toy_sha = set(toys.sha256)
    for p in cfg["inputs"]["extra_toys4k_lists"]:
        toy_sha |= set(pd.read_csv(p).sha256)
    o["toys4k_sha256"] = o.sha256.isin(toy_sha)
    o["toys4k_name"] = [mentions_toys4k(n, tags, ds) for n, tags, ds in zip(o.name, o.tags, o.description)]
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(c) for c in o.captions]
    tdocs = [" ".join(json.loads(c) if isinstance(c, str) else c) for c in toys.captions]
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words="english", dtype=np.float32)
    X = vec.fit_transform(docs + tdocs)
    Xo, Xt = X[: len(docs)], X[len(docs):]
    has_t = np.asarray(Xt.sum(1)).ravel() > 0
    Xt = Xt[has_t]
    tnames = toys.file_identifier.values[has_t]
    best = np.zeros(len(docs), np.float32); arg = np.zeros(len(docs), np.int64)
    for i in range(0, len(docs), 20000):
        S = (Xo[i:i + 20000] @ Xt.T).toarray()
        best[i:i + 20000] = S.max(1); arg[i:i + 20000] = S.argmax(1)
    o["toys4k_caption_cos"] = best
    o["toys4k_caption_nn"] = tnames[arg]
    # exact-caption duplicates: identical shortest AND longest caption with any Toys4K object
    tset = set()
    for c in toys.captions:
        cc = json.loads(c) if isinstance(c, str) else c
        if cc:
            tset.add((min(cc, key=len).strip().lower(), max(cc, key=len).strip().lower()))
    o["toys4k_caption_exact"] = [(s.strip().lower(), l.strip().lower()) in tset if s else False for s, l in zip(o.shortest_caption, o.longest_caption)]
    print("toys4k", f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump({"objects": o, "bad_users": bad_users}, open(common.work(cfg, "flags_raw.pkl"), "wb"))
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
