#!/usr/bin/env python
"""Step 1: join the TRELLIS-500K metadata of every subset with the Sketchfab table and the Objaverse
1.0 per-object metadata. Reads only; writes <work_dir>/joined.pkl.

Inputs (config.yaml, ``inputs``):
  trellis_metadata/{ObjaverseXL_sketchfab,ABO,HSSD,3D-FUTURE,Toys4k}.csv
      sha256, file_identifier, aesthetic_score, captions (JSON list)
  sketchfab_manifest
      uid, file_identifier, sha256, object_path, face_count, vertex_count, glb_size, archive_glb_size, license
  objaverse_metadata/metadata/000-*.json.gz   (Sketchfab API records, per uid)
      name, tags[].name, categories[].name, description, animationCount, isAgeRestricted, likeCount, viewCount,
      staffpickedAt, publishedAt, user.username, archives.glb.{textureCount,textureMaxResolution,faceCount,vertexCount}
  objaverse_metadata/lvis-annotations.json.gz   (LVIS category -> uids; joined and kept, not used by the rules)
"""
import csv, glob, gzip, json, os, sys, time, pickle
from multiprocessing import Pool
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

csv.field_size_limit(sys.maxsize)
SUBSET_NAME = {"ObjaverseXL_sketchfab": "sketchfab", "HSSD": "hssd", "ABO": "abo", "3D-FUTURE": "3dfuture", "Toys4k": "toys4k"}


def read_trellis(meta, fname):
    rows = []
    with open(os.path.join(meta, fname + ".csv"), newline="") as f:
        for r in csv.DictReader(f):
            try:
                caps = json.loads(r["captions"]) if r["captions"] else []
            except json.JSONDecodeError:
                caps = []
            caps = [c for c in caps if isinstance(c, str) and c.strip()]
            rows.append({"subset": SUBSET_NAME[fname], "sha256": r["sha256"], "file_identifier": r["file_identifier"],
                         "aesthetic_score": float(r["aesthetic_score"]), "captions": caps})
    return pd.DataFrame(rows)


def load_shard(path):
    d = json.load(gzip.open(path))
    out = {}
    for uid, v in d.items():
        glb = (v.get("archives") or {}).get("glb") or {}
        out[uid] = {
            "name": v.get("name") or "",
            "tags": [t.get("name", "") for t in (v.get("tags") or [])],
            "categories": [c.get("name", "") for c in (v.get("categories") or [])],
            "description": (v.get("description") or "")[:2000],
            "animation_count": v.get("animationCount"),
            "is_age_restricted": v.get("isAgeRestricted"),
            "like_count": v.get("likeCount"), "view_count": v.get("viewCount"),
            "staffpicked": v.get("staffpickedAt") is not None,
            "published_at": v.get("publishedAt"),
            "username": ((v.get("user") or {}).get("username") or ""),
            "meta_face_count": v.get("faceCount"), "meta_vertex_count": v.get("vertexCount"),
            "glb_texture_count": glb.get("textureCount"), "glb_texture_max_res": glb.get("textureMaxResolution"),
            "glb_face_count": glb.get("faceCount"), "glb_vertex_count": glb.get("vertexCount"),
            "meta_license": v.get("license"),
        }
    return out


def raw_prefix(path):
    return path.rstrip("/") + "/"


def main():
    cfg = common.config(__doc__.split("\n")[0])
    inp = cfg["inputs"]
    t0 = time.time()
    subsets = {}
    for fname in ("ObjaverseXL_sketchfab", "HSSD", "ABO", "3D-FUTURE", "Toys4k"):
        subsets[fname] = read_trellis(inp["trellis_metadata"], fname)
        print(fname, len(subsets[fname]), f"{time.time()-t0:.0f}s", flush=True)
    man = pd.read_csv(inp["sketchfab_manifest"], dtype={"uid": str, "hf_blob_id": str, "license": str})
    print("manifest", len(man), f"{time.time()-t0:.0f}s", flush=True)
    sk = subsets["ObjaverseXL_sketchfab"].merge(
        man[["uid", "sha256", "object_path", "face_count", "vertex_count", "glb_size", "archive_glb_size", "license"]],
        on="sha256", how="left", validate="1:1")
    assert sk["uid"].notna().all()
    shards = sorted(glob.glob(os.path.join(inp["objaverse_metadata"], "metadata", "000-*.json.gz")))
    need = set(sk["uid"])
    meta = {}
    with Pool(cfg["processes"]) as p:
        for d in p.imap_unordered(load_shard, shards):
            for k, v in d.items():
                if k in need:
                    meta[k] = v
    print("objaverse meta", len(meta), "of", len(need), f"{time.time()-t0:.0f}s", flush=True)
    lvis = json.load(gzip.open(os.path.join(inp["objaverse_metadata"], "lvis-annotations.json.gz")))
    uid2lvis = {}
    for cat, uids in lvis.items():
        for u in uids:
            uid2lvis.setdefault(u, []).append(cat)
    mdf = pd.DataFrame.from_dict(meta, orient="index")
    mdf.index.name = "uid"
    sk = sk.merge(mdf.reset_index(), on="uid", how="left")
    sk["lvis_categories"] = sk["uid"].map(lambda u: uid2lvis.get(u, []))
    sk["raw_path"] = raw_prefix(inp["raw"]["ObjaverseXL_sketchfab"]) + sk["object_path"]
    frames = [sk]
    for fname in ("HSSD", "ABO", "3D-FUTURE"):
        df = subsets[fname].copy()
        df["raw_path"] = raw_prefix(inp["raw"][fname]) + df["file_identifier"]
        df["license"] = {"HSSD": "CC-BY-NC-4.0", "ABO": "CC-BY-4.0", "3D-FUTURE": "3D-FUTURE-research-only"}[fname]
        frames.append(df)
    allobj = pd.concat(frames, ignore_index=True)
    toys = subsets["Toys4k"]
    pickle.dump({"objects": allobj, "toys4k": toys}, open(common.work(cfg, "joined.pkl"), "wb"))
    print("done", allobj.groupby("subset").size().to_dict(), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
