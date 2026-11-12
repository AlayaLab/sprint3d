#!/usr/bin/env python
"""Step 2: cheap geometry statistics of the HSSD, ABO and 3D-FUTURE assets without loading meshes.
HSSD/ABO GLB: only the 12-byte header and the JSON chunk are read; triangles are counted from the index
accessors of triangle primitives (mode 4), vertices from the POSITION accessors (instancing is not
expanded). 3D-FUTURE: sizes of raw_model.obj / normalized_model.obj / texture.png (no parsing).
Writes <work_dir>/geom_headers.pkl."""
import json, os, pickle, struct, sys
from multiprocessing import Pool
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def glb_stats(path):
    out = {"raw_path": path, "exists": False}
    try:
        st = os.stat(path)
        out["exists"] = True
        out["file_size"] = st.st_size
        with open(path, "rb") as f:
            magic, ver, total = struct.unpack("<III", f.read(12))
            if magic != 0x46546C67:
                out["error"] = "not_glb"; return out
            clen, ctype = struct.unpack("<II", f.read(8))
            js = json.loads(f.read(clen))
        acc = js.get("accessors", [])
        tri = 0; verts = 0; prims = 0; nonTri = 0
        for m in js.get("meshes", []):
            for p in m.get("primitives", []):
                prims += 1
                mode = p.get("mode", 4)
                pos = p.get("attributes", {}).get("POSITION")
                nv = acc[pos]["count"] if pos is not None and pos < len(acc) else 0
                verts += nv
                if mode == 4:
                    tri += (acc[p["indices"]]["count"] // 3) if "indices" in p else nv // 3
                else:
                    nonTri += 1
        out.update({"face_count": tri, "vertex_count": verts, "n_meshes": len(js.get("meshes", [])),
                    "n_nodes": len(js.get("nodes", [])), "n_prims": prims, "n_nontri_prims": nonTri,
                    "n_animations": len(js.get("animations", [])), "n_skins": len(js.get("skins", [])),
                    "n_images": len(js.get("images", [])), "n_materials": len(js.get("materials", []))})
    except Exception as e:  # noqa
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def future_stats(path):
    out = {"raw_path": path, "exists": os.path.isdir(path)}
    for fn in ("raw_model.obj", "normalized_model.obj", "texture.png", "model.mtl", "image.jpg"):
        p = os.path.join(path, fn)
        try:
            out[fn.replace(".", "_") + "_size"] = os.stat(p).st_size
        except OSError:
            out[fn.replace(".", "_") + "_size"] = None
    return out


def main():
    cfg = common.config(__doc__.split("\n")[0])
    d = pickle.load(open(common.work(cfg, "joined.pkl"), "rb"))["objects"]
    glbs = d[d.subset.isin(["hssd", "abo"])].raw_path.tolist()
    fut = d[d.subset == "3dfuture"].raw_path.tolist()
    with Pool(cfg["processes"]) as p:
        g = p.map(glb_stats, glbs, chunksize=8)
        f = p.map(future_stats, fut, chunksize=8)
    pickle.dump({"glb": pd.DataFrame(g), "future": pd.DataFrame(f)}, open(common.work(cfg, "geom_headers.pkl"), "wb"))
    print("glb", len(g), "missing", sum(not x["exists"] for x in g), "errors", sum("error" in x for x in g))
    print("future", len(f), "missing", sum(not x["exists"] for x in f))


if __name__ == "__main__":
    main()
