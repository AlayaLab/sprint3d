#!/usr/bin/env python
"""Step 3: polygon ('f ') and vertex ('v ') line counts of every 3D-FUTURE raw_model.obj (the face counts
of this subset). Writes <work_dir>/future_faces.tsv (uuid, f, v; tab separated)."""
import os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def count(model_dir):
    path = os.path.join(model_dir, "raw_model.obj")
    f = v = 0
    try:
        with open(path, "rb") as handle:
            for line in handle:
                if line.startswith(b"f "):
                    f += 1
                elif line.startswith(b"v "):
                    v += 1
    except OSError:
        return os.path.basename(model_dir), "", ""
    return os.path.basename(model_dir), str(f), str(v)


def main():
    cfg = common.config(__doc__.split("\n")[0])
    root = os.path.join(cfg["inputs"]["raw"]["3D-FUTURE"], "3D-FUTURE-model")
    dirs = [os.path.join(root, name) for name in sorted(os.listdir(root))]
    with Pool(cfg["processes"]) as p:
        rows = p.map(count, dirs, chunksize=64)
    with open(common.work(cfg, "future_faces.tsv"), "w") as out:
        for row in rows:
            out.write("\t".join(row) + "\n")
    print("3D-FUTURE models", len(rows))


if __name__ == "__main__":
    main()
