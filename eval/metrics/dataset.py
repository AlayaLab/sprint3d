"""dataset.py -- ground-truth layout of the evaluation set (Pixal3D / TRELLIS data-toolkit layout).

  <root>/<source dir>/renders_cond/<sha256>/<view:03d>.png     conditioning render (RGBA) = the model input
  <root>/<source dir>/renders_cond/<sha256>/transforms.json     camera of every view (camera_angle_x, transform_matrix)
  <root>/<source dir>/mesh_dumps/<sha256>.pickle                ground-truth mesh in the render's world frame
  <root>/<source dir>/ss_support_view_512/<sha256>/view<view:02d>_scale.json, view<view:02d>.npz
                                                                voxel-frame scale and 32^3 support of the view
                                                                (needed by `true_camera` rows only)

The object list (CSV: id, sha256, source, licence, view) names the source of every object; `source_dirs` maps a source
to its sub-directory.  Without a list, <root>/metadata.csv (sha256[, object_id, category]) describes a flat root.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ViewRecord:
    sha256: str
    view: int
    object_id: str
    category: str
    image_png: Path
    transforms_json: Path
    mesh_dump: Path
    support_npz: Path
    scale_json: Path


class GTDataset:
    def __init__(self, root, list_csv=None, source_dirs=None):
        self.root = Path(root)
        self.rows = {}
        if list_csv is not None:
            sd = source_dirs or {}
            for r in csv.DictReader(open(list_csv)):
                sha = r["sha256"]
                src = r.get("source", "")
                d = self.root / sd.get(src, src) if src else self.root
                self.rows[sha] = {"sha256": sha, "object_id": r.get("id") or sha[:12], "category": src, "dir": d,
                                  "view": int(r["view"]) if r.get("view") not in (None, "") else None}
        elif (self.root / "metadata.csv").is_file():
            for r in csv.DictReader(open(self.root / "metadata.csv")):
                sha = r["sha256"]
                self.rows[sha] = {"sha256": sha, "object_id": r.get("object_id") or sha[:12],
                                  "category": r.get("category") or r.get("subset") or "", "dir": self.root, "view": None}
        else:
            raise FileNotFoundError(f"{self.root}/metadata.csv missing and no object list given")
        self._by_id = {v["object_id"]: k for k, v in self.rows.items()}

    # ---- lookups ----------------------------------------------------------------------------
    def shas(self):
        return sorted(self.rows)

    def has(self, sha):
        return sha in self.rows

    def resolve(self, key: str) -> str | None:
        """sha256, list id or sha256 prefix -> sha256"""
        if key in self.rows:
            return key
        if key in self._by_id:
            return self._by_id[key]
        hits = [s for s in self.rows if s.startswith(key)] if len(key) >= 8 else []
        return hits[0] if len(hits) == 1 else None

    def object_id(self, sha):
        return self.rows[sha]["object_id"]

    def category(self, sha):
        return self.rows[sha]["category"]

    def listed_view(self, sha):
        return self.rows[sha]["view"]

    def base(self, sha) -> Path:
        return self.rows[sha]["dir"]

    def renders_dir(self, sha) -> Path:
        return self.base(sha) / "renders_cond" / sha

    def image_path(self, sha, view) -> Path:
        return self.renders_dir(sha) / f"{int(view):03d}.png"

    def transforms(self, sha) -> Path:
        return self.renders_dir(sha) / "transforms.json"

    def transforms_payload(self, sha) -> dict:
        return json.loads(self.transforms(sha).read_text())

    def mesh_dump(self, sha) -> Path:
        return self.base(sha) / "mesh_dumps" / f"{sha}.pickle"

    def support_npz(self, sha, view) -> Path:
        return self.base(sha) / "ss_support_view_512" / sha / f"view{int(view):02d}.npz"

    def scale_json(self, sha, view) -> Path:
        return self.base(sha) / "ss_support_view_512" / sha / f"view{int(view):02d}_scale.json"

    def record(self, sha, view) -> ViewRecord:
        return ViewRecord(sha256=sha, view=int(view), object_id=self.object_id(sha), category=self.category(sha),
                          image_png=self.image_path(sha, view), transforms_json=self.transforms(sha),
                          mesh_dump=self.mesh_dump(sha), support_npz=self.support_npz(sha, view),
                          scale_json=self.scale_json(sha, view))

    def missing(self, sha, view, with_support=False) -> list:
        """missing ground-truth files of a view; the support files are needed by `true_camera` rows only."""
        r = self.record(sha, view)
        need = [r.image_png, r.transforms_json, r.mesh_dump] + ([r.support_npz, r.scale_json] if with_support else [])
        return [str(p) for p in need if not Path(p).is_file()]
