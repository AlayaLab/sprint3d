"""synthetic.py -- a tiny ground-truth dataset in the data-toolkit layout and products that reproduce it exactly.

Object: an icosphere stretched along z in the world frame (Z-up), one view looking at it from azimuth 30 deg,
elevation 20 deg, distance 2.5, FoV 40 deg.  The conditioning image is a 64x64 RGBA PNG whose alpha covers the whole
frame except one corner pixel, so the official crop is centred and slightly larger than the image (ndc sx = 1.0625).

  official product   : vertices chosen so that the product's own inference-time projection (camera_angle_x, distance,
                       mesh_scale, official crop) reproduces the ground-truth camera image exactly
  true_camera product: vertices = ViewFrame.world_to_product(V_world) (ground-truth total scale)
"""
from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

SHA = "ab" * 32
VIEW = 0
FOV = math.radians(40.0)


def world_mesh():
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    V = np.asarray(m.vertices, np.float64) * np.array([1.0, 0.8, 1.2]) + np.array([0.02, -0.03, 0.01])
    return V, np.asarray(m.faces, np.int64)


def camera_c2w():
    from metrics.render import lookat_c2w
    return lookat_c2w(30.0, 20.0, 2.5, up_axis="z")


def official_camera_and_product(V_world, c2w, sx=68 / 64):
    """camera {camera_angle_x, distance, mesh_scale} and product vertices whose own projection equals the GT image."""
    from metrics.frame import P2D
    R, t = c2w[:3, :3], c2w[:3, 3]
    v_cam = (V_world - t) @ R
    d, m = 1.7, 1.0
    fov_own = 2.0 * math.atan(sx * math.tan(FOV / 2.0))
    v_prod = ((v_cam + np.array([0.0, 0.0, d])) * m) @ P2D
    return {"camera_angle_x": fov_own, "distance": d, "mesh_scale": m}, v_prod


def build_dataset(root: Path) -> dict:
    import trimesh
    from PIL import Image
    from metrics.frame import ViewFrame
    root = Path(root)
    base = root / "gt" / "SRC"
    rd = base / "renders_cond" / SHA
    rd.mkdir(parents=True)
    (base / "mesh_dumps").mkdir()
    sd = base / "ss_support_view_512" / SHA
    sd.mkdir(parents=True)
    V, F = world_mesh()
    with open(base / "mesh_dumps" / f"{SHA}.pickle", "wb") as fh:
        pickle.dump({"objects": [{"vertices": V.astype(np.float32), "faces": F}]}, fh)
    c2w = camera_c2w()
    (rd / "transforms.json").write_text(json.dumps({"frames": [{"file_path": "000.png", "camera_angle_x": FOV,
                                                               "transform_matrix": c2w.tolist()}]}))
    a = np.full((64, 64, 4), 200, np.uint8)
    a[..., 3] = 255
    a[0, 0, 3] = 0
    Image.fromarray(a, "RGBA").save(rd / "000.png")
    vf = ViewFrame.from_files(rd / "transforms.json", VIEW, mesh_vertices=V, sha256=SHA)
    (sd / "view00_scale.json").write_text(json.dumps({"sphere_radius": vf.sphere_radius, "box_scale": vf.box_scale,
                                                      "total_scale": vf.total_scale}))
    from metrics.frame import dense_surface_points, support32_from_points
    sup, _ = support32_from_points(vf.world_to_voxel(dense_surface_points(V, F, n=50_000, seed=0)))
    np.savez(sd / "view00.npz", support32=sup)
    lst = root / "list.csv"
    with open(lst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sha256", "source", "licence", "view"])
        w.writerow([SHA[:12], SHA, "src", "test", VIEW])
    cam, v_off = official_camera_and_product(V, c2w)
    prod = root / "products"
    prod.mkdir()
    trimesh.Trimesh(v_off, F, process=False).export(prod / "official.glb")
    trimesh.Trimesh(vf.world_to_product(V), F, process=False).export(prod / "true_camera.glb")
    return {"root": root, "gt_root": root / "gt", "list": lst, "camera": cam, "V": V, "F": F, "c2w": c2w,
            "official_glb": prod / "official.glb", "true_camera_glb": prod / "true_camera.glb", "view_frame": vf}
