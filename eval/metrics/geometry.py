"""geometry.py -- single-view 3D geometry: Chamfer / F-score vs ground truth, 32^3 support IoU / precision / recall.

  * Chamfer: non-squared L2, symmetric mean 0.5 * (mean_p d(p, G) + mean_g d(g, P)); 100k surface samples per side
    (trimesh area-weighted sampling, seeded per (object, view)); reported x1e-3 in tables.
  * F@tau: 2PR / (P + R), P = frac(d(p, G) <= tau), R = frac(d(g, P) <= tau); tau = 0.005, 0.01, 0.02.
  * own-bbox frame (keys *_nb, every condition): both meshes normalised by their own bounding box in ground-truth
    world axes (metrics/projection.normalise_own_bbox) -- computed by run_eval.py with the functions below.
  * voxel frame (keys *_vox, true_camera rows only): the voxel frame of the input view (frame.ViewFrame); product
    GLB -> voxel frame by the involutive P2D only.
  * support (true_camera rows only): the product's 32^3 structure support (a decoded dump when the manifest provides
    one) vs the dataset's support32; without a dump a proxy support is derived from the product mesh (dense surface
    samples -> floor((v + 0.5) * 32)) and flagged `support_source = "mesh_proxy"`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .frame import ViewFrame, coords_iou, dense_surface_points, load_mesh_dump, support32_from_points

N_SAMPLES = 100_000
F_THRESHOLDS = (0.005, 0.01, 0.02)
SUPPORT_RES = 32
# salts of the per-(object, view) sampling seeds; part of the reported numbers, do not change
SEED_SALT_3D = "evalv2_3d"
SEED_SALT_POINTS = "evalv2_pc"
SEED_SALT_REFERENCE = "evalv2_3d_vsA"


def stable_seed(*parts) -> int:
    raw = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def load_product_mesh(path) -> trimesh.Trimesh:
    """GLB in the generator's product frame (vertices used as stored)."""
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry()
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"product mesh empty or not a triangle mesh: {path}")
    return m


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=int(seed))
    return np.asarray(pts, np.float64)


def chamfer_fscore(pts_pred: np.ndarray, pts_ref: np.ndarray, thresholds=F_THRESHOLDS) -> dict:
    d_pr, _ = cKDTree(pts_ref).query(pts_pred, workers=1)
    d_rp, _ = cKDTree(pts_pred).query(pts_ref, workers=1)
    out = {"chamfer": 0.5 * (float(d_pr.mean()) + float(d_rp.mean())),
           "mean_d_pred_to_ref": float(d_pr.mean()), "mean_d_ref_to_pred": float(d_rp.mean())}
    for tau in thresholds:
        p = float((d_pr <= tau).mean())
        r = float((d_rp <= tau).mean())
        out[f"f_{tau}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[f"precision_{tau}"] = p
        out[f"recall_{tau}"] = r
    return out


def gt_points_voxel(view_frame: ViewFrame, mesh_dump, n=N_SAMPLES, seed=0, mesh_scale=None):
    V, F = load_mesh_dump(mesh_dump)
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    return view_frame.world_to_voxel(sample_surface(m, n, seed), mesh_scale)


def product_points_voxel(product_glb, n=N_SAMPLES, seed=0):
    m = load_product_mesh(product_glb)
    return ViewFrame.product_to_voxel(sample_surface(m, n, seed)), m


def geometry_vs_gt(product_glb, view_frame: ViewFrame, mesh_dump, *, seed_key, n=N_SAMPLES, mesh_scale=None,
                   gt_pts=None) -> dict:
    """Chamfer / F-score of one product vs ground truth in the voxel frame (true_camera rows).  `gt_pts` may be passed
    to reuse the ground-truth sample across arms of the same (object, view) (paired design)."""
    seed = stable_seed(SEED_SALT_3D, seed_key)
    if gt_pts is None:
        gt_pts = gt_points_voxel(view_frame, mesh_dump, n=n, seed=seed, mesh_scale=mesh_scale)
    pred_pts, mesh = product_points_voxel(product_glb, n=n, seed=seed + 1)
    out = chamfer_fscore(pred_pts, gt_pts)
    out.update({"n_samples": int(n), "product_vertices": int(len(mesh.vertices)), "product_faces": int(len(mesh.faces)),
                "product_abs_max_voxel": float(np.abs(pred_pts).max()), "gt_abs_max_voxel": float(np.abs(gt_pts).max())})
    return out


def geometry_vs_product(product_glb, reference_glb, *, seed_key, n=N_SAMPLES, ref_pts=None) -> dict:
    """Chamfer / F-score between two products of the same input in the shared product frame."""
    seed = stable_seed(SEED_SALT_REFERENCE, seed_key)
    if ref_pts is None:
        ref_pts, _ = product_points_voxel(reference_glb, n=n, seed=seed + 2)
    pred_pts, _ = product_points_voxel(product_glb, n=n, seed=seed + 1)
    return chamfer_fscore(pred_pts, ref_pts)


# ---------------------------------------------------------------------------- support (32^3)
def load_gt_support32(support_npz) -> np.ndarray:
    with np.load(support_npz) as z:
        return np.asarray(z["support32"], np.int64)


def load_dumped_support(npz_path) -> np.ndarray:
    with np.load(npz_path) as z:
        c = np.asarray(z["coords"], np.int64)
    if c.ndim != 2 or c.shape[1] not in (3, 4):
        raise ValueError(f"bad support dump {npz_path}: {c.shape}")
    return np.unique(c[:, -3:], axis=0)


def proxy_support_from_mesh(mesh: trimesh.Trimesh, seed=7, n=400_000) -> np.ndarray:
    pts = dense_surface_points(np.asarray(mesh.vertices), np.asarray(mesh.faces), n=n, seed=seed)
    coords, _ = support32_from_points(ViewFrame.product_to_voxel(pts), SUPPORT_RES)
    return coords


def support_metrics(pred_coords: np.ndarray, gt_coords: np.ndarray) -> dict:
    r = coords_iou(pred_coords, gt_coords, SUPPORT_RES)
    return {"support_iou": r["iou"], "support_precision": r["precision"], "support_recall": r["recall"],
            "n_support_pred": r["n_a"], "n_support_gt": r["n_b"]}


def support_vs_gt(product_glb, support_npz_gt, support_dump=None) -> dict:
    gt = load_gt_support32(support_npz_gt)
    if support_dump is not None and Path(support_dump).is_file():
        pred = load_dumped_support(support_dump)
        src = "decoded_dump"
    else:
        pred = proxy_support_from_mesh(load_product_mesh(product_glb))
        src = "mesh_proxy"
    out = support_metrics(pred, gt)
    out["support_source"] = src
    return out
