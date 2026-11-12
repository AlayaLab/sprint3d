"""frame.py -- coordinate frames of the evaluation (analytic, registration-free).

Frames (row-vector convention throughout: v_out = v_in @ M):

  world   : the normalised mesh dump (mesh_dumps/<sha256>.pickle): bbox-centred, max span 1, the frame in which
            Blender rendered renders_cond/<sha256>/<view>.png with the camera transform_matrix (c2w) of transforms.json.
  camera  : Blender camera frame of that view (x right, y up, -z forward): v_cam = (v_world - t) @ R_c2w.
  voxel   : the data toolkit's view-aligned frame (ss_support_view_512):
                v_voxel = box_scale * transform_mesh(v_world / sphere_radius)
                        = (v_world @ R_view^T) * total_scale,     total_scale = box_scale / sphere_radius
            with R_view the rotation of the toolkit's transform_mesh (a pure rotation equal to the w2c rotation, i.e.
            voxel == camera frame up to the isotropic total_scale).  support32 = floor((v_voxel + 0.5) * 32).
  product : the exported GLB frame of the generator (involutive diag(-1, 1, -1)): v_voxel = v_product @ P2D.
            A product generated with conditioning mesh_scale = m lives in v = (v_world @ R_view^T) * m.

No per-object fitted parameter anywhere (pose error is generation error, as in the Pixal3D paper).
The voxel-frame scale (scale json) is needed only by `true_camera` rows; without it the scale fields are None.
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

P2D = np.diag([-1.0, 1.0, -1.0])  # v_voxel = v_product @ P2D

_R_INIT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_R_BACK = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_R_PLY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
BOX_TARGET = 0.49999  # data-toolkit abs-max rescale target


def _h(R, t=None):
    T = np.eye(4)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def camera_c2w_new(radius: float, yaw: float = -math.pi / 2, pitch: float = 0.0) -> np.ndarray:
    """numpy port of the data toolkit's get_new_camera_matrix (LookAt at origin, up (0,1,0), +90 deg twist about the
    local z to match Blender's Track-To axes)."""
    eye = np.array([radius * math.cos(yaw) * math.cos(pitch), radius * math.sin(yaw) * math.cos(pitch),
                    radius * math.sin(pitch)])
    f = -eye / np.linalg.norm(eye)
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    Rz90 = np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return T @ Rz90


def toolkit_view_transform(frame: dict) -> np.ndarray:
    """4x4 T_final of the data toolkit's transform_mesh (column-vector form: v' = T @ v)."""
    c2w = np.asarray(frame["transform_matrix"], np.float64)
    radius = float(np.linalg.norm(c2w[:3, 3]))
    T_cam = camera_c2w_new(radius) @ np.linalg.inv(c2w) @ _h(_R_PLY)
    return _h(_R_BACK) @ T_cam @ _h(_R_INIT)


def load_mesh_dump(path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    vs, fs, off = [], [], 0
    for o in d["objects"]:
        v = np.asarray(o["vertices"], np.float64)
        f = np.asarray(o["faces"], np.int64)
        if v.size == 0 or f.size == 0:
            continue
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs, 0), np.concatenate(fs, 0)


def sphere_radius_of(V: np.ndarray) -> float:
    """the data toolkit's sphere normalisation: max distance from the bbox centre (divides only, no recentring;
    asserts the bbox centre is near the origin)."""
    c = (V.min(0) + V.max(0)) / 2
    if np.abs(c).max() >= 0.25:
        raise ValueError(f"mesh dump bbox centre far from origin: {c}")
    return float(np.linalg.norm(V - c, axis=1).max())


@dataclass
class ViewFrame:
    sha256: str
    view: int
    camera_angle_x: float          # horizontal FoV (rad)
    c2w: np.ndarray                # 4x4 Blender camera-to-world
    distance: float                # |c2w[:3,3]|
    R_view: np.ndarray             # 3x3, v_aligned = v_world @ R_view.T (column form: R_view @ v)
    sphere_radius: float | None
    box_scale: float | None
    total_scale: float | None

    # ---- factory -------------------------------------------------------------------------------
    @classmethod
    def from_files(cls, transforms_json, view: int, scale_json=None, mesh_vertices: np.ndarray | None = None,
                   sha256: str = "") -> "ViewFrame":
        payload = json.loads(Path(transforms_json).read_text())
        frame = payload["frames"][view]
        if frame.get("file_path") not in (None, f"{view:03d}.png"):
            raise ValueError(f"frame {view} of {transforms_json} is {frame.get('file_path')}")
        c2w = np.asarray(frame["transform_matrix"], np.float64)
        T = toolkit_view_transform(frame)
        if np.abs(T[:3, 3]).max() > 1e-5:
            raise ValueError(f"view transform is not a pure rotation (t={T[:3, 3]}) for {transforms_json}")
        R_view = T[:3, :3]
        sr = bs = ts = None
        if scale_json is not None:
            s = json.loads(Path(scale_json).read_text())
            sr, bs, ts = float(s["sphere_radius"]), float(s["box_scale"]), float(s["total_scale"])
            if abs(bs / sr - ts) > 1e-6 * max(1.0, ts):
                raise ValueError(f"inconsistent scale json {scale_json}")
        elif mesh_vertices is not None:
            sr = sphere_radius_of(mesh_vertices)
            aligned = (mesh_vertices / sr) @ R_view.T
            bs = BOX_TARGET / float(np.abs(aligned).max())
            ts = bs / sr
        return cls(sha256=sha256, view=view, camera_angle_x=float(frame["camera_angle_x"]), c2w=c2w,
                   distance=float(np.linalg.norm(c2w[:3, 3])), R_view=R_view, sphere_radius=sr,
                   box_scale=bs, total_scale=ts)

    # ---- camera ---------------------------------------------------------------------------------
    @property
    def R_c2w(self) -> np.ndarray:
        return self.c2w[:3, :3]

    @property
    def R_w2c(self) -> np.ndarray:
        return self.c2w[:3, :3].T

    def world_to_camera(self, v: np.ndarray) -> np.ndarray:
        return (np.asarray(v, np.float64) - self.c2w[:3, 3]) @ self.R_c2w

    def camera_normal(self, n_world: np.ndarray) -> np.ndarray:
        return np.asarray(n_world, np.float64) @ self.R_c2w

    # ---- voxel / product ------------------------------------------------------------------------
    def _m(self, mesh_scale):
        m = self.total_scale if mesh_scale is None else float(mesh_scale)
        if m is None:
            raise ValueError("the voxel-frame scale of this view is unknown (no scale json): needed by true_camera rows")
        return m

    def world_to_voxel(self, v, mesh_scale=None) -> np.ndarray:
        return (np.asarray(v, np.float64) @ self.R_view.T) * self._m(mesh_scale)

    def voxel_to_world(self, v, mesh_scale=None) -> np.ndarray:
        return (np.asarray(v, np.float64) / self._m(mesh_scale)) @ self.R_view

    def world_to_product(self, v, mesh_scale=None) -> np.ndarray:
        return self.world_to_voxel(v, mesh_scale) @ P2D

    def product_to_world(self, v, mesh_scale=None) -> np.ndarray:
        return self.voxel_to_world(np.asarray(v, np.float64) @ P2D, mesh_scale)

    @staticmethod
    def product_to_voxel(v) -> np.ndarray:
        return np.asarray(v, np.float64) @ P2D

    def world_normal_from_product(self, n) -> np.ndarray:
        """normals: product -> world (rotation part only; the isotropic scale drops out)."""
        return (np.asarray(n, np.float64) @ P2D) @ self.R_view

    def summary(self) -> dict:
        return {"sha256": self.sha256, "view": self.view, "camera_angle_x": self.camera_angle_x,
                "fov_deg": math.degrees(self.camera_angle_x), "distance": self.distance,
                "sphere_radius": self.sphere_radius, "box_scale": self.box_scale, "total_scale": self.total_scale,
                "R_view_equals_w2c": bool(np.allclose(self.R_view, self.R_w2c, atol=1e-6))}


def support32_from_points(pts_voxel: np.ndarray, res: int = 32) -> tuple[np.ndarray, int]:
    """floor((v+0.5)*res) occupancy of a dense point set; returns (unique int coords [M,3], n_out_of_grid)."""
    ijk = np.floor((np.asarray(pts_voxel, np.float64) + 0.5) * res).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < res)).all(1)
    return np.unique(ijk[ok], axis=0), int((~ok).sum())


def dense_surface_points(V, F, n=400_000, seed=0) -> np.ndarray:
    import trimesh
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=int(seed))
    return np.concatenate([np.asarray(m.vertices, np.float64), np.asarray(pts, np.float64)], 0)


def coords_iou(a: np.ndarray, b: np.ndarray, res: int = 32) -> dict:
    la = np.unique(a[:, 0] * res * res + a[:, 1] * res + a[:, 2])
    lb = np.unique(b[:, 0] * res * res + b[:, 1] * res + b[:, 2])
    inter = len(np.intersect1d(la, lb, assume_unique=True))
    union = len(la) + len(lb) - inter
    return {"iou": inter / union if union else 0.0, "n_a": int(len(la)), "n_b": int(len(lb)), "inter": int(inter),
            "precision": inter / len(la) if len(la) else 0.0, "recall": inter / len(lb) if len(lb) else 0.0}
