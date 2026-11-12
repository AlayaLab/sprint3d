"""projection.py -- how a row's mesh reaches the input-view image.

Three placements (conventions.CONVENTIONS["conditions"]):

  true_camera  product generated with the ground-truth camera (mesh_scale = total_scale):
               v_world = ((v_product @ P2D) / total_scale) @ R_view, rendered with the ground-truth camera.
  official / manual_fov
               product generated without a ground-truth camera: rendered with its own inference-time projection
               (Pixal3D paper Sec. 4.1).  The official inference builds that camera as {camera_angle_x (MoGe-2 on
               the official crop, or manual), distance = distance_from_fov(...) = 0.5 / tan(fov / 2) / mesh_scale,
               mesh_scale = 1.0} and the conditioning back-projection is a pinhole at (0, -distance, 0) of the Z-up
               generation cube looking at its centre.  In the exported product frame (v_voxel = v_product @ P2D;
               voxel axes == camera axes) this reads
                   v_cam = (v_product @ P2D) / mesh_scale - (0, 0, distance)
               and the image it projects to is the official crop of the input.  The crop -> raw-image map is an exact
               NDC affine (official_crop.crop_to_raw_ndc), so the normal map is rasterised directly in the raw
               conditioning image frame where the ground-truth normal map lives.
  baseline     external mesh: fixed per-method frame map -> ground-truth world axes -> own-bbox normalisation ->
               rendered with the ground-truth camera pose (paper Sec. 4.1).

For the 3D metrics every mesh is taken to ground-truth world axes (rotation only) and normalised by its own bounding
box (`normalise_own_bbox`), identically for all methods and the ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .frame import P2D, ViewFrame
from .official_crop import IDENTITY_NDC, crop_to_raw_ndc, official_crop_box


# ---------------------------------------------------------------------------- normalisation
def normalise_own_bbox(V: np.ndarray) -> tuple[np.ndarray, dict]:
    """v' = (v - bbox_centre) / max_extent  (max extent 1, bbox-centred).  The one normalisation rule."""
    V = np.asarray(V, np.float64)
    lo, hi = V.min(0), V.max(0)
    centre = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("degenerate mesh bounding box")
    return (V - centre) / extent, {"bbox_centre": centre.tolist(), "bbox_max_extent": extent}


# ---------------------------------------------------------------------------- own projection
@dataclass
class OwnProjection:
    camera_angle_x: float
    distance: float
    mesh_scale: float
    ndc: dict = field(default_factory=lambda: dict(IDENTITY_NDC))   # crop -> raw image NDC affine
    crop: dict | None = None
    crop_source: str = "identity"

    def product_to_cam(self, v_product) -> np.ndarray:
        v = (np.asarray(v_product, np.float64) @ P2D) / float(self.mesh_scale)
        v[:, 2] -= float(self.distance)
        return v

    def summary(self) -> dict:
        return {"camera_angle_x": self.camera_angle_x, "fov_deg": math.degrees(self.camera_angle_x),
                "distance": self.distance, "mesh_scale": self.mesh_scale, "ndc": self.ndc,
                "crop_source": self.crop_source, "crop_box": None if self.crop is None else self.crop.get("crop_box"),
                "distance_matches_official_formula": bool(abs(
                    self.distance - 0.5 / math.tan(self.camera_angle_x / 2.0) / self.mesh_scale) < 1e-4 * self.distance)}


def own_projection_for_row(row: dict, *, identity_crop=False) -> OwnProjection:
    """Inference-time projection of a product row: camera = the row's recorded inference camera; crop = the official
    crop recomputed from the raw input image."""
    cam = row.get("camera") or {}
    if not all(k in cam for k in ("camera_angle_x", "distance", "mesh_scale")):
        raise ValueError(f"row {row.get('object_id')} has no recorded inference camera")
    proj = OwnProjection(float(cam["camera_angle_x"]), float(cam["distance"]), float(cam["mesh_scale"]))
    if identity_crop:
        return proj
    from PIL import Image
    crop = official_crop_box(Image.open(row["input_png"]))
    proj.crop_source = "recomputed_from_input"
    proj.crop = crop
    proj.ndc = crop_to_raw_ndc(crop["crop_box"], crop["source_size"])
    return proj


# ---------------------------------------------------------------------------- placements
def product_to_world_axes(view_frame: ViewFrame, v_product) -> np.ndarray:
    """rotation only: product -> voxel axes (P2D) -> ground-truth world axes (R_view); no scale, no translation."""
    return (np.asarray(v_product, np.float64) @ P2D) @ view_frame.R_view


def world_axes_to_productlike(view_frame: ViewFrame, v_world) -> np.ndarray:
    """inverse rotation: ground-truth world axes -> the product-frame orientation convention (for the 3D scorers)."""
    return (np.asarray(v_world, np.float64) @ view_frame.R_view.T) @ P2D


class WorldPlacement:
    """duck-typed stand-in for ViewFrame in renderers that call `.product_to_world(vertices)`: returns a fixed world
    placement (own-bbox normalised, ground-truth world axes) of the mesh."""

    def __init__(self, view_frame: ViewFrame, to_world_axes):
        self.c2w = view_frame.c2w
        self.camera_angle_x = view_frame.camera_angle_x
        self._to_world_axes = to_world_axes
        self.info = {}

    def product_to_world(self, v, mesh_scale=None) -> np.ndarray:
        Vn, self.info = normalise_own_bbox(self._to_world_axes(v))
        return Vn


# ---------------------------------------------------------------------------- rendering
def perspective_clip_affine(v_cam, fov_x: float, ndc: dict, near=0.01, far=100.0):
    """pinhole (centre principal point) followed by the NDC affine x' = sx x + tx, y' = sy y + ty."""
    import torch
    f = 1.0 / math.tan(fov_x / 2.0)
    x, y, z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]
    w = -z
    a = -(far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    return torch.stack([f * x * ndc["sx"] + ndc["tx"] * w, f * y * ndc["sy"] + ndc["ty"] * w, a * z + b, w], dim=1)


def render_own_projection_normals(rz, proj: OwnProjection, mesh, res=512, device="cuda", bg=(0.5, 0.5, 0.5)) -> dict:
    import torch
    from .render import render_normals
    V_cam = torch.as_tensor(proj.product_to_cam(np.asarray(mesh.vertices)), dtype=torch.float32, device=device)
    F = torch.as_tensor(np.asarray(mesh.faces, np.int64), dtype=torch.int64, device=device)
    clip = perspective_clip_affine(V_cam, proj.camera_angle_x, proj.ndc)
    return render_normals(rz, V_cam, F, clip, res, perspective=True, bg=bg)


def render_world_normals(rz, view_frame: ViewFrame, V_world, faces, res=512, device="cuda", bg=(0.5, 0.5, 0.5)) -> dict:
    """a mesh given in ground-truth world coordinates, rendered with the ground-truth camera pose (baselines)."""
    import torch
    from .render import perspective_clip, render_normals, world_to_cam
    Vt = torch.as_tensor(np.asarray(V_world), dtype=torch.float32, device=device)
    F = torch.as_tensor(np.asarray(faces, np.int64), dtype=torch.int64, device=device)
    V_cam = world_to_cam(Vt, view_frame.c2w)
    clip = perspective_clip(V_cam, view_frame.camera_angle_x)
    return render_normals(rz, V_cam, F, clip, res, perspective=True, bg=bg)
