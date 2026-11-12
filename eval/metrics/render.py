"""render.py -- nvdiffrast rendering used by the normal-map, perception and reference-agreement metrics.

nvdiffrast 0.4.0, RasterizeCudaContext, float32, deterministic; with (a) the input-view perspective camera of
transforms.json (normal-map metrics), (b) yaw-offset perspective cameras around the world up axis (CLIP / CLIP-N
views) and (c) orthographic 4-azimuth cameras (DINOv2 agreement with a reference product).

Conventions
  * camera frame = Blender camera (x right, y up, -z forward); pixel row 0 = top of the image (nvdiffrast rasterizes
    bottom-up; we flip), matching the PNG renders.
  * normals: per-face (flat) geometric normals of the rendered mesh, transformed to camera space and flipped to face
    the camera (dot(n, view_dir) >= 0) -- identical treatment for prediction and ground truth, so inconsistent winding
    of ground-truth dumps or generated meshes does not count as a normal error.
  * normal RGB encoding: rgb = n * 0.5 + 0.5 (camera-space n); background = `bg`.
  * textured RGB (CLIP): glTF base-colour texture sampled bilinearly, Lambertian headlight
    (ambient 0.35 + 0.65 * max(n . v, 0)), white background.  trimesh returns uv with the origin bottom-left (its glTF
    loader flips v); nvdiffrast / glTF sample with the origin top-left -> gltf_uv_from_trimesh() before dr.texture.
"""
from __future__ import annotations

import math

import numpy as np
import torch

NEAR, FAR = 0.01, 100.0


class Raster:
    _ctx = None

    def __init__(self, device="cuda"):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.device = torch.device(device)
        if Raster._ctx is None:
            Raster._ctx = dr.RasterizeCudaContext(device=self.device)
        self.ctx = Raster._ctx

    def rasterize(self, clip, faces, res):
        faces_i = faces.to(torch.int32).contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip[None].contiguous(), faces_i, (res, res), grad_db=False)
        return rast[0], faces_i

    def interpolate(self, attr, rast, faces_i):
        out, _ = self.dr.interpolate(attr[None].contiguous(), rast[None].contiguous(), faces_i)
        return out[0]

    def texture(self, tex, uv):
        return self.dr.texture(tex[None].contiguous(), uv[None].contiguous(), filter_mode="linear")[0]


# ---------------------------------------------------------------------------- cameras
def perspective_clip(v_cam: torch.Tensor, fov_x: float, near=NEAR, far=FAR) -> torch.Tensor:
    """camera-space [V,3] -> OpenGL clip coords [V,4] (square image, horizontal FoV fov_x)."""
    f = 1.0 / math.tan(fov_x / 2.0)
    x, y, z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]
    a = -(far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    return torch.stack([f * x, f * y, a * z + b, -z], dim=1)


def orthographic_clip(v_cam: torch.Tensor, scale: float, near=-10.0, far=10.0) -> torch.Tensor:
    x, y, z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]
    return torch.stack([x * 2.0 / scale, y * 2.0 / scale, (-z - near) / (far - near) * 2.0 - 1.0,
                        torch.ones_like(x)], dim=1)


def yaw_c2w(c2w: np.ndarray, yaw_deg: float, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """rotate a camera pose about the world up axis (Blender world: z up) by yaw_deg (about the origin)."""
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    ux, uy, uz = up
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], np.float64)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    out = np.eye(4)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = R @ c2w[:3, 3]
    return out


def lookat_c2w(azimuth_deg: float, elevation_deg: float, radius: float, up_axis="y") -> np.ndarray:
    """camera on a sphere looking at the origin."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    if up_axis == "y":
        eye = np.array([radius * math.cos(e) * math.sin(a), radius * math.sin(e), radius * math.cos(e) * math.cos(a)])
        up = np.array([0.0, 1.0, 0.0])
    else:
        eye = np.array([radius * math.cos(e) * math.cos(a), radius * math.cos(e) * math.sin(a), radius * math.sin(e)])
        up = np.array([0.0, 0.0, 1.0])
    f = -eye / np.linalg.norm(eye)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, u, -f, eye
    return c2w


def world_to_cam(V_world: torch.Tensor, c2w: np.ndarray) -> torch.Tensor:
    R = torch.as_tensor(c2w[:3, :3], dtype=torch.float32, device=V_world.device)
    t = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=V_world.device)
    return (V_world - t) @ R  # == R^T (v - t)


# ---------------------------------------------------------------------------- normals
def face_normals(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = torch.cross(b - a, c - a, dim=1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def render_normals(rz: Raster, V_cam: torch.Tensor, F: torch.Tensor, clip: torch.Tensor, res: int,
                   perspective: bool = True, bg=(1.0, 1.0, 1.0)) -> dict:
    """Flat camera-space normals.  Returns dict(mask [H,W] bool, normal [H,W,3] float (0 outside),
    rgb [H,W,3] float in [0,1] with bg outside, depth [H,W] camera -z (nan outside), tri [H,W] int)."""
    rast, faces_i = rz.rasterize(clip, F, res)
    rast = torch.flip(rast, dims=[0])                         # bottom-up -> top-down rows
    tri = rast[..., 3].long() - 1
    mask = tri >= 0
    fn = face_normals(V_cam, F)                              # camera-space face normals
    n = torch.zeros((res, res, 3), dtype=torch.float32, device=V_cam.device)
    n[mask] = fn[tri[mask]]
    # per-pixel camera-space position (for the view direction) via barycentric interpolation
    pos = rz.interpolate(V_cam, rast, faces_i)               # per-pixel barycentrics; row flip is harmless
    if perspective:
        view_dir = -pos / pos.norm(dim=-1, keepdim=True).clamp_min(1e-12)   # pixel -> camera
    else:
        view_dir = torch.tensor([0.0, 0.0, 1.0], device=V_cam.device).expand_as(pos)
    flip = (n * view_dir).sum(-1, keepdim=True) < 0
    n = torch.where(flip, -n, n)
    n = torch.where(mask[..., None], n, torch.zeros_like(n))
    rgb = (n * 0.5 + 0.5).clamp(0, 1)
    bgt = torch.tensor(bg, dtype=torch.float32, device=V_cam.device).view(1, 1, 3)
    rgb = torch.where(mask[..., None], rgb, bgt.expand_as(rgb))
    depth = torch.where(mask, -pos[..., 2], torch.full_like(pos[..., 2], float("nan")))
    return {"mask": mask, "normal": n, "rgb": rgb, "depth": depth, "tri": tri}


# ---------------------------------------------------------------------------- textured
def gltf_uv_from_trimesh(uv) -> np.ndarray:
    """trimesh uv (origin bottom-left: trimesh's glTF loader sets v = 1 - v on load) -> glTF uv as stored in the file
    (origin top-left), which is what nvdiffrast dr.texture expects for an image tensor whose row 0 is the top row."""
    uv = np.asarray(uv, np.float32)
    return np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1).astype(np.float32)


def mesh_texture_tensors(mesh, device) -> dict | None:
    """base-colour texture + per-vertex uv of a trimesh (glTF), or None when untextured.
    Returned uv are in the glTF / nvdiffrast convention (origin top-left, see gltf_uv_from_trimesh)."""
    vis = getattr(mesh, "visual", None)
    try:
        uv = gltf_uv_from_trimesh(vis.uv) if vis is not None and getattr(vis, "uv", None) is not None else None
        mat = getattr(vis, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if uv is None or img is None or len(uv) != len(mesh.vertices):
            return None
        tex = np.ascontiguousarray(np.asarray(img.convert("RGB"), np.float32) / 255.0)
        base = None
        if mat is not None and getattr(mat, "baseColorFactor", None) is not None:
            base = np.asarray(mat.baseColorFactor, np.float32)[:3] / (255.0 if np.max(mat.baseColorFactor) > 1 else 1.0)
        return {"tex": torch.as_tensor(tex, device=device), "uv": torch.as_tensor(uv, device=device),
                "base": None if base is None else torch.as_tensor(base, device=device), "uv_convention": "gltf_top_left"}
    except Exception:
        return None


@torch.no_grad()
def render_textured(rz: Raster, V_cam: torch.Tensor, F: torch.Tensor, clip: torch.Tensor, res: int,
                    texinfo: dict | None, bg=(1.0, 1.0, 1.0)) -> dict:
    nm = render_normals(rz, V_cam, F, clip, res, perspective=True, bg=bg)
    rast, faces_i = rz.rasterize(clip, F, res)
    rast = torch.flip(rast, dims=[0])
    mask = nm["mask"]
    if texinfo is not None:
        uv = rz.interpolate(texinfo["uv"], rast, faces_i)      # glTF uv (origin top-left, see mesh_texture_tensors)
        albedo = rz.texture(texinfo["tex"], uv)
        if texinfo.get("base") is not None:
            albedo = albedo * texinfo["base"].view(1, 1, 3)
    else:
        albedo = torch.full((res, res, 3), 0.8, device=V_cam.device)
    pos = rz.interpolate(V_cam, rast, faces_i)
    vdir = -pos / pos.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    lam = (nm["normal"] * vdir).sum(-1, keepdim=True).clamp_min(0.0)
    shade = albedo * (0.35 + 0.65 * lam)
    bgt = torch.tensor(bg, dtype=torch.float32, device=V_cam.device).view(1, 1, 3)
    rgb = torch.where(mask[..., None], shade.clamp(0, 1), bgt.expand_as(shade))
    return {"rgb": rgb, "mask": mask, "normal_rgb": nm["rgb"], "textured": texinfo is not None}


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def composite_rgba_png(path, bg=(255, 255, 255), size=None) -> np.ndarray:
    """RGBA render -> RGB uint8 composited on bg (the conditioning image as CLIP sees it)."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", im.size, (*bg, 255))
    canvas.alpha_composite(im)
    return np.asarray(canvas.convert("RGB"))
