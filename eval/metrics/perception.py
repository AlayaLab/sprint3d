"""perception.py -- appendix perception metrics: CLIP, CLIP-N (TRELLIS.2 definition), ULIP-2 / Uni3D (Pixal3D definition).

CLIP / CLIP-N (TRELLIS.2, arXiv 2512.14692, Tab. 2: "visual alignment is measured with the CLIP score", "-N: measured
with normal map"; the paper fixes neither the CLIP model nor the views, so the choice below is declared):
  * model: openai/clip-vit-large-patch14 (transformers CLIPModel, pinned revision), image embeddings L2-normalised
  * input image: the conditioning render composited on white, 512^2
  * views: 4 perspective renders of the product with the input camera intrinsics, yaw offsets 0 / 90 / 180 / 270 deg
    about the world up axis (same elevation and distance), 512^2, white background
  * CLIP   = mean over the 4 views of cos(emb(textured render), emb(input))
  * CLIP-N = mean over the 4 views of cos(emb(normal render), emb(input))
  * per-view values are stored too (view 0 = the input view).

ULIP-2 (arXiv 2305.08275, ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g) and Uni3D (arXiv 2310.06773, uni3d-g): image-3D
consistency = cos(point-cloud embedding, image embedding) with each model's own image tower (OpenCLIP ViT-bigG-14 for
ULIP-2, EVA02-E-14-plus for Uni3D).  Point cloud: 10,000 (ULIP-2) / 8,192 (Uni3D) surface samples with colour from the
textured product (xyz + rgb).  Missing code or checkpoints are reported per row (`<name>_status`), never faked.
"""
from __future__ import annotations

import numpy as np
import torch

from .render import Raster, composite_rgba_png, mesh_texture_tensors, perspective_clip, render_textured, world_to_cam, \
    yaw_c2w

CLIP_VIEW_YAWS = (0.0, 90.0, 180.0, 270.0)
CLIP_RES = 512


class CLIPScorer:
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14", revision=None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_id, revision=revision).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id, revision=revision)
        self.model_id = model_id

    @torch.no_grad()
    def embed(self, images_uint8) -> torch.Tensor:
        """list of HxWx3 uint8 arrays -> [N,D] L2-normalised image embeddings."""
        from PIL import Image
        pil = [Image.fromarray(np.asarray(im)) for im in images_uint8]
        inp = self.proc(images=pil, return_tensors="pt").to(self.device)
        e = self.model.get_image_features(**inp)
        return torch.nn.functional.normalize(e, dim=-1)


@torch.no_grad()
def render_clip_views(rz: Raster, view_frame, product_mesh, device="cuda", yaws=CLIP_VIEW_YAWS, res=CLIP_RES,
                      mesh_scale=None) -> dict:
    """textured RGB + normal RGB renders of the product from the input camera and yaw-offset cameras."""
    Vw = view_frame.product_to_world(np.asarray(product_mesh.vertices, np.float64), mesh_scale)
    Vt = torch.as_tensor(Vw, dtype=torch.float32, device=device)
    Ft = torch.as_tensor(np.asarray(product_mesh.faces, np.int64), dtype=torch.int64, device=device)
    tex = mesh_texture_tensors(product_mesh, device)
    rgb, nrm = [], []
    for yaw in yaws:
        c2w = yaw_c2w(view_frame.c2w, yaw, up=(0.0, 0.0, 1.0))
        V_cam = world_to_cam(Vt, c2w)
        clip = perspective_clip(V_cam, view_frame.camera_angle_x)
        r = render_textured(rz, V_cam, Ft, clip, res, tex, bg=(1.0, 1.0, 1.0))
        rgb.append(r["rgb"])
        nrm.append(r["normal_rgb"])
    return {"rgb": torch.stack(rgb), "normal": torch.stack(nrm), "textured": tex is not None, "yaws": list(yaws)}


@torch.no_grad()
def clip_scores(scorer: CLIPScorer, views: dict, input_png) -> dict:
    from .render import to_uint8
    inp = composite_rgba_png(input_png, bg=(255, 255, 255), size=CLIP_RES)
    e_in = scorer.embed([inp])
    e_rgb = scorer.embed([to_uint8(v) for v in views["rgb"]])
    e_n = scorer.embed([to_uint8(v) for v in views["normal"]])
    s_rgb = (e_rgb @ e_in.T).squeeze(-1)
    s_n = (e_n @ e_in.T).squeeze(-1)
    return {"clip": float(s_rgb.mean()), "clip_n": float(s_n.mean()),
            "clip_per_view": [float(v) for v in s_rgb], "clip_n_per_view": [float(v) for v in s_n],
            "clip_input_view": float(s_rgb[0]), "clip_n_input_view": float(s_n[0]),
            "clip_textured": bool(views["textured"]), "clip_model": scorer.model_id}


# ---------------------------------------------------------------------------- ULIP-2 / Uni3D
def colored_point_cloud(product_mesh, n: int, seed: int) -> np.ndarray:
    """[n,6] xyz (unit sphere) + rgb (0..1) surface samples; colour from the base-colour texture when present
    (per-sample uv), else 0.5 gray."""
    import trimesh
    pts, fidx = trimesh.sample.sample_surface(product_mesh, n, seed=int(seed))
    pts = np.asarray(pts, np.float64)
    rgb = np.full((n, 3), 0.5, np.float32)
    vis = getattr(product_mesh, "visual", None)
    try:
        if vis is not None and getattr(vis, "uv", None) is not None and vis.material is not None:
            img = getattr(vis.material, "baseColorTexture", None)
            if img is not None:
                tri = product_mesh.faces[fidx]
                bary = trimesh.triangles.points_to_barycentric(product_mesh.triangles[fidx], pts)
                uv = (np.asarray(vis.uv)[tri] * bary[..., None]).sum(1)
                im = np.asarray(img.convert("RGB"), np.float32) / 255.0
                H, W = im.shape[:2]
                px = np.clip((uv[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
                py = np.clip((1.0 - (uv[:, 1] % 1.0)) * (H - 1), 0, H - 1).astype(int)
                rgb = im[py, px]
    except Exception:
        pass
    c = (pts.min(0) + pts.max(0)) / 2
    pts = pts - c
    pts = pts / max(float(np.linalg.norm(pts, axis=1).max()), 1e-12)
    return np.concatenate([pts.astype(np.float32), rgb.astype(np.float32)], 1)


class _Image3DScorer:
    """cos(point-cloud embedding, input-image embedding) with one of the 3D models (metrics/scorers3d.py)."""
    name = "base"
    n_points = 8192
    model_cls = None

    def __init__(self, device="cuda", paths: dict | None = None):
        self.device = torch.device(device)
        self.model = None
        self.paths = paths or {}
        self.missing = list(self.model_cls.missing(self.paths))
        self.load_error = None
        if not self.missing:
            try:
                self.model = self.model_cls(device, self.paths)
                self.n_points = self.model.n_points
            except Exception as e:  # report, never fake
                self.load_error = repr(e)
                self.missing = [f"load failed: {self.load_error}"]

    @torch.no_grad()
    def score(self, product_mesh, input_png, seed: int) -> dict:
        if self.missing:
            return {self.name: None, f"{self.name}_status": "unavailable: " + "; ".join(self.missing)}
        pc = colored_point_cloud(product_mesh, self.n_points, seed)
        e_pc = self.model.embed_points(pc)
        e_im = self.model.embed_image(composite_rgba_png(input_png, bg=(255, 255, 255), size=224))
        return {self.name: float((e_pc * e_im).sum()), f"{self.name}_status": "ok"}


class ULIP2Scorer(_Image3DScorer):
    """ULIP-2 PointBERT (10k xyz+rgb, vit_g) + OpenCLIP ViT-bigG-14 image tower."""
    name = "ulip2"
    n_points = 10000
    from .scorers3d import ULIP2Model as model_cls  # noqa: E402


class Uni3DScorer(_Image3DScorer):
    """Uni3D-g + EVA02-E-14-plus image tower."""
    name = "uni3d"
    n_points = 8192
    from .scorers3d import Uni3DModel as model_cls  # noqa: E402
