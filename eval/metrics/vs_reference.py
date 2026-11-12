"""vs_reference.py -- agreement of a product with the reference product of the same input (appendix).

  * cd_to_ref / f_0.005_to_ref / f_0.01_to_ref: geometry.chamfer_fscore between the two products in the shared product
    frame (both generated for the same (image, camera); no transform at all).
  * DINOv2 ViT-L/14 CLS and patch distances over 4 independent rendered views: orthographic cameras (scale 1.5,
    elevation 20 deg, azimuths 0 / 90 / 180 / 270), 518^2, camera-space normal RGB, white background, object centred in
    the unit cube; features from the facebookresearch/dinov2 code (local checkout) + dinov2_vitl14_pretrain.pth.
        dino_cls   : 1 - cos(CLS_cand, CLS_ref) averaged over the 4 views
        dino_patch : mean over the 37x37 patch tokens of 1 - cos(patch_cand, patch_ref) (same position), averaged over views
  * noise band: the same statistics between two reference samples of different seeds when a second reference arm is
    supplied; flips = objects whose cd_to_ref exceeds flip_threshold_from_noise_band (default: 95th percentile of the
    noise-band cd_to_ref).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .render import Raster, lookat_c2w, orthographic_clip, render_normals, world_to_cam

DINO_RES = 518
DINO_AZIMUTHS = (0, 90, 180, 270)
DINO_ELEVATION = 20.0
ORTHO_SCALE = 1.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_unit_cube(V: np.ndarray) -> np.ndarray:
    """bbox centre -> 0, longest axis -> 1."""
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    s = float((hi - lo).max())
    return (V - c) / max(s, 1e-12)


class DinoV2:
    def __init__(self, device="cuda", code_dir=None, weights=None):
        if not code_dir or not weights:
            raise FileNotFoundError("DINOv2 needs models.dinov2_code and models.dinov2_weights in the configuration")
        self.device = torch.device(device)
        model = torch.hub.load(str(code_dir), "dinov2_vitl14", source="local", pretrained=False)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, imgs: torch.Tensor) -> dict:
        """imgs: [N,H,W,3] in [0,1] at 518^2 -> {'cls': [N,1024] L2-normalised, 'patch': [N,1369,1024] L2-normalised}"""
        x = imgs.permute(0, 3, 1, 2).to(self.device)
        if x.shape[-1] != DINO_RES:
            x = torch.nn.functional.interpolate(x, size=(DINO_RES, DINO_RES), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model.forward_features(x)
        cls = torch.nn.functional.normalize(out["x_norm_clstoken"], dim=-1)
        patch = torch.nn.functional.normalize(out["x_norm_patchtokens"], dim=-1)
        return {"cls": cls, "patch": patch}


@torch.no_grad()
def render_dino_views(rz: Raster, V_product: np.ndarray, F: np.ndarray, device="cuda") -> torch.Tensor:
    """4 orthographic camera-space normal renders [4,518,518,3] of a product (unit-cube normalised)."""
    Vn = normalize_unit_cube(V_product)
    Vt = torch.as_tensor(Vn, dtype=torch.float32, device=device)
    Ft = torch.as_tensor(F, dtype=torch.int64, device=device)
    views = []
    for az in DINO_AZIMUTHS:
        c2w = lookat_c2w(az, DINO_ELEVATION, radius=3.0, up_axis="y")
        V_cam = world_to_cam(Vt, c2w)
        clip = orthographic_clip(V_cam, ORTHO_SCALE)
        r = render_normals(rz, V_cam, Ft, clip, DINO_RES, perspective=False, bg=(1.0, 1.0, 1.0))
        views.append(r["rgb"])
    return torch.stack(views, 0)


def dino_distances(fa: dict, fb: dict) -> dict:
    cls = 1.0 - (fa["cls"] * fb["cls"]).sum(-1)                  # [4]
    patch = 1.0 - (fa["patch"] * fb["patch"]).sum(-1).mean(-1)   # [4]
    return {"dino_cls": float(cls.mean()), "dino_patch": float(patch.mean()),
            "dino_cls_per_view": [float(v) for v in cls], "dino_patch_per_view": [float(v) for v in patch]}


def flip_threshold_from_noise_band(noise_cd_to_ref: dict, quantile=0.95) -> float | None:
    v = np.asarray([x for x in noise_cd_to_ref.values() if x is not None and math.isfinite(x)])
    return float(np.quantile(v, quantile)) if len(v) else None
