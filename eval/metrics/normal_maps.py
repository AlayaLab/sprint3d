"""normal_maps.py -- normal-map metrics in the input-image frame (Pixal3D arXiv 2605.10922, Tab. 1).

Per (object, view, arm): the generated mesh and the ground-truth mesh are rendered as flat camera-space normal maps at
512^2 in the input view (metrics/render.py), then

  mask_iou                             |pred & gt| / |pred | gt| of the two masks
  psnr_full / ssim_full / lpips_full   full 512^2 image, prediction and ground truth each composited on the constant
                                       background `bg_rgb` (declared: white); LPIPS = AlexNet v0.1
  lpips_full_vgg                       same images, LPIPS VGG v0.1
  psnr_ov / ssim_ov / lpips_ov         overlap region (pred & gt): PSNR over the overlap pixels; SSIM map averaged over
                                       the overlap; LPIPS(alex) with both images 0.5-gray outside the overlap
  ang_mean / ang_median                angular error acos(<n_pred, n_gt>) in degrees over the overlap
  ang_mean_b                           angular error over overlap & pixels within 10 px of the ground-truth mask boundary
  acc_11.25 / acc_22.5 / acc_30.0      fraction of overlap pixels with angular error below the threshold

Full-image metrics are defined even when the prediction misses the ground truth completely (a penalty, not a dropped
row); overlap and angular metrics are None without overlap.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy import ndimage

from .frame import ViewFrame, load_mesh_dump
from .render import Raster, perspective_clip, render_normals, world_to_cam

RES = 512
BAND_PX = 10
ACC_THRESHOLDS = (11.25, 22.5, 30.0)
GRAY = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------- SSIM map
def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim_map(x: torch.Tensor, y: torch.Tensor, data_range=1.0, size=11, sigma=1.5, K1=0.01, K2=0.03):
    """x, y: [1,C,H,W] in [0, data_range]; returns the per-pixel SSIM map [1,C,H',W'] ('valid' conv, identical to
    pytorch_msssim's formulation)."""
    C = x.shape[1]
    g = _gaussian_window(size, sigma, x.device)
    win = (g[:, None] @ g[None, :]).expand(C, 1, size, size).contiguous()
    mu_x = Fnn.conv2d(x, win, groups=C)
    mu_y = Fnn.conv2d(y, win, groups=C)
    sxx = Fnn.conv2d(x * x, win, groups=C) - mu_x ** 2
    syy = Fnn.conv2d(y * y, win, groups=C) - mu_y ** 2
    sxy = Fnn.conv2d(x * y, win, groups=C) - mu_x * mu_y
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    cs = (2 * sxy + C2) / (sxx + syy + C2)
    return ((2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)) * cs


# ---------------------------------------------------------------------------- LPIPS
class LPIPSScorer:
    def __init__(self, device="cuda", net="alex"):
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device
        self.net = net

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """a, b: [H,W,3] float in [0,1] -> LPIPS distance (inputs scaled to [-1,1])."""
        x = a.permute(2, 0, 1)[None].to(self.device) * 2 - 1
        y = b.permute(2, 0, 1)[None].to(self.device) * 2 - 1
        return float(self.model(x, y).item())


# ---------------------------------------------------------------------------- renders
def render_gt_normals(rz: Raster, view_frame: ViewFrame, mesh_dump, res=RES, device="cuda") -> dict:
    V, F = load_mesh_dump(mesh_dump)
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    Ft = torch.as_tensor(F, dtype=torch.int64, device=device)
    V_cam = world_to_cam(Vt, view_frame.c2w)
    clip = perspective_clip(V_cam, view_frame.camera_angle_x)
    return render_normals(rz, V_cam, Ft, clip, res, perspective=True, bg=GRAY)


def render_product_normals(rz: Raster, view_frame: ViewFrame, product_mesh, res=RES, device="cuda",
                           mesh_scale=None) -> dict:
    """true_camera rows: product -> world by the analytic chain -> ground-truth camera."""
    Vw = view_frame.product_to_world(np.asarray(product_mesh.vertices, np.float64), mesh_scale)
    Vt = torch.as_tensor(Vw, dtype=torch.float32, device=device)
    Ft = torch.as_tensor(np.asarray(product_mesh.faces, np.int64), dtype=torch.int64, device=device)
    V_cam = world_to_cam(Vt, view_frame.c2w)
    clip = perspective_clip(V_cam, view_frame.camera_angle_x)
    return render_normals(rz, V_cam, Ft, clip, res, perspective=True, bg=GRAY)


# ---------------------------------------------------------------------------- metrics
def boundary_band(gt_mask: np.ndarray, band_px=BAND_PX) -> np.ndarray:
    """pixels within band_px of the ground-truth mask boundary (inside or outside)."""
    inside = ndimage.distance_transform_edt(gt_mask)          # distance to nearest background pixel
    outside = ndimage.distance_transform_edt(~gt_mask)        # distance to nearest foreground pixel
    dist = np.where(gt_mask, inside, outside)
    return dist <= band_px


def composite(render: dict, bg_rgb) -> torch.Tensor:
    bg = torch.tensor(list(bg_rgb), dtype=torch.float32, device=render["rgb"].device).view(1, 1, 3)
    return torch.where(render["mask"][..., None], render["rgb"], bg.expand_as(render["rgb"]))


def full_image_metrics(p_full: torch.Tensor, g_full: torch.Tensor, lpips_main=None, lpips_secondary=None) -> dict:
    """p_full, g_full: [H,W,3] in [0,1] (already composited on the constant background)."""
    mse = float(((p_full - g_full) ** 2).mean())
    out = {"psnr_full": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
           "ssim_full": float(ssim_map(p_full.permute(2, 0, 1)[None], g_full.permute(2, 0, 1)[None]).mean()),
           "lpips_full": lpips_main(p_full, g_full) if lpips_main is not None else None,
           "lpips_full_vgg": lpips_secondary(p_full, g_full) if lpips_secondary is not None else None}
    return out


@torch.no_grad()
def normal_map_metrics(pred: dict, gt: dict, *, bg_rgb, lpips_main=None, lpips_secondary=None,
                       band_px=BAND_PX, acc_thresholds=ACC_THRESHOLDS) -> dict:
    pm, gm = pred["mask"], gt["mask"]
    overlap = pm & gm
    union = pm | gm
    n_ov, n_un = int(overlap.sum()), int(union.sum())
    out = {"mask_iou": n_ov / n_un if n_un else 0.0, "n_overlap_px": n_ov, "n_pred_px": int(pm.sum()),
           "n_gt_px": int(gm.sum()), "n_union_px": n_un, "n_px": int(pm.numel())}
    # ---- full image, constant background -----------------------------------------------------
    out.update(full_image_metrics(composite(pred, bg_rgb), composite(gt, bg_rgb), lpips_main, lpips_secondary))
    # ---- overlap region + angular metrics ------------------------------------------------------
    none_keys = ("psnr_ov", "ssim_ov", "lpips_ov", "ang_mean", "ang_median", "ang_mean_b")
    if n_ov == 0:
        out.update({k: None for k in none_keys})
        out.update({f"acc_{t}": None for t in acc_thresholds})
        out["n_band_px"] = 0
        return out
    gray = torch.tensor(GRAY, dtype=torch.float32, device=pm.device).view(1, 1, 3)
    p_ov = torch.where(overlap[..., None], pred["rgb"], gray)
    g_ov = torch.where(overlap[..., None], gt["rgb"], gray)
    mse = ((pred["rgb"] - gt["rgb"]) ** 2)[overlap].mean()
    out["psnr_ov"] = float(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))
    smap = ssim_map(p_ov.permute(2, 0, 1)[None], g_ov.permute(2, 0, 1)[None])
    ov_c = overlap[5:-5, 5:-5]
    out["ssim_ov"] = float(smap[0].mean(0)[ov_c].mean()) if int(ov_c.sum()) else float(smap.mean())
    out["lpips_ov"] = lpips_main(p_ov, g_ov) if lpips_main is not None else None
    dot = (pred["normal"] * gt["normal"]).sum(-1).clamp(-1.0, 1.0)
    ang_all = torch.rad2deg(torch.acos(dot))
    ang = ang_all[overlap]
    out["ang_mean"] = float(ang.mean())
    out["ang_median"] = float(ang.median())
    for t in acc_thresholds:
        out[f"acc_{t}"] = float((ang < t).float().mean())
    band = torch.as_tensor(boundary_band(gm.cpu().numpy(), band_px), device=pm.device)
    ob = overlap & band
    out["n_band_px"] = int(ob.sum())
    out["ang_mean_b"] = float(ang_all[ob].mean()) if out["n_band_px"] else None
    return out


def save_pair_png(path, pred: dict, gt: dict):
    """pred | ground-truth normal maps on white (pure white == background)."""
    from PIL import Image
    from .render import to_uint8
    pr, gr = composite(pred, (1.0, 1.0, 1.0)), composite(gt, (1.0, 1.0, 1.0))
    Image.fromarray(np.concatenate([to_uint8(pr), to_uint8(gr)], 1)).save(path)
