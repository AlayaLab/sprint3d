"""Normal-map metrics on tiny synthetic maps with closed-form answers."""
import math

import numpy as np
import pytest
import torch

from metrics import normal_maps as nm

H = 64


def _maps(mask_p, mask_g, n_p=None, n_g=None):
    n = torch.zeros(H, H, 3)
    n[..., 2] = 1.0
    n_p = n if n_p is None else n_p
    n_g = n if n_g is None else n_g
    mk = lambda mask, nn: {"mask": mask, "rgb": torch.where(mask[..., None], nn * 0.5 + 0.5, torch.full_like(nn, 0.5)),  # noqa: E731
                           "normal": torch.where(mask[..., None], nn, torch.zeros_like(nn))}
    return mk(mask_p, n_p), mk(mask_g, n_g)


def _squares(shift=4):
    p = torch.zeros(H, H, dtype=torch.bool)
    g = torch.zeros(H, H, dtype=torch.bool)
    p[16:48, 16:48] = True
    g[16:48, 16 + shift:48 + shift] = True
    return p, g


def test_full_image_psnr_closed_form_and_keys():
    p, g = _squares()
    pred, gt = _maps(p, g)
    out = nm.normal_map_metrics(pred, gt, bg_rgb=(1.0, 1.0, 1.0))
    # pixels where exactly one map has the object: |(0.5,0.5,1) - (1,1,1)|^2 summed = 0.5 -> mean over channels 1/6
    mse = int((p ^ g).sum()) * (0.5 / 3) / (H * H)
    assert out["psnr_full"] == pytest.approx(10 * math.log10(1 / mse), abs=1e-4)
    assert out["mask_iou"] == pytest.approx(int((p & g).sum()) / int((p | g).sum()))
    assert out["ang_mean"] == pytest.approx(0.0, abs=1e-3)
    assert out["psnr_ov"] > 100.0                        # identical inside the overlap
    assert out["ssim_ov"] == pytest.approx(1.0, abs=1e-6)
    for k in ("psnr", "ssim", "lpips"):
        assert k not in out                              # only region-qualified keys
    black = nm.normal_map_metrics(pred, gt, bg_rgb=(0.0, 0.0, 0.0))
    assert abs(black["psnr_full"] - out["psnr_full"]) > 0.01   # the background is a real parameter


def test_angular_error_and_accuracies():
    p = torch.zeros(H, H, dtype=torch.bool)
    p[8:56, 8:56] = True
    t = math.radians(20.0)
    n_g = torch.zeros(H, H, 3)
    n_g[..., 2] = 1.0
    n_p = torch.zeros(H, H, 3)
    n_p[..., 0], n_p[..., 2] = math.sin(t), math.cos(t)
    pred, gt = _maps(p, p, n_p, n_g)
    out = nm.normal_map_metrics(pred, gt, bg_rgb=(1.0, 1.0, 1.0))
    assert out["ang_mean"] == pytest.approx(20.0, abs=1e-3)
    assert out["ang_median"] == pytest.approx(20.0, abs=1e-3)
    assert out["ang_mean_b"] == pytest.approx(20.0, abs=1e-3)
    assert (out["acc_11.25"], out["acc_22.5"], out["acc_30.0"]) == (0.0, 1.0, 1.0)
    assert out["mask_iou"] == 1.0


def test_overlap_psnr_closed_form():
    p = torch.zeros(H, H, dtype=torch.bool)
    p[8:56, 8:56] = True
    n_g = torch.zeros(H, H, 3)
    n_g[..., 2] = 1.0
    n_p = torch.zeros(H, H, 3)
    n_p[..., 1] = 1.0
    pred, gt = _maps(p, p, n_p, n_g)
    out = nm.normal_map_metrics(pred, gt, bg_rgb=(1.0, 1.0, 1.0))
    # rgb (0.5,1,0.5) vs (0.5,0.5,1): squared error 0.25 + 0.25 over 3 channels
    assert out["psnr_ov"] == pytest.approx(10 * math.log10(1 / (0.5 / 3)), abs=1e-4)
    assert out["ang_mean"] == pytest.approx(90.0, abs=1e-3)


def test_boundary_band():
    m = np.zeros((32, 32), bool)
    m[8:24, 8:24] = True
    band = nm.boundary_band(m, band_px=2)
    assert band[8, 16] and band[6, 16] and band[9, 16]       # within 2 px of the edge, both sides
    assert not band[10, 16] and not band[5, 16]
    assert not band[16, 16] and not band[2, 16]


def test_ssim_map_identity_and_shape():
    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    s = nm.ssim_map(x, x)
    assert s.shape == (1, 3, 22, 22)                      # 'valid' convolution with an 11x11 window
    assert float(s.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(nm.ssim_map(x, 1 - x).mean()) < 0.5


def test_no_overlap_is_a_penalty_not_a_dropped_row():
    p, g = _squares()
    empty = torch.zeros_like(p)
    pred, gt = _maps(empty, g)
    out = nm.normal_map_metrics(pred, gt, bg_rgb=(1.0, 1.0, 1.0))
    assert out["ang_mean"] is None and out["psnr_ov"] is None
    assert out["psnr_full"] is not None and out["mask_iou"] == 0.0


def test_lpips_plumbing_uses_the_overlap_gray_images():
    seen = []

    def fake_lpips(a, b):
        seen.append((a.clone(), b.clone()))
        return 0.25

    p, g = _squares()
    pred, gt = _maps(p, g)
    out = nm.normal_map_metrics(pred, gt, bg_rgb=(1.0, 1.0, 1.0), lpips_main=fake_lpips, lpips_secondary=fake_lpips)
    assert out["lpips_full"] == out["lpips_full_vgg"] == out["lpips_ov"] == 0.25
    a_ov, _ = seen[-1]                                    # overlap call: 0.5 gray outside the overlap
    outside = ~(p & g)
    assert torch.allclose(a_ov[outside], torch.full_like(a_ov[outside], 0.5))
