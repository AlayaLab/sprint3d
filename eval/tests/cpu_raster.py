"""cpu_raster.py -- a small CPU stand-in for metrics.render.Raster (nvdiffrast), used only by the tests.  Same interface:
  rasterize(clip [V,4], faces [F,3], res) -> (rast [H,W,4] = (u, v, z_ndc, tri_id + 1), faces_int32)   rows bottom-up
  interpolate(attr [V,C], rast, faces)    -> [H,W,C]   perspective-correct, like nvdiffrast
  texture(tex [h,w,3], uv [H,W,2])        -> [H,W,3]   bilinear
Pixel centres are sampled at ndc = (i + 0.5) / res * 2 - 1; no near-plane clipping (triangles with a vertex behind the
camera are dropped); no top-left rule.  Numbers agree with nvdiffrast up to edge pixels.
"""
from __future__ import annotations

import numpy as np
import torch


class CPURaster:
    def __init__(self, device="cpu"):
        self.device = torch.device("cpu")

    def rasterize(self, clip, faces, res):
        c = clip.detach().cpu().numpy().astype(np.float64)
        f = faces.detach().cpu().numpy().astype(np.int64)
        w = c[:, 3]
        ok = (w[f] > 1e-9).all(1)
        ndc = c[:, :3] / np.where(np.abs(w) > 1e-12, w, 1e-12)[:, None]
        px = (ndc[:, 0] + 1.0) * res / 2.0 - 0.5           # continuous pixel index (centre of pixel i at i)
        py = (ndc[:, 1] + 1.0) * res / 2.0 - 0.5
        X, Y, Z, W = px[f], py[f], ndc[:, 2][f], w[f]
        x0 = np.clip(np.floor(X.min(1)).astype(np.int64), 0, res - 1)
        x1 = np.clip(np.ceil(X.max(1)).astype(np.int64), 0, res - 1)
        y0 = np.clip(np.floor(Y.min(1)).astype(np.int64), 0, res - 1)
        y1 = np.clip(np.ceil(Y.max(1)).astype(np.int64), 0, res - 1)
        ok &= (X.max(1) >= 0) & (X.min(1) <= res - 1) & (Y.max(1) >= 0) & (Y.min(1) <= res - 1)
        area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
        ok &= np.abs(area) > 1e-12
        size = np.maximum(x1 - x0, y1 - y0) + 1
        frag_pix, frag_z, frag_tri, frag_u, frag_v = [], [], [], [], []
        K = 1
        while K <= 2 * res:
            sel = np.nonzero(ok & (size <= K) & (size > K // 2))[0]
            if len(sel):
                off = np.arange(K)
                oy, ox = np.meshgrid(off, off, indexing="ij")
                ox, oy = ox.ravel(), oy.ravel()
                step = max(1, int(4_000_000 // (K * K)))
                for s in range(0, len(sel), step):
                    t = sel[s:s + step]
                    gx = x0[t][:, None] + ox[None]
                    gy = y0[t][:, None] + oy[None]
                    inb = (gx <= x1[t][:, None]) & (gy <= y1[t][:, None])
                    Xa, Ya = X[t], Y[t]
                    a = area[t][:, None]
                    b0 = ((Xa[:, 1, None] - gx) * (Ya[:, 2, None] - gy) - (Xa[:, 2, None] - gx) * (Ya[:, 1, None] - gy)) / a
                    b1 = ((Xa[:, 2, None] - gx) * (Ya[:, 0, None] - gy) - (Xa[:, 0, None] - gx) * (Ya[:, 2, None] - gy)) / a
                    b2 = 1.0 - b0 - b1
                    cov = inb & (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
                    ti, pi = np.nonzero(cov)
                    if len(ti) == 0:
                        continue
                    tt = t[ti]
                    B0, B1, B2 = b0[ti, pi], b1[ti, pi], b2[ti, pi]
                    z = B0 * Z[tt, 0] + B1 * Z[tt, 1] + B2 * Z[tt, 2]
                    q0, q1, q2 = B0 / W[tt, 0], B1 / W[tt, 1], B2 / W[tt, 2]
                    qs = q0 + q1 + q2
                    frag_pix.append(gy[ti, pi] * res + gx[ti, pi])
                    frag_z.append(z)
                    frag_tri.append(tt)
                    frag_u.append(q0 / qs)
                    frag_v.append(q1 / qs)
            K *= 2
        rast = np.zeros((res * res, 4), np.float32)
        if frag_pix:
            pix = np.concatenate(frag_pix)
            z = np.concatenate(frag_z)
            tri = np.concatenate(frag_tri)
            u = np.concatenate(frag_u)
            v = np.concatenate(frag_v)
            keep = (z >= -1.0) & (z <= 1.0)
            pix, z, tri, u, v = pix[keep], z[keep], tri[keep], u[keep], v[keep]
            order = np.lexsort((z, pix))
            pix, z, tri, u, v = pix[order], z[order], tri[order], u[order], v[order]
            first = np.ones(len(pix), bool)
            first[1:] = pix[1:] != pix[:-1]
            rast[pix[first], 0] = u[first]
            rast[pix[first], 1] = v[first]
            rast[pix[first], 2] = z[first]
            rast[pix[first], 3] = tri[first] + 1
        return torch.from_numpy(rast.reshape(res, res, 4)), faces.to(torch.int32)

    def interpolate(self, attr, rast, faces_i):
        tri = rast[..., 3].long() - 1
        f = faces_i.long()[tri.clamp_min(0)]
        u, v = rast[..., 0:1], rast[..., 1:2]
        a = attr.to(torch.float32)
        out = u * a[f[..., 0]] + v * a[f[..., 1]] + (1.0 - u - v) * a[f[..., 2]]
        return torch.where((tri >= 0)[..., None], out, torch.zeros_like(out))

    def texture(self, tex, uv):
        import torch.nn.functional as Fnn
        g = (uv * 2.0 - 1.0)[None]
        t = tex.permute(2, 0, 1)[None]
        return Fnn.grid_sample(t, g, mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
