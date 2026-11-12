"""GPU UV atlas: single-pass chart segmentation, projection parameterisation and
shelf packing into one texture atlas (lever ``gpu_chart_segmentation``).

Pipeline (batched torch ops on the GPU, apart from the shelf packer's
sequential loop which runs on host arrays):

1. Segmentation. Every face is assigned to the nearest of 26 fixed
   directions (axes, edges and corners of a cube) by one argmax over
   ``normals @ directions.T``; faces of the same direction bucket are split
   into connected components (union-find by min-label hooking and pointer
   jumping). Sliver charts of at most 32 faces are then folded into the
   neighbouring chart they share the longest world-space boundary with (if
   that chart's mean normal is within 45 degrees), for up to 14 rounds.
   Charts with more than 1,500 faces are cut in half along their in-plane axis
   of larger area-weighted variance, for up to 6 rounds (all oversized charts
   in the same round). Finally charts smaller than 8 faces are absorbed by a
   neighbour whose axis is within 70 degrees, and every chart is made
   connected.
2. Parameterisation. Each chart is projected orthographically onto the plane
   orthogonal to its area-weighted mean normal; the in-plane frame is the one
   of 16 rotations (0 to 90 degrees) with the smallest bounding box.
3. Packing. One texel density for all charts, chosen so that the padded
   rectangles fill 90 % of the atlas (2-texel gutter, 1 texel for charts of
   at most 4 faces), rectangles rotated so that width >= height, shelf
   next-fit-decreasing-height packing with a scale search.
4. Verification. The packed layout is rasterised at atlas resolution; charts
   that fold over themselves in projection are split (depth layer or normal
   sign) and steps 2-3 repeated, up to 6 passes.

:func:`uv_phase` consumes the tensors the mesh GPU phase produced and returns
output vertices, faces, UVs and the output-to-input vertex map.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class AtlasConfig:
    pad: int = 2
    pad_small: int = 1
    small_chart_faces: int = 4
    fill_target: float = 0.90
    absorb_angle_deg: float = 70.0
    absorb_iters: int = 4
    min_chart_faces: int = 8
    angle_candidates: int = 16
    fold_rounds: int = 2
    max_rounds: int = 6
    fold_dilate: int = 1
    max_cc_iters: int = 64
    check_overlap: bool = True
    scale_iters: int = 6


@dataclass
class SegmentConfig:
    bucket_dirs: int = 26
    merge_small_chart_faces: int = 32
    merge_absorb_angle_deg: float = 45.0
    merge_max_rounds: int = 14
    max_chart_faces: int = 1500
    max_split_rounds: int = 6
    max_cc_iters: int = 64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def face_normals_areas(P: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tri = P[faces]
    c = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = c.norm(dim=-1)
    return c / n.clamp_min(1e-20)[:, None], 0.5 * n


def face_adjacency(faces: torch.Tensor, V: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairs of faces sharing an edge (for non-manifold edges, consecutive pairs in sorted order)."""
    a = faces.reshape(-1)
    b = faces.roll(-1, dims=1).reshape(-1)
    key = torch.minimum(a, b) * V + torch.maximum(a, b)
    order = torch.argsort(key)
    ks = key[order]
    same = ks[1:] == ks[:-1]
    fa = torch.div(order[:-1][same], 3, rounding_mode="floor")
    fb = torch.div(order[1:][same], 3, rounding_mode="floor")
    keep = fa != fb
    return fa[keep], fb[keep]


def edge_adjacency_with_length(P: torch.Tensor, faces: torch.Tensor):
    """Like :func:`face_adjacency`, plus the world-space length of each shared edge."""
    V = P.shape[0]
    a = faces.reshape(-1)
    b = faces.roll(-1, dims=1).reshape(-1)
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    key = lo * V + hi
    order = torch.argsort(key)
    ks = key[order]
    same = ks[1:] == ks[:-1]
    fa = torch.div(order[:-1][same], 3, rounding_mode="floor")
    fb = torch.div(order[1:][same], 3, rounding_mode="floor")
    keep = fa != fb
    fa, fb = fa[keep], fb[keep]
    e_idx = order[:-1][same][keep]
    elen = (P[lo[e_idx]] - P[hi[e_idx]]).norm(dim=-1)
    return fa, fb, elen


def connected_components(F: int, fa: torch.Tensor, fb: torch.Tensor, key: Optional[torch.Tensor],
                         max_iters: int = 64) -> Tuple[torch.Tensor, int]:
    """Dense component id per face; edges only connect faces with equal ``key``."""
    if key is not None:
        m = key[fa] == key[fb]
        fa, fb = fa[m], fb[m]
    comp = torch.arange(F, device=fa.device)
    iters = 0
    for iters in range(1, max_iters + 1):
        ca, cb = comp[fa], comp[fb]
        lo, hi = torch.minimum(ca, cb), torch.maximum(ca, cb)
        changed = lo != hi
        if not bool(changed.any()):
            break
        comp.scatter_reduce_(0, hi[changed], lo[changed], reduce="amin")
        comp = comp[comp]
        comp = comp[comp]
    _, dense = torch.unique(comp, return_inverse=True)
    return dense, iters


def bucket_directions(n: int, device, dtype) -> torch.Tensor:
    if n == 6:
        d = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=dtype)
    else:
        g = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
        d = torch.cartesian_prod(g, g, g)
        d = d[(d != 0).any(dim=1)]
        d = _normalize(d)
    return d.to(device)


def chart_axes(fn: torch.Tensor, area: torch.Tensor, label: torch.Tensor, C: int) -> torch.Tensor:
    acc = torch.zeros(C, 3, device=fn.device, dtype=fn.dtype)
    acc.index_add_(0, label, fn * area[:, None])
    bad = acc.norm(dim=-1) < 1e-12  # normals cancel exactly: fall back to the summed unit normals
    if bool(bad.any()):
        first = torch.zeros(C, 3, device=fn.device, dtype=fn.dtype)
        first.index_add_(0, label, fn)
        acc = torch.where(bad[:, None], first, acc)
    return _normalize(acc)


def absorb_faces(label: torch.Tensor, flag: torch.Tensor, fn: torch.Tensor, area: torch.Tensor,
                 fa: torch.Tensor, fb: torch.Tensor, cos_min: float, iters: int) -> torch.Tensor:
    """Label propagation: flagged faces adopt the label of an unflagged neighbour whose
    chart axis best matches their normal (if within ``cos_min``), layer by layer."""
    F = label.shape[0]
    device = label.device
    ea = torch.cat([fa, fb]); eb = torch.cat([fb, fa])
    valid = fn.norm(dim=-1) > 0.5
    for _ in range(iters):
        if not bool(flag.any()):
            break
        C = int(label.max().item()) + 1
        axis = chart_axes(fn, area, label, C)
        m = flag[ea] & ~flag[eb]
        if not bool(m.any()):
            break
        f, g = ea[m], eb[m]
        score = (fn[f] * axis[label[g]]).sum(-1)
        score = torch.where(valid[f], score, torch.ones_like(score))
        ok = score >= cos_min
        f, g, score = f[ok], g[ok], score[ok]
        if f.numel() == 0:
            break
        best = torch.full((F,), -2.0, device=device, dtype=score.dtype).scatter_reduce_(0, f, score, "amax")
        pick = score >= best[f]
        f, g = f[pick], g[pick]
        new_label = label.clone()
        new_label[f] = label[g]
        adopted = torch.zeros(F, dtype=torch.bool, device=device)
        adopted[f] = True
        label = new_label
        flag = flag & ~adopted
    return label


# ---------------------------------------------------------------------------
# 1. segmentation
# ---------------------------------------------------------------------------
def merge_small_charts(P: torch.Tensor, faces: torch.Tensor, label: torch.Tensor, small_chart_faces: int,
                       absorb_angle_deg: float, max_rounds: int, stats: Dict[str, Any]) -> torch.Tensor:
    """Fold every chart with <= ``small_chart_faces`` faces into its longest-boundary,
    normal-compatible neighbour (one hop per round; merges only go towards a chart that
    is not small or has a larger id, so no swap cycles)."""
    device = label.device
    F = faces.shape[0]
    fn, area = face_normals_areas(P, faces)
    fa, fb, elen = edge_adjacency_with_length(P, faces)
    _, label = torch.unique(label, return_inverse=True)
    charts_before = int(label.max().item()) + 1
    cos_min = math.cos(math.radians(absorb_angle_deg))
    faces_folded = 0
    rounds_used = 0
    for rnd in range(max_rounds):
        rounds_used = rnd + 1
        C = int(label.max().item()) + 1
        fc = torch.bincount(label, minlength=C)
        small = fc <= small_chart_faces
        if not bool(small.any()):
            rounds_used = rnd
            break
        axis = chart_axes(fn, area, label, C)
        ca, cb = label[fa], label[fb]
        cross = ca != cb
        ca, cb, w = ca[cross], cb[cross], elen[cross]
        src = torch.cat([ca, cb]); dst = torch.cat([cb, ca]); wv = torch.cat([w, w])
        m = small[src]
        src, dst, wv = src[m], dst[m], wv[m]
        if src.numel() == 0:
            break
        key = src * C + dst
        uk, inv = torch.unique(key, return_inverse=True)
        tot_w = torch.zeros(uk.shape[0], device=device, dtype=wv.dtype).index_add_(0, inv, wv)
        u_src = torch.div(uk, C, rounding_mode="floor")
        u_dst = uk - u_src * C
        cosang = (axis[u_src] * axis[u_dst]).sum(-1)
        ok = cosang >= cos_min
        masked = torch.where(ok, tot_w, torch.full_like(tot_w, -1.0))
        best = torch.full((C,), -1.0, device=device, dtype=wv.dtype).scatter_reduce_(0, u_src, masked, "amax")
        pick = (best[u_src] >= 0) & (masked >= best[u_src])
        if not bool(pick.any()):
            break
        p_src, p_dst = u_src[pick], u_dst[pick]
        keep = (~small[p_dst]) | (p_dst > p_src)
        p_src, p_dst = p_src[keep], p_dst[keep]
        if p_src.numel() == 0:
            break
        chart_map = torch.arange(C, device=device)
        chart_map[p_src] = p_dst
        new_label = chart_map[label]
        moved = new_label != label
        if not bool(moved.any()):
            break
        faces_folded += int(moved.sum().item())
        _, label = torch.unique(new_label, return_inverse=True)
    label, _ = connected_components(F, fa, fb, label)
    stats.update({"merge_charts_before": charts_before, "merge_charts_after": int(label.max().item()) + 1,
                  "merge_faces_folded": faces_folded, "merge_rounds_used": rounds_used})
    return label


def segment_charts(P: torch.Tensor, faces: torch.Tensor, cfg: SegmentConfig, stats: Dict[str, Any]) -> torch.Tensor:
    """P [V,3], faces [F,3] int64 -> dense chart id per face."""
    device = P.device
    F = faces.shape[0]
    t0 = time.perf_counter()
    fn, area = face_normals_areas(P, faces)
    fa, fb = face_adjacency(faces, P.shape[0])
    dirs = bucket_directions(cfg.bucket_dirs, fn.device, fn.dtype)
    bucket = torch.argmax(fn @ dirs.T, dim=1)
    label, cc_iters = connected_components(F, fa, fb, bucket, cfg.max_cc_iters)
    stats["segment_bucket_charts"] = int(label.max().item()) + 1
    stats["segment_bucket_cc_iters"] = cc_iters

    label = merge_small_charts(P, faces, label, cfg.merge_small_chart_faces, cfg.merge_absorb_angle_deg,
                               cfg.merge_max_rounds, stats)

    centroid = P[faces].mean(dim=1)
    split_rounds_used = 0
    for rnd in range(cfg.max_split_rounds):
        C = int(label.max().item()) + 1
        fc = torch.bincount(label, minlength=C)
        big = fc[label] > cfg.max_chart_faces
        if not bool(big.any()):
            break
        split_rounds_used = rnd + 1
        axis = chart_axes(fn, area, label, C)
        helper = torch.nn.functional.one_hot(axis.abs().argmin(dim=1), 3).to(P.dtype)
        e1 = _normalize(torch.cross(axis, helper, dim=-1))
        e2 = torch.cross(axis, e1, dim=-1)
        u = (centroid * e1[label]).sum(-1)
        v = (centroid * e2[label]).sum(-1)
        w = area
        sw = torch.zeros(C, device=device, dtype=P.dtype).index_add_(0, label, w).clamp_min(1e-30)
        mu = torch.zeros(C, device=device, dtype=P.dtype).index_add_(0, label, w * u) / sw
        mv = torch.zeros(C, device=device, dtype=P.dtype).index_add_(0, label, w * v) / sw
        varu = (torch.zeros(C, device=device, dtype=P.dtype).index_add_(0, label, w * u * u) / sw - mu * mu).clamp_min(0)
        varv = (torch.zeros(C, device=device, dtype=P.dtype).index_add_(0, label, w * v * v) / sw - mv * mv).clamp_min(0)
        use_u = varu >= varv
        thresh = torch.where(use_u, mu, mv)[label]
        val = torch.where(use_u[label], u, v)
        side = val > thresh
        new_label = label * 2 + (big & side).long()
        _, label = torch.unique(new_label, return_inverse=True)
    stats["segment_split_rounds_used"] = split_rounds_used
    stats["segment_s"] = round(time.perf_counter() - t0, 6)
    stats["segment_charts_out"] = int(label.max().item()) + 1
    return label


def finalize_charts(label: torch.Tensor, fn: torch.Tensor, area: torch.Tensor, fa: torch.Tensor, fb: torch.Tensor,
                    cfg: AtlasConfig, stats: Dict[str, Any]) -> torch.Tensor:
    """Absorb charts smaller than ``min_chart_faces`` into neighbours, then make every chart connected."""
    F = label.shape[0]
    cos_absorb = math.cos(math.radians(cfg.absorb_angle_deg))
    if cfg.min_chart_faces > 1:
        C = int(label.max().item()) + 1
        fc = torch.bincount(label, minlength=C)
        tiny = (fc[label] < cfg.min_chart_faces)
        n_tiny = int(tiny.sum().item())
        if n_tiny:
            label = absorb_faces(label, tiny, fn, area, fa, fb, cos_absorb, cfg.absorb_iters)
    label, _ = connected_components(F, fa, fb, label, cfg.max_cc_iters)
    return label


# ---------------------------------------------------------------------------
# 2. parameterisation
# ---------------------------------------------------------------------------
def parameterize(P: torch.Tensor, faces: torch.Tensor, label: torch.Tensor, fn: torch.Tensor, area: torch.Tensor,
                 cfg: AtlasConfig, frame: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    """Orthographic projection per chart. ``frame``: per-face frozen axis/angle from the
    previous pass; a chart whose faces are all frozen keeps its projection."""
    device = P.device
    V = P.shape[0]
    C = int(label.max().item()) + 1
    axis = chart_axes(fn, area, label, C)
    frozen_c = None
    if frame is not None:
        unfrozen = torch.zeros(C, dtype=torch.long, device=device).index_add_(0, label, (~frame["frozen"]).long())
        frozen_c = unfrozen == 0
        axis_prev = torch.zeros(C, 3, device=device, dtype=axis.dtype)
        axis_prev[label] = frame["axis"]
        axis = torch.where(frozen_c[:, None], axis_prev, axis)
    helper = torch.nn.functional.one_hot(axis.abs().argmin(dim=1), 3).to(P.dtype)
    e1 = _normalize(torch.cross(axis, helper, dim=-1))
    e2 = torch.cross(axis, e1, dim=-1)

    key = label[:, None] * V + faces
    uniq, inv = torch.unique(key.reshape(-1), return_inverse=True)
    out_faces = inv.reshape(-1, 3)
    out_chart = torch.div(uniq, V, rounding_mode="floor")
    out_vid = uniq - out_chart * V
    pos = P[out_vid]
    u0 = (pos * e1[out_chart]).sum(-1)
    v0 = (pos * e2[out_chart]).sum(-1)
    depth = (pos * axis[out_chart]).sum(-1)

    K = max(1, int(cfg.angle_candidates))
    ang = torch.arange(K, device=device, dtype=P.dtype) * (math.pi / 2 / K)
    ca, sa = torch.cos(ang), torch.sin(ang)
    ur = u0[:, None] * ca - v0[:, None] * sa
    vr = u0[:, None] * sa + v0[:, None] * ca
    idx = (out_chart[:, None] * K + torch.arange(K, device=device)).reshape(-1)
    inf = float("inf")
    umin = torch.full((C * K,), inf, device=device, dtype=P.dtype).scatter_reduce_(0, idx, ur.reshape(-1), "amin")
    umax = torch.full((C * K,), -inf, device=device, dtype=P.dtype).scatter_reduce_(0, idx, ur.reshape(-1), "amax")
    vmin = torch.full((C * K,), inf, device=device, dtype=P.dtype).scatter_reduce_(0, idx, vr.reshape(-1), "amin")
    vmax = torch.full((C * K,), -inf, device=device, dtype=P.dtype).scatter_reduce_(0, idx, vr.reshape(-1), "amax")
    w_all = (umax - umin).reshape(C, K)
    h_all = (vmax - vmin).reshape(C, K)
    best = torch.argmin(w_all * h_all, dim=1)
    if frozen_c is not None:
        best_prev = torch.zeros(C, dtype=torch.long, device=device)
        best_prev[label] = frame["best"]
        best = torch.where(frozen_c, best_prev, best)
    sel = torch.arange(C, device=device) * K + best
    w = w_all.reshape(-1)[sel]
    h = h_all.reshape(-1)[sel]
    bv = best[out_chart]
    ar = torch.arange(out_chart.shape[0], device=device)
    lu = ur[ar, bv] - umin[sel][out_chart]
    lv = vr[ar, bv] - vmin[sel][out_chart]
    cosang = (fn * axis[label]).sum(-1)
    return {"C": C, "axis": axis, "best": best, "frozen_c": frozen_c, "out_faces": out_faces, "out_chart": out_chart,
            "out_vid": out_vid, "lu": lu, "lv": lv, "umin": umin[sel], "vmin": vmin[sel],
            "depth": depth, "w": w, "h": h, "cosang": cosang}


# ---------------------------------------------------------------------------
# 3. packing
# ---------------------------------------------------------------------------
def shelf_pack(w: np.ndarray, h: np.ndarray, T: int, pad, rot=None):
    """Shelf next-fit-decreasing-height of integer rectangles into a T-wide strip.
    Returns x, y (origin of the padded cell), rot (rotated by 90 degrees) and the height used."""
    n = w.shape[0]
    rot = (h > w) if rot is None else np.asarray(rot, dtype=bool)
    w2 = np.where(rot, h, w)
    h2 = np.where(rot, w, h)
    pad = np.broadcast_to(np.asarray(pad, dtype=np.int64), (n,))
    cw = w2 + 2 * pad
    ch = h2 + 2 * pad
    order = np.argsort(-ch, kind="stable")
    cw_s = cw[order]
    ch_s = ch[order]
    csum = np.cumsum(cw_s)
    x = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    start, ycur = 0, 0
    while start < n:
        base = csum[start - 1] if start > 0 else 0
        end = int(np.searchsorted(csum, base + T, side="right"))
        end = max(end, start + 1)
        seg = order[start:end]
        x[seg] = csum[start:end] - cw_s[start:end] - base
        y[seg] = ycur
        ycur += int(ch_s[start])
        start = end
    return x, y, rot, ycur


def pack_charts(w_world: torch.Tensor, h_world: torch.Tensor, T: int, cfg: AtlasConfig, stats: Dict[str, Any],
                pad_c: Optional[torch.Tensor] = None, s_fixed: Optional[float] = None,
                rot_c: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Common texel density + shelf packing into a T x T atlas, with a scale search."""
    wf = w_world.detach().to(torch.float64)
    hf = h_world.detach().to(torch.float64)
    C = wf.shape[0]
    if pad_c is None:
        pad_c = torch.full((C,), int(cfg.pad), dtype=torch.int64, device=w_world.device)
    padf = pad_c.to(torch.float64)
    # s from  sum (s w + 2 pad)(s h + 2 pad) = fill * T^2
    A2 = float((wf * hf).sum()); A1 = float((2 * padf * (wf + hf)).sum()); A0 = float(((2 * padf) ** 2).sum())
    target = cfg.fill_target * T * T
    if A2 <= 0:
        s = 1.0
    else:
        disc = A1 * A1 - 4 * A2 * (A0 - target)
        s = (-A1 + math.sqrt(max(disc, 0.0))) / (2 * A2)
    if s_fixed is not None:
        s = float(s_fixed)
    best = None
    t_pack = 0.0
    wt, ht, pad = wf.cpu().numpy(), hf.cpu().numpy(), pad_c.cpu().numpy()
    rot_o = None if rot_c is None else rot_c.cpu().numpy()
    for it in range(cfg.scale_iters):
        t0 = time.perf_counter()
        # +1: chart origins are snapped to the texel grid, which can push content one texel further
        wi = np.maximum(np.ceil(wt * s).astype(np.int64), 1) + 1
        hi = np.maximum(np.ceil(ht * s).astype(np.int64), 1) + 1
        x, y, rot, used = shelf_pack(wi, hi, T, pad, rot=rot_o)
        max_cw = int((np.maximum(wi, hi) + 2 * pad).max()) if C else 0
        t_pack += time.perf_counter() - t0
        fits = used <= T and max_cw <= T
        if fits and (best is None or s > best["s"]):
            best = {"s": s, "x": x, "y": y, "rot": rot, "wi": wi, "hi": hi, "used": used, "iter": it}
        if fits:
            if used > 0.985 * T or s_fixed is not None:
                break
            s_new = s * math.sqrt(T / max(used, 1)) * 0.985
        else:
            s_new = s * math.sqrt(T / max(used, 1)) * 0.97
            if max_cw > T:
                s_new = min(s_new, s * (T / max_cw) * 0.97)
        s = s_new
    if best is None:
        raise RuntimeError("shelf packing failed to fit the atlas")
    stats["pack_iters"] = best["iter"] + 1
    stats["pack_s"] = round(stats.get("pack_s", 0.0) + t_pack, 6)
    stats["texels_per_unit"] = float(best["s"])
    return best


# ---------------------------------------------------------------------------
# 4. overlap check (pixel-centre sampling, top-left fill rule)
# ---------------------------------------------------------------------------
def rasterize_count(uv_tex: torch.Tensor, faces: torch.Tensor, T: int, max_pairs: int = 60_000_000):
    """Count, per texel centre, the triangles covering it. Returns (count [T*T],
    texel id per covered pair, face id per covered pair)."""
    device = uv_tex.device
    tri = uv_tex[faces]
    x0 = torch.floor(tri[..., 0].min(1).values).clamp(0, T - 1).to(torch.int64)
    x1 = torch.ceil(tri[..., 0].max(1).values).clamp(1, T).to(torch.int64)
    y0 = torch.floor(tri[..., 1].min(1).values).clamp(0, T - 1).to(torch.int64)
    y1 = torch.ceil(tri[..., 1].max(1).values).clamp(1, T).to(torch.int64)
    bw = (x1 - x0).clamp_min(1)
    bh = (y1 - y0).clamp_min(1)
    npx = bw * bh
    total = int(npx.sum().item())
    if total > max_pairs:
        raise RuntimeError(f"rasterize_count: too many candidate pairs ({total})")
    fid = torch.repeat_interleave(torch.arange(faces.shape[0], device=device), npx)
    off = torch.cumsum(npx, 0) - npx
    local = torch.arange(total, device=device) - off[fid]
    px = x0[fid] + local % bw[fid]
    py = y0[fid] + torch.div(local, bw[fid], rounding_mode="floor")
    cx = px.to(uv_tex.dtype) + 0.5
    cy = py.to(uv_tex.dtype) + 0.5
    t = tri[fid].to(torch.float64)
    cx = cx.to(torch.float64); cy = cy.to(torch.float64)
    ax, ay = t[:, 0, 0], t[:, 0, 1]
    bx, by = t[:, 1, 0], t[:, 1, 1]
    qx, qy = t[:, 2, 0], t[:, 2, 1]
    area2 = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
    sgn = torch.where(area2 < 0, -1.0, 1.0)

    def edge(x1_, y1_, x2_, y2_):
        e = ((x2_ - x1_) * (cy - y1_) - (y2_ - y1_) * (cx - x1_)) * sgn
        dx, dy = (x2_ - x1_) * sgn, (y2_ - y1_) * sgn
        topleft = (dy < 0) | ((dy == 0) & (dx > 0))
        return (e > 0) | ((e == 0) & topleft)

    inside = edge(ax, ay, bx, by) & edge(bx, by, qx, qy) & edge(qx, qy, ax, ay) & (area2 != 0)
    pid = (py * T + px)[inside]
    fin = fid[inside]
    count = torch.bincount(pid, minlength=T * T).to(torch.int32)
    return count, pid, fin


# ---------------------------------------------------------------------------
# full atlas
# ---------------------------------------------------------------------------
def build_atlas(P: torch.Tensor, faces: torch.Tensor, label: torch.Tensor, T: int, cfg: AtlasConfig,
                stats: Dict[str, Any]) -> Dict[str, Any]:
    """P [V,3] float32, faces [F,3] int64, label [F] initial chart id. Returns out_faces [F,3]
    int32, out_vid [Vo] int64 (output -> input vertex), uv [Vo,2] float32 in [0,1]^2."""
    device = P.device
    F = faces.shape[0]
    V = P.shape[0]
    t0 = time.perf_counter()
    fn, area = face_normals_areas(P, faces)
    fa, fb = face_adjacency(faces, V)
    _, label = torch.unique(label, return_inverse=True)
    stats["charts_in"] = int(label.max().item()) + 1
    label = finalize_charts(label, fn, area, fa, fb, cfg, stats)
    t1 = time.perf_counter()

    fold_splits = 0
    overlap_texels = None
    frame = None
    s = None
    rnd = 0
    for rnd in range(cfg.max_rounds):
        par = parameterize(P, faces, label, fn, area, cfg, frame)
        fcount = torch.bincount(label, minlength=par["C"])
        pad_c = torch.where(fcount <= cfg.small_chart_faces, torch.full_like(fcount, int(cfg.pad_small)),
                            torch.full_like(fcount, int(cfg.pad)))
        rot_c = None
        if frame is not None:
            rot_prev = torch.zeros(par["C"], dtype=torch.bool, device=device)
            rot_prev[label] = frame["rot"]
            rot_c = torch.where(par["frozen_c"], rot_prev, par["h"] > par["w"])
        pk = pack_charts(par["w"], par["h"], T, cfg, stats, pad_c, s_fixed=s, rot_c=rot_c)
        s = pk["s"]
        oc = par["out_chart"]
        x = torch.as_tensor(np.asarray(pk["x"]), device=device)
        y = torch.as_tensor(np.asarray(pk["y"]), device=device)
        rot = torch.as_tensor(np.asarray(pk["rot"]), device=device)
        # chart-local texel coordinates with the origin snapped to the texel grid, so a chart's
        # texel-space geometry is invariant across passes up to an integer translation
        ur = par["lu"] + par["umin"][oc]
        vr = par["lv"] + par["vmin"][oc]
        rv = rot[oc]
        au = torch.where(rv, -vr, ur) * s           # 90-degree rotation (u, v) -> (-v, u)
        av = torch.where(rv, ur, vr) * s
        omin_u = torch.full((par["C"],), float("inf"), device=device, dtype=P.dtype).scatter_reduce_(0, oc, au, "amin")
        omin_v = torch.full((par["C"],), float("inf"), device=device, dtype=P.dtype).scatter_reduce_(0, oc, av, "amin")
        lu2 = au - torch.floor(omin_u)[oc]
        lv2 = av - torch.floor(omin_v)[oc]
        padv = pad_c[oc].to(P.dtype)
        u_tex = (x[oc].to(P.dtype) + padv) + lu2
        v_tex = (y[oc].to(P.dtype) + padv) + lv2
        uv_tex = torch.stack([u_tex, v_tex], dim=1)
        frame = {"axis": par["axis"][label], "best": par["best"][label], "rot": rot[label],
                 "frozen": torch.ones(F, dtype=torch.bool, device=device)}
        if not cfg.check_overlap:
            break
        count, pid, fin = rasterize_count(uv_tex, par["out_faces"], T)
        over = count > 1
        overlap_texels = int(over.sum().item())
        if overlap_texels == 0 or rnd == cfg.max_rounds - 1:
            break
        if rnd >= cfg.fold_rounds:
            # resolve: every face still covering an overlapped texel becomes its own chart
            bad = torch.zeros(F, dtype=torch.bool, device=device)
            bad[fin[over[pid]]] = True
            key = label * (F + 1) + torch.where(bad, torch.arange(F, device=device) + 1, torch.zeros_like(label))
            _, label = torch.unique(key, return_inverse=True)
            frame["frozen"] = ~bad
            continue
        # fold fix restricted to the overlapping faces: the face farther along the chart axis is
        # the back layer; coplanar overlaps fall back to a normal-bucket split of those faces
        over_pairs = over[pid]
        p_o, f_o = pid[over_pairs], fin[over_pairs]
        fdepth = par["depth"][par["out_faces"]].mean(dim=1)
        dmax = torch.full((T * T,), -float("inf"), device=device, dtype=P.dtype).scatter_reduce_(0, p_o, fdepth[f_o], "amax")
        dmin = torch.full((T * T,), float("inf"), device=device, dtype=P.dtype).scatter_reduce_(0, p_o, fdepth[f_o], "amin")
        eps = 1e-7
        is_back = (fdepth[f_o] >= dmax[p_o] - eps) & (dmax[p_o] - dmin[p_o] > eps)
        back = torch.zeros(F, dtype=torch.bool, device=device)
        back[f_o[is_back]] = True
        flat = torch.zeros(F, dtype=torch.bool, device=device)
        flat[f_o[dmax[p_o] - dmin[p_o] <= eps]] = True
        flat &= ~back
        bad_charts = torch.zeros(par["C"], dtype=torch.bool, device=device)
        bad_charts[label[f_o]] = True
        fold_splits += int(bad_charts.sum().item())
        flipped = (par["cosang"] < 0) & bad_charts[label]
        has_flipped = torch.zeros(par["C"], dtype=torch.bool, device=device)
        has_flipped[label[flipped]] = True
        back = torch.where(has_flipped[label], flipped, back)
        for _ in range(cfg.fold_dilate):
            grow = torch.zeros(F, dtype=torch.bool, device=device)
            m = back[fa] & (label[fa] == label[fb]); grow[fb[m]] = True
            m = back[fb] & (label[fa] == label[fb]); grow[fa[m]] = True
            back |= grow
        dirs26 = bucket_directions(26, fn.device, fn.dtype)
        bucket = torch.where(flat, torch.argmax(fn @ dirs26.T, dim=1) + 1, torch.zeros_like(label))
        key = (label * 2 + back.to(label.dtype)) * 27 + bucket
        frame["frozen"] = ~(back | flat)
        label, _ = connected_components(F, fa, fb, key, cfg.max_cc_iters)
    t2 = time.perf_counter()
    uv = uv_tex / T
    C = par["C"]
    stats.update({"charts": C, "fold_splits": fold_splits, "passes": rnd + 1, "overlap_texels": overlap_texels})
    if cfg.check_overlap:
        stats["occupancy"] = round(float((count > 0).float().mean().item()), 4)
    # atlas quality statistics (projection stretch = 1/|cos(normal, chart axis)|, area weighted)
    validf = fn.norm(dim=-1) > 0.5
    cosabs = torch.where(validf, par["cosang"].abs(), torch.ones_like(par["cosang"])).clamp_min(1e-6)
    stretch = 1.0 / cosabs
    stats["stretch_mean"] = round(float((stretch * area).sum().item() / float(area.sum().item())), 4)
    stats["stretch_max"] = round(float(stretch.max().item()), 3)
    stats["stretch_p99"] = round(float(torch.quantile(stretch, 0.99).item()), 3) if F < 16_000_000 else None
    stats["flipped_faces"] = int(((par["cosang"] < 0) & validf).sum().item())
    fc = torch.bincount(label, minlength=C)
    stats["chart_faces_median"] = int(fc.median().item())
    stats["tiny_charts_le4"] = int((fc <= 4).sum().item())
    stats["out_vertices"] = int(par["out_vid"].shape[0])
    stats["refine_s"] = round(t1 - t0, 6)
    stats["param_pack_s"] = round(t2 - t1, 6)
    return {"out_faces": par["out_faces"].to(torch.int32), "out_vid": par["out_vid"],
            "uv": uv.to(torch.float32), "stats": stats}


def uv_phase(new_vertices_cpu: torch.Tensor, chart_faces: torch.Tensor, chart_vmap: torch.Tensor,
             texture_size: int, device="cuda", segment: Optional[SegmentConfig] = None,
             atlas: Optional[AtlasConfig] = None, use_stream: bool = True) -> Dict[str, Any]:
    """Segment, parameterise and pack on the GPU (on a private CUDA stream, synchronised
    before returning). Returns out_vertices / out_faces / out_uvs / out_vmaps (on the device)."""
    t0 = time.perf_counter()
    dev = torch.device(device)
    segment = segment or SegmentConfig()
    atlas = atlas or AtlasConfig()
    stats: Dict[str, Any] = {}

    def impl():
        P = new_vertices_cpu.to(dev, non_blocking=True)
        cf = chart_faces.to(dev, non_blocking=True)
        vm = chart_vmap.to(dev, non_blocking=True)
        faces = vm[cf.long()].long()
        label = segment_charts(P, faces, segment, stats)
        res = build_atlas(P, faces, label, int(texture_size), atlas, stats)
        out_vid = res["out_vid"]
        out_vertices = P[out_vid]
        if dev.type == "cuda":
            torch.cuda.current_stream(dev).synchronize()
        return {"out_vertices": out_vertices, "out_faces": res["out_faces"], "out_uvs": res["uv"],
                "out_vmaps": out_vid}

    if dev.type == "cuda" and use_stream:
        stream = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(stream):
            out = impl()
        stream.synchronize()
    else:
        out = impl()
    stats["total_s"] = round(time.perf_counter() - t0, 6)
    stats["segment_config"] = asdict(segment)
    out["stats"] = stats
    return out
