"""Tiny stand-ins for the Pixal3D flows, used by the training tests (CPU, no Pixal3D needed).

They have the layer names of the real flows that receive adapters (``blocks.<i>.self_attn.to_qkv``,
``...to_out``, ``mlp.mlp.0``, ``mlp.mlp.2``) and an adaLN modulation layer that must be skipped.
The reference numbers in ``test_train_objective.py`` were produced by the original training code on
exactly these models and data.
"""
import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.to_qkv = nn.Linear(c, 3 * c)
        self.self_attn.to_out = nn.Linear(c, c)
        self.mlp = nn.Module()
        self.mlp.mlp = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))

    def forward(self, h, temb, ctx):
        shift, scale = self.adaLN_modulation(temb).chunk(2, -1)
        x = h * (1 + scale) + shift
        q, k, v = self.self_attn.to_qkv(x).chunk(3, -1)
        h = h + self.self_attn.to_out(torch.tanh(q * k) * v + ctx)
        return h + self.mlp.mlp(h)


class TinyDenseFlow(nn.Module):
    """Dense ``[1, 8, R, R, R]`` in and out, like the structure flow."""

    def __init__(self, c=8, nb=2, cdim=6):
        super().__init__()
        self.t_embed = nn.Linear(1, c)
        self.cond_proj = nn.Linear(cdim, c)
        self.blocks = nn.ModuleList([TinyBlock(c) for _ in range(nb)])

    def forward(self, dense, t1000, cond):
        c = dense.shape[1]
        h = dense.reshape(c, -1).t()
        temb = self.t_embed((t1000 / 1000.0).reshape(1, 1))
        ctx = self.cond_proj(cond["global"].mean(1)) + cond["proj"].mean(1)[:, :c]
        for b in self.blocks:
            h = b(h, temb, ctx)
        return h.t().reshape(dense.shape)


class Sparse:
    """Minimal sparse tensor: ``feats`` [N, C], ``coords`` [N, 4], batch size 1."""

    def __init__(self, feats, coords):
        self.feats, self.coords = feats, coords
        self.shape = (1, feats.shape[1])

    def replace(self, feats):
        return Sparse(feats, self.coords)


class TinySparseFlow(nn.Module):
    """Sparse latent in and out with ``concat_cond``, like the texture flow."""

    def __init__(self, c=32, nb=2, cdim=6):
        super().__init__()
        self.t_embed = nn.Linear(1, c)
        self.cond_proj = nn.Linear(cdim, c)
        self.in_proj = nn.Linear(2 * c, c)
        self.blocks = nn.ModuleList([TinyBlock(c) for _ in range(nb)])

    def forward(self, x, t1000, cond, concat_cond=None):
        h = self.in_proj(torch.cat([x.feats, concat_cond.feats], -1))
        temb = self.t_embed((t1000 / 1000.0).reshape(1, 1))
        ctx = self.cond_proj(cond["global"].mean(1))
        for b in self.blocks:
            h = b(h, temb, ctx)
        return x.replace(h)


class TinyShapeFlow(nn.Module):
    """Sparse latent in and out with ``{"global", "proj"}`` conditioning on the latent's coordinates, like
    the coarse- and fine-shape flows (no ``concat_cond``)."""

    def __init__(self, c=32, nb=2, cdim=6, pdim=12):
        super().__init__()
        self.t_embed = nn.Linear(1, c)
        self.cond_proj = nn.Linear(cdim, c)
        self.proj_in = nn.Linear(pdim, c)
        self.blocks = nn.ModuleList([TinyBlock(c) for _ in range(nb)])

    def forward(self, x, t1000, cond, concat_cond=None):
        h = x.feats + self.proj_in(cond["proj"].feats)
        temb = self.t_embed((t1000 / 1000.0).reshape(1, 1))
        ctx = self.cond_proj(cond["global"].mean(1))
        for b in self.blocks:
            h = b(h, temb, ctx)
        return x.replace(h)


def shape_objects(gen, n_objects=5, cdim=6, pdim=12):
    """Random supports and conditioning of ``n_objects`` objects (coords on a 16^3 grid, batch column 0)."""
    out = {}
    for k in range(n_objects):
        n = 40 + 9 * k
        xyz = torch.randint(0, 16, (n, 3), generator=gen, dtype=torch.int32)
        coords = torch.cat([torch.zeros(n, 1, dtype=torch.int32), xyz], 1)
        out[f"obj{k:02d}"] = {"coords": coords, "proj": torch.randn(n, pdim, generator=gen),
                              "global": torch.randn(1, 5, cdim, generator=gen)}
    return out


def structure_batches(gen, n_updates=3, accum=4, res=16):
    out = []
    for u in range(n_updates):
        batch = []
        for a in range(accum):
            ti, tj = [(1.0, 0.9), (0.9, 0.75), (0.75, 0.6), (0.6, 0.0)][(u + a) % 4]
            x = torch.randn(res ** 3, 8, generator=gen)
            tgt = torch.randn(res ** 3, 8, generator=gen)
            g = torch.randn(1, 5, 6, generator=gen)
            pj = torch.randn(1, 7, 12, generator=gen)
            batch.append((x, ti, tj, tgt, {"global": g, "proj": pj}))
        out.append(batch)
    return out


def texture_batches(gen, n_updates=3, accum=4):
    from fastpixal3d.training.schedule import texture_initial_noise

    out = []
    for u in range(n_updates):
        batch = []
        for a in range(accum):
            n = 50 + 7 * ((u * accum + a) % 5)
            i, j = [(0, 6), (6, 12)][(u + a) % 2]
            ti, tj = [1.0, 0.75, 0.0][i // 6], [1.0, 0.75, 0.0][j // 6]
            crd = torch.cat([torch.zeros(n, 1, dtype=torch.int32),
                             torch.randint(0, 64, (n, 3), generator=gen, dtype=torch.int32)], 1)
            x = texture_initial_noise(n) if i == 0 else torch.randn(n, 32, generator=gen)
            batch.append({"coords": crd, "x": x, "ti": ti, "tj": tj, "target": torch.randn(n, 32, generator=gen),
                          "concat": torch.randn(n, 32, generator=gen), "global": torch.randn(1, 5, 6, generator=gen),
                          "i": i})
        out.append(batch)
    return out
