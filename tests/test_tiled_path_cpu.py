"""CPU tests of the tiled mesh path: chart splitting, tile planning, gather and native build selection."""
from types import SimpleNamespace

import pytest
import torch

from fastpixal3d.config import Levers
from fastpixal3d.mesh import tiled_path as T


def _grid_chart(nx, ny, offset=(0.0, 0.0, 0.0), scale=1.0):
    xs, ys = torch.meshgrid(torch.arange(nx + 1, dtype=torch.float32), torch.arange(ny + 1, dtype=torch.float32),
                            indexing="ij")
    v = torch.stack([xs.reshape(-1) * scale, ys.reshape(-1) * scale, torch.zeros((nx + 1) * (ny + 1))], 1)
    v = v + torch.tensor(offset)
    idx = torch.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = torch.cat([torch.stack([a, b, c], -1).reshape(-1, 3), torch.stack([b, d, c], -1).reshape(-1, 3)])
    return v, f.to(torch.int32)


def _gpu_ctx(sizes, texture_size=256):
    """A gpu_ctx as geometry_gpu_phase(gpu_charts=False) returns it, built from grid charts."""
    verts, faces, vmaps, voff, foff = [], [], [], [0], [0]
    base = 0
    for k, (nx, ny) in enumerate(sizes):
        v, f = _grid_chart(nx, ny, offset=(0.0, 0.0, float(k)), scale=0.5 + 0.1 * k)
        verts.append(v)
        faces.append(f + voff[-1])          # chart faces index chart_vertices (global positions)
        vmaps.append(torch.arange(v.shape[0]) + base)
        base += v.shape[0]
        voff.append(voff[-1] + v.shape[0])
        foff.append(foff[-1] + f.shape[0])
    chart_vertices = torch.cat(verts)
    return {"num_charts": len(sizes), "chart_vertices": chart_vertices, "chart_faces": torch.cat(faces),
            "chart_vertex_offset": torch.tensor(voff), "chart_face_offset": torch.tensor(foff),
            "chart_vmap": torch.cat(vmaps), "new_vertices_cpu": chart_vertices.clone(), "texture_size": texture_size}


class FakeAtlas:
    """Deterministic stand-in for cumesh.xatlas.Atlas: one chart per mesh, laid out on a diagonal."""

    def __init__(self):
        self.meshes = []

    def add_mesh(self, vertices, faces):
        assert vertices.dtype == torch.float32 and faces.dtype == torch.int32
        self.meshes.append((vertices, faces))

    def compute_charts(self):
        pass

    def pack_charts(self, resolution, padding):
        self.resolution, self.padding = resolution, padding

    def get_mesh(self, k):
        v, f = self.meshes[k]
        xy = v[:, :2] - v[:, :2].min(0).values
        xy = xy / xy.max().clamp_min(1e-6)
        uv = (xy + k) / (len(self.meshes) + 1)
        return torch.arange(v.shape[0], dtype=torch.int32), f.clone(), uv.to(torch.float32)


def test_face_area_slices_equal_mesh_area_bitwise():
    torch.manual_seed(0)
    v = torch.randn(200, 3)
    f = torch.randint(0, 200, (300, 3), dtype=torch.int32)
    f[0] = f[0, 0]  # a degenerate face
    norms = T.face_area_norms(v, f)
    for a, b in ((0, 1), (0, 300), (17, 140), (140, 300)):
        assert float(norms[a:b].sum().item() * 0.5) == T.mesh_area(v, f[a:b])


def test_split_chart_respects_both_caps():
    v, f = _grid_chart(40, 30)
    vmap = torch.arange(v.shape[0])
    total = T.mesh_area(v, f)
    pieces = T.split_chart(v, f, vmap, max_faces=500, max_area=total / 5)
    assert sum(p[1].shape[0] for p in pieces) == f.shape[0]
    for pv, pf, pm in pieces:
        assert pf.shape[0] <= 500 and T.mesh_area(pv, pf) <= total / 5 + 1e-9
        assert torch.equal(v[pm], pv)  # vmap points back to the chart's vertices


def test_uv_phase_tiles_and_fast_split_gather_is_identical():
    ctx = _gpu_ctx([(30, 20), (5, 5), (70, 40), (3, 2), (12, 12)] + [(4, 4)] * 20)
    outs = []
    for fast in (False, True):
        timings = {}
        out = T.uv_phase(ctx, timings, fast_split_gather=fast, atlas_factory=FakeAtlas)
        outs.append(out)
        uv = timings["uv"]
        assert uv["grid"] == 4 and uv["tile_res"] == 64 and uv["num_charts"] == 25
        assert uv["fast_split_gather"] is fast and uv["total_s"] >= 0
    a, b = outs
    for key in ("out_vertices", "out_faces", "out_uvs", "out_vmaps"):
        assert torch.equal(a[key], b[key]), key
    assert torch.equal(a["out_vertices"], ctx["new_vertices_cpu"][a["out_vmaps"]])
    assert a["out_uvs"].min() >= 0 and a["out_uvs"].max() <= 1
    assert int(a["out_faces"].max()) < a["out_vertices"].shape[0]
    n_faces = int(ctx["chart_face_offset"][-1])
    assert a["out_faces"].shape == (n_faces, 3)


def test_plan_tiles_balances_faces_over_sixteen_groups():
    ctx = _gpu_ctx([(10, 10)] * 40)
    tasks, groups, grid, info = T.plan_tiles(ctx["chart_vertices"], ctx["chart_faces"], ctx["chart_vertex_offset"],
                                             ctx["chart_face_offset"], ctx["chart_vmap"], 40, uv_workers=16)
    assert grid == 4 and len(groups) == 16 and sorted(sum(groups, [])) == list(range(len(tasks)))
    assert max(info["group_faces"]) - min(info["group_faces"]) <= 200


def test_host_bvh_build_selection():
    assert T.host_bvh_build(False, False) == "openmp"
    assert T.host_bvh_build(True, False) == "openmp_nogil"
    assert T.host_bvh_build(True, True) == "openmp_pinned"
    assert T.host_bvh_build(False, True) == "openmp_pinned"
    lv = Levers.baseline()
    assert T.bvh_impl_name(lv) == "host_openmp"
    assert T.bvh_impl_name(lv.with_enabled(["mesh_prep_worker_thread"])) == "host_openmp_nogil"
    assert T.bvh_impl_name(lv.with_enabled(["gpu_bvh_build"])) == "gpu"


def test_missing_legacy_build_names_the_build_script(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name.startswith("fastpixal3d_bvh_"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(T.importlib, "import_module", fake)
    with pytest.raises(ImportError, match="build_legacy.sh"):
        T.bvh_class(Levers.baseline())


def test_fast_chart_kwargs_extend_the_stock_ones():
    calls = []
    mesh = SimpleNamespace(compute_charts=lambda **kw: calls.append(kw),
                           read=lambda: (torch.zeros(3, 3), torch.zeros(1, 3, dtype=torch.int32)),
                           read_atlas_charts=lambda: (1, None, torch.arange(3), torch.zeros(1, 3, dtype=torch.int32),
                                                      torch.tensor([0, 3]), torch.tensor([0, 1])))
    T.compute_charts(mesh, fast=False)
    T.compute_charts(mesh, fast=True)
    assert calls[0] == T.CHART_KWARGS
    assert calls[1] == {**T.CHART_KWARGS, **T.FAST_CHART_KWARGS}
