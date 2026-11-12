"""Tiled mesh path: the UV atlas of rows 01-07 of the cumulative measurement (16 tiles, CPU xatlas).

Before ``gpu_chart_segmentation``, the export path built its UV atlas in two halves:

GPU half (part of the mesh GPU phase, :func:`fastpixal3d.mesh.bake.geometry_gpu_phase`)
    after simplification, CuMesh clusters the faces into charts on the GPU
    (:func:`compute_charts`: cone half-angle 90 degrees, no refinement, one global
    iteration) and the chart data are copied to host memory.
CPU half (:func:`uv_phase`, on a worker thread while the texture is sampled)
    oversized charts are cut by recursive median splits until each piece has at most
    ``max(10_000, F / 32)`` faces and at most 1/32 of the surface area; the pieces are
    distributed over 16 groups (largest first, each to the group with the fewest faces);
    every group is unwrapped by its own xatlas ``Atlas`` (compute_charts + pack_charts at
    1/4 of the texture size, 4 px padding) on a 16-thread pool, and the group atlases are
    placed as the tiles of a 4 x 4 grid.

The texture bake (:func:`fastpixal3d.mesh.bake.bake_and_finalize`, BVH reprojection to the
pre-remesh surface, trilinear volume sampling, GPU pull-push fill of the gutter) is the same
as in the deployed configuration.

Levers that act on this path only:

* ``per_tile_single_thread_unwrap``: the xatlas ``Atlas`` of each tile runs without its own
  internal thread pool (:func:`single_thread_xatlas`, legacy build ``fastpixal3d_xatlas_single``).
* ``pinned_bvh_transfers``: the host-built BVH is uploaded through pinned memory in chunks
  (legacy build ``fastpixal3d_bvh_openmp_pinned``), and the split and gather steps of
  :func:`uv_phase` compute the chart areas once and gather all tiles with one index
  (``fast_split_gather``; same decisions, same output).

The BVH of this path (:func:`bvh_class`) is built on the host by an OpenMP-parallel build
of cubvh. With ``mesh_prep_worker_thread`` the build that releases the GIL is used (the BVH
is then built on the worker thread while the main thread samples the texture), with
``pinned_bvh_transfers`` the build with pinned transfers, and with ``gpu_bvh_build`` the
GPU build of the deployed configuration. The three host builds and the single-thread
xatlas are produced by ``native/legacy/build_legacy.sh``; the path refuses to run without
the build a configuration asks for.
"""
from __future__ import annotations

import contextlib
import importlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch

UV_WORKERS = 16          # number of tiles (a 4 x 4 grid)
UV_CONCURRENCY = 16      # tile atlases unwrapped at the same time
UV_PADDING = 4           # xatlas pack padding (texels)
CHART_KWARGS = dict(threshold_cone_half_angle_rad=math.radians(90.0), refine_iterations=0,
                    global_iterations=1, smooth_strength=1)
# GPU chart clustering of the patched CuMesh build (fast_gpu_simplify): the round loop of the
# clustering runs on the device, with the compact sort key and fused round kernels.
FAST_CHART_KWARGS = dict(sync_free=True, max_rounds=4096, sync_every=4, compact_chart_bits=True,
                         skip_collapse_cone_writeback=True, fuse_round_kernels=True)

HOST_BVH_MODULES = {
    "openmp": "fastpixal3d_bvh_openmp",                 # OpenMP-parallel host build
    "openmp_nogil": "fastpixal3d_bvh_openmp_nogil",     # + GIL released during build and queries
    "openmp_pinned": "fastpixal3d_bvh_openmp_pinned",   # + pinned, chunked host-to-device copies
}
XATLAS_SINGLE_MODULE = "fastpixal3d_xatlas_single"


# ---------------------------------------------------------------------------
# native builds
# ---------------------------------------------------------------------------
def host_bvh_build(mesh_prep_worker_thread: bool, pinned_bvh_transfers: bool) -> str:
    """Which host BVH build the tiled path uses (keys of :data:`HOST_BVH_MODULES`)."""
    if pinned_bvh_transfers:
        return "openmp_pinned"
    if mesh_prep_worker_thread:
        return "openmp_nogil"
    return "openmp"


def _import_legacy(module: str, what: str):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(f"the tiled mesh path needs the legacy build {module!r} ({what}); "
                          "run native/legacy/build_legacy.sh (see native/legacy/README.md)") from error


def bvh_class(levers):
    """BVH class of the tiled path for a :class:`fastpixal3d.config.Levers`."""
    if levers.gpu_bvh_build:
        from .bvh import bvh_class as deployed

        return deployed(True)
    name = host_bvh_build(levers.mesh_prep_worker_thread, levers.pinned_bvh_transfers)
    return _import_legacy(HOST_BVH_MODULES[name], "host BVH build").cuBVH


def bvh_impl_name(levers) -> str:
    if levers.gpu_bvh_build:
        return "gpu"
    return "host_" + host_bvh_build(levers.mesh_prep_worker_thread, levers.pinned_bvh_transfers)


@contextlib.contextmanager
def single_thread_xatlas():
    """Within the block, new ``cumesh.xatlas.Atlas`` objects use the xatlas build without an
    internal thread pool (``per_tile_single_thread_unwrap``). Process-wide; one request at a time."""
    xa = importlib.import_module("cumesh.xatlas")
    private = _import_legacy(XATLAS_SINGLE_MODULE, "xatlas without internal threads")
    saved = xa._xatlas
    xa._xatlas = private
    try:
        yield {"inner_threads": 1, "module": XATLAS_SINGLE_MODULE}
    finally:
        xa._xatlas = saved


# ---------------------------------------------------------------------------
# GPU half: chart clustering
# ---------------------------------------------------------------------------
def compute_charts(mesh, fast: bool):
    """Cluster the faces of ``mesh`` (a CuMesh, degenerate faces already removed) into charts and
    return ``(new_vertices, num_charts, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset)``
    on the device. ``fast``: the device-side round loop of the patched CuMesh build."""
    kwargs = dict(CHART_KWARGS)
    if fast:
        kwargs.update(FAST_CHART_KWARGS)
    mesh.compute_charts(**kwargs)
    new_vertices, _new_faces = mesh.read()
    num_charts, _charts_id, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = mesh.read_atlas_charts()
    return new_vertices, int(num_charts), chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset


# ---------------------------------------------------------------------------
# CPU half: split, group, unwrap per tile, gather
# ---------------------------------------------------------------------------
def mesh_area(vertices: torch.Tensor, faces: torch.Tensor) -> float:
    """Sum of the triangle areas (float64 accumulation)."""
    tri = vertices[faces.long()].double()
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return float(cross.norm(dim=-1).sum().item() * 0.5)


def face_area_norms(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-face ``|cross|`` (twice the area) with the same operations as :func:`mesh_area`;
    the sum of a contiguous slice, halved, equals :func:`mesh_area` of that chart bit for bit."""
    tri = vertices[faces.long()].double()
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return cross.norm(dim=-1)


def split_chart(vertices: torch.Tensor, faces: torch.Tensor, vmap: torch.Tensor, max_faces: int, max_area: float):
    """Cut one chart by recursive median splits of the face centroids along the longest axis
    until every piece has at most ``max_faces`` faces and at most ``max_area`` surface area.
    Pieces of one face are kept as they are. Returns ``[(vertices, faces, vmap), ...]``."""
    out = []
    stack = [(vertices, faces, vmap)]
    while stack:
        v, f, m = stack.pop()
        if f.shape[0] <= 1:
            out.append((v, f, m))
            continue
        if f.shape[0] <= max_faces and mesh_area(v, f) <= max_area:
            out.append((v, f, m))
            continue
        centroids = v[f.long()].mean(dim=1)
        axis = int((centroids.max(0).values - centroids.min(0).values).argmax())
        selector = centroids[:, axis] <= centroids[:, axis].median()
        if bool(selector.all()) or bool((~selector).all()):
            order = torch.argsort(centroids[:, axis])
            selector = torch.zeros(f.shape[0], dtype=torch.bool)
            selector[order[: f.shape[0] // 2]] = True
        for side in (selector, ~selector):
            side_faces = f[side]
            used, local = torch.unique(side_faces.reshape(-1).long(), return_inverse=True)
            stack.append((v[used].contiguous(), local.reshape(-1, 3).to(torch.int32).contiguous(), m[used]))
    return out


def plan_tiles(chart_vertices: torch.Tensor, chart_faces: torch.Tensor, chart_vertex_offset: torch.Tensor,
               chart_face_offset: torch.Tensor, chart_vmap: torch.Tensor, num_charts: int, uv_workers: int,
               fast_split: bool = False):
    """Split oversized charts and assign the pieces to ``grid x grid`` tile groups.

    Returns ``(tasks, groups, grid, info)``: ``tasks`` = ``[(vertices, local faces int32, vmap)]``,
    ``groups[g]`` = task indices of tile ``g`` (row-major in the grid)."""
    total_faces = int(chart_face_offset[-1])
    face_cap = max(10_000, total_faces // max(1, int(uv_workers) * 2))
    norms = face_area_norms(chart_vertices, chart_faces) if fast_split else None
    clusters = []
    total_area = 0.0
    for i in range(num_charts):
        v0, v1 = int(chart_vertex_offset[i]), int(chart_vertex_offset[i + 1])
        f0, f1 = int(chart_face_offset[i]), int(chart_face_offset[i + 1])
        verts_i = chart_vertices[v0:v1].contiguous()
        faces_i = (chart_faces[f0:f1] - chart_vertex_offset[i]).contiguous()
        vmap_i = chart_vmap[v0:v1]
        area_i = float(norms[f0:f1].sum().item() * 0.5) if norms is not None else mesh_area(verts_i, faces_i)
        total_area += area_i
        clusters.append((verts_i, faces_i, vmap_i, area_i))
    area_cap = total_area / max(1, int(uv_workers) * 2)

    tasks = []
    n_split = 0
    for verts_i, faces_i, vmap_i, area_i in clusters:
        if faces_i.shape[0] > face_cap or area_i > area_cap:
            n_split += 1
            tasks.extend(split_chart(verts_i, faces_i, vmap_i, face_cap, area_cap))
        else:
            tasks.append((verts_i, faces_i, vmap_i))
    if not fast_split:
        # The areas of the final pieces; computed (and not used for any decision) as in the
        # measured configurations without pinned_bvh_transfers.
        _ = [mesh_area(v, f) for v, f, _m in tasks]

    n_groups = max(1, min(int(uv_workers), len(tasks)))
    grid = math.ceil(math.sqrt(n_groups))
    n_groups = grid * grid
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    group_faces = [0] * n_groups
    for ti in sorted(range(len(tasks)), key=lambda i: -tasks[i][1].shape[0]):
        gi = min(range(n_groups), key=lambda g: group_faces[g])
        groups[gi].append(ti)
        group_faces[gi] += int(tasks[ti][1].shape[0])
    info = {"num_charts": int(num_charts), "num_tasks": len(tasks), "n_split_charts": n_split, "face_cap": face_cap,
            "area_cap": area_cap, "grid": grid, "group_faces": group_faces}
    return tasks, groups, grid, info


def gather_tiles(tasks, results, grid: int, new_vertices_cpu: torch.Tensor, fast_gather: bool = False):
    """Concatenate the per-tile xatlas outputs into one mesh with UVs in [0, 1]^2.

    ``results[g]`` = ``[(task index, xrefs, faces, uvs), ...]`` of tile ``g``. ``fast_gather``
    offsets all ``xrefs`` into one concatenated vertex map and gathers once (same values, same order)."""
    vmaps, out_faces, uvs, global_refs = [], [], [], []
    cnt = 0
    if fast_gather:
        offsets = [0]
        for _v, _f, m in tasks:
            offsets.append(offsets[-1] + int(m.shape[0]))
        all_vmaps = torch.cat([m for _v, _f, m in tasks], dim=0) if tasks else torch.empty(0, dtype=torch.int64)
    for gi, outs in enumerate(results):
        row, col = divmod(gi, grid)
        tile_offset = torch.tensor([col, row], dtype=torch.float32)
        for ti, xrefs, xfaces, xuvs in outs:
            if fast_gather:
                global_refs.append(xrefs.long() + offsets[ti])
            else:
                vmaps.append(tasks[ti][2][xrefs.long()])
            out_faces.append(xfaces + cnt)
            cnt += int(xrefs.shape[0])
            uvs.append((xuvs + tile_offset) / grid)
    if fast_gather:
        refs = torch.cat(global_refs, dim=0) if global_refs else torch.empty(0, dtype=torch.int64)
        vmap = all_vmaps[refs]
    else:
        vmap = torch.cat(vmaps, dim=0)
    faces = torch.cat(out_faces, dim=0)
    uv = torch.cat(uvs, dim=0)
    return {"out_vertices": new_vertices_cpu[vmap], "out_faces": faces, "out_uvs": uv, "out_vmaps": vmap}


def uv_phase(gpu_ctx: Dict[str, Any], timings: Dict[str, Any], *, fast_split_gather: bool = False,
             uv_workers: int = UV_WORKERS, uv_concurrency: Optional[int] = UV_CONCURRENCY, padding: int = UV_PADDING,
             atlas_factory=None) -> Dict[str, Any]:
    """CPU half of the tiled path (host tensors only). ``gpu_ctx`` is the output of
    :func:`fastpixal3d.mesh.bake.geometry_gpu_phase` with ``gpu_charts=False``. Returns
    out_vertices / out_faces / out_uvs / out_vmaps (host tensors) for
    :func:`fastpixal3d.mesh.bake.geometry_finish`; the step times go to ``timings["uv"]``."""
    if atlas_factory is None:
        from cumesh.xatlas import Atlas as atlas_factory  # resolved per call (see single_thread_xatlas)
    t0 = time.perf_counter()
    tasks, groups, grid, info = plan_tiles(
        gpu_ctx["chart_vertices"], gpu_ctx["chart_faces"], gpu_ctx["chart_vertex_offset"],
        gpu_ctx["chart_face_offset"], gpu_ctx["chart_vmap"], int(gpu_ctx["num_charts"]), uv_workers,
        fast_split=fast_split_gather)
    tile_res = int(gpu_ctx["texture_size"]) // grid
    t_split = time.perf_counter()

    def unwrap_group(gi: int):
        started = time.perf_counter()
        members = groups[gi]
        if not members:
            return [], 0.0
        atlas = atlas_factory()
        for ti in members:
            atlas.add_mesh(tasks[ti][0], tasks[ti][1])
        atlas.compute_charts()
        atlas.pack_charts(resolution=tile_res, padding=padding)
        return [(ti, *atlas.get_mesh(k)) for k, ti in enumerate(members)], time.perf_counter() - started

    concurrency = len(groups) if uv_concurrency is None else int(uv_concurrency)
    with ThreadPoolExecutor(max_workers=min(concurrency, len(groups))) as pool:
        results = list(pool.map(unwrap_group, range(len(groups))))
    t_xatlas = time.perf_counter()
    out = gather_tiles(tasks, [outs for outs, _ in results], grid, gpu_ctx["new_vertices_cpu"],
                       fast_gather=fast_split_gather)
    stats = dict(info)
    stats.update(tile_res=tile_res, uv_concurrency=concurrency, fast_split_gather=bool(fast_split_gather),
                 split_s=round(t_split - t0, 3), xatlas_parallel_s=round(t_xatlas - t_split, 3),
                 gather_s=round(time.perf_counter() - t_xatlas, 3),
                 group_seconds=[round(d, 3) for _o, d in results], total_s=round(time.perf_counter() - t0, 6))
    timings["uv"] = stats
    return out
