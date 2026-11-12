"""Mesh preparation (GPU phase, finish) and texture baking.

The export path turns the decoded shape mesh and the decoded texture voxels
into a textured triangle mesh:

GPU phase (worker thread when ``mesh_prep_worker_thread`` is on)
    fill small holes -> BVH of the cleaned mesh -> narrow-band remesh (full or
    half grid) -> simplify to the face budget -> drop degenerate faces (->
    GPU chart clustering on the tiled mesh path).
UV phase (worker thread, concurrent with the texture flow)
    UV atlas: on the GPU (:mod:`fastpixal3d.mesh.uv_atlas`), or 16 tiles
    unwrapped on the CPU on the tiled mesh path (:mod:`fastpixal3d.mesh.tiled_path`).
finish (main thread)
    vertex normals of the simplified mesh, gathered to the atlas vertices.
bake (main thread)
    rasterise the atlas in UV space, map every texel to the closest point of
    the pre-remesh surface (BVH query), trilinearly sample the texture volume
    there, pull-push fill the gutter, and assemble a ``trimesh.Trimesh`` with a
    PBR material (base colour + alpha, metallic-roughness). The float maps are
    converted to 8 bit on the host in buffers kept across requests
    (:func:`texture_map_to_uint8`).

:func:`fixed_bake_launch_config` is the ``fixed_bake_launch_config`` lever: a
fixed tile configuration for FlexGEMM's indexed weighted-sum kernel (6
channels, 8 corners), bypassing its autotuner; same kernel and summation
order, so the values are unchanged.
"""
from __future__ import annotations

import contextlib
import importlib
import math
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as tnf

AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
FILL_HOLES_MAX_PERIMETER = 3e-2
REMESH_BAND = 1
REMESH_PROJECT = 0
RASTER_CHUNK = 100_000

_RASTER_CTX = None


# ---------------------------------------------------------------------------
# fixed launch configuration for the bake's trilinear sampling kernel
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def fixed_bake_launch_config(bm: int = 32, bk: int = 8, num_warps: int = 2):
    """Within the block, FlexGEMM's ``indice_weighed_sum_fwd`` launches its JIT kernel
    directly with (BM, BK, num_warps) for 6-channel / 8-corner inputs instead of
    running the autotuner; other shapes go through the original function."""
    import triton

    top = importlib.import_module("flex_gemm.kernels.triton")
    pkg = importlib.import_module("flex_gemm.kernels.triton.grid_sample")
    mod = importlib.import_module("flex_gemm.kernels.triton.grid_sample.indice_weighed_sum_fwd")
    original = mod.indice_weighed_sum_fwd
    kernel = mod.indice_weighed_sum_fwd_kernel
    if not hasattr(kernel, "fn"):
        raise RuntimeError("FlexGEMM's autotuned kernel exposes no underlying JIT function")
    stats = Counter(calls=0, accelerated=0, fallback=0)

    def fixed(input, indices, weight):
        stats["calls"] += 1
        eligible = (
            input.ndim == 2 and indices.ndim == 2 and weight.ndim == 2
            and input.shape[1] == 6 and indices.shape[1] == 8
            and indices.shape == weight.shape and input.shape[0] > 0
            and indices.shape[0] > 0 and input.is_cuda
            and input.is_contiguous() and indices.is_contiguous()
            and weight.is_contiguous())
        if not eligible:
            stats["fallback"] += 1
            return original(input, indices, weight)
        if torch.is_grad_enabled() and input.requires_grad:
            raise RuntimeError("fixed_bake_launch_config is inference-only")
        n, c = input.shape
        m, v = indices.shape
        out = torch.empty((m, c), dtype=input.dtype, device=input.device)
        grid = (triton.cdiv(c, bk) * triton.cdiv(m, bm),)
        kernel.fn[grid](input, indices, weight, out, int(math.log2(n)), m, c, v, BM=bm, BK=bk, num_warps=num_warps)
        stats["accelerated"] += 1
        return out

    modules = (top, pkg, mod)
    saved = [(m, m.indice_weighed_sum_fwd) for m in modules]
    try:
        for m in modules:
            m.indice_weighed_sum_fwd = fixed
        yield stats
    finally:
        for m, fn in saved:
            m.indice_weighed_sum_fwd = fn


# ---------------------------------------------------------------------------
# GPU phase
# ---------------------------------------------------------------------------
def geometry_gpu_phase(vertices: torch.Tensor, faces: torch.Tensor, grid_size: int, decimation_target: int,
                       texture_size: int, *, mesh_module, bvh_cls, remesh_divisor: int, fast_simplify: bool,
                       gpu_charts: bool, timings: Dict[str, Any]) -> Dict[str, Any]:
    from .remesh import remesh
    from .simplify import simplify

    aabb = torch.tensor(np.array(AABB), dtype=torch.float32, device="cuda")
    grid = torch.tensor(np.array([grid_size] * 3), dtype=torch.int32, device="cuda")
    voxel_size = (aabb[1] - aabb[0]) / grid

    started = time.perf_counter()
    vertices = vertices.cuda()
    faces = faces.cuda()
    mesh = mesh_module.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=FILL_HOLES_MAX_PERIMETER)
    vertices, faces = mesh.read()
    timings["fill_holes_s"] = round(time.perf_counter() - started, 3)

    torch.cuda.synchronize()
    started = time.perf_counter()
    bvh = bvh_cls(vertices, faces)
    torch.cuda.synchronize()
    timings["bvh_build_s"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    center = aabb.mean(dim=0)
    scale_extent = (aabb[1] - aabb[0]).max().item()
    resolution = grid.max().item()
    remesh_stats: Dict[str, Any] = {}
    mesh.init(*remesh(mesh_module.remeshing, vertices, faces, center, scale_extent, resolution, REMESH_BAND,
                      REMESH_PROJECT, bvh, divisor=remesh_divisor, stats=remesh_stats))
    simplify(mesh, decimation_target, fast=fast_simplify)
    timings["simplify_remesh_s"] = round(time.perf_counter() - started, 3)
    timings["remesh"] = remesh_stats

    t0 = time.perf_counter()
    mesh.remove_degenerate_faces()
    if gpu_charts:
        # Single-chart layout: the GPU atlas re-segments the mesh itself and only needs its topology.
        chart_v, chart_f = mesh.read()
        num_v, num_f = int(chart_v.shape[0]), int(chart_f.shape[0])
        num_charts = 1
        chart_vmap = torch.arange(num_v, dtype=torch.int64, device=chart_f.device)
        chart_faces = chart_f.to(torch.int64)
        chart_vertex_offset = torch.tensor([0, num_v], dtype=torch.int64, device=chart_f.device)
        chart_face_offset = torch.tensor([0, num_f], dtype=torch.int64, device=chart_f.device)
        new_vertices, _new_faces = mesh.read()
    else:
        # Tiled mesh path: CuMesh's GPU chart clustering; the charts are unwrapped on the CPU
        # (fastpixal3d.mesh.tiled_path.uv_phase).
        from .tiled_path import compute_charts

        new_vertices, num_charts, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = compute_charts(
            mesh, fast=fast_simplify)
    new_vertices_cpu = new_vertices.cpu()
    chart_vertices = new_vertices[chart_vmap].cpu()
    chart_faces = chart_faces.cpu()
    chart_vertex_offset = chart_vertex_offset.cpu()
    chart_face_offset = chart_face_offset.cpu()
    chart_vmap = chart_vmap.cpu()
    timings["cluster_s"] = round(time.perf_counter() - t0, 3)
    return {"mesh": mesh, "bvh": bvh, "vertices": vertices, "faces": faces, "aabb": aabb, "voxel_size": voxel_size,
            "grid_size": grid, "texture_size": texture_size, "new_vertices_cpu": new_vertices_cpu,
            "chart_vertices": chart_vertices, "chart_faces": chart_faces, "chart_vmap": chart_vmap,
            "chart_vertex_offset": chart_vertex_offset, "chart_face_offset": chart_face_offset,
            "num_charts": num_charts}


def geometry_uv_phase(gpu_ctx: Dict[str, Any], timings: Dict[str, Any]) -> Dict[str, Any]:
    from .uv_atlas import uv_phase

    out = uv_phase(gpu_ctx["new_vertices_cpu"], gpu_ctx["chart_faces"], gpu_ctx["chart_vmap"], gpu_ctx["texture_size"])
    timings["uv"] = out.pop("stats")
    return out


def geometry_finish(gpu_ctx: Dict[str, Any], uv: Dict[str, Any], timings: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    out_vertices = uv["out_vertices"].cuda()
    out_faces = uv["out_faces"].cuda()
    out_uvs = uv["out_uvs"].cuda()
    out_vmaps = uv["out_vmaps"].cuda()
    mesh = gpu_ctx["mesh"]
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    timings["finish_geometry_s"] = round(time.perf_counter() - started, 3)
    return {"out_vertices": out_vertices, "out_faces": out_faces, "out_uvs": out_uvs, "out_vmaps": out_vmaps,
            "out_normals": out_normals, "bvh": gpu_ctx["bvh"], "vertices": gpu_ctx["vertices"],
            "faces": gpu_ctx["faces"], "aabb": gpu_ctx["aabb"], "voxel_size": gpu_ctx["voxel_size"],
            "grid_size": gpu_ctx["grid_size"], "texture_size": gpu_ctx["texture_size"]}


# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------
def pullpush_fill(attrs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill texels where ``mask`` is False by pull-push (mip dilation); valid texels are unchanged."""
    x = (attrs * mask[..., None]).permute(2, 0, 1).unsqueeze(0)
    w = mask[None, None].to(attrs.dtype)
    levels = [(x, w)]
    while levels[-1][0].shape[-1] > 1:
        x_prev, w_prev = levels[-1]
        levels.append((tnf.avg_pool2d(x_prev, 2), tnf.avg_pool2d(w_prev, 2)))
    x_c, w_c = levels[-1]
    filled = x_c / w_c.clamp_min(1e-8)
    for x_l, w_l in reversed(levels[:-1]):
        up = tnf.interpolate(filled, size=x_l.shape[-2:], mode="bilinear", align_corners=False)
        filled = torch.where(w_l > 0, x_l / w_l.clamp_min(1e-8), up)
    out = filled[0].permute(1, 2, 0)
    return torch.where(mask[..., None], attrs, out)


class _HostBuffers(threading.local):
    """Host buffers of the texture conversion (per role, shape, dtype and thread), kept across requests."""

    def __init__(self):
        self.buffers: Dict[tuple, torch.Tensor] = {}

    def get(self, role: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        key = (role, shape, dtype)
        buffer = self.buffers.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype)
            self.buffers[key] = buffer
        return buffer


_HOST_BUFFERS = _HostBuffers()


def texture_map_to_uint8(values: torch.Tensor) -> np.ndarray:
    """``np.clip(values.cpu().numpy() * 255, 0, 255).astype(np.uint8)`` for a texture map.

    Same operations in the same precision (so the same result), but the host copy and the two float
    intermediate results are written to two host buffers that later requests reuse. Fresh arrays
    would be backed by new pages that the kernel has to fault in and zero on every request (about
    1.2 GB for the four maps of a 4096^2 texture, 0.1-0.2 s of CPU time); the reused buffers stay
    resident after the first request."""
    shape = tuple(values.shape)
    copy = _HOST_BUFFERS.get("copy", shape, values.dtype)
    scratch = _HOST_BUFFERS.get("scratch", shape, values.dtype).numpy()
    copy.copy_(values)
    array = copy.numpy()
    np.multiply(array, 255, out=scratch)
    np.clip(scratch, 0, 255, out=array)
    return array.astype(np.uint8)


def raster_context():
    global _RASTER_CTX
    if _RASTER_CTX is None:
        import nvdiffrast.torch as dr

        _RASTER_CTX = dr.RasterizeCudaContext()
    return _RASTER_CTX


def bake_and_finalize(geom: Dict[str, Any], attr_volume: torch.Tensor, coords: torch.Tensor, attr_layout,
                      on_texture: Optional[Callable[[Any], None]] = None):
    """Returns ``(trimesh.Trimesh, timings)``. ``on_texture(image)`` is called for each of the two
    PBR texture images right after it is created (used to start PNG encoding early)."""
    import nvdiffrast.torch as dr
    import trimesh
    import trimesh.visual
    from PIL import Image
    from flex_gemm.ops.grid_sample import grid_sample_3d

    timings: Dict[str, Any] = {}
    out_vertices, out_faces, out_uvs = geom["out_vertices"], geom["out_faces"], geom["out_uvs"]
    out_normals, bvh = geom["out_normals"], geom["bvh"]
    vertices, faces = geom["vertices"], geom["faces"]
    aabb, voxel_size, grid_size = geom["aabb"], geom["voxel_size"], geom["grid_size"]
    texture_size = geom["texture_size"]

    started = time.perf_counter()
    ctx = raster_context()
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])],
                         dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device="cuda", dtype=torch.float32)
    for i in range(0, out_faces.shape[0], RASTER_CHUNK):
        rast_chunk, _ = dr.rasterize(ctx, uvs_rast, out_faces[i:i + RASTER_CHUNK], resolution=[texture_size, texture_size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)
    mask = rast[0, ..., 3] > 0

    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device="cuda")
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode="trilinear",
    )
    torch.cuda.synchronize()
    timings["bake_s"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    attrs = pullpush_fill(attrs, mask)
    torch.cuda.synchronize()
    timings["fill_s"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    base_color = texture_map_to_uint8(attrs[..., attr_layout["base_color"]])
    metallic = texture_map_to_uint8(attrs[..., attr_layout["metallic"]])
    roughness = texture_map_to_uint8(attrs[..., attr_layout["roughness"]])
    alpha = texture_map_to_uint8(attrs[..., attr_layout["alpha"]])

    base_color_image = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
    if on_texture is not None:
        on_texture(base_color_image)
    metallic_roughness_image = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
    if on_texture is not None:
        on_texture(metallic_roughness_image)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base_color_image,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=metallic_roughness_image,
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=False,  # remeshed output is watertight-ish and consistently oriented
    )

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1]
    textured_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, vertex_normals=normals_np, process=False,
                                    visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material))
    timings["finalize_s"] = round(time.perf_counter() - started, 3)
    return textured_mesh, timings
