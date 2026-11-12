"""TRELLIS.2 with the rebuilt mesh path of this package.

TRELLIS.2-4B keeps its released weights, samplers, seed, pipeline type (``1024_cascade``) and export targets
(1,000,000 faces, 4096 x 4096 texture, remeshing with band 1 and projection 0, the cube [-0.5, 0.5]^3, GLB in
TRELLIS.2's own export frame). Only the post-processing after the flows is replaced: instead of
``o_voxel.postprocess.to_glb`` and trimesh's exporter, the output is built by ``fastpixal3d.mesh`` and written by
``fastpixal3d.export.glb`` (hole filling, GPU BVH, narrow-band remeshing, GPU simplification, single-pass GPU chart
segmentation and atlas packing, texture bake, lean GLB writer).

Form of the port: a wrapper, no patch. TRELLIS.2 is imported from an unmodified checkout at its pinned commit and
no attribute of its modules is reassigned. The request below calls the pipeline's public stage methods in the
order of ``Trellis2ImageTo3DPipeline.run()`` (same arguments, same seed position) up to the shape latent, then
decodes the shape before the texture flow (the shape decoding depends only on the shape latent), so that the mesh
work can run on a worker thread while the texture flow samples. A patch series would have to change ``run()`` in
TRELLIS.2 and ``to_glb`` in o-voxel; the wrapper needs neither, and the latents it produces are bitwise equal to
those of ``run()`` (recorded per input as sha256 of the raw tensors).

Arms (levers of docs/LEVERS.md; ``--disable`` removes one):

  mesh_path                  mesh_prep_worker_thread, fixed_bake_launch_config, fast_gpu_simplify, gpu_bvh_build,
                             gpu_chart_segmentation, lean_glb_writer       (report row "+ rebuilt mesh path")
  mesh_path_adaptive_remesh  the same + adaptive_remesh                    (report row "+ adaptive half-resolution
                                                                            remesh"; output-changing)
  mesh_path_serial           mesh_path without mesh_prep_worker_thread: TRELLIS.2's run() unchanged, then the
                             mesh path on the main thread (control: what the overlap contributes)

Without ``mesh_prep_worker_thread`` an arm always takes the serial form. The levers ``per_tile_single_thread_unwrap``
and ``pinned_bvh_transfers`` of the measured mesh path act on code paths that the GPU chart segmentation and the GPU
BVH replace, so they have no counterpart here (docs/LEVERS.md); ``gpu_chart_segmentation`` cannot be switched off
(the tiled CPU UV path is not part of this package).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

BASELINES = Path(__file__).resolve().parents[1] / "baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from baseline_io import Stages, add_package_to_path  # noqa: E402
from trellis2_pipeline import (MAX_NUM_TOKENS, SIMPLIFY_BEFORE_EXPORT, WEIGHTS, Trellis2Base,  # noqa: E402
                               Trellis2Stock)

MESH_LEVERS = ("mesh_prep_worker_thread", "fixed_bake_launch_config", "fast_gpu_simplify", "gpu_bvh_build",
               "gpu_chart_segmentation", "lean_glb_writer", "adaptive_remesh")
ARMS = {
    "mesh_path": ("mesh_prep_worker_thread", "fixed_bake_launch_config", "fast_gpu_simplify", "gpu_bvh_build",
                  "gpu_chart_segmentation", "lean_glb_writer"),
    "mesh_path_adaptive_remesh": ("mesh_prep_worker_thread", "fixed_bake_launch_config", "fast_gpu_simplify",
                                  "gpu_bvh_build", "gpu_chart_segmentation", "lean_glb_writer", "adaptive_remesh"),
    "mesh_path_serial": ("fixed_bake_launch_config", "fast_gpu_simplify", "gpu_bvh_build", "gpu_chart_segmentation",
                         "lean_glb_writer"),
}
ALL_ARMS = ("stock",) + tuple(ARMS)


def arm_levers(arm: str, disable: Iterable[str] = ()) -> frozenset:
    """Levers of a mesh-path arm after ``--disable``."""
    if arm not in ARMS:
        raise KeyError(f"unknown mesh-path arm {arm!r}; choose from {sorted(ARMS)}")
    disable = list(disable)
    for name in disable:
        if name not in MESH_LEVERS:
            raise KeyError(f"{name!r} is not a lever of the mesh path; choose from {list(MESH_LEVERS)}")
        if name == "gpu_chart_segmentation":
            raise ValueError("gpu_chart_segmentation cannot be switched off: the tiled CPU UV path it replaces is "
                             "not part of this package")
    return frozenset(ARMS[arm]) - set(disable)


class Trellis2MeshPath(Trellis2Base):
    """TRELLIS.2 generation unchanged; post-processing by fastpixal3d.mesh / fastpixal3d.export."""

    def __init__(self, arm: str = "mesh_path", disable: Iterable[str] = (), source: Optional[str] = None,
                 weights: str = WEIGHTS, record_latents: bool = True):
        super().__init__(source=source, weights=weights, record_latents=record_latents)
        self.arm = arm
        self.levers = arm_levers(arm, disable)
        self.mesh_module = self.bvh_cls = None

    def load(self) -> Dict[str, Any]:
        info = super().load()
        add_package_to_path()
        from fastpixal3d.export.glb import fpng_module
        from fastpixal3d.mesh.bvh import bvh_class, bvh_impl_name
        from fastpixal3d.mesh.simplify import mesh_module

        # resolve the native modules now, so that a missing build fails before the first request
        self.mesh_module = mesh_module("fast_gpu_simplify" in self.levers)
        self.bvh_cls = bvh_class("gpu_bvh_build" in self.levers)
        import flex_gemm.ops.grid_sample  # noqa: F401
        import nvdiffrast.torch  # noqa: F401

        info.update(levers=sorted(self.levers), mesh_module=self.mesh_module.__name__,
                    bvh=bvh_impl_name("gpu_bvh_build" in self.levers),
                    png_encoder="fpng" if ("lean_glb_writer" in self.levers and fpng_module() is not None) else "pillow")
        return info

    def settings(self) -> Dict[str, Any]:
        out = super().settings()
        out.update(arm=self.arm, levers=sorted(self.levers),
                   schedule="overlapped" if "mesh_prep_worker_thread" in self.levers else "serial")
        return out

    # ------------------------------------------------------------------------------------------- helpers
    def _launch_config(self):
        from fastpixal3d.mesh import bake

        return bake.fixed_bake_launch_config() if "fixed_bake_launch_config" in self.levers else contextlib.nullcontext()

    def _divisor(self, shape_slat, resolution: int) -> int:
        from fastpixal3d.config import Sampling
        from fastpixal3d.mesh.remesh import remesh_divisor

        return remesh_divisor(int(shape_slat.coords.shape[0]), int(resolution), Sampling().remesh_density_threshold,
                              adaptive="adaptive_remesh" in self.levers)

    def _gpu_phase(self, mesh, resolution: int, divisor: int, mesh_timings: Dict[str, Any]):
        from fastpixal3d.config import EXPORT   # 1,000,000 faces, 4096 x 4096: also TRELLIS.2's own export targets
        from fastpixal3d.mesh import bake

        return bake.geometry_gpu_phase(mesh.vertices, mesh.faces, int(resolution), EXPORT.decimation_target,
                                       EXPORT.texture_size, mesh_module=self.mesh_module, bvh_cls=self.bvh_cls,
                                       remesh_divisor=divisor, fast_simplify="fast_gpu_simplify" in self.levers,
                                       gpu_charts=True, timings=mesh_timings)

    def _bake_and_write(self, stages: Stages, gpu_ctx, uv, attr_volume, coords, attr_layout, out_dir: Path,
                        stats: Dict[str, Any]) -> None:
        from fastpixal3d.export.glb import export_glb
        from fastpixal3d.mesh import bake

        mesh_timings = stats["mesh_timings"]
        with stages.stage("finish_bake"):
            geom = bake.geometry_finish(gpu_ctx, uv, mesh_timings)
            output, bake_timings = bake.bake_and_finalize(geom, attr_volume=attr_volume, coords=coords,
                                                          attr_layout=attr_layout)
        with stages.stage("write"):
            export_stats: Dict[str, Any] = {}
            export_glb(output, out_dir / "model.glb", lean="lean_glb_writer" in self.levers, rotation=None,
                       stats=export_stats)
        stats.update(output_vertices=int(len(output.vertices)), output_faces=int(len(output.faces)),
                     bake_timings=bake_timings, export=export_stats)

    # ------------------------------------------------------------------------------------------- requests
    def run(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        import torch

        self.pipe.stages = stages
        try:
            with torch.no_grad():
                if "mesh_prep_worker_thread" in self.levers:
                    return self._run_overlapped(image_path, out_dir, seed, stages)
                return self._run_serial(image_path, out_dir, seed, stages)
        finally:
            self.pipe.stages = None

    def _run_overlapped(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        import torch
        from PIL import Image

        from fastpixal3d.mesh import bake
        from fastpixal3d.runtime.streams import join_thread, run_in_thread

        pipe = self.pipe
        stats: Dict[str, Any] = {"mesh_timings": {}}
        image = Image.open(image_path)
        # Trellis2ImageTo3DPipeline.run() for pipeline type 1024_cascade (one sample, the pipeline's sampler
        # parameters), statement by statement up to the shape latent.
        image = pipe.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = pipe.get_cond([image], 512)
        cond_1024 = pipe.get_cond([image], 1024)
        coords = pipe.sample_sparse_structure(cond_512, 32, 1, {})
        shape_slat, resolution = pipe.sample_shape_slat_cascade(
            cond_512, cond_1024, pipe.models["shape_slat_flow_model_512"], pipe.models["shape_slat_flow_model_1024"],
            512, 1024, coords, {}, MAX_NUM_TOKENS)
        del coords
        divisor = self._divisor(shape_slat, resolution)
        with self._launch_config():
            # The shape is decoded before the texture flow; the hole filling of decode_latent() and the
            # simplification before the export move with it.
            with stages.stage("decode_shape"):
                meshes, subs = pipe.decode_shape_slat(shape_slat, resolution)
                if len(meshes) != 1:
                    raise RuntimeError(f"expected one decoded mesh, got {len(meshes)}")
                mesh = meshes[0]
                mesh.fill_holes()
                mesh.simplify(SIMPLIFY_BEFORE_EXPORT)
            stats["source_faces"] = int(mesh.faces.shape[0])

            def worker():
                ctx = self._gpu_phase(mesh, resolution, divisor, stats["mesh_timings"])
                torch.cuda.synchronize()
                return ctx, bake.geometry_uv_phase(ctx, stats["mesh_timings"])

            with stages.stage("texture_and_mesh"):
                thread, holder = run_in_thread(worker, "mesh_prep")
                try:
                    tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, {})
                    torch.cuda.empty_cache()          # as run() does after the flows
                    tex_voxels = pipe.decode_tex_slat(tex_slat, subs)
                finally:
                    thread.join()
                gpu_ctx, uv = join_thread(thread, holder)
            stats["mesh_worker_s"] = holder.get("wall_s")
            voxels = tex_voxels[0]
            self._bake_and_write(stages, gpu_ctx, uv, voxels.feats, voxels.coords[:, 1:], pipe.pbr_attr_layout,
                                 out_dir, stats)
        self.keep_latents(shape_slat, tex_slat, resolution)
        stats.update(resolution=int(resolution), shape_tokens=int(shape_slat.coords.shape[0]), remesh_divisor=divisor)
        return stats

    def _run_serial(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        from PIL import Image

        from fastpixal3d.mesh import bake

        pipe = self.pipe
        stats: Dict[str, Any] = {"mesh_timings": {}}
        meshes, (shape_slat, tex_slat, resolution) = pipe.run(Image.open(image_path), seed=seed,
                                                             pipeline_type=pipe.default_pipeline_type,
                                                             return_latent=True)
        mesh = meshes[0]
        divisor = self._divisor(shape_slat, resolution)
        with self._launch_config(), stages.stage("export"):
            with stages.stage("simplify"):
                mesh.simplify(SIMPLIFY_BEFORE_EXPORT)
            stats["source_faces"] = int(mesh.faces.shape[0])
            with stages.stage("mesh_gpu_phase"):
                gpu_ctx = self._gpu_phase(mesh, resolution, divisor, stats["mesh_timings"])
            with stages.stage("mesh_uv_phase"):
                uv = bake.geometry_uv_phase(gpu_ctx, stats["mesh_timings"])
            self._bake_and_write(stages, gpu_ctx, uv, mesh.attrs, mesh.coords, mesh.layout, out_dir, stats)
        self.keep_latents(shape_slat, tex_slat, resolution)
        stats.update(resolution=int(resolution), shape_tokens=int(shape_slat.coords.shape[0]), remesh_divisor=divisor)
        return stats


def build_arm(arm: str, disable: Iterable[str] = (), source: Optional[str] = None, weights: str = WEIGHTS,
              record_latents: bool = True):
    if arm == "stock":
        return Trellis2Stock(source=source, weights=weights, record_latents=record_latents)
    return Trellis2MeshPath(arm, disable, source=source, weights=weights, record_latents=record_latents)
