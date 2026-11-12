"""Image -> textured GLB with the distilled students and the speed levers.

    from fastpixal3d.pipeline import FastPixal3D
    pipe = FastPixal3D.load()                           # or FastPixal3D.load(Config(...))
    result = pipe.run("input.png", "output.glb")
    print(result["timings"]["e2e_s"])

Execution order of one request (stage keys of :mod:`fastpixal3d.runtime.timing`):

1. preprocess the image (alpha crop or background removal), write it as PNG,
   estimate the camera field of view with MoGe-2 on that PNG;
2. structure: conditioning -> student (4 steps) -> support decode;
3. coarse shape: conditioning -> student (1 step);
4. fine support from the upsampled coarse latent; fine shape: conditioning ->
   student (1 step, FlashAttention-3, folded LoRA);
5. decode the shape latent to a mesh; then, concurrently,
   - worker thread: hole filling, GPU BVH, adaptive remesh, GPU simplification,
     GPU UV atlas;
   - main thread: texture conditioning -> texture student (2 steps, side CUDA
     stream) -> texture decode;
6. vertex normals, texture bake (fixed launch configuration), PNG encoding
   started as soon as the textures exist;
7. GLB export (lean writer).

With ``mesh_prep_worker_thread`` off, the mesh GPU phase runs on the main thread
first and only the UV atlas is built on the worker thread while the texture is
sampled. With ``gpu_chart_segmentation`` off, the UV atlas is the tiled mesh path
of the first rows of the cumulative measurement (:mod:`fastpixal3d.mesh.tiled_path`).

Every stage boundary synchronises the GPU; the end-to-end time is measured
from the start of step 1 to the closed GLB file. Model loading and warm-up are
not part of a request.
"""
from __future__ import annotations

import contextlib
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import Config, Levers, stable_fine_shape_seed
from .weights import WeightPaths


def configure_process(threads: int = 8, autotune_cache: Optional[str] = None, tf32_override: bool = True) -> None:
    """Process-wide settings of the measured configuration. Call before CUDA is initialised
    (``scripts/infer.py`` does this before importing torch)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["ATTN_BACKEND"] = "flash_attn"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # The measured configuration ran float32 cuBLAS matmuls with TF32 allowed (this also applies to the
    # LoRA folding at load time). tf32_override=False removes the variable: full float32 matmuls.
    if tf32_override:
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    else:
        os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    if autotune_cache:
        os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(autotune_cache)


class FastPixal3D:
    def __init__(self, config: Config, weights: WeightPaths):
        self.config = config
        self.weights = weights
        self.levers: Levers = config.levers
        self.pipe = None
        self.moge = None
        self.inference = None
        self.info: Dict[str, Any] = {}
        self._keep: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, config: Optional[Config] = None, weights: Optional[WeightPaths] = None, **config_kwargs):
        config = config or Config(**config_kwargs)
        weights = weights or WeightPaths.from_env()
        self = cls(config, weights)
        self._load()
        return self

    def _load(self) -> None:
        import torch

        cfg, lv = self.config, self.levers
        started = time.perf_counter()
        self.weights.apply_environment()
        src = str(self.weights.source_dir())
        if src not in sys.path:
            sys.path.insert(0, src)
        cache = os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH")
        import flex_gemm  # noqa: F401  (binds the autotune cache path before Pixal3D's inference.py sets its own)
        import inference  # Pixal3D's inference.py (camera estimation, extractor configs, pipeline init)

        if cache:  # undo inference.py's import-time setting (a file inside the Pixal3D checkout)
            os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = cache
        else:
            os.environ.pop("FLEX_GEMM_AUTOTUNE_CACHE_PATH", None)
        self.info["autotune_cache"] = getattr(sys.modules.get("flex_gemm"), "AUTOTUNE_CACHE_PATH", None)
        os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "0"
        torch.set_num_threads(cfg.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.inference = inference
        for key in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
            del sys.modules[key]  # NAF's torch.hub code imports a top-level package named "src"
        from .auxiliary import use_pinned_auxiliary_models

        self.info["auxiliary_models"] = aux = use_pinned_auxiliary_models(inference, self.weights.root_dir())
        self.pipe = inference.init_pipeline(self.weights.model_dir(), device=cfg.device, low_vram=False)
        self.moge = inference.load_moge_model(device=cfg.device, model_name=aux["moge2_model"])
        nrm = self.pipe.shape_slat_normalization
        self.slat_std = torch.tensor(nrm["std"]).float().reshape(-1)
        self.slat_mean = torch.tensor(nrm["mean"]).float().reshape(-1)
        self.info["pipeline_load_s"] = round(time.perf_counter() - started, 3)

        started = time.perf_counter()
        self._attach_students()
        self.info["students_load_s"] = round(time.perf_counter() - started, 3)

        if lv.shared_conditioning:
            from .stages.conditioning import verify_shared_weights

            self.info["shared_conditioning"] = verify_shared_weights(self.pipe)

        from .runtime.attention import fa3_status, fast_backend

        self.fast_attention = fast_backend(cfg.attention_fallback)
        self.info["fa3_status"] = fa3_status()
        self.info["fast_attention_backend"] = self.fast_attention

        from .stages.structure import CONSISTENCY, StructureStudent, student_t_values

        self.structure_student = None
        if lv.structure_student:
            s = cfg.sampling
            if self._structure_sampler == CONSISTENCY:
                t_values = [float(t) for t in s.consistency_structure_t]
            else:
                t_values = student_t_values(self._structure_grid, s.structure_grid_rescale_t,
                                            s.teacher_structure_steps)
            self.structure_student = StructureStudent(self.pipe.models["sparse_structure_flow_model"], t_values,
                                                      use_graph=cfg.use_structure_graph,
                                                      verify=cfg.verify_structure_graph,
                                                      sampler=self._structure_sampler,
                                                      renoise_seed=s.consistency_structure_renoise_seed)
            self.info["structure_t_values"] = t_values
            self.info["structure_sampler"] = self._structure_sampler
        torch.cuda.synchronize()

    def _check_student_times(self, name: str, spec) -> None:
        """The manifest's sampling times must equal the times used here (absolute tolerance 1e-6).

        ``steps.grid_indices`` (indices into Pixal3D's 12-step grid at time rescale 3) is the source of
        truth; ``steps.t`` is used only when the indices are absent. The structure student samples at
        exactly these grid points; the texture student runs Pixal3D's sampler with 2 steps, whose times
        must coincide with them."""
        from .stages.structure import CONSISTENCY, student_t_values, teacher_t_grid

        if name not in ("structure", "texture"):
            return
        s = self.config.sampling
        if name == "structure" and self._structure_sampler == CONSISTENCY:
            listed = spec.steps.get("t")
            ours = [float(t) for t in s.consistency_structure_t]
            if listed is not None:
                listed = [float(v) for v in listed]
                if len(listed) == len(ours) + 1 and listed[-1] == 0.0:
                    listed = listed[:-1]  # the final 0.0 of a step list is not a query time
                if len(listed) != len(ours) or any(abs(a - b) > 1e-6 for a, b in zip(listed, ours)):
                    raise ValueError(f"structure student {spec.key!r}: students.json gives query times {listed}, "
                                     f"this configuration uses {ours}")
            return
        indices = spec.grid_indices()
        listed = (student_t_values(indices, s.structure_grid_rescale_t, s.teacher_structure_steps)
                  if indices is not None else spec.steps.get("t"))
        if listed is None:
            return
        if name == "structure":
            ours = student_t_values(self._structure_grid, s.structure_grid_rescale_t, s.teacher_structure_steps)
        else:
            ours = [float(t) for t in teacher_t_grid(s.texture_student_steps, s.texture_student_rescale_t)]
        if len(listed) != len(ours) or any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(listed, ours)):
            raise ValueError(f"{name} student: students.json gives t = {listed}, this configuration uses {ours}")

    def _attach_students(self) -> None:
        """Wrap the pipeline's own flows with the student adapters (one copy of each flow)."""
        from .students.fold import fold_lora
        from .students.load import (FOLD_BF16, PIPELINE_MODEL_KEYS, check_target_modules, load_release,
                                    load_training_checkpoints, relative_state)
        from .students.lora import apply_lora, bind_lora_state, lora_parameter_names, lora_target_names
        from .stages.structure import TRAJECTORY, sampler_of

        lv = self.levers
        wanted = [name for name, on in (("structure", lv.structure_student), ("coarse_shape", lv.coarse_shape_student),
                                        ("fine_shape", lv.fine_shape_student), ("texture", lv.texture_student)) if on]
        self._structure_grid = tuple(self.config.sampling.structure_grid_indices)
        self._structure_sampler = TRAJECTORY
        self.info["students"] = {}
        if not wanted:
            return
        keys = {"structure": self.config.structure_adapter}
        students = self.weights.students_dir()
        if students:
            loaded = load_release(students, wanted, keys=keys)
        elif self.weights.student_checkpoints:
            loaded = load_training_checkpoints(self.weights.student_checkpoints, wanted, keys=keys)
        else:
            raise ValueError("student levers are on but no student weights were found (--students, "
                             "FASTPIXAL3D_STUDENTS or <weights directory>/students)")
        for name in wanted:
            weights = loaded[name]
            spec = weights.spec
            if spec.model_key != PIPELINE_MODEL_KEYS[name]:
                raise ValueError(f"{name}: students.json names pipeline model {spec.model_key!r}, "
                                 f"expected {PIPELINE_MODEL_KEYS[name]!r}")
            model = self.pipe.models[spec.model_key]
            check_target_modules(spec, lora_target_names(model))
            apply_lora(model, rank=spec.rank, alpha=spec.alpha)
            state, prefix = relative_state(weights.state, lora_parameter_names(model))
            n = bind_lora_state(model, state)
            fold = spec.fold == FOLD_BF16
            if name == "fine_shape":
                fold = bool(lv.fold_lora_fine_shape)
            folded = fold_lora(model) if fold else 0
            model.eval()
            if name == "structure":
                self._structure_sampler = sampler_of(spec.steps) if spec.steps else TRAJECTORY
                if spec.grid_indices() is not None and self._structure_sampler == TRAJECTORY:
                    self._structure_grid = spec.grid_indices()
            self._check_student_times(name, spec)
            self.info["students"][name] = {"tensors": n, "rank": spec.rank, "alpha": spec.alpha, "folded": folded,
                                           "name_prefix": prefix, "source": Path(weights.source).name,
                                           "key": spec.key}
            del weights, state
        del loaded

    # ------------------------------------------------------------------ request
    def run(self, image_path: Union[str, Path], output_path: Union[str, Path], object_id: Optional[str] = None,
            seed: int = 0, work_dir: Optional[Union[str, Path]] = None,
            dump_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Generate one GLB. ``object_id`` (default: the image file stem) seeds the fine-shape
        noise; ``seed`` offsets all per-request seeds. Returns timings and statistics.

        ``dump_dir``: also write the intermediate results used by ``scripts/parity_check.py``
        (structure support, normalised coarse and fine latents), after the request has finished."""
        import torch

        self._keep = {} if dump_dir is not None else None
        try:
            with torch.no_grad():
                result = self._run(Path(image_path), Path(output_path), object_id or Path(image_path).stem, int(seed),
                                   Path(work_dir) if work_dir else Path(output_path).resolve().parent)
            if dump_dir is not None:
                write_parity_record(Path(dump_dir), self._keep, result)
            return result
        finally:
            self._keep = None
            # Between requests (outside the timed window): collect garbage and release cached
            # allocator blocks, so that no collection or cache growth from this request spills
            # into the next one.
            gc.collect()
            torch.cuda.empty_cache()

    def _run(self, image_path: Path, output_path: Path, object_id: str, seed: int, work_dir: Path) -> Dict[str, Any]:
        import torch

        from .export.glb import PngPrewarm, export_glb
        from .runtime.streams import RequestExecutor
        from .runtime.timing import StageTimer
        from .stages.conditioning import conditioning_for

        lv = self.levers
        timer = StageTimer()
        export = self.config.export
        stats: Dict[str, Any] = {"object_id": object_id, "seed": seed,
                                 "export_settings": {"decimation_target": export.decimation_target,
                                                     "texture_size": export.texture_size}}
        executor = RequestExecutor(max_workers=2, defer_empty_cache=lv.texture_side_stream)
        prewarm = PngPrewarm(executor.submit) if (lv.texture_side_stream and lv.lean_glb_writer) else None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Request-scoped helpers are set up before the clock starts and torn down after it stops.
        try:
            with conditioning_for(self.pipe, lv.shared_conditioning) as cond:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                request_started = time.perf_counter()
                image, camera = self._prepare_image(image_path, object_id, work_dir, timer)
                stats["camera"] = {k: float(v) for k, v in camera.items()}
                stats["prepared_size"] = [int(v) for v in image.size]
                shape_slat, resolution = self._shape(image, camera, object_id, seed, cond, timer, executor, stats)
                output = self._decode(image, camera, shape_slat, resolution, seed, cond, timer, executor, prewarm, stats)
                timer.timed("export_s", lambda: export_glb(output, output_path, lean=lv.lean_glb_writer,
                                                             prewarm=prewarm, stats=stats.setdefault("export", {})))
                timer.record("e2e_s", time.perf_counter() - request_started)
        finally:
            executor.join()
        stats["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        stats["glb"] = str(output_path)
        stats["glb_bytes"] = output_path.stat().st_size
        return {"timings": timer.summary(), "stats": stats}

    # -- stage 1: image and camera --------------------------------------------------------
    def _prepare_image(self, image_path: Path, object_id: str, work_dir: Path, timer):
        from PIL import Image

        prepared_dir = work_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = prepared_dir / f"{object_id}.png"

        def preprocess():
            image = self.pipe.preprocess_image(Image.open(image_path))
            image.save(prepared_path)
            return image

        image = timer.timed("preprocess_s", preprocess)
        camera = timer.timed("camera_s", lambda: self.inference.get_camera_params_wild_moge(
            os.fspath(prepared_path), self.moge, device=self.config.device, mesh_scale=1.0, extend_pixel=0,
            image_resolution=512))
        return image, camera

    # -- stages 2-4: structure and shape --------------------------------------------------
    def _shape(self, image, camera, object_id: str, seed: int, cond, timer, executor, stats):
        import torch

        from .runtime.attention import attention_backend
        from .stages import shape as sh
        from .stages import structure as st

        lv, s = self.levers, self.config.sampling
        pipe = self.pipe
        device = torch.device(pipe.device)
        cam = dict(camera_angle_x=camera["camera_angle_x"], distance=camera["distance"],
                   mesh_scale=camera.get("mesh_scale", 1.0))

        # structure
        cond_ss = timer.timed("ss_condition_s", lambda: cond.structure([image], **cam))
        ss_initial = timer.timed("ss_initial_noise_s", lambda: st.initial_noise(device, s.structure_noise_seed))
        structure_stats: Dict[str, Any] = {}
        if lv.structure_student:
            dense = timer.timed("ss_rollout_s", lambda: self.structure_student(cond_ss["cond"], ss_initial, structure_stats))
        elif s.teacher_structure_grid_indices is not None:  # same-step teacher on the student's grid
            t_values = st.student_t_values(s.teacher_structure_grid_indices, s.structure_grid_rescale_t,
                                           s.teacher_structure_steps)
            structure_stats["teacher_t_values"] = t_values
            dense = timer.timed("ss_rollout_s", lambda: st.teacher_rollout_on_grid(
                pipe, cond_ss, ss_initial, t_values, guidance=s.teacher_structure_guidance))
        else:
            dense = timer.timed("ss_rollout_s", lambda: st.teacher_rollout(pipe, cond_ss, ss_initial,
                                                                           s.teacher_structure_steps,
                                                                           s.teacher_structure_rescale_t,
                                                                           guidance=s.teacher_structure_guidance))
        ss_coords = timer.timed("ss_decode_support_s",
                                lambda: st.decode_support(pipe.models["sparse_structure_decoder"], dense, device))
        if int(ss_coords.shape[0]) == 0:
            raise RuntimeError("the sparse structure is empty")
        stats["ss_tokens"] = int(ss_coords.shape[0])
        stats["structure"] = structure_stats
        del cond_ss, dense, ss_initial
        executor.empty_cache()

        # coarse shape
        cond_lr = timer.timed("lr_condition_s", lambda: cond.sparse(pipe.image_cond_model_shape_512, [image], ss_coords, **cam))
        lr_initial = timer.timed("lr_initial_noise_s", lambda: sh.coarse_initial_noise(ss_coords.shape[0], s.coarse_noise_seed))
        if lv.coarse_shape_student:
            lr_slat, lr_normalized = timer.timed("lr_rollout_s", lambda: sh.coarse_student(
                pipe.models["shape_slat_flow_model_512"], cond_lr["cond"], ss_coords, lr_initial, 1.0,
                s.coarse_noise_seed, self.slat_std, self.slat_mean))
        else:
            lr_slat = timer.timed("lr_rollout_s", lambda: sh.coarse_teacher(
                pipe, cond_lr, ss_coords, s.teacher_shape_steps, s.teacher_shape_rescale_t, s.coarse_noise_seed,
                guidance=s.teacher_shape_guidance))
            lr_normalized = None
        if self._keep is not None:  # references only; written after the request
            self._keep["support"] = ss_coords
            self._keep["coarse"] = (lr_normalized.coords, lr_normalized.feats) if lr_normalized is not None else None
        del cond_lr, lr_initial
        executor.empty_cache()

        # fine support
        def fine_support():
            upsampled = pipe.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
            return sh.select_fine_support(upsampled, s.hr_requested_resolution, s.hr_max_tokens)

        hr_coords, meta = timer.timed("hr_upsample_support_s", fine_support)
        hr_coords = hr_coords.to(device=device, dtype=torch.int32)
        resolution = int(meta["actual_resolution"])
        stats["hr_tokens"] = int(hr_coords.shape[0])
        stats["resolution"] = resolution
        del lr_slat
        executor.empty_cache()

        # fine shape
        cond_hr = timer.timed("hr_condition_s", lambda: cond.sparse(
            pipe.image_cond_model_shape_1024, [image], hr_coords, grid_resolution_override=int(meta["grid_resolution"]), **cam))
        hr_seed = stable_fine_shape_seed(object_id, seed)
        hr_initial, rng_state = timer.timed("hr_initial_noise_s", lambda: sh.fine_initial_noise(hr_coords.shape[0], hr_seed, device))
        if lv.fine_shape_student:
            def fine():
                ctx = attention_backend(self.fast_attention) if lv.fa3_fine_shape_flow else contextlib.nullcontext()
                with ctx:
                    return sh.fine_student(pipe.models["shape_slat_flow_model_1024"], cond_hr["cond"], hr_coords,
                                           hr_initial, rng_state, 1.0)
        else:  # Pixal3D's fine-shape sampler (FlashAttention-3 too when fa3_fine_shape_flow is on)
            def fine():
                ctx = attention_backend(self.fast_attention) if lv.fa3_fine_shape_flow else contextlib.nullcontext()
                with ctx:
                    return sh.fine_teacher(pipe, cond_hr, hr_coords, hr_initial, s.teacher_shape_steps,
                                           s.teacher_shape_rescale_t, guidance=s.teacher_shape_guidance)
        hr_features = timer.timed("hr_rollout_s", fine)
        if self._keep is not None:
            self._keep["fine"] = (hr_coords, hr_features)
        std = self.slat_std.to(device=device, dtype=torch.float32)[None]
        mean = self.slat_mean.to(device=device, dtype=torch.float32)[None]
        shape_slat = sh._sparse(hr_features.float() * std + mean, hr_coords)
        if self._keep is not None:
            self._keep["digests"] = {"fine_coords": shape_slat.coords, "fine_latent": shape_slat.feats}
        stats["hr_seed"] = hr_seed
        del cond_hr, hr_initial, hr_features, rng_state
        executor.empty_cache()
        return shape_slat, resolution

    # -- stages 5-6: decode, mesh, texture, bake ------------------------------------------
    def _sample_texture(self, cond_tex, shape_slat):
        from .runtime.attention import attention_backend
        from .runtime.streams import SideStream
        from .stages.texture import texture_kv_reuse, texture_projection_reuse

        lv, s = self.levers, self.config.sampling
        pipe = self.pipe
        model = pipe.models["tex_slat_flow_model_1024"]
        params = ({"steps": s.texture_student_steps, "rescale_t": s.texture_student_rescale_t} if lv.texture_student
                  else {"steps": s.teacher_texture_steps, "rescale_t": s.teacher_texture_rescale_t})
        with contextlib.ExitStack() as stack:
            if lv.fa3_texture_flow:
                stack.enter_context(attention_backend(self.fast_attention))
            if lv.texture_kv_reuse:
                stack.enter_context(texture_kv_reuse(model, shape_slat))

            def call():
                ctx = texture_projection_reuse(model) if lv.texture_projection_reuse else contextlib.nullcontext()
                with ctx:
                    return pipe.sample_tex_slat(cond_tex, model, shape_slat, params)

            if lv.texture_side_stream:
                return SideStream(pipe.device).run(call)
            return call()

    def _decode(self, image, camera, shape_slat, resolution: int, seed: int, cond, timer, executor, prewarm, stats):
        import torch

        from .mesh import bake, tiled_path
        from .mesh.bvh import bvh_class
        from .mesh.remesh import remesh_divisor
        from .mesh.simplify import mesh_module
        from .runtime.streams import join_thread, run_in_thread
        from .stages.shape import fork_cuda_rng

        lv, s = self.levers, self.config.sampling
        pipe = self.pipe
        export = self.config.export
        divisor = remesh_divisor(int(shape_slat.coords.shape[0]), resolution, s.remesh_density_threshold,
                                 adaptive=lv.adaptive_remesh)
        stats["remesh_divisor"] = divisor
        prep_timings: Dict[str, Any] = {}
        mesh_mod = mesh_module(lv.fast_gpu_simplify)
        tiled = not lv.gpu_chart_segmentation
        bvh_cls = tiled_path.bvh_class(lv) if tiled else bvh_class(lv.gpu_bvh_build)
        stats["mesh_path"] = "tiled" if tiled else "gpu_atlas"
        stats["mesh_schedule"] = "worker_thread" if lv.mesh_prep_worker_thread else "uv_overlap"

        launch_cfg = bake.fixed_bake_launch_config() if lv.fixed_bake_launch_config else contextlib.nullcontext()
        unwrap_cfg = (tiled_path.single_thread_xatlas() if tiled and lv.per_tile_single_thread_unwrap
                      else contextlib.nullcontext())
        with launch_cfg, unwrap_cfg:
            meshes, subs = timer.timed("shape_decode_s", lambda: pipe.decode_shape_slat(shape_slat, resolution))
            if len(meshes) != 1:
                raise RuntimeError(f"expected one decoded mesh, got {len(meshes)}")
            source_mesh = meshes[0]
            timer.timed("shape_fill_holes_s", source_mesh.fill_holes)
            stats["source_vertices"] = int(source_mesh.vertices.shape[0])
            stats["source_faces"] = int(source_mesh.faces.shape[0])
            span_started = time.perf_counter()

            def gpu_phase():
                return bake.geometry_gpu_phase(
                    source_mesh.vertices, source_mesh.faces, resolution, export.decimation_target, export.texture_size,
                    mesh_module=mesh_mod, bvh_cls=bvh_cls, remesh_divisor=divisor, fast_simplify=lv.fast_gpu_simplify,
                    gpu_charts=lv.gpu_chart_segmentation, timings=prep_timings)

            if tiled:
                def uv_phase(ctx):
                    return tiled_path.uv_phase(ctx, prep_timings, fast_split_gather=lv.pinned_bvh_transfers)
            else:
                def uv_phase(ctx):
                    return bake.geometry_uv_phase(ctx, prep_timings)

            if lv.mesh_prep_worker_thread:  # the whole mesh preparation on a worker thread
                def worker():
                    ctx = gpu_phase()
                    torch.cuda.synchronize()
                    return ctx, uv_phase(ctx)
                thread, holder = run_in_thread(worker, "mesh_prep")
            else:  # GPU phase on the main thread first; only the UV phase overlaps the texture
                first = timer.timed("glb_gpu_phase_s", gpu_phase)
                thread, holder = run_in_thread(lambda: (first, uv_phase(first)), "uv_phase")

            window_started = time.perf_counter()
            tex_started = time.perf_counter()
            try:
                cond_tex = timer.timed("tex_condition_s", lambda: cond.sparse(
                    pipe.image_cond_model_tex_1024, [image], shape_slat.coords,
                    camera_angle_x=camera["camera_angle_x"], distance=camera["distance"],
                    mesh_scale=camera.get("mesh_scale", 1.0), grid_resolution_override=resolution // 16))

                def tex_rollout():
                    with fork_cuda_rng(torch.device(pipe.device)):
                        torch.manual_seed(s.texture_seed_base + seed)
                        return self._sample_texture(cond_tex, shape_slat)

                tex_slat = timer.timed("tex_rollout_s", tex_rollout)
                if self._keep is not None:
                    self._keep.setdefault("digests", {})["texture_latent"] = tex_slat.feats
                del cond_tex
                executor.empty_cache()
                tex_voxels = timer.timed("tex_decode_s", lambda: pipe.decode_tex_slat(tex_slat, subs))
                torch.cuda.synchronize()
                stats["tex_stream_s"] = round(time.perf_counter() - tex_started, 6)
            finally:
                thread.join()
            stats["mesh_overlap_window_s"] = round(time.perf_counter() - window_started, 6)
            stats["mesh_worker_s"] = holder.get("wall_s")
            gpu_ctx, uv = join_thread(thread, holder)

            geom = timer.timed("glb_finish_s", lambda: bake.geometry_finish(gpu_ctx, uv, prep_timings))
            voxel = tex_voxels[0]
            on_texture = prewarm if prewarm is not None else None
            output, bake_timings = timer.timed("glb_bake_finalize_s", lambda: bake.bake_and_finalize(
                geom, attr_volume=voxel.feats, coords=voxel.coords[:, 1:], attr_layout=pipe.pbr_attr_layout,
                on_texture=on_texture))
            timer.record("to_glb_s", time.perf_counter() - span_started)
            stats["output_vertices"] = int(len(output.vertices))
            stats["output_faces"] = int(len(output.faces))
            stats["mesh_timings"] = prep_timings
            stats["bake_timings"] = bake_timings
            del meshes, subs, tex_voxels, tex_slat, source_mesh, gpu_ctx, uv, geom
            executor.empty_cache()
        return output


def write_parity_record(directory: Path, keep: Dict[str, Any], result: Dict[str, Any]) -> None:
    """support.npy (int32 [M, 4]), coarse.npz / fine.npz (coords int32 [N, 4], feats float32 [N, 32]),
    timing.json -- the per-input layout read by scripts/parity_check.py. ``timing.json`` also holds
    the sha256 of the raw bytes of the de-normalised fine latent, its coordinates and the texture
    latent as passed to the decoders (bitwise comparison between runs)."""
    import json

    import numpy as np

    import hashlib

    directory.mkdir(parents=True, exist_ok=True)
    digests = {name: hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes(order="C")).hexdigest()
               for name, t in (keep.get("digests") or {}).items()}
    if keep.get("support") is not None:
        np.save(directory / "support.npy", keep["support"].detach().cpu().numpy().astype(np.int32))
    for name in ("coarse", "fine"):
        if keep.get(name) is not None:
            coords, feats = keep[name]
            np.savez(directory / f"{name}.npz", coords=coords.detach().cpu().numpy().astype(np.int32),
                     feats=feats.detach().float().cpu().numpy())
    (directory / "timing.json").write_text(json.dumps({"e2e_s": result["timings"]["e2e_s"],
                                                       "timings": result["timings"],
                                                       "output_faces": result["stats"].get("output_faces"),
                                                       "sha256": digests}, indent=1))


def camera_record(result: Dict[str, Any], object_id: str, image_path) -> Dict[str, Any]:
    """The inference-time camera of one request, as written next to its GLB (``<stem>.json``).

    ``camera`` holds ``camera_angle_x`` (horizontal field of view of the pinhole camera, radians,
    estimated by MoGe-2 on the preprocessed image), ``distance`` (camera distance from the origin of the
    generation cube along the view axis) and ``mesh_scale`` (scale of that cube, 1.0); the same three
    values are repeated at the top level. ``prepared_size`` is the size of the preprocessed (cropped)
    image the camera refers to."""
    camera = {k: float(v) for k, v in result["stats"]["camera"].items()}
    return {"object_id": object_id, "image": str(image_path), "camera": camera, **camera,
            "fov_source": "moge2", "prepared_size": result["stats"].get("prepared_size")}
