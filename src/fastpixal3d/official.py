"""Pixal3D (official): the released Pixal3D inference script, called at its defaults.

This is the reference row of the comparison. What runs, exactly, for every input::

    inference.run_inference(image_path=<input>, output_path=<out>.glb, model_path=<weights>)

from Pixal3D's own ``inference.py`` (the unmodified ``third_party/Pixal3D`` checkout), with every other
argument at its default. That is the same call as
``python inference.py --image <input> --output <out>.glb --model_path <weights>``:
seed 42; 12 sampling steps for the structure, shape and texture flows (guidance 7.5 / 7.5 / 1.0,
guidance rescale 0.7 / 0.5 / 0.0, time rescale 5 / 3 / 3); the 1536 cascade with at most 49,152 tokens;
the camera field of view estimated by MoGe-2; ``o_voxel.postprocess.to_glb`` with 1,000,000 faces, a
4096 x 4096 texture and remeshing; the script's rotation; GLB export with WebP textures. ``model_path``
is the only argument that differs from the script's default (a local copy of the pinned weights instead
of the Hugging Face id).

Additions, none of which changes the official code path:

* the models are loaded once per process: ``inference.init_pipeline`` and ``inference.load_moge_model``
  are memoised (the script loads them on every call). The loading time is reported separately and
  excluded from the end-to-end time. ``run_inference`` moves MoGe-2 to the CPU after use, so the memoised
  model is moved back to the GPU on each call (inside the timed window, as in the reference measurement);
* the auxiliary models (DINOv3, MoGe-2, RMBG-2.0, NAF) are read from the weights directory at their pinned
  revisions instead of being fetched by hub id (:mod:`fastpixal3d.auxiliary`; the same files);
* the camera returned by ``inference.get_camera_params_wild_moge`` is recorded;
* per-stage wall times, with CUDA synchronisation around each timed call (record only);
* the FlexGEMM autotune cache is a file outside the Pixal3D checkout (``inference.py`` points it at a
  file inside the checkout when it is imported): the path given by the caller, or else a copy of the
  checkout's cache file in the work directory; and the working directory is the output directory
  (``run_inference`` writes a temporary PNG next to its output).

End-to-end time of one input: wall time of the ``run_inference`` call, from reading the input image to
the closed GLB file, minus the model loading of that call (non-zero only for the first input).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .auxiliary import use_pinned_auxiliary_models
from .weights import WeightPaths

STAGE_METHODS = (("preprocess_image", "preprocess_image"), ("get_proj_cond_ss", "cond_ss"),
                 ("get_proj_cond_shape", "cond_shape"), ("sample_sparse_structure", "sparse_structure"),
                 ("decode_latent", "decode"))


def configure_official_process(threads: int = 8, tf32_override: bool = False,
                               autotune_cache: Optional[str] = None) -> None:
    """Process environment of the reference measurement. Call before PyTorch is imported.

    ``tf32_override=False`` removes ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` (stock PyTorch matmul precision).
    The allocator and attention settings are the values ``inference.py`` sets itself when imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    # the values inference.py sets itself when imported, set here before PyTorch is imported
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    if not tf32_override:
        os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    if autotune_cache:
        os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(autotune_cache)


class OfficialPixal3D:
    def __init__(self, weights: Optional[WeightPaths] = None, work_dir: Optional[Union[str, Path]] = None):
        self.weights = weights or WeightPaths.from_env()
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.inference = None
        self.model_path: Optional[str] = None
        self.info: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {"stages": [], "camera": None, "prepared_size": None, "load_s": 0.0}

    @classmethod
    def load(cls, weights: Optional[WeightPaths] = None, work_dir=None) -> "OfficialPixal3D":
        self = cls(weights, work_dir)
        self._load()
        return self

    # ------------------------------------------------------------------ setup
    def _load(self) -> None:
        self.weights.apply_environment()
        src = self.weights.source_dir()
        self.model_path = self.weights.model_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        cache = os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH")
        if not cache:  # never the file inside the Pixal3D checkout: a copy of it in the work directory
            cache = str(self.work_dir / "autotune_cache.json")
            seed = src / "autotune_cache.json"
            if not Path(cache).exists() and seed.is_file():
                Path(cache).write_bytes(seed.read_bytes())
            os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = cache
        import flex_gemm  # binds the autotune cache path before inference.py overrides the variable
        import torch

        if os.path.abspath(flex_gemm.AUTOTUNE_CACHE_PATH) != os.path.abspath(cache):
            raise RuntimeError("flex_gemm was imported before the autotune cache path was set "
                               f"({flex_gemm.AUTOTUNE_CACHE_PATH}); load OfficialPixal3D in a fresh process")

        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        os.chdir(self.work_dir)
        spec = importlib.util.spec_from_file_location("pixal3d_official_inference", src / "inference.py")
        inf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = inf
        spec.loader.exec_module(inf)
        os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = cache
        self.inference = inf
        self._instrument(torch)
        started = time.perf_counter()
        aux = use_pinned_auxiliary_models(inf, self.weights.root_dir())
        inf.init_pipeline(self.model_path)  # memoised: loads once, counted as loading
        # same; run_inference moves MoGe-2 to the GPU and back
        inf.load_moge_model(device="cuda", model_name=aux["moge2_model"]).cpu()
        torch.cuda.synchronize()
        self.info.update(load_s=round(time.perf_counter() - started, 3), model_path=self.model_path,
                         inference_py=str(src / "inference.py"),
                         tf32_override=os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"), autotune_cache=cache,
                         auxiliary_models=aux)
        self._state["load_s"] = 0.0

    def _instrument(self, torch) -> None:
        inf, state, models = self.inference, self._state, self._models

        def sync():
            torch.cuda.synchronize()

        def timed(name, fn):
            def wrapper(*args, **kwargs):
                sync()
                started = time.perf_counter()
                out = fn(*args, **kwargs)
                sync()
                state["stages"].append((name, time.perf_counter() - started))
                return out
            return wrapper

        orig_init, orig_moge, orig_camera = inf.init_pipeline, inf.load_moge_model, inf.get_camera_params_wild_moge

        def init_pipeline(*args, **kwargs):
            if "pipe" not in models:
                started = time.perf_counter()
                pipe = orig_init(*args, **kwargs)
                for method, name in STAGE_METHODS:
                    setattr(pipe, method, timed(name, getattr(pipe, method)))
                # coarse shape level; the sampler calls of both shape levels are summed separately
                pipe.sample_shape_slat = timed("shape_coarse", pipe.sample_shape_slat)
                pipe.shape_slat_sampler.sample = timed("shape_sampler_calls", pipe.shape_slat_sampler.sample)
                pipe.sample_tex_slat = timed("texture", pipe.sample_tex_slat)
                models["pipe"] = pipe
                state["load_s"] += time.perf_counter() - started
            return models["pipe"]

        def load_moge_model(*args, **kwargs):
            if "moge" not in models:
                started = time.perf_counter()
                models["moge"] = orig_moge(*args, **kwargs)
                state["load_s"] += time.perf_counter() - started
            return models["moge"].to(kwargs.get("device", "cuda"))

        def get_camera(image_path, *args, **kwargs):
            from PIL import Image

            with Image.open(image_path) as im:
                state["prepared_size"] = [int(v) for v in im.size]
            camera = timed("camera", orig_camera)(image_path, *args, **kwargs)
            state["camera"] = {k: float(v) for k, v in camera.items()}
            return camera

        inf.init_pipeline, inf.load_moge_model, inf.get_camera_params_wild_moge = init_pipeline, load_moge_model, get_camera
        inf.o_voxel.postprocess.to_glb = timed("to_glb", inf.o_voxel.postprocess.to_glb)
        import trimesh

        for cls in (trimesh.Trimesh, trimesh.Scene):
            if "export" in cls.__dict__ and not getattr(cls.__dict__["export"], "_official_timed", False):
                original = cls.__dict__["export"]

                def export(self_, *args, _original=original, **kwargs):
                    return timed("export", _original)(self_, *args, **kwargs)
                export._official_timed = True
                cls.export = export

    # ------------------------------------------------------------------ request
    def run(self, image_path: Union[str, Path], output_path: Union[str, Path], object_id: Optional[str] = None,
            seed: Optional[int] = None) -> Dict[str, Any]:
        """Run the official script on one input; returns timings and statistics (same layout as
        :meth:`fastpixal3d.pipeline.FastPixal3D.run`)."""
        import torch

        image_path, output_path = Path(image_path).resolve(), Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state
        state.update(stages=[], camera=None, prepared_size=None)
        load_before = state["load_s"]
        kwargs = {"image_path": str(image_path), "output_path": str(output_path), "model_path": self.model_path}
        if seed is not None:
            kwargs["seed"] = int(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.inference.run_inference(**kwargs)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        load = state["load_s"] - load_before
        timings: Dict[str, float] = {}
        for name, seconds in state["stages"]:
            key = f"{name}_s"
            timings[key] = round(timings.get(key, 0.0) + seconds, 6)
        timings["e2e_s"] = round(wall - load, 6)
        stats = {"object_id": object_id or image_path.stem, "camera": state["camera"],
                 "prepared_size": state["prepared_size"], "load_s_this_request": round(load, 3),
                 "wall_s_incl_load": round(wall, 3), "seed": kwargs.get("seed", 42),
                 "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                 "glb": str(output_path), "glb_bytes": output_path.stat().st_size}
        return {"timings": timings, "stats": stats}
