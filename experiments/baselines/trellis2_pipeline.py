"""TRELLIS.2 (microsoft/TRELLIS.2 @ 75fbf018) at its released defaults, with stage timers.

The TRELLIS.2 checkout is used as it is: nothing in it is modified, and no attribute of its modules or classes is
reassigned. :func:`timed_pipeline_class` derives a subclass of ``Trellis2ImageTo3DPipeline`` whose stage methods
call the inherited method inside a CUDA-synchronised stage timer; ``from_pretrained`` builds instances of the
subclass (the base class constructs ``cls``), so the pipeline's own ``run()`` executes unchanged and is timed per
stage.

The stock request is the one of TRELLIS.2's ``example.py`` with the export settings of the report's runs:
``run(image, seed=42)`` with the pipeline's default type (``1024_cascade``), ``mesh.simplify(16777216)``, then
``o_voxel.postprocess.to_glb`` with 1,000,000 faces, a 4096 x 4096 texture, remeshing (band 1, projection 0) and
the cube [-0.5, 0.5]^3, written with trimesh's GLB exporter (PNG textures; ``example.py`` passes
``extension_webp=True``, the measured runs did not).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from baseline_io import Method, Stages, parts, tensor_sha256

COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
O_VOXEL_COMMIT = "de38fdd4082638cb4e6d2285f2e215280bd3f485"
WEIGHTS = "microsoft/TRELLIS.2-4B"
WEIGHTS_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
PIPELINE_TYPE = "1024_cascade"
MAX_NUM_TOKENS = 49152                      # run()'s default
SIMPLIFY_BEFORE_EXPORT = 16_777_216         # example.py: "nvdiffrast limit"
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
TO_GLB = {"decimation_target": 1_000_000, "texture_size": 4096, "remesh": True, "remesh_band": 1,
          "remesh_project": 0}

# Methods of Trellis2ImageTo3DPipeline that become stage timers (stage name = method name).
STAGE_METHODS = ("preprocess_image", "get_cond", "sample_sparse_structure", "sample_shape_slat",
                 "sample_shape_slat_cascade", "sample_tex_slat", "decode_shape_slat", "decode_tex_slat",
                 "decode_latent")
# Sampling of the networks (image conditioning and the three flows); everything else after preprocessing is
# post-processing in the report's tables.
FLOW_STAGES = ("get_cond", "sample_sparse_structure", "sample_shape_slat", "sample_shape_slat_cascade",
               "sample_tex_slat")


def default_source() -> str:
    from baseline_io import REPO

    return os.environ.get("FASTPIXAL3D_TRELLIS2_SRC") or str(REPO / "third_party" / "TRELLIS.2")


def timed_pipeline_class(base):
    """Subclass of ``base`` whose :data:`STAGE_METHODS` time themselves into ``self.stages`` (a
    :class:`baseline_io.Stages`, or None for no timing)."""

    def make(name):
        inherited = getattr(base, name)

        def method(self, *args, **kwargs):
            stages = self.stages
            if stages is None:
                return inherited(self, *args, **kwargs)
            with stages.stage(name):
                return inherited(self, *args, **kwargs)

        method.__name__ = name
        method.__qualname__ = f"Timed{base.__name__}.{name}"
        method.__doc__ = inherited.__doc__
        return method

    namespace = {name: make(name) for name in STAGE_METHODS if hasattr(base, name)}
    namespace["stages"] = None
    return type(f"Timed{base.__name__}", (base,), namespace)


_PIPELINES: Dict[tuple, Any] = {}


def load_pipeline(source: str, weights: str, device: str = "cuda"):
    """The timed TRELLIS.2 image-to-3D pipeline from a checkout at :data:`COMMIT` and a weight directory or
    Hugging Face id. Loaded once per process: every arm of one run shares the same pipeline object."""
    source = str(Path(source).expanduser().resolve())
    key = (source, str(weights), device)
    if key in _PIPELINES:
        return _PIPELINES[key]
    if not (Path(source) / "trellis2" / "pipelines").is_dir():
        raise FileNotFoundError(f"{source} is not a TRELLIS.2 checkout (see experiments/baselines/README.md)")
    if source not in sys.path:
        sys.path.insert(0, source)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = timed_pipeline_class(Trellis2ImageTo3DPipeline).from_pretrained(weights)
    if device == "cuda":
        pipe.cuda()
    else:
        pipe.to(device)
    if pipe.default_pipeline_type != PIPELINE_TYPE:
        raise RuntimeError(f"pipeline type {pipe.default_pipeline_type!r}, expected {PIPELINE_TYPE!r}")
    _PIPELINES[key] = pipe
    return pipe


def checkout_commit(source: str) -> Optional[str]:
    import subprocess

    try:
        return subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def latent_hashes(shape_slat, tex_slat, resolution: int) -> Dict[str, Any]:
    """sha256 of the raw latent bytes (computed outside the timed window by the callers)."""
    return {"shape_feats": tensor_sha256(shape_slat.feats), "shape_coords": tensor_sha256(shape_slat.coords),
            "tex_feats": tensor_sha256(tex_slat.feats), "tex_coords": tensor_sha256(tex_slat.coords),
            "resolution": int(resolution), "shape_tokens": int(shape_slat.coords.shape[0])}


def flow_times(timings: Dict[str, float]) -> Dict[str, float]:
    """``flows_s`` (image conditioning + the three flows) and ``post_processing_s`` = stage sum - preprocessing -
    flows: everything after the flows on the critical path (latent decoding, mesh processing, baking, export)."""
    flows = sum(timings.get(f"{name}_s", 0.0) for name in FLOW_STAGES)
    total = timings.get("stage_sum_s", 0.0)
    return {"flows_s": flows, "post_processing_s": total - timings.get("preprocess_image_s", 0.0) - flows}


class Trellis2Base(Method):
    """Loading and settings shared by the stock adapter and the mesh-path port."""

    name = "trellis2"

    def __init__(self, source: Optional[str] = None, weights: str = WEIGHTS, record_latents: bool = True):
        self.source = source or default_source()
        self.weights = weights
        self.record_latents = record_latents
        self.pipe = None
        self._latents = None

    def keep_latents(self, shape_slat, tex_slat, resolution: int) -> None:
        if self.record_latents:
            self._latents = (shape_slat, tex_slat, int(resolution))

    def after_request(self) -> Dict[str, Any]:
        if self._latents is None:
            return {}
        try:
            return {"latents": latent_hashes(*self._latents)}
        finally:
            self._latents = None

    def load(self) -> Dict[str, Any]:
        self.pipe = load_pipeline(self.source, self.weights)
        commit = checkout_commit(self.source)
        if commit and commit != COMMIT:
            print(f"[trellis2] warning: checkout at {commit}, the measured commit is {COMMIT}", flush=True)
        return {"trellis2_commit": commit, "weights": self.weights, "pipeline_type": self.pipe.default_pipeline_type,
                "low_vram": bool(getattr(self.pipe, "low_vram", None))}

    def settings(self) -> Dict[str, Any]:
        pipe = self.pipe
        out = {"pipeline_type": PIPELINE_TYPE, "max_num_tokens": MAX_NUM_TOKENS,
               "simplify_before_export": SIMPLIFY_BEFORE_EXPORT, "aabb": AABB, "export": dict(TO_GLB)}
        if pipe is not None:
            out.update(sparse_structure_sampler_params=pipe.sparse_structure_sampler_params,
                       shape_slat_sampler_params=pipe.shape_slat_sampler_params,
                       tex_slat_sampler_params=pipe.tex_slat_sampler_params)
        return out

    def derived(self, timings: Dict[str, float]) -> Dict[str, float]:
        return flow_times(timings)


class Trellis2Stock(Trellis2Base):
    """TRELLIS.2 as released: ``run()`` and ``o_voxel.postprocess.to_glb``."""

    arm = "stock"

    def derived(self, timings: Dict[str, float]) -> Dict[str, float]:
        out = flow_times(timings)
        out.update(parts(timings, sampling=out["flows_s"], decoding=timings.get("decode_latent_s", 0.0),
                         meshwork=timings.get("export_s", 0.0)))
        return out

    def run(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        import o_voxel
        from PIL import Image

        pipe = self.pipe
        pipe.stages = stages
        try:
            image = Image.open(image_path)
            meshes, (shape_slat, tex_slat, resolution) = pipe.run(image, seed=seed,
                                                                 pipeline_type=pipe.default_pipeline_type,
                                                                 return_latent=True)
            mesh = meshes[0]
            with stages.stage("export"):
                with stages.stage("simplify"):
                    mesh.simplify(SIMPLIFY_BEFORE_EXPORT)
                with stages.stage("to_glb"):
                    glb = o_voxel.postprocess.to_glb(
                        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=AABB, verbose=False, **TO_GLB)
                with stages.stage("write"):
                    glb.export(str(out_dir / "model.glb"))
        finally:
            pipe.stages = None
        self.keep_latents(shape_slat, tex_slat, resolution)
        return {"resolution": int(resolution), "shape_tokens": int(shape_slat.coords.shape[0]),
                "output_vertices": int(len(glb.vertices)), "output_faces": int(len(glb.faces))}
