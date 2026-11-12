"""Configuration: the export settings, lever flags and the fixed sampling constants.

The defaults reproduce the deployed configuration (all levers on). The export
uses the settings of Pixal3D's own ``inference.py``: a target of 1,000,000
faces and a 4096 x 4096 texture (:data:`EXPORT`).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .levers import LEVERS, LEVERS_BY_NAME


@dataclass(frozen=True)
class ExportSettings:
    decimation_target: int   # face budget of the GPU simplification
    texture_size: int        # edge length of the square base-colour and metallic-roughness textures


# The export settings of Pixal3D's inference.py; every reported number uses them.
EXPORT = ExportSettings(decimation_target=1_000_000, texture_size=4096)


@dataclass(frozen=True)
class Levers:
    """One boolean per lever (see :mod:`fastpixal3d.levers`); all on by default."""

    mesh_prep_worker_thread: bool = True
    fixed_bake_launch_config: bool = True
    fast_gpu_simplify: bool = True
    gpu_bvh_build: bool = True
    adaptive_remesh: bool = True
    lean_glb_writer: bool = True
    gpu_chart_segmentation: bool = True
    per_tile_single_thread_unwrap: bool = True
    pinned_bvh_transfers: bool = True
    fa3_texture_flow: bool = True
    texture_kv_reuse: bool = True
    shared_conditioning: bool = True
    fa3_fine_shape_flow: bool = True
    texture_side_stream: bool = True
    structure_student: bool = True
    texture_projection_reuse: bool = True
    coarse_shape_student: bool = True
    fine_shape_student: bool = True
    fold_lora_fine_shape: bool = True
    texture_student: bool = True

    @classmethod
    def deployed(cls) -> "Levers":
        return cls()

    @classmethod
    def baseline(cls) -> "Levers":
        """Every lever off (teacher sampling, stock mesh tools)."""
        return cls(**{f.name: False for f in fields(cls)})

    def with_disabled(self, names: Iterable[str]) -> "Levers":
        names = list(names)
        for name in names:
            if name not in LEVERS_BY_NAME:
                raise KeyError(f"unknown lever {name!r}")
        return replace(self, **{name: False for name in names})

    def with_enabled(self, names: Iterable[str]) -> "Levers":
        names = list(names)
        for name in names:
            if name not in LEVERS_BY_NAME:
                raise KeyError(f"unknown lever {name!r}")
        return replace(self, **{name: True for name in names})

    def as_dict(self) -> Dict[str, bool]:
        return dataclasses.asdict(self)

    def disabled(self):
        return [name for name, on in self.as_dict().items() if not on]


assert [f.name for f in fields(Levers)] == [lv.name for lv in LEVERS], "Levers fields must follow the lever table"


@dataclass(frozen=True)
class Sampling:
    """Seeds and sampler settings of the deployed configuration (all fixed)."""

    structure_noise_seed: int = 42          # CPU generator, dense [1, 8, 16, 16, 16] initial noise
    coarse_noise_seed: int = 0              # CPU generator, [N, 32] initial noise
    texture_seed_base: int = 4242           # torch.manual_seed(texture_seed_base + seed) before texture sampling
    # Structure student: Euler steps on these noise fractions (a subset of the 12-step grid at rescale_t=3).
    structure_grid_indices: tuple = (0, 3, 6, 8, 12)
    structure_grid_rescale_t: float = 3.0
    shape_student_rescale_t: float = 3.0    # one-step students: query time 1.0
    texture_student_steps: int = 2
    texture_student_rescale_t: float = 3.0
    # Teacher settings (used when a student lever is off).
    teacher_structure_steps: int = 12
    teacher_structure_rescale_t: float = 5.0
    teacher_shape_steps: int = 12           # coarse and fine shape
    teacher_shape_rescale_t: float = 3.0
    teacher_texture_steps: int = 12
    teacher_texture_rescale_t: float = 3.0
    # Same-step teacher comparisons (defaults: Pixal3D's own samplers). With grid indices, the structure
    # teacher takes Euler steps on these indices of the 12-step grid at structure_grid_rescale_t (the
    # structure student's grid) instead of its own schedule. Guidance off = conditional prediction only
    # (one forward pass per step), as the students.
    teacher_structure_grid_indices: Optional[Tuple[int, ...]] = None
    teacher_structure_guidance: bool = True
    teacher_shape_guidance: bool = True
    # Structure adapter with a consistency sampler (an entry of students.json selected by
    # Config.structure_adapter whose steps say x0 prediction): query times and the seed of its re-noising.
    consistency_structure_t: tuple = (1.0, 0.9, 0.75, 0.5)
    consistency_structure_renoise_seed: int = 42
    # Shape support selection (Pixal3D's 1536 cascade with its token cap).
    hr_requested_resolution: int = 1536
    hr_max_tokens: int = 49_152
    # Adaptive remesh: keep the native grid iff hr_tokens / resolution**2 >= threshold.
    remesh_density_threshold: float = 0.025


@dataclass(frozen=True)
class Config:
    levers: Levers = field(default_factory=Levers)
    sampling: Sampling = field(default_factory=Sampling)
    device: str = "cuda"
    threads: int = 8                        # torch / OpenMP CPU threads
    attention_fallback: Optional[str] = None  # None: auto (FA3 on sm_90, else FA2); "fa2": force FA2
    verify_structure_graph: bool = True     # compare the first graphed structure rollout with an eager one
    use_structure_graph: bool = True        # static CUDA graph for the structure student (bit-exact, scheduling only)
    structure_adapter: str = "structure"    # students.json key of the structure student

    @property
    def export(self) -> ExportSettings:
        return EXPORT

    def replace(self, **kwargs) -> "Config":
        return replace(self, **kwargs)


def config_from_mapping(spec: Mapping[str, Any]) -> Config:
    """A :class:`Config` from a plain mapping (a parsed YAML or JSON file), e.g.::

        levers: {gpu_chart_segmentation: false, texture_student: false}   # or a list of the levers that are ON
        sampling: {teacher_shape_steps: 1, teacher_shape_guidance: false}
        structure_adapter: structure               # students.json key of the structure student

    ``levers`` as a mapping changes only the named flags of the deployed configuration (all on);
    as a list it names exactly the levers that are on (all others off). Other keys of ``spec``
    (names, descriptions) are ignored here; unknown keys inside ``levers`` / ``sampling`` raise."""
    kwargs: Dict[str, Any] = {}
    for key in ("attention_fallback", "threads", "use_structure_graph", "verify_structure_graph",
                "structure_adapter"):
        if key in spec and spec[key] is not None:
            kwargs[key] = spec[key]
    raw = spec.get("levers")
    levers = Levers()
    if isinstance(raw, Mapping):
        unknown = sorted(set(raw) - set(LEVERS_BY_NAME))
        if unknown:
            raise KeyError(f"unknown levers {unknown}")
        levers = replace(levers, **{k: bool(v) for k, v in raw.items()})
    elif raw is not None:
        names = list(raw)
        unknown = sorted(set(names) - set(LEVERS_BY_NAME))
        if unknown:
            raise KeyError(f"unknown levers {unknown}")
        levers = Levers.baseline().with_enabled(names)
    kwargs["levers"] = levers
    sampling = dict(spec.get("sampling") or {})
    known = {f.name for f in fields(Sampling)}
    unknown = sorted(set(sampling) - known)
    if unknown:
        raise KeyError(f"unknown sampling settings {unknown}")
    for key, value in list(sampling.items()):
        if isinstance(value, list):
            sampling[key] = tuple(value)
    kwargs["sampling"] = Sampling(**sampling)
    return Config(**kwargs)


def load_config_file(path) -> Dict[str, Any]:
    """Parse a YAML (or JSON) configuration file into a mapping (see :func:`config_from_mapping`)."""
    import json
    from pathlib import Path

    text = Path(path).read_text()
    if str(path).endswith(".json"):
        return json.loads(text)
    import yaml

    return yaml.safe_load(text) or {}


def stable_fine_shape_seed(object_id: str, seed: int = 0) -> int:
    """Per-object seed of the fine-shape initial noise (sha256 of a fixed namespaced string)."""
    import hashlib

    raw = f"shape_hr|{object_id}|{int(seed)}|hr_initial_noise".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % ((1 << 63) - 1)
