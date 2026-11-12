"""Loading the four student adapters.

Release format (a directory)::

    students.json
    structure_student.safetensors     coarse_shape_student.safetensors
    fine_shape_student.safetensors    texture_student.safetensors

``students.json``::

    {"format": 1,
     "base_model": {"repo": "TencentARC/Pixal3D", "revision": "<sha>"},
     "key_convention": "<how tensor names map to the adapter parameters>",
     "students": {"structure": {"file": ..., "sha256": ..., "base_submodel": "<path in the base repo>",
                                "lora": {"rank": 256, "alpha": 256, "target_modules": [...]},
                                "steps": {...}, "fold": "none" | "bf16", "dtype": "float32",
                                "source_checkpoint_sha256": ...},
                  "coarse_shape": {...}, "fine_shape": {...}, "texture": {...}}}

Tensor names are not hard-coded here. Every name must end in the name of an
adapter parameter of the LoRA-wrapped base flow (``blocks.<i>.<...>.lora_A.weight``
or ``...lora_B.weight``); a common prefix in front of that (whatever the
``key_convention`` documents) is detected and removed, and the resulting set
of names must equal the set of adapter parameters exactly.

Further entries may hold other adapters of the same flows (for example a
consistency-distilled structure student, entry ``structure_consistency``, which no
configuration of this repository selects). A configuration selects them by key
(``Config.structure_adapter``); the stage an entry adapts is its ``stage`` field
or else the stage name its key starts with.

A second, internal entry point (:func:`load_training_checkpoint`) reads the
exponential-moving-average adapters straight out of a training checkpoint
(``.pt``); it is used for development before the release files exist.
"""
from __future__ import annotations
import os
import pickle

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

STUDENTS = ("structure", "coarse_shape", "fine_shape", "texture")
FOLD_NONE, FOLD_BF16 = "none", "bf16"

# Which Pixal3D pipeline model each student adapts (pipeline.json "models" keys).
PIPELINE_MODEL_KEYS = {
    "structure": "sparse_structure_flow_model",
    "coarse_shape": "shape_slat_flow_model_512",
    "fine_shape": "shape_slat_flow_model_1024",
    "texture": "tex_slat_flow_model_1024",
}
# Fold rule of the deployed configuration (fine shape: switchable lever; texture: always).
DEFAULT_FOLD = {"structure": FOLD_NONE, "coarse_shape": FOLD_NONE, "fine_shape": FOLD_BF16, "texture": FOLD_BF16}


@dataclass
class StudentSpec:
    name: str                       # the stage: structure, coarse_shape, fine_shape or texture
    rank: int = 256
    alpha: float = 256.0
    fold: str = FOLD_NONE
    file: Optional[str] = None
    sha256: Optional[str] = None
    base_submodel: Optional[str] = None
    target_modules: Optional[list] = None
    steps: Dict[str, Any] = field(default_factory=dict)
    dtype: str = "float32"
    pipeline_key: Optional[str] = None
    key: Optional[str] = None       # the students.json key (default: the stage name)
    role: Optional[str] = None

    @property
    def model_key(self) -> str:
        """The Pixal3D pipeline model this student adapts."""
        return self.pipeline_key or PIPELINE_MODEL_KEYS.get(self.name) or PIPELINE_MODEL_KEYS[stage_of(self.name)]

    def grid_indices(self) -> Optional[Tuple[int, ...]]:
        """Teacher-grid indices of the student's sampling times (``steps.grid_indices``, or
        ``steps.teacher_grid.indices``), or None if the manifest does not give them."""
        indices = self.steps.get("grid_indices")
        grid = self.steps.get("teacher_grid")
        if indices is None and isinstance(grid, Mapping):
            indices = grid.get("indices")
        return None if indices is None else tuple(int(i) for i in indices)


@dataclass
class StudentWeights:
    spec: StudentSpec
    state: Dict[str, torch.Tensor]   # raw names as stored
    source: str


def sha256_file(path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# release format
# ---------------------------------------------------------------------------
def read_manifest(directory) -> Dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads((directory / "students.json").read_text())
    if int(manifest.get("format", -1)) != 1:
        raise ValueError(f"unsupported students.json format {manifest.get('format')!r} (expected 1)")
    students = manifest.get("students") or {}
    missing = [name for name in STUDENTS if name not in students]
    if missing:
        raise ValueError(f"students.json lacks entries for {missing}")
    return manifest


def stage_of(key: str, entry: Optional[Mapping[str, Any]] = None) -> str:
    """The stage a students.json entry adapts: its ``stage`` field, else the stage its key starts with."""
    stage = (entry or {}).get("stage")
    if stage is None:
        stage = next((s for s in STUDENTS if key == s or key.startswith(s + "_")), None)
    if stage not in STUDENTS:
        raise ValueError(f"students.json entry {key!r}: cannot tell which stage it adapts")
    return stage


def adapter_key(name: str, keys: Optional[Mapping[str, str]]) -> str:
    """The students.json key that serves stage ``name`` (``keys`` maps stage -> key; default: the stage)."""
    key = (keys or {}).get(name, name)
    if key != name and stage_of(key) != name:
        raise ValueError(f"adapter {key!r} does not adapt the {name} stage")
    return key


def spec_from_manifest(name: str, entry: Mapping[str, Any], key: Optional[str] = None) -> StudentSpec:
    lora = entry.get("lora") or {}
    fold = str(entry.get("fold", DEFAULT_FOLD.get(name, FOLD_NONE))).lower()
    if fold not in (FOLD_NONE, FOLD_BF16):
        raise ValueError(f"{key or name}: fold must be 'none' or 'bf16', got {fold!r}")
    return StudentSpec(name=name, rank=int(lora.get("rank", 256)), alpha=float(lora.get("alpha", 256.0)),
                       fold=fold, file=entry.get("file"), sha256=entry.get("sha256"),
                       base_submodel=entry.get("base_submodel"), target_modules=lora.get("target_modules"),
                       steps=dict(entry.get("steps") or {}), dtype=str(entry.get("dtype", "float32")),
                       pipeline_key=entry.get("pipeline_key"), key=key or name, role=entry.get("role"))


def load_release(directory, names: Iterable[str] = STUDENTS, verify_sha256: bool = False,
                 device: str = "cpu", keys: Optional[Mapping[str, str]] = None) -> Dict[str, StudentWeights]:
    """The adapters of the stages ``names`` (keyed by stage). ``keys`` maps a stage to another
    students.json entry than the stage's own (e.g. ``{"structure": "structure_consistency"}``)."""
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = read_manifest(directory)
    out = {}
    for name in names:
        key = adapter_key(name, keys)
        if key not in manifest["students"]:
            raise KeyError(f"{directory / 'students.json'} has no entry {key!r}, which this configuration "
                           f"uses for the {name} stage; available: {sorted(manifest['students'])}")
        spec = spec_from_manifest(name, manifest["students"][key], key=key)
        path = directory / spec.file
        if verify_sha256 and spec.sha256:
            got = sha256_file(path)
            if got != spec.sha256:
                raise ValueError(f"{name}: sha256 of {path.name} is {got}, students.json says {spec.sha256}")
        state = load_file(str(path), device=device)
        out[name] = StudentWeights(spec=spec, state=state, source=str(path))
    return out


# ---------------------------------------------------------------------------
# training checkpoints (internal / development)
# ---------------------------------------------------------------------------
def _ema_from_payload(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """EMA adapter weights keyed by the generator's parameter names. The EMA shadow is
    either name-keyed or positional; a positional shadow follows the insertion order
    of ``payload['gen']``."""
    gen = payload.get("gen")
    ema = payload.get("ema")
    shadow = ema.get("shadow") if isinstance(ema, Mapping) else None
    if (not isinstance(gen, Mapping) or not gen) and isinstance(shadow, Mapping) and shadow:
        return dict(shadow)
    if not isinstance(gen, Mapping) or not gen:
        raise ValueError("checkpoint has no name-keyed 'gen' weights")
    names = list(gen.keys())
    if isinstance(shadow, Mapping) and shadow:
        if set(shadow) != set(names):
            raise ValueError("name-keyed EMA names differ from the generator names")
        out = {n: shadow[n] for n in names}
    elif isinstance(shadow, (list, tuple)) and shadow:
        if len(shadow) != len(names):
            raise ValueError(f"positional EMA has {len(shadow)} tensors, generator has {len(names)}")
        out = dict(zip(names, shadow))
    else:
        raise ValueError("checkpoint has no EMA shadow")
    for n in names:
        if tuple(out[n].shape) != tuple(gen[n].shape) or out[n].dtype != gen[n].dtype:
            raise ValueError(f"EMA tensor {n!r} does not match the generator tensor")
    return out


def load_training_checkpoint(path, name: str, key: Optional[str] = None) -> StudentWeights:
    """EMA adapters from a training checkpoint (``torch.load`` on CPU, memory-mapped).

    Restricted deserialization (``weights_only=True``) by default; a checkpoint that holds other
    Python objects is unpickled in full only when ``FASTPIXAL3D_TRUST_CHECKPOINT=1`` is set (full unpickling
    can execute code: trusted files only)."""
    trust = os.environ.get("FASTPIXAL3D_TRUST_CHECKPOINT") == "1"

    def _load(weights_only: bool):
        try:
            return torch.load(str(path), map_location="cpu", mmap=True, weights_only=weights_only)
        except (TypeError, RuntimeError):
            return torch.load(str(path), map_location="cpu", weights_only=weights_only)

    try:
        payload = _load(True)
    except pickle.UnpicklingError as error:
        if not trust:
            raise RuntimeError(f"{path} holds objects other than tensors and plain containers; set "
                               f"FASTPIXAL3D_TRUST_CHECKPOINT=1 to load it with full unpickling if you trust the file") from error
        payload = _load(False)
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    state = {k: v.detach().clone() for k, v in _ema_from_payload(payload).items()}
    steps: Dict[str, Any] = {}
    if provenance.get("grid_indices") is not None:
        steps["grid_indices"] = [int(i) for i in provenance["grid_indices"]]
    spec = StudentSpec(name=name, rank=int(provenance.get("lora_rank", 256)),
                       alpha=float(provenance.get("lora_alpha", 256.0)), fold=DEFAULT_FOLD[name], steps=steps,
                       key=key or name)
    del payload
    return StudentWeights(spec=spec, state=state, source=str(path))


def load_training_checkpoints(paths: Mapping[str, str], names: Iterable[str] = STUDENTS,
                              keys: Optional[Mapping[str, str]] = None) -> Dict[str, StudentWeights]:
    out = {}
    for name in names:
        key = adapter_key(name, keys)
        if key not in paths:
            raise KeyError(f"no training checkpoint given for {key!r} (the {name} stage of this configuration)")
        out[name] = load_training_checkpoint(paths[key], name, key=key)
    return out


# ---------------------------------------------------------------------------
# name mapping
# ---------------------------------------------------------------------------
def relative_state(state: Mapping[str, torch.Tensor], model_names: Iterable[str]) -> Tuple[Dict[str, torch.Tensor], str]:
    """Strip the common prefix in front of the adapter parameter names. Returns
    (state keyed by model-relative names, detected prefix)."""
    model_names = set(model_names)
    keys = list(state)
    if not keys:
        raise ValueError("empty adapter state")
    if set(keys) == model_names:
        return dict(state), ""
    first = keys[0]
    candidates = sorted({first[: len(first) - len(m)] for m in model_names if first.endswith(m)}, key=len)
    for prefix in candidates:
        if all(k.startswith(prefix) for k in keys):
            stripped = {k[len(prefix):]: v for k, v in state.items()}
            if set(stripped) == model_names:
                return stripped, prefix
    raise KeyError(f"cannot map adapter tensor names onto the model (first name {first!r}; "
                   f"{len(keys)} tensors, {len(model_names)} adapter parameters)")


def check_target_modules(spec: StudentSpec, target_names: Iterable[str]) -> None:
    """If the manifest lists target modules, every wrapped layer must match one of them
    (by full name or by its last name components)."""
    if not spec.target_modules:
        return
    patterns = [str(t) for t in spec.target_modules]
    for name in target_names:
        if not any(name == p or name.endswith("." + p) or p in name.split(".") for p in patterns):
            raise ValueError(f"{spec.name}: wrapped layer {name!r} matches none of target_modules {patterns}")
