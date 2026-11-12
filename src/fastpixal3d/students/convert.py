"""Convert a student training checkpoint into the released weight format.

Released format
---------------
A directory with ``students.json`` and one float32 safetensors file per student. Each file holds
only the exponential-moving-average (EMA) LoRA weights of the student. The keys are the parameter
names of the base flow model after LoRA wrapping::

    backbone.model.<path>.lora_A.weight    shape [rank, in_features]
    backbone.model.<path>.lora_B.weight    shape [out_features, rank]

where ``<path>`` is the name of an ``nn.Linear`` in the base flow model (as in its state dict, e.g.
``blocks.0.self_attn.to_qkv``). The weight update of that layer is ``(alpha / rank) * B @ A``.

Training checkpoint
-------------------
A training checkpoint (``torch.save`` of a dict) stores the trainable LoRA weights under ``gen``
(name -> tensor, in the order of ``named_parameters()``) and their EMA under ``ema["shadow"]``.
The shadow is either name-keyed (a dict with the same names as ``gen``) or positional (a list in
the insertion order of ``gen``). A positional shadow is named by zipping it with the keys of
``gen`` in insertion order; every EMA tensor must then have the shape and dtype of the ``gen``
tensor it is paired with. The naming is the loader's own (``load._ema_from_payload``), and the
bf16 fold that ``students.json`` prescribes for some students is implemented in ``fold.py``.

Usage::

    python -m fastpixal3d.students.convert --student fine_shape --checkpoint CKPT.pt --out DIR

The command writes ``<out>/<student file>``, adds or replaces the student's entry in
``<out>/students.json``, and re-reads the written file to check it bit for bit against the
checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .load import DEFAULT_FOLD, FOLD_BF16, FOLD_NONE, PIPELINE_MODEL_KEYS, _ema_from_payload, sha256_file

FORMAT_VERSION = 1

BASE_MODEL = {"repo": "TencentARC/Pixal3D", "revision": "0b31f9160aa400719af409098bff7936a932f726"}

KEY_PREFIX = "backbone.model."
METADATA_KEY = "student_lora"
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")

KEY_CONVENTION = (
    "Each key is 'backbone.model.<path>.lora_A.weight' or 'backbone.model.<path>.lora_B.weight', where "
    "<path> names an nn.Linear in the state dict of the base flow model given by base_submodel (minus its "
    "'.weight' suffix); lora_A has shape [rank, in_features], lora_B has shape [out_features, rank], and "
    "the layer's weight update is (alpha / rank) * lora_B @ lora_A; example: "
    "'backbone.model.blocks.0.self_attn.to_qkv.lora_A.weight'."
)

FOLD_RULE = {
    FOLD_NONE: (
        "The adapters stay separate in float32: y = base(x) + (lora_B(lora_A(x.float())) * (alpha / rank))"
        ".to(base(x).dtype)."
    ),
    FOLD_BF16: (
        "At load time each adapted weight W (bfloat16, the dtype of the base model's transformer blocks) is "
        "replaced by W + ((lora_B.float() @ lora_A.float()) * (alpha / rank)).to(bfloat16), the sum being "
        "taken in bfloat16, and the adapters are removed."
    ),
}

_TEACHER_GRID = "t = linspace(1, 0, 13) warped by t' = 3t / (1 + 2t); grid_indices index this grid"
_X0_FORMULA = "x0 = (1 - s) * x_t - (s + (1 - s) * t) * v with s = 1e-5, in float64"

_STUDENT_FILES = {
    "structure": ("structure_student.safetensors", "ckpts/ss_flow_img_dit_1_3B_64_bf16", {
        "num_steps": 4, "t": [1.0, 0.9, 0.75, 0.6, 0.0], "grid_indices": [0, 3, 6, 8, 12],
        "teacher_grid": _TEACHER_GRID, "update": "euler", "guidance_strength": 1.0}),
    "coarse_shape": ("coarse_shape_student.safetensors", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16", {
        "num_steps": 1, "t": [1.0, 0.0], "update": "x0_prediction", "guidance_strength": 1.0,
        "x0_formula": _X0_FORMULA}),
    "fine_shape": ("fine_shape_student.safetensors", "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16", {
        "num_steps": 1, "t": [1.0, 0.0], "update": "x0_prediction", "guidance_strength": 1.0,
        "x0_formula": _X0_FORMULA}),
    "texture": ("texture_student.safetensors", "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16", {
        "num_steps": 2, "k": 2, "m": 6, "t": [1.0, 0.75, 0.0], "grid_indices": [0, 6, 12],
        "teacher_grid": _TEACHER_GRID, "update": "euler", "guidance_strength": 1.0}),
}
# pipeline key and fold rule come from the loader, so both sides use one definition
STUDENTS: dict[str, dict[str, Any]] = {
    name: {"file": file, "base_submodel": submodel, "pipeline_key": PIPELINE_MODEL_KEYS[name],
           "fold": DEFAULT_FOLD[name], "steps": steps}
    for name, (file, submodel, steps) in _STUDENT_FILES.items()
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_IN_BLOCK = re.compile(r"(?:^|\.)blocks\.(\d+)\.(.+)$")


class ConversionError(ValueError):
    """The checkpoint or the release files do not have the expected structure."""


# --------------------------------------------------------------------------------------------- #
# reading a training checkpoint
# --------------------------------------------------------------------------------------------- #

def load_training_checkpoint(path: str | os.PathLike, allow_pickle: bool = False) -> dict:
    """Load a training checkpoint on CPU (memory-mapped when the file format allows it).

    Checkpoints that contain more than tensors and plain containers (for example random-number
    generator states) can only be read with full unpickling, which can execute code; pass
    ``allow_pickle=True`` only for checkpoints you trust.
    """
    import torch

    kwargs = {"map_location": "cpu", "weights_only": not allow_pickle}
    try:
        try:
            return torch.load(str(path), mmap=True, **kwargs)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            return torch.load(str(path), **kwargs)
    except pickle.UnpicklingError as error:
        raise ConversionError(
            f"{path} holds objects other than tensors; if you trust the file, load it with "
            "allow_pickle=True (command line: --allow-pickle)") from error


# --------------------------------------------------------------------------------------------- #
# LoRA structure
# --------------------------------------------------------------------------------------------- #

def module_path(key: str) -> str:
    """'backbone.model.blocks.0.mlp.mlp.0.lora_A.weight' -> 'blocks.0.mlp.mlp.0'."""
    if not key.startswith(KEY_PREFIX) or not key.endswith(LORA_SUFFIXES):
        raise ConversionError(f"key {key!r} does not follow '{KEY_PREFIX}<path>.lora_{{A,B}}.weight'")
    return key[len(KEY_PREFIX):-len(".lora_A.weight")]


def describe_lora(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Rank, target modules and counts of a LoRA state dict; checks that A/B pairs are complete."""
    paths: dict[str, dict[str, Any]] = {}
    for key, tensor in state_dict.items():
        part = "A" if key.endswith(".lora_A.weight") else "B"
        paths.setdefault(module_path(key), {})[part] = tensor
    ranks = set()
    for path, pair in paths.items():
        if set(pair) != {"A", "B"}:
            raise ConversionError(f"module {path!r} lacks lora_A or lora_B")
        a, b = pair["A"], pair["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ConversionError(f"module {path!r}: lora_A {tuple(a.shape)} and lora_B {tuple(b.shape)} do not chain")
        ranks.add(int(a.shape[0]))
    if len(ranks) != 1:
        raise ConversionError(f"expected one LoRA rank, found {sorted(ranks)}")
    # target modules are layer names inside a transformer block, e.g. "self_attn.to_qkv"; a wrapped
    # layer such as "blocks.7.self_attn.to_qkv" ends with one of them
    patterns, blocks = [], set()
    for path in paths:
        match = _IN_BLOCK.search(path)
        pattern = match.group(2) if match else path
        if match:
            blocks.add(int(match.group(1)))
        if pattern not in patterns:
            patterns.append(pattern)
    dtypes = sorted({str(t.dtype).replace("torch.", "") for t in state_dict.values()})
    return {
        "rank": ranks.pop(),
        "target_modules": patterns,
        "num_blocks": len(blocks),
        "num_adapters": len(paths),
        "num_tensors": len(state_dict),
        "num_elements": int(sum(t.numel() for t in state_dict.values())),
        "dtypes": dtypes,
    }


# --------------------------------------------------------------------------------------------- #
# students.json
# --------------------------------------------------------------------------------------------- #

def empty_students_json() -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "base_model": dict(BASE_MODEL),
        "key_convention": KEY_CONVENTION,
        "fold_rule": dict(FOLD_RULE),
        "students": {},
    }


def validate_students_json(doc: Mapping[str, Any], require_all: bool = True) -> None:
    """Raise ``ConversionError`` if ``doc`` does not follow the students.json schema."""

    def need(condition: bool, message: str) -> None:
        if not condition:
            raise ConversionError(f"students.json: {message}")

    need(doc.get("format") == FORMAT_VERSION, f"format must be {FORMAT_VERSION}")
    base = doc.get("base_model")
    need(isinstance(base, Mapping) and isinstance(base.get("repo"), str) and base["repo"], "base_model.repo missing")
    need(isinstance(base.get("revision"), str) and bool(_HEX40.match(base["revision"])),
         "base_model.revision must be a full 40-character commit hash")
    need(isinstance(doc.get("key_convention"), str) and KEY_PREFIX in doc["key_convention"],
         "key_convention must describe the key prefix")
    students = doc.get("students")
    need(isinstance(students, Mapping), "students must be a mapping")
    if require_all:
        need(set(students) == set(STUDENTS), f"students must be exactly {sorted(STUDENTS)}")
    for name, entry in students.items():
        need(name in STUDENTS, f"unknown student {name!r}")
        need(isinstance(entry.get("file"), str) and entry["file"].endswith(".safetensors"), f"{name}.file")
        need(isinstance(entry.get("sha256"), str) and bool(_HEX64.match(entry["sha256"])), f"{name}.sha256")
        need(isinstance(entry.get("size_bytes"), int) and entry["size_bytes"] > 0, f"{name}.size_bytes")
        need(isinstance(entry.get("base_submodel"), str) and entry["base_submodel"], f"{name}.base_submodel")
        lora = entry.get("lora")
        need(isinstance(lora, Mapping), f"{name}.lora")
        need(isinstance(lora.get("rank"), int) and lora["rank"] > 0, f"{name}.lora.rank")
        need(isinstance(lora.get("alpha"), (int, float)) and lora["alpha"] > 0, f"{name}.lora.alpha")
        need(isinstance(lora.get("target_modules"), list) and lora["target_modules"], f"{name}.lora.target_modules")
        need(lora.get("num_tensors") == 2 * lora.get("num_adapters", -1), f"{name}.lora.num_tensors")
        steps = entry.get("steps")
        need(isinstance(steps, Mapping) and steps.get("num_steps", 0) >= 1, f"{name}.steps")
        if "grid_indices" in steps:
            need(len(steps["grid_indices"]) == len(steps.get("t", [])) == steps["num_steps"] + 1,
                 f"{name}.steps.grid_indices must have one index per t value")
        need(entry.get("fold") in FOLD_RULE, f"{name}.fold must be one of {sorted(FOLD_RULE)}")
        need(entry.get("dtype") == "float32", f"{name}.dtype must be float32")
        need(isinstance(entry.get("source_checkpoint_sha256"), str)
             and bool(_HEX64.match(entry["source_checkpoint_sha256"])), f"{name}.source_checkpoint_sha256")


# --------------------------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------------------------- #

def _plain_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def convert(student: str, checkpoint: str | os.PathLike, out_dir: str | os.PathLike, *,
            lora_alpha: float | None = None, allow_pickle: bool = False, overwrite: bool = False,
            checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Write the EMA LoRA of ``checkpoint`` as ``<out_dir>/<file>`` and update students.json.

    ``lora_alpha`` defaults to the value stored in the checkpoint's provenance, else to the rank.
    Returns the students.json entry.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if student not in STUDENTS:
        raise ConversionError(f"unknown student {student!r}; expected one of {sorted(STUDENTS)}")
    spec = STUDENTS[student]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / spec["file"]
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (pass overwrite=True to replace it)")

    source_sha = checkpoint_sha256 or sha256_file(checkpoint)
    payload = load_training_checkpoint(checkpoint, allow_pickle=allow_pickle)
    lora = _ema_from_payload(payload)  # positional EMA is named in the insertion order of payload["gen"]
    info = describe_lora(lora)
    if info["dtypes"] != ["float32"]:
        raise ConversionError(f"EMA tensors must be float32, found {info['dtypes']}")

    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    if provenance.get("lora_rank") is not None and int(provenance["lora_rank"]) != info["rank"]:
        raise ConversionError(f"provenance rank {provenance['lora_rank']} != tensor rank {info['rank']}")
    stored_alpha = provenance.get("lora_alpha")
    if lora_alpha is None:
        lora_alpha = float(stored_alpha) if stored_alpha is not None else float(info["rank"])
    elif stored_alpha is not None and float(stored_alpha) != float(lora_alpha):
        raise ConversionError(f"lora_alpha {lora_alpha} differs from the checkpoint's {stored_alpha}")

    steps = json.loads(json.dumps(spec["steps"]))
    if provenance.get("grid_t_values") is not None and [float(t) for t in provenance["grid_t_values"]] != steps["t"]:
        raise ConversionError(f"checkpoint step grid {provenance['grid_t_values']} != {steps['t']}")
    if provenance.get("grid_indices") is not None and [int(i) for i in provenance["grid_indices"]] != steps.get(
            "grid_indices"):
        raise ConversionError(f"checkpoint grid indices {provenance['grid_indices']} != {steps.get('grid_indices')}")
    if provenance.get("k") is not None and int(provenance["k"]) != steps["num_steps"]:
        raise ConversionError(f"checkpoint was trained for {provenance['k']} steps, expected {steps['num_steps']}")

    tensors = {name: tensor.contiguous() for name, tensor in lora.items()}
    # One metadata entry only: safetensors writes several entries in an arbitrary order, which would
    # make the file bytes (and its sha256) differ between runs.
    metadata = {METADATA_KEY: json.dumps({
        "student": student,
        "base_model": f"{BASE_MODEL['repo']}@{BASE_MODEL['revision']}",
        "base_submodel": spec["base_submodel"],
        "lora_rank": info["rank"],
        "lora_alpha": _plain_number(lora_alpha),
    }, sort_keys=True, separators=(",", ":"))}
    tmp = target.with_name(target.name + ".tmp")
    save_file(tensors, str(tmp), metadata=metadata)
    with safe_open(str(tmp), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise ConversionError("written file has different keys")
        for name in handle.keys():
            written = handle.get_tensor(name)
            source = tensors[name]
            if written.dtype != source.dtype or written.shape != source.shape or not torch.equal(
                    written.view(torch.int32), source.view(torch.int32)):
                raise ConversionError(f"written tensor {name!r} differs from the checkpoint")
    os.replace(tmp, target)

    entry = {
        "file": spec["file"],
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "base_submodel": spec["base_submodel"],
        "pipeline_key": spec["pipeline_key"],
        "lora": {
            "rank": info["rank"],
            "alpha": _plain_number(lora_alpha),
            "target_modules": info["target_modules"],
            "num_blocks": info["num_blocks"],
            "num_adapters": info["num_adapters"],
            "num_tensors": info["num_tensors"],
            "num_elements": info["num_elements"],
        },
        "steps": steps,
        "fold": spec["fold"],
        "dtype": "float32",
        "source_checkpoint_sha256": source_sha,
        "training_updates": int(payload.get("completed_updates", -1)),
    }
    manifest = out_dir / "students.json"
    # the shared fields are always rewritten from this module; only other students' entries are kept
    doc = empty_students_json()
    previous = json.loads(manifest.read_text()).get("students", {}) if manifest.exists() else {}
    previous[student] = entry
    doc["students"] = {name: previous[name] for name in STUDENTS if name in previous}
    validate_students_json(doc, require_all=False)
    tmp_manifest = manifest.with_name(manifest.name + ".tmp")
    tmp_manifest.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp_manifest, manifest)
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--student", required=True, choices=sorted(STUDENTS))
    parser.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="output directory (students.json + safetensors)")
    parser.add_argument("--lora-alpha", type=float, default=None,
                        help="LoRA alpha (default: from the checkpoint provenance, else the rank)")
    parser.add_argument("--checkpoint-sha256", default=None, help="skip hashing the checkpoint again")
    parser.add_argument("--allow-pickle", action="store_true",
                        help="unpickle non-tensor objects in the checkpoint (only for trusted files)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    entry = convert(args.student, args.checkpoint, args.out, lora_alpha=args.lora_alpha,
                    allow_pickle=args.allow_pickle, overwrite=args.overwrite,
                    checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps({args.student: entry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
