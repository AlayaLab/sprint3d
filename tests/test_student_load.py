import json

import pytest
import torch

from fastpixal3d.students import load as L


def test_relative_state_strips_prefix_without_hardcoding():
    names = ["blocks.0.to_q.lora_A.weight", "blocks.0.to_q.lora_B.weight", "blocks.1.mlp.0.lora_A.weight",
             "blocks.1.mlp.0.lora_B.weight"]
    for prefix in ("", "backbone.model.", "student.flow."):
        state = {prefix + n: torch.zeros(1) for n in names}
        rel, got = L.relative_state(state, names)
        assert got == prefix and sorted(rel) == sorted(names)
    with pytest.raises(KeyError):
        L.relative_state({"x." + names[0]: torch.zeros(1)}, names)


def test_positional_ema_follows_generator_order():
    gen = {"b": torch.zeros(2), "a": torch.zeros(3)}
    payload = {"gen": gen, "ema": {"shadow": [torch.ones(2), torch.full((3,), 2.0)]}}
    ema = L._ema_from_payload(payload)
    assert list(ema) == ["b", "a"] and float(ema["a"][0]) == 2.0
    with pytest.raises(ValueError):
        L._ema_from_payload({"gen": gen, "ema": {"shadow": [torch.ones(2)]}})


def test_release_manifest_roundtrip(tmp_path):
    from safetensors.torch import save_file

    entries = {}
    for name in L.STUDENTS:
        fname = f"{name}_student.safetensors"
        save_file({"backbone.model.blocks.0.to_q.lora_A.weight": torch.zeros(2, 4)}, str(tmp_path / fname))
        entries[name] = {"file": fname, "sha256": L.sha256_file(tmp_path / fname), "base_submodel": "ckpts/x",
                         "lora": {"rank": 2, "alpha": 2, "target_modules": ["to_q"]}, "steps": {},
                         "fold": L.DEFAULT_FOLD[name], "dtype": "float32"}
    (tmp_path / "students.json").write_text(json.dumps({"format": 1, "base_model": {"repo": "r", "revision": "x"},
                                                        "key_convention": "prefix + parameter name",
                                                        "students": entries}))
    loaded = L.load_release(tmp_path, verify_sha256=True)
    assert set(loaded) == set(L.STUDENTS)
    assert loaded["fine_shape"].spec.fold == "bf16" and loaded["structure"].spec.fold == "none"
    assert loaded["texture"].spec.rank == 2
    entries["texture"]["sha256"] = "0" * 64
    (tmp_path / "students.json").write_text(json.dumps({"format": 1, "students": entries}))
    with pytest.raises(ValueError):
        L.load_release(tmp_path, names=["texture"], verify_sha256=True)


def test_target_module_check():
    spec = L.StudentSpec(name="texture", target_modules=["to_q", "mlp.0"])
    L.check_target_modules(spec, ["blocks.0.to_q", "blocks.3.mlp.0"])
    with pytest.raises(ValueError):
        L.check_target_modules(spec, ["blocks.0.to_out"])


def test_manifest_grid_indices_and_pipeline_key():
    from fastpixal3d.stages.structure import student_t_values
    from fastpixal3d.students.load import spec_from_manifest

    entry = {"file": "s.safetensors", "pipeline_key": "sparse_structure_flow_model", "lora": {"rank": 4, "alpha": 4},
             "steps": {"t": [1.0, 0.9, 0.75, 0.6, 0.0], "teacher_grid": {"indices": [0, 3, 6, 8, 12]}}}
    spec = spec_from_manifest("structure", entry)
    assert spec.grid_indices() == (0, 3, 6, 8, 12)
    assert spec.model_key == "sparse_structure_flow_model"
    assert student_t_values(spec.grid_indices(), 3.0, 12) == pytest.approx(entry["steps"]["t"], abs=1e-9)
    entry["steps"]["grid_indices"] = [0, 6, 12]
    assert spec_from_manifest("structure", entry).grid_indices() == (0, 6, 12)
    assert spec_from_manifest("texture", {"file": "t"}).model_key == "tex_slat_flow_model_1024"
