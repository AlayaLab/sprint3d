"""The earlier (consistency) structure student and the selection of adapters by students.json key."""
import json

import pytest
import torch

from fastpixal3d.stages import structure as S
from fastpixal3d.students import load as L

SIGMA_MIN = 1e-5


class FakeFlow(torch.nn.Module):
    """A deterministic dense 'velocity' depending on x, t and the conditioning."""

    def forward(self, x, t, cond):
        assert x.shape == (1, 8, 16, 16, 16) and t.shape == (1,) and t.dtype == torch.float32
        return torch.tanh(x * 0.7 + t.view(1, 1, 1, 1, 1) / 1000.0) * cond["scale"] - 0.1 * x


def _reference_rows(noise, cond, query_t, seed):
    """The rollout as the measurement code writes it: rows of the full grid in coordinate order,
    velocity via scatter -> dense forward -> gather, x0 in float64, re-noising with [N, C] draws."""
    R = noise.shape[2]
    grid = torch.stack(torch.meshgrid(torch.arange(R), torch.arange(R), torch.arange(R), indexing="ij"), -1).reshape(-1, 3)
    gather = lambda d: d[0, :, grid[:, 0], grid[:, 1], grid[:, 2]].T  # noqa: E731

    def scatter(rows):
        d = rows.new_zeros(1, rows.shape[1], R, R, R)
        d[0, :, grid[:, 0], grid[:, 1], grid[:, 2]] = rows.T
        return d

    gen = torch.Generator(device=noise.device)
    gen.manual_seed(seed)
    model = FakeFlow()
    x = gather(noise)
    for i, t in enumerate(query_t):
        t1000 = torch.as_tensor(t, dtype=torch.float32).reshape(-1) * 1000.0
        v = gather(model(scatter(x), t1000, cond))
        xd, vd = x.double(), v.double()
        tt = torch.as_tensor(t, dtype=torch.float64).reshape(1, 1).expand(x.shape[0], 1)
        x0 = ((1.0 - SIGMA_MIN) * xd - (SIGMA_MIN + (1.0 - SIGMA_MIN) * tt) * vd).to(x.dtype)
        if i == len(query_t) - 1:
            return scatter(x0)
        eps = torch.randn(x0.shape, dtype=x0.dtype, generator=gen)
        tn = torch.as_tensor(query_t[i + 1], dtype=x0.dtype).reshape(1, 1).expand(x0.shape[0], 1)
        x = (1.0 - tn) * x0 + (SIGMA_MIN + (1.0 - SIGMA_MIN) * tn) * eps


def test_consistency_rollout_matches_the_row_formulation_bitwise():
    noise = S.initial_noise("cpu", 42)
    cond = {"scale": 0.9}
    q = (1.0, 0.9, 0.75, 0.5)
    ours = S.consistency_rollout(FakeFlow(), noise, cond, q, renoise_seed=42)
    ref = _reference_rows(noise, cond, q, 42)
    assert torch.equal(ours, ref)
    other = S.consistency_rollout(FakeFlow(), noise, cond, q, renoise_seed=7)
    assert not torch.equal(ours, other)
    student = S.StructureStudent(FakeFlow(), q, use_graph=False, sampler=S.CONSISTENCY, renoise_seed=42)
    stats = {}
    assert torch.equal(student(cond, noise, stats), ours) and stats["sampler"] == "consistency"


def test_sampler_of_students_json_steps():
    assert S.sampler_of({"update": "euler"}) == S.TRAJECTORY
    assert S.sampler_of({}) == S.TRAJECTORY
    for update in ("x0_prediction", "consistency", "x0_prediction_renoise"):
        assert S.sampler_of({"update": update}) == S.CONSISTENCY
    with pytest.raises(ValueError):
        S.sampler_of({"update": "heun"})


def test_adapter_keys_and_stage_of():
    assert L.stage_of("structure_consistency") == "structure"
    assert L.stage_of("anything", {"stage": "texture"}) == "texture"
    with pytest.raises(ValueError):
        L.stage_of("unknown_adapter")
    assert L.adapter_key("structure", {"structure": "structure_consistency"}) == "structure_consistency"
    assert L.adapter_key("coarse_shape", {"structure": "structure_consistency"}) == "coarse_shape"
    with pytest.raises(ValueError):
        L.adapter_key("structure", {"structure": "texture_other"})


def test_missing_adapter_entry_names_the_key(tmp_path):
    entry = {"file": "x.safetensors", "lora": {"rank": 4, "alpha": 4}, "fold": "none"}
    (tmp_path / "students.json").write_text(json.dumps({"format": 1, "students": {
        name: dict(entry) for name in L.STUDENTS}}))
    with pytest.raises(KeyError, match="structure_consistency"):
        L.load_release(tmp_path, ["structure"], keys={"structure": "structure_consistency"})
    with pytest.raises(KeyError, match="structure_consistency"):
        L.load_training_checkpoints({"structure": "a.pt"}, ["structure"], keys={"structure": "structure_consistency"})
    spec = L.spec_from_manifest("structure", {**entry, "role": "experiments-only"}, key="structure_consistency")
    assert spec.key == "structure_consistency" and spec.name == "structure" and spec.role == "experiments-only"


STRUCTURE_CONSISTENCY_ENTRY = {
    "role": "experiments-only", "file": "structure_student_consistency.safetensors",
    "base_submodel": "ckpts/ss_flow_img_dit_1_3B_64_bf16", "pipeline_key": "sparse_structure_flow_model",
    "lora": {"rank": 256, "alpha": 256}, "fold": "none", "dtype": "float32",
    "steps": {"num_steps": 4, "t": [1.0, 0.9, 0.75, 0.5, 0.0], "update": "x0_prediction_renoise",
              "guidance_strength": 1.0}}


def test_spec_of_the_experiments_only_structure_entry():
    for name in ("structure_consistency", "structure"):  # by its own key, or as the structure stage
        spec = L.spec_from_manifest(name, STRUCTURE_CONSISTENCY_ENTRY, key="structure_consistency")
        assert spec.fold == L.FOLD_NONE and spec.model_key == "sparse_structure_flow_model"
        assert spec.role == "experiments-only" and spec.key == "structure_consistency"
        assert S.sampler_of(spec.steps) == S.CONSISTENCY
    no_fold = {k: v for k, v in STRUCTURE_CONSISTENCY_ENTRY.items() if k not in ("fold", "pipeline_key")}
    spec = L.spec_from_manifest("structure_consistency", no_fold)
    assert spec.fold == L.FOLD_NONE and spec.model_key == "sparse_structure_flow_model"
    # the four deployed students keep their defaults
    assert L.spec_from_manifest("fine_shape", {"lora": {}}).fold == L.FOLD_BF16
    assert L.spec_from_manifest("texture", {"lora": {}}).fold == L.FOLD_BF16
    assert L.spec_from_manifest("coarse_shape", {"lora": {}}).fold == L.FOLD_NONE
