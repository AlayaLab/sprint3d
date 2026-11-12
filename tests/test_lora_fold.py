import torch
import torch.nn as nn

from fastpixal3d.students.fold import fold_into, fold_lora, lora_delta
from fastpixal3d.students.lora import (LoRALinear, apply_lora, bind_lora_state, is_lora_target,
                                       lora_parameter_names, lora_target_names)


class Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.to_qkv = nn.Linear(d, 3 * d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))


class Tiny(nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.input_layer = nn.Linear(d, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])


def _model(dtype=torch.bfloat16):
    torch.manual_seed(0)
    m = Tiny()
    for b in m.blocks:
        b.to(dtype)
    return m


def test_targets_exclude_modulation_and_outside_blocks():
    m = _model()
    names = lora_target_names(m)
    assert names == ["blocks.0.to_qkv", "blocks.0.mlp.0", "blocks.0.mlp.2",
                     "blocks.1.to_qkv", "blocks.1.mlp.0", "blocks.1.mlp.2"]
    assert not any("adaLN" in n for n in names)
    assert not is_lora_target("input_layer", m.input_layer)


def test_apply_bind_and_fold_rule_is_exact():
    m = _model()
    adapters = apply_lora(m, rank=4, alpha=4.0)
    assert len(adapters) == 6 and all(a.scaling == 1.0 for a in adapters)
    g = torch.Generator().manual_seed(1)
    state = {n: torch.randn(p.shape, generator=g) * 0.05 for n, p in m.named_parameters() if "lora_" in n}
    assert sorted(state) == sorted(lora_parameter_names(m))
    assert bind_lora_state(m, state) == 12
    expected = {}
    for name, module in m.named_modules():
        if isinstance(module, LoRALinear):
            w = module.base.weight.detach().clone()
            delta = (module.lora_B.weight.float() @ module.lora_A.weight.float()) * float(module.scaling)
            expected[name] = w + delta.to(torch.bfloat16)
    x = torch.randn(5, 16, generator=g).to(torch.bfloat16)
    unfolded = m.blocks[0].to_qkv(x)
    assert fold_lora(m) == 6
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    for name, w in expected.items():
        folded = m.get_submodule(name).weight
        assert folded.dtype == torch.bfloat16
        assert torch.equal(folded, w)
    folded_out = m.blocks[0].to_qkv(x)
    assert torch.allclose(folded_out.float(), unfolded.float(), atol=0.1, rtol=0.05)


def test_fold_into_matches_formula_and_zero_delta_is_noop():
    g = torch.Generator().manual_seed(2)
    w = torch.randn(8, 6, generator=g).to(torch.bfloat16)
    a = torch.randn(3, 6, generator=g)
    b = torch.zeros(8, 3)
    w0 = w.clone()
    fold_into(w, a, b, 1.0)
    assert torch.equal(w, w0)
    b = torch.randn(8, 3, generator=g)
    expected = w0 + lora_delta(a, b, 2.0).to(torch.bfloat16)
    fold_into(w, a, b, 2.0)
    assert torch.equal(w, expected)


def test_bind_is_strict():
    m = _model()
    apply_lora(m, rank=2, alpha=2.0)
    state = {n: torch.zeros_like(p) for n, p in m.named_parameters() if "lora_" in n}
    state.pop(next(iter(state)))
    try:
        bind_lora_state(m, state)
    except KeyError:
        pass
    else:
        raise AssertionError("missing adapter tensor not detected")
