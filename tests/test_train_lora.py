"""Adapters attached for training: initialisation order, names, and the layout of the released students.

The architecture test builds the real Pixal3D flows on the ``meta`` device (no weights) and needs
the Pixal3D source checkout and the model configuration files; it is skipped without them:

    FASTPIXAL3D_PIXAL3D_SRC   Pixal3D source checkout (default: third_party/Pixal3D)
    FASTPIXAL3D_PIXAL3D       local directory with ckpts/*.json (a repository id is not downloaded here)
    FASTPIXAL3D_STUDENTS      optional: released students; their tensor names and shapes are compared too
"""
import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fastpixal3d.training import student as ST
from train_tiny_models import TinyDenseFlow

REPO = Path(__file__).resolve().parents[1]

# Layout of the released students (students.json of the release; docs/WEIGHTS.md).
TARGET_MODULES = ["self_attn.to_qkv", "self_attn.to_out", "cross_attn.cross_attn_block.to_q",
                  "cross_attn.cross_attn_block.to_kv", "cross_attn.cross_attn_block.to_out", "cross_attn.proj_linear",
                  "mlp.mlp.0", "mlp.mlp.2"]
RELEASED = {
    "structure": {"submodel": ST.STRUCTURE_SUBMODEL, "num_blocks": 30, "num_adapters": 240, "num_tensors": 480,
                  "num_elements": 318504960, "file": "structure_student.safetensors"},
    "texture": {"submodel": ST.TEXTURE_SUBMODEL, "num_blocks": 30, "num_adapters": 240, "num_tensors": 480,
                "num_elements": 326369280, "file": "texture_student.safetensors"},
}


def test_initialisation_order_matches_reference():
    """CPU draws in module order: Linear(in, r) and Linear(r, out) default init, then Kaiming on A, B = 0."""
    torch.manual_seed(7)
    flow = TinyDenseFlow()
    adapters = ST.attach_adapters(flow, 4, 4.0, 0.0)
    torch.manual_seed(7)
    ref_flow = TinyDenseFlow()
    targets = [m for n, m in ref_flow.named_modules()
               if isinstance(m, nn.Linear) and "blocks." in n and "adaLN_modulation" not in n]
    assert len(targets) == len(adapters) == 8
    for a, base in zip(adapters, targets):
        lin_a = nn.Linear(base.in_features, 4, bias=False)
        nn.Linear(4, base.out_features, bias=False)
        nn.init.kaiming_uniform_(lin_a.weight, a=math.sqrt(5))
        assert torch.equal(a.lora_A.weight, lin_a.weight)
        assert torch.count_nonzero(a.lora_B.weight) == 0
        assert a.scaling == 1.0 and a.lora_A.weight.dtype == torch.float32


def test_names_and_frozen_base():
    torch.manual_seed(0)
    flow = TinyDenseFlow().requires_grad_(False)
    net = ST.StructureStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))
    names = list(net.trainable())
    assert names[:2] == ["backbone.model.blocks.0.self_attn.to_qkv.lora_A.weight",
                         "backbone.model.blocks.0.self_attn.to_qkv.lora_B.weight"]
    assert len(names) == 16 and all(".lora_" in n for n in names)
    assert not any("adaLN" in n for n in names)


def test_student_equals_teacher_at_initialisation():
    torch.manual_seed(0)
    flow = TinyDenseFlow().requires_grad_(False).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 16, 16, 16, generator=g)
    cond = {"global": torch.randn(1, 5, 6, generator=g), "proj": torch.randn(1, 7, 12, generator=g)}
    t1000 = torch.tensor([900.0])
    before = flow(x, t1000, cond)
    ST.attach_adapters(flow, 4, 4.0, 0.0)
    assert torch.equal(flow(x, t1000, cond), before)


def _pixal3d_model_dir():
    src = Path(os.environ.get("FASTPIXAL3D_PIXAL3D_SRC") or REPO / "third_party" / "Pixal3D")
    model = os.environ.get("FASTPIXAL3D_PIXAL3D")
    if not (src / "pixal3d").is_dir() or not model or not Path(model).is_dir():
        pytest.skip("needs FASTPIXAL3D_PIXAL3D_SRC and a local FASTPIXAL3D_PIXAL3D directory")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from pixal3d import models  # noqa: F401
    except Exception as error:  # the package needs its compiled dependencies
        pytest.skip(f"pixal3d not importable: {error}")
    return Path(model)


def _safetensors_header(path):
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return {k: tuple(v["shape"]) for k, v in header.items()}


@pytest.mark.parametrize("student", sorted(RELEASED))
def test_adapter_layout_of_released_students(student):
    model_dir = _pixal3d_model_dir()
    from pixal3d import models

    spec = RELEASED[student]
    config = json.loads((model_dir / f"{spec['submodel']}.json").read_text())
    with torch.device("meta"):
        flow = getattr(models, config["name"])(**config["args"])
    flow.requires_grad_(False)
    net = ST.StudentNetwork(flow, ST.attach_adapters(flow, 256, 256.0, 0.0, init_device="meta"))
    params = net.trainable()
    assert len(net.adapters) == spec["num_adapters"] and len(params) == spec["num_tensors"]
    assert sum(p.numel() for p in params.values()) == spec["num_elements"]
    blocks = {n.split(".")[3] for n in params}
    assert len(blocks) == spec["num_blocks"]
    suffixes = []
    for n in params:
        layer = n.split(".", 4)[4].rsplit(".lora_", 1)[0]
        if layer not in suffixes:
            suffixes.append(layer)
    assert suffixes == TARGET_MODULES
    assert all(a.lora_A.weight.shape[0] == 256 and a.scaling == 1.0 for a in net.adapters)
    students = os.environ.get("FASTPIXAL3D_STUDENTS")
    if students and (Path(students) / spec["file"]).is_file():
        header = _safetensors_header(Path(students) / spec["file"])
        assert header == {n: tuple(p.shape) for n, p in params.items()}
        manifest = json.loads((Path(students) / "students.json").read_text())["students"][student]["lora"]
        assert manifest["target_modules"] == TARGET_MODULES and manifest["rank"] == 256 and manifest["alpha"] == 256
        assert manifest["num_elements"] == spec["num_elements"]
