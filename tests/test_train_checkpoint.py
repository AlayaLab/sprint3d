"""Checkpoints written by the training loops: resume, and conversion to the released weight format."""
import importlib.util
import json
import random
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")

from fastpixal3d.students import convert as CV  # noqa: E402
from fastpixal3d.students import load as LD  # noqa: E402
from fastpixal3d.training import checkpoint as CK  # noqa: E402
from fastpixal3d.training import config as C  # noqa: E402
from fastpixal3d.training import consistency as CD  # noqa: E402
from fastpixal3d.training import loop  # noqa: E402
from fastpixal3d.training import student as ST  # noqa: E402
from fastpixal3d.training.objective import MicroBatch  # noqa: E402
from train_tiny_models import Sparse, TinyDenseFlow, TinyShapeFlow, TinySparseFlow, shape_objects  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _defaults(name):
    spec = importlib.util.spec_from_file_location(f"trainer_{name}", REPO / "train" / name / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DEFAULTS


def _run(kind, out_dir, updates, resume="none"):
    name = {"structure": "structure_traj", "texture": "texture_traj"}[kind]
    cfg = C.load_config(None, _defaults(name), overrides=[
        "lora.rank=4", "lora.alpha=4.0", f"optim.updates={updates}", "optim.grad_accum=2", "optim.lr=0.01",
        "optim.autocast=null", f"output.dir={out_dir}", "output.checkpoint_every=2", "output.log_every=1",
        f"output.resume={resume}", "runtime.device=cpu"])
    run = loop.Run(cfg=cfg, device=torch.device("cpu"), rank=0, world_size=1, local_rank=0, model_dir=Path("."),
                   weights=None)
    s = cfg["schedule"]
    from fastpixal3d.training.schedule import segment_pairs, teacher_t_grid

    run.grid, run.pairs = teacher_t_grid(s["teacher_steps"], s["rescale_t"]), segment_pairs(s["grid_indices"])
    run.pool = [f"obj{i}.view00" for i in range(5)]
    torch.manual_seed(int(cfg["optim"]["seed"]))
    if kind == "structure":
        flow = TinyDenseFlow().requires_grad_(False).eval()
        run.net = ST.StructureStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))
    else:
        flow = TinySparseFlow().requires_grad_(False).eval()
        run.net = ST.TextureStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))
    run.params = run.net.trainable()

    def prepare(item):
        view, (i, j) = item
        g = torch.Generator().manual_seed(random.Random(view).randrange(1 << 30) + i)
        if kind == "structure":
            x, tgt = torch.randn(4096, 8, generator=g), torch.randn(4096, 8, generator=g)
            inputs = {"cond": {"global": torch.randn(1, 5, 6, generator=g), "proj": torch.randn(1, 3, 12, generator=g)}}
        else:
            n = 40
            crd = torch.zeros(n, 4, dtype=torch.int32)
            x, tgt = torch.randn(n, 32, generator=g), torch.randn(n, 32, generator=g)
            inputs = {"x": Sparse(x, crd), "concat": Sparse(torch.randn(n, 32, generator=g), crd),
                      "cond": {"global": torch.randn(1, 5, 6, generator=g)}}
        return MicroBatch(view, i, j, float(run.grid[i]), float(run.grid[j]), x, tgt, inputs)

    def forward(mb):
        if kind == "structure":
            return run.net(mb.x_i, mb.t_i, mb.inputs["cond"])
        return run.net(mb.inputs["x"], mb.t_i, mb.inputs["cond"], mb.inputs["concat"]).feats

    prov = loop.provenance_common(run)
    prov["student"] = kind
    schema = CK.STRUCTURE_SCHEMA if kind == "structure" else CK.TEXTURE_SCHEMA
    summary = loop.train(run, schema=schema, prepare=prepare, forward=forward, provenance=prov)
    return run, summary


def _run_shape(kind, out_dir, updates, resume="none", world_size=1):
    """The consistency trainer (shape students) on the tiny shape flow, one process."""
    spec = importlib.util.spec_from_file_location("trainer_shape_consistency", REPO / "train" / "shape_consistency" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = C.load_config(None, module.DEFAULTS, overrides=[
        f"model.student={kind}", "lora.rank=4", "lora.alpha=4.0", f"optim.updates={updates}", "optim.lr=0.01",
        "consistency.anchor_steps=2", f"output.dir={out_dir}", "output.checkpoint_every=2", "output.log_every=1",
        f"output.resume={resume}", "runtime.device=cpu", "data.conditioning=unused"])
    cfg = module.fill_released(cfg)
    CD.check(cfg)
    run = loop.Run(cfg=cfg, device=torch.device("cpu"), rank=0, world_size=world_size, local_rank=0, model_dir=Path("."),
                   weights=None)
    objects = shape_objects(torch.Generator().manual_seed(99))
    run.pool = sorted(objects)
    torch.manual_seed(int(cfg["optim"]["seed"]))
    flow = TinyShapeFlow().requires_grad_(False).eval()
    run.net = ST.ShapeStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))
    run.params = run.net.trainable()

    def load_object(object_id):
        rec = objects[object_id]
        coords = rec["coords"].clone()
        coords[:, 0] = 0
        return coords, {"global": rec["global"], "proj": Sparse(rec["proj"], coords)}

    prov = CD.provenance(cfg, world_size=world_size, n_objects=len(run.pool), trainer="test")
    schema = CK.COARSE_SHAPE_SCHEMA if kind == "coarse_shape" else CK.FINE_SHAPE_SCHEMA
    summary = CD.train(run, schema=schema, pool=run.pool, load_object=load_object, sparse=Sparse, provenance=prov)
    return run, summary


@pytest.mark.parametrize("kind", ["coarse_shape", "fine_shape"])
def test_shape_checkpoint_converts_to_release_format(kind, tmp_path):
    run, summary = _run_shape(kind, tmp_path / "run", updates=3)
    assert summary["last_step"] == 3 and summary["anchors"] == 3  # three distinct objects visited
    names = sorted(p.name for p in (tmp_path / "run").glob("completed_updates_*.pt"))
    assert names == ["completed_updates_000000000002.pt", "completed_updates_000000000003.pt"]
    payload = CK.load(tmp_path / "run" / names[-1])
    assert payload["schema_version"] == (3 if kind == "coarse_shape" else 4)
    assert payload["provenance"]["data_position"] == {"epoch": 0, "cursor": 3}
    assert payload["provenance"]["objective"] == "consistency_distillation" and payload["provenance"]["k"] == 1
    entry = CV.convert(kind, tmp_path / "run" / names[-1], tmp_path / "release")
    assert entry["steps"]["num_steps"] == 1 and entry["steps"]["update"] == "x0_prediction"
    assert entry["lora"]["rank"] == 4 and entry["training_updates"] == 3
    from safetensors.torch import load_file

    released = load_file(str(tmp_path / "release" / entry["file"]))
    for n, t in payload["ema"]["shadow"].items():
        assert torch.equal(released[n], t)
    rows = [json.loads(line) for line in (tmp_path / "run" / "loss.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3] and all(0.0 < r["ema_decay"] <= 0.99 for r in rows)
    assert rows[0]["ema_decay"] == 0.1 and rows[1]["ema_decay"] == 2 / 11


def test_shape_resume_continues_the_data_order(tmp_path):
    whole, _ = _run_shape("coarse_shape", tmp_path / "whole", updates=5)
    _run_shape("coarse_shape", tmp_path / "split", updates=3)
    resumed, summary = _run_shape("coarse_shape", tmp_path / "split", updates=5, resume="auto")
    assert summary["updates"] == 2 and summary["last_step"] == 5
    a = CK.load(tmp_path / "whole" / "completed_updates_000000000005.pt")
    b = CK.load(tmp_path / "split" / "completed_updates_000000000005.pt")
    assert a["provenance"]["data_position"] == b["provenance"]["data_position"] == {"epoch": 1, "cursor": 0}
    rows_a = [json.loads(l)["object"] for l in (tmp_path / "whole" / "loss.jsonl").read_text().splitlines()]
    rows_b = [json.loads(l)["object"] for l in (tmp_path / "split" / "loss.jsonl").read_text().splitlines()]
    assert rows_a == rows_b  # the same objects in the same order
    for n in a["gen"]:  # the same adapters: resume replays the draws of updates 4 and 5
        assert torch.allclose(a["gen"][n], b["gen"][n], atol=1e-6) and torch.allclose(a["ema"]["shadow"][n], b["ema"]["shadow"][n], atol=1e-6)
    with pytest.raises(RuntimeError):
        _run_shape("coarse_shape", tmp_path / "split", updates=6, resume="auto", world_size=2)  # another world size


@pytest.mark.parametrize("kind", ["structure", "texture"])
def test_checkpoint_converts_to_release_format(kind, tmp_path):
    run, summary = _run(kind, tmp_path / "run", updates=3)
    assert summary["last_step"] == 3
    names = sorted(p.name for p in (tmp_path / "run").glob("completed_updates_*.pt"))
    assert names == ["completed_updates_000000000002.pt", "completed_updates_000000000003.pt"]
    ckpt = tmp_path / "run" / names[-1]
    payload = CK.load(ckpt)
    assert payload["completed_updates"] == 3 and payload["schema_version"] == (1 if kind == "structure" else 2)
    assert list(payload["gen"]) == list(run.params) and list(payload["ema"]["shadow"]) == list(run.params)
    entry = CV.convert(kind, ckpt, tmp_path / "release")  # weights-only loading: no pickled objects needed
    assert entry["lora"]["rank"] == 4 and entry["lora"]["alpha"] == 4 and entry["training_updates"] == 3
    assert entry["lora"]["target_modules"] == ["self_attn.to_qkv", "self_attn.to_out", "mlp.mlp.0", "mlp.mlp.2"]
    from safetensors.torch import load_file

    released = load_file(str(tmp_path / "release" / entry["file"]))
    assert set(released) == set(payload["ema"]["shadow"])
    for n, t in payload["ema"]["shadow"].items():
        assert torch.equal(released[n], t)
    # the released file binds to a freshly wrapped flow under the loader's name mapping
    rel, prefix = LD.relative_state(released, [n for n, _ in run.net.backbone.model.named_parameters() if ".lora_" in n])
    assert prefix == "backbone.model."


def test_resume_continues_from_the_newest_checkpoint(tmp_path):
    _run("structure", tmp_path / "run", updates=3)
    first = CK.load(tmp_path / "run" / "completed_updates_000000000003.pt")
    run, summary = _run("structure", tmp_path / "run", updates=5, resume="auto")
    assert summary["updates"] == 2 and summary["last_step"] == 5
    final = CK.load(tmp_path / "run" / "completed_updates_000000000005.pt")
    assert final["completed_updates"] == 5
    assert final["opt_gen"]["state"][0]["step"].item() == 5  # AdamW state carried over
    assert not torch.equal(final["gen"][next(iter(final["gen"]))], first["gen"][next(iter(first["gen"]))])
