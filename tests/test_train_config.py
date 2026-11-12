"""Configuration files of the trainers and the object lists they point to."""
import csv
import importlib.util
from pathlib import Path

import pytest

from fastpixal3d.training import config as C
from fastpixal3d.training.records import split_view_name, view_name, views_from_list

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "train" / "configs"


def trainer(name):
    spec = importlib.util.spec_from_file_location(f"trainer_{name}", REPO / "train" / name / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,config", [("structure_traj", "structure_student.yaml"),
                                         ("texture_traj", "texture_student.yaml")])
def test_shipped_configs_load_and_equal_the_defaults(name, config):
    """The YAML files spell out the released settings; they must agree with the script defaults."""
    module = trainer(name)
    cfg = module.load(["--config", str(CONFIGS / config)])
    defaults = C._strip_markers(module.DEFAULTS)
    for section in ("lora", "schedule", "optim"):
        assert cfg[section] == defaults[section], section
    assert cfg["model"]["base_submodel"] == defaults["model"]["base_submodel"]
    assert Path(cfg["data"]["views"]).is_file()  # resolved against train/configs/


def test_released_settings():
    s = trainer("structure_traj").load(["--config", str(CONFIGS / "structure_student.yaml")])
    t = trainer("texture_traj").load(["--config", str(CONFIGS / "texture_student.yaml")])
    assert s["schedule"]["grid_indices"] == [0, 3, 6, 8, 12] and t["schedule"]["grid_indices"] == [0, 6, 12]
    assert (s["optim"]["updates"], s["optim"]["grad_accum"], s["optim"]["seed"]) == (500, 4, 20260915)
    assert (t["optim"]["updates"], t["optim"]["grad_accum"], t["optim"]["seed"]) == (3000, 4, 20260912)
    for cfg in (s, t):
        assert cfg["lora"] == {"rank": 256, "alpha": 256.0, "dropout": 0.0}
        o = cfg["optim"]
        assert (o["lr"], o["betas"], o["weight_decay"], o["grad_clip"], o["ema_decay"]) == (1e-4, [0.9, 0.999], 0.0, 10.0, 0.99)


@pytest.mark.parametrize("config,student,seed,submodel,checkpointing", [
    ("coarse_shape_student.yaml", "coarse_shape", 20260716, "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16", False),
    ("fine_shape_student.yaml", "fine_shape", 20260723, "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16", True),
])
def test_shape_configs_spell_out_the_released_runs(config, student, seed, submodel, checkpointing, tmp_path):
    module = trainer("shape_consistency")
    cfg = module.load(["--config", str(CONFIGS / config)])
    defaults = C._strip_markers(module.DEFAULTS)
    assert cfg["model"]["student"] == student and cfg["model"]["base_submodel"] == submodel
    assert cfg["model"]["gradient_checkpointing"] is checkpointing
    assert cfg["lora"] == {"rank": 256, "alpha": 256.0, "dropout": 0.0}
    assert cfg["consistency"] == defaults["consistency"]
    assert cfg["consistency"] == {"num_timesteps": 50, "loss_weight": 80.0, "anchor_steps": 12, "anchor_rescale_t": 3.0,
                                  "guidance_strength": 7.5, "guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0]}
    o = cfg["optim"]
    assert (o["updates"], o["lr"], o["betas"], o["weight_decay"], o["grad_clip"], o["ema_decay"], o["seed"]) == \
        (3000, 2e-6, [0.0, 0.999], 1e-3, 10.0, 0.99, seed)
    assert cfg["output"]["checkpoint_every"] == 250 and cfg["runtime"]["tf32_override"] is False
    assert Path(cfg["data"]["objects"]).is_file()
    # a configuration that names only the student gets the released student's submodel, checkpointing and seed
    partial = tmp_path / "partial.yaml"
    partial.write_text(f"model:\n  student: {student}\ndata:\n  conditioning: cond_dir\n")
    filled = module.load(["--config", str(partial)])
    assert (filled["model"]["base_submodel"], filled["model"]["gradient_checkpointing"], filled["optim"]["seed"]) == \
        (submodel, checkpointing, seed)
    with pytest.raises(C.ConfigError):
        module.load(["--config", str(CONFIGS / config), "--set", "model.student=texture"])
    with pytest.raises(C.ConfigError):
        module.load(["--config", str(CONFIGS / config), "--set", "consistency.num_timesteps=1"])


def test_overrides_and_errors(tmp_path):
    module = trainer("structure_traj")
    base = ["--config", str(CONFIGS / "structure_student.yaml")]
    cfg = module.load(base + ["--set", "optim.updates=7", "--set", "data.max_views=3", "--set", "output.dir=rel/out"])
    assert cfg["optim"]["updates"] == 7 and cfg["data"]["max_views"] == 3
    assert Path(cfg["output"]["dir"]).is_absolute()
    with pytest.raises(C.ConfigError):
        module.load(base + ["--set", "optim.update=7"])          # misspelt key
    with pytest.raises(C.ConfigError):
        module.load(base + ["--set", "schedule.grid_indices=[0, 5, 12]"])  # no stored anchor at index 5
    bad = tmp_path / "bad.yaml"
    bad.write_text("optim:\n  lr: 1.0e-4\n  momentum: 0.9\n")
    with pytest.raises(C.ConfigError):
        module.load(["--config", str(bad)])
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n")
    with pytest.raises(C.ConfigError):
        module.load(["--config", str(empty)])                  # data paths are required
    partial = tmp_path / "partial.yaml"
    partial.write_text("data:\n  anchors: anchors_dir\n  conditioning: cond_dir\n")
    cfg = module.load(["--config", str(partial)])
    assert cfg["data"]["anchors"] == str((tmp_path / "anchors_dir").resolve())
    assert cfg["optim"]["updates"] == 500  # untouched keys keep the defaults


def test_object_lists():
    lists = REPO / "data" / "lists"
    rows = {name: list(csv.DictReader((lists / f"{name}.csv").open())) for name in ("curated_10k", "matched_5243", "train_5k")}
    assert {k: len(v) for k, v in rows.items()} == {"curated_10k": 10000, "matched_5243": 5243, "train_5k": 5243}
    for name, rs in rows.items():
        assert list(rs[0]) == ["id", "sha256", "source", "licence"]
        assert len({r["sha256"] for r in rs}) == len(rs)
    assert {r["sha256"] for r in rows["matched_5243"]} <= {r["sha256"] for r in rows["curated_10k"]}
    assert all(r["id"] == r["sha256"][:12] for r in rows["curated_10k"])
    assert len(views_from_list(lists / "train_5k.csv", "id", [0, 1])) == 10486
    views = views_from_list(lists / "curated_10k.csv", "sha256", [0, 1])
    assert len(views) == 20000 and split_view_name(views[1]) == (rows["curated_10k"][0]["sha256"], 1)
    assert view_name("abc", 3) == "abc.view03"
