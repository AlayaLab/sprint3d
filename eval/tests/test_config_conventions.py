"""Configuration, conventions hash, record keys, resume rule, manifests."""
import json
from pathlib import Path

import pytest

from metrics import config, conventions
from metrics.dataset import GTDataset
from metrics.manifest import attach_gt, rows_from_manifest

SUITE = Path(__file__).resolve().parents[1]


def test_example_config_loads_and_resolves_paths(tmp_path):
    cfg = config.load_config(SUITE / "config.example.yaml")
    assert cfg["dataset"]["list"] == str((SUITE / "lists/validation_400.csv").resolve())
    assert cfg["evaluation"]["frame_maps"] == str((SUITE / "frame_maps/baseline_frame_maps.json").resolve())
    assert cfg["models"]["uni3d_checkpoint"].endswith("uni3d/modelzoo/uni3d-g/model.pt")
    assert cfg["evaluation"]["table1_region"] == "overlap" and cfg["evaluation"]["background"] == "white"
    assert conventions.sha256_file(cfg["dataset"]["list"]) == cfg["dataset"]["list_sha256"]
    own = tmp_path / "c.yaml"
    own.write_text("dataset:\n  root: gt\n  list: my.csv\n")
    c2 = config.load_config(own)
    assert c2["dataset"]["root"] == str(tmp_path / "gt") and c2["dataset"]["list"] == str(tmp_path / "my.csv")
    assert c2["evaluation"]["frame_maps"] == str((SUITE / "frame_maps/baseline_frame_maps.json").resolve())
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset:\n  rot: x\n")
    with pytest.raises(ValueError):
        config.load_config(bad)


def test_apply_environment(monkeypatch, tmp_path):
    for k in ("HF_HOME", "TORCH_HOME", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    cfg = config.load_config(None)
    cfg["models"].update({"hf_home": str(tmp_path / "hf"), "torch_home": str(tmp_path / "t")})
    env = config.apply_environment(cfg)
    assert env == {"HF_HOME": str(tmp_path / "hf"), "TORCH_HOME": str(tmp_path / "t"), "HF_HUB_OFFLINE": "1"}


def test_conventions_hash_changes_with_conventions_and_code(tmp_path):
    w, b = conventions.effective_conventions("white"), conventions.effective_conventions("black")
    assert w["image_metrics"]["background_rgb"] == [1.0, 1.0, 1.0]
    assert conventions.conventions_hash(w) != conventions.conventions_hash(b)
    assert conventions.conventions_hash(w) == conventions.conventions_hash(conventions.effective_conventions("white"))
    assert w["baselines"]["frame_map_sha256"] == conventions.sha256_file(conventions.FRAME_MAP_PATH)
    (tmp_path / "metrics").mkdir()
    (tmp_path / "run_eval.py").write_text("a = 1\n")
    (tmp_path / "metrics" / "x.py").write_text("x = 1\n")
    h1, files = conventions.code_hash(tmp_path)
    assert sorted(files) == ["metrics/x.py", "run_eval.py"]
    (tmp_path / "metrics" / "x.py").write_text("x = 2\n")
    assert conventions.code_hash(tmp_path)[0] != h1


def test_declared_values():
    im = conventions.CONVENTIONS["image_metrics"]
    assert (im["lpips_main"]["net"], im["lpips_secondary"]["net"], im["background_name"]) == ("alex", "vgg", "white")
    assert conventions.CONVENTIONS["normal_maps"]["resolution"] == 512
    assert conventions.CONVENTIONS["geometry_3d"]["registration"] == "none"


def test_record_key_and_resume(tmp_path):
    import run_eval
    row = {"arm": "a@official", "sha256": "ab", "view": 1, "repeat": 0}
    k = conventions.record_key(row, "1" * 64, "2" * 64)
    assert k == "a@official|ab|1|0|code:1111111111111111|conv:2222222222222222"
    pv = tmp_path / "per_view.json"
    cur = {"key": k, "code_hash": "1" * 64, "conventions_hash": "2" * 64}
    other = dict(cur, key="x", code_hash="9" * 64)
    pv.write_text(json.dumps([cur, other, {"key": "legacy"}]))
    done = run_eval.load_resumable(pv, "1" * 64, "2" * 64)
    assert list(done) == [k]
    stale = list(tmp_path.glob("per_view.stale_*.json"))
    assert len(stale) == 1 and len(json.loads(stale[0].read_text())) == 2


def test_manifest_rows_need_the_inference_camera(synthetic, tmp_path):
    ds = GTDataset(synthetic["gt_root"], list_csv=synthetic["list"], source_dirs={"src": "SRC"})
    sha = "ab" * 32
    assert ds.resolve(sha[:12]) == sha and ds.resolve(sha[:10]) == sha and ds.resolve("zz") is None
    m = tmp_path / "m.json"
    m.write_text(json.dumps([{"object_id": sha[:12], "view": 0, "product_glb": "x.glb", "arm": "ours"}]))
    with pytest.raises(ValueError):
        rows_from_manifest(m, ds)
    m.write_text(json.dumps([{"object_id": sha[:12], "view": 0, "product_glb": "x.glb", "arm": "ours",
                              **synthetic["camera"]}]))
    rows = attach_gt(rows_from_manifest(m, ds), ds)
    assert rows[0]["arm"] == "ours@official" and rows[0]["condition"] == "official" and rows[0]["missing_gt"] == []
    m.write_text(json.dumps([{"object_id": sha, "view": 0, "product_glb": "x.glb", "arm": "ours", "condition": "true_camera"}]))
    rows = attach_gt(rows_from_manifest(m, ds), ds)
    assert rows[0]["arm"] == "ours@true_camera" and rows[0]["camera"] is None and rows[0]["missing_gt"] == []


def test_lists():
    import csv
    rows = list(csv.DictReader(open(SUITE / "lists/validation_400.csv")))
    assert len(rows) == 400 and list(rows[0]) == ["id", "sha256", "source", "licence", "view", "in_earlier_5k_training_set",
                                                  "gt_image_sha256", "gt_mesh_dump_sha256"]
    assert len({r["sha256"] for r in rows}) == 400 and all(r["id"] == r["sha256"][:12] for r in rows)
    assert sum(r["in_earlier_5k_training_set"] == "true" for r in rows) == 7
    assert {r["in_earlier_5k_training_set"] for r in rows} == {"true", "false"}
    hexd = set("0123456789abcdef")
    assert all(len(r[k]) == 64 and set(r[k]) <= hexd for r in rows for k in ("gt_image_sha256", "gt_mesh_dump_sha256"))
    assert len({r["gt_image_sha256"] for r in rows}) == 400
    assert {r["source"] for r in rows} == {"sketchfab", "abo", "hssd", "3dfuture"}
    timing = list(csv.DictReader(open(SUITE / "lists/timing_80.csv")))
    assert len(timing) == 80 and list(timing[0]) == ["id", "view"]
    assert len({t["id"] for t in timing}) == 40 and {t["id"] for t in timing} <= {r["id"] for r in rows}


def test_toys4k_lists():
    import csv
    import hashlib
    rows = list(csv.DictReader(open(SUITE / "lists/toys4k_dev_500.csv")))
    assert len(rows) == 500 and list(rows[0]) == ["id", "sha256", "source", "licence", "view", "object_id", "category",
                                                  "mesh_sha256"]
    assert all(r["id"] == r["sha256"][:12] and r["source"] == "toys4k" and r["view"] == "4" for r in rows)
    assert len({r["sha256"] for r in rows}) == 500 and len({r["category"] for r in rows}) == 105
    ids = "".join(i + "\n" for i in sorted(r["object_id"] for r in rows))
    assert hashlib.sha256(ids.encode()).hexdigest() == "84c3d2d5031a280a1c469ec848119489cd6371eb97247ff44209dd1758907027"
