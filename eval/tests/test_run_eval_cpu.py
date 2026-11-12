"""End to end on the synthetic dataset with the CPU rasteriser: an official-condition product built to reproduce the
ground-truth image through its own inference-time projection and the official crop, and a true_camera product."""
import json
import shutil

import pytest

import make_manifest
import run_eval


@pytest.fixture()
def cpu_raster(monkeypatch):
    import metrics.render
    from cpu_raster import CPURaster
    monkeypatch.setattr(metrics.render, "Raster", CPURaster)


def _config(synthetic, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"dataset:\n  root: {synthetic['gt_root']}\n  list: {synthetic['list']}\n  source_dirs: {{src: SRC}}\n"
                   "evaluation:\n  device: cpu\n  skip: [lpips, perception, vs_reference]\n  reference_arm: ref\n  n_boot: 200\n")
    return cfg


def test_end_to_end(synthetic, tmp_path, cpu_raster, capsys):
    sha = "ab" * 32
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps([
        {"object_id": sha[:12], "view": 0, "product_glb": str(synthetic["official_glb"]), "arm": "ref", **synthetic["camera"]},
        {"object_id": sha[:12], "view": 0, "product_glb": str(synthetic["official_glb"]), "arm": "ours", "e2e_s": 4.0,
         **synthetic["camera"]},
        {"object_id": sha, "view": 0, "product_glb": str(synthetic["true_camera_glb"]), "arm": "ours", "condition": "true_camera"}]))
    out = tmp_path / "out"
    run_eval.main(["--config", str(_config(synthetic, tmp_path)), "--manifest", str(man), "--out", str(out)])
    recs = {r["arm"]: r for r in json.loads((out / "per_view.json").read_text())}
    assert set(recs) == {"ref@official", "ours@official", "ours@true_camera"}
    for arm, r in recs.items():
        assert r["errors"] == {}, (arm, r["errors"])
        assert r["mask_iou"] > 0.99 and r["ang_mean"] < 1.0, arm          # reproduces the ground-truth image
        assert r["psnr_ov"] > 35 and r["chamfer_nb"] < 0.01 and r["f_0.01_nb"] > 0.9
        assert r["lpips_ov"] is None                                       # skipped
    off = recs["ours@official"]
    assert off["placement"]["kind"] == "own_projection" and off["placement"]["ndc"]["sx"] == pytest.approx(68 / 64)
    assert abs(off["fov_error_deg"]) < 1e-6
    tc = recs["ours@true_camera"]
    assert tc["support_source"] == "mesh_proxy" and tc["support_iou"] > 0.8 and tc["chamfer_vox"] < 0.01
    corpus = json.loads((out / "corpus.json").read_text())
    assert corpus["reference"]["official"] == "ref@official"
    assert "ours@official" in corpus["paired"] and corpus["speed"]["ours@official"]["mean"] == 4.0
    assert "ours@official" in corpus["condition_gap"]
    per_object = json.loads((out / "per_object.json").read_text())
    assert per_object["ours@official"][sha]["mask_iou"] == off["mask_iou"]
    t1 = (out / "tables/table1.md").read_text()
    assert "ref@official" in t1 and "**overlap**" in t1
    # resume: nothing is recomputed; tables-only with the other region
    run_eval.main(["--config", str(_config(synthetic, tmp_path)), "--manifest", str(man), "--out", str(tmp_path / "out2")])
    shutil.rmtree(tmp_path / "out2")
    capsys.readouterr()
    (out / "corpus.json").unlink()
    run_eval.main(["--config", str(_config(synthetic, tmp_path)), "--manifest", str(man), "--out", str(out)])
    assert "[resume] 3 record(s) reused" in capsys.readouterr().out
    run_eval.main(["--config", str(_config(synthetic, tmp_path)), "--out", str(out), "--tables-only", "--table1-region", "full"])
    assert "**full**" in (out / "tables/table1.md").read_text()


def test_finished_results_of_other_conventions_are_not_overwritten(synthetic, tmp_path, cpu_raster):
    out = tmp_path / "out"
    out.mkdir()
    (out / "corpus.json").write_text(json.dumps({"protocol": {"code_hash": "x", "conventions_hash": "y"}}))
    man = tmp_path / "m.json"
    man.write_text(json.dumps([{"object_id": "ab" * 32, "view": 0, "product_glb": "x.glb", "arm": "ref", **synthetic["camera"]}]))
    with pytest.raises(SystemExit):
        run_eval.main(["--config", str(_config(synthetic, tmp_path)), "--manifest", str(man), "--out", str(out)])


def test_list_checksum_is_enforced(synthetic, tmp_path):
    cfg = _config(synthetic, tmp_path)
    cfg.write_text(cfg.read_text().replace("dataset:\n", "dataset:\n  list_sha256: " + "f" * 64 + "\n"))
    with pytest.raises(SystemExit):
        run_eval.main(["--config", str(cfg), "--manifest", "m.json", "--out", str(tmp_path / "o")])


def test_make_manifest(synthetic, tmp_path):
    d = tmp_path / "glbs"
    d.mkdir()
    sha = "ab" * 32
    shutil.copy(synthetic["official_glb"], d / f"{sha[:12]}_v0.glb")
    (d / f"{sha[:12]}_v0.json").write_text(json.dumps({"camera": synthetic["camera"], "e2e_s": 3.5}))
    out = tmp_path / "m.json"
    make_manifest.main(["--dir", str(d), "--arm", "ours", "--list", str(synthetic["list"]), "--out", str(out)])
    rows = json.loads(out.read_text())
    assert len(rows) == 1 and rows[0]["object_id"] == sha and rows[0]["e2e_s"] == 3.5
    assert rows[0]["camera_angle_x"] == pytest.approx(synthetic["camera"]["camera_angle_x"])
