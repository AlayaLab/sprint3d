"""Experiment analyses (experiments/analysis) on synthetic timing runs."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "experiments" / "analysis"


def _module(name):
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    spec = importlib.util.spec_from_file_location(f"exp_{name}", ANALYSIS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _module("analysis_common")


def _fast_row(index, oid, scale, tex=5.0, worker=30.0, schedule="worker_thread"):
    t = {"preprocess_s": 0.1, "camera_s": 0.2, "ss_condition_s": 0.05, "ss_initial_noise_s": 0.0,
         "ss_rollout_s": 0.4, "ss_decode_support_s": 0.02, "lr_condition_s": 0.05, "lr_initial_noise_s": 0.0,
         "lr_rollout_s": 0.1, "hr_upsample_support_s": 0.1, "hr_condition_s": 0.1, "hr_initial_noise_s": 0.0,
         "hr_rollout_s": 0.8 * scale, "shape_decode_s": 0.5, "shape_fill_holes_s": 0.05,
         "tex_condition_s": 0.3, "tex_rollout_s": tex - 0.5, "tex_decode_s": 0.2,
         "glb_finish_s": 0.3, "glb_bake_finalize_s": 2.0, "export_s": 1.0}
    stats = {"tex_stream_s": tex, "mesh_worker_s": worker, "mesh_schedule": schedule, "output_faces": 990_000,
             "mesh_timings": {"cluster_s": 0.4, "uv": {"total_s": worker * 0.8}}}
    gpu = 1.5 if schedule == "uv_overlap" else 0.0
    if gpu:
        t["glb_gpu_phase_s"] = gpu
    window = max(tex, worker) + 0.01
    stats["mesh_overlap_window_s"] = window
    t["to_glb_s"] = gpu + window + t["glb_finish_s"] + t["glb_bake_finalize_s"] + 0.02
    keys = ("preprocess_s", "camera_s", "ss_condition_s", "ss_initial_noise_s", "ss_rollout_s", "ss_decode_support_s",
            "lr_condition_s", "lr_initial_noise_s", "lr_rollout_s", "hr_upsample_support_s", "hr_condition_s",
            "hr_initial_noise_s", "hr_rollout_s", "shape_decode_s", "shape_fill_holes_s", "to_glb_s", "export_s")
    accounted = sum(t[k] for k in keys)
    t["e2e_s"] = accounted + 0.03
    t["timing_accounted_s"], t["residual_s"] = accounted, 0.03
    return {"index": index, "id": oid, "status": "ok", "e2e_s": t["e2e_s"], "timings": t, "stats": stats}


def _official_row(index, oid):
    t = {"preprocess_image_s": 0.3, "camera_s": 0.4, "cond_ss_s": 0.1, "sparse_structure_s": 1.0,
         "cond_shape_s": 0.4, "shape_coarse_s": 1.5, "shape_sampler_calls_s": 11.5, "texture_s": 5.0,
         "decode_s": 3.0, "to_glb_s": 100.0, "export_s": 6.0}
    t["e2e_s"] = sum(t.values()) + 0.5
    return {"index": index, "id": oid, "status": "ok", "e2e_s": t["e2e_s"], "timings": t, "stats": {}}


@pytest.fixture()
def synthetic(tmp_path):
    """A chain of four configurations x 12 inputs (6 objects x 2 views) -- official, tiled path (slow UV),
    + FA3 (a flow lever hidden by the UV branch), Ours -- in three blocks, plus one control of the FA3 row."""
    cfg_dir, runs = tmp_path / "configs", tmp_path / "runs"
    cfg_dir.mkdir()
    specs = [
        ("00_pixal3d_rerun", "Pixal3D re-run", "name: pixal3d_rerun\nmethod: official\nadds: []\n"),
        ("01_tiled_mesh_path", "+ tiled mesh path (16-tile atlas)",
         "name: tiled_mesh_path\nmethod: fast\nblock: post_processing\nadds: []\nlevers: []\n"),
        ("02_fa3_texture_flow", "+ FlashAttention-3, texture flow",
         "name: fa3_texture_flow\nmethod: fast\nblock: flow_engineering\nadds: [fa3_texture_flow]\n"
         "levers: [fa3_texture_flow]\n"),
        ("03_ours", "Ours", "name: ours\nmethod: fast\nblock: distillation\nadds: [texture_student]\nlevers: {}\n"),
        ("x1_without_fa3", "(control) without FA3",
         "name: without_fa3\nmethod: fast\ncontrol_of: fa3_texture_flow\nadds: []\nremoves: [fa3_texture_flow]\n"
         "levers: []\n"),
    ]
    ids = [f"obj{o}_v{v:02d}" for o in range(6) for v in (0, 3)]
    for stem, paper, body in specs:
        (cfg_dir / f"{stem}.yaml").write_text(f"# Paper row: {paper}\n{body}")
        d = runs / stem
        d.mkdir(parents=True)
        rows = []
        for i, oid in enumerate(ids, start=1):
            if stem == "00_pixal3d_rerun":
                rows.append(_official_row(i, oid))
            elif stem in ("01_tiled_mesh_path", "x1_without_fa3"):
                rows.append(_fast_row(i, oid, 1.0 + 0.01 * i, tex=5.0, worker=30.0 + i, schedule="uv_overlap"))
            elif stem == "02_fa3_texture_flow":
                rows.append(_fast_row(i, oid, 1.0 + 0.01 * i, tex=4.0, worker=30.0 + i, schedule="uv_overlap"))
            else:
                rows.append(_fast_row(i, oid, 1.0 + 0.01 * i, tex=1.0, worker=0.8))
        rows[5]["status"] = "error" if stem == "03_ours" else rows[5]["status"]
        (d / "timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return cfg_dir, runs


def test_attribution_sums_to_e2e(synthetic):
    cfg_dir, runs = synthetic
    for entry in C.load_table([str(cfg_dir)], runs, exclude_first=2):
        for oid, cp in entry["attribution"].items():
            assert sum(cp.values()) == pytest.approx(float(entry["measured"][oid]["e2e_s"]), abs=1e-9)
    table = C.load_table([str(cfg_dir)], runs, exclude_first=2)
    tiled = next(iter(table[1]["attribution"].values()))
    assert tiled["uv"] > 30 and tiled["texture"] == 0  # the CPU UV branch is the longer one
    deployed = next(iter(table[3]["attribution"].values()))
    assert deployed["texture"] == pytest.approx(1.0) and deployed["uv"] == 0
    official = next(iter(table[0]["attribution"].values()))
    assert official["unsplit"] == 100.0 and official["fine_shape"] == pytest.approx(10.0)


def test_chain_blocks_and_controls(synthetic):
    cfg_dir, runs = synthetic
    table = C.load_table([str(cfg_dir)], runs, exclude_first=2)
    chain, controls = C.split_chain(table)
    assert [C.name_of(e["config"]) for e in chain] == ["pixal3d_rerun", "tiled_mesh_path", "fa3_texture_flow", "ours"]
    assert [C.name_of(e["config"]) for e in controls] == ["without_fa3"]
    assert C.block_spans(chain) == [("post_processing", "mesh post-processing", 0, 1),
                                    ("flow_engineering", "flow engineering", 1, 2), ("distillation", "distillation", 2, 3)]
    (pair,) = C.control_pairs(table)
    assert C.name_of(pair["without"]["config"]) == "without_fa3" and C.name_of(pair["with"]["config"]) == "fa3_texture_flow"
    assert pair["levers"] == ["fa3_texture_flow"]
    bad = [dict(e, config=dict(e["config"], block="post_processing")) if k == 3 else e for k, e in enumerate(chain)]
    with pytest.raises(ValueError, match="contiguous"):
        C.block_spans(bad)


def test_waterfall_lever_and_stage_tables(synthetic, tmp_path):
    cfg_dir, runs = synthetic
    out = tmp_path / "out"
    W, L, S = _module("waterfall"), _module("lever_table"), _module("stage_deltas")
    assert W.main([str(cfg_dir), "--runs", str(runs), "--out", str(out), "--exclude-first", "2", "--figure"]) == 0
    doc = json.loads((out / "waterfall.json").read_text())
    data = doc["rows"]
    assert [r["name"] for r in data] == ["pixal3d_rerun", "tiled_mesh_path", "fa3_texture_flow", "ours"]
    assert [r["row"] for r in data] == [0, 1, 2, 3]
    assert [r["block"] for r in data] == ["", "post_processing", "flow_engineering", "distillation"]
    assert [r["n"] for r in data] == [10, 10, 10, 9]
    assert data[3]["n_failed"] == 1
    # FA3 shortens the texture flow by 1 s, but the CPU UV branch is longer: no end-to-end change
    assert data[2]["delta_prev_mean"] == pytest.approx(0.0, abs=1e-9)
    assert data[1]["delta_prev_mean"] < -60
    assert data[1]["e2e_median_ci_lo"] <= data[1]["e2e_median"] <= data[1]["e2e_median_ci_hi"]
    assert (out / "waterfall.png").stat().st_size > 0 and (out / "waterfall.pdf").stat().st_size > 0
    assert (out / "waterfall.csv").read_text().splitlines()[0].startswith("row,name,paper_row,block")
    blocks = doc["blocks"]
    assert [(b["kind"], b["block"], b["from"], b["to"]) for b in blocks] == [
        ("block", "post_processing", "pixal3d_rerun", "tiled_mesh_path"),
        ("block", "flow_engineering", "tiled_mesh_path", "fa3_texture_flow"),
        ("block", "distillation", "fa3_texture_flow", "ours"), ("total", "", "pixal3d_rerun", "ours")]
    assert blocks[1]["factor"] == pytest.approx(1.0) and blocks[0]["factor"] > 2
    product = blocks[0]["factor"] * blocks[1]["factor"] * blocks[2]["factor"]
    assert blocks[3]["factor"] == pytest.approx(product, rel=0.2)  # medians over 9 vs 10 inputs
    assert all(b["factor_ci_lo"] <= b["factor"] <= b["factor_ci_hi"] for b in blocks)
    assert (out / "waterfall_blocks.csv").read_text().splitlines()[0].startswith("kind,block,label,from,to")
    assert doc["controls"][0]["name"] == "without_fa3" and doc["controls"][0]["n"] == 10

    assert L.main([str(cfg_dir), "--runs", str(runs), "--out", str(out), "--exclude-first", "2"]) == 0
    levers = json.loads((out / "levers.json").read_text())["rows"]
    assert [(r["kind"], r["row"]) for r in levers] == [("step", "01"), ("step", "02"), ("step", "03"), ("control", "x1")]
    assert levers[0]["levers"] == "" and levers[0]["report_name"] == "tiled mesh path (16-tile atlas)"
    fa3 = next(r for r in levers if r["kind"] == "step" and r["levers"] == "fa3_texture_flow")
    assert fa3["output_class"] == "near-identical" and fa3["stage_key"] == "tex_rollout_s"
    assert fa3["stage_delta_mean"] == pytest.approx(-1.0) and fa3["saved_s"] == pytest.approx(0.0, abs=1e-9)
    assert fa3["block"] == "flow_engineering"
    control = levers[-1]
    assert (control["from"], control["to"], control["levers"]) == ("without_fa3", "fa3_texture_flow", "fa3_texture_flow")
    assert control["stage_delta_mean"] == pytest.approx(-1.0)
    assert next(r for r in levers if r["name"] == "ours")["output_class"] == "output-changing"

    assert S.main([str(cfg_dir), "--runs", str(runs), "--out", str(out), "--exclude-first", "2"]) == 0
    stages = json.loads((out / "stage_deltas.json").read_text())["rows"]
    kinds = {(r["from"], r["to"], r["kind"]) for r in stages}
    assert ("pixal3d_rerun", "tiled_mesh_path", "adjacent") in kinds and ("fa3_texture_flow", "ours", "block") in kinds
    assert ("pixal3d_rerun", "ours", "total") in kinds and ("without_fa3", "fa3_texture_flow", "control") in kinds
    tex = next(r for r in stages if (r["from"], r["to"], r["stage"]) == ("fa3_texture_flow", "ours", "texture"))
    assert tex["delta_mean"] == pytest.approx(1.0)  # hidden before, on the critical path after
    uv = next(r for r in stages if (r["from"], r["to"], r["stage"]) == ("fa3_texture_flow", "ours", "uv"))
    assert uv["delta_mean"] < -30
    assert S.main([str(cfg_dir), "--runs", str(runs), "--out", str(out / "obj"), "--exclude-first", "2",
                   "--unit", "object"]) == 0
    assert json.loads((out / "obj" / "stage_deltas.json").read_text())["rows"][0]["unit"] == "object"


def test_paired_table_resamples_objects(synthetic, tmp_path):
    cfg_dir, runs = synthetic
    out = tmp_path / "paired"
    P = _module("paired_table")
    # reference = the tiled mesh path, compared with the FA3 row
    paths = [str(cfg_dir / "01_tiled_mesh_path.yaml"), str(cfg_dir / "02_fa3_texture_flow.yaml")]
    assert P.main(paths + ["--runs", str(runs), "--out", str(out), "--exclude-first", "2"]) == 0
    rows = json.loads((out / "paired.json").read_text())["rows"]
    assert rows[1]["texture_sampling_delta_median"] == pytest.approx(-1.0)
    assert rows[1]["n_objects"] == 5 and rows[1]["n"] == 10


def test_boot_ci_groups_and_seed():
    import numpy as np

    values = [1.0, 1.1, 0.9, 5.0, 5.1, 4.9]
    a = C.boot_ci(values, np.mean)
    assert a == C.boot_ci(values, np.mean)
    g = C.boot_ci(values, np.mean, groups=["a", "a", "a", "b", "b", "b"])
    assert g[0] <= 3.0 <= g[1] and g[1] - g[0] > a[1] - a[0]
    assert C.object_of("abc_def_v03") == "abc_def" and C.object_of("plain") == "plain"


def test_experiment_defaults_count_every_input(synthetic, tmp_path):
    """run_arms.py and the analyses count every input (the report's cumulative measurement and one-at-a-time removals);
    the dedicated timing run keeps its convention of leaving out the first 3 inputs."""
    import re

    cfg_dir, runs = synthetic
    W = _module("waterfall")
    assert W.main([str(cfg_dir), "--runs", str(runs), "--out", str(tmp_path / "all")]) == 0
    rows = json.loads((tmp_path / "all" / "waterfall.json").read_text())["rows"]
    assert [r["n"] for r in rows] == [12, 12, 12, 11]
    repo = ANALYSIS.parents[1]
    default = re.compile(r'"--exclude-first", type=int, default=(\d+)')
    for path in [repo / "experiments" / "run_arms.py"] + sorted(ANALYSIS.glob("*.py")):
        found = default.findall(path.read_text())
        assert found in ([], ["0"]), (path.name, found)
    assert default.findall((repo / "scripts" / "time_e2e.py").read_text()) == ["3"]
    assert default.findall((repo / "scripts" / "analyze_timing.py").read_text()) == ["3"]
