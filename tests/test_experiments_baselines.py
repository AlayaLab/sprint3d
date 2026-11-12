"""experiments/baselines and experiments/trellis2_mesh_path: arguments, records and summaries (CPU only)."""
import csv
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for sub in ("experiments/baselines", "experiments/trellis2_mesh_path"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

import baseline_io as bio  # noqa: E402
import make_baseline_manifest as mbm  # noqa: E402
import make_input_list as mil  # noqa: E402
import mesh_path as mp  # noqa: E402
import run_hunyuan3d21  # noqa: E402
import run_trellis2  # noqa: E402
import run_triposg  # noqa: E402
import summarize_timing  # noqa: E402
import summarize_trellis2  # noqa: E402
import time_trellis2  # noqa: E402
import trellis2_pipeline as t2p  # noqa: E402

FRAME_MAPS = json.loads((REPO / "eval" / "frame_maps" / "baseline_frame_maps.json").read_text())


@pytest.fixture
def clean_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def write_list(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------------------------------------- inputs and arguments
def test_read_list(tmp_path):
    lst = write_list(tmp_path / "in.jsonl", [{"id": "a_v0", "path": "a.png"}, {"id": "b_v1", "path": "b.png"}])
    rows = bio.read_list(lst)
    assert [r[0] for r in rows] == ["a_v0", "b_v1"] and all(Path(p).is_absolute() for _, p in rows)
    for bad in ([{"id": "a", "path": "x"}, {"id": "a", "path": "y"}], [{"id": "../x", "path": "x"}],
                [{"id": "a"}], []):
        with pytest.raises(ValueError):
            bio.read_list(write_list(tmp_path / "bad.jsonl", bad))


@pytest.mark.parametrize("module, warmup", [(run_trellis2, 1), (run_triposg, 1), (run_hunyuan3d21, 1),
                                            (time_trellis2, 0)])
def test_parsers(module, warmup, tmp_path):
    ap = module.parser()
    a = ap.parse_args(["--image", "x.png", "--out-dir", str(tmp_path)])
    assert a.warmup_requests == warmup and a.exclude_first == 3 and a.seed == 42 and a.tf32 == "on"
    rows, list_path = bio.inputs_from_args(a)
    assert rows[0][0] == "x" and list_path is None
    a = ap.parse_args(["--image", "x.png", "--id", "obj_v2", "--out-dir", str(tmp_path)])
    assert bio.inputs_from_args(a)[0][0][0] == "obj_v2"
    with pytest.raises(SystemExit):
        ap.parse_args(["--image", "x.png", "--list", "l.jsonl", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        ap.parse_args(["--image", "x.png"])
    a = ap.parse_args(["--list", "l.jsonl", "--id", "z", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        bio.check_common_arguments(a)


def test_arms_and_levers():
    assert time_trellis2.parse_arms("stock,mesh_path") == ["stock", "mesh_path"]
    for bad in ("", "stock,stock", "mesh_path,unknown"):
        with pytest.raises(SystemExit):
            time_trellis2.parse_arms(bad)
    with pytest.raises(SystemExit):
        time_trellis2.parse_arms("mesh_path", ["gpu_chart_segmentation"])
    with pytest.raises(SystemExit):
        time_trellis2.parse_arms("mesh_path", ["texture_kv_reuse"])
    with pytest.raises(SystemExit):
        time_trellis2.parse_arms("stock", ["fast_gpu_simplify"])
    full = mp.arm_levers("mesh_path")
    assert "mesh_prep_worker_thread" in full and "adaptive_remesh" not in full
    assert mp.arm_levers("mesh_path_adaptive_remesh") == full | {"adaptive_remesh"}
    assert mp.arm_levers("mesh_path_serial") == full - {"mesh_prep_worker_thread"}
    assert mp.arm_levers("mesh_path", ["fast_gpu_simplify"]) == full - {"fast_gpu_simplify"}
    from fastpixal3d.levers import LEVERS_BY_NAME

    assert set(mp.MESH_LEVERS) <= set(LEVERS_BY_NAME)


def test_configure_process(clean_env, tmp_path):
    env = bio.configure_process(threads=3, tf32=False, offline=True, autotune_cache=str(tmp_path / "c.json"))
    assert env["OMP_NUM_THREADS"] == "3" and env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] is None
    assert env["HF_HUB_OFFLINE"] == "1" and env["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] == str(tmp_path / "c.json")
    assert bio.configure_process(tf32=True)["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "1"


# ------------------------------------------------------------------------------------------- timers
def test_stages_nesting_and_repeats():
    s = bio.Stages(sync=lambda: None)
    with s.stage("a"):
        with s.stage("inner"):
            pass
    with s.stage("a"):
        pass
    s.add("b", 2.0)
    s.add("c", 5.0, nested=True)
    t = s.timings()
    assert s.calls["a_s"] == 2 and "inner_s" in t
    assert t["stage_sum_s"] == pytest.approx(t["a_s"] + 2.0, abs=1e-6)


def test_timed_calls_restores_and_times_outermost_only():
    module = types.ModuleType("m")
    module.f = lambda x: x + 1

    class Obj:
        def g(self, x):
            return module.f(x) * 2

    obj, models = Obj(), {"net": lambda x: -x}
    s = bio.Stages(sync=lambda: None)
    originals = (module.f, models["net"])
    with bio.timed_calls(s, [(module, "f", "f"), (obj, "g", "g"), (models, "net", "net")]):
        assert obj.g(1) == 4 and models["net"](3) == -3 and module.f(0) == 1
    assert (module.f, models["net"]) == originals and "g" not in vars(obj)
    assert s.calls == {"g_s": 1, "net_s": 1, "f_s": 1}      # f inside g is not timed separately


def test_timed_pipeline_class():
    class Base:
        def __init__(self, models=None):
            self.models = models

        @classmethod
        def from_pretrained(cls, path):
            return cls({"path": path})

        def get_cond(self, x):
            return x * 2

        def run(self, x):
            return self.get_cond(x) + self.get_cond(x)

    Timed = t2p.timed_pipeline_class(Base)
    pipe = Timed.from_pretrained("w")
    assert isinstance(pipe, Timed) and pipe.models == {"path": "w"}
    assert pipe.run(1) == 4 and pipe.stages is None
    pipe.stages = bio.Stages(sync=lambda: None)
    assert pipe.run(2) == 8 and pipe.stages.calls == {"get_cond_s": 2}
    assert Timed.get_cond is not Base.get_cond and Timed.__mro__[1] is Base


def test_flow_and_part_times():
    t = {"stage_sum_s": 100.0, "preprocess_image_s": 1.0, "get_cond_s": 2.0, "sample_sparse_structure_s": 3.0,
         "sample_shape_slat_cascade_s": 10.0, "sample_tex_slat_s": 4.0, "decode_latent_s": 5.0, "export_s": 75.0}
    d = t2p.Trellis2Stock(source="x").derived(t)
    assert d["flows_s"] == 19.0 and d["post_processing_s"] == 80.0
    assert (d["part_sampling_s"], d["part_decoding_s"], d["part_meshwork_s"], d["part_other_s"]) == (19.0, 5.0, 75.0, 1.0)
    h = run_hunyuan3d21.Hunyuan3D21.derived(None, {"stage_sum_s": 60.0, "shape_s": 15.0, "shape_preprocess_s": 0.1,
                                                    "shape_condition_s": 0.5, "shape_decode_s": 4.0,
                                                    "shape_write_s": 0.4, "paint_multiview_diffusion_s": 6.0,
                                                    "paint_remesh_s": 3.0, "export_s": 0.5})
    assert h["part_sampling_s"] == pytest.approx(0.5 + 10.4 + 6.0)
    assert h["part_decoding_s"] == pytest.approx(4.4) and h["part_meshwork_s"] == pytest.approx(3.5)


# ------------------------------------------------------------------------------------------- records
@pytest.mark.parametrize("method", ["trellis2", "triposg", "hunyuan3d21"])
def test_camera_record_carries_frame_map(method):
    rec = bio.camera_record("abc_v1", "/in/abc.png", method, "stock", {"e2e_s": 2.5, "stage_sum_s": 2.4})
    entry = FRAME_MAPS["methods"][method]
    assert rec["camera"] is None and rec["object_id"] == "abc_v1" and rec["e2e_s"] == 2.5
    assert all(rec["frame_map"][k] == v for k, v in entry.items())
    raw = (REPO / "eval" / "frame_maps" / "baseline_frame_maps.json").read_bytes()
    assert rec["frame_map"]["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert rec["geometry_glb"] == entry["geometry_file"] and rec["textured_glb"] == entry["textured_file"]
    assert "model.glb" in bio.expected_files(method)


class FakeMethod(bio.Method):
    name = "triposg"

    def __init__(self, arm, fail_on=()):
        self.arm, self.fail_on, self.calls = arm, set(fail_on), []

    def load(self):
        return {"fake": True}

    def run(self, image_path, out_dir, seed, stages):
        self.calls.append(out_dir.name)
        if out_dir.name in self.fail_on:
            raise RuntimeError("boom")
        with stages.stage("generate"):
            (out_dir / "model.glb").write_bytes(b"glTF-fake")
        return {"output_faces": 12}

    def derived(self, timings):
        return bio.parts(timings, sampling=timings["generate_s"], decoding=0.0, meshwork=0.0)


def test_driver_writes_records_and_resumes(tmp_path, clean_env):
    images = [tmp_path / f"{i}.png" for i in "abc"]
    lst = write_list(tmp_path / "in.jsonl", [{"id": f"o{i}_v0", "path": str(p)} for i, p in enumerate(images)])
    out = tmp_path / "out"
    first, second = FakeMethod("one", fail_on={"o2_v0"}), FakeMethod("two")
    argv = ["--list", str(lst), "--out-dir", str(out), "--sampler-interval", "0", "--exclude-first", "1"]
    assert bio.main_for(lambda a: [first, second], run_trellis2.parser(), argv) == 1      # one failure recorded
    assert first.calls[0] == "o0_v0" and len(first.calls) == 4                              # warm-up + 3 inputs
    rows = [json.loads(x) for x in (out / "one" / "timing.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["ok", "ok", "error"] and [r["warmup"] for r in rows] == [True, False, False]
    assert [r["arm_position"] for r in rows] == [0, 1, 0]                                   # rotated order
    ok = rows[0]
    assert ok["timings"]["stage_sum_s"] == ok["timings"]["generate_s"] and "part_sampling_s" in ok["timings"]
    cam = json.loads((out / "one" / "o0_v0" / "camera.json").read_text())
    assert cam["method"] == "triposg" and cam["arm"] == "one" and cam["camera"] is None
    run = json.loads((out / "two" / "run.json").read_text())
    assert run["n_inputs"] == 3 and run["warmup_request_e2e_s"] and run["failed"] == 0 and run["rotate"]
    with pytest.raises(SystemExit):
        bio.main_for(lambda a: [FakeMethod("one"), FakeMethod("two")], run_trellis2.parser(), argv)
    again = FakeMethod("one")
    assert bio.main_for(lambda a: [again, FakeMethod("two")], run_trellis2.parser(),
                        argv + ["--resume", "--warmup-requests", "0"]) == 0
    assert again.calls == ["o2_v0"]
    assert (out / "one" / "run.json").is_file() and (out / "one" / "run.resume1.json").is_file()
    assert len((out / "one" / "timing.jsonl").read_text().splitlines()) == 4


def test_missing_output_is_an_error(tmp_path, clean_env):
    class NoOutput(FakeMethod):
        def run(self, image_path, out_dir, seed, stages):
            return {}

    lst = write_list(tmp_path / "in.jsonl", [{"id": "o_v0", "path": "x.png"}])
    argv = ["--list", str(lst), "--out-dir", str(tmp_path / "o"), "--sampler-interval", "0", "--warmup-requests", "0"]
    assert bio.main_for(lambda a: [NoOutput("stock")], run_triposg.parser(), argv) == 1
    row = json.loads((tmp_path / "o" / "timing.jsonl").read_text())
    assert row["status"] == "error" and "missing" in row["error"]


# ------------------------------------------------------------------------------------------- lists and manifests
def write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_make_input_list(tmp_path):
    sha = "ab" * 32
    lookup = write_csv(tmp_path / "val.csv", ["id", "sha256", "source", "view"], [["abababababab", sha, "abo", "3"]])
    timing = write_csv(tmp_path / "t.csv", ["id", "view"], [["abababababab", "0"], ["abababababab", "3"]])
    render = tmp_path / "gt" / "ABO" / "renders_cond" / sha
    render.mkdir(parents=True)
    (render / "000.png").write_bytes(b"x")
    with pytest.raises(SystemExit):
        mil.build_rows(timing, tmp_path / "gt", {"abo": "ABO"}, lookup_csv=lookup)
    (render / "003.png").write_bytes(b"x")
    rows = mil.build_rows(timing, tmp_path / "gt", {"abo": "ABO"}, lookup_csv=lookup)
    assert [r["id"] for r in rows] == ["abababababab_v0", "abababababab_v3"] and rows[1]["path"].endswith("003.png")


def test_baseline_manifest(tmp_path):
    sha = "cd" * 32
    lst = write_csv(tmp_path / "list.csv", ["id", "sha256", "view"], [["cdcdcdcdcdcd", sha, "1"], ["eeeeeeeeeeee", "e" * 64, "0"]])
    item = tmp_path / "run" / "cdcdcdcdcdcd_v1"
    item.mkdir(parents=True)
    for name in ("model.glb", "shape_untextured.glb"):
        (item / name).write_bytes(b"glb")
    rec = bio.camera_record("cdcdcdcdcdcd_v1", "/in/x.png", "hunyuan3d21", "stock", {"e2e_s": 3.0, "stage_sum_s": 2.9})
    (item / "camera.json").write_text(json.dumps(rec))
    rows, missing = mbm.build(tmp_path / "run", lst, "stage_sum_s")
    assert missing == ["eeeeeeeeeeee_v0"] and len(rows) == 1
    r = rows[0]
    assert r["method"] == "hunyuan3d21" and r["sha256"] == sha and r["view"] == 1 and r["e2e_s"] == 2.9
    assert r["geometry_glb"].endswith("shape_untextured.glb") and r["textured_glb"].endswith("model.glb")
    with pytest.raises(SystemExit):
        mbm.build(tmp_path / "run", lst, "stage_sum_s", method="trellis2")
    rec["frame_map"]["family"] = "view_aligned"
    (item / "camera.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        mbm.build(tmp_path / "run", lst, "stage_sum_s")


# ------------------------------------------------------------------------------------------- summaries
def write_timing(directory: Path, values):
    directory.mkdir(parents=True)
    with open(directory / "timing.jsonl", "w") as f:
        for i, (ident, t) in enumerate(values.items(), start=1):
            timings = {"stage_sum_s": t["total"], "e2e_s": t["total"] + 0.01, "preprocess_image_s": 0.1,
                       "flows_s": t["flows"], "sample_tex_slat_s": t["tex"],
                       "post_processing_s": t["total"] - 0.1 - t["flows"],
                       "part_sampling_s": t["flows"], "part_decoding_s": 1.0,
                       "part_meshwork_s": t["total"] - t["flows"] - 1.1, "part_other_s": 0.1}
            f.write(json.dumps({"index": i, "id": ident, "status": "ok", "e2e_s": t["total"] + 0.01,
                                "timings": timings, "stats": {"output_faces": 1000}}) + "\n")


def test_summaries(tmp_path):
    ids = [f"o{k}_v{v}" for k in range(4) for v in (0, 3)]
    stock = {i: {"total": 100.0 + n, "flows": 15.0, "tex": 4.0} for n, i in enumerate(ids)}
    fast = {i: {"total": 16.0 + n / 10, "flows": 15.0, "tex": 4.4} for n, i in enumerate(ids)}
    serial = {i: {"total": 16.5, "flows": 15.0, "tex": 4.0} for i in ids}
    for name, data in (("stock", stock), ("fast", fast), ("serial", serial)):
        write_timing(tmp_path / name, data)
    out = tmp_path / "t2.json"
    assert summarize_trellis2.main(["--run", f"stock={tmp_path / 'stock'}", "--run", f"fast={tmp_path / 'fast'}",
                                    "--serial", str(tmp_path / "serial"), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    p = rep["paired"]["fast"]
    assert p["n"] == 8 and p["n_objects"] == 4 and p["flows_ratio_median"] == 1.0
    assert p["speedup_ratio_of_medians"] == pytest.approx(103.5 / 16.35)
    lo, hi = p["speedup_ratio_of_medians_ci95"]
    assert lo <= p["speedup_ratio_of_medians"] <= hi
    assert p["conservative_speedup"] < p["speedup_ratio_of_medians"]       # uncontended texture flow charged
    assert rep["runs"]["stock"]["post_processing_median"] == pytest.approx(103.5 - 15.1)
    out2 = tmp_path / "b.json"
    assert summarize_timing.main(["--run", f"A={tmp_path / 'stock'}", "--run", f"B={tmp_path / 'fast'}",
                                  "--exclude-first", "3", "--out", str(out2)]) == 0
    rows = json.loads(out2.read_text())["rows"]
    assert rows["A"]["n"] == 5 and rows["A"]["sampling_median_s"] == 15.0
    assert rows["A"]["sampling_share_mean"] + rows["A"]["post_processing_share_mean"] + \
        rows["A"]["other_share_mean"] == pytest.approx(1.0)


def test_quality_deltas(tmp_path):
    pytest.importorskip("scipy")
    for name, shift in (("ref", 0.0), ("run", 0.5)):
        d = tmp_path / name
        d.mkdir()
        objs = {f"{k:064x}": {"psnr_full": 18.0 + k + shift, "ang_mean": 30.0 - shift} for k in range(6)}
        (d / "per_object.json").write_text(json.dumps({"TRELLIS.2": objs}))
    q = summarize_trellis2.quality(tmp_path / "ref", tmp_path / "run", ["psnr_full", "ang_mean"], None)
    assert q["psnr_full"]["bootstrap"]["mean_diff"] == pytest.approx(0.5)
    assert q["ang_mean"]["bootstrap"]["mean_diff"] == pytest.approx(-0.5)
    assert q["psnr_full"]["sign_test"]["n_pos"] == 6


# ------------------------------------------------------------------------------------------- GLB frame
def test_export_glb_without_rotation(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from PIL import Image

    from fastpixal3d.export import glb

    def textured():
        box = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
        rng = np.random.default_rng(0)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(rng.integers(0, 255, (8, 8, 4), dtype=np.uint8)),
            metallicRoughnessTexture=Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)))
        return trimesh.Trimesh(vertices=np.asarray(box.vertices), faces=np.asarray(box.faces), process=False,
                               visual=trimesh.visual.TextureVisuals(uv=rng.random((len(box.vertices), 2)),
                                                                    material=material))

    kept, rotated = textured(), textured()
    original = np.asarray(kept.vertices).copy()
    glb.export_glb(kept, tmp_path / "kept.glb", rotation=None)
    glb.export_glb(rotated, tmp_path / "rotated.glb")
    load = lambda p: np.asarray(trimesh.load(p, force="mesh", process=False).vertices)  # noqa: E731
    assert np.allclose(load(tmp_path / "kept.glb"), original, atol=1e-6)
    assert np.allclose(load(tmp_path / "rotated.glb"),
                       original @ glb.PRODUCT_EXPORT_ROTATION[:3, :3].T, atol=1e-6)
