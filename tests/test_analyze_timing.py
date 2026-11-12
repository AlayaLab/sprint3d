"""scripts/analyze_timing.py on synthetic runs (CPU only)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def at():
    spec = importlib.util.spec_from_file_location("analyze_timing", REPO / "scripts" / "analyze_timing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(directory: Path, e2e, method="fast"):
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({"method": method, "load_s": 1.0}))
    with open(directory / "timing.jsonl", "w") as f:
        for i, t in enumerate(e2e, start=1):
            timings = {"e2e_s": t, "preprocess_s": 0.1, "to_glb_s": t / 2, "glb_bake_finalize_s": 0.2, "export_s": 0.1}
            f.write(json.dumps({"index": i, "id": f"x{i}", "status": "ok", "e2e_s": t, "timings": timings,
                                "stats": {"tex_stream_s": t / 3, "mesh_worker_s": t / 4}}) + "\n")


def test_percentile_and_bootstrap(at):
    assert at.pct([1, 2, 3, 4], 0.5) == 2.5
    ci1 = at.boot_ci([1.0, 2.0, 3.0, 4.0, 5.0], lambda s: sum(s) / len(s), n=2000)
    ci2 = at.boot_ci([1.0, 2.0, 3.0, 4.0, 5.0], lambda s: sum(s) / len(s), n=2000)
    assert ci1 == ci2 and 1.0 <= ci1[0] <= 3.0 <= ci1[1] <= 5.0


def test_summary_excludes_warmup_and_pairs(at, tmp_path):
    write_run(tmp_path / "ref", [100.0, 9.0, 9.0, 4.0, 4.4, 3.6, 4.0, 4.2], method="official")
    write_run(tmp_path / "fast", [50.0, 2.0, 2.0, 1.0, 1.1, 0.9, 1.0, 1.05])
    out = tmp_path / "summary.json"
    assert at.main([str(tmp_path / "ref"), str(tmp_path / "fast"), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    fast = report["runs"]["fast"]
    assert fast["excluded_warmup_ids"] == ["x1", "x2", "x3"]
    assert fast["e2e_s"]["n"] == 5 and fast["e2e_s"]["median"] == pytest.approx(1.0)
    assert fast["e2e_s"]["ci95_median"][0] <= 1.0 <= fast["e2e_s"]["ci95_median"][1]
    assert "critical_path" in fast and fast["critical_path"]["export"]["median"] == pytest.approx(0.1)
    pair = report["paired"]["ref / fast"]
    assert pair["n"] == 5 and pair["ratio_of_medians"] == pytest.approx(4.0)
