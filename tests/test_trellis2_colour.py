"""experiments/trellis2_mesh_path/colour_metrics.py and its summary on a tiny textured mesh (CPU rasteriser, no LPIPS)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

REPO = Path(__file__).resolve().parents[1]
for sub in ("eval", "eval/tests", "experiments/trellis2_mesh_path"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

import colour_metrics as cm  # noqa: E402
import summarize_trellis2 as st  # noqa: E402

FRAME = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # trellis2 frame map: world = glb @ FRAME


@pytest.fixture
def clean_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def write_product(directory: Path, ident: str, V_world, F, rgb):
    from PIL import Image

    v_glb = np.asarray(V_world) @ FRAME.T
    uv = (v_glb[:, [0, 2]] - v_glb[:, [0, 2]].min(0)) / np.ptp(v_glb[:, [0, 2]], axis=0)
    image = Image.fromarray(np.tile(np.array(rgb, np.uint8), (8, 8, 1)))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0.0, roughnessFactor=1.0)
    mesh = trimesh.Trimesh(v_glb, F, process=False, visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    (directory / ident).mkdir(parents=True)
    mesh.export(directory / ident / "model.glb")


def test_colour_metrics_and_summary(tmp_path, clean_env):
    from synthetic import SHA, build_dataset

    data = build_dataset(tmp_path / "ds")
    ident = f"{SHA[:12]}_v0"
    for name, rgb in (("ref", (200, 60, 60)), ("same", (200, 60, 60)), ("green", (120, 150, 90))):
        write_product(tmp_path / name, ident, data["V"], data["F"], rgb)
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"dataset": {"root": str(data["gt_root"]), "list": str(data["list"]),
                                              "source_dirs": {"src": "SRC"}}}))
    lst = tmp_path / "inputs.jsonl"
    lst.write_text(json.dumps({"id": ident, "path": "unused.png", "sha256": SHA, "view": 0}) + "\n")
    out = tmp_path / "colour.json"
    argv = ["--config", str(config), "--list", str(lst), "--reference", f"ref={tmp_path / 'ref'}",
            "--run", f"same={tmp_path / 'same'}", "--run", f"green={tmp_path / 'green'}", "--pair", "same=green",
            "--out", str(out), "--cpu", "--no-lpips"]
    assert cm.main(argv) == 0
    rep = json.loads(out.read_text())
    e = rep["per_input"][ident]
    assert rep["reference"] == "ref" and not e["errors"]
    extent = float(np.ptp(data["V"], axis=0).max())
    assert e["placement"]["bbox_max_extent"] == pytest.approx(extent, rel=1e-6)   # GLB stores float32
    assert set(e["pairs"]) == {"same__vs__ref", "green__vs__ref", "same__vs__green"}
    same = e["pairs"]["same__vs__ref"]
    assert set(same["per_view"]) == {"in", "y90", "y180", "y270", "under"}
    assert same["mean"]["tex_psnr"] == pytest.approx(120.0) and same["mean"]["mask_iou"] == 1.0
    green = e["pairs"]["green__vs__ref"]["mean"]
    assert green["mask_iou"] == 1.0 and green["tex_psnr"] < 20.0 and green["tex_ssim"] < same["mean"]["tex_ssim"]
    assert e["vs_input"]["same"] == e["vs_input"]["ref"] and e["vs_input"]["green"] != e["vs_input"]["ref"]
    assert e["products"]["ref"]["textured"] and e["products"]["ref"]["faces"] == len(data["F"])
    s = st.colour(out)
    assert s["reference"] == "ref"
    assert s["delta_vs_input"]["same"]["tex_psnr"]["bootstrap"]["mean_diff"] == 0.0
    assert s["agreement"]["green__vs__ref"]["tex_psnr"]["mean"] == pytest.approx(green["tex_psnr"])
    assert s["agreement"]["same__vs__ref"]["n_objects"] == 1


def test_colour_summary_aggregation(tmp_path):
    def entry(sha, ref, run, pair):
        return {"sha256": sha, "vs_input": {"official": {"tex_psnr": ref}, "arm": {"tex_psnr": run}},
                "pairs": {"arm__vs__official": {"mean": {"tex_psnr": pair}}}, "errors": {}}

    per = {"a_v0": entry("a", 20.0, 20.5, 40.0), "a_v1": entry("a", 22.0, 22.5, 42.0),
           "b_v0": entry("b", 18.0, 18.0, 44.0)}
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"per_input": per}))           # no "reference" key: defaults to "official"
    s = st.colour(path)
    d = s["delta_vs_input"]["arm"]["tex_psnr"]
    assert d["bootstrap"]["n"] == 2 and d["bootstrap"]["mean_diff"] == pytest.approx(0.25)   # (0.5 + 0.0) / 2
    a = s["agreement"]["arm__vs__official"]["tex_psnr"]
    assert a["mean"] == pytest.approx((41.0 + 44.0) / 2) and a["ci95"][0] <= a["mean"] <= a["ci95"][1]
    assert st.mean_ci([1.0, 2.0, 3.0]) == st.mean_ci([1.0, 2.0, 3.0])
    with pytest.raises(SystemExit):
        cm.main(["--list", str(path), "--reference", "x", "--out", str(tmp_path / "o.json")])
