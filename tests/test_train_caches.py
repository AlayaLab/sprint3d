"""Cache builders: input-image recipe, fine-support quantisation, work lists and the build driver (CPU)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fastpixal3d.training import cachework as W
from fastpixal3d.training import records as R

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _png(size, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size, size, 4), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue(), array


def test_input_image_recipe():
    png, array = _png(512)
    rgba, size = R.resize_rgba(png, 512)
    assert size == (512, 512) and np.array_equal(np.asarray(rgba), array)  # 512 renders are used as they are
    img = R.composite_on_black(rgba)
    a = array.astype(np.float32) / 255.0
    want = (np.clip(a[:, :, :3] * a[:, :, 3:4], 0, 1) * 255.0).astype(np.uint8)
    assert img.mode == "RGB" and np.array_equal(np.asarray(img), want)
    big, big_array = _png(1024, seed=1)
    rgba, size = R.resize_rgba(big, 512)
    ref = Image.fromarray(big_array, mode="RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    assert size == (1024, 1024) and np.array_equal(np.asarray(rgba), np.asarray(ref))
    assert R.conditioning_image(big, hashlib.sha256(big).hexdigest()).size == (512, 512)
    with pytest.raises(ValueError):
        R.conditioning_image(big, "0" * 64)


def test_find_render(tmp_path):
    key = "ab" * 32
    path = R.render_path(tmp_path / "root", "ObjaverseXL_sketchfab", key, 1)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    render = {"subset": "ObjaverseXL_sketchfab", "sha256": key, "view": 1, "image_path": "/nonexistent/001.png"}
    assert R.find_render(render, [tmp_path / "elsewhere", tmp_path / "root"]) == path
    flat = R.render_path(tmp_path / "flat", "", key, 1)
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"y")
    assert R.find_render(render, [tmp_path / "flat"]) == flat
    with pytest.raises(FileNotFoundError):
        R.find_render(render, [tmp_path / "missing"])


def _builder(name):
    spec = importlib.util.spec_from_file_location(f"cache_{name}", REPO / "train" / "caches" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fine_support_quantisation():
    q = _builder("fine_shape_conditioning").quantize_fine_coords
    g = torch.Generator().manual_seed(0)
    coords = torch.cat([torch.zeros(3000, 1, dtype=torch.int32), torch.randint(0, 512, (3000, 3), generator=g,
                                                                              dtype=torch.int32)], 1)
    unique, resolution, grid = q(coords)
    assert (resolution, grid) == (1024, 64) and unique.dtype == torch.int32
    assert int(unique[:, 1:].max()) <= 63 and unique.shape[0] == torch.unique(unique, dim=0).shape[0]
    want = ((coords[:, 1:] + 0.5) / 512 * 63).round().int()
    assert torch.equal(unique, torch.unique(torch.cat([coords[:, :1], want], 1), dim=0))
    _, resolution, grid = q(coords, max_tokens=100)  # too many tokens: 1024 -> 896 -> 768 (floor)
    assert (resolution, grid) == (768, 48)


def test_work_list_and_driver(tmp_path):
    lst = tmp_path / "list.csv"
    lst.write_text("id,sha256,source,licence\nobj1,aa,sketchfab,by\nobj2,bb,hssd,CC-BY-NC-4.0\n")
    ds = {"root": str(tmp_path), "subset_dirs": {"sketchfab": "ObjaverseXL_sketchfab"}, "list": str(lst),
          "key": "id", "views": [0, 1]}
    rows = W.list_rows(ds)
    assert sorted(rows) == ["obj1.view00", "obj1.view01", "obj2.view00", "obj2.view01"]
    assert rows["obj1.view01"] == {"sha256": "aa", "source": "sketchfab", "view": 1, "subset_dir": "ObjaverseXL_sketchfab"}
    assert rows["obj2.view00"]["subset_dir"] == ""

    out = tmp_path / "cache"
    out.mkdir()
    torch.save({"x": 1}, out / "obj1.view00.pt")  # already built
    args = W.parse_args("t", ["--config", "unused.yaml", "--shard", "0", "--nshard", "1", "--verify-n", "1"])
    assert W.pending(args, sorted(rows), out) == ["obj1.view01", "obj2.view00", "obj2.view01"]
    args2 = W.parse_args("t", ["--config", "u", "--shard", "1", "--nshard", "2"])
    assert W.pending(args2, sorted(rows), out) == ["obj2.view00"]

    def build(view):
        if view == "obj2.view01":
            raise FileNotFoundError("no render")
        return {"object_id": view, "value": torch.ones(2)}

    summary = W.run(args, out, sorted(rows), build, verify=lambda v, r: 0.0, tag="test")
    assert summary.created == 2 and [s["view"] for s in summary.skipped] == ["obj2.view01"]
    assert summary.verify == [{"view": "obj1.view01", "rel_drift": 0.0}]
    assert R.record_names(out) == ["obj1.view00", "obj1.view01", "obj2.view00"]
    assert torch.load(out / "obj2.view00.pt", weights_only=True)["object_id"] == "obj2.view00"
    assert json.loads((out / "_summary.shard0of1.json").read_text())["created"] == 2


def test_cache_config_example_loads():
    args = W.parse_args("t", ["--config", str(REPO / "train" / "configs" / "caches.example.yaml"), "--dataset", "train_5k"])
    cfg = W.load(args)
    ds = W.dataset(cfg, "train_5k")
    assert ds["key"] == "id" and ds["subset_dirs"] == {} and Path(ds["list"]).name == "train_5k.csv"
    assert len(W.list_rows(ds)) == 10486
    assert W.dataset(cfg, "curated_10k")["subset_dirs"]["3dfuture"] == "3D-FUTURE"
    assert W.output_dir(cfg, "texture_anchors", "curated_10k").name == "texture_anchors_10k"
