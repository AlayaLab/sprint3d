"""scripts/parity_check.py quantities and the per-input records, on synthetic data (CPU only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pc():
    spec = importlib.util.spec_from_file_location("parity_check", REPO / "scripts" / "parity_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_identical_ignores_order(pc):
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=np.int32)
    assert pc.support_identical(coords, coords[::-1])["identical"]
    assert not pc.support_identical(coords, coords[:2])["identical"]


def test_latent_rel_fro_matches_rows_by_coordinates(pc):
    rng = np.random.default_rng(0)
    coords = np.stack([np.zeros(50), rng.permutation(50), np.arange(50), np.arange(50)], axis=1).astype(np.int64)
    feats = rng.standard_normal((50, 32))
    perm = rng.permutation(50)
    out = pc.latent_rel_fro((coords, feats), (coords[perm], feats[perm]))
    assert out["rel_fro"] == 0.0 and out["rows_matched"] == 50
    noisy = pc.latent_rel_fro((coords, feats), (coords, feats * (1 + 1e-4)))
    assert noisy["rel_fro"] == pytest.approx(1e-4, rel=1e-6)


def test_psnr(pc):
    a = np.zeros((8, 8, 3))
    assert pc.image_psnr(a, a) == float("inf")
    assert pc.image_psnr(a, a + 1.0) == pytest.approx(20 * np.log10(255.0))
    assert pc.image_psnr(a, np.zeros((4, 4, 3))) is None


def test_compare_on_records(pc, tmp_path):
    for run, scale in (("ref", 1.0), ("cand", 1.0 + 5e-4)):
        d = tmp_path / run / "obj"
        d.mkdir(parents=True)
        coords = np.array([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32)
        np.save(d / "support.npy", coords)
        for name in ("coarse", "fine"):
            np.savez(d / f"{name}.npz", coords=coords, feats=np.ones((2, 32), np.float32) * scale)
        (d / "timing.json").write_text(json.dumps({"e2e_s": 4.0 * scale, "sha256": {"fine_latent": str(scale)}}))
    report = pc.compare(tmp_path / "ref", tmp_path / "cand", ["obj"], None, None, exclude_first=0, samples=10)
    assert report["checks"]["support"] is True
    assert report["checks"]["coarse"] is True and report["checks"]["fine"] is True
    assert report["checks"]["e2e"] is True
    assert report["checks"]["faces"] is None  # no GLBs given
    assert report["summary"]["latents_identical"][0] is False and report["checks"]["latents"] is False


def test_camera_record():
    from fastpixal3d.pipeline import camera_record

    result = {"stats": {"camera": {"camera_angle_x": 0.5, "distance": 2.0, "mesh_scale": 1.0}, "prepared_size": [518, 518]}}
    record = camera_record(result, "obj_v00", "in.png")
    assert record["camera"] == {"camera_angle_x": 0.5, "distance": 2.0, "mesh_scale": 1.0}
    assert record["camera_angle_x"] == 0.5 and record["distance"] == 2.0 and record["mesh_scale"] == 1.0
    assert record["object_id"] == "obj_v00"
