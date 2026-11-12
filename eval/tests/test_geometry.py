"""Chamfer / F-score, seeds, own-bbox normalisation and 32^3 support on synthetic point sets."""
import numpy as np
import pytest

from metrics import geometry, projection
from metrics.frame import coords_iou, support32_from_points


def test_chamfer_identical_and_shifted():
    g = np.stack(np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 3) * 0.1
    out = geometry.chamfer_fscore(g, g)
    assert out["chamfer"] == 0.0 and out["f_0.01"] == 1.0
    d = 0.003                                             # shift << grid spacing: nearest neighbour = own copy
    out = geometry.chamfer_fscore(g + np.array([d, 0, 0]), g)
    assert out["chamfer"] == pytest.approx(d, abs=1e-12)
    assert out["f_0.005"] == 1.0 and out["f_0.01"] == 1.0
    out = geometry.chamfer_fscore(g + np.array([0.007, 0, 0]), g)
    assert out["f_0.005"] == 0.0 and out["f_0.01"] == 1.0


def test_fscore_precision_recall():
    ref = np.zeros((4, 3))
    pred = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]], float)
    out = geometry.chamfer_fscore(pred, ref)
    assert out["precision_0.01"] == 0.5 and out["recall_0.01"] == 1.0
    assert out["f_0.01"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_stable_seed_is_pinned():
    # the salts are part of the reported numbers: changing them changes every sampled point set
    assert geometry.stable_seed(geometry.SEED_SALT_3D, "ab|0|0") == geometry.stable_seed("evalv2_3d", "ab|0|0")
    assert geometry.stable_seed("evalv2_3d", "x") == int.from_bytes(__import__("hashlib").sha256(b"evalv2_3d|x").digest()[:4], "big")


def test_own_bbox_normalisation_invariance():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(500, 3)) * [3.0, 1.0, 0.5] + [10, -4, 2]
    Vn, info = projection.normalise_own_bbox(V)
    assert float((Vn.max(0) - Vn.min(0)).max()) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose((Vn.max(0) + Vn.min(0)) / 2, 0.0, atol=1e-12)
    Vn2, _ = projection.normalise_own_bbox(V * 7.3 + 5.0)
    np.testing.assert_allclose(Vn, Vn2, atol=1e-9)
    with pytest.raises(ValueError):
        projection.normalise_own_bbox(np.zeros((3, 3)))


def test_support_occupancy_and_iou():
    pts = np.array([[-0.49, -0.49, -0.49], [0.0, 0.0, 0.0], [0.49, 0.49, 0.49], [0.6, 0.0, 0.0]])
    c, n_out = support32_from_points(pts)
    assert n_out == 1
    assert sorted(map(tuple, c)) == [(0, 0, 0), (16, 16, 16), (31, 31, 31)]
    r = coords_iou(c, c[:2])
    assert (r["iou"], r["precision"], r["recall"]) == pytest.approx((2 / 3, 2 / 3, 1.0))
    m = geometry.support_metrics(c, c)
    assert m["support_iou"] == 1.0


def test_sample_surface_is_seeded():
    import trimesh
    m = trimesh.creation.box()
    a = geometry.sample_surface(m, 1000, 5)
    b = geometry.sample_surface(m, 1000, 5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, geometry.sample_surface(m, 1000, 6))
