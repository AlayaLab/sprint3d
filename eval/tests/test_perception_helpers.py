"""Perception helpers that run without models: point clouds, uv convention, CUDA-extension stand-ins, CLIP plumbing."""
import numpy as np
import pytest
import torch

from metrics import perception, scorers3d
from metrics.render import gltf_uv_from_trimesh


def test_uv_convention_flip():
    uv = np.array([[0.25, 0.0], [0.5, 1.0], [0.75, 0.3]], np.float32)
    out = gltf_uv_from_trimesh(uv)
    np.testing.assert_allclose(out, [[0.25, 1.0], [0.5, 0.0], [0.75, 0.7]], atol=1e-7)


def test_colored_point_cloud_unit_sphere_and_texture_colour():
    import trimesh
    from PIL import Image
    m = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    pc = perception.colored_point_cloud(m, 2000, seed=3)
    assert pc.shape == (2000, 6)
    assert float(np.linalg.norm(pc[:, :3], axis=1).max()) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(pc[:, 3:], 0.5)                       # untextured -> gray
    tex = Image.new("RGB", (4, 4), (255, 0, 0))
    uv = np.full((len(m.vertices), 2), 0.5)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
    pc = perception.colored_point_cloud(m, 500, seed=3)
    np.testing.assert_allclose(pc[:, 3:], [[1.0, 0.0, 0.0]] * 500)


def test_fps_stub_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    xyz = torch.rand(1, 200, 3, generator=g)
    idx = scorers3d._fps_index(xyz, 16)[0].long()
    assert idx[0] == 0
    chosen = [0]
    d = ((xyz[0] - xyz[0, 0]) ** 2).sum(-1)
    for _ in range(15):
        nxt = int(d.argmax())
        chosen.append(nxt)
        d = torch.minimum(d, ((xyz[0] - xyz[0, nxt]) ** 2).sum(-1))
    assert idx.tolist() == chosen


def test_knn_stub():
    ref = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    query = torch.tensor([[[0.9, 0.0, 0.0]]])
    dist, idx = scorers3d._KNN(k=2, transpose_mode=True)(ref, query)
    assert idx[0, 0].tolist() == [1, 0]
    assert dist[0, 0, 0] == pytest.approx(0.1, abs=1e-6)


def test_pc_normalize():
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]])
    p = scorers3d.pc_normalize(x)
    np.testing.assert_allclose(p.mean(0), 0.0, atol=1e-12)
    assert float(np.sqrt((p ** 2).sum(1)).max()) == pytest.approx(1.0)


def test_scorer_reports_missing_models_instead_of_scores(tmp_path):
    s = perception.ULIP2Scorer("cpu", {"ulip_code": str(tmp_path), "ulip2_checkpoint": str(tmp_path / "none.pt")})
    out = s.score(None, None, 0)
    assert out["ulip2"] is None and out["ulip2_status"].startswith("unavailable")


def test_clip_scores_with_a_fake_scorer(tmp_path):
    from PIL import Image
    png = tmp_path / "in.png"
    Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(png)

    class Fake:
        model_id = "fake"

        def embed(self, images):
            e = torch.tensor([[float(np.asarray(im).mean() > 128), 1.0] for im in images])
            return torch.nn.functional.normalize(e, dim=-1)

    views = {"rgb": torch.ones(4, 8, 8, 3), "normal": torch.zeros(4, 8, 8, 3), "textured": True}
    out = perception.clip_scores(Fake(), views, png)
    assert out["clip"] == pytest.approx(1 / np.sqrt(2), abs=1e-6)    # white renders vs a dark input
    assert out["clip_n"] == pytest.approx(1.0, abs=1e-6)
    assert len(out["clip_per_view"]) == 4 and out["clip_input_view"] == out["clip_per_view"][0]
