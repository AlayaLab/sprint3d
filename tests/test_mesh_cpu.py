import math

import pytest
import torch

from fastpixal3d.mesh import remesh as R
from fastpixal3d.mesh import uv_atlas as U


def test_remesh_divisor_rule():
    assert R.remesh_divisor(int(0.03 * 1536 ** 2), 1536) == R.DIVISOR_FULL
    assert R.remesh_divisor(int(0.01 * 1536 ** 2), 1536) == R.DIVISOR_HALF
    assert R.remesh_divisor(10, 1536, adaptive=False) == R.DIVISOR_FULL


def test_remesh_arguments_full_and_half():
    scale, res = R.remesh_arguments(1.0, 1536, 1, R.DIVISOR_FULL)
    assert res == 1536 and scale == (1536 + 3) / 1536 * 1.0
    scale2, res2 = R.remesh_arguments(1.0, 1536, 1, R.DIVISOR_HALF)
    assert res2 == 768
    assert math.isclose(scale2, (768 + 3) / 768, rel_tol=1e-12)


def _sphere(n_lat=12, n_lon=24):
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            verts.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    verts.append((0.0, 0.0, -1.0))
    faces = []
    for j in range(n_lon):
        faces.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a = 1 + i * n_lon + j
            b = 1 + i * n_lon + (j + 1) % n_lon
            faces += [(a, a + n_lon, b), (b, a + n_lon, b + n_lon)]
    last = len(verts) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append((base + j, last, base + (j + 1) % n_lon))
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)


def test_uv_atlas_on_cpu_sphere():
    P, F = _sphere()
    out = U.uv_phase(P, F, torch.arange(P.shape[0]), texture_size=256, device="cpu", use_stream=False)
    uv, faces, vid = out["out_uvs"], out["out_faces"], out["out_vmaps"]
    assert faces.shape == F.shape and faces.dtype == torch.int32
    assert uv.min() >= 0 and uv.max() <= 1
    assert torch.equal(P[vid], out["out_vertices"])
    # every output face maps back to the input face's vertices
    assert torch.equal(vid[faces.long()], F)
    stats = out["stats"]
    assert stats["charts"] >= 1 and stats["overlap_texels"] == 0


def test_connected_components_and_adjacency():
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    fa, fb = U.face_adjacency(faces, 7)
    label, _ = U.connected_components(3, fa, fb, None)
    assert label[0] == label[1] and label[2] != label[0]


def test_shelf_pack_fits():
    import numpy as np

    w = np.array([10, 20, 5, 7]); h = np.array([4, 8, 9, 3])
    x, y, rot, used = U.shelf_pack(w, h, 32, 1)
    assert used <= 32 and (x >= 0).all()


def test_texture_map_to_uint8_matches_direct_conversion():
    import numpy as np

    from fastpixal3d.mesh.bake import texture_map_to_uint8

    def direct(values):
        return np.clip(values.cpu().numpy() * 255, 0, 255).astype(np.uint8)

    g = torch.Generator().manual_seed(0)
    attrs = torch.rand((64, 48, 6), generator=g) * 1.4 - 0.2
    attrs[0, :6, 0] = torch.tensor([0.0, 1.0, -0.0, float("inf"), -float("inf"), 0.5 / 255])
    outputs = []
    for sl in (slice(0, 3), slice(3, 4), slice(4, 5), slice(5, 6), slice(0, 3)):
        view = attrs[..., sl]  # strided view, as in the bake
        got = texture_map_to_uint8(view)
        assert got.dtype == np.uint8 and got.shape == tuple(view.shape)
        assert np.array_equal(got, direct(view))
        outputs.append((got, direct(view)))
    # results handed out earlier are not overwritten by later conversions that reuse the buffer
    for got, expected in outputs:
        assert np.array_equal(got, expected)
