import io

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
from PIL import Image

from fastpixal3d.export import glb


def _textured_cube():
    box = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
    vertices = np.asarray(box.vertices, dtype=np.float64)
    faces = np.asarray(box.faces, dtype=np.int64)
    rng = np.random.default_rng(0)
    uv = rng.random((len(vertices), 2))
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    base = Image.fromarray(rng.integers(0, 255, (16, 16, 4), dtype=np.uint8))
    mr = Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base, baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=mr, metallicFactor=1.0, roughnessFactor=1.0, alphaMode="OPAQUE", doubleSided=False)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))


def _load(data):
    return trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)


@pytest.mark.parametrize("use_fpng", [False, True])
def test_lean_writer_matches_trimesh_content(use_fpng):
    mesh = _textured_cube()
    assert glb.is_writer_shape(mesh)
    ours = _load(glb.glb_bytes(mesh, use_fpng=use_fpng))
    reference = _load(mesh.export(file_type="glb"))
    assert np.array_equal(ours.faces, reference.faces)
    assert np.allclose(ours.vertices, reference.vertices)
    assert np.allclose(ours.visual.uv, reference.visual.uv)
    assert np.allclose(ours.vertex_normals, reference.vertex_normals, atol=1e-6)
    for key in ("baseColorTexture", "metallicRoughnessTexture"):
        a = np.asarray(getattr(ours.visual.material, key))
        b = np.asarray(getattr(reference.visual.material, key))
        assert np.array_equal(a, b), key


def _chunks(data):
    import json
    import struct

    _magic, _version, _total, json_len, _json_magic = struct.unpack_from("<5I", data, 0)
    tree = json.loads(data[20:20 + json_len])
    bin_len, _bin_magic = struct.unpack_from("<2I", data, 20 + json_len)
    return tree, data[28 + json_len:28 + json_len + bin_len]


def test_without_fpng_the_container_equals_trimesh_export():
    """Same glTF tree (up to JSON key order) and byte-identical binary chunk."""
    mesh = _textured_cube()
    with glb.png_compress_level(1):
        reference = mesh.export(file_type="glb")
    tree, binary = _chunks(glb.glb_bytes(mesh, use_fpng=False))
    ref_tree, ref_binary = _chunks(reference)
    assert tree == ref_tree
    assert binary == ref_binary


def test_export_glb_rotates_and_writes(tmp_path):
    mesh = _textured_cube()
    before = mesh.vertices.copy()
    path = tmp_path / "out.glb"
    glb.export_glb(mesh, path, lean=True, use_fpng=False)
    assert path.stat().st_size > 0 and not (tmp_path / "out.partial.glb").exists()
    rotated = trimesh.transformations.transform_points(before, glb.PRODUCT_EXPORT_ROTATION)
    assert np.allclose(mesh.vertices, rotated)
    reloaded = trimesh.load(path, force="mesh", process=False)
    assert len(reloaded.faces) == 12


def test_prewarm_reuses_encoded_bytes():
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        prewarm = glb.PngPrewarm(pool.submit)
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        prewarm(img)
        data = prewarm.take(img)
        if glb.fpng_module() is None:
            assert data is None
        else:
            assert data is not None and data[:8] == b"\x89PNG\r\n\x1a\n"
        assert prewarm.take(img) is None
    finally:
        pool.shutdown()
