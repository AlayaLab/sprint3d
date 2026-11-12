"""Official crop, own inference-time projection, frame chain and baseline frame maps."""
import math
import types

import numpy as np
import pytest

from metrics import baselines, projection
from metrics.frame import P2D, ViewFrame
from metrics.official_crop import crop_to_raw_ndc, official_crop_box


def _rgba(size, box):
    from PIL import Image
    a = np.zeros((size[1], size[0], 4), np.uint8)
    a[box[1]:box[3] + 1, box[0]:box[2] + 1, 3] = 255
    return Image.fromarray(a, "RGBA")


def test_crop_formula():
    c = official_crop_box(_rgba((1024, 1024), (301, 117, 780, 851)))
    # bbox w = 479, h = 734 -> size int(734 * 1.1) = 807; centre (540.5, 484); box rounded half to even
    assert c["alpha_bbox"] == [301, 117, 780, 851]
    assert c["crop_box"] == [round(540.5 - 403), round(484 - 403), round(540.5 + 403), round(484 + 403)]
    assert c["crop_size"][0] == c["crop_size"][1]
    big = official_crop_box(_rgba((2048, 2048), (900, 400, 1500, 1301)))
    assert big["resize_scale"] == 0.5 and big["source_size"] == [1024, 1024]
    with pytest.raises(ValueError):
        from PIL import Image
        official_crop_box(Image.new("RGBA", (8, 8), (0, 0, 0, 255)))   # no usable alpha


def test_crop_matches_official_source_when_available():
    """runs the official preprocess_image source text when a Pixal3D checkout is present (third_party/Pixal3D)"""
    import ast
    import textwrap
    from pathlib import Path
    src_path = Path(__file__).resolve().parents[2] / "third_party/Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py"
    if not src_path.is_file():
        pytest.skip("no Pixal3D checkout")
    from PIL import Image
    src = src_path.read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "preprocess_image")
    ns = {"np": np, "Image": Image}
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    dummy = types.SimpleNamespace(low_vram=False, rembg_model=None, device="cpu")
    for size, box in [((1024, 1024), (301, 117, 780, 851)), ((640, 480), (100, 50, 399, 301)), ((2048, 2048), (900, 400, 1500, 1301))]:
        im = _rgba(size, box)
        assert list(ns["preprocess_image"](dummy, im).size) == official_crop_box(im)["crop_size"]


def test_ndc_affine():
    n = crop_to_raw_ndc([0, 0, 1024, 1024], [1024, 1024])
    assert (n["sx"], n["sy"], n["tx"], n["ty"]) == (1.0, 1.0, 0.0, 0.0)
    n = crop_to_raw_ndc([512, 0, 1024, 512], [1024, 1024])      # top-right quadrant
    assert (n["sx"], n["sy"], n["tx"], n["ty"]) == (0.5, 0.5, 0.5, 0.5)


def test_own_projection_matches_official_formula():
    fov, ms = 0.7, 1.0
    d = 0.5 / math.tan(fov / 2) / ms
    proj = projection.OwnProjection(fov, d, ms)
    assert proj.summary()["distance_matches_official_formula"]
    v_vox = np.array([[-0.5, 0.0, 0.0], [0.5, 0.25, 0.0]])          # voxel frame: x right, y up, z toward the camera
    v_cam = proj.product_to_cam(v_vox @ P2D)
    ndc_x = (1.0 / math.tan(fov / 2)) * v_cam[:, 0] / -v_cam[:, 2]
    assert ndc_x[0] == pytest.approx(-1.0, abs=1e-9)                 # the cube's mid-plane edge maps to the image edge
    assert ndc_x[1] == pytest.approx(1.0, abs=1e-9)


def test_view_frame_is_a_pure_rotation_and_chains_invert(synthetic):
    vf = synthetic["view_frame"]
    assert vf.summary()["R_view_equals_w2c"]
    V = synthetic["V"]
    np.testing.assert_allclose(vf.product_to_world(vf.world_to_product(V)), V, atol=1e-12)
    no_scale = ViewFrame.from_files(synthetic["gt_root"] / "SRC/renders_cond" / ("ab" * 32) / "transforms.json", 0)
    assert no_scale.total_scale is None
    with pytest.raises(ValueError):
        no_scale.world_to_voxel(V)                                    # only true_camera rows need the scale


@pytest.fixture()
def tilted_view():
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = [0.0, 1.0, 0.0], [-0.5, 0.0, math.sqrt(0.75)], [math.sqrt(0.75), 0.0, 0.5]
    return types.SimpleNamespace(R_view=c2w[:3, :3].T.copy(), c2w=c2w)   # elevation 30 deg


def test_frame_map_file():
    fm = baselines.load_frame_maps()
    m = fm["methods"]
    assert set(m) == {"trellis2", "triposg", "hunyuan3d21", "direct3d_s2"}
    assert m["hunyuan3d21"]["geometry_file"] == "shape_untextured.glb"
    assert {k for k, v in m.items() if v["textured"]} == {"trellis2", "hunyuan3d21"}
    assert (m["triposg"]["family"], m["hunyuan3d21"]["family"]) == ("view_aligned", "azimuth_aligned")


def test_trellis2_axis_swap(tilted_view):
    spec = baselines.load_frame_maps()["methods"]["trellis2"]
    out = baselines.to_world_axes(spec, tilted_view, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(out, [[1.0, -3.0, 2.0]])                # (gx, -gz, gy): GLB +Y -> world +Z
    assert float(np.linalg.det(np.asarray(spec["matrix"]))) == pytest.approx(1.0)


def test_camera_referenced_families(tilted_view):
    v = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])      # GLB up, GLB +z, GLB right
    va = baselines.to_world_axes({"family": "view_aligned", "depth_sign": 1}, tilted_view, v)
    az = baselines.to_world_axes({"family": "azimuth_aligned", "depth_sign": 1}, tilted_view, v)
    np.testing.assert_allclose(va[0], tilted_view.c2w[:3, 1], atol=1e-12)  # view-aligned: GLB up = camera up
    np.testing.assert_allclose(az[0], [0.0, 0.0, 1.0], atol=1e-12)         # azimuth-aligned: GLB up = world up
    np.testing.assert_allclose(az[1], [1.0, 0.0, 0.0], atol=1e-12)         # GLB +z = horizontal direction to the camera
    np.testing.assert_allclose(va[2], az[2], atol=1e-12)
    R = baselines.azimuth_rotation(tilted_view)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    neg = baselines.to_world_axes({"family": "azimuth_aligned", "depth_sign": -1}, tilted_view, v)
    np.testing.assert_allclose(neg[1], -az[1], atol=1e-12)
