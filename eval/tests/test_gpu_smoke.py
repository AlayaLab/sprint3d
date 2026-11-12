"""GPU smoke test: nvdiffrast renders of the synthetic object agree with the CPU rasteriser.  Skipped without a CUDA
device or nvdiffrast."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


def _need_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    pytest.importorskip("nvdiffrast.torch")


def test_nvdiffrast_matches_cpu_raster(synthetic):
    _need_gpu()
    from cpu_raster import CPURaster
    from metrics import normal_maps, precision
    from metrics.render import Raster
    precision.enforce_exact_fp32()
    vf = synthetic["view_frame"]
    dump = synthetic["gt_root"] / "SRC/mesh_dumps" / f"{'ab' * 32}.pickle"
    g = normal_maps.render_gt_normals(Raster("cuda"), vf, dump, device="cuda")
    c = normal_maps.render_gt_normals(CPURaster(), vf, dump, device="cpu")
    gm, cm = g["mask"].cpu(), c["mask"]
    iou = float((gm & cm).sum()) / float((gm | cm).sum())
    assert iou > 0.99
    both = (gm & cm).numpy()
    diff = (g["normal"].cpu().numpy() - c["normal"].numpy())[both]
    assert float(np.abs(diff).mean()) < 1e-3
    assert precision.matmul_probe("cuda")["max_abs_error"] < precision.MAX_PROBE_ERROR
