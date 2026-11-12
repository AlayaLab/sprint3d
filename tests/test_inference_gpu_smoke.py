"""End-to-end smoke test on a GPU (one image; 1,000,000-face / 4096 x 4096 export).

Skipped unless a CUDA GPU is visible and the weights are found: in the weights directory of
scripts/download_weights.py (FASTPIXAL3D_WEIGHTS, by default weights/), or given by

    FASTPIXAL3D_PIXAL3D        Pixal3D weights (local directory or Hugging Face repo id)
    FASTPIXAL3D_STUDENTS       directory with students.json
    FASTPIXAL3D_PIXAL3D_SRC    Pixal3D source checkout (default: third_party/Pixal3D)
    FASTPIXAL3D_SMOKE_IMAGE    input image (default: an example image of the Pixal3D checkout)
    FASTPIXAL3D_SMOKE_ARGS     extra arguments for scripts/infer.py (optional, whitespace separated)

Run with ``pytest -q -m gpu tests/test_inference_gpu_smoke.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _smoke_image():
    if os.environ.get("FASTPIXAL3D_SMOKE_IMAGE"):
        return Path(os.environ["FASTPIXAL3D_SMOKE_IMAGE"])
    src = Path(os.environ.get("FASTPIXAL3D_PIXAL3D_SRC") or REPO / "third_party" / "Pixal3D")
    candidates = sorted((src / "assets" / "images").glob("*.png")) if (src / "assets" / "images").is_dir() else []
    return candidates[0] if candidates else None


def _weights():
    from fastpixal3d.weights import WeightPaths

    return WeightPaths.from_env()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not _weights().students_dir() and "--student-checkpoints"
                       not in os.environ.get("FASTPIXAL3D_SMOKE_ARGS", ""),
                       reason="no student weights (weights/students or FASTPIXAL3D_STUDENTS)"),
    pytest.mark.skipif(not os.environ.get("FASTPIXAL3D_PIXAL3D") and _weights().downloaded("pixal3d") is None,
                       reason="no Pixal3D weights (weights/pixal3d or FASTPIXAL3D_PIXAL3D)"),
]


def test_end_to_end(tmp_path):
    image = _smoke_image()
    if image is None or not image.is_file():
        pytest.skip("no smoke image (set FASTPIXAL3D_SMOKE_IMAGE)")
    output = tmp_path / "smoke_v00.glb"
    timing = tmp_path / "timing.json"
    cmd = [sys.executable, str(REPO / "scripts" / "infer.py"), "--image", str(image), "--output", str(output),
           "--warmup", "0", "--runs", "1", "--timing-json", str(timing),
           "--dump-dir", str(tmp_path / "dump")]
    cmd += os.environ.get("FASTPIXAL3D_SMOKE_ARGS", "").split()
    subprocess.run(cmd, check=True, cwd=REPO, timeout=3600)

    import trimesh

    assert output.is_file() and output.stat().st_size > 0
    mesh = trimesh.load(output, force="mesh", process=False)
    assert 0 < len(mesh.faces) <= 1_000_000 * 1.01
    material = mesh.visual.material
    assert material.baseColorTexture is not None and material.baseColorTexture.size == (4096, 4096)
    assert material.metallicRoughnessTexture is not None

    report = json.loads(timing.read_text())
    assert report["export_settings"] == {"decimation_target": 1_000_000, "texture_size": 4096}
    timings = report["timings"]
    for key in ("preprocess_s", "ss_rollout_s", "hr_rollout_s", "tex_rollout_s", "glb_bake_finalize_s", "to_glb_s", "export_s", "e2e_s"):
        assert key in timings and timings[key] > 0, key

    camera = json.loads(output.with_suffix(".json").read_text())
    for key in ("camera_angle_x", "distance", "mesh_scale"):
        assert isinstance(camera[key], float) and camera[key] > 0, key

    dump = tmp_path / "dump" / "smoke_v00"
    for name in ("support.npy", "coarse.npz", "fine.npz", "timing.json"):
        assert (dump / name).is_file(), name
