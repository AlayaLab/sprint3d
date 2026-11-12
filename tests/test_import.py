import importlib

import pytest


@pytest.mark.parametrize("name", [
    "fastpixal3d", "fastpixal3d.config", "fastpixal3d.levers", "fastpixal3d.weights",
    "fastpixal3d.students.lora", "fastpixal3d.students.fold", "fastpixal3d.students.load",
    "fastpixal3d.mesh.uv_atlas", "fastpixal3d.mesh.remesh", "fastpixal3d.mesh.simplify", "fastpixal3d.mesh.bvh",
    "fastpixal3d.mesh.tiled_path", "fastpixal3d.export.glb", "fastpixal3d.runtime.timing",
    "fastpixal3d.runtime.streams", "fastpixal3d.runtime.attention", "fastpixal3d.stages.shape",
    "fastpixal3d.stages.structure",
])
def test_import(name):
    importlib.import_module(name)


def test_lazy_pipeline_attribute():
    import fastpixal3d

    assert "FastPixal3D" in fastpixal3d.__all__


def test_tiled_path_names_its_legacy_builds():
    from fastpixal3d.mesh import tiled_path

    assert set(tiled_path.HOST_BVH_MODULES) == {"openmp", "openmp_nogil", "openmp_pinned"}
    assert tiled_path.XATLAS_SINGLE_MODULE == "fastpixal3d_xatlas_single"
