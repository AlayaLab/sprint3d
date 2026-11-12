import dataclasses
import re

import pytest

from fastpixal3d import config as C
from fastpixal3d import levers as L


def test_export_settings_are_pixal3d_defaults():
    assert C.EXPORT == C.ExportSettings(decimation_target=1_000_000, texture_size=4096)
    assert C.Config().export == C.EXPORT


def test_levers_default_all_on_and_switchable():
    lv = C.Levers()
    assert all(lv.as_dict().values())
    off = lv.with_disabled(["adaptive_remesh", "texture_student"])
    assert off.disabled() == ["adaptive_remesh", "texture_student"]
    assert off.with_enabled(["adaptive_remesh"]).disabled() == ["texture_student"]
    assert not any(C.Levers.baseline().as_dict().values())
    with pytest.raises(KeyError):
        lv.with_disabled(["no_such_lever"])


def test_lever_table_matches_flags():
    names = [f.name for f in dataclasses.fields(C.Levers)]
    assert names == [lv.name for lv in L.LEVERS]
    assert len(set(names)) == len(names) == 20
    for lv in L.LEVERS:
        assert lv.output in L.OUTPUT_CLASSES
        assert re.fullmatch(r"[a-z0-9_]+", lv.name)
        assert lv.description and lv.paper_name


def test_output_classes_of_students_and_remesh():
    for name in ("structure_student", "coarse_shape_student", "fine_shape_student", "texture_student",
                 "adaptive_remesh", "gpu_chart_segmentation"):
        assert L.lever(name).output == L.OUTPUT_CHANGING
    for name in ("fa3_texture_flow", "fa3_fine_shape_flow", "fold_lora_fine_shape", "fast_gpu_simplify", "gpu_bvh_build"):
        assert L.lever(name).output == L.NEAR_IDENTICAL


def test_markdown_table_lists_every_lever():
    table = L.markdown_table()
    for lv in L.LEVERS:
        assert f"`{lv.name}`" in table


def test_fine_shape_seed_is_stable():
    a = C.stable_fine_shape_seed("object", 0)
    assert a == C.stable_fine_shape_seed("object", 0)
    assert a != C.stable_fine_shape_seed("object", 1)
    assert 0 <= a < (1 << 63) - 1


def test_sampling_constants():
    s = C.Sampling()
    assert s.structure_grid_indices == (0, 3, 6, 8, 12)
    assert (s.texture_student_steps, s.texture_student_rescale_t) == (2, 3.0)
    assert (s.hr_requested_resolution, s.hr_max_tokens) == (1536, 49_152)
