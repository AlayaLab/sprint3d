"""The experiment configurations (experiments/configs): flags exist, order and lever sets are consistent."""
import re
from pathlib import Path

import pytest

from fastpixal3d import config as C
from fastpixal3d import levers as L

CONFIGS = Path(__file__).resolve().parents[1] / "experiments" / "configs"
SETS = ("cumulative", "ablations", "same_step_teacher")
PAPER_ROW = re.compile(r"^# Paper row: \S")


def files(name):
    return sorted((CONFIGS / name).glob("*.yaml"))


def load(path):
    return C.load_config_file(path)


def on_set(spec):
    return {k for k, v in C.config_from_mapping(spec).levers.as_dict().items() if v}


CHAIN_ROWS = [  # (file stem, paper row) of the cumulative chain, in the report's order
    ("00_pixal3d_rerun", "Pixal3D re-run"),
    ("01_tiled_mesh_path", "+ tiled mesh path (16-tile atlas)"),
    ("02_mesh_prep_worker_thread", "+ mesh pre-processing on a worker thread"),
    ("03_fixed_bake_launch_config", "+ fixed bake-kernel launch configuration"),
    ("04_fast_gpu_simplify", "+ faster GPU simplification and charting"),
    ("05_gpu_bvh_build", "+ GPU BVH build"),
    ("06_adaptive_remesh", "+ adaptive half-resolution remesh"),
    ("07_lean_glb_writer", "+ lean GLB writer, parallel PNG"),
    ("08_gpu_chart_segmentation", "+ single-pass GPU chart segmentation"),
    ("09_fa3_texture_flow", "+ FlashAttention-3, texture flow"),
    ("10_texture_kv_reuse", "+ texture K/V reuse across steps"),
    ("11_shared_conditioning", "+ shared image conditioning"),
    ("12_fa3_fine_shape_flow", "+ FlashAttention-3, fine-shape flow only"),
    ("13_texture_side_stream", "+ side-stream texture sampling, early PNG"),
    ("14_structure_student", "+ structure student: trajectory, 4 steps"),
    ("15_texture_projection_reuse", "+ texture projection reuse across steps"),
    ("16_shape_students", "+ shape students: consistency, 1 step each"),
    ("17_fold_lora_fine_shape", "+ LoRA folded, fine-shape student"),
    ("18_ours", "Ours (+ texture student: trajectory, 2 steps)"),
]
BLOCKS = {"post_processing": range(1, 9), "flow_engineering": range(9, 14), "distillation": range(14, 19)}
STUDENTS = {"structure_student", "coarse_shape_student", "fine_shape_student", "texture_student"}


def chain_files():
    return [p for p in files("cumulative") if not p.name.startswith("x")]


def control_files():
    return [p for p in files("cumulative") if p.name.startswith("x")]


@pytest.mark.parametrize("name", SETS)
def test_every_file_parses_names_are_unique_and_rows_are_numbered(name):
    paths = files(name)
    assert paths, name
    names = []
    numbered = [p for p in paths if not p.name.startswith("x")]
    for k, path in enumerate(numbered):
        assert path.name.startswith(f"{k:02d}_"), path.name
    for k, path in enumerate(p for p in paths if p.name.startswith("x")):
        assert name == "cumulative" and path.name.startswith(f"x{k + 1}_"), path.name
    for path in paths:
        text = path.read_text()
        assert PAPER_ROW.match(text.splitlines()[0]), f"{path.name}: first line must be '# Paper row: ...'"
        spec = load(path)
        assert spec["method"] in ("fast", "official")
        assert set(spec) <= {"name", "method", "adds", "removes", "levers", "sampling", "block", "control_of",
                             "use_structure_graph", "structure_adapter"}, path.name
        if spec["method"] == "fast":
            cfg = C.config_from_mapping(spec)  # raises on unknown levers / sampling keys
            assert cfg.export == C.EXPORT
        names.append(spec["name"])
        assert re.fullmatch(r"[a-z0-9_]+", spec["name"])
    assert len(names) == len(set(names)), name


def test_lever_names_in_configs_exist():
    for path in CONFIGS.rglob("*.yaml"):
        spec = load(path)
        for key in ("adds", "removes"):
            for lever in spec.get(key) or []:
                assert lever in L.LEVERS_BY_NAME, (path.name, lever)
        levers = spec.get("levers") or []
        for lever in (levers if isinstance(levers, list) else levers.keys()):
            assert lever in L.LEVERS_BY_NAME, (path.name, lever)


def test_unknown_lever_or_setting_is_rejected():
    with pytest.raises(KeyError):
        C.config_from_mapping({"levers": ["no_such_lever"]})
    with pytest.raises(KeyError):
        C.config_from_mapping({"levers": {"no_such_lever": False}})
    with pytest.raises(KeyError):
        C.config_from_mapping({"sampling": {"no_such_setting": 1}})


def test_cumulative_chain_is_the_reports_order():
    paths = chain_files()
    assert [(p.stem, p.read_text().splitlines()[0][len("# Paper row: "):]) for p in paths] == CHAIN_ROWS


def test_cumulative_order_adds_one_step_at_a_time():
    specs = [load(p) for p in chain_files()]
    assert specs[0]["method"] == "official" and specs[0]["name"] == "pixal3d_rerun"
    assert all(s["method"] == "fast" for s in specs[1:])
    assert on_set(specs[1]) == set(), "the tiled mesh path row has every lever off"
    assert not C.config_from_mapping(specs[1]).levers.gpu_chart_segmentation
    prev = set()
    for k, spec in enumerate(specs[1:], start=1):
        cur = on_set(spec)
        adds = set(spec["adds"])
        assert prev <= cur, f"{spec['name']}: a lever of an earlier row is off"
        assert cur - prev == adds, f"{spec['name']}: adds {sorted(adds)} but changes {sorted(cur - prev)}"
        assert spec["levers"] == [lv.name for lv in L.LEVERS if lv.name in cur], f"{spec['name']}: lever order"
        for lever in adds:
            assert L.lever(lever).row == f"{k:02d}", (spec["name"], lever)
        prev = cur
    assert prev == set(L.LEVERS_BY_NAME), "the last row is the deployed configuration"
    assert [lv.name for lv in L.LEVERS] == [a for s in specs[1:] for a in s["adds"]], "lever table in chain order"


def test_cumulative_blocks():
    specs = [load(p) for p in chain_files()]
    assert "block" not in specs[0]
    for block, rows in BLOCKS.items():
        assert [k for k, s in enumerate(specs) if s.get("block") == block] == list(rows), block


def test_released_flows_until_row_13_students_from_row_14():
    specs = [load(p) for p in chain_files()]
    for k, spec in enumerate(specs[1:], start=1):
        cfg = C.config_from_mapping(spec)
        students = {name for name in STUDENTS if getattr(cfg.levers, name)}
        if k <= 13:
            assert not students, spec["name"]
            assert cfg.sampling == C.Sampling(), spec["name"]  # Pixal3D's 12-step samplers
        expected = {14: {"structure_student"}, 15: {"structure_student"},
                    16: STUDENTS - {"texture_student"}, 17: STUDENTS - {"texture_student"}, 18: STUDENTS}
        if k >= 14:
            assert students == expected[k], spec["name"]
    assert C.Sampling().teacher_structure_steps == C.Sampling().teacher_shape_steps == 12
    assert C.Sampling().teacher_texture_steps == 12


def test_cumulative_rows_up_to_chart_segmentation_use_the_tiled_path():
    specs = [load(p) for p in chain_files()][1:]
    tiled = [s["name"] for s in specs if not C.config_from_mapping(s).levers.gpu_chart_segmentation]
    assert tiled == ["tiled_mesh_path", "mesh_prep_worker_thread", "fixed_bake_launch_config", "fast_gpu_simplify",
                     "gpu_bvh_build", "adaptive_remesh", "lean_glb_writer"]
    # the two levers of the tiled CPU unwrap enter with the row that replaces it
    for spec in specs:
        cfg = C.config_from_mapping(spec)
        assert cfg.levers.per_tile_single_thread_unwrap == cfg.levers.pinned_bvh_transfers == \
            cfg.levers.gpu_chart_segmentation, spec["name"]


def test_controls_differ_from_their_row_by_the_listed_levers():
    chain = {s["name"]: s for s in (load(p) for p in chain_files())}
    controls = {p.name.split("_", 1)[0]: load(p) for p in control_files()}
    assert sorted(controls) == ["x1", "x2", "x3"]
    expected = {"x1": ("gpu_chart_segmentation", set(), {"per_tile_single_thread_unwrap", "pinned_bvh_transfers"}),
                "x2": ("texture_side_stream", {"fold_lora_fine_shape"}, set()),
                "x3": ("gpu_chart_segmentation", set(), {"adaptive_remesh"})}
    for key, (ref, adds, removes) in expected.items():
        spec = controls[key]
        assert spec["control_of"] == ref and "block" not in spec
        assert set(spec["adds"]) == adds and set(spec["removes"]) == removes
        assert on_set(spec) == (on_set(chain[ref]) | adds) - removes, key


def test_structure_student_uses_the_trajectory_adapter_everywhere():
    """Every configuration with the structure student selects the students.json entry 'structure'
    (rows 14-18 name it explicitly); no configuration names another structure adapter."""
    for path in CONFIGS.rglob("*.yaml"):
        spec = load(path)
        assert "structure_consistency" not in path.read_text(), path.name
        if spec["method"] != "fast":
            continue
        cfg = C.config_from_mapping(spec)
        assert cfg.structure_adapter == "structure", path.name
        if path.parent.name == "cumulative" and cfg.levers.structure_student:
            assert spec.get("structure_adapter") == "structure", path.name
    assert C.config_from_mapping({}).structure_adapter == "structure"


def test_structure_graph_is_on_wherever_the_structure_student_is():
    for path in CONFIGS.rglob("*.yaml"):
        spec = load(path)
        if spec["method"] == "fast" and C.config_from_mapping(spec).levers.structure_student:
            assert C.config_from_mapping(spec).use_structure_graph, path.name


def test_ablations_remove_one_lever_from_ours():
    specs = [load(p) for p in files("ablations")]
    assert on_set(specs[0]) == set(L.LEVERS_BY_NAME)
    removed = []
    for spec in specs[1:]:
        off = set(L.LEVERS_BY_NAME) - on_set(spec)
        assert off == set(spec["removes"]) and len(off) == 1, spec["name"]
        removed += spec["removes"]
    assert removed == ["mesh_prep_worker_thread", "fa3_texture_flow", "texture_kv_reuse", "texture_projection_reuse"]


def test_same_step_teachers_replace_one_student():
    specs = {s["name"]: s for s in (load(p) for p in files("same_step_teacher"))}
    assert on_set(specs["ours"]) == set(L.LEVERS_BY_NAME)
    ss = C.config_from_mapping(specs["structure_4_steps"])
    assert not ss.levers.structure_student and not ss.levers.texture_projection_reuse
    assert ss.sampling.teacher_structure_grid_indices == (0, 3, 6, 8, 12)
    assert ss.sampling.teacher_structure_guidance is False
    for name, lever in (("coarse_shape_1_step", "coarse_shape_student"), ("fine_shape_1_step", "fine_shape_student")):
        cfg = C.config_from_mapping(specs[name])
        assert set(L.LEVERS_BY_NAME) - on_set(specs[name]) == {lever}
        assert cfg.sampling.teacher_shape_steps == 1 and cfg.sampling.teacher_shape_guidance is False


def test_config_from_mapping_list_and_mapping_forms():
    cfg = C.config_from_mapping({"levers": ["texture_student"], "sampling": {"structure_grid_indices": [0, 6, 12]}})
    assert cfg.levers.disabled() == [lv.name for lv in L.LEVERS if lv.name != "texture_student"]
    assert cfg.sampling.structure_grid_indices == (0, 6, 12)
    cfg = C.config_from_mapping({"levers": {"adaptive_remesh": False}})
    assert cfg.levers.disabled() == ["adaptive_remesh"] and cfg.export == C.EXPORT
    assert C.config_from_mapping({}) == C.Config()
