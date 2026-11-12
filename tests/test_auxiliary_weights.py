"""The weights directory of scripts/download_weights.py is where the package reads every model from."""
from pathlib import Path

import pytest

from fastpixal3d import auxiliary
from fastpixal3d.weights import WeightPaths


def _source(name, files):
    return {"name": name, "kind": "huggingface", "repo": f"org/{name}", "revision": "0" * 40,
            "local_dir": name, "files": [{"path": p, "size": n, "sha256": "0" * 64} for p, n in files]}


def _write(root: Path, source: dict, sizes=None):
    for entry in source["files"]:
        path = root / source["local_dir"] / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (entry["size"] if sizes is None else sizes[entry["path"]]))


def test_sources_cover_the_auxiliary_models():
    sources = auxiliary.load_sources()
    for name in ("students", "pixal3d", "dinov3", "moge2", "rmbg2", "naf"):
        assert sources[name]["local_dir"] == name
    assert sources["moge2"]["files"][0]["path"] == "model.pt"
    assert sources["naf"]["code"]["commit"] and sources["naf"]["files"][0]["path"] == "naf_release.pth"
    assert sources["rmbg2"]["repo"] == "briaai/RMBG-2.0"  # the model_name in Pixal3D's pipeline.json


def test_local_copy_needs_every_file_with_its_size(tmp_path):
    source = _source("dinov3", [("config.json", 3), ("model.safetensors", 5)])
    assert auxiliary.local_copy(tmp_path, source) is None
    _write(tmp_path, source, sizes={"config.json": 3, "model.safetensors": 4})
    assert auxiliary.local_copy(tmp_path, source) is None          # truncated file
    _write(tmp_path, source)
    assert auxiliary.local_copy(tmp_path, source) == tmp_path / "dinov3"


def test_weight_paths_default_to_the_weights_directory(tmp_path, monkeypatch):
    sources = {"students": _source("students", [("students.json", 2)]),
               "pixal3d": _source("pixal3d", [("pipeline.json", 2)])}
    monkeypatch.setattr(auxiliary, "load_sources", lambda *a: sources)
    weights = WeightPaths(root=str(tmp_path))
    assert weights.root_dir() == tmp_path
    assert weights.students_dir() is None
    for source in sources.values():
        _write(tmp_path, source)
    assert weights.students_dir() == str(tmp_path / "students")
    assert weights.model_dir() == str((tmp_path / "pixal3d").resolve())
    explicit = WeightPaths(root=str(tmp_path), students="elsewhere", pixal3d=str(tmp_path))
    assert explicit.students_dir() == "elsewhere" and explicit.model_dir() == str(tmp_path.resolve())
    checkpoints = WeightPaths(root=str(tmp_path), student_checkpoints={"structure": "a.pt"})
    assert checkpoints.students_dir() is None                       # training checkpoints win over the default


def test_weights_directory_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTPIXAL3D_WEIGHTS", str(tmp_path))
    assert WeightPaths.from_env().root_dir() == tmp_path
    monkeypatch.delenv("FASTPIXAL3D_WEIGHTS")
    assert WeightPaths.from_env().root_dir().name == "weights"


def test_missing_background_removal_fails_only_when_used():
    stand_in = auxiliary.MissingBackgroundRemoval("GatedRepoError: no access")
    assert stand_in.to("cuda") is stand_in and stand_in.cuda() is stand_in and stand_in.cpu() is stand_in
    with pytest.raises(RuntimeError, match="download_weights.py --only rmbg2"):
        stand_in(object())
