"""CPU tests of the student weight format: students.json schema, conversion, and that the loader
reads what the converter writes (the fold itself is tested in test_lora_fold.py).

No GPU and no real weights are needed; the checkpoints here are small random tensors.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from safetensors.torch import load_file  # noqa: E402

from fastpixal3d.students import convert as C  # noqa: E402
from fastpixal3d.students import load as L  # noqa: E402
import verify_weights as vw  # noqa: E402

TARGETS = ("self_attn.to_qkv", "cross_attn.proj_linear", "mlp.mlp.0")
DIMS = {"self_attn.to_qkv": (12, 36), "cross_attn.proj_linear": (16, 12), "mlp.mlp.0": (12, 20)}  # (in, out)


def fake_gen(rank=4, blocks=2, seed=0):
    """Name -> fp32 tensor, inserted in named_parameters() order (A then B per adapter)."""
    g = torch.Generator().manual_seed(seed)
    gen = {}
    for i in range(blocks):
        for target in TARGETS:
            fan_in, fan_out = DIMS[target]
            prefix = f"backbone.model.blocks.{i}.{target}"
            gen[prefix + ".lora_A.weight"] = torch.randn(rank, fan_in, generator=g)
            gen[prefix + ".lora_B.weight"] = torch.randn(fan_out, rank, generator=g) * 1e-2
    return gen


def fake_checkpoint(tmp_path, positional, provenance=None, name="ckpt.pt", seed=0):
    gen = fake_gen(seed=seed)
    ema = {n: t + 0.001 * torch.randn_like(t) for n, t in gen.items()}
    shadow = list(ema.values()) if positional else dict(reversed(list(ema.items())))
    payload = {"schema_version": 2, "completed_updates": 7, "gen": gen,
               "ema": {"shadow": shadow, "n": 7, "updates": 7}}
    if provenance is not None:
        payload["provenance"] = provenance
    path = tmp_path / name
    torch.save(payload, path)
    return path, gen, ema


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_describe_lora():
    info = C.describe_lora(fake_gen(rank=4, blocks=3))
    assert info["rank"] == 4
    assert info["target_modules"] == list(TARGETS)
    assert (info["num_blocks"], info["num_adapters"], info["num_tensors"]) == (3, 9, 18)
    assert info["dtypes"] == ["float32"]
    assert C.module_path("backbone.model.blocks.0.mlp.mlp.0.lora_B.weight") == "blocks.0.mlp.mlp.0"
    with pytest.raises(C.ConversionError):
        C.module_path("model.blocks.0.mlp.mlp.0.weight")


def test_convert_writes_bit_exact_ema_and_valid_students_json(tmp_path):
    out = tmp_path / "students"
    entries = {}
    for i, (student, positional) in enumerate([("structure", False), ("coarse_shape", True),
                                               ("fine_shape", True), ("texture", False)]):
        provenance = {"lora_rank": 4, "lora_alpha": 4.0} if not positional else None
        ckpt, gen, ema = fake_checkpoint(tmp_path, positional, provenance, name=f"{student}.pt", seed=i)
        entries[student] = C.convert(student, ckpt, out, lora_alpha=None if provenance else 4.0)
        tensors = load_file(str(out / C.STUDENTS[student]["file"]))
        assert set(tensors) == set(gen)
        assert all(bits_equal(tensors[n], ema[n]) for n in gen)
        assert entries[student]["sha256"] == C.sha256_file(out / C.STUDENTS[student]["file"])
        assert entries[student]["source_checkpoint_sha256"] == C.sha256_file(ckpt)

    doc = json.loads((out / "students.json").read_text())
    C.validate_students_json(doc)
    assert list(doc["students"]) == ["structure", "coarse_shape", "fine_shape", "texture"]
    assert doc["base_model"] == C.BASE_MODEL
    assert re.search(r"'backbone\.model\.blocks\.0\.[a-z_.]+\.lora_A\.weight'", doc["key_convention"])
    assert [doc["students"][s]["fold"] for s in doc["students"]] == ["none", "none", "bf16", "bf16"]
    assert doc["students"]["structure"]["steps"]["t"] == [1.0, 0.9, 0.75, 0.6, 0.0]
    assert doc["students"]["structure"]["steps"]["grid_indices"] == [0, 3, 6, 8, 12]
    assert doc["students"]["texture"]["steps"]["t"] == [1.0, 0.75, 0.0]
    assert doc["students"]["texture"]["steps"]["grid_indices"] == [0, 6, 12]
    assert doc["students"]["fine_shape"]["steps"]["num_steps"] == 1
    assert doc["students"]["coarse_shape"]["lora"] == {
        "rank": 4, "alpha": 4, "target_modules": list(TARGETS),
        "num_blocks": 2, "num_adapters": 6, "num_tensors": 12,
        "num_elements": sum(t.numel() for t in fake_gen().values())}

    # the inference loader reads the directory: target modules, key prefix, sha256, grid indices
    loaded = L.load_release(out, verify_sha256=True)
    wrapped = [f"blocks.{i}.{t}" for i in range(2) for t in TARGETS]
    parameters = [f"{n}.lora_{p}.weight" for n in wrapped for p in "AB"]
    for student, weights in loaded.items():
        L.check_target_modules(weights.spec, wrapped)
        relative, prefix = L.relative_state(weights.state, parameters)
        assert prefix == C.KEY_PREFIX and set(relative) == set(parameters)
        assert weights.spec.fold == C.STUDENTS[student]["fold"] and weights.spec.alpha == 4.0
    assert loaded["structure"].spec.steps["grid_indices"] == [0, 3, 6, 8, 12]

    # the standalone verifier accepts the directory and catches a corrupted file
    assert vw.main(["--students-dir", str(out)]) == 0
    target = out / C.STUDENTS["texture"]["file"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    assert vw.main(["--students-dir", str(out)]) == 1


def test_convert_is_deterministic(tmp_path):
    ckpt, _, _ = fake_checkpoint(tmp_path, True)
    first = C.convert("fine_shape", ckpt, tmp_path / "a", lora_alpha=4.0)
    second = C.convert("fine_shape", ckpt, tmp_path / "b", lora_alpha=4.0)
    assert first["sha256"] == second["sha256"]
    from safetensors import safe_open
    with safe_open(str(tmp_path / "a" / first["file"]), framework="pt") as handle:
        meta = json.loads(handle.metadata()[C.METADATA_KEY])
    assert meta == {"student": "fine_shape", "base_submodel": C.STUDENTS["fine_shape"]["base_submodel"],
                    "base_model": f"{C.BASE_MODEL['repo']}@{C.BASE_MODEL['revision']}",
                    "lora_rank": 4, "lora_alpha": 4}


def test_convert_refreshes_shared_fields(tmp_path):
    ckpt, _, _ = fake_checkpoint(tmp_path, True)
    out = tmp_path / "s"
    C.convert("fine_shape", ckpt, out, lora_alpha=4.0)
    manifest = out / "students.json"
    doc = json.loads(manifest.read_text())
    doc["key_convention"] = "stale"
    doc["fold_rule"] = {"none": "stale"}
    manifest.write_text(json.dumps(doc))
    C.convert("coarse_shape", ckpt, out, lora_alpha=4.0)
    doc = json.loads(manifest.read_text())
    assert doc["key_convention"] == C.KEY_CONVENTION and doc["fold_rule"] == C.FOLD_RULE
    assert list(doc["students"]) == ["coarse_shape", "fine_shape"]


def test_convert_checks_provenance_and_overwrite(tmp_path):
    ckpt, _, _ = fake_checkpoint(tmp_path, False, {"lora_rank": 4, "lora_alpha": 4.0, "k": 4,
                                                   "grid_indices": [0, 3, 6, 8, 12],
                                                   "grid_t_values": [1.0, 0.9, 0.75, 0.6, 0.0]})
    C.convert("structure", ckpt, tmp_path / "a")
    with pytest.raises(FileExistsError):
        C.convert("structure", ckpt, tmp_path / "a")
    with pytest.raises(C.ConversionError, match="lora_alpha"):
        C.convert("structure", ckpt, tmp_path / "b", lora_alpha=8.0)
    with pytest.raises(C.ConversionError, match="step"):
        C.convert("texture", ckpt, tmp_path / "c")  # trained for 4 steps, texture student has 2
    other, _, _ = fake_checkpoint(tmp_path, False, {"lora_rank": 4, "lora_alpha": 4.0, "grid_indices": [0, 2, 6, 8, 12]},
                                  name="other.pt")
    with pytest.raises(C.ConversionError, match="grid indices"):
        C.convert("structure", other, tmp_path / "d")


def test_validate_students_json_rejects_bad_documents(tmp_path):
    ckpt, _, _ = fake_checkpoint(tmp_path, True)
    C.convert("fine_shape", ckpt, tmp_path / "s", lora_alpha=4.0)
    doc = json.loads((tmp_path / "s" / "students.json").read_text())
    C.validate_students_json(doc, require_all=False)
    with pytest.raises(C.ConversionError, match="exactly"):
        C.validate_students_json(doc)
    for mutate, message in [
        (lambda d: d["base_model"].update(revision="main"), "revision"),
        (lambda d: d["students"]["fine_shape"].update(fold="fp8"), "fold"),
        (lambda d: d["students"]["fine_shape"].update(dtype="bfloat16"), "dtype"),
        (lambda d: d["students"]["fine_shape"].update(sha256="xyz"), "sha256"),
        (lambda d: d["students"]["fine_shape"]["steps"].update(grid_indices=[0, 12, 3]), "grid_indices"),
    ]:
        bad = json.loads(json.dumps(doc))
        mutate(bad)
        with pytest.raises(C.ConversionError, match=message):
            C.validate_students_json(bad, require_all=False)


def test_pickled_checkpoint_needs_explicit_permission(tmp_path):
    gen = fake_gen()
    payload = {"gen": gen, "ema": {"shadow": list(gen.values())},
               "rank_states": [{"numpy": __import__("numpy").random.get_state()}]}
    path = tmp_path / "pickled.pt"
    torch.save(payload, path)
    with pytest.raises(C.ConversionError, match="allow_pickle"):
        C.load_training_checkpoint(path)
    assert set(C.load_training_checkpoint(path, allow_pickle=True)) == set(payload)


def test_sources_json_is_pinned():
    doc = json.loads((REPO / "weights" / "sources.json").read_text())
    assert doc["format"] == 1
    names = [s["name"] for s in doc["sources"]]
    assert names == ["students", "pixal3d", "dinov3", "moge2", "rmbg2", "naf"]
    for source in doc["sources"]:
        assert source["files"], source["name"]
        for entry in source["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), (source["name"], entry["path"])
            assert isinstance(entry["size"], int) and entry["size"] > 0
        if source["kind"] == "huggingface":
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"]), source["name"]
    by_name = {s["name"]: s for s in doc["sources"]}
    assert by_name["pixal3d"]["repo"] == C.BASE_MODEL["repo"]
    assert by_name["pixal3d"]["revision"] == C.BASE_MODEL["revision"]
    pixal3d_files = {e["path"] for e in by_name["pixal3d"]["files"]}
    for spec in C.STUDENTS.values():
        assert spec["base_submodel"] + ".safetensors" in pixal3d_files
        assert spec["base_submodel"] + ".json" in pixal3d_files
    student_files = {e["path"] for e in by_name["students"]["files"]}
    assert student_files == {"students.json"} | {spec["file"] for spec in C.STUDENTS.values()}
    assert re.fullmatch(r"[0-9a-f]{40}", by_name["naf"]["code"]["commit"])


def test_verify_weights_on_a_fake_source(tmp_path):
    root = tmp_path / "w"
    (root / "m").mkdir(parents=True)
    (root / "m" / "a.bin").write_bytes(b"hello")
    sources = {"format": 1, "sources": [{"name": "m", "kind": "url", "url": "", "local_dir": "m", "files": [
        {"path": "a.bin", "size": 5, "sha256": vw.sha256_file(root / "m" / "a.bin")}]}]}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources))
    assert vw.main(["--sources", str(path), "--root", str(root)]) == 0
    (root / "m" / "a.bin").write_bytes(b"hellO")
    assert vw.main(["--sources", str(path), "--root", str(root)]) == 1
    assert vw.main(["--sources", str(path), "--root", str(root), "--size-only"]) == 0
    (root / "m" / "a.bin").unlink()
    assert vw.main(["--sources", str(path), "--root", str(root), "--size-only"]) == 1
