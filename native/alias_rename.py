#!/usr/bin/env python3
"""Give an upstream CuMesh or cubvh checkout private names so that the patched
build can be imported next to the stock package in the same process.

Two compiled pybind11 modules that register the same C++ types (same namespace,
same class names) cannot be loaded together, and the stock ``cumesh`` package is
needed by other dependencies (o-voxel, Pixal3D). This script therefore renames,
purely textually and before the patches are applied:

  CuMesh  -> python package ``<alias>`` (default ``fastpixal3d_cumesh``),
             C++ namespaces ``cumesh`` -> ``<alias>_ns`` and
             ``cumesh_xatlas`` -> ``<alias>_xatlas_ns``.
  cubvh   -> python package ``<alias>`` (default ``fastpixal3d_cubvh``),
             extension module ``_cubvh`` -> ``_<alias>``,
             C++ namespace ``cumesh`` -> ``<alias>_ns``.

Nothing else is changed. The patch series under native/<name>/patches is
written against the renamed tree.

usage: alias_rename.py {cumesh,cubvh} SOURCE_DIR [--alias NAME]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

CXX_SUFFIXES = {".h", ".hpp", ".cuh", ".cu", ".cpp", ".cc", ".c"}


def _rewrite(path: Path, rules) -> bool:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    new = text
    for pattern, repl in rules:
        new = re.sub(pattern, repl, new)
    if new != text:
        path.write_text(new, encoding="utf-8", errors="surrogateescape")
        return True
    return False


def _cxx_files(root: Path, subdirs):
    for sub in subdirs:
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in CXX_SUFFIXES and "third_party/eigen" not in p.as_posix():
                yield p


def rename_cumesh(root: Path, alias: str) -> list:
    changed = []
    ns, xns = f"{alias}_ns", f"{alias}_xatlas_ns"
    cxx_rules = [
        (r"\bcumesh_xatlas\b", xns),
        (r"\bnamespace cumesh\b", f"namespace {ns}"),
        (r"\bcumesh::", f"{ns}::"),
    ]
    for p in _cxx_files(root, ["src", "third_party/cubvh/src", "third_party/cubvh/include", "third_party/xatlas"]):
        if _rewrite(p, cxx_rules):
            changed.append(p)
    pkg_old, pkg_new = root / "cumesh", root / alias
    if pkg_old.exists():
        shutil.move(str(pkg_old), str(pkg_new))
        changed.append(pkg_new)
    setup_rules = [
        (r'name="cumesh\.', f'name="{alias}.'),
        (r'name="cumesh"', f'name="{alias}"'),
        (r'packages=\["cumesh"\]', f'packages=["{alias}"]'),
    ]
    if _rewrite(root / "setup.py", setup_rules):
        changed.append(root / "setup.py")
    if (root / "pyproject.toml").exists() and _rewrite(root / "pyproject.toml", [(r'(?m)^name = "cumesh"', f'name = "{alias}"')]):
        changed.append(root / "pyproject.toml")
    return changed


def rename_cubvh(root: Path, alias: str) -> list:
    changed = []
    ns = f"{alias}_ns"
    cxx_rules = [
        (r"\bnamespace cumesh\b", f"namespace {ns}"),
        (r"\bcumesh::", f"{ns}::"),
    ]
    for p in _cxx_files(root, ["src", "include"]):
        if _rewrite(p, cxx_rules):
            changed.append(p)
    pkg_old, pkg_new = root / "cubvh", root / alias
    if pkg_old.exists():
        shutil.move(str(pkg_old), str(pkg_new))
        changed.append(pkg_new)
    for p in sorted(pkg_new.glob("*.py")):
        if _rewrite(p, [(r"\bimport _cubvh as _backend", f"import _{alias} as _backend")]):
            changed.append(p)
    if _rewrite(root / "setup.py", [(r"name='_cubvh'", f"name='_{alias}'")]):
        changed.append(root / "setup.py")
    if (root / "pyproject.toml").exists() and _rewrite(root / "pyproject.toml", [
            (r'(?m)^name = "cubvh"', f'name = "{alias}"'),
            (r'include = \["cubvh"\]', f'include = ["{alias}"]')]):
        changed.append(root / "pyproject.toml")
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", choices=("cumesh", "cubvh"))
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("--alias", default=None)
    args = ap.parse_args(argv)
    alias = args.alias or f"fastpixal3d_{args.project}"
    root = args.source_dir.resolve()
    fn = rename_cumesh if args.project == "cumesh" else rename_cubvh
    changed = fn(root, alias)
    print(f"[alias_rename] {args.project} -> {alias}: {len(changed)} paths changed", file=sys.stderr)
    return 0 if changed else 1


if __name__ == "__main__":
    sys.exit(main())
