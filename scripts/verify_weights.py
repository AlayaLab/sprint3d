#!/usr/bin/env python3
"""Check the size and sha256 of every weight file against the pinned list in weights/sources.json.

Usage:
    python scripts/verify_weights.py                    # all sources under weights/
    python scripts/verify_weights.py --only students,pixal3d --root /data/weights
    python scripts/verify_weights.py --students-dir path/to/students   # a directory made by convert.py

For the students, the entries of students.json (file name, size, sha256) are also checked against
the files. With --students-dir only students.json in that directory is used as the reference, which
is how a locally converted set of student weights is checked.

Exit status: 0 if every checked file matches, 1 otherwise. Needs only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_SOURCES = Path(__file__).resolve().parents[1] / "weights" / "sources.json"
STUDENT_NAMES = ("structure", "coarse_shape", "fine_shape", "texture")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_sources(path: Path) -> list[dict]:
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != 1 or not isinstance(doc.get("sources"), list):
        raise ValueError(f"{path}: not a format-1 source list")
    return doc["sources"]


def select(sources: list[dict], only: str | None) -> list[dict]:
    if not only:
        return sources
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    known = {source["name"] for source in sources}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; known: {sorted(known)}")
    return [source for source in sources if source["name"] in wanted]


def file_status(path: Path, size: int, sha256: str | None, actual_sha256: str | None) -> str:
    """'ok', 'missing', 'size ...' or 'sha256 ...'; sha256=None checks the size only."""
    if not path.is_file():
        return "missing"
    actual_size = path.stat().st_size
    if actual_size != size:
        return f"size {actual_size} != {size}"
    if sha256 is not None and actual_sha256 != sha256:
        return f"sha256 {actual_sha256} != {sha256}"
    return "ok"


def check_many(items: list[tuple[str, Path, int, str | None]], jobs: int) -> list[tuple[str, Path, str]]:
    """items: (label, path, size, sha256 or None) -> (label, path, status), in input order.

    Each file is hashed at most once, even when several items refer to it."""
    to_hash = sorted({path for _, path, size, sha in items
                      if sha is not None and path.is_file() and path.stat().st_size == size})
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        hashes = dict(zip(to_hash, pool.map(sha256_file, to_hash)))
    return [(label, path, file_status(path, size, sha, hashes.get(path))) for label, path, size, sha in items]


def students_json_items(directory: Path, size_only: bool) -> tuple[list, list[str]]:
    """Items for the files listed in <directory>/students.json, plus schema problems found."""
    manifest = directory / "students.json"
    if not manifest.is_file():
        return [], [f"{manifest}: missing"]
    doc = json.loads(manifest.read_text())
    problems = []
    if doc.get("format") != 1:
        problems.append(f"{manifest}: format is {doc.get('format')!r}, expected 1")
    students = doc.get("students", {})
    if set(students) != set(STUDENT_NAMES):
        problems.append(f"{manifest}: students {sorted(students)} != {sorted(STUDENT_NAMES)}")
    items = []
    for name in STUDENT_NAMES:
        entry = students.get(name)
        if not entry:
            continue
        items.append((f"students.json:{name}", directory / entry["file"], int(entry["size_bytes"]),
                      None if size_only else entry["sha256"]))
    return items, problems


def run(root: Path, sources: list[dict], size_only: bool = False, jobs: int = 4,
        students_dir: Path | None = None) -> int:
    items, problems = [], []
    if students_dir is not None:
        more, problems = students_json_items(students_dir, size_only)
        items += more
    for source in sources:
        base = root / source["local_dir"]
        for entry in source["files"]:
            items.append((source["name"], base / entry["path"], int(entry["size"]),
                          None if size_only else entry["sha256"]))
        if source["name"] == "students" and (base / "students.json").is_file():
            more, found = students_json_items(base, size_only)
            items += more
            problems += found
    results = check_many(items, jobs)
    width = max((len(label) for label, _, _ in results), default=8)
    failures = 0
    for label, path, status in results:
        failures += status != "ok"
        print(f"{status if status == 'ok' else 'FAIL':4s}  {label:<{width}s}  {path}"
              + ("" if status == "ok" else f"  ({status})"))
    for problem in problems:
        print(f"FAIL  {problem}")
    failures += len(problems)
    checked = "size" if size_only else "size and sha256"
    print(f"{len(results) - sum(s != 'ok' for _, _, s in results)}/{len(results)} files match ({checked}); "
          f"{failures} problem(s)")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify weight files against pinned sizes and sha256.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="pinned source list (JSON)")
    parser.add_argument("--root", type=Path, default=None,
                        help="weights root (default: $FASTPIXAL3D_WEIGHTS or the directory of --sources)")
    parser.add_argument("--only", default=None, help="comma-separated source names (default: all)")
    parser.add_argument("--students-dir", type=Path, default=None,
                        help="check only the student files listed in <dir>/students.json")
    parser.add_argument("--size-only", action="store_true", help="skip hashing")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="files hashed in parallel")
    args = parser.parse_args(argv)
    if args.students_dir is not None:
        return run(args.students_dir, [], size_only=args.size_only, jobs=args.jobs, students_dir=args.students_dir)
    root = args.root or Path(os.environ.get("FASTPIXAL3D_WEIGHTS") or args.sources.parent)
    sources = select(load_sources(args.sources), args.only)
    return run(root, sources, size_only=args.size_only, jobs=args.jobs)


if __name__ == "__main__":
    sys.exit(main())
