#!/usr/bin/env python3
"""Download the student weights and the upstream models at the revisions pinned in weights/sources.json.

Usage:
    python scripts/download_weights.py                       # everything, into weights/
    python scripts/download_weights.py --root /data/weights  # elsewhere (or set FASTPIXAL3D_WEIGHTS)
    python scripts/download_weights.py --only students,pixal3d
    python scripts/download_weights.py --students-repo USER/REPO   # another copy of the student repository

Hugging Face files are fetched with huggingface_hub at the pinned revision; other files are fetched
from their URL. Files already present with the pinned size and sha256 are skipped, and every
downloaded file is checked against the pinned sha256 (scripts/verify_weights.py repeats the check).

The gated upstream repositories need `hf auth login` with a token that has read
access to it. Some upstream repositories are gated (see docs/WEIGHTS.md): accept their terms on the
model page with the same login.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_weights as vw  # noqa: E402

PLACEHOLDER_MARK = "<"


def fetch_url(url: str, target: Path, sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as tmp:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    digest.update(block)
                    tmp.write(block)
        except BaseException:
            os.unlink(tmp.name)
            raise
    if digest.hexdigest() != sha256:
        os.unlink(tmp.name)
        raise RuntimeError(f"{url}: sha256 {digest.hexdigest()} != pinned {sha256}")
    os.replace(tmp.name, target)


def fetch_hf(repo: str, revision: str, filename: str, target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(target_dir))
    except GatedRepoError as error:
        raise SystemExit(f"{repo} is gated: accept its terms at https://huggingface.co/{repo} "
                         f"and log in with `hf auth login`, then rerun.\n{error}") from error
    except RepositoryNotFoundError as error:
        raise SystemExit(f"{repo} not found (or not accessible with the current login; a private repository "
                         f"needs `hf auth login` with a token that has read access to it).\n{error}") from error
    return Path(path)


def download_source(source: dict, root: Path, verify: bool) -> int:
    """Fetch the missing or mismatching files of one source. Returns the number of files fetched."""
    base = root / source["local_dir"]
    fetched = 0
    for entry in source["files"]:
        target = base / entry["path"]
        expected = entry["sha256"]
        if target.is_file() and target.stat().st_size == entry["size"] and vw.sha256_file(target) == expected:
            print(f"have  {source['name']}: {entry['path']}")
            continue
        print(f"fetch {source['name']}: {entry['path']} ({entry['size'] / 1e9:.3f} GB)", flush=True)
        if source["kind"] == "huggingface":
            path = fetch_hf(source["repo"], source["revision"], entry["path"], base)
            if path.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            if verify:
                status = vw.check_many([(source["name"], target, entry["size"], expected)], jobs=1)[0][2]
                if status != "ok":
                    raise SystemExit(f"{target}: {status} (pinned revision {source['revision']})")
        elif source["kind"] == "url":
            fetch_url(source["url"], target, expected)
        else:
            raise SystemExit(f"unknown source kind {source['kind']!r}")
        fetched += 1
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download pinned weight files.")
    parser.add_argument("--sources", type=Path, default=vw.DEFAULT_SOURCES, help="pinned source list (JSON)")
    parser.add_argument("--root", type=Path, default=None,
                        help="weights root (default: $FASTPIXAL3D_WEIGHTS or the directory of --sources)")
    parser.add_argument("--only", default=None, help="comma-separated source names (default: all)")
    parser.add_argument("--students-repo", default=None,
                        help="Hugging Face repository of the student weights (overrides sources.json)")
    parser.add_argument("--students-revision", default=None, help="revision of the student repository")
    parser.add_argument("--no-verify", action="store_true", help="do not hash newly downloaded files")
    args = parser.parse_args(argv)

    root = args.root or Path(os.environ.get("FASTPIXAL3D_WEIGHTS") or args.sources.parent)
    sources = vw.select(vw.load_sources(args.sources), args.only)
    for source in sources:
        if source["name"] == "students":
            source["repo"] = args.students_repo or source["repo"]
            source["revision"] = args.students_revision or source["revision"]
            if PLACEHOLDER_MARK in source["repo"]:
                raise SystemExit("the student repository is not set: pass --students-repo <user>/<repo> "
                                 "or fill in 'repo' for 'students' in weights/sources.json")
    root.mkdir(parents=True, exist_ok=True)
    fetched = sum(download_source(source, root, verify=not args.no_verify) for source in sources)
    files = sum(len(source["files"]) for source in sources)
    checked = "" if args.no_verify else ", all with the pinned size and sha256"
    print(f"{files} file(s) under {root} ({fetched} fetched now{checked})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
