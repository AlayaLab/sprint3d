#!/usr/bin/env python
"""fetch_metric_models.py -- download the third-party metric models at the pinned revisions and check their sha256.

Everything goes into one directory (--dest), in the layout metrics/config.model_paths() expects; set models.dir of the
configuration to the same directory.  Pins and checksums: metrics/config.PINS.  Licences: docs/EVALUATION.md.

  group      files                                                                 needed for
  table1     torchvision AlexNet + VGG16 ImageNet weights (LPIPS backbones);       normal-map table (LPIPS)
             LPIPS v0.1 linear layers ship with the pip package lpips==0.1.4 (checked here)
  clip       openai/clip-vit-large-patch14                                         appendix: CLIP / CLIP-N
  ulip2      ULIP-2 PointBERT checkpoint + OpenCLIP ViT-bigG-14 + ULIP code         appendix: ULIP-2
  uni3d      Uni3D-g checkpoint + OpenCLIP EVA02-E-14-plus + Uni3D code             appendix: Uni3D
  dinov2     DINOv2 ViT-L/14 weights + dinov2 code                                  appendix: agreement with a reference

usage: python eval/fetch_metric_models.py --dest /path/to/metric_models [--only table1,clip] [--verify-only]
About 28 GB for all groups (the two OpenCLIP towers are about 10 GB each).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE))

from metrics.config import PINS, model_paths  # noqa: E402

GROUPS = {"table1": ["torchvision_alexnet", "torchvision_vgg16", "lpips_linear_layers"],
          "clip": ["clip"],
          "ulip2": ["ulip2_checkpoint", "openclip_bigg", "code_ulip"],
          "uni3d": ["uni3d_checkpoint", "openclip_eva02", "code_uni3d"],
          "dinov2": ["dinov2_weights", "code_dinov2"]}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, want: str) -> bool:
    ok = path.is_file() and sha256_of(path) == want
    print(f"  [{'ok' if ok else 'BAD' if path.is_file() else 'missing'}] {path}")
    return ok


def fetch_url(url: str, dest: Path, want: str, verify_only: bool) -> bool:
    if not verify_only and not (dest.is_file() and sha256_of(dest) == want):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 22)
        tmp.replace(dest)
    return check(dest, want)


def fetch_hf(pin: dict, hub_cache: Path, verify_only: bool) -> bool:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    ok = True
    for fname, want in pin["files"].items():
        if verify_only:
            p = try_to_load_from_cache(pin["repo"], fname, cache_dir=str(hub_cache), revision=pin["revision"])
            p = Path(p) if isinstance(p, str) else hub_cache / f"{pin['repo']}@{pin['revision'][:8]}/{fname}"
        else:
            p = Path(hf_hub_download(pin["repo"], fname, revision=pin["revision"], cache_dir=str(hub_cache)))
        ok &= check(p, want)
    return ok


def fetch_hf_file(pin: dict, local_dir: Path, verify_only: bool) -> bool:
    from huggingface_hub import hf_hub_download
    dest = local_dir / pin["file"]
    if not verify_only and not (dest.is_file() and sha256_of(dest) == pin["sha256"]):
        hf_hub_download(pin["repo"], pin["file"], repo_type=pin.get("repo_type", "model"), revision=pin["revision"],
                        local_dir=str(local_dir))
    return check(dest, pin["sha256"])


def fetch_git(pin: dict, dest: Path, verify_only: bool) -> bool:
    def head():
        r = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    if not verify_only and head() != pin["commit"]:
        dest.mkdir(parents=True, exist_ok=True)
        run = lambda *c: subprocess.run(["git", "-C", str(dest), *c], check=True)  # noqa: E731
        if not (dest / ".git").exists():
            run("init", "-q")
            run("remote", "add", "origin", pin["git"])
        run("fetch", "-q", "--depth", "1", "origin", pin["commit"])
        run("checkout", "-q", "--detach", "FETCH_HEAD")
    got = head()
    ok = got == pin["commit"]
    print(f"  [{'ok' if ok else 'missing' if got is None else 'BAD'}] {dest} @ {pin['commit'][:12]}")
    return ok


def fetch_lpips() -> bool:
    try:
        import lpips
    except ImportError:
        print("  [missing] python package lpips==0.1.4 (pip install lpips==0.1.4)")
        return False
    root = Path(lpips.__file__).resolve().parent
    return all([check(root / rel, want) for rel, want in PINS["lpips_linear_layers"]["files"].items()])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="destination directory (= models.dir of the configuration)")
    ap.add_argument("--only", default=",".join(GROUPS), help="comma list of groups: " + ",".join(GROUPS))
    ap.add_argument("--verify-only", action="store_true", help="check files and checksums, download nothing")
    a = ap.parse_args(argv)
    paths = model_paths(a.dest)
    hub = Path(paths["hf_home"]) / "hub"
    torch_ckpt = Path(paths["torch_home"]) / "hub" / "checkpoints"
    groups = [g.strip() for g in a.only.split(",") if g.strip()]
    bad = set(groups) - set(GROUPS)
    if bad:
        raise SystemExit(f"unknown groups {sorted(bad)}")
    ok = True
    for g in groups:
        print(f"[{g}]")
        for item in GROUPS[g]:
            pin = PINS[item]
            if item.startswith("torchvision_"):
                ok &= fetch_url(pin["url"], torch_ckpt / pin["url"].rsplit("/", 1)[1], pin["sha256"], a.verify_only)
            elif item == "lpips_linear_layers":
                ok &= fetch_lpips()
            elif item == "dinov2_weights":
                ok &= fetch_url(pin["url"], Path(paths["dinov2_weights"]), pin["sha256"], a.verify_only)
            elif item in ("clip", "openclip_bigg", "openclip_eva02"):
                ok &= fetch_hf(pin, hub, a.verify_only)
            elif item == "ulip2_checkpoint":
                ok &= fetch_hf_file(pin, Path(a.dest) / "ulip2", a.verify_only)
            elif item == "uni3d_checkpoint":
                ok &= fetch_hf_file(pin, Path(a.dest) / "uni3d", a.verify_only)
            elif item.startswith("code_"):
                key = {"code_ulip": "ulip_code", "code_uni3d": "uni3d_code", "code_dinov2": "dinov2_code"}[item]
                ok &= fetch_git(pin, Path(paths[key]), a.verify_only)
    print("all files present and verified" if ok else "SOME FILES ARE MISSING OR DIFFER (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
