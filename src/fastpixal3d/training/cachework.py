"""Shared driver of the cache builders in ``train/caches/``.

Every builder writes one ``<view>.pt`` record per rendered view into an output directory. The
driver turns an object list and render indices into view names, keeps this process's share
(``--shard i --nshard n``: every n-th pending view), skips views whose record already exists
(records are written atomically, so an existing file is complete), stops after ``--limit`` views or
``--max-minutes``, optionally re-runs the first ``--verify-n`` views to check that the teacher
rollout is deterministic, and writes a JSON summary next to the records.

Configuration (``train/configs/caches.example.yaml``): ``pixal3d`` (weights and source checkout),
``datasets`` (render roots in the Pixal3D data-toolkit layout and the object lists) and ``caches``
(one output directory per cache). Each builder takes ``--config``, ``--dataset`` and ``--set``.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config as C
from .records import record_names, views_from_list

DATASET_DEFAULTS = {
    "root": None,          # dataset root: <root>/<subset dir>/{renders_cond,mesh_dumps}/...
    "subset_dirs": C.ANY,  # `source` column of the list -> sub-directory of root ({} = flat layout)
    "list": None,          # object list (data/lists/*.csv) or a file of view names
    "key": "sha256",       # list column used in view names <key>.viewNN
    "views": [0, 1],       # render indices per object
    "image_name": "image.png",  # shape_conditioning_wild.py: <root>/<subset dir>/<key>/<image_name>
}

DEFAULTS = {
    "pixal3d": {"model": None, "source": None},
    "runtime": {"device": "cuda", "threads": 4, "tf32_override": True},
    "datasets": {"curated_10k": dict(DATASET_DEFAULTS), "train_5k": dict(DATASET_DEFAULTS, key="id"),
                 "train_5k_images": dict(DATASET_DEFAULTS, key="id", views=[])},
    "caches": {
        "structure_conditioning": {"curated_10k": None, "train_5k": None},
        "structure_anchors": {"curated_10k": None, "train_5k": None},
        "coarse_shape_conditioning": {"curated_10k": None},
        "fine_shape_conditioning": {"curated_10k": None},
        "fine_shape_anchors": {"curated_10k": None},
        "texture_anchors": {"curated_10k": None},
        "coarse_shape_conditioning_wild": {"train_5k_images": None},   # shape students (shape_conditioning_wild.py)
        "fine_shape_conditioning_wild": {"train_5k_images": None},
    },
}
DATASET_NAMES = tuple(DEFAULTS["datasets"])
PATH_KEYS = [f"datasets.{d}.{k}" for d in DATASET_NAMES for k in ("root", "list")] + \
    [f"caches.{c}.{d}" for c, ds in DEFAULTS["caches"].items() for d in ds]
OPTIONAL_PATH_KEYS = ["pixal3d.model", "pixal3d.source"]


def parse_args(description: str, argv=None, extra: Optional[Callable[[argparse.ArgumentParser], None]] = None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="cache configuration (train/configs/caches.example.yaml)")
    p.add_argument("--dataset", default="curated_10k", help="dataset section of the configuration")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--only", default=None, help="file with the view names to build (default: list x views)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="build at most this many views")
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this wall time")
    p.add_argument("--verify-n", type=int, default=0, help="rebuild the first N views and report the drift")
    p.add_argument("--max-drift", type=float, default=5e-3, help="maximum relative drift of a verified rebuild")
    if extra is not None:
        extra(p)
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.nshard:
        raise SystemExit("--shard must be in [0, --nshard)")
    return args


def load(args) -> Dict[str, Any]:
    cfg = C.load_config(args.config, DEFAULTS, path_keys=PATH_KEYS, optional_path_keys=OPTIONAL_PATH_KEYS,
                        overrides=args.overrides)
    C.require(args.dataset in cfg["datasets"], f"unknown dataset {args.dataset!r} (known: {sorted(cfg['datasets'])})")
    return cfg


def dataset(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    d = dict(cfg["datasets"][name])
    d["subset_dirs"] = dict(d.get("subset_dirs") or {})
    return d


def output_dir(cfg: Dict[str, Any], cache: str, dataset_name: str) -> Path:
    path = (cfg["caches"].get(cache) or {}).get(dataset_name)
    C.require(path is not None, f"caches.{cache}.{dataset_name} is not set")
    return Path(path)


def list_rows(ds: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """view name -> {"sha256", "source", "view", "subset_dir"} for every object x render index of the list."""
    import csv

    C.require(ds["list"] is not None, "the dataset has no object list")
    out = {}
    with open(ds["list"], newline="") as handle:
        for row in csv.DictReader(handle):
            for v in ds["views"]:
                name = f"{row[ds['key']]}.view{int(v):02d}"
                out[name] = {"sha256": row["sha256"], "source": row.get("source", ""), "view": int(v),
                             "subset_dir": ds["subset_dirs"].get(row.get("source", ""), "")}
    return out


def pending(args, names: List[str], out_dir: Path) -> List[str]:
    if args.only:
        wanted = set(views_from_list(args.only, "", []))
        names = [n for n in names if n in wanted]
    have = set(record_names(out_dir)) if out_dir.is_dir() else set()
    todo = [n for n in names if n not in have]
    todo = [n for i, n in enumerate(todo) if i % args.nshard == args.shard]
    if args.limit:
        todo = todo[: args.limit]
    return todo


@dataclass
class Summary:
    requested: int = 0
    created: int = 0
    skipped: List[Dict[str, str]] = field(default_factory=list)
    verify: List[Dict[str, Any]] = field(default_factory=list)
    walls: List[float] = field(default_factory=list)
    stop_reason: str = "list_exhausted"


def run(args, out_dir: Path, names: List[str], build: Callable[[str], Dict[str, Any]],
        verify: Optional[Callable[[str, Dict[str, Any]], float]] = None, tag: str = "cache") -> Summary:
    """Build every pending view: ``build(view)`` returns the record (or raises ``FileNotFoundError``
    when an input is missing, which skips the view). ``verify(view, record)`` returns a relative
    drift for the first ``--verify-n`` built views."""
    from .records import atomic_torch_save

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = pending(args, names, out_dir)
    s = Summary(requested=len(todo))
    print(json.dumps({"event": "pending", "cache": tag, "views": len(todo), "out_dir": str(out_dir),
                      "shard": f"{args.shard}/{args.nshard}"}), flush=True)
    t_start = time.perf_counter()
    for view in todo:
        if args.max_minutes and (time.perf_counter() - t_start) / 60.0 > args.max_minutes:
            s.stop_reason = "max_minutes"
            break
        path = out_dir / f"{view}.pt"
        if path.exists():
            continue
        t0 = time.perf_counter()
        try:
            record = build(view)
        except FileNotFoundError as error:
            s.skipped.append({"view": view, "reason": f"missing input: {error}"})
            continue
        except Exception as error:  # one bad view must not stop a long build
            s.skipped.append({"view": view, "reason": f"{type(error).__name__}: {error}"})
            print(traceback.format_exc()[-1500:], flush=True)
            continue
        atomic_torch_save(record, path)
        s.created += 1
        s.walls.append(time.perf_counter() - t0)
        if verify is not None and len(s.verify) < args.verify_n:
            drift = float(verify(view, record))
            s.verify.append({"view": view, "rel_drift": drift})
            print(json.dumps({"event": "verify", "view": view, "rel_drift": drift}), flush=True)
            if drift > args.max_drift:
                raise SystemExit(f"rebuild of {view} drifted by {drift:.3g} > {args.max_drift}")
        if s.created % 25 == 0:
            print(json.dumps({"event": "progress", "created": s.created, "skipped": len(s.skipped),
                              "s_per_view": round(sum(s.walls[-50:]) / len(s.walls[-50:]), 3)}), flush=True)
    summary = {"cache": tag, "requested": s.requested, "created": s.created, "skipped": s.skipped,
               "verify": s.verify, "stop_reason": s.stop_reason,
               "s_per_view": (sum(s.walls) / len(s.walls)) if s.walls else None,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps({"event": "summary", **{k: v for k, v in summary.items() if k != "skipped"},
                      "n_skipped": len(s.skipped)}), flush=True)
    (out_dir / f"_summary.shard{args.shard}of{args.nshard}.json").write_text(json.dumps(summary, indent=1))
    return s


def rel_drift(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))
