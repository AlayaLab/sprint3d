#!/usr/bin/env python3
"""Train the coarse-shape or fine-shape student: consistency distillation of a Pixal3D shape flow.

    torchrun --nproc_per_node 4 train/shape_consistency/train.py --config train/configs/coarse_shape_student.yaml
    torchrun --nproc_per_node 4 train/shape_consistency/train.py --config train/configs/fine_shape_student.yaml
    python train/shape_consistency/train.py --config train/configs/coarse_shape_student.yaml \\
        --set optim.updates=3 --set data.max_objects=4 --set output.dir=/tmp/coarse_smoke   # a short check

The student is the coarse (512) or fine (1024) shape flow with rank-256 adapters, deployed as a one-step
predictor of the clean latent at t = 1 (docs/TRAINING.md, "consistency distillation"). It reads one record per
object (train/caches/shape_conditioning_wild.py): the support ``coords`` and the DINOv3 ``global`` and projected
``proj`` features of the object's conditioning image. The frozen flow itself is the teacher. Every process
trains on one object per update and the processes average their gradients; the released students were
trained with 16 processes (``torchrun --nnodes 4 --nproc_per_node 4``).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from fastpixal3d.training import config as C  # noqa: E402
from fastpixal3d.training import consistency as CD  # noqa: E402
from fastpixal3d.training import loop  # noqa: E402

# The released runs (docs/TRAINING.md): the fields a configuration may leave null are filled from here.
RELEASED = {
    "coarse_shape": {"base_submodel": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16", "seed": 20260716,
                     "gradient_checkpointing": False},
    "fine_shape": {"base_submodel": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16", "seed": 20260723,
                   "gradient_checkpointing": True},
}

DEFAULTS = CD.defaults(student="coarse_shape", submodel=None, seed=None, gradient_checkpointing=None,
                       output_dir="runs/shape_student")
DEFAULTS["data"] = {
    "conditioning": None,    # directory of <object>.pt records (coords, global, proj)
    "objects": None,         # optional: object list (.csv with column `object_key`) or one id per line; default: all records
    "object_key": "id",      # CSV column that names the record files
    "exclude": None,         # optional: ids to leave out (.csv or one id per line)
    "max_objects": None,     # use only the first N objects of the sorted pool (short checks)
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML configuration (train/configs/{coarse,fine}_shape_student.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override one configuration key, e.g. --set optim.updates=10 (repeatable)")
    p.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    return p.parse_args(argv)


def fill_released(cfg: dict) -> dict:
    """Null ``model.base_submodel``, ``model.gradient_checkpointing`` and ``optim.seed`` take the released
    student's values."""
    student = cfg["model"]["student"]
    C.require(student in RELEASED, f"model.student must be one of {sorted(RELEASED)}")
    released = RELEASED[student]
    if cfg["model"]["base_submodel"] is None:
        cfg["model"]["base_submodel"] = released["base_submodel"]
    if cfg["model"]["gradient_checkpointing"] is None:
        cfg["model"]["gradient_checkpointing"] = released["gradient_checkpointing"]
    if cfg["optim"]["seed"] is None:
        cfg["optim"]["seed"] = released["seed"]
    return cfg


def load(argv=None) -> dict:
    args = parse_args(argv)
    cfg = C.load_config(args.config, DEFAULTS, path_keys=CD.PATH_KEYS, optional_path_keys=CD.OPTIONAL_PATH_KEYS,
                        overrides=args.overrides)
    cfg = fill_released(cfg)
    CD.check(cfg)
    if args.print_config:
        print(C.dump(cfg))
        raise SystemExit(0)
    return cfg


def object_ids(path, key: str):
    """Object ids of a CSV list (column ``key``) or of a text file (one per line, ``#`` comments)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and key not in rows[0]:
            raise KeyError(f"{path} has no column {key!r} (columns: {list(rows[0])})")
        return [row[key] for row in rows]
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def check_record(record, object_id: str) -> None:
    import torch

    coords, g, proj = record["coords"], record["global"], record["proj"]
    if not (torch.is_tensor(coords) and coords.ndim == 2 and coords.shape[1] == 4 and coords.shape[0] > 0):
        raise ValueError(f"{object_id}: coords must be a non-empty [N, 4] tensor")
    if coords.dtype.is_floating_point or bool((coords[:, 0] != 0).any()):
        raise ValueError(f"{object_id}: coords must be integer with a zero batch column")
    if not (torch.is_tensor(proj) and proj.ndim == 2 and proj.shape[0] == coords.shape[0]):
        raise ValueError(f"{object_id}: proj must be [N, D] with one row per support voxel")
    if not (torch.is_tensor(g) and g.ndim == 2):
        raise ValueError(f"{object_id}: global must be [K, D]")


def main(argv=None) -> dict:
    cfg = load(argv)
    run = loop.start(cfg)
    import torch

    from fastpixal3d.training.checkpoint import COARSE_SHAPE_SCHEMA, FINE_SHAPE_SCHEMA
    from fastpixal3d.training.records import load_record, training_pool

    d = cfg["data"]
    cond_dir = Path(d["conditioning"])
    wanted = set(object_ids(d["objects"], d["object_key"])) if d["objects"] else None
    pool = training_pool(cond_dir, wanted, None)
    if d["exclude"]:
        dropped = set(object_ids(d["exclude"], d["object_key"]))
        pool = [name for name in pool if name not in dropped]
    if d["max_objects"]:
        pool = pool[: int(d["max_objects"])]
    run.pool = pool
    student = cfg["model"]["student"]
    loop.build(run, student)  # seeded adapters, broadcast from rank 0
    from pixal3d.modules import sparse as sp

    device = run.device

    def sparse(feats, coords):
        return sp.SparseTensor(feats=feats, coords=coords)

    def load_object(object_id: str):
        record = load_record(cond_dir / f"{object_id}.pt")
        check_record(record, object_id)
        coords = record["coords"].to(device).int()
        cond = {"global": record["global"].to(torch.float32).unsqueeze(0).to(device),
                "proj": sparse(record["proj"].to(torch.float32).to(device), coords)}
        return coords, cond

    schema = {"coarse_shape": COARSE_SHAPE_SCHEMA, "fine_shape": FINE_SHAPE_SCHEMA}[student]
    provenance = CD.provenance(cfg, world_size=run.world_size, n_objects=len(pool),
                               trainer="train/shape_consistency/train.py")
    if pool:
        first = load_record(cond_dir / f"{pool[0]}.pt")
        provenance["conditioning_schema"] = first.get("schema")
    summary = CD.train(run, schema=schema, pool=pool, load_object=load_object, sparse=sparse, provenance=provenance)
    if run.is_main:
        print(json.dumps({"event": "done", **{k: v for k, v in summary.items() if k != "step_s"}}), flush=True)
    return summary


if __name__ == "__main__":
    main()
