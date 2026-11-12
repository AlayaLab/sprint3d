#!/usr/bin/env python3
r"""Input list (JSON lines with ``id`` and ``path``) of the generation scripts from an object list of eval/lists/.

    python experiments/baselines/make_input_list.py --csv eval/lists/timing_80.csv --config eval/config.yaml \
        --out inputs/timing_80.jsonl
    python experiments/baselines/make_input_list.py --csv eval/lists/validation_400.csv \
        --dataset-root /path/to/validation_ground_truth --out inputs/validation_400.jsonl

One line per row of the CSV, in its order: ``{"id": "<id>_v<view>", "path": <conditioning render>, "sha256", "view"}``
with the conditioning render ``<root>/<source dir>/renders_cond/<sha256>/<view:03d>.png`` of docs/DATA.md. Rows
without ``sha256`` / ``source`` (timing_80.csv has ``id`` and ``view`` only) take them from
eval/lists/validation_400.csv by ``id``. The dataset root and the ``source`` -> directory map come from the
evaluation configuration (``--config``, keys ``dataset.root`` and ``dataset.source_dirs``) or ``--dataset-root``.
The same list serves scripts/time_e2e.py, scripts/infer.py and the scripts of experiments/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
VALIDATION = REPO / "eval" / "lists" / "validation_400.csv"


def load_config(path: Optional[str]) -> Dict:
    sys.path.insert(0, str(REPO / "eval"))
    from metrics.config import load_config as load

    cfg = load(path)
    return cfg["dataset"]


def build_rows(csv_path, root: Path, source_dirs: Dict[str, str], check: bool = True,
               lookup_csv=VALIDATION) -> List[Dict]:
    lookup = {r["id"]: r for r in csv.DictReader(open(lookup_csv))}
    out, missing = [], []
    for r in csv.DictReader(open(csv_path)):
        ref = lookup.get(r["id"], {})
        sha = r.get("sha256") or ref.get("sha256")
        source = r.get("source") or ref.get("source")
        if not sha or source is None:
            raise SystemExit(f"{csv_path}: no sha256 / source for id {r['id']!r} (not in {lookup_csv})")
        view = int(r["view"])
        path = root / source_dirs.get(source, source) / "renders_cond" / sha / f"{view:03d}.png"
        if check and not path.is_file():
            missing.append(str(path))
        out.append({"id": f"{r['id']}_v{view}", "path": str(path), "sha256": sha, "view": view})
    if missing:
        raise SystemExit(f"{len(missing)} conditioning render(s) not found, e.g. {missing[0]} (--no-check skips this)")
    ids = [x["id"] for x in out]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{csv_path}: duplicate (id, view) rows")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="object list with id and view (e.g. eval/lists/timing_80.csv)")
    ap.add_argument("--config", default=None, help="evaluation configuration (eval/config.example.yaml)")
    ap.add_argument("--dataset-root", default=None, help="ground-truth root (overrides dataset.root of --config)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-check", action="store_true", help="do not check that the renders exist")
    args = ap.parse_args(argv)
    dataset = load_config(args.config)
    root = args.dataset_root or dataset.get("root")
    if not root:
        raise SystemExit("give --dataset-root or a --config with dataset.root")
    root = Path(root).expanduser().resolve()      # a relative dataset.root is resolved by load_config already
    rows = build_rows(args.csv, root, dataset.get("source_dirs") or {}, check=not args.no_check)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} inputs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
