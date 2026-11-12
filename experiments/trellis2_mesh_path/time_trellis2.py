#!/usr/bin/env python3
r"""Timing run of TRELLIS.2 and of TRELLIS.2 with the rebuilt mesh path (protocol: docs/TIMING.md).

    # TRELLIS.2 as released
    python experiments/trellis2_mesh_path/time_trellis2.py --list inputs.jsonl --out-dir runs/trellis2_stock \
        --arms stock --trellis2-src third_party/TRELLIS.2 --trellis2-weights weights/trellis2_4b
    # the mesh path, with and without the adaptive remesh, and the serial control, in one process
    python experiments/trellis2_mesh_path/time_trellis2.py --list inputs.jsonl --out-dir runs/trellis2_mesh \
        --arms mesh_path,mesh_path_adaptive_remesh,mesh_path_serial --trellis2-src ... --trellis2-weights ...
    python experiments/trellis2_mesh_path/summarize_trellis2.py --run stock=runs/trellis2_stock \
        --run mesh_path=runs/trellis2_mesh/mesh_path --run mesh_path_adaptive_remesh=runs/trellis2_mesh/mesh_path_adaptive_remesh \
        --serial runs/trellis2_mesh/mesh_path_serial

One image: ``--image input.png --out-dir out/`` (the id is the image stem).

Arms (mesh_path.py): ``stock`` (TRELLIS.2's run() and o-voxel's to_glb), ``mesh_path``, ``mesh_path_adaptive_remesh``,
``mesh_path_serial``. TRELLIS.2 is loaded once per process and shared by the arms. With several arms, every input
runs with every arm, in an order rotated by the input's list position so that each arm runs first equally often
(``--no-rotate`` keeps the given order); each arm writes its own directory ``<out-dir>/<arm>/``.

Per arm: run.json, timing.jsonl (the format of scripts/time_e2e.py, also read by scripts/analyze_timing.py),
<id>/model.glb and <id>/camera.json (see experiments/baselines/baseline_io.py); sampler.jsonl in <out-dir>.
Every timing line holds ``e2e_s`` (wall clock, docs/TIMING.md), ``stage_sum_s`` (sum of the synchronised stages,
the definition of the report's table), ``flows_s`` and ``post_processing_s``, the stage times, the face counts and
the sha256 of the shape and texture latents (equal across arms for the same input and seed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "baselines"))

import baseline_io  # noqa: E402
from mesh_path import ALL_ARMS, MESH_LEVERS, arm_levers  # noqa: E402
from trellis2_pipeline import WEIGHTS, default_source  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    baseline_io.add_common_arguments(ap)
    ap.set_defaults(warmup_requests=0)
    ap.add_argument("--arms", default="stock,mesh_path",
                    help=f"comma-separated arms, from {','.join(ALL_ARMS)} (default: stock,mesh_path)")
    ap.add_argument("--disable", action="append", default=[], metavar="LEVER",
                    help=f"switch a lever of the mesh-path arms off (repeatable): {', '.join(MESH_LEVERS)}")
    ap.add_argument("--trellis2-src", default=None,
                    help="TRELLIS.2 checkout at commit 75fbf018 (default: $FASTPIXAL3D_TRELLIS2_SRC or "
                         "third_party/TRELLIS.2)")
    ap.add_argument("--trellis2-weights", default=WEIGHTS,
                    help="TRELLIS.2-4B: local directory or Hugging Face id (default: microsoft/TRELLIS.2-4B)")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="run the arms in the given order for every input")
    ap.add_argument("--no-latent-hashes", dest="latent_hashes", action="store_false",
                    help="do not record the sha256 of the latents (computed after the clock stops)")
    return ap


def parse_arms(text: str, disable=()):
    arms = [a.strip() for a in text.split(",") if a.strip()]
    if not arms:
        raise SystemExit("--arms is empty")
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ALL_ARMS)}")
    if len(set(arms)) != len(arms):
        raise SystemExit(f"--arms lists an arm twice: {arms}")
    for arm in arms:
        if arm != "stock":
            try:
                arm_levers(arm, disable)
            except (KeyError, ValueError) as error:
                raise SystemExit(str(error.args[0] if error.args else error))
    if disable and arms == ["stock"]:
        raise SystemExit("--disable applies to the mesh-path arms only")
    return arms


def build(args):
    from mesh_path import build_arm

    source = args.trellis2_src or default_source()
    return [build_arm(arm, args.disable, source=source, weights=args.trellis2_weights,
                      record_latents=args.latent_hashes) for arm in parse_arms(args.arms, args.disable)]


def main(argv=None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    parse_arms(args.arms, args.disable)          # argument errors before anything is loaded
    return baseline_io.main_for(build, ap, argv)


if __name__ == "__main__":
    sys.exit(main())
