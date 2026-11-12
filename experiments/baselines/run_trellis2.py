#!/usr/bin/env python3
r"""TRELLIS.2 baseline: image(s) -> GLB at the released defaults (trellis2_pipeline.py), timed per stage.

    python experiments/baselines/run_trellis2.py --list inputs.jsonl --out-dir runs/trellis2 \
        --trellis2-src third_party/TRELLIS.2 --trellis2-weights weights/trellis2_4b
    python experiments/baselines/run_trellis2.py --image input.png --out-dir out/trellis2

Per input: ``<out-dir>/<id>/model.glb`` and ``<out-dir>/<id>/camera.json`` (no inference camera; the frame map of
eval/frame_maps for ``trellis2``: canonical, GLB axes (x, y, z) -> (x, -z, y)). Timing records as in
baseline_io.py. Runs in this package's environment; only the TRELLIS.2 checkout is added (README.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_io  # noqa: E402
from trellis2_pipeline import WEIGHTS, Trellis2Stock, default_source  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    baseline_io.add_common_arguments(ap)
    ap.add_argument("--trellis2-src", default=None,
                    help="TRELLIS.2 checkout at commit 75fbf018 (default: $FASTPIXAL3D_TRELLIS2_SRC or "
                         "third_party/TRELLIS.2)")
    ap.add_argument("--trellis2-weights", default=WEIGHTS,
                    help="TRELLIS.2-4B: local directory or Hugging Face id (default: microsoft/TRELLIS.2-4B)")
    return ap


def build(args):
    return [Trellis2Stock(source=args.trellis2_src or default_source(), weights=args.trellis2_weights)]


def main(argv=None) -> int:
    return baseline_io.main_for(build, parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
