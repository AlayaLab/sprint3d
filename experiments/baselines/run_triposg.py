#!/usr/bin/env python3
r"""TripoSG baseline (VAST-AI-Research/TripoSG @ fc5c4099): image(s) -> untextured GLB at the released defaults.

    python experiments/baselines/run_triposg.py --list inputs.jsonl --out-dir runs/triposg \
        --triposg-src third_party/TripoSG --triposg-weights weights/triposg --rmbg-weights briaai/RMBG-1.4

The request is the one of the checkout's scripts/inference_triposg.py: ``prepare_image`` of scripts/image_process.py
(the alpha channel when it is usable, otherwise RMBG-1.4 background removal; white background, crop to the object,
10 % padding), the pipeline in float16 with 50 steps, guidance 7.0 and a generator seeded with ``--seed`` (42), the
pipeline's own geometry extraction (flash decoder, octree depth 9, bounds 1.005), no face reduction (the
script's default ``--faces -1``), and trimesh's GLB export of the mesh.

Per input: ``<out-dir>/<id>/model.glb`` and ``<out-dir>/<id>/camera.json`` (no inference camera; the frame map of
eval/frame_maps for ``triposg``: camera-referenced, view aligned). Stages: ``preprocess``; ``generate`` (the
pipeline call) with the nested ``image_condition`` (encode_image) and ``geometry_extraction`` (latent decoding and
iso-surface extraction, timed around the pipeline module's extraction function while the call runs); ``export``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_io  # noqa: E402
from baseline_io import Method, Stages, parts, timed_calls  # noqa: E402

COMMIT = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
WEIGHTS = "VAST-AI/TripoSG"
WEIGHTS_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
RMBG = "briaai/RMBG-1.4"
RMBG_REVISION = "2ceba5a5efaec153162aedea169f76caf9b46cf8"
STEPS = 50
GUIDANCE = 7.0
DTYPE = "float16"


class TripoSG(Method):
    name = "triposg"
    arm = "stock"

    def __init__(self, source: str, weights: str = WEIGHTS, rmbg: str = RMBG):
        self.source = str(Path(source).expanduser().resolve())
        self.weights = weights
        self.rmbg_weights = rmbg
        self.pipe = self.rmbg = None

    def load(self) -> Dict[str, Any]:
        if not (Path(self.source) / "triposg" / "pipelines").is_dir():
            raise FileNotFoundError(f"{self.source} is not a TripoSG checkout (see experiments/baselines/README.md)")
        for p in (self.source, os.path.join(self.source, "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        import triposg.pipelines.pipeline_triposg as pipeline_module
        from briarmbg import BriaRMBG
        from image_process import prepare_image
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        self.torch, self.pipeline_module, self.prepare_image = torch, pipeline_module, prepare_image
        self.rmbg = BriaRMBG.from_pretrained(self.rmbg_weights).to("cuda").eval()
        self.pipe = TripoSGPipeline.from_pretrained(self.weights).to("cuda", getattr(torch, DTYPE))
        from trellis2_pipeline import checkout_commit

        commit = checkout_commit(self.source)
        if commit and commit != COMMIT:
            print(f"[triposg] warning: checkout at {commit}, the measured commit is {COMMIT}", flush=True)
        return {"triposg_commit": commit, "weights": self.weights, "rmbg": self.rmbg_weights}

    def settings(self) -> Dict[str, Any]:
        return {"num_inference_steps": STEPS, "guidance_scale": GUIDANCE, "dtype": DTYPE, "faces": -1,
                "geometry": "pipeline defaults (flash decoder, octree depth 9, bounds 1.005)",
                "preprocess": "alpha if usable, else RMBG-1.4; white background; crop; 10 % padding"}

    def run(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        import numpy as np
        import trimesh

        torch = self.torch
        with stages.stage("preprocess"):
            image = self.prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=self.rmbg)
        targets = [(self.pipe, "encode_image", "image_condition")]
        targets += [(self.pipeline_module, fn, "geometry_extraction")
                    for fn in ("flash_extract_geometry", "hierarchical_extract_geometry")
                    if hasattr(self.pipeline_module, fn)]
        with torch.no_grad(), stages.stage("generate"), timed_calls(stages, targets):
            output = self.pipe(image=image, generator=torch.Generator(device=self.pipe.device).manual_seed(seed),
                               num_inference_steps=STEPS, guidance_scale=GUIDANCE).samples[0]
        with stages.stage("export"):
            mesh = trimesh.Trimesh(output[0].astype(np.float32), np.ascontiguousarray(output[1]))
            mesh.export(str(out_dir / "model.glb"))
        return {"output_vertices": int(len(mesh.vertices)), "output_faces": int(len(mesh.faces))}

    def derived(self, timings: Dict[str, float]) -> Dict[str, float]:
        extraction = timings.get("geometry_extraction_s", 0.0)
        return parts(timings, sampling=timings.get("generate_s", 0.0) - extraction, decoding=extraction,
                     meshwork=timings.get("export_s", 0.0))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    baseline_io.add_common_arguments(ap)
    ap.add_argument("--triposg-src", default=os.environ.get("FASTPIXAL3D_TRIPOSG_SRC"), required=False,
                    help="TripoSG checkout at commit fc5c4099 (or $FASTPIXAL3D_TRIPOSG_SRC)")
    ap.add_argument("--triposg-weights", default=WEIGHTS, help="local directory or Hugging Face id of the weights")
    ap.add_argument("--rmbg-weights", default=RMBG, help="RMBG-1.4: local directory or Hugging Face id")
    return ap


def build(args):
    if not args.triposg_src:
        raise SystemExit("--triposg-src (or FASTPIXAL3D_TRIPOSG_SRC) is required")
    return [TripoSG(args.triposg_src, args.triposg_weights, args.rmbg_weights)]


def main(argv=None) -> int:
    return baseline_io.main_for(build, parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
