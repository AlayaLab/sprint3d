#!/usr/bin/env python3
r"""Hunyuan3D-2.1 baseline (Tencent-Hunyuan/Hunyuan3D-2.1 @ 82920d64): image(s) -> PBR-textured GLB. OPTIONAL.

Hunyuan3D-2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement, which does not apply in
the European Union, the United Kingdom and South Korea and sets further conditions (README.md). No file of
Hunyuan3D-2.1 is part of this repository: this adapter imports it from a checkout obtained separately.

    python experiments/baselines/run_hunyuan3d21.py --list inputs.jsonl --out-dir runs/hunyuan3d21 \
        --hunyuan-src /path/to/Hunyuan3D-2.1 --hy3dgen-models /path/to/hy3dgen \
        --realesrgan /path/to/RealESRGAN_x4plus.pth

The request is the one of the checkout's demo.py with the settings of its README:

  shape    ``Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")`` called on the image
           converted to RGBA (as demo.py does: inputs need an alpha channel) with a generator seeded with ``--seed``,
           at the defaults of its ``__call__`` (50 steps, guidance 5.0, octree resolution 384, box 1.01, 8000 chunks);
           the mesh is written as ``shape_untextured.glb`` (the geometry file of the frame map)
  texture  ``Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))`` with the RealESRGAN
           x4plus checkpoint and the multiview configuration and custom pipeline of the checkout, on
           ``shape_untextured.glb`` and the input image (its own defaults: remeshing to 40,000 faces, 2048 renders,
           a 4096 texture, back-sample baking, inpainting)
  export   the textured OBJ is converted to ``model.glb`` with Blender's Python module ``bpy`` when it can be
           imported (the repository's own conversion), and with trimesh otherwise (base colour only; the choice is
           recorded as ``obj_to_glb``). Without ``bpy``, an empty stand-in module lets the repository's renderer import.

Stages: ``shape`` (nested: ``shape_preprocess``, ``shape_condition``, ``shape_decode`` = VAE decoding, iso-surface
extraction and mesh conversion), ``shape_write``, ``paint`` (nested, timed around the paint pipeline's own calls
while it runs: ``paint_remesh``, ``paint_uv_wrap``, ``paint_load_mesh``, ``paint_view_selection``,
``paint_render_normals``, ``paint_render_positions``, ``paint_multiview_diffusion``, ``paint_super_resolution``,
``paint_bake``, ``paint_inpaint``, ``paint_set_texture``, ``paint_set_texture_mr``, ``paint_save_mesh``,
``paint_convert_glb``), ``export``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_io  # noqa: E402
from baseline_io import Method, Stages, parts, timed_calls  # noqa: E402

COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
WEIGHTS = "tencent/Hunyuan3D-2.1"
WEIGHTS_REVISION = "0b94677654c57bb9a6b6845cd7b704ccf551d327"
MAX_NUM_VIEW = 6
RESOLUTION = 512
# Parts of the report's table: which paint sub-steps sample networks (or render their conditioning) and which
# process the mesh and the texture.
PAINT_SAMPLING = ("paint_multiview_diffusion", "paint_super_resolution", "paint_render_normals",
                  "paint_render_positions", "paint_view_selection")
PAINT_MESHWORK = ("paint_remesh", "paint_uv_wrap", "paint_load_mesh", "paint_bake", "paint_inpaint",
                  "paint_set_texture", "paint_set_texture_mr", "paint_save_mesh", "paint_convert_glb")


class Hunyuan3D21(Method):
    name = "hunyuan3d21"
    arm = "stock"

    def __init__(self, source: str, realesrgan: str, weights: str = WEIGHTS):
        self.source = str(Path(source).expanduser().resolve())
        self.realesrgan = str(Path(realesrgan).expanduser().resolve())
        self.weights = weights
        self.shape_pipe = self.paint_pipe = None
        self.bpy = False

    def load(self) -> Dict[str, Any]:
        if not (Path(self.source) / "hy3dshape").is_dir() or not (Path(self.source) / "hy3dpaint").is_dir():
            raise FileNotFoundError(f"{self.source} is not a Hunyuan3D-2.1 checkout (see experiments/baselines/README.md)")
        if not Path(self.realesrgan).is_file():
            raise FileNotFoundError(f"RealESRGAN checkpoint {self.realesrgan} not found")
        for p in (self.source, os.path.join(self.source, "hy3dshape"), os.path.join(self.source, "hy3dpaint")):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from torchvision_fix import apply_fix   # as demo.py

            apply_fix()
        except Exception as error:  # demo.py continues without the fix as well
            print(f"[hunyuan3d21] torchvision fix not applied: {error!r}", flush=True)
        try:
            import bpy  # noqa: F401

            self.bpy = True
        except Exception:
            self.bpy = False
            sys.modules["bpy"] = types.ModuleType("bpy")
        import torch
        import textureGenPipeline
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        self.torch, self.paint_module = torch, textureGenPipeline
        self.shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.weights)
        config = Hunyuan3DPaintConfig(MAX_NUM_VIEW, RESOLUTION)
        config.realesrgan_ckpt_path = self.realesrgan
        config.multiview_cfg_path = os.path.join(self.source, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
        config.custom_pipeline = os.path.join(self.source, "hy3dpaint", "hunyuanpaintpbr")
        self.paint_config = config
        self.paint_pipe = Hunyuan3DPaintPipeline(config)
        from trellis2_pipeline import checkout_commit

        commit = checkout_commit(self.source)
        if commit and commit != COMMIT:
            print(f"[hunyuan3d21] warning: checkout at {commit}, the measured commit is {COMMIT}", flush=True)
        return {"hunyuan3d21_commit": commit, "weights": self.weights, "obj_to_glb": "bpy" if self.bpy else "trimesh",
                "hy3dgen_models": os.environ.get("HY3DGEN_MODELS")}

    def settings(self) -> Dict[str, Any]:
        c = self.paint_config if self.paint_pipe is not None else None
        out = {"shape": "Hunyuan3DDiTFlowMatchingPipeline.__call__ defaults (50 steps, guidance 5.0, octree "
                        "resolution 384, box 1.01, 8000 chunks)",
               "paint": {"max_num_view": MAX_NUM_VIEW, "resolution": RESOLUTION}}
        if c is not None:
            out["paint"].update(render_size=c.render_size, texture_size=c.texture_size, bake_mode=c.bake_mode,
                                raster_mode=c.raster_mode)
        return out

    def _paint_targets(self):
        pp, tg = self.paint_pipe, self.paint_module
        return [(tg, "remesh_mesh", "paint_remesh"), (tg, "mesh_uv_wrap", "paint_uv_wrap"),
                (tg, "convert_obj_to_glb", "paint_convert_glb"),
                (pp.render, "load_mesh", "paint_load_mesh"), (pp.render, "set_texture", "paint_set_texture"),
                (pp.render, "set_texture_mr", "paint_set_texture_mr"), (pp.render, "save_mesh", "paint_save_mesh"),
                (pp.view_processor, "bake_view_selection", "paint_view_selection"),
                (pp.view_processor, "render_normal_multiview", "paint_render_normals"),
                (pp.view_processor, "render_position_multiview", "paint_render_positions"),
                (pp.view_processor, "bake_from_multiview", "paint_bake"),
                (pp.view_processor, "texture_inpaint", "paint_inpaint"),
                (pp.models, "multiview_model", "paint_multiview_diffusion"),
                (pp.models, "super_model", "paint_super_resolution")]

    def run(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        import trimesh
        from PIL import Image

        torch = self.torch
        shape_glb = out_dir / "shape_untextured.glb"
        textured_obj = out_dir / "textured.obj"
        image = Image.open(image_path).convert("RGBA")
        shape_targets = [(self.shape_pipe, "prepare_image", "shape_preprocess"),
                         (self.shape_pipe, "encode_cond", "shape_condition"),
                         (self.shape_pipe, "_export", "shape_decode")]
        with stages.stage("shape"), timed_calls(stages, shape_targets):
            mesh = self.shape_pipe(image=image, generator=torch.Generator(device="cuda").manual_seed(seed))[0]
        with stages.stage("shape_write"):
            mesh.export(str(shape_glb))
        with stages.stage("paint"), timed_calls(stages, self._paint_targets()):
            out_obj = self.paint_pipe(mesh_path=str(shape_glb), image_path=image_path,
                                      output_mesh_path=str(textured_obj), save_glb=self.bpy)
        with stages.stage("export"):
            converted = textured_obj.with_suffix(".glb")
            if self.bpy and converted.is_file():
                shutil.copyfile(converted, out_dir / "model.glb")
            else:
                trimesh.load(out_obj, force="scene").export(str(out_dir / "model.glb"))
        return {"shape_faces": int(len(mesh.faces)), "obj_to_glb": "bpy" if self.bpy else "trimesh"}

    def derived(self, timings: Dict[str, float]) -> Dict[str, float]:
        g = lambda name: timings.get(f"{name}_s", 0.0)  # noqa: E731
        shape_sampling = g("shape") - g("shape_preprocess") - g("shape_condition") - g("shape_decode")
        sampling = g("shape_condition") + shape_sampling + sum(g(n) for n in PAINT_SAMPLING)
        meshwork = g("export") + sum(g(n) for n in PAINT_MESHWORK)
        return parts(timings, sampling=sampling, decoding=g("shape_decode") + g("shape_write"), meshwork=meshwork)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    baseline_io.add_common_arguments(ap)
    ap.add_argument("--hunyuan-src", default=os.environ.get("FASTPIXAL3D_HUNYUAN3D21_SRC"),
                    help="Hunyuan3D-2.1 checkout at commit 82920d64 (or $FASTPIXAL3D_HUNYUAN3D21_SRC)")
    ap.add_argument("--hy3dgen-models", default=None,
                    help="directory with tencent/Hunyuan3D-2.1 below it (HY3DGEN_MODELS of the repository)")
    ap.add_argument("--realesrgan", required=False, default=None, help="RealESRGAN_x4plus.pth")
    return ap


def build(args):
    if not args.hunyuan_src:
        raise SystemExit("--hunyuan-src (or FASTPIXAL3D_HUNYUAN3D21_SRC) is required")
    if not args.realesrgan:
        raise SystemExit("--realesrgan is required")
    if args.hy3dgen_models:
        os.environ["HY3DGEN_MODELS"] = str(Path(args.hy3dgen_models).expanduser().resolve())
    return [Hunyuan3D21(args.hunyuan_src, args.realesrgan)]


def main(argv=None) -> int:
    return baseline_io.main_for(build, parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
