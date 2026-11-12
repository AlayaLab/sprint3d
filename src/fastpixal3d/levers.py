"""The speed levers of the deployed configuration.

Every lever has a plain name, which is also the name of its boolean flag in
:class:`fastpixal3d.config.Levers`. ``row`` is the row of the cumulative
measurement that adds the lever (``experiments/configs/cumulative/<row>_*.yaml``,
the order of the report: mesh post-processing in rows 01-08, flow engineering
in rows 09-13, distilled students in rows 14-18; row 01 is the tiled mesh path
with every lever off). The two levers that act only on the tiled CPU unwrap
enter with the row that replaces it (08). ``output`` is the output class used
in the report:

* ``identical``       -- designed to leave the output unchanged (scheduling,
                         caching, launch configuration);
* ``near-identical``  -- numerically equivalent up to floating-point
                         rounding (different attention kernel, different
                         reduction order, bf16 weight folding);
* ``output-changing`` -- changes the output by design (students, adaptive
                         remesh, GPU charting).

All levers are on in the deployed configuration and every lever can be switched
off. ``requires_legacy`` marks the levers that act only on the tiled mesh path
(``gpu_chart_segmentation`` off, :mod:`fastpixal3d.mesh.tiled_path`), whose
native builds are produced by ``native/legacy/build_legacy.sh``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

IDENTICAL = "identical"
NEAR_IDENTICAL = "near-identical"
OUTPUT_CHANGING = "output-changing"
OUTPUT_CLASSES = (IDENTICAL, NEAR_IDENTICAL, OUTPUT_CHANGING)


@dataclass(frozen=True)
class Lever:
    name: str
    row: str
    paper_name: str
    output: str
    stage: str
    description: str
    requires_legacy: bool = False


LEVERS: Tuple[Lever, ...] = (
    Lever("mesh_prep_worker_thread", "02", "Mesh pre-processing on a worker thread", IDENTICAL, "mesh",
          "Run mesh cleaning, remeshing, simplification and UV charting on a worker thread while the "
          "main thread conditions, samples and decodes the texture."),
    Lever("fixed_bake_launch_config", "03", "Fixed bake-kernel launch configuration", IDENTICAL, "bake",
          "Launch the trilinear volume-sampling kernel of the texture bake with a fixed tile "
          "configuration instead of running the autotuner sweep."),
    Lever("fast_gpu_simplify", "04", "Faster GPU simplification and charting", NEAR_IDENTICAL, "mesh",
          "Mesh simplification whose round loop runs on the GPU without a host synchronisation per "
          "round (patched CuMesh build)."),
    Lever("gpu_bvh_build", "05", "GPU BVH build", NEAR_IDENTICAL, "mesh",
          "Build the BVH used by remeshing and texture baking on the GPU (linear BVH, patched cubvh "
          "build) instead of on the host."),
    Lever("adaptive_remesh", "06", "Adaptive half-resolution remesh", OUTPUT_CHANGING, "mesh",
          "Remesh at half the decoder resolution unless the object is thin (many support tokens per "
          "unit of resolution squared)."),
    Lever("lean_glb_writer", "07", "Lean GLB writer, parallel PNG", IDENTICAL, "export",
          "Write the GLB container directly and encode the two PBR textures concurrently with a fast "
          "PNG encoder."),
    Lever("gpu_chart_segmentation", "08", "Single-pass GPU chart segmentation", OUTPUT_CHANGING, "mesh",
          "Segment UV charts in one pass on the GPU and pack them into a single atlas on the GPU "
          "(replaces the tiled CPU unwrapping)."),
    Lever("per_tile_single_thread_unwrap", "08", "Per-tile single-thread CPU unwrapping", IDENTICAL, "mesh",
          "One inner thread per tile in the CPU UV unwrapper of the tiled mesh path. No effect when "
          "gpu_chart_segmentation is on (no CPU unwrapping happens).", requires_legacy=True),
    Lever("pinned_bvh_transfers", "08", "Pinned asynchronous BVH transfers", IDENTICAL, "mesh",
          "Pinned, chunked host-to-device copies for the host-built BVH of the tiled mesh path, and a "
          "split and gather of its CPU unwrapper that compute each chart area once. No effect when "
          "gpu_chart_segmentation is on.", requires_legacy=True),
    Lever("fa3_texture_flow", "09", "FlashAttention-3, texture flow", NEAR_IDENTICAL, "texture",
          "Use FlashAttention-3 kernels for the attention layers of the texture flow."),
    Lever("texture_kv_reuse", "10", "Texture K/V reuse across steps", IDENTICAL, "texture",
          "Compute the cross-attention keys/values and the projected image features of the texture "
          "flow once per request and reuse them in every sampling step."),
    Lever("shared_conditioning", "11", "Shared image conditioning", IDENTICAL, "conditioning",
          "Run the DINOv3 backbone once per image size, share the NAF guide encoding, and sample the "
          "projected features only at the occupied voxels."),
    Lever("fa3_fine_shape_flow", "12", "FlashAttention-3, fine-shape flow only", NEAR_IDENTICAL, "shape",
          "Use FlashAttention-3 kernels for the fine-shape flow only."),
    Lever("texture_side_stream", "13", "Side-stream texture sampling, early PNG", IDENTICAL, "texture",
          "Sample the texture on its own CUDA stream so it overlaps the mesh work, start PNG encoding "
          "as soon as the textures exist, and defer allocator cache releases."),
    Lever("structure_student", "14", "Structure student: trajectory, 4 steps", OUTPUT_CHANGING, "structure",
          "Four-step trajectory-distilled student for the sparse structure (replaces 12 guided steps)."),
    Lever("texture_projection_reuse", "15", "Texture projection reuse across steps", IDENTICAL, "texture",
          "Cache the per-block context projections of the texture flow (and the conditioning cast) "
          "for the duration of one sampling call."),
    Lever("coarse_shape_student", "16", "Coarse-shape student: consistency, 1 step", OUTPUT_CHANGING, "shape",
          "One-step consistency-distilled student for the 512 shape flow."),
    Lever("fine_shape_student", "16", "Fine-shape student: consistency, 1 step", OUTPUT_CHANGING, "shape",
          "One-step consistency-distilled student for the 1024/1536 shape flow."),
    Lever("fold_lora_fine_shape", "17", "LoRA folded, fine-shape student", NEAR_IDENTICAL, "shape",
          "Fold the fine-shape student's LoRA into the bf16 base weights at load time. No effect when "
          "fine_shape_student is off (no adapter is loaded)."),
    Lever("texture_student", "18", "Texture student: trajectory, 2 steps", OUTPUT_CHANGING, "texture",
          "Two-step trajectory-distilled student for the texture flow (replaces 12 steps)."),
)

LEVERS_BY_NAME: Dict[str, Lever] = {lever.name: lever for lever in LEVERS}


def lever(name: str) -> Lever:
    try:
        return LEVERS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown lever {name!r}; known: {sorted(LEVERS_BY_NAME)}") from None


def markdown_table() -> str:
    """The lever table as Markdown (the body of docs/LEVERS.md)."""
    lines = ["| Lever (flag) | Row | Name in the report | Output class | Stage | What it does |",
             "|---|---|---|---|---|---|"]
    for lv in LEVERS:
        lines.append(f"| `{lv.name}` | {lv.row} | {lv.paper_name} | {lv.output} | {lv.stage} | {lv.description} |")
    return "\n".join(lines)
