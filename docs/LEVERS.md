# Speed levers

The deployed configuration turns every lever below on. Each lever is a boolean field of
`fastpixal3d.config.Levers` with the same name, and `scripts/infer.py --disable <name>` (repeatable)
switches it off. `Row` is the row of the report's cumulative measurement that adds the lever
(`experiments/configs/cumulative/<row>_*.yaml`; [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md)), and `Name in
the report` is how the report refers to it. The measurement starts from Pixal3D re-run (*Base (Pixal3D), re-run* in the report; row 00) and the
tiled mesh path with every lever off (row 01, Pixal3D's 12-step flows) and adds the levers in three
blocks: mesh post-processing (rows 02-08), flow engineering (rows 09-13) and the distilled students
(rows 14-18). The report's table of optimizations (Table 1) lists the texture projection reuse (row 15) with the
flow engineering and merges the two shape students (row 16) into one line. The two levers that act only
on the tiled CPU unwrap (`per_tile_single_thread_unwrap`, `pinned_bvh_transfers`) are switched on with the
row that replaces that unwrap (08); the report does not list them as steps.

Output classes:

* **identical**: scheduling, caching or launch configuration; designed to leave the output unchanged.
* **near-identical**: equal up to floating-point rounding (another attention kernel, another
  reduction order, bf16 weight folding, closest-face ties in the BVH).
* **output-changing**: changes the output by design (students, adaptive remesh, GPU charting).

| Lever (flag) | Row | Name in the report | Output class | Stage | What it does |
|---|---|---|---|---|---|
| `mesh_prep_worker_thread` | 02 | Mesh pre-processing on a worker thread | identical | mesh | Run mesh cleaning, remeshing, simplification and UV charting on a worker thread while the main thread conditions, samples and decodes the texture. |
| `fixed_bake_launch_config` | 03 | Fixed bake-kernel launch configuration | identical | bake | Launch the trilinear volume-sampling kernel of the texture bake with a fixed tile configuration instead of running the autotuner sweep. |
| `fast_gpu_simplify` | 04 | Faster GPU simplification and charting | near-identical | mesh | Mesh simplification whose round loop runs on the GPU without a host synchronisation per round (patched CuMesh build). |
| `gpu_bvh_build` | 05 | GPU BVH build | near-identical | mesh | Build the BVH used by remeshing and texture baking on the GPU (linear BVH, patched cubvh build) instead of on the host. |
| `adaptive_remesh` | 06 | Adaptive half-resolution remesh | output-changing | mesh | Remesh at half the decoder resolution unless the object is thin (many support tokens per unit of resolution squared). |
| `lean_glb_writer` | 07 | Lean GLB writer, parallel PNG | identical | export | Write the GLB container directly and encode the two PBR textures concurrently with a fast PNG encoder. |
| `gpu_chart_segmentation` | 08 | Single-pass GPU chart segmentation | output-changing | mesh | Segment UV charts in one pass on the GPU and pack them into a single atlas on the GPU (replaces the tiled CPU unwrapping). |
| `per_tile_single_thread_unwrap` | 08 | Per-tile single-thread CPU unwrapping | identical | mesh | One inner thread per tile in the CPU UV unwrapper of the tiled mesh path. No effect when gpu_chart_segmentation is on (no CPU unwrapping happens). |
| `pinned_bvh_transfers` | 08 | Pinned asynchronous BVH transfers | identical | mesh | Pinned, chunked host-to-device copies for the host-built BVH of the tiled mesh path, and a split and gather of its CPU unwrapper that compute each chart area once. No effect when gpu_chart_segmentation is on. |
| `fa3_texture_flow` | 09 | FlashAttention-3, texture flow | near-identical | texture | Use FlashAttention-3 kernels for the attention layers of the texture flow. |
| `texture_kv_reuse` | 10 | Texture K/V reuse across steps | identical | texture | Compute the cross-attention keys/values and the projected image features of the texture flow once per request and reuse them in every sampling step. |
| `shared_conditioning` | 11 | Shared image conditioning | identical | conditioning | Run the DINOv3 backbone once per image size, share the NAF guide encoding, and sample the projected features only at the occupied voxels. |
| `fa3_fine_shape_flow` | 12 | FlashAttention-3, fine-shape flow only | near-identical | shape | Use FlashAttention-3 kernels for the fine-shape flow only. |
| `texture_side_stream` | 13 | Side-stream texture sampling, early PNG | identical | texture | Sample the texture on its own CUDA stream so it overlaps the mesh work, start PNG encoding as soon as the textures exist, and defer allocator cache releases. |
| `structure_student` | 14 | Structure student: trajectory, 4 steps | output-changing | structure | Four-step trajectory-distilled student for the sparse structure (replaces 12 guided steps). |
| `texture_projection_reuse` | 15 | Texture projection reuse across steps | identical | texture | Cache the per-block context projections of the texture flow (and the conditioning cast) for the duration of one sampling call. |
| `coarse_shape_student` | 16 | Coarse-shape student: consistency, 1 step | output-changing | shape | One-step consistency-distilled student for the 512 shape flow. |
| `fine_shape_student` | 16 | Fine-shape student: consistency, 1 step | output-changing | shape | One-step consistency-distilled student for the 1024/1536 shape flow. |
| `fold_lora_fine_shape` | 17 | LoRA folded, fine-shape student | near-identical | shape | Fold the fine-shape student's LoRA into the bf16 base weights at load time. No effect when fine_shape_student is off (no adapter is loaded). |
| `texture_student` | 18 | Texture student: trajectory, 2 steps | output-changing | texture | Two-step trajectory-distilled student for the texture flow (replaces 12 steps). |

## Switching levers off

| lever | when off |
|---|---|
| `mesh_prep_worker_thread` | mesh cleaning, BVH, remeshing and simplification run on the main thread before the texture is sampled; the UV atlas is still built on a worker thread while the texture is sampled |
| `fixed_bake_launch_config` | FlexGEMM's autotuner chooses the launch configuration of the bake's volume-sampling kernel |
| `per_tile_single_thread_unwrap`, `pinned_bvh_transfers` | act only on the tiled mesh path (`gpu_chart_segmentation` off): each tile's xatlas runs its own thread pool; the host-built BVH is copied with plain synchronous copies and the CPU unwrap computes the chart areas per chart |
| `fa3_texture_flow`, `fa3_fine_shape_flow` | the flow runs on FlashAttention-2 (`fa3_fine_shape_flow` also applies to Pixal3D's fine-shape sampler when `fine_shape_student` is off) |
| `texture_kv_reuse`, `texture_projection_reuse` | keys, values and projections are recomputed in every sampling step |
| `fast_gpu_simplify` | the stock CuMesh simplification (one host synchronisation per round) |
| `gpu_bvh_build` | the BVH bundled with the stock CuMesh package (host build, copied to the GPU); on the tiled mesh path one of the host builds of `native/legacy` |
| `shared_conditioning` | each stage computes its image conditioning with Pixal3D's own code |
| `fold_lora_fine_shape` | the fine-shape adapter stays a separate low-rank branch evaluated in every forward pass (with `fine_shape_student` off the lever has nothing to act on) |
| `adaptive_remesh` | narrow-band remeshing always runs at the decoder resolution |
| `lean_glb_writer` | trimesh's GLB exporter with Pillow's PNG encoder |
| `gpu_chart_segmentation` | the tiled mesh path of rows 01-07 of the cumulative measurement (`fastpixal3d.mesh.tiled_path`): GPU chart clustering, then 16 tiles unwrapped by xatlas on a CPU thread pool while the texture is sampled. It needs the builds of `native/legacy` (see the README there for which configuration needs which) |
| `texture_side_stream` | the texture is sampled on the default stream, PNG encoding starts at export, allocator cache releases run immediately |
| `structure_student` | Pixal3D's sparse-structure sampler (12 steps) |
| `coarse_shape_student`, `fine_shape_student` | Pixal3D's shape samplers (12 steps) from the same initial noise |
| `texture_student` | Pixal3D's texture sampler (12 steps) |

The export is not a lever: it always uses the settings of Pixal3D's `inference.py` (1,000,000
faces, 4096 x 4096 textures), the setting of all comparisons with the baseline.

On GPUs other than Hopper (sm_90), the two FlashAttention-3 levers fall back to FlashAttention-2
(`--fa2` forces this); see [INSTALL.md](INSTALL.md#other-gpus).
