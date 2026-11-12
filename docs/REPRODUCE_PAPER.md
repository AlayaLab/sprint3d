# Reproducing the report's experiments

Each section maps rows and numbers of the report to the commands that produce them. Installation:
[INSTALL.md](INSTALL.md); timing protocol: [TIMING.md](TIMING.md); evaluation: [EVALUATION.md](EVALUATION.md).

The report calls Pixal3D (official) *Base (Pixal3D)* and the Pixal3D re-run *Base (Pixal3D), re-run*; the tables below
and the `# Paper row:` lines of the configurations keep the shorter names.

## Cumulative measurement and ablations

The speed analysis of the report rests on three sets of configurations, all at the official export
(1,000,000 faces, 4096 x 4096 texture):

| set | configurations | inputs | what it gives |
|---|---|---|---|
| cumulative measurement | `experiments/configs/cumulative/` (19 rows and 3 controls) | the 80 timing inputs | the waterfall figure and its three factors (Table 2), the table of every step (Table 8), the table of optimizations (Table 1), the stage differences (Appendix A) |
| one-lever-off ablations | `experiments/configs/ablations/` (Ours and 4 rows) | the 80 timing inputs | the one-at-a-time removals (Appendix A) |
| same-step teachers | `experiments/configs/same_step_teacher/` (Ours and 3 rows) | the 400 validation inputs | the second block of Table 5 (time here, quality with `eval/`) |

Each configuration file names the report's row in its first line (`# Paper row: ...`) and sets the lever
flags (`fastpixal3d.config.Levers`, docs/LEVERS.md) and, for the same-step teachers, the sampling settings
(`fastpixal3d.config.Sampling`).

### Rows of the cumulative measurement

The files `00_*` to `18_*` are the chain, in the order of the report: each row keeps every lever of the
row above and adds the levers listed under `adds` (`levers` lists the full set). Row 00 is Pixal3D re-run;
row 01 replaces its export by the tiled mesh path with every lever off; the other rows form three blocks
(the `block` key), which the report summarises by one factor each:

- mesh post-processing, rows 01-08 (`post_processing`), ending at the row the report calls "mesh path
  rebuilt";
- flow engineering, rows 09-13 (`flow_engineering`), ending at "+ flow optimizations";
- distillation, rows 14-18 (`distillation`), ending at Ours, the deployed configuration.

Rows 00-13 sample all four flows with Pixal3D's released 12-step samplers; no student is loaded. Rows
14-18 add the students: the trajectory-distilled structure student (row 14; `students.json` entry
`structure`, with its static CUDA graph), the coarse- and fine-shape students (row 16, one step each),
the folded fine-shape adapter (row 17) and the texture student (row 18). Later steps supersede earlier
ones: from the GPU BVH build (row 05) on no host BVH is built, and from the single-pass GPU chart
segmentation (row 08) on the 16-tile CPU unwrap of rows 01-07 no longer runs. Row 08 also switches on the
two levers that act only on that unwrap (`per_tile_single_thread_unwrap`, `pinned_bvh_transfers`), so
that row 18 is the deployed configuration; the report does not list them as steps. The texture projection
reuse (row 15) is listed with the flow engineering in the report's table of optimizations (Table 1); the
measurement code could only switch it on together with the structure student, so the measurement adds it
right after that student.

The released flows cannot simply be run with one step instead of the students: with one step in each of
the four released flows, the mesh extraction runs out of GPU memory (CuMesh reports a CUDA out-of-memory
error), which is why the report's figure of quality against time samples the released flows of row 13
with 12, 8, 4 and 2 steps only. Those runs are not part of this set.

| file | block | report row | report median (s) |
|---|---|---|---|
| `00_pixal3d_rerun` | | Pixal3D re-run | 138.4 |
| `01_tiled_mesh_path` | post-processing | + tiled mesh path (16-tile atlas) | 55.3 |
| `02_mesh_prep_worker_thread` | post-processing | + mesh pre-processing on a worker thread | 53.5 |
| `03_fixed_bake_launch_config` | post-processing | + fixed bake-kernel launch configuration | 52.8 |
| `04_fast_gpu_simplify` | post-processing | + faster GPU simplification and charting | 52.4 |
| `05_gpu_bvh_build` | post-processing | + GPU BVH build | 52.6 |
| `06_adaptive_remesh` | post-processing | + adaptive half-resolution remesh | 55.5 |
| `07_lean_glb_writer` | post-processing | + lean GLB writer, parallel PNG | 52.4 |
| `08_gpu_chart_segmentation` | post-processing | + single-pass GPU chart segmentation | 19.0 |
| `09_fa3_texture_flow` | flow engineering | + FlashAttention-3, texture flow | 17.9 |
| `10_texture_kv_reuse` | flow engineering | + texture K/V reuse across steps | 17.7 |
| `11_shared_conditioning` | flow engineering | + shared image conditioning | 17.3 |
| `12_fa3_fine_shape_flow` | flow engineering | + FlashAttention-3, fine-shape flow only | 15.6 |
| `13_texture_side_stream` | flow engineering | + side-stream texture sampling, early PNG | 15.4 |
| `14_structure_student` | distillation | + structure student: trajectory, 4 steps | 15.2 |
| `15_texture_projection_reuse` | distillation | + texture projection reuse across steps | 14.9 |
| `16_shape_students` | distillation | + shape students: consistency, 1 step each | 7.23 |
| `17_fold_lora_fine_shape` | distillation | + LoRA folded, fine-shape student | 7.07 |
| `18_ours` | distillation | Ours (+ texture student: trajectory, 2 steps) | 4.00 |

The report's factors of the three blocks are 7.26x (row 00 to row 08), 1.24x (row 08 to row 13) and
3.84x (row 13 to row 18), each the ratio of the two medians over the 80 inputs.

The three controls (files `x1_*` to `x3_*`, key `control_of`) are not steps of the chain; each is compared
with its own row only, and the analyses below report the effect of the levers in which they differ
(the row with the levers minus the row without them):

| file | compared with | differs by | what it shows |
|---|---|---|---|
| `x1_without_superseded_levers` | row 08 | without `per_tile_single_thread_unwrap` and `pinned_bvh_transfers` | the two levers have nothing to act on once the GPU chart segmentation replaces the tiled unwrap: medians 19.05 s (row 08) and 19.03 s (x1), mean difference +0.2 s [+0.05, +0.49] |
| `x2_fold_lora_with_released_fine_shape_flow` | row 13 | with `fold_lora_fine_shape` | the fold has nothing to act on under the released 12-step fine-shape flow: medians 15.41 s (x2) and 15.39 s (row 13) |
| `x3_without_adaptive_remesh` | row 08 | without `adaptive_remesh` | the effect of the adaptive half-resolution remesh in the final mesh path, which the report measures apart from its position in the chain (row 06) |

The numbers of x1 and x2 were cut from the current report; the configurations stay here because they are part of the
parity record ([PARITY.md](PARITY.md)).

### Requirements

- The installation of [INSTALL.md](INSTALL.md), including `native/build_native.sh`, and for rows 01-04 of
  the cumulative set the builds of `native/legacy/build_legacy.sh` (`native/legacy/README.md` lists which
  row needs which build).
- The student weights (`students.json` with the entries `structure`, `coarse_shape`, `fine_shape` and
  `texture`; docs/WEIGHTS.md) for rows 14-18, the ablations and the same-step teachers.
- An input list as in [TIMING.md](TIMING.md): JSON lines with `id` and `path` in a fixed order. Name the
  inputs `<object id>_v<view>` (e.g. `a89c72706302_v00`): the analyses resample the views of one object
  together where the report does. The 80 timing inputs are `eval/lists/timing_80.csv`, the 400 validation
  inputs `eval/lists/validation_400.csv` (images: docs/DATA.md).
- One GPU per process with nothing else running on it (protocol of TIMING.md); the report used one Hopper GPU
  (141 GB) and 15 CPU cores. Rows 01-07 build the UV atlas with 16 xatlas atlases on a CPU thread pool, so on
  fewer cores these rows get longer (the UV branch, not the GPU stages).
- Time: about 21 GPU-hours for the cumulative set on one Hopper GPU (Pixal3D re-run about 4.4 h, each of
  rows 01-07 about 1.3-1.5 h, every later row and control at most 0.7 h), 1 h for the ablations. The rows can
  be split over several machines; keep each pair of adjacent rows, and each control with its row, on one
  machine (run a boundary row on both).

### Commands

`experiments/run_arms.py` runs the configurations one after the other, each in its own process through
`scripts/time_e2e.py` (same inputs, models loaded once and not timed, one untimed warm-up request,
per-stage timers, machine load record), and
writes one directory per configuration plus `arms.json`. A finished configuration is skipped when the
command is repeated, so an interrupted list resumes where it stopped.

```bash
STUDENTS=/path/to/students          # directory with students.json
SEED=/path/to/autotune_cache.json   # optional FlexGEMM autotune cache, copied into every run

# cumulative measurement: 22 processes (the 19 rows, then the 3 controls)
python experiments/run_arms.py experiments/configs/cumulative --list timing_inputs.jsonl \
    --out-dir runs/cumulative --students $STUDENTS --autotune-seed $SEED
# or a part of it on one machine, e.g. row 08 with its two controls:
#   ... --only gpu_chart_segmentation,without_superseded_levers,without_adaptive_remesh
python experiments/analysis/waterfall.py   experiments/configs/cumulative --runs runs/cumulative --out results/cumulative --figure
python experiments/analysis/lever_table.py experiments/configs/cumulative --runs runs/cumulative --out results/cumulative
python experiments/analysis/stage_deltas.py experiments/configs/cumulative --runs runs/cumulative --out results/cumulative \
    --unit object

# one-lever-off ablations: 5 processes
python experiments/run_arms.py experiments/configs/ablations --list timing_inputs.jsonl \
    --out-dir runs/ablations --students $STUDENTS --autotune-seed $SEED
python experiments/analysis/paired_table.py experiments/configs/ablations --runs runs/ablations --out results/ablations

# same-step teachers on the 400 validation inputs: 4 processes; quality with the evaluation suite
python experiments/run_arms.py experiments/configs/same_step_teacher --list validation_inputs.jsonl \
    --out-dir runs/same_step_teacher --students $STUDENTS --autotune-seed $SEED
python experiments/analysis/paired_table.py experiments/configs/same_step_teacher --runs runs/same_step_teacher \
    --out results/same_step_teacher
python eval/make_manifest.py --dir runs/same_step_teacher/01_structure_4_steps --arm structure_4_steps \
    --out manifests/structure_4_steps.json        # likewise for the other rows; then eval/run_eval.py (docs/EVALUATION.md)
```

`--only NAME[,NAME...]` runs a subset; arguments after `--` go to every `time_e2e.py` call. Every analysis
script writes CSV and JSON next to each other; the bootstrap settings are those of TIMING.md (10,000
resamples, seed 0, percentile intervals). The analyses read the chain and the controls from the
configuration directory: `waterfall.py` draws the 19 rows and the factors of the three blocks,
`lever_table.py` lists one line per step (`kind = step`) and per control (`kind = control`), and
`stage_deltas.py` compares adjacent rows, the blocks (`kind = block`), the whole chain (`kind = total`) and
the controls (`kind = control`).

### Report items and where they come from

| report item | configurations | script | output (columns) |
|---|---|---|---|
| Waterfall figure (cumulative measurement at the official export) | `cumulative/00-18` | `waterfall.py --figure` | `waterfall.png`, `waterfall.pdf`; bars from `waterfall.csv` (`*_median` per critical-path stage, `e2e_median`) |
| Factors of the three blocks (waterfall figure, Table 2) and their product | `cumulative/00, 08, 13, 18` | `waterfall.py` | `waterfall_blocks.csv` (`kind = block` / `total`: `factor`, `factor_ci_lo`, `factor_ci_hi`; objects resampled) |
| Table 8 (every step): E2E median and p90 | `cumulative/00-18` | `waterfall.py` | `waterfall.csv` (`e2e_median`, `e2e_p90`); quality columns: `eval/run_eval.py` on the GLBs of the same run directories |
| Table 1 (optimizations): effect (s), class | `cumulative/01-18` | `lever_table.py` | `levers.csv` (`kind = step`: `delta_mean`, `delta_ci_lo`, `delta_ci_hi`, `output_class`, `block`); the report lists row 15 with the flow engineering |
| Controls: superseded levers, LoRA fold under the 12-step flow (both no longer printed), adaptive remesh in the final mesh path (Section 3.2) | `cumulative/08, 13, x1-x3` | `lever_table.py` | `levers.csv` (`kind = control`) |
| Texture-flow levers measured on the texture stage (no longer printed) | `cumulative/09, 10, 15` | `lever_table.py` | `levers.csv` (`stage_delta_mean`, key `tex_rollout_s`) |
| Appendix A, paragraph "Cumulative measurement": end-to-end and per-stage differences | `cumulative/00, 08, 13, 18` | `stage_deltas.py --unit object` | `stage_deltas.csv` (`kind = block` / `total`) |
| Appendix A, paragraph "One-at-a-time removal" | `ablations/*` | `paired_table.py` | `paired.csv` (`e2e_delta_median`, `texture_sampling_delta_median`) |
| Table 5, second block: E2E | `same_step_teacher/*` | `paired_table.py` | `paired.csv` (`e2e_median`) |
| Table 5, second block: quality metrics | `same_step_teacher/*` | `eval/run_eval.py` | docs/EVALUATION.md |

### Differences between these configurations and the measurement in the report

- Processes. In the report, each of eight machines ran every row of the chain and the controls x1 and x2 in
  one process on the same ten inputs, with the order of the configurations rotated from input to input;
  x3 ran in a separate run together with a second measurement of row 08. `run_arms.py` runs one process per
  configuration.
- Pixal3D re-run. The report's row runs Pixal3D's flows and export inside the measurement code; the
  configuration here runs Pixal3D's inference script unmodified (as the Base (Pixal3D) row of Table 4,
  133.0 s in the dedicated timing run; the report finds the two equal within run-to-run variation).
- Inputs in the statistics. As in the report's cumulative measurement and one-at-a-time removals, `run_arms.py` and the
  analyses count every input of the list (each process warms up with one untimed request first). The
  dedicated timing run of TIMING.md leaves out the first 3 inputs instead (`scripts/time_e2e.py` and
  `scripts/analyze_timing.py` default to `--exclude-first 3`); pass `--exclude-first 3` to `run_arms.py`
  and to the analyses for that convention.

## TRELLIS.2 port

Report: the table "The rebuilt mesh path in TRELLIS.2" and the paragraph "A second pipeline". TRELLIS.2-4B runs
with its released weights, samplers, seed, pipeline type and export targets (1,000,000 faces, 4096 x 4096 texture);
only the post-processing after the flows is replaced by this package's mesh path. Code:
`experiments/trellis2_mesh_path/` (TRELLIS.2 itself at commit `75fbf018`, unmodified; installation in
[experiments/baselines/README.md](../experiments/baselines/README.md)).

### How the port is built

`mesh_path.py` is a wrapper, not a patch. It imports TRELLIS.2 from an unmodified checkout and reassigns nothing in
it. The request calls the pipeline's public stage methods (`preprocess_image`, `get_cond`, `sample_sparse_structure`,
`sample_shape_slat_cascade`, `decode_shape_slat`, `sample_tex_slat`, `decode_tex_slat`) in the order of its `run()`,
with the same arguments and the same seed position, and decodes the shape before the texture flow (the decoding
depends only on the shape latent). The mesh work then runs on a worker thread while the texture flow samples. The
output is built by `fastpixal3d.mesh` and written by `fastpixal3d.export.glb` in TRELLIS.2's own export frame
(`export_glb(..., rotation=None)`). A patch series would have to change `run()` in TRELLIS.2 and `to_glb` in
o-voxel; the wrapper needs neither. For every input the scripts record the sha256 of the shape and texture latents,
which are equal in all arms. Stage times come from a subclass of the pipeline class whose stage methods time the
inherited ones (`experiments/baselines/trellis2_pipeline.py`).

| arm | levers ([LEVERS.md](LEVERS.md)) | report row |
|---|---|---|
| `stock` | none: TRELLIS.2's `run()`, `mesh.simplify(16777216)`, `o_voxel.postprocess.to_glb`, trimesh export | TRELLIS.2 (official) |
| `mesh_path` | `mesh_prep_worker_thread`, `fixed_bake_launch_config`, `fast_gpu_simplify`, `gpu_bvh_build`, `gpu_chart_segmentation`, `lean_glb_writer` | + rebuilt mesh path |
| `mesh_path_adaptive_remesh` | the same + `adaptive_remesh` (output-changing) | + adaptive half-resolution remesh |
| `mesh_path_serial` | `mesh_path` without `mesh_prep_worker_thread`: `run()` unchanged, then the mesh path on the main thread | text: the overlap with the texture flow gains nothing in TRELLIS.2 |

The measured mesh path also had `per_tile_single_thread_unwrap` and `pinned_bvh_transfers` switched on; they act on
code paths that the GPU chart segmentation and the GPU BVH replace, so they have no counterpart here.

### Commands

```bash
# the 80 timing inputs (40 objects x 2 views), in the order of the list
python experiments/baselines/make_input_list.py --csv eval/lists/timing_80.csv --config eval/config.yaml \
    --out inputs/timing_80.jsonl

# generation and timing: TRELLIS.2 as released, then the three mesh-path arms in one process
python experiments/trellis2_mesh_path/time_trellis2.py --list inputs/timing_80.jsonl --out-dir runs/trellis2_stock \
    --arms stock --trellis2-src third_party/TRELLIS.2 --trellis2-weights weights/trellis2_4b
python experiments/trellis2_mesh_path/time_trellis2.py --list inputs/timing_80.jsonl --out-dir runs/trellis2_mesh \
    --arms mesh_path,mesh_path_adaptive_remesh,mesh_path_serial \
    --trellis2-src third_party/TRELLIS.2 --trellis2-weights weights/trellis2_4b

# quality: one evaluation per arm (every arm is evaluated as method "trellis2", with TRELLIS.2's frame map)
for arm in stock mesh_path mesh_path_adaptive_remesh; do
  dir=runs/trellis2_mesh/$arm; [ $arm = stock ] && dir=runs/trellis2_stock
  python experiments/baselines/make_baseline_manifest.py --dir $dir --method trellis2 \
      --list inputs/timing_80.jsonl --out manifests/trellis2_$arm.json
  python eval/run_eval.py --config eval/config.yaml --baseline-manifest manifests/trellis2_$arm.json \
      --out results/trellis2_$arm --skip vs_reference,perception
done

# base colour: every arm against TRELLIS.2 as released, and the mesh path against its serial export
python experiments/trellis2_mesh_path/colour_metrics.py --config eval/config.yaml --list inputs/timing_80.jsonl \
    --reference stock=runs/trellis2_stock --run mesh_path=runs/trellis2_mesh/mesh_path \
    --run mesh_path_adaptive_remesh=runs/trellis2_mesh/mesh_path_adaptive_remesh \
    --run mesh_path_serial=runs/trellis2_mesh/mesh_path_serial --pair mesh_path=mesh_path_serial \
    --out results/trellis2_colour.json

# the table
python experiments/trellis2_mesh_path/summarize_trellis2.py \
    --run stock=runs/trellis2_stock --run mesh_path=runs/trellis2_mesh/mesh_path \
    --run mesh_path_adaptive_remesh=runs/trellis2_mesh/mesh_path_adaptive_remesh \
    --serial runs/trellis2_mesh/mesh_path_serial \
    --eval stock=results/trellis2_stock --eval mesh_path=results/trellis2_mesh_path \
    --eval mesh_path_adaptive_remesh=results/trellis2_mesh_path_adaptive_remesh \
    --colour results/trellis2_colour.json \
    --out results/trellis2_table.json --markdown results/trellis2_table.md
```

### Numbers of the report and where they come from

Table 3 and the text of Section 4.2 print the medians, the speed-ups, the differences and the two agreement values. The
means, the p90 values, the ratio of the generation stages and the 1.4 % of the adaptive remesh come from the same runs
but are not printed.

| report | value | output of `summarize_trellis2.py` |
|---|---|---|
| End-to-end time, TRELLIS.2 (official): median / mean / p90 | 112.7 / 137.2 / 234.0 s | `runs.stock` median, mean, p90 |
| + rebuilt mesh path | 16.9 / 19.3 / 34.4 s | `runs.mesh_path` |
| + adaptive half-resolution remesh | 16.5 / 19.5 / 34.8 s | `runs.mesh_path_adaptive_remesh` |
| Post-processing (median) | 91.4 / 2.2 / 2.1 s | `post_processing_median` |
| Speed-up, ratio of medians, 95 % CI | 6.7x [5.4, 8.6]; 6.8x [5.4, 8.5] | `paired.<arm>.speedup_ratio_of_medians(_ci95)` |
| Generation stages, per-input ratio | 0.99 [0.97, 1.00] | `paired.mesh_path.flows_ratio_median(_ci95)` |
| Conservative estimate | 5.7x [4.7, 6.9] | `paired.mesh_path.conservative_speedup(_ci95)` (needs `--serial`) |
| Adaptive remesh, median of per-input ratios | 1.4 % faster | `speedup_median_of_ratios` of the two mesh-path arms against each other (`--run mesh_path=... --run mesh_path_adaptive_remesh=...`) |
| Delta normal map, PSNR (full image) and mean angular error: + rebuilt mesh path | +0.00 [-0.00, +0.01] dB; +0.00 [-0.01, +0.02] deg | `quality.mesh_path.psnr_full`, `quality.mesh_path.ang_mean` |
| the same, + adaptive half-resolution remesh | -0.07 [-0.18, +0.00] dB; +0.02 [-0.07, +0.11] deg | `quality.mesh_path_adaptive_remesh.*` |
| Delta color: base colour rendered at the input view, PSNR against the input image, paired difference to TRELLIS.2 (official) | -0.00 [-0.02, +0.01] dB; -0.06 [-0.09, -0.04] dB | `colour.delta_vs_input.<arm>.tex_psnr` |
| Text: base-colour renders of the mesh path and of TRELLIS.2 (official) agree at | 42.7 dB | `colour.agreement.mesh_path__vs__stock.tex_psnr` |
| Text: two exports of the same latents through the mesh path agree at | 43.1 dB | `colour.agreement.mesh_path__vs__mesh_path_serial.tex_psnr` |

Definitions used by the table:

- End-to-end time = `stage_sum_s`: the sum of the device-synchronised stages from preprocessing to the written GLB,
  model loading excluded (the timing lines also hold `e2e_s`, the wall clock of TIMING.md). Post-processing =
  `stage_sum_s` minus preprocessing minus image conditioning and the three flows: everything after the flows on the
  critical path.
- All inputs count (`--exclude-first 0`, the default of `summarize_trellis2.py`), and no warm-up request precedes
  the list (`--warmup-requests 0`, the default of `time_trellis2.py`).
- Confidence intervals: 95 % percentile intervals of a bootstrap over the 40 objects (both views of an object
  resampled together; 10,000 resamples, seed 20260918). The quality deltas are the paired per-object differences of
  `eval/metrics/stats.py` (mean over objects, bootstrap over objects).
- Conservative estimate: per input, the reference's own preprocessing and flow times plus the arm's post-processing
  beyond an uncontended texture flow (the serial arm's texture-flow time on the same input), so that no difference
  on the generation side is credited to the mesh path.
- Base colour (`colour_metrics.py`): every product is placed with the frame map of `trellis2` and the own-bounding-box
  normalisation of TRELLIS.2 (official)'s product on the same input (the other products get the same transform), and
  its unlit base colour is rendered with the renderer of `eval/metrics/render.py` at 512 x 512 on white. Delta color
  compares the input-view render with the input image composited on white (PSNR over the full image), as a paired
  difference per object; the agreement is the PSNR between two products averaged over five views (the input view,
  the input view turned by 90, 180 and 270 degrees about the vertical axis, and a view from 60 degrees below), then
  over objects. The input is a shaded render and the products are unlit, so the absolute values against the input
  image are not meaningful; only the paired differences are. LPIPS needs the AlexNet weights of
  `eval/fetch_metric_models.py --only table1` (`models.dir` of the configuration).
- Agreement with the measurement code: applied to the stored products of one input of the report's run (TRELLIS.2
  (official) and the three mesh-path arms), `colour_metrics.py` reproduces every stored value of that input (placement,
  face counts, and PSNR, SSIM, LPIPS and mask IoU of every view and pair); applied to the stored per-input values of
  all 80 inputs, `summarize_trellis2.py` reproduces the Delta color values and the 42.7 / 43.1 dB above.

How the report's run was set up: one Hopper GPU (sm_90, 141 GB) and 15 CPU cores per process, `OMP_NUM_THREADS=8`,
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` for every arm (the defaults of these scripts); the stock arm ran in three
processes on three machines, each input once; the three mesh-path arms ran in one process with the arm order
rotated per input (the default of `time_trellis2.py` for several arms); the outputs were written to a
network-mounted directory inside the timed window. The stock export runs its UV unwrapping and parts of its mesh
cleanup on the CPU, so its time depends on the CPU resources of the machine more than the mesh path does. The mesh
path of the report ran the measurement code's export path; `fastpixal3d.mesh` is its rewrite for this package
(agreement criteria for Pixal3D: [PARITY.md](PARITY.md)).

## Baselines

The report runs TRELLIS.2, TripoSG and Hunyuan3D-2.1 at their released defaults. Adapters, pinned commits, weights,
installation and licences: [experiments/baselines/README.md](../experiments/baselines/README.md). Hunyuan3D-2.1 is
optional (its licence does not apply in the European Union, the United Kingdom and South Korea); its adapter is
code of this repository and no file of Hunyuan3D-2.1 is included.

### Table of the released pipelines (Table 9)

| report row | median / mean / p90 (s) | sampling, median s (mean share) | post-processing, median s (mean share) |
|---|---|---|---|
| TRELLIS.2 (1M faces, 4096^2 texture) | 105.3 / 128.5 / 197.7 | 14.3 (16 %) | 85.8 (84 %) |
| Hunyuan3D-2.1 (40k faces, 2048^2 texture) | 62.2 / 64.1 / 81.8 | 13.5 (22 %) | 48.4 (78 %) |
| TripoSG (1.2M faces, no texture) | 7.9 / 7.8 / 10.6 | 2.7 (37 %) | 4.9 (60 %) |

The row of Base (Pixal3D) in the same table comes from the dedicated timing run
([TIMING.md](TIMING.md)).

```bash
python experiments/baselines/make_input_list.py --csv eval/lists/timing_80.csv --config eval/config.yaml \
    --out inputs/timing_80.jsonl
python experiments/baselines/run_trellis2.py --list inputs/timing_80.jsonl --out-dir runs/trellis2 \
    --trellis2-src third_party/TRELLIS.2 --trellis2-weights weights/trellis2_4b
envs/triposg/bin/python experiments/baselines/run_triposg.py --list inputs/timing_80.jsonl --out-dir runs/triposg \
    --triposg-src third_party/TripoSG --triposg-weights weights/triposg
envs/hunyuan3d21/bin/python experiments/baselines/run_hunyuan3d21.py --list inputs/timing_80.jsonl \
    --out-dir runs/hunyuan3d21 --hunyuan-src /path/to/Hunyuan3D-2.1 --hy3dgen-models /path/to/hy3dgen \
    --realesrgan /path/to/RealESRGAN_x4plus.pth
python experiments/baselines/summarize_timing.py --run TRELLIS.2=runs/trellis2 \
    --run Hunyuan3D-2.1=runs/hunyuan3d21 --run TripoSG=runs/triposg --exclude-first 3 \
    --markdown results/baselines_timing.md
```

- The statistics cover the 77 measured inputs: the 80 timing inputs without the first 3 of the list, which are the
  warm-up inputs of the dedicated timing run.
- End-to-end time = `stage_sum_s` (model loading excluded); medians over the inputs. Shares are the means of the
  per-input shares, which is why they need not match the ratio of the medians.
- Sampling = image conditioning and flow or diffusion sampling. For Hunyuan3D-2.1 it also contains the multi-view
  diffusion and super-resolution of the texture stage and the normal and position renders and view selection that
  condition the diffusion. Post-processing = latent decoding and iso-surface extraction plus remeshing,
  simplification, UV unwrapping, baking, inpainting and export. The rest is input preparation.
  - TripoSG: sampling = the pipeline call minus its geometry extraction.
  - Hunyuan3D-2.1, shape stage: sampling = the call minus preprocessing, conditioning and decoding (the report took
    the time of the diffusion loop itself; the difference is the setup of the latents and time steps).
- In the report these times were recorded while generating the baseline outputs for the validation set (the 400
  inputs of `eval/lists/validation_400.csv` and the 54 further timing inputs), on several machines of the same type.
  Each process loaded its model once and ran one untimed warm-up generation (`--warmup-requests 1`, the default of
  these scripts). Outputs were written to the local disk.
- The TRELLIS.2 row here (105.3 s) and the TRELLIS.2 (official) row of the TRELLIS.2 port (112.7 s) come from
  different runs: other machines, 77 or 80 inputs.

### Quality of the baselines (Tables 4 and 10)

The report prints the baseline rows of Tables 4 and 10; `eval/make_table.py` prints them as "not reported". The
commands below use the validation set; the evaluation code handles external baselines: it
renders them with the ground-truth camera pose after a fixed per-method frame map
(`eval/frame_maps/baseline_frame_maps.json`) and normalises every mesh by its own bounding box
([EVALUATION.md](EVALUATION.md), `eval/metrics/baselines.py`). To compute their values:

```bash
python experiments/baselines/make_input_list.py --csv eval/lists/validation_400.csv --config eval/config.yaml \
    --out inputs/validation_400.jsonl
python experiments/baselines/run_trellis2.py --list inputs/validation_400.jsonl --out-dir runs/val_trellis2 \
    --trellis2-src third_party/TRELLIS.2 --trellis2-weights weights/trellis2_4b       # likewise TripoSG, Hunyuan3D-2.1
python experiments/baselines/make_baseline_manifest.py --dir runs/val_trellis2 --method trellis2 \
    --out manifests/trellis2.json
python eval/run_eval.py --config eval/config.yaml --manifest manifests/table1.json \
    --baseline-manifest manifests/trellis2.json --out results/table1_with_baselines --skip vs_reference
```

Every `camera.json` written by the adapters records the frame map it was generated for (copied from
`eval/frame_maps/baseline_frame_maps.json` with that file's sha256). `make_baseline_manifest.py` refuses a product
whose record differs from the frame map that the evaluation applies. One `--baseline-manifest` file can hold the
rows of several methods (concatenate the JSON lists); each method becomes one arm.
