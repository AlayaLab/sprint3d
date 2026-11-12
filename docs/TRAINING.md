# Training the students

The pipeline replaces four slow sampling stages of Pixal3D by *students*: the Pixal3D flow of the stage
with a low-rank adapter (LoRA) that is trained to replace the teacher's 12-step sampling by one, two or four
steps. This document describes how the four students were trained and how to train them again with this
repository: the **structure** and **texture** students by trajectory distillation, the **coarse-shape** and
**fine-shape** students by consistency distillation.

| Student | Base flow (Pixal3D) | Student steps (t) | Training set | Updates | Processes x accumulation |
|---|---|---|---|---:|---|
| structure | `ckpts/ss_flow_img_dit_1_3B_64_bf16` | 4 (1.0, 0.9, 0.75, 0.6, 0) | 5k set, 10,486 views | 500 | 1 x 4 |
| coarse shape | `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16` | 1 (1.0, 0) | 5k set, one conditioning image per object, 4,995 objects | 3,000 | 16 x 1 |
| fine shape | `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16` | 1 (1.0, 0) | 5k set, one conditioning image per object, 4,990 objects | 2,000 (of a 3,000-update run) | 16 x 1 |
| texture | `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16` | 2 (1.0, 0.75, 0) | 10k set; 13,398 views in updates 1-1,500, 19,998 views in updates 1,501-3,000 | 3,000 (two segments) | 4 x 1, then 1 x 4 |

Layout:

```
src/fastpixal3d/training/    shared training code (configuration, time grid, loss and update, adapters,
                             checkpoints, cache records, the update loop of the trajectory trainers,
                             consistency.py: the consistency distillation and its update loop)
train/structure_traj/train.py        structure trainer
train/texture_traj/train.py          texture trainer
train/shape_consistency/train.py     coarse- and fine-shape trainer (model.student selects the stage)
train/caches/                cache builders (conditioning and teacher trajectories; shape_conditioning_wild.py
                             for the shape students)
train/configs/               structure_student.yaml, texture_student.yaml, coarse_shape_student.yaml,
                             fine_shape_student.yaml, caches.example.yaml
data/lists/                  object lists of the training sets (docs/DATA.md)
data/select/                 selection of the 10k set (docs/DATA.md)
```

## Method: one-step trajectory distillation (structure and texture students)

Pixal3D samples each stage with Euler steps of a flow model on the noise fraction t (1 = noise, 0 = data).
With `rescale_t = 3` the 12-step teacher grid is

    t_j = 3 s_j / (1 + 2 s_j),   s = linspace(1, 0, 13)

and every k-step grid with k dividing 12 is a subsequence of it. The teacher's states on this grid
(*anchors*) are computed once and cached. The student runs on a coarser grid of teacher-grid indices:
`0, 3, 6, 8, 12` for the structure student (not uniform; t = 1.0, 0.9, 0.75, 0.6, 0.0) and `0, 6, 12` for the
texture student. A training sample is one view and one segment `(i, j)` of the student grid, drawn
uniformly:

    v      = student(x_i, t_i)                 x_i: teacher state at index i (index 0: the view's noise)
    x_pred = x_i - (t_i - t_j) * v             Pixal3D's Euler step
    loss   = mean((x_pred - x_j)^2)            x_j: teacher state at index j

The student is run conditional-only (no classifier-free guidance); for the structure stage the
teacher's trajectory includes its guidance, so the student also learns to replace it.

Settings of both students (`train/configs/*.yaml`, checked by `tests/test_train_config.py`):

| | |
|---|---|
| adapters | rank 256, alpha 256 (scale 1), no dropout, on every linear layer of the 30 transformer blocks except the adaLN modulation: `self_attn.to_qkv`, `self_attn.to_out`, `cross_attn.cross_attn_block.{to_q,to_kv,to_out}`, `cross_attn.proj_linear`, `mlp.mlp.0`, `mlp.mlp.2` (240 adapters); float32 weights; `lora_B = 0` at start |
| base flow | frozen (bfloat16 transformer blocks), evaluation mode, activation checkpointing in every block |
| forward | bfloat16 autocast; loss in float32 |
| optimizer | AdamW, learning rate 1e-4 (constant, no warm-up), betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| update | gradients of `grad_accum` micro-batches of one view each (loss / `grad_accum`), averaged over processes, global norm clipped to 10, one AdamW step |
| EMA | `ema = 0.99 * ema + 0.01 * adapters` after every update; the released weights are the EMA |
| sampling | `random.Random(seed + 1000 * rank)`: per micro-batch one `choice` of the view, then one `randrange` of the segment; seeds 20260915 (structure), 20260912 (texture) |
| initialisation | `torch.manual_seed(seed)`, then the base flow is built (`pixal3d.models.from_pretrained`), then the adapters are created on the CPU in module order and moved to the GPU |

The adapter initialisation, the loss, the gradient accumulation, clipping, AdamW and EMA of this
repository were compared with the code that trained the released students on the same small models
and inputs: the adapters, losses, gradient norms, EMA and optimizer state agree bit for bit on the CPU
and on the GPU with bfloat16 autocast (`tests/test_train_objective.py` keeps the reference numbers).
The adapters attached to the real Pixal3D flows have the names and shapes of the released files
(`tests/test_train_lora.py`), and a checkpoint written by the trainers converts to a file with the same
tensor names, shapes and size as the released one.

## Method: consistency distillation (shape students)

The coarse- and fine-shape students are one-step predictors of the clean latent: at inference the pipeline
runs the adapted flow once at t = 1 on the object's support and recovers the latent as
`x0 = (1 - s) x - (s + (1 - s) t) v` with `s = 1e-5`, in float64 (`fastpixal3d.stages.shape`). They were trained
by consistency distillation with the frozen flow itself as the teacher (`fastpixal3d.training.consistency`). One
update on one process takes one object (its support `coords` and conditioning `global`, `proj` from the cache):

    x0       the teacher's own sample on the support: 12 guided Euler steps (rescale_t 3, guidance 7.5 /
             rescale 0.5 / interval [0.6, 1.0]) from a noise drawn once per object from a seed derived from the
             run seed and the object id; computed once per object and kept in CPU memory ("anchor")
    n        uniform in {1, ..., 49}:  t_n = 1 - n / 50  (student),  t_next = t_n - 1 / 50  (target)
    x_n      = (1 - t_n) x0 + (s + (1 - s) t_n) eps,  eps ~ N(0, I)
    x_next   = x_n - (t_n - t_next) v_teacher(x_n, t_n)           one guided Euler step of the teacher
    target   = x0_from_v(x_next, v_EMA(x_next, t_next), t_next)   the student with its EMA weights, no gradient
    pred     = x0_from_v(x_n, v_student(x_n, t_n), t_n)
    loss     = 80 * mean((pred - target)^2)                        in float64

The student is conditional-only. With `torchrun` the processes train on different objects and average their
gradients; the released students used 16 processes (4 nodes x 4 GPUs), so one update saw 16 objects.

Settings of both shape students (`train/configs/{coarse,fine}_shape_student.yaml`, checked by
`tests/test_train_config.py`):

| | |
|---|---|
| adapters | as above: rank 256, alpha 256, no dropout, the same 240 layers; `lora_B = 0` at start |
| base flow | frozen (bfloat16 blocks), evaluation mode; activation checkpointing for the fine-shape student only |
| forward | no autocast: bfloat16 blocks, float32 adapters, float64 clean-latent recovery and loss |
| teacher | the same flow with the adapters switched off, guided as Pixal3D's shape sampler |
| optimizer | AdamW, learning rate 2e-6 (constant), betas (0, 0.999), eps 1e-8, weight decay 1e-3 |
| update | one object per process, gradients averaged over processes, global norm clipped to 10, one AdamW step |
| EMA | `ema = d * ema + (1 - d) * adapters` with `d = min(0.99, (1 + n) / (10 + n))`, `n` = updates done before (0.1 at the first update, 0.99 from the 891st); the released weights are the EMA |
| data order | PyTorch's `DistributedSampler(shuffle=True, drop_last=True)` seeded with the run seed, one object per process, re-shuffled every epoch (`consistency.distributed_order`); 3,000 updates x 16 = 48,000 visits, about 9.6 passes over the 4,990-5,000 objects |
| draws | per update: the time index, then the noise, from a generator seeded from (run seed, update, rank, object id); the anchor noise from (run seed, object id) |
| seeds | 20260716 (coarse shape), 20260723 (fine shape) |

The anchor, the guided teacher step, the loss, the AdamW update and the EMA ramp of this repository were
compared with the original training code on the same tiny model and inputs: losses, gradient norms, adapters
and EMA agree bit for bit on the CPU (`tests/test_train_consistency.py` keeps the reference numbers), and the
guided velocity agrees inside and outside the guidance interval.

## Requirements

The installation of [INSTALL.md](INSTALL.md) (Pixal3D source checkout, its dependencies and FlashAttention-2)
and the Pixal3D weights. The trainers set `ATTN_BACKEND=flash_attn` and, as in the training runs,
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (`runtime.tf32_override`; the container image of the runs set this
variable, so float32 cuBLAS matmuls could use TF32). The texture trainer keeps the whole Pixal3D pipeline on
the GPU because it recomputes the image conditioning at every step. The cache builders need the same
environment; they load the full Pixal3D pipeline (flows, decoders, the DINOv3 extractors and the NAF
upsampler), and the structure-conditioning builder also uses `data_toolkit/utils.py` of the Pixal3D source
checkout to recompute the cameras.

## Data

The object lists, licences and the selection of the 10k set are described in [DATA.md](DATA.md). The renders
are not produced by this repository:

- 10k set: two views per object, rendered with the Pixal3D data toolkit (changed only in the render device and
  threads and in loading 3D-FUTURE and HSSD assets; `render_cond.py`, Blender 3.0.1,
  Cycles, 1024 x 1024 RGBA, views on a Hammersley sequence, field of view drawn in [10, 70] degrees, with
  `transforms.json`) and mesh dumps of `dump_mesh.py`, in the toolkit's layout
  `<root>/<subset dir>/{renders_cond/<sha256>/, mesh_dumps/<sha256>.pickle}`;
- 5k set: two views per object, 512 x 512 RGBA (Blender 3.0.1, yaw 45 and 225 degrees, pitch 25 degrees,
  field of view 40 degrees), with `transforms.json` and mesh dumps in the same layout without subset
  directories.

The scripts that rendered them are not part of this release.

## Caches

Each cache holds one `<view>.pt` record per rendered view. Copy `train/configs/caches.example.yaml`, set
the dataset roots and the cache directories, and run the builders in order. All builders skip views that
already have a record, write atomically, set their numeric environment themselves, and can be split over
processes or GPUs with
`--shard i --nshard n`; `--verify-n N` samples the first N views twice and reports the relative
difference of the final state.

```bash
C=train/configs/caches.yaml
# structure student (5k set)
python train/caches/structure_conditioning.py --config $C --dataset train_5k
python train/caches/structure_anchors.py      --config $C --dataset train_5k --verify-n 3
# texture student (10k set)
python train/caches/structure_conditioning.py --config $C --dataset curated_10k
python train/caches/fine_shape_conditioning.py --config $C --dataset curated_10k
python train/caches/fine_shape_anchors.py      --config $C --dataset curated_10k --verify-n 2
python train/caches/texture_anchors.py         --config $C --dataset curated_10k --verify-n 2
# shape students (5k set, the objects' conditioning images): coarse and fine records in one pass
python train/caches/shape_conditioning_wild.py --config $C --dataset train_5k_images --stage both --verify-n 2
```

| Cache (builder) | Per view | Size per view |
|---|---|---|
| structure conditioning (`structure_conditioning.py`) | DINOv3 global [5, 1024] and projected [4096, 1024] features of the input image, float32; the true camera; deterministic algorithms with PyTorch's TF32 switches off | 17 MB |
| structure anchors (`structure_anchors.py`) | teacher states after steps 2, 3, 4, 6, 8, 9, 10, 12 as bfloat16 [4096, 8]; 12 steps, `rescale_t` 3, guidance 7.5 / rescale 0.7 / interval [0.6, 1.0] | 0.5 MB |
| fine-shape conditioning (`fine_shape_conditioning.py`) | Pixal3D's cascade with fixed seeds (structure seed 42 with the pipeline's structure sampler settings, coarse shape seed 0, 12 steps each), the fine support (1024 grid, lowered while it has 32,768 tokens or more) and its 1024 conditioning, stored as bfloat16 | about 46 MB on average |
| fine-shape anchors (`fine_shape_anchors.py`) | teacher states of the fine-shape flow after steps 3, 4, 6, 8, 9, 12, bfloat16; 12 steps, `rescale_t` 3, guidance 7.5 / 0.5 / [0.6, 1.0] | 0.5-4 MB |
| texture anchors (`texture_anchors.py`) | fine support, the normalised fine-shape latent (the final state of the fine-shape teacher trajectory) as `concat_cond`, teacher texture states after steps 3, 4, 6, 8, 9, 12 (float32; 12 steps, `rescale_t` 3, conditional only, noise seed 0), camera, render reference | about 10 MB on average |
| shape conditioning, coarse and fine (`shape_conditioning_wild.py`) | one record per object, named by its id: the coarse support (structure flow at seed 42, 12 steps, the pipeline's default guidance) with the 512 conditioning, and the fine support (coarse flow at seed 0, 12 steps, upsampled x4, quantised to the 1024 grid, lowered by 128 to at least 768 while 32,768 tokens or more) with the 1024 conditioning; `global` [5, 1024] and `proj` [N, 2048], bfloat16 by default (lossless for the flows; `--storage-dtype float32` = the original caches); the MoGe-2 camera | coarse about 10 MB, fine about 50 MB (float32: 20 / 100 MB) |

The shape caches take the object's conditioning image through Pixal3D's **in-the-wild input path**, not the
raw-render / true-camera path of the other caches: `pipeline.preprocess_image` (background removal when the
image has no usable alpha, crop) and the camera estimated by MoGe-2 on the prepared image
(`get_camera_params_wild_moge`, mesh scale 1), as the released students saw them (`<root>/<subset dir>/<id>/image.png`
of the 5k set, one image per object; `train_5k_images` in `caches.example.yaml`). Objects whose structure sample
is empty are skipped: 250 of the 5,243 objects for the coarse cache and 253 for the fine cache of the released
runs. The released runs read float32 caches; bfloat16 storage gives the same flow inputs, because the shape
flows cast their conditioning to bfloat16.

The input image of every cache and of the texture trainer is the raw render: resized to 512 x 512 with
Lanczos when larger, composited on black, no crop; the camera is the render's own (field of view and
distance from `transforms.json`, mesh scale recomputed from the mesh dump). The structure-stage noise and
the fine-shape noise are drawn from seeds derived from the view name, so step 0 is never stored.

The fine-shape latent of a texture record comes from the fine-shape anchor record when one exists (rounded
to bfloat16 there) and is otherwise computed in the texture builder with the same function, without rounding
(`--fine-shape-source auto`, the rule of the original build; `anchor` or `rollout` force one source). In the
original build the fine-shape anchors were still being computed while the texture records were built, so the
texture records of the released student use both sources (each record's `provenance.concat_cond_source`
says which): 12,704 of the 19,998 training views (all 10,486 views of `matched_5243` and 2,218 others) read the
stored fine-shape anchors, and 7,294 computed the latent in the builder.

Checks against the caches of the released runs (one view per row, rebuilt here from the same inputs):

| Cache | Result |
|---|---|
| structure conditioning, 5k set and 10k set | bit-identical (`global`, `proj`, camera) |
| structure anchors, 5k set | bit-identical (all eight states) |
| fine-shape conditioning | bit-identical (fine support, `global`, `proj`, resolution) |
| texture anchors from the released fine-shape latent (`anchor`) | bit-identical (all six states) |
| texture anchors with the fine-shape latent computed in the builder (`rollout`) | bit-identical, `concat_cond` included |
| fine-shape anchors | **not identical**: relative difference 0.3-0.4 % at step 3 growing to 6-9 % at step 12 (two views) |

The stored fine-shape anchors were computed in a separate job. On the machine of these checks the original
builder reproduces exactly the records of this repository's builder, and the fine-shape rollout inside the
texture builder reproduces the texture records that were built that way; the difference therefore comes from
the environment of that job (not identified), not from the code. The 12,704 texture records whose fine-shape
latent was read from the stored anchors cannot be rebuilt bit for bit: their `concat_cond` and texture states
differ by about this much. The 7,294 records with the latent computed in the builder can be (one checked).

## Training

Set the paths marked `set me` in the configuration (or pass them with `--set`). Every key is documented in
the YAML file; `--print-config` prints the resolved configuration.

```bash
# structure student: 1 GPU, gradient accumulation 4, 500 updates
python train/structure_traj/train.py --config train/configs/structure_student.yaml

# texture student, second segment as released: resume from the update-1,500 checkpoint in output.dir,
# 1 GPU, gradient accumulation 4, up to update 3,000
python train/texture_traj/train.py --config train/configs/texture_student.yaml

# a texture run from scratch with the effective batch of the first segment on 4 GPUs (data parallel)
torchrun --nproc_per_node 4 train/texture_traj/train.py --config train/configs/texture_student.yaml \
    --set optim.grad_accum=1 --set output.checkpoint_every=500

# a short check (a few updates on a few cached views)
python train/structure_traj/train.py --config train/configs/structure_student.yaml \
    --set optim.updates=6 --set data.max_views=4 --set output.dir=/tmp/structure_check --set output.resume=none

# shape students as released: 16 processes (4 nodes x 4 GPUs), one object each; the fine-shape run is
# configured for 3,000 updates and the released weights are its update-2,000 checkpoint
torchrun --nnodes 4 --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint $MASTER:29500 \
    train/shape_consistency/train.py --config train/configs/coarse_shape_student.yaml
torchrun --nnodes 4 --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint $MASTER:29500 \
    train/shape_consistency/train.py --config train/configs/fine_shape_student.yaml
# a short check of a shape student (one GPU, a few objects)
python train/shape_consistency/train.py --config train/configs/coarse_shape_student.yaml \
    --set optim.updates=6 --set data.max_objects=4 --set output.dir=/tmp/coarse_check --set output.resume=none
```

With fewer processes the effective batch shrinks (one object per process per update) and the data order
changes; a resumed run must keep the number of processes of its checkpoints.

Each run writes `config.yaml`, `loss.jsonl` (one line per logged update: mean loss, micro-batch losses,
segment, view, gradient norm, step time) and `completed_updates_<n>.pt` checkpoints into `output.dir`
(`output.checkpoint_every`, the last update and an early stop by `runtime.max_minutes`). With
`output.resume: auto` a restarted run continues from the newest checkpoint: adapters, EMA and AdamW state
are restored; the sampling generator is created again from the seed.

A checkpoint holds the adapters (`gen`), their EMA (`ema.shadow`), the AdamW state and the run's settings.
Convert it to the released format with

```bash
PYTHONPATH=src python -m fastpixal3d.students.convert --student structure \
    --checkpoint runs/structure_student/completed_updates_000000000500.pt --out my_students
PYTHONPATH=src python -m fastpixal3d.students.convert --student texture \
    --checkpoint runs/texture_student/completed_updates_000000003000.pt --out my_students
PYTHONPATH=src python -m fastpixal3d.students.convert --student coarse_shape \
    --checkpoint runs/coarse_shape_student/completed_updates_000000003000.pt --out my_students
PYTHONPATH=src python -m fastpixal3d.students.convert --student fine_shape \
    --checkpoint runs/fine_shape_student/completed_updates_000000002000.pt --out my_students
```

([WEIGHTS.md](WEIGHTS.md) describes the format; the texture student is folded into bfloat16 weights at load
time.)

## The released runs

**Structure student.** One process, gradient accumulation 4 (4 views per update), 500 updates, checkpoints
every 250 updates, pool of 10,486 views (all records of the 5k set), seed 20260915, the settings above. The
released weights are the EMA of the checkpoint after update 500 (sha256
`be04198a3751c999343510611e627ff92a696b02bdeb83ec8ea93c966663c1f8`). The run took 918 s (1.84 s per update,
cache reads included) on one GPU.

**Texture student**, in two segments:

| | Updates 1-1,500 | Updates 1,501-3,000 |
|---|---|---|
| processes x accumulation | 4 x 1 (gradients averaged over 4 GPUs) | 1 x 4 |
| effective batch | 4 views | 4 views |
| training pool | 13,398 views: the records of the texture-anchor cache that existed when the run started (the cache was still being built) | 19,998 views: all views of the 10k set with records (object `7f9dc2a3bc46` had none, see [DATA.md](DATA.md); the builders of this repository do not need the files it lacked, so leave its two views out to get the same pool) |
| sampling streams | 4 streams, `seed + 1000 * rank` | 1 stream, `seed`, created again at the resume (it repeats rank 0's sequence of the first segment from its start, on a larger pool) |
| trainer | an earlier version of the texture trainer without the accumulation option; with one micro-batch per process its update is the same arithmetic as this trainer's | this repository's `train/texture_traj/train.py` |
| checkpoints | every 500 updates; the run was started for 3,000 updates and ended after the checkpoint of update 1,500 | resumed from the update-1,500 checkpoint (adapters, EMA, AdamW state); every 250 updates |
| time | 3,840 s (2.56 s per update) | 9,880 s (6.59 s per update) |

The released weights are the EMA of the checkpoint after update 3,000 (sha256
`4d8c96e111239c08670fe7cf88391ac024e2aa70b952e9f3714149c36ccb970f`). Both segments used seed 20260912, the
10k-set caches above and the settings above.

**Coarse-shape and fine-shape students.** 16 processes (4 nodes x 4 H100 GPUs), one object per process per
update, the settings of "Method: consistency distillation", checkpoints every 250 updates, trained with the
original code (its `cd_init` stage, which this repository's `consistency.py` reproduces).

| | Coarse shape | Fine shape |
|---|---|---|
| training pool | 4,995 objects of the 5k set with a coarse record | 4,990 objects with a fine record (4,983 at resolution 1024, 7 at 896) |
| updates | 3,000 | the run was configured for 3,000; the released weights are the checkpoint after update 2,000 |
| seed | 20260716 | 20260723 |
| activation checkpointing | no | yes |
| source checkpoint of the released weights | update 3,000, sha256 `46bc526388cc55f226d5747e64d7cd8a32fbe9153fdd0ff67eac7695ed148e1f` | update 2,000, sha256 `9fa67dd330e2dcfaf751e281d19ed8cc3e9c949c97a1a0d0d1a5940bafda070c` |

Both runs stopped inside the consistency stage of the original trainer; no later training stage was entered.

Smoke run of this trainer (2026-11-04, one H100 node, the original caches): coarse student 6 updates on 4 objects
(1.9 s per update after the first, which also builds the anchors), resumed to 8 with the data position restored,
and 4 updates with 4 processes on 8 objects (the first update of rank 0 gives the same loss as the one-process run,
0.5565, since the anchor and the draws depend only on the seed, the update and the object); fine student 3 updates on
2 objects of 21,669 and 13,939 tokens (18 s for the first update, 5-9 s after). Both checkpoints convert to
release files of 1,305,542,328 bytes with 240 adapters, the size and layout of the released students.
The 4-step consistency structure student of the report's ablations (15,000 updates, dense structure flow) is an
experiments-only adapter and has no trainer here.

## What cannot be reproduced exactly

- **Platform.** The GPU kernels are not bitwise deterministic across runs, GPUs, drivers and library
  versions (FlashAttention's backward pass, TF32 matmuls, atomic additions), so a rerun of the same
  configuration produces adapters that differ in the last bits at first and drift apart over the updates.
  The released numbers were measured with the released weights, not with a retrained student.
- **Random streams.** The sampling order is fixed by the seed, the pool and the number of processes, and the
  noise by the view names. The pool of the texture student's first segment (13,398 views present at its start)
  is not recorded view by view, so that segment cannot be replayed exactly; the second segment's pool is the
  full 19,998-view list. The rendering of the training views involved random camera draws that this
  repository does not reproduce.
- **Caches.** Rebuilt from the same renders and weights, the structure caches, the fine-shape conditioning and
  the texture anchors were bit-identical to the released ones on the views checked, but the stored fine-shape
  anchors were not, so 12,704 of the 19,998 texture records (those that read them) cannot be rebuilt bit for
  bit (see "Caches"). Records built on another platform can differ as well.
- The original texture trainer also loaded Pixal3D's camera model (MoGe-2), which training does not use; this
  trainer does not load it.
- **Shape students.** The original runs drew the time index and the noise of every update from the process-global
  generator (seeded 1234 + rank at start and consumed by model construction before), which cannot be replayed;
  this trainer derives them from the run seed, the update, the rank and the object id, so a resumed run repeats
  its draws exactly. The original adapters were initialised on the GPU from seed 1234 and broadcast from rank 0;
  this trainer initialises them on the CPU from `optim.seed` (equally valid starting values, as for the other
  students). The teacher anchors are reproducible: their noise comes from a generator on the training device
  seeded from the run seed and the object id, as in the original runs (a CUDA generator gives other values than a
  CPU one for the same seed, so a CPU run has other anchors). The original runs kept no TF32 override and no
  autocast; neither does this trainer. The per-object anchors stay in CPU memory for the whole run (about 0.6 MB
  per coarse object, 2-4 MB per fine object; a 16-process run holds them once per process).

## Tests

CPU (`pytest -q tests`): `test_train_config.py` (configurations and object lists), `test_train_objective.py`
(update arithmetic against the reference numbers, accumulation, sampling order, grids, noise seeds),
`test_train_consistency.py` (the consistency distillation against the reference numbers of the original code:
anchor, guided teacher step, loss, AdamW and EMA ramp; seeds; the data order against PyTorch's
`DistributedSampler`), `test_train_lora.py` (initialisation order, names; with `FASTPIXAL3D_PIXAL3D_SRC`, a local
`FASTPIXAL3D_PIXAL3D` and optionally `FASTPIXAL3D_STUDENTS`, the adapter layout of the real flows against the
released students), `test_train_checkpoint.py` (checkpoint, resume, conversion to the released format, for all
four students).
