# Parity with the measurement code (pre-registered 2026-10-03, before any run)

The numbers in the report come from the internal measurement code. The public package must reproduce the
deployed configuration within the tolerances below, measured on the 80 timing inputs (40 objects x 2 views)
of the development subset, one process per machine, one H200, warm-up 3 excluded.

Deployed configuration (all levers on):
- structure support coordinates: identical set for every input;
- coarse-shape and fine-shape latents: relative Frobenius error <= 1e-3 against the measurement code;
- face count of the GLB: within +-1 %;
- texture: PSNR >= 40 dB against the measurement code's texture of the same input;
- end-to-end time: median within +-5 % of the measurement code on the same machine type
  (reference: 4.05 s for the official export, 3.20 s for the 200k export).

Cumulative measurement rows: for each adjacent pair of rows, the sign of the difference matches the
measurement code and the 95 % CI of the public package contains the measurement code's point estimate.

Reference version of the measurement code: the files used by the dedicated timing run
(crossover_r3 at sha256 83866031..., plus the 15 chained files; full list in the internal snapshot).
Bit-identical output is not expected (non-deterministic GPU simplification, BVH build and FA3 kernels).
Results of the parity run are reported in this file when available; the criteria above are not changed
after the run.

## Amendment, 2026-10-03 (before the parity run; after a one-image smoke check)

The smoke check on one example image showed that the texture criterion above was ill-posed: the atlas layout
of two runs differs even for the measurement code against itself (image-space PSNR about 16 dB), while the
colour sampled at the same surface points agrees to about 40 dB (official export) and 35 dB (200k export).
The texture criterion is therefore replaced, before any run on the 80 timing inputs, by:
- the structure support set and the decoder-input latents of the shape and texture stages are bit-identical
  to the measurement code (sha256 of the tensors), and
- the surface-sampled texture PSNR between the package and the measurement code is at least the
  measurement code's own run-to-run surface PSNR on the same input minus 1 dB.
The face-count and end-to-end criteria are unchanged. Smoke-check observation on the single image: latents
identical, face count within the measurement code's own run-to-run spread, end-to-end median +2.5 % (official
export) and +0.2 % (200k export) on a shared machine; the official run on the timing machine type decides.

## Amendment, 2026-10-05 (before the parity run)

The 200k export (200,000 faces, 2048 x 2048 texture) was removed from the released package by the author's
decision before the parity run. The package has one export, the official export (1,000,000 faces, 4096 x 4096
texture), which is also the export of every number in the report that this repository reproduces. The parity run
therefore covers the official export only: the end-to-end criterion applies to the official export alone
(reference: 4.05 s), and the other criteria are evaluated on the same runs. The statements about the 200k export
above are kept as registered and are not evaluated. The tolerances are unchanged.

## Results, 2026-10-05

**Setup.** One machine: one H200 GPU, 15 CPU cores, `OMP_NUM_THREADS=8`. The run used the timing protocol of
TIMING.md on the 80 timing inputs, with the official export.

Each run was a separate process, in this order:

1. the measurement code (run r0);
2. this package;
3. the measurement code again (run r1).

| run | version |
|---|---|
| measurement code | the files of the dedicated timing run, each checked against its recorded sha256 before the run |
| this package | commit `5ce2013` with the released student weights (loaded through `students.json`) |

Commits after `5ce2013` have not been through this check.

The end-to-end times exclude the first 3 inputs as warm-up (n = 77). All other quantities use all 80 inputs.

For context, the face counts are also compared with a third run of the measurement code: the dedicated timing
run of the report, on the same inputs. The surface-sampled PSNR is computed with 200,000 points per direction.

| criterion | outcome |
|---|---|
| structure support identical | **met**: identical for 80 / 80 inputs |
| decoder-input latents bit-identical (amendment) | **met**: sha256 of the fine-shape coordinates, the fine-shape latent and the texture latent identical for 80 / 80 |
| coarse- and fine-shape latents, relative error <= 1e-3 | fine: **met**, error 0 for 80 / 80 (identical bytes). Coarse: not evaluated, because the measurement code does not record it; the fine latent, which is computed from it, is bit-identical |
| face count within +-1 % | **not met**: 74 / 80 within +-1 % |
| texture, surface PSNR >= own run-to-run PSNR - 1 dB (amendment) | **not met**: 78 / 80 |
| end-to-end median within +-5 % | **met**: +4.8 % |
| cumulative measurement rows | not part of this run (evaluated separately: see "Cumulative measurement rows, 2026-10-06" below) |

### Face count

Absolute relative difference of the face count, with the measurement code's r0 as reference:

| comparison | median | p90 | max | inputs beyond 1 % |
|---|---|---|---|---|
| package vs r0 | 0.08 % | 0.22 % | 8.45 % | 6 |
| measurement r1 vs r0 (own run-to-run) | 0.09 % | 0.30 % | 8.42 % | 6 |

Face counts of the median input:

| run | median face count | inputs with at least 990,000 faces |
|---|---|---|
| measurement r0 | 968,770 | 17 |
| measurement r1 | 975,120 | 22 |
| timing run | 975,422 | 21 |
| package | 974,759 | 22 |

The six inputs beyond 1 %:

| input | r0 | r1 | timing run | package | package vs r0 |
|---|---|---|---|---|---|
| `da89c72706302_v03` | 921,980 | 999,576 | 999,262 | 999,894 | +8.45 % |
| `d38613d1ca525_v02` | 932,237 | 932,941 | 931,214 | 999,459 | +7.21 % |
| `d74d243167b88_v03` | 952,650 | 986,846 | 985,650 | 986,346 | +3.54 % |
| `d500359f38b57_v00` | 965,769 | 998,613 | 999,623 | 998,089 | +3.35 % |
| `dc6f8eda86a3f_v00` | 969,787 | 971,351 | 994,465 | 992,541 | +2.35 % |
| `df1acb2fa444d_v00` | 948,508 | 964,872 | 964,896 | 963,750 | +1.61 % |

Five of the six are inputs where the measurement code itself gives two different face counts across its three
runs. On each of them the package lies within 1 % of one of the measurement code's runs.

On the sixth input, `d38613d1ca525_v02`, the measurement code gave 931,214 to 932,941 faces in all three runs,
while the package gave 999,459 (+7.2 %). Repeated runs in the additional check below show that the measurement
code itself also gives about 999,000 faces on this input.

Over all 80 inputs, the package is within 1 % of at least one of the three measurement runs on 79 inputs.

### Texture

Surface-sampled base-colour PSNR:

| comparison | median | p90 | min | max |
|---|---|---|---|---|
| package vs r0 | 43.51 dB | 52.55 dB | 32.45 dB | 61.32 dB |
| measurement r1 vs r0 (own run-to-run) | 43.46 dB | 52.70 dB | 32.53 dB | 60.98 dB |

The margin is the package value minus the measurement code's own value on the same input:

| statistic | value |
|---|---|
| median | +0.04 dB |
| p10 | -0.44 dB |
| p90 | +0.56 dB |
| min | -1.39 dB |
| max | +1.74 dB |
| inputs with a negative margin | 36 |
| inputs with a positive margin | 44 |
| inputs below -1 dB | 2 |
| inputs above +1 dB | 2 |

The two inputs below the criterion:

| input | package vs r0 | r1 vs r0 | margin |
|---|---|---|---|
| `db14844a7806d_v00` | 50.18 dB | 51.57 dB | -1.39 dB |
| `dc6f8eda86a3f_v01` | 42.58 dB | 43.92 dB | -1.34 dB |

They miss the criterion by 0.39 dB and 0.34 dB. Their face counts are stable across all four runs: within 0.08 %
on `db14844a7806d_v00` and 0.11 % on `dc6f8eda86a3f_v01`.

With one pair of measurement runs per input, the measurement code's own spread of this pairwise PSNR cannot be
measured for a single input. Across inputs, the margin is spread on both sides of zero.

The pixel-by-pixel PSNR of the texture images (median 15.3 dB) remains informational only, because the atlas
layouts differ between runs.

### End-to-end time

| run | median | p90 | max | 95 % CI of the median |
|---|---|---|---|---|
| measurement r0 | 4.002 s | 6.070 s | 17.344 s | 3.740-4.318 s |
| measurement r1 | 4.170 s | 6.251 s | 17.604 s | 3.952-4.370 s |
| package | 4.195 s | 6.231 s | 17.488 s | 4.018-4.439 s |

Package against measurement r0:

- the package median is 4.8 % above r0, within the +-5 % criterion;
- the 95 % bootstrap interval of the ratio of medians (package / r0) is 1.028-1.090;
- the per-input median ratio is 1.035.

Measurement code against itself:

- r1 is 4.2 % above r0 in the median (ratio of medians 1.042, interval 1.013-1.059);
- its per-input median ratio is 1.028.

Package against r1: the medians differ by +0.6 %, and the per-input median ratio is 1.008.

Package against the registered reference value of 4.05 s: +3.6 %.

### Conclusion

**Numerics.** The structure support and all decoder-input latents are bit-identical to the measurement code on
all 80 inputs.

**Timing.** The end-to-end median is within the criterion at +4.8 %. The measurement code's own two runs differ
by 4.2 %.

**Face count and texture.** Neither criterion is met as registered: face count 74 / 80, texture 78 / 80. The
inputs outside the criteria are:

- `da89c72706302_v03`, `d74d243167b88_v03`, `d500359f38b57_v00`, `dc6f8eda86a3f_v00` and `df1acb2fa444d_v00`
  (face count). The measurement code itself alternates between two face counts on these inputs, and the package
  matches one of its runs.
- `d38613d1ca525_v02` (face count, +7.2 %). The measurement code gave about 932,000 faces in its three runs
  here. Repeated runs in the additional check below show that it also gives about 999,000 faces on this input,
  as the package did.
- `db14844a7806d_v00` and `dc6f8eda86a3f_v01` (texture). They miss the -1 dB allowance by 0.39 dB and 0.34 dB.

### Additional check, 2026-10-05

**Current version.** The deployed path of the current main branch reproduces the results of commit `5ce2013`, the
commit tested above. Two inputs were checked: the example image used in the smoke check, and the parity input
`d38613d1ca525_v02`. On both, the structure support set and the three decoder-input digests are identical.

**`d38613d1ca525_v02`.** The measurement code (the same version as above) was run 5 more times on this input in
one process. It gave 931,370, 999,347, 999,667, 999,069 and 932,639 faces. The package was also run 5 times (one
process), and the same package at `5ce2013` another 5 times; all ten package runs gave 931,295 to 932,675 faces.

In every run of both codes, the inputs to the mesh simplification were the same:

- the same decoded mesh (8,907,524 faces);
- the same remesh resolution (half resolution, divisor 2);
- the same remeshed mesh (4,041,944 faces);
- the same target (1,000,000 faces);
- the same simplification settings.

The simplification stops after the first round that brings the face count to the target or below. The number of
faces removed in a round varies slightly from run to run with GPU scheduling. On this input, the 19th round ends
within about 1,000 faces of the target: the 19-round runs ended 110 to 930 faces below it. When the 19th round
ends just below the target, the result is about 999,000 faces. When it ends just above, a 20th round removes
about 68,000 more faces, and the result is about 932,000.

Both codes produce both outcomes. The +7.2 % on this input is this run-to-run bimodality, not a difference
between the codes.

### Note, 2026-10-05: the autotune cache

Bit-identity of the fine-shape stage also requires the same FlexGEMM autotune cache: on the example image, two
package runs that each started from an empty cache gave different fine-shape digests (35,465 and 35,462 fine
tokens), while all runs that started from copies of one cache file gave identical digests. For this reason the
parity runs start every package run from a copy of the autotune cache that the measurement code's Pixal3D
checkout uses (`--autotune-cache`).

### Re-run with the current package, 2026-10-05

**Setup.** The protocol, the machine type and the measurement code are the same as above, and the measurement code
again ran twice (r0, r1).

| run | version |
|---|---|
| this package | commit `fae3c8b`: the single export, one weights directory, the structure-adapter selection by key, and a faster texture finalisation in the bake |
| student weights | the released files, read through the weights directory |

Times use the 77 measured inputs. Everything else uses all 80. The face-count comparison uses all five runs of the
measurement code on these inputs:

- r0 and r1 of this re-run;
- r0 and r1 of the first parity run;
- the dedicated timing run.

| criterion | outcome |
|---|---|
| structure support identical | **met**: 80 / 80 |
| decoder-input latents bit-identical | **met**: 80 / 80 |
| fine-shape latent, relative error <= 1e-3 | **met**: error 0. Coarse: not evaluated, as above |
| face count within +-1 % | **not met**: 74 / 80 |
| texture, surface PSNR >= own run-to-run PSNR - 1 dB | **not met**: 77 / 80 |
| end-to-end median within +-5 % | **met**: -0.15 % |

**End-to-end time.**

| run | median | 95 % CI of the median |
|---|---|---|
| measurement r0 | 3.986 s | 3.745-4.237 s |
| measurement r1 | 3.986 s | 3.747-4.274 s |
| package | 3.980 s | 3.877-4.259 s |

- Package against r0: ratio of medians 0.9985 (95 % interval 0.966-1.053). The per-input median ratio of r0 to the
  package is 1.0016.
- Measurement code against itself: the medians of r0 and r1 agree to 0.01 %.

**Texture finalisation.** The stage that bakes the texture and builds the textured mesh (`glb_bake_finalize_s`) is
where the package was slower in the first run. Medians:

| run | package | measurement code |
|---|---|---|
| first parity run | 0.915 s | 0.747 s and 0.850 s |
| this re-run | 0.752 s | 0.733 s and 0.743 s |

Within that stage, the final conversion step takes 0.703 s in the package and 0.694 s and 0.699 s in the
measurement code.

**Face count.** Absolute relative difference of the face count, with r0 as reference:

| comparison | median | p90 | max | inputs beyond 1 % |
|---|---|---|---|---|
| package vs r0 | 0.07 % | 0.25 % | 5.32 % | 6 |
| measurement r1 vs r0 | 0.07 % | 0.22 % | 6.69 % | 4 |

The six inputs beyond 1 %:

| input | package vs r0 | measurement code, five runs | package, first run | package, this run |
|---|---|---|---|---|
| `d500359f38b57_v00` | +3.45 % | 950,137-999,623 | 998,089 | 998,567 |
| `d500359f38b57_v02` | +5.32 % | 942,014-997,886 | 991,354 | 993,047 |
| `dc6f8eda86a3f_v00` | +1.09 % | 969,787-994,465 | 992,541 | 995,039 |
| `dd75fa0402316_v00` | +4.82 % | 944,074-990,012 | 945,080 | 989,852 |
| `d454ac8fb7e02_v02` | -5.00 % | 982,103-994,479 | 982,017 | 944,453 |
| `d7c1db9fd3ce6_v02` | -3.68 % | 957,930-960,118 | 958,116 | 924,592 |

On the first four inputs, the measurement code's own runs take both a higher and a lower face count, and the
package lies within 1 % of one of its runs.

On `d454ac8fb7e02_v02` and `d7c1db9fd3ce6_v02` it is the other way round:

- the package's two runs differ from each other by 3.8 % and 3.5 %;
- the package's first run matched the measurement code;
- the lower values of this re-run did not occur in the measurement code's five runs.

The records of this run do not include the number of simplification rounds, so the cause of these two steps is not
established. Over all 80 inputs, the package is within 1 % of at least one of the measurement code's five runs on 78
inputs.

**Texture.** Surface-sampled base-colour PSNR:

| comparison | median | p90 | min | max |
|---|---|---|---|---|
| package vs r0 | 43.66 dB | 52.59 dB | 32.96 dB | 61.33 dB |
| measurement r1 vs r0 | 43.52 dB | 52.19 dB | 32.59 dB | 60.61 dB |

The margin (package value minus own run-to-run value):

| statistic | value |
|---|---|
| median | +0.04 dB |
| p10 | -0.45 dB |
| p90 | +0.56 dB |
| min | -1.70 dB |
| max | +1.91 dB |
| inputs with a negative margin | 38 |
| inputs with a positive margin | 42 |
| inputs below -1 dB | 3 |
| inputs above +1 dB | 2 |

The three inputs below the criterion:

| input | package vs r0 | r1 vs r0 | margin | misses the criterion by |
|---|---|---|---|---|
| `d0d9a02a369f9_v01` | 40.48 dB | 41.56 dB | -1.08 dB | 0.08 dB |
| `db14844a7806d_v03` | 46.34 dB | 48.05 dB | -1.70 dB | 0.70 dB |
| `dccc9fbe17b75_v01` | 52.66 dB | 54.23 dB | -1.58 dB | 0.58 dB |

Their face counts agree within 0.2 % across all runs.

These are not the inputs that failed in the first run. In the first run these three met the criterion (margins
+0.16, -0.93 and +0.86 dB). The two inputs that failed then, `db14844a7806d_v00` and `dc6f8eda86a3f_v01`, meet it
now (+1.79 and +0.27 dB).

The reference value itself, the measurement code's own r1-vs-r0 PSNR, changed between the two parity runs:

| statistic | absolute change |
|---|---|
| median | 0.24 dB |
| p90 | 0.78 dB |
| max | 2.48 dB |
| inputs that changed by more than 1 dB | 4 |

**Conclusion.**

- **Numerics.** The current package is bit-identical to the measurement code on all 80 inputs.
- **Timing.** The end-to-end median is within the criterion at -0.15 %, and the texture finalisation now takes
  about as long as in the measurement code.
- **Face count and texture.** Neither criterion is met as registered: face count 74 / 80, texture 77 / 80. The
  inputs outside the criteria are:
  - `d500359f38b57_v00`, `d500359f38b57_v02`, `dc6f8eda86a3f_v00` and `dd75fa0402316_v00` (face count). They lie
    within the measurement code's own run-to-run variation.
  - `d454ac8fb7e02_v02` and `d7c1db9fd3ce6_v02` (face count). The package's value dropped by 3.5 to 3.8 %
    between its two runs, and the lower value was not seen in the measurement code's five runs.
  - `d0d9a02a369f9_v01`, `db14844a7806d_v03` and `dccc9fbe17b75_v01` (texture). They miss by 0.08 to 0.70 dB. The
    reference value itself varies between runs by up to 2.5 dB.

## Cumulative measurement rows, 2026-10-06

**Setup.** This package (commit `1f1f4d6`, released student weights) ran every configuration of the report's
cumulative table (Table 8):

- the Pixal3D re-run (row 00);
- rows 01 to 17;
- Ours (row 18);
- the three controls x1 to x3 (docs/REPRODUCE_PAPER.md);
- the four ablations on Ours.

| setting | value |
|---|---|
| inputs | the 80 timing inputs, all counted, as in the report's cumulative measurement and one-at-a-time removals (no warm-up exclusion) |
| processes | one per configuration |
| machines | four of the same type as above (one H200, 15 CPU cores), started together on 2026-10-05 |
| package commit | `1f1f4d6` |

Each machine ran one contiguous segment of the chain:

| machine | configurations | wall time |
|---|---|---|
| s1 | rows 00-01 and the ablations | 5 h 45 min |
| s2 | rows 01-04 | 4 h 19 min, ended incomplete (see below) |
| s3 | rows 04-08 and x1 | 6 h 14 min |
| s4 | rows 08-18, x2 and x3 | 5 h 34 min |

The stored values are the measurement code's runs of 2026-10-05 behind the report's cumulative table. The stored
pair of x3 comes from a separate run of the same day with its own row 08.

How each kind of pair is computed:

| kind | statistic | interval |
|---|---|---|
| step, block, control | mean over inputs of the per-input difference (later row minus earlier row) | 95 % bootstrap over inputs (10,000 resamples, seed 0) |
| ablation | median of the per-input differences, as in the report's one-at-a-time removals (Appendix A) | 95 % object bootstrap |

The stored point estimate is computed on the same inputs. Both rows of every step pair were run on the same
machine.

**Machine s2.** Row 01 stopped after 20 of 80 inputs. The stored log of that process ends with "Fatal Python error:
Segmentation fault". At that moment the texture flow was sampling input 21 and the CPU unwrapping threads (xatlas)
of the tiled mesh path were running. The log does not identify the faulting thread, and the cause is not
determined. Rows 02 to 04 on s2 completed.

How the pairs that involve row 01 were formed:

- 00 -> 01 uses row 01 from s1 (80 inputs).
- 01 -> 02 was compared within s2, on the 20 inputs that row 01 completed there (n = 20).

Every other pair uses n = 80.

**Outcome of the registered criterion.** The criterion requires, for every adjacent pair, the same sign as the
measurement code **and** a 95 % CI that contains its point estimate. It is **not met**:

| check | result |
|---|---|
| same sign | 18 / 18 pairs |
| CI contains the stored value | 15 / 18 pairs |

Step pairs:

| step | lever(s) added | package, s [95 % CI] | measurement code, s | sign | CI contains |
|---|---|---|---|---|---|
| 00 -> 01 | tiled mesh path | -106.0 [-136.5, -81.7] | -116.1 | yes | yes |
| 01 -> 02 (n = 20) | mesh_prep_worker_thread | -0.49 [-2.37, +1.44] | -0.67 | yes | yes |
| 02 -> 03 | fixed_bake_launch_config | -2.11 [-3.55, -0.75] | -1.91 | yes | yes |
| 03 -> 04 | fast_gpu_simplify | -0.46 [-2.13, +1.21] | -0.71 | yes | yes |
| 04 -> 05 | gpu_bvh_build | -0.96 [-2.36, +0.38] | -1.66 | yes | yes |
| 05 -> 06 | adaptive_remesh | +1.79 [+0.34, +3.22] | +2.83 | yes | yes |
| 06 -> 07 | lean_glb_writer | -1.29 [-2.43, -0.13] | -2.68 | yes | **no** |
| 07 -> 08 | gpu_chart_segmentation | -30.66 [-33.95, -27.44] | -30.49 | yes | yes |
| 08 -> 09 | fa3_texture_flow | -1.61 [-1.95, -1.31] | -1.87 | yes | yes |
| 09 -> 10 | texture_kv_reuse | -0.151 [-0.179, -0.123] | -0.158 | yes | yes |
| 10 -> 11 | shared_conditioning | -0.369 [-0.393, -0.345] | -0.409 | yes | **no** |
| 11 -> 12 | fa3_fine_shape_flow | -2.89 [-3.49, -2.35] | -2.84 | yes | yes |
| 12 -> 13 | texture_side_stream | -0.067 [-0.090, -0.044] | -0.075 | yes | yes |
| 13 -> 14 | structure_student | -1.13 [-1.72, -0.67] | -0.92 | yes | yes |
| 14 -> 15 | texture_projection_reuse | -0.152 [-0.180, -0.126] | -0.264 | yes | **no** |
| 15 -> 16 | coarse_shape_student, fine_shape_student | -9.13 [-10.28, -8.07] | -9.07 | yes | yes |
| 16 -> 17 | fold_lora_fine_shape | -0.157 [-0.215, -0.100] | -0.167 | yes | yes |
| 17 -> 18 | texture_student (Ours) | -3.59 [-4.11, -3.11] | -3.58 | yes | yes |

All three misses are small levers:

| step | package vs stored | stored value outside the package's CI by | row median |
|---|---|---|---|
| 06 -> 07 | -1.29 s vs -2.68 s | 0.25 s | 54 s |
| 10 -> 11 | -0.369 s vs -0.409 s | 0.016 s | 17.5 s |
| 14 -> 15 | -0.152 s vs -0.264 s | 0.084 s | 15 s |

Block pairs (4 / 4 meet sign and CI):

| block | package, s [95 % CI] | measurement code, s |
|---|---|---|
| A: mesh post-processing (00 -> 08) | -140.6 [-171.6, -115.9] | -150.9 |
| B: flow engineering (08 -> 13) | -5.09 [-6.04, -4.24] | -5.34 |
| C: distillation (13 -> 18) | -14.15 [-16.02, -12.44] | -14.00 |
| whole chain (00 -> 18) | -159.8 [-191.0, -134.7] | -170.3 |

Blocks A and the whole chain pair rows from different machines (s1 against s3 and s4).

Row medians, package vs measurement code:

| row | package | measurement code |
|---|---|---|
| 00 Pixal3D re-run | 122.8 s | 138.4 s |
| 01 | 54.1 s | 55.3 s |
| 08 | 19.10 s | 19.05 s |
| 13 | 15.64 s | 15.39 s |
| 16 | 7.24 s | 7.23 s |
| 18 Ours | 3.997 s | 4.005 s |

Row 00 is dominated by Pixal3D's CPU UV unwrapping, so it depends on the machine's CPU. The dedicated timing run of
Pixal3D (official) gave 133.0 s.

Controls (mean over inputs of the difference row minus control):

| control | package, s [95 % CI] | measurement code, s | sign | CI contains |
|---|---|---|---|---|
| x1: row 08 without the superseded levers | -0.012 [-0.032, +0.007] | +0.234 | no | no |
| x2: row 13 with fold_lora_fine_shape | -0.006 [-0.030, +0.017] | -0.017 | yes | yes |
| x3: row 08 without adaptive_remesh | -0.469 [-0.544, -0.399] | -0.833 | yes | no |

- x1: both values are within 0.25 s of zero.
- x3 is the effect of the adaptive remesh in the final mesh path. The measurement code's own interval for x3, with
  the same bootstrap, is [-1.50, -0.44] s, and it contains the package's value.

Ablations on Ours (median of the per-input difference "without the lever" minus Ours; object bootstrap):

| ablation | package, s [95 % CI] | measurement code, s | sign | CI contains |
|---|---|---|---|---|
| without mesh_prep_worker_thread | +0.038 [+0.029, +0.051] | +0.034 | yes | yes |
| without fa3_texture_flow | +0.148 [+0.101, +0.228] | +0.129 | yes | yes |
| without texture_kv_reuse | +0.027 [+0.011, +0.040] | +0.006 | yes | no |
| without texture_projection_reuse | +0.070 [+0.054, +0.093] | -0.008 | **no** | no |

Removing texture_projection_reuse costs the package 0.07 s, where the measurement code found no cost. In the
chain, adding the lever saves 0.15 s in the package against 0.26 s in the measurement code.

**Conclusion.**

- **Registered criterion.** The criterion for the cumulative rows is not met: the sign agrees on all 18 adjacent
  pairs, but the package's interval misses the measurement code's value on three of them, all small levers
  (lean_glb_writer, shared_conditioning, texture_projection_reuse). All four blocks agree.
- **Final configuration.** Ours runs in 3.997 s against the measurement code's 4.005 s.
- **Controls and ablations.** Of the controls, x1 differs in sign at magnitudes below 0.25 s and x3 misses the
  package's interval. Of the ablations, texture_kv_reuse misses the interval, and texture_projection_reuse differs
  in sign.
- **The report.** The report's cumulative table (Table 8), block factors (Table 2) and one-at-a-time removals (Appendix A) keep the measurement code's
  values. The differences of the public package are the ones documented in this section.

