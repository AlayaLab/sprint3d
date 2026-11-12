# Evaluation

This is the evaluation behind the report's single-image tables: normal maps of the generated mesh compared with the
ground truth in the input-image frame (the protocol of the Pixal3D paper, Sec. 4.1), plus the appendix metrics
(single-view 3D geometry, CLIP / CLIP-N, ULIP-2, Uni3D).  The protocol is written out in
`eval/protocol/protocol.json`; the declared conventions are in `eval/metrics/conventions.py`.

The report applies this protocol to three sets:

| Set | Tables of the report | Object list |
|---|---|---|
| validation set, 400 objects | Table 10 (a) | `eval/lists/validation_400.csv` |
| Toys4K test part, 2,653 objects (the main comparison) | Tables 4, 11 and 12 | not in this repository |
| Toys4K development part, 500 objects | Table 10 (b) | `eval/lists/toys4k_dev_500.csv` |

The sections below follow the validation set.  For a Toys4K list, set `dataset.list` to the list, `dataset.list_sha256`
to its checksum or `null`, and `dataset.source_dirs.toys4k` to its directory (docs/DATA.md).  The report's Toys4K numbers
come from the internal evaluation code; these lists have not been run through this package.  The report calls Pixal3D (official) *Base (Pixal3D)* and this package
*Ours*.  The name `table1` in file names and options is kept from an earlier numbering of the report; its columns are
those of Table 1 of the Pixal3D paper.

```
eval/
  run_eval.py               evaluate GLBs against the ground truth -> per_view.json, per_object.json, corpus.json, tables/
  make_manifest.py          manifest of one arm from a directory of GLBs (+ camera sidecars)
  make_table.py             LaTeX rows of the normal-map table from per_object.json (no manual editing)
  fetch_metric_models.py    third-party metric models at pinned revisions, with sha256 checks
  config.example.yaml       configuration (every machine-specific path)
  requirements.txt          Python packages
  metrics/                  metric implementations
  frame_maps/               fixed per-method frame maps of external baselines
  protocol/protocol.json    the evaluation protocol
  lists/                    validation_400.csv (Table 10 (a)), toys4k_dev_500.csv (Table 10 (b)), timing_80.csv (timing run); see docs/DATA.md
  tests/                    CPU unit tests (pytest -q eval/tests) and a GPU smoke test
```

## 1. Install

Python 3.10, PyTorch 2.7.0 (CUDA 12.8), nvdiffrast 0.4.0 and the packages of `eval/requirements.txt`; a CUDA GPU is
needed for rendering (nvdiffrast).  nvdiffrast is under the NVIDIA Source Code License (non-commercial use).

```bash
pip install -r eval/requirements.txt
pytest -q eval/tests              # CPU tests; the GPU smoke test runs when a GPU is present
```

## 2. Metric models

```bash
python eval/fetch_metric_models.py --dest /path/to/metric_models                 # everything, about 28 GB
python eval/fetch_metric_models.py --dest /path/to/metric_models --only table1   # normal-map table only (LPIPS backbones)
python eval/fetch_metric_models.py --dest /path/to/metric_models --verify-only   # check an existing download
```

Set `models.dir` of the configuration to the same directory.  Every file is checked against the sha256 in
`eval/metrics/config.py` (`PINS`); Hugging Face files are read offline from the pinned revisions at evaluation time.

| Model | Used for | Source and pinned revision | sha256 (weights) | Licence |
|---|---|---|---|---|
| LPIPS v0.1 linear layers (alex, vgg) | normal-map table, LPIPS | pip `lpips==0.1.4` | alex `df73285e...35c0`, vgg `a78928a0...2868` | BSD-2-Clause |
| AlexNet, VGG16 (ImageNet) | LPIPS backbones | torchvision `alexnet-owt-7be5be79.pth`, `vgg16-397923af.pth` | `7be5be79...ee02`, `397923af...5bf0` | torchvision: BSD-3-Clause (no separate licence for the weights) |
| CLIP ViT-L/14 | CLIP, CLIP-N | `openai/clip-vit-large-patch14` @ `32bd6428` | `a2bf730a...6dcb` (model.safetensors) | MIT (OpenAI CLIP repository); no licence field on the model card |
| OpenCLIP ViT-bigG-14 laion2b_s39b_b160k | ULIP-2 image tower | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` @ `743c27bd` | `db71bd27...ec34` | MIT |
| OpenCLIP EVA02-E-14-plus laion2b_s9b_b144k | Uni3D image tower | `timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k` @ `0162f50c` | `dcdd4372...9d97` | MIT |
| ULIP-2 PointBERT 10k xyzrgb (vit_g) | ULIP-2 | dataset repo `SFXX/ulip` @ `3f33d96e` (linked from the ULIP README) | `a4b5ed97...4adb` | ULIP code: BSD-3-Clause; no licence stated for the checkpoint |
| Uni3D-g | Uni3D | `BAAI/Uni3D` @ `3d8233b7`, `modelzoo/uni3d-g/model.pt` | `aa1f163b...8c3f` | Uni3D code: MIT; no licence stated on the model card |
| ULIP code | ULIP-2 | github.com/salesforce/ULIP @ `95d480fe` | -- | BSD-3-Clause (PointNeXt part: MIT) |
| Uni3D code | Uni3D | github.com/baaivision/Uni3D @ `64e03c3c` | -- | MIT |
| DINOv2 ViT-L/14 + code | agreement with a reference (not in the normal-map tables) | `dinov2_vitl14_pretrain.pth`, github.com/facebookresearch/dinov2 @ `7764ea0f` | `d5383ea8...f428` | Apache-2.0 |

Full revisions and checksums: `eval/metrics/config.py`.  The two CUDA extensions imported by the ULIP / Uni3D code
(`pointnet2_ops`, `knn_cuda`) are replaced by pure-PyTorch equivalents (`eval/metrics/scorers3d.py`); nothing is compiled.

## 3. Ground truth and configuration

The ground truth (conditioning renders, cameras, mesh dumps) is not distributed; `docs/DATA.md` describes the expected
layout and what is known about how it was produced.  Copy `eval/config.example.yaml` and set at least `dataset.root`
and `models.dir`.

| Key | Meaning |
|---|---|
| `dataset.root` | ground-truth root (one sub-directory per source) |
| `dataset.list` | object list (`lists/validation_400.csv`) |
| `dataset.list_sha256` | optional startup check of the list file; `null` disables it |
| `dataset.source_dirs` | `source` column of the list -> sub-directory of the root |
| `models.dir` | destination of `fetch_metric_models.py`; the model keys below default to its layout |
| `models.hf_home`, `models.torch_home` | caches exported as `HF_HOME` / `TORCH_HOME` |
| `models.offline` | `true`: `HF_HUB_OFFLINE=1`, only the pinned local files are used |
| `models.ulip_code`, `models.ulip2_checkpoint`, `models.uni3d_code`, `models.uni3d_checkpoint` | ULIP-2 / Uni3D code checkouts and checkpoints |
| `models.dinov2_code`, `models.dinov2_weights` | DINOv2 code checkout and weights |
| `evaluation.device` | `cuda` |
| `evaluation.background` | constant background of the full-image PSNR / SSIM / LPIPS (`white`) |
| `evaluation.table1_region` | region of PSNR / SSIM / LPIPS in `tables/table1.md` (`overlap`); both are always stored |
| `evaluation.frame_maps` | per-method frame maps of external baselines |
| `evaluation.reference_arm` | base name of the reference arm for paired statistics (`pixal3d_official`) |
| `evaluation.skip` | metric groups to skip: `2d`, `3d`, `perception`, `vs_reference`, `lpips`, `lpips_vgg`, `dino`, `clip`, `ulip2`, `uni3d` |
| `evaluation.n_boot` | bootstrap resamples (10,000) |

Relative paths are resolved against the directory of the configuration file.  Command-line options of `run_eval.py`
(`--dataset-root`, `--device`, `--table1-region`, `--reference-arm`, `--skip`, ...) override the file.

## 4. Generate the products

One GLB per row of `eval/lists/validation_400.csv`, generated from the conditioning render
`renders_cond/<sha256>/<view:03d>.png` of that row, without a ground-truth camera (condition `official`).  The two
rows that this repository generates are

| Row | Generator | Arm name used below |
|---|---|---|
| Pixal3D (official); *Base (Pixal3D)* in the report | the released Pixal3D `inference.py` at its defaults (1M faces, 4096^2 texture) | `pixal3d_official` |
| Ours | this repository (`scripts/infer.py`; the same export: 1M faces, 4096^2 texture) | `ours` |

An earlier version of the report's table had one more row, "Ours, 200k export": the same pipeline with a 200,000-face
target and a 2048^2 texture; the current table has no such row.  That export setting is not part of this repository, so the row is optional and off by default: it is
evaluated only when a manifest of such GLBs is passed explicitly (arm `ours_200k` in step 5) and printed only with
`--extra-row` (step 6).

The evaluation renders each product with its own inference-time projection, so every GLB needs the camera the
generator used: horizontal FoV `camera_angle_x` (radians, MoGe-2 estimate on the official crop), `distance` and
`mesh_scale`.  Store the products as

```
<glb_dir>/<id>_v<view>.glb
<glb_dir>/<id>_v<view>.json      {"camera_angle_x": ..., "distance": ..., "mesh_scale": ..., "e2e_s": ...}
```

(`<id>` = the list id or the full sha256; `<glb_dir>/<id>_v<view>/model.glb` with `camera.json` next to it also works.)

## 5. Run the evaluation

```bash
python eval/make_manifest.py --dir /path/to/pixal3d_official_glbs --arm pixal3d_official --out manifests/pixal3d_official.json
python eval/make_manifest.py --dir /path/to/ours_glbs             --arm ours             --out manifests/ours.json
python - <<'PY'
import json
rows = sum((json.load(open(f"manifests/{a}.json")) for a in ("pixal3d_official", "ours")), [])
json.dump(rows, open("manifests/table1.json", "w"), indent=1)
PY
python eval/run_eval.py --config eval/config.yaml --manifest manifests/table1.json --out results/table1 \
    --skip vs_reference              # add lpips_vgg,perception,3d to compute the normal-map table only
```

For the optional 200k-export row, write a third manifest (`--arm ours_200k`) and add `"ours_200k"` to the list above.

Outputs in `results/table1/`:

- `per_view.json`: one record per (arm, object, view) with every metric, the inference camera and crop, the
  ground-truth frame, and the code and conventions hashes;
- `per_object.json`: `{arm: {sha256: {metric: value}}}` (views averaged per object; one view per object here);
- `corpus.json`: per-arm summaries, paired statistics against `pixal3d_official@<condition>`, the conventions, the
  code hash and the measured float32 matmul error;
- `tables/*.md`: the normal-map table (both regions), 3D geometry, perception, paired statistics, speed and face counts.

A rerun into the same directory resumes; records computed by other code or under other conventions are never reused
(`python eval/run_eval.py --print-hashes` prints both hashes).  `--tables-only` re-renders the tables of an existing
directory, e.g. with `--table1-region full`.  `--save-renders` writes prediction | ground-truth normal-map pairs.
Rendering runs in exact float32: TF32 is switched off before the first CUDA matmul and the run stops if a probe still
measures TF32-level error.  With all metric groups except `vs_reference`, one object with two 1M-face products takes
about 8 s on one GPU once the models are loaded.

## 6. The normal-map table

```bash
python eval/make_table.py --per-object results/table1/per_object.json \
    --row "Pixal3D (official)=pixal3d_official@official" --row "Ours=ours@official" \
    --e2e "Pixal3D (official)=<s>" --e2e "Ours=<s>" \
    --out-dir results/table1/latex
```

`--row` is given exactly twice (reference, candidate).  The optional 200k-export row is off by default and is printed
only when it is passed explicitly: `--extra-row "Ours, 200k export=ours_200k@official"` (with its own
`--e2e "Ours, 200k export=<s>"`), below a rule, unbolded and in italics.

`table1.tex` is the normal-map table (PSNR / SSIM / LPIPS in the overlap region; the better of Pixal3D (official) and Ours
in bold per column), `table1_full_region.tex` the same rows on the full image (asterisk / dagger = significantly
better / worse than Pixal3D (official): paired per-object sign test, Holm-adjusted over the 13 normal-map metrics,
p < 0.05), `table1_values.csv` every mean with its bootstrap CI and every paired test.  Every cell except the E2E
column comes from `per_object.json`.  The E2E column is the median end-to-end time of the dedicated timing run on the
80 inputs of `eval/lists/timing_80.csv` (warm-up excluded; docs/TIMING.md) and is passed with `--e2e`; it is measured
on different inputs than the quality columns.  The external baselines are printed as "not reported"; the report's baseline rows come from evaluating their outputs ([REPRODUCE_PAPER.md](REPRODUCE_PAPER.md)).

Applied to the per-object results behind the report, `make_table.py` reproduced the tabular body of the validation-set
table and of its full-image variant character for character, given the E2E medians and the 200k-export row passed
with `--extra-row`.  That check dates from 2026-10-03, when this table was the report's main table and its baseline
rows said "not reported".  Its Base (Pixal3D) and Ours rows are now in Table 10 (a) with the same values; the 200k-export
row and the full-image variant for the validation set are no longer printed.

## 7. Metric definitions

| Column | Definition |
|---|---|
| IoU | mask IoU of the predicted and ground-truth normal maps (512^2), in % |
| PSNR / SSIM / LPIPS (main table) | overlap region: PSNR over the pixels covered by both; SSIM (gaussian 11x11, sigma 1.5) map averaged over the overlap; LPIPS AlexNet v0.1 with both images 0.5-gray outside the overlap |
| PSNR / SSIM / LPIPS (appendix) | full 512^2 image, each map composited on a white background |
| Mean / Median | angular error (degrees) over the overlap |
| Mean_B | angular error over overlap pixels within 10 px of the ground-truth mask boundary |
| 11.25 / 22.5 / 30 | % of overlap pixels with angular error below the threshold |

Normal maps are flat (per-face) camera-space normals flipped to face the camera, encoded as rgb = n / 2 + 1/2.  The
ground truth is rendered with the ground-truth camera of the input view; a Pixal3D-family product is rendered with its
own inference-time camera, mapped from the official crop to the raw image by the exact affine of the crop box (no
ground-truth pose, no ICP).  Statistics: objects weighted equally; percentile bootstrap (10,000 resamples, seed
20260918); exact two-sided sign test on per-object differences with ties dropped.

Random sampling (surface points for Chamfer / F-score, point clouds for ULIP-2 / Uni3D) is seeded per (object, view,
repeat) by `stable_seed(salt, "<sha256>|<view>|<repeat>")` in `eval/metrics/geometry.py`, with three fixed salt
strings: `evalv2_3d` (3D metrics), `evalv2_pc` (point clouds of the 3D scorers) and `evalv2_3d_vsA` (agreement with a
reference product).  They are part of the reported numbers and must not be changed: another salt draws other points.

## 8. Agreement with the measurement code

The metric code is a port of the internal evaluation code that computed the report's numbers, with the same libraries,
resolutions, overlap definition, seeds and camera handling.  On 3 validation objects (both Pixal3D (official) and Ours,
the stored GLBs of the report's run) every per-view value of every metric listed above, including CLIP, ULIP-2 and
Uni3D, is identical to the stored value.  The code hash and conventions hash of this version differ from the internal
ones because the code was rewritten for publication.
