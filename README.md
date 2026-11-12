# Sprint3D

Sprint3D (Python package `fastpixal3d`) turns a single image into a textured 3D mesh (GLB) in about four seconds. It is built on
[Pixal3D](https://github.com/TencentARC/Pixal3D) and runs the same model family:

- four small distilled *students* replace Pixal3D's slow sampling stages;
- a set of system-level changes (the *levers*) remove most of the remaining time: overlapped
  execution, reuse of conditioning and attention keys, faster GPU kernels for mesh simplification,
  BVH construction and UV charting, and a lean GLB writer.

End-to-end time from reading the image to the closed GLB file, on one NVIDIA H200 GPU, both with the export
settings of Pixal3D's own script (1,000,000 faces, 4096 x 4096 texture):

| method | end-to-end time |
|---|---|
| Pixal3D (official) | 133.0 s |
| this package | 4.05 s |

These are median times over the 80 timing inputs, with the first 3 excluded as warm-up, measured with the
protocol of [docs/TIMING.md](docs/TIMING.md).
The quality evaluation and all other numbers are in the report.

## Status

- Inference: parity re-run with the current package on the 80 timing inputs ([docs/PARITY.md](docs/PARITY.md)): support and decoder-input latents identical on 80 / 80, end-to-end median -0.15 % (criterion +-5 %); face count (74 / 80 within +-1 %) and texture (77 / 80) within the measurement code's run-to-run variation except the inputs listed there.
- Evaluation: reproduces the Base (Pixal3D) and Ours rows of the report's validation-set table (Table 10 (a)) exactly. The report's main comparison (Table 4) applies the same protocol to the Toys4K test part, whose object list is not in this package; the list of the development part (Table 10 (b)) is ([docs/EVALUATION.md](docs/EVALUATION.md)).
- Cumulative measurement: reproduced on the 80 timing inputs, and the pre-registered criterion is not met (the sign matches on 18 / 18 adjacent rows, the 95 % CI contains the measured value on 15 / 18); all 4 blocks match, and the final row gives 3.997 s against the measured 4.005 s ([docs/PARITY.md](docs/PARITY.md)).
- Other experiments (TRELLIS.2 port, baselines): ported, checked on one input; full runs pending.
- Training: trainers and cache builders of all four students; the shape students' consistency trainer reproduces the original training code bit for bit on the CPU tests and passed a GPU smoke run on 2026-11-04 (coarse: 6 updates, resume to 8, and 4 processes; fine: 3 updates; both convert to release files of the released size; [docs/TRAINING.md](docs/TRAINING.md)); no full retraining has been run with it.
- Licence: Apache License 2.0 for the code in this repository (see the end of this page).

## Requirements

- Linux with an NVIDIA GPU and CUDA 12.8.
- The reported numbers use a Hopper GPU (sm_90). FlashAttention-3 needs sm_90; on other GPUs the package
  falls back to FlashAttention-2 (`--fa2`), which gives the same results up to rounding but is slower.
- Python 3.10 and PyTorch 2.7.0.
- Several source-built CUDA extensions (FlexGEMM, CuMesh, o-voxel, nvdiffrast, FlashAttention-3) and two patched
  native extensions from this repository.

## Installation

```bash
git clone --recursive https://github.com/AlayaLab/sprint3d.git && cd sprint3d
bash env/install.sh                       # conda environment, pinned packages, source builds, patched extensions
```

The installation takes 1 to 1.5 hours with 4 CPU cores and about 30 GB of disk. It works with conda, mamba or
micromamba, and `ENV_PREFIX=/path bash env/install.sh` puts the environment in a directory; when a step fails,
running the script again resumes at that step. `--recursive` fetches all pinned upstream sources (2.0 GB); only
`third_party/Pixal3D` is needed to run inference.

| file | what it is |
|---|---|
| [`env/environment.yml`](env/environment.yml) | the environment |
| [`env/requirements-runtime.lock`](env/requirements-runtime.lock) | the exact package versions of the measured environment |
| [`env/build_fa3.sh`](env/build_fa3.sh) | builds FlashAttention-3 at the pinned commit |
| [`env/Dockerfile`](env/Dockerfile) | the same steps as a container recipe |

[docs/INSTALL.md](docs/INSTALL.md) describes every step, including how to build the patched native extensions
with `native/build_native.sh`.

## Weights

The student adapters are distributed by this project from the Hugging Face model repository
`AlayaLab/sprint3d`, pinned to one commit in
[`weights/sources.json`](weights/sources.json): the four deployed students (about 5.2 GB) and one adapter used only
by the experiments. Pixal3D and the auxiliary models come from their own repositories at pinned revisions.
Pixal3D is gated in the EU and RMBG-2.0 is gated; accept their terms first. The package reads every model from the
directory these scripts fill (`weights/`, or `$FASTPIXAL3D_WEIGHTS`); RMBG-2.0 is needed only for input images
without an alpha channel.

```bash
python scripts/download_weights.py         # students from AlayaLab/sprint3d, upstream models at pinned revisions
python scripts/verify_weights.py           # size and sha256 of every file
```

[docs/WEIGHTS.md](docs/WEIGHTS.md) lists every file, its revision and its licence.

## Quick start: image to GLB

```bash
python scripts/infer.py --image third_party/Pixal3D/assets/images/4_img.png --output out/4_img.glb \
    --students weights/students --pixal3d weights/pixal3d --timing-json out/timing.json
```

- The GLB uses the export settings of Pixal3D's own script (1,000,000 faces, 4096 x 4096 texture).
- The Pixal3D repository includes 19 example images under `assets/images`.
- Next to the GLB, `infer.py` writes `<stem>.json` with the camera used for the input.
- `--timing-json` writes per-stage times.
- `--list inputs.jsonl --out-dir out/` processes many inputs in one process.
- The first request of a process is slower: it compiles kernels and tunes the sparse convolutions. Use
  `--warmup 1` (the default) before timing.

To produce the reference output, run Pixal3D's own script, unmodified and at its defaults:
`python scripts/run_official.py --image <image> --output <out>.glb`.

`bash scripts/check.sh` runs the CPU tests and one image through `infer.py` and prints a short summary;
use it to check an installation.

## Further documentation

| topic | where |
|---|---|
| timing protocol behind the numbers above (`scripts/time_e2e.py`, `scripts/analyze_timing.py`) | [docs/TIMING.md](docs/TIMING.md) |
| evaluation of the report's tables (`eval/`); object lists with checksums and licences; no data is distributed | [docs/EVALUATION.md](docs/EVALUATION.md), [docs/DATA.md](docs/DATA.md) |
| retraining the four students (`data/`, `train/`) | [docs/TRAINING.md](docs/TRAINING.md) |
| the cumulative measurement, ablations, TRELLIS.2 port and baselines (`experiments/`) | [docs/REPRODUCE_PAPER.md](docs/REPRODUCE_PAPER.md) |
| the speed levers: flags of `fastpixal3d.config.Levers`, their output class and their row in the report | [docs/LEVERS.md](docs/LEVERS.md) |
| agreement between this package and the internal measurement code (criteria registered before the check) | [docs/PARITY.md](docs/PARITY.md) |

## Repository layout

```
src/fastpixal3d/     inference package (pipeline, stages, students, mesh, export, runtime, official reference)
native/              patch series on pinned upstream commits (CuMesh, cubvh) and their build script
scripts/             infer.py, run_official.py, time_e2e.py, analyze_timing.py, parity_check.py, parity_run.sh,
                     download_weights.py, verify_weights.py
eval/                evaluation suite
data/                object selection scripts and object lists (no data)
train/               caches and trainers of the four students (trajectory and consistency distillation)
experiments/         cumulative measurement, ablations, TRELLIS.2 port, baselines
weights/             sources.json (pinned weight sources and checksums)
env/                 environment files, install script, FlashAttention-3 build, Dockerfile recipe
docs/                INSTALL, WEIGHTS, TIMING, EVALUATION, DATA, TRAINING, LEVERS, PARITY, REPRODUCE_PAPER
tests/               CPU tests and one GPU smoke test (pytest -q)
third_party/         upstream repositories at pinned commits (git submodules)
```

## Release contents and data handling

* **Checkpoint loading:** the deployed students are safetensors files, verified by size and sha256 against
  `weights/sources.json`. Every `torch.load` in this package uses restricted deserialization
  (`weights_only=True`) by default; the loaders of training checkpoints, training records and the Uni3D / ULIP-2
  evaluation checkpoints fall back to full unpickling only when `FASTPIXAL3D_TRUST_CHECKPOINT=1` is set, and say
  so in their error message. PyTorch 2.7.0 is pinned (2.6 or newer includes the fix for CVE-2025-32434).
* **Pickle files:** `data/select/` reads the intermediate pickles that its own earlier steps wrote, and the
  evaluation and cache builders read mesh dumps produced by the evaluation data preparation. Load only files you
  produced yourself or obtained from a trusted source.
* **Configuration parsing:** YAML is read with `yaml.safe_load`; command-line overrides are parsed as values, never
  evaluated as code.
* **Network behaviour:** no telemetry and no hard-coded endpoints in the inference path. Downloads happen only in
  the scripts and steps that exist for that purpose: `scripts/download_weights.py` and `eval/fetch_metric_models.py`
  (public Hugging Face and GitHub release files, pinned revisions and sha256 checked), `torch.hub` for the NAF
  feature upsampler at a pinned commit (as the upstream Pixal3D loader does), and `env/install.sh` (public PyPI,
  the official PyTorch and NATTEN wheel indexes, and pinned public GitHub commits).
* **Processes:** subprocesses are started in list form without a shell (`git rev-parse` for provenance, `nvidia-smi`
  for the timing sampler, the project's own scripts in `experiments/run_arms.py` and the GPU smoke test).
  `eval/tests/test_frames_projection.py` executes a function extracted from the pinned Pixal3D source.
* **Archive contents:** source, configurations, object lists, tests and documentation only; no git history, no
  credentials, no weights, no datasets, no renders. `.gitignore` excludes `.env*`, `.netrc`, weight and cache files
  and the download directories under `weights/`.
* **Licence files:** `LICENSE`, `NOTICE` and `THIRD_PARTY_LICENSES.md` are in the archive and declared in
  `pyproject.toml` (`license-files`) for source and wheel distributions.

## License

Sprint3D's own code (`src/`, `scripts/`, `eval/`, `data/`, `train/`, `experiments/`, `tests/`, `docs/`, the patch
files in `native/`) is released under the Apache License 2.0 (`LICENSE`, Copyright 2026 Alaya Lab); `NOTICE`
carries the attribution notices required by the licence. The student weights are proposed for release under CC BY-NC 4.0
(pending the lab's open-source review), because their training data includes assets under non-commercial and
research-only licences (`docs/WEIGHTS.md`, `docs/DATA.md`).

Running the full system also involves third-party components under **non-commercial** terms, whatever
licence this repository's own code has:

- nvdiffrast (texture baking and evaluation rendering): NVIDIA Source Code License;
- the cubvh code in the patched native extension built from `native/cubvh`: covered by its
  `LICENSE_NVIDIA` (NVIDIA Source Code License) in addition to MIT;
- RMBG-2.0 (background removal, used only for inputs without an alpha channel): CC BY-NC 4.0;
- the DINOv3 weights (image conditioning): DINOv3 License.

The student weights were trained on data that includes assets under non-commercial and research-only
licences. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for all third-party components, their licences and
the full texts in `LICENSES/`.
