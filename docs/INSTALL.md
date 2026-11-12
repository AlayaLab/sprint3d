# Installation and use

The package runs on Linux with an NVIDIA GPU. The measured configuration is a Hopper GPU (sm_90)
with CUDA 12.8; the two FlashAttention-3 levers need sm_90 (see [Other GPUs](#other-gpus)).

## 1. Installation with `env/install.sh`

Needs: Linux x86-64; an NVIDIA driver for CUDA 12.8 (570 or later); `conda`, `mamba` or `micromamba`;
`git`; a C++ compiler that CUDA 12.8 supports (gcc 11 to 14); network access to conda-forge, PyPI,
download.pytorch.org, GitHub, GitLab (Eigen) and whl.natten.org; about 30 GB of disk.

```bash
git clone --recursive https://github.com/AlayaLab/sprint3d.git && cd sprint3d   # with third_party/ (section 3)
bash env/install.sh              # creates the conda environment "fastpixal3d" and builds everything
conda activate fastpixal3d
python scripts/download_weights.py   # section 4
bash scripts/check.sh                # CPU tests, then one example image on the GPU
```

`env/install.sh` takes its settings from the environment:

| variable | default | meaning |
|---|---|---|
| `ENV_NAME` | `fastpixal3d` | name of the conda environment |
| `ENV_PREFIX` | -- | a directory for the environment instead of a name (`ENV_PREFIX=/path/env bash env/install.sh`) |
| `CONDA` | first of `conda`, `mamba`, `micromamba` on `PATH` | the conda frontend |
| `MAX_JOBS` | 4 | parallel compile jobs |
| `TORCH_CUDA_ARCH_LIST` | 9.0 | CUDA architectures to compile for (Hopper) |
| `WORK_DIR` | `build/env` | checkouts and build trees |
| `SKIP_FA3` | -- | `1` on GPUs other than Hopper (the package then uses FlashAttention-2, `--fa2`) |

The script prints the duration of every step. A step that has finished is not repeated when the script
is run again (the record is kept inside the environment), so after a failure fix the cause and run the
same command again; an environment whose creation was interrupted is completed in place. The steps,
with the times of a fresh installation with 4 CPU cores (`MAX_JOBS=4`):

| step | what it does | time |
|---|---|---:|
| environment | the conda packages of `env/environment.yml` (Python 3.10.21, CUDA 12.8 compiler and libraries, git, ninja) and the pip packages of `env/requirements-runtime.lock` (PyTorch 2.7.0+cu128 and the exact versions of the measured environment) | 5 min on a local disk, about 30 min on a network file system |
| `flexgemm` | FlexGEMM, sparse convolution | 5 min |
| `cumesh` | CuMesh, stock (o-voxel imports it) | 11 min |
| `o-voxel` | o-voxel from TRELLIS.2 | 7 min |
| `nvdiffrast` | nvdiffrast | 5 min |
| `pipeline`, `moge`, `utils3d` | MoGe-2 code and its helpers; utils3d with `env/patches/utils3d-moge2-compat.patch` | 2 min |
| `flash_attn`, `natten` | FlashAttention-2 2.8.3 (prebuilt wheel, fetched by its setup script) and NATTEN 0.21.0 (wheel) | 2 min |
| `flash_attn_3` | FlashAttention-3, `env/build_fa3.sh` (Hopper only) | 5 min |
| `native` | the two patched native extensions, `native/build_native.sh` | 22 min |
| final | the Pixal3D submodule, this package with `pytest`, an import check of every module | 2 min |
| total | | 1 to 1.5 h |

Disk: the environment 16 GB (13 GB before the source builds); `build/env` 1.6 GB; the conda package cache
5.7 GB and the pip cache 2.6 GB (outside the repository; `conda clean -a` and `pip cache purge` remove them
afterwards); the repository with its submodules 2.0 GB.

Build tools: `env/environment.yml` does not pin pip, setuptools and wheel. The source builds work with the
versions of the measurement environment (pip 25.3, setuptools 80.9.0, wheel 0.45.1) and with those a fresh
installation received on 2026-10-05 (pip 26.2.1, setuptools 84.0.0, wheel 0.48.0).

Pitfalls:

* `TORCH_CUDA_ARCH_LIST` inherited from the shell (CUDA containers often list every architecture) makes
  every build compile for all of them, which takes much longer; set it to your GPU's architecture.
* CUDA containers may set `PIP_CONSTRAINT` to pin their own PyTorch build, and pip then cannot resolve the
  lock file (`ResolutionImpossible`). `install.sh` ignores it; unset it before running pip in this
  environment yourself.
* conda's CUDA packages keep their headers in `<env>/targets/x86_64-linux/include`. `install.sh` adds that
  directory to `CPATH` (CuMesh needs `cuda.h` there); a manual build needs the same, or `CUDA_HOME`
  pointing at a CUDA 12.8 toolkit.
* Only the headless OpenCV build is installed. With `opencv-python` as well, its `cv2` can replace the
  headless one and fail to import without `libGL` / `libGLX`.
* Creating the environment on a network file system is slow: most of its time is spent unpacking and
  linking about 13 GB of files.

## 2. The steps one by one

For another environment manager or a manual installation: an environment with Python 3.10 and the pinned
packages, then the source builds against the installed PyTorch (`--no-build-isolation`), in this order.

```bash
conda env create -f env/environment.yml && conda activate fastpixal3d   # or: pip install -r env/requirements-runtime.lock
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0 CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
```

Pin the commits; later upstream versions are untested.

| package | source | commit |
|---|---|---|
| flex_gemm 1.0.0 | `https://github.com/JeffreyXiang/FlexGEMM` | `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| cumesh 0.0.1 (stock, needed by o_voxel) | `https://github.com/JeffreyXiang/CuMesh` | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| o_voxel 0.0.1 | `https://github.com/microsoft/TRELLIS.2`, directory `o-voxel/` | `de38fdd4082638cb4e6d2285f2e215280bd3f485` |
| nvdiffrast 0.4.0 | `https://github.com/NVlabs/nvdiffrast` | `253ac4fcea7de5f396371124af597e6cc957bfae` |
| pipeline 1.0.0 | `https://github.com/EasternJournalist/pipeline` | `866f059d2a05cde05e4a52211ec5051fd5f276d6` |
| moge 2.0.0 | `https://github.com/microsoft/MoGe` | `07444410f1e33f402353b99d6ccd26bd31e469e8` |
| utils3d 0.0.2 | `https://github.com/EasternJournalist/utils3d` + `env/patches/utils3d-moge2-compat.patch` | `9a4eb15e4021b67b12c460c7057d642626897ec8` |
| flash_attn 2.8.3 (FlashAttention-2) | PyPI | `flash_attn==2.8.3` |
| natten 0.21.0 | `https://whl.natten.org`, build for torch 2.7.0 / CUDA 12.8 | `natten==0.21.0+torch270cu128` |
| flash_attn_3 3.0.0b1 (FlashAttention-3) | `https://github.com/Dao-AILab/flash-attention`, directory `hopper/` | tag `v2.8.3.post1` (`a8aa52b`) |

```bash
build() {  # build <url> <commit> [subdirectory]
  d=$(mktemp -d) && git clone "$1" "$d" && git -C "$d" checkout "$2" \
    && git -C "$d" submodule update --init --recursive && pip install --no-build-isolation --no-deps "$d/${3:-}"
}
build https://github.com/JeffreyXiang/FlexGEMM   6dd94a859c26ee8246888502eada3dd8ad85532e
build https://github.com/JeffreyXiang/CuMesh     12289e1062f0603f2f0d0771b02e1395d247f26f
build https://github.com/microsoft/TRELLIS.2     de38fdd4082638cb4e6d2285f2e215280bd3f485 o-voxel
build https://github.com/NVlabs/nvdiffrast       253ac4fcea7de5f396371124af597e6cc957bfae
build https://github.com/EasternJournalist/pipeline 866f059d2a05cde05e4a52211ec5051fd5f276d6
pip install --no-deps "git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8"
d=$(mktemp -d) && git clone https://github.com/EasternJournalist/utils3d "$d" \
  && git -C "$d" checkout 9a4eb15e4021b67b12c460c7057d642626897ec8 \
  && git -C "$d" apply "$PWD/env/patches/utils3d-moge2-compat.patch" && pip install --no-deps "$d"
pip install --no-deps --no-build-isolation flash_attn==2.8.3
pip install --no-deps "natten==0.21.0+torch270cu128" -f https://whl.natten.org
```

The measurement environment used utils3d at commit `9a4eb15` with a local compatibility patch for MoGe-2, shipped
as `env/patches/utils3d-moge2-compat.patch`: it adds the `utils3d.pt` alias and `depth_map_to_point_map`, which
MoGe-2 calls. Without it the camera estimate fails with `No module named 'utils3d.pt'`.

FlashAttention-3 (Hopper only): `env/build_fa3.sh` builds `hopper/` of flash-attention v2.8.3.post1
(`a8aa52b`, CUTLASS `dc48179`). By default it compiles only the kernels the pipeline calls (forward, bf16, head
dimension 128; 5 minutes with 4 jobs), as in the measured build; `FULL=1` builds everything.

```bash
PYTHON=$(which python) MAX_JOBS=4 bash env/build_fa3.sh
python -c "import flash_attn_interface"      # the module the package looks for
```

Then the patched native extensions (below), `git submodule update --init third_party/Pixal3D` and
`pip install -e ".[test]"`. `env/Dockerfile` repeats the steps of `install.sh` as a container recipe.

### Patched native extensions

Two levers need patched builds of CuMesh (`fast_gpu_simplify`) and cubvh (`gpu_bvh_build`).
`native/build_native.sh` clones the pinned upstream commits, renames the modules to
`fastpixal3d_cumesh` and `fastpixal3d_cubvh` (so they coexist with the stock `cumesh` that
o_voxel imports), applies the patch series under `native/<package>/patches/` and installs both (22 minutes
with 4 jobs):

```bash
PYTHON=$(which python) MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0 native/build_native.sh
# or into a separate directory:  native/build_native.sh --target /path/to/site  (then add it to PYTHONPATH)
```

The upstream licenses are under `native/cumesh/` and `native/cubvh/`. cubvh carries the NVIDIA
Source Code License (`native/cubvh/LICENSE_NVIDIA`, non-commercial use); nvdiffrast is under the
NVIDIA Source Code License as well.

Only to reproduce the tiled-mesh-path rows of the cumulative measurement (configurations with
`gpu_chart_segmentation` off, see [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md)): `native/legacy/build_legacy.sh`
builds three host BVH variants and a single-thread xatlas the same way (patch series, upstream licenses
and the list of which configuration needs which build in `native/legacy/README.md`).

## 3. Pixal3D source and the other pinned sources

The pipeline imports the unmodified Pixal3D code (`TencentARC/Pixal3D` at
`28efad66fdcbd8174a8538d9baf71fe34fe4b6d2`) from `third_party/Pixal3D`. Another checkout of the same
commit can be used with `--pixal3d-src` or `FASTPIXAL3D_PIXAL3D_SRC`.

All upstream sources are git submodules under `third_party/`, at the commits the results were obtained
with. Only `third_party/Pixal3D` is needed to run inference (`git submodule update --init
third_party/Pixal3D`, 0.16 GB); `env/install.sh` builds from its own clones of the pinned commits (in
`build/env`), so the submodule checkouts stay unmodified.

| submodule | upstream | commit | used for |
|---|---|---|---|
| `third_party/Pixal3D` | `https://github.com/TencentARC/Pixal3D` | `28efad66fdcbd8174a8538d9baf71fe34fe4b6d2` | the pipeline (inference, training) |
| `third_party/TRELLIS.2` | `https://github.com/microsoft/TRELLIS.2` | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` | the TRELLIS.2 port and baseline (`--trellis2-src`); its `o-voxel/` is identical to that of the build commit `de38fdd` |
| `third_party/CuMesh` | `https://github.com/JeffreyXiang/CuMesh` | `12289e1062f0603f2f0d0771b02e1395d247f26f` | base of the patched simplification (nested: cubvh `ce92267`, Eigen `e63d9f6`) |
| `third_party/cubvh` | `https://github.com/JeffreyXiang/cubvh` (branch `trellis.2`) | `ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70` | base of the patched GPU BVH build |
| `third_party/flash-attention` | `https://github.com/Dao-AILab/flash-attention` | `a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2` (`v2.8.3.post1`) | FlashAttention-3 (`hopper/`) |
| `third_party/ULIP` | `https://github.com/salesforce/ULIP` | `95d480fe2b16c06d0558c60b5cfea981b4cdc8eb` | code of the appendix metric ULIP-2 (`eval/fetch_metric_models.py` fetches the same commit) |
| `third_party/Uni3D` | `https://github.com/baaivision/Uni3D` | `64e03c3c42c196e8cb5ed03857810af9fc9ac39c` | code of the appendix metric Uni3D (likewise) |
| `third_party/TripoSG` | `https://github.com/VAST-AI-Research/TripoSG` | `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` | baseline (`--triposg-src third_party/TripoSG`) |

`git clone --recursive` fetches all of them with their own submodules (CUTLASS, composable_kernel, Eigen;
2.0 GB). After a plain `git clone`, `git submodule update --init` fetches only the top level (1.3 GB), which is
all that this repository uses.
Hunyuan3D-2.1, the third baseline, is not a submodule; `experiments/baselines/README.md` gives its commit.

## 4. Weights

`scripts/download_weights.py` downloads every file pinned in `weights/sources.json` into one directory,
the *weights directory*: `weights/` by default, or `$FASTPIXAL3D_WEIGHTS` (both scripts also take `--root`).
The package reads every model from there; it is the only copy needed:

| subdirectory | model | how the package uses it |
|---|---|---|
| `students/` | the four student adapters (`students.json` and `.safetensors` files) | default of `--students` / `FASTPIXAL3D_STUDENTS` |
| `pixal3d/` | Pixal3D flows and decoders (`TencentARC/Pixal3D`; gated in the EU) | default of `--pixal3d` / `FASTPIXAL3D_PIXAL3D` |
| `dinov3/`, `moge2/`, `rmbg2/`, `naf/` | DINOv3 ViT-L/16, MoGe-2 ViT-L, RMBG-2.0 (gated), NAF | given to Pixal3D's own loaders in place of their hub ids |

`--students` and `--pixal3d` (or the variables) point elsewhere when needed; `--pixal3d` also accepts a
Hugging Face repository id. Pixal3D's `inference.py` loads the four auxiliary models by hub id at their
latest revision, into `HF_HOME` and `TORCH_HOME`; the package passes the same loaders the files of the
weights directory instead (`src/fastpixal3d/auxiliary.py`), so nothing is downloaded twice and the pinned
revisions are used. A model whose files are not all in the weights directory is loaded as Pixal3D's code
loads it, by hub id through `HF_HOME` / `TORCH_HOME`. NAF's model code, which is not a weight file, is
fetched once by `torch.hub` at the pinned commit into `TORCH_HOME/hub`.

Access: the DINOv3 weights come from a public mirror (the gated `facebook` repository named in Pixal3D's
`pipeline.json` is not used), and MoGe-2 and NAF are public. RMBG-2.0 is gated and only needed for input
images without an alpha channel: without `rmbg2/` the pipeline still loads and runs RGBA inputs, and an
RGB input stops with a message saying how to get it. With all files present, `--offline` (or
`HF_HUB_OFFLINE=1`) runs without contacting the hub. [WEIGHTS.md](WEIGHTS.md) lists every file.

The format of `students.json` is described in [WEIGHTS.md](WEIGHTS.md); the loader checks every file's
sha256 and the adapter rank and target modules against the base model. The adapters of the fine-shape and
texture students are folded into the bf16 weights of their base flows when the model is loaded; the
structure and coarse-shape adapters stay separate.

## 5. Running

```bash
python scripts/infer.py --image input.png --output out/obj_v00.glb \
    --students /path/to/students --timing-json out/timing.json
```

The export uses the settings of Pixal3D's `inference.py`: a target of 1,000,000 faces and a 4096 x 4096
texture (`fastpixal3d.config.EXPORT`), as in all comparisons.

Useful options:

* `--list inputs.jsonl --out-dir out/` processes many inputs in one process; each line is a JSON
  object with `id` (names the outputs) and `path`.
* `--warmup N` runs N untimed requests on the first input first (default 1). The first request of
  a process compiles kernels, runs the sparse-convolution autotuner and captures the CUDA graph of
  the structure student, so it is several times slower than the following ones.
* `--autotune-cache FILE`: the FlexGEMM autotune cache. Point it at a copy of a cache from an
  earlier run to skip most of the autotuning.
* `--disable LEVER` (repeatable) switches a lever off, see [LEVERS.md](LEVERS.md).
* `--threads N` (default 8) sets the CPU thread count (OpenMP and PyTorch).
* `--seed`, `--object-id`: the fine-shape noise is seeded from the object id and the seed; the
  default object id is the output file stem.

`scripts/infer.py` sets the process environment of the measured configuration before PyTorch is
imported: `OMP_NUM_THREADS` (and the MKL/OpenBLAS equivalents) to `--threads`,
`ATTN_BACKEND=flash_attn`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (float32 cuBLAS matmuls,
including the adapter folding at load time, may use TF32, as in the measured configuration).
`--tf32 off` removes that variable for full float32 matmuls; outputs then differ slightly from the
measured configuration.

From Python:

```python
from fastpixal3d import Config
from fastpixal3d.pipeline import FastPixal3D, configure_process

configure_process(threads=8)                     # before CUDA is initialised
model = FastPixal3D.load(Config())               # weights from the environment variables
result = model.run("input.png", "out/obj_v00.glb", object_id="obj")
print(result["timings"]["e2e_s"], result["stats"]["output_faces"])
```

### Pixal3D (official) and timing

`scripts/run_official.py` runs Pixal3D's own `inference.py` unmodified at its defaults (the reference of
the comparisons) and writes the same `<stem>.glb` / `<stem>.json` pair as `scripts/infer.py`; what it
calls is documented at the top of `src/fastpixal3d/official.py`. The timing protocol and its scripts
(`scripts/time_e2e.py`, `scripts/analyze_timing.py`, `scripts/parity_run.sh`) are described in
[TIMING.md](TIMING.md).

### Other GPUs

FlashAttention-3 needs sm_90. On other GPUs the FA3 levers fall back to FlashAttention-2
automatically; `--fa2` forces the fallback. Outputs agree up to rounding, the run is slower. Set
`TORCH_CUDA_ARCH_LIST` to the GPU's architecture for the source builds.

## 6. Output format

For every input, next to each other:

* `<stem>.glb` -- the textured mesh: one primitive with positions, normals and UVs, a PBR material
  with a base-colour texture (RGBA) and a metallic-roughness texture, both PNG, 4096 x 4096. The
  mesh is in the coordinate frame of Pixal3D's own GLB export.
* `<stem>.json` -- the inference-time camera of that input, needed to render the output from the
  input view (for example by the evaluation). With `--list`, `<stem>` is the input's `id`
  (evaluation inputs use `<id>_v<view>`); with `--image`, it is the stem of `--output`.
  `scripts/run_official.py` and `scripts/time_e2e.py` write the same file.

  ```json
  {"object_id": "obj_v00", "image": "input.png",
   "camera": {"camera_angle_x": 0.5103, "distance": 1.9169, "mesh_scale": 1.0},
   "camera_angle_x": 0.5103, "distance": 1.9169, "mesh_scale": 1.0,
   "fov_source": "moge2", "prepared_size": [1066, 1066]}
  ```

  `camera_angle_x` is the horizontal field of view in radians, estimated by MoGe-2 on the
  preprocessed image; `distance` is the camera distance from the centre of the generation cube
  derived from that field of view; `mesh_scale` is the scale of the generation cube. These are the
  values the pipeline conditions on; the same three appear under `camera` and at the top level.

* `--timing-json FILE` -- one report per invocation: `load_s`, `warmup_e2e_s`, `runs` (per timed
  request: `timings` with one `<stage>_s` entry per stage, `e2e_s` and `residual_s`; `stats` with
  token counts, remesh and atlas statistics, `output_faces`, the camera) and the environment.
  `e2e_s` runs from reading the input image to the closed GLB file; the stages are timed with
  device synchronisation at their boundaries. Work that overlaps the main thread (mesh worker,
  texture stream) is reported separately (`mesh_worker_s`, `tex_stream_s`) and is not part of the
  stage sum.
* `--dump-dir DIR` -- per input, `DIR/<stem>/` with `support.npy` (structure support coordinates,
  int32 [M, 4]), `coarse.npz` and `fine.npz` (normalised shape latents: `coords` int32 [N, 4],
  `feats` float32 [N, 32]) and `timing.json` (timings, face count, and the sha256 of the raw bytes
  of the latents passed to the shape and texture decoders). `scripts/parity_check.py` compares two
  such directories (criteria in [PARITY.md](PARITY.md)).

## 7. Tests

```bash
bash scripts/check.sh                  # the CPU tests, then one example image on the GPU, with a summary
pytest -q tests eval/tests             # CPU tests only; no GPU or weights needed
pytest -q -m gpu tests/test_inference_gpu_smoke.py   # end-to-end on one image (GPU and weights)
```

`env/install.sh` installs `pytest`. The evaluation itself needs a few more packages
(`pip install -r eval/requirements.txt`); its CPU tests do not.
