# Baseline adapters

Run scripts for the external image-to-3D methods of the report, each at its released defaults and at the pinned
commit of the report's runs. No code or weight of these methods is part of this repository: each script imports the
method from a checkout you obtain yourself.

| method | script | upstream commit | output (per input) | frame map in `eval/frame_maps` |
|---|---|---|---|---|
| TRELLIS.2 | `run_trellis2.py` | microsoft/TRELLIS.2 @ `75fbf0183001ed9876c8dbb35de6b68552ee08bd` | `model.glb` (1M faces, 4096 x 4096 texture) | `trellis2`: canonical (GLB axes (x, y, z) -> world (x, -z, y)) |
| TripoSG | `run_triposg.py` | VAST-AI-Research/TripoSG @ `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` | `model.glb` (untextured) | `triposg`: camera-referenced, view aligned |
| Hunyuan3D-2.1 (**optional**) | `run_hunyuan3d21.py` | Tencent-Hunyuan/Hunyuan3D-2.1 @ `82920d643c0dc2f7bfd7255f45f62d386edfe60c` | `shape_untextured.glb`, `model.glb` (PBR texture) | `hunyuan3d21`: camera-referenced, azimuth aligned |

`sources.json` lists the same commits with the weight revisions, the licences and the settings.

## What the scripts do

All three share `baseline_io.py`, which follows the protocol of [docs/TIMING.md](../../docs/TIMING.md): one process,
the models loaded once and not timed, the inputs one at a time in list order, `--warmup-requests` untimed requests on
the first input (default 1), the device synchronised before the clock starts and before it stops, garbage
collection and a release of cached GPU memory after every request outside the timed window, and a machine-load
sampler. The process environment is set before PyTorch is imported: `OMP_NUM_THREADS=MKL_NUM_THREADS=8`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `ATTN_BACKEND=flash_attn` and
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (the setting of the report's runs of these methods; `--tf32 off` removes it).

Inputs: `--image input.png` (id = the file stem, or `--id`) or `--list inputs.jsonl` (JSON lines with `id` and `path`).
`make_input_list.py` writes such a list from an object list of `eval/lists/` and the ground-truth root, with ids
`<id>_v<view>`.

Output directory (`--out-dir`):

| file | content |
|---|---|
| `<id>/model.glb` | the method's output (and the further files its frame map names) |
| `<id>/camera.json` | the per-input record in the layout of the package's camera file (docs/INSTALL.md, "Output format"): these methods estimate no camera, so `camera` is null; the method's fixed frame map is copied from `eval/frame_maps/baseline_frame_maps.json` together with that file's sha256, and the end-to-end times are recorded |
| `timing.jsonl` | one line per input in the format of `scripts/time_e2e.py`: `e2e_s` (wall clock from reading the image to the closed GLB file), `stage_sum_s` (sum of the device-synchronised stage timers: the end-to-end definition of the report's baseline table), every stage, and the split of the report's table (`part_sampling_s`, `part_decoding_s`, `part_meshwork_s`, `part_other_s`) |
| `run.json`, `sampler.jsonl` | settings, command line, list checksum, load time, warm-up times, machine and environment records; machine load |

`--resume` continues an interrupted run (inputs recorded with status ok are skipped). Each run starts with an empty
FlexGEMM autotune cache (`<out-dir>/autotune_cache.json`, or `--autotune-cache`).

| script | stages |
|---|---|
| `run_trellis2.py` | TRELLIS.2's own `run()`, timed per pipeline method by a subclass of `Trellis2ImageTo3DPipeline` (`trellis2_pipeline.py`; the checkout is not modified): `preprocess_image`, `get_cond` (twice), `sample_sparse_structure`, `sample_shape_slat_cascade`, `sample_tex_slat`, `decode_latent`; then `export` = `simplify` (16,777,216 faces, as in `example.py`) + `to_glb` (o-voxel: 1,000,000 faces, 4096 x 4096, remesh band 1, projection 0) + `write` (trimesh, PNG textures) |
| `run_triposg.py` | `preprocess` (the checkout's `prepare_image`), `generate` (the pipeline call; nested `image_condition` and `geometry_extraction`), `export` (trimesh) |
| `run_hunyuan3d21.py` | `shape` (nested `shape_preprocess`, `shape_condition`, `shape_decode`), `shape_write`, `paint` (nested sub-steps of the texture stage: remeshing, UV unwrapping, view selection, normal and position renders, multi-view diffusion, super-resolution, baking, inpainting, texture assignment, mesh save, OBJ to GLB), `export` |

TripoSG and Hunyuan3D-2.1 offer no hook to split their single pipeline call; while that call runs, the scripts wrap
the named callables of the third-party code with timers that call them unchanged, and restore them afterwards
(`baseline_io.timed_calls`).

## Installation

TRELLIS.2 runs in this package's environment (`env/install.sh`, which also installs o-voxel from TRELLIS.2 @
`de38fdd4082638cb4e6d2285f2e215280bd3f485`); only the checkout is needed:

```bash
git clone https://github.com/microsoft/TRELLIS.2 third_party/TRELLIS.2
git -C third_party/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
huggingface-cli download microsoft/TRELLIS.2-4B --revision af44b45f2e35a493886929c6d786e563ec68364d \
    --local-dir weights/trellis2_4b
```

The pipeline also loads `microsoft/TRELLIS-image-large` (sparse-structure decoder), `facebook/dinov3-vitl16-pretrain-lvd1689m`
(gated) and `briaai/RMBG-2.0` (gated, CC BY-NC 4.0) from the Hugging Face hub; accept their terms first. Their
revisions in the report's runs are in `sources.json`.

TripoSG needs a virtual environment on top of this package's environment:

```bash
git clone https://github.com/VAST-AI-Research/TripoSG third_party/TripoSG
git -C third_party/TripoSG checkout fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
python -m venv --system-site-packages envs/triposg
envs/triposg/bin/pip install scikit-image==0.25.2 peft==0.21.0 jaxtyping==0.3.7 typeguard==4.6.0
envs/triposg/bin/pip install --no-build-isolation --no-binary :all: diso==0.1.4     # CUDA extension, built from source
huggingface-cli download VAST-AI/TripoSG --revision 2c1c516d22d58db486a058d98d31bb6177344e06 --local-dir weights/triposg
```

### Hunyuan3D-2.1 (optional)

**Optional.** Hunyuan3D-2.1 is licensed under the Tencent Hunyuan 3D 2.1 Community License Agreement, which does not
apply in the European Union, the United Kingdom and South Korea, requires a copy of the agreement and a notice of
modifications on redistribution, requires a separate licence above one million monthly active users and prohibits
using its outputs to improve other AI models. No file of Hunyuan3D-2.1 is included in this repository;
`run_hunyuan3d21.py` is an adapter written for this repository that imports Hunyuan3D-2.1 from a checkout you
obtain yourself after reading and accepting that licence:

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 /path/to/Hunyuan3D-2.1
git -C /path/to/Hunyuan3D-2.1 checkout 82920d643c0dc2f7bfd7255f45f62d386edfe60c
python -m venv --system-site-packages envs/hunyuan3d21
envs/hunyuan3d21/bin/pip install transformers==4.46.0 diffusers==0.30.0 huggingface-hub==0.30.2 trimesh==4.4.7 \
    realesrgan==0.3.0 basicsr==1.4.2 pymeshlab xatlas pygltflib rembg onnxruntime cupy-cuda12x pytorch-lightning \
    torchmetrics timm torchdiffeq scikit-image pybind11 fast_simplification omegaconf configargparse
envs/hunyuan3d21/bin/pip install numpy==2.2.6          # trimesh 4.4.7 downgrades numpy; restore the environment's
# the two extensions of the checkout, built as its README describes, inside the virtual environment:
source envs/hunyuan3d21/bin/activate
(cd /path/to/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install --no-build-isolation .)
(cd /path/to/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh)
deactivate
huggingface-cli download tencent/Hunyuan3D-2.1 --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 \
    --local-dir /path/to/hy3dgen/tencent/Hunyuan3D-2.1
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth   # sha256 4fa0d389...82f1
```

Blender's Python module (`bpy`) converts the textured OBJ to GLB in the repository's own code; the report's runs had
no `bpy` for Python 3.10, and the adapter then converts with trimesh (base colour only; recorded as `obj_to_glb`).

## Running

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
```

`docs/REPRODUCE_PAPER.md` ("Baselines") maps the report's rows to these commands, to `summarize_timing.py` and to the
evaluation with `make_baseline_manifest.py` and `eval/run_eval.py --baseline-manifest`.

## Licences

| component | licence |
|---|---|
| these adapters | the licence of this repository |
| TRELLIS.2 code and TRELLIS.2-4B weights | MIT (Microsoft) |
| RMBG-2.0 (loaded by TRELLIS.2) | CC BY-NC 4.0, gated |
| DINOv3 ViT-L/16 (loaded by TRELLIS.2) | DINOv3 License, gated |
| TripoSG code and weights | MIT (VAST-AI-Research); its NOTICE states that parts of the code derive from HunyuanDiT and FlashVDM (Tencent Hunyuan community licences), RMBG (bria-rmbg-1.4) and Diffusers (Apache-2.0) |
| RMBG-1.4 (used by TripoSG for inputs without alpha) | bria-rmbg-1.4 (non-commercial) |
| diso (TripoSG's iso-surface extraction) | CC BY-NC 4.0 |
| Hunyuan3D-2.1 code and weights (optional) | Tencent Hunyuan 3D 2.1 Community License Agreement |
| DINOv2 giant (Hunyuan3D-2.1 texture stage) | Apache-2.0 |
| Real-ESRGAN (code and x4plus checkpoint) | BSD-3-Clause |
