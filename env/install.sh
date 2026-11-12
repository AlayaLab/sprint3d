#!/usr/bin/env bash
# Set up the measured configuration: conda environment, pinned packages, source-built dependencies,
# FlashAttention-3, the patched native extensions, the Pixal3D submodule and this package.
#
# usage: bash env/install.sh     (from a clone of this repository; needs conda, mamba or micromamba, git,
#                                 a C++ compiler supported by CUDA 12.8 (gcc 11 to 14) and an NVIDIA driver for CUDA 12.8)
# environment:
#   ENV_NAME              name of the conda environment (default: fastpixal3d)
#   ENV_PREFIX            or a directory for the environment (overrides ENV_NAME)
#   CONDA                 conda, mamba or micromamba executable (default: the first one found on PATH)
#   MAX_JOBS              parallel compile jobs (default: 4)
#   TORCH_CUDA_ARCH_LIST  CUDA architectures to compile for (default: 9.0, Hopper)
#   WORK_DIR              checkouts and build trees (default: build/env)
#   SKIP_FA3=1            on GPUs other than Hopper (the package then uses FlashAttention-2: --fa2)
# A step that has finished is skipped when the script is run again (stamps in <environment>/.fastpixal3d-install).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
export MAX_JOBS=${MAX_JOBS:-4} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
WORK_DIR=${WORK_DIR:-$REPO/build/env}
log() { echo "[install $(date +%H:%M:%S)] $*"; }

# pip constraints of a system Python (set in some CUDA containers) do not apply to the new environment
if [ -n "${PIP_CONSTRAINT:-}" ]; then log "ignoring PIP_CONSTRAINT=$PIP_CONSTRAINT"; unset PIP_CONSTRAINT; fi

# 1. environment (Python 3.10, CUDA 12.8 compiler, pinned pip packages)
CONDA=${CONDA:-$(command -v conda || command -v mamba || command -v micromamba || true)}
[ -n "$CONDA" ] || { echo "install.sh: conda, mamba or micromamba is needed (or set CONDA)" >&2; exit 1; }
if [ -n "${ENV_PREFIX:-}" ]; then ENV=(-p "$ENV_PREFIX"); else ENV=(-n "${ENV_NAME:-fastpixal3d}"); fi
env_prefix() { "$CONDA" run "${ENV[@]}" python -c "import sys; print(sys.prefix)" 2> /dev/null; }
conda_frontend() { [ "$(basename "$CONDA")" = conda ]; }
t0=$SECONDS
PREFIX=${ENV_PREFIX:-$(env_prefix || true)}
if [ -z "$PREFIX" ] || [ ! -d "$PREFIX/conda-meta" ]; then
  log "creating the environment ${ENV[*]} with $CONDA (about 10 GB)"
  if conda_frontend; then "$CONDA" env create "${ENV[@]}" -f "$HERE/environment.yml"
  else "$CONDA" create -y "${ENV[@]}" -f "$HERE/environment.yml"; fi
  PREFIX=${ENV_PREFIX:-$(env_prefix)}
elif [ ! -e "$PREFIX/.fastpixal3d-install/environment" ]; then
  log "completing the environment $PREFIX (its creation did not finish)"
  if conda_frontend; then "$CONDA" env update "${ENV[@]}" -f "$HERE/environment.yml"
  else "$CONDA" install -y "${ENV[@]}" -f "$HERE/environment.yml"; fi
fi
STAMPS=$PREFIX/.fastpixal3d-install
if [ ! -e "$STAMPS/environment" ]; then
  mkdir -p "$STAMPS"; echo "$((SECONDS - t0))" > "$STAMPS/environment"
  log "environment: done in $((SECONDS - t0)) s"
fi
PY=$PREFIX/bin/python
export PATH=$PREFIX/bin:$PATH CUDA_HOME=${CUDA_HOME:-$PREFIX}
# conda's CUDA packages keep their headers under targets/; the host compiler needs them too (cuda.h)
if [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
  export CPATH=$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}
fi
mkdir -p "$WORK_DIR"
log "python: $PY ($($PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'))"
log "CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"

step() {  # step NAME COMMAND...: run COMMAND once per environment and report its duration
  local name=$1 t0=$SECONDS; shift
  if [ -e "$STAMPS/$name" ]; then log "$name: already done"; return; fi
  log "$name ..."
  "$@"
  echo "$((SECONDS - t0))" > "$STAMPS/$name"
  log "$name: done in $((SECONDS - t0)) s"
}

# 2. source-built dependencies at the measured commits
build() {  # build NAME URL COMMIT [SUBDIRECTORY]
  local name=$1 url=$2 commit=$3 sub=${4:-} dir=$WORK_DIR/$1
  [ -d "$dir/.git" ] || git clone --quiet "$url" "$dir"
  git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
  git -C "$dir" submodule update --init --recursive --quiet
  (cd "$dir/$sub" && "$PY" -m pip install --no-deps --no-build-isolation .)
}
step flexgemm-6dd94a8   build flexgemm   https://github.com/JeffreyXiang/FlexGEMM       6dd94a859c26ee8246888502eada3dd8ad85532e
step cumesh-12289e1     build cumesh     https://github.com/JeffreyXiang/CuMesh         12289e1062f0603f2f0d0771b02e1395d247f26f
step o-voxel-de38fdd    build trellis2   https://github.com/microsoft/TRELLIS.2         de38fdd4082638cb4e6d2285f2e215280bd3f485 o-voxel
step nvdiffrast-253ac4f build nvdiffrast https://github.com/NVlabs/nvdiffrast           253ac4fcea7de5f396371124af597e6cc957bfae
step pipeline-866f059   build pipeline   https://github.com/EasternJournalist/pipeline  866f059d2a05cde05e4a52211ec5051fd5f276d6
step moge-0744441    "$PY" -m pip install --no-deps "git+https://github.com/microsoft/MoGe@07444410f1e33f402353b99d6ccd26bd31e469e8"
# utils3d as measured: the pinned commit plus env/patches/utils3d-moge2-compat.patch (the utils3d.pt alias and
# depth_map_to_point_map that MoGe-2 calls)
patched_utils3d() {
  local dir=$WORK_DIR/utils3d
  [ -d "$dir/.git" ] || git clone --quiet https://github.com/EasternJournalist/utils3d "$dir"
  git -C "$dir" -c advice.detachedHead=false checkout --quiet --force 9a4eb15e4021b67b12c460c7057d642626897ec8
  git -C "$dir" apply "$HERE/patches/utils3d-moge2-compat.patch"
  "$PY" -m pip install --no-deps --force-reinstall "$dir"
}
step utils3d-9a4eb15-patched patched_utils3d
step flash_attn-2.8.3 "$PY" -m pip install --no-deps --no-build-isolation flash_attn==2.8.3
step natten-0.21.0   "$PY" -m pip install --no-deps "natten==0.21.0+torch270cu128" -f https://whl.natten.org

# 3. FlashAttention-3 (Hopper only)
if [ -z "${SKIP_FA3:-}" ]; then
  step flash_attn_3-a8aa52b env PYTHON="$PY" WORK_DIR="$WORK_DIR/fa3" bash "$HERE/build_fa3.sh"
fi

# 4. patched native extensions (fast GPU simplification, GPU BVH build)
step native env PYTHON="$PY" WORK_DIR="$WORK_DIR/native" bash "$REPO/native/build_native.sh"

# 5. Pixal3D source (unmodified, pinned) and this package with its test dependencies
git -C "$REPO" submodule update --init third_party/Pixal3D \
  || log "third_party/Pixal3D: set FASTPIXAL3D_PIXAL3D_SRC to a checkout of TencentARC/Pixal3D @ 28efad66"
"$PY" -m pip install -e "$REPO[test]"

"$PY" - <<'PYCHECK'
import importlib
for name in ("torch", "flex_gemm", "cumesh", "o_voxel", "nvdiffrast.torch", "moge", "utils3d", "flash_attn", "natten",
             "fpng", "fastpixal3d_cumesh", "fastpixal3d_cubvh", "fastpixal3d"):
    importlib.import_module(name)
import utils3d
assert hasattr(utils3d.pt, "depth_map_to_point_map"), "utils3d without env/patches/utils3d-moge2-compat.patch"
try:
    importlib.import_module("flash_attn_interface")
    fa3 = "FlashAttention-3 available"
except ImportError:
    fa3 = "FlashAttention-3 not installed (use --fa2)"
print(f"[install] all imports ok; {fa3}")
PYCHECK
log "done: conda activate ${ENV_PREFIX:-${ENV_NAME:-fastpixal3d}}; see docs/INSTALL.md for the weights and a first run"
