#!/usr/bin/env bash
# Build FlashAttention-3 (Hopper, sm_90 only) from the pinned commit, as used in the measured configuration.
#
#   flash-attention  v2.8.3.post1 = a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2, directory hopper/
#   CUTLASS          dc4817921edda44a549197ff3a9dcf5df0636e7b (the csrc/cutlass submodule of that commit)
#   result           flash_attn_3 3.0.0b1 (module flash_attn_interface), gencode compute_90a
#
# By default only the kernels the pipeline calls are compiled (forward, bf16, head dim 128, sm_90; about two
# minutes with 4 jobs), as in the measured build; the compiled kernels are the same as in a full build.
# FULL=1 builds the complete kernel set (much longer).
#
# usage: env/build_fa3.sh [--target DIR]      environment: PYTHON (python), MAX_JOBS (4), WORK_DIR (build/fa3), FULL
set -euo pipefail
PYTHON=${PYTHON:-python}
MAX_JOBS=${MAX_JOBS:-4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR=${WORK_DIR:-$HERE/../build/fa3}
FA_COMMIT=a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2
CUTLASS_COMMIT=dc4817921edda44a549197ff3a9dcf5df0636e7b
TARGET=()
[ "${1:-}" = "--target" ] && TARGET=(--target "$2")

src=$WORK_DIR/flash-attention
if [ ! -d "$src/.git" ]; then
  mkdir -p "$WORK_DIR"
  git clone https://github.com/Dao-AILab/flash-attention "$src"
fi
git -C "$src" checkout --quiet "$FA_COMMIT"
git -C "$src" submodule update --init csrc/cutlass
got=$(git -C "$src/csrc/cutlass" rev-parse HEAD)
[ "$got" = "$CUTLASS_COMMIT" ] || { echo "csrc/cutlass is at $got, expected $CUTLASS_COMMIT" >&2; exit 1; }

export FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS NVCC_THREADS=${NVCC_THREADS:-2}
if [ -z "${FULL:-}" ]; then
  for f in BACKWARD SPLIT PAGEDKV APPENDKV LOCAL SOFTCAP PACKGQA FP16 FP8 HDIM64 HDIM96 HDIM192 HDIM256 SM80; do
    export "FLASH_ATTENTION_DISABLE_$f=TRUE"
  done
fi
(cd "$src/hopper" && "$PYTHON" -m pip install --no-deps --no-build-isolation ${TARGET[@]+"${TARGET[@]}"} .)
PYTHONPATH="${TARGET[1]:-}${TARGET[1]:+:}${PYTHONPATH:-}" "$PYTHON" -c "import flash_attn_interface; print('[build_fa3] flash_attn_interface: ok')"
