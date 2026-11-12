#!/usr/bin/env bash
# Build the two patched native extensions fastpixal3d needs:
#
#   fastpixal3d_cumesh  CuMesh  @ CUMESH_COMMIT + patches/cumesh   (sync-free GPU simplification)
#   fastpixal3d_cubvh   cubvh   @ CUBVH_COMMIT  + patches/cubvh    (GPU LBVH construction)
#
# Each upstream checkout is pinned, renamed to a private module/namespace name
# (alias_rename.py; this lets the patched builds coexist with the stock `cumesh`
# package other dependencies import), patched with `git apply`, and built with
# pip. Nothing outside the build directory and the install target is modified.
#
# usage: native/build_native.sh [--target DIR] [--only cumesh|cubvh] [--keep-going]
#
#   --target DIR   install into DIR (pip --target) instead of the active environment;
#                  add DIR to PYTHONPATH afterwards.
#   --only NAME    build a single extension.
#
# environment:
#   PYTHON           python interpreter of the target environment   (default: python)
#   TORCH_CUDA_ARCH_LIST  CUDA architectures to compile for          (default: 9.0)
#   MAX_JOBS         parallel compile jobs                            (default: 4)
#   WORK_DIR         scratch directory for checkouts and builds      (default: native/build)
#   CUMESH_REPO      CuMesh git URL or local mirror                   (default: GitHub)
#   CUBVH_REPO       cubvh git URL or local mirror (also used for CuMesh's cubvh submodule)
#   EIGEN_REPO       Eigen git URL or local mirror (cubvh's submodule)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
WORK_DIR=${WORK_DIR:-$HERE/build}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-4}

CUMESH_REPO=${CUMESH_REPO:-https://github.com/JeffreyXiang/CuMesh.git}
CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
CUBVH_REPO=${CUBVH_REPO:-https://github.com/JeffreyXiang/cubvh.git}
CUBVH_COMMIT=ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70
EIGEN_REPO=${EIGEN_REPO:-https://gitlab.com/libeigen/eigen.git}
EIGEN_COMMIT=e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b

TARGET=""
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET=$2; shift 2;;
    --only) ONLY=$2; shift 2;;
    -h|--help) sed -n '2,27p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

GIT=(git -c advice.detachedHead=false -c protocol.file.allow=always)
log() { echo "[build_native] $*"; }

checkout() {  # checkout URL COMMIT DIR
  local url=$1 commit=$2 dir=$3
  if [ ! -d "$dir/.git" ]; then
    "${GIT[@]}" clone --quiet --no-checkout "$url" "$dir"
  fi
  "${GIT[@]}" -C "$dir" checkout --quiet --force "$commit"
  "${GIT[@]}" -C "$dir" clean -qfdx
}

submodule() {  # submodule PARENT_DIR PATH URL COMMIT
  local parent=$1 path=$2 url=$3 commit=$4
  "${GIT[@]}" -C "$parent" config "submodule.$path.url" "$url"
  "${GIT[@]}" -C "$parent" submodule update --init --quiet -- "$path"
  local got
  got=$("${GIT[@]}" -C "$parent/$path" rev-parse HEAD)
  if [ "$got" != "$commit" ]; then
    echo "submodule $path is at $got, expected $commit" >&2; exit 1
  fi
}

install_tree() {  # install_tree SOURCE_DIR
  local src=$1
  local args=(-m pip install --no-deps --no-build-isolation --no-cache-dir)
  if [ -n "$TARGET" ]; then
    mkdir -p "$TARGET"
    args+=(--upgrade --target "$TARGET")
  fi
  (cd "$src" && "$PYTHON" "${args[@]}" .)
}

build_cumesh() {
  local src=$WORK_DIR/cumesh
  log "CuMesh @ ${CUMESH_COMMIT:0:7} -> fastpixal3d_cumesh"
  checkout "$CUMESH_REPO" "$CUMESH_COMMIT" "$src"
  submodule "$src" third_party/cubvh "$CUBVH_REPO" "$CUBVH_COMMIT"
  submodule "$src/third_party/cubvh" third_party/eigen "$EIGEN_REPO" "$EIGEN_COMMIT"
  "$PYTHON" "$HERE/alias_rename.py" cumesh "$src"
  for p in "$HERE"/cumesh/patches/*.patch; do
    log "  apply $(basename "$p")"
    "${GIT[@]}" -C "$src" apply --whitespace=nowarn "$p"
  done
  install_tree "$src"
}

build_cubvh() {
  local src=$WORK_DIR/cubvh
  log "cubvh @ ${CUBVH_COMMIT:0:7} -> fastpixal3d_cubvh"
  checkout "$CUBVH_REPO" "$CUBVH_COMMIT" "$src"
  submodule "$src" third_party/eigen "$EIGEN_REPO" "$EIGEN_COMMIT"
  "$PYTHON" "$HERE/alias_rename.py" cubvh "$src"
  for p in "$HERE"/cubvh/patches/*.patch; do
    log "  apply $(basename "$p")"
    "${GIT[@]}" -C "$src" apply --whitespace=nowarn "$p"
  done
  install_tree "$src"
}

mkdir -p "$WORK_DIR"
log "python=$("$PYTHON" -c 'import sys; print(sys.executable)') TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS target=${TARGET:-<environment>}"
case "$ONLY" in
  "") build_cubvh; build_cumesh;;
  cumesh) build_cumesh;;
  cubvh) build_cubvh;;
  *) echo "--only must be cumesh or cubvh" >&2; exit 2;;
esac

check_path=${TARGET:+$TARGET:}
PYTHONPATH="${check_path}${PYTHONPATH:-}" "$PYTHON" - <<'EOF'
import importlib
for name in ("fastpixal3d_cubvh", "fastpixal3d_cumesh"):
    try:
        m = importlib.import_module(name)
        print(f"[build_native] import {name}: ok ({m.__file__})")
    except ImportError as e:
        print(f"[build_native] import {name}: not available ({e})")
EOF
