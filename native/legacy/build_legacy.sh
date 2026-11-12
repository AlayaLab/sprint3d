#!/usr/bin/env bash
# Build the four native extensions that only the tiled mesh path needs (rows 01-07 of the
# cumulative measurement; see README.md in this directory for which row needs which build):
#
#   fastpixal3d_bvh_openmp         cubvh  @ CUBVH_COMMIT  + cubvh/patches/0001
#   fastpixal3d_bvh_openmp_nogil   cubvh  @ CUBVH_COMMIT  + cubvh/patches/0001..0002
#   fastpixal3d_bvh_openmp_pinned  cubvh  @ CUBVH_COMMIT  + cubvh/patches/0001..0003
#   fastpixal3d_xatlas_single      CuMesh @ CUMESH_COMMIT, third_party/xatlas + xatlas/patches/0001
#
# The cubvh patches are written against the pinned upstream tree; each build is then
# renamed to its own module / namespace with ../alias_rename.py, so the three BVH builds,
# the stock packages and the deployed builds of ../build_native.sh can all be installed
# in one environment. The xatlas build compiles CuMesh's bundled xatlas on its own,
# without its internal thread pool, under a renamed C++ namespace.
#
# usage: native/legacy/build_legacy.sh [--target DIR] [--only NAME]
#
#   --target DIR   install into DIR (pip --target) instead of the active environment;
#                  add DIR to PYTHONPATH afterwards.
#   --only NAME    build one extension: openmp | nogil | pinned | xatlas
#
# environment:
#   PYTHON                python interpreter of the target environment   (default: python)
#   TORCH_CUDA_ARCH_LIST  CUDA architectures to compile for          (default: 9.0)
#   MAX_JOBS              parallel compile jobs                            (default: 4)
#   WORK_DIR              scratch directory for checkouts and builds      (default: native/legacy/build)
#   CUBVH_REPO            cubvh git URL or local mirror                   (default: GitHub)
#   EIGEN_REPO            Eigen git URL or local mirror (cubvh's submodule)
#   CUMESH_REPO           CuMesh git URL or local mirror                  (default: GitHub)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NATIVE=$(cd "$HERE/.." && pwd)
PYTHON=${PYTHON:-python}
WORK_DIR=${WORK_DIR:-$HERE/build}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-4}

CUBVH_REPO=${CUBVH_REPO:-https://github.com/JeffreyXiang/cubvh.git}
CUBVH_COMMIT=ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70
EIGEN_REPO=${EIGEN_REPO:-https://gitlab.com/libeigen/eigen.git}
EIGEN_COMMIT=e63d9f6ccb7f6f29f31241b87c542f3f0ab3112b
CUMESH_REPO=${CUMESH_REPO:-https://github.com/JeffreyXiang/CuMesh.git}
CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
XATLAS_MODULE=fastpixal3d_xatlas_single

TARGET=""
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET=$2; shift 2;;
    --only) ONLY=$2; shift 2;;
    -h|--help) sed -n '2,29p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

GIT=(git -c advice.detachedHead=false -c protocol.file.allow=always)
log() { echo "[build_legacy] $*"; }

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

build_bvh() {  # build_bvh ALIAS N_PATCHES
  local alias=$1 n=$2 src=$WORK_DIR/$1 i=0
  log "cubvh @ ${CUBVH_COMMIT:0:7} + patches 0001..000$n -> $alias"
  checkout "$CUBVH_REPO" "$CUBVH_COMMIT" "$src"
  submodule "$src" third_party/eigen "$EIGEN_REPO" "$EIGEN_COMMIT"
  for p in "$HERE"/cubvh/patches/*.patch; do
    i=$((i + 1))
    [ "$i" -le "$n" ] || break
    log "  apply $(basename "$p")"
    "${GIT[@]}" -C "$src" apply --whitespace=nowarn "$p"
  done
  "$PYTHON" "$NATIVE/alias_rename.py" cubvh "$src" --alias "$alias"
  install_tree "$src"
}

build_xatlas() {
  local src=$WORK_DIR/cumesh_xatlas
  log "CuMesh @ ${CUMESH_COMMIT:0:7} third_party/xatlas (single-thread) -> $XATLAS_MODULE"
  checkout "$CUMESH_REPO" "$CUMESH_COMMIT" "$src"   # xatlas is a plain directory of CuMesh, no submodule needed
  "$PYTHON" - "$src/third_party/xatlas" "${XATLAS_MODULE}_ns" <<'EOF'
import re, sys
from pathlib import Path
root, ns = Path(sys.argv[1]), sys.argv[2]
for name in ("xatlas.cpp", "xatlas.h", "binding.cpp"):
    path = root / name
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    path.write_text(re.sub(r"\bcumesh_xatlas\b", ns, text), encoding="utf-8", errors="surrogateescape")
print(f"[build_legacy]   namespace cumesh_xatlas -> {ns}")
EOF
  for p in "$HERE"/xatlas/patches/*.patch; do
    log "  apply $(basename "$p")"
    "${GIT[@]}" -C "$src" apply --whitespace=nowarn "$p"
  done
  install_tree "$src/third_party/xatlas"
}

mkdir -p "$WORK_DIR"
log "python=$("$PYTHON" -c 'import sys; print(sys.executable)') TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS target=${TARGET:-<environment>}"
case "$ONLY" in
  "") build_bvh fastpixal3d_bvh_openmp 1; build_bvh fastpixal3d_bvh_openmp_nogil 2
      build_bvh fastpixal3d_bvh_openmp_pinned 3; build_xatlas;;
  openmp) build_bvh fastpixal3d_bvh_openmp 1;;
  nogil) build_bvh fastpixal3d_bvh_openmp_nogil 2;;
  pinned) build_bvh fastpixal3d_bvh_openmp_pinned 3;;
  xatlas) build_xatlas;;
  *) echo "--only must be openmp, nogil, pinned or xatlas" >&2; exit 2;;
esac

check_path=${TARGET:+$TARGET:}
PYTHONPATH="${check_path}${PYTHONPATH:-}" "$PYTHON" - <<'EOF'
import importlib
for name in ("fastpixal3d_bvh_openmp", "fastpixal3d_bvh_openmp_nogil", "fastpixal3d_bvh_openmp_pinned",
             "fastpixal3d_xatlas_single"):
    try:
        m = importlib.import_module(name)
        print(f"[build_legacy] import {name}: ok ({m.__file__})")
    except ImportError as e:
        print(f"[build_legacy] import {name}: not available ({e})")
EOF
