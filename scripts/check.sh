#!/usr/bin/env bash
# One-command check of an installation.
#
#   1. the CPU test suite (tests/ and eval/tests/, GPU tests deselected, GPUs hidden);
#   2. one image through scripts/infer.py at the official export (1,000,000 faces, 4096 x 4096 texture),
#      skipped with a message when no CUDA GPU is visible.
#
# Prints a short summary: tests passed, face count, texture size, end-to-end time.
#
# usage: scripts/check.sh [--image IMAGE] [--out DIR] [--skip-tests] [--skip-gpu] [-- extra arguments for infer.py]
#
#   --image IMAGE  input image (default: assets/images/4_img.png of the Pixal3D checkout)
#   --out DIR      output directory (default: a new temporary directory)
#   extra arguments, e.g. `-- --threads 4` on a machine with fewer cores, or `-- --autotune-cache FILE`
#                  with a copy of a FlexGEMM autotune cache (default: a new cache in DIR)
#
# environment:
#   PYTHON                   interpreter (default: python)
#   FASTPIXAL3D_WEIGHTS      weights directory of scripts/download_weights.py (default: weights/)
#   FASTPIXAL3D_STUDENTS     student weights, a directory with students.json (default: <weights dir>/students)
#   FASTPIXAL3D_PIXAL3D      Pixal3D weights, a local directory or a Hugging Face repo id
#                            (default: <weights dir>/pixal3d if present, else the pinned Hugging Face revision)
#   FASTPIXAL3D_PIXAL3D_SRC  Pixal3D source checkout (default: third_party/Pixal3D)
#   CUDA_VISIBLE_DEVICES     selects the GPU of step 2
#
# The end-to-end time is that of one request after one untimed warm-up request; it is not the
# protocol of docs/TIMING.md (80 inputs, dedicated machine). Exit code 0 when every step that ran passed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PYTHON=${PYTHON:-python}
IMAGE="" OUT="" SKIP_TESTS=0 SKIP_GPU=0 EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --skip-tests) SKIP_TESTS=1; shift;;
    --skip-gpu) SKIP_GPU=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2;;
  esac
done
OUT=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/fastpixal3d_check.XXXXXX")}
mkdir -p "$OUT"
cd "$REPO"

TESTS_LINE="not run" TESTS_OK=1 GPU_OK=1 SUMMARY_GPU=""

# ---- 1. CPU test suite ------------------------------------------------------------------------------------------------
if [ "$SKIP_TESTS" = 1 ]; then
  TESTS_LINE="skipped (--skip-tests)"
elif ! "$PYTHON" -c "import pytest" > /dev/null 2>&1; then
  TESTS_LINE="not run: pytest is not installed for $PYTHON (pip install pytest)"
  TESTS_OK=0
else
  echo "[check] CPU tests: $PYTHON -m pytest -q -m 'not gpu' tests eval/tests  (log: $OUT/tests.log)"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="" "$PYTHON" -m pytest -q -p no:cacheprovider -m "not gpu" tests eval/tests \
    > "$OUT/tests.log" 2>&1
  rc=$?
  TESTS_LINE="$(grep -E '[0-9]+ (passed|failed|error)' "$OUT/tests.log" | tail -n 1 | sed -E 's/=//g; s/ in [0-9.]+s( \([0-9:]+\))?//; s/^ *//; s/ *$//')"
  TESTS_LINE="${TESTS_LINE:-no summary line (see $OUT/tests.log)} (CPU, $(( $(date +%s) - start )) s)"
  if [ "$rc" -ne 0 ]; then
    TESTS_OK=0
    tail -n 30 "$OUT/tests.log"
  fi
fi

# ---- 2. one image at the official export ------------------------------------------------------------------------------
NGPU=$("$PYTHON" -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
if [ "$SKIP_GPU" = 1 ]; then
  SUMMARY_GPU="inference : skipped (--skip-gpu)"
elif [ "${NGPU:-0}" = 0 ]; then
  SUMMARY_GPU="inference : skipped: no CUDA GPU visible to $PYTHON"
else
  SRC=${FASTPIXAL3D_PIXAL3D_SRC:-$REPO/third_party/Pixal3D}
  IMAGE=${IMAGE:-$SRC/assets/images/4_img.png}
  if [ ! -f "$IMAGE" ]; then
    SUMMARY_GPU="inference : FAILED: input image not found: $IMAGE (pass --image, or initialise third_party/Pixal3D)"
    GPU_OK=0
  else
    STEM=$(basename "${IMAGE%.*}")
    GLB="$OUT/$STEM.glb"
    echo "[check] inference: scripts/infer.py --image $IMAGE --output $GLB  (log: $OUT/infer.log)"
    # the autotune cache goes to OUT (otherwise Pixal3D's inference.py writes one into its checkout);
    # a later --autotune-cache in the extra arguments takes precedence
    "$PYTHON" scripts/infer.py --image "$IMAGE" --output "$GLB" --warmup 1 --runs 1 \
      --timing-json "$OUT/timing.json" --autotune-cache "$OUT/autotune_cache.json" \
      ${EXTRA[@]+"${EXTRA[@]}"} > "$OUT/infer.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$GLB" ]; then
      tail -n 30 "$OUT/infer.log"
      SUMMARY_GPU="inference : FAILED (exit code $rc; see $OUT/infer.log)"
      GPU_OK=0
    else
      SUMMARY_GPU=$("$PYTHON" - "$GLB" "$OUT/timing.json" <<'PY'
import io, json, struct, sys

from PIL import Image

glb_path, timing_path = sys.argv[1], sys.argv[2]
data = open(glb_path, "rb").read()
magic, _, _ = struct.unpack_from("<4sII", data, 0)
assert magic == b"glTF", "not a GLB file"
length, kind = struct.unpack_from("<I4s", data, 12)
gltf = json.loads(data[20:20 + length])
offset = 20 + length
binary = b""
if offset < len(data):
    blen, _ = struct.unpack_from("<I4s", data, offset)
    binary = data[offset + 8: offset + 8 + blen]

faces = 0
for mesh in gltf.get("meshes", []):
    for prim in mesh.get("primitives", []):
        if "indices" in prim:
            faces += gltf["accessors"][prim["indices"]]["count"] // 3
        else:
            faces += gltf["accessors"][prim["attributes"]["POSITION"]]["count"] // 3


def texture_size(info):
    if not info:
        return None
    texture = gltf["textures"][info["index"]]
    source = texture.get("source")
    if source is None:  # WebP and other extension images
        source = next(iter(texture.get("extensions", {}).values()), {}).get("source")
    image = gltf["images"][source]
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).size


sizes = {}
for material in gltf.get("materials", []):
    pbr = material.get("pbrMetallicRoughness", {})
    sizes.setdefault("base colour", texture_size(pbr.get("baseColorTexture")))
    sizes.setdefault("metallic-roughness", texture_size(pbr.get("metallicRoughnessTexture")))

timing = json.load(open(timing_path))
e2e = (timing.get("timings") or {}).get("e2e_s")
load_s = timing.get("load_s")
warm = timing.get("warmup_e2e_s") or []

ok = 0 < faces <= 1_010_000 and sizes.get("base colour") == (4096, 4096)
fmt = lambda s: "missing" if s is None else f"{s[0]} x {s[1]}"
print(f"faces     : {faces:,} (target 1,000,000)")
print("texture   : " + ", ".join(f"{fmt(v)} ({k})" for k, v in sizes.items()))
print(f"end-to-end: {e2e:.2f} s (one request after {len(warm)} warm-up request(s); "
      f"model loading {load_s:.0f} s not included)" if e2e is not None else "end-to-end: missing")
print("export    : " + ("OK (official export)" if ok else "FAILED: expected <= 1,000,000 faces (+1 %) and a 4096 x 4096 texture"))
sys.exit(0 if ok else 1)
PY
)
      [ $? -eq 0 ] || GPU_OK=0
      SUMMARY_GPU="inference : $IMAGE -> $GLB
$SUMMARY_GPU"
    fi
  fi
fi

# ---- summary ------------------------------------------------------------------------------------------------------------
echo
echo "== fastpixal3d check =="
echo "tests     : $TESTS_LINE"
echo "$SUMMARY_GPU"
if [ "$TESTS_OK" = 1 ] && [ "$GPU_OK" = 1 ]; then
  echo "result    : OK (outputs and logs in $OUT)"
  exit 0
fi
echo "result    : FAILED (outputs and logs in $OUT)"
exit 1
