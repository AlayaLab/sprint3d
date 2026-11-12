#!/usr/bin/env bash
# Parity run on one machine (criteria: docs/PARITY.md; protocol: docs/TIMING.md).
#
# Runs this package on the input list with the timing protocol, writes the per-input parity records,
# summarises the timings and compares everything with a run of the measurement code.
#
# usage: scripts/parity_run.sh --list inputs.jsonl --reference REF_DIR --out OUT_DIR \
#            [--reference-repeat REF2_DIR] [--exclude-first 3] \
#            [-- extra arguments for scripts/time_e2e.py, e.g. --students DIR --autotune-cache FILE]
#
#   REF_DIR, REF2_DIR  two runs of the measurement code on the same list, in the per-input layout read by
#                      scripts/parity_check.py (support.npy, timing.json with e2e_s and the decoder-input
#                      sha256 values, and the GLB). REF2_DIR is only used for the texture criterion
#                      (the reference's own run-to-run surface PSNR).
#   OUT_DIR            new directory: run/ (time_e2e.py output, run/dump/ parity records),
#                      timing_summary.json, per_input.csv, parity_report.json
#
# environment: PYTHON (default: python). Select the GPU with CUDA_VISIBLE_DEVICES. Nothing else may run
# on the GPU during the run. Exit code: that of parity_check.py (0 = no criterion failed).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
LIST="" REF="" REF2="" OUT="" EXCLUDE=3
while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST=$2; shift 2;;
    --reference) REF=$2; shift 2;;
    --reference-repeat) REF2=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --exclude-first) EXCLUDE=$2; shift 2;;
    --) shift; break;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2;;
  esac
done
[ -n "$LIST" ] && [ -n "$REF" ] && [ -n "$OUT" ] || { echo "--list, --reference and --out are required" >&2; exit 2; }
[ -e "$OUT/run/timing.jsonl" ] && { echo "$OUT/run already holds a run; use a new --out" >&2; exit 2; }
mkdir -p "$OUT"

echo "[parity_run] timing run: list $LIST"
"$PYTHON" "$HERE/time_e2e.py" --method fast --list "$LIST" --out-dir "$OUT/run" \
  --exclude-first "$EXCLUDE" --dump-dir "$OUT/run/dump" "$@"

echo "[parity_run] timing summary"
"$PYTHON" "$HERE/analyze_timing.py" "$OUT/run" --labels package --exclude-first "$EXCLUDE" \
  --out "$OUT/timing_summary.json" --csv "$OUT/per_input.csv"

echo "[parity_run] parity check against $REF"
REPEAT_ARGS=()
[ -n "$REF2" ] && REPEAT_ARGS=(--reference-repeat "$REF2")
set +e
"$PYTHON" "$HERE/parity_check.py" --reference "$REF" ${REPEAT_ARGS[@]+"${REPEAT_ARGS[@]}"} --candidate "$OUT/run/dump" \
  --candidate-glb-dir "$OUT/run" --list "$LIST" --exclude-first "$EXCLUDE" --out "$OUT/parity_report.json"
rc=$?
set -e
echo "[parity_run] parity_check exit code $rc (report: $OUT/parity_report.json)"
exit $rc
