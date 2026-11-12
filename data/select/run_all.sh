#!/bin/bash
# Selection of the 10k set from TRELLIS-500K metadata (docs/DATA.md). CPU only; inputs are read only.
#   bash data/select/run_all.sh data/select/config.yaml
# Optional steps (pilot list, precision estimate, Toys4K caption-neighbour audit) run when their inputs are set.
set -euo pipefail
CONFIG=$1
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
run() { echo "== $1"; "$PY" "$HERE/$1" --config "$CONFIG"; }
run join_metadata.py        # TRELLIS-500K metadata + Sketchfab table + Objaverse 1.0 metadata
run geometry_headers.py     # HSSD / ABO GLB headers, 3D-FUTURE file sizes
run future_face_counts.py   # 3D-FUTURE OBJ face counts
run flags.py                # rule flags (rules.py), Toys4K overlap
run select.py               # HARD rules -> keep / reject
run pilot.py                # optional QA sample
run sets.py                 # stratified nested sets and the validation split
run write_outputs.py        # candidates and set lists in <work_dir>/out
"$PY" "$HERE/toys4k_neighbours.py" --config "$CONFIG" || echo "(audit skipped)"
"$PY" "$HERE/precision.py" --config "$CONFIG" || echo "(precision estimate skipped)"
run stats.py
run stats_extra.py
