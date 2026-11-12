# End-to-end timing

This is the protocol behind the end-to-end times of the report, and the scripts that implement it.
`scripts/time_e2e.py` runs one method on a list of inputs, and `scripts/analyze_timing.py` summarises one
or more runs.

## Protocol

| item | setting |
|---|---|
| machine | one GPU, one timing process per machine and nothing else running on that GPU. The reported runs used one Hopper GPU (sm_90, 141 GB) and 15 CPU cores |
| inputs | the 80 timing inputs (40 objects x 2 views of the development subset), always in the same fixed order |
| model loading | done once per process, not timed, and reported separately (`load_s`) |
| warm-up | this package runs one untimed request on the first input before the list (`--warmup-requests 1`). The official script gets no extra request. In both cases the first 3 inputs of the list are run and recorded but excluded from the statistics (`--exclude-first 3`) |
| clock | starts when the input image is read and stops when the GLB file is closed. The CUDA device is synchronised before the clock starts and before it stops. Per-stage timers synchronise at their boundaries |
| CPU threads | `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` |
| TF32 | this package: `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (float32 cuBLAS matmuls may use TF32; `--tf32 off` removes it). Official script: the variable is removed, giving stock PyTorch precision (`--tf32 on` keeps it) |
| allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for both methods (Pixal3D's `inference.py` sets the same value itself) |
| attention | `ATTN_BACKEND=flash_attn`. This package also uses FlashAttention-3 for the fine-shape and texture flows (see LEVERS.md) |
| cuBLAS workspace | this package: `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| autotuning | each run gets its own copy of a FlexGEMM autotune cache (`--autotune-cache`). Kernels that are not yet in the cache are tuned during the warm-up inputs |
| between requests | this package runs garbage collection and releases cached GPU memory after every request, outside the timed window |

All of these settings are applied by the script before PyTorch is imported. The script then records:

- the effective environment;
- the PyTorch TF32 flags, the GPU, the driver and the CPU quota (`run.json`);
- the machine load every 30 s from a separate low-priority process: load average, CPU throttling, CPU pressure, GPU utilisation, SM clock, power and temperature (`sampler.jsonl`).

The load record makes it possible to tell contention apart from method differences.

### Methods

- `--method fast` runs this package (all levers on unless `--disable` is given, or the levers and
  sampling settings of a configuration file with `--config`, see `experiments/configs/`), with the export
  settings of Pixal3D's script (1,000,000 faces, 4096 x 4096 texture).
- `--method official` runs Pixal3D's `inference.py` unmodified at its defaults, through
  `fastpixal3d.official` (the same code as `scripts/run_official.py`). The models are loaded once, and the
  end-to-end time is the wall time of `run_inference` minus model loading. That window includes:
  - reading the image;
  - writing and removing the temporary preprocessed image;
  - moving MoGe-2 back to the GPU;
  - the WebP GLB export.

## Running

```bash
# Pixal3D (official)
python scripts/time_e2e.py --method official --list timing_inputs.jsonl --out-dir runs/official \
    --autotune-cache runs/official_autotune.json

# this package
python scripts/time_e2e.py --method fast --list timing_inputs.jsonl \
    --out-dir runs/fast --students /path/to/students --autotune-cache runs/fast_autotune.json

python scripts/analyze_timing.py runs/official runs/fast \
    --out runs/summary.json --csv runs/per_input.csv
```

The input list is a JSON lines file with `id` and `path` per input, in the order to run.

### Output of `scripts/time_e2e.py`

| file | content |
|---|---|
| `run.json` | method, export settings, command line, sha256 of the input list, machine record at start and end, environment, load time, warm-up request times |
| `timing.jsonl` | one line per input: list position, id, warm-up flag, `e2e_s`, all per-stage timings, statistics (token counts, face counts, peak GPU memory, camera) |
| `sampler.jsonl` | machine load samples |
| `<id>.glb`, `<id>.json` | output and inference-time camera of every input (format in INSTALL.md) |
| `dump/<id>/` | with `--dump-dir`: the parity records of PARITY.md |

## Statistics (`scripts/analyze_timing.py`)

The statistics cover the measured inputs, which excludes the first 3 in list order. The same numbers over all inputs are reported too.

For each run:

- end-to-end time: median, mean, standard deviation, 10th, 25th, 75th and 90th percentiles (linear interpolation), interquartile range, minimum and maximum;
- 95 % bootstrap intervals of the median and of the mean over inputs: 10,000 resamples of the inputs with replacement, seed 0, percentile interval;
- every stage timer: median, mean and the bootstrap interval of the mean.

For this package the stages are also grouped along the critical path:

1. image and camera;
2. structure;
3. coarse shape;
4. fine shape;
5. shape decode;
6. the overlapped window, where texture sampling and mesh preparation run concurrently and the longer branch is counted;
7. bake;
8. export;
9. other.

With two or more runs, the first run is the reference, and every other run is compared with it on the measured inputs present in both. The comparison reports the per-input ratio reference / run (median of ratios, geometric mean), the ratio of medians and the ratio of means, each with a bootstrap interval over inputs.

## Parity run

`scripts/parity_run.sh` runs the timing protocol for this package with `--dump-dir`, summarises it, and calls `scripts/parity_check.py` against two runs of the measurement code on the same inputs. The criteria are in PARITY.md.
