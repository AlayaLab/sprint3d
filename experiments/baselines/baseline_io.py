"""Shared driver of the baseline adapters and of the TRELLIS.2 mesh-path port.

Every adapter is a :class:`Method` (load the models once, run one input) and is driven by :func:`run_methods`
with the protocol of docs/TIMING.md: one process, the models loaded once and not timed, the inputs one at a time
in list order, optional untimed warm-up requests on the first input, CUDA synchronised before the clock starts
and before it stops, and garbage collection plus a release of cached GPU memory after every request (outside the
timed window).

Input list: JSON lines with ``id`` and ``path`` per input (the format of scripts/time_e2e.py). Evaluation inputs
use ``<list id>_v<view>`` as id (make_input_list.py writes such a list).

Output directory of one method:

  run.json          method, configuration, command line, list sha256, machine and environment records, load time,
                    warm-up times (a process that resumes the run with --resume writes run.resume<k>.json)
  timing.jsonl      one line per input, in the format of scripts/time_e2e.py (read by scripts/analyze_timing.py
                    and summarize_timing.py)
  sampler.jsonl     machine load every --sampler-interval seconds
  <id>/model.glb    the output (some methods write further files, e.g. shape_untextured.glb)
  <id>/camera.json  the per-input record of camera_record()

Times of one input (seconds, in ``timings`` of the timing.jsonl line):

  e2e_s             wall clock from reading the input image to the closed GLB file (docs/TIMING.md)
  stage_sum_s       sum of the top-level stage timers; each stage is bracketed by torch.cuda.synchronize(). This is
                    the end-to-end definition of the report's TRELLIS.2 and baseline tables
  <stage>_s         every stage timer, nested ones included (a stage entered twice is summed)
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
FRAME_MAPS = REPO / "eval" / "frame_maps" / "baseline_frame_maps.json"
FRAME_MAPS_REL = "eval/frame_maps/baseline_frame_maps.json"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SIDECAR = "camera.json"
OUTPUT = "model.glb"


def add_package_to_path() -> None:
    """Make ``fastpixal3d`` (src/) importable without installing the package."""
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


# ----------------------------------------------------------------------------------------------- inputs
def read_list(path) -> List[Tuple[str, str]]:
    """``[(id, absolute image path)]`` from a JSON lines file with ``id`` and ``path``."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = str(row.get("id") or row.get("object_id") or "")
        if not ID_PATTERN.match(ident):
            raise ValueError(f"{path}:{number}: id {ident!r} is not usable as a directory name")
        if not row.get("path"):
            raise ValueError(f"{path}:{number}: no 'path'")
        rows.append((ident, str(Path(row["path"]).expanduser().resolve())))
    if not rows:
        raise ValueError(f"{path}: no inputs")
    ids = [i for i, _ in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return rows


def inputs_from_args(args) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """(rows, list path or None) for ``--image [--id]`` or ``--list``."""
    if args.list:
        return read_list(args.list), str(Path(args.list).resolve())
    ident = args.id or Path(args.image).stem
    if not ID_PATTERN.match(ident):
        raise ValueError(f"--id {ident!r} is not usable as a directory name")
    return [(ident, str(Path(args.image).expanduser().resolve()))], None


# ----------------------------------------------------------------------------------------------- arguments
def add_common_arguments(ap: argparse.ArgumentParser, default_seed: int = 42) -> None:
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="one input image")
    src.add_argument("--list", help="JSON lines with 'id' and 'path', run in this order")
    ap.add_argument("--id", default=None, help="(--image) id of the output (default: the image stem)")
    ap.add_argument("--out-dir", required=True, help="output directory (see experiments/baselines/README.md)")
    ap.add_argument("--warmup-requests", type=int, default=1,
                    help="untimed requests on the first input before the list (default 1)")
    ap.add_argument("--exclude-first", type=int, default=3,
                    help="inputs marked as warm-up in timing.jsonl (still run and recorded; default 3)")
    ap.add_argument("--seed", type=int, default=default_seed)
    ap.add_argument("--threads", type=int, default=8, help="OMP_NUM_THREADS / MKL_NUM_THREADS")
    ap.add_argument("--tf32", choices=("on", "off"), default="on",
                    help="on: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (the setting of the report's runs of these methods); "
                         "off: stock PyTorch precision")
    ap.add_argument("--sampler-interval", type=float, default=30.0, help="seconds; 0 disables the load sampler")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: inputs already recorded with status ok are skipped")
    ap.add_argument("--autotune-cache", default=None,
                    help="FlexGEMM autotune cache file, updated during the run (default: <out-dir>/autotune_cache.json, "
                         "i.e. every run starts without tuned kernels)")
    ap.add_argument("--hf-home", default=None, help="Hugging Face cache (HF_HOME)")
    ap.add_argument("--offline", action="store_true", help="do not contact the Hugging Face hub (HF_HUB_OFFLINE=1)")


def check_common_arguments(args) -> None:
    if args.warmup_requests < 0 or args.exclude_first < 0:
        raise SystemExit("--warmup-requests and --exclude-first must be >= 0")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.id is not None and not args.image:
        raise SystemExit("--id is only used with --image")


def configure_process(threads: int = 8, tf32: bool = True, hf_home: Optional[str] = None,
                      offline: bool = False, autotune_cache: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Process environment of the measured runs. Call before PyTorch is imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["ATTN_BACKEND"] = "flash_attn"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if tf32:
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    else:
        os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    if hf_home:
        os.environ["HF_HOME"] = str(Path(hf_home).expanduser().resolve())
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if autotune_cache:
        os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = str(Path(autotune_cache).expanduser().resolve())
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG", "ATTN_BACKEND",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "HF_HOME", "HF_HUB_OFFLINE", "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
            "CUDA_VISIBLE_DEVICES")
    return {k: os.environ.get(k) for k in keys}


# ----------------------------------------------------------------------------------------------- timing
def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class Stages:
    """Stage timer with a device synchronisation at both boundaries of every stage.

    A stage entered while another one is open is *nested*: it is reported but not part of ``stage_sum_s``. A stage
    entered several times is reported as the sum of its calls (``calls`` counts them). Single-threaded use only.
    """

    def __init__(self, sync: Callable[[], None] = cuda_sync):
        self.sync = sync
        self.times: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self.top_level_s = 0.0
        self._depth = 0

    @contextlib.contextmanager
    def stage(self, name: str):
        nested = self._depth > 0
        self.sync()
        started = time.perf_counter()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            self.sync()
            self.add(name, time.perf_counter() - started, nested)

    def add(self, name: str, seconds: float, nested: bool = False) -> None:
        key = f"{name}_s"
        self.times[key] = self.times.get(key, 0.0) + float(seconds)
        self.calls[key] = self.calls.get(key, 0) + 1
        if not nested:
            self.top_level_s += float(seconds)

    def get(self, name: str) -> float:
        return self.times.get(f"{name}_s", 0.0)

    def timings(self) -> Dict[str, float]:
        out = {k: round(v, 6) for k, v in self.times.items()}
        out["stage_sum_s"] = round(self.top_level_s, 6)
        return out


class _CallableProxy:
    """Stands in for a callable object stored in a dict: calls go through ``call``, attributes to the object."""

    def __init__(self, target, call):
        self.__dict__["_target"] = target
        self.__dict__["_call"] = call

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__dict__["_target"], name)


@contextlib.contextmanager
def timed_calls(stages: "Stages", targets: Sequence[Tuple[Any, str, str]]):
    """Time the calls of third-party callables inside the block, for pipelines whose single call offers no other
    way to split its time.

    ``targets``: ``(owner, attribute, stage name)`` with ``owner`` a module, an object or a dict (then
    ``attribute`` is a key). Each callable is replaced by a wrapper that calls it unchanged inside
    ``stages.stage(name)`` (with its device synchronisations); only the outermost instrumented call is timed when
    one calls another. Every original is restored when the block exits."""
    depth = [0]
    saved = []

    def wrap(original, name):
        def timed(*args, **kwargs):
            if depth[0]:
                return original(*args, **kwargs)
            depth[0] += 1
            try:
                with stages.stage(name):
                    return original(*args, **kwargs)
            finally:
                depth[0] -= 1
        return timed

    try:
        for owner, attr, name in targets:
            if isinstance(owner, dict):
                original = owner[attr]
                saved.append((owner, attr, original, True))
                owner[attr] = _CallableProxy(original, wrap(original, name))
            else:
                original = getattr(owner, attr)
                own = attr in vars(owner)
                saved.append((owner, attr, original if own else None, own))
                setattr(owner, attr, wrap(original, name))
        yield
    finally:
        for owner, attr, original, own in reversed(saved):
            if isinstance(owner, dict):
                owner[attr] = original
            elif own:
                setattr(owner, attr, original)
            else:
                delattr(owner, attr)


# ----------------------------------------------------------------------------------------------- frame maps
def frame_map(method: str, path=None) -> Dict[str, Any]:
    """The fixed frame map of ``method`` in eval/frame_maps/baseline_frame_maps.json, with the file's sha256."""
    path = Path(path) if path else FRAME_MAPS
    raw = path.read_bytes()
    data = json.loads(raw)
    if method not in data["methods"]:
        raise KeyError(f"{method!r} has no frame map in {path}")
    return {"entry": dict(data["methods"][method]), "convention": data.get("convention"),
            "sha256": hashlib.sha256(raw).hexdigest()}


def camera_record(ident: str, image: str, method: str, arm: str, timings: Dict[str, Any],
                  frame_maps_path=None) -> Dict[str, Any]:
    """Per-input record written next to the output as ``<id>/camera.json``.

    It has the layout of the package's camera file (docs/INSTALL.md, "Output format"), but the external methods
    estimate no camera: ``camera`` is null. The evaluation renders their outputs with the ground-truth camera pose
    after the method's fixed frame map (eval/metrics/baselines.py); that frame map is copied here, together with
    the sha256 of the frame-map file, so that a product records the convention it was generated for."""
    fm = frame_map(method, frame_maps_path)
    entry = fm["entry"]
    return {"object_id": ident, "image": str(image), "method": method, "arm": arm,
            "camera": None, "fov_source": None,
            "frame_map": {"method": method, **entry, "convention": fm["convention"], "file": FRAME_MAPS_REL,
                          "file_sha256": fm["sha256"]},
            "geometry_glb": entry.get("geometry_file"), "textured_glb": entry.get("textured_file"),
            "e2e_s": timings.get("e2e_s"), "stage_sum_s": timings.get("stage_sum_s")}


def expected_files(method: str) -> List[str]:
    entry = frame_map(method)["entry"]
    return sorted({f for f in (OUTPUT, entry.get("geometry_file"), entry.get("textured_file")) if f})


# ----------------------------------------------------------------------------------------------- methods
class Method:
    """One configuration of one external pipeline. Subclasses set ``name`` (key of the frame-map file: the frame of
    the GLB they write) and ``arm``, and implement :meth:`load` and :meth:`run`."""

    name = "?"
    arm = "stock"

    def load(self) -> Dict[str, Any]:
        """Load the models (not timed); returns facts recorded in run.json."""
        raise NotImplementedError

    def run(self, image_path: str, out_dir: Path, seed: int, stages: Stages) -> Dict[str, Any]:
        """Generate ``out_dir/model.glb`` (and any further file of the frame map) from ``image_path``; return
        statistics. The caller synchronises the device and reads the clock before and after."""
        raise NotImplementedError

    def after_request(self) -> Dict[str, Any]:
        """Untimed work after the clock has stopped (e.g. checksums of latents kept by :meth:`run`); returns
        statistics."""
        return {}

    def derived(self, timings: Dict[str, float]) -> Dict[str, float]:
        """Extra time keys computed from the stage times (e.g. sampling and post-processing)."""
        return {}

    def settings(self) -> Dict[str, Any]:
        return {}


def parts(timings: Dict[str, float], sampling: float, decoding: float, meshwork: float) -> Dict[str, float]:
    """The split of the report's table of the released pipelines, from the stage sum: ``part_sampling_s`` (image
    conditioning and flow or diffusion sampling), ``part_decoding_s`` (latent decoding and iso-surface extraction),
    ``part_meshwork_s`` (remeshing, simplification, UV unwrapping, baking, export) and ``part_other_s`` (input
    preparation and the unattributed remainder)."""
    total = float(timings.get("stage_sum_s", 0.0))
    return {"part_sampling_s": sampling, "part_decoding_s": decoding, "part_meshwork_s": meshwork,
            "part_other_s": total - sampling - decoding - meshwork}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor) -> str:
    """sha256 of the raw bytes of a tensor (bitwise comparison of latents between runs)."""
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes(order="C")).hexdigest()


def _recorded_ok(timing_path: Path) -> set:
    done = set()
    if timing_path.is_file():
        for line in timing_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(row["id"])
    return done


def run_one(method: Method, ident: str, image: str, out_dir: Path, seed: int) -> Dict[str, Any]:
    """One timed request. Returns ``{"timings", "stats"}``; the GLB and camera.json are in ``out_dir/<ident>/``."""
    import torch

    cuda = torch.cuda.is_available()
    item = out_dir / ident
    item.mkdir(parents=True, exist_ok=True)
    stages = Stages()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stats = method.run(image, item, seed, stages) or {}
    if cuda:
        torch.cuda.synchronize()
    e2e = time.perf_counter() - started
    stats.update(method.after_request())
    for name in expected_files(method.name):
        path = item / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{method.name}: expected output {path} is missing or empty")
    timings = stages.timings()
    timings["e2e_s"] = round(e2e, 6)
    timings.update({k: round(v, 6) for k, v in method.derived(timings).items()})
    if cuda:
        stats["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    stats["stage_calls"] = dict(stages.calls)
    stats["glb_bytes"] = (item / OUTPUT).stat().st_size
    (item / SIDECAR).write_text(json.dumps(camera_record(ident, image, method.name, method.arm, timings), indent=1))
    return {"timings": timings, "stats": stats}


def release_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def method_dirs(methods: Sequence[Method], out_dir: Path) -> Dict[str, Path]:
    """One directory per method: ``out_dir`` itself for a single method, ``out_dir/<arm>`` for several."""
    arms = [m.arm for m in methods]
    if len(set(arms)) != len(arms):
        raise ValueError(f"arms must be distinct: {arms}")
    if len(methods) == 1:
        return {methods[0].arm: out_dir}
    return {m.arm: out_dir / m.arm for m in methods}


def rotated(methods: Sequence[Method], index: int, rotate: bool) -> List[Method]:
    """The order of the methods for the input at list position ``index`` (0-based): with ``rotate``, method k runs
    first for the inputs with index % len(methods) == k, so that each pays first-call costs equally often."""
    methods = list(methods)
    if not rotate or len(methods) < 2:
        return methods
    k = index % len(methods)
    return methods[k:] + methods[:k]


def run_methods(methods: Sequence[Method], args, env: Dict[str, Optional[str]], rotate: bool = True) -> int:
    """Load ``methods`` once, run the warm-up requests, then every input of the list with every method."""
    add_package_to_path()
    from fastpixal3d.runtime.environment import environment_record, machine_record, start_sampler

    machine_at_start = machine_record()
    rows, list_path = inputs_from_args(args)
    out_dir = Path(args.out_dir).resolve()
    dirs = method_dirs(methods, out_dir)
    for arm, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        if (d / "timing.jsonl").exists() and not args.resume:
            raise SystemExit(f"{d / 'timing.jsonl'} exists; use a new --out-dir or --resume")
    done = {arm: _recorded_ok(d / "timing.jsonl") for arm, d in dirs.items()}
    # run.json of this process; a resumed run keeps the record of the earlier process(es) as run.json
    run_files = {}
    for arm, d in dirs.items():
        path, k = d / "run.json", 1
        while path.exists():
            path, k = d / f"run.resume{k}.json", k + 1
        run_files[arm] = path
    sampler = (start_sampler(str(out_dir / "sampler.jsonl"), args.sampler_interval)
               if args.sampler_interval > 0 else None)
    runs: Dict[str, Dict[str, Any]] = {}
    try:
        for method in methods:
            started = time.perf_counter()
            info = method.load()
            runs[method.arm] = {
                "method": method.name, "arm": method.arm, "settings": method.settings(), "load_info": info,
                "load_s": round(time.perf_counter() - started, 3), "argv": sys.argv, "list": list_path,
                "list_sha256": sha256_file(list_path) if list_path else None, "n_inputs": len(rows),
                "exclude_first": args.exclude_first, "warmup_requests": args.warmup_requests, "seed": args.seed,
                "rotate": bool(rotate and len(methods) > 1), "process_env": env,
                "machine_at_start": machine_at_start, "started_unix": time.time()}
            print(f"[{method.arm}] loaded in {runs[method.arm]['load_s']:.1f} s", flush=True)
        for method in methods:
            warm = []
            first_id, first_path = rows[0]
            for i in range(args.warmup_requests):
                res = run_one(method, f"{first_id}", first_path, dirs[method.arm] / "_warmup" / str(i), args.seed)
                warm.append(res["timings"]["e2e_s"])
                release_memory()
                print(f"[{method.arm}] warm-up request {i}: {warm[-1]:.3f} s", flush=True)
            runs[method.arm]["warmup_request_e2e_s"] = warm
            runs[method.arm]["environment"] = environment_record()
            run_files[method.arm].write_text(json.dumps(runs[method.arm], indent=1, default=str))
        failed = {m.arm: 0 for m in methods}
        logs = {arm: open(d / "timing.jsonl", "a") for arm, d in dirs.items()}
        try:
            for index, (ident, image) in enumerate(rows):
                for position, method in enumerate(rotated(methods, index, rotate)):
                    if ident in done[method.arm]:
                        continue
                    record = {"index": index + 1, "id": ident, "image": image, "arm": method.arm,
                              "warmup": index + 1 <= args.exclude_first, "arm_position": position}
                    try:
                        res = run_one(method, ident, image, dirs[method.arm], args.seed)
                        record.update(status="ok", e2e_s=res["timings"]["e2e_s"], timings=res["timings"],
                                      stats=res["stats"])
                    except Exception as error:  # recorded; the run continues with the next input
                        failed[method.arm] += 1
                        record.update(status="error", error=f"{type(error).__name__}: {error}",
                                      traceback=traceback.format_exc()[-3000:])
                    finally:
                        release_memory()
                    record["finished_unix"] = time.time()
                    logs[method.arm].write(json.dumps(record, default=str) + "\n")
                    logs[method.arm].flush()
                    shown = (f"{record['e2e_s']:.3f} s (stage sum {record['timings']['stage_sum_s']:.3f} s)"
                             if record["status"] == "ok" else record["error"])
                    print(f"[{method.arm}] {index + 1}/{len(rows)} {ident}: {shown}", flush=True)
        finally:
            for log in logs.values():
                log.close()
        machine_at_end = machine_record()
        for method in methods:
            runs[method.arm].update(finished_unix=time.time(), failed=failed[method.arm], machine_at_end=machine_at_end)
            run_files[method.arm].write_text(json.dumps(runs[method.arm], indent=1, default=str))
        return 0 if not any(failed.values()) else 1
    finally:
        if sampler is not None:
            sampler.terminate()


def main_for(build: Callable[[argparse.Namespace], Sequence[Method]], ap: argparse.ArgumentParser,
             argv: Optional[Iterable[str]] = None, rotate_default: bool = True) -> int:
    """Parse the arguments, set the process environment (before PyTorch is imported), build the methods, run."""
    args = ap.parse_args(None if argv is None else list(argv))
    check_common_arguments(args)
    cache = args.autotune_cache or str(Path(args.out_dir) / "autotune_cache.json")
    Path(cache).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    env = configure_process(threads=args.threads, tf32=args.tf32 == "on", hf_home=args.hf_home,
                            offline=args.offline, autotune_cache=cache)
    methods = build(args)
    return run_methods(methods, args, env, rotate=getattr(args, "rotate", rotate_default))
