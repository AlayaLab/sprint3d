"""Shared loading, critical-path attribution and bootstrap helpers of the experiment analyses.

Inputs are the run directories written by ``experiments/run_arms.py`` (one per configuration, each
the output of ``scripts/time_e2e.py``) and the configuration files that produced them. A row of an
analysis is one configuration; its runs directory is ``<runs>/<file stem>``.

The cumulative set is a chain plus controls. The chain is every configuration without a ``control_of``
key, in file order (the order of the measurement): the first row is the reference, every later row
names its ``block`` (``post_processing``, ``flow_engineering``, ``distillation``; each block a contiguous
run of rows). A control (``control_of: <name of a chain row>``) is the chain row with the levers of its
``removes`` switched off or those of its ``adds`` switched on; it is compared with that row only.

Statistics: the measured inputs are all successful inputs of the list, as in the report's cumulative
measurement and one-at-a-time removals (``exclude_first`` > 0 leaves out the first ones, the convention of the dedicated
timing run in docs/TIMING.md); bootstrap intervals are 95 % percentile intervals over 10,000 resamples with a fixed seed (0),
drawn per statistic with ``numpy.random.default_rng(0)``. Paired differences are taken on the inputs
measured in both runs. Two resampling units are used:

* ``input`` -- every input is resampled on its own (the waterfall and lever tables);
* ``object`` -- the inputs of one object (ids ``<object>_v<view>``) are resampled together (the
  one-lever-off and same-step tables).
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

N_BOOT = 10_000
SEED = 0
PAPER_ROW = re.compile(r"^#\s*Paper row:\s*(.+?)\s*$")
OBJECT_OF = re.compile(r"^(.*)_v\d+$")

# Critical-path stages (column order of every table and of the figure).
STAGES = ("image", "structure", "coarse_shape", "fine_shape", "mesh", "uv", "texture", "bake", "export",
          "unsplit", "other")
STAGE_LABELS = {"image": "image and camera", "structure": "structure", "coarse_shape": "coarse shape",
                "fine_shape": "fine shape", "mesh": "mesh preparation", "uv": "UV atlas", "texture": "texture",
                "bake": "bake", "export": "export", "unsplit": "mesh to GLB (not split)", "other": "other"}


# Blocks of the cumulative measurement (``block`` key of a chain row) and their labels in the report.
BLOCK_LABELS = {"post_processing": "mesh post-processing", "flow_engineering": "flow engineering",
                "distillation": "distillation"}


# ----------------------------------------------------------------------------------------------- configurations
def read_config(path: Path) -> Dict[str, Any]:
    import yaml

    text = Path(path).read_text()
    spec = yaml.safe_load(text) or {}
    paper = next((m.group(1) for m in map(PAPER_ROW.match, text.splitlines()) if m), None)
    return {**spec, "paper_row": paper, "file": str(path), "stem": Path(path).stem}


def config_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        files.extend(sorted(p.glob("*.yaml")) if p.is_dir() else [p])
    return files


# ----------------------------------------------------------------------------------------------- runs
def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    log = Path(run_dir) / "timing.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def measured(rows: Sequence[Mapping[str, Any]], exclude_first: int) -> Dict[str, Mapping[str, Any]]:
    """Successful inputs after the warm-up ones, by input id."""
    return {r["id"]: r for r in rows if r.get("status") == "ok" and int(r["index"]) > exclude_first}


def object_of(input_id: str) -> str:
    m = OBJECT_OF.match(input_id)
    return m.group(1) if m else input_id


# ----------------------------------------------------------------------------------------------- attribution
def _g(d: Mapping[str, Any], key: str) -> float:
    v = d.get(key)
    return float(v) if v is not None else 0.0


def attribute(row: Mapping[str, Any]) -> Dict[str, float]:
    """Seconds of each critical-path stage of one input (they sum to ``e2e_s``).

    This package: everything runs on the main thread except the overlap window, where the texture
    flow (main thread) runs concurrently with a worker thread -- the whole mesh preparation
    (``mesh_prep_worker_thread``) or only the UV atlas (otherwise, after the GPU phase on the main
    thread). Inside the window the longer branch is charged to the critical path; the waiting time
    of the window beyond it goes to ``other``. Pixal3D's own script runs serially; its mesh-to-GLB
    step is not split (``unsplit``)."""
    t, s = row.get("timings") or {}, row.get("stats") or {}
    cp = {k: 0.0 for k in STAGES}
    e2e = _g(t, "e2e_s")
    if "to_glb_s" in t and "glb_bake_finalize_s" in t:  # this package
        cp["image"] = _g(t, "preprocess_s") + _g(t, "camera_s")
        cp["structure"] = sum(_g(t, k) for k in ("ss_condition_s", "ss_initial_noise_s", "ss_rollout_s",
                                                 "ss_decode_support_s"))
        cp["coarse_shape"] = sum(_g(t, k) for k in ("lr_condition_s", "lr_initial_noise_s", "lr_rollout_s"))
        cp["fine_shape"] = sum(_g(t, k) for k in ("hr_upsample_support_s", "hr_condition_s", "hr_initial_noise_s",
                                                  "hr_rollout_s"))
        decode = _g(t, "shape_decode_s") + _g(t, "shape_fill_holes_s")
        bake = _g(t, "glb_finish_s") + _g(t, "glb_bake_finalize_s")
        mt = s.get("mesh_timings") or {}
        cluster = _g(mt, "cluster_s")
        uv_total = _g(mt.get("uv") or {}, "total_s")
        tex = _g(s, "tex_stream_s")
        worker = _g(s, "mesh_worker_s")
        gpu_phase = _g(t, "glb_gpu_phase_s")
        window = s.get("mesh_overlap_window_s")
        if s.get("mesh_schedule", "worker_thread") == "worker_thread":
            window = float(window) if window is not None else _g(t, "to_glb_s") - bake
            uv_w = cluster + uv_total
            mesh_w = max(worker - uv_w, 0.0)
            if tex >= worker:
                cp["texture"] += tex
                longest = tex
            else:
                cp["mesh"] += mesh_w
                cp["uv"] += worker - mesh_w
                longest = worker
            accounted = window + bake
        else:
            window = float(window) if window is not None else _g(t, "to_glb_s") - bake - gpu_phase
            cp["mesh"] += max(gpu_phase - cluster, 0.0)
            cp["uv"] += min(cluster, gpu_phase)
            if tex >= worker:
                cp["texture"] += tex
                longest = tex
            else:
                cp["uv"] += worker
                longest = worker
            accounted = gpu_phase + window + bake
        cp["mesh"] += decode
        cp["bake"] = bake
        cp["export"] = _g(t, "export_s")
        cp["other"] = _g(t, "residual_s") + (_g(t, "to_glb_s") - accounted) + (window - longest)
    else:  # Pixal3D's inference script (fastpixal3d.official)
        cp["image"] = _g(t, "preprocess_image_s") + _g(t, "camera_s")
        cp["structure"] = _g(t, "cond_ss_s") + _g(t, "sparse_structure_s")
        cp["coarse_shape"] = _g(t, "shape_coarse_s")
        cp["fine_shape"] = max(_g(t, "shape_sampler_calls_s") - _g(t, "shape_coarse_s"), 0.0)
        cp["texture"] = _g(t, "texture_s")
        cp["mesh"] = _g(t, "decode_s")
        cp["unsplit"] = _g(t, "to_glb_s")
        cp["export"] = _g(t, "export_s")
        cp["other"] = e2e - sum(cp.values())
    return cp


# ----------------------------------------------------------------------------------------------- bootstrap
def boot_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
            groups: Optional[Sequence[str]] = None, n: int = N_BOOT, seed: int = SEED) -> Optional[List[float]]:
    """95 % percentile interval of ``stat`` over ``n`` resamples. With ``groups``, whole groups (objects)
    are resampled with replacement and all their values enter each resample."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return None
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, x.size, size=(n, x.size))
        boots = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(axis=1)
    else:
        keys = sorted(set(groups))
        members = [np.flatnonzero(np.asarray(groups) == k) for k in keys]
        draws = rng.integers(0, len(keys), size=(n, len(keys)))
        boots = np.array([stat(x[np.concatenate([members[j] for j in d])]) for d in draws])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(lo), float(hi)]


def pct(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def paired(prev: Mapping[str, Mapping[str, Any]], cur: Mapping[str, Mapping[str, Any]],
           value: Callable[[Mapping[str, Any]], float] = lambda r: float(r["e2e_s"]),
           unit: str = "input") -> Dict[str, Any]:
    """Paired difference cur - prev over the inputs measured in both: mean and median with bootstrap
    intervals (resampling unit ``input`` or ``object``)."""
    ids = sorted(set(prev) & set(cur))
    if not ids:
        return {"n": 0}
    d = np.array([value(cur[i]) - value(prev[i]) for i in ids])
    groups = [object_of(i) for i in ids] if unit == "object" else None
    out = {"n": len(ids), "n_objects": len({object_of(i) for i in ids}), "unit": unit,
           "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
           "ci95_mean": boot_ci(d, np.mean, groups), "ci95_median": boot_ci(d, np.median, groups),
           "median_prev": st.median(value(prev[i]) for i in ids), "median_cur": st.median(value(cur[i]) for i in ids)}
    return out


def fmt_ci(ci: Optional[Sequence[float]], digits: int = 3) -> str:
    return "" if not ci else f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"


# ----------------------------------------------------------------------------------------------- table rows
def load_table(config_paths: Iterable[str], runs: Path, exclude_first: int = 0) -> List[Dict[str, Any]]:
    """One entry per configuration (file order): config, measured inputs and per-input attribution."""
    table = []
    for path in config_files(config_paths):
        cfg = read_config(path)
        rows = load_rows(Path(runs) / cfg["stem"])
        meas = measured(rows, exclude_first)
        table.append({"config": cfg, "rows": rows, "measured": meas, "n_failed": sum(r.get("status") != "ok" for r in rows),
                      "attribution": {i: attribute(r) for i, r in meas.items()}})
    return table


def name_of(cfg: Mapping[str, Any]) -> str:
    return cfg.get("name", cfg["stem"])


def split_chain(table: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(chain, controls) of :func:`load_table`'s entries: the configurations without ``control_of`` in file
    order, and the controls."""
    chain = [e for e in table if not e["config"].get("control_of")]
    controls = [e for e in table if e["config"].get("control_of")]
    return chain, controls


def block_spans(chain: List[Dict[str, Any]]) -> List[Tuple[str, str, int, int]]:
    """``(block, label, from, to)`` per block of the chain, in order: ``from`` is the index of the row
    before the block's first row, ``to`` the index of its last row (so that consecutive blocks share
    their boundary row). Rows without ``block`` (the reference) are skipped."""
    spans: List[List[Any]] = []
    for k, entry in enumerate(chain):
        block = entry["config"].get("block")
        if not block:
            continue
        if spans and spans[-1][0] == block and spans[-1][3] == k - 1:
            spans[-1][3] = k
            continue
        if any(s[0] == block for s in spans):
            raise ValueError(f"{entry['config']['stem']}: block {block!r} is not a contiguous run of rows")
        if k == 0:
            raise ValueError(f"{entry['config']['stem']}: the first row of the chain is the reference, without block")
        spans.append([block, BLOCK_LABELS.get(block, block), k - 1, k])
    return [tuple(s) for s in spans]


def control_pairs(table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One record per control: ``without`` and ``with`` (the entries ordered so that ``with`` has the
    levers that differ), ``levers`` (those levers) and the control's entry. The effect of the levers is
    ``with - without``."""
    chain, controls = split_chain(table)
    by_name = {name_of(e["config"]): e for e in chain}
    out = []
    for entry in controls:
        cfg = entry["config"]
        ref = by_name.get(cfg["control_of"])
        if ref is None:
            raise ValueError(f"{cfg['stem']}: control_of {cfg['control_of']!r} is not a row of the chain")
        adds, removes = list(cfg.get("adds") or []), list(cfg.get("removes") or [])
        if bool(adds) == bool(removes):
            raise ValueError(f"{cfg['stem']}: a control either adds or removes levers (not both, not neither)")
        without, with_ = (entry, ref) if removes else (ref, entry)
        out.append({"control": entry, "reference": ref, "without": without, "with": with_,
                    "levers": removes or adds})
    return out


def paired_ratio(prev: Mapping[str, Mapping[str, Any]], cur: Mapping[str, Mapping[str, Any]],
                 unit: str = "object", n: int = N_BOOT, seed: int = SEED) -> Dict[str, Any]:
    """Speed-up ``median(prev) / median(cur)`` of the end-to-end time over the inputs measured in both,
    with a 95 % percentile interval (inputs, or the inputs of one object together, resampled with
    replacement; both rows of a resample use the same inputs)."""
    ids = sorted(set(prev) & set(cur))
    if not ids:
        return {"n": 0}
    a = np.array([float(prev[i]["e2e_s"]) for i in ids])
    b = np.array([float(cur[i]["e2e_s"]) for i in ids])
    rng = np.random.default_rng(seed)
    if unit == "object":
        groups = [object_of(i) for i in ids]
        keys = sorted(set(groups))
        members = [np.flatnonzero(np.asarray(groups) == k) for k in keys]
        draws = rng.integers(0, len(keys), size=(n, len(keys)))
        idx = [np.concatenate([members[j] for j in d]) for d in draws]
    else:
        idx = list(rng.integers(0, len(ids), size=(n, len(ids))))
    boots = np.array([np.median(a[i]) / np.median(b[i]) for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(ids), "n_objects": len({object_of(i) for i in ids}), "unit": unit,
            "factor": float(np.median(a) / np.median(b)), "ci95": [float(lo), float(hi)],
            "median_prev": float(np.median(a)), "median_cur": float(np.median(b))}


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
