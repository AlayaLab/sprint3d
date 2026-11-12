"""Records of the machine, the process environment and the machine load during a timing run."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE",
            "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG", "ATTN_BACKEND", "CUDA_VISIBLE_DEVICES",
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH")
GPU_QUERY = "utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu"


def _read(path) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _run(cmd, timeout: int = 30) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_select() -> list:
    """``nvidia-smi -i`` arguments for the GPUs this process can see (all GPUs if not restricted)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and all(part.strip().isdigit() for part in visible.split(",")):
        return ["-i", visible]
    return []


def machine_record() -> Dict[str, Any]:
    """CPU, memory, CPU quota, GPU and load at the time of the call (no PyTorch needed)."""
    cpu_model = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    mem_total = next((l for l in (_read("/proc/meminfo") or "").splitlines() if l.startswith("MemTotal")), None)
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kernel": platform.release(), "python": sys.version.split()[0], "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(), "cpus_allowed": len(os.sched_getaffinity(0)), "mem_total": mem_total,
        "cpu_quota": {"cgroup_v2_cpu_max": _read("/sys/fs/cgroup/cpu.max"),
                      "cgroup_v1_cfs_quota_us": _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
                      or _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
                      "cgroup_v1_cfs_period_us": _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
                      or _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")},
        "loadavg": _read("/proc/loadavg"), "cpu_pressure": _read("/proc/pressure/cpu"),
        "gpu": _run(["nvidia-smi", *_gpu_select(),
                     "--query-gpu=index,name,memory.total,driver_version,clocks.max.sm,power.limit",
                     "--format=csv,noheader"]),
        "gpu_processes": _run(["nvidia-smi", *_gpu_select(), "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"]),
    }


def environment_record() -> Dict[str, Any]:
    """Process environment and the effective PyTorch settings (call after PyTorch is imported)."""
    record: Dict[str, Any] = {"env": {k: os.environ.get(k) for k in ENV_KEYS}}
    try:
        import torch

        record.update(torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                      device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                      torch_threads=torch.get_num_threads(),
                      matmul_allow_tf32_flag=bool(torch.backends.cuda.matmul.allow_tf32),
                      cudnn_allow_tf32_flag=bool(torch.backends.cudnn.allow_tf32),
                      float32_matmul_precision=torch.get_float32_matmul_precision())
    except ImportError:
        pass
    return record


def _cpu_stat() -> Optional[str]:
    text = _read("/sys/fs/cgroup/cpu.stat") or _read("/sys/fs/cgroup/cpu,cpuacct/cpu.stat")
    return None if text is None else " ".join(text.split())


def sample_forever(path: str, interval: float = 30.0) -> None:
    """Append one JSON line per interval: load average, CPU throttling, CPU pressure, GPU state."""
    with open(path, "a") as out:
        while True:
            gpu = _run(["nvidia-smi", *_gpu_select(), f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                       timeout=20)
            out.write(json.dumps({"unix": time.time(), "loadavg": _read("/proc/loadavg"), "cpu_stat": _cpu_stat(),
                                  "cpu_pressure": _read("/proc/pressure/cpu"), "gpu": gpu}) + "\n")
            out.flush()
            time.sleep(interval)


def start_sampler(path: str, interval: float = 30.0):
    """Start :func:`sample_forever` in a separate low-priority process; returns the process handle."""
    code = ("import sys; from fastpixal3d.runtime.environment import sample_forever; "
            "sample_forever(sys.argv[1], float(sys.argv[2]))")
    src = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
    return subprocess.Popen([sys.executable, "-c", code, str(path), str(interval)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=lambda: os.nice(19))
