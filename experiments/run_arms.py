#!/usr/bin/env python3
"""Run a list of configurations (experiments/configs/*/*.yaml) with the timing protocol of docs/TIMING.md.

    python experiments/run_arms.py experiments/configs/cumulative --list timing_inputs.jsonl \
        --out-dir runs/cumulative --students /path/to/students --autotune-seed autotune_cache.json

Every configuration runs in its own process, one after the other, through ``scripts/time_e2e.py``:
the same input list in the same order, models loaded once per process (not timed), one untimed
warm-up request on the first input (this package; none for Pixal3D's own script), per-stage timers,
the machine record and load sampler. Every input counts (``--exclude-first 0``, as the report's
cumulative measurement and one-at-a-time removals); ``--exclude-first N`` marks the first N inputs as warm-up, the
convention of the dedicated timing run (``scripts/time_e2e.py``, docs/TIMING.md). ``method: official`` runs Pixal3D's inference script at its defaults; every other
configuration runs this package with the configuration's levers and sampling settings.

Arguments: configuration files or directories (a directory means all ``*.yaml`` in it, in file-name
order; the file-name order is the order of the measurement, and in the cumulative set the controls
``x1_*``-``x3_*`` sort after the chain). Output, one directory per configuration (named after the file
without its extension), plus ``arms.json`` in ``--out-dir``: the order, the name, the report's row name
(the ``# Paper row:`` comment), the block or the row a control belongs to, the sha256 of each
configuration file, the command, exit code and wall time of each process. A directory that already holds a finished run (``timing.jsonl``
with one line per input) is skipped, so an interrupted list can be resumed; ``--force`` reruns it into a
fresh directory name (``<name>.<n>``).

Arguments after ``--`` go to every ``time_e2e.py`` call unchanged (e.g. ``-- --tf32 off``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
TIME_E2E = REPO / "scripts" / "time_e2e.py"
PAPER_ROW = re.compile(r"^#\s*Paper row:\s*(.+?)\s*$")


def config_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.yaml")))
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"no such configuration: {p}")
    if not files:
        raise SystemExit("no configuration files given")
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        raise SystemExit(f"two configurations with the same file name: {sorted({s for s in stems if stems.count(s) > 1})}")
    return files


def read_config(path: Path) -> Dict[str, Any]:
    """The parsed configuration plus ``paper_row`` (from the ``# Paper row:`` comment) and ``sha256``."""
    import yaml

    text = path.read_text()
    spec = yaml.safe_load(text) or {}
    paper = next((m.group(1) for m in map(PAPER_ROW.match, text.splitlines()) if m), None)
    return {**spec, "paper_row": paper, "file": str(path), "stem": path.stem,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def n_inputs(list_path: Path) -> int:
    return sum(1 for line in list_path.read_text().splitlines() if line.strip())


def finished(run_dir: Path, expected: int) -> bool:
    log = run_dir / "timing.jsonl"
    return log.is_file() and sum(1 for line in log.read_text().splitlines() if line.strip()) == expected


def command(cfg: Dict[str, Any], args, run_dir: Path) -> List[str]:
    method = cfg.get("method", "fast")
    if method not in ("fast", "official"):
        raise SystemExit(f"{cfg['file']}: method must be fast or official, not {method!r}")
    cmd = [args.python, str(TIME_E2E), "--method", method, "--list", str(Path(args.list).resolve()),
           "--out-dir", str(run_dir), "--exclude-first", str(args.exclude_first),
           "--autotune-cache", str(run_dir / "autotune_cache.json")]
    if method == "fast":
        cmd += ["--config", str(Path(cfg["file"]).resolve())]
        if args.students:
            cmd += ["--students", args.students]
        if args.student_checkpoints:
            cmd += ["--student-checkpoints", args.student_checkpoints]
        if args.dump:
            cmd += ["--dump-dir", str(run_dir / "dump")]
    for flag in ("pixal3d", "pixal3d_src", "hf_home", "torch_home"):
        value = getattr(args, flag)
        if value:
            cmd += ["--" + flag.replace("_", "-"), value]
    if args.offline:
        cmd.append("--offline")
    return cmd + list(args.extra)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="configuration files or directories")
    ap.add_argument("--list", required=True, help="JSON lines with 'id' and 'path' (the same for every configuration)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None, help="comma-separated configuration names or file stems to run")
    ap.add_argument("--exclude-first", type=int, default=0,
                    help="inputs marked as warm-up in every run (default: none, as the report)")
    ap.add_argument("--autotune-seed", default=None,
                    help="FlexGEMM autotune cache copied into each run directory before the run (each run gets "
                         "its own copy; default: each run starts from an empty cache)")
    ap.add_argument("--students", default=None)
    ap.add_argument("--student-checkpoints", default=None)
    ap.add_argument("--pixal3d", default=None)
    ap.add_argument("--pixal3d-src", default=None)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--torch-home", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dump", action="store_true", help="also write the parity records (dump/<id>/) of every input")
    ap.add_argument("--force", action="store_true", help="rerun finished configurations into a new directory")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    args.extra = extra

    files = config_files(args.configs)
    configs = [{**read_config(f), "order": k} for k, f in enumerate(files)]
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c.get("name") in wanted or c["stem"] in wanted]
        if not configs:
            raise SystemExit(f"--only {args.only}: no configuration matches")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = n_inputs(Path(args.list))
    manifest_path = out_dir / "arms.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"arms": []}
    manifest.update(list=str(Path(args.list).resolve()),
                    list_sha256=hashlib.sha256(Path(args.list).read_bytes()).hexdigest(), n_inputs=expected,
                    exclude_first=args.exclude_first, extra_args=extra)
    by_dir = {a["run_dir"]: a for a in manifest["arms"]}

    failed = 0
    for position, cfg in enumerate(configs):
        order = cfg["order"]
        run_dir = out_dir / cfg["stem"]
        if finished(run_dir, expected) and not args.force:
            print(f"[run_arms] {cfg['stem']}: finished, skipped", flush=True)
            continue
        if args.force and run_dir.exists():
            k = 1
            while (out_dir / f"{cfg['stem']}.{k}").exists():
                k += 1
            run_dir = out_dir / f"{cfg['stem']}.{k}"
        elif run_dir.exists() and (run_dir / "timing.jsonl").exists():
            # interrupted earlier: keep it for inspection, start over next to it
            shutil.move(str(run_dir), str(out_dir / f"{cfg['stem']}.interrupted.{int(time.time())}"))
        cmd = command(cfg, args, run_dir)
        print(f"[run_arms] {position + 1}/{len(configs)} {cfg['stem']}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.autotune_seed:
            shutil.copyfile(args.autotune_seed, run_dir / "autotune_cache.json")
        started = time.time()
        with open(run_dir / "stdout.log", "w") as log:
            code = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)).returncode
        complete = finished(run_dir, expected)
        failed += int(code != 0 or not complete)
        entry = {"order": order, "name": cfg.get("name", cfg["stem"]), "stem": cfg["stem"],
                 "paper_row": cfg.get("paper_row"), "block": cfg.get("block"), "control_of": cfg.get("control_of"),
                 "method": cfg.get("method", "fast"), "config": cfg["file"],
                 "config_sha256": cfg["sha256"], "run_dir": str(run_dir), "command": cmd, "exit_code": code,
                 "complete": complete, "started_unix": started, "wall_s": round(time.time() - started, 1)}
        by_dir[str(run_dir)] = entry
        manifest["arms"] = sorted(by_dir.values(), key=lambda a: (a["order"], a["run_dir"]))
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[run_arms] {cfg['stem']}: exit {code}, {'complete' if complete else 'INCOMPLETE'}, "
              f"{entry['wall_s']:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
