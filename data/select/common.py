"""Configuration of the selection scripts (``--config config.yaml``) and shared helpers."""
import argparse
import os
from pathlib import Path


def config(description=""):
    import yaml

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="data/select/config.yaml (see config.example.yaml)")
    args = p.parse_args()
    path = Path(args.config).resolve()
    cfg = yaml.safe_load(path.read_text())
    base = path.parent

    def resolve(value):
        if value is None:
            return None
        v = Path(value).expanduser()
        return str(v if v.is_absolute() else (base / v).resolve())

    inp = cfg["inputs"]
    for key in ("trellis_metadata", "sketchfab_manifest", "objaverse_metadata", "toys4k_audit_list", "earlier_5k_list",
                "precision_dir"):
        inp[key] = resolve(inp.get(key))
    inp["raw"] = {k: resolve(v) for k, v in (inp.get("raw") or {}).items()}
    inp["extra_toys4k_lists"] = [resolve(v) for v in (inp.get("extra_toys4k_lists") or [])]
    cfg["work_dir"] = resolve(cfg["work_dir"])
    cfg["processes"] = int(cfg.get("processes", 4))
    os.makedirs(cfg["work_dir"], exist_ok=True)
    os.makedirs(out_dir(cfg), exist_ok=True)
    return cfg


def work(cfg, name):
    return os.path.join(cfg["work_dir"], name)


def out_dir(cfg):
    return os.path.join(cfg["work_dir"], "out")
