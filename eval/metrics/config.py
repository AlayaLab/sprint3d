"""config.py -- the evaluation configuration (YAML) and the pinned third-party metric models.

Every machine-specific path lives in the configuration file (see eval/config.example.yaml for the documented keys).
Relative paths in the file are resolved against the directory that contains the file.  The model revisions and
sha256 sums below are the ones used for the reported numbers; eval/fetch_metric_models.py downloads exactly these
files into the layout `model_paths()` expects.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

# ---------------------------------------------------------------------------- pinned metric models
PINS = {
    "clip": {"repo": "openai/clip-vit-large-patch14", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
             "files": {"model.safetensors": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
                       "config.json": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
                       "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
                       "tokenizer.json": "a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4",
                       "tokenizer_config.json": "deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129",
                       "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
                       "vocab.json": "3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69",
                       "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"}},
    "openclip_bigg": {"repo": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                      "revision": "743c27bd53dfe508a0ade0f50698f99b39d03bec",
                      "files": {"open_clip_model.safetensors":
                                "db71bd27925917762c506c1b8a08e191d236e5586b5cd99f77d11aced578ec34",
                                "open_clip_config.json":
                                "5bd7ad69ec151ba9e4a5e92a622655cb88cd6c76764df97a1ea0279062755e7b"}},
    "openclip_eva02": {"repo": "timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k",
                       "revision": "0162f50c2742be3c90b3a81e4e8066fc5a90e125",
                       "files": {"open_clip_model.safetensors":
                                 "dcdd43722d13420fd9682b162167bc10e1d2afe89f3da90224adf67658669d97",
                                 "open_clip_config.json":
                                 "7dd195154d6f3c5d1d00ef8a80f84dd54c6384e34e3df1dbe71b3142c505a2ac"}},
    "ulip2_checkpoint": {"repo": "SFXX/ulip", "repo_type": "dataset", "revision": "3f33d96e6bfec2a793d06acf0df4c591ddd47489",
                         "file": "ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt",
                         "sha256": "a4b5ed9799d5841a1646e0fb7d24cb8dcdd3b3e6fab2e2575de2531f71274adb"},
    "uni3d_checkpoint": {"repo": "BAAI/Uni3D", "revision": "3d8233b76aa350d72f6213ecd2123c2026b42355",
                         "file": "modelzoo/uni3d-g/model.pt",
                         "sha256": "aa1f163bb8c34d7eb4bbb7187e8aaf78879b29df02d08acfbeb0671e2a218c3f"},
    "dinov2_weights": {"url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                       "sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"},
    "torchvision_alexnet": {"url": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
                            "sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"},
    "torchvision_vgg16": {"url": "https://download.pytorch.org/models/vgg16-397923af.pth",
                          "sha256": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"},
    "lpips_linear_layers": {"package": "lpips==0.1.4",
                            "files": {"weights/v0.1/alex.pth": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
                                      "weights/v0.1/vgg.pth": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"}},
    "code_ulip": {"git": "https://github.com/salesforce/ULIP.git", "commit": "95d480fe2b16c06d0558c60b5cfea981b4cdc8eb"},
    "code_uni3d": {"git": "https://github.com/baaivision/Uni3D.git", "commit": "64e03c3c42c196e8cb5ed03857810af9fc9ac39c"},
    "code_dinov2": {"git": "https://github.com/facebookresearch/dinov2.git", "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8"},
}


def model_paths(models_dir) -> dict:
    """the directory layout written by fetch_metric_models.py"""
    d = Path(models_dir)
    return {"hf_home": d / "hf", "torch_home": d / "torch",
            "ulip2_checkpoint": d / "ulip2" / PINS["ulip2_checkpoint"]["file"],
            "uni3d_checkpoint": d / "uni3d" / PINS["uni3d_checkpoint"]["file"],
            "dinov2_weights": d / "dinov2" / "dinov2_vitl14_pretrain.pth",
            "ulip_code": d / "code" / "ULIP", "uni3d_code": d / "code" / "Uni3D", "dinov2_code": d / "code" / "dinov2"}


# ---------------------------------------------------------------------------- configuration file
DEFAULTS = {
    "dataset": {"root": None, "list": "lists/validation_400.csv", "list_sha256": None,
                "source_dirs": {"sketchfab": "ObjaverseXL_sketchfab", "abo": "ABO", "hssd": "HSSD", "3dfuture": "3D-FUTURE"}},
    "models": {"dir": None, "hf_home": None, "torch_home": None, "offline": True,
               "ulip_code": None, "ulip2_checkpoint": None, "uni3d_code": None, "uni3d_checkpoint": None,
               "dinov2_code": None, "dinov2_weights": None},
    "evaluation": {"device": "cuda", "background": "white", "table1_region": "overlap",
                   "frame_maps": "frame_maps/baseline_frame_maps.json", "reference_arm": "pixal3d_official",
                   "skip": ["vs_reference"], "n_boot": 10000},
}
PATH_KEYS = {"dataset": ("root", "list"),
             "models": ("dir", "hf_home", "torch_home", "ulip_code", "ulip2_checkpoint", "uni3d_code", "uni3d_checkpoint",
                        "dinov2_code", "dinov2_weights"),
             "evaluation": ("frame_maps",)}
SUITE_DIR = Path(__file__).resolve().parent.parent


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "source_dirs":
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None) -> dict:
    """defaults <- YAML file.  Relative paths of the file resolve against its directory; relative default paths
    against eval/."""
    import yaml
    cfg = copy.deepcopy(DEFAULTS)
    user = {}
    if path is not None:
        user = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        for sec, vals in user.items():
            bad = set(vals or {}) - set(DEFAULTS[sec])
            if bad:
                raise ValueError(f"unknown keys in section {sec!r}: {sorted(bad)}")
        cfg = _merge(cfg, user)
    for sec, keys in PATH_KEYS.items():
        for k in keys:
            v = cfg[sec].get(k)
            if v is None:
                continue
            p = Path(os.path.expanduser(str(v)))
            base = Path(path).resolve().parent if k in (user.get(sec) or {}) else SUITE_DIR
            cfg[sec][k] = str(p if p.is_absolute() else (base / p).resolve())
    m = cfg["models"]
    if m["dir"]:
        for k, p in model_paths(m["dir"]).items():
            if m.get(k) is None:
                m[k] = str(p)
    return cfg


def apply_environment(cfg: dict) -> dict:
    """model caches -> environment (must run before transformers / open_clip / torch.hub are used)."""
    m = cfg["models"]
    env = {}
    if m.get("hf_home"):
        env["HF_HOME"] = m["hf_home"]
    if m.get("torch_home"):
        env["TORCH_HOME"] = m["torch_home"]
    if m.get("offline"):
        env["HF_HUB_OFFLINE"] = "1"
    os.environ.update(env)
    return env
