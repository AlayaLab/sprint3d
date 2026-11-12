"""scorers3d.py -- ULIP-2 and Uni3D image-3D consistency.

The official code of both models is used unchanged from a checkout at a pinned commit (paths in the configuration:
models.ulip_code / models.uni3d_code; eval/fetch_metric_models.py clones them).  The two CUDA extensions they import
(pointnet2_ops for furthest point sampling, knn_cuda) are replaced by pure-torch equivalents registered in sys.modules
before the model code is imported, so no compilation is needed and the numerics are those of the reference
implementations (FPS started at point 0 as in pointnet2_ops; exact k-NN by torch.topk / torch.cdist).

Checkpoints (Hugging Face Hub, pinned in metrics/config.PINS):
  ULIP-2 : SFXX/ulip  ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt
           (PointBERT, 10k xyz+rgb points, 512 groups x 32, trans_dim 384 depth 18; pc_projection 768 -> 1280)
           image tower: OpenCLIP ViT-bigG-14 laion2b_s39b_b160k (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
  Uni3D  : BAAI/Uni3D  modelzoo/uni3d-g/model.pt (EVA-giant point transformer, 8192 xyz+rgb points, 512 groups x 64,
           pc_feat_dim 1408 -> embed 1024)
           image tower: OpenCLIP EVA02-E-14-plus laion2b_s9b_b144k (timm/eva02_enormous_patch14_plus_clip_224...)
Both repositories train the image side with Resize / Crop 224 + ImageNet mean / std normalisation -> the same test-time
transform here.  Point clouds: centroid-centred, max-norm unit sphere (pc_normalize in both repositories); colours in
[0, 1].  Score = cos(pc_embed, image_embed) with both embeddings L2-normalised (logit_scale not applied).
"""
from __future__ import annotations
import pickle

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

from .config import PINS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------- CUDA-extension stubs
def _load_checkpoint(path, **kwargs):
    """``torch.load`` with restricted deserialization first; the third-party checkpoints of Uni3D and
    ULIP-2 may hold objects other than tensors, which are read in full only when
    FASTPIXAL3D_TRUST_CHECKPOINT=1 is set (full unpickling can execute code: trusted files only)."""
    try:
        return torch.load(str(path), weights_only=True, **kwargs)
    except pickle.UnpicklingError as error:
        if os.environ.get("FASTPIXAL3D_TRUST_CHECKPOINT") != "1":
            raise RuntimeError(f"{path} holds objects other than tensors; set FASTPIXAL3D_TRUST_CHECKPOINT=1 to load it "
                               f"with full unpickling if you trust the file") from error
        return torch.load(str(path), weights_only=False, **kwargs)


def _fps_index(xyz: torch.Tensor, n: int) -> torch.Tensor:
    """pointnet2_ops.furthest_point_sample semantics: [B,N,3] -> [B,n] int32, first pick = index 0."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, n, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(n):
        idx[:, i] = far
        c = xyz[ar, far][:, None, :]
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx.to(torch.int32)


def _gather_operation(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """[B,C,N], [B,n] -> [B,C,n]"""
    B, C, _ = feat.shape
    return torch.gather(feat, 2, idx.long()[:, None, :].expand(B, C, idx.shape[1]))


class _KNN:
    def __init__(self, k=4, transpose_mode=False):
        self.k, self.transpose = k, transpose_mode

    def __call__(self, ref, query):
        if not self.transpose:            # [B,3,N], [B,3,M]
            ref, query = ref.transpose(1, 2), query.transpose(1, 2)
        d = torch.cdist(query.float(), ref.float())
        dist, idx = d.topk(self.k, dim=-1, largest=False)   # [B,M,k]
        if not self.transpose:
            return dist.transpose(1, 2), idx.transpose(1, 2)
        return dist, idx


def install_extension_stubs():
    if "pointnet2_ops" not in sys.modules:
        m = types.ModuleType("pointnet2_ops")
        u = types.ModuleType("pointnet2_ops.pointnet2_utils")
        u.furthest_point_sample = _fps_index
        u.gather_operation = _gather_operation
        m.pointnet2_utils = u
        sys.modules["pointnet2_ops"] = m
        sys.modules["pointnet2_ops.pointnet2_utils"] = u
    if "knn_cuda" not in sys.modules:
        k = types.ModuleType("knn_cuda")
        k.KNN = _KNN
        sys.modules["knn_cuda"] = k


def _load_package(name: str, path: Path):
    """import a package directory of a checkout under a private top-level name (relative imports work)."""
    if name in sys.modules:
        return sys.modules[name]
    init = path / "__init__.py"
    if init.is_file():
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(path)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    else:  # namespace package without __init__ (Uni3D/models)
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod.__package__ = name
        sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------- common helpers
def hf_local_file(repo: str, filename: str, revision: str | None = None) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo, filename, revision=revision)
        return Path(p) if isinstance(p, str) and os.path.exists(p) else None
    except Exception:
        return None


def image_tensor(img_uint8: np.ndarray, device) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return tf(Image.fromarray(np.asarray(img_uint8)))[None].to(device)


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    c = xyz.mean(0)
    p = xyz - c
    return p / max(float(np.sqrt((p ** 2).sum(1)).max()), 1e-12)


def _open_clip_model(model_name: str, weight_file: Path, device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=str(weight_file), device="cpu")
    model = model.eval().to(device=device, dtype=torch.float16)
    return model


def _missing_common(paths: dict, code_key: str, code_probe: str, ckpt_key: str, tower: str, mods) -> list:
    out = []
    code = paths.get(code_key)
    if not code or not (Path(code) / code_probe).is_file():
        out.append(f"model code ({code_key} = {code})")
    if not paths.get(ckpt_key) or not Path(paths[ckpt_key]).is_file():
        out.append(f"checkpoint ({ckpt_key} = {paths.get(ckpt_key)})")
    pin = PINS[tower]
    if hf_local_file(pin["repo"], "open_clip_model.safetensors", pin["revision"]) is None:
        out.append(f"OpenCLIP image tower {pin['repo']}@{pin['revision'][:8]} in HF_HOME")
    for mod in mods:
        if importlib.util.find_spec(mod) is None:
            out.append(f"python package {mod}")
    return out


# ---------------------------------------------------------------------------- Uni3D
class Uni3DModel:
    cfg = dict(pc_model="eva_giant_patch14_560", pretrained_pc="", drop_path_rate=0.0, pc_feat_dim=1408,
               embed_dim=1024, group_size=64, num_group=512, pc_encoder_dim=512, patch_dropout=0.0)
    n_points = 8192

    @staticmethod
    def missing(paths: dict) -> list:
        return _missing_common(paths, "uni3d_code", "models/uni3d.py", "uni3d_checkpoint", "openclip_eva02",
                               ("open_clip", "easydict", "timm"))

    def __init__(self, device="cuda", paths: dict | None = None):
        from easydict import EasyDict
        paths = paths or {}
        install_extension_stubs()
        self.device = torch.device(device)
        import timm
        _load_package("uni3d_code_models", Path(paths["uni3d_code"]) / "models")
        # models/uni3d.py pulls in losses.py -> the repository's `utils` package (distributed training helpers);
        # create_uni3d() is 3 lines, reproduced here on the repository's PointcloudEncoder instead.
        pe = importlib.import_module("uni3d_code_models.point_encoder")
        args = EasyDict(self.cfg)
        point_transformer = timm.create_model(args.pc_model, checkpoint_path=args.pretrained_pc,
                                              drop_path_rate=args.drop_path_rate)

        class _Uni3D(torch.nn.Module):
            def __init__(self, point_encoder):
                super().__init__()
                self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                self.point_encoder = point_encoder

            def encode_pc(self, pc):
                return self.point_encoder(pc[:, :, :3].contiguous(), pc[:, :, 3:].contiguous())

        self.model = _Uni3D(pe.PointcloudEncoder(point_transformer, args))
        ck = _load_checkpoint(paths["uni3d_checkpoint"], map_location="cpu", mmap=True)
        sd = ck["module"] if "module" in ck else ck.get("state_dict", ck)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.to(self.device).eval()
        pin = PINS["openclip_eva02"]
        self.clip = _open_clip_model("EVA02-E-14-plus", hf_local_file(pin["repo"], "open_clip_model.safetensors",
                                                                      pin["revision"]), self.device)

    @torch.no_grad()
    def embed_points(self, pc6: np.ndarray) -> torch.Tensor:
        xyz = pc_normalize(pc6[:, :3].astype(np.float32))
        rgb = pc6[:, 3:6].astype(np.float32)
        pc = torch.from_numpy(np.concatenate([xyz, rgb], 1))[None].to(self.device)
        e = self.model.encode_pc(pc)
        return torch.nn.functional.normalize(e.float(), dim=-1)[0]

    @torch.no_grad()
    def embed_image(self, img_uint8: np.ndarray) -> torch.Tensor:
        x = image_tensor(img_uint8, self.device).to(torch.float16)
        e = self.clip.encode_image(x)
        return torch.nn.functional.normalize(e.float(), dim=-1)[0]


# ---------------------------------------------------------------------------- ULIP-2
class ULIP2Model:
    n_points = 10000

    @staticmethod
    def missing(paths: dict) -> list:
        return _missing_common(paths, "ulip_code", "models/pointbert/point_encoder.py", "ulip2_checkpoint",
                               "openclip_bigg", ("open_clip", "easydict", "timm", "yaml"))

    def __init__(self, device="cuda", paths: dict | None = None):
        import yaml
        from easydict import EasyDict
        paths = paths or {}
        install_extension_stubs()
        self.device = torch.device(device)
        # ULIP uses absolute imports 'models.pointbert.*' -> import its 'models' package under its own name, scoped to
        # this process (Uni3D's package is loaded under a private name, so there is no clash).
        ulip_root = str(Path(paths["ulip_code"]))
        if ulip_root not in sys.path:
            sys.path.insert(0, ulip_root)
        if "models" in sys.modules and not getattr(sys.modules["models"], "__file__", "").startswith(ulip_root):
            del sys.modules["models"]
        pe = importlib.import_module("models.pointbert.point_encoder")
        cfg = EasyDict(yaml.safe_load(open(Path(ulip_root) / "models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml")))
        self.point_encoder = pe.PointTransformer_Colored(cfg.model, args=EasyDict(evaluate_3d=True))
        ck = _load_checkpoint(paths["ulip2_checkpoint"], map_location="cpu")
        sd = {k[len("module."):]: v for k, v in ck["state_dict"].items()}
        pe_sd = {k[len("point_encoder."):]: v for k, v in sd.items() if k.startswith("point_encoder.")}
        self.point_encoder.load_state_dict(pe_sd, strict=True)
        self.point_encoder = self.point_encoder.to(self.device).eval()
        self.pc_projection = sd["pc_projection"].to(self.device)
        pin = PINS["openclip_bigg"]
        self.clip = _open_clip_model("ViT-bigG-14", hf_local_file(pin["repo"], "open_clip_model.safetensors",
                                                                  pin["revision"]), self.device)

    @torch.no_grad()
    def embed_points(self, pc6: np.ndarray) -> torch.Tensor:
        xyz = pc_normalize(pc6[:, :3].astype(np.float32))
        rgb = pc6[:, 3:6].astype(np.float32)
        pc = torch.from_numpy(np.concatenate([xyz, rgb], 1))[None].to(self.device)
        feat = self.point_encoder(pc)
        e = feat @ self.pc_projection
        return torch.nn.functional.normalize(e.float(), dim=-1)[0]

    @torch.no_grad()
    def embed_image(self, img_uint8: np.ndarray) -> torch.Tensor:
        x = image_tensor(img_uint8, self.device).to(torch.float16)
        e = self.clip.encode_image(x)
        return torch.nn.functional.normalize(e.float(), dim=-1)[0]
