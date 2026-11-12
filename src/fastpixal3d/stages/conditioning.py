"""Image conditioning for the four flows (lever ``shared_conditioning``).

Pixal3D builds four DINOv3 projection extractors from the same frozen
backbone (structure 512 px / grid 16, coarse shape 512 px / grid 32 + NAF to
512, fine shape 1024 px / grid 64 + NAF to 512, texture 1024 px / grid 64 +
NAF to 1024). Each extractor resizes the image, runs the ViT, optionally
upsamples the patch map with NAF, projects every point of a dense grid into
the image and samples the feature maps there; the shape and texture stages
then gather the rows of the occupied voxels.

:class:`SharedConditioning` produces exactly the same tensors with less work:

1. the resized image tensor and the backbone output are computed once per
   image size (512 and 1024) and reused by the stages that share that size;
2. NAF's guide-image encoding, which depends only on the guide image, is
   computed once for the two 1024 px stages (fine shape and texture) and
   reused;
3. for the sparse stages, the projected points are still computed for the
   whole grid (identical tensors), but the bilinear feature sampling is only
   evaluated at the occupied voxels instead of the full ``R^3`` grid
   (``grid_sample`` computes each output location independently).

The four extractors' weights are compared once at load time; sharing is
refused if two extractors that would share a result differ.
"""
from __future__ import annotations

import collections
import contextlib
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

STAGE_ATTRS = ("image_cond_model_ss", "image_cond_model_shape_512", "image_cond_model_shape_1024",
               "image_cond_model_tex_1024")


def extractors_of(pipe):
    return [(attr, getattr(pipe, attr)) for attr in STAGE_ATTRS if getattr(pipe, attr, None) is not None]


def _state_dicts_equal(m1, m2) -> bool:
    s1, s2 = m1.state_dict(), m2.state_dict()
    if list(s1.keys()) != list(s2.keys()):
        return False
    return all(s1[k].shape == s2[k].shape and s1[k].dtype == s2[k].dtype and torch.equal(s1[k], s2[k]) for k in s1)


def verify_shared_weights(pipe) -> Dict[str, Any]:
    """Raise unless every pair of extractors that would share a backbone pass (same image
    size) or a NAF guide encoding (same guide size) has identical weights."""
    extractors = extractors_of(pipe)
    by_size = collections.defaultdict(list)
    for attr, m in extractors:
        by_size[int(m.image_size)].append(m)
    bad = []
    for size, ms in by_size.items():
        for other in ms[1:]:
            if not _state_dicts_equal(ms[0].model, other.model):
                bad.append(("backbone", size))
    naf_by_guide = collections.defaultdict(list)
    for attr, m in extractors:
        if getattr(m, "use_naf_upsample", False):
            m._load_naf()
            naf_by_guide[int(m.image_size)].append(m)
    for size, ms in naf_by_guide.items():
        for other in ms[1:]:
            if not _state_dicts_equal(ms[0].naf_model, other.naf_model):
                bad.append(("naf", size))
    if bad:
        raise RuntimeError(f"shared conditioning: extractors that would share results differ: {bad}")
    return {"backbone_groups": {k: len(v) for k, v in by_size.items()},
            "naf_guide_groups": {k: len(v) for k, v in naf_by_guide.items()}}


# ---------------------------------------------------------------------------
# DinoV3ProjFeatureExtractor.forward, in pieces (identical operations)
# ---------------------------------------------------------------------------
def _proj_module(pg):
    return sys.modules[type(pg).__module__]


def flat_index(coords: torch.Tensor, R: int) -> torch.Tensor:
    """Row of voxel (x, y, z) in the dense [B, R^3, C] projection (x slowest)."""
    x, y, z = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    return (x * R + y) * R + z


def dense_image_points_norm(pg, camera_angle_x, distance, mesh_scale, B):
    """ProjGrid.forward up to the [-1, 1] normalisation of the projected points (all grid points)."""
    grid_points = pg.grid_points
    grid_points = grid_points.expand(B, -1, -1)
    grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
    transform_matrix = pg.front_view_transform_matrix
    transform_matrix = transform_matrix.expand(B, -1, -1).clone()
    transform_matrix[:, 1, 3] = -distance
    image_points, _depth, _valid = _proj_module(pg).project_points_to_image_batch(
        grid_points, transform_matrix, camera_angle_x, pg.image_resolution)
    return (image_points + 0.5) / pg.image_resolution * 2 - 1


def sample_sparse(pg, features_map, image_points_norm, idx, BHWC=True):
    """ProjGrid.forward's feature sampling restricted to grid rows ``idx``; returns [B, N, C]."""
    if BHWC:
        features_map = features_map.permute(0, 3, 1, 2)
    queries = image_points_norm[:, idx]
    x = _proj_module(pg).sample_features(features_map, queries)
    return x.permute(0, 2, 1)


_GRID_CACHE: Dict[Any, Any] = {}


def proj_grid_for(extractor, R: int):
    """The ProjGrid for grid resolution ``R`` (the extractor's own, or a cached one built
    with the same expression the pipeline uses for a resolution override)."""
    pg = extractor.proj_grid
    if int(R) == int(pg.grid_resolution):
        return pg
    device = pg.grid_points.device
    key = (type(pg), int(R), int(pg.image_resolution), str(device))
    cached = _GRID_CACHE.get(key)
    if cached is None:
        cached = pg.__class__(grid_resolution=int(R), image_resolution=pg.image_resolution).to(device)
        _GRID_CACHE[key] = cached
    return cached


def image_tensor(extractor, image):
    if isinstance(image, torch.Tensor):
        assert image.ndim == 4, "image tensor must be batched (B, C, H, W)"
        return image
    image = [i.resize((extractor.image_size, extractor.image_size), Image.LANCZOS) for i in image]
    image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
    image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
    return torch.stack(image).cuda()


def backbone_features(extractor, image_t):
    image = extractor.transform(image_t)
    with torch.no_grad():
        return extractor.extract_features(image)


def num_register_tokens(extractor) -> int:
    return getattr(extractor.model.config, "num_register_tokens", 4)


def _coords_on_contract(coords, R: int) -> bool:
    if not torch.is_tensor(coords) or coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        return False
    lo, hi = torch.stack([coords.amin(0), coords.amax(0)]).tolist()
    return lo[0] == 0 and hi[0] == 0 and min(lo[1:]) >= 0 and max(hi[1:]) < R


class SharedConditioning:
    """Per-request conditioning with shared backbone/NAF work. Use as a context manager
    (installs the NAF guide-encoding memo for the duration of the request)."""

    def __init__(self, pipe, sparse_tensor_cls=None):
        self.pipe = pipe
        self.sparse_tensor_cls = sparse_tensor_cls or sys.modules[type(pipe).__module__].SparseTensor
        extractors = extractors_of(pipe)
        sizes = collections.Counter(int(m.image_size) for _, m in extractors)
        naf_keys = collections.Counter((int(m.image_size), tuple(int(v) for v in m.naf_target_size))
                                       for _, m in extractors if getattr(m, "use_naf_upsample", False))
        guide_sizes = collections.Counter(int(m.image_size) for _, m in extractors if getattr(m, "use_naf_upsample", False))
        self.shareable_naf = {k for k, n in naf_keys.items() if n >= 2}
        self.shareable_guides = {s: n for s, n in guide_sizes.items() if n >= 2}
        self.images: Dict[Any, Any] = {}
        self.features: Dict[Any, Any] = {}
        self.naf: Dict[Any, Any] = {}
        self.naf_enc: Dict[Any, Any] = {}
        self.naf_enc_uses: Dict[Any, int] = {}
        self.guide_key = None
        self.stats: Dict[str, Any] = {"backbone_calls": 0, "backbone_calls_saved": 0, "naf_calls": 0,
                                      "naf_encoder_calls": 0, "naf_encoder_calls_saved": 0, "stages": []}
        self._installed: List[Any] = []

    # -- request scope -------------------------------------------------------------------
    def __enter__(self):
        for _attr, m in extractors_of(self.pipe):
            if not getattr(m, "use_naf_upsample", False):
                continue
            m._load_naf()
            ie = getattr(m.naf_model, "image_encoder", None)
            if ie is None or not callable(getattr(ie, "forward_encoder", None)):
                continue
            had = "forward_encoder" in ie.__dict__
            old = ie.__dict__.get("forward_encoder")
            ie.forward_encoder = self._make_forward_encoder(ie)
            self._installed.append((ie, had, old))
        return self

    def __exit__(self, *exc):
        for ie, had, old in self._installed:
            if had:
                ie.forward_encoder = old
            else:
                del ie.forward_encoder
        self._installed.clear()
        self.images.clear()
        self.features.clear()
        self.naf.clear()
        self.naf_enc.clear()
        self.naf_enc_uses.clear()
        self.guide_key = None
        return False

    def _make_forward_encoder(self, ie):
        def forward_encoder(x, output_size):
            if ie.use_encoder:
                x = self._guide_encoding(ie, x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=output_size)
            return x
        return forward_encoder

    def _guide_encoding(self, ie, x):
        key = (self.guide_key, tuple(x.shape), x.dtype) if self.guide_key is not None else None
        enc = self.naf_enc.get(key) if key is not None else None
        if enc is not None:
            self.stats["naf_encoder_calls_saved"] += 1
            self.naf_enc_uses[key] += 1
            if self.naf_enc_uses[key] >= self.shareable_guides.get(self.guide_key[1], 1):
                del self.naf_enc[key]
                del self.naf_enc_uses[key]
            return enc
        enc = torch.cat([ie.encoder(x), ie.sem_encoder(x)], dim=1)
        self.stats["naf_encoder_calls"] += 1
        if key is not None and self.guide_key[1] in self.shareable_guides:
            self.naf_enc[key] = enc
            self.naf_enc_uses[key] = 1
        return enc

    # -- shared pieces -------------------------------------------------------------------
    @staticmethod
    def _image_key(image, extractor):
        first = image[0] if isinstance(image, list) else image
        return (id(first), int(extractor.image_size))

    def _image_for(self, extractor, image):
        key = self._image_key(image, extractor)
        hit = self.images.get(key)
        if hit is None:
            img = image_tensor(extractor, image)
            self.images[key] = (image[0] if isinstance(image, list) else image, img)
            return img
        return hit[1]

    def _features_for(self, extractor, image):
        key = self._image_key(image, extractor)
        z = self.features.get(key)
        if z is not None:
            self.stats["backbone_calls_saved"] += 1
            return z
        z = backbone_features(extractor, self._image_for(extractor, image))
        self.stats["backbone_calls"] += 1
        self.features[key] = z
        return z

    def _forward(self, extractor, image, camera_angle_x, distance, mesh_scale, coords=None, grid_resolution=None):
        """DinoV3ProjFeatureExtractor.forward with shared pieces; ``coords`` -> sparse rows [N, C]."""
        key = self._image_key(image, extractor)
        S = int(extractor.image_size)
        self._image_for(extractor, image)
        z = self._features_for(extractor, image)
        B = z.shape[0]
        z_clstoken = z[:, 0:1]
        num_reg = num_register_tokens(extractor)
        z_regtokens = z[:, 1:1 + num_reg]
        z_patchtokens = z[:, 1 + num_reg:]
        z_spatial = z_patchtokens.reshape(B, extractor.patch_number, extractor.patch_number, -1)
        R = int(extractor.grid_resolution if grid_resolution is None else grid_resolution)
        pg = proj_grid_for(extractor, R)
        if coords is None:
            z_proj_lr = pg(z_spatial, camera_angle_x, distance, mesh_scale, None)
            pts = idx = None
        else:
            pts = dense_image_points_norm(pg, camera_angle_x, distance, mesh_scale, B)
            idx = flat_index(coords, R)
            z_proj_lr = sample_sparse(pg, z_spatial, pts, idx, BHWC=True)
        if extractor.use_naf_upsample:
            target = tuple(int(v) for v in extractor.naf_target_size)
            naf_key = (key, target)
            hr_features = self.naf.get(naf_key)
            if hr_features is None:
                lr_features_bchw = z_spatial.permute(0, 3, 1, 2)
                image_for_naf = self._image_for(extractor, image).clone()
                self.guide_key = key
                try:
                    extractor._load_naf()
                    hr_features = extractor.naf_model(image_for_naf, lr_features_bchw, extractor.naf_target_size)
                finally:
                    self.guide_key = None
                if (S, target) in self.shareable_naf:
                    self.naf[naf_key] = hr_features
                self.stats["naf_calls"] += 1
            if coords is None:
                z_proj_hr = pg(hr_features, camera_angle_x, distance, mesh_scale, None, BHWC=False)
            else:
                z_proj_hr = sample_sparse(pg, hr_features, pts, idx, BHWC=False)
            z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            del hr_features
        else:
            z_proj = z_proj_lr
        z_global = torch.cat([z_clstoken, z_regtokens], dim=1)
        if coords is not None:
            z_proj = z_proj[0]
        return z_global, z_proj

    # -- the two pipeline entry points ---------------------------------------------------
    @torch.no_grad()
    def structure(self, image, camera_angle_x, distance, mesh_scale=1.0) -> Dict[str, Any]:
        started = time.perf_counter()
        device = self.pipe.device
        cam = torch.tensor([camera_angle_x], device=device)
        dist = torch.tensor([distance], device=device)
        scale = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = self._forward(self.pipe.image_cond_model_ss, image, cam, dist, scale)
        out = {"cond": {"global": z_global, "proj": z_proj},
               "neg_cond": {"global": torch.zeros_like(z_global), "proj": torch.zeros_like(z_proj)}}
        torch.cuda.synchronize(device)
        self.stats["stages"].append({"stage": "structure", "wall_s": round(time.perf_counter() - started, 6)})
        return out

    @torch.no_grad()
    def sparse(self, extractor, image, coords, camera_angle_x, distance, mesh_scale=1.0,
               grid_resolution_override=None) -> Dict[str, Any]:
        R = int(extractor.grid_resolution if grid_resolution_override is None else grid_resolution_override)
        if not _coords_on_contract(coords, R):  # batch index 0, inside the grid (one host sync)
            return self.pipe.get_proj_cond_shape(extractor, image, coords, camera_angle_x=camera_angle_x,
                                                 distance=distance, mesh_scale=mesh_scale,
                                                 grid_resolution_override=grid_resolution_override)
        started = time.perf_counter()
        device = self.pipe.device
        cam = torch.tensor([camera_angle_x], device=device)
        dist = torch.tensor([distance], device=device)
        scale = torch.tensor([mesh_scale], device=device)
        z_global, z_proj_sparse = self._forward(extractor, image, cam, dist, scale, coords=coords, grid_resolution=R)
        st = self.sparse_tensor_cls
        out = {"cond": {"global": z_global, "proj": st(feats=z_proj_sparse, coords=coords)},
               "neg_cond": {"global": torch.zeros_like(z_global),
                            "proj": st(feats=torch.zeros_like(z_proj_sparse), coords=coords)}}
        torch.cuda.synchronize(device)
        self.stats["stages"].append({"stage": f"grid{R}", "wall_s": round(time.perf_counter() - started, 6)})
        return out


class PipelineConditioning:
    """The pipeline's own conditioning methods (``shared_conditioning`` off)."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.stats: Dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def structure(self, image, camera_angle_x, distance, mesh_scale=1.0):
        return self.pipe.get_proj_cond_ss(image, camera_angle_x=camera_angle_x, distance=distance, mesh_scale=mesh_scale)

    def sparse(self, extractor, image, coords, camera_angle_x, distance, mesh_scale=1.0, grid_resolution_override=None):
        return self.pipe.get_proj_cond_shape(extractor, image, coords, camera_angle_x=camera_angle_x, distance=distance,
                                             mesh_scale=mesh_scale, grid_resolution_override=grid_resolution_override)


def conditioning_for(pipe, shared: bool):
    return SharedConditioning(pipe) if shared else PipelineConditioning(pipe)
