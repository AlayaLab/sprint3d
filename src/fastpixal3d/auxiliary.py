"""The auxiliary models of Pixal3D's inference code, read from the weights directory.

Pixal3D's ``inference.py`` loads four auxiliary models through their upstream loaders, by hub id and at
the latest revision: the DINOv3 ViT-L/16 image encoder (``transformers``), MoGe-2 (``moge``), RMBG-2.0
(``transformers``; built together with the pipeline although only inputs without an alpha channel use it)
and the NAF feature upsampler (``torch.hub``). :func:`use_pinned_auxiliary_models` hands the same loaders
the files that ``scripts/download_weights.py`` put in the weights directory (``<root>/<local_dir>`` of
``weights/sources.json``, where the root is ``$FASTPIXAL3D_WEIGHTS`` or the repository's ``weights/``):
pinned revisions, checked by sha256, and no second copy in ``HF_HOME`` or ``TORCH_HOME``.

A model whose files are not all in the weights directory is loaded as Pixal3D's code loads it: by hub id,
through the Hugging Face cache (``HF_HOME``) or ``torch.hub`` (``TORCH_HOME``). NAF's model code is not a
weight file; it is loaded with ``torch.hub`` at the pinned commit (fetched once into ``TORCH_HOME``), or,
offline when that commit is not cached, from the copy Pixal3D's own loader uses (``valeoai/NAF``, branch
``main``). RMBG-2.0 is gated: when it can be loaded from neither place, the pipeline loads without it and
only an input without an alpha channel fails, with a message saying how to get it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

SOURCES = Path(__file__).resolve().parents[2] / "weights" / "sources.json"


def load_sources(path: Path = SOURCES) -> Dict[str, dict]:
    """``weights/sources.json`` by source name."""
    return {source["name"]: source for source in json.loads(Path(path).read_text())["sources"]}


def local_copy(root: Path, source: dict) -> Optional[Path]:
    """``<root>/<local_dir>`` if every file of the source is there with its pinned size, else None."""
    base = Path(root) / source["local_dir"]
    for entry in source["files"]:
        path = base / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return None
    return base


class MissingBackgroundRemoval:
    """Stands in for RMBG-2.0 when it is not available; fails only when an input needs it."""

    def __init__(self, reason: str):
        self.reason = reason

    def to(self, device):
        return self

    def cuda(self):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        raise RuntimeError(
            "this input image has no alpha channel, so it needs RMBG-2.0 for background removal, which is not "
            f"available ({self.reason}). Accept its terms at https://huggingface.co/briaai/RMBG-2.0, log in "
            "(`hf auth login`) and run `python scripts/download_weights.py --only rmbg2`, or pass an RGBA image.")


def use_pinned_auxiliary_models(inference, root: Path, sources: Optional[Dict[str, dict]] = None) -> Dict[str, str]:
    """Point the loaders of Pixal3D's ``inference`` module (already imported) at the weights directory.

    Call before ``inference.init_pipeline``; load MoGe-2 with ``inference.load_moge_model(model_name=
    result["moge2_model"])``. Returns where each model is read from (for the run record)."""
    import torch
    from pixal3d.pipelines import rembg
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    sources = sources or load_sources()
    info: Dict[str, str] = {}

    # DINOv3: the extractors are built from inference.IMAGE_COND_CONFIGS (model_name = hub id)
    dinov3 = local_copy(root, sources["dinov3"])
    if dinov3 is not None:
        for config in inference.IMAGE_COND_CONFIGS.values():
            config["model_name"] = str(dinov3)
    info["dinov3"] = str(dinov3 or next(iter(inference.IMAGE_COND_CONFIGS.values()))["model_name"])

    # MoGe-2: MoGeModel.from_pretrained accepts the path of model.pt
    moge2 = local_copy(root, sources["moge2"])
    info["moge2_model"] = str(moge2 / "model.pt") if moge2 is not None else inference.MOGE_MODEL_NAME

    # RMBG-2.0: the pipeline builds rembg.BiRefNet(model_name="briaai/RMBG-2.0") from its pipeline.json
    rmbg = sources["rmbg2"]
    rmbg_dir = local_copy(root, rmbg)
    original = getattr(rembg.BiRefNet, "_original", rembg.BiRefNet)

    def background_removal(model_name: str = rmbg["repo"]):
        if model_name != rmbg["repo"]:
            return original(model_name)
        name = str(rmbg_dir) if rmbg_dir is not None else model_name
        try:
            model = original(name)
        except OSError as error:  # gated without access, offline without a cached copy, no network
            reason = f"{type(error).__name__}: {str(error).splitlines()[0].split('. ')[0][:160]}"
            info["rmbg2"] = f"not available ({reason})"
            return MissingBackgroundRemoval(reason)
        info["rmbg2"] = name
        return model

    background_removal._original = original
    rembg.BiRefNet = background_removal

    # NAF: code at the pinned commit, weights from the weights directory (else the release URL, as upstream)
    naf = sources["naf"]
    naf_dir = local_copy(root, naf)
    naf_file = naf_dir / naf["files"][0]["path"] if naf_dir is not None else None
    repo, commit = naf["code"]["repo"], naf["code"]["commit"]
    hub_ref = f"{repo}:{commit}"
    if os.environ.get("HF_HUB_OFFLINE") == "1" and not (Path(torch.hub.get_dir()) / (
            repo.replace("/", "_") + "_" + commit)).is_dir():
        hub_ref = repo  # offline: the cached copy that Pixal3D's own loader uses

    def _load_naf(self):
        if self.naf_model is None:
            device = next(self.model.parameters()).device
            model = torch.hub.load(hub_ref, "naf", pretrained=False, device=device, trust_repo=True,
                                   skip_validation=True)
            if naf_file is not None:
                state = torch.load(naf_file, map_location=device, weights_only=True)
            else:
                state = torch.hub.load_state_dict_from_url(naf["url"], map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            model.requires_grad_(False)
            self.naf_model = model

    DinoV3ProjFeatureExtractor._load_naf = _load_naf
    info["naf"] = f"code {hub_ref}, weights {naf_file or naf['url']}"
    return info
