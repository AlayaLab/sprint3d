"""Where the model weights come from.

``scripts/download_weights.py`` puts every file of ``weights/sources.json`` under one weights directory
(``$FASTPIXAL3D_WEIGHTS``, by default the repository's ``weights/``), and the package reads them from
there; this is the only copy it needs:

* ``students/``: the four student adapters (``AlayaLab/sprint3d``);
* ``pixal3d/``: Pixal3D's flows and decoders (``TencentARC/Pixal3D``);
* ``dinov3/``: the DINOv3 ViT-L/16 image encoder (``camenduru/dinov3-vitl16-pretrain-lvd1689m``);
* ``moge2/``: MoGe-2 ViT-L, the camera field of view (``Ruicheng/moge-2-vitl``);
* ``rmbg2/``: RMBG-2.0, background removal for inputs without alpha (``briaai/RMBG-2.0``, gated);
* ``naf/``: the NAF feature upsampler (release asset of ``valeoai/NAF``).

Explicit settings take precedence: ``FASTPIXAL3D_STUDENTS`` / ``--students`` (student directory),
``FASTPIXAL3D_PIXAL3D`` / ``--pixal3d`` (local model directory or Hugging Face repo id, with
``FASTPIXAL3D_PIXAL3D_REVISION``), ``FASTPIXAL3D_PIXAL3D_SRC`` (Pixal3D source checkout). Without a local
copy, Pixal3D comes from the Hugging Face cache at its pinned revision, and an auxiliary model is loaded as
Pixal3D's own code loads it, by hub id through ``HF_HOME`` / ``TORCH_HOME``. See :mod:`fastpixal3d.auxiliary`
for how the auxiliary models are passed to Pixal3D's loaders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

PIXAL3D_REPO = "TencentARC/Pixal3D"
PIXAL3D_REVISION = "0b31f9160aa400719af409098bff7936a932f726"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class WeightPaths:
    pixal3d_src: Optional[str] = None        # Pixal3D source checkout (third_party/Pixal3D by default)
    pixal3d: Optional[str] = None            # local model directory or Hugging Face repo id
    pixal3d_revision: Optional[str] = None   # revision when ``pixal3d`` is a repo id
    students: Optional[str] = None           # directory with students.json
    student_checkpoints: Optional[Dict[str, str]] = None  # development: training checkpoints per student
    hf_home: Optional[str] = None
    torch_home: Optional[str] = None
    offline: bool = False
    root: Optional[str] = None               # weights directory of scripts/download_weights.py

    @classmethod
    def from_env(cls, **overrides) -> "WeightPaths":
        env = os.environ
        values = dict(
            pixal3d_src=env.get("FASTPIXAL3D_PIXAL3D_SRC"),
            pixal3d=env.get("FASTPIXAL3D_PIXAL3D"),
            pixal3d_revision=env.get("FASTPIXAL3D_PIXAL3D_REVISION"),
            students=env.get("FASTPIXAL3D_STUDENTS"),
            hf_home=env.get("HF_HOME"),
            torch_home=env.get("TORCH_HOME"),
            offline=env.get("HF_HUB_OFFLINE", "0") == "1",
            root=env.get("FASTPIXAL3D_WEIGHTS"),
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def source_dir(self) -> Path:
        path = Path(self.pixal3d_src) if self.pixal3d_src else _repo_root() / "third_party" / "Pixal3D"
        if not (path / "pixal3d").is_dir() or not (path / "inference.py").is_file():
            raise FileNotFoundError(f"Pixal3D source checkout not found at {path} (set FASTPIXAL3D_PIXAL3D_SRC or "
                                    "initialise the third_party/Pixal3D submodule)")
        return path

    def apply_environment(self) -> None:
        """Export cache locations / offline mode for the upstream loaders (call before loading)."""
        if self.hf_home:
            os.environ["HF_HOME"] = str(self.hf_home)
        if self.torch_home:
            os.environ["TORCH_HOME"] = str(self.torch_home)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def root_dir(self) -> Path:
        """The weights directory (``$FASTPIXAL3D_WEIGHTS``, else the repository's ``weights/``)."""
        return Path(self.root) if self.root else _repo_root() / "weights"

    def downloaded(self, name: str) -> Optional[Path]:
        """``<weights directory>/<name>`` if it holds every file of that source of ``weights/sources.json``."""
        from .auxiliary import load_sources, local_copy

        return local_copy(self.root_dir(), load_sources()[name])

    def model_dir(self) -> str:
        """Local directory of the Pixal3D weights: ``pixal3d`` if given (a directory, or a repo id downloaded
        at the pinned revision), else ``<weights directory>/pixal3d``, else the pinned Hugging Face snapshot."""
        spec = self.pixal3d
        if spec is None:
            local = self.downloaded("pixal3d")
            if local is not None:
                return str(local.resolve())
            spec = PIXAL3D_REPO
        if Path(spec).is_dir():
            return str(Path(spec).resolve())
        from huggingface_hub import snapshot_download

        return snapshot_download(spec, revision=self.pixal3d_revision or PIXAL3D_REVISION)

    def students_dir(self) -> Optional[str]:
        """``students`` if given, else ``<weights directory>/students`` when no training checkpoints are given."""
        if self.students or self.student_checkpoints:
            return self.students
        local = self.downloaded("students")
        return str(local) if local is not None else None
