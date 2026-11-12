"""Cache records: view names, view lists, atomic writes, the input image and the camera.

View names. One record per rendered view, named ``<object key>.view<NN>`` (``NN`` = render index,
two digits). The object key is the asset ``sha256`` for the 10k set and the object ``id`` for the
earlier 5k set (``data/lists``). The name also seeds the view's initial noise
(:mod:`fastpixal3d.training.schedule`), so it must not be changed.

Input image ("raw render, true camera"). The conditioning image of a view is its RGBA render,
resized to 512 x 512 with PIL's Lanczos filter when it has another size (PIL resizes RGBA with
premultiplied alpha), then composited on black: ``rgb * alpha``, rounded down to 8 bits. No crop
and no background removal. The camera is the one the render was made with: the horizontal field
of view and the distance ``|c2w[:3, 3]|`` from ``transforms.json``, and the mesh scale recomputed
from the mesh dump with the Pixal3D data toolkit's ``sphere_normalize_torch`` / ``transform_mesh``:
``mesh_scale = (0.49999 / max|transform_mesh(sphere_normalize(V))|) / sphere_radius``.
"""
from __future__ import annotations
import pickle

import csv
import hashlib
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

VIEW_NAME = re.compile(r"^(?P<key>[0-9A-Za-z_-]+)\.view(?P<view>\d{2})$")


# ----------------------------------------------------------------------------------- view names
def view_name(key: str, view: int) -> str:
    return f"{key}.view{int(view):02d}"


def split_view_name(name: str) -> tuple[str, int]:
    match = VIEW_NAME.match(name)
    if not match:
        raise ValueError(f"malformed view name {name!r} (expected <key>.viewNN)")
    return match.group("key"), int(match.group("view"))


def views_from_list(path: str | Path, key_column: str, views: Sequence[int]) -> List[str]:
    """View names of every row of an object list (CSV) for the given render indices, or the lines
    of a text file (one view name per line) when ``path`` does not end in ``.csv``."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and key_column not in rows[0]:
        raise KeyError(f"{path} has no column {key_column!r} (columns: {list(rows[0])})")
    return [view_name(row[key_column], v) for row in rows for v in views]


def record_names(directory: str | Path) -> List[str]:
    """Sorted names of the ``*.pt`` records in ``directory`` (files starting with ``_`` or ``.`` are
    bookkeeping and are skipped)."""
    directory = Path(directory)
    out = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith(".pt") and not entry.name.startswith(("_", ".")):
                out.append(entry.name[:-3])
    return sorted(out)


def training_pool(record_dir: str | Path, wanted: Optional[Iterable[str]] = None,
                  limit: Optional[int] = None) -> List[str]:
    """Sorted record names, restricted to ``wanted`` when given, then cut to the first ``limit``."""
    names = record_names(record_dir)
    if wanted is not None:
        wanted = set(wanted)
        names = [n for n in names if n in wanted]
    if limit:
        names = names[: int(limit)]
    return names


# ------------------------------------------------------------------------------------ file I/O
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(record: Mapping[str, Any], destination: str | Path) -> int:
    """``torch.save`` to a temporary file in the same directory, fsync, rename. Returns the size."""
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(record), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return destination.stat().st_size


def load_record(path: str | Path, mmap: bool = False) -> Dict[str, Any]:
    """Records hold tensors and plain Python values written by this project's cache builders.

    Restricted deserialization (``weights_only=True``) by default, which covers those records; a file
    that holds other Python objects is unpickled in full only when ``FASTPIXAL3D_TRUST_CHECKPOINT=1`` is set
    (full unpickling can execute code: trusted files only)."""
    import torch

    def _load(weights_only: bool):
        kwargs = {"map_location": "cpu", "weights_only": weights_only}
        if mmap:
            try:
                return torch.load(str(path), mmap=True, **kwargs)
            except RuntimeError:
                pass
        return torch.load(str(path), **kwargs)

    try:
        return _load(True)
    except pickle.UnpicklingError as error:
        if os.environ.get("FASTPIXAL3D_TRUST_CHECKPOINT") != "1":
            raise RuntimeError(f"{path} holds objects other than tensors and plain containers; set "
                               f"FASTPIXAL3D_TRUST_CHECKPOINT=1 to load it with full unpickling if you trust the file") from error
        return _load(False)


# ------------------------------------------------------------------------------- input image
def resize_rgba(png_bytes: bytes, size: int = 512):
    """RGBA image of ``size`` x ``size`` (Lanczos when the render has another size; a copy otherwise)."""
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as opened:
        rgba = opened.convert("RGBA")
        source_size = rgba.size
        if rgba.size != (size, size):
            rgba = rgba.resize((size, size), Image.Resampling.LANCZOS)
        else:
            rgba = rgba.copy()
    return rgba, source_size


def composite_on_black(rgba):
    """``rgb * alpha`` on black, float32 arithmetic, truncated to uint8 (512 x 512 RGB)."""
    from PIL import Image

    if rgba.size != (512, 512):
        raise ValueError(f"conditioning image is not 512x512: {rgba.size}")
    array = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = array[:, :, :3] * array[:, :, 3:4]
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def conditioning_image(png_bytes: bytes, expected_sha256: Optional[str] = None):
    if expected_sha256 and sha256_bytes(png_bytes) != expected_sha256:
        raise ValueError("render image sha256 differs from the one recorded in the cache record")
    rgba, _ = resize_rgba(png_bytes, 512)
    return composite_on_black(rgba)


def render_path(root: str | Path, subset_dir: str, key: str, view: int) -> Path:
    """``<root>/<subset dir>/renders_cond/<key>/<view:03d>.png`` (``subset_dir`` may be empty)."""
    base = Path(root) / subset_dir if subset_dir else Path(root)
    return base / "renders_cond" / key / f"{int(view):03d}.png"


def find_render(render: Mapping[str, Any], image_roots: Sequence[str | Path]) -> Path:
    """Locate the render of a record from its ``render`` block (``subset``, ``sha256``, ``view``)
    under the configured dataset roots, falling back to the absolute path stored at build time."""
    subset, key, view = render.get("subset") or "", render.get("sha256"), render.get("view")
    candidates = []
    if key is not None and view is not None:
        for root in image_roots:
            candidates += [render_path(root, subset, key, view), render_path(root, "", key, view)]
    if render.get("image_path"):
        candidates.append(Path(render["image_path"]))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"render of {key}.view{view} not found under {list(map(str, image_roots))}")


# ---------------------------------------------------------------------------------- camera
def concatenate_mesh_dump(dump: Mapping[str, Any]) -> np.ndarray:
    """Vertices of all triangulated objects of a Pixal3D mesh dump (``{"objects": [...]}``)."""
    vertices = []
    for item in dump.get("objects", []):
        v = np.asarray(item.get("vertices"), dtype=np.float32).reshape(-1, 3)
        f = np.asarray(item.get("faces"), dtype=np.int64).reshape(-1, 3)
        if len(v) and len(f):
            vertices.append(v)
    if not vertices:
        raise ValueError("mesh dump has no triangulated geometry")
    return np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32)


def true_camera(frame: Mapping[str, Any], vertices: np.ndarray, toolkit_utils) -> Dict[str, float]:
    """``{camera_angle_x, distance, mesh_scale}`` of one render from its ``transforms.json`` frame and
    the mesh dump; ``toolkit_utils`` is Pixal3D's ``data_toolkit/utils.py`` module."""
    import torch

    camera_angle_x = float(frame["camera_angle_x"])
    c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    distance = float(torch.linalg.vector_norm(c2w[:3, 3]))
    sphere_vertices, _center, sphere_radius = toolkit_utils.sphere_normalize_torch(torch.from_numpy(vertices))
    aligned = toolkit_utils.transform_mesh(sphere_vertices, frame)
    if float(aligned.abs().max().item()) <= 0.0:
        raise ValueError("degenerate view transform")
    box_scale = torch.tensor(0.49999, dtype=aligned.dtype) / aligned.abs().max()
    return {"camera_angle_x": camera_angle_x, "distance": distance,
            "mesh_scale": float((box_scale / sphere_radius).item())}
