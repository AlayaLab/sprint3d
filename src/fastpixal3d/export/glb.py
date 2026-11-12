"""GLB export.

The export rotates the mesh into the product frame (the same rotation as
Pixal3D's ``inference.py``) and writes one binary glTF file.

``lean_glb_writer`` on: :func:`write_glb` builds the glTF container for the
one shape the bake produces (a single mesh with positions, normals, one UV
set and one PBR material with a base-colour and a metallic-roughness
texture) without trimesh's generic export machinery (scene traversal and
content-hash de-duplication of every buffer); the accessor, buffer-view and
material layout is the one trimesh writes. The two textures are PNG-encoded
concurrently with ``fpng`` (falling back to Pillow, compression level 1, if
``fpng`` is unavailable). Encodes started earlier (see
:class:`PngPrewarm`) are reused.

Off: ``trimesh.Trimesh.export`` with PNG compression level 1.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
import os
import struct
import threading
from typing import Any, Dict, Optional

import numpy as np

# Rotation from the generation frame to the product frame (as in Pixal3D's inference.py).
PRODUCT_EXPORT_ROTATION = np.asarray([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
PNG_COMPRESS_LEVEL = 1

_FPNG = "unprobed"


def fpng_module():
    """The ``fpng`` module (py-fpng-nb) or None if it cannot be imported."""
    global _FPNG
    if _FPNG == "unprobed":
        try:
            import fpng

            fpng.from_pil  # noqa: B018
            _FPNG = fpng
        except Exception:
            _FPNG = None
    return _FPNG


def encode_png(img, compress_level: int = PNG_COMPRESS_LEVEL, use_fpng: bool = True):
    """(png bytes, encoder name)."""
    if use_fpng:
        module = fpng_module()
        if module is not None:
            try:
                return module.from_pil(img), "fpng"
            except Exception:
                pass
    with io.BytesIO() as f:
        img.save(f, format="PNG", compress_level=compress_level)
        return f.getvalue(), "pillow"


class PngPrewarm:
    """Starts PNG encoding of texture images in the background as soon as they exist;
    :meth:`take` returns the finished bytes (or None if the image was not submitted)."""

    def __init__(self, submit, use_fpng: bool = True):
        self._submit = submit
        self._use_fpng = use_fpng
        self._pending: Dict[int, concurrent.futures.Future] = {}
        self._lock = threading.Lock()

    def __call__(self, img) -> None:
        if fpng_module() is None or not self._use_fpng:
            return
        key = id(img)
        with self._lock:
            if key in self._pending:
                return
            self._pending[key] = self._submit(fpng_module().from_pil, img)

    def take(self, img) -> Optional[bytes]:
        with self._lock:
            future = self._pending.pop(id(img), None)
        if future is None:
            return None
        try:
            return future.result()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# lean writer
# ---------------------------------------------------------------------------
def _gltf():
    import trimesh.exchange.gltf as gltf

    return gltf


def _append_accessor(gltf, accessors, buffer_items, blob, data):
    as_bytes = data.tobytes()
    buffer_items.append(gltf._byte_pad(as_bytes))
    blob["bufferView"] = len(buffer_items) - 1
    dtype = gltf._dtypes[blob["componentType"]]
    kind = blob["type"]
    if kind == "SCALAR":
        integer = dtype.startswith("<u") or dtype.startswith("<i")
        blob["count"] = int(data.size)
        blob["max"] = [int(data.max())] if integer else [float(data.max())]
        blob["min"] = [int(data.min())] if integer else [float(data.min())]
    else:
        shape = {"VEC2": 2, "VEC3": 3, "VEC4": 4}[kind]
        resh = data.reshape((-1, shape))
        blob["count"] = len(resh)
        blob["max"] = resh.max(axis=0).tolist()
        blob["min"] = resh.min(axis=0).tolist()
    accessors.append(blob)
    return len(accessors) - 1


def _append_material(gltf, mat, tree, buffer_items, prewarm: Optional[PngPrewarm], use_fpng: bool,
                     encoder_stats: Optional[dict]):
    from trimesh.visual import color as visual_color

    result: Dict[str, Any] = {"pbrMetallicRoughness": {}}
    try:
        result["baseColorFactor"] = visual_color.to_float(mat.baseColorFactor).reshape(4).tolist()
    except Exception:
        pass
    try:
        result["emissiveFactor"] = mat.emissiveFactor.reshape(3).tolist()
    except Exception:
        pass
    if isinstance(mat.name, str):
        result["name"] = mat.name
    if isinstance(mat.alphaMode, str):
        result["alphaMode"] = mat.alphaMode
    if isinstance(mat.alphaCutoff, float):
        result["alphaCutoff"] = mat.alphaCutoff
    if isinstance(mat.doubleSided, bool):
        result["doubleSided"] = mat.doubleSided
    if isinstance(mat.metallicFactor, float):
        result["metallicFactor"] = mat.metallicFactor
    if isinstance(mat.roughnessFactor, float):
        result["roughnessFactor"] = mat.roughnessFactor

    image_mapping = {
        "baseColorTexture": mat.baseColorTexture,
        "emissiveTexture": mat.emissiveTexture,
        "normalTexture": mat.normalTexture,
        "occlusionTexture": mat.occlusionTexture,
        "metallicRoughnessTexture": mat.metallicRoughnessTexture,
    }
    candidates = [(key, img) for key, img in image_mapping.items() if img is not None]

    def encode(img):
        if prewarm is not None:
            data = prewarm.take(img)
            if data is not None:
                return data, "fpng"
        return encode_png(img, PNG_COMPRESS_LEVEL, use_fpng)

    encoded: Dict[str, tuple] = {}
    if len(candidates) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = {key: executor.submit(encode, img) for key, img in candidates}
            for key, future in futures.items():
                encoded[key] = future.result()
    else:
        for key, img in candidates:
            encoded[key] = encode(img)

    for key, _img in candidates:
        image_bytes, encoder = encoded[key]
        buffer_items.append(gltf._byte_pad(image_bytes))
        tree["images"].append({"bufferView": len(buffer_items) - 1, "mimeType": "image/png"})
        result[key] = {"index": len(tree["textures"])}
        tree["textures"].append({"source": len(tree["images"]) - 1})
        if encoder_stats is not None:
            encoder_stats[key] = {"encoder": encoder, "bytes": len(image_bytes)}

    for key in ["baseColorTexture", "baseColorFactor", "roughnessFactor", "metallicFactor", "metallicRoughnessTexture"]:
        if key in result:
            result["pbrMetallicRoughness"][key] = result.pop(key)
    if len(result["pbrMetallicRoughness"]) == 0:
        result.pop("pbrMetallicRoughness")
    tree["materials"].append(result)
    return len(tree["materials"]) - 1


def glb_bytes(mesh, prewarm: Optional[PngPrewarm] = None, use_fpng: bool = True,
              encoder_stats: Optional[dict] = None) -> bytes:
    """GLB bytes of a single textured ``trimesh.Trimesh`` (vertex normals in its cache)."""
    from trimesh import util as tm_util

    gltf = _gltf()
    scene = mesh.scene()
    name = next(iter(scene.geometry.keys()))
    tree: Dict[str, Any] = {"scene": 0, "scenes": [{"nodes": [0]}],
                            "asset": {"version": "2.0", "generator": "https://github.com/mikedh/trimesh"},
                            "accessors": [], "meshes": [], "images": [], "textures": [], "materials": []}
    buffer_items: list = []
    acc_face = _append_accessor(gltf, tree["accessors"], buffer_items, {"componentType": 5125, "type": "SCALAR"},
                                mesh.faces.astype(np.uint32))
    acc_vertex = _append_accessor(gltf, tree["accessors"], buffer_items,
                                  {"componentType": 5126, "type": "VEC3", "byteOffset": 0},
                                  mesh.vertices.astype(np.float32))
    current: Dict[str, Any] = {"name": name, "extras": {},
                               "primitives": [{"attributes": {"POSITION": acc_vertex}, "indices": acc_face,
                                               "mode": gltf._GL_TRIANGLES}]}
    try:
        current["extras"] = gltf._jsonify(mesh.metadata)
        extensions = current["extras"].pop("gltf_extensions", None)
        if isinstance(extensions, dict):
            current["extensions"] = extensions
        if mesh.units not in [None, "m", "meters", "meter"]:
            current["extras"]["units"] = str(mesh.units)
    except Exception:
        pass
    if hasattr(mesh.visual, "material"):
        material = _append_material(gltf, mesh.visual.material, tree, buffer_items, prewarm, use_fpng, encoder_stats)
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices):
            uv = mesh.visual.uv.copy()[:, :2]
            uv[:, 1] = 1.0 - uv[:, 1]
            acc_uv = _append_accessor(gltf, tree["accessors"], buffer_items,
                                      {"componentType": 5126, "type": "VEC2", "byteOffset": 0}, uv.astype(np.float32))
            current["primitives"][0]["attributes"]["TEXCOORD_0"] = acc_uv
        current["primitives"][0]["material"] = material
    if "vertex_normals" in mesh._cache.cache:
        normals = tm_util.unitize(mesh.vertex_normals)
        acc_norm = _append_accessor(gltf, tree["accessors"], buffer_items,
                                    {"componentType": 5126, "type": "VEC3", "byteOffset": 0},
                                    normals.astype(np.float32))
        current["primitives"][0]["attributes"]["NORMAL"] = acc_norm
    tree["meshes"].append(current)
    tree.update(scene.graph.to_gltf(scene=scene, mesh_index={name: 0}))
    for key in ["textures", "materials", "images", "accessors", "meshes"]:
        if len(tree[key]) == 0:
            tree.pop(key)
    views = gltf._build_views({i: b for i, b in enumerate(buffer_items)})
    buffer_data = b"".join(buffer_items)
    if len(buffer_data) > 0:
        tree["buffers"] = [{"byteLength": len(buffer_data)}]
        tree["bufferViews"] = views
    content = tm_util.jsonify(tree, separators=(",", ":"))
    content += (4 - ((len(content) + 20) % 4)) * " "
    content = content.encode("utf-8")
    header = struct.pack("<5I", gltf._magic["gltf"], 2, len(content) + len(buffer_data) + 28, len(content),
                         gltf._magic["json"])
    bin_header = struct.pack("<2I", len(buffer_data), 0x004E4942)
    return b"".join([header, content, bin_header, buffer_data])


def is_writer_shape(mesh) -> bool:
    """True for the one mesh shape :func:`glb_bytes` supports."""
    try:
        material = mesh.visual.material
        fields = ("baseColorTexture", "emissiveTexture", "normalTexture", "occlusionTexture", "metallicRoughnessTexture")
        present = {f for f in fields if getattr(material, f, None) is not None}
        has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices)
        return present == {"baseColorTexture", "metallicRoughnessTexture"} and has_uv
    except Exception:
        return False


@contextlib.contextmanager
def png_compress_level(level: int):
    """Default PNG compression level for ``PIL.Image.save`` inside the block (trimesh export path)."""
    from PIL import Image

    original_save = Image.Image.save

    def patched_save(self, fp, format=None, **params):
        if format is not None and str(format).upper() == "PNG" and "compress_level" not in params:
            params["compress_level"] = level
        return original_save(self, fp, format=format, **params)

    Image.Image.save = patched_save
    try:
        yield
    finally:
        Image.Image.save = original_save


def export_glb(mesh, path, lean: bool = True, prewarm: Optional[PngPrewarm] = None, use_fpng: bool = True,
               stats: Optional[dict] = None, rotation=PRODUCT_EXPORT_ROTATION) -> None:
    """Rotate ``mesh`` into the product frame (in place) and write ``path`` atomically.

    ``rotation=None`` keeps the mesh in the frame it has (used by the TRELLIS.2 port, whose GLB stays in
    TRELLIS.2's own export frame)."""
    if rotation is not None:
        mesh.apply_transform(rotation)
    path = os.fspath(path)
    partial = path[:-4] + ".partial.glb" if path.endswith(".glb") else path + ".partial"
    if lean and is_writer_shape(mesh):
        data = glb_bytes(mesh, prewarm=prewarm, use_fpng=use_fpng, encoder_stats=stats)
        with open(partial, "wb") as f:
            f.write(data)
    else:
        with png_compress_level(PNG_COMPRESS_LEVEL):
            mesh.export(partial, file_type="glb")
    os.replace(partial, path)
    if os.stat(path).st_size <= 0:
        raise RuntimeError(f"empty GLB written to {path}")
