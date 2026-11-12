"""Triangle BVH used by narrow-band remeshing and by the texture bake.

``gpu_bvh_build`` on: the patched cubvh build (``fastpixal3d_cubvh``) builds a
binary linear BVH (Karras 2012) on the GPU; the triangles never leave the
device. Off: the BVH bundled with the stock ``cumesh`` package (host
top-down build). Both expose ``unsigned_distance(points, return_uvw)`` with
the same semantics; closest-face ids can differ only on exact distance ties.
"""
from __future__ import annotations


def bvh_class(gpu_build: bool = True):
    if gpu_build:
        try:
            import fastpixal3d_cubvh
        except ImportError as error:
            raise ImportError("gpu_bvh_build needs the patched cubvh build (module 'fastpixal3d_cubvh'); "
                              "run native/build_native.sh or switch the lever off") from error
        return fastpixal3d_cubvh.cuBVH
    import cumesh

    return cumesh.cuBVH


def bvh_impl_name(gpu_build: bool = True) -> str:
    return "gpu" if gpu_build else "host"
