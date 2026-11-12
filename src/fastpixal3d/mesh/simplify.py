"""Mesh library selection and GPU simplification settings.

``fast_gpu_simplify`` on: the patched CuMesh build (``fastpixal3d_cumesh``),
whose ``simplify(sync_free=True)`` runs the whole edge-collapse round loop on
the GPU (threshold escalation and the target check are decided on the device;
the host polls a done flag every ``sync_every`` rounds), with a compact edge
sort key, fused per-round decision kernels and a faster threshold ramp
(``ratio_thresh=5e-2``, ``escalation_factor=20``). The faster ramp changes the
order in which edges are collapsed (not the acceptance criterion), so the
result is near-identical to the stock loop rather than identical.

Off: the stock ``cumesh`` package and its Python round loop.
"""
from __future__ import annotations

# Settings of the deployed configuration.
FAST_SIMPLIFY_KWARGS = dict(
    sync_free=True, max_rounds=4096, sync_every=4,
    compact_edge_bits=True, ratio_thresh=5e-2, escalation_factor=20.0,
    fuse_round_kernels=True, use_cuda_graph=False,
)


def mesh_module(fast: bool = True):
    """The module providing ``CuMesh`` and ``remeshing``."""
    if fast:
        try:
            import fastpixal3d_cumesh
        except ImportError as error:
            raise ImportError("fast_gpu_simplify needs the patched CuMesh build (module 'fastpixal3d_cumesh'); "
                              "run native/build_native.sh or switch the lever off") from error
        return fastpixal3d_cumesh
    import cumesh

    return cumesh


def simplify(mesh, target_faces: int, fast: bool = True) -> None:
    """Simplify ``mesh`` (a CuMesh) to at most ``target_faces`` faces."""
    if fast:
        mesh.simplify(int(target_faces), verbose=False, options={}, **FAST_SIMPLIFY_KWARGS)
    else:
        mesh.simplify(int(target_faces), verbose=False)
