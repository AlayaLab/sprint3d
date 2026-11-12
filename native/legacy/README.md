# Legacy native builds (tiled mesh path of the cumulative measurement)

The cumulative measurement starts (row 01) from the tiled mesh path (`fastpixal3d.mesh.tiled_path`): a
host-built BVH and a UV atlas of 16 tiles unwrapped by xatlas on a CPU thread pool. Until the GPU BVH
build (row 05) and the single-pass GPU chart segmentation (row 08) replace them, the rows of that
measurement use builds that the deployed configuration never loads. They are produced from pinned
upstream commits by the patch series in this directory:

| module | upstream | patches | what changes |
|---|---|---|---|
| `fastpixal3d_bvh_openmp` | cubvh `ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70` | `cubvh/patches/0001` | The host BVH build runs in parallel (OpenMP tasks over independent subtrees, then a serial pass that numbers the nodes in the order of the original builder); the triangle gather is parallel. The node array is identical to the serial build. |
| `fastpixal3d_bvh_openmp_nogil` | same | `0001`, `0002` | As above, and the BVH build and the distance queries release the GIL, so the BVH can be built on the mesh worker thread while the main thread samples the texture. |
| `fastpixal3d_bvh_openmp_pinned` | same | `0001`, `0002`, `0003` | As above, and the host-to-device copies of the BVH (triangles and nodes) go through one pinned host buffer in chunks of at most 256 MiB on the current PyTorch stream; the Python wrapper passes float32 vertices and uint32 triangles so no conversion copy is made. Device contents are identical. |
| `fastpixal3d_xatlas_single` | CuMesh `12289e1062f0603f2f0d0771b02e1395d247f26f`, `third_party/xatlas` | `xatlas/patches/0001` | CuMesh's bundled xatlas built as a separate module without its internal thread pool (`XA_MULTITHREADED=0`), C++ namespace renamed. Same charts and packing, computed on the calling thread. |

The three BVH builds are renamed to their module names with `../alias_rename.py` after the patches are
applied (so they can be installed next to each other and next to the builds of `../build_native.sh`).
Build all four into the active environment, or into a directory added to `PYTHONPATH`:

```bash
PYTHON=$(which python) MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0 native/legacy/build_legacy.sh
native/legacy/build_legacy.sh --target /path/to/site      # then add /path/to/site to PYTHONPATH
native/legacy/build_legacy.sh --only pinned                # one build: openmp | nogil | pinned | xatlas
```

## Which rows need which build

Rows of `experiments/configs/cumulative/` (file prefix). A configuration that asks for a build that is not
installed stops with an `ImportError` naming this directory; there is no silent fallback.

| rows | configuration (the lever added) | host BVH build | single-thread xatlas |
|---|---|---|---|
| 01 | `tiled_mesh_path` (every lever off) | `fastpixal3d_bvh_openmp` | -- |
| 02-04 | `mesh_prep_worker_thread`, `fixed_bake_launch_config`, `fast_gpu_simplify` | `fastpixal3d_bvh_openmp_nogil` | -- |
| 05-07 | `gpu_bvh_build`, `adaptive_remesh`, `lean_glb_writer` | -- (GPU build of `../build_native.sh`) | -- |
| 00, 08-18, x1-x3 | `pixal3d_rerun`, `gpu_chart_segmentation` and later, the three controls | -- | -- |

Rows 04 and later also need `fastpixal3d_cumesh` and rows 05 and later `fastpixal3d_cubvh` (both from
`../build_native.sh`), as the deployed configuration does. In general, for a configuration with
`gpu_chart_segmentation` off: `gpu_bvh_build` off selects `..._pinned` if `pinned_bvh_transfers` is on,
else `..._nogil` if `mesh_prep_worker_thread` is on, else `..._openmp`; `per_tile_single_thread_unwrap`
on needs `fastpixal3d_xatlas_single`. The cumulative set therefore needs only `..._openmp` and
`..._nogil`; `..._pinned` and `fastpixal3d_xatlas_single` serve configurations that switch
`gpu_chart_segmentation` off while keeping the two levers of the tiled unwrap on (for example
`scripts/infer.py --disable gpu_chart_segmentation`).

## Licences

The upstream licences are in `licenses/`: cubvh (MIT, Copyright (c) 2022 kiui) and the NVIDIA Source Code
License of the instant-ngp code it contains (`cubvh_LICENSE_NVIDIA`, non-commercial use), CuMesh (MIT) and
xatlas (MIT, Copyright (c) 2018-2020 Jonathan Young). The patches are distributed under the licence of the
file they modify.

Licence: the cubvh patches build on upstream cubvh, whose `LICENSE_NVIDIA` (non-commercial) applies to these builds as it does to the main BVH build; see THIRD_PARTY_LICENSES.md.
