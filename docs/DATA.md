# Evaluation data

No evaluation data is distributed with this repository: no renders, meshes, ground-truth files or generated GLBs.
The repository distributes object lists with identifiers, checksums and licences, and the evaluation code.

## Object lists

| File | Rows | Columns | Use |
|---|---|---|---|
| `eval/lists/validation_400.csv` | 400 objects | `id`, `sha256`, `source`, `licence`, `view`, `in_earlier_5k_training_set`, `gt_image_sha256`, `gt_mesh_dump_sha256` | validation set of Table 10 (a) (one input view per object) |
| `eval/lists/toys4k_dev_500.csv` | 500 objects | `id`, `sha256`, `source`, `licence`, `view`, `object_id`, `category`, `mesh_sha256` | Toys4K development part of Table 10 (b) (one fixed view per object) |
| `eval/lists/timing_80.csv` | 80 inputs (40 objects x 2 views) | `id`, `view` | inputs of the dedicated timing run, in run order |

- `sha256` is the object key of TRELLIS-500K (the sha256 of the raw asset file); `id` is its first 12 hex digits and is
  used in file names and tables.
- `source` is the TRELLIS-500K subset the object comes from: `sketchfab` (Objaverse-XL, Sketchfab part; 349 objects),
  `3dfuture` (24), `hssd` (16), `abo` (11).
- `licence` is the licence of the asset as given in the TRELLIS-500K metadata: `by` 312, `by-nc` 16, `by-nc-sa` 11,
  `by-sa` 10 (Sketchfab Creative Commons codes), `CC-BY-4.0` 11 (ABO), `CC-BY-NC-4.0` 16 (HSSD),
  `3D-FUTURE-research-only` 24.  Several licences are non-commercial or research-only; every asset stays under its own
  licence, and anything derived from it (renders, meshes) inherits that licence.
- `view` is the index of the conditioning render that is used as the input image of every method (one of four renders
  per object, drawn once with a fixed seed).
- `in_earlier_5k_training_set`: 7 of the 400 validation objects (5 Sketchfab, 2 HSSD, matched by their source
  identifiers) also belong to the earlier 5k-object training set, which was used to train the structure, coarse-shape
  and fine-shape students and the alternative texture student trained on that set; the training set of the deployed
  texture student contains none of them.  Table 10 (a) uses all 400 objects.
- `gt_image_sha256` and `gt_mesh_dump_sha256`: sha256 of the conditioning render `renders_cond/<sha256>/<view:03d>.png`
  of the listed view and of the mesh dump `mesh_dumps/<sha256>.pickle` used for Table 10 (a).  Compare them with your own
  ground-truth files to check whether you evaluate on the same inputs (two objects, `15b316c6c6cf` and `8cb934857c4f`,
  have identical mesh dumps).  `transforms.json` is not hashed.
- The 40 timing objects are a subset of the 400 validation objects; their two timing views are not necessarily the
  view used in Table 10 (a).
- `eval/config.example.yaml` pins the sha256 of `validation_400.csv`; `run_eval.py` stops if the file differs.
- Toys4K lists. TRELLIS-500K lists 3,229 Toys4K objects; 76 of them were set aside for checks during development, and the
  other 3,153 were split once (seed 20261001, proportional allocation per category) into the development part of 500 and
  the test part of 2,653. `sha256` is the TRELLIS-500K key, `object_id` and `category` the names of Toys4K, and
  `mesh_sha256` the sha256 of the object's `mesh.obj` in the official archive `toys4k_obj_files.zip`. `view` 4 is the same
  fixed view for every object: yaw 45 deg, pitch 25 deg, FoV 40 deg, radius 2.532. TRELLIS-500K gives no per-asset licence
  for these objects, so `licence` only says `Toys4K`: the terms of the Toys4K authors apply (their README: "Please do not distribute the
  data made available here in any way"), and no asset is distributed here. The ground truth follows the layout below (`renders_cond/<sha256>/004.png`, `transforms.json`,
  `mesh_dumps/<sha256>.pickle`); the script that rendered it is not part of this release.

## Ground-truth layout expected by the evaluation

The evaluation reads one directory per source under a root (`dataset.root` and `dataset.source_dirs` of the
configuration), in the layout of the Pixal3D / TRELLIS.2 data toolkits:

```
<root>/<source dir>/renders_cond/<sha256>/<view:03d>.png       RGBA conditioning render, 1024 x 1024 (the model input)
<root>/<source dir>/renders_cond/<sha256>/transforms.json       per view: camera_angle_x (horizontal FoV, rad) and
                                                               transform_matrix (Blender camera-to-world)
<root>/<source dir>/mesh_dumps/<sha256>.pickle                 ground-truth mesh, {"objects": [{"vertices", "faces"}, ...]},
                                                               in the world frame of the renders (bbox-centred)
<root>/<source dir>/ss_support_view_512/<sha256>/view<view:02d>_scale.json
<root>/<source dir>/ss_support_view_512/<sha256>/view<view:02d>.npz
```

Default source directories: `sketchfab` -> `ObjaverseXL_sketchfab`, `abo` -> `ABO`, `hssd` -> `HSSD`,
`3dfuture` -> `3D-FUTURE`.

Table 10 (a) (condition `official`) needs only the conditioning render, `transforms.json` and the mesh dump of each listed
(object, view).  The two `ss_support_view_512` files (voxel-frame scale with the keys `sphere_radius`, `box_scale`,
`total_scale`, and the 32^3 structure support `support32`) are needed only for rows generated with the ground-truth
camera (`true_camera`, an appendix condition).

## How the ground truth was produced (to be verified)

**Status: not verified.**  The script and the run that produced the ground-truth files used for Table 10 (a) have not been
identified, and we have not checked that the recipe below reproduces them.  Do not assume that files produced by this
recipe are identical to ours; the conditioning renders in particular depend on random camera draws.  The two sha256
columns of `validation_400.csv` tell whether your files match the ones used for Table 10 (a).

Candidate upstream scripts are those of the Pixal3D data toolkit (`data_toolkit/` of the Pixal3D repository, commit
28efad6), which is derived from the TRELLIS.2 data toolkit:

| Files | Candidate script | What the script does (as written upstream) |
|---|---|---|
| asset download and metadata | `build_metadata.py`, `download.py` | the toolkit ships dataset modules for Objaverse-XL (`--source sketchfab`) and ABO; how the HSSD and 3D-FUTURE assets were obtained is not identified |
| `mesh_dumps/<sha256>.pickle` | `dump_mesh.py` | mesh dump with Blender (4.5.1 at that commit) |
| `renders_cond/<sha256>/` | `render_cond.py` | Blender 3.0.1, Cycles, 32 samples with denoising, 1024 px RGBA; views on a Hammersley sequence with a random offset; FoV drawn in [10, 70] deg with the camera radius tied to the FoV (uniform in 1 / r^2); writes `transforms.json`; upstream default 2 views per object (our data has 4) |
| `ss_support_view_512/<sha256>/view<view:02d>_scale.json` | `dual_grid_view.py` writes scale files with the same keys | view-aligned normalisation: sphere normalisation, rotation into the view, rescale to abs-max 0.49999 |
| `ss_support_view_512/<sha256>/view<view:02d>.npz` | not identified | 32^3 occupancy of the view-aligned dual grid |

If you regenerate the ground truth, the evaluation is well defined on your files, but the resulting numbers are not
guaranteed to match Table 10 (a).

# Training data

No training data is distributed either: no renders, meshes or caches. The repository distributes the object
lists of the training sets, the scripts that selected the 10k set from public metadata (`data/select/`), and
the builders of the training caches (`train/caches/`, see [TRAINING.md](TRAINING.md)).

## Object lists

| File | Rows | Used for |
|---|---:|---|
| `data/lists/curated_10k.csv` | 10,000 objects | the 10k set: training set of the texture student |
| `data/lists/matched_5243.csv` | 5,243 objects | a nested subset of the 10k set with the size of the 5k set (data comparisons) |
| `data/lists/train_5k.csv` | 5,243 objects | the earlier 5k set: training set of the structure, coarse-shape and fine-shape students |

All three files have the columns `id`, `sha256`, `source`, `licence`.

- 10k set and its subset: `sha256` is the TRELLIS-500K object key (the sha256 of the raw asset file), `id` its
  first 12 hex digits (as in `eval/lists/`); `source` is the TRELLIS-500K subset (`sketchfab` = Objaverse-XL,
  Sketchfab part, `3dfuture`, `hssd`, `abo`); `licence` is the licence in the TRELLIS-500K metadata.
  `matched_5243` is contained in `curated_10k`.
- 5k set: `id` is the object identifier (the Objaverse 1.0 uid for the 3,743 Sketchfab objects, the HSSD object
  id for the 1,500 HSSD objects), `sha256` is the sha256 of the asset file that was rendered (the Objaverse 1.0
  GLB of `allenai/objaverse`; the HSSD GLB of `hssd/hssd-hab`), `source` is `sketchfab` or `hssd`.
- Licences. 10k set: `by` 7,944, `3D-FUTURE-research-only` 598, `CC-BY-NC-4.0` 410 (HSSD), `by-nc` 317,
  `CC-BY-4.0` 279 (ABO), `by-nc-sa` 262, `by-sa` 134, `cc0` 56. 5k set: `by` 3,335, `CC-BY-NC-4.0` 1,500 (HSSD),
  `by-nc` 172, `by-nc-sa` 164, `by-sa` 38, `cc0` 34. Several licences are non-commercial or research-only; every
  asset stays under its own licence, and renders, caches and models trained on them are subject to these terms.
- Views. Every object was rendered from two views; the training caches hold one record per view, named
  `<key>.view00` and `<key>.view01` with `key` = `sha256` (10k set) or `id` (5k set). The 5k set has 10,486
  views. In the 10k set one object (`7f9dc2a3bc46`) has renders but no structure-support files, so the cache
  builders of the original runs skipped it: the full training pool of the texture student is 9,999 objects
  (19,998 views).
- Overlaps. The 10k set and the validation set (`eval/lists/validation_400.csv`) are disjoint by construction.
  The earlier 5k set shares 316 objects with the 10k set (158 of them in `matched_5243`) and 7 with the
  validation set (matched by Objaverse uid and HSSD id; `in_earlier_5k_training_set` in `validation_400.csv`).

## Selection of the 10k set

The 10k set, its 5,243-object subset and the 400-object validation set were drawn by `data/select/` from
TRELLIS-500K (`JeffreyXiang/TRELLIS-500K`, revision `288b0611`), subsets ObjaverseXL_sketchfab, 3D-FUTURE, HSSD
and ABO (the GitHub part of Objaverse-XL is not used; Toys4K is used for evaluation only). Only metadata is read:
the TRELLIS-500K captions (the aesthetic score is recorded, no threshold beyond TRELLIS-500K's own), the
Objaverse 1.0 name, tags, categories, description, animation count and uploader of the Sketchfab objects, face
counts (Sketchfab table; triangle counts from the GLB header of HSSD and ABO assets; polygon lines of the
3D-FUTURE OBJ files) and file sizes. No render is inspected at this stage.

```bash
cp data/select/config.example.yaml data/select/config.yaml    # set the input paths
bash data/select/run_all.sh data/select/config.yaml            # CPU only; outputs in <work_dir>/out/
```

| Step | Script | Output |
|---|---|---|
| 1 | `join_metadata.py` | TRELLIS-500K metadata joined with the Sketchfab table and the Objaverse 1.0 metadata |
| 2 | `geometry_headers.py` | triangle / vertex counts of the HSSD and ABO GLB files from their headers; 3D-FUTURE file sizes |
| 3 | `future_face_counts.py` | polygon and vertex line counts of the 3D-FUTURE OBJ files |
| 4 | `flags.py` (lexicons in `rules.py`) | per-object rule flags, Toys4K overlap |
| 5 | `select.py` | keep / reject, funnel statistics |
| 6 | `sets.py` | stratified nested sets and the validation split |
| 7 | `write_outputs.py` | candidate tables and the set lists |
| 8, 9 | `stats.py`, `stats_extra.py` | `selection_stats.json` |
| optional | `pilot.py`, `toys4k_neighbours.py`, `precision.py` | a QA sample for renders, a caption-neighbour audit of a Toys4K evaluation subset, a precision estimate from blind labels (labels not distributed) |

The Sketchfab table (`inputs.sketchfab_manifest`) maps every ObjaverseXL_sketchfab object of TRELLIS-500K to its
Objaverse 1.0 uid and GLB file, with face and vertex counts, file sizes and licence; the expected columns are
listed in `config.example.yaml`. It is not distributed.

### Rules

An object is rejected if any HARD family fires; the families are checked in this order (counts = objects
removed by each family after the previous ones, per subset Sketchfab / 3D-FUTURE / HSSD / ABO):

| Family | Rule | Removed |
|---|---|---|
| `toys4k_dup` | overlap with Toys4K, see below | 24 / 0 / 1 / 0 |
| `missing_or_broken_file` | 3D-FUTURE model without `normalized_model.obj` | – / 3 / – / – |
| `no_caption_sketchfab` | Sketchfab object without captions | 669 / – / – / – |
| `scene` | head noun of the shortest caption is a scene word (173 words, `rules.py` `SCENE_HEAD`; compounds such as "kitchen island" excepted); strong scene phrases in any caption ("diorama", "terrain", "game level", "a scene", ...; on the furniture subsets without the "a scene" family); Sketchfab names or tags such as diorama, terrain, cityscape; all objects of Sketchfab uploaders with at least 20 objects of which at least 50 % carry a scene signal (71 uploaders) | 20,461 / 31 / 32 / 2 |
| `scene_context` | an object with its surroundings: context phrases in the captions ("on grass", "with trees", "surrounded by ..."); on Sketchfab also ground or surrounding nouns after on / in / with and comma lists such as "House, pool, garden." | 5,964 / 0 / 0 / 0 |
| `multi_object_set` | collections, sets, packs, several objects or several poses of one object ("Two chairs.", "set of ...", "various poses"; TV / drum / swing / radio sets excepted) | 2,067 / 60 / 144 / 23 |
| `flat_or_text` | flat items: rug, mat, wall, tile, poster, painting, logo, text, card, decal, ... | 137 / 3 / 66 / 75 |
| `point_cloud` | Sketchfab names or tags pointcloud, potree, lidar, splat, 3dgs, or "point cloud" in a caption | 229 / – / – / – |
| `world_export` | Sketchfab tags of game-world exports (mineways, jmc2obj, worldpainter, minecraft-world) | 161 / – / – / – |
| `extreme_face_count` | Sketchfab: fewer than 12 faces or a GLB larger than 200 MB; all subsets: more than 2M faces | 1,965 / 0 / 2 / 4 |

**Toys4K deduplication** (`flags.py`, family `toys4k_dup`): an object is rejected if any of four signals fires:

1. its sha256 equals that of a Toys4K object (the Toys4K list of TRELLIS-500K, plus optional extra lists);
2. **name rule**: the regular expression `toys?\s?4k` matches the lower-cased Sketchfab name, tags or
   description of the object ("toy4k", "toys4k", "toys 4k", ...);
3. its shortest and longest captions are identical to those of a Toys4K object;
4. the TF-IDF cosine similarity between its captions and the captions of a Toys4K object is at least 0.85.

In this population signals 1-3 matched no object; all 25 rejected objects (24 Sketchfab, 1 HSSD) were rejected by
the caption similarity. Objects with a similarity between 0.70 and 0.85 are kept and flagged (1,249 objects of the population).

SOFT flags are recorded but do not reject (for stricter variants): comma lists in the shortest caption,
architecture / places categories, animated (rigged) objects, single buildings, photogrammetry and phone scans,
"A and B" captions, plural head nouns, 1M-2M faces, fewer than 100 faces, weak name/tag scene or pack words,
Toys4K caption similarity 0.70-0.85, no caption (non-Sketchfab subsets).

Result: 156,811 of 188,934 objects kept (Sketchfab 136,630 of 168,307; 3D-FUTURE 9,375 of 9,472; HSSD 6,425 of
6,670; ABO 4,381 of 4,485).

A metadata-only check (a blind labelling, by a language model, of 100 kept and 100 rejected Sketchfab objects
from name, tags, categories, captions and counts; not ground truth) estimated that 91.4 % of kept objects are
single objects (95 % CI 84.5-95.4 %) and that 66.3 % of rejected objects are scenes, multi-object sets or junk
(56.3-75.0 %); most wrongly rejected objects are single buildings removed with their uploader.

### Sets

- Source shares follow the keep pool (Sketchfab 87.13 %, 3D-FUTURE 5.98 %, HSSD 4.10 %, ABO 2.79 %), rounded by the
  largest-remainder method; within a source the objects are stratified by face count (bins < 1k, 1k-5k, 5k-20k,
  20k-100k, 100k-500k, 500k-2M) in proportion to the source's keep pool.
- Within each (source, bin) cell the objects are ordered by `sha256("trellis500k_curated_v1/selection_v1:" + sha256)`.
  The first objects of a cell form the validation set (400 objects); the following ones form the nested training
  sets: `matched_5243` (5,243) within `curated_10k` (10,000) within a 50,000-object set that was not used.
- 10k set per source: Sketchfab 8,713, 3D-FUTURE 598, HSSD 410, ABO 279.
  5,243-object subset: Sketchfab 4,568, 3D-FUTURE 313, HSSD 215, ABO 147.
- The selection defines a replacement order for objects that fail a later render check (the next objects of the
  same cell); the released lists contain no replacements.
