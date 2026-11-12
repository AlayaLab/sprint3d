"""Training of the four students: trajectory distillation (structure, texture) and consistency
distillation (coarse shape, fine shape).

The modules here are shared by the command-line scripts under ``train/``:

``config``      YAML configuration files: defaults, overrides, path resolution, checks
``schedule``    the teacher's 12-step time grid, segment pairs, per-view noise
``objective``   one-step trajectory loss, gradient accumulation, clipping, AdamW, EMA
``student``     base flow + rank-256 LoRA, wrappers that give the released parameter names
``checkpoint``  training checkpoints (the format ``fastpixal3d.students.convert`` reads)
``records``     cache records: atomic writes, view names, the input-image recipe, cameras
``environment`` process settings and loading Pixal3D from its source checkout
``loop``        the update loop shared by the two trajectory trainers (seeding order, sampling, logging)
``consistency`` the consistency distillation of the shape students: anchor, guided teacher step, loss,
                EMA ramp, data order and its own update loop

Only ``config`` and ``schedule`` work without torch; nothing here imports Pixal3D at module level.
"""
