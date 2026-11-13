# Weights

The pipeline uses the public Pixal3D model and four small *students* that replace its slow sampling
stages. Only the students are distributed with this project: they are LoRA adapters on top of the
Pixal3D flow models, stored in the Hugging Face model repository
[`AlayaLab/sprint3d`](https://huggingface.co/AlayaLab/sprint3d) at commit
`6c3864784f21a5da225ea2a78563f66f676228c5`. Everything else is fetched from its original repository at a
pinned revision. All files, revisions, sizes and sha256 values are listed in
[`weights/sources.json`](../weights/sources.json).

## Download and verify

```bash
python scripts/download_weights.py                    # about 32.6 GB in total
python scripts/download_weights.py --only students    # only the student files (5.19 GB)
python scripts/verify_weights.py                      # size + sha256 of every file
```

Both scripts accept `--root DIR` (default: the `weights/` directory, or `$FASTPIXAL3D_WEIGHTS`) and
`--only NAME[,NAME]` with the source names below. Files that are already present with the pinned
sha256 are not downloaded again. `verify_weights.py` needs only the Python standard library; it also
checks that the hashes recorded in `students.json` match the student files.
`download_weights.py --students-repo USER/REPO [--students-revision REV]` reads the student files from
another copy of the repository; they are still checked against the pinned sha256 values.

Access:

- Downloading the student repository needs no token once it is public; `hf auth login` is only needed for the
  gated upstream repositories below. Without access to a gated repository the script
  stops with a "not found (or not accessible ...)" message.
- Two upstream repositories are gated: Pixal3D (in the EU) and RMBG-2.0. Accept their terms on the
  model pages, with the same login.

Layout after download:

```
weights/
  students/   students.json, structure_student.safetensors, coarse_shape_student.safetensors,
              fine_shape_student.safetensors, texture_student.safetensors
  pixal3d/    pipeline.json, ckpts/*.json, ckpts/*.safetensors, LICENSE, NOTICE
  dinov3/     config.json, preprocessor_config.json, model.safetensors, LICENSE.md
  moge2/      model.pt
  rmbg2/      config.json, BiRefNet_config.py, birefnet.py, preprocessor_config.json, model.safetensors
  naf/        naf_release.pth
```

This directory is the only copy of the weights the package needs: `scripts/infer.py`, `scripts/time_e2e.py`,
`scripts/run_official.py` and the training code read every model from it (`$FASTPIXAL3D_WEIGHTS`, by
default `weights/`). `--students` / `FASTPIXAL3D_STUDENTS` and `--pixal3d` / `FASTPIXAL3D_PIXAL3D` point
elsewhere if needed. Pixal3D's own inference code loads DINOv3, MoGe-2, RMBG-2.0 and NAF by their hub ids;
the package hands the same loaders the downloaded files instead (`src/fastpixal3d/auxiliary.py`), so
nothing is downloaded a second time into `HF_HOME` or `TORCH_HOME`. A model whose files are not all in the
weights directory is loaded as Pixal3D's own code loads it: by hub id, through the Hugging Face cache or
`torch.hub` (then at whatever revision the cache holds). NAF's model code is not a weight file:
`torch.hub` fetches it once at the pinned commit into `TORCH_HOME/hub`; offline, when that commit is not
cached, the copy that Pixal3D's own loader uses (branch `main`) is taken.

## Sources

| Source (`--only`) | Repository @ revision | Files | Size | License | Used for |
|---|---|---:|---:|---|---|
| `students` | [`AlayaLab/sprint3d`](https://huggingface.co/AlayaLab/sprint3d) @ `6c3864784f21` | 5 | 5.19 GB | CC BY-NC 4.0 | EMA LoRA of the four students |
| `pixal3d` | [`TencentARC/Pixal3D`](https://huggingface.co/TencentARC/Pixal3D) @ `0b31f9160aa4` | 17 | 24.04 GB | MIT (gated in the EU) | base flow models, decoders, `pipeline.json` |
| `dinov3` | [`camenduru/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) @ `3c276edd87d6` | 4 | 1.21 GB | DINOv3 License | image encoder for conditioning |
| `moge2` | [`Ruicheng/moge-2-vitl`](https://huggingface.co/Ruicheng/moge-2-vitl) @ `39c4d5e957af` | 1 | 1.31 GB | MIT | camera field of view |
| `rmbg2` | [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) @ `5df4c9c76d81` | 5 | 0.88 GB | CC BY-NC 4.0 (gated) | background removal, only for inputs without alpha |
| `naf` | [`valeoai/NAF`](https://github.com/valeoai/NAF) release `model`, code @ `37f2dfc180f2` | 1 | 2.7 MB | Apache-2.0 | feature upsampler |

Full revisions are in `weights/sources.json`. Notes:

- The student weights are released under CC BY-NC 4.0 (`LICENSES/CC-BY-NC-4.0.txt`). They were trained on
  renders of assets that include non-commercial and research-only licences:
  15.9 % of the 10k set (texture student) and 35.0 % of the 5k set (structure and shape students); see
  [DATA.md](DATA.md).
- The DINOv3 repository is a mirror of `facebook/dinov3-vitl16-pretrain-lvd1689m`, the repository named
  in Pixal3D's `pipeline.json`; the Pixal3D inference code builds its image encoders from this mirror, so
  the gated `facebook` repository is not needed. The two repositories list the same git object ids for all
  files, so the weights are identical (checked 2026-11-05). The DINOv3 License applies.
- RMBG-2.0 is licensed for non-commercial use only. It is not run for RGBA inputs, and it is optional:
  without `rmbg2/` (for example without access to the gated repository) the pipeline still loads and
  runs RGBA inputs; an input without an alpha channel then stops with a message.
- NAF's model code is loaded with `torch.hub` at the pinned commit; its weights come from the
  release asset `naf_release.pth`. Part of the NAF code (`rope.py`) carries DINOv3 License terms.
- The Pixal3D repository's `NOTICE` lists further third-party components and is kept next to the
  weights.

## Student files

| Student | File | Size (bytes) | sha256 | Base submodel (in `TencentARC/Pixal3D`) | Fold |
|---|---|---:|---|---|---|
| structure | `structure_student.safetensors` | 1,274,085,008 | `a938130271c96eaaf67643bb428aa2aa8e53033b09608ea3eedc5531d9b97396` | `ckpts/ss_flow_img_dit_1_3B_64_bf16` | none |
| coarse shape | `coarse_shape_student.safetensors` | 1,305,542,328 | `59be306522816b4d1defa525cfecf3f9666a161902b245fee387cd7a34fb9a66` | `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16` | none |
| fine shape | `fine_shape_student.safetensors` | 1,305,542,328 | `6beb37c82f5f618f84d9bc988564d41e3971ff85c42ef06217e580186e7dd583` | `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16` | bf16 |
| texture | `texture_student.safetensors` | 1,305,542,328 | `ecc4ae096358120cde9ef2a6bbdff8bfee5c04e5f3bc49db84183d0f7d37887c` | `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16` | bf16 |

Each file holds 480 float32 tensors: LoRA factors (rank 256, alpha 256) for 240 linear layers, eight
in each of the 30 transformer blocks (`self_attn.to_qkv`, `self_attn.to_out`,
`cross_attn.cross_attn_block.{to_q,to_kv,to_out}`, `cross_attn.proj_linear`, `mlp.mlp.0`,
`mlp.mlp.2`); the adaLN modulation layers are not adapted. The structure student has 318,504,960
parameters, the other three 326,369,280 each. The tensors are the exponential moving average of the
trained adapters; optimizer state and other training data are not included.

| Student | Network evaluations | Time grid t (1 = noise, 0 = data) | Update |
|---|---:|---|---|
| structure | 4 | 1.0, 0.9, 0.75, 0.6, 0 | Euler, no guidance |
| coarse shape | 1 | 1.0 → 0 | x0 prediction |
| fine shape | 1 | 1.0 → 0 | x0 prediction |
| texture | 2 | 1.0, 0.75, 0 | Euler, no guidance |

The structure and texture grids are subsets of Pixal3D's 12-step grid with `rescale_t = 3`
(indices 0/3/6/8/12 and 0/6/12, recorded as `steps.grid_indices`).

### `students.json`

```json
{
  "format": 1,
  "base_model": {"repo": "TencentARC/Pixal3D", "revision": "0b31f9160aa400719af409098bff7936a932f726"},
  "key_convention": "...",
  "fold_rule": {"none": "...", "bf16": "..."},
  "students": {
    "fine_shape": {
      "file": "fine_shape_student.safetensors", "sha256": "...", "size_bytes": 1305542328,
      "base_submodel": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
      "pipeline_key": "shape_slat_flow_model_1024",
      "lora": {"rank": 256, "alpha": 256, "target_modules": ["self_attn.to_qkv", "..."],
               "num_blocks": 30, "num_adapters": 240, "num_tensors": 480, "num_elements": 326369280},
      "steps": {"num_steps": 1, "t": [1.0, 0.0], "update": "x0_prediction", "...": "..."},
      "fold": "bf16", "dtype": "float32",
      "source_checkpoint_sha256": "...", "training_updates": 2000
    },
    "...": {}
  }
}
```

`base_submodel` is the path of the flow model in the Pixal3D repository without its `.json` /
`.safetensors` extension, and `pipeline_key` is its key in Pixal3D's `pipeline.json`.
`target_modules` are layer names inside a transformer block: every adapted layer
(`blocks.<i>.<target>`) ends with one of them. `source_checkpoint_sha256` identifies the training
checkpoint the file was converted from.

**Keys.** Each tensor is named `backbone.model.<path>.lora_A.weight` or
`backbone.model.<path>.lora_B.weight`, where `<path>` is the name of an `nn.Linear` in the base flow
model's state dict without the `.weight` suffix, for example
`backbone.model.blocks.0.self_attn.to_qkv.lora_A.weight` for the layer whose weight is
`blocks.0.self_attn.to_qkv.weight`. `lora_A` has shape `[rank, in_features]`, `lora_B` has shape
`[out_features, rank]`, and the update of the layer weight is `(alpha / rank) * lora_B @ lora_A`.
The same naming is used for all four students; the `backbone.model.` prefix comes from the module
that wraps the flow model during training.

**Fold.** For `fold: "none"` the adapters stay separate and compute in float32:
`y = base(x) + (lora_B(lora_A(x.float())) * alpha / rank).to(base(x).dtype)`.
For `fold: "bf16"` each adapted weight `W` (bfloat16, like all weights in the transformer blocks) is
replaced once at load time by

```python
W + ((lora_B.float() @ lora_A.float()) * (alpha / rank)).to(torch.bfloat16)   # sum taken in bfloat16
```

and the adapters are removed, which makes the student as fast as the base model. Rounding to
bfloat16 removes part of the small LoRA update: 48.2 % of its Frobenius norm (summed over all 240
layers) for the fine-shape student and 5.7 % for the texture student. The reported results were
measured with this fold, so it is part of the model: load these two students with exactly this
formula. `fastpixal3d.students.fold.fold_lora` implements it.

## Converting a training checkpoint

`src/fastpixal3d/students/convert.py` turns a training checkpoint of a student into this format:

```bash
PYTHONPATH=src python -m fastpixal3d.students.convert --student fine_shape \
    --checkpoint completed_updates_000000002000.pt --out my_students
python scripts/verify_weights.py --students-dir my_students
```

A training checkpoint stores the trainable LoRA weights under `gen` (name → tensor, in parameter
order) and their moving average under `ema["shadow"]`, either name-keyed or as a list in the order of
`gen`. A list is named by pairing it with the keys of `gen` in order; every pair must agree in shape
and dtype. The converter writes the EMA tensors unchanged, re-reads the file to check every tensor bit
for bit, and adds or replaces the student's entry in `students.json`. The output is deterministic: the
same checkpoint gives the same file and sha256. Checkpoints that contain more than tensors (for
example random-number generator states) need `--allow-pickle`, which unpickles arbitrary objects;
use it only for files you trust.
