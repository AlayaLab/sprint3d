# Notes on checkpoint loading, data files and network behaviour

What this repository loads, executes and downloads, for readers who review it before running it.

* **Checkpoint loading:** the deployed students are safetensors files, verified by size and sha256 against
  `weights/sources.json`. Every `torch.load` in this package uses restricted deserialization
  (`weights_only=True`) by default; the loaders of training checkpoints, training records and the Uni3D / ULIP-2
  evaluation checkpoints fall back to full unpickling only when `FASTPIXAL3D_TRUST_CHECKPOINT=1` is set, and say
  so in their error message; the converter of training checkpoints (`fastpixal3d.students.convert`) does so only
  with `--allow-pickle`. PyTorch 2.7.0 is pinned (2.6 or newer includes the fix for CVE-2025-32434).
* **Pickle files:** `data/select/` reads the intermediate pickles that its own earlier steps wrote, and the
  evaluation and the cache builders read mesh dumps produced by the data preparation. Load only files you
  produced yourself or obtained from a trusted source.
* **Configuration parsing:** YAML is read with `yaml.safe_load`; command-line overrides are parsed as values, never
  evaluated as code.
* **Network behaviour:** no telemetry. With a complete weights directory, the inference path fetches only the
  model code of the NAF feature upsampler, through `torch.hub` at a pinned commit (as the upstream Pixal3D loader
  does). A model whose files are not all in the weights directory is loaded as Pixal3D's own code loads it, by hub
  id from the Hugging Face hub or `torch.hub` (Pixal3D itself at its pinned revision); `scripts/infer.py --offline`
  keeps the run from contacting the Hugging Face hub. All other downloads happen in the scripts that exist for
  that purpose: `scripts/download_weights.py` (Hugging Face repositories at pinned revisions and one GitHub release
  file, sha256 checked), `eval/fetch_metric_models.py` (Hugging Face, download.pytorch.org, dl.fbaipublicfiles.com
  and pinned GitHub commits, sha256 checked) and `env/install.sh` (conda-forge and nvidia channels, PyPI,
  download.pytorch.org, whl.natten.org, pinned GitHub commits, and Eigen from GitLab).
* **Processes:** subprocesses are started in list form without a shell (`git` for provenance and, in
  `eval/fetch_metric_models.py`, for fetching metric-model code at pinned commits; `nvidia-smi` and a Python child
  process for the timing sampler; the project's own scripts in `experiments/run_arms.py` and the GPU smoke test).
  `eval/tests/test_frames_projection.py` executes a function extracted from the pinned Pixal3D source.
* **Repository contents:** source, configurations, object lists, tests, documentation and the images of the README;
  no credentials, no weights, no datasets, no renders. `.gitignore` excludes `.env*`, `.netrc`, weight and cache
  files and the download directories under `weights/`.
* **Licence files:** `LICENSE`, `NOTICE` and `THIRD_PARTY_LICENSES.md` are declared in `pyproject.toml`
  (`license-files`) for source and wheel distributions.
