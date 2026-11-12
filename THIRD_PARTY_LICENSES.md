# Third-party licences

The first table indexes the full licence texts in `LICENSES/`, retrieved on 2026-11-04 from the upstream repositories
named in its last column. The component list that follows it gives every component with its version or commit and how it
is used. The patched native extensions built from `native/` also carry the licences of the code they patch
(`native/cumesh/`, `native/cubvh/`, `native/legacy/`).

| Component | Role in this package | Licence | Text | Source of the text |
|---|---|---|---|---|
| Pixal3D (code and weights) | base model: flows, decoders, inference pipeline (submodule `third_party/Pixal3D`, weights `TencentARC/Pixal3D`) | MIT, Copyright (c) 2026 Tencent | `LICENSES/Pixal3D.txt`, `LICENSES/Pixal3D-NOTICE.txt` | github.com/TencentARC/Pixal3D (LICENSE, NOTICE) |
| TRELLIS.2 incl. o-voxel | structured-latent backbone, export code (submodule `third_party/TRELLIS.2`) | MIT, Copyright (c) Microsoft Corporation | `LICENSES/TRELLIS.2.txt` | github.com/microsoft/TRELLIS.2 |
| CuMesh | remeshing, simplification, UV (submodule `third_party/CuMesh`; patched in `native/cumesh/`) | MIT, Copyright (c) 2025 Jianfeng Xiang | `LICENSES/CuMesh.txt` | github.com/JeffreyXiang/CuMesh |
| cubvh | BVH (submodule `third_party/cubvh`; patched in `native/cubvh/`) | MIT, Copyright (c) 2022 kiui, **and** NVIDIA Source Code License (non-commercial) | `LICENSES/cubvh.txt`, `LICENSES/cubvh-NVIDIA.txt` | github.com/JeffreyXiang/cubvh (branch trellis.2) |
| xatlas | CPU UV unwrapping inside CuMesh and `native/legacy/` | MIT, Copyright (c) 2018-2020 Jonathan Young | `LICENSES/xatlas.txt` | github.com/jpcy/xatlas |
| FlexGEMM | sparse convolutions (built from source at install) | MIT, Copyright (c) 2025 Jianfeng Xiang | `LICENSES/FlexGEMM.txt` | github.com/JeffreyXiang/FlexGEMM |
| MoGe / MoGe-2 weights | camera field of view | MIT, Copyright (c) Microsoft Corporation | `LICENSES/MoGe.txt` | github.com/microsoft/MoGe |
| utils3d | geometry utilities (with `env/patches/utils3d-moge2-compat.patch`) | MIT, Copyright (c) 2022 EasternJournalist | `LICENSES/utils3d.txt` | github.com/EasternJournalist/utils3d |
| pipeline | MoGe dependency | MIT, Copyright (c) 2025 Ruicheng Wang | `LICENSES/pipeline.txt` | github.com/EasternJournalist/pipeline |
| NAF | feature upsampler (`rope.py` carries DINOv3 License terms) | Apache-2.0 | `LICENSES/NAF.txt` | github.com/valeoai/NAF |
| nvdiffrast | texture baking, evaluation rendering (built from source at install, not distributed here) | NVIDIA Source Code License (1-Way Commercial): **non-commercial use only** | `LICENSES/nvdiffrast.txt` | github.com/NVlabs/nvdiffrast |
| FlashAttention-2 / -3 | attention kernels (submodule `third_party/flash-attention`) | BSD-3-Clause | `LICENSES/flash-attention.txt` | github.com/Dao-AILab/flash-attention |
| NATTEN | neighbourhood attention (pip) | MIT, Copyright (c) 2022-2026 Ali Hassani | `LICENSES/NATTEN.txt` | github.com/SHI-Labs/NATTEN |
| DINOv3 ViT-L/16 weights | image conditioning (downloaded by the user, not distributed here) | DINOv3 License (Meta): redistribution only under the same licence, acceptable-use and trade-control terms | `LICENSES/DINOv3.txt` | github.com/facebookresearch/dinov3 (LICENSE.md) |
| RMBG-2.0 weights | background removal for inputs without alpha (optional; gated, downloaded by the user) | CC BY-NC 4.0: **non-commercial use only** | `LICENSES/CC-BY-NC-4.0.txt` | creativecommons.org (legal code) |

Evaluation-only and baseline components (ULIP, Uni3D, TripoSG, Hunyuan3D-2.1, DINOv2, LPIPS, CLIP and the metric
weights) are listed with their licences in `THIRD_PARTY_LICENSES.md`; none of their files is distributed in this repository.

---

# Component list (versions, commits and use)
## Non-commercial terms that apply when the full system runs

Whatever licence this repository's own code has, running the complete pipeline uses the following components,
whose terms restrict use to non-commercial (research or evaluation) purposes:

| component | used for | licence |
|---|---|---|
| nvdiffrast (NVlabs/nvdiffrast @ 253ac4f) | texture baking (rasterising the UV atlas); evaluation rendering | NVIDIA Source Code License (1-Way Commercial): non-commercial use only. Copyright (c) 2020, NVIDIA Corporation |
| cubvh (inside CuMesh, and the patched extension built from `native/cubvh`) | BVH for remeshing and texture baking | MIT (Copyright (c) 2022 kiui) **and** `LICENSE_NVIDIA` (NVIDIA Source Code License, Copyright (c) 2022, NVIDIA Corporation & affiliates; non-commercial use only). The files it covers are not stated, so the whole extension is treated as covered. Both files are in `native/cubvh/` and `native/cumesh/third_party_licenses/` |
| RMBG-2.0 weights (briaai/RMBG-2.0) | background removal, only for input images without an alpha channel | CC BY-NC 4.0 (gated model) |
| DINOv3 ViT-L/16 weights | image conditioning | DINOv3 License (Meta). Redistribution only under the same licence; publications must acknowledge DINOv3; trade-control restrictions apply. Pixal3D loads the weights from the mirror `camenduru/dinov3-vitl16-pretrain-lvd1689m` of `facebook/dinov3-vitl16-pretrain-lvd1689m` | The same terms apply to the legacy BVH builds under `native/legacy/`, which are needed only by intermediate rows of the cumulative measurement.

The student weights were trained on 3D assets whose licences include non-commercial and research-only terms
(3D-FUTURE: research only; HSSD: CC BY-NC 4.0; Sketchfab assets under CC BY-NC and CC BY-NC-SA, among others).
See [docs/DATA.md](docs/DATA.md).

`eval/lists/toys4k_dev_500.csv` names objects of Toys4K by identifiers and checksums only. The authors of Toys4K ask
not to distribute their data (github.com/rehg-lab/lowshot-shapebias), and no file of it is in this repository.

## Pixal3D

Pixal3D (code and weights) is MIT-licensed, Copyright (c) 2026 Tencent. This project uses the unmodified Pixal3D
code at commit `28efad66` (`third_party/Pixal3D`) and its weights from `TencentARC/Pixal3D` at revision
`0b31f916`. Pixal3D's NOTICE is reproduced unchanged:

```text
Pixal3D
Copyright (c) 2026 Tencent.

Tencent is pleased to support the community by making Pixal3D available.

The open-source software and/or models included in this distribution may have
been modified by Tencent (“Tencent Modifications”). All Tencent Modifications
are Copyright (c) Tencent.

For the avoidance of doubt, Pixal3D refers solely to code, parameters, weights,
and associated documentation made publicly available by Tencent in accordance
with the MIT License (see the LICENSE file).

Responsible Use
This model is released to support research, creative production, and practical 3D generation. Users are responsible for ensuring that their use complies with applicable laws, rights, consent requirements, and platform policies.

We do not support use of this model to create non-consensual sexual content, sexualized depictions of real people, content involving minors, harassment, impersonation, or privacy-violating applications.

The model is provided “as is” under the MIT License. This guidance does not modify the license terms but describes the intended and responsible use of the project.

================================================================================
Third-Party Components
================================================================================

This product includes software developed by third parties. The third-party
components listed below remain licensed under their respective original terms.
Pixal3D does not impose any additional restrictions beyond those specified in
the original licenses of these third-party components. Users are required to
comply with all applicable terms and conditions of the original licenses and
to ensure that the use of these third-party components conforms to all
relevant laws and regulations.

In case you believe there have been errors in the attribution below, you may
submit the concerns to us for review and correction.


================================================================================
Open Model Licensed under Apache-2.0
================================================================================

1. dinov2
   Copyright (c) 2025 dinov2 original author and authors


================================================================================
Open Source Software Licensed under the MIT License
================================================================================

2. TRELLIS.2
   Copyright (c) Microsoft Corporation.

3. Direct3D-S2
   Copyright (c) DreamTech.

4. MoGe
   Copyright (c) Microsoft Corporation.


================================================================================
Terms of the MIT License
================================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


================================================================================
Terms of the Apache License, Version 2.0
================================================================================

Apache License
                       Version 2.0, January 2004
                    http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

Definitions.

"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.

"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

Grant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

Redistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:

(a) You must give any other recipients of the Work or Derivative Works a copy of this License; and (b) You must cause any modified files to carry prominent notices stating that You changed the files; and (c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and (d) If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License. You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

Submission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.

Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.

Accepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

```

## Other third-party components

Runtime (inference):

| component | version / commit | licence |
|---|---|---|
| TRELLIS.2, including o-voxel (microsoft/TRELLIS.2) | o-voxel @ de38fdd | MIT, Copyright (c) Microsoft Corporation |
| CuMesh (JeffreyXiang/CuMesh) | 12289e1 (stock build; the patched build of `native/cumesh` starts from the same commit) | MIT, Copyright (c) 2025 Jianfeng Xiang. Bundles cubvh (see above) and xatlas |
| xatlas (inside CuMesh) | as bundled | MIT, Copyright (c) 2018-2020 Jonathan Young |
| FlexGEMM (JeffreyXiang/FlexGEMM) | 6dd94a8 | MIT, Copyright (c) 2025 Jianfeng Xiang |
| MoGe (microsoft/MoGe) and the MoGe-2 ViT-L weights (Ruicheng/moge-2-vitl) | 0744441 / 39c4d5e | MIT, Copyright (c) Microsoft Corporation |
| utils3d (EasternJournalist/utils3d) | 9a4eb15 (version 0.0.2) with a local compatibility patch for MoGe-2, `env/patches/utils3d-moge2-compat.patch`, as in the measurement environment | MIT, Copyright (c) 2022 EasternJournalist |
| pipeline (EasternJournalist/pipeline) | 866f059 | MIT, Copyright (c) 2025 Ruicheng Wang |
| NAF feature upsampler (valeoai/NAF) | code 37f2dfc, release weights | Apache-2.0; `rope.py` carries DINOv3 License terms |
| FlashAttention-2 and FlashAttention-3 (Dao-AILab/flash-attention) | 2.8.3 / v2.8.3.post1 (a8aa52b, `hopper/`) | BSD-3-Clause |
| NATTEN | 0.21.0 | MIT |
| py-fpng-nb (`fpng`) and lazy_loader | 0.1.0 / 0.4 | MIT / BSD-3-Clause |
| PyTorch, torchvision, Triton | 2.7.0 / 0.22.0 / 3.3.0 | BSD-3-Clause / BSD-3-Clause / MIT |
| trimesh, Pillow (MIT-CMU), PyYAML, tqdm (MPL-2.0 and MIT) | see `env/requirements-runtime.lock` | MIT unless stated |
| numpy, scipy | see `env/requirements-runtime.lock` | BSD-3-Clause |
| OpenCV (headless), transformers, timm, safetensors, huggingface_hub, kornia, diffusers, accelerate | see `env/requirements-runtime.lock` | Apache-2.0 |
| plyfile | 1.1.3 | GPL-3.0-or-later. Imported by o-voxel; installed as a separate package, not distributed with this repository |
| easydict | 1.13 | LGPL-3.0. Installed as a separate package |

Evaluation (`eval/`):

| component | licence |
|---|---|
| DINOv2 (metric backbone) | Apache-2.0 |
| LPIPS | BSD-2-Clause |
| ULIP / ULIP-2 code | BSD-3-Clause (its PointNeXt part: MIT) |
| Uni3D code | MIT |
| CLIP, OpenCLIP and EVA weights, Uni3D and ULIP-2 weights, torchvision and Inception weights | fetched from their own repositories by `eval/fetch_metric_models.py`; their licences apply and are not restated here |

Baselines used in the report (not part of this repository's code):

| component | licence |
|---|---|
| TripoSG | MIT, with a NOTICE: parts derived from HunyuanDiT / FlashVDM fall under the Tencent Hunyuan community licence, RMBG-1.4 under the bria-rmbg-1.4 licence |
| Hunyuan3D-2.1 | Tencent Hunyuan 3D 2.1 Community License: not available in the EU, the United Kingdom and South Korea, and further conditions apply. Not vendored; only its commit is referenced |

### Baselines

The adapters in `experiments/baselines/` and the TRELLIS.2 port in `experiments/trellis2_mesh_path/` are code of
this repository. They import the following components from checkouts and downloads made by the user; none of these
files is distributed with this repository (commits and revisions: `experiments/baselines/sources.json`).

| component | version / commit | licence |
|---|---|---|
| TRELLIS.2 code (microsoft/TRELLIS.2) | 75fbf018 | MIT, Copyright (c) Microsoft Corporation |
| TRELLIS.2-4B weights (microsoft/TRELLIS.2-4B) | af44b45 | MIT |
| TripoSG code (VAST-AI-Research/TripoSG) | fc5c4099 | MIT, Copyright (c) 2025 VAST-AI-Research and contributors; its NOTICE: code derived from HunyuanDiT and FlashVDM (Tencent Hunyuan community licences), RMBG (bria-rmbg-1.4) and Diffusers (Apache-2.0) |
| TripoSG weights (VAST-AI/TripoSG) | 2c1c516 | MIT |
| RMBG-1.4 weights (briaai/RMBG-1.4), used by TripoSG for inputs without alpha | 2ceba5a | bria-rmbg-1.4 (non-commercial) |
| diso (iso-surface extraction of TripoSG) | 0.1.4 | CC BY-NC 4.0 |
| Hunyuan3D-2.1 code and weights (Tencent-Hunyuan/Hunyuan3D-2.1, tencent/Hunyuan3D-2.1), optional | 82920d64 / 0b94677 | Tencent Hunyuan 3D 2.1 Community License Agreement: not applicable in the European Union, the United Kingdom and South Korea; further conditions apply |
| DINOv2 giant weights (facebook/dinov2-giant), used by Hunyuan3D-2.1 | 611a9d4 | Apache-2.0 |
| Real-ESRGAN x4plus checkpoint, used by Hunyuan3D-2.1 | v0.1.0 release | BSD-3-Clause |

Data: no data is distributed. Object lists carry each asset's licence; every asset and anything rendered or derived
from it stays under its own licence (see [docs/DATA.md](docs/DATA.md)). Rendering uses Blender (GPL), which is not
distributed.
