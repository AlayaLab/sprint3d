"""tables.py -- Markdown tables of one evaluation run (presentation only; every value comes from the corpus).

  tables/table1.md            normal-map columns (IoU / PSNR / SSIM / LPIPS / Mean / Median / Mean_B / 11.25 / 22.5 / 30)
                              for the official-condition and baseline rows; PSNR / SSIM / LPIPS in the region chosen by
                              --table1-region, followed by both regions side by side
  tables/true_camera.md       rows generated with the ground-truth camera + the per-arm condition gap
  tables/geometry_3d.md       single-view 3D geometry vs ground truth (own-bbox frame; voxel frame for true_camera rows)
  tables/perception.md        CLIP, CLIP-N, ULIP-2 / Uni3D (coloured and geometry-only)
  tables/vs_reference.md      agreement with the reference arm of the same condition
  tables/speed.md             end-to-end seconds recorded by the generators and face counts
  tables/paired_statistics.md paired statistics vs the reference arm of the same condition
The LaTeX rows of the report's normal-map table are produced by eval/make_table.py from per_object.json.
"""
from __future__ import annotations

import json
from pathlib import Path

TABLE1_REGIONS = ("overlap", "full")
TABLE1_REGION_DEFAULT = "overlap"
REGION_KEYS = {"overlap": {"psnr": "psnr_ov", "ssim": "ssim_ov", "lpips": "lpips_ov"},
               "full": {"psnr": "psnr_full", "ssim": "ssim_full", "lpips": "lpips_full"}}
TABLE1_CONDITIONS = ("official", "baseline")


def check_region(region: str) -> str:
    if region not in TABLE1_REGIONS:
        raise ValueError(f"table1_region must be one of {TABLE1_REGIONS}, got {region!r}")
    return region


def table1_columns(region: str) -> list:
    """[(header, metric key, display scale, decimals)] -- the 10 columns of the normal-map table (Table 1 of the Pixal3D paper); PSNR / SSIM / LPIPS keys follow
    the region."""
    k = REGION_KEYS[check_region(region)]
    return [("IoU↑", "mask_iou", 100.0, 2), ("PSNR↑", k["psnr"], 1.0, 2), ("SSIM↑", k["ssim"], 1.0, 3),
            ("LPIPS↓", k["lpips"], 1.0, 3), ("Mean↓", "ang_mean", 1.0, 2), ("Median↓", "ang_median", 1.0, 2),
            ("Mean_B↓", "ang_mean_b", 1.0, 2), ("11.25°↑", "acc_11.25", 100.0, 2), ("22.5°↑", "acc_22.5", 100.0, 2),
            ("30°↑", "acc_30.0", 100.0, 2)]


def _fmt(v, nd=3, scale=1.0):
    if v is None:
        return "—"
    try:
        x = float(v) * scale
    except (TypeError, ValueError):
        return "—"
    if x != x or x in (float("inf"), float("-inf")):
        return "—"
    return f"{x:.{nd}f}"


def _mean(corpus, arm, key):
    return corpus["summary"].get(arm, {}).get(key, {}).get("mean")


def _n(corpus, arm, key):
    return corpus["summary"].get(arm, {}).get(key, {}).get("n")


def _cond(corpus, arm):
    return corpus.get("arm_condition", {}).get(arm)


def arms_of(corpus, arms, conditions) -> list:
    return [a for a in arms if _cond(corpus, a) in conditions]


def _t1_rows(corpus, arms, region) -> list:
    cols = table1_columns(region)
    L = ["| Method | n | " + " | ".join(h for h, *_ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for arm in arms:
        L.append(f"| {arm} | {_n(corpus, arm, 'mask_iou') or 0} | "
                 + " | ".join(_fmt(_mean(corpus, arm, k), nd, s) for _, k, s, nd in cols) + " |")
    return L


def table1_record(corpus, arm) -> dict:
    """both region conventions of one Table-1 row (display units: IoU / accuracies in %)."""
    out = {"arm": arm, "condition": _cond(corpus, arm), "n_objects": _n(corpus, arm, "mask_iou")}
    for region in TABLE1_REGIONS:
        out[region] = {h: (None if _mean(corpus, arm, k) is None else _mean(corpus, arm, k) * s)
                       for h, k, s, _ in table1_columns(region)}
    return out


def render_table1(corpus, arms, region) -> str:
    rows = arms_of(corpus, arms, TABLE1_CONDITIONS)
    bg = corpus["protocol"]["conventions"]["image_metrics"]["background_name"]
    L = ["# Normal maps in the input-image frame\n",
         f"PSNR / SSIM / LPIPS region: **{region}**. Pixal3D-family rows: `official` condition (rendered with the "
         "product's own inference-time projection). Baselines: ground-truth camera pose, fixed per-method frame map. "
         "Objects weighted equally, views averaged per object.\n"]
    L += _t1_rows(corpus, rows, region) if rows else ["(no official-condition or baseline rows in this run)"]
    L.append("\nBoth regions (`overlap`: PSNR over pred & gt pixels, SSIM map averaged over the overlap, LPIPS with both "
             f"images 0.5-gray outside the overlap; `full`: all 512^2 pixels on a constant {bg} background):\n")
    L.append("| Method | Condition | PSNR overlap↑ | PSNR full↑ | SSIM overlap↑ | SSIM full↑ | LPIPS overlap↓ | LPIPS full↓ | "
             "LPIPS full (vgg)↓ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for arm in arms_of(corpus, arms, ("official", "baseline", "true_camera", "manual_fov")):
        m = lambda k, nd=3: _fmt(_mean(corpus, arm, k), nd)  # noqa: E731
        L.append(f"| {arm} | {_cond(corpus, arm)} | {m('psnr_ov', 2)} | {m('psnr_full', 2)} | {m('ssim_ov')} | "
                 f"{m('ssim_full')} | {m('lpips_ov')} | {m('lpips_full')} | {m('lpips_full_vgg')} |")
    return "\n".join(L) + "\n"


def render_other(corpus, arms, region) -> dict:
    out = {}
    tc = arms_of(corpus, arms, ("true_camera", "manual_fov"))
    L = ["# Rows generated with the ground-truth camera (true_camera) and manual-FoV rows\n"]
    L += _t1_rows(corpus, tc, region) if tc else ["(no true_camera or manual_fov rows in this run)"]
    gap = corpus.get("condition_gap") or {}
    if gap:
        L.append("\nCondition gap per arm (paired per object, other − true_camera; bootstrap 95% CI; sign test +/−/=)\n")
        L.append("| Arm | metric | true_camera | other | Δ | 95% CI | sign +/−/= |")
        L.append("|---|---|---|---|---|---|---|")
        for arm, g in gap.items():
            for m, x in g["metrics"].items():
                b, s = x.get("bootstrap"), x.get("sign_test") or {}
                if not b:
                    continue
                L.append(f"| {arm} vs {g['true_camera_arm']} | {m} | {_fmt(x['true_camera_mean'], 4)} | "
                         f"{_fmt(x['other_mean'], 4)} | {b['mean_diff']:+.4g} | [{b['ci95_lo']:+.4g}, {b['ci95_hi']:+.4g}] | "
                         f"{s.get('n_pos')}/{s.get('n_neg')}/{s.get('n_tie')} |")
    out["true_camera.md"] = "\n".join(L) + "\n"

    faces = corpus.get("faces", {})
    L = ["# Single-view 3D geometry vs ground truth (every mesh normalised by its own bbox; no registration)\n",
         "| Method | Condition | CD↓ (×1e-3) | F@0.005↑ | F@0.01↑ | F@0.02↑ | #faces | true_camera only: CD_vox (×1e-3) | "
         "F@0.01_vox | Support IoU@32³ | Prec. | Rec. |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        m = lambda k, nd=3, s=1.0: _fmt(_mean(corpus, arm, k), nd, s)  # noqa: E731
        L.append(f"| {arm} | {_cond(corpus, arm)} | {m('chamfer_nb', 2, 1000)} | {m('f_0.005_nb')} | {m('f_0.01_nb')} | "
                 f"{m('f_0.02_nb')} | {_fmt((faces.get(arm) or {}).get('mean'), 0)} | {m('chamfer_vox', 2, 1000)} | "
                 f"{m('f_0.01_vox')} | {m('support_iou')} | {m('support_precision')} | {m('support_recall')} |")
    out["geometry_3d.md"] = "\n".join(L) + "\n"

    L = ["# Perception: CLIP (textured renders, 4 yaw views), CLIP-N (normal renders), ULIP-2 / Uni3D (coloured and "
         "geometry-only)\n",
         "| Method | Condition | CLIP↑ | CLIP view-0↑ | CLIP-N↑ | CLIP-N view-0↑ | ULIP-2↑ | ULIP-2 geo↑ | Uni3D↑ | Uni3D geo↑ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        m = lambda k: _fmt(_mean(corpus, arm, k), 4)  # noqa: E731
        L.append(f"| {arm} | {_cond(corpus, arm)} | {m('clip')} | {m('clip_input_view')} | {m('clip_n')} | "
                 f"{m('clip_n_input_view')} | {m('ulip2')} | {m('ulip2_geo')} | {m('uni3d')} | {m('uni3d_geo')} |")
    out["perception.md"] = "\n".join(L) + "\n"

    L = ["# Agreement with the reference arm of the same condition (CD / F to the reference product, DINOv2 4 views)\n",
         f"reference per condition: {json.dumps(corpus.get('reference', {}))}\n",
         "| Method | Condition | CD-to-ref↓ (×1e-3) | F@0.01-to-ref↑ | DINO CLS↓ | DINO patch↓ | flips |",
         "|---|---|---|---|---|---|---|"]
    fl = (corpus.get("flips") or {}).get("per_arm", {})
    for arm in arms:
        if arm not in fl:
            continue
        f = fl.get(arm, {})
        fs = f"{f['n_flips']}/{f['n']}" if f.get("n_flips") is not None else "n/a (no noise band)"
        L.append(f"| {arm} | {_cond(corpus, arm)} | {_fmt(_mean(corpus, arm, 'cd_to_ref'), 2, 1000)} | "
                 f"{_fmt(_mean(corpus, arm, 'f_0.01_to_ref'), 3)} | {_fmt(_mean(corpus, arm, 'dino_cls'), 4)} | "
                 f"{_fmt(_mean(corpus, arm, 'dino_patch'), 4)} | {fs} |")
    out["vs_reference.md"] = "\n".join(L) + "\n"

    L = ["# End-to-end seconds as recorded by the generators, and face counts of the evaluated products\n",
         "| Method | Condition | n | E2E mean (s) | median | p10 | p90 | #faces mean |", "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        sp = (corpus.get("speed") or {}).get(arm) or {}
        L.append(f"| {arm} | {_cond(corpus, arm)} | {sp.get('n', 0)} | {_fmt(sp.get('mean'), 2)} | {_fmt(sp.get('median'), 2)} | "
                 f"{_fmt(sp.get('p10'), 2)} | {_fmt(sp.get('p90'), 2)} | {_fmt((faces.get(arm) or {}).get('mean'), 0)} |")
    out["speed.md"] = "\n".join(L) + "\n"

    other = "full" if region == "overlap" else "overlap"
    keys = [k for _, k, _, _ in table1_columns(region)] + [REGION_KEYS[other][x] for x in ("psnr", "ssim", "lpips")] + \
           ["chamfer_nb", "f_0.01_nb", "clip", "clip_n", "ulip2", "ulip2_geo", "uni3d", "uni3d_geo", "cd_to_ref", "dino_cls"]
    L = ["# Paired statistics vs the reference arm of the same condition (per-object deltas; bootstrap 95% CI; sign "
         "test, Holm over the metrics of each arm)\n",
         "| Method | metric | mean Δ | 95% CI | sign +/−/= | p (Holm) | verdict |", "|---|---|---|---|---|---|---|"]
    for arm, rowsm in (corpus.get("paired") or {}).items():
        for mk in keys:
            x = rowsm.get(mk)
            if not x or not x.get("bootstrap"):
                continue
            b, s = x["bootstrap"], x.get("sign_test") or {}
            verdict = "n.s."
            if b.get("significant"):
                verdict = "better" if x.get("candidate_better_on_mean") else "worse"
            L.append(f"| {arm} | {mk} | {b['mean_diff']:+.4g} | [{b['ci95_lo']:+.4g}, {b['ci95_hi']:+.4g}] | "
                     f"{s.get('n_pos')}/{s.get('n_neg')}/{s.get('n_tie')} | {_fmt(s.get('p_holm', s.get('p_value')), 3)} | {verdict} |")
    out["paired_statistics.md"] = "\n".join(L) + "\n"
    return out


def write_tables(out_dir, corpus, arms, region) -> dict:
    """writes <out>/tables/*.md; returns the index stored in corpus['tables']."""
    region = check_region(region)
    td = Path(out_dir) / "tables"
    td.mkdir(parents=True, exist_ok=True)
    files = {"table1.md": render_table1(corpus, arms, region), **render_other(corpus, arms, region)}
    for name, text in files.items():
        (td / name).write_text(text)
    return {"table1_region": region, "table1_columns": [h for h, *_ in table1_columns(region)],
            "table1_keys": {r: [k for _, k, _, _ in table1_columns(r)] for r in TABLE1_REGIONS},
            "table1": [table1_record(corpus, arm) for arm in arms_of(corpus, arms, TABLE1_CONDITIONS)],
            "files": sorted(files)}
