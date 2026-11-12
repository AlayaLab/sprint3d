#!/usr/bin/env python
"""run_eval.py -- evaluation of generated GLBs against the ground truth of the evaluation set (Pixal3D protocol).

Inputs
  --config CFG              YAML configuration (keys documented in eval/config.example.yaml): ground-truth root,
                            object list, metric-model locations, evaluation options
  --manifest ROWS           products of Pixal3D-family pipelines (JSON list or CSV; see metrics/manifest.py and
                            eval/make_manifest.py): object_id, view, product_glb, arm, condition, inference camera
  --baseline-manifest ROWS  external baselines (method, object_id, view, geometry_glb[, textured_glb])

Outputs (--out DIR)
  per_view.json    one record per (arm, object, view, repeat) with every metric, keyed by the code and conventions hash
  per_object.json  {arm: {sha256: {metric: mean over the object's views}}}
  corpus.json      summaries, paired statistics vs the reference arm of the same condition, conventions, hashes
  tables/*.md      Markdown tables (eval/make_table.py writes the LaTeX rows of the report's normal-map table)
  renders/         pred | ground-truth normal-map pairs (--save-renders)

Every record carries the code hash and the conventions hash; a rerun into the same --out resumes and never reuses
records of other code or other conventions (they are set aside in per_view.stale_<time>.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE))

from metrics import conventions as conv_mod  # noqa: E402
from metrics import stats, tables  # noqa: E402
from metrics.config import apply_environment, load_config  # noqa: E402
from metrics.dataset import GTDataset  # noqa: E402
from metrics.frame import ViewFrame  # noqa: E402
from metrics.manifest import attach_gt, base_arm, rows_from_manifest  # noqa: E402

METRICS_2D = ["mask_iou", "psnr_full", "ssim_full", "lpips_full", "lpips_full_vgg", "psnr_ov", "ssim_ov", "lpips_ov",
              "ang_mean", "ang_median", "ang_mean_b", "acc_11.25", "acc_22.5", "acc_30.0"]
METRICS_3D = ["chamfer_nb", "f_0.005_nb", "f_0.01_nb", "f_0.02_nb", "chamfer_vox", "f_0.005_vox", "f_0.01_vox", "f_0.02_vox",
              "support_iou", "support_precision", "support_recall", "product_faces"]
METRICS_PERC = ["clip", "clip_n", "clip_input_view", "clip_n_input_view", "ulip2", "uni3d", "ulip2_geo", "uni3d_geo"]
METRICS_REF = ["cd_to_ref", "f_0.01_to_ref", "dino_cls", "dino_patch"]
METRICS_DIAG = ["fov_used_deg", "fov_true_crop_deg", "fov_error_deg"]
LOWER_BETTER = {"lpips_full", "lpips_full_vgg", "lpips_ov", "ang_mean", "ang_median", "ang_mean_b", "chamfer_nb", "chamfer_vox",
                "cd_to_ref", "dino_cls", "dino_patch", "e2e_s"}
HIGHER_BETTER = {m: (m not in LOWER_BETTER) for m in METRICS_2D + METRICS_3D + METRICS_PERC + METRICS_REF + ["e2e_s"]}
HIGHER_BETTER["product_faces"] = None
HIGHER_BETTER.update({m: None for m in METRICS_DIAG})
GAP_METRICS = ["mask_iou", "psnr_full", "ssim_full", "lpips_full", "psnr_ov", "ssim_ov", "lpips_ov", "ang_mean", "ang_median",
               "acc_22.5", "chamfer_nb", "f_0.01_nb", "clip", "clip_n"]
CONDITION_ORDER = list(conv_mod.CONDITION_LABELS)   # deterministic order only, not a ranking
SKIP_WORDS = {"2d", "3d", "perception", "vs_reference", "lpips", "lpips_vgg", "dino", "clip", "ulip2", "uni3d"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML configuration (default: built-in defaults; see eval/config.example.yaml)")
    ap.add_argument("--manifest", help="product rows (JSON list or CSV)")
    ap.add_argument("--baseline-manifest", help="external-baseline rows (JSON list or CSV)")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--dataset-root", help="override dataset.root of the configuration")
    ap.add_argument("--device", help="override evaluation.device")
    ap.add_argument("--background", choices=sorted(conv_mod.BACKGROUNDS), help="override evaluation.background")
    ap.add_argument("--table1-region", choices=list(tables.TABLE1_REGIONS), help="override evaluation.table1_region "
                    "(presentation only: both regions are always stored)")
    ap.add_argument("--reference-arm", help="override evaluation.reference_arm (base arm name, without @condition)")
    ap.add_argument("--noise-arm", default=None, help="second sample of the reference arm (noise band for flips)")
    ap.add_argument("--skip", default=None, help="override evaluation.skip: comma list of " + ",".join(sorted(SKIP_WORDS)))
    ap.add_argument("--save-renders", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N (object, view) groups only")
    ap.add_argument("--only", default="", help="comma list of object ids / sha256 prefixes to evaluate")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--flip-quantile", type=float, default=0.95)
    ap.add_argument("--title", default="evaluation")
    ap.add_argument("--print-hashes", action="store_true", help="print the code / conventions hashes and exit")
    ap.add_argument("--tables-only", action="store_true",
                    help="re-render corpus / tables of an existing --out from its per_view.json (same code and "
                         "conventions), e.g. with another --table1-region; nothing is computed")
    return ap.parse_args(argv)


def effective_settings(a):
    cfg = load_config(a.config)
    ev = cfg["evaluation"]
    for k in ("device", "background", "table1_region", "reference_arm"):
        v = getattr(a, k)
        if v is not None:
            ev[k] = v
    if a.n_boot is not None:
        ev["n_boot"] = a.n_boot
    if a.skip is not None:
        ev["skip"] = [s.strip() for s in a.skip.split(",") if s.strip()]
    bad = set(ev["skip"]) - SKIP_WORDS
    if bad:
        raise SystemExit(f"unknown --skip words {sorted(bad)}")
    if a.dataset_root:
        cfg["dataset"]["root"] = str(Path(a.dataset_root).resolve())
    return cfg


def build_conventions(cfg):
    ev = cfg["evaluation"]
    conv = conv_mod.effective_conventions(ev["background"], ev["frame_maps"])
    code_h, per_file = conv_mod.code_hash(SUITE)
    return conv, conv_mod.conventions_hash(conv), code_h, per_file


def check_list(cfg):
    """optional startup check of the object list against the sha256 pinned in the configuration"""
    want = cfg["dataset"].get("list_sha256")
    if want is not None:
        got = conv_mod.sha256_file(cfg["dataset"]["list"])
        if got != str(want):
            raise SystemExit(f"object list {cfg['dataset']['list']} has sha256 {got}, the configuration pins {want}")


def load_resumable(pv_path: Path, code_h: str, conv_h: str) -> dict:
    """{key: record} of the records in per_view.json computed by THIS code under THESE conventions.  Everything else is
    never reused: it is copied to per_view.stale_<time>.json next to it and left out of the run."""
    if not pv_path.exists():
        return {}
    old = json.loads(pv_path.read_text())
    same = lambda rec: rec.get("code_hash") == code_h and rec.get("conventions_hash") == conv_h  # noqa: E731
    keep, stale = [r for r in old if same(r)], [r for r in old if not same(r)]
    if stale:
        sp = pv_path.with_name(f"per_view.stale_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.json")
        sp.write_text(json.dumps(stale, indent=1))
        print(f"[resume] {len(stale)} record(s) computed by other code / conventions are NOT reused; set aside in {sp.name}", flush=True)
    print(f"[resume] {len(keep)} record(s) reused (same code + conventions hash)", flush=True)
    return {rec["key"]: rec for rec in keep}


def main(argv=None):
    a = parse_args(argv)
    cfg = effective_settings(a)
    apply_environment(cfg)
    ev = cfg["evaluation"]
    conv, conv_h, code_h, code_files = build_conventions(cfg)
    if a.print_hashes:
        print(json.dumps({"suite_version": conv_mod.SUITE_VERSION, "code_hash": code_h, "conventions_hash": conv_h,
                          "code_files": code_files, "background": ev["background"]}, indent=1))
        return
    if not a.out:
        raise SystemExit("need --out")
    if a.tables_only:
        return tables_only(a, cfg, conv, conv_h, code_h, code_files)
    if not cfg["dataset"]["root"]:
        raise SystemExit("need dataset.root in the configuration (or --dataset-root)")
    check_list(cfg)
    skip = set(ev["skip"])
    out = Path(a.out)
    if (out / "corpus.json").is_file():          # finished results are never overwritten by other code / conventions
        prev = json.loads((out / "corpus.json").read_text()).get("protocol", {})
        if prev.get("code_hash") != code_h or prev.get("conventions_hash") != conv_h:
            raise SystemExit(f"refused: {out} holds finished results of other code / conventions "
                             f"(code {str(prev.get('code_hash'))[:16]}, conv {str(prev.get('conventions_hash'))[:16]}; "
                             f"running code {code_h[:16]}, conv {conv_h[:16]}). Results are never overwritten: choose a new --out.")
    out.mkdir(parents=True, exist_ok=True)
    if a.save_renders:
        (out / "renders").mkdir(exist_ok=True)
    ds = GTDataset(cfg["dataset"]["root"], list_csv=cfg["dataset"]["list"], source_dirs=cfg["dataset"]["source_dirs"])
    rows, frame_maps = [], None
    if a.manifest:
        rows += rows_from_manifest(a.manifest, ds)
    if a.baseline_manifest:
        from metrics import baselines
        frame_maps = baselines.load_frame_maps(ev["frame_maps"])
        rows += baselines.rows_from_baseline_manifest(a.baseline_manifest, ds, frame_maps)
    if not rows:
        raise SystemExit("no product rows (need --manifest and / or --baseline-manifest)")
    rows = attach_gt(rows, ds)
    missing = [r for r in rows if r["missing_gt"]]
    if missing:
        print(f"[warn] {len(missing)} row(s) without ground truth are skipped, e.g. {missing[0]['missing_gt'][:2]}", flush=True)
    rows = [r for r in rows if not r["missing_gt"]]
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    if only:
        rows = [r for r in rows if any(r["object_id"] == o or r["sha256"].startswith(o) for o in only)]
    for r in rows:
        r["key"] = conv_mod.record_key(r, code_h, conv_h)
    print(f"[manifest] {len(rows)} product rows; arms: {sorted({r['arm'] for r in rows})}; code {code_h[:16]} conv {conv_h[:16]}", flush=True)
    (out / "manifest.json").write_text(json.dumps(rows, indent=1))

    # ---- resume: only records of THIS code and THESE conventions ---------------------------------------
    pv_path = out / "per_view.json"
    done = load_resumable(pv_path, code_h, conv_h)

    # ---- exact FP32 before the first CUDA matmul (metrics/precision.py) -----------------------------
    from metrics import precision
    dev = ev["device"]
    precision_state = precision.enforce_exact_fp32()
    precision_state["matmul_probe"] = precision.matmul_probe(dev)
    err = precision_state["matmul_probe"]["max_abs_error"]
    print(f"[precision] {json.dumps(precision_state)}", flush=True)
    if err is not None and err > precision.MAX_PROBE_ERROR:
        raise SystemExit(f"float32 GPU matmul error {err:.2e} > {precision.MAX_PROBE_ERROR:.0e}: TF32 is still active")

    # ---- models (lazy) -----------------------------------------------------------------------------------
    rz = lpips_fn = lpips_vgg = clip_scorer = dino = ulip = uni = None
    from metrics import geometry, normal_maps, perception, projection, vs_reference
    mcfg = cfg["models"]
    if not ({"2d", "perception", "vs_reference"} <= skip):
        from metrics.render import Raster
        rz = Raster(dev)
    if "2d" not in skip and "lpips" not in skip:
        lpips_fn = normal_maps.LPIPSScorer(dev, net=conv["image_metrics"]["lpips_main"]["net"])
        if "lpips_vgg" not in skip:
            lpips_vgg = normal_maps.LPIPSScorer(dev, net=conv["image_metrics"]["lpips_secondary"]["net"])
    if "perception" not in skip and "clip" not in skip:
        from metrics.config import PINS
        clip_scorer = perception.CLIPScorer(dev, PINS["clip"]["repo"], PINS["clip"]["revision"])
    if "perception" not in skip and "ulip2" not in skip:
        ulip = perception.ULIP2Scorer(dev, mcfg)
        print("[ulip2]", "ready" if not ulip.missing else "missing: " + "; ".join(ulip.missing), flush=True)
    if "perception" not in skip and "uni3d" not in skip:
        uni = perception.Uni3DScorer(dev, mcfg)
        print("[uni3d]", "ready" if not uni.missing else "missing: " + "; ".join(uni.missing), flush=True)
    if "vs_reference" not in skip and "dino" not in skip:
        dino = vs_reference.DinoV2(dev, mcfg.get("dinov2_code"), mcfg.get("dinov2_weights"))
    bg_rgb = conv["image_metrics"]["background_rgb"]

    # ---- per (sha, view) groups ----------------------------------------------------------------------------
    groups = {}
    for r in rows:
        groups.setdefault((r["sha256"], r["view"]), []).append(r)
    keys = sorted(groups)
    if a.limit:
        keys = keys[: a.limit]
    records = list(done.values())
    t_start = time.time()
    for gi, (sha, view) in enumerate(keys):
        grp = groups[(sha, view)]
        pending = [r for r in grp if r["key"] not in done]
        if not pending:
            continue
        try:
            scale_json = grp[0]["scale_json"] if Path(grp[0]["scale_json"]).is_file() else None
            vf = ViewFrame.from_files(grp[0]["transforms_json"], view, scale_json=scale_json, sha256=sha)
        except Exception as e:
            print(f"[frame] {sha[:12]} v{view}: {e}", flush=True)
            continue
        cache = {"gt_pts_vox": None, "gt_pts_nb": None, "gt_render": None, "ref": {}}
        for r in pending:
            cond = r["condition"]
            rec = {"key": r["key"], "code_hash": code_h, "conventions_hash": conv_h, "suite_version": conv_mod.SUITE_VERSION,
                   **{k: r.get(k) for k in ("arm", "arm_label", "condition", "method", "object_id", "sha256", "view", "repeat",
                                           "product_glb", "textured_glb", "e2e_s", "category", "dataset_object_id")},
                   "support_source": None, "errors": {}, "frame": vf.summary()}
            t0 = time.time()
            try:
                mesh = geometry.load_product_mesh(r["product_glb"])
                rec["product_faces"], rec["product_vertices"] = int(len(mesh.faces)), int(len(mesh.vertices))
                V = np.asarray(mesh.vertices, np.float64)
                own = None
                if cond == "baseline":
                    from metrics import baselines
                    spec = frame_maps["methods"][r["method"]]
                    to_world_axes = (lambda v, _s=spec: baselines.to_world_axes(_s, vf, v))
                    rec["placement"] = {"kind": "baseline", "family": spec["family"], "depth_sign": spec.get("depth_sign"),
                                        "matrix": spec.get("matrix")}
                else:
                    to_world_axes = (lambda v: projection.product_to_world_axes(vf, v))
                    if cond != "true_camera":
                        own = projection.own_projection_for_row(r)
                        rec["placement"] = {"kind": "own_projection", **own.summary()}
                        # diagnostic only (never used by a metric): the FoV the crop really has under the GT camera
                        import math
                        true_fov = 2.0 * math.atan(own.ndc["sx"] * math.tan(vf.camera_angle_x / 2.0))
                        rec["fov_used_deg"], rec["fov_true_crop_deg"] = math.degrees(own.camera_angle_x), math.degrees(true_fov)
                        rec["fov_error_deg"] = rec["fov_used_deg"] - rec["fov_true_crop_deg"]
                    else:
                        rec["placement"] = {"kind": "true_camera_chain"}
                V_nb, nb_info = projection.normalise_own_bbox(to_world_axes(V))     # GT-world axes, own bbox
                rec["own_bbox"] = nb_info
            except Exception as e:
                rec["errors"]["load"] = repr(e) + "\n" + traceback.format_exc()[-600:]
                records.append(rec)
                continue
            seed_key = f"{sha}|{view}|{r['repeat']}"
            # ---- 3D ------------------------------------------------------------------------------------------
            if "3d" not in skip:
                try:
                    import trimesh
                    seed = geometry.stable_seed(geometry.SEED_SALT_3D, seed_key)
                    if cache["gt_pts_nb"] is None:
                        from metrics.frame import load_mesh_dump
                        Vg, Fg = load_mesh_dump(r["mesh_dump"])
                        Vg_nb, gt_nb_info = projection.normalise_own_bbox(Vg)
                        cache["gt_pts_nb"] = geometry.sample_surface(trimesh.Trimesh(Vg_nb, Fg, process=False), geometry.N_SAMPLES, seed)
                        cache["gt_nb_info"] = gt_nb_info
                    pred_pts = geometry.sample_surface(trimesh.Trimesh(V_nb, np.asarray(mesh.faces), process=False), geometry.N_SAMPLES, seed + 1)
                    g = geometry.chamfer_fscore(pred_pts, cache["gt_pts_nb"])
                    rec.update({"chamfer_nb": g["chamfer"], "f_0.005_nb": g["f_0.005"], "f_0.01_nb": g["f_0.01"], "f_0.02_nb": g["f_0.02"],
                                "precision_0.01_nb": g["precision_0.01"], "recall_0.01_nb": g["recall_0.01"],
                                "gt_own_bbox": cache["gt_nb_info"]})
                    if cond == "true_camera":     # voxel-frame values + 32^3 support: only meaningful with the GT scale
                        if cache["gt_pts_vox"] is None:
                            cache["gt_pts_vox"] = geometry.gt_points_voxel(vf, r["mesh_dump"], seed=seed)
                        gv = geometry.geometry_vs_gt(r["product_glb"], vf, r["mesh_dump"], seed_key=seed_key, gt_pts=cache["gt_pts_vox"])
                        rec.update({"chamfer_vox": gv["chamfer"], "f_0.005_vox": gv["f_0.005"], "f_0.01_vox": gv["f_0.01"],
                                    "f_0.02_vox": gv["f_0.02"], "product_abs_max_voxel": gv["product_abs_max_voxel"],
                                    "gt_abs_max_voxel": gv["gt_abs_max_voxel"]})
                        rec.update(geometry.support_vs_gt(r["product_glb"], r["gt_support_npz"], r.get("support_npz")))
                    else:
                        rec["support_source"] = "n/a (no ground-truth scale: the generation frame is not the voxel frame)"
                except Exception as e:
                    rec["errors"]["3d"] = repr(e) + "\n" + traceback.format_exc()[-600:]
            # ---- 2D ------------------------------------------------------------------------------------------
            if "2d" not in skip:
                try:
                    if cache["gt_render"] is None:
                        cache["gt_render"] = normal_maps.render_gt_normals(rz, vf, r["mesh_dump"], device=dev)
                    if cond == "true_camera":
                        pred = normal_maps.render_product_normals(rz, vf, mesh, device=dev)
                    elif cond == "baseline":
                        pred = projection.render_world_normals(rz, vf, V_nb, mesh.faces, device=dev)
                    else:
                        pred = projection.render_own_projection_normals(rz, own, mesh, device=dev)
                    rec.update(normal_maps.normal_map_metrics(pred, cache["gt_render"], bg_rgb=bg_rgb,
                                                              lpips_main=lpips_fn, lpips_secondary=lpips_vgg))
                    if a.save_renders:
                        png = out / "renders" / f"{r['arm'].replace('@', '-')}__{sha[:12]}_v{view:02d}_r{r['repeat']}.png"
                        normal_maps.save_pair_png(str(png), pred, cache["gt_render"])
                except Exception as e:
                    rec["errors"]["2d"] = repr(e) + "\n" + traceback.format_exc()[-600:]
            # ---- perception ----------------------------------------------------------------------------------
            if "perception" not in skip:
                try:
                    import trimesh
                    placed = vf if cond == "true_camera" else projection.WorldPlacement(vf, to_world_axes)
                    score_mesh = mesh
                    if cond == "baseline":   # same orientation convention as the product frame for the 3D scorers
                        score_mesh = mesh.copy()
                        score_mesh.vertices = projection.world_axes_to_productlike(vf, V_nb)
                    tex_mesh, tex_score_mesh = mesh, score_mesh
                    if cond == "baseline":
                        tex_mesh = tex_score_mesh = None
                        if r.get("textured_glb") and frame_maps["methods"][r["method"]].get("textured"):
                            tex_mesh = mesh if r["textured_glb"] == r["product_glb"] else geometry.load_product_mesh(r["textured_glb"])
                            tex_score_mesh = tex_mesh.copy()
                            Vt_nb, _ = projection.normalise_own_bbox(to_world_axes(np.asarray(tex_mesh.vertices, np.float64)))
                            tex_score_mesh.vertices = projection.world_axes_to_productlike(vf, Vt_nb)
                    if clip_scorer is not None:
                        vg = perception.render_clip_views(rz, placed, mesh, device=dev)
                        cs = perception.clip_scores(clip_scorer, vg, r["input_png"])
                        textured = bool(vg["textured"])
                        if tex_mesh is not None and tex_mesh is not mesh:
                            vt = perception.render_clip_views(rz, projection.WorldPlacement(vf, to_world_axes), tex_mesh, device=dev)
                            ct = perception.clip_scores(clip_scorer, vt, r["input_png"])
                            textured = bool(vt["textured"])
                            for k in ("clip", "clip_per_view", "clip_input_view"):
                                cs[k] = ct[k]
                        if tex_mesh is None or not textured:          # CLIP is defined for textured methods only
                            for k in ("clip", "clip_per_view", "clip_input_view"):
                                cs[k] = None
                        cs["clip_textured"] = bool(textured and tex_mesh is not None)
                        rec.update(cs)
                    pc_seed = geometry.stable_seed(geometry.SEED_SALT_POINTS, seed_key)
                    geo_only = trimesh.Trimesh(np.asarray(score_mesh.vertices), np.asarray(score_mesh.faces), process=False)
                    for scorer in (ulip, uni):
                        if scorer is None:
                            continue
                        rec.update(scorer.score(tex_score_mesh if tex_score_mesh is not None else geo_only, r["input_png"], pc_seed))
                        gsc = scorer.score(geo_only, r["input_png"], pc_seed)
                        rec[f"{scorer.name}_geo"] = gsc.get(scorer.name)
                except Exception as e:
                    rec["errors"]["perception"] = repr(e) + "\n" + traceback.format_exc()[-600:]
            # ---- agreement with the reference arm of the SAME condition --------------------------------------
            ref_arm = ev["reference_arm"] + r["arm"][len(base_arm(r["arm"])):]        # same condition suffix as this row
            ref = next((x for x in grp if x["arm"] == ref_arm and x["repeat"] == 0 and x["condition"] == cond), None)
            if "vs_reference" not in skip and cond != "baseline" and ref is not None and ref["key"] != r["key"]:
                try:
                    rc = cache["ref"].setdefault(ref_arm, {"mesh": None, "pts": None, "dino": None})
                    if rc["mesh"] is None:
                        rc["mesh"] = geometry.load_product_mesh(ref["product_glb"])
                    if rc["pts"] is None:
                        rc["pts"], _ = geometry.product_points_voxel(
                            ref["product_glb"], seed=geometry.stable_seed(geometry.SEED_SALT_REFERENCE, seed_key) + 2)
                    g = geometry.geometry_vs_product(r["product_glb"], ref["product_glb"], seed_key=seed_key, ref_pts=rc["pts"])
                    rec["cd_to_ref"], rec["f_0.01_to_ref"], rec["f_0.005_to_ref"] = g["chamfer"], g["f_0.01"], g["f_0.005"]
                    rec["reference_arm"] = ref_arm
                    if dino is not None:
                        if rc["dino"] is None:
                            rm = rc["mesh"]
                            rc["dino"] = dino.features(vs_reference.render_dino_views(rz, np.asarray(rm.vertices), np.asarray(rm.faces), dev))
                        fc = dino.features(vs_reference.render_dino_views(rz, np.asarray(mesh.vertices), np.asarray(mesh.faces), dev))
                        rec.update(vs_reference.dino_distances(fc, rc["dino"]))
                except Exception as e:
                    rec["errors"]["vs_reference"] = repr(e) + "\n" + traceback.format_exc()[-600:]
            rec["wall_s"] = round(time.time() - t0, 2)
            records.append(rec)
            done[rec["key"]] = rec
        pv_path.write_text(json.dumps(records, indent=1))
        print(f"[{gi + 1}/{len(keys)}] {sha[:12]} v{view} rows={len(pending)} elapsed {(time.time() - t_start) / 60:.1f} min", flush=True)

    finalize(records, a, cfg, out, conv, conv_h, code_h, code_files, precision_state)


# ------------------------------------------------------------------------------------------------ aggregation
def finalize(records, a, cfg, out: Path, conv, conv_h, code_h, code_files, precision_state=None):
    ev = cfg["evaluation"]
    ref_base = ev["reference_arm"]
    metric_keys = METRICS_2D + METRICS_3D + METRICS_PERC + METRICS_REF + METRICS_DIAG + ["e2e_s"]
    by_arm = {}
    for rec in records:
        by_arm.setdefault(rec["arm"], []).append(rec)
    arm_cond = {arm: conv_mod.canonical_condition(recs[0].get("condition")) for arm, recs in by_arm.items()}
    # order: Pixal3D-family arms (reference first) before baselines; inside an arm the conditions in CONDITION_ORDER
    arms = sorted(by_arm, key=lambda x: (1 if arm_cond[x] == "baseline" else 0, 0 if base_arm(x) == ref_base else 1, base_arm(x),
                                         CONDITION_ORDER.index(arm_cond[x]), x))
    per_object = {arm: stats.views_to_objects(by_arm[arm], object_key="sha256", metric_keys=metric_keys) for arm in arms}
    (out / "per_object.json").write_text(json.dumps(per_object, indent=1))
    corpus = {"reference": {}, "summary": {}, "paired": {}}
    for cond in dict.fromkeys(arm_cond[x] for x in arms):
        sub = {arm: per_object[arm] for arm in arms if arm_cond[arm] == cond}
        ref = next((arm for arm in sub if base_arm(arm) == ref_base), None) if cond != "baseline" else None
        c = stats.compare_arms(sub, ref or "__none__", metric_keys, HIGHER_BETTER, n_boot=ev["n_boot"])
        corpus["summary"].update(c["summary"])
        if ref is not None:
            corpus["paired"].update(c["paired"])
        corpus["reference"][cond] = ref
    # ---- true_camera vs other condition per arm (paired per object: other - true_camera) ------------------
    gap = {}
    for arm in arms:
        if arm_cond[arm] in ("true_camera", "baseline"):
            continue
        tc = base_arm(arm) + conv_mod.CONDITION_SUFFIX["true_camera"]
        if tc not in per_object:
            continue
        gap[arm] = {"true_camera_arm": tc, "metrics": {}}
        for m in GAP_METRICS:
            cand = {o: v.get(m) for o, v in per_object[arm].items()}
            refv = {o: v.get(m) for o, v in per_object[tc].items()}
            gap[arm]["metrics"][m] = {"true_camera_mean": corpus["summary"][tc].get(m, {}).get("mean"),
                                      "other_mean": corpus["summary"][arm].get(m, {}).get("mean"),
                                      "bootstrap": stats.paired_bootstrap(cand, refv, n_boot=ev["n_boot"]),
                                      "sign_test": stats.sign_test(cand, refv)}
    corpus["condition_gap"] = gap
    corpus["arm_condition"] = arm_cond
    corpus["n_products"] = {arm: len(by_arm[arm]) for arm in arms}
    corpus["n_errors"] = {arm: sum(1 for r in by_arm[arm] if r.get("errors")) for arm in arms}
    corpus["support_source"] = {arm: sorted({str(r.get("support_source")) for r in by_arm[arm]}) for arm in arms}
    corpus["speed"] = {arm: stats.summarize([r.get("e2e_s") for r in by_arm[arm]]) for arm in arms}
    corpus["faces"] = {arm: stats.summarize([r.get("product_faces") for r in by_arm[arm]]) for arm in arms}
    corpus["precision"] = precision_state
    # flips / noise band (cd_to_ref above the noise band of a second reference sample)
    flips, thr = {}, None
    noise = {}
    if a.noise_arm:
        noise = next((per_object[x] for x in arms if base_arm(x) == a.noise_arm), {})
    if noise:
        from metrics.vs_reference import flip_threshold_from_noise_band
        thr = flip_threshold_from_noise_band({o: v.get("cd_to_ref") for o, v in noise.items()}, a.flip_quantile)
    for arm in arms:
        if base_arm(arm) == ref_base or arm_cond[arm] == "baseline":
            continue
        vals = {o: v.get("cd_to_ref") for o, v in per_object[arm].items()}
        flips[arm] = stats.count_flips(vals, thr) if thr is not None else {"n_flips": None, "note": "no noise band (--noise-arm)", "n": len(vals)}
    corpus["flips"] = {"threshold_cd_to_ref": thr, "quantile": a.flip_quantile, "per_arm": flips}
    corpus["protocol"] = {"suite_version": conv_mod.SUITE_VERSION, "reference_arm": ref_base, "n_boot": ev["n_boot"],
                          "seed": stats.BOOT_SEED, "title": a.title, "code_hash": code_h, "conventions_hash": conv_h,
                          "code_files": code_files, "conventions": conv, "table1_region": ev["table1_region"],
                          "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    corpus["tables"] = tables.write_tables(out, corpus, arms, ev["table1_region"])
    (out / "corpus.json").write_text(json.dumps(corpus, indent=1))
    print(f"[done] {out}/corpus.json, {out}/tables/ (table1_region={ev['table1_region']})", flush=True)


def tables_only(a, cfg, conv, conv_h, code_h, code_files):
    """re-render corpus.json / per_object.json / tables of an existing --out (records of THIS code and conventions)."""
    out = Path(a.out)
    records = list(load_resumable(out / "per_view.json", code_h, conv_h).values())
    if not records:
        raise SystemExit(f"--tables-only: no records of code {code_h[:16]} / conventions {conv_h[:16]} in {out}/per_view.json")
    prev = json.loads((out / "corpus.json").read_text()) if (out / "corpus.json").is_file() else {}
    finalize(records, a, cfg, out, conv, conv_h, code_h, code_files, prev.get("precision"))


if __name__ == "__main__":
    main()
