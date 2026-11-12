#!/usr/bin/env python3
"""Compare a run of this package with a run of the measurement code, input by input, with the parity
criteria of docs/PARITY.md (as amended before the parity run).

    python scripts/parity_check.py --reference REF_DIR --reference-repeat REF2_DIR --candidate CAND_DIR \
        [--candidate-glb-dir DIR] [--list inputs.jsonl] [--exclude-first 3] [--out report.json]

Each run directory holds, per input id (``scripts/infer.py`` / ``scripts/time_e2e.py --dump-dir`` write
this layout; files that a run does not have are skipped):

    <dir>/<id>/support.npy   int32 [M, 4]   structure support coordinates (b, x, y, z)
    <dir>/<id>/coarse.npz    coords int32 [N, 4], feats float32 [N, 32]   normalised coarse latent
    <dir>/<id>/fine.npz      coords int32 [N, 4], feats float32 [N, 32]   normalised fine latent
    <dir>/<id>/timing.json   {"e2e_s": seconds or [seconds, ...],
                              "sha256": {"fine_coords", "fine_latent", "texture_latent"}}
    GLB: <glb-dir>/<id>.glb, <dir>/<id>/output.glb or <dir>/<id>.glb

The sha256 values are those of the raw bytes of the decoder inputs: the de-normalised fine latent
(float32 [N, 32]), its coordinates (int32 [N, 4]) and the texture latent (float32 [N, 32]).

Criteria (a check is ``null`` when the data for it is missing):

* support: identical coordinate set for every input;
* latents: the three decoder-input sha256 values are identical for every input;
* coarse / fine latents: relative Frobenius error ||c - r|| / ||r|| <= 1e-3 where both runs have the
  arrays (rows matched by coordinates); identical fine-latent digests count as error 0;
* faces: face count of the GLB within +-1 %;
* texture: the surface-sampled base-colour PSNR between candidate and reference is at least the
  reference's own run-to-run value (reference against ``--reference-repeat``, same input) minus 1 dB.
  Surface-sampled: area-uniform points on one mesh, closest points on the other, both directions;
  independent of the atlas layout. The pixel-by-pixel PSNR of the texture images is reported for
  information only (atlas layouts differ between runs);
* e2e: median end-to-end time over the inputs after the first ``--exclude-first`` (list order) within
  +-5 % of the reference median.

Exit code 0 when no check fails, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CRITERIA = {"latent_rel_fro": 1e-3, "face_rel": 0.01, "texture_psnr_margin_db": 1.0, "e2e_rel": 0.05}
DIGESTS = ("fine_coords", "fine_latent", "texture_latent")


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def glb_path(run_dir: Optional[Path], glb_dir: Optional[Path], item: str) -> Optional[Path]:
    candidates = [glb_dir / f"{item}.glb"] if glb_dir else []
    if run_dir is not None:
        candidates += [run_dir / item / "output.glb", run_dir / f"{item}.glb"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_npz(path: Path):
    if not path.is_file():
        return None
    data = np.load(path)
    return data["coords"].astype(np.int64), data["feats"].astype(np.float64)


def load_e2e(run_dir: Path, item: str) -> Optional[float]:
    p = run_dir / item / "timing.json"
    if not p.is_file():
        return None
    value = json.loads(p.read_text()).get("e2e_s")
    if isinstance(value, list):
        return float(np.median(value)) if value else None
    return None if value is None else float(value)


def load_digests(run_dir: Path, item: str) -> Dict[str, str]:
    p = run_dir / item / "timing.json"
    return json.loads(p.read_text()).get("sha256") or {} if p.is_file() else {}


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float64)
    return mesh, texture


# ---------------------------------------------------------------------------
# quantities
# ---------------------------------------------------------------------------
def support_identical(ref: np.ndarray, cand: np.ndarray) -> Dict[str, Any]:
    r = {tuple(row) for row in ref.reshape(-1, 4).tolist()}
    c = {tuple(row) for row in cand.reshape(-1, 4).tolist()}
    return {"identical": r == c, "reference": len(r), "candidate": len(c), "common": len(r & c)}


def latent_rel_fro(ref, cand) -> Dict[str, Any]:
    (rc, rf), (cc, cf) = ref, cand
    if rc.shape == cc.shape and np.array_equal(rc, cc):
        a, b, matched = cf, rf, rc.shape[0]
    else:
        key = lambda c: c[:, 0] * (1 << 48) + c[:, 1] * (1 << 32) + c[:, 2] * (1 << 16) + c[:, 3]  # noqa: E731
        rk, ck = key(rc), key(cc)
        common, ri, ci = np.intersect1d(rk, ck, return_indices=True)
        a, b, matched = cf[ci], rf[ri], common.shape[0]
    denom = float(np.linalg.norm(b))
    rel = float(np.linalg.norm(a - b)) / denom if denom > 0 else float("inf")
    return {"rel_fro": rel, "rows_reference": int(rc.shape[0]), "rows_candidate": int(cc.shape[0]),
            "rows_matched": int(matched)}


def psnr(mse: float, peak: float = 255.0) -> float:
    return float("inf") if mse <= 0 else 10.0 * math.log10(peak * peak / mse)


def image_psnr(ref_tex: np.ndarray, cand_tex: np.ndarray) -> Optional[float]:
    if ref_tex.shape != cand_tex.shape:
        return None
    return psnr(float(np.mean((ref_tex - cand_tex) ** 2)))


def _bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample an (H, W, 3) texture at glTF-loaded UVs (v up, as trimesh returns them)."""
    h, w = texture.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip((1.0 - uv[:, 1]) * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    top = texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx
    bottom = texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _colours_at(mesh, texture, face_id: np.ndarray, bary: np.ndarray) -> np.ndarray:
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    tri_uv = uv[np.asarray(mesh.faces)[face_id]]
    return _bilinear(texture, np.einsum("nk,nkd->nd", bary, tri_uv))


def _closest(mesh, points: np.ndarray):
    """(face id, barycentric) of the closest surface point; GPU BVH when available."""
    try:
        import torch
        import cumesh

        if not torch.cuda.is_available():
            raise ImportError
        bvh = cumesh.cuBVH(torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32).cuda(),
                           torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32).cuda())
        _, face_id, uvw = bvh.unsigned_distance(torch.as_tensor(points, dtype=torch.float32).cuda(), return_uvw=True)
        return face_id.long().cpu().numpy(), uvw.double().cpu().numpy()
    except ImportError:
        import trimesh

        closest, _, face_id = trimesh.proximity.closest_point(mesh, points)
        tri = np.asarray(mesh.triangles)[face_id]
        return face_id, trimesh.triangles.points_to_barycentric(tri, closest)


def surface_psnr(ref, cand, samples: int = 200_000, seed: int = 0) -> float:
    import trimesh

    mse = []
    for (src, src_tex), (dst, dst_tex) in ((ref, cand), (cand, ref)):
        points, face_id = trimesh.sample.sample_surface(src, samples, seed=seed)
        bary = trimesh.triangles.points_to_barycentric(np.asarray(src.triangles)[face_id], points)
        a = _colours_at(src, src_tex, face_id, bary)
        dst_face, dst_bary = _closest(dst, points)
        b = _colours_at(dst, dst_tex, dst_face, dst_bary)
        mse.append(float(np.mean((a - b) ** 2)))
    return psnr(float(np.mean(mse)))


# ---------------------------------------------------------------------------
def compare(ref_dir: Path, cand_dir: Path, items: List[str], ref_glb: Optional[Path], cand_glb: Optional[Path],
            exclude_first: int, samples: int, ref2_dir: Optional[Path] = None,
            ref2_glb: Optional[Path] = None) -> Dict[str, Any]:
    rows = []
    for item in items:
        row: Dict[str, Any] = {"id": item}
        rs, cs = ref_dir / item / "support.npy", cand_dir / item / "support.npy"
        if rs.is_file() and cs.is_file():
            row["support"] = support_identical(np.load(rs), np.load(cs))
        rd, cd = load_digests(ref_dir, item), load_digests(cand_dir, item)
        if rd and cd:
            row["digests_equal"] = {k: rd.get(k) == cd.get(k) for k in DIGESTS if k in rd or k in cd}
        for name in ("coarse", "fine"):
            r, c = load_npz(ref_dir / item / f"{name}.npz"), load_npz(cand_dir / item / f"{name}.npz")
            if r is not None and c is not None:
                row[name] = latent_rel_fro(r, c)
            elif name == "fine" and row.get("digests_equal", {}).get("fine_latent"):
                row[name] = {"rel_fro": 0.0, "from": "identical sha256"}
        rg, cg = glb_path(ref_dir, ref_glb, item), glb_path(cand_dir, cand_glb, item)
        if rg and cg:
            ref_mesh, cand_mesh = load_mesh(rg), load_mesh(cg)
            fr, fc = len(ref_mesh[0].faces), len(cand_mesh[0].faces)
            row["faces"] = {"reference": fr, "candidate": fc, "rel": (fc - fr) / fr}
            texture = {"image": image_psnr(ref_mesh[1], cand_mesh[1]), "surface": surface_psnr(ref_mesh, cand_mesh, samples)}
            r2 = glb_path(ref2_dir, ref2_glb, item) if (ref2_dir or ref2_glb) else None
            if r2:
                texture["surface_reference_run_to_run"] = surface_psnr(ref_mesh, load_mesh(r2), samples)
                texture["pass"] = texture["surface"] >= texture["surface_reference_run_to_run"] - CRITERIA["texture_psnr_margin_db"]
            row["texture_psnr_db"] = texture
        row["e2e_s"] = {"reference": load_e2e(ref_dir, item), "candidate": load_e2e(cand_dir, item)}
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)

    def all_of(key, pred):
        values = [pred(r[key]) for r in rows if key in r and r[key] is not None]
        return (all(values) if values else None), len(values)

    timed = [r for r in rows[exclude_first:] if None not in r["e2e_s"].values()]
    e2e = None
    if timed:
        mr = float(np.median([r["e2e_s"]["reference"] for r in timed]))
        mc = float(np.median([r["e2e_s"]["candidate"] for r in timed]))
        e2e = {"n": len(timed), "median_reference_s": mr, "median_candidate_s": mc, "rel": (mc - mr) / mr,
               "pass": abs(mc - mr) / mr <= CRITERIA["e2e_rel"]}
    textures = [r["texture_psnr_db"] for r in rows if "texture_psnr_db" in r]
    summary = {
        "support_identical": all_of("support", lambda v: v["identical"]),
        "latents_identical": all_of("digests_equal", lambda v: bool(v) and all(v.values())),
        "coarse_rel_fro_max": max([r["coarse"]["rel_fro"] for r in rows if "coarse" in r], default=None),
        "fine_rel_fro_max": max([r["fine"]["rel_fro"] for r in rows if "fine" in r], default=None),
        "face_rel_max_abs": max([abs(r["faces"]["rel"]) for r in rows if "faces" in r], default=None),
        "texture_surface_min_db": min([t["surface"] for t in textures], default=None),
        "texture_surface_min_margin_db": min([t["surface"] - t["surface_reference_run_to_run"] for t in textures
                                              if "surface_reference_run_to_run" in t], default=None),
        "texture_image_min_db_informational": min([t["image"] for t in textures if t["image"] is not None], default=None),
        "e2e": e2e,
    }
    texture_checks = [t["pass"] for t in textures if "pass" in t]
    checks = {
        "support": summary["support_identical"][0],
        "latents": summary["latents_identical"][0],
        "coarse": None if summary["coarse_rel_fro_max"] is None else summary["coarse_rel_fro_max"] <= CRITERIA["latent_rel_fro"],
        "fine": None if summary["fine_rel_fro_max"] is None else summary["fine_rel_fro_max"] <= CRITERIA["latent_rel_fro"],
        "faces": None if summary["face_rel_max_abs"] is None else summary["face_rel_max_abs"] <= CRITERIA["face_rel"],
        "texture": (all(texture_checks) if texture_checks else None),
        "e2e": None if e2e is None else e2e["pass"],
    }
    return {"criteria": CRITERIA, "summary": summary, "checks": checks, "inputs": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--reference-glb-dir", type=Path, default=None)
    ap.add_argument("--candidate-glb-dir", type=Path, default=None)
    ap.add_argument("--reference-repeat", type=Path, default=None,
                    help="a second run of the reference on the same inputs (run-to-run texture floor)")
    ap.add_argument("--reference-repeat-glb-dir", type=Path, default=None)
    ap.add_argument("--list", type=Path, default=None, help="JSON lines with 'id' (defines the order)")
    ap.add_argument("--exclude-first", type=int, default=3, help="inputs excluded from the time median (warm-up)")
    ap.add_argument("--samples", type=int, default=200_000, help="surface samples per direction for the texture PSNR")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.list:
        items = [str(json.loads(line).get("id") or json.loads(line).get("object_id"))
                 for line in args.list.read_text().splitlines() if line.strip()]
    else:
        items = sorted(p.name for p in args.reference.iterdir() if p.is_dir())
    report = compare(args.reference, args.candidate, items, args.reference_glb_dir, args.candidate_glb_dir,
                     args.exclude_first, args.samples, args.reference_repeat, args.reference_repeat_glb_dir)
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        args.out.write_text(text)
    print(json.dumps({"summary": report["summary"], "checks": report["checks"]}, indent=1, default=str))
    return 0 if all(v is not False for v in report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
