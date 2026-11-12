"""make_table.py on synthetic per-object results: values, bold rule, significance markers, E2E column,
the optional additional row."""
import json

import numpy as np
import pytest

import make_table


def _po(n, shift, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        base = rng.normal(size=1)[0] * 0.01
        out[f"{i:064x}"] = {"mask_iou": 0.96 + base, "psnr_ov": 19.0 + shift + base, "ssim_ov": 0.61 + shift / 10,
                            "lpips_ov": 0.053 - shift / 100, "psnr_full": 24.8 + shift, "ssim_full": 0.9 + shift / 100,
                            "lpips_full": 0.06 - shift / 100, "ang_mean": 17.2 - shift, "ang_median": 11.7 - shift,
                            "ang_mean_b": 19.4 - shift, "acc_11.25": 0.53 + shift / 10, "acc_22.5": 0.77 + shift / 10,
                            "acc_30.0": 0.85 + shift / 10, "n_views": 1}
    return out


def _per_object(tmp_path):
    po = {"ref@official": _po(40, 0.0, 0), "ours@official": _po(40, 0.5, 0), "ours_200k@official": _po(40, 0.6, 0)}
    for o in po["ours@official"].values():
        o["mask_iou"] -= 0.003                                         # the reference is better on IoU
    p = tmp_path / "per_object.json"
    p.write_text(json.dumps(po))
    return p


ROWS = ["--row", "Pixal3D (official)=ref@official", "--row", "Ours=ours@official",
        "--e2e", "Pixal3D (official)=133.019", "--e2e", "Ours=4.047"]


def test_table_rows(tmp_path):
    p = _per_object(tmp_path)
    make_table.main(["--per-object", str(p), *ROWS, "--out-dir", str(tmp_path / "t")])
    main = (tmp_path / "t/table1.tex").read_text().splitlines()
    ref_row = next(l for l in main if l.startswith("Pixal3D (official) &"))
    ours_row = next(l for l in main if l.startswith(r"\textbf{Ours} &"))
    assert ref_row.split(" & ")[1].startswith(r"\textbf{")                  # IoU: reference better -> bold
    assert ours_row.split(" & ")[2] == r"\textbf{19.50}"                    # PSNR overlap: candidate better -> bold
    assert ref_row.endswith("& 133.0 \\\\") and ours_row.endswith(r"& \textbf{4.05} \\")
    assert main[main.index(ours_row) - 1] == r"\rowcolor{rowgray}"
    assert main[main.index(ours_row) + 1] == r"\bottomrule"                # no additional row by default
    assert sum("not reported" in l for l in main) == 3
    full = (tmp_path / "t/table1_full_region.tex").read_text()
    assert r"25.30$^{\ast}$" in full and r"\textbf{2" not in full           # markers, no bold, full-region PSNR
    rows = (tmp_path / "t/table1_values.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * 13


def test_additional_row_only_when_passed(tmp_path):
    p = _per_object(tmp_path)
    make_table.main(["--per-object", str(p), *ROWS, "--extra-row", "Ours, 200k export=ours_200k@official",
                     "--e2e", "Ours, 200k export=3.203", "--out-dir", str(tmp_path / "t")])
    main = (tmp_path / "t/table1.tex").read_text().splitlines()
    extra = next(l for l in main if l.startswith(r"\textit{Ours, 200k export} &"))
    assert extra.endswith("& 3.20 \\\\") and r"\textbf" not in extra and "ast" not in extra
    assert main[main.index(extra) - 1] == r"\midrule"
    rows = (tmp_path / "t/table1_values.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 * 13
    with pytest.raises(SystemExit):                                         # a third --row is refused
        make_table.main(["--per-object", str(p), *ROWS, "--row", "Ours, 200k export=ours_200k@official",
                         "--out-dir", str(tmp_path / "u")])
