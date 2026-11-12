"""Aggregation and paired statistics."""
import math

import numpy as np
import pytest

from metrics import stats


def test_views_to_objects_and_summary():
    rows = [{"sha256": "a", "m": 1.0}, {"sha256": "a", "m": 3.0}, {"sha256": "b", "m": None}, {"sha256": "b", "m": 5.0}]
    po = stats.views_to_objects(rows, object_key="sha256", metric_keys=["m"])
    assert po == {"a": {"n_views": 2, "m": 2.0}, "b": {"n_views": 2, "m": 5.0}}
    s = stats.summarize([1.0, 2.0, None, float("nan"), 3.0])
    assert (s["n"], s["mean"], s["median"]) == (3, 2.0, 2.0)


def test_paired_bootstrap_constant_shift():
    ref = {str(i): float(i) for i in range(50)}
    cand = {k: v + 0.5 for k, v in ref.items()}
    b = stats.paired_bootstrap(cand, ref, n_boot=2000)
    assert b["n"] == 50 and b["mean_diff"] == pytest.approx(0.5)
    assert b["ci95_lo"] == pytest.approx(0.5) and b["ci95_hi"] == pytest.approx(0.5) and b["significant"]
    assert stats.paired_bootstrap({"x": 1.0}, {"y": 1.0}) is None       # no common object


def test_sign_test_and_holm():
    ref = {str(i): 0.0 for i in range(12)}
    cand = {str(i): (1.0 if i < 10 else (-1.0 if i < 11 else 0.0)) for i in range(12)}
    s = stats.sign_test(cand, ref)
    assert (s["n_pos"], s["n_neg"], s["n_tie"]) == (10, 1, 1)
    p = sum(math.comb(11, k) for k in (0, 1, 10, 11)) / 2 ** 11
    assert s["p_value"] == pytest.approx(p)
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_compare_arms_marks_direction():
    ref = {str(i): {"lpips": 0.2, "psnr": 20.0} for i in range(30)}
    cand = {str(i): {"lpips": 0.1, "psnr": 21.0} for i in range(30)}
    c = stats.compare_arms({"ref": ref, "cand": cand}, "ref", ["lpips", "psnr"], {"lpips": False, "psnr": True}, n_boot=500)
    assert c["paired"]["cand"]["lpips"]["candidate_better_on_mean"] is True
    assert c["paired"]["cand"]["psnr"]["candidate_better_on_mean"] is True
    assert c["summary"]["cand"]["psnr"]["mean"] == 21.0
    assert np.isclose(c["paired"]["cand"]["psnr"]["sign_test"]["p_holm"], min(1.0, 2 * 2 / 2 ** 30))
