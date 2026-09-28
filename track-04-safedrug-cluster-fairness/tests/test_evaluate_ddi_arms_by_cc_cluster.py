"""Tests for scripts/evaluate_ddi_arms_by_cc_cluster.py (stage 1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_ddi_arms_by_cc_cluster as E  # noqa: E402


def test_visit_metrics_hand_computed():
    A = np.zeros((5, 5), dtype=int)
    A[0, 1] = A[1, 0] = 1
    A[2, 3] = A[3, 2] = 1
    gt = np.array([1, 1, 0, 1, 0], dtype=np.uint8)      # {0,1,3}
    pred = np.array([1, 0, 1, 1, 0], dtype=np.uint8)    # {0,2,3}
    m = E.visit_metrics(gt, pred, A)
    assert m["jaccard"] == pytest.approx(2 / 4)          # inter {0,3}, union {0,1,2,3}
    assert m["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    assert m["n_med_pred"] == 3 and m["n_med_gt"] == 3
    assert m["dd_cnt_pred"] == 1 and m["ddi_pred"] == pytest.approx(1 / 3)   # (2,3) among 3 pairs
    assert m["dd_cnt_true"] == 1 and m["ddi_true"] == pytest.approx(1 / 3)   # (0,1) among 3 pairs
    empty = E.visit_metrics(gt, np.zeros(5, dtype=np.uint8), A)
    assert empty["jaccard"] == 0.0 and empty["f1"] == 0.0 and np.isnan(empty["ddi_pred"])


def _arm(arm, seeds, hadm_gt, jac):
    rows = []
    for s in seeds:
        for h, (subj, n_gt, dd) in hadm_gt.items():
            rows.append({"HADM_ID": h, "SUBJECT_ID": subj, "seed": s, "arm": arm, "split": "test",
                         "n_med_gt": n_gt, "dd_cnt_true": dd, "jaccard": jac[h] + 0.01 * s, "f1": jac[h],
                         "n_med_pred": 10, "ddi_pred": 0.05, "dd_cnt_pred": 2, "ddi_true": 0.1})
    return pd.DataFrame(rows)


def test_check_alignment_and_seed_average():
    gt = {100: (1, 5, 1), 200: (1, 6, 2), 300: (2, 7, 0)}
    on = _arm("ON", [0, 1], gt, {100: 0.5, 200: 0.6, 300: 0.7})
    off = _arm("OFF", [0, 1], gt, {100: 0.55, 200: 0.6, 300: 0.65})
    st = E.check_alignment(on, off)
    assert st["n_visits"] == 3 and st["seeds_per_arm"]["ON"] == {0: 3, 1: 3}
    avg = E.seed_average(on, ["jaccard"]).set_index("HADM_ID")
    assert avg.loc[100, "jaccard"] == pytest.approx(0.505) and (avg["n_seeds"] == 2).all()
    bad = off.copy(); bad.loc[bad["HADM_ID"] == 300, "n_med_gt"] = 99
    with pytest.raises(AssertionError):
        E.check_alignment(on, bad)


def test_spearman_permutation_detects_planted_negative_association():
    rng = np.random.default_rng(0)
    n_groups, n_pat = 8, 400
    patient_group = rng.integers(0, n_groups, size=n_pat)
    premise = np.arange(n_groups, dtype=float)            # cluster "GT DDI" increases with id
    subj, grp, vals = [], [], []
    for p in range(n_pat):
        for _ in range(3):                                  # 3 visits per patient
            subj.append(p); grp.append(patient_group[p])
            vals.append(-0.02 * patient_group[p] + rng.normal(scale=0.05))   # loss grows with premise
    grp, subj = np.array(grp), np.array(subj)
    res = E.spearman_permutation(grp, subj, np.array(vals), premise, n_groups, n_perm=300, seed=0)
    assert res["rho"] < -0.8 and res["p_one_sided_negative"] < 0.05
    # Calibration under the null: with 8 groups a single noise draw can land on
    # rho = -0.86 by chance (true tail probability ~0.5%), so check the rate of
    # p < 0.05 over 30 independent noise draws instead of one draw. Expected
    # 1.5 rejections; P(> 6) under Binomial(30, 0.05) is < 0.001.
    rejections = 0
    for draw in range(30):
        null_vals = rng.normal(scale=0.05, size=len(vals))
        r0 = E.spearman_permutation(grp, subj, null_vals, premise, n_groups, n_perm=200, seed=draw)
        rejections += r0["p_one_sided_negative"] < 0.05
    assert rejections <= 6


def test_patient_bootstrap_mean_contains_point_estimate():
    vals = np.random.default_rng(1).normal(size=200)
    pats = np.repeat(np.arange(50), 4)
    b = E.patient_bootstrap_mean(vals, pats, n_boot=100)
    assert b["ci_low"] <= b["mean"] <= b["ci_high"] and b["n_patients"] == 50


def test_run_analysis_plumbing_on_real_on_dumps_self_comparison():
    """ON vs ON: every delta is exactly 0; checks the join, cluster table and
    that visit counts add up. Skips if the ON dumps or stage-0 manifest are absent."""
    if not (E.STAGE0.exists() and all((d / "per_visit_predictions.npz").exists() for d in E.arm_dirs("ON").values())):
        pytest.skip("real inputs not available")
    import dill
    ddi_A = np.asarray(dill.load(open(E.DDI_A, "rb")))
    on, _ = E.load_arm("ON", ["test"], ddi_A)
    cc = E.load_cc_labels()
    stage0 = json.loads(E.STAGE0.read_text(encoding="utf-8"))
    res = E.run_analysis(on, on.copy(), cc, stage0, n_perm=20, seed=0)
    assert res["n_visits_primary"] == on["HADM_ID"].nunique()
    assert sum(r["n_visits"] for r in res["cluster_table"]) == res["n_visits_primary"]
    assert all(r["d_jaccard"] == 0.0 for r in res["cluster_table"])
    assert res["verdict"]["H1"] is False and res["verdict"]["supported"] is False
