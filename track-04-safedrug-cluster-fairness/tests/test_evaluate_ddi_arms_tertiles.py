import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_ddi_arms_tertiles as T  # noqa: E402


def test_tertile_map_is_8_8_8_and_ordered_by_gt_ddi():
    rows = [{"cluster_label": f"{i:02d}", "ddi_rate_mean": 0.05 + 0.002 * i, "n_visits": 100} for i in range(24)]
    rows.append({"cluster_label": "NO_CC", "ddi_rate_mean": 0.08, "n_visits": 50})
    tmap, table = T.tertile_map({"cluster_table": rows})
    counts = pd.Series(tmap).value_counts().to_dict()
    assert counts == {"HIGH": 8, "MID": 8, "LOW": 8}
    assert tmap["23"] == "HIGH" and tmap["00"] == "LOW" and tmap["12"] == "MID"
    assert "NO_CC" not in tmap
    assert list(table["rank"]) == list(range(1, 25))


def test_contrast_permutation_detects_planted_high_minus_low():
    rng = np.random.default_rng(0)
    n_pat = 600
    pat_group = rng.integers(0, 3, size=n_pat)          # 0=HIGH 1=MID 2=LOW
    subj = np.repeat(np.arange(n_pat), 3)
    grp = np.repeat(pat_group, 3)
    effect = np.array([-0.03, -0.015, 0.0])
    vals = effect[grp] + rng.normal(scale=0.05, size=len(grp))
    r = T.contrast_permutation(grp, subj, vals, 3, hi=0, lo=2, n_perm=400, seed=0)
    assert r["observed"] < -0.02 and r["p_one_sided_negative"] < 0.01
    # calibration: noise-only draws should rarely reject
    rej = 0
    for d in range(20):
        noise = rng.normal(scale=0.05, size=len(grp))
        rej += T.contrast_permutation(grp, subj, noise, 3, 0, 2, n_perm=200, seed=d)["p_one_sided_negative"] < 0.05
    assert rej <= 4


def test_contrast_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"SUBJECT_ID": np.repeat(np.arange(120), 2),
                       "tertile": rng.choice(["HIGH", "MID", "LOW"], size=240),
                       "d_jaccard": rng.normal(scale=0.05, size=240)})
    b = T.contrast_bootstrap(df, "d_jaccard", n_boot=200)
    assert b["ci_low"] <= b["mean"] <= b["ci_high"]
