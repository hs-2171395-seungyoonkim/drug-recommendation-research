import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import (
    adjusted_group_means,
    cluster_robust_ols,
    gap_statistics,
    largest_remainder,
    permutation_p,
)


def test_cluster_robust_ols_hand_computed():
    # 2 clusters, 4 observations: cluster 0 = (x=0,y=0),(x=1,y=2);
    # cluster 1 = (x=2,y=3),(x=3,y=7). Worked by hand in the design doc /
    # planner report -- OLS beta = [-0.3, 2.2]; CR1 (Stata vce(cluster))
    # small-sample-corrected SE = [0.61237244, 0.24494897].
    X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
    y = np.array([0, 2, 3, 7], dtype=float)
    clusters = np.array([0, 0, 1, 1])
    beta, se, vcov = cluster_robust_ols(X, y, clusters)
    assert beta == pytest.approx([-0.3, 2.2], abs=1e-9)
    assert se == pytest.approx([0.6123724357, 0.2449489743], abs=1e-8)
    assert vcov.shape == (2, 2)


def test_cluster_robust_ols_requires_at_least_two_clusters():
    X = np.array([[1, 0], [1, 1]], dtype=float)
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        cluster_robust_ols(X, y, np.array([0, 0]))


def test_largest_remainder_sums_to_total():
    props = np.array([0.5, 0.3, 0.2])
    out = largest_remainder(props, 10)
    assert out.sum() == 10
    assert list(out) == [5, 3, 2]


def test_gap_statistics_excludes_small_groups():
    # group 0: 40 visits (mean 0.5), group 1: 5 visits (small, excluded),
    # group 2: 35 visits (mean 0.7).
    group_codes = np.array([0] * 40 + [1] * 5 + [2] * 35)
    values = np.concatenate([np.full(40, 0.5), np.full(5, 0.9), np.full(35, 0.7)])
    range_, wsd, n_used = gap_statistics(group_codes, values, n_groups=3, min_visits=30)
    assert n_used == 2
    assert range_ == pytest.approx(0.2)


def test_gap_statistics_returns_nan_with_fewer_than_two_big_groups():
    group_codes = np.array([0] * 40 + [1] * 5)
    values = np.concatenate([np.full(40, 0.5), np.full(5, 0.9)])
    range_, wsd, n_used = gap_statistics(group_codes, values, n_groups=2, min_visits=30)
    assert n_used == 1
    assert np.isnan(range_)
    assert np.isnan(wsd)


def test_permutation_p_reasonable_under_null():
    rng = np.random.default_rng(7)
    n_patients = 40
    n_groups = 4
    subject = np.repeat(np.arange(n_patients), 2)  # 2 visits per patient
    group = rng.integers(0, n_groups, size=len(subject))  # unrelated to outcome
    values = rng.normal(0, 1, size=len(subject))
    result = permutation_p(group, subject, values, n_groups, min_visits=5, n_perm=500, seed=0)
    assert 0.02 <= result["p_raw_range"] <= 1.0
    assert result["n_groups_used"] >= 2


def test_adjusted_group_means_recovers_planted_effect():
    # adjusted_group_means() reports each group's mean *predicted at the
    # sample's average covariate values* (design D6), not at covariates=0 --
    # so its absolute level carries a shared "covariate effect at the mean"
    # offset that has nothing to do with the group effect. What the group
    # dummies actually recover is the *difference* between groups, which is
    # what this test checks.
    rng = np.random.default_rng(42)
    n_patients = 300
    true_effect = {0: 0.0, 1: 0.2, 2: 0.4}
    rows = []
    for sid in range(n_patients):
        group = sid % 3
        n_dx = rng.integers(3, 15)
        n_med_gt = rng.integers(1, 20)
        visit_index = rng.integers(0, 4)  # must vary -- a constant column makes X singular
        noise = rng.normal(0, 0.01)
        outcome = (
            true_effect[group] + 0.01 * np.log(n_dx) + 0.001 * n_med_gt
            + 0.0 * visit_index + noise
        )
        rows.append(
            {
                "SUBJECT_ID": sid, "grp": group, "n_dx": n_dx,
                "n_med_gt": n_med_gt, "visit_index": visit_index, "y": outcome,
            }
        )
    df = pd.DataFrame(rows)
    adjusted, coef_table = adjusted_group_means(df, "grp", "y", [0, 1, 2])
    assert (adjusted[1] - adjusted[0]) == pytest.approx(
        true_effect[1] - true_effect[0], abs=0.02
    )
    assert (adjusted[2] - adjusted[0]) == pytest.approx(
        true_effect[2] - true_effect[0], abs=0.02
    )
    assert set(coef_table["term"]) == {0, 1, 2, "log_n_dx", "n_med_gt", "visit_index"}
