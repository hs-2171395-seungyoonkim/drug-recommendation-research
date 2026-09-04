import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import (
    _figure_group_labels,
    adjusted_group_means,
    cluster_robust_ols,
    gap_statistics,
    largest_remainder,
    make_figures,
    permutation_p,
    write_report,
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


def test_write_report_renders_adjusted_means_section(tmp_path):
    # Fix-round regression test: write_report() previously read only
    # table_group_summary.csv/table_permutation.csv/per_visit_metrics.csv and never
    # rendered table_adjusted_means.csv/table_ols_coefficients.csv, so D9's required
    # "D6 raw-vs-adjusted" content never made it into the report. This test builds a
    # tiny synthetic out-dir with every file write_report() reads (including the two
    # it was missing), plants a distinctive adjusted_mean value, and asserts it shows
    # up in the rendered Markdown -- it fails if the adjusted-means section is removed
    # or if write_report() goes back to ignoring those two CSVs.
    eval_dir = tmp_path

    (eval_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "best_epoch": 5,
                "best_eval_jaccard": 0.51,
                "official_metrics": {
                    "test": {"ja": 0.50, "prauc": 0.70, "avg_f1": 0.60, "ddi_rate": 0.06, "avg_med": 8.0},
                    "eval": {"ja": 0.50, "prauc": 0.70, "avg_f1": 0.60, "ddi_rate": 0.06, "avg_med": 8.0},
                },
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "partition": ["long_k10", "long_k10"],
            "scope": ["test", "test"],
            "group": [0, 1],
            "n_visits": [40, 35],
            "n_patients": [30, 25],
            "small": [False, False],
        }
    ).to_csv(eval_dir / "table_group_summary.csv", index=False)

    pd.DataFrame(
        {
            "partition": ["long_k10"], "scope": ["test"], "outcome": ["jaccard"],
            "source": ["raw"], "statistic": ["range"], "observed": [0.1],
            "null_mean": [0.05], "null_p95": [0.09], "z": [1.5],
            "p_raw": [0.2], "p_bonferroni": [1.0],
        }
    ).to_csv(eval_dir / "table_permutation.csv", index=False)

    pd.DataFrame({"split": ["test", "eval"], "has_label": [True, True]}).to_csv(
        eval_dir / "per_visit_metrics.csv", index=False
    )

    # Planted group means -- group 0's adjusted_mean (0.5500) is the value under test.
    pd.DataFrame(
        {
            "partition": ["long_k10", "long_k10"],
            "scope": ["test", "test"],
            "outcome": ["jaccard", "jaccard"],
            "group": [0, 1],
            "raw_mean": [0.4000, 0.6000],
            "adjusted_mean": [0.5500, 0.5800],
        }
    ).to_csv(eval_dir / "table_adjusted_means.csv", index=False)

    pd.DataFrame(
        {
            "partition": ["long_k10"] * 3,
            "scope": ["test"] * 3,
            "outcome": ["jaccard"] * 3,
            "term": ["log_n_dx", "n_med_gt", "visit_index"],
            "estimate": [0.0123, 0.0012, 0.0001],
            "cluster_robust_se": [0.0045, 0.0006, 0.0002],
        }
    ).to_csv(eval_dir / "table_ols_coefficients.csv", index=False)

    write_report(eval_dir)
    report = (eval_dir / "REPORT_SAFEDRUG_CLUSTER_KO.md").read_text(encoding="utf-8")

    assert "조정 전/후 군집별 평균" in report
    assert "long_k10 / test" in report
    assert "| group | n_visits | raw_mean | adjusted_mean |" in report
    # The planted adjusted_mean value -- absent from every other section, so its
    # presence here can only come from the adjusted-means section actually rendering.
    assert "0.5500" in report
    assert "0.4000" in report
    assert "공변량 계수" in report
    assert "log_n_dx" in report


def test_figure_group_labels_normalizes_float_formatted_group_ids():
    # Fix-round-2 regression test. table_group_summary.csv's "group" column mixes
    # integer cluster ids (long_k10/short_k10/concise_k10/long_k25) with CCS
    # group-name strings (ccs_group) in the same column; attach_labels' left join
    # puts NaN into the numeric label columns for unlabeled visits, which forces
    # pandas to upcast those columns to float64 -- so a numeric group id is a float
    # (6.0) by the time it is written to CSV, and because the column overall also
    # holds non-numeric CCS strings, it round-trips through pd.read_csv as the
    # literal *string* '6.0', not the float 6.0. The original _figure_group_labels()
    # called bare int(g), which raises ValueError on that string -- this is exactly
    # the crash seen running the real pipeline (out/safedrug_eval/gap_stderr.log).
    # This test fails against that code (int('6.0') raises) and passes once group
    # ids are normalized before the theme-dict lookup.
    labels = _figure_group_labels("long_k10", ["6.0", 6, "기타(소규모)"])

    # '6.0' (string, as read back from CSV) and 6 (a clean int) must resolve to the
    # identical themed label -- not two different values, and not the str(g) fallback.
    assert labels[0] == labels[1]
    assert labels[0] != "6.0"
    assert " / " in labels[0]  # themed label format is "{빈도1} / {top_lift}"

    # A genuine (non-numeric) CCS group name is not in the long_k10 theme dict and
    # must pass through unchanged, exactly like the pre-fix str(g) fallback did.
    assert labels[2] == "기타(소규모)"


def test_make_figures_smoke_writes_pngs_with_short_labels(tmp_path):
    # Task F item 2: make_figures previously called _figure_group_labels(),
    # whose long_k10 output ("code title (pct) / code title xlift (pct)")
    # overlaps badly once tick-labeled on an axis -- it now uses the shared
    # safedrug_percluster.labels.short_group_label() helper instead. This is
    # a smoke test (tiny synthetic table under tmp_path, matplotlib 3.11
    # under py -3.12) -- it only checks the figures actually get written,
    # not their pixel content.
    eval_dir = tmp_path
    groups = [0, 1, 2]

    summary_rows = [
        {
            "partition": "long_k10", "scope": "test", "group": g,
            "mean_jaccard": 0.40 + 0.05 * g,
            "ci_low_jaccard": 0.35 + 0.05 * g,
            "ci_high_jaccard": 0.45 + 0.05 * g,
            "mean_ddi_rate_visit": 0.05 + 0.01 * g,
            "mean_n_med_pred": 15 + g,
        }
        for g in groups
    ]
    pd.DataFrame(summary_rows).to_csv(eval_dir / "table_group_summary.csv", index=False)

    adjusted_rows = [
        {"partition": "long_k10", "scope": "test", "group": g, "outcome": "jaccard",
         "adjusted_mean": 0.42 + 0.03 * g}
        for g in groups
    ]
    pd.DataFrame(adjusted_rows).to_csv(eval_dir / "table_adjusted_means.csv", index=False)

    pd.DataFrame(
        {"epoch": [0, 1, 2], "eval_ja": [0.40, 0.45, 0.50], "eval_ddi_rate": [0.07, 0.065, 0.06]}
    ).to_csv(eval_dir / "train_log.csv", index=False)

    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    make_figures(eval_dir, figs_dir)

    assert (figs_dir / "fig_long_k10_raw_vs_adjusted.png").exists()
    assert (figs_dir / "fig_long_k10_ddi_nmed.png").exists()
    assert (figs_dir / "fig_train_log.png").exists()
    # Other partitions (short_k10/concise_k10/long_k25/ccs_group) have no
    # rows in this tiny fixture -- make_figures must skip them, not crash.
    assert not (figs_dir / "fig_ccs_group_raw_vs_adjusted.png").exists()


def test_make_figures_smoke_lang_ko_writes_ko_suffixed_pngs(tmp_path):
    # Task G item 3: lang="ko" smoke test on the same tiny synthetic fixture
    # as test_make_figures_smoke_writes_pngs_with_short_labels above. Only
    # checks the _ko-suffixed files get written (and that the plain
    # English-suffix files are NOT also written by this call) -- matplotlib
    # never raises for a missing font family, so this passes even on a
    # machine without Malgun Gothic installed.
    eval_dir = tmp_path
    groups = [0, 1, 2]

    summary_rows = [
        {
            "partition": "long_k10", "scope": "test", "group": g,
            "mean_jaccard": 0.40 + 0.05 * g,
            "ci_low_jaccard": 0.35 + 0.05 * g,
            "ci_high_jaccard": 0.45 + 0.05 * g,
            "mean_ddi_rate_visit": 0.05 + 0.01 * g,
            "mean_n_med_pred": 15 + g,
        }
        for g in groups
    ]
    pd.DataFrame(summary_rows).to_csv(eval_dir / "table_group_summary.csv", index=False)

    adjusted_rows = [
        {"partition": "long_k10", "scope": "test", "group": g, "outcome": "jaccard",
         "adjusted_mean": 0.42 + 0.03 * g}
        for g in groups
    ]
    pd.DataFrame(adjusted_rows).to_csv(eval_dir / "table_adjusted_means.csv", index=False)

    pd.DataFrame(
        {"epoch": [0, 1, 2], "eval_ja": [0.40, 0.45, 0.50], "eval_ddi_rate": [0.07, 0.065, 0.06]}
    ).to_csv(eval_dir / "train_log.csv", index=False)

    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    make_figures(eval_dir, figs_dir, lang="ko")

    assert (figs_dir / "fig_long_k10_raw_vs_adjusted_ko.png").exists()
    assert (figs_dir / "fig_long_k10_ddi_nmed_ko.png").exists()
    assert (figs_dir / "fig_train_log_ko.png").exists()
    # lang="ko" must not also write the plain (English-filename) figures.
    assert not (figs_dir / "fig_long_k10_raw_vs_adjusted.png").exists()
    assert not (figs_dir / "fig_long_k10_ddi_nmed.png").exists()
    assert not (figs_dir / "fig_train_log.png").exists()
