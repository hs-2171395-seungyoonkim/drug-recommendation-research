import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_cluster_gap import covariate_only_residuals, gap_statistics
from safedrug_mechanism import (
    adjusted_group_means_v2,
    covariate_only_residuals_v2,
    drug_training_frequency,
    make_mechanism_figure,
    mechanism_correlations,
    precision_recall_gap_table,
    precision_recall_permutation,
    rare_drug_threshold,
    rarity_features,
    training_representation,
)


def test_drug_training_frequency_hand_computed():
    train_med_sets = [{0, 1}, {0, 1}, {0, 2}, {3}]
    freq = drug_training_frequency(train_med_sets, n_med=4)
    assert freq == pytest.approx([0.75, 0.5, 0.25, 0.25])


def test_rare_drug_threshold_and_rarity_features_hand_computed():
    drug_freq = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    threshold = rare_drug_threshold(drug_freq)
    assert threshold == pytest.approx(np.percentile(drug_freq, 10))

    y_gt_100 = np.zeros(5)
    y_gt_100[[0, 1]] = 1  # drugs 0,1 -> freq 10, 20
    gt_lookup = {100: y_gt_100}
    out = rarity_features(gt_lookup, [100], drug_freq, threshold)
    row = out.iloc[0]
    assert row["mean_gt_freq"] == pytest.approx(15.0)
    assert row["min_gt_freq"] == pytest.approx(10.0)
    assert row["n_rare_gt"] == 1  # only drug 0's freq (10) is below the threshold


def test_rarity_features_zero_gt_drugs_gives_nan_mean_min():
    drug_freq = np.array([10.0, 20.0, 30.0])
    y_gt_empty = np.zeros(3)
    out = rarity_features({1: y_gt_empty}, [1], drug_freq, rare_drug_threshold(drug_freq))
    row = out.iloc[0]
    assert np.isnan(row["mean_gt_freq"])
    assert np.isnan(row["min_gt_freq"])
    assert row["n_rare_gt"] == 0


def test_load_training_visits_filters_by_split_point_and_joins_labels(tmp_path):
    # 4 visits across 3 patients (safedrug_patient_index 0, 0, 1, 2); split_point=2
    # keeps only patients with safedrug_patient_index < 2 -- patients 0 and 1 --
    # i.e. HADM_IDs 100, 101, 102 (not 103, patient 2's visit).
    master = pd.DataFrame({
        "HADM_ID": [100, 101, 102, 103], "SUBJECT_ID": [1, 1, 2, 3],
        "safedrug_patient_index": [0, 0, 1, 2], "safedrug_visit_index": [0, 1, 0, 0],
    })
    master.to_csv(tmp_path / "master_visits.csv", index=False)

    dxtext_path = tmp_path / "dxtext.csv"
    pd.DataFrame({"HADM_ID": [100, 101, 102, 103]}).to_csv(
        dxtext_path, index=False, encoding="utf-8-sig"
    )
    labels_path = tmp_path / "labels.npz"
    np.savez(
        labels_path,
        long_k10=np.array([0, 0, 1, 1], dtype=np.int32),
        long_k25=np.array([0, 0, 1, 1], dtype=np.int32),
        short_k10=np.array([0, 0, 1, 1], dtype=np.int32),
        concise_k10=np.array([0, 0, 1, 1], dtype=np.int32),
    )
    ccs_path = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": [100, 101, 102, 103], "group": ["A", "A", "B", "B"]}).to_csv(
        ccs_path, index=False, encoding="utf-8-sig"
    )

    import safedrug_mechanism as sm

    backup = sm.DXTEXT_CSV, sm.LABELS_NPZ, sm.CCS_CSV
    sm.DXTEXT_CSV, sm.LABELS_NPZ, sm.CCS_CSV = dxtext_path, labels_path, ccs_path
    try:
        out = sm.load_training_visits(tmp_path, split_point=2)
    finally:
        sm.DXTEXT_CSV, sm.LABELS_NPZ, sm.CCS_CSV = backup

    assert sorted(out["HADM_ID"]) == [100, 101, 102]
    assert out[out["HADM_ID"] == 100]["long_k10"].iloc[0] == 0
    assert out[out["HADM_ID"] == 102]["long_k10"].iloc[0] == 1


def test_precision_recall_gap_table_columns_and_values():
    df = pd.DataFrame({
        "grp": ["a"] * 5 + ["b"] * 5,
        "SUBJECT_ID": list(range(10)),
        "precision": [0.5] * 5 + [0.9] * 5,
        "recall": [0.6] * 5 + [0.4] * 5,
        "f1": [0.55] * 5 + [0.55] * 5,
        "n_med_pred": [10] * 5 + [5] * 5,
        "n_med_gt": [8] * 5 + [8] * 5,
    })
    out = precision_recall_gap_table(df, "grp").set_index("group")
    assert out.loc["a", "n_visits"] == 5
    assert out.loc["a", "mean_precision"] == pytest.approx(0.5)
    assert out.loc["b", "mean_recall"] == pytest.approx(0.4)
    assert out.loc["a", "mean_over_under"] == pytest.approx(2.0)  # 10-8
    assert out.loc["b", "mean_over_under"] == pytest.approx(-3.0)  # 5-8


def test_precision_recall_permutation_separates_precision_and_recall():
    rng = np.random.default_rng(5)
    n_per_group = 40
    rows = []
    sid = 0
    for g in range(3):
        for _ in range(n_per_group):
            rows.append({
                "grp": g, "SUBJECT_ID": sid,
                # precision has no group gap; recall differs sharply by group.
                "precision": rng.normal(0.6, 0.05),
                "recall": rng.normal(0.3 + 0.2 * g, 0.03),
            })
            sid += 1
    df = pd.DataFrame(rows)
    out = precision_recall_permutation(df, "grp")
    range_precision = out[(out["source_metric"] == "precision")
                           & (out["statistic"] == "range")].iloc[0]
    range_recall = out[(out["source_metric"] == "recall")
                        & (out["statistic"] == "range")].iloc[0]
    assert range_recall["observed"] > range_precision["observed"]
    assert range_recall["p_raw"] < range_precision["p_raw"]


def test_training_representation_hand_computed():
    train_labeled = pd.DataFrame({"grp": ["a", "a", "a", "b", "b"]})
    test_group_counts = pd.Series({"a": 10, "b": 30})

    out = training_representation(train_labeled, "grp", test_group_counts).set_index("group")
    assert out.loc["a", "train_n_visits"] == 3
    assert out.loc["a", "train_share"] == pytest.approx(0.6)
    assert out.loc["a", "test_share"] == pytest.approx(0.25)
    assert out.loc["a", "share_ratio"] == pytest.approx(2.4)
    assert out.loc["b", "train_share"] == pytest.approx(0.4)
    assert out.loc["b", "test_share"] == pytest.approx(0.75)


def test_mechanism_correlations_perfect_rank_agreement():
    group_table = pd.DataFrame(
        {
            "group": [0, 1, 2, 3],
            "small": [False, False, False, False],
            "adjusted_mean_jaccard": [0.5, 0.6, 0.7, 0.8],
            "train_share": [0.1, 0.2, 0.3, 0.4],
            "mean_gt_freq": [0.4, 0.3, 0.2, 0.1],  # perfectly inversely related
            "mean_n_med_gt": [5, 5, 5, 5],  # constant -> undefined correlation
        }
    )
    out = mechanism_correlations(group_table, "long_k10", "seed0").set_index("predictor")
    assert out.loc["train_share", "spearman_r"] == pytest.approx(1.0)
    assert out.loc["mean_gt_freq", "spearman_r"] == pytest.approx(-1.0)


def test_adjusted_group_means_v2_recovers_planted_effect():
    rng = np.random.default_rng(7)
    n_patients = 300
    rows = []
    planted = {0: 0.0, 1: 0.15, 2: 0.30}
    for sid in range(n_patients):
        group = sid % 3
        n_dx = rng.integers(3, 15)
        n_med_gt = rng.integers(1, 20)
        visit_index = rng.integers(0, 4)
        mean_gt_freq = rng.uniform(0.1, 0.9)
        n_rare_gt = rng.integers(0, 5)
        noise = rng.normal(0, 0.01)
        outcome = (
            0.3 * mean_gt_freq - 0.01 * n_rare_gt
            + 0.01 * np.log(n_dx) + 0.001 * n_med_gt + 0.0 * visit_index
            + planted[group] + noise
        )
        rows.append({"SUBJECT_ID": sid, "grp": group, "n_dx": n_dx, "n_med_gt": n_med_gt,
                     "visit_index": visit_index, "mean_gt_freq": mean_gt_freq,
                     "n_rare_gt": n_rare_gt, "y": outcome})
    df = pd.DataFrame(rows)
    adjusted, coef_table = adjusted_group_means_v2(df, "grp", "y", [0, 1, 2])
    assert (adjusted[1] - adjusted[0]) == pytest.approx(0.15, abs=0.03)
    assert (adjusted[2] - adjusted[0]) == pytest.approx(0.30, abs=0.03)
    assert set(coef_table["term"]) == {0, 1, 2, "log_n_dx", "n_med_gt", "visit_index",
                                        "mean_gt_freq", "n_rare_gt"}


def test_covariate_only_residuals_v2_shrinks_gap_when_effect_is_mediated():
    rng = np.random.default_rng(11)
    n_patients = 300
    rows = []
    freq_by_group = {0: 0.2, 1: 0.5, 2: 0.8}
    for sid in range(n_patients):
        group = sid % 3
        mean_gt_freq = freq_by_group[group] + rng.normal(0, 0.03)
        n_dx = rng.integers(3, 15)
        n_med_gt = rng.integers(1, 20)
        visit_index = rng.integers(0, 4)
        n_rare_gt = rng.integers(0, 5)
        noise = rng.normal(0, 0.01)
        # outcome depends only on mean_gt_freq (+ tiny covariates) -- NOT on group
        # directly; the group effect is entirely mediated by mean_gt_freq.
        outcome = 0.5 * mean_gt_freq + 0.01 * np.log(n_dx) + 0.001 * n_med_gt + noise
        rows.append({"SUBJECT_ID": sid, "grp": group, "n_dx": n_dx, "n_med_gt": n_med_gt,
                     "visit_index": visit_index, "mean_gt_freq": mean_gt_freq,
                     "n_rare_gt": n_rare_gt, "y": outcome})
    df = pd.DataFrame(rows)
    groups = [0, 1, 2]
    group_code_map = {g: i for i, g in enumerate(groups)}

    resid_v1, idx_v1 = covariate_only_residuals(df, "y")
    gc_v1 = df.loc[idx_v1, "grp"].map(group_code_map).to_numpy()
    range_v1, _, _ = gap_statistics(gc_v1, resid_v1.to_numpy(), len(groups))

    resid_v2, idx_v2 = covariate_only_residuals_v2(df, "y")
    gc_v2 = df.loc[idx_v2, "grp"].map(group_code_map).to_numpy()
    range_v2, _, _ = gap_statistics(gc_v2, resid_v2.to_numpy(), len(groups))

    assert range_v2 < range_v1 * 0.3


def test_make_mechanism_figure_smoke_lang_ko_writes_ko_suffixed_png(tmp_path):
    # Task G item 3: tiny synthetic groups_df with every column
    # make_mechanism_figure reads, lang="ko" -- checks the _ko-suffixed file
    # is written and the plain (English-filename) file is not. Only checks
    # the PNG gets written, not its pixel content (matplotlib 3.11 under
    # py -3.12; a missing Korean font on the test machine renders tofu boxes
    # but does not raise).
    groups_df = pd.DataFrame(
        {
            "group": [0, 1, 2],
            "n_visits": [40, 35, 50],
            "mean_precision": [0.6, 0.65, 0.55],
            "mean_recall": [0.5, 0.55, 0.45],
            "train_share": [0.3, 0.4, 0.2],
            "adjusted_mean_jaccard": [0.45, 0.50, 0.40],
            "mean_gt_freq": [0.1, 0.2, 0.05],
        }
    )
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    make_mechanism_figure(groups_df, figs_dir, lang="ko")

    assert (figs_dir / "fig_mechanism_long_k10_ko.png").exists()
    assert not (figs_dir / "fig_mechanism_long_k10.png").exists()
