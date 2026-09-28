import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_group_ddi_targets import (
    build_visit_target_csv,
    group_ground_truth_ddi_rates,
    load_training_gt_med_sets,
    pairwise_ddi_rate,
    parse_args,
)

# Meds 0 and 3 are the one DDI pair -- the repo's existing DDI test fixture.
DDI_ADJ = np.array(
    [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)


def test_pairwise_ddi_rate_hand_computed():
    med_sets = [{0, 3}, {1, 2}]  # 1 DDI pair out of 2 total pairs
    assert pairwise_ddi_rate(med_sets, DDI_ADJ) == pytest.approx(0.5)


def test_pairwise_ddi_rate_zero_pairs_returns_zero():
    med_sets = [{0}, set()]
    assert pairwise_ddi_rate(med_sets, DDI_ADJ) == pytest.approx(0.0)


def test_group_ground_truth_ddi_rates_clips_high_and_low():
    train_labeled = pd.DataFrame({
        "HADM_ID": [100, 101, 200, 201],
        "long_k10": [0, 0, 1, 1],
    })
    gt_med_sets = {100: {0, 3}, 101: {1, 2}, 200: {0, 1, 2}, 201: {3}}
    out = group_ground_truth_ddi_rates(train_labeled, gt_med_sets, DDI_ADJ, clip_low=0.03, clip_high=0.10)
    out = out.set_index("group")
    assert out.loc[0, "raw_ddi_rate"] == pytest.approx(0.5)
    assert out.loc[0, "target_ddi"] == pytest.approx(0.10)  # clipped down
    assert out.loc[1, "raw_ddi_rate"] == pytest.approx(0.0)
    assert out.loc[1, "target_ddi"] == pytest.approx(0.03)  # clipped up
    assert out.loc[0, "n_train_visits"] == 2
    assert out.loc[1, "n_train_visits"] == 2


def test_group_ground_truth_ddi_rates_drops_unlabeled_rows():
    train_labeled = pd.DataFrame({
        "HADM_ID": [100, 101],
        "long_k10": [0.0, float("nan")],
    })
    gt_med_sets = {100: {0, 3}, 101: {1, 2}}
    out = group_ground_truth_ddi_rates(train_labeled, gt_med_sets, DDI_ADJ)
    assert list(out["group"]) == [0]


def test_build_visit_target_csv_labeled_and_unlabeled_fallback():
    train_labeled = pd.DataFrame({
        "HADM_ID": [100, 101, 999],
        "long_k10": [0.0, 1.0, float("nan")],
    })
    group_table = pd.DataFrame({"group": [0, 1], "target_ddi": [0.10, 0.03]})
    out = build_visit_target_csv(train_labeled, group_table, default_target=0.06).set_index("HADM_ID")
    assert out.loc[100, "target_ddi"] == pytest.approx(0.10)
    assert out.loc[101, "target_ddi"] == pytest.approx(0.03)
    assert out.loc[999, "target_ddi"] == pytest.approx(0.06)  # unlabeled -> flat default


def test_group_ground_truth_ddi_rates_partition_ccs_group_joins_by_hadm_id(tmp_path):
    # Mirrors tests/test_safedrug_mechanism.py's
    # test_load_training_visits_filters_by_split_point_and_joins_labels
    # fixture exactly, so this test exercises the SAME attach_labels-based
    # HADM_ID join safedrug_mechanism.load_training_visits performs, then
    # feeds its real output into group_ground_truth_ddi_rates(partition=
    # "ccs_group") -- proving the ccs_group column reaching this function
    # came from an actual CSV join on HADM_ID, not a stand-in DataFrame.
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
    # The "tiny synthetic assignments CSV": training HADM_IDs 100/101 -> "A",
    # 102 -> "B" (103 is a test-split visit and must not affect the result).
    ccs_path = tmp_path / "ccs.csv"
    pd.DataFrame({"HADM_ID": [100, 101, 102, 103], "group": ["A", "A", "B", "B"]}).to_csv(
        ccs_path, index=False, encoding="utf-8-sig"
    )

    import safedrug_mechanism as sm

    backup = sm.DXTEXT_CSV, sm.LABELS_NPZ, sm.CCS_CSV
    sm.DXTEXT_CSV, sm.LABELS_NPZ, sm.CCS_CSV = dxtext_path, labels_path, ccs_path
    try:
        train_labeled = sm.load_training_visits(tmp_path, split_point=2)
    finally:
        sm.DXTEXT_CSV, sm.LABELS_NPZ, sm.CCS_CSV = backup

    assert sorted(train_labeled["HADM_ID"]) == [100, 101, 102]

    # 100:{0,3} is a DDI pair, 101:{1,2} is not, 102:{0,3} is a DDI pair.
    gt_med_sets = {100: {0, 3}, 101: {1, 2}, 102: {0, 3}}
    out = group_ground_truth_ddi_rates(
        train_labeled, gt_med_sets, DDI_ADJ, clip_low=0.0, clip_high=1.0, partition="ccs_group"
    ).set_index("group")

    assert list(out.index) == ["A", "B"]
    assert out.loc["A", "n_train_visits"] == 2
    assert out.loc["A", "raw_ddi_rate"] == pytest.approx(0.5)  # 1 DDI pair of 2 (100 + 101)
    assert out.loc["B", "n_train_visits"] == 1
    assert out.loc["B", "raw_ddi_rate"] == pytest.approx(1.0)  # 102 alone, its 1 pair is DDI


def test_group_ground_truth_ddi_rates_shrink_k1_n1_gives_midpoint():
    # group 0 has exactly one training visit (n_g=1) whose ground-truth set
    # {0,3} is a DDI pair -> raw_ddi_rate = 1.0. group 1 has two visits
    # ({1,2} not DDI, {0,3} DDI) -> raw_ddi_rate = 0.5. r_global (pooled over
    # all three labeled visits) = 2 DDI pairs / 3 total pairs = 2/3.
    # shrink_k=1 on group 0 (n_g=1): shrunk = (1*1.0 + 1*(2/3)) / (1+1),
    # exactly the arithmetic midpoint of raw_ddi_rate and r_global.
    train_labeled = pd.DataFrame({
        "HADM_ID": [100, 101, 102],
        "long_k10": [0, 1, 1],
    })
    gt_med_sets = {100: {0, 3}, 101: {1, 2}, 102: {0, 3}}
    r_global = 2.0 / 3.0

    out = group_ground_truth_ddi_rates(
        train_labeled, gt_med_sets, DDI_ADJ, clip_low=0.0, clip_high=1.0, shrink_k=1.0
    ).set_index("group")

    expected_midpoint = (1.0 + r_global) / 2.0
    assert out.loc[0, "n_train_visits"] == 1
    assert out.loc[0, "raw_ddi_rate"] == pytest.approx(1.0)
    assert out.loc[0, "shrunk_ddi_rate"] == pytest.approx(expected_midpoint)
    assert out.loc[0, "shrunk_ddi_rate"] == pytest.approx((1.0 * 1.0 + 1.0 * r_global) / (1.0 + 1.0))
    assert out.loc[0, "target_ddi"] == pytest.approx(expected_midpoint)  # clip range [0,1] is a no-op here


def test_group_ground_truth_ddi_rates_shrink_k0_identical_to_raw_rate():
    train_labeled = pd.DataFrame({
        "HADM_ID": [100, 101, 200, 201],
        "long_k10": [0, 0, 1, 1],
    })
    gt_med_sets = {100: {0, 3}, 101: {1, 2}, 200: {0, 1, 2}, 201: {3}}
    out = group_ground_truth_ddi_rates(
        train_labeled, gt_med_sets, DDI_ADJ, clip_low=0.03, clip_high=0.10, shrink_k=0.0
    ).set_index("group")
    # shrink_k=0 (the default) must reproduce the pre-shrinkage behaviour
    # exactly: shrunk_ddi_rate == raw_ddi_rate (bit-exact, not just close),
    # and target_ddi unchanged from test_..._clips_high_and_low above.
    assert out.loc[0, "shrunk_ddi_rate"] == out.loc[0, "raw_ddi_rate"]
    assert out.loc[1, "shrunk_ddi_rate"] == out.loc[1, "raw_ddi_rate"]
    assert out.loc[0, "target_ddi"] == pytest.approx(0.10)
    assert out.loc[1, "target_ddi"] == pytest.approx(0.03)


def test_parse_args_partition_and_shrink_k_defaults_unchanged():
    args = parse_args(["--out-csv", "out.csv"])
    assert args.partition == "long_k10"
    assert args.shrink_k == 0.0


def test_parse_args_accepts_ccs_group_and_shrink_k():
    args = parse_args(["--out-csv", "out.csv", "--partition", "ccs_group", "--shrink-k", "25"])
    assert args.partition == "ccs_group"
    assert args.shrink_k == pytest.approx(25.0)


def test_build_visit_target_csv_partition_ccs_group():
    train_labeled = pd.DataFrame({
        "HADM_ID": [100, 101, 999],
        "ccs_group": ["A", "B", float("nan")],
    })
    group_table = pd.DataFrame({"group": ["A", "B"], "target_ddi": [0.10, 0.03]})
    out = build_visit_target_csv(
        train_labeled, group_table, default_target=0.06, partition="ccs_group"
    ).set_index("HADM_ID")
    assert out.loc[100, "target_ddi"] == pytest.approx(0.10)
    assert out.loc[101, "target_ddi"] == pytest.approx(0.03)
    assert out.loc[999, "target_ddi"] == pytest.approx(0.06)  # unlabeled -> flat default


def test_load_training_gt_med_sets_joins_by_position(tmp_path):
    # 3 patients (2 training, split_point=2), 1 visit each -- records[p][v] = [diag, proc, meds].
    records = [
        [[[1], [], [0, 3]]],
        [[[2], [], [1, 2]]],
        [[[3], [], [2]]],  # patient_index 2 -- test/eval, must NOT appear in output
    ]
    with (tmp_path / "records_reconstructed.pkl").open("wb") as fh:
        pickle.dump(records, fh)

    master = pd.DataFrame({
        "HADM_ID": [100, 101, 102],
        "SUBJECT_ID": [1, 2, 3],
        "safedrug_patient_index": [0, 1, 2],
        "safedrug_visit_index": [0, 0, 0],
    })
    master.to_csv(tmp_path / "master_visits.csv", index=False)

    gt_med_sets, split_point = load_training_gt_med_sets(tmp_path)
    assert split_point == 2
    assert gt_med_sets == {100: {0, 3}, 101: {1, 2}}
    assert 102 not in gt_med_sets
