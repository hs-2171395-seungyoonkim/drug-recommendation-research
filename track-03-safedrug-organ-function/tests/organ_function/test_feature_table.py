import math

import pandas as pd

from organ_function.feature_table import compute_lab_features

COLUMNS = ["subject_id", "itemid", "charttime", "valuenum", "ref_range_lower", "ref_range_upper"]


def _labs(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_no_prior_labs_returns_missing():
    labs = _labs([])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-05 10:24:00"))
    assert result["missing"] is True
    assert math.isnan(result["value"])
    assert math.isnan(result["deviation"])
    assert math.isnan(result["delta"])


def test_single_prior_lab_has_value_but_no_delta():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["missing"] is False
    assert result["value"] == 0.8
    assert math.isnan(result["delta"])


def test_two_prior_labs_computes_delta_as_latest_minus_previous():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-06 18:24:00"), 1.3, 0.4, 1.1],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-07 10:24:00"))
    assert result["value"] == 1.3
    assert result["delta"] == 1.3 - 0.8


def test_value_within_range_has_zero_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.7, 0.4, 1.1]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["deviation"] == 0.0


def test_value_above_range_has_positive_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 2.1, 0.4, 1.1]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["deviation"] == (2.1 - 1.1) / (1.1 - 0.4)


def test_value_below_range_has_negative_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.1, 0.4, 1.1]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["deviation"] == (0.1 - 0.4) / (1.1 - 0.4)


def test_missing_ref_range_gives_nan_deviation():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, float("nan"), float("nan")]])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert math.isnan(result["deviation"])
    assert result["value"] == 0.8


def test_excludes_labs_at_or_after_index_time():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 18:24:00"), 0.8, 0.4, 1.1],
        [100, 50912, pd.Timestamp("2240-11-06 18:24:00"), 9.9, 0.4, 1.1],  # in/after the visit
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 18:24:00"))
    assert result["value"] == 0.8  # the 9.9 reading is not strictly before index_time


def test_ignores_rows_for_other_itemids():
    labs = _labs([
        [100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1],
        [100, 51006, pd.Timestamp("2240-11-05 10:24:00"), 15.0, 7, 20],
    ])
    result = compute_lab_features(labs, 50912, pd.Timestamp("2240-11-06 10:24:00"))
    assert result["value"] == 0.8


from organ_function.feature_table import build_visit_features, build_feature_table
from organ_function.lab_config import LAB_NAMES


def test_build_visit_features_has_all_expected_keys():
    labs = _labs([[100, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1]])
    result = build_visit_features(
        subject_labs=labs,
        index_time=pd.Timestamp("2240-11-06 10:24:00"),
        current_diag_ids=[1, 2],
        prior_diag_ids_seen=set(),
    )
    for name in LAB_NAMES:
        assert f"{name}_value" in result
        assert f"{name}_deviation" in result
        assert f"{name}_delta" in result
        assert f"{name}_missing" in result
    assert "new_diagnosis_flag" in result
    assert result["creatinine_value"] == 0.8
    assert result["bun_missing"] is True
    assert result["new_diagnosis_flag"] is False  # empty prior history


def test_build_feature_table_matches_records_shape_and_uses_previous_visit_only():
    # Two patients: patient 0 has 2 visits, patient 1 has 1 visit.
    records = [
        [
            [[1], [], [0]],       # patient 0, visit 0: diag=[1]
            [[1, 2], [], [0]],    # patient 0, visit 1: diag gains code 2 (new)
        ],
        [
            [[9], [], [0]],       # patient 1, visit 0
        ],
    ]
    hadm_ids = [[100, 101], [200]]

    lab_subset = _labs([
        # subject 5 (patient 0): one creatinine reading before visit 0's index time
        [5, 50912, pd.Timestamp("2240-11-05 10:24:00"), 0.8, 0.4, 1.1],
        # and a later one, before visit 1's index time but after visit 0's
        [5, 50912, pd.Timestamp("2240-11-14 10:24:00"), 1.5, 0.4, 1.1],
    ])

    hadm_to_subject = pd.Series({100: 5, 101: 5, 200: 7})
    admission_times = pd.Series({
        100: pd.Timestamp("2240-11-09 10:24:00"),
        101: pd.Timestamp("2240-11-19 10:24:00"),
        200: pd.Timestamp("2240-12-06 10:24:00"),
    })

    table = build_feature_table(records, hadm_ids, lab_subset, admission_times, hadm_to_subject)

    assert len(table) == len(records)
    assert len(table[0]) == 2
    assert len(table[1]) == 1

    # visit 0 (index_time 11-09 10:24): only the 11-05 10:24 reading is prior -> value 0.8
    assert table[0][0]["creatinine_value"] == 0.8
    # visit 1 (index_time 11-19 10:24): both readings are prior -> latest is 1.5
    assert table[0][1]["creatinine_value"] == 1.5
    assert table[0][1]["new_diagnosis_flag"] is True  # code 2 is new vs. {1}

    # patient 1 has no labs for subject 7 -> missing
    assert table[1][0]["creatinine_missing"] is True
