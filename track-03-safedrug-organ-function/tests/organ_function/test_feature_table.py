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
