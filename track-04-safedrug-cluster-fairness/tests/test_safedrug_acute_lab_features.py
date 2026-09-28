import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safedrug_acute_lab_features import (
    LAB_PANEL,
    bin_value,
    compute_quantile_edges,
    first_value_in_window,
    load_labevents_for_cohort,
    main,
    split_boundaries,
)


def _rows(charttimes, valuenums):
    return pd.DataFrame({"CHARTTIME": pd.to_datetime(charttimes), "VALUENUM": valuenums})


def test_first_value_in_window_picks_earliest_within_window():
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows(
        ["2248-04-11 23:04:00", "2248-04-11 17:04:00", "2248-04-13 03:04:00"],
        [5.0, 3.0, 9.0],
    )
    assert first_value_in_window(rows, admittime) == 3.0


def test_first_value_in_window_includes_pre_admission_ed_draw():
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows(["2248-04-11 11:04:00"], [7.0])  # 4h before admission, inside the 6h lookback
    assert first_value_in_window(rows, admittime) == 7.0


def test_first_value_in_window_excludes_before_lookback():
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows(["2248-04-11 08:04:00"], [7.0])  # 7h before, outside the 6h lookback
    assert first_value_in_window(rows, admittime) is None


def test_first_value_in_window_excludes_after_24h():
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows(["2248-04-12 16:04:00"], [7.0])  # 25h after admission
    assert first_value_in_window(rows, admittime) is None


def test_first_value_in_window_ignores_null_valuenum():
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows(["2248-04-11 16:04:00", "2248-04-11 17:04:00"], [np.nan, 4.0])
    assert first_value_in_window(rows, admittime) == 4.0


def test_first_value_in_window_returns_none_for_zero_value_correctly():
    # regression guard: 0.0 is falsy in Python but a valid lab result -- must
    # not be treated as "no result found".
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows(["2248-04-11 16:04:00"], [0.0])
    assert first_value_in_window(rows, admittime) == 0.0


def test_first_value_in_window_empty_rows_returns_none():
    admittime = pd.Timestamp("2248-04-11 15:04:00")
    rows = _rows([], [])
    assert first_value_in_window(rows, admittime) is None


def test_compute_quantile_edges_hand_computed():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    edges = compute_quantile_edges(values, n_bins=5)
    assert len(edges) == 4
    expected = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    assert np.allclose(edges, expected, atol=1e-4)


def test_compute_quantile_edges_ignores_nan():
    values = np.array([1, 2, 3, 4, 5, np.nan, np.nan])
    edges = compute_quantile_edges(values, n_bins=5)
    expected = np.quantile([1, 2, 3, 4, 5], [0.2, 0.4, 0.6, 0.8])
    assert np.allclose(edges, expected, atol=1e-4)


def test_compute_quantile_edges_raises_on_all_nan():
    with pytest.raises(ValueError):
        compute_quantile_edges(np.array([np.nan, np.nan]), n_bins=5)


def test_bin_value_below_first_edge_is_bin_zero():
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    assert bin_value(5.0, edges, missing_bin=5) == 0


def test_bin_value_above_last_edge_is_bin_four():
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    assert bin_value(50.0, edges, missing_bin=5) == 4


def test_bin_value_on_edge_goes_to_the_upper_bin():
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    assert bin_value(20.0, edges, missing_bin=5) == 2  # side="right": ties go up


def test_bin_value_missing_returns_missing_bin():
    edges = np.array([10.0, 20.0, 30.0, 40.0])
    assert bin_value(None, edges, missing_bin=5) == 5
    assert bin_value(float("nan"), edges, missing_bin=5) == 5


def test_split_boundaries_matches_known_cohort_size():
    assert split_boundaries(6350) == (4233, 1058)


def test_load_labevents_for_cohort_filters_item_and_hadm_and_parses_charttime(tmp_path):
    csv_path = tmp_path / "LABEVENTS.csv"
    pd.DataFrame(
        {
            "SUBJECT_ID": [1, 1, 2, 1],
            "HADM_ID": [100, 100, 200, 999],  # 999 is not in the cohort
            "ITEMID": [50912, 99999, 50912, 50912],  # 99999 is not in the panel
            "CHARTTIME": [
                "2248-04-11 15:04:00", "2248-04-11 15:04:00",
                "2248-04-11 15:04:00", "2248-04-11 15:04:00",
            ],
            "VALUENUM": [1.1, 2.2, 3.3, 4.4],
        }
    ).to_csv(csv_path, index=False)
    out = load_labevents_for_cohort(csv_path, itemids={50912}, hadm_ids={100, 200}, chunksize=2)
    assert sorted(out["HADM_ID"].tolist()) == [100, 200]
    assert (out["ITEMID"] == 50912).all()
    assert out["CHARTTIME"].dtype.kind == "M"  # parsed to datetime


def test_load_labevents_for_cohort_reads_gzip_compressed_csv(tmp_path):
    # Controller override: the real LABEVENTS file on this machine is
    # LABEVENTS.csv.gz (gzip-compressed) -- prove the compression-inferred
    # read path works, not just plain .csv.
    csv_gz_path = tmp_path / "LABEVENTS.csv.gz"
    pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2],
            "HADM_ID": [100, 200],
            "ITEMID": [50912, 50912],
            "CHARTTIME": ["2248-04-11 15:04:00", "2248-04-11 15:04:00"],
            "VALUENUM": [1.1, 3.3],
        }
    ).to_csv(csv_gz_path, index=False, compression="gzip")
    out = load_labevents_for_cohort(csv_gz_path, itemids={50912}, hadm_ids={100, 200}, chunksize=2)
    assert sorted(out["HADM_ID"].tolist()) == [100, 200]
    assert out["CHARTTIME"].dtype.kind == "M"


def test_lab_panel_has_thirty_items():
    assert len(LAB_PANEL) == 30


def test_main_computes_quantile_edges_from_train_split_only(tmp_path):
    # 7 one-visit patients -> split_boundaries(7) == (4, 1): patients 0-3
    # train (HADM_ID 1-4), patients 4-5 test (HADM_ID 5-6), patient 6 eval
    # (HADM_ID 7). Plant extreme test/eval VALUENUMs (99999, 88888, -99999)
    # for the panel's creatinine item that would drag the quantile edges far
    # outside the train-only range if they leaked into edge computation.
    assert split_boundaries(7) == (4, 1)
    itemid = 50912  # creatinine
    hadm_ids_nested = [[1], [2], [3], [4], [5], [6], [7]]
    train_valuenums = {1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0}
    other_valuenums = {5: 99999.0, 6: 88888.0, 7: -99999.0}
    all_valuenums = {**train_valuenums, **other_valuenums}

    hadm_ids_path = tmp_path / "records_final_hadm_ids.pkl"
    with open(hadm_ids_path, "wb") as fh:
        dill.dump(hadm_ids_nested, fh)

    admissions = pd.DataFrame(
        {
            "HADM_ID": list(all_valuenums),
            "ADMITTIME": [f"2248-04-1{h} 15:04:00" for h in all_valuenums],
        }
    )
    admissions_csv = tmp_path / "ADMISSIONS.csv"
    admissions.to_csv(admissions_csv, index=False)

    labevents = pd.DataFrame(
        {
            "SUBJECT_ID": list(all_valuenums),
            "HADM_ID": list(all_valuenums),
            "ITEMID": [itemid] * len(all_valuenums),
            "CHARTTIME": [f"2248-04-1{h} 16:04:00" for h in all_valuenums],  # 1h after admission
            "VALUENUM": list(all_valuenums.values()),
        }
    )
    labevents_csv = tmp_path / "LABEVENTS.csv"
    labevents.to_csv(labevents_csv, index=False)

    d_labitems_csv = tmp_path / "D_LABITEMS.csv"
    pd.DataFrame({"ITEMID": [itemid], "LABEL": ["Creatinine"]}).to_csv(d_labitems_csv, index=False)

    out_dir = tmp_path / "out"
    main(
        [
            "--labevents", str(labevents_csv),
            "--d-labitems", str(d_labitems_csv),
            "--admissions", str(admissions_csv),
            "--hadm-ids-pkl", str(hadm_ids_path),
            "--out-dir", str(out_dir),
        ]
    )

    loaded = np.load(out_dir / "lab_query.npz", allow_pickle=True)
    items = sorted(LAB_PANEL)
    col = items.index(itemid)
    actual_edges = loaded["edges"][col]

    expected_edges = compute_quantile_edges(list(train_valuenums.values()), n_bins=5)
    assert np.allclose(actual_edges, expected_edges, atol=1e-4)
    # Sanity: if the planted test/eval outliers had leaked into edge
    # computation, the top edge would be dragged up near 88888/99999 --
    # confirm it stayed inside the train-only value range instead.
    assert actual_edges.max() < 1000.0
